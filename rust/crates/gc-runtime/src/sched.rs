//! Guest thread scheduler, interrupts, alarms and time.
//!
//! The SDK's own scheduler ends in `OSLoadContext` + `rfi`, which a static recompilation can't
//! follow. Instead every OSThread runs on its own host thread and a single token (`running`)
//! decides which one executes guest code. The HLE layer maps OSSleepThread / OSWakeupThread /
//! OSResumeThread / ... onto this, using the SDK's priority rules (lower value = higher prio).
//!
//! Interrupts are raised as pending bits and delivered at the next function entry of the
//! running thread (when MSR.EE is set), or by the idle thread when nothing is runnable. Handlers
//! come from the OS interrupt table at 0x80003040, so the game's own handlers run.

use crate::{Ctx, CtxExt, Machine, MSR_EE, OS_CURRENT_THREAD, REG_MSR};
use gc_core::ATTENTION;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::*};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// Token holder markers (real guest threads are OSThread pointers, always >= 0x80000000).
pub const NONE: u32 = 0;
pub const IDLE: u32 = 1;
pub const BOOT: u32 = 2;

pub const IRQ_DSP_AI: u32 = 5;
pub const IRQ_DSP_ARAM: u32 = 6;
pub const IRQ_DSP_DSP: u32 = 7;
pub const IRQ_AI: u32 = 8;
pub const IRQ_PE_TOKEN: u32 = 18;
pub const IRQ_PE_FINISH: u32 = 19;
pub const IRQ_SI: u32 = 20;
pub const IRQ_DI: u32 = 21;
pub const IRQ_VI: u32 = 24;
const INTERRUPT_TABLE: u32 = 0x8000_3040;

/// OSThread field offsets / states.
const T_STATE: u32 = 0x2C8;
const T_SUSPEND: u32 = 0x2CC;
const T_PRIORITY: u32 = 0x2D0;
const T_QUEUE_JOIN: u32 = 0x2E8;
const GS_READY: u64 = 1;
const GS_RUNNING: u64 = 2;
const GS_WAITING: u64 = 4;
const GS_MORIBUND: u64 = 8;

/// Time base runs at bus clock / 4 = 40.5 MHz.
pub const TB_HZ: u64 = 40_500_000;
const VI_PERIOD: u64 = TB_HZ * 1001 / 60_000; // 59.94 Hz

static PENDING_IRQ: AtomicU32 = AtomicU32::new(0);
/// AI DMA pacing: time-base ticks per buffer (0 = stopped) and when the next one is due.
static AI_PERIOD: AtomicU64 = AtomicU64::new(0);
static AI_NEXT: AtomicU64 = AtomicU64::new(0);

pub fn set_ai_period(ticks: u64) {
    let was = AI_PERIOD.swap(ticks, Relaxed);
    if ticks > 0 && was == 0 {
        AI_NEXT.store(timebase() + ticks, Relaxed);
    }
}
/// Stack pointer for interrupt handlers run on the idle context (reserved RAM set at boot).
static IDLE_SP: AtomicU32 = AtomicU32::new(0);

pub fn set_idle_stack(sp: u32) {
    IDLE_SP.store(sp, Relaxed);
}
static ALARM_DUE: AtomicBool = AtomicBool::new(false);
pub static VI_COUNT: AtomicU32 = AtomicU32::new(0);

fn start() -> Instant {
    static S: OnceLock<Instant> = OnceLock::new();
    *S.get_or_init(Instant::now)
}

pub fn timebase() -> u64 {
    (start().elapsed().as_nanos() as u64).wrapping_mul(81) / 2000
}

#[inline]
pub fn pending() -> bool {
    PENDING_IRQ.load(Relaxed) != 0 || ALARM_DUE.load(Relaxed)
}

pub fn raise(irq: u32) {
    crate::probe::irq_raised(irq);
    PENDING_IRQ.fetch_or(1 << irq, Relaxed);
    ATTENTION.store(true, Relaxed);
    // If every guest thread is asleep (e.g. waiting for this very interrupt), only the idle
    // thread can deliver it: wake it now instead of at its next timed tick.
    if let Some(m) = crate::try_machine() {
        m.sched.cv.notify_all();
    }
}

/// Timebase of the most recent VI retrace (for the beam position / field registers).
pub static LAST_VI_TB: AtomicU64 = AtomicU64::new(0);

/// VI beam position as the hardware reports it: (vertical count 1..525 over a full
/// interlaced frame = two retraces, horizontal count). The field is the parity of the
/// retrace counter, so VIGetNextField alternates every retrace like on hardware.
pub fn vi_beam() -> (u32, u32) {
    let field = VI_COUNT.load(Relaxed) & 1;
    let since = timebase().saturating_sub(LAST_VI_TB.load(Relaxed));
    let within = (since * 262 / VI_PERIOD).min(261) as u32;
    (field * 263 + within + 1, 1)
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum TS {
    Ready,
    Running,
    Waiting(u32),
    Suspended,
    Dead,
}

struct Th {
    prio: i32,
    state: TS,
    seq: u64,
    suspend: i32,
    started: bool,
    /// Incarnation: OSThread structs get reused (static THP threads, heap-allocated loaders), so
    /// a host thread only owns `running == addr` while its generation matches.
    incarnation: u64,
}

thread_local! {
    /// Generation of the guest thread this host thread runs (0 = the boot thread).
    static MY_INCARNATION: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// True if this host thread's guest thread was cancelled/exited or its OSThread re-created:
/// it must never run guest code again.
fn stale(g: &St, me: u32) -> bool {
    g.threads.get(&me).is_some_and(|t| t.state == TS::Dead || t.incarnation != MY_INCARNATION.with(|c| c.get()))
}

fn park_stale() -> ! {
    loop {
        std::thread::park();
    }
}

struct Alarm {
    ptr: u32,
    handler: u32,
    fire: u64,
    period: u64,
}

struct St {
    next_incarnation: u64,
    running: u32,
    threads: BTreeMap<u32, Th>,
    seq: u64,
    alarms: Vec<Alarm>,
    next_vi: u64,
}

pub struct Sched {
    st: Mutex<St>,
    cv: Condvar,
    /// Latest small-data-area bases (r2 << 32 | r13) seen on a guest thread, for the idle
    /// context that runs interrupt handlers when every thread is blocked.
    sda: AtomicU64,
}

impl Default for Sched {
    fn default() -> Self {
        let mut threads = BTreeMap::new();
        threads.insert(BOOT, Th { prio: 16, state: TS::Running, seq: 0, suspend: 0, started: true, incarnation: 0 });
        Sched {
            st: Mutex::new(St { next_incarnation: 1, running: BOOT, threads, seq: 1, alarms: Vec::new(), next_vi: VI_PERIOD }),
            cv: Condvar::new(),
            sda: AtomicU64::new(0),
        }
    }
}

fn set_gstate(m: &Machine, t: u32, s: u64) {
    if t >= 0x8000_0000 {
        m.write_be(t + T_STATE, 2, s);
    }
}

impl Sched {
    fn lock(&self) -> MutexGuard<'_, St> {
        self.st.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Re-key the boot context to the OS default thread once OSInit has created it, and record
    /// the SDA bases for the idle context.
    fn adopt(&self, c: &mut Ctx) {
        let sda = (c.r(crate::REG_R2, 4) << 32) | c.r(crate::REG_R13, 4);
        if sda != 0 {
            self.sda.store(sda, Relaxed);
        }
        if c.thread == BOOT {
            let p = c.m().read_be(OS_CURRENT_THREAD, 4) as u32;
            if p >= 0x8000_0000 {
                let mut g = self.lock();
                crate::hostprof::rekey(BOOT, p);
                if let Some(mut th) = g.threads.remove(&BOOT) {
                    th.prio = c.m().read_be(p + T_PRIORITY, 4) as i32;
                    g.threads.insert(p, th);
                }
                if g.running == BOOT {
                    g.running = p;
                }
                c.thread = p;
            }
        }
    }

    fn pick(g: &St) -> Option<u32> {
        g.threads
            .iter()
            .filter(|(_, t)| t.state == TS::Ready)
            .min_by_key(|(_, t)| (t.prio, t.seq))
            .map(|(k, _)| *k)
    }

    fn make_ready(g: &mut St, m: &Machine, t: u32) {
        g.seq += 1;
        let seq = g.seq;
        if let Some(th) = g.threads.get_mut(&t) {
            th.state = TS::Ready;
            th.seq = seq;
        }
        set_gstate(m, t, GS_READY);
    }

    /// Hand the token to the best ready thread and block until it comes back to `me`.
    fn switch<'a>(&'a self, m: &Machine, me: u32, mut g: MutexGuard<'a, St>) -> MutexGuard<'a, St> {
        match Self::pick(&g) {
            Some(n) => {
                g.running = n;
                if let Some(th) = g.threads.get_mut(&n) {
                    th.state = TS::Running;
                }
                if n >= 0x8000_0000 {
                    m.write_be(OS_CURRENT_THREAD, 4, n as u64);
                    set_gstate(m, n, GS_RUNNING);
                }
            }
            None => g.running = NONE,
        }
        self.cv.notify_all();
        let dead = matches!(g.threads.get(&me).map(|t| t.state), Some(TS::Dead) | None);
        if dead {
            return g;
        }
        while g.running != me || stale(&g, me) {
            if stale(&g, me) {
                drop(g);
                park_stale();
            }
            g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        g
    }

    /// If a higher-priority thread is ready, yield to it (never inside an interrupt handler).
    fn maybe_preempt<'a>(&'a self, c: &Ctx, mut g: MutexGuard<'a, St>) -> MutexGuard<'a, St> {
        if c.in_irq || c.thread < 0x8000_0000 {
            return g;
        }
        let my = g.threads.get(&c.thread).map(|t| t.prio).unwrap_or(i32::MAX);
        let best = Self::pick(&g).and_then(|n| g.threads.get(&n)).map(|t| t.prio);
        if best.is_some_and(|b| b < my) {
            Self::make_ready(&mut g, c.m(), c.thread);
            g = self.switch(c.m(), c.thread, g);
        }
        g
    }

    pub fn wait_turn(&self, me: u32, incarnation: u64) {
        MY_INCARNATION.with(|c| c.set(incarnation));
        let mut g = self.lock();
        while g.running != me || stale(&g, me) {
            if stale(&g, me) {
                drop(g);
                park_stale();
            }
            g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }

    // ---- OS thread API (called from hle.rs) ----

    pub fn create(&self, c: &mut Ctx, t: u32) {
        self.adopt(c);
        let prio = c.m().read_be(t + T_PRIORITY, 4) as i32;
        let mut g = self.lock();
        g.next_incarnation += 1;
        let incarnation = g.next_incarnation;
        g.threads.insert(t, Th { prio, state: TS::Suspended, seq: 0, suspend: 1, started: false, incarnation });
    }

    /// OSResumeThread: returns the previous suspend count.
    pub fn resume(&self, c: &mut Ctx, t: u32) -> i32 {
        self.adopt(c);
        let mut g = self.lock();
        let Some(th) = g.threads.get_mut(&t) else {
            crate::diag::log(format!("OSResumeThread on unknown thread {t:#010x}"));
            return 0;
        };
        let prev = th.suspend;
        th.suspend = (th.suspend - 1).max(0);
        c.m().write_be(t + T_SUSPEND, 4, th.suspend as u64);
        if th.suspend == 0 && th.state == TS::Suspended {
            let spawn = !th.started;
            let incarnation = th.incarnation;
            th.started = true;
            Self::make_ready(&mut g, c.m(), t);
            if spawn {
                spawn_guest_thread(c, t, incarnation);
            }
        }
        drop(self.maybe_preempt(c, g));
        prev
    }

    /// OSSuspendThread: returns the previous suspend count.
    pub fn suspend(&self, c: &mut Ctx, t: u32) -> i32 {
        self.adopt(c);
        let mut g = self.lock();
        let Some(th) = g.threads.get_mut(&t) else { return 0 };
        let prev = th.suspend;
        th.suspend += 1;
        c.m().write_be(t + T_SUSPEND, 4, th.suspend as u64);
        if matches!(th.state, TS::Ready | TS::Running) {
            th.state = TS::Suspended;
            if t == c.thread && !c.in_irq {
                drop(self.switch(c.m(), t, g));
            }
        }
        prev
    }

    pub fn sleep(&self, c: &mut Ctx, queue: u32) {
        self.adopt(c);
        let me = c.thread;
        let mut g = self.lock();
        if c.in_irq || me < 0x8000_0000 {
            // Sleeping outside a real thread (boot/idle/irq): just let others run once.
            drop(g);
            return;
        }
        if let Some(th) = g.threads.get_mut(&me) {
            th.state = TS::Waiting(queue);
        }
        set_gstate(c.m(), me, GS_WAITING);
        g = self.switch(c.m(), me, g);
        drop(g);
    }

    pub fn wakeup(&self, c: &mut Ctx, queue: u32) {
        self.adopt(c);
        let mut g = self.lock();
        let woken: Vec<u32> = g
            .threads
            .iter()
            .filter(|(_, t)| t.state == TS::Waiting(queue))
            .map(|(k, _)| *k)
            .collect();
        for t in woken {
            Self::make_ready(&mut g, c.m(), t);
        }
        drop(self.maybe_preempt(c, g));
    }

    pub fn yield_now(&self, c: &mut Ctx) {
        self.adopt(c);
        if c.in_irq || c.thread < 0x8000_0000 {
            return;
        }
        let mut g = self.lock();
        Self::make_ready(&mut g, c.m(), c.thread);
        drop(self.switch(c.m(), c.thread, g));
    }

    pub fn reschedule(&self, c: &mut Ctx) {
        self.adopt(c);
        let g = self.lock();
        drop(self.maybe_preempt(c, g));
    }

    /// OSExitThread / thread function returned: wake joiners and hand the token on.
    pub fn exit(&self, c: &mut Ctx) {
        let me = c.thread;
        let mut g = self.lock();
        if let Some(th) = g.threads.get_mut(&me) {
            th.state = TS::Dead;
        }
        set_gstate(c.m(), me, GS_MORIBUND);
        let join = me + T_QUEUE_JOIN;
        let woken: Vec<u32> =
            g.threads.iter().filter(|(_, t)| t.state == TS::Waiting(join)).map(|(k, _)| *k).collect();
        for t in woken {
            Self::make_ready(&mut g, c.m(), t);
        }
        drop(self.switch(c.m(), me, g));
    }

    pub fn cancel(&self, c: &mut Ctx, t: u32) {
        if t == c.thread {
            self.exit(c);
            return;
        }
        let mut g = self.lock();
        if let Some(th) = g.threads.get_mut(&t) {
            th.state = TS::Dead;
        }
        set_gstate(c.m(), t, GS_MORIBUND);
    }

    // ---- alarms ----

    pub fn set_alarm(&self, ptr: u32, handler: u32, fire: u64, period: u64) {
        let mut g = self.lock();
        g.alarms.retain(|a| a.ptr != ptr);
        g.alarms.push(Alarm { ptr, handler, fire, period });
        self.cv.notify_all();
    }

    pub fn cancel_alarm(&self, ptr: u32) {
        self.lock().alarms.retain(|a| a.ptr != ptr);
    }

    fn take_due_alarms(&self) -> Vec<(u32, u32)> {
        let now = timebase();
        let mut g = self.lock();
        let mut due = Vec::new();
        g.alarms.retain_mut(|a| {
            if a.fire > now {
                return true;
            }
            due.push((a.ptr, a.handler));
            if a.period > 0 {
                while a.fire <= now {
                    a.fire += a.period;
                }
                true
            } else {
                false
            }
        });
        due
    }

    /// Generate time-driven interrupts; returns how long until the next event.
    fn tick(&self, g: &mut St) -> Duration {
        let now = timebase();
        if now >= g.next_vi {
            raise(IRQ_VI);
            VI_COUNT.fetch_add(1, Relaxed);
            LAST_VI_TB.store(now, Relaxed);
            if let Some(m) = crate::try_machine() {
                m.frames.scanout();
            }
            // Keep a fixed cadence; if we fell more than a period behind, skip whole periods.
            g.next_vi += VI_PERIOD;
            if g.next_vi <= now {
                g.next_vi += (now - g.next_vi) / VI_PERIOD * VI_PERIOD + VI_PERIOD;
            }
        }
        let mut next = g.next_vi;
        let ai = AI_PERIOD.load(Relaxed);
        if ai > 0 {
            let due = AI_NEXT.load(Relaxed);
            if now >= due {
                raise(IRQ_DSP_AI);
                AI_NEXT.store(if now - due > 4 * ai { now + ai } else { due + ai }, Relaxed);
            }
            next = next.min(AI_NEXT.load(Relaxed));
        }
        for a in &g.alarms {
            if a.fire <= now {
                ALARM_DUE.store(true, Relaxed);
                ATTENTION.store(true, Relaxed);
            }
            next = next.min(a.fire);
        }
        Duration::from_nanos(next.saturating_sub(now) * 2000 / 81).clamp(Duration::from_micros(200), Duration::from_millis(16))
    }

    /// Current token holder (for the probe's sampler).
    pub fn running(&self) -> u32 {
        self.lock().running
    }

    pub fn describe(&self) -> String {
        let g = self.lock();
        let mut s = format!("running={:#x} ", g.running);
        for (k, t) in &g.threads {
            s += &format!("[{k:#010x} p{} {:?} s{}] ", t.prio, t.state, t.suspend);
        }
        s
    }
}

fn spawn_guest_thread(c: &Ctx, t: u32, incarnation: u64) {
    let m = c.m();
    let mut nc = c.spawn_like(t);
    // The SDK's OSCreateThread already filled the thread's OSContext (gpr[1]=sp, gpr[3]=arg,
    // srr0=entry); start from there.
    let sp = m.read_be(t + 0x04, 4) as u32;
    let arg = m.read_be(t + 0x0C, 4) as u32;
    let entry = m.read_be(t + 0x198, 4) as u32;
    nc.set_gpr(1, sp);
    nc.set_gpr(3, arg);
    let msr = nc.r(REG_MSR, 4) | MSR_EE;
    nc.wr(REG_MSR, 4, msr);
    std::thread::Builder::new()
        .name(format!("guest-{t:08x}"))
        .stack_size(64 << 20)
        .spawn(move || {
            crate::hostprof::register(t);
            m.sched.wait_turn(t, incarnation);
            crate::diag::log(format!("thread {t:#010x} starts at {entry:#010x} {}", m.name(entry)));
            nc.call(entry as u64);
            m.sched.exit(&mut nc);
        })
        .expect("spawn guest thread");
}

/// Run pending interrupt handlers and due alarms on `c` (which holds the token).
pub fn deliver(c: &mut Ctx) {
    // Clear first: anything raised while handlers run sets it again.
    ATTENTION.store(false, Relaxed);
    let saved = c.regs.to_vec();
    c.in_irq = true;
    let msr = c.r(REG_MSR, 4) & !MSR_EE;
    c.wr(REG_MSR, 4, msr);
    let ctx_ptr = c.m().read_be(OS_CURRENT_THREAD, 4) as u32;

    let bits = PENDING_IRQ.swap(0, AcqRel);
    for irq in 0..32 {
        if bits & (1 << irq) == 0 {
            continue;
        }
        crate::probe::irq_delivered(irq);
        crate::hw::before_interrupt(c, irq);
        let handler = c.m().read_be(INTERRUPT_TABLE + 4 * irq, 4) as u32;
        if handler != 0 {
            c.call_guest(handler, &[irq, ctx_ptr]);
        }
    }
    if ALARM_DUE.swap(false, AcqRel) {
        for (alarm, handler) in c.m().sched.take_due_alarms() {
            if handler != 0 {
                c.call_guest(handler, &[alarm, ctx_ptr]);
            }
        }
    }
    c.regs.copy_from_slice(&saved);
    c.in_irq = false;
    if c.thread >= 0x8000_0000 {
        let g = c.m().sched.lock();
        drop(c.m().sched.maybe_preempt(c, g));
    }
}

#[cfg(windows)]
#[link(name = "winmm")]
unsafe extern "system" {
    fn timeBeginPeriod(period_ms: u32) -> u32;
}

/// Start the timer/idle thread: generates VI and alarm interrupts, and runs handlers itself when
/// no guest thread is runnable.
pub fn start_idle(m: &'static Machine) {
    // The idle thread paces VI (16.7 ms) and AI (17.5 ms at 32 kHz) with timed waits. Windows'
    // default timer tick is 15.6 ms, which makes those events land late and in bursts (frame
    // judder, audio starving between bursts); ask for 1 ms resolution for the process.
    #[cfg(windows)]
    unsafe {
        timeBeginPeriod(1);
    }
    std::thread::Builder::new()
        .name("bmx-idle".into())
        .stack_size(64 << 20)
        .spawn(move || {
            let mut c = m.new_ctx();
            c.thread = IDLE;
            idle_loop(c)
        })
        .expect("spawn idle thread");
    std::thread::Builder::new()
        .name("bmx-timer".into())
        .spawn(move || timer_loop(m))
        .expect("spawn timer thread");
}

/// Timer thread: raises VI / AI / alarm events at their exact times (sleep, then spin the last
/// stretch, since OS sleeps overshoot by up to a millisecond or more). Kept separate from the
/// idle thread, whose interrupt-handler work would otherwise delay retraces.
fn timer_loop(m: &'static Machine) {
    let s = &m.sched;
    loop {
        let wait = {
            let mut g = s.lock();
            s.tick(&mut g)
        };
        s.cv.notify_all();
        let deadline = Instant::now() + wait;
        if wait > Duration::from_micros(1500) {
            std::thread::sleep(wait - Duration::from_micros(1200));
        }
        while Instant::now() < deadline {
            std::thread::yield_now();
        }
    }
}

fn idle_loop(mut c: Ctx) {
    let s = &c.m().sched;
    loop {
        let mut g = s.lock();
        if g.running == NONE && pending() {
            g.running = IDLE;
            drop(g);
            let sda = s.sda.load(Relaxed);
            c.wr(crate::REG_R2, 4, sda >> 32);
            c.wr(crate::REG_R13, 4, sda & 0xFFFF_FFFF);
            c.set_gpr(1, IDLE_SP.load(Relaxed));
            deliver(&mut c);
            let mut g = s.lock();
            if g.running == IDLE {
                match Sched::pick(&g) {
                    Some(n) => {
                        g.running = n;
                        if let Some(th) = g.threads.get_mut(&n) {
                            th.state = TS::Running;
                        }
                        if n >= 0x8000_0000 {
                            c.m().write_be(OS_CURRENT_THREAD, 4, n as u64);
                            set_gstate(c.m(), n, GS_RUNNING);
                        }
                    }
                    None => g.running = NONE,
                }
                s.cv.notify_all();
            }
            continue;
        }
        let _ = s.cv.wait_timeout(g, Duration::from_millis(2)).unwrap_or_else(|e| e.into_inner());
    }
}
