//! Performance probe: frame pacing ("choppiness"), interrupt latency, a sampling profiler and
//! host-side costs. Enabled with `BMX_PROBE=1` (or by the preflight/app probe modes); each
//! interval report is appended to `bmx_probe.txt`, and `report()` returns a summary.
//!
//! * Guest frames: interval between finished GX frames (fps, p50/p99/max, hitches > 1.5/2/3
//!   VI periods). Choppiness = stdev of the interval.
//! * Host frames: the same for the renderer (`host_frame`), plus how many guest frames it showed.
//! * IRQ latency: raise → handler delivery, per interrupt line (VI / AI / DSP / ARAM / PE).
//! * Profiler: a 2 kHz sampler reads the last-entered guest function and the token holder, so
//!   the report shows where guest time goes per function and per thread.
//! * Host costs: time inside the GX decoder and texture decode on the guest's thread.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

pub static ENABLED: AtomicBool = AtomicBool::new(false);
/// Host nanoseconds spent decoding GX commands / textures (guest-thread cost).
pub static GX_NS: AtomicU64 = AtomicU64::new(0);
pub static TEX_NS: AtomicU64 = AtomicU64::new(0);
/// Breakdown of GX_NS: primitives (incl. material), material/texture lookup, frame end.
pub static GX_DRAW_NS: AtomicU64 = AtomicU64::new(0);
pub static GX_MAT_NS: AtomicU64 = AtomicU64::new(0);
pub static GX_FRAME_NS: AtomicU64 = AtomicU64::new(0);
pub static GX_VTX_NS: AtomicU64 = AtomicU64::new(0);
pub static GX_EMIT_NS: AtomicU64 = AtomicU64::new(0);
pub static GX_VERTS: AtomicU64 = AtomicU64::new(0);
pub static GX_DRAWS: AtomicU64 = AtomicU64::new(0);
/// Longest single GX FIFO push (ns) this interval.
pub static GX_MAX_PUSH_NS: AtomicU64 = AtomicU64::new(0);

const VI_US: f64 = 1_000_000.0 / 59.94;

#[derive(Default)]
struct Series {
    last: Option<Instant>,
    gaps_us: Vec<u32>,
}

impl Series {
    fn tick(&mut self) {
        let now = Instant::now();
        if let Some(p) = self.last.replace(now) {
            self.gaps_us.push(now.duration_since(p).as_micros().min(u32::MAX as u128) as u32);
        }
    }
}

#[derive(Default)]
struct St {
    guest: Series,
    host: Series,
    host_new_frames: u64,
    irq_lat: HashMap<u32, (u64, u64, u64)>, // irq -> (count, sum µs, max µs)
    samples_fn: HashMap<u32, u64>,
    samples_thread: HashMap<u32, u64>,
    samples: u64,
    /// Longest stall this interval: (µs, last function, thread).
    stall: (u64, u32, u32),
    // totals over the whole run
    all_guest: Vec<u32>,
    all_host: Vec<u32>,
}

fn st() -> &'static Mutex<St> {
    static S: OnceLock<Mutex<St>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(St::default()))
}

/// Raise timestamps per IRQ line (µs since probe start; 0 = none pending).
static RAISED: [AtomicU64; 32] = [const { AtomicU64::new(0) }; 32];

fn t0() -> Instant {
    static T: OnceLock<Instant> = OnceLock::new();
    *T.get_or_init(Instant::now)
}

fn now_us() -> u64 {
    t0().elapsed().as_micros() as u64 + 1
}

pub fn enabled() -> bool {
    ENABLED.load(Relaxed)
}

/// Turn the probe on (idempotent) and start its sampler/reporter.
pub fn start(m: &'static crate::Machine) {
    if ENABLED.swap(true, Relaxed) {
        return;
    }
    t0();
    let _ = std::fs::write("bmx_probe.txt", "");
    std::thread::Builder::new()
        .name("bmx-probe-sampler".into())
        .spawn(move || {
            let (mut last_entries, mut stall_since) = (0u64, Instant::now());
            loop {
            std::thread::sleep(Duration::from_micros(500));
            let f = crate::LAST_FN.load(Relaxed);
            let t = m.sched.running();
            // Stall detector: a thread holds the token but enters no guest function (long host
            // call such as a big DVD read, or a tight loop without calls).
            let entries = crate::FN_ENTRIES.load(Relaxed);
            if entries != last_entries || t == crate::sched::NONE {
                last_entries = entries;
                stall_since = Instant::now();
            } else {
                let us = stall_since.elapsed().as_micros() as u64;
                let mut s = st().lock().unwrap();
                if us > s.stall.0 {
                    s.stall = (us, f, t);
                }
            }
            crate::hostprof::sample(t);
            let mut s = st().lock().unwrap();
            s.samples += 1;
            *s.samples_fn.entry(f).or_default() += 1;
            *s.samples_thread.entry(t).or_default() += 1;
            }
        })
        .expect("probe sampler");
    std::thread::Builder::new()
        .name("bmx-probe-report".into())
        .spawn(move || {
            let mut k = 0;
            loop {
                std::thread::sleep(Duration::from_secs(5));
                k += 5;
                let text = interval(m, k);
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open("bmx_probe.txt") {
                    let _ = f.write_all(text.as_bytes());
                }
            }
        })
        .expect("probe reporter");
}

pub fn from_env(m: &'static crate::Machine) {
    if std::env::var("BMX_PROBE").is_ok_and(|v| v != "0") {
        start(m);
    }
}

/// A guest frame finished (called from the GX frame publish).
pub fn guest_frame() {
    if enabled() {
        st().lock().unwrap().guest.tick();
    }
}

/// The host renderer finished a frame; `new_guest` = it showed a new guest frame.
pub fn host_frame(new_guest: bool) {
    if enabled() {
        let mut s = st().lock().unwrap();
        s.host.tick();
        s.host_new_frames += new_guest as u64;
    }
}

pub fn irq_raised(irq: u32) {
    if enabled() {
        let _ = RAISED[irq as usize].compare_exchange(0, now_us(), Relaxed, Relaxed);
    }
}

pub fn irq_delivered(irq: u32) {
    if enabled() {
        let r = RAISED[irq as usize].swap(0, Relaxed);
        if r != 0 {
            let lat = now_us().saturating_sub(r);
            let mut s = st().lock().unwrap();
            let e = s.irq_lat.entry(irq).or_default();
            e.0 += 1;
            e.1 += lat;
            e.2 = e.2.max(lat);
        }
    }
}

/// Timer for host-side cost accumulators.
pub struct Timer(Option<Instant>, &'static AtomicU64);
impl Timer {
    pub fn new(acc: &'static AtomicU64) -> Timer {
        Timer(enabled().then(Instant::now), acc)
    }
}
impl Drop for Timer {
    fn drop(&mut self) {
        if let Some(t) = self.0 {
            let ns = t.elapsed().as_nanos() as u64;
            self.1.fetch_add(ns, Relaxed);
            if std::ptr::eq(self.1, &GX_NS) {
                GX_MAX_PUSH_NS.fetch_max(ns, Relaxed);
            }
        }
    }
}

pub struct Stats {
    pub n: usize,
    pub fps: f64,
    pub mean_ms: f64,
    pub stdev_ms: f64,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub hitch15: usize,
    pub hitch2: usize,
    pub hitch3: usize,
}

pub fn stats(gaps: &[u32]) -> Option<Stats> {
    if gaps.is_empty() {
        return None;
    }
    let mut v: Vec<u32> = gaps.to_vec();
    v.sort_unstable();
    let n = v.len();
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n as f64;
    let pct = |p: f64| v[((n as f64 - 1.0) * p).round() as usize] as f64 / 1000.0;
    let over = |k: f64| v.iter().filter(|&&x| x as f64 > VI_US * k).count();
    Some(Stats {
        n,
        fps: 1_000_000.0 / mean,
        mean_ms: mean / 1000.0,
        stdev_ms: var.sqrt() / 1000.0,
        p50_ms: pct(0.5),
        p99_ms: pct(0.99),
        max_ms: *v.last().unwrap() as f64 / 1000.0,
        hitch15: over(1.5),
        hitch2: over(2.0),
        hitch3: over(3.0),
    })
}

fn fmt_stats(name: &str, s: Option<Stats>) -> String {
    match s {
        None => format!("{name}: no frames\n"),
        Some(s) => format!(
            "{name}: {:.1} fps over {} frames | interval mean {:.2} ms, stdev {:.2} ms (choppiness), p50 {:.2}, p99 {:.2}, max {:.1} ms | hitches >1.5 VI {}, >2 VI {}, >3 VI {}\n",
            s.fps, s.n, s.mean_ms, s.stdev_ms, s.p50_ms, s.p99_ms, s.max_ms, s.hitch15, s.hitch2, s.hitch3
        ),
    }
}

fn irq_name(i: u32) -> String {
    match i {
        crate::sched::IRQ_VI => "VI".into(),
        crate::sched::IRQ_DSP_AI => "AI".into(),
        crate::sched::IRQ_DSP_DSP => "DSP".into(),
        crate::sched::IRQ_DSP_ARAM => "ARAM".into(),
        _ => format!("irq{i}"),
    }
}

fn interval(m: &'static crate::Machine, t: u64) -> String {
    let mut s = st().lock().unwrap();
    let g = std::mem::take(&mut s.guest.gaps_us);
    let h = std::mem::take(&mut s.host.gaps_us);
    s.all_guest.extend_from_slice(&g);
    s.all_host.extend_from_slice(&h);
    let mut out = format!("=== probe t={t}s ===\n");
    out += &fmt_stats("guest", stats(&g));
    if !h.is_empty() {
        out += &fmt_stats("host ", stats(&h));
        out += &format!("host showed {} new guest frames\n", std::mem::take(&mut s.host_new_frames));
    }
    let mut irqs: Vec<_> = s.irq_lat.drain().collect();
    irqs.sort();
    out += "irq latency:";
    for (i, (n, sum, max)) in irqs {
        out += &format!(" {} n={} avg {:.2} ms max {:.1} ms;", irq_name(i), n, sum as f64 / n as f64 / 1000.0, max as f64 / 1000.0);
    }
    out += "\n";
    let gx = GX_NS.swap(0, Relaxed) as f64 / 1e6;
    let tex = TEX_NS.swap(0, Relaxed) as f64 / 1e6;
    out += &format!(
        "GPU thread: GX decode {:.0} ms/5s ({:.1}%), texture decode (render side) {:.0} ms/5s; THP native {} frames; audio underruns {} queue {}\n",
        gx, gx / 50.0, tex, crate::thp::FRAMES.load(Relaxed), crate::audio::UNDERRUNS.load(Relaxed), m.audio.queued()
    );
    out += &format!(
        "  GX breakdown ms/5s: draw {:.0} (material {:.0}, vertices {:.0}, emit {:.0}; {} draws, {} verts), frame end {:.0}; longest single FIFO push {:.1} ms
",
        GX_DRAW_NS.swap(0, Relaxed) as f64 / 1e6,
        GX_MAT_NS.swap(0, Relaxed) as f64 / 1e6,
        GX_VTX_NS.swap(0, Relaxed) as f64 / 1e6,
        GX_EMIT_NS.swap(0, Relaxed) as f64 / 1e6,
        GX_DRAWS.swap(0, Relaxed),
        GX_VERTS.swap(0, Relaxed),
        GX_FRAME_NS.swap(0, Relaxed) as f64 / 1e6,
        GX_MAX_PUSH_NS.swap(0, Relaxed) as f64 / 1e6
    );
    let total = s.samples.max(1) as f64;
    let mut th: Vec<_> = s.samples_thread.drain().collect();
    th.sort_by(|a, b| b.1.cmp(&a.1));
    let misses = gc_core::BLOCK_MISSES.load(Relaxed);
    if misses > 0 {
        out += &format!("WARNING: {misses} mid-block entries (computed jumps to non-leaders; see bmx_trace.log)
");
    }
    let (us, sf, stt) = std::mem::take(&mut s.stall);
    if us > 0 {
        out += &format!("longest stall: {:.1} ms in {sf:#010x} {} on {}
", us as f64 / 1000.0, m.name(sf), thread_name(stt));
    }
    // Interval histogram (guest frames): shows the cadence pattern behind the stdev.
    let mut hist = [0usize; 6];
    for &x in &g {
        let ms = x as f64 / 1000.0;
        hist[if ms < 8.0 { 0 } else if ms < 14.0 { 1 } else if ms < 20.0 { 2 } else if ms < 26.0 { 3 } else if ms < 40.0 { 4 } else { 5 }] += 1;
    }
    out += &format!("guest intervals <8ms {} | 8-14 {} | 14-20 {} | 20-26 {} | 26-40 {} | >40 {}
", hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]);
    out += "cpu by thread:";
    for (t, n) in th.iter().take(8) {
        out += &format!(" {} {:.1}%;", thread_name(*t), *n as f64 * 100.0 / total);
    }
    out += "\n";
    let mut f: Vec<_> = s.samples_fn.drain().collect();
    f.sort_by(|a, b| b.1.cmp(&a.1));
    out += "profile (last-entered guest function, % of samples):\n";
    for (a, n) in f.iter().take(25) {
        out += &format!("  {:5.1}%  {a:#010x} {}\n", *n as f64 * 100.0 / total, m.name(*a));
    }
    s.samples = 0;
    drop(s);
    out += &crate::hostprof::report();
    out
}

fn thread_name(t: u32) -> String {
    match t {
        0 => "none(idle-wait)".into(),
        1 => "idle(irq)".into(),
        2 => "boot".into(),
        _ => format!("{t:#010x}"),
    }
}

/// Whole-run summary (guest and host pacing).
pub fn report() -> String {
    let s = st().lock().unwrap();
    let mut g = s.all_guest.clone();
    g.extend_from_slice(&s.guest.gaps_us);
    let mut h = s.all_host.clone();
    h.extend_from_slice(&s.host.gaps_us);
    let mut out = fmt_stats("guest (whole run)", stats(&g));
    if !h.is_empty() {
        out += &fmt_stats("host  (whole run)", stats(&h));
    }
    out
}
