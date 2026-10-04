//! Function-level high-level emulation.
//!
//! Every generated function asks `gc_core::hooked` on entry; functions listed in
//! `sheets/hle.csv` (installed at boot via `gc_core::install_hooks`) are routed to [`call`]. Handlers are matched by the
//! function's name from `sheets/functions.csv`, can run the original recompiled body with
//! [`Ctx::call_orig`], and set r3/r4 as the return value. Returning `false` falls through to
//! the original code. Adding an HLE function only needs this file + the sheet, not a rebuild of
//! the generated crates.

/// Probe counters for the watchdog's pacing line (audio streaming health).
pub static DVD_READS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ARQ_REQUESTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PADREADS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
use crate::{Ctx, CtxExt};

/// `BMX_PAD_SCRIPT=600:0,4:1000,60:0,...`: scripted port-0 buttons for automated runs. Each
/// `reads:hexmask` segment holds the mask for that many PADRead calls (one per frame); after the
/// last segment the script releases everything. Segment changes are logged with the frame count.
fn pad_script(n: u32) -> Option<u16> {
    static SCRIPT: std::sync::OnceLock<Vec<(u32, u16)>> = std::sync::OnceLock::new();
    let segs = SCRIPT.get_or_init(|| {
        std::env::var("BMX_PAD_SCRIPT")
            .unwrap_or_default()
            .split(',')
            .filter_map(|seg| {
                let (k, m) = seg.trim().split_once(':')?;
                Some((k.parse().ok()?, u16::from_str_radix(m, 16).ok()?))
            })
            .collect()
    });
    if segs.is_empty() {
        return None;
    }
    let mut start = 0;
    for (i, &(len, mask)) in segs.iter().enumerate() {
        if n < start + len {
            if n == start {
                crate::diag::log(format!("pad script: frame {n} segment {i} mask {mask:#06x}"));
            }
            return Some(mask);
        }
        start += len;
    }
    Some(0)
}

/// The CPU's HID2 (bit 32 set once some thread has published it).
static HID2: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

const DVD_STATE_END: u64 = 0;
const DVD_COMMAND_READ: u64 = 1;

/// The hottest hooks (hundreds of thousands of calls per second in 3D scenes), resolved by
/// address once so they skip the name lookup and the string match below.
#[derive(Clone, Copy)]
enum Hot {
    Disable,
    Enable,
    Restore,
    Mtx(&'static str),
}

fn hot(addr: u32) -> Option<Hot> {
    static T: std::sync::OnceLock<Vec<(u32, Hot)>> = std::sync::OnceLock::new();
    let t = T.get_or_init(|| {
        let m = crate::machine();
        let mut v: Vec<(u32, Hot)> = [
            ("OSDisableInterrupts", Hot::Disable),
            ("OSEnableInterrupts", Hot::Enable),
            ("OSRestoreInterrupts", Hot::Restore),
            ("PSMTXCopy", Hot::Mtx("PSMTXCopy")),
            ("PSMTXConcat", Hot::Mtx("PSMTXConcat")),
            ("PSMTXMultVec", Hot::Mtx("PSMTXMultVec")),
            ("PSVECCrossProduct", Hot::Mtx("PSVECCrossProduct")),
            ("PSVECMag", Hot::Mtx("PSVECMag")),
            ("PSMTXInverse", Hot::Mtx("PSMTXInverse")),
        ]
        .iter()
        .filter_map(|(n, h)| (m.addr_of)(n).map(|a| (a, *h)))
        .collect();
        v.sort_by_key(|e| e.0);
        v
    });
    t.binary_search_by_key(&addr, |e| e.0).ok().map(|i| t[i].1)
}

pub fn call(c: &mut Ctx, addr: u32) -> bool {
    if let Some(h) = hot(addr) {
        let msr = c.r(gc_core::REG_MSR, 4);
        let was = ((msr & gc_core::MSR_EE) != 0) as u32;
        match h {
            Hot::Disable => {
                c.wr(gc_core::REG_MSR, 4, msr & !gc_core::MSR_EE);
                c.set_gpr(3, was);
            }
            // Setting MSR.EE with an interrupt pending takes it right away, as on the CPU.
            Hot::Enable => {
                c.wr(gc_core::REG_MSR, 4, msr | gc_core::MSR_EE);
                c.set_gpr(3, was);
                c.poll();
            }
            Hot::Restore => {
                let new = if c.gpr(3) != 0 { msr | gc_core::MSR_EE } else { msr & !gc_core::MSR_EE };
                c.wr(gc_core::REG_MSR, 4, new);
                c.set_gpr(3, was);
                c.poll();
            }
            Hot::Mtx(name) => {
                crate::psmtx::call(c, name);
            }
        }
        return true;
    }
    let m = c.m();
    let s = &m.sched;
    match m.name(addr) {
        // ---- threads ----
        "OSCreateThread" => {
            let t = c.gpr(3);
            c.call_orig(addr); // fills the OSThread/OSContext; returns TRUE in r3
            s.create(c, t);
        }
        "OSResumeThread" => {
            crate::diag::log(format!("OSResumeThread {:#x} from {:#x} (lr {:#x})", c.gpr(3), c.thread, c.r(0x1020, 4) as u32));
            let prev = s.resume(c, c.gpr(3));
            c.set_gpr(3, prev as u32);
        }
        "OSSuspendThread" => {
            crate::diag::log(format!("OSSuspendThread {:#x} from {:#x} (lr {:#x})", c.gpr(3), c.thread, c.r(0x1020, 4) as u32));
            let prev = s.suspend(c, c.gpr(3));
            c.set_gpr(3, prev as u32);
        }
        "OSSleepThread" => s.sleep(c, c.gpr(3)),
        "OSWakeupThread" => s.wakeup(c, c.gpr(3)),
        "OSYieldThread" => s.yield_now(c),
        "__OSReschedule" => s.reschedule(c),
        "OSExitThread" => {
            let t = c.thread;
            m.write_be(t + 0x2D8, 4, c.gpr(3) as u64);
            s.exit(c);
            park_forever();
        }
        "OSCancelThread" => {
            if c.gpr(3) == c.thread {
                s.exit(c);
                park_forever();
            }
            s.cancel(c, c.gpr(3));
        }
        "SelectThread" | "OSLoadContext" => {
            crate::diag::log(format!("{} reached (unexpected with HLE scheduler)", m.name(addr)));
        }

        // ---- alarms ----
        "OSSetAlarm" => {
            let (alarm, handler) = (c.gpr(3), c.gpr(7));
            let fire = crate::sched::timebase() + arg64(c, 5);
            m.write_be(alarm, 4, handler as u64);
            m.write_be(alarm + 0x08, 8, fire);
            m.write_be(alarm + 0x18, 8, 0);
            s.set_alarm(alarm, handler, fire, 0);
        }
        "OSSetPeriodicAlarm" => {
            let (alarm, handler) = (c.gpr(3), c.gpr(9));
            let (start, period) = (arg64(c, 5), arg64(c, 7).max(1));
            let now = crate::sched::timebase();
            let fire = if start > now { start } else { start + ((now - start) / period + 1) * period };
            m.write_be(alarm, 4, handler as u64);
            m.write_be(alarm + 0x08, 8, fire);
            m.write_be(alarm + 0x18, 8, period);
            m.write_be(alarm + 0x20, 8, start);
            s.set_alarm(alarm, handler, fire, period);
        }
        "OSCancelAlarm" => {
            s.cancel_alarm(c.gpr(3));
            m.write_be(c.gpr(3), 4, 0);
        }

        // ---- DVD (reads go straight to the disc image) ----
        "DVDInit" => {
            if let Some(a) = (m.addr_of)("__DVDFSInit") {
                c.call(a as u64);
            }
        }
        "DVDReadAbsAsyncPrio" => {
            let (block, dst, len, off, cb) = (c.gpr(3), c.gpr(4), c.gpr(5), c.gpr(6), c.gpr(7));
            DVD_READS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            dvd_read(c, off as u64, dst, len);
            for (o, v) in [(0x08, DVD_COMMAND_READ), (0x0C, DVD_STATE_END), (0x10, off as u64),
                           (0x14, len as u64), (0x18, dst as u64), (0x1C, len as u64),
                           (0x20, len as u64), (0x28, cb as u64)] {
                m.write_be(block + o, 4, v);
            }
            if cb != 0 {
                c.call_guest(cb, &[len, block]);
            }
            c.set_gpr(3, 1);
        }
        "DVDReadDiskID" => {
            let (block, dst, cb) = (c.gpr(3), c.gpr(4), c.gpr(5));
            dvd_read(c, 0, dst, 0x20);
            m.write_be(block + 0x0C, 4, DVD_STATE_END);
            if cb != 0 {
                c.call_guest(cb, &[0x20, block]);
            }
            c.set_gpr(3, 1);
        }
        "DVDInquiryAsync" => {
            let (block, info, cb) = (c.gpr(3), c.gpr(4), c.gpr(5));
            for i in 0..8 {
                m.write_be(info + 4 * i, 4, 0);
            }
            m.write_be(block + 0x0C, 4, DVD_STATE_END);
            if cb != 0 {
                c.call_guest(cb, &[0x20, block]);
            }
            c.set_gpr(3, 1);
        }
        "DVDCancelAsync" => {
            let (block, cb) = (c.gpr(3), c.gpr(4));
            m.write_be(block + 0x0C, 4, DVD_STATE_END);
            if cb != 0 {
                c.call_guest(cb, &[0, block]);
            }
            c.set_gpr(3, 1);
        }
        "DVDCancel" | "DVDGetDriveStatus" | "DVDReset" | "DVDResume" => c.set_gpr(3, 0),
        "DVDCheckDisk" => c.set_gpr(3, 1),
        "DVDGetCommandBlockStatus" => {
            let st = m.read_be(c.gpr(3) + 0x0C, 4) as u32;
            c.set_gpr(3, st);
        }

        // ---- controllers ----
        "PADInit" | "PADReset" | "PADRecalibrate" => c.set_gpr(3, 1),
        "PADControlMotor" => {}
        "PADSetSamplingCallback" => c.set_gpr(3, 0),
        "PADRead" => {
            let dst = c.gpr(3);
            let mut ports = *m.pad.ports.lock().unwrap();
            if let Ok(v) = std::env::var("BMX_FAKE_PAD") {
                // Diagnostic: hold a button mask on port 0 (hex), e.g. 1000 = Start.
                let n = PADREADS.load(std::sync::atomic::Ordering::Relaxed);
                if (n / 30) % 2 == 1 {
                    ports[0].button = u16::from_str_radix(&v, 16).unwrap_or(0);
                }
                ports[0].connected = true;
            }
            let n = PADREADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if let Some(mask) = pad_script(n) {
                ports[0].button = mask;
                ports[0].connected = true;
            }
            for (i, p) in ports.iter().enumerate() {
                for (j, b) in p.to_bytes().iter().enumerate() {
                    m.write_be(dst + 12 * i as u32 + j as u32, 1, *b as u64);
                }
            }
            c.set_gpr(3, 0);
        }

        "SIProbe" | "SIGetType" => {
            // SI_GC_CONTROLLER for connected ports, SI_ERROR_NO_RESPONSE otherwise.
            let ch = (c.gpr(3) & 3) as usize;
            let connected = m.pad.ports.lock().unwrap()[ch].connected
                || ch == 0 && (std::env::var_os("BMX_FAKE_PAD").is_some() || std::env::var_os("BMX_PAD_SCRIPT").is_some());
            c.set_gpr(3, if connected { 0x0900_0000 } else { 0x08 });
        }

        // ---- GX ----
        "__GXFifoLink" => {
            m.hw().gx_link = c.gpr(3) & 0xFF != 0;
            c.call_orig(addr);
        }

        // ---- audio: the init-ucode handshake and AI sample-rate calibration are skipped; the
        // DSP itself is emulated at the mailbox level (dsp.rs) ----
        "__AI_SRC_INIT" | "__OSInitAudioSystem" | "__OSStopAudioSystem" => {}
        // ---- diagnostics / system ----
        // ---- MTX library (native; top of the profile in 3D scenes) ----
        "PSMTXCopy" | "PSMTXConcat" | "PSMTXMultVec" | "PSVECCrossProduct" | "PSVECMag" | "PSMTXInverse" => {
            crate::psmtx::call(c, m.name(addr));
        }
        // ---- MSR.EE ----
        // Ghidra's mtmsr p-code keeps the old EE (and ME) bits, so the recompiled versions of
        // these never change the interrupt-enable state: every OS critical section could be
        // interrupted (the ARAM queue's chunk bookkeeping got corrupted that way).
        "OSDisableInterrupts" => {
            let msr = c.r(gc_core::REG_MSR, 4);
            c.wr(gc_core::REG_MSR, 4, msr & !gc_core::MSR_EE);
            c.set_gpr(3, ((msr & gc_core::MSR_EE) != 0) as u32);
        }
        "OSEnableInterrupts" => {
            let msr = c.r(gc_core::REG_MSR, 4);
            c.wr(gc_core::REG_MSR, 4, msr | gc_core::MSR_EE);
            c.set_gpr(3, ((msr & gc_core::MSR_EE) != 0) as u32);
            c.poll();
        }
        "OSRestoreInterrupts" => {
            let msr = c.r(gc_core::REG_MSR, 4);
            let new = if c.gpr(3) != 0 { msr | gc_core::MSR_EE } else { msr & !gc_core::MSR_EE };
            c.wr(gc_core::REG_MSR, 4, new);
            c.set_gpr(3, ((msr & gc_core::MSR_EE) != 0) as u32);
            c.poll();
        }
        "PPCMtmsr" => {
            c.wr(gc_core::REG_MSR, 4, c.gpr(3) as u64);
            c.poll();
        }
        // HID2 (locked cache enable, paired-single enable, ...) belongs to the CPU, not to a
        // thread, but every host thread has its own register file. The few functions that
        // touch it see the shared copy on entry and publish it on exit.
        "PPCMfhid2" | "PPCMthid2" | "__OSFPRInit" | "__LCEnable" | "LCDisable" | "LCQueueWait"
        | "__OSLoadFPUContext" | "__OSSaveFPUContext" | "OSFillFPUContext" | "FUN_801454a0"
        | "FUN_80145658" => {
            const REG_HID2: usize = 0x1E60;
            let g = HID2.load(std::sync::atomic::Ordering::SeqCst);
            if g >> 32 != 0 {
                c.wr(REG_HID2, 4, g & 0xFFFF_FFFF);
            }
            c.call_orig(addr);
            HID2.store(1 << 32 | c.r(REG_HID2, 4), std::sync::atomic::Ordering::SeqCst);
        }
        "OSReport" => {
            let msg = format_guest(c, c.gpr(3), 4);
            crate::diag::log(format!("OSReport: {}", msg.trim_end()));
        }
        "OSPanic" => {
            let file = read_cstr(c, c.gpr(3));
            let line = c.gpr(4);
            let msg = format_guest(c, c.gpr(5), 6);
            crate::diag::log(format!("OSPanic {file}:{line}: {}", msg.trim_end()));
            park_forever();
        }
        "OSResetSystem" | "__OSDoHotReset" => {
            crate::diag::log(format!("{} called (ignored)", m.name(addr)));
        }
        "PPCHalt" => {
            crate::diag::log("PPCHalt".into());
            park_forever();
        }
        // ---- "Unlock all" stays out of the memory card (cheats.rs) ----
        "bmxSaveSerialize" => crate::cheats::around_save(|| c.call_orig(addr)),
        "bmxSaveDeserialize" => crate::cheats::around_load(|| c.call_orig(addr)),
        name if name.starts_with("CARD") => match crate::memcard::call(c, name) {
            Some(r) => {
                crate::diag::log(format!("{name} -> {r}"));
                c.set_gpr(3, r as u32);
            }
            None => return false,
        },
        _ => return false,
    }
    true
}

fn park_forever() -> ! {
    loop {
        std::thread::park();
    }
}

/// 64-bit argument in a register pair (PPC EABI: rN = high word, rN+1 = low word).
fn arg64(c: &Ctx, n: usize) -> u64 {
    ((c.gpr(n) as u64) << 32) | c.gpr(n + 1) as u64
}

fn dvd_read(c: &mut Ctx, off: u64, dst: u32, len: u32) {
    let m = c.m();
    let Some(o) = crate::Machine::ram_off(dst) else { return };
    let len = (len as usize).min(crate::RAM_SIZE - o);
    match &m.disc {
        Some(d) => d.read(off, &mut m.ram()[o..o + len]),
        None => crate::diag::log(format!("DVD read {off:#x}+{len:#x} but no disc image is loaded")),
    }
}

fn read_cstr(c: &Ctx, addr: u32) -> String {
    let mut out = Vec::new();
    for i in 0..1024 {
        let b = c.m().read_be(addr + i, 1) as u8;
        if b == 0 {
            break;
        }
        out.push(b);
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Minimal printf for OSReport/OSPanic: integer, char and string conversions.
fn format_guest(c: &Ctx, fmt_addr: u32, first_arg: usize) -> String {
    let fmt = read_cstr(c, fmt_addr);
    let mut out = String::new();
    let mut arg = first_arg;
    let mut next = |c: &Ctx| {
        let v = if arg <= 10 { c.gpr(arg) } else { 0 };
        arg += 1;
        v
    };
    let mut chars = fmt.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '%' {
            out.push(ch);
            continue;
        }
        let mut spec = String::new();
        while let Some(&n) = chars.peek() {
            chars.next();
            if n.is_ascii_alphabetic() && !matches!(n, 'l' | 'h') || n == '%' {
                spec.push(n);
                break;
            }
            spec.push(n);
        }
        match spec.chars().last() {
            Some('d') | Some('i') => out += &(next(c) as i32).to_string(),
            Some('u') => out += &next(c).to_string(),
            Some('x') | Some('p') => out += &format!("{:x}", next(c)),
            Some('X') => out += &format!("{:X}", next(c)),
            Some('c') => out.push(next(c) as u8 as char),
            Some('s') => {
                let a = next(c);
                out += &read_cstr(c, a);
            }
            Some('%') => out.push('%'),
            _ => out += &format!("%{spec}"),
        }
    }
    out
}

