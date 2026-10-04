//! Headless preflight: boot the recompiled game without a window and check that it keeps
//! making progress. Prints a report (also written to `preflight_report.txt`) and exits non-zero
//! on failure.
//!
//!   cargo run -p bmx-recomp --bin preflight -- [seconds]
//!
//! Checks: disc image opens and is GB3E51; no Rust panic, OSPanic or unresolved-call flood;
//! the game produces GX frames and keeps producing them (no stall); the DSP/AI audio path and
//! memory card get exercised. After 20 s it presses Start / A periodically to get past the
//! title and menus.

use gc_runtime::{diag, dsp, hw, pad, sched, FN_ENTRIES, LAST_FN, UNKNOWN_TARGETS};
use std::fmt::Write as _;
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

struct Report {
    text: String,
    failures: Vec<String>,
    warnings: Vec<String>,
}

impl Report {
    fn line(&mut self, s: impl AsRef<str>) {
        println!("{}", s.as_ref());
        self.text += s.as_ref();
        self.text.push('\n');
    }
}

fn main() {
    let secs: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(90);
    let mut rep = Report { text: String::new(), failures: Vec::new(), warnings: Vec::new() };
    rep.line(format!("BMX XXX preflight ({secs} s headless boot)"));

    // ---- static checks ----
    let path = bmx_recomp::disc_path();
    match gc_runtime::disc::Disc::open(&path) {
        Ok(d) => {
            let mut id = [0u8; 6];
            d.read(0, &mut id);
            let id = String::from_utf8_lossy(&id).to_string();
            if id == "GB3E51" {
                rep.line(format!("[ok]   disc image {} ({id})", path.display()));
            } else {
                rep.failures.push(format!("disc image is {id:?}, expected GB3E51"));
            }
        }
        Err(e) => rep.failures.push(format!("cannot open disc image {}: {e} (set BMX_DISC)", path.display())),
    }
    if !rep.failures.is_empty() {
        finish(rep);
    }

    // ---- boot ----
    let mut cpu = bmx_recomp::boot();
    let m = gc_runtime::machine();
    std::thread::Builder::new()
        .name("bmx-cpu".into())
        .stack_size(256 << 20)
        .spawn(move || bmx_recomp::run(&mut cpu))
        .expect("spawn cpu thread");

    let start = Instant::now();
    let mut last_frames = 0;
    let mut last_progress = Instant::now();
    let mut first_frame: Option<f32> = None;
    let mut max_batches = 0usize;
    let mut textured = 0usize;
    let mut tick = 0u64;
    while start.elapsed() < Duration::from_secs(secs) {
        std::thread::sleep(Duration::from_secs(1));
        tick += 1;
        let t = start.elapsed().as_secs_f32();
        // Navigate: hold Start / A briefly every other second after the boot screens.
        if t > 20.0 && std::env::var_os("BMX_NO_INPUT").is_none() {
            let mut ports = m.pad.ports.lock().unwrap();
            ports[0] = pad::PadStatus { connected: true, ..Default::default() };
            if tick % 2 == 0 {
                ports[0].button = if tick % 4 == 0 { pad::BUTTON_START } else { pad::BUTTON_A };
            }
        }
        let frames = m.frames.count();
        if frames != last_frames {
            if first_frame.is_none() {
                first_frame = Some(t);
            }
            last_frames = frames;
            last_progress = Instant::now();
            let f = m.frames.latest();
            max_batches = max_batches.max(f.batches.len());
            textured = textured.max(f.batches.iter().filter(|b| b.tev.textures.iter().any(|t| t.is_some())).count());
        }
        if tick % 10 == 0 {
            let last = LAST_FN.load(Relaxed);
            rep.line(format!(
                "  t={t:>5.0}s frames={frames} fn_entries={} last={:#010x} {} dsp_subframes={} ai_buffers={}",
                FN_ENTRIES.load(Relaxed),
                last,
                m.name(last),
                dsp::SUBFRAMES_RENDERED.load(Relaxed),
                hw::AI_BUFFERS.load(Relaxed),
            ));
        }
        if diag::tail(20000).iter().any(|l| l.starts_with("PANIC") || l.starts_with("OSPanic")) {
            break;
        }
        if first_frame.is_some() && last_progress.elapsed() > Duration::from_secs(15) {
            break;
        }
    }

    // ---- verdict ----
    let log = diag::tail(20000);
    for l in log.iter().filter(|l| l.starts_with("PANIC") || l.starts_with("OSPanic")) {
        rep.failures.push(l.clone());
    }
    match first_frame {
        None => rep.failures.push("no GX frame was completed".into()),
        Some(t) => rep.line(format!("[ok]   first GX frame after {t:.1} s")),
    }
    if first_frame.is_some() && last_progress.elapsed() > Duration::from_secs(15) {
        rep.failures.push(format!("frames stalled at {last_frames} for 15 s"));
    } else if first_frame.is_some() {
        rep.line(format!("[ok]   frames kept coming: {last_frames} total"));
    }
    rep.line(format!("[info] max batches per frame {max_batches}, textured batches {textured}"));
    let sub = dsp::SUBFRAMES_RENDERED.load(Relaxed);
    let ai = hw::AI_BUFFERS.load(Relaxed);
    if sub > 0 && ai > 0 {
        rep.line(format!("[ok]   audio: {sub} DSP sub-frames rendered, {ai} AI buffers output (peak voices {}, channel activations {})",
            dsp::VOICES_MAX.load(Relaxed), dsp::ACTIVE_SEEN.load(Relaxed)));
    } else {
        rep.warnings.push(format!("audio path not exercised (DSP sub-frames {sub}, AI buffers {ai})"));
    }
    let unresolved = UNKNOWN_TARGETS.load(Relaxed);
    if unresolved > 0 {
        rep.warnings.push(format!("{unresolved} calls to addresses with no recompiled function"));
    }
    if log.iter().any(|l| l.starts_with("CARD")) {
        rep.line("[ok]   memory card API used");
    }
    if gc_runtime::probe::enabled() {
        rep.line(gc_runtime::probe::report());
    }
    rep.line(format!("threads: {}", m.sched.describe()));
    rep.line("hot functions (last interval):");
    for (a, k) in diag::hottest(10) {
        rep.line(format!("  {k:>10}  {a:#010x} {}", m.name(a)));
    }
    rep.line("log (first 40 distinct lines):");
    for l in log.iter().take(40) {
        rep.line(format!("  {l}"));
    }
    let _ = sched::VI_COUNT.load(Relaxed);
    finish(rep);
}

fn finish(mut rep: Report) -> ! {
    for w in rep.warnings.clone() {
        rep.line(format!("[warn] {w}"));
    }
    for f in rep.failures.clone() {
        rep.line(format!("[FAIL] {f}"));
    }
    let ok = rep.failures.is_empty();
    rep.line(if ok { "PREFLIGHT PASSED" } else { "PREFLIGHT FAILED" });
    let mut text = String::new();
    let _ = write!(text, "{}", rep.text);
    let _ = std::fs::write("preflight_report.txt", text);
    std::process::exit(if ok { 0 } else { 1 });
}
