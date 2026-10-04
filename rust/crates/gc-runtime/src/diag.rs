//! Diagnostics for testers: hot-function counters, a log, and a stall watchdog that writes
//! `bmx_trace.log` next to the executable's working directory.

use std::collections::BTreeMap;
use std::io::Write;
use std::sync::atomic::Ordering;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

struct LogState {
    lines: Vec<String>,
    intrinsics: BTreeMap<&'static str, u64>,
    /// How often each distinct message was logged; repeats are counted, not printed.
    seen: BTreeMap<String, u64>,
}

fn state() -> &'static Mutex<LogState> {
    static S: OnceLock<Mutex<LogState>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(LogState { lines: Vec::new(), intrinsics: BTreeMap::new(), seen: BTreeMap::new() }))
}

pub fn log(mut msg: String) {
    if msg.starts_with("unknown call target") || msg.starts_with("unmapped") {
        let f = crate::LAST_FN.load(Ordering::Relaxed);
        msg += &format!(" in {:#010x} {}", f, crate::machine().name(f));
    }
    let mut s = state().lock().unwrap();
    let n = s.seen.entry(msg.clone()).or_default();
    *n += 1;
    if *n == 1 && s.lines.len() < 20_000 {
        eprintln!("[bmx] {msg}");
        s.lines.push(msg);
    }
}

pub fn intrinsic(name: &'static str) {
    *state().lock().unwrap().intrinsics.entry(name).or_default() += 1;
}

/// Last `n` log lines (for the on-screen overlay).
pub fn tail(n: usize) -> Vec<String> {
    let s = state().lock().unwrap();
    s.lines.iter().rev().take(n).rev().cloned().collect()
}

/// Top `n` functions by entry count since the last call (counters are reset).
pub fn hottest(n: usize) -> Vec<(u32, u32)> {
    let mut v: Vec<(u32, u32)> = gc_core::hit_counters()
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let k = c.swap(0, Ordering::Relaxed);
            (k > 0).then_some((0x8000_0000 + 4 * i as u32, k))
        })
        .collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    v.truncate(n);
    v
}

/// Background watchdog: every 5 s, append a snapshot (hot functions, log tail, threads) to
/// `bmx_trace.log`. If the game stalls, this file shows where.
pub fn start_watchdog(m: &'static crate::Machine) {
    // `BMX_HOT=n`: count function entries for the "hot functions" list (costs a table update
    // per guest call, so it is off otherwise).
    if std::env::var_os("BMX_HOT").is_some() {
        gc_core::COUNT_HITS.store(true, Ordering::Relaxed);
    }
    std::thread::Builder::new()
        .name("bmx-watchdog".into())
        .spawn(move || {
            let start = Instant::now();
            let mut f = std::fs::File::create("bmx_trace.log").ok();
            let mut prev = [0u64; 11];
            loop {
                std::thread::sleep(Duration::from_secs(5));
                // Stutter probe: per-interval deltas of frames / audio / THP counters.
                let now = [
                    m.frames.count(),
                    crate::hw::AI_BUFFERS.load(Ordering::Relaxed),
                    crate::audio::UNDERRUNS.load(Ordering::Relaxed),
                    crate::audio::DROPPED_FRAMES.load(Ordering::Relaxed),
                    crate::thp::FRAMES.load(Ordering::Relaxed),
                    crate::thp::MICROS.load(Ordering::Relaxed),
                    crate::audio::PUSHED.load(Ordering::Relaxed),
                    crate::audio::SUMSQ.load(Ordering::Relaxed),
                    crate::hle::DVD_READS.load(Ordering::Relaxed),
                    crate::hle::ARQ_REQUESTS.load(Ordering::Relaxed),
                    crate::gx::GPU_BUSY_NS.load(Ordering::Relaxed),
                ];
                let dv: Vec<u64> = now.iter().zip(prev).map(|(a, b)| a - b).collect();
                prev = now;
                let pacing = format!(
                    "pacing: {} frames/5s (worst gap {:.1} ms), {} AI buffers/5s, audio queue {} frames, \
                     underruns {}, dropped {}, THP native frames {} (avg {:.2} ms), audio rms {:.0}, \
                     voices {}, DVD reads {}, ARQ requests {}, GX thread busy {:.0}%\n",
                    dv[0],
                    crate::gx::MAX_FRAME_GAP_US.swap(0, Ordering::Relaxed) as f64 / 1000.0,
                    dv[1],
                    m.audio.queued(),
                    dv[2],
                    dv[3],
                    dv[4],
                    if dv[4] > 0 { dv[5] as f64 / dv[4] as f64 / 1000.0 } else { 0.0 },
                    if dv[6] > 0 { (dv[7] as f64 * 256.0 / (2.0 * dv[6] as f64)).sqrt() } else { 0.0 },
                    crate::dsp::VOICES_LAST.load(Ordering::Relaxed),
                    dv[8],
                    dv[9],
                    dv[10] as f64 / 5e9 * 100.0,
                );
                let mut out = format!(
                    "=== t={:.0}s entries={} calls={} unresolved={} frames={} gx_bytes={}\n",
                    start.elapsed().as_secs_f32(),
                    crate::FN_ENTRIES.load(Ordering::Relaxed),
                    crate::CALLS.load(Ordering::Relaxed),
                    crate::UNKNOWN_TARGETS.load(Ordering::Relaxed),
                    m.frames.count(),
                    crate::hw::GX_BYTES.load(Ordering::Relaxed),
                );
                out += &pacing;
                out += &format!("last fn: {:#010x} {}\n", crate::LAST_FN.load(Ordering::Relaxed),
                    m.name(crate::LAST_FN.load(Ordering::Relaxed)));
                out += "hot functions (last 5 s):\n";
                for (a, k) in hottest(std::env::var("BMX_HOT").ok().and_then(|v| v.parse().ok()).unwrap_or(15)) {
                    out += &format!("  {k:>10}  {a:#010x} {}\n", m.name(a));
                }
                out += &format!("threads: {}\n", m.sched.describe());
                out += &frame_summary(m);
                let s = state().lock().unwrap();
                let repeated: Vec<String> = s.seen.iter().filter(|(_, n)| **n > 1)
                    .map(|(m, n)| format!("  repeated x{n}: {m}
")).collect();
                out += &repeated.concat();
                if !s.intrinsics.is_empty() {
                    out += &format!("intrinsics: {:?}\n", s.intrinsics);
                }
                for l in s.lines.iter().rev().take(10).rev() {
                    out += &format!("  log: {l}\n");
                }
                drop(s);
                if let Some(f) = f.as_mut() {
                    let _ = f.write_all(out.as_bytes());
                    let _ = f.flush();
                }
            }
        })
        .expect("spawn watchdog");
}

/// What the latest GX frame contains, for diagnosing rendering problems from a trace log.
fn frame_summary(m: &crate::Machine) -> String {
    let f = m.frames.latest();
    let mut out = format!(
        "frame: {} batches, {} draws, {} tris, clear {:?}\n",
        f.batches.len(), f.draws, f.triangles(), f.clear
    );
    out += &format!("  proj {:?}\n  viewport {:?} efb {:?}\n", f.proj, f.viewport, f.efb);
    for (at, v, src, size, dst) in &f.copies {
        out += &format!(
            "  copy before batch {at}: bp52 {v:#08x} (xfb {} clear {}) src ({},{}) size {}x{} dest {:#x}\n",
            (v >> 14) & 1, (v >> 11) & 1, src & 0x3FF, (src >> 10) & 0x3FF,
            (size & 0x3FF) + 1, ((size >> 10) & 0x3FF) + 1, (dst & 0xFF_FFFF) << 5
        );
    }
    for (i, b) in f.batches.iter().enumerate().take(16) {
        let n = b.indices.len();
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in b.verts.iter().map(|v| v.pos) {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k] / p[3]);
                hi[k] = hi[k].max(p[k] / p[3]);
            }
        }
        let t = &b.tev;
        out += &format!("  batch {i} proj {:?} viewport {:?}\n", b.proj, b.viewport);
        out += &format!(
            "    first vertex raw {:?} mtx#{} view {:?}\n    mtx {:?}\n    vcd/vat/matidx {:08x?}\n",
            b.dbg_raw, b.dbg_mtx_idx, b.dbg_view, b.dbg_mtx, b.dbg_vtx
        );
        out += &format!(
            "  batch {i}: verts {n} pos x[{:.2},{:.2}] y[{:.2},{:.2}] z[{:.3},{:.3}] tex {:?} phase {:?}\n",
            lo[0], hi[0], lo[1], hi[1], lo[2], hi[2],
            t.textures.map(|x| x.is_some()), t.blend,
        );
        out += &format!(
            "    stages {} alphacmp {:#08x} zmode {:#x} blend {:#x} swap {:#010x}\n",
            t.params[0], t.params[1], t.pipe[0], t.pipe[1], t.params[2]
        );
        for s in 0..(t.params[0] as usize).min(4) {
            let st = t.stages[s];
            out += &format!("    stage{s}: cenv {:#08x} aenv {:#08x} order {:#x} ksel {:#x}\n", st[0], st[1], st[2], st[3]);
        }
        out += &format!("    regs {:?} konst0 {:?}\n", t.regs, t.konst[0]);
        if let Some(v) = b.verts.first() {
            out += &format!("    v0 pos {:?} c0 {:?} c1 {:?} uv0 {:?}\n", v.pos, v.color0, v.color1, v.uv);
        }
    }
    out
}
