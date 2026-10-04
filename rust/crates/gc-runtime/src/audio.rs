//! Audio output path: AI DMA buffers (interleaved stereo s16, big-endian, as the game's DAC
//! mixer writes them) are copied here in real time and drained by the host audio stream.
//!
//! The game delivers audio in bursts (one AI buffer, ~17.5 ms, per AI interrupt, and interrupt
//! delivery itself is jittery), while the host device pulls at a steady rate. A small jitter
//! buffer absorbs that: playback starts once `TARGET_MS` is queued, an underrun outputs silence
//! and re-primes instead of crackling sample by sample, and the consumer nudges its rate
//! (at most ±0.5 %) to keep the queue near the target rather than dropping audio on overflow.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

/// Queue depth to (re)start playback at, and the level the rate control steers towards.
const TARGET_MS: usize = 60;
/// Hard cap before old audio is dropped (only reached if the host stops pulling).
const MAX_MS: usize = 500;

/// Diagnostics for the trace: underruns (queue ran dry while playing) and overflow drops.
pub static UNDERRUNS: AtomicU64 = AtomicU64::new(0);
pub static DROPPED_FRAMES: AtomicU64 = AtomicU64::new(0);
/// Output level probe: frames pushed and their summed squared amplitude (both channels, >> 8),
/// so the watchdog can report the RMS level per window (silence = music stopped).
pub static PUSHED: AtomicU64 = AtomicU64::new(0);
pub static SUMSQ: AtomicU64 = AtomicU64::new(0);

pub struct AudioShared {
    samples: Mutex<VecDeque<[i16; 2]>>,
    /// Sample rate of what is queued (32000 or 48000).
    rate: AtomicU32,
    /// Playing (false while (re)priming the jitter buffer).
    primed: AtomicBool,
}

impl Default for AudioShared {
    fn default() -> Self {
        AudioShared { samples: Mutex::new(VecDeque::new()), rate: AtomicU32::new(32000), primed: AtomicBool::new(false) }
    }
}

impl AudioShared {
    pub fn rate(&self) -> u32 {
        self.rate.load(Relaxed)
    }

    fn frames(&self, ms: usize) -> usize {
        self.rate() as usize * ms / 1000
    }

    pub fn push_be(&self, bytes: &[u8], rate: u32) {
        self.rate.store(rate, Relaxed);
        let max = self.frames(MAX_MS);
        let mut q = self.samples.lock().unwrap();
        let mut sq = 0u64;
        for f in bytes.chunks_exact(4) {
            let s = [i16::from_be_bytes([f[0], f[1]]), i16::from_be_bytes([f[2], f[3]])];
            sq += ((s[0] as i64 * s[0] as i64 + s[1] as i64 * s[1] as i64) >> 8) as u64;
            q.push_back(s);
        }
        PUSHED.fetch_add((bytes.len() / 4) as u64, Relaxed);
        SUMSQ.fetch_add(sq, Relaxed);
        if q.len() > max {
            let n = q.len() - max;
            DROPPED_FRAMES.fetch_add(n as u64, Relaxed);
            q.drain(..n);
        }
    }

    /// Frames currently queued.
    pub fn queued(&self) -> usize {
        self.samples.lock().unwrap().len()
    }

    /// Resampling correction for the consumer: >1 drains faster when the queue is above the
    /// target, <1 slower when below (bounded to ±0.5 %, inaudible).
    pub fn rate_trim(&self) -> f64 {
        let target = self.frames(TARGET_MS).max(1) as f64;
        let err = (self.queued() as f64 - target) / target;
        1.0 + (err * 0.005).clamp(-0.005, 0.005)
    }

    /// Pop up to `n` frames; while priming or after an underrun, outputs silence.
    pub fn pop(&self, n: usize, out: &mut Vec<[i16; 2]>) {
        let target = self.frames(TARGET_MS);
        let mut q = self.samples.lock().unwrap();
        if !self.primed.load(Relaxed) {
            if q.len() < target {
                out.extend(std::iter::repeat_n([0, 0], n));
                return;
            }
            self.primed.store(true, Relaxed);
        }
        for i in 0..n {
            match q.pop_front() {
                Some(s) => out.push(s),
                None => {
                    UNDERRUNS.fetch_add(1, Relaxed);
                    self.primed.store(false, Relaxed);
                    out.extend(std::iter::repeat_n([0, 0], n - i));
                    return;
                }
            }
        }
    }
}
