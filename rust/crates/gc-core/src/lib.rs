//! The stable ABI between generated code and the console runtime.
//!
//! The 48 generated `bmx-part-NN` crates depend only on this crate. It holds the CPU context,
//! p-code operator semantics, the inline MEM1 fast path and a table of function pointers
//! ([`Host`]) through which everything else (MMIO, interrupts, HLE, dispatch) reaches
//! `gc-runtime`. Changing the runtime therefore never rebuilds the generated crates; only
//! changing this file does.

pub mod ops;
pub mod ps;

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;

/// 24 MiB of MEM1.
pub const RAM_SIZE: usize = 0x0180_0000;
/// Register-space offsets (from sheets/registers.csv).
pub const REG_R2: usize = 0x08;
pub const REG_R13: usize = 0x34;
pub const REG_MSR: usize = 0x700;
pub const REG_GQR0: usize = 0x1E40;
pub const REG_DMAU: usize = 0x1E68;
pub const REG_DMAL: usize = 0x1E6C;
pub const MSR_EE: u64 = 0x8000;

/// Runtime services, installed once by `gc-runtime`.
pub struct Host {
    /// Loads/stores outside MEM1 (MMIO, locked cache, EFB...).
    pub mmio_read: fn(&mut Ctx, u32, usize) -> u64,
    pub mmio_write: fn(&mut Ctx, u32, usize, u64),
    /// Run pending interrupts/alarms on this context.
    pub deliver: fn(&mut Ctx),
    /// HLE handler for a hooked function; true if it handled the call.
    pub hle: fn(&mut Ctx, u32) -> bool,
    /// Run the recompiled function containing an address; false if none.
    pub dispatch: fn(&mut Ctx, u32) -> bool,
    /// Unhandled CALLOTHER intrinsics (counted).
    pub intrinsic: fn(&mut Ctx, &'static str, &[u64]),
    pub log: fn(String),
    /// Guest time base (40.5 MHz).
    pub timebase: fn() -> u64,
}

pub static FN_ENTRIES: AtomicU64 = AtomicU64::new(0);
pub static CALLS: AtomicU64 = AtomicU64::new(0);
/// Entries into a function at a non-leader address (see `Ctx::block_miss`).
pub static BLOCK_MISSES: AtomicU64 = AtomicU64::new(0);
pub static UNKNOWN_TARGETS: AtomicU64 = AtomicU64::new(0);
pub static LAST_FN: AtomicU32 = AtomicU32::new(0);
/// Set by the runtime whenever an interrupt or alarm is waiting to be delivered.
pub static ATTENTION: AtomicBool = AtomicBool::new(false);
/// Per-function entry counting ([`hit_counters`]) for the hot-function diagnostics. Off by
/// default: it touches a 4 MiB table on every call.
pub static COUNT_HITS: AtomicBool = AtomicBool::new(false);

/// Size of the register space (Ghidra's register address space for the Gekko, incl. PS1).
pub const REG_BYTES: usize = 0x10010;

/// Bump a statistics counter. Guest code runs on one host thread at a time (scheduler token),
/// so a plain load + store is enough; a locked read-modify-write on every call is not free.
#[inline(always)]
fn bump(c: &AtomicU64) {
    c.store(c.load(Relaxed).wrapping_add(1), Relaxed);
}

const TEXT_BASE: u32 = 0x8000_0000;
const TEXT_WORDS: usize = 0x0040_0000 / 4;

/// Per-function entry counters (indexed by address / 4) for diagnostics.
pub fn hit_counters() -> &'static [AtomicU32] {
    static C: OnceLock<Vec<AtomicU32>> = OnceLock::new();
    C.get_or_init(|| (0..TEXT_WORDS).map(|_| AtomicU32::new(0)).collect())
}

#[cold]
fn count_hit(addr: u32) {
    if let Some(c) = hit_counters().get((addr.wrapping_sub(TEXT_BASE) / 4) as usize) {
        c.store(c.load(Relaxed).wrapping_add(1), Relaxed);
    }
}

/// One bit per code word: functions routed to `Host::hle`.
static HOOKED: OnceLock<Vec<u64>> = OnceLock::new();

pub fn install_hooks(addrs: &[u32]) {
    let mut bits = vec![0u64; TEXT_WORDS / 64];
    for a in addrs {
        let i = (a.wrapping_sub(TEXT_BASE) / 4) as usize;
        if let Some(w) = bits.get_mut(i / 64) {
            *w |= 1 << (i % 64);
        }
    }
    let _ = HOOKED.set(bits);
}

#[inline]
pub fn hooked(addr: u32) -> bool {
    let i = (addr.wrapping_sub(TEXT_BASE) / 4) as usize;
    HOOKED.get().and_then(|b| b.get(i / 64)).is_some_and(|w| w & (1 << (i % 64)) != 0)
}

#[inline]
pub fn get_be(buf: &[u8], off: usize, size: usize) -> u64 {
    match size {
        1 => buf[off] as u64,
        2 => u16::from_be_bytes([buf[off], buf[off + 1]]) as u64,
        4 => u32::from_be_bytes(buf[off..off + 4].try_into().unwrap()) as u64,
        8 => u64::from_be_bytes(buf[off..off + 8].try_into().unwrap()),
        _ => buf[off..off + size].iter().fold(0u64, |v, b| (v << 8) | *b as u64),
    }
}
#[inline]
pub fn put_be(buf: &mut [u8], off: usize, size: usize, v: u64) {
    match size {
        1 => buf[off] = v as u8,
        2 => buf[off..off + 2].copy_from_slice(&(v as u16).to_be_bytes()),
        4 => buf[off..off + 4].copy_from_slice(&(v as u32).to_be_bytes()),
        8 => buf[off..off + 8].copy_from_slice(&v.to_be_bytes()),
        _ => {
            for i in 0..size {
                buf[off + i] = (v >> (8 * (size - 1 - i))) as u8;
            }
        }
    }
}

/// Translate a GameCube effective/physical address into a MEM1 offset.
#[inline]
pub fn ram_off(addr: u32) -> Option<usize> {
    let phys = (addr & 0x3FFF_FFFF) as usize;
    (phys < RAM_SIZE).then_some(phys)
}

/// One guest CPU thread.
///
/// The register file lives inline (not behind a pointer): generated functions take
/// `&mut Ctx`, which tells LLVM nothing else aliases it, so guest registers can stay in host
/// registers across loads/stores to guest RAM, and fixed offsets need no bounds checks.
pub struct Ctx {
    pub regs: [u8; REG_BYTES],
    /// MEM1 (RAM_SIZE bytes), owned by the runtime's machine.
    pub ram: *mut u8,
    pub host: &'static Host,
    /// Guest OSThread pointer this context runs (or a scheduler marker below 0x80000000).
    pub thread: u32,
    pub in_irq: bool,
    pub depth: u32,
    /// Set by [`Ctx::call_orig`]: the next entry to this address skips its HLE handler.
    pub bypass: u32,
}

// SAFETY: guest code runs on exactly one host thread at a time (the runtime's scheduler token).
unsafe impl Send for Ctx {}

impl Ctx {
    pub fn new(regs: usize, ram: *mut u8, host: &'static Host) -> Self {
        assert!(regs <= REG_BYTES, "register space {regs:#x} larger than REG_BYTES");
        Ctx { regs: [0; REG_BYTES], ram, host, thread: 0, in_irq: false, depth: 0, bypass: 0 }
    }

    #[inline]
    pub fn ram(&self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ram, RAM_SIZE) }
    }

    // Register space.
    #[inline] pub fn r(&self, off: usize, size: usize) -> u64 { get_be(&self.regs, off, size) }
    #[inline] pub fn wr(&mut self, off: usize, size: usize, v: u64) { put_be(&mut self.regs, off, size, v) }
    #[inline] pub fn gpr(&self, n: usize) -> u32 { self.r(4 * n, 4) as u32 }
    #[inline] pub fn set_gpr(&mut self, n: usize, v: u32) { self.wr(4 * n, 4, v as u64) }

    /// `mftb`: guest time base backed by the host clock.
    #[inline] pub fn tb_lo(&self) -> u64 { (self.host.timebase)() & 0xFFFF_FFFF }
    #[inline] pub fn tb_hi(&self) -> u64 { (self.host.timebase)() >> 32 }

    #[inline]
    pub fn ld(&mut self, addr: u64, size: usize) -> u64 {
        let a = addr as u32;
        match ram_off(a) {
            Some(o) if o + size <= RAM_SIZE => get_be(self.ram(), o, size),
            _ => (self.host.mmio_read)(self, a, size),
        }
    }

    #[inline]
    pub fn st(&mut self, addr: u64, size: usize, v: u64) {
        let a = addr as u32;
        match ram_off(a) {
            Some(o) if o + size <= RAM_SIZE => put_be(self.ram(), o, size, v),
            _ => (self.host.mmio_write)(self, a, size, v),
        }
    }

    /// Called at the top of every generated function: bookkeeping, interrupt delivery and the
    /// HLE check. Returns true if an HLE handler fully handled the call.
    #[inline]
    pub fn enter(&mut self, addr: u32, entry: u32) -> bool {
        bump(&FN_ENTRIES);
        LAST_FN.store(addr, Relaxed);
        if COUNT_HITS.load(Relaxed) {
            count_hit(addr);
        }
        // Consume the bypass before interrupt delivery can run other guest code.
        let bypass = self.bypass == addr;
        if bypass {
            self.bypass = 0;
        }
        if ATTENTION.load(Relaxed) && !self.in_irq && self.r(REG_MSR, 4) & MSR_EE != 0 {
            (self.host.deliver)(self);
        }
        if bypass || entry != addr || !hooked(addr) {
            return false;
        }
        (self.host.hle)(self, addr)
    }

    /// Loop back-edge: deliver pending interrupts, as the CPU would between any two
    /// instructions. Without this, a loop that polls a flag set by an interrupt handler
    /// (`while (!__AXDSPInitFlag);`) never sees the interrupt, since it enters no function.
    #[inline(always)]
    pub fn poll(&mut self) {
        if ATTENTION.load(Relaxed) && !self.in_irq && self.r(REG_MSR, 4) & MSR_EE != 0 {
            (self.host.deliver)(self);
        }
    }

    /// A function was entered at an address that isn't one of its block leaders (a computed
    /// jump the generator didn't know about). Logged; the call returns without running code.
    /// Fix by making the address a leader (it's then data-referenced or a branch target).
    #[cold]
    pub fn block_miss(&mut self, func: u32, pc: u32) {
        BLOCK_MISSES.fetch_add(1, Relaxed);
        (self.host.log)(format!("mid-block entry {pc:#010x} in function {func:#010x} (not a block leader)"));
    }

    /// Call through the dispatch table (indirect calls, tail jumps, cross-crate calls).
    pub fn call(&mut self, target: u64) {
        bump(&CALLS);
        if self.depth > 20_000 {
            (self.host.log)(format!("call depth limit hit at {target:#010x}"));
            return;
        }
        self.depth += 1;
        let found = (self.host.dispatch)(self, target as u32);
        self.depth -= 1;
        if !found {
            UNKNOWN_TARGETS.fetch_add(1, Relaxed);
            (self.host.log)(format!("unknown call target {target:#010x}"));
        }
    }

    /// From an HLE handler: run the original recompiled function at `addr`.
    pub fn call_orig(&mut self, addr: u32) {
        self.bypass = addr;
        self.call(addr as u64);
        self.bypass = 0;
    }

    /// Call a guest function with up to 8 integer arguments; returns r3.
    pub fn call_guest(&mut self, addr: u32, args: &[u32]) -> u32 {
        for (i, a) in args.iter().enumerate() {
            self.set_gpr(3 + i, *a);
        }
        self.call(addr as u64);
        self.gpr(3)
    }

    /// CALLOTHER intrinsics. Cache-block zeroing is real; the rest go to the runtime.
    pub fn intrinsic(&mut self, name: &'static str, args: &[u64]) -> u64 {
        match name {
            "dataCacheBlockClearToZero" | "dataCacheBlockSetToZeroLocked" => {
                let a = args.first().copied().unwrap_or(0) & !31;
                for i in 0..4 {
                    self.st(a + 8 * i, 8, 0);
                }
            }
            _ => (self.host.intrinsic)(self, name, args),
        }
        0
    }

    /// `mtspr DMAL` with the trigger bit: copy between RAM and the locked cache.
    pub fn lc_dma(&mut self) {
        let (u, l) = (self.r(REG_DMAU, 4) as u32, self.r(REG_DMAL, 4) as u32);
        if l & 2 == 0 {
            return;
        }
        let blocks = match ((u & 0x1F) << 2) | ((l >> 2) & 3) {
            0 => 128,
            n => n,
        };
        let (mem, lc) = ((u & !0x1F) as u64, (l & !0x1F) as u64);
        for i in 0..(blocks * 4) as u64 {
            if l & 0x10 != 0 {
                let v = self.ld(mem + 8 * i, 8);
                self.st(lc + 8 * i, 8, v);
            } else {
                let v = self.ld(lc + 8 * i, 8);
                self.st(mem + 8 * i, 8, v);
            }
        }
        self.wr(REG_DMAL, 4, (l & !2) as u64);
    }
}
