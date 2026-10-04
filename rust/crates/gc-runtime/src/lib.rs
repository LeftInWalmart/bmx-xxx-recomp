//! The console runtime behind the generated code.
//!
//! * [`Machine`] is the shared console: MEM1, ARAM, MMIO devices, the guest-thread scheduler,
//!   the disc and the GX frame output. There is one per process ([`machine`]).
//! * The CPU context ([`Ctx`]) and the ABI the generated crates use live in `gc-core`; this
//!   crate plugs into it through [`HOST`]. Editing anything here rebuilds only this crate and
//!   its dependents (`bmx-recomp`, `bmx-bevy`), not the 48 generated crates.
//!
//! Guest code only ever runs on the host thread that holds the scheduler token (see
//! `sched.rs`), which is what makes the unsynchronised RAM/MMIO access sound in practice.
#![allow(clippy::missing_safety_doc)]

pub mod audio;
pub mod cheats;
pub mod diag;
pub mod disc;
pub mod dsp;
pub mod gx;
pub mod hle;
pub mod hw;
pub mod memcard;
pub mod pad;
pub mod hostprof;
pub mod probe;
pub mod psmtx;
pub mod sched;
pub mod thp;

pub use gc_core::{
    get_be, ops, put_be, Ctx, CALLS, FN_ENTRIES, LAST_FN, MSR_EE, RAM_SIZE, REG_GQR0, REG_MSR,
    REG_R13, REG_R2, UNKNOWN_TARGETS,
};

use std::cell::UnsafeCell;
use std::sync::{Arc, OnceLock};

/// 16 MiB of ARAM.
pub const ARAM_SIZE: usize = 0x0100_0000;
/// OS global: pointer to the current OSThread.
pub const OS_CURRENT_THREAD: u32 = 0x8000_00E4;

/// Resolves a guest address to a recompiled function and runs it. Returns false if unknown.
pub type Dispatch = fn(&mut Ctx, u32) -> bool;

pub struct Machine {
    ram: UnsafeCell<Vec<u8>>,
    aram: UnsafeCell<Vec<u8>>,
    hw: UnsafeCell<hw::Hardware>,
    pub dispatch: Dispatch,
    pub sched: sched::Sched,
    pub disc: Option<disc::Disc>,
    pub frames: Arc<gx::FrameShared>,
    pub pad: Arc<pad::PadShared>,
    pub audio: Arc<audio::AudioShared>,
    /// Function name lookup (for HLE binding and diagnostics), and its inverse.
    pub names: fn(u32) -> Option<&'static str>,
    pub addr_of: fn(&str) -> Option<u32>,
    pub reg_space: usize,
}

// SAFETY: guest code runs on exactly one host thread at a time (scheduler token); the
// Bevy side only touches `frames` and `pad`, which are internally synchronised.
unsafe impl Sync for Machine {}
unsafe impl Send for Machine {}

static MACHINE: OnceLock<&'static Machine> = OnceLock::new();

/// The console, if it has been created.
pub fn try_machine() -> Option<&'static Machine> {
    MACHINE.get().copied()
}

/// The console. Panics if called before [`Machine::new`].
pub fn machine() -> &'static Machine {
    MACHINE.get().expect("machine not created")
}

fn dispatch_hook(c: &mut Ctx, addr: u32) -> bool {
    let found = (machine().dispatch)(c, addr);
    if !found {
        // Context for bad indirect calls: caller (LR), stack, thread, this-pointer, and the
        // stack's saved-LR chain (who called the caller).
        let m = machine();
        let (lr, sp) = (c.r(0x1020, 4) as u32, c.gpr(1));
        let mut chain = String::new();
        let mut fp = sp;
        for _ in 0..6 {
            let next = m.read_be(fp, 4) as u32;
            if !(0x8000_0000..0x8180_0000).contains(&next) || next <= fp {
                break;
            }
            let saved_lr = m.read_be(next + 4, 4) as u32;
            chain += &format!(" <- {saved_lr:#010x}");
            fp = next;
        }
        diag::log(format!(
            "bad call {addr:#010x}: lr {lr:#010x} sp {sp:#010x} thread {:#x} r3 {:#010x} [r3]={:#010x} chain{chain}",
            c.thread, c.gpr(3), m.read_be(c.gpr(3), 4)
        ));
    }
    found
}
fn intrinsic_hook(_c: &mut Ctx, name: &'static str, _args: &[u64]) {
    diag::intrinsic(name);
}

/// The services `gc-core` calls into.
pub static HOST: gc_core::Host = gc_core::Host {
    mmio_read: hw::read,
    mmio_write: hw::write,
    deliver: sched::deliver,
    hle: hle::call,
    dispatch: dispatch_hook,
    intrinsic: intrinsic_hook,
    log: diag::log,
    timebase: sched::timebase,
};

impl Machine {
    pub fn new(
        reg_space: usize,
        dispatch: Dispatch,
        names: fn(u32) -> Option<&'static str>,
        addr_of: fn(&str) -> Option<u32>,
        disc: Option<disc::Disc>,
    ) -> &'static Machine {
        let m: &'static Machine = Box::leak(Box::new(Machine {
            ram: UnsafeCell::new(vec![0; RAM_SIZE]),
            aram: UnsafeCell::new(vec![0; ARAM_SIZE]),
            hw: UnsafeCell::new(hw::Hardware::default()),
            dispatch,
            sched: sched::Sched::default(),
            disc,
            frames: Arc::new(gx::FrameShared::default()),
            pad: Arc::new(pad::PadShared::default()),
            audio: Arc::new(audio::AudioShared::default()),
            names,
            addr_of,
            reg_space,
        }));
        if MACHINE.set(m).is_err() {
            panic!("only one Machine per process");
        }
        m
    }

    #[allow(clippy::mut_from_ref)]
    pub fn ram(&self) -> &mut [u8] {
        unsafe { &mut *self.ram.get() }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn aram(&self) -> &mut [u8] {
        unsafe { &mut *self.aram.get() }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn hw(&self) -> &mut hw::Hardware {
        unsafe { &mut *self.hw.get() }
    }

    /// Translate a GameCube effective/physical address into a MEM1 offset.
    #[inline]
    pub fn ram_off(addr: u32) -> Option<usize> {
        gc_core::ram_off(addr)
    }

    pub fn read_be(&self, addr: u32, size: usize) -> u64 {
        match Self::ram_off(addr) {
            Some(o) if o + size <= RAM_SIZE => get_be(self.ram(), o, size),
            _ => 0,
        }
    }
    pub fn write_be(&self, addr: u32, size: usize, v: u64) {
        if let Some(o) = Self::ram_off(addr) {
            if o + size <= RAM_SIZE {
                put_be(self.ram(), o, size, v)
            }
        }
    }
    pub fn load(&self, addr: u32, bytes: &[u8]) {
        if let Some(o) = Self::ram_off(addr) {
            let n = bytes.len().min(RAM_SIZE - o);
            self.ram()[o..o + n].copy_from_slice(&bytes[..n]);
        }
    }
    pub fn name(&self, addr: u32) -> &'static str {
        (self.names)(addr).unwrap_or("?")
    }

    /// A fresh CPU context attached to this machine.
    pub fn new_ctx(&'static self) -> Ctx {
        Ctx::new(self.reg_space, self.ram().as_mut_ptr(), &HOST)
    }
}

/// Runtime-side conveniences on the `gc-core` context.
pub trait CtxExt {
    fn m(&self) -> &'static Machine;
    /// A fresh context for a new guest thread, inheriting SDA bases, MSR and GQRs.
    fn spawn_like(&self, thread: u32) -> Ctx;
}

impl CtxExt for Ctx {
    #[inline]
    fn m(&self) -> &'static Machine {
        machine()
    }

    fn spawn_like(&self, thread: u32) -> Ctx {
        let mut c = machine().new_ctx();
        for off in [REG_R2, REG_R13, REG_MSR] {
            c.wr(off, 4, self.r(off, 4));
        }
        c.regs[REG_GQR0..REG_GQR0 + 32].copy_from_slice(&self.regs[REG_GQR0..REG_GQR0 + 32]);
        c.thread = thread;
        c
    }
}
