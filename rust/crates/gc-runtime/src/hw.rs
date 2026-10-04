//! Hardware / MMIO boundary. Everything outside MEM1 lands here.
//!
//! Registers at 0xCC000000..0xCC010000 are backed by a byte array (so reads return what the
//! game last wrote) with device behaviour layered on top where polling loops or DMA need it:
//!
//! | Block | Behaviour |
//! |---|---|
//! | CP  0xCC000000 | status always "idle" |
//! | PE  0xCC001000 | interrupt status bits are write-1-to-clear; token register |
//! | VI  0xCC002000 | beam position advances with time; retrace status set on VI interrupt |
//! | PI  0xCC003000 | reset switch reads "released"; CPU FIFO write pointer advances |
//! | DSP 0xCC005000 | mailboxes talk to the DSP emulation (`dsp.rs`); ARAM DMA is instant + interrupt |
//! | DI  0xCC006000 | read DMA from the disc image + interrupt (fallback; DVD is HLE'd) |
//! | SI/EXI | transfers complete instantly; no devices attached |
//! | AI  0xCC006C00 | sample counter advances |
//! | GX  0xCC008000 | write-gather pipe → RAM FIFO + `gx` decoder |
//! | LC  0xE0000000 | 16 KiB locked cache scratch |

use crate::sched::{self, IRQ_DI, IRQ_DSP_ARAM, IRQ_PE_FINISH, IRQ_PE_TOKEN, IRQ_VI};
use crate::{get_be, gx, put_be, Ctx, CtxExt, Machine, ARAM_SIZE, RAM_SIZE};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub static MMIO_READS: AtomicU64 = AtomicU64::new(0);
pub static MMIO_WRITES: AtomicU64 = AtomicU64::new(0);
pub static GX_BYTES: AtomicU64 = AtomicU64::new(0);
/// AI DMA buffers handed to the host audio output.
pub static AI_BUFFERS: AtomicU64 = AtomicU64::new(0);

pub const GX_FIFO: u32 = 0xCC00_8000;
const MMIO_BASE: u32 = 0xCC00_0000;
const MMIO_SIZE: u32 = 0x1_0000;
const LC_BASE: u32 = 0xE000_0000;
const LC_SIZE: u32 = 0x4000;

const CP_SR: u32 = 0x0000;
const PE_ISR: u32 = 0x100A;
const PE_TOKEN: u32 = 0x100E;
const VI_DISP_POS: u32 = 0x202C;
const VI_DI0: u32 = 0x2030;
const PI_INTSR: u32 = 0x3000;
const PI_FIFO_BASE: u32 = 0x300C;
const PI_FIFO_END: u32 = 0x3010;
const PI_FIFO_WPTR: u32 = 0x3014;
const DSP_MBOX_IN_H: u32 = 0x5000;
const DSP_MBOX_IN_L: u32 = 0x5002;
const DSP_MBOX_OUT_H: u32 = 0x5004;
const DSP_MBOX_OUT_L: u32 = 0x5006;
const AI_DMA_HI: u32 = 0x5030;
const AI_DMA_LO: u32 = 0x5032;
const AI_DMA_CTRL: u32 = 0x5036;
const AI_CR: u32 = 0x6C00;
const DSP_CR: u32 = 0x500A;
const AR_MODE: u32 = 0x5016;
const AR_MMADDR: u32 = 0x5020;
const AR_ARADDR: u32 = 0x5024;
const AR_CNT: u32 = 0x5028;
const DI_SR: u32 = 0x6000;
const DI_CMD0: u32 = 0x6008;
const DI_CMD1: u32 = 0x600C;
const DI_MAR: u32 = 0x6014;
const DI_LEN: u32 = 0x6018;
const DI_CR: u32 = 0x601C;
const SI_COMCSR: u32 = 0x6434;
const EXI_CR: [u32; 3] = [0x680C, 0x6820, 0x6834];
const AI_SCNT: u32 = 0x6C08;

pub struct Hardware {
    mmio: Vec<u8>,
    lc: Vec<u8>,
    pub gx: gx::Gx,
    /// CPU FIFO attached to the GP (false while it is redirected to build a display list).
    pub gx_link: bool,
    pub dsp: crate::dsp::Dsp,
    /// Write-gather pipe: bytes are flushed to the RAM FIFO in 32-byte bursts, like hardware.
    gather: Vec<u8>,
    ai_samples: u32,
    unmapped_logged: u32,
}

impl Default for Hardware {
    fn default() -> Self {
        Hardware {
            mmio: vec![0; MMIO_SIZE as usize],
            lc: vec![0; LC_SIZE as usize],
            gx: gx::Gx::default(),
            gx_link: true,
            dsp: crate::dsp::Dsp::default(),
            gather: Vec::with_capacity(32),
            ai_samples: 0,
            unmapped_logged: 0,
        }
    }
}

impl Hardware {
    fn get(&self, off: u32, size: usize) -> u64 {
        get_be(&self.mmio, off as usize, size)
    }
    fn set(&mut self, off: u32, size: usize, v: u64) {
        put_be(&mut self.mmio, off as usize, size, v)
    }
    fn get16(&self, off: u32) -> u32 {
        self.get(off, 2) as u32
    }
    fn set16(&mut self, off: u32, v: u32) {
        self.set(off, 2, v as u64)
    }
    fn get32(&self, off: u32) -> u32 {
        self.get(off, 4) as u32
    }
}

/// Does the access `[off, off+size)` touch register byte `reg`?
#[inline]
fn hits(off: u32, size: usize, reg: u32, reg_size: u32) -> bool {
    off < reg + reg_size && reg < off + size as u32
}

pub fn read(c: &mut Ctx, addr: u32, size: usize) -> u64 {
    MMIO_READS.fetch_add(1, Relaxed);
    let hw = c.m().hw();
    if (LC_BASE..LC_BASE + LC_SIZE).contains(&addr) {
        let o = (addr - LC_BASE) as usize;
        return if o + size <= hw.lc.len() { get_be(&hw.lc, o, size) } else { 0 };
    }
    if addr & 0xFFFF_0000 != MMIO_BASE {
        if hw.unmapped_logged < 32 {
            hw.unmapped_logged += 1;
            crate::diag::log(format!("unmapped read {addr:#010x} ({size} bytes)"));
        }
        return 0;
    }
    let off = addr & 0xFFFF;
    if off as usize + size > MMIO_SIZE as usize {
        return 0;
    }
    // Live registers: refresh the backing bytes before the generic read.
    if hits(off, size, CP_SR, 2) {
        // Read idle + command idle once the GPU thread has drained the FIFO.
        hw.set16(CP_SR, if gx::gpu().idle() { 0x000C } else { 0 });
    }
    if hits(off, size, PE_TOKEN, 2) || hits(off, size, PE_ISR, 2) {
        apply_gx_events(c.m());
    }
    if hits(off, size, VI_DISP_POS, 4) {
        let (v, h) = sched::vi_beam();
        hw.set(VI_DISP_POS, 4, ((v as u64) << 16) | h as u64);
    }
    if hits(off, size, PI_INTSR, 4) {
        let v = hw.get32(PI_INTSR) | 0x0001_0000; // RSWST: reset switch released
        hw.set(PI_INTSR, 4, v as u64);
    }
    if hits(off, size, DSP_MBOX_IN_H, 2) {
        let v = hw.get16(DSP_MBOX_IN_H) & !0x8000; // DSP consumed our mail
        hw.set16(DSP_MBOX_IN_H, v);
    }
    // DSP → CPU mail comes from the DSP emulation's queue; reading the low half consumes it.
    if hits(off, size, DSP_MBOX_OUT_H, 2) {
        let v = hw.dsp.read_hi();
        hw.set16(DSP_MBOX_OUT_H, v);
    }
    if hits(off, size, DSP_MBOX_OUT_L, 2) {
        let v = hw.dsp.read_lo();
        hw.set16(DSP_MBOX_OUT_L, v);
    }
    if hits(off, size, DSP_CR, 2) {
        let v = hw.get16(DSP_CR) & !0x0201; // reset done, ARAM DMA idle
        hw.set16(DSP_CR, v);
    }
    if hits(off, size, AR_MODE, 2) {
        let v = hw.get16(AR_MODE) | 1;
        hw.set16(AR_MODE, v);
    }
    for (r, mask) in [(DI_CR, 1u32), (SI_COMCSR, 1), (EXI_CR[0], 1), (EXI_CR[1], 1), (EXI_CR[2], 1)] {
        if hits(off, size, r, 4) {
            let v = hw.get32(r) & !mask; // transfer complete
            hw.set(r, 4, v as u64);
        }
    }
    if hits(off, size, AI_SCNT, 4) {
        hw.ai_samples = hw.ai_samples.wrapping_add(32);
        hw.set(AI_SCNT, 4, hw.ai_samples as u64);
    }
    hw.get(off, size)
}

pub fn write(c: &mut Ctx, addr: u32, size: usize, v: u64) {
    MMIO_WRITES.fetch_add(1, Relaxed);
    let m = c.m();
    let hw = m.hw();
    if addr == GX_FIFO || addr & 0xFFFF_FFF8 == GX_FIFO {
        gather_write(m, size, v);
        return;
    }
    if (LC_BASE..LC_BASE + LC_SIZE).contains(&addr) {
        let o = (addr - LC_BASE) as usize;
        if o + size <= hw.lc.len() {
            put_be(&mut hw.lc, o, size, v);
        }
        return;
    }
    if addr & 0xFFFF_0000 != MMIO_BASE {
        if hw.unmapped_logged < 32 {
            hw.unmapped_logged += 1;
            crate::diag::log(format!("unmapped write {addr:#010x} = {v:#x} ({size} bytes)"));
        }
        return;
    }
    let off = addr & 0xFFFF;
    if off as usize + size > MMIO_SIZE as usize {
        return;
    }
    // Write-1-to-clear status bits.
    let w1c = |hw: &Hardware, reg: u32, bits: u32| -> Option<u32> {
        hits(off, size, reg, 2).then(|| hw.get16(reg) & bits)
    };
    let pe_status = w1c(hw, PE_ISR, 0x000C);
    let dsp_status = w1c(hw, DSP_CR, 0x00A8);
    hw.set(off, size, v);
    if let Some(prev) = pe_status {
        let written = hw.get16(PE_ISR);
        hw.set16(PE_ISR, (written & !0x000C) | (prev & !(written & 0x000C)));
    }
    if let Some(prev) = dsp_status {
        let written = hw.get16(DSP_CR);
        hw.set16(DSP_CR, (written & !0x00A8) | (prev & !(written & 0x00A8)));
    }
    // CPU → DSP mail: complete once the low half is written.
    if hits(off, size, DSP_MBOX_IN_L, 2) {
        let mail = hw.get32(DSP_MBOX_IN_H);
        hw.dsp.mail_to_dsp(m, mail);
        let v = hw.get16(DSP_MBOX_IN_H) & !0x8000;
        hw.set16(DSP_MBOX_IN_H, v);
    }
    // DSP control: reset (bit 0) restarts the DSP ROM; halt (bit 2) freezes its mailbox.
    if hits(off, size, DSP_CR, 2) {
        let cr = hw.get16(DSP_CR);
        if cr & 1 != 0 {
            hw.dsp.reset();
            hw.set16(DSP_CR, cr & !1);
        }
        hw.dsp.set_halted(cr & 4 != 0);
    }
    // ARAM DMA: triggered by writing the count register (low half completes it).
    if hits(off, size, AR_CNT + 2, 2) {
        aram_dma(m);
    }
    // AI DMA: (re)start pacing when the enable bit is written.
    if hits(off, size, AI_DMA_CTRL, 2) {
        let ctrl = hw.get16(AI_DMA_CTRL);
        let bytes = (ctrl & 0x7FFF) as u64 * 32;
        let period = if ctrl & 0x8000 != 0 && bytes > 0 {
            bytes / 4 * sched::TB_HZ / ai_rate(hw) as u64
        } else {
            0
        };
        sched::set_ai_period(period);
    }
    // DI command start.
    if hits(off, size, DI_CR, 4) && hw.get32(DI_CR) & 1 != 0 {
        di_command(m);
    }
}

fn aram_dma(m: &Machine) {
    let hw = m.hw();
    let mm = (hw.get32(AR_MMADDR) & 0x03FF_FFE0) as usize;
    let ar = (hw.get32(AR_ARADDR) & 0x00FF_FFE0) as usize;
    let cnt = hw.get32(AR_CNT);
    if mm >= RAM_SIZE || ar >= ARAM_SIZE {
        crate::diag::log(format!("ARAM DMA out of range: mm {mm:#x} ar {ar:#x}"));
        return;
    }
    let len = (cnt & 0x7FFF_FFE0) as usize;
    let len = len.min(RAM_SIZE - mm).min(ARAM_SIZE - ar);
    if cnt & 0x8000_0000 != 0 {
        let (ram, aram) = (m.ram(), m.aram());
        ram[mm..mm + len].copy_from_slice(&aram[ar..ar + len]);
    } else {
        let (ram, aram) = (m.ram(), m.aram());
        aram[ar..ar + len].copy_from_slice(&ram[mm..mm + len]);
    }
    let cr = hw.get16(DSP_CR) | 0x0020;
    hw.set16(DSP_CR, cr);
    sched::raise(IRQ_DSP_ARAM);
}

fn di_command(m: &Machine) {
    let hw = m.hw();
    let cmd = hw.get32(DI_CMD0);
    if cmd >> 24 == 0xA8 {
        let off = (hw.get32(DI_CMD1) as u64) << 2;
        let dst = ((hw.get32(DI_MAR) & 0x03FF_FFE0) as usize).min(RAM_SIZE);
        let len = (hw.get32(DI_LEN) as usize).min(RAM_SIZE - dst);
        if let Some(d) = &m.disc {
            d.read(off, &mut m.ram()[dst..dst + len]);
        }
        hw.set(DI_LEN, 4, 0);
    }
    let cr = hw.get32(DI_CR) & !1;
    hw.set(DI_CR, 4, cr as u64);
    let sr = hw.get32(DI_SR) | 0x10; // TCINT
    hw.set(DI_SR, 4, sr as u64);
    sched::raise(IRQ_DI);
}

/// CPU write-gather pipe: queue bytes for the RAM FIFO and feed the GPU when it is linked.
fn gather_write(m: &Machine, size: usize, v: u64) {
    GX_BYTES.fetch_add(size as u64, Relaxed);
    let hw = m.hw();
    let be = v.to_be_bytes();
    hw.gather.extend_from_slice(&be[8 - size..]);
    // Like the hardware gather pipe, data leaves in 32-byte bursts: to the RAM FIFO, and (when
    // the CPU FIFO is linked to the GP) to the GPU. GXFlush pads with 32 zero bytes, so every
    // command the game waits on (draw done, tokens, copies) is pushed through.
    while hw.gather.len() >= 32 {
        let mut burst = [0u8; 32];
        burst.copy_from_slice(&hw.gather[..32]);
        hw.gather.drain(..32);
        let base = hw.get32(PI_FIFO_BASE) & 0x03FF_FFE0;
        let end = hw.get32(PI_FIFO_END) & 0x03FF_FFE0;
        let wptr_reg = hw.get32(PI_FIFO_WPTR);
        let mut wptr = wptr_reg & 0x03FF_FFE0;
        if (wptr as usize) + 32 <= RAM_SIZE && wptr != 0 {
            m.ram()[wptr as usize..wptr as usize + 32].copy_from_slice(&burst);
        }
        wptr += 32;
        let mut wrap = wptr_reg & 0x0400_0000;
        if end > base && wptr >= end {
            wptr = base;
            wrap = 0x0400_0000;
        }
        hw.set(PI_FIFO_WPTR, 4, (wptr | wrap) as u64);
        if hw.gx_link {
            gx::gpu().submit(&burst);
        }
    }
}

/// Apply PE register effects of events the GPU thread produced (on the CPU side; the GPU
/// thread already raised the interrupts).
fn apply_gx_events(m: &Machine) {
    let hw = m.hw();
    for e in gx::gpu().take_events() {
        match e {
            gx::Event::DrawDone => {
                let v = hw.get16(PE_ISR) | 0x0008;
                hw.set16(PE_ISR, v);
            }
            gx::Event::Token(t) => hw.set16(PE_TOKEN, t as u32),
            gx::Event::TokenInt(t) => {
                hw.set16(PE_TOKEN, t as u32);
                let v = hw.get16(PE_ISR) | 0x0004;
                hw.set16(PE_ISR, v);
            }
        }
    }
}

fn ai_rate(hw: &Hardware) -> u32 {
    if hw.get32(AI_CR) & 0x40 != 0 { 32000 } else { 48000 }
}

/// Device side effects that accompany an interrupt, applied just before its handler runs.
pub fn before_interrupt(c: &mut Ctx, irq: u32) {
    let m = c.m();
    let hw = m.hw();
    match irq {
        IRQ_PE_FINISH | IRQ_PE_TOKEN => apply_gx_events(m),
        IRQ_VI => {
            let v = hw.get16(VI_DI0) | 0x8000;
            hw.set16(VI_DI0, v);
        }
        sched::IRQ_DSP_DSP => {
            let v = hw.get16(DSP_CR) | 0x0080;
            hw.set16(DSP_CR, v);
        }
        sched::IRQ_DSP_AI => {
            // The AI fetched the buffer the registers point at: hand it to the host and
            // signal the game to prepare the next one.
            let addr = ((hw.get16(AI_DMA_HI) & 0x03FF) << 16) | (hw.get16(AI_DMA_LO) & 0xFFE0);
            let len = (hw.get16(AI_DMA_CTRL) & 0x7FFF) as usize * 32;
            let a = addr as usize;
            if len > 0 && a + len <= RAM_SIZE {
                m.audio.push_be(&m.ram()[a..a + len], ai_rate(hw));
                AI_BUFFERS.fetch_add(1, Relaxed);
            }
            let v = hw.get16(DSP_CR) | 0x0008;
            hw.set16(DSP_CR, v);
        }
        _ => {}
    }
}
