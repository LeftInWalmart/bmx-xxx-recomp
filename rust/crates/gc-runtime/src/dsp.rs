//! The audio DSP, emulated at the mailbox level, so the game's own SDK `DSP` driver and `AX`
//! library run unmodified (BMX XXX uses AX; there is no game-side HLE).
//!
//! * **ROM**: after a reset the DSP mails `0x8071FEED`. `__DSP_boot_task` then sends the boot
//!   parameters (`0x80F3A001` IRAM source, `0x80F3A002` length, `0x80F3B002` DRAM length,
//!   `0x80F3C002` IRAM destination, `0x80F3D001` start PC). The task's IRAM image is
//!   identified by its Ector checksum, as Dolphin's DSP HLE does.
//! * **AX ucode**: announces itself with `DSP_INIT` (0xDCD10000). Every 5 ms the CPU sends
//!   `0xBABE0000 | n` and then the address of an `n`-word command list. The list is executed:
//!   parameter-block (voice) processing with the hardware accelerator's ADPCM / PCM8 / PCM16
//!   decoding, sample-rate conversion, volume envelope, low-pass filter, the 9 mix buses
//!   (main/auxA/auxB × L/R/S), aux send/return, the compressor and the final interleaved 16-bit
//!   output. The DSP then mails `DSP_YIELD` (0xDCD10002) with an interrupt, and the SDK driver
//!   answers with `0xCDD10003` (continue).
//!
//! Mails that request an interrupt raise `IRQ_DSP_DSP`. A mail queued behind others raises it
//! when the mail ahead of it is read, like Dolphin's mail handler.

use crate::sched::{self, IRQ_DSP_DSP};
use crate::{Machine, ARAM_SIZE};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// AX frames (5 ms command lists) processed so far.
pub static SUBFRAMES_RENDERED: AtomicU64 = AtomicU64::new(0);
/// Voices that were running in the last frame.
pub static VOICES_LAST: AtomicU64 = AtomicU64::new(0);
/// Most voices running in any frame so far.
pub static VOICES_MAX: AtomicU64 = AtomicU64::new(0);
/// Voice-frames processed with `running == 1`, cumulative.
pub static ACTIVE_SEEN: AtomicU64 = AtomicU64::new(0);

const DSP_INIT: u32 = 0xDCD1_0000;
const DSP_RESUME: u32 = 0xDCD1_0001;
const DSP_YIELD: u32 = 0xDCD1_0002;
const MAIL_RESUME: u32 = 0xCDD1_0000;
const MAIL_NEW_UCODE: u32 = 0xCDD1_0001;
const MAIL_RESET: u32 = 0xCDD1_0002;
const MAIL_CONTINUE: u32 = 0xCDD1_0003;
const MAIL_CMDLIST: u32 = 0xBABE_0000;

/// The oldest GameCube AX ucode: no low-pass filter in the PB and the old mixer-control bits.
const CRC_AX_OLD: u32 = 0x4E8A_8B21;
const CRC_CARD: u32 = 0x65D6_CC6F;

/// Samples per millisecond and milliseconds per AX frame.
const SPMS: usize = 32;
const MS: usize = 5;
const FRAME: usize = SPMS * MS;

pub struct Dsp {
    /// DSP → CPU mail queue: (mail, raise an interrupt once this mail has been read).
    mail: VecDeque<(u32, bool)>,
    last_mail: u32,
    halted: bool,
    ucode: Ucode,
    rom: RomBoot,
    ax: Box<Ax>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Ucode {
    Rom,
    Ax,
    /// A task we don't emulate (logged once); mails are ignored.
    Unknown(u32),
}

#[derive(Default)]
struct RomBoot {
    param: u32,
    ram_addr: u32,
    length: u32,
    imem_addr: u32,
}

impl Default for Dsp {
    fn default() -> Self {
        let mut d = Dsp {
            mail: VecDeque::new(),
            last_mail: 0,
            halted: false,
            ucode: Ucode::Rom,
            rom: RomBoot::default(),
            ax: Box::new(Ax::default()),
        };
        d.push(0x8071_FEED, false);
        d
    }
}

impl Dsp {
    /// DSP control register written with the reset bit: back to the ROM, which mails 0x8071FEED.
    pub fn reset(&mut self) {
        self.mail.clear();
        self.ucode = Ucode::Rom;
        self.rom = RomBoot::default();
        self.push(0x8071_FEED, false);
    }

    pub fn set_halted(&mut self, halted: bool) {
        self.halted = halted;
    }

    fn push(&mut self, mail: u32, interrupt: bool) {
        if interrupt {
            match self.mail.front_mut() {
                None => sched::raise(IRQ_DSP_DSP),
                Some(front) => front.1 = true,
            }
        }
        self.mail.push_back((mail, false));
    }

    /// DSP → CPU mailbox, high half (bit 15 = mail pending).
    pub fn read_hi(&mut self) -> u32 {
        if !self.halted {
            if let Some(&(m, _)) = self.mail.front() {
                self.last_mail = m;
            }
        }
        self.last_mail >> 16
    }

    /// DSP → CPU mailbox, low half: consumes the mail.
    pub fn read_lo(&mut self) -> u32 {
        if !self.halted {
            if let Some((m, irq)) = self.mail.pop_front() {
                self.last_mail = m;
                if irq {
                    sched::raise(IRQ_DSP_DSP);
                }
            }
        }
        self.last_mail &= 0x7FFF_FFFF;
        self.last_mail & 0xFFFF
    }

    /// CPU → DSP mail (complete once the low half is written).
    pub fn mail_to_dsp(&mut self, m: &Machine, mail: u32) {
        match self.ucode {
            Ucode::Rom => self.rom_mail(m, mail),
            Ucode::Ax => {
                let mut ax = std::mem::take(&mut self.ax);
                ax.mail(self, m, mail);
                self.ax = ax;
            }
            Ucode::Unknown(_) => {}
        }
    }

    fn rom_mail(&mut self, m: &Machine, mail: u32) {
        if self.rom.param == 0 {
            if mail & 0xFFFF_0000 == 0x80F3_0000 {
                self.rom.param = mail;
            } else {
                self.push(0xFEEE_0000 | (mail & 0xFFFF), false);
            }
            return;
        }
        match self.rom.param {
            0x80F3_A001 => self.rom.ram_addr = mail,
            0x80F3_A002 => self.rom.length = mail & 0xFFFF,
            0x80F3_C002 => self.rom.imem_addr = mail & 0xFFFF,
            0x80F3_D001 => {
                self.rom.param = 0;
                let crc = ector_crc(m, self.rom.ram_addr, self.rom.length);
                self.boot(crc);
                return;
            }
            _ => {}
        }
        self.rom.param = 0;
    }

    fn boot(&mut self, crc: u32) {
        self.mail.clear();
        let known = crc != CRC_CARD;
        crate::diag::log(format!(
            "DSP: booting task (IRAM {:#010x} len {:#x} -> {:#06x}), ucode crc {crc:#010x}{}",
            self.rom.ram_addr,
            self.rom.length,
            self.rom.imem_addr,
            if known { " -> AX HLE" } else { " (CARD ucode: not emulated)" }
        ));
        if known {
            self.ucode = Ucode::Ax;
            self.ax = Box::new(Ax { crc, ..Ax::default() });
            self.push(DSP_INIT, true);
        } else {
            self.ucode = Ucode::Unknown(crc);
        }
    }
}

/// Dolphin's `HashEctor`: the checksum DSP HLE uses to identify a ucode image.
fn ector_crc(m: &Machine, addr: u32, len: u32) -> u32 {
    let mut crc = 0u32;
    for i in 0..len {
        crc ^= m.read_be(addr.wrapping_add(i), 1) as u32;
        crc = crc.rotate_left(3);
    }
    crc
}

// ---------------------------------------------------------------------------------------------
// AX
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Default)]
enum MailState {
    #[default]
    CmdListSize,
    CmdListAddr,
    NextTask,
}

/// Mix buses, in the AX order main L/R/S, auxA L/R/S, auxB L/R/S.
const MAIN_L: usize = 0;
const MAIN_R: usize = 1;
const MAIN_S: usize = 2;
const AUXA_L: usize = 3;
const AUXB_L: usize = 6;
const AUXB_R: usize = 7;
const AUXB_S: usize = 8;

struct Ax {
    crc: u32,
    state: MailState,
    cmdlist: [u16; 512],
    cmdlist_size: u16,
    buses: [[i32; FRAME]; 9],
    compressor_pos: u16,
    /// MAIL_NEW_UCODE: the next 10 mails describe a ucode switch (not emulated; logged).
    upload_steps: u32,
}

impl Default for Ax {
    fn default() -> Self {
        Ax {
            crc: 0,
            state: MailState::CmdListSize,
            cmdlist: [0; 512],
            cmdlist_size: 0,
            buses: [[0; FRAME]; 9],
            compressor_pos: 0,
            upload_steps: 0,
        }
    }
}

fn rd16(m: &Machine, a: u32) -> u16 {
    m.read_be(a, 2) as u16
}
fn rd32(m: &Machine, a: u32) -> u32 {
    m.read_be(a, 4) as u32
}
fn wr16(m: &Machine, a: u32, v: u16) {
    m.write_be(a, 2, v as u64)
}
fn wr32(m: &Machine, a: u32, v: u32) {
    m.write_be(a, 4, v as u64)
}
fn clamp16(v: i64) -> i16 {
    v.clamp(-0x8000, 0x7FFF) as i16
}

impl Ax {
    fn mail(&mut self, dsp: &mut Dsp, m: &Machine, mail: u32) {
        if self.upload_steps > 0 {
            self.upload_steps -= 1;
            if self.upload_steps == 0 {
                crate::diag::log("DSP: ucode switch requested (not emulated); staying on AX".into());
                dsp.push(DSP_RESUME, true);
            }
            return;
        }
        match self.state {
            MailState::CmdListSize => {
                if mail & 0xFFFF_0000 == MAIL_CMDLIST {
                    self.cmdlist_size = mail as u16;
                    self.state = MailState::CmdListAddr;
                } else {
                    crate::diag::log(format!("DSP/AX: unexpected mail {mail:#010x} (want command list)"));
                }
            }
            MailState::CmdListAddr => {
                self.copy_cmdlist(m, mail, self.cmdlist_size);
                self.run_cmdlist(m);
                self.cmdlist_size = 0;
                SUBFRAMES_RENDERED.fetch_add(1, Relaxed);
                dsp.push(DSP_YIELD, true);
                self.state = MailState::NextTask;
            }
            MailState::NextTask => match 0xCDD1_0000 | (mail & 0xFFFF) {
                MAIL_RESUME => {
                    dsp.push(DSP_RESUME, true);
                    self.state = MailState::CmdListSize;
                }
                MAIL_NEW_UCODE => {
                    self.upload_steps = 10;
                    self.state = MailState::CmdListSize;
                }
                MAIL_RESET => {
                    dsp.mail.clear();
                    dsp.ucode = Ucode::Rom;
                    dsp.rom = RomBoot::default();
                    dsp.push(0x8071_FEED, false);
                }
                MAIL_CONTINUE => self.state = MailState::CmdListSize,
                _ => crate::diag::log(format!("DSP/AX: unknown task mail {mail:#010x}")),
            },
        }
    }

    fn copy_cmdlist(&mut self, m: &Machine, addr: u32, size: u16) {
        if size as usize >= self.cmdlist.len() {
            crate::diag::log(format!("DSP/AX: command list at {addr:#010x} too large ({size})"));
            return;
        }
        for i in 0..size as u32 {
            self.cmdlist[i as usize] = rd16(m, addr + 2 * i);
        }
    }

    fn run_cmdlist(&mut self, m: &Machine) {
        let mut i = 0usize;
        let mut pb_addr = 0u32;
        let mut voices = 0u64;
        loop {
            let next = |i: &mut usize, l: &[u16; 512]| -> u16 {
                let v = l[*i & 511];
                *i += 1;
                v
            };
            let cl = self.cmdlist;
            let addr = |i: &mut usize| -> u32 {
                let hi = next(i, &cl) as u32;
                (hi << 16) | next(i, &cl) as u32
            };
            match next(&mut i, &cl) {
                0x00 => {
                    let a = addr(&mut i);
                    self.setup(m, a);
                }
                0x01 => {
                    let a = addr(&mut i);
                    let (vm, va, vb) = (next(&mut i, &cl), next(&mut i, &cl), next(&mut i, &cl));
                    self.download_and_mix(m, a, [vm, va, vb]);
                }
                0x02 => pb_addr = addr(&mut i),
                0x03 => voices += self.process_pbs(m, pb_addr),
                c @ (0x04 | 0x05) => {
                    let (w, r) = (addr(&mut i), addr(&mut i));
                    self.mix_aux(m, (c - 4) as usize, w, r);
                }
                0x06 => {
                    let a = addr(&mut i);
                    self.upload_lrs(m, a);
                }
                0x07 => {
                    let a = addr(&mut i);
                    self.set_main_lr(m, a, false);
                }
                0x08 => i += 10,
                0x09 => {
                    let r = addr(&mut i);
                    self.mix_aux(m, 1, 0, r);
                }
                0x0A..=0x0C => {}
                0x0D => {
                    let a = addr(&mut i);
                    let size = next(&mut i, &cl);
                    self.copy_cmdlist(m, a, size);
                    i = 0;
                }
                0x0E => {
                    let (surround, lr) = (addr(&mut i), addr(&mut i));
                    self.output(m, lr, surround);
                }
                0x0F => break,
                0x10 => {
                    let (ul, dl) = (addr(&mut i), addr(&mut i));
                    self.mix_auxb_lr(m, ul, dl);
                }
                0x11 => {
                    let a = addr(&mut i);
                    self.set_main_lr(m, a, true);
                }
                0x12 => {
                    let threshold = next(&mut i, &cl);
                    let frames = next(&mut i, &cl);
                    let a = addr(&mut i);
                    self.compressor(m, threshold, frames, a);
                }
                0x13 => {
                    let a: [u32; 6] = std::array::from_fn(|_| addr(&mut i));
                    self.send_aux_and_mix(m, a);
                }
                c => {
                    crate::diag::log(format!("DSP/AX: unknown command {c:#06x}"));
                    break;
                }
            }
            if i > 512 {
                break;
            }
        }
        VOICES_LAST.store(voices, Relaxed);
        VOICES_MAX.fetch_max(voices, Relaxed);
    }

    /// CMD_SETUP: each bus starts from a (value, delta) ramp given by the CPU.
    fn setup(&mut self, m: &Machine, init: u32) {
        for (b, bus) in self.buses.iter_mut().enumerate() {
            let at = init + 6 * b as u32;
            let value = ((rd16(m, at) as u32) << 16 | rd16(m, at + 2) as u32) as i32;
            let delta = rd16(m, at + 4) as i16 as i32;
            for (j, s) in bus.iter_mut().enumerate() {
                *s = if value == 0 { 0 } else { value.wrapping_add(j as i32 * delta) };
            }
        }
    }

    /// CMD_DL_AND_VOL_MIX. Each of main/auxA/auxB adds the same downloaded L/R/S block scaled
    /// by its own volume (as the ucode does: the source pointer is not advanced per group).
    fn download_and_mix(&mut self, m: &Machine, addr: u32, vols: [u16; 3]) {
        for (g, vol) in vols.iter().enumerate() {
            for ch in 0..3 {
                for k in 0..FRAME {
                    let s = rd32(m, addr + 4 * (ch * FRAME + k) as u32) as i32 as i64;
                    let bus = &mut self.buses[g * 3 + ch][k];
                    *bus = bus.wrapping_add(((s * *vol as i64) >> 15) as i32);
                }
            }
        }
    }

    fn upload(&self, m: &Machine, mut addr: u32, buses: &[usize]) {
        for &b in buses {
            for &s in &self.buses[b] {
                wr32(m, addr, s as u32);
                addr += 4;
            }
        }
    }

    /// CMD_MIX_AUXA/B: send the aux bus to the CPU (effects), add the CPU's return into main.
    fn mix_aux(&mut self, m: &Machine, aux: usize, write: u32, read: u32) {
        let base = AUXA_L + 3 * aux;
        if write != 0 {
            self.upload(m, write, &[base, base + 1, base + 2]);
        }
        for ch in 0..3 {
            for k in 0..FRAME {
                let s = rd32(m, read + 4 * (ch * FRAME + k) as u32) as i32;
                let bus = &mut self.buses[MAIN_L + ch][k];
                *bus = bus.wrapping_add(s);
            }
        }
    }

    fn upload_lrs(&self, m: &Machine, dst: u32) {
        self.upload(m, dst, &[MAIN_L, MAIN_R, MAIN_S]);
    }

    /// CMD_SET_LR (mono to both sides) / CMD_SET_OPPOSITE_LR (left inverted).
    fn set_main_lr(&mut self, m: &Machine, src: u32, opposite: bool) {
        for k in 0..FRAME {
            let s = rd32(m, src + 4 * k as u32) as i32;
            self.buses[MAIN_L][k] = if opposite { s.wrapping_neg() } else { s };
            self.buses[MAIN_R][k] = s;
            self.buses[MAIN_S][k] = 0;
        }
    }

    fn compressor(&mut self, m: &Machine, threshold: u16, release_frames: u16, table: u32) {
        let triggered = (0..FRAME).any(|i| {
            self.buses[MAIN_L][i].unsigned_abs() > threshold as u32
                || self.buses[MAIN_R][i].unsigned_abs() > threshold as u32
        });
        let frame_bytes = (FRAME * 2) as u32;
        let offset = if triggered {
            let o = self.compressor_pos as u32 * frame_bytes;
            self.compressor_pos = release_frames;
            o
        } else if self.compressor_pos != 0 {
            self.compressor_pos -= 1;
            (11 + self.compressor_pos as u32) * frame_bytes
        } else {
            return;
        };
        for i in 0..FRAME {
            let coef = rd16(m, table + offset + 2 * i as u32) as i64;
            for b in [MAIN_L, MAIN_R] {
                self.buses[b][i] = ((self.buses[b][i] as i64 * coef) >> 15) as i32;
            }
        }
    }

    /// CMD_OUTPUT: surround to RAM as 32-bit, main L/R clamped and interleaved R, L (16-bit).
    fn output(&self, m: &Machine, lr: u32, surround: u32) {
        self.upload(m, surround, &[MAIN_S]);
        for i in 0..FRAME {
            let l = clamp16(self.buses[MAIN_L][i] as i64);
            let r = clamp16(self.buses[MAIN_R][i] as i64);
            wr16(m, lr + 4 * i as u32, r as u16);
            wr16(m, lr + 4 * i as u32 + 2, l as u16);
        }
    }

    fn mix_auxb_lr(&mut self, m: &Machine, ul: u32, dl: u32) {
        self.upload(m, ul, &[AUXB_L, AUXB_R]);
        for (n, (aux, main)) in [(AUXB_L, MAIN_L), (AUXB_R, MAIN_R)].into_iter().enumerate() {
            for k in 0..FRAME {
                let s = rd32(m, dl + 4 * (n * FRAME + k) as u32) as i32;
                self.buses[aux][k] = s;
                self.buses[main][k] = self.buses[main][k].wrapping_add(s);
            }
        }
    }

    fn send_aux_and_mix(&mut self, m: &Machine, a: [u32; 6]) {
        self.upload(m, a[0], &[AUXA_L, AUXA_L + 1, AUXA_L + 2]);
        self.upload(m, a[1], &[AUXB_S]);
        for (n, bus) in [MAIN_L, MAIN_R, AUXB_L, AUXB_R].into_iter().enumerate() {
            for k in 0..FRAME {
                let s = rd32(m, a[2 + n] + 4 * k as u32) as i32;
                self.buses[bus][k] = self.buses[bus][k].wrapping_add(s);
            }
        }
    }

    /// CMD_PROCESS: run every parameter block in the linked list for 5 × 1 ms.
    fn process_pbs(&mut self, m: &Machine, mut pb_addr: u32) -> u64 {
        let has_lpf = self.crc != CRC_AX_OLD;
        let mut running = 0;
        let mut guard = 0;
        while pb_addr != 0 && guard < 256 {
            guard += 1;
            let mut pb = Pb::read(m, pb_addr, has_lpf);
            let updates = pb.load_updates(m);
            let was_running = pb.w[pb::RUNNING] == 1;
            for ms in 0..MS {
                pb.apply_updates(ms, &updates);
                let mctrl = mixer_control(self.crc, pb.w[pb::MIXER_CONTROL]);
                process_voice(m, &mut pb, &mut self.buses, ms * SPMS, mctrl);
            }
            if was_running {
                running += 1;
                ACTIVE_SEEN.fetch_add(1, Relaxed);
            }
            pb.write(m, pb_addr, has_lpf);
            pb_addr = (pb.w[pb::NEXT_HI] as u32) << 16 | pb.w[pb::NEXT_LO] as u32;
        }
        running
    }
}

/// Parameter-block word indices (the AXPB layout with the low-pass filter; the oldest ucode's
/// PB lacks the 4 LPF words, so its later words shift down by 4 in memory).
mod pb {
    pub const NEXT_HI: usize = 0;
    pub const NEXT_LO: usize = 1;
    pub const SRC_TYPE: usize = 4;
    pub const MIXER_CONTROL: usize = 6;
    pub const RUNNING: usize = 7;
    pub const IS_STREAM: usize = 8;
    /// Mixer: (volume, delta) pairs: main L, main R, auxA L, auxA R, auxB L, auxB R, auxB S,
    /// main S, auxA S.
    pub const MIXER: usize = 9;
    pub const UPDATES_NUM: usize = 34;
    pub const UPDATES_HI: usize = 39;
    pub const UPDATES_LO: usize = 40;
    /// Depop: main L, auxA L, auxB L, main R, auxA R, auxB R, main S, auxA S, auxB S.
    pub const DPOP: usize = 41;
    pub const VOL_CUR: usize = 50;
    pub const VOL_DELTA: usize = 51;
    pub const LOOPING: usize = 55;
    pub const FORMAT: usize = 56;
    pub const LOOP_HI: usize = 57;
    pub const END_HI: usize = 59;
    pub const CUR_HI: usize = 61;
    pub const CUR_LO: usize = 62;
    pub const COEFS: usize = 63;
    pub const GAIN: usize = 79;
    pub const PRED_SCALE: usize = 80;
    pub const YN1: usize = 81;
    pub const YN2: usize = 82;
    pub const RATIO_HI: usize = 83;
    pub const RATIO_LO: usize = 84;
    pub const FRAC: usize = 85;
    pub const LAST: usize = 86;
    pub const LOOP_PRED_SCALE: usize = 90;
    pub const LOOP_YN1: usize = 91;
    pub const LOOP_YN2: usize = 92;
    pub const LPF_ON: usize = 93;
    pub const LPF_YN1: usize = 94;
    pub const LPF_A0: usize = 95;
    pub const LPF_B0: usize = 96;
    pub const LOOP_COUNTER: usize = 97;
    pub const WORDS: usize = 122;
}

struct Pb {
    w: [u16; pb::WORDS],
}

impl Pb {
    fn read(m: &Machine, addr: u32, has_lpf: bool) -> Pb {
        let mut w = [0u16; pb::WORDS];
        for (i, v) in w.iter_mut().enumerate() {
            let mem = if has_lpf || i < pb::LPF_ON {
                i
            } else if i < pb::LOOP_COUNTER {
                continue;
            } else {
                i - 4
            };
            if has_lpf || mem < pb::WORDS - 4 {
                *v = rd16(m, addr + 2 * mem as u32);
            }
        }
        Pb { w }
    }

    fn write(&self, m: &Machine, addr: u32, has_lpf: bool) {
        for (i, &v) in self.w.iter().enumerate() {
            let mem = if has_lpf || i < pb::LPF_ON {
                i
            } else if i < pb::LOOP_COUNTER {
                continue;
            } else {
                i - 4
            };
            if has_lpf || mem < pb::WORDS - 4 {
                wr16(m, addr + 2 * mem as u32, v);
            }
        }
    }

    fn load_updates(&self, m: &Machine) -> [(u16, u16); 32] {
        let a = (self.w[pb::UPDATES_HI] as u32) << 16 | self.w[pb::UPDATES_LO] as u32;
        std::array::from_fn(|i| (rd16(m, a + 4 * i as u32), rd16(m, a + 4 * i as u32 + 2)))
    }

    fn apply_updates(&mut self, ms: usize, updates: &[(u16, u16); 32]) {
        let start: usize = (0..ms).map(|i| self.w[pb::UPDATES_NUM + i] as usize).sum();
        let count = self.w[pb::UPDATES_NUM + ms] as usize;
        if start < updates.len() && count <= updates.len() - start {
            for &(off, val) in &updates[start..start + count] {
                if (off as usize) < pb::WORDS {
                    self.w[off as usize] = val;
                }
            }
        }
    }

    fn addr(&self, hi: usize) -> u32 {
        (self.w[hi] as u32) << 16 | self.w[hi + 1] as u32
    }
}

// Mix-control bits (Dolphin's AXMixControl).
const MIX_MAIN_L: u32 = 0x000001;
const MIX_MAIN_R: u32 = 0x000004;
const MIX_MAIN_S: u32 = 0x000010;
const MIX_AUXA_L: u32 = 0x000040;
const MIX_AUXA_R: u32 = 0x000100;
const MIX_AUXA_S: u32 = 0x000400;
const MIX_AUXB_L: u32 = 0x001000;
const MIX_AUXB_R: u32 = 0x004000;
const MIX_AUXB_S: u32 = 0x010000;
const MIX_ALL_RAMPS: u32 = 0xAAAAAA;

fn mixer_control(crc: u32, mc: u16) -> u32 {
    let mc = mc as u32;
    let mut r = 0;
    if crc == CRC_AX_OLD {
        if mc & 0x10 != 0 {
            r |= MIX_MAIN_L | MIX_MAIN_R;
            if mc & 0x6 == 0 {
                r |= MIX_AUXB_L | MIX_AUXB_R;
            }
            if mc & 0x7 == 1 {
                r |= MIX_AUXA_L | MIX_AUXA_R | MIX_AUXA_S;
            }
        } else {
            r |= MIX_MAIN_L | MIX_MAIN_R;
            if mc & 1 != 0 {
                r |= MIX_AUXA_L | MIX_AUXA_R;
            }
            if mc & 2 != 0 {
                r |= MIX_AUXB_L | MIX_AUXB_R;
            }
            if mc & 4 != 0 {
                r |= MIX_MAIN_S;
                if r & MIX_AUXA_L != 0 {
                    r |= MIX_AUXA_S;
                }
                if r & MIX_AUXB_L != 0 {
                    r |= MIX_AUXB_S;
                }
            }
        }
        if mc & 8 != 0 {
            r |= MIX_ALL_RAMPS;
        }
    } else {
        for (bit, flags) in [
            (0x0001, MIX_MAIN_L),
            (0x0002, MIX_MAIN_R),
            (0x0004, MIX_MAIN_S),
            (0x0008, (MIX_MAIN_L | MIX_MAIN_R | MIX_MAIN_S) << 1),
            (0x0010, MIX_AUXA_L),
            (0x0020, MIX_AUXA_R),
            (0x0040, (MIX_AUXA_L | MIX_AUXA_R) << 1),
            (0x0080, MIX_AUXA_S),
            (0x0100, MIX_AUXA_S << 1),
            (0x0200, MIX_AUXB_L),
            (0x0400, MIX_AUXB_R),
            (0x0800, (MIX_AUXB_L | MIX_AUXB_R) << 1),
            (0x1000, MIX_AUXB_S),
            (0x2000, MIX_AUXB_S << 1),
        ] {
            if mc & bit != 0 {
                r |= flags;
            }
        }
    }
    r
}

/// The DSP's sample accelerator, set up from a parameter block for one millisecond.
struct Accel {
    start: u32,
    end: u32,
    cur: u32,
    format: u16,
    gain: i16,
    yn1: i16,
    yn2: i16,
    pred_scale: u16,
    stopped: bool,
}

impl Accel {
    fn new(pb: &Pb) -> Accel {
        Accel {
            start: pb.addr(pb::LOOP_HI) & 0x3FFF_FFFF,
            end: pb.addr(pb::END_HI) & 0x3FFF_FFFF,
            cur: pb.addr(pb::CUR_HI) & 0xBFFF_FFFF,
            format: pb.w[pb::FORMAT],
            gain: pb.w[pb::GAIN] as i16,
            yn1: pb.w[pb::YN1] as i16,
            yn2: pb.w[pb::YN2] as i16,
            pred_scale: pb.w[pb::PRED_SCALE] & 0x7F,
            stopped: false,
        }
    }

    fn aram(m: &Machine, a: u32) -> u8 {
        m.aram().get(a as usize % ARAM_SIZE).copied().unwrap_or(0)
    }

    fn current_sample(&self, m: &Machine) -> u16 {
        match self.format & 3 {
            0 => {
                let v = Self::aram(m, self.cur >> 1) as u16;
                if self.cur & 1 != 0 { v & 0xF } else { v >> 4 }
            }
            1 => Self::aram(m, self.cur) as u16,
            2 => (Self::aram(m, self.cur.wrapping_mul(2)) as u16) << 8 | Self::aram(m, self.cur.wrapping_mul(2) + 1) as u16,
            _ => 0,
        }
    }

    fn read_sample(&mut self, m: &Machine, pb: &mut Pb) -> i16 {
        if self.stopped {
            return 0;
        }
        let decode = (self.format >> 2) & 3;
        let raw = if decode == 1 || decode == 3 { 0 } else { self.current_sample(m) as i16 };
        let ci = ((self.pred_scale >> 4) & 7) as usize;
        let coef1 = pb.w[pb::COEFS + 2 * ci] as i16 as i32;
        let coef2 = pb.w[pb::COEFS + 2 * ci + 1] as i16 as i32;
        let val: i16;
        let mut step: u32;
        match decode {
            0 => {
                let mut nib = (raw & 0xF) as i32;
                if nib >= 8 {
                    nib -= 16;
                }
                let scale = 1i32 << (self.pred_scale & 0xF);
                let v = scale * nib + ((0x400 + coef1 * self.yn1 as i32 + coef2 * self.yn2 as i32) >> 11);
                val = v.clamp(-0x8000, 0x7FFF) as i16;
                step = 2;
                self.yn2 = self.yn1;
                self.yn1 = val;
                self.cur = self.cur.wrapping_add(1);
                if self.end & 0xF == 0 && self.cur == self.end {
                    self.cur = self.start + 1;
                } else if self.end & 0xF == 1 && self.cur == self.end.wrapping_sub(1) {
                    self.cur = self.start;
                } else if self.cur & 15 == 0 {
                    self.pred_scale = Self::aram(m, (self.cur & !15) >> 1) as u16;
                    self.cur = self.cur.wrapping_add(2);
                    step += 2;
                }
            }
            _ => {
                let shift = match (self.format >> 4) & 3 {
                    1 => 0,
                    2 => 16,
                    _ => 11,
                };
                let v = ((self.gain as i32 * raw as i32) >> shift)
                    + ((coef1 * self.yn1 as i32) >> shift)
                    + ((coef2 * self.yn2 as i32) >> shift);
                val = v as i16;
                self.yn2 = self.yn1;
                self.yn1 = val;
                step = 2;
                if decode != 1 {
                    self.cur = self.cur.wrapping_add(1);
                }
            }
        }
        if self.cur == self.end.wrapping_add(step).wrapping_sub(1) {
            self.cur = self.start;
            self.stopped = true;
            // End of sample: loop (reloading the loop predictor) or stop the voice.
            if pb.w[pb::LOOPING] != 0 {
                self.pred_scale = pb.w[pb::LOOP_PRED_SCALE] & 0x7F;
                if pb.w[pb::IS_STREAM] != 1 {
                    self.yn1 = pb.w[pb::LOOP_YN1] as i16;
                    self.yn2 = pb.w[pb::LOOP_YN2] as i16;
                } else {
                    pb.w[pb::LOOP_COUNTER] = pb.w[pb::LOOP_COUNTER].wrapping_add(1);
                }
                self.stopped = false;
            } else {
                pb.w[pb::RUNNING] = 0;
            }
        }
        self.cur &= 0xBFFF_FFFF;
        val
    }
}

/// One millisecond of one voice: decode + resample, envelope, filter, mix into the buses.
fn process_voice(m: &Machine, pb: &mut Pb, buses: &mut [[i32; FRAME]; 9], at: usize, mctrl: u32) {
    if pb.w[pb::RUNNING] != 1 {
        return;
    }
    let mut acc = Accel::new(pb);
    let mut samples = [0i16; SPMS];
    let ratio = (pb.w[pb::RATIO_HI] as u32) << 16 | pb.w[pb::RATIO_LO] as u32;
    let mut pos = pb.w[pb::FRAC] as u32;
    let mut last: [i16; 4] = std::array::from_fn(|i| pb.w[pb::LAST + i] as i16);
    let src = pb.w[pb::SRC_TYPE];
    if src == 2 {
        // Nearest: one input sample per output sample.
        for s in samples.iter_mut() {
            *s = acc.read_sample(m, pb);
        }
        last.copy_from_slice(&samples[SPMS - 4..]);
    } else {
        // Linear (also used for polyphase: the DSP ROM's polyphase coefficients aren't shipped).
        let mut temp = last;
        let mut idx = 4usize;
        for out in samples.iter_mut() {
            pos = pos.wrapping_add(ratio);
            while pos >= 0x10000 {
                temp[idx & 3] = acc.read_sample(m, pb);
                idx += 1;
                pos -= 0x10000;
            }
            let frac = pos & 0xFFFF;
            if frac != 0 {
                let s0 = temp[idx & 3] as i32;
                let s1 = temp[(idx + 1) & 3] as i32;
                let inv = (0x10000 - frac) & 0xFFFF;
                *out = ((s0 * inv as i32 + s1 * frac as i32) >> 16) as i16;
            } else {
                *out = temp[idx & 3];
            }
        }
        for k in 0..4 {
            last[3 - k] = temp[(idx + 3 - k) & 3];
        }
    }
    pb.w[pb::FRAC] = pos as u16;
    for (i, v) in last.iter().enumerate() {
        pb.w[pb::LAST + i] = *v as u16;
    }
    pb.w[pb::CUR_HI] = (acc.cur >> 16) as u16;
    pb.w[pb::CUR_LO] = acc.cur as u16;
    pb.w[pb::YN1] = acc.yn1 as u16;
    pb.w[pb::YN2] = acc.yn2 as u16;
    pb.w[pb::PRED_SCALE] = acc.pred_scale;

    for s in samples.iter_mut() {
        let vol = pb.w[pb::VOL_CUR] as i16 as i32;
        *s = clamp16(((*s as i32 * vol) >> 15) as i64);
        pb.w[pb::VOL_CUR] = pb.w[pb::VOL_CUR].wrapping_add(pb.w[pb::VOL_DELTA]);
    }
    if pb.w[pb::LPF_ON] != 0 {
        let a0 = pb.w[pb::LPF_A0] as i32;
        let b0 = pb.w[pb::LPF_B0] as i16 as i32;
        for s in samples.iter_mut() {
            let y = clamp16(((a0 * *s as i32 + b0 * pb.w[pb::LPF_YN1] as i16 as i32) >> 15) as i64);
            *s = y;
            pb.w[pb::LPF_YN1] = y as u16;
        }
    }
    // (mix flag, bus, mixer pair index, depop index)
    const ROUTES: [(u32, usize, usize, usize); 9] = [
        (MIX_MAIN_L, MAIN_L, 0, 0),
        (MIX_MAIN_R, MAIN_R, 1, 3),
        (MIX_MAIN_S, MAIN_S, 7, 6),
        (MIX_AUXA_L, AUXA_L, 2, 1),
        (MIX_AUXA_R, AUXA_L + 1, 3, 4),
        (MIX_AUXA_S, AUXA_L + 2, 8, 7),
        (MIX_AUXB_L, AUXB_L, 4, 2),
        (MIX_AUXB_R, AUXB_R, 5, 5),
        (MIX_AUXB_S, AUXB_S, 6, 8),
    ];
    for (flag, bus, pair, dpop) in ROUTES {
        if mctrl & flag == 0 {
            continue;
        }
        let ramp = mctrl & (flag << 1) != 0;
        let vi = pb::MIXER + 2 * pair;
        let delta = if ramp { pb.w[vi + 1] } else { 0 };
        let mut last = 0i16;
        for (k, &s) in samples.iter().enumerate() {
            let v = clamp16((s as i64 * pb.w[vi] as i64) >> 15);
            buses[bus][at + k] = buses[bus][at + k].wrapping_add(v as i32);
            pb.w[vi] = pb.w[vi].wrapping_add(delta);
            last = v;
        }
        pb.w[pb::DPOP + dpop] = last as u16;
    }
}
