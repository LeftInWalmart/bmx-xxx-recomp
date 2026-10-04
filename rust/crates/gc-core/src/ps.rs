//! Native Gekko paired-single instructions.
//!
//! The GameCubeLoader SLEIGH spec gets several of these wrong (e.g. `ps_merge00 f0,f1,f0`
//! writes ps0 before reading it for ps1), which corrupts the matrix math the game's 3D library
//! relies on. The generator therefore emits `c.ps(word)` for every `ps_*` / `psq_*` / `dcbz_l`
//! instruction, and this module implements them from the instruction encoding, reading all
//! operands before writing (alias-safe) and rounding arithmetic results to single precision
//! like the hardware.
//!
//! Everything here is `#[inline(always)]`: the generator passes the instruction word as a
//! constant, so after inlining the decode folds away and only the arithmetic remains.

use crate::Ctx;

const FPR: usize = 0x4000;
const PS1: usize = 0x5000;
const GQR: usize = 0x1E40;
const CR: usize = 0x900;

#[inline(always)]
fn ps0(c: &Ctx, r: u32) -> f64 {
    f64::from_bits(c.r(FPR + 8 * r as usize, 8))
}
#[inline(always)]
fn ps1(c: &Ctx, r: u32) -> f64 {
    f64::from_bits(c.r(PS1 + 8 * r as usize, 8))
}
#[inline(always)]
fn set(c: &mut Ctx, r: u32, a: f64, b: f64) {
    c.wr(FPR + 8 * r as usize, 8, a.to_bits());
    c.wr(PS1 + 8 * r as usize, 8, b.to_bits());
}
#[inline(always)]
fn single(x: f64) -> f64 {
    x as f32 as f64
}
#[inline(always)]
fn gpr(c: &Ctx, r: u32) -> u32 {
    c.gpr(r as usize)
}

/// 2^n for the quantisation scales (n in -32..=31), exactly.
#[inline(always)]
fn pow2(n: i32) -> f64 {
    f64::from_bits(((1023 + n) as u64) << 52)
}

/// Dequantise per GQR load type/scale.
#[inline(always)]
fn load_q(c: &mut Ctx, ea: u32, ty: u32, scale: i32) -> (f64, u32) {
    match ty {
        4 => (c.ld(ea as u64, 1) as u8 as f64 * pow2(-scale), 1),
        5 => (c.ld(ea as u64, 2) as u16 as f64 * pow2(-scale), 2),
        6 => (c.ld(ea as u64, 1) as u8 as i8 as f64 * pow2(-scale), 1),
        7 => (c.ld(ea as u64, 2) as u16 as i16 as f64 * pow2(-scale), 2),
        _ => (f32::from_bits(c.ld(ea as u64, 4) as u32) as f64, 4),
    }
}

/// Quantise per GQR store type/scale (truncate, then saturate).
#[inline(always)]
fn store_q(c: &mut Ctx, ea: u32, v: f64, ty: u32, scale: i32) -> u32 {
    let x = v * pow2(scale);
    match ty {
        4 => {
            c.st(ea as u64, 1, x.clamp(0.0, 255.0) as u64);
            1
        }
        5 => {
            c.st(ea as u64, 2, x.clamp(0.0, 65535.0) as u64);
            2
        }
        6 => {
            c.st(ea as u64, 1, (x.clamp(-128.0, 127.0) as i64 as u64) & 0xFF);
            1
        }
        7 => {
            c.st(ea as u64, 2, (x.clamp(-32768.0, 32767.0) as i64 as u64) & 0xFFFF);
            2
        }
        _ => {
            c.st(ea as u64, 4, (v as f32).to_bits() as u64);
            4
        }
    }
}

#[inline(always)]
fn sext6(v: u32) -> i32 {
    ((v << 26) as i32) >> 26
}

#[inline(always)]
fn psq_load(c: &mut Ctx, fd: u32, ea: u32, w: bool, i: u32) {
    let g = c.r(GQR + 4 * i as usize, 4) as u32;
    let (ty, scale) = ((g >> 16) & 7, sext6((g >> 24) & 0x3F));
    let (a, size) = load_q(c, ea, ty, scale);
    let b = if w { 1.0 } else { load_q(c, ea.wrapping_add(size), ty, scale).0 };
    set(c, fd, a, b);
}

#[inline(always)]
fn psq_store(c: &mut Ctx, fs: u32, ea: u32, w: bool, i: u32) {
    let g = c.r(GQR + 4 * i as usize, 4) as u32;
    let (ty, scale) = (g & 7, sext6((g >> 8) & 0x3F));
    let (a, b) = (ps0(c, fs), ps1(c, fs));
    let size = store_q(c, ea, a, ty, scale);
    if !w {
        store_q(c, ea.wrapping_add(size), b, ty, scale);
    }
}

#[inline(always)]
fn cmp(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        1
    } else if a < b {
        8
    } else if a > b {
        4
    } else {
        2
    }
}

#[cold]
#[inline(never)]
fn ps_unknown(c: &mut Ctx, w: u32) {
    (c.host.log)(format!("ps: unhandled encoding {w:#010x}"));
}

impl Ctx {
    /// Execute one paired-single (or related) instruction word.
    #[inline(always)]
    pub fn ps(&mut self, w: u32) {
        let c = self;
        let op = w >> 26;
        let d = (w >> 21) & 31;
        let a = (w >> 16) & 31;
        let b = (w >> 11) & 31;
        match op {
            56 | 57 | 60 | 61 => {
                let off = (((w & 0xFFF) << 20) as i32 >> 20) as u32;
                let base = if a == 0 && (op == 56 || op == 60) { 0 } else { gpr(c, a) };
                let ea = base.wrapping_add(off);
                let (wb, i) = ((w >> 15) & 1 != 0, (w >> 12) & 7);
                if op < 60 {
                    psq_load(c, d, ea, wb, i);
                } else {
                    psq_store(c, d, ea, wb, i);
                }
                if op == 57 || op == 61 {
                    c.set_gpr(a as usize, ea);
                }
                return;
            }
            4 => {}
            _ => {
                ps_unknown(c, w);
                return;
            }
        }
        // Indexed quantised loads/stores (XO in bits 1-6).
        let xo6 = (w >> 1) & 0x3F;
        if matches!(xo6, 6 | 7 | 38 | 39) {
            let base = if a == 0 && xo6 < 32 { 0 } else { gpr(c, a) };
            let ea = base.wrapping_add(gpr(c, b));
            let (wb, i) = ((w >> 10) & 1 != 0, (w >> 7) & 7);
            if xo6 & 1 == 0 {
                psq_load(c, d, ea, wb, i);
            } else {
                psq_store(c, d, ea, wb, i);
            }
            if xo6 >= 32 {
                c.set_gpr(a as usize, ea);
            }
            return;
        }
        let xo5 = (w >> 1) & 0x1F;
        let (a0, a1, b0, b1) = (ps0(c, a), ps1(c, a), ps0(c, b), ps1(c, b));
        if matches!(xo5, 0 | 8 | 16 | 22) {
            let xo10 = (w >> 1) & 0x3FF;
            match xo10 {
                0 | 32 => c.wr(CR + (d >> 2) as usize, 1, cmp(a0, b0)),
                64 | 96 => c.wr(CR + (d >> 2) as usize, 1, cmp(a1, b1)),
                40 => set(c, d, -b0, -b1),
                72 => set(c, d, b0, b1),
                136 => set(c, d, -b0.abs(), -b1.abs()),
                264 => set(c, d, b0.abs(), b1.abs()),
                528 => set(c, d, a0, b0),
                560 => set(c, d, a0, b1),
                592 => set(c, d, a1, b0),
                624 => set(c, d, a1, b1),
                1014 => {
                    let base = if a == 0 { 0 } else { gpr(c, a) };
                    let ea = (base.wrapping_add(gpr(c, b)) & !31) as u64;
                    for k in 0..4 {
                        c.st(ea + 8 * k, 8, 0);
                    }
                }
                _ => ps_unknown(c, w),
            }
            return;
        }
        let cr = (w >> 6) & 31;
        let (c0, c1) = (ps0(c, cr), ps1(c, cr));
        let (r0, r1) = match xo5 {
            10 => (a0 + b1, c1),
            11 => (c0, a0 + b1),
            12 => (a0 * c0, a1 * c0),
            13 => (a0 * c1, a1 * c1),
            14 => (a0 * c0 + b0, a1 * c0 + b1),
            15 => (a0 * c1 + b0, a1 * c1 + b1),
            18 => (a0 / b0, a1 / b1),
            20 => (a0 - b0, a1 - b1),
            21 => (a0 + b0, a1 + b1),
            23 => {
                let sel = |x: f64, t: f64, f: f64| if x >= 0.0 { t } else { f };
                set(c, d, sel(a0, c0, b0), sel(a1, c1, b1));
                return;
            }
            24 => (1.0 / b0, 1.0 / b1),
            25 => (a0 * c0, a1 * c1),
            26 => (1.0 / b0.sqrt(), 1.0 / b1.sqrt()),
            28 => (a0 * c0 - b0, a1 * c1 - b1),
            29 => (a0 * c0 + b0, a1 * c1 + b1),
            30 => (-(a0 * c0 - b0), -(a1 * c1 - b1)),
            31 => (-(a0 * c0 + b0), -(a1 * c1 + b1)),
            _ => {
                ps_unknown(c, w);
                return;
            }
        };
        set(c, d, single(r0), single(r1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Host;

    fn nop_rw(_: &mut Ctx, _: u32, _: usize) -> u64 { 0 }
    fn nop_w(_: &mut Ctx, _: u32, _: usize, _: u64) {}
    fn nop(_: &mut Ctx) {}
    fn no_hle(_: &mut Ctx, _: u32) -> bool { false }
    fn no_intr(_: &mut Ctx, _: &'static str, _: &[u64]) {}
    fn no_log(m: String) { panic!("{m}") }
    fn tb() -> u64 { 0 }
    static HOST: Host = Host {
        mmio_read: nop_rw, mmio_write: nop_w, deliver: nop, hle: no_hle,
        dispatch: no_hle, intrinsic: no_intr, log: no_log, timebase: tb,
    };

    fn ctx(ram: &mut Vec<u8>) -> Ctx {
        Ctx::new(0x5200, ram.as_mut_ptr(), &HOST)
    }
    fn x(op: u32, d: u32, a: u32, b: u32, xo10: u32) -> u32 {
        (op << 26) | (d << 21) | (a << 16) | (b << 11) | (xo10 << 1)
    }
    fn af(d: u32, a: u32, b: u32, c: u32, xo5: u32) -> u32 {
        (4 << 26) | (d << 21) | (a << 16) | (b << 11) | (c << 6) | (xo5 << 1)
    }

    #[test]
    fn merge_alias_safe() {
        let mut ram = vec![0u8; crate::RAM_SIZE];
        let mut c = ctx(&mut ram);
        set(&mut c, 0, 1.0, 2.0);
        set(&mut c, 1, 3.0, 4.0);
        c.ps(x(4, 0, 1, 0, 528)); // ps_merge00 f0,f1,f0 -> (f1.ps0, old f0.ps0)
        assert_eq!((ps0(&c, 0), ps1(&c, 0)), (3.0, 1.0));
        set(&mut c, 0, 1.0, 2.0);
        c.ps(x(4, 0, 0, 1, 624)); // ps_merge11 f0,f0,f1 -> (f0.ps1, f1.ps1)
        assert_eq!((ps0(&c, 0), ps1(&c, 0)), (2.0, 4.0));
    }

    #[test]
    fn arithmetic() {
        let mut ram = vec![0u8; crate::RAM_SIZE];
        let mut c = ctx(&mut ram);
        set(&mut c, 1, 2.0, 3.0);
        set(&mut c, 2, 10.0, 20.0);
        set(&mut c, 3, 0.5, 4.0);
        c.ps(af(4, 1, 2, 3, 12)); // ps_muls0 f4 = f1 * f3.ps0
        assert_eq!((ps0(&c, 4), ps1(&c, 4)), (1.0, 1.5));
        c.ps(af(4, 1, 2, 3, 14)); // ps_madds0 f4 = f1*f3.ps0 + f2
        assert_eq!((ps0(&c, 4), ps1(&c, 4)), (11.0, 21.5));
        c.ps(af(4, 1, 2, 3, 10)); // ps_sum0 f4 = (f1.ps0 + f2.ps1, f3.ps1)
        assert_eq!((ps0(&c, 4), ps1(&c, 4)), (22.0, 4.0));
        c.ps(af(1, 1, 2, 3, 29)); // ps_madd f1 = f1*f3 + f2 (dest aliases A)
        assert_eq!((ps0(&c, 1), ps1(&c, 1)), (11.0, 32.0));
        set(&mut c, 5, -1.0, 1.0);
        c.ps(af(6, 5, 2, 3, 23)); // ps_sel f6 = f5 >= 0 ? f3 : f2
        assert_eq!((ps0(&c, 6), ps1(&c, 6)), (10.0, 4.0));
    }

    #[test]
    fn quantized_load_store() {
        let mut ram = vec![0u8; crate::RAM_SIZE];
        let mut c = ctx(&mut ram);
        // GQR1: load s16 scale 8, store s16 scale 8.
        c.wr(GQR + 4, 4, (8 << 24) | (7 << 16) | (8 << 8) | 7);
        c.set_gpr(3, 0x8000_1000);
        set(&mut c, 2, 1.5, -2.25);
        let st = (60 << 26) | (2 << 21) | (3 << 16) | (1 << 12); // psq_st f2,0(r3),0,1
        c.ps(st);
        let ld = (56 << 26) | (7 << 21) | (3 << 16) | (1 << 12); // psq_l f7,0(r3),0,1
        c.ps(ld);
        assert_eq!((ps0(&c, 7), ps1(&c, 7)), (1.5, -2.25));
        // GQR0 = float, W=1 loads ps1 = 1.0.
        c.wr(GQR, 4, 0);
        c.st(0x8000_1100, 4, 3.25f32.to_bits() as u64);
        c.set_gpr(4, 0x8000_1100);
        c.ps((56 << 26) | (8 << 21) | (4 << 16) | (1 << 15));
        assert_eq!((ps0(&c, 8), ps1(&c, 8)), (3.25, 1.0));
    }
}
