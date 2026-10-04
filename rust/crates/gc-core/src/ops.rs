//! P-code operator semantics. Values are carried in `u64`, `s` is the operand size in bytes.

#[inline] pub fn mask(v: u64, s: usize) -> u64 { if s >= 8 { v } else { v & ((1u64 << (8 * s)) - 1) } }
#[inline] pub fn sext(v: u64, s: usize) -> i64 { if s >= 8 { v as i64 } else { let sh = 64 - 8 * s; ((v << sh) as i64) >> sh } }
#[inline] fn b(x: bool) -> u64 { x as u64 }

#[inline] pub fn int_add(a: u64, c: u64, s: usize) -> u64 { mask(a.wrapping_add(c), s) }
#[inline] pub fn int_sub(a: u64, c: u64, s: usize) -> u64 { mask(a.wrapping_sub(c), s) }
#[inline] pub fn int_mult(a: u64, c: u64, s: usize) -> u64 { mask(a.wrapping_mul(c), s) }
#[inline] pub fn int_div(a: u64, c: u64, s: usize) -> u64 { if c == 0 { 0 } else { mask(mask(a, s) / mask(c, s), s) } }
#[inline] pub fn int_rem(a: u64, c: u64, s: usize) -> u64 { if c == 0 { 0 } else { mask(mask(a, s) % mask(c, s), s) } }
#[inline] pub fn int_sdiv(a: u64, c: u64, s: usize) -> u64 { let d = sext(c, s); if d == 0 { 0 } else { mask(sext(a, s).wrapping_div(d) as u64, s) } }
#[inline] pub fn int_srem(a: u64, c: u64, s: usize) -> u64 { let d = sext(c, s); if d == 0 { 0 } else { mask(sext(a, s).wrapping_rem(d) as u64, s) } }
#[inline] pub fn int_and(a: u64, c: u64, _: usize) -> u64 { a & c }
#[inline] pub fn int_or(a: u64, c: u64, _: usize) -> u64 { a | c }
#[inline] pub fn int_xor(a: u64, c: u64, _: usize) -> u64 { a ^ c }
#[inline] pub fn int_negate(a: u64, s: usize) -> u64 { mask(!a, s) }
#[inline] pub fn int_2comp(a: u64, s: usize) -> u64 { mask(a.wrapping_neg(), s) }
#[inline] pub fn int_left(a: u64, c: u64, s: usize) -> u64 { if c >= 64 { 0 } else { mask(a << c, s) } }
#[inline] pub fn int_right(a: u64, c: u64, s: usize) -> u64 { if c >= 64 { 0 } else { mask(a, s) >> c } }
#[inline] pub fn int_sright(a: u64, c: u64, s: usize) -> u64 { mask((sext(a, s) >> c.min(63)) as u64, s) }

#[inline] pub fn int_equal(a: u64, c: u64, _: usize) -> u64 { b(a == c) }
#[inline] pub fn int_notequal(a: u64, c: u64, _: usize) -> u64 { b(a != c) }
#[inline] pub fn int_less(a: u64, c: u64, s: usize) -> u64 { b(mask(a, s) < mask(c, s)) }
#[inline] pub fn int_lessequal(a: u64, c: u64, s: usize) -> u64 { b(mask(a, s) <= mask(c, s)) }
#[inline] pub fn int_sless(a: u64, c: u64, s: usize) -> u64 { b(sext(a, s) < sext(c, s)) }
#[inline] pub fn int_slessequal(a: u64, c: u64, s: usize) -> u64 { b(sext(a, s) <= sext(c, s)) }
#[inline] pub fn int_carry(a: u64, c: u64, s: usize) -> u64 { b(mask(a, s) > mask(!c, s)) }
#[inline] pub fn int_scarry(a: u64, c: u64, s: usize) -> u64 { let r = int_add(a, c, s); let t = 8 * s - 1; b(((a ^ r) & (c ^ r)) >> t & 1 == 1) }
#[inline] pub fn int_sborrow(a: u64, c: u64, s: usize) -> u64 { let r = int_sub(a, c, s); let t = 8 * s - 1; b(((a ^ c) & (a ^ r)) >> t & 1 == 1) }
#[inline] pub fn int_zext(a: u64, s: usize) -> u64 { mask(a, s) }
#[inline] pub fn int_sext(a: u64, s: usize) -> u64 { sext(a, s) as u64 }

#[inline] pub fn bool_negate(a: u64, _: usize) -> u64 { b(a & 1 == 0) }
#[inline] pub fn bool_and(a: u64, c: u64, _: usize) -> u64 { a & c & 1 }
#[inline] pub fn bool_or(a: u64, c: u64, _: usize) -> u64 { (a | c) & 1 }
#[inline] pub fn bool_xor(a: u64, c: u64, _: usize) -> u64 { (a ^ c) & 1 }

#[inline] pub fn popcount(a: u64, s: usize) -> u64 { mask(a, s).count_ones() as u64 }
#[inline] pub fn lzcount(a: u64, s: usize) -> u64 { (mask(a, s).leading_zeros() as u64).saturating_sub(64 - 8 * s as u64) }
/// SUBPIECE: drop `c` low bytes, truncate to output size.
#[inline] pub fn subpiece(a: u64, c: u64, out: usize) -> u64 { mask(if c >= 8 { 0 } else { a >> (8 * c) }, out) }
/// PIECE: `hi` concatenated above `lo` (lo has size `ls`).
#[inline] pub fn piece(hi: u64, lo: u64, ls: usize) -> u64 { if ls >= 8 { lo } else { (hi << (8 * ls)) | mask(lo, ls) } }

// Floating point: size 4 => f32 bits, size 8 => f64 bits.
#[inline] pub fn tof(a: u64, s: usize) -> f64 { if s == 4 { f32::from_bits(a as u32) as f64 } else { f64::from_bits(a) } }
#[inline] pub fn fromf(v: f64, s: usize) -> u64 { if s == 4 { (v as f32).to_bits() as u64 } else { v.to_bits() } }

#[inline] pub fn float_add(a: u64, c: u64, s: usize) -> u64 { fromf(tof(a, s) + tof(c, s), s) }
#[inline] pub fn float_sub(a: u64, c: u64, s: usize) -> u64 { fromf(tof(a, s) - tof(c, s), s) }
#[inline] pub fn float_mult(a: u64, c: u64, s: usize) -> u64 { fromf(tof(a, s) * tof(c, s), s) }
#[inline] pub fn float_div(a: u64, c: u64, s: usize) -> u64 { fromf(tof(a, s) / tof(c, s), s) }
#[inline] pub fn float_neg(a: u64, s: usize) -> u64 { fromf(-tof(a, s), s) }
#[inline] pub fn float_abs(a: u64, s: usize) -> u64 { fromf(tof(a, s).abs(), s) }
#[inline] pub fn float_sqrt(a: u64, s: usize) -> u64 { fromf(tof(a, s).sqrt(), s) }
#[inline] pub fn float_ceil(a: u64, s: usize) -> u64 { fromf(tof(a, s).ceil(), s) }
#[inline] pub fn float_floor(a: u64, s: usize) -> u64 { fromf(tof(a, s).floor(), s) }
#[inline] pub fn float_round(a: u64, s: usize) -> u64 { fromf(tof(a, s).round(), s) }
#[inline] pub fn float_nan(a: u64, s: usize) -> u64 { b(tof(a, s).is_nan()) }
#[inline] pub fn float_equal(a: u64, c: u64, s: usize) -> u64 { b(tof(a, s) == tof(c, s)) }
#[inline] pub fn float_notequal(a: u64, c: u64, s: usize) -> u64 { b(tof(a, s) != tof(c, s)) }
#[inline] pub fn float_less(a: u64, c: u64, s: usize) -> u64 { b(tof(a, s) < tof(c, s)) }
#[inline] pub fn float_lessequal(a: u64, c: u64, s: usize) -> u64 { b(tof(a, s) <= tof(c, s)) }
/// FLOAT2FLOAT: input size `s` -> output size `out`.
#[inline] pub fn float2float(a: u64, s: usize, out: usize) -> u64 { fromf(tof(a, s), out) }
#[inline] pub fn int2float(a: u64, s: usize, out: usize) -> u64 { fromf(sext(a, s) as f64, out) }
#[inline] pub fn trunc(a: u64, s: usize, out: usize) -> u64 {
    let v = tof(a, s);
    let i = if out <= 4 { v as i32 as i64 } else { v as i64 };
    mask(i as u64, out)
}

/// GameCubeLoader `ldexpf` intrinsic: the paired-single dequantization scale 2^n.
#[inline] pub fn ldexp_pow2(a: u64, s: usize, out: usize) -> u64 { fromf(2f64.powi(sext(a, s) as i32), out) }
