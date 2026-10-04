//! Native versions of the SDK's paired-single matrix/vector library (MTX), HLE'd because they
//! sit at the top of the guest profile in 3D scenes (every ps instruction otherwise goes through
//! the runtime paired-single decoder).
//!
//! Matrices are 3x4 row-major f32 (`Mtx`), vectors 3 f32 (`Vec`). Results are rounded to single
//! precision after every multiply-add, in the order the SDK assembly accumulates, so values
//! match the paired-single code. Outputs may alias inputs (all inputs are read first).

use crate::{Ctx, CtxExt, Machine};

const FPR1: usize = 0x4000 + 8;
const PS1_1: usize = 0x5000 + 8;

fn rd<const N: usize>(m: &Machine, a: u32) -> [f32; N] {
    std::array::from_fn(|i| f32::from_bits(m.read_be(a + 4 * i as u32, 4) as u32))
}

fn wr(m: &Machine, a: u32, v: &[f32]) {
    for (i, x) in v.iter().enumerate() {
        m.write_be(a + 4 * i as u32, 4, x.to_bits() as u64);
    }
}

/// Fused multiply-add rounded to single, like ps_madd / fmadds.
#[inline]
fn madd(a: f32, b: f32, c: f32) -> f32 {
    (a as f64 * b as f64 + c as f64) as f32
}

#[inline]
fn mul(a: f32, b: f32) -> f32 {
    (a as f64 * b as f64) as f32
}

fn set_f1(c: &mut Ctx, v: f32) {
    let bits = (v as f64).to_bits();
    c.wr(FPR1, 8, bits);
    c.wr(PS1_1, 8, bits);
}

/// Handle a MTX library function by name; false if not one of ours.
pub fn call(c: &mut Ctx, name: &str) -> bool {
    let m = c.m();
    match name {
        "PSMTXCopy" => {
            let v = rd::<12>(m, c.gpr(3));
            wr(m, c.gpr(4), &v);
        }
        "PSMTXConcat" => {
            let (a, b) = (rd::<12>(m, c.gpr(3)), rd::<12>(m, c.gpr(4)));
            let mut o = [0f32; 12];
            for r in 0..3 {
                for k in 0..4 {
                    let mut v = mul(a[4 * r], b[k]);
                    v = madd(a[4 * r + 1], b[4 + k], v);
                    v = madd(a[4 * r + 2], b[8 + k], v);
                    if k == 3 {
                        v = (v as f64 + a[4 * r + 3] as f64) as f32;
                    }
                    o[4 * r + k] = v;
                }
            }
            wr(m, c.gpr(5), &o);
        }
        "PSMTXMultVec" => {
            let (mt, v) = (rd::<12>(m, c.gpr(3)), rd::<3>(m, c.gpr(4)));
            let mut o = [0f32; 3];
            for r in 0..3 {
                let mut x = madd(mt[4 * r], v[0], mt[4 * r + 3]);
                x = madd(mt[4 * r + 1], v[1], x);
                o[r] = madd(mt[4 * r + 2], v[2], x);
            }
            wr(m, c.gpr(5), &o);
        }
        "PSVECCrossProduct" => {
            let (a, b) = (rd::<3>(m, c.gpr(3)), rd::<3>(m, c.gpr(4)));
            let o = [
                madd(a[1], b[2], -mul(a[2], b[1])),
                madd(a[2], b[0], -mul(a[0], b[2])),
                madd(a[0], b[1], -mul(a[1], b[0])),
            ];
            wr(m, c.gpr(5), &o);
        }
        "PSVECMag" => {
            let v = rd::<3>(m, c.gpr(3));
            let d = madd(v[2], v[2], madd(v[1], v[1], mul(v[0], v[0])));
            set_f1(c, if d > 0.0 { d.sqrt() } else { 0.0 });
        }
        "PSMTXInverse" => {
            let a = rd::<12>(m, c.gpr(3));
            let g = |r: usize, k: usize| a[4 * r + k] as f64;
            let det = g(0, 0) * g(1, 1) * g(2, 2) + g(0, 1) * g(1, 2) * g(2, 0) + g(0, 2) * g(1, 0) * g(2, 1)
                - g(2, 0) * g(1, 1) * g(0, 2)
                - g(1, 0) * g(0, 1) * g(2, 2)
                - g(0, 0) * g(2, 1) * g(1, 2);
            if det == 0.0 {
                c.set_gpr(3, 0);
                return true;
            }
            let d = 1.0 / det;
            let mut o = [0f64; 12];
            o[0] = (g(1, 1) * g(2, 2) - g(2, 1) * g(1, 2)) * d;
            o[1] = -(g(0, 1) * g(2, 2) - g(2, 1) * g(0, 2)) * d;
            o[2] = (g(0, 1) * g(1, 2) - g(1, 1) * g(0, 2)) * d;
            o[4] = -(g(1, 0) * g(2, 2) - g(2, 0) * g(1, 2)) * d;
            o[5] = (g(0, 0) * g(2, 2) - g(2, 0) * g(0, 2)) * d;
            o[6] = -(g(0, 0) * g(1, 2) - g(1, 0) * g(0, 2)) * d;
            o[8] = (g(1, 0) * g(2, 1) - g(2, 0) * g(1, 1)) * d;
            o[9] = -(g(0, 0) * g(2, 1) - g(2, 0) * g(0, 1)) * d;
            o[10] = (g(0, 0) * g(1, 1) - g(1, 0) * g(0, 1)) * d;
            for r in 0..3 {
                o[4 * r + 3] = -(o[4 * r] * g(0, 3) + o[4 * r + 1] * g(1, 3) + o[4 * r + 2] * g(2, 3));
            }
            let out: [f32; 12] = o.map(|x| x as f32);
            wr(m, c.gpr(4), &out);
            c.set_gpr(3, 1);
        }
        _ => return false,
    }
    true
}

#[cfg(test)]
mod tests {
    #[test]
    fn madd_rounds_once() {
        assert_eq!(super::madd(1.5, 2.0, 0.25), 3.25);
    }
}
