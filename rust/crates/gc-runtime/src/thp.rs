//! Native THP video frame decoder (HLE of the SDK's `THPVideoDecode`).
//!
//! THP video frames are baseline JPEG (Huffman, 8-bit, usually 4:2:0) with one quirk: the
//! entropy-coded segment is not byte-stuffed. The SDK decoder writes the Y, U and V planes as
//! GX I8 textures (8x4-texel tiles, 32 bytes each) with the usual +128 level shift. The
//! recompiled SDK decoder works but runs well below real time (it leans on paired singles and
//! the locked cache), which starves the movie's audio; this does the same job natively.

use crate::Machine;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Frames decoded natively, and total host time spent (µs), for the trace.
pub static FRAMES: AtomicU64 = AtomicU64::new(0);
pub static MICROS: AtomicU64 = AtomicU64::new(0);

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
    54, 47, 55, 62, 63,
];

#[derive(Clone)]
struct Huff {
    maxcode: [i32; 18],
    valptr: [i32; 17],
    mincode: [i32; 17],
    vals: Vec<u8>,
}

impl Huff {
    fn new(bits: &[u8], vals: Vec<u8>) -> Huff {
        let mut h = Huff { maxcode: [-1; 18], valptr: [0; 17], mincode: [0; 17], vals };
        let (mut code, mut k) = (0i32, 0i32);
        for l in 1..=16 {
            let n = bits[l - 1] as i32;
            if n > 0 {
                h.valptr[l] = k;
                h.mincode[l] = code;
                code += n;
                k += n;
                h.maxcode[l] = code - 1;
            }
            code <<= 1;
        }
        h.maxcode[17] = i32::MAX;
        h
    }
}

struct Bits<'a> {
    d: &'a [u8],
    pos: usize,
    acc: u32,
    n: u32,
}

impl Bits<'_> {
    #[inline]
    fn bit(&mut self) -> i32 {
        if self.n == 0 {
            self.acc = *self.d.get(self.pos).unwrap_or(&0) as u32;
            self.pos += 1;
            self.n = 8;
        }
        self.n -= 1;
        ((self.acc >> self.n) & 1) as i32
    }
    fn get(&mut self, s: u32) -> i32 {
        let mut v = 0;
        for _ in 0..s {
            v = (v << 1) | self.bit();
        }
        v
    }
    fn decode(&mut self, h: &Huff) -> Option<u8> {
        let mut code = self.bit();
        let mut l = 1;
        while code > h.maxcode[l] {
            code = (code << 1) | self.bit();
            l += 1;
            if l > 16 {
                return None;
            }
        }
        h.vals.get((h.valptr[l] + code - h.mincode[l]) as usize).copied()
    }
    /// JPEG "receive + extend" of an `s`-bit magnitude.
    fn extend(&mut self, s: u32) -> i32 {
        if s == 0 {
            return 0;
        }
        let v = self.get(s);
        if v < 1 << (s - 1) { v - (1 << s) + 1 } else { v }
    }
    fn align(&mut self) {
        self.n = 0;
    }
}

struct Comp {
    h: usize,
    v: usize,
    tq: usize,
    td: usize,
    ta: usize,
    pred: i32,
}

fn idct_table() -> &'static [[f32; 8]; 8] {
    static T: std::sync::OnceLock<[[f32; 8]; 8]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[0f32; 8]; 8];
        for (x, row) in t.iter_mut().enumerate() {
            for (u, c) in row.iter_mut().enumerate() {
                let cu = if u == 0 { std::f32::consts::FRAC_1_SQRT_2 } else { 1.0 };
                *c = cu / 2.0 * (((2 * x + 1) as f32 * u as f32 * std::f32::consts::PI) / 16.0).cos();
            }
        }
        t
    })
}

/// 8x8 inverse DCT of `blk` (natural order), level-shifted to 0..255.
fn idct(blk: &[f32; 64], out: &mut [u8; 64]) {
    let t = idct_table();
    let mut tmp = [0f32; 64];
    for y in 0..8 {
        for x in 0..8 {
            let mut s = 0.0;
            for u in 0..8 {
                s += t[x][u] * blk[y * 8 + u];
            }
            tmp[y * 8 + x] = s;
        }
    }
    for x in 0..8 {
        for y in 0..8 {
            let mut s = 0.0;
            for v in 0..8 {
                s += t[y][v] * tmp[v * 8 + x];
            }
            out[y * 8 + x] = (s + 128.0).round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Decode one THP video frame at `file` into the three I8 textures. Returns false if the
/// stream uses something this decoder doesn't handle (the caller then runs the SDK code).
pub fn decode(m: &Machine, file: u32, dst: [u32; 3]) -> bool {
    let t0 = std::time::Instant::now();
    let ok = decode_inner(m, file, dst).is_some();
    if ok {
        FRAMES.fetch_add(1, Relaxed);
        MICROS.fetch_add(t0.elapsed().as_micros() as u64, Relaxed);
    }
    ok
}

fn decode_inner(m: &Machine, file: u32, dst: [u32; 3]) -> Option<()> {
    let ram = m.ram();
    let start = Machine::ram_off(file)?;
    let d = &ram[start..];
    let be16 = |p: usize| -> Option<usize> { Some(((*d.get(p)? as usize) << 8) | *d.get(p + 1)? as usize) };

    let mut qt = [[0f32; 64]; 4];
    let mut dc: Vec<Option<Huff>> = vec![None; 4];
    let mut ac: Vec<Option<Huff>> = vec![None; 4];
    let mut comps: Vec<Comp> = Vec::new();
    let (mut width, mut height, mut restart) = (0usize, 0usize, 0usize);
    let mut p = 0usize;
    let scan_start;
    loop {
        if *d.get(p)? != 0xFF {
            return None;
        }
        while *d.get(p)? == 0xFF {
            p += 1;
        }
        let marker = d[p];
        p += 1;
        match marker {
            0xD8 => {}
            0xC0 => {
                let len = be16(p)?;
                height = be16(p + 3)?;
                width = be16(p + 5)?;
                let n = *d.get(p + 7)? as usize;
                comps.clear();
                for i in 0..n {
                    let c = p + 8 + 3 * i;
                    let hv = *d.get(c + 1)? as usize;
                    comps.push(Comp { h: hv >> 4, v: hv & 15, tq: (*d.get(c + 2)? as usize) & 3, td: 0, ta: 0, pred: 0 });
                }
                p += len;
            }
            0xC4 => {
                let len = be16(p)?;
                let end = p + len;
                let mut q = p + 2;
                while q < end {
                    let tc_th = *d.get(q)?;
                    let bits = d.get(q + 1..q + 17)?;
                    let total: usize = bits.iter().map(|&b| b as usize).sum();
                    let vals = d.get(q + 17..q + 17 + total)?.to_vec();
                    let h = Huff::new(bits, vals);
                    let id = (tc_th & 3) as usize;
                    if tc_th >> 4 == 0 { dc[id] = Some(h) } else { ac[id] = Some(h) }
                    q += 17 + total;
                }
                p = end;
            }
            0xDB => {
                let len = be16(p)?;
                let end = p + len;
                let mut q = p + 2;
                while q < end {
                    let pq_tq = *d.get(q)?;
                    let id = (pq_tq & 3) as usize;
                    q += 1;
                    for k in 0..64 {
                        let v = if pq_tq >> 4 == 0 {
                            *d.get(q + k)? as f32
                        } else {
                            be16(q + 2 * k)? as f32
                        };
                        qt[id][ZIGZAG[k]] = v;
                    }
                    q += if pq_tq >> 4 == 0 { 64 } else { 128 };
                }
                p = end;
            }
            0xDD => {
                restart = be16(p + 2)?;
                p += be16(p)?;
            }
            0xDA => {
                let len = be16(p)?;
                let ns = *d.get(p + 2)? as usize;
                // Map scan components to frame components (THP always scans all of them in order).
                if ns != comps.len() {
                    return None;
                }
                for i in 0..ns {
                    let t = *d.get(p + 4 + 2 * i)?;
                    comps[i].td = (t >> 4) as usize & 3;
                    comps[i].ta = (t & 15) as usize & 3;
                }
                scan_start = p + len;
                break;
            }
            0xE0..=0xEF | 0xFE => p += be16(p)?,
            _ => return None,
        }
    }
    if width == 0 || height == 0 || comps.len() != 3 || width % 8 != 0 {
        return None;
    }
    let hmax = comps.iter().map(|c| c.h).max()?;
    let vmax = comps.iter().map(|c| c.v).max()?;
    if hmax == 0 || vmax == 0 {
        return None;
    }
    let mcus_x = width.div_ceil(8 * hmax);
    let mcus_y = height.div_ceil(8 * vmax);
    // Destination planes (RAM offsets) and their sizes.
    let mut planes = [(0usize, 0usize, 0usize); 3];
    for (i, c) in comps.iter().enumerate() {
        let pw = width * c.h / hmax;
        let ph = height * c.v / vmax;
        let off = Machine::ram_off(dst[i])?;
        let size = pw * ph;
        if pw % 8 != 0 || off + size > crate::RAM_SIZE {
            return None;
        }
        planes[i] = (off, pw, ph);
    }
    for c in comps.iter() {
        if dc[c.td].is_none() || ac[c.ta].is_none() {
            return None;
        }
    }

    let mut bits = Bits { d, pos: scan_start, acc: 0, n: 0 };
    let mut out = Vec::new();
    // Decode into a scratch list of (plane, block x, block y, pixels), then write tiles.
    let mut blk = [0f32; 64];
    let mut px = [0u8; 64];
    let mut mcu = 0usize;
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            if restart > 0 && mcu > 0 && mcu % restart == 0 {
                bits.align();
                if d.get(bits.pos) == Some(&0xFF) && d.get(bits.pos + 1).is_some_and(|b| (0xD0..=0xD7).contains(b)) {
                    bits.pos += 2;
                }
                for c in comps.iter_mut() {
                    c.pred = 0;
                }
            }
            mcu += 1;
            for (ci, c) in comps.iter_mut().enumerate() {
                let (dct, act) = (dc[c.td].as_ref()?, ac[c.ta].as_ref()?);
                let q = &qt[c.tq];
                for v in 0..c.v {
                    for h in 0..c.h {
                        blk.fill(0.0);
                        let s = bits.decode(dct)? as u32;
                        c.pred += bits.extend(s);
                        blk[0] = c.pred as f32 * q[0];
                        let mut k = 1;
                        while k < 64 {
                            let rs = bits.decode(act)?;
                            let (r, s) = ((rs >> 4) as usize, (rs & 15) as u32);
                            if s == 0 {
                                if r == 15 {
                                    k += 16;
                                    continue;
                                }
                                break;
                            }
                            k += r;
                            if k > 63 {
                                break;
                            }
                            let z = ZIGZAG[k];
                            blk[z] = bits.extend(s) as f32 * q[z];
                            k += 1;
                        }
                        idct(&blk, &mut px);
                        out.push((ci, (mx * c.h + h) * 8, (my * c.v + v) * 8, px));
                    }
                }
            }
        }
    }
    // Write the planes as I8 textures: 8x4 tiles in row-major tile order, 32 bytes per tile.
    let ram = m.ram();
    for (ci, bx, by, px) in &out {
        let (off, pw, ph) = planes[*ci];
        let tiles_per_row = pw / 8;
        for y in 0..8 {
            let yy = by + y;
            if yy >= ph {
                break;
            }
            let x0 = *bx;
            if x0 >= pw {
                continue;
            }
            let tile = (yy / 4) * tiles_per_row + x0 / 8;
            let o = off + tile * 32 + (yy % 4) * 8;
            ram[o..o + 8].copy_from_slice(&px[y * 8..y * 8 + 8]);
        }
    }
    Some(())
}
