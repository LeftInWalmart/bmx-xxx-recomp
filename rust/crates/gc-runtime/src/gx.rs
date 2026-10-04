//! GX command-stream decoder.
//!
//! Parses the FIFO the game writes to 0xCC008000 (and display lists it calls): CP / XF / BP
//! register loads and draw commands. For each draw it:
//!
//! * decodes vertices (direct and indexed attributes, all component/colour formats),
//! * transforms positions by the position matrix and XF projection to clip space,
//! * computes the colour channels with the game's XF lighting setup (ambient, material,
//!   directional/spot lights, diffuse and attenuation functions),
//! * generates texture coordinates through the texgen matrices,
//! * evaluates the TEV stages per vertex with the texture taken as white, which gives the
//!   factor the sampled texture is modulated by (exact for replace/modulate/decal setups,
//!   an approximation for exotic combiner use),
//! * decodes the referenced texture from RAM (I4, I8, IA4, IA8, RGB565, RGB5A3, RGBA8, C4, C8,
//!   C14X2, CMPR; TLUTs through emulated TMEM) and caches it,
//! * groups triangles into batches by texture + blend/alpha-test state.
//!
//! A display copy (`BP 0x52` with the copy-to-XFB bit) ends the frame and publishes it to
//! [`FrameShared`] for the Bevy renderer. Not handled: indirect textures, fog, EFB-copy
//! render targets (drawn untextured), per-pixel TEV, lines/points.

use crate::Machine;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

/// Device events the decoder produces for `hw.rs` (PE interrupts).
pub enum Event {
    DrawDone,
    Token(u16),
    TokenInt(u16),
}

/// Which render phase a batch needs: depth-tested opaque draws can be reordered freely;
/// blended or depth-test-disabled draws must stay in submission order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Blend {
    #[default]
    Opaque,
    Sorted,
}

/// Complete pixel-pipeline state of a batch, evaluated per pixel by the Bevy TEV shader.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct TevState {
    /// Per stage: [colour env, alpha env, order, ksel] where
    /// order = texture map (0..7) | texcoord (0..7) << 4 | enable << 7 | ras channel << 8,
    /// ksel = konst colour select | konst alpha select << 8.
    pub stages: [[u32; 4]; 16],
    /// TEV registers PREV, C0, C1, C2 (RGBA, signed 11-bit range scaled to 0..1 = 0..255).
    pub regs: [[f32; 4]; 4],
    pub konst: [[f32; 4]; 4],
    /// [number of stages, alpha compare (BP 0xF3), swap tables (4 × 8 bits), 0]
    pub params: [u32; 4],
    /// Fixed-function pixel state: [Z mode (BP 0x40), blend mode (BP 0x41), 0, 0].
    pub pipe: [u32; 4],
    /// Fog: [A (float), C (float), B magnitude, B shift] and [r, g, b, fsel | proj << 3].
    pub fog: [f32; 4],
    pub fog_color: [f32; 4],
    /// Texture maps 0..7 (decoder cache keys) referenced by the stages.
    pub textures: [Option<u64>; 8],
    pub blend: Blend,
}

impl TevState {
    pub fn key(&self) -> u64 {
        let mut h = 0xCBF2_9CE4_8422_2325u64;
        let mut mix = |v: u64| h = (h ^ v).wrapping_mul(0x100_0000_01B3);
        for st in &self.stages {
            for v in st {
                mix(*v as u64);
            }
        }
        for v in self.regs.iter().chain(self.konst.iter()).flatten() {
            mix(v.to_bits() as u64);
        }
        for v in self.params.iter().chain(self.pipe.iter()) {
            mix(*v as u64);
        }
        for v in self.fog.iter().chain(self.fog_color.iter()) {
            mix(v.to_bits() as u64);
        }
        for t in &self.textures {
            mix(t.unwrap_or(0));
        }
        mix(self.blend as u64);
        h
    }
}

/// One transformed vertex, in exactly the layout the renderer uploads (28 floats; matches the
/// vertex input of gx_efb.wgsl), so a frame's vertices go to the GPU as a straight copy.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct GpuVertex {
    /// Homogeneous clip coordinates over the whole EFB: after the divide x, y in [-1, 1]
    /// (y up) and z in [0, 1] (0 = near); the GPU clips (GX clips against the frustum too).
    pub pos: [f32; 4],
    /// Rasterised colour channels 0 and 1 (lit).
    pub color0: [f32; 4],
    pub color1: [f32; 4],
    /// Texture coordinates 0..7 (only the texgens the stages use are computed).
    pub uv: [[f32; 2]; 8],
}

#[derive(Clone, Default)]
pub struct Batch {
    pub key: u64,
    pub tev: Arc<TevState>,
    /// Each vertex of the batch's draws once; `indices` (into `verts`) lists the triangles in
    /// submission order (strips and fans share their vertices instead of repeating them).
    pub verts: Vec<GpuVertex>,
    pub indices: Vec<u32>,
    /// Pixel rectangle the batch may touch: scissor box & viewport, in EFB pixels
    /// [x0, y0, x1, y1) (y down).
    pub rect: [u32; 4],
    /// Diagnostics: projection and viewport when the batch started.
    pub proj: [f32; 7],
    pub viewport: [f32; 6],
    /// Diagnostics for the batch's first vertex: raw position, matrix index, matrix, view pos,
    /// and the vertex descriptor / attribute format registers.
    pub dbg_raw: [f32; 3],
    pub dbg_mtx_idx: u32,
    pub dbg_mtx: [f32; 12],
    pub dbg_view: [f32; 3],
    pub dbg_vtx: [u32; 6],
}

pub struct Texture {
    pub width: u32,
    pub height: u32,
    /// Raw GX texture bytes + palette; decoded to RGBA lazily by the consumer (the render
    /// thread), so big uploads don't stall the game thread.
    fmt: u32,
    raw: Vec<u8>,
    tlut: Vec<u8>,
    tlut_fmt: u32,
    rgba: std::sync::OnceLock<Vec<u8>>,
    /// 0 clamp, 1 repeat, 2 mirror.
    pub wrap_s: u8,
    pub wrap_t: u8,
    pub linear: bool,
}

#[derive(Clone, Default)]
pub struct Frame {
    pub batches: Vec<Batch>,
    pub clear: [f32; 4],
    pub draws: u32,
    /// Vertices emitted so far (budget check without walking the batches).
    pub verts: usize,
    /// Diagnostics: last projection (XF 0x1020-0x1026) and viewport (XF 0x101A-0x101F),
    /// and the EFB size used for normalisation.
    pub proj: [f32; 7],
    pub viewport: [f32; 6],
    pub efb: [f32; 2],
    /// Diagnostics: EFB copies in order, as (batch index at the time, BP 0x52 value,
    /// source rect BP 0x49, size BP 0x4A, destination BP 0x4B).
    pub copies: Vec<(usize, u32, u32, u32, u32)>,
    /// Diagnostics: draws that sampled an EFB-copy texture, as (batch index, address, w, h).
    pub rt_reads: Vec<(usize, u32, u32, u32)>,
    /// Registers in effect at each copy: PE control (BP 0x43), clear colour AR/GB (0x4F/0x50),
    /// clear Z (0x51), Z mode (0x40), blend mode (0x41), copy filter (0x53/0x54).
    pub copy_regs: Vec<[u32; 8]>,
}

impl Frame {
    pub fn triangles(&self) -> usize {
        self.batches.iter().map(|b| b.indices.len() / 3).sum()
    }
}

#[derive(Default)]
pub struct FrameShared {
    /// Textures by key, plus insertion order and total raw bytes for budgeted eviction.
    tex_order: Mutex<(std::collections::VecDeque<(u64, usize)>, usize)>,
    /// The frame on screen. Shared by pointer: readers (renderer, probe, trace) don't copy the
    /// vertex arrays.
    latest: Mutex<Arc<Frame>>,
    /// Last copied XFB frame, waiting for the next VI retrace to be "scanned out".
    pending: Mutex<Option<Frame>>,
    count: AtomicU64,
    textures: Mutex<HashMap<u64, Arc<Texture>>>,
}

impl FrameShared {
    pub fn count(&self) -> u64 {
        self.count.load(Relaxed)
    }
    pub fn latest(&self) -> Arc<Frame> {
        self.latest.lock().unwrap().clone()
    }
    pub fn texture(&self, key: u64) -> Option<Arc<Texture>> {
        self.textures.lock().unwrap().get(&key).cloned()
    }
    /// Display copy finished: like the real VI, the new XFB is shown at the next retrace.
    fn publish(&self, f: Frame) {
        *self.pending.lock().unwrap() = Some(f);
    }

    /// VI retrace: present the most recent copied frame, if there is a new one. Presenting on
    /// the retrace (not when the copy happens) keeps the output cadence locked to 59.94 Hz
    /// however the game's render work jitters within the frame.
    pub fn scanout(&self) {
        let Some(f) = self.pending.lock().unwrap().take() else { return };
        // Swap under the lock, free the previous frame after releasing it.
        let previous = std::mem::replace(&mut *self.latest.lock().unwrap(), Arc::new(f));
        drop(previous);
        self.count.fetch_add(1, Relaxed);
        crate::probe::guest_frame();
        // Frame pacing diagnostics: longest gap between finished frames since last read.
        static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
        let now = std::time::Instant::now();
        if let Some(prev) = LAST.lock().unwrap().replace(now) {
            MAX_FRAME_GAP_US.fetch_max(now.duration_since(prev).as_micros() as u64, Relaxed);
        }
    }
    fn add_texture(&self, key: u64, t: Texture) {
        // Budgeted, oldest-first eviction: a few entries per insert instead of clearing
        // thousands at once (THP movies add ~400 KB of textures every frame).
        const BUDGET: usize = 192 << 20;
        let bytes = t.raw.len() + t.tlut.len();
        let mut evicted = Vec::new();
        let mut map = self.textures.lock().unwrap();
        {
            let mut order = self.tex_order.lock().unwrap();
            order.0.push_back((key, bytes));
            order.1 += bytes;
            while order.1 > BUDGET && order.0.len() > 1 {
                let (k, b) = order.0.pop_front().unwrap();
                order.1 -= b;
                if let Some(old) = map.remove(&k) {
                    evicted.push(old);
                }
            }
        }
        map.insert(key, Arc::new(t));
    }
}

/// `BMX_GX_PROBE=n`: every n-th finished frame, log its EFB copies, the draws that sample
/// EFB-copy textures, and per-batch screen coverage (finds render-to-texture effects the
/// renderer doesn't reproduce yet).
fn gx_probe(f: &Frame) {
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    static N: AtomicU64 = AtomicU64::new(0);
    let every = *EVERY.get_or_init(|| std::env::var("BMX_GX_PROBE").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
    if every == 0 || N.fetch_add(1, Relaxed) % every != 0 {
        return;
    }
    let mut out = format!("gx probe: frame {} batches {} tris {} clear {:?}
", N.load(Relaxed), f.batches.len(), f.triangles(), f.clear);
    for (k, (at, cmd, src, size, dst)) in f.copies.iter().enumerate() {
        out += &format!(
            "  copy @batch {at}: pe {:#x} {} clear {} fmt {} src ({},{}) size {}x{} -> {:#010x} mipscale {}
",
            f.copy_regs.get(k).map_or(0, |r| r[0]), if cmd & (1 << 14) != 0 { "XFB" } else { "TEX" },
            (cmd >> 11) & 1, ((cmd >> 4) & 7) | ((cmd >> 3) & 1) << 3,
            bits(*src, 0, 10), bits(*src, 10, 10), bits(*size, 0, 10) + 1, bits(*size, 10, 10) + 1,
            0x8000_0000 | ((dst & 0xFF_FFFF) << 5), (cmd >> 9) & 1
        );
    }
    for (b, a, w, h) in &f.rt_reads {
        out += &format!("  rt read @batch {b}: {a:#010x} {w}x{h}
");
    }
    for (i, b) in f.batches.iter().enumerate() {
        if b.indices.is_empty() {
            continue;
        }
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in b.verts.iter().map(|v| v.pos) {
            for k in 0..3 {
                let v = p[k] / p[3];
                lo[k] = lo[k].min(v);
                hi[k] = hi[k].max(v);
            }
        }
        let area = ((hi[0].min(1.0) - lo[0].max(-1.0)).max(0.0) * (hi[1].min(1.0) - lo[1].max(-1.0)).max(0.0)) / 4.0;
        static ALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if area > 0.25 || i < 3 || *ALL.get_or_init(|| std::env::var_os("BMX_GX_PROBE_ALL").is_some()) {
            out += &format!(
                "  batch {i}: {} tris, cover {:.0}%, x {:.2}..{:.2} y {:.2}..{:.2} z {:.3}..{:.3}, tex {:?}, blend {:?} pipe {:#x}/{:#x} stages {} vp {:?} raw {:?} view {:?}
",
                b.indices.len() / 3, area * 100.0, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], b.tev.textures, b.tev.blend, b.tev.pipe[0], b.tev.pipe[1], b.tev.params[0],
                b.viewport, b.dbg_raw, b.dbg_view
            );
        }
    }
    crate::diag::log(out);
}

/// Time the GPU thread spends decoding FIFO data (ns, cumulative; always measured, once per
/// batch). The watchdog reports it as a busy percentage next to the frame rate.
pub static GPU_BUSY_NS: AtomicU64 = AtomicU64::new(0);

/// Longest time between two finished frames (µs); the watchdog reads and resets it.
pub static MAX_FRAME_GAP_US: AtomicU64 = AtomicU64::new(0);

const MAX_FRAME_VERTS: usize = 2_000_000;
const MAX_BATCHES: usize = 8192;
const TMEM_SIZE: usize = 0x10_0000;

pub struct Gx {
    /// Feed bytes to the decoder (false while the CPU FIFO is redirected to a display list).
    pub link: bool,
    buf: Vec<u8>,
    /// Bytes of `buf` already decoded (the rest is an incomplete command).
    pos: usize,
    cp: [u32; 256],
    xf_mem: Vec<u32>,
    xf_reg: [u32; 0x100],
    bp: [u32; 256],
    bp_mask: u32,
    /// TEV colour registers (C0..C2, PREV) and konst colours, as RGBA in 0..1 (signed range).
    tev_reg: [[f32; 4]; 4],
    konst: [[f32; 4]; 4],
    tmem: Vec<u8>,
    /// Texture cache: (addr, fmt, w, h, tlut addr, tlut fmt) -> (content hash, key).
    tex_cache: HashMap<(u32, u32, u32, u32, u32, u32), (u64, u64)>,
    /// Per-frame memo of texture lookups (image regs + mode + palette regs -> key): a texture
    /// bound for many draws is hashed once per frame, not once per draw. Cleared at frame end.
    tex_memo: HashMap<[u32; 4], Option<u64>>,
    /// Bumped on every BP write / texture-memo reset: draws with an unchanged generation reuse
    /// the previous draw's material, UV routing and batch key (J3D issues long runs of draws
    /// with identical state).
    state_gen: u64,
    mat_cache: Option<(u64, Arc<TevState>, u8, u64)>,
    /// Decoded vertex formats per VAT (CP 0x50-0x9F decide them; cleared on writes there).
    fmt_cache: [Option<VtxFormat>; 8],
    /// Lighting setup, rebuilt only after an XF register or light memory write
    /// (`light_gen` counts those writes).
    light_gen: u64,
    light_cache: Option<(u64, Box<LightSetup>)>,
    /// Reused per-draw vertex buffer (no allocation per draw).
    scratch: Vec<GpuVertex>,
    /// Addresses written by EFB-to-texture copies (their contents aren't emulated).
    copy_targets: std::collections::HashSet<u32>,
    frame: Frame,
    /// EFB width/height (from the last display copy), for viewport normalisation.
    efb: [f32; 2],
    /// Frames completed during the current `push`, published afterwards.
    done: Vec<Frame>,
    unknown_logged: u32,
    /// Position (12) + normal (9) matrices per matrix index, decoded from XF memory on first
    /// use; `mtx_valid` bit n = entry n is current (any write to matrix memory clears it).
    mtx_cache: Box<[[f32; 21]; 64]>,
    mtx_valid: u64,
}

impl Default for Gx {
    fn default() -> Self {
        Gx {
            link: true,
            buf: Vec::new(),
            pos: 0,
            cp: [0; 256],
            xf_mem: vec![0; 0x1000],
            xf_reg: [0; 0x100],
            bp: [0; 256],
            bp_mask: 0xFF_FFFF,
            tev_reg: [[0.0; 4]; 4],
            konst: [[1.0; 4]; 4],
            tmem: vec![0; TMEM_SIZE],
            tex_cache: HashMap::new(),
            tex_memo: HashMap::new(),
            state_gen: 0,
            mat_cache: None,
            fmt_cache: [None; 8],
            light_gen: 0,
            light_cache: None,
            scratch: Vec::new(),
            copy_targets: Default::default(),
            frame: Frame::default(),
            efb: [640.0, 480.0],
            done: Vec::new(),
            unknown_logged: 0,
            mtx_cache: Box::new([[0.0; 21]; 64]),
            mtx_valid: 0,
        }
    }
}

fn be16(d: &[u8], o: usize) -> u32 {
    u16::from_be_bytes([d[o], d[o + 1]]) as u32
}
fn be32(d: &[u8], o: usize) -> u32 {
    u32::from_be_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
}
fn bits(v: u32, lo: u32, n: u32) -> u32 {
    (v >> lo) & ((1 << n) - 1)
}
fn rgba_u32(v: u32) -> [f32; 4] {
    [
        ((v >> 24) & 0xFF) as f32 / 255.0,
        ((v >> 16) & 0xFF) as f32 / 255.0,
        ((v >> 8) & 0xFF) as f32 / 255.0,
        (v & 0xFF) as f32 / 255.0,
    ]
}
fn sext11(v: u32) -> f32 {
    (((v & 0x7FF) << 21) as i32 >> 21) as f32 / 255.0
}

/// Bytes per component for GX component formats u8, s8, u16, s16, f32.
fn comp_size(fmt: u32) -> usize {
    match fmt {
        0 | 1 => 1,
        2 | 3 => 2,
        _ => 4,
    }
}
fn color_size(fmt: u32) -> usize {
    [2, 3, 4, 2, 3, 4].get(fmt as usize).copied().unwrap_or(4)
}

#[derive(Clone, Copy, Default)]
struct AttrDesc {
    /// 0 none, 1 direct, 2 index8, 3 index16
    mode: u32,
    fmt: u32,
    count: usize,
    /// 1 / 2^shift (fixed-point scale of integer components).
    scale: f32,
    /// Byte size of the inline data for this attribute.
    inline: usize,
}

#[derive(Clone, Copy)]
struct VtxFormat {
    pnmtx: bool,
    texmtx: [bool; 8],
    pos: AttrDesc,
    nrm: AttrDesc,
    col: [AttrDesc; 2],
    tex: [AttrDesc; 8],
    size: usize,
}

/// Per-draw texgen setup (XF 0x1040+i texgen info, 0x1050+i post-transform), decoded once
/// per draw instead of once per vertex.
#[derive(Clone, Copy, Default)]
struct TexGen {
    stq: bool,
    /// Texgen type (0 = regular matrix transform; others pass the source through).
    ty: u32,
    /// Source row: 0 position, 1 normal, 5..12 texcoord 0..7, others constant.
    row: usize,
    /// Dual-texture post transform enabled, its normalisation flag and 3x4 matrix.
    dual: bool,
    normalize: bool,
    post: [f32; 12],
}

impl Gx {
    /// Feed FIFO bytes; returns PE events. Incomplete commands are buffered.
    pub fn push(&mut self, m: &Machine, bytes: &[u8]) -> Vec<Event> {
        self.buf.extend_from_slice(bytes);
        let mut ev = Vec::new();
        // Decode from the read offset; an incomplete trailing command waits for more bytes
        // without the buffer being copied (large immediate-mode draws arrive over many bursts).
        let data = std::mem::take(&mut self.buf);
        let used = {
            let _t = crate::probe::Timer::new(&crate::probe::GX_NS);
            self.exec(m, &data[self.pos..], &mut ev, 0)
        };
        self.buf = data;
        self.pos += used;
        if self.pos == self.buf.len() {
            self.buf.clear();
            self.pos = 0;
        } else if self.pos >= 1 << 16 {
            self.buf.drain(..self.pos);
            self.pos = 0;
        }
        for f in self.done.drain(..) {
            m.frames.publish(f);
        }
        if self.buf.len() - self.pos > 1 << 20 {
            crate::diag::log("gx: FIFO desync, dropping buffered bytes".into());
            self.buf.clear();
            self.pos = 0;
        }
        ev
    }

    /// Execute commands; returns bytes consumed (stops at an incomplete command).
    fn exec(&mut self, m: &Machine, d: &[u8], ev: &mut Vec<Event>, depth: u32) -> usize {
        let mut p = 0;
        while p < d.len() {
            let cmd = d[p] as u32;
            let need = match cmd {
                0x00 | 0x44 | 0x48 => 1,
                0x08 => 6,
                0x10 => {
                    if p + 5 > d.len() {
                        return p;
                    }
                    5 + 4 * (((be32(d, p + 1) >> 16) & 0xFFFF) as usize + 1)
                }
                0x20 | 0x28 | 0x30 | 0x38 => 5,
                0x40 => 9,
                0x61 => 5,
                0x80..=0xBF => {
                    if p + 3 > d.len() {
                        return p;
                    }
                    let n = be16(d, p + 1) as usize;
                    3 + n * self.format(cmd & 7).size
                }
                _ => {
                    if self.unknown_logged < 16 {
                        self.unknown_logged += 1;
                        crate::diag::log(format!("gx: unknown FIFO command {cmd:#04x}"));
                    }
                    1
                }
            };
            if p + need > d.len() {
                return p;
            }
            let c = &d[p..p + need];
            match cmd {
                0x08 => {
                    let reg = c[1] as usize;
                    self.cp[reg] = be32(c, 2);
                    if (0x50..0xA0).contains(&reg) {
                        self.fmt_cache = [None; 8];
                    }
                }
                0x10 => {
                    let addr = be32(c, 1) & 0xFFFF;
                    for i in 0..(need - 5) / 4 {
                        self.xf_write(addr + i as u32, be32(c, 5 + 4 * i));
                    }
                }
                0x20 | 0x28 | 0x30 | 0x38 => {
                    let v = be32(c, 1);
                    let array = 12 + ((cmd - 0x20) >> 3) as usize;
                    let idx = v >> 16;
                    let n = ((v >> 12) & 0xF) + 1;
                    let addr = v & 0xFFF;
                    let src = (self.cp[0xA0 + array] & 0x03FF_FFFF) + idx * self.cp[0xB0 + array];
                    for i in 0..n {
                        let val = m.read_be(0x8000_0000 | (src + 4 * i), 4) as u32;
                        self.xf_write(addr + i, val);
                    }
                }
                0x40 => {
                    let addr = be32(c, 1) & 0x03FF_FFFF;
                    let size = be32(c, 5) as usize;
                    if depth < 4 && (addr as usize) + size <= crate::RAM_SIZE {
                        // Decoded in place from RAM (J3D calls a display list per shape).
                        let ram: &[u8] = m.ram();
                        self.exec(m, &ram[addr as usize..addr as usize + size], ev, depth + 1);
                    }
                }
                0x61 => self.bp_write(m, be32(c, 1), ev),
                0x80..=0xBF => self.draw(m, cmd, c),
                _ => {}
            }
            p += need;
        }
        p
    }

    fn xf_write(&mut self, addr: u32, v: u32) {
        match addr {
            0..=0x0FFF => {
                self.xf_mem[addr as usize] = v;
                if addr < 0x500 {
                    self.mtx_valid = 0;
                } else if (0x600..0x680).contains(&addr) {
                    self.light_gen += 1;
                }
            }
            0x1000..=0x10FF => {
                self.xf_reg[(addr - 0x1000) as usize] = v;
                if (0x1009..=0x1011).contains(&addr) {
                    self.light_gen += 1;
                }
            }
            _ => {}
        }
    }

    fn bp_write(&mut self, m: &Machine, v: u32, ev: &mut Vec<Event>) {
        self.state_gen += 1;
        let reg = (v >> 24) as usize;
        let val = v & 0xFF_FFFF;
        if reg == 0xFE {
            self.bp_mask = val;
            return;
        }
        let new = (self.bp[reg] & !self.bp_mask) | (val & self.bp_mask);
        self.bp_mask = 0xFF_FFFF;
        self.bp[reg] = new;
        match reg {
            0x45 => ev.push(Event::DrawDone),
            0x47 => ev.push(Event::Token(new as u16)),
            0x48 => ev.push(Event::TokenInt(new as u16)),
            0x52 => {
                let at = self.frame.batches.len();
                self.frame.copies.push((at, new, self.bp[0x49], self.bp[0x4A], self.bp[0x4B]));
                self.frame.copy_regs.push([
                    self.bp[0x43], self.bp[0x4F], self.bp[0x50], self.bp[0x51], self.bp[0x40], self.bp[0x41], self.bp[0x53], self.bp[0x54],
                ]);
                if new & (1 << 14) != 0 {
                    self.end_frame();
                } else {
                    self.copy_targets.insert((self.bp[0x4B] & 0x00FF_FFFF) << 5);
                    self.tex_memo.clear();
        self.state_gen += 1;
                }
            }
            0x65 => {
                // TX_LOADTLUT1: copy a palette from RAM into TMEM.
                self.tex_memo.clear();
        self.state_gen += 1;
                let src = (self.bp[0x64] & 0x00FF_FFFF) << 5;
                let dst = ((new & 0x3FF) << 9) as usize;
                let len = (((new >> 10) & 0x7FF) * 32) as usize;
                let len = len.min(TMEM_SIZE.saturating_sub(dst));
                let ram: &[u8] = m.ram();
                let src = src as usize;
                let avail = ram.len().saturating_sub(src).min(len);
                if avail > 0 {
                    self.tmem[dst..dst + avail].copy_from_slice(&ram[src..src + avail]);
                }
                // Bytes past the end of RAM read as zero.
                self.tmem[dst + avail..dst + len].fill(0);
            }
            0xE0..=0xE7 => {
                let i = (reg - 0xE0) / 2;
                let konst = (new >> 23) & 1 != 0;
                let dst = if konst { &mut self.konst[i] } else { &mut self.tev_reg[i] }; // 0 = PREV, 1..3 = C0..C2
                if reg & 1 == 0 {
                    dst[0] = sext11(new);
                    dst[3] = sext11(new >> 12);
                } else {
                    dst[2] = sext11(new);
                    dst[1] = sext11(new >> 12);
                }
            }
            _ => {}
        }
    }

    fn end_frame(&mut self) {
        let _t = crate::probe::Timer::new(&crate::probe::GX_FRAME_NS);
        self.tex_memo.clear();
        self.state_gen += 1;
        let ar = self.bp[0x4F];
        let gb = self.bp[0x50];
        let size = self.bp[0x4A];
        let (w, h) = ((bits(size, 0, 10) + 1) as f32, (bits(size, 10, 10) + 1) as f32);
        if w >= 64.0 && h >= 64.0 {
            self.efb = [w, h];
        }
        let mut f = std::mem::take(&mut self.frame);
        for i in 0..7 {
            f.proj[i] = f32::from_bits(self.xf_reg[0x20 + i]);
        }
        for i in 0..6 {
            f.viewport[i] = f32::from_bits(self.xf_reg[0x1A + i]);
        }
        f.efb = self.efb;
        gx_probe(&f);
        f.clear = [
            (ar & 0xFF) as f32 / 255.0,
            ((gb >> 8) & 0xFF) as f32 / 255.0,
            (gb & 0xFF) as f32 / 255.0,
            ((ar >> 8) & 0xFF) as f32 / 255.0,
        ];
        self.done.push(f);
    }

    fn format(&mut self, vat: u32) -> VtxFormat {
        let i = (vat & 7) as usize;
        if let Some(f) = self.fmt_cache[i] {
            return f;
        }
        let f = self.decode_format(vat);
        self.fmt_cache[i] = Some(f);
        f
    }

    fn decode_format(&self, vat: u32) -> VtxFormat {
        let lo = self.cp[0x50];
        let hi = self.cp[0x60];
        let a = self.cp[0x70 + vat as usize];
        let b = self.cp[0x80 + vat as usize];
        let c = self.cp[0x90 + vat as usize];
        let idx_size = |mode: u32| match mode {
            2 => 1,
            3 => 2,
            _ => 0,
        };
        let mk = |mode: u32, fmt: u32, count: usize, shift: u32, elem: usize| {
            let inline = if mode == 1 { elem } else { idx_size(mode) };
            AttrDesc { mode, fmt, count, scale: 1.0 / (1u32 << shift) as f32, inline }
        };
        let pnmtx = lo & 1 != 0;
        let mut texmtx = [false; 8];
        for (i, t) in texmtx.iter_mut().enumerate() {
            *t = lo & (1 << (1 + i)) != 0;
        }
        let pos_cnt = if a & 1 != 0 { 3 } else { 2 };
        let pos_fmt = bits(a, 1, 3);
        let pos = mk(bits(lo, 9, 2), pos_fmt, pos_cnt, bits(a, 4, 5), pos_cnt * comp_size(pos_fmt));
        let nbt = bits(a, 9, 1) != 0;
        let nrm_fmt = bits(a, 10, 3);
        let nrm_cnt = if nbt { 9 } else { 3 };
        let nrm_shift = if nrm_fmt == 1 { 6 } else if nrm_fmt == 3 { 14 } else { 0 };
        let mut nrm = mk(bits(lo, 11, 2), nrm_fmt, nrm_cnt, nrm_shift, nrm_cnt * comp_size(nrm_fmt));
        // NBT with three separate indices: the normal is the first of them.
        if nrm.mode >= 2 && nbt && (a >> 31) & 1 != 0 {
            nrm.inline *= 3;
        }
        let col = [
            mk(bits(lo, 13, 2), bits(a, 14, 3), 4, 0, color_size(bits(a, 14, 3))),
            mk(bits(lo, 15, 2), bits(a, 18, 3), 4, 0, color_size(bits(a, 18, 3))),
        ];
        // (count bit, fmt, shift) per texcoord, spread across VAT A/B/C.
        let tfields = [
            (bits(a, 21, 1), bits(a, 22, 3), bits(a, 25, 5)),
            (bits(b, 0, 1), bits(b, 1, 3), bits(b, 4, 5)),
            (bits(b, 9, 1), bits(b, 10, 3), bits(b, 13, 5)),
            (bits(b, 18, 1), bits(b, 19, 3), bits(b, 22, 5)),
            (bits(b, 27, 1), bits(b, 28, 3), bits(c, 0, 5)),
            (bits(c, 5, 1), bits(c, 6, 3), bits(c, 9, 5)),
            (bits(c, 14, 1), bits(c, 15, 3), bits(c, 18, 5)),
            (bits(c, 23, 1), bits(c, 24, 3), bits(c, 27, 5)),
        ];
        let mut tex = [AttrDesc::default(); 8];
        for i in 0..8 {
            let (cnt, fmt, shift) = tfields[i];
            let n = if cnt != 0 { 2 } else { 1 };
            tex[i] = mk(bits(hi, 2 * i as u32, 2), fmt, n, shift, n * comp_size(fmt));
        }
        let mut size = pnmtx as usize + texmtx.iter().filter(|t| **t).count();
        size += pos.inline + nrm.inline + col[0].inline + col[1].inline;
        size += tex.iter().map(|t| t.inline).sum::<usize>();
        VtxFormat { pnmtx, texmtx, pos, nrm, col, tex, size }
    }

    /// Locate an attribute's data: inline bytes or (array, index).
    fn attr_bytes<'a>(&self, m: &'a Machine, d: &'a [u8], o: usize, at: &AttrDesc, array: usize, elem: usize) -> Option<&'a [u8]> {
        match at.mode {
            1 => Some(&d[o..o + elem]),
            2 | 3 => {
                let idx = if at.mode == 2 { d[o] as u32 } else { be16(d, o) };
                let base = self.cp[0xA0 + array] & 0x03FF_FFFF;
                let addr = (base + idx * self.cp[0xB0 + array]) as usize;
                (addr + elem <= crate::RAM_SIZE).then(|| &m.ram()[addr..addr + elem])
            }
            _ => None,
        }
    }

    /// Z texture (GXSetZTexture): BP 0xF5 op 1 = add, 2 = replace, using the last TEV stage's
    /// texture. The texel at (0,0) is used for the whole draw, which matches how the SDK and
    /// JFramework use it (a uniform "clear Z" texture). Returns (op, value in 24-bit units).
    fn ztexture(&self, m: &Machine) -> Option<(u32, f32)> {
        let ztex2 = self.bp[0xF5];
        let op = bits(ztex2, 2, 2);
        if op == 0 || op == 3 {
            return None;
        }
        let stage = bits(self.bp[0x00], 10, 4);
        let order = self.bp[0x28 + (stage / 2) as usize];
        let (map, on) = if stage & 1 == 0 { (bits(order, 0, 3), bits(order, 6, 1)) } else { (bits(order, 12, 3), bits(order, 18, 1)) };
        if on == 0 {
            return None;
        }
        let (img0_r, img3_r) = if map < 4 { (0x88 + map, 0x94 + map) } else { (0xA8 + map - 4, 0xB4 + map - 4) };
        let fmt = bits(self.bp[img0_r as usize], 20, 4);
        let addr = ((self.bp[img3_r as usize] & 0x00FF_FFFF) << 5) as usize;
        let ram = m.ram();
        if addr + 64 > ram.len() {
            return None;
        }
        let d = &ram[addr..addr + 64];
        let texel = match bits(ztex2, 0, 2) {
            0 => d[0] as u32,                                                    // u8 (I8 layout)
            1 => (d[0] as u32) << 8 | d[1] as u32,                               // u16 (IA8 layout)
            _ if fmt == 6 => (d[1] as u32) << 16 | (d[32] as u32) << 8 | d[33] as u32, // u24x8 (RGBA8 layout)
            _ => (d[0] as u32) << 16 | (d[1] as u32) << 8 | d[2] as u32,
        };
        let bias = self.bp[0xF4] & 0x00FF_FFFF;
        Some((op, (texel.wrapping_add(bias) & 0x00FF_FFFF) as f32))
    }

    fn xf_f(&self, addr: usize) -> f32 {
        f32::from_bits(self.xf_mem.get(addr).copied().unwrap_or(0))
    }

    fn draw(&mut self, m: &Machine, cmd: u32, c: &[u8]) {
        let _t = crate::probe::Timer::new(&crate::probe::GX_DRAW_NS);
        let prim = cmd & 0xF8;
        let genmode = self.bp[0x00];
        if bits(genmode, 14, 2) == 3 {
            return; // cull all
        }
        let n = be16(c, 1) as usize;
        // Triangles the primitive produces (lines and points are not drawn).
        let tris = match prim {
            0x80 | 0x88 => n / 4 * 2,
            0x90 => n / 3,
            0x98 | 0xA0 => n.saturating_sub(2),
            _ => 0,
        };
        if tris == 0 {
            return;
        }
        let f = self.format(cmd & 7);
        if crate::probe::enabled() {
            crate::probe::GX_DRAWS.fetch_add(1, Relaxed);
            crate::probe::GX_VERTS.fetch_add(n as u64, Relaxed);
        }
        let (tev, uv_src, key) = match &self.mat_cache {
            Some((g, t, u, k)) if *g == self.state_gen => (t.clone(), *u, *k),
            _ => {
                let _t = crate::probe::Timer::new(&crate::probe::GX_MAT_NS);
                let (t, u) = self.material(m);
                let t = Arc::new(t);
                let k = t.key();
                self.mat_cache = Some((self.state_gen, t.clone(), u, k));
                (t, u, k)
            }
        };
        let vtx_timer = crate::probe::Timer::new(&crate::probe::GX_VTX_NS);

        // Everything that is constant for the draw is decoded once here: lighting (cached
        // across draws until XF lighting state changes), texgens and their matrices,
        // projection, viewport and the EFB mapping.
        let lights = match self.light_cache.take() {
            Some((g, l)) if g == self.light_gen => l,
            _ => Box::new(self.light_setup()),
        };
        let has_vcol = [f.col[0].mode != 0, f.col[1].mode != 0];
        let dual = self.xf_reg[0x12] & 1 != 0;
        let mut gens = [TexGen::default(); 8];
        let mut tex_mtx = [[0f32; 12]; 8];
        for i in 0..8 {
            if uv_src & (1 << i) == 0 {
                continue;
            }
            let info = self.xf_reg[0x40 + i];
            let g = &mut gens[i];
            g.stq = bits(info, 1, 1) != 0;
            g.ty = bits(info, 4, 3);
            g.row = bits(info, 7, 5) as usize;
            // Dual texture transform (XF 0x1012): optional normalisation, then the
            // post-transform matrix (XF memory 0x500 + 4 * index). J3D uses it for environment
            // / reflection mapping (normal-sourced texgens), e.g. item boxes and kart shine.
            g.dual = dual;
            if dual {
                let post = self.xf_reg[0x50 + i];
                g.normalize = bits(post, 8, 1) != 0;
                g.post = self.xf_block(0x500 + (bits(post, 0, 6) * 4) as usize);
            }
            let idx = if i < 4 { bits(self.cp[0x30], 6 + 6 * i as u32, 6) } else { bits(self.cp[0x40], 6 * (i as u32 - 4), 6) };
            tex_mtx[i] = self.xf_block((idx * 4) as usize);
        }
        let pr: [f32; 6] = std::array::from_fn(|i| f32::from_bits(self.xf_reg[0x20 + i]));
        let ortho = self.xf_reg[0x26] != 0;
        let ztex = self.ztexture(m);
        // XF viewport: screen = offset + scale * ndc (x/y in EFB pixels + 342, z in 24-bit units).
        let vf = |i: usize| f32::from_bits(self.xf_reg[0x1A + i]);
        let (vp_xs, vp_ys, vp_zscale, vp_xo, vp_yo, vp_zoff) = (vf(0), vf(1), vf(2), vf(3), vf(4), vf(5));
        let [efb_w, efb_h] = self.efb;
        let (xoff, yoff) = self.screen_offset();
        // Window position inside the EFB = viewport origin - screen offset + scale * ndc; mapped
        // to [-1, 1] over the whole EFB and kept homogeneous (multiplied by w) so the GPU does
        // the perspective divide and frustum clipping like the GX setup unit.
        let (ax, bx) = (vp_xs / efb_w * 2.0, (vp_xo - xoff) / efb_w * 2.0 - 1.0);
        let (ay, by) = (-vp_ys / efb_h * 2.0, 1.0 - (vp_yo - yoff) / efb_h * 2.0);
        let default_pmtx = self.cp[0x30] & 0x3F;
        let fixed_mt = if f.pnmtx { [0f32; 21] } else { self.matrices(default_pmtx) };

        let mut out = std::mem::take(&mut self.scratch);
        out.clear();
        out.reserve(n);
        let mut first = None;
        let mut o = 3;
        for vi in 0..n {
            let mut pmtx = default_pmtx;
            let vm;
            let mt: &[f32; 21] = if f.pnmtx {
                pmtx = c[o] as u32 & 0x3F;
                o += 1;
                vm = self.matrices(pmtx);
                &vm
            } else {
                &fixed_mt
            };
            // Per-vertex texture matrix indices override the CP defaults.
            let mut tmtx_vtx = [u8::MAX; 8];
            for (i, present) in f.texmtx.iter().enumerate() {
                if *present {
                    tmtx_vtx[i] = c[o] & 0x3F;
                    o += 1;
                }
            }
            let mut pos = [0f32; 3];
            if let Some(b) = self.attr_bytes(m, c, o, &f.pos, 0, f.pos.count * comp_size(f.pos.fmt)) {
                for (i, p) in pos.iter_mut().enumerate().take(f.pos.count) {
                    *p = read_comp(b, i, f.pos.fmt, f.pos.scale);
                }
            }
            o += f.pos.inline;
            let mut nrm = [0f32, 0.0, 1.0];
            if let Some(b) = self.attr_bytes(m, c, o, &f.nrm, 1, 3 * comp_size(f.nrm.fmt)) {
                for (i, v) in nrm.iter_mut().enumerate() {
                    *v = read_comp(b, i, f.nrm.fmt, f.nrm.scale);
                }
            }
            o += f.nrm.inline;
            let mut vcol = [[1f32; 4]; 2];
            for ch in 0..2 {
                if let Some(b) = self.attr_bytes(m, c, o, &f.col[ch], 2 + ch, color_size(f.col[ch].fmt)) {
                    vcol[ch] = read_color(b, f.col[ch].fmt);
                }
                o += f.col[ch].inline;
            }
            let mut tc = [[0f32; 2]; 8];
            for i in 0..8 {
                let t = &f.tex[i];
                if t.mode != 0 {
                    if let Some(b) = self.attr_bytes(m, c, o, t, 4 + i, t.count * comp_size(t.fmt)) {
                        for k in 0..t.count {
                            tc[i][k] = read_comp(b, k, t.fmt, t.scale);
                        }
                    }
                    o += t.inline;
                }
            }

            let view = [
                mt[0] * pos[0] + mt[1] * pos[1] + mt[2] * pos[2] + mt[3],
                mt[4] * pos[0] + mt[5] * pos[1] + mt[6] * pos[2] + mt[7],
                mt[8] * pos[0] + mt[9] * pos[1] + mt[10] * pos[2] + mt[11],
            ];
            let mut vn = [
                mt[12] * nrm[0] + mt[13] * nrm[1] + mt[14] * nrm[2],
                mt[15] * nrm[0] + mt[16] * nrm[1] + mt[17] * nrm[2],
                mt[18] * nrm[0] + mt[19] * nrm[1] + mt[20] * nrm[2],
            ];
            let len = (vn[0] * vn[0] + vn[1] * vn[1] + vn[2] * vn[2]).sqrt();
            if len > 1e-6 {
                vn = [vn[0] / len, vn[1] / len, vn[2] / len];
            }
            let p = project(&pr, ortho, view);
            let w = p[3];
            let zc = match ztex {
                Some((2, z)) => z * w,
                Some((_, z)) => vp_zoff * w + vp_zscale * p[2] + z * w,
                None => vp_zoff * w + vp_zscale * p[2],
            } / 16_777_215.0;
            let mut v = GpuVertex { pos: [ax * p[0] + bx * w, ay * p[1] + by * w, zc, w], ..Default::default() };
            v.color0 = light_channel(&lights, 0, view, vn, vcol[0], has_vcol[0]);
            v.color1 = light_channel(&lights, 1, view, vn, vcol[1], has_vcol[1]);
            // Only the texgens the TEV stages read are consumed downstream.
            for i in 0..8 {
                if uv_src & (1 << i) != 0 {
                    let tm = if tmtx_vtx[i] != u8::MAX { self.xf_block(tmtx_vtx[i] as usize * 4) } else { tex_mtx[i] };
                    v.uv[i] = texgen(&gens[i], &tm, pos, nrm, &tc);
                }
            }
            if vi == 0 {
                first = Some((pos, pmtx, view));
            }
            out.push(v);
        }
        self.light_cache = Some((self.light_gen, lights));

        drop(vtx_timer);
        let _emit_timer = crate::probe::Timer::new(&crate::probe::GX_EMIT_NS);

        let dbg = first.map(|(raw, mtx, view)| {
            let m12 = self.xf_block((mtx * 4) as usize);
            let vat = (cmd & 7) as usize;
            let vtx = [self.cp[0x50], self.cp[0x60], self.cp[0x70 + vat], self.cp[0x80 + vat], self.cp[0x90 + vat], self.cp[0x30]];
            (raw, mtx, view, m12, vtx)
        });
        let rect = self.pixel_rect(xoff, yoff);
        if let Some(b) = self.batch_for(key, tev, rect) {
            if b.verts.is_empty() {
                if let Some((raw, mtx, view, m12, vtx)) = dbg {
                    b.dbg_raw = raw;
                    b.dbg_mtx_idx = mtx;
                    b.dbg_view = view;
                    b.dbg_mtx = m12;
                    b.dbg_vtx = vtx;
                }
            }
            let base = b.verts.len() as u32;
            b.verts.extend_from_slice(&out);
            let ix = &mut b.indices;
            ix.reserve(tris * 3);
            let n = n as u32;
            match prim {
                0x80 | 0x88 => {
                    for q in 0..n / 4 {
                        let v = base + 4 * q;
                        ix.extend_from_slice(&[v, v + 1, v + 2, v, v + 2, v + 3]);
                    }
                }
                0x90 => {
                    for t in 0..n / 3 {
                        let v = base + 3 * t;
                        ix.extend_from_slice(&[v, v + 1, v + 2]);
                    }
                }
                0x98 => {
                    for i in 2..n {
                        ix.extend_from_slice(&[base + i - 2, base + i - 1, base + i]);
                    }
                }
                _ => {
                    for i in 2..n {
                        ix.extend_from_slice(&[base, base + i - 1, base + i]);
                    }
                }
            }
            self.frame.verts += tris * 3;
            self.frame.draws += 1;
        }
        self.scratch = out;
    }

    /// The batch to append to: the last one if its key matches (keeps draw order), else new.
    /// Screen-space offset subtracted from window coordinates (GXSetScissorBoxOffset, BP 0x59,
    /// in units of 2 pixels and biased by 342; the reset value means no offset).
    fn screen_offset(&self) -> (f32, f32) {
        let r = self.bp[0x59];
        if r == 0 {
            return (342.0, 342.0);
        }
        ((bits(r, 0, 10) * 2) as f32, (bits(r, 10, 10) * 2) as f32)
    }

    /// Scissor box (BP 0x20/0x21) intersected with the viewport, in EFB pixels.
    fn pixel_rect(&self, xoff: f32, yoff: f32) -> [u32; 4] {
        let (tl, br) = (self.bp[0x20], self.bp[0x21]);
        let sc = if tl == 0 && br == 0 {
            [0.0, 0.0, 1024.0, 1024.0]
        } else {
            [
                bits(tl, 12, 11) as f32 - xoff,
                bits(tl, 0, 11) as f32 - yoff,
                bits(br, 12, 11) as f32 - xoff + 1.0,
                bits(br, 0, 11) as f32 - yoff + 1.0,
            ]
        };
        let vf = |i: usize| f32::from_bits(self.xf_reg[0x1A + i]);
        let (cx, cy, hw, hh) = (vf(3) - xoff, vf(4) - yoff, vf(0).abs(), vf(1).abs());
        let vp = [cx - hw, cy - hh, cx + hw, cy + hh];
        let [ew, eh] = self.efb;
        let x0 = sc[0].max(vp[0]).clamp(0.0, ew).round() as u32;
        let y0 = sc[1].max(vp[1]).clamp(0.0, eh).round() as u32;
        let x1 = sc[2].min(vp[2]).clamp(0.0, ew).round() as u32;
        let y1 = sc[3].min(vp[3]).clamp(0.0, eh).round() as u32;
        [x0, y0, x1.max(x0), y1.max(y0)]
    }

    fn batch_for(&mut self, key: u64, tev: Arc<TevState>, rect: [u32; 4]) -> Option<&mut Batch> {
        if self.frame.verts >= MAX_FRAME_VERTS {
            return None;
        }
        // Never append across an EFB copy: the copy must see exactly the draws before it.
        let after_copy = self.frame.copies.last().is_some_and(|c| c.0 == self.frame.batches.len());
        let reuse = !after_copy && self.frame.batches.last().is_some_and(|b| b.key == key && b.rect == rect);
        if !reuse {
            if self.frame.batches.len() >= MAX_BATCHES {
                return None;
            }
            let mut b = Batch { key, tev, rect, ..Default::default() };
            for i in 0..7 {
                b.proj[i] = f32::from_bits(self.xf_reg[0x20 + i]);
            }
            for i in 0..6 {
                b.viewport[i] = f32::from_bits(self.xf_reg[0x1A + i]);
            }
            self.frame.batches.push(b);
        }
        self.frame.batches.last_mut()
    }

    /// 12 consecutive XF memory words as floats (a 3x4 matrix).
    fn xf_block(&self, base: usize) -> [f32; 12] {
        std::array::from_fn(|k| self.xf_f(base + k))
    }

    /// XF colour channel `ch` (0/1): material/ambient sources, lighting, diffuse + attenuation.
    /// Position + normal matrix for matrix index `pmtx` (cached until matrix memory changes).
    fn matrices(&mut self, pmtx: u32) -> [f32; 21] {
        let i = (pmtx & 63) as usize;
        if self.mtx_valid & (1 << i) == 0 {
            let mut m = [0f32; 21];
            let base = i * 4;
            for k in 0..12 {
                m[k] = self.xf_f(base + k);
            }
            let nb = 0x400 + (i & 31) * 3;
            for k in 0..9 {
                m[12 + k] = self.xf_f(nb + k);
            }
            self.mtx_cache[i] = m;
            self.mtx_valid |= 1 << i;
        }
        self.mtx_cache[i]
    }

    /// Decode the XF lighting registers and enabled lights for both channels (colour + alpha).
    fn light_setup(&self) -> LightSetup {
        let mut setup = LightSetup { chans: bits(self.xf_reg[0x09], 0, 2) as usize, parts: Default::default() };
        for ch in 0..2 {
            for (part, ctrl_reg) in [(0usize, 0x0E + ch), (1, 0x10 + ch)] {
                let ctrl = self.xf_reg[ctrl_reg];
                let p = &mut setup.parts[ch][part];
                p.mat_vtx = bits(ctrl, 0, 1) != 0;
                p.amb_vtx = bits(ctrl, 6, 1) != 0;
                p.lighting = bits(ctrl, 1, 1) != 0;
                p.mat = rgba_u32(self.xf_reg[0x0C + ch]);
                p.amb = rgba_u32(self.xf_reg[0x0A + ch]);
                p.diff_fn = bits(ctrl, 7, 2);
                p.attn_fn = bits(ctrl, 9, 2);
                let mask = bits(ctrl, 2, 4) | (bits(ctrl, 11, 4) << 4);
                for l in 0..8 {
                    if mask & (1 << l) == 0 {
                        continue;
                    }
                    let lb = 0x600 + l * 0x10;
                    let lf = |i: usize| self.xf_f(lb + i);
                    p.lights[p.count] = LightParams {
                        col: rgba_u32(self.xf_mem[lb + 3]),
                        cos: [lf(4), lf(5), lf(6)],
                        dist: [lf(7), lf(8), lf(9)],
                        pos: [lf(10), lf(11), lf(12)],
                        dir: [lf(13), lf(14), lf(15)],
                    };
                    p.count += 1;
                }
            }
        }
        setup
    }

    /// Snapshot the TEV / blend state of the current draw (all 8 texture maps and texcoords, as
    /// the hardware has them). Returns the state and the mask of texgens the stages read.
    fn material(&mut self, m: &Machine) -> (TevState, u8) {
        let genmode = self.bp[0x00];
        let n = (bits(genmode, 10, 4) + 1) as usize;
        let mut t = TevState::default();
        let mut maps = 0u8;
        let mut uvs = 0u8;
        for s in 0..n {
            let order = self.bp[0x28 + s / 2] >> (12 * (s & 1));
            let (map, tc, en, chan) = (bits(order, 0, 3), bits(order, 3, 3), bits(order, 6, 1), bits(order, 7, 3));
            if en != 0 {
                maps |= 1 << map;
                uvs |= 1 << tc;
            }
            let ksel = self.bp[0xF6 + s / 2];
            let (kc, ka) = if s & 1 == 0 { (bits(ksel, 4, 5), bits(ksel, 9, 5)) } else { (bits(ksel, 14, 5), bits(ksel, 19, 5)) };
            t.stages[s] = [
                self.bp[0xC0 + 2 * s],
                self.bp[0xC1 + 2 * s],
                map | (tc << 4) | (en << 7) | (chan << 8),
                kc | (ka << 8),
            ];
        }
        for map in 0..8u32 {
            if maps & (1 << map) != 0 {
                t.textures[map as usize] = self.texture(m, map);
            }
        }
        let mut swap = 0u32;
        for tbl in 0..4 {
            let (rg, ba) = (self.bp[0xF6 + 2 * tbl], self.bp[0xF7 + 2 * tbl]);
            let packed = bits(rg, 0, 2) | (bits(rg, 2, 2) << 2) | (bits(ba, 0, 2) << 4) | (bits(ba, 2, 2) << 6);
            swap |= packed << (8 * tbl);
        }
        t.regs = self.tev_reg;
        t.konst = self.konst;
        t.params = [n as u32, self.bp[0xF3], swap, 0];
        // Fog (BP 0xEE-0xF2): A and C are packed floats (11-bit mantissa, 8-bit exponent, sign).
        let packed_float = |v: u32| f32::from_bits((bits(v, 19, 1) << 31) | (bits(v, 11, 8) << 23) | (bits(v, 0, 11) << 12));
        let p3 = self.bp[0xF1];
        let fsel = bits(p3, 21, 3);
        let proj = bits(p3, 20, 1);
        t.fog = [
            packed_float(self.bp[0xEE]),
            packed_float(p3),
            (self.bp[0xEF] & 0xFF_FFFF) as f32,
            bits(self.bp[0xF0], 0, 5) as f32,
        ];
        let fc = self.bp[0xF2];
        t.fog_color = [
            bits(fc, 16, 8) as f32 / 255.0,
            bits(fc, 8, 8) as f32 / 255.0,
            bits(fc, 0, 8) as f32 / 255.0,
            (fsel | (proj << 3)) as f32,
        ];
        let (zmode, blend_reg) = (self.bp[0x40], self.bp[0x41]);
        t.pipe = [zmode, blend_reg, 0, 0];
        let blending = blend_reg & 1 != 0 || bits(blend_reg, 11, 1) != 0;
        let ztest = zmode & 1 != 0;
        t.blend = if blending || !ztest { Blend::Sorted } else { Blend::Opaque };
        (t, uvs)
    }

    /// Decode (or reuse) the texture bound to `map`; returns its cache key.
    fn texture(&mut self, m: &Machine, map: u32) -> Option<u64> {
        let (mode_r, img0_r, img3_r, tlut_r) = if map < 4 {
            (0x80 + map, 0x88 + map, 0x94 + map, 0x98 + map)
        } else {
            (0xA0 + map - 4, 0xA8 + map - 4, 0xB4 + map - 4, 0xB8 + map - 4)
        };
        let memo = [self.bp[mode_r as usize], self.bp[img0_r as usize], self.bp[img3_r as usize], self.bp[tlut_r as usize]];
        if let Some(k) = self.tex_memo.get(&memo) {
            return *k;
        }
        let t0 = std::time::Instant::now();
        let r = self.texture_inner(m, map);
        self.tex_memo.insert(memo, r);
        let el = t0.elapsed();
        if el.as_millis() >= 5 {
            let img0 = self.bp[if map < 4 { 0x88 + map } else { 0xA8 + map - 4 } as usize];
            crate::diag::log(format!(
                "gx: slow texture lookup {:.1} ms (map {map}, fmt {}, {}x{}, cache entries {})",
                el.as_secs_f64() * 1000.0, bits(img0, 20, 4), bits(img0, 0, 10) + 1, bits(img0, 10, 10) + 1, self.tex_cache.len()
            ));
        }
        r
    }

    fn texture_inner(&mut self, m: &Machine, map: u32) -> Option<u64> {
        let (mode_r, img0_r, img3_r, tlut_r) = if map < 4 {
            (0x80 + map, 0x88 + map, 0x94 + map, 0x98 + map)
        } else {
            (0xA0 + map - 4, 0xA8 + map - 4, 0xB4 + map - 4, 0xB8 + map - 4)
        };
        let img0 = self.bp[img0_r as usize];
        let (w, h, fmt) = (bits(img0, 0, 10) + 1, bits(img0, 10, 10) + 1, bits(img0, 20, 4));
        let addr = (self.bp[img3_r as usize] & 0x00FF_FFFF) << 5;
        if addr == 0 {
            return None;
        }
        if self.copy_targets.contains(&addr) {
            if self.frame.rt_reads.len() < 64 {
                self.frame.rt_reads.push((self.frame.batches.len(), addr, w, h));
            }
            // Rendered by an EFB copy: the renderer binds its copy texture for this address.
            return Some(rt_key(addr, self.bp[mode_r as usize]));
        }
        let size = tex_size(fmt, w, h);
        if size == 0 || addr as usize + size > crate::RAM_SIZE {
            return None;
        }
        let tlut = self.bp[tlut_r as usize];
        let (tlut_addr, tlut_fmt) = (((tlut & 0x3FF) << 9), bits(tlut, 10, 2));
        let data = &m.ram()[addr as usize..addr as usize + size];
        let mut h64 = fnv(&[addr.to_be_bytes(), fmt.to_be_bytes(), w.to_be_bytes(), h.to_be_bytes()].concat());
        let step = (size / 512).max(1);
        for i in (0..size).step_by(step) {
            h64 = (h64 ^ data[i] as u64).wrapping_mul(0x100_0000_01B3);
        }
        let paletted = matches!(fmt, 8 | 9 | 10);
        if paletted {
            let n = if fmt == 8 { 32 } else { 512 };
            let t = tlut_addr as usize;
            if t + n <= TMEM_SIZE {
                h64 = h64 ^ fnv(&self.tmem[t..t + n]);
            }
        }
        let ck = (addr, fmt, w, h, if paletted { tlut_addr } else { 0 }, tlut_fmt);
        if let Some((hash, key)) = self.tex_cache.get(&ck) {
            if *hash == h64 {
                return Some(*key);
            }
        }
        let mode = self.bp[mode_r as usize];
        let t0 = (tlut_addr as usize).min(TMEM_SIZE - 512);
        let tex = Texture {
            width: w,
            height: h,
            fmt,
            raw: data.to_vec(),
            tlut: if paletted { self.tmem[t0..(t0 + 512).min(TMEM_SIZE)].to_vec() } else { Vec::new() },
            tlut_fmt,
            rgba: std::sync::OnceLock::new(),
            wrap_s: bits(mode, 0, 2) as u8,
            wrap_t: bits(mode, 2, 2) as u8,
            linear: bits(mode, 4, 1) != 0,
        };
        let key = h64 ^ ((mode as u64) << 40);
        m.frames.add_texture(key, tex);
        self.tex_cache.insert(ck, (h64, key));
        Some(key)
    }
}

/// Position (view space) → clip space through the XF projection (XF 0x1020-0x1026).
#[inline(always)]
fn project(pr: &[f32; 6], ortho: bool, v: [f32; 3]) -> [f32; 4] {
    if !ortho {
        [pr[0] * v[0] + pr[1] * v[2], pr[2] * v[1] + pr[3] * v[2], pr[4] * v[2] + pr[5], -v[2]]
    } else {
        [pr[0] * v[0] + pr[1], pr[2] * v[1] + pr[3], pr[4] * v[2] + pr[5], 1.0]
    }
}

/// Texgen: matrix × source row (position, normal or texcoord), the optional dual-texture post
/// transform, projected if STQ.
#[inline(always)]
fn texgen(g: &TexGen, m: &[f32; 12], pos: [f32; 3], nrm: [f32; 3], tc: &[[f32; 2]; 8]) -> [f32; 2] {
    let input = match g.row {
        0 => [pos[0], pos[1], pos[2], 1.0],
        1 => [nrm[0], nrm[1], nrm[2], 1.0],
        5..=12 => [tc[g.row - 5][0], tc[g.row - 5][1], 1.0, 1.0],
        _ => [0.0, 0.0, 1.0, 1.0],
    };
    if g.ty != 0 {
        return [input[0], input[1]]; // emboss / colour texgens: pass through
    }
    let r = |k: usize| m[4 * k] * input[0] + m[4 * k + 1] * input[1] + m[4 * k + 2] * input[2] + m[4 * k + 3] * input[3];
    // 2x4 (ST) or 3x4 (STQ) texture matrix.
    let mut v = [r(0), r(1), if g.stq { r(2) } else { 1.0 }];
    if g.dual {
        if g.normalize {
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if len > 1e-12 {
                v = [v[0] / len, v[1] / len, v[2] / len];
            }
        }
        let p = &g.post;
        let q = |k: usize| p[4 * k] * v[0] + p[4 * k + 1] * v[1] + p[4 * k + 2] * v[2] + p[4 * k + 3];
        v = [q(0), q(1), q(2)];
    }
    if g.stq && v[2].abs() > 1e-6 {
        return [v[0] / v[2], v[1] / v[2]];
    }
    [v[0], v[1]]
}

#[derive(Clone, Copy, Default)]
struct LightParams {
    col: [f32; 4],
    /// Angular (cos) attenuation A0..A2 and distance attenuation K0..K2.
    cos: [f32; 3],
    dist: [f32; 3],
    pos: [f32; 3],
    /// Direction (spot) or half-angle vector (specular).
    dir: [f32; 3],
}

#[derive(Clone, Copy, Default)]
struct LitPart {
    lighting: bool,
    mat_vtx: bool,
    amb_vtx: bool,
    mat: [f32; 4],
    amb: [f32; 4],
    diff_fn: u32,
    attn_fn: u32,
    lights: [LightParams; 8],
    count: usize,
}

/// XF lighting state of a draw: [channel][colour, alpha].
#[derive(Clone)]
struct LightSetup {
    chans: usize,
    parts: [[LitPart; 2]; 2],
}

/// XF colour channel `ch` (0/1): material/ambient sources, lighting with the enabled lights,
/// diffuse and attenuation functions (same formulas as Dolphin's LightingShaderGen).
fn light_channel(setup: &LightSetup, ch: usize, pos: [f32; 3], n: [f32; 3], vcol: [f32; 4], has_vcol: bool) -> [f32; 4] {
    if ch >= setup.chans.max(1) {
        return [0.0; 4];
    }
    let dot3 = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let norm3 = |v: [f32; 3]| {
        let l = dot3(v, v).sqrt();
        if l > 0.0 { [v[0] / l, v[1] / l, v[2] / l] } else { v }
    };
    let mut out = [0f32; 4];
    for part in 0..2 {
        let p = &setup.parts[ch][part];
        let material = if p.mat_vtx && has_vcol { vcol } else { p.mat };
        let mut lit = if !p.lighting {
            [1.0; 4]
        } else {
            let mut acc = if p.amb_vtx && has_vcol { vcol } else { p.amb };
            for l in &p.lights[..p.count] {
                let to_light = [l.pos[0] - pos[0], l.pos[1] - pos[1], l.pos[2] - pos[2]];
                let (ldir, attn) = match p.attn_fn {
                    // Specular: the light's "direction" is the half-angle vector.
                    1 => {
                        let ldir = norm3(to_light);
                        let a = if dot3(n, ldir) >= 0.0 { dot3(n, l.dir).max(0.0) } else { 0.0 };
                        let k = if p.diff_fn == 0 { l.dist } else { norm3(l.dist) };
                        let num = (l.cos[0] + l.cos[1] * a + l.cos[2] * a * a).max(0.0);
                        let den = k[0] + k[1] * a + k[2] * a * a;
                        (ldir, if den != 0.0 { num / den } else { 0.0 })
                    }
                    // Spot: angular (cos) and distance attenuation.
                    3 => {
                        let dist2 = dot3(to_light, to_light);
                        let dist = dist2.sqrt();
                        let ldir = if dist > 0.0 { [to_light[0] / dist, to_light[1] / dist, to_light[2] / dist] } else { to_light };
                        let c = dot3(ldir, l.dir).max(0.0);
                        let num = (l.cos[0] + l.cos[1] * c + l.cos[2] * c * c).max(0.0);
                        let den = l.dist[0] + l.dist[1] * dist + l.dist[2] * dist2;
                        (ldir, if den != 0.0 { num / den } else { 0.0 })
                    }
                    // None / directional: no attenuation.
                    _ => {
                        let ldir = norm3(to_light);
                        (if dot3(ldir, ldir) == 0.0 { n } else { ldir }, 1.0)
                    }
                };
                let ndl = dot3(ldir, n);
                let diff = match p.diff_fn {
                    0 => 1.0,
                    1 => ndl,
                    _ => ndl.max(0.0),
                };
                // Each light's contribution is rounded to the 8-bit accumulator.
                for k in 0..4 {
                    acc[k] += (attn * diff * l.col[k] * 255.0).round() / 255.0;
                }
            }
            acc.map(|x| x.clamp(0.0, 1.0))
        };
        if part == 0 {
            for k in 0..3 {
                lit[k] *= material[k];
                out[k] = lit[k];
            }
        } else {
            out[3] = lit[3] * material[3];
        }
    }
    out
}

/// Texture keys with this tag name an EFB-copy render target instead of a decoded texture:
/// bits 0-31 = RAM address, bits 32-47 = texture mode (wrap / filter bits).
pub const RT_TAG: u64 = 0xEFB0 << 48;

pub fn rt_key(addr: u32, mode: u32) -> u64 {
    RT_TAG | ((mode as u64 & 0xFFFF) << 32) | addr as u64
}

fn fnv(d: &[u8]) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for b in d {
        h = (h ^ *b as u64).wrapping_mul(0x100_0000_01B3);
    }
    h
}

#[inline(always)]
fn read_comp(b: &[u8], i: usize, fmt: u32, scale: f32) -> f32 {
    match fmt {
        0 => b[i] as f32 * scale,
        1 => b[i] as i8 as f32 * scale,
        2 => be16(b, 2 * i) as f32 * scale,
        3 => be16(b, 2 * i) as u16 as i16 as f32 * scale,
        _ => f32::from_bits(be32(b, 4 * i)),
    }
}

fn read_color(b: &[u8], fmt: u32) -> [f32; 4] {
    let n = |v: u32, bits: u32| v as f32 / ((1u32 << bits) - 1) as f32;
    match fmt {
        0 => {
            let v = be16(b, 0);
            [n(v >> 11, 5), n((v >> 5) & 63, 6), n(v & 31, 5), 1.0]
        }
        1 | 2 => [n(b[0] as u32, 8), n(b[1] as u32, 8), n(b[2] as u32, 8), 1.0],
        3 => {
            let v = be16(b, 0);
            [n(v >> 12, 4), n((v >> 8) & 15, 4), n((v >> 4) & 15, 4), n(v & 15, 4)]
        }
        4 => {
            let v = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
            [n(v >> 18, 6), n((v >> 12) & 63, 6), n((v >> 6) & 63, 6), n(v & 63, 6)]
        }
        _ => [n(b[0] as u32, 8), n(b[1] as u32, 8), n(b[2] as u32, 8), n(b[3] as u32, 8)],
    }
}

// ---- texture decoding ----

/// (block width, block height, bytes per block) for a GX texture format.
fn block_info(fmt: u32) -> Option<(u32, u32, u32)> {
    Some(match fmt {
        0 | 8 => (8, 8, 32),        // I4, C4
        1 | 2 | 9 => (8, 4, 32),    // I8, IA4, C8
        3 | 4 | 5 | 10 => (4, 4, 32), // IA8, RGB565, RGB5A3, C14X2
        6 => (4, 4, 64),            // RGBA8
        14 => (8, 8, 32),           // CMPR
        _ => return None,
    })
}

pub fn tex_size(fmt: u32, w: u32, h: u32) -> usize {
    match block_info(fmt) {
        Some((bw, bh, bytes)) => (w.div_ceil(bw) * h.div_ceil(bh) * bytes) as usize,
        None => 0,
    }
}

fn rgb565(v: u16) -> [u8; 4] {
    let r = ((v >> 11) & 31) as u32;
    let g = ((v >> 5) & 63) as u32;
    let b = (v & 31) as u32;
    [(r * 255 / 31) as u8, (g * 255 / 63) as u8, (b * 255 / 31) as u8, 255]
}

fn rgb5a3(v: u16) -> [u8; 4] {
    if v & 0x8000 != 0 {
        let c = |x: u16| ((x as u32 & 31) * 255 / 31) as u8;
        [c(v >> 10), c(v >> 5), c(v), 255]
    } else {
        let c = |x: u16| ((x as u32 & 15) * 255 / 15) as u8;
        [c(v >> 8), c(v >> 4), c(v), ((v as u32 >> 12 & 7) * 255 / 7) as u8]
    }
}

fn tlut_color(tlut: &[u8], idx: usize, fmt: u32) -> [u8; 4] {
    let o = 2 * idx;
    if o + 1 >= tlut.len() {
        return [0, 0, 0, 0];
    }
    let v = u16::from_be_bytes([tlut[o], tlut[o + 1]]);
    match fmt {
        0 => {
            let (a, i) = ((v >> 8) as u8, v as u8);
            [i, i, i, a]
        }
        1 => rgb565(v),
        _ => rgb5a3(v),
    }
}

impl Texture {
    /// RGBA8 texels (decoded on first use).
    pub fn rgba(&self) -> &[u8] {
        self.rgba.get_or_init(|| {
            let _t = crate::probe::Timer::new(&crate::probe::TEX_NS);
            decode_texture(self.fmt, self.width, self.height, &self.raw, &self.tlut, self.tlut_fmt)
                .unwrap_or_else(|| vec![255; (self.width * self.height * 4) as usize])
        })
    }
}

fn decode_texture(fmt: u32, w: u32, h: u32, d: &[u8], tlut: &[u8], tlut_fmt: u32) -> Option<Vec<u8>> {
    let (bw, bh, bytes) = block_info(fmt)?;
    let mut out = vec![0u8; (w * h * 4) as usize];
    let bx_n = w.div_ceil(bw);
    let by_n = h.div_ceil(bh);
    let mut put = |x: u32, y: u32, c: [u8; 4]| {
        if x < w && y < h {
            let o = ((y * w + x) * 4) as usize;
            out[o..o + 4].copy_from_slice(&c);
        }
    };
    for by in 0..by_n {
        for bx in 0..bx_n {
            let base = ((by * bx_n + bx) * bytes) as usize;
            let blk = &d[base..base + bytes as usize];
            let (x0, y0) = (bx * bw, by * bh);
            match fmt {
                0 | 8 => {
                    for i in 0..64u32 {
                        let b = blk[(i / 2) as usize];
                        let v = if i & 1 == 0 { b >> 4 } else { b & 15 };
                        let c = if fmt == 0 {
                            let l = v * 17;
                            [l, l, l, l]
                        } else {
                            tlut_color(tlut, v as usize, tlut_fmt)
                        };
                        put(x0 + i % 8, y0 + i / 8, c);
                    }
                }
                1 | 9 => {
                    for i in 0..32u32 {
                        let v = blk[i as usize];
                        let c = if fmt == 1 { [v, v, v, v] } else { tlut_color(tlut, v as usize, tlut_fmt) };
                        put(x0 + i % 8, y0 + i / 8, c);
                    }
                }
                2 => {
                    for i in 0..32u32 {
                        let v = blk[i as usize];
                        let (a, l) = ((v >> 4) * 17, (v & 15) * 17);
                        put(x0 + i % 8, y0 + i / 8, [l, l, l, a]);
                    }
                }
                3 => {
                    for i in 0..16u32 {
                        let (a, l) = (blk[2 * i as usize], blk[2 * i as usize + 1]);
                        put(x0 + i % 4, y0 + i / 4, [l, l, l, a]);
                    }
                }
                4 | 5 | 10 => {
                    for i in 0..16u32 {
                        let v = u16::from_be_bytes([blk[2 * i as usize], blk[2 * i as usize + 1]]);
                        let c = match fmt {
                            4 => rgb565(v),
                            5 => rgb5a3(v),
                            _ => tlut_color(tlut, (v & 0x3FFF) as usize, tlut_fmt),
                        };
                        put(x0 + i % 4, y0 + i / 4, c);
                    }
                }
                6 => {
                    for i in 0..16u32 {
                        let ar = 2 * i as usize;
                        let gb = 32 + 2 * i as usize;
                        put(x0 + i % 4, y0 + i / 4, [blk[ar + 1], blk[gb], blk[gb + 1], blk[ar]]);
                    }
                }
                14 => {
                    for sub in 0..4u32 {
                        let s = &blk[(sub * 8) as usize..(sub * 8 + 8) as usize];
                        let c0 = u16::from_be_bytes([s[0], s[1]]);
                        let c1 = u16::from_be_bytes([s[2], s[3]]);
                        let (p0, p1) = (rgb565(c0), rgb565(c1));
                        let mix = |a: u8, b: u8, wa: u32, wb: u32, div: u32| ((a as u32 * wa + b as u32 * wb) / div) as u8;
                        let pal: [[u8; 4]; 4] = if c0 > c1 {
                            [p0, p1,
                             [mix(p0[0], p1[0], 2, 1, 3), mix(p0[1], p1[1], 2, 1, 3), mix(p0[2], p1[2], 2, 1, 3), 255],
                             [mix(p0[0], p1[0], 1, 2, 3), mix(p0[1], p1[1], 1, 2, 3), mix(p0[2], p1[2], 1, 2, 3), 255]]
                        } else {
                            [p0, p1,
                             [mix(p0[0], p1[0], 1, 1, 2), mix(p0[1], p1[1], 1, 1, 2), mix(p0[2], p1[2], 1, 1, 2), 255],
                             [0, 0, 0, 0]]
                        };
                        let (sx, sy) = (x0 + (sub % 2) * 4, y0 + (sub / 2) * 4);
                        for row in 0..4u32 {
                            let bitsrow = s[4 + row as usize];
                            for col in 0..4u32 {
                                let idx = (bitsrow >> (6 - 2 * col)) & 3;
                                put(sx + col, sy + row, pal[idx as usize]);
                            }
                        }
                    }
                }
                _ => return None,
            }
        }
    }
    Some(out)
}


/// Asynchronous GPU, like the real GP consuming the FIFO in parallel with the CPU: the CPU's
/// gather pipe submits 32-byte bursts here and a dedicated thread decodes them. PE events
/// (draw done, tokens) are queued for the CPU side and announced with their interrupts; the
/// register updates are applied by the CPU side (`hw::apply_gx_events`), never concurrently.
pub struct GpuQueue {
    q: Mutex<(Vec<u8>, bool)>, // pending bytes, worker busy
    work: std::sync::Condvar,
    space: std::sync::Condvar,
    events: Mutex<Vec<Event>>,
}

/// The CPU stalls when this much FIFO data is waiting, like the hardware FIFO high watermark.
const GPU_QUEUE_LIMIT: usize = 4 << 20;

pub fn gpu() -> &'static GpuQueue {
    static G: std::sync::OnceLock<GpuQueue> = std::sync::OnceLock::new();
    G.get_or_init(|| GpuQueue {
        q: Mutex::new((Vec::with_capacity(1 << 20), false)),
        work: std::sync::Condvar::new(),
        space: std::sync::Condvar::new(),
        events: Mutex::new(Vec::new()),
    })
}

impl GpuQueue {
    pub fn submit(&self, bytes: &[u8]) {
        let mut q = self.q.lock().unwrap();
        while q.0.len() > GPU_QUEUE_LIMIT {
            q = self.space.wait(q).unwrap();
        }
        q.0.extend_from_slice(bytes);
        self.work.notify_one();
    }
    /// No FIFO data waiting and the decoder isn't working (CP status "idle").
    pub fn idle(&self) -> bool {
        let q = self.q.lock().unwrap();
        q.0.is_empty() && !q.1
    }
    pub fn take_events(&self) -> Vec<Event> {
        std::mem::take(&mut *self.events.lock().unwrap())
    }
}

/// Start the GPU thread (owns the decoder state in `Hardware::gx`).
pub fn start_gpu(m: &'static Machine) {
    std::thread::Builder::new()
        .name("bmx-gpu".into())
        .stack_size(64 << 20)
        .spawn(move || {
            crate::hostprof::register(crate::hostprof::GPU);
            let g = gpu();
            let gx = &mut m.hw().gx;
            let mut data = Vec::with_capacity(1 << 20);
            loop {
                {
                    let mut q = g.q.lock().unwrap();
                    while q.0.is_empty() {
                        q.1 = false;
                        q = g.work.wait(q).unwrap();
                    }
                    q.1 = true;
                    std::mem::swap(&mut q.0, &mut data);
                }
                g.space.notify_all();
                let t0 = std::time::Instant::now();
                let ev = gx.push(m, &data);
                GPU_BUSY_NS.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
                data.clear();
                if !ev.is_empty() {
                    let finish = ev.iter().any(|e| matches!(e, Event::DrawDone));
                    let token = ev.iter().any(|e| matches!(e, Event::TokenInt(_)));
                    g.events.lock().unwrap().extend(ev);
                    if finish {
                        crate::sched::raise(crate::sched::IRQ_PE_FINISH);
                    }
                    if token {
                        crate::sched::raise(crate::sched::IRQ_PE_TOKEN);
                    }
                }
            }
        })
        .expect("spawn gpu thread");
}
