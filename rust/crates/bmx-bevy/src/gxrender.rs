//! GX renderer: replays each decoded GX frame on an emulated embedded framebuffer (EFB).
//!
//! The GameCube draws into a 640×528 on-chip framebuffer (colour + 24-bit depth) and gets
//! results out of it only through EFB copies (BP 0x52): to a texture in RAM, which later draws
//! sample (render-to-texture: blur, depth effects, previews), or to the external framebuffer
//! (XFB) for display, optionally clearing the EFB afterwards. This module does the same on the
//! host GPU, from the frame `gc_runtime::gx` decodes:
//!
//! * batches are drawn strictly in submission order into the EFB colour/depth targets, each with
//!   its GX blend / Z state, scissor ∩ viewport rectangle and per-pixel TEV (`gx_efb.wgsl`);
//! * at each copy the EFB rectangle is converted (copy format, colour or Z source, 2×2 box
//!   filter, intensity) into a texture kept per destination address, which draws referencing
//!   that address sample (`gx_copy.wgsl`); a copy with the clear bit then clears the rectangle;
//! * the XFB copy is what the window shows (`gx_blit.wgsl`), letterboxed to 4:3.
//!
//! Runs in the render world as part of the camera's Core3d schedule, before the UI pass, so
//! the overlay text still draws on top.

use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::view::ViewTarget;
use bevy::render::RenderApp;
use gc_runtime::gx::{Frame, FrameShared, TevState, RT_TAG};
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};
use wgpu::*;

/// The decoder's frame queue, made available to the render world.
#[derive(Resource, Clone, ExtractResource)]
pub struct GxLink(pub Arc<FrameShared>);

pub struct GxRenderPlugin;

impl Plugin for GxRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((ExtractResourcePlugin::<GxLink>::default(), ExtractResourcePlugin::<crate::settings::Settings>::default()));
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else { return };
        render_app.init_resource::<GxState>().add_systems(Core3d, gx_pass.in_set(Core3dSystems::PostProcess));
    }
}

#[derive(Resource, Default)]
struct GxState {
    gpu: Option<Gpu>,
    shown: u64,
}

const TEV_UNIFORM_SIZE: u64 = 432;
const TEV_STRIDE: u64 = 512;
const COPY_UNIFORM_SIZE: u64 = 80;
const COPY_STRIDE: u64 = 256;
/// Floats per vertex: clip position (4), colour 0 (4), colour 1 (4), texcoords 0..7 (16);
/// the decoder already produces vertices in this layout (`gx::GpuVertex`).
const VERTEX_FLOATS: usize = 28;
const _: () = assert!(std::mem::size_of::<gc_runtime::gx::GpuVertex>() == VERTEX_FLOATS * 4);
const EFB_COLOR: TextureFormat = TextureFormat::Rgba8Unorm;
const EFB_DEPTH: TextureFormat = TextureFormat::Depth32Float;

struct Target {
    _tex: Texture,
    view: TextureView,
    size: (u32, u32),
}

impl Target {
    fn new(device: &Device, label: &str, size: (u32, u32), format: TextureFormat) -> Target {
        let tex = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d { width: size.0.max(1), height: size.1.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&TextureViewDescriptor::default());
        Target { _tex: tex, view, size }
    }
}

struct GameTexture {
    view: TextureView,
    sampler_key: (u8, u8, bool),
    used: u64,
}

struct Gpu {
    tev_module: ShaderModule,
    tev_bgl: BindGroupLayout,
    tev_layout: PipelineLayout,
    /// Pipelines per (Z mode, blend mode). Shared with the background prewarm threads.
    tev_pipelines: Arc<Mutex<HashMap<(u32, u32), RenderPipeline>>>,
    copy_bgl: BindGroupLayout,
    copy_pipeline: RenderPipeline,
    clear_bgl: BindGroupLayout,
    clear_module: ShaderModule,
    clear_layout: PipelineLayout,
    clear_pipelines: HashMap<(bool, bool, bool), RenderPipeline>,
    blit_module: ShaderModule,
    blit_bgl: BindGroupLayout,
    blit_layout: PipelineLayout,
    blit_pipelines: HashMap<TextureFormat, RenderPipeline>,
    blit_sampler: Sampler,
    blit_group: Option<BindGroup>,
    /// Display parameters for the blit (gamma).
    display_buf: Buffer,

    efb: Option<(Target, Target)>,
    xfb: Option<Target>,
    /// EFB-copy textures by destination address.
    rts: HashMap<u32, Target>,
    textures: HashMap<u64, GameTexture>,
    samplers: HashMap<(u8, u8, bool), Sampler>,
    white: TextureView,
    vertex_buf: Option<Buffer>,
    index_buf: Option<Buffer>,
    tev_buf: Option<Buffer>,
    copy_buf: Buffer,
    tick: u64,
}

fn shader(device: &Device, label: &str, src: &str) -> ShaderModule {
    device.create_shader_module(ShaderModuleDescriptor { label: Some(label), source: ShaderSource::Wgsl(src.into()) })
}

fn uniform_entry(binding: u32, size: u64) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::VERTEX_FRAGMENT,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: NonZeroU64::new(size),
        },
        count: None,
    }
}

fn texture_entry(binding: u32, sample_type: TextureSampleType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture { sample_type, view_dimension: TextureViewDimension::D2, multisampled: false },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Sampler(SamplerBindingType::Filtering),
        count: None,
    }
}

fn layout(device: &Device, bgl: &BindGroupLayout) -> PipelineLayout {
    device.create_pipeline_layout(&PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(bgl)], immediate_size: 0 })
}

/// GX compare function (Z mode bits 1-3).
fn compare(f: u32) -> CompareFunction {
    match f {
        0 => CompareFunction::Never,
        1 => CompareFunction::Less,
        2 => CompareFunction::Equal,
        3 => CompareFunction::LessEqual,
        4 => CompareFunction::Greater,
        5 => CompareFunction::NotEqual,
        6 => CompareFunction::GreaterEqual,
        _ => CompareFunction::Always,
    }
}

/// GX blend factor (the source side reads the destination colour for 2/3, like hardware).
fn blend_factor(f: u32, src_side: bool) -> BlendFactor {
    match f {
        0 => BlendFactor::Zero,
        1 => BlendFactor::One,
        2 if src_side => BlendFactor::Dst,
        2 => BlendFactor::Src,
        3 if src_side => BlendFactor::OneMinusDst,
        3 => BlendFactor::OneMinusSrc,
        4 => BlendFactor::SrcAlpha,
        5 => BlendFactor::OneMinusSrcAlpha,
        6 => BlendFactor::DstAlpha,
        _ => BlendFactor::OneMinusDstAlpha,
    }
}

/// Colour target state for a GX blend mode register (BP 0x41).
fn color_target(blend: u32) -> ColorTargetState {
    let enabled = blend & 1 != 0;
    let logic = (blend >> 1) & 1 != 0;
    let subtract = (blend >> 11) & 1 != 0;
    let logic_op = (blend >> 12) & 15;
    let mut write_mask = ColorWrites::empty();
    if (blend >> 3) & 1 != 0 {
        write_mask |= ColorWrites::COLOR;
    }
    if (blend >> 4) & 1 != 0 {
        write_mask |= ColorWrites::ALPHA;
    }
    let comp = |src, dst, operation| {
        let c = BlendComponent { src_factor: src, dst_factor: dst, operation };
        Some(BlendState { color: c, alpha: c })
    };
    let state = if subtract {
        comp(BlendFactor::One, BlendFactor::One, BlendOperation::ReverseSubtract)
    } else if enabled {
        comp(blend_factor((blend >> 8) & 7, true), blend_factor((blend >> 5) & 7, false), BlendOperation::Add)
    } else if logic {
        // Logic ops the blend unit can express: CLEAR, COPY, NOOP, SET.
        match logic_op {
            0 => comp(BlendFactor::Zero, BlendFactor::Zero, BlendOperation::Add),
            5 => {
                write_mask = ColorWrites::empty();
                None
            }
            15 => comp(BlendFactor::One, BlendFactor::One, BlendOperation::Max),
            _ => None,
        }
    } else {
        None
    };
    ColorTargetState { format: EFB_COLOR, blend: state, write_mask }
}

fn tev_bytes(t: &TevState) -> Vec<u8> {
    let mut b = Vec::with_capacity(TEV_STRIDE as usize);
    for st in &t.stages {
        for v in st {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    for v in t.regs.iter().chain(t.konst.iter()).flatten() {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for v in &t.params {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for v in t.fog.iter().chain(t.fog_color.iter()) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    debug_assert_eq!(b.len() as u64, TEV_UNIFORM_SIZE);
    b.resize(TEV_STRIDE as usize, 0);
    b
}

/// A decoded copy: where it reads, where it goes, and how.
struct Copy {
    at: usize,
    xfb: bool,
    clear: bool,
    src: [u32; 4],
    dst_size: (u32, u32),
    addr: u32,
    regs: [u32; 8],
    params: Vec<u8>,
}

fn decode_copies(f: &Frame, scale: u32) -> Vec<Copy> {
    let bits = |v: u32, lo: u32, n: u32| (v >> lo) & ((1 << n) - 1);
    f.copies
        .iter()
        .enumerate()
        .map(|(k, &(at, cmd, src, size, dst))| {
            let regs = f.copy_regs.get(k).copied().unwrap_or_default();
            let xfb = cmd & (1 << 14) != 0;
            let rect = [bits(src, 0, 10), bits(src, 10, 10), bits(size, 0, 10) + 1, bits(size, 10, 10) + 1];
            let half = !xfb && (cmd >> 9) & 1 != 0;
            // Copies keep the internal resolution (like Dolphin's scaled EFB copies): the
            // destination is the source rectangle times the scale (halved by the box filter).
            let (sw, sh) = (rect[2] * scale, rect[3] * scale);
            let dst_size = if half { ((sw / 2).max(1), (sh / 2).max(1)) } else { (sw, sh) };
            let fmt = if xfb { 6 } else { bits(cmd, 4, 3) | bits(cmd, 3, 1) << 3 };
            let pixfmt = regs[0] & 7;
            let depth = pixfmt == 3;
            // flags: intensity conversion, EFB pixel format, and for display copies the output
            // gamma (BP 0x52 bits 7-8: 1.0 / 1.7 / 2.2).
            let gamma = if xfb { (cmd >> 7) & 3 } else { 0 };
            let flags = ((cmd >> 15) & 1) | (pixfmt << 8) | (gamma << 16);
            let (ar, gb, z) = (regs[1], regs[2], regs[3]);
            let clear_color = [
                (ar & 0xFF) as f32 / 255.0,
                ((gb >> 8) & 0xFF) as f32 / 255.0,
                (gb & 0xFF) as f32 / 255.0,
                ((ar >> 8) & 0xFF) as f32 / 255.0,
            ];
            let mut params = Vec::with_capacity(COPY_STRIDE as usize);
            for v in rect.iter().chain([fmt, half as u32, depth as u32, flags].iter()) {
                params.extend_from_slice(&v.to_le_bytes());
            }
            for v in clear_color.iter().chain([(z & 0xFF_FFFF) as f32 / 16_777_215.0, scale as f32, 0.0, 0.0].iter()) {
                params.extend_from_slice(&v.to_le_bytes());
            }
            // Vertical copy filter (BP 0x53/0x54, 7 x 6-bit coefficients, sum normally 64):
            // weights of the rows above / at / below each pixel. Colour copies only; reset
            // state (never programmed) acts as a plain copy.
            let (f0, f1) = (regs[6], regs[7]);
            let w = |r: u32, i: u32| ((r >> (6 * i)) & 0x3F) as f32;
            let filter = if (f0 | f1) & 0xFF_FFFF == 0 {
                [0.0, 64.0, 0.0]
            } else {
                [w(f0, 0) + w(f0, 1) + w(f0, 2), w(f0, 3), w(f1, 0) + w(f1, 1) + w(f1, 2)]
            };
            // BP 0x52 bits 0/1: clamp the filter's top / bottom rows to the source rectangle.
            let clamp = (cmd & 3) as f32;
            for v in [filter[0] / 64.0, filter[1] / 64.0, filter[2] / 64.0, clamp] {
                params.extend_from_slice(&v.to_le_bytes());
            }
            params.resize(COPY_STRIDE as usize, 0);
            Copy {
                at,
                xfb,
                clear: (cmd >> 11) & 1 != 0,
                src: rect,
                dst_size,
                addr: (dst & 0xFF_FFFF) << 5,
                regs,
                params,
            }
        })
        .collect()
}

impl Gpu {
    fn new(device: &Device, queue: &Queue) -> Gpu {
        let tev_module = shader(device, "gx_tev", include_str!("gx_efb.wgsl"));
        let copy_module = shader(device, "gx_copy", include_str!("gx_copy.wgsl"));
        let blit_module = shader(device, "gx_blit", include_str!("gx_blit.wgsl"));
        let float = TextureSampleType::Float { filterable: true };

        let mut tev_entries = vec![uniform_entry(0, TEV_UNIFORM_SIZE)];
        for i in 0..8 {
            tev_entries.push(texture_entry(1 + 2 * i, float));
            tev_entries.push(sampler_entry(2 + 2 * i));
        }
        let tev_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor { label: Some("gx_tev"), entries: &tev_entries });
        let tev_layout = layout(device, &tev_bgl);

        let copy_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gx_copy"),
            entries: &[
                uniform_entry(0, COPY_UNIFORM_SIZE),
                texture_entry(1, TextureSampleType::Float { filterable: false }),
                texture_entry(2, TextureSampleType::Depth),
            ],
        });
        let copy_layout = layout(device, &copy_bgl);
        let copy_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("gx_copy"),
            layout: Some(&copy_layout),
            vertex: VertexState { module: &copy_module, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &copy_module,
                entry_point: Some("fs_copy"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState { format: EFB_COLOR, blend: None, write_mask: ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let clear_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gx_clear"),
            entries: &[uniform_entry(0, COPY_UNIFORM_SIZE)],
        });
        let clear_layout = layout(device, &clear_bgl);

        let blit_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gx_blit"),
            entries: &[
                texture_entry(0, float),
                sampler_entry(1),
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: NonZeroU64::new(16) },
                    count: None,
                },
            ],
        });
        let blit_layout = layout(device, &blit_bgl);
        let blit_sampler = device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        let white_tex = device.create_texture(&TextureDescriptor {
            label: Some("gx_white"),
            size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo { texture: &white_tex, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
            &[255; 4],
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: None },
            Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let white = white_tex.create_view(&TextureViewDescriptor::default());
        let copy_buf = device.create_buffer(&BufferDescriptor {
            label: Some("gx_copy_params"),
            size: COPY_STRIDE * 256,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Prebuild the pipelines for every GX pixel state seen in earlier sessions (plus the
        // common ones) on background threads: compiling one takes long enough to drop frames
        // when a new state first appears mid-game (e.g. a new menu or effect).
        let tev_pipelines: Arc<Mutex<HashMap<(u32, u32), RenderPipeline>>> = Default::default();
        let mut keys: Vec<(u32, u32)> = COMMON_PIPELINES.to_vec();
        for k in load_pipeline_keys() {
            if !keys.contains(&k) {
                keys.push(k);
            }
        }
        let keys = Arc::new(Mutex::new(keys));
        for _ in 0..2 {
            let (device, module, layout, keys, map) =
                (device.clone(), tev_module.clone(), tev_layout.clone(), keys.clone(), tev_pipelines.clone());
            let _ = std::thread::Builder::new().name("gx-pipeline-prewarm".into()).spawn(move || loop {
                let Some((z, b)) = keys.lock().unwrap().pop() else { break };
                if map.lock().unwrap().contains_key(&(z, b)) {
                    continue;
                }
                let p = build_tev_pipeline(&device, &module, &layout, z, b);
                map.lock().unwrap().entry((z, b)).or_insert(p);
            });
        }

        let display_buf = device.create_buffer(&BufferDescriptor {
            label: Some("gx_display"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut gpu = Gpu {
            display_buf,
            tev_module,
            tev_bgl,
            tev_layout,
            tev_pipelines,
            copy_bgl,
            copy_pipeline,
            clear_bgl,
            clear_module: copy_module,
            clear_layout,
            clear_pipelines: HashMap::new(),
            blit_module,
            blit_bgl,
            blit_layout,
            blit_pipelines: HashMap::new(),
            blit_sampler,
            blit_group: None,
            efb: None,
            xfb: None,
            rts: HashMap::new(),
            textures: HashMap::new(),
            samplers: HashMap::new(),
            white,
            vertex_buf: None,
            index_buf: None,
            tev_buf: None,
            copy_buf,
            tick: 0,
        };
        // All 8 copy-clear variants up front (small; avoids a stall on first use).
        for k in 0..8u32 {
            gpu.clear_pipeline(device, (k & 1 != 0, k & 2 != 0, k & 4 != 0));
        }
        gpu
    }

    /// Make sure the pipeline for (zmode, blend) exists. A state never seen before compiles
    /// here (once per install: it is recorded and prebuilt in the background next time).
    fn tev_pipeline(&mut self, device: &Device, zmode: u32, blend: u32) {
        if self.tev_pipelines.lock().unwrap().contains_key(&(zmode, blend)) {
            return;
        }
        let p = build_tev_pipeline(device, &self.tev_module, &self.tev_layout, zmode, blend);
        self.tev_pipelines.lock().unwrap().insert((zmode, blend), p);
        remember_pipeline_key(zmode, blend);
    }
}

/// Where per-user data lives (same folder as controls.json).
fn data_dir() -> std::path::PathBuf {
    std::env::var_os("APPDATA").map(std::path::PathBuf::from).unwrap_or_else(|| ".".into()).join("BMX-XXX-Recomp")
}

fn pipeline_cache_path() -> std::path::PathBuf {
    data_dir().join("pipelines.json")
}

/// Common GX pixel states (inherited from the MKDD port: opaque, alpha-blended, additive, Z
/// variants). BMX's own states are learned at runtime into pipelines.json.
const COMMON_PIPELINES: &[(u32, u32)] = &[
    (0x1f, 0x501c), (0x1f, 0x5018), (0x16, 0x3118), (0x17, 0x3118), (0x0d, 0x5935), (0x0d, 0x5135),
    (0x00, 0x54bd), (0x00, 0x54ad), (0x07, 0xf4ad), (0x07, 0x34a9), (0x06, 0x34b9), (0x17, 0x34a9),
    (0x16, 0x34a9), (0x07, 0x3118), (0x00, 0x34b9), (0x00, 0x3118), (0x17, 0x34b9), (0x06, 0x3118),
];

fn load_pipeline_keys() -> Vec<(u32, u32)> {
    std::fs::read_to_string(pipeline_cache_path())
        .ok()
        .and_then(|t| serde_json::from_str::<Vec<(u32, u32)>>(&t).ok())
        .unwrap_or_default()
}

fn remember_pipeline_key(zmode: u32, blend: u32) {
    let mut keys = load_pipeline_keys();
    if keys.contains(&(zmode, blend)) {
        return;
    }
    keys.push((zmode, blend));
    let _ = std::fs::create_dir_all(data_dir());
    if let Ok(t) = serde_json::to_string(&keys) {
        let _ = std::fs::write(pipeline_cache_path(), t);
    }
}

fn build_tev_pipeline(device: &Device, module: &ShaderModule, layout: &PipelineLayout, zmode: u32, blend: u32) -> RenderPipeline {
    {
        let attrs = vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4];
        let (depth_compare, depth_write) = if zmode & 1 != 0 {
            (compare((zmode >> 1) & 7), (zmode >> 4) & 1 != 0)
        } else {
            (CompareFunction::Always, false)
        };
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("gx_tev"),
            layout: Some(layout),
            vertex: VertexState {
                module,
                entry_point: Some("vs_tev"),
                compilation_options: Default::default(),
                buffers: &[VertexBufferLayout {
                    array_stride: (VERTEX_FLOATS * 4) as u64,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &attrs,
                }],
            },
            primitive: PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(DepthStencilState {
                format: EFB_DEPTH,
                depth_write_enabled: Some(depth_write),
                depth_compare: Some(depth_compare),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module,
                entry_point: Some("fs_tev"),
                compilation_options: Default::default(),
                targets: &[Some(color_target(blend))],
            }),
            multiview_mask: None,
            cache: None,
        })
    }
}

impl Gpu {
    /// Clear pipeline for (colour update, alpha update, Z update).
    fn clear_pipeline(&mut self, device: &Device, key: (bool, bool, bool)) {
        if self.clear_pipelines.contains_key(&key) {
            return;
        }
        let mut write_mask = ColorWrites::empty();
        if key.0 {
            write_mask |= ColorWrites::COLOR;
        }
        if key.1 {
            write_mask |= ColorWrites::ALPHA;
        }
        let p = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("gx_clear"),
            layout: Some(&self.clear_layout),
            vertex: VertexState { module: &self.clear_module, entry_point: Some("vs_clear"), compilation_options: Default::default(), buffers: &[] },
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: EFB_DEPTH,
                depth_write_enabled: Some(key.2),
                depth_compare: Some(CompareFunction::Always),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &self.clear_module,
                entry_point: Some("fs_clear"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState { format: EFB_COLOR, blend: None, write_mask })],
            }),
            multiview_mask: None,
            cache: None,
        });
        self.clear_pipelines.insert(key, p);
    }

    fn sampler(&mut self, device: &Device, key: (u8, u8, bool)) -> Sampler {
        let mode = |w: u8| match w {
            1 => AddressMode::Repeat,
            2 => AddressMode::MirrorRepeat,
            _ => AddressMode::ClampToEdge,
        };
        self.samplers
            .entry(key)
            .or_insert_with(|| {
                let filter = if key.2 { FilterMode::Linear } else { FilterMode::Nearest };
                device.create_sampler(&SamplerDescriptor {
                    address_mode_u: mode(key.0),
                    address_mode_v: mode(key.1),
                    mag_filter: filter,
                    min_filter: filter,
                    ..Default::default()
                })
            })
            .clone()
    }

    /// View + sampler for a texture slot key (decoded texture, EFB copy, or white).
    fn slot(&mut self, device: &Device, queue: &Queue, frames: &FrameShared, key: Option<u64>) -> (TextureView, Sampler) {
        let Some(key) = key else {
            return (self.white.clone(), self.sampler(device, (0, 0, false)));
        };
        if key & 0xFFFF_0000_0000_0000 == RT_TAG {
            let mode = ((key >> 32) & 0xFFFF) as u32;
            let s = self.sampler(device, ((mode & 3) as u8, ((mode >> 2) & 3) as u8, (mode >> 4) & 1 != 0));
            let view = self.rts.get(&(key as u32)).map_or_else(|| self.white.clone(), |t| t.view.clone());
            return (view, s);
        }
        let tick = self.tick;
        if let Some(t) = self.textures.get_mut(&key) {
            t.used = tick;
            let (v, sk) = (t.view.clone(), t.sampler_key);
            return (v, self.sampler(device, sk));
        }
        let Some(t) = frames.texture(key) else {
            return (self.white.clone(), self.sampler(device, (0, 0, false)));
        };
        let size = Extent3d { width: t.width, height: t.height, depth_or_array_layers: 1 };
        let tex = device.create_texture(&TextureDescriptor {
            label: Some("gx_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            // Non-sRGB: TEV math runs on the raw (gamma-space) values like the hardware does.
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
            t.rgba(),
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * t.width), rows_per_image: None },
            size,
        );
        let view = tex.create_view(&TextureViewDescriptor::default());
        let sampler_key = (t.wrap_s, t.wrap_t, t.linear);
        self.textures.insert(key, GameTexture { view: view.clone(), sampler_key, used: tick });
        (view, self.sampler(device, sampler_key))
    }

    fn ensure_buffer(device: &Device, buf: &mut Option<Buffer>, size: u64, usage: BufferUsages, label: &str) {
        if buf.as_ref().is_none_or(|b| b.size() < size) {
            *buf = Some(device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: size.next_power_of_two().max(1 << 16),
                usage: usage | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
    }

    /// Replay one GX frame: draws, EFB copies, clears; ends with the XFB updated.
    /// `scale`: internal resolution multiplier (the EFB and every copy are rendered that many
    /// times larger than the console's 640x528 and downsampled for display: supersampling).
    fn render_frame(&mut self, device: &Device, queue: &Queue, enc: &mut CommandEncoder, frames: &FrameShared, f: &Frame, scale: u32) {
        self.tick += 1;
        if self.tick % 60 == 0 {
            let cutoff = self.tick.saturating_sub(600);
            self.textures.retain(|_, t| t.used >= cutoff);
        }
        let scale = scale.clamp(1, 8);
        let efb_size = (f.efb[0] as u32 * scale, f.efb[1] as u32 * scale);
        if self.efb.as_ref().is_none_or(|(c, _)| c.size != efb_size) {
            let color = Target::new(device, "gx_efb_color", efb_size, EFB_COLOR);
            let depth = Target::new(device, "gx_efb_depth", efb_size, EFB_DEPTH);
            // Power-on state: black, depth at the far plane.
            enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("gx_efb_init"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &color.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(wgpu::Color::BLACK), store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth.view,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.efb = Some((color, depth));
        }

        let copies = decode_copies(f, scale);
        for c in &copies {
            if c.xfb {
                if self.xfb.as_ref().is_none_or(|t| t.size != c.dst_size) {
                    self.xfb = Some(Target::new(device, "gx_xfb", c.dst_size, EFB_COLOR));
                    self.blit_group = None;
                }
            } else if self.rts.get(&c.addr).is_none_or(|t| t.size != c.dst_size) {
                self.rts.insert(c.addr, Target::new(device, "gx_efb_copy", c.dst_size, EFB_COLOR));
            }
        }

        // Vertices, triangle indices and TEV uniforms for every batch. The decoder's vertices
        // are already in the GPU layout: each batch is one block copy, drawn indexed with its
        // vertex block as the base.
        let total_v: usize = f.batches.iter().map(|b| b.verts.len()).sum();
        let total_i: usize = f.batches.iter().map(|b| b.indices.len()).sum();
        let mut verts: Vec<gc_runtime::gx::GpuVertex> = Vec::with_capacity(total_v);
        let mut indices: Vec<u32> = Vec::with_capacity(total_i);
        let mut tev: Vec<u8> = Vec::with_capacity(f.batches.len() * TEV_STRIDE as usize);
        let mut ranges = Vec::with_capacity(f.batches.len());
        for b in &f.batches {
            let base = verts.len() as i32;
            let start = indices.len() as u32;
            verts.extend_from_slice(&b.verts);
            indices.extend_from_slice(&b.indices);
            ranges.push((start..indices.len() as u32, base));
            tev.extend_from_slice(&tev_bytes(&b.tev));
        }
        let vbytes = verts.len() * VERTEX_FLOATS * 4;
        Self::ensure_buffer(device, &mut self.vertex_buf, vbytes as u64, BufferUsages::VERTEX, "gx_vertices");
        Self::ensure_buffer(device, &mut self.index_buf, (indices.len() * 4) as u64, BufferUsages::INDEX, "gx_indices");
        Self::ensure_buffer(device, &mut self.tev_buf, tev.len() as u64, BufferUsages::UNIFORM, "gx_tev_uniforms");
        let (vbuf, ibuf, tbuf) = (self.vertex_buf.clone().unwrap(), self.index_buf.clone().unwrap(), self.tev_buf.clone().unwrap());
        if !verts.is_empty() {
            // SAFETY: GpuVertex is repr(C) plain f32 data, viewed as bytes (the GPU wants
            // little-endian f32, which is the host layout on every supported target).
            let bytes = unsafe { std::slice::from_raw_parts(verts.as_ptr() as *const u8, vbytes) };
            queue.write_buffer(&vbuf, 0, bytes);
        }
        if !indices.is_empty() {
            // SAFETY: u32 data viewed as bytes (little-endian host, as above).
            let bytes = unsafe { std::slice::from_raw_parts(indices.as_ptr() as *const u8, indices.len() * 4) };
            queue.write_buffer(&ibuf, 0, bytes);
        }
        if !tev.is_empty() {
            queue.write_buffer(&tbuf, 0, &tev);
        }
        let copy_params: Vec<u8> = copies.iter().take(256).flat_map(|c| c.params.iter().copied()).collect();
        if !copy_params.is_empty() {
            queue.write_buffer(&self.copy_buf, 0, &copy_params);
        }

        // Pipelines and bind groups (textures resolved now; EFB-copy views exist already).
        let mut groups: HashMap<[Option<u64>; 8], BindGroup> = HashMap::new();
        let mut batch_group = Vec::with_capacity(f.batches.len());
        for b in &f.batches {
            self.tev_pipeline(device, b.tev.pipe[0], b.tev.pipe[1]);
            let key = b.tev.textures;
            if !groups.contains_key(&key) {
                let slots: Vec<(TextureView, Sampler)> = key.iter().map(|k| self.slot(device, queue, frames, *k)).collect();
                let mut entries = vec![BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding { buffer: &tbuf, offset: 0, size: NonZeroU64::new(TEV_UNIFORM_SIZE) }),
                }];
                for (i, (v, s)) in slots.iter().enumerate() {
                    entries.push(BindGroupEntry { binding: 1 + 2 * i as u32, resource: BindingResource::TextureView(v) });
                    entries.push(BindGroupEntry { binding: 2 + 2 * i as u32, resource: BindingResource::Sampler(s) });
                }
                let g = device.create_bind_group(&BindGroupDescriptor { label: Some("gx_tev"), layout: &self.tev_bgl, entries: &entries });
                groups.insert(key, g);
            }
            batch_group.push(key);
        }
        let pipes: HashMap<(u32, u32), RenderPipeline> = {
            let map = self.tev_pipelines.lock().unwrap();
            f.batches.iter().map(|b| (b.tev.pipe[0], b.tev.pipe[1])).filter_map(|k| map.get(&k).map(|p| (k, p.clone()))).collect()
        };
        for c in copies.iter().filter(|c| c.clear) {
            let (z, b) = (c.regs[4], c.regs[5]);
            self.clear_pipeline(device, ((b >> 3) & 1 != 0, (b >> 4) & 1 != 0, (z >> 4) & 1 != 0));
        }
        let (efb_color, efb_depth) = {
            let (c, d) = self.efb.as_ref().unwrap();
            (c.view.clone(), d.view.clone())
        };
        let copy_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("gx_copy"),
            layout: &self.copy_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding { buffer: &self.copy_buf, offset: 0, size: NonZeroU64::new(COPY_UNIFORM_SIZE) }),
                },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&efb_color) },
                BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&efb_depth) },
            ],
        });
        let clear_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("gx_clear"),
            layout: &self.clear_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding { buffer: &self.copy_buf, offset: 0, size: NonZeroU64::new(COPY_UNIFORM_SIZE) }),
            }],
        });

        let efb_pass = |enc: &mut CommandEncoder, label: &'static str| -> RenderPass<'static> {
            enc.begin_render_pass(&RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &efb_color,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Load, store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &efb_depth,
                    depth_ops: Some(Operations { load: LoadOp::Load, store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            })
            .forget_lifetime()
        };

        let mut next = 0usize;
        let mut draw_until = |enc: &mut CommandEncoder, end: usize| {
            if next >= end {
                return;
            }
            let mut pass = efb_pass(enc, "gx_draws");
            pass.set_vertex_buffer(0, vbuf.slice(..));
            pass.set_index_buffer(ibuf.slice(..), IndexFormat::Uint32);
            for i in next..end.min(f.batches.len()) {
                let b = &f.batches[i];
                let (r, base) = &ranges[i];
                let [x0, y0, x1, y1] = b.rect.map(|v| v * scale);
                let (x1, y1) = (x1.min(efb_size.0), y1.min(efb_size.1));
                if r.is_empty() || x1 <= x0 || y1 <= y0 {
                    continue;
                }
                pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
                let Some(pipe) = pipes.get(&(b.tev.pipe[0], b.tev.pipe[1])) else { continue };
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, &groups[&batch_group[i]], &[(i as u64 * TEV_STRIDE) as u32]);
                pass.draw_indexed(r.clone(), *base, 0..1);
            }
            next = end;
        };

        for (k, c) in copies.iter().enumerate().take(256) {
            draw_until(enc, c.at);
            let offset = (k as u64 * COPY_STRIDE) as u32;
            let dst = if c.xfb { self.xfb.as_ref() } else { self.rts.get(&c.addr) };
            if let Some(dst) = dst {
                let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
                    label: Some("gx_efb_copy"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &dst.view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations { load: LoadOp::Load, store: StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.copy_pipeline);
                pass.set_bind_group(0, &copy_group, &[offset]);
                pass.draw(0..3, 0..1);
            }
            if c.clear {
                let [x, y, w, h] = c.src.map(|v| v * scale);
                let (x1, y1) = ((x + w).min(efb_size.0), (y + h).min(efb_size.1));
                if x1 > x && y1 > y {
                    let (z, b) = (c.regs[4], c.regs[5]);
                    let key = ((b >> 3) & 1 != 0, (b >> 4) & 1 != 0, (z >> 4) & 1 != 0);
                    let mut pass = efb_pass(enc, "gx_efb_clear");
                    pass.set_scissor_rect(x, y, x1 - x, y1 - y);
                    pass.set_pipeline(&self.clear_pipelines[&key]);
                    pass.set_bind_group(0, &clear_group, &[offset]);
                    pass.draw(0..3, 0..1);
                }
            }
        }
        draw_until(enc, f.batches.len());
    }

    /// Show the XFB in the window (4:3, letterboxed).
    fn blit(&mut self, device: &Device, queue: &Queue, enc: &mut CommandEncoder, target: &ViewTarget, gamma: f32) {
        let disp: Vec<u8> = [gamma, 0.0, 0.0, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        queue.write_buffer(&self.display_buf, 0, &disp);
        let format = target.main_texture_format();
        if !self.blit_pipelines.contains_key(&format) {
            let p = device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("gx_blit"),
                layout: Some(&self.blit_layout),
                vertex: VertexState { module: &self.blit_module, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &self.blit_module,
                    entry_point: Some(if format.is_srgb() { "fs_blit_srgb" } else { "fs_blit" }),
                    compilation_options: Default::default(),
                    targets: &[Some(ColorTargetState { format, blend: None, write_mask: ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache: None,
            });
            self.blit_pipelines.insert(format, p);
        }
        let Some(xfb) = self.xfb.as_ref() else { return };
        if self.blit_group.is_none() {
            self.blit_group = Some(device.create_bind_group(&BindGroupDescriptor {
                label: Some("gx_blit"),
                layout: &self.blit_bgl,
                entries: &[
                    BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&xfb.view) },
                    BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&self.blit_sampler) },
                    BindGroupEntry { binding: 2, resource: self.display_buf.as_entire_binding() },
                ],
            }));
        }
        let size = target.main_texture().size();
        let (w, h) = (size.width as f32, size.height as f32);
        let (vw, vh) = if w / h > 4.0 / 3.0 { (h * 4.0 / 3.0, h) } else { (w, w * 3.0 / 4.0) };
        let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
            label: Some("gx_blit"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: target.main_texture_view(),
                depth_slice: None,
                resolve_target: None,
                ops: Operations { load: LoadOp::Clear(wgpu::Color::BLACK), store: StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport((w - vw) / 2.0, (h - vh) / 2.0, vw, vh, 0.0, 1.0);
        pass.set_pipeline(&self.blit_pipelines[&format]);
        pass.set_bind_group(0, self.blit_group.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
    }
}

fn gx_pass(
    view: ViewQuery<&ViewTarget>,
    link: Option<Res<GxLink>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut st: ResMut<GxState>,
    settings: Option<Res<crate::settings::Settings>>,
    mut ctx: RenderContext,
) {
    let target = view.into_inner();
    let Some(link) = link else { return };
    let device = device.wgpu_device();
    let GxState { gpu, shown } = &mut *st;
    let gpu = gpu.get_or_insert_with(|| Gpu::new(device, &queue));
    let n = link.0.count();
    if n != *shown {
        *shown = n;
        let f = link.0.latest();
        let scale = settings.as_ref().map_or(1, |s| s.internal_res);
        gpu.render_frame(device, &queue, ctx.command_encoder(), &link.0, &f, scale);
    }
    let gamma = settings.map_or(1.0, |s| s.gamma);
    gpu.blit(device, &queue, ctx.command_encoder(), target, gamma);
}

#[cfg(test)]
mod tests {
    /// The GX shaders parse and validate (what wgpu does at pipeline creation).
    #[test]
    fn shaders_validate() {
        for (name, src) in [
            ("gx_efb.wgsl", include_str!("gx_efb.wgsl")),
            ("gx_copy.wgsl", include_str!("gx_copy.wgsl")),
            ("gx_blit.wgsl", include_str!("gx_blit.wgsl")),
        ] {
            let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }
}
