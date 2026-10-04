// EFB copies and copy-clears (see gxrender.rs).


// ---- EFB copies, clears and display ----

struct CopyParams {
    // x, y, w, h of the source rectangle in EFB pixels
    src: vec4<u32>,
    // x = copy format, y = 2x2 box filter, z = source (0 colour, 1 depth), w = flags:
    // bit 0 intensity (YUV) conversion, bits 8-10 EFB pixel format, bits 16-17 output gamma
    mode: vec4<u32>,
    // clear colour (gamma space RGBA) and clear depth (clear_z.x) for fs_clear; clear_z.y is
    // the internal resolution scale (src is in console EFB pixels)
    clear_color: vec4<f32>,
    clear_z: vec4<f32>,
    // xyz = vertical copy filter weights for the rows above / at / below (colour copies),
    // w = clamp flags (bit 0 top, bit 1 bottom: rows outside the source rect clamp to it)
    vfilter: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cp: CopyParams;
@group(0) @binding(1) var efb_color: texture_2d<f32>;
@group(0) @binding(2) var efb_depth: texture_depth_2d;

struct FullOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> FullOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: FullOut;
    o.pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    o.uv = uv;
    return o;
}

@vertex
fn vs_clear(@builtin(vertex_index) i: u32) -> FullOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: FullOut;
    o.pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, cp.clear_z.x, 1.0);
    o.uv = uv;
    return o;
}

@fragment
fn fs_clear(i: FullOut) -> @location(0) vec4<f32> {
    return cp.clear_color;
}

fn q(v: f32, levels: f32) -> f32 {
    return round(clamp(v, 0.0, 1.0) * levels) / levels;
}

// One EFB texel as the copy unit reads it (pixel format applied).
fn efb_texel(p: vec2<i32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(efb_color));
    let c = clamp(p, vec2<i32>(0), dims - vec2<i32>(1));
    if (cp.mode.z == 1u) {
        // Depth: 24-bit value split into bytes R (high), G (middle), B (low).
        let z = u32(round(clamp(textureLoad(efb_depth, c, 0), 0.0, 1.0) * 16777215.0));
        return vec4<f32>(f32(z >> 16u), f32((z >> 8u) & 255u), f32(z & 255u), 255.0) / 255.0;
    }
    var t = textureLoad(efb_color, c, 0);
    let pixfmt = (cp.mode.w >> 8u) & 7u;
    if (pixfmt == 0u) { t.a = 1.0; }                        // RGB8_Z24: no alpha
    if (pixfmt == 1u) {                                      // RGBA6_Z24
        t = vec4<f32>(q(t.r, 63.0), q(t.g, 63.0), q(t.b, 63.0), q(t.a, 63.0));
    }
    if (pixfmt == 2u) {                                      // RGB565_Z16
        t = vec4<f32>(q(t.r, 31.0), q(t.g, 63.0), q(t.b, 31.0), 1.0);
    }
    return t;
}

fn scale() -> i32 {
    return max(i32(cp.clear_z.y), 1);
}

// Source row `y` (scaled pixels) with the copy's top/bottom clamping applied.
fn copy_row(y: i32) -> i32 {
    let flags = u32(cp.vfilter.w);
    var r = y;
    if ((flags & 1u) != 0u) { r = max(r, i32(cp.src.y) * scale()); }
    if ((flags & 2u) != 0u) { r = min(r, i32(cp.src.y + cp.src.w) * scale() - 1); }
    return r;
}

// Colour at EFB position `s` (top-left of the 2x2 block when scaling by half) for row offset
// `dy`: the box filter averages the block, like the hardware's half-size copy.
fn copy_tap(s: vec2<i32>, dy: i32) -> vec4<f32> {
    if (cp.mode.y == 1u) {
        let r0 = copy_row(s.y + dy);
        let r1 = copy_row(s.y + 1 + dy);
        return (efb_texel(vec2<i32>(s.x, r0)) + efb_texel(vec2<i32>(s.x + 1, r0))
              + efb_texel(vec2<i32>(s.x, r1)) + efb_texel(vec2<i32>(s.x + 1, r1))) * 0.25;
    }
    return efb_texel(vec2<i32>(s.x, copy_row(s.y + dy)));
}

@fragment
fn fs_copy(i: FullOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(i.pos.xy));
    let base = vec2<i32>(cp.src.xy) * scale();
    let step = select(1, 2, cp.mode.y == 1u);
    let s = base + step * p;
    var t: vec4<f32>;
    if (cp.mode.z == 1u) {
        // Depth copies: point sampled, never filtered (a blend of depths across an edge is a
        // depth that exists nowhere in the scene).
        t = efb_texel(s);
    } else {
        // Colour copies: the vertical copy filter over three rows (one console line apart).
        let f = cp.vfilter;
        t = copy_tap(s, -scale()) * f.x + copy_tap(s, 0) * f.y + copy_tap(s, scale()) * f.z;
        t = clamp(t, vec4<f32>(0.0), vec4<f32>(1.0));
    }
    let gidx = (cp.mode.w >> 16u) & 3u;
    if (gidx == 1u) { t = vec4<f32>(pow(t.rgb, vec3<f32>(1.0 / 1.7)), t.a); }
    if (gidx >= 2u) { t = vec4<f32>(pow(t.rgb, vec3<f32>(1.0 / 2.2)), t.a); }
    let fmt = cp.mode.x;
    // Intensity formats can convert RGB to luma (Y of YUV) first.
    if ((cp.mode.w & 1u) == 1u && fmt <= 3u) {
        let y = clamp(dot(t.rgb, vec3<f32>(0.257, 0.504, 0.098)) + 16.0 / 255.0, 0.0, 1.0);
        t = vec4<f32>(y, y, y, t.a);
    }
    // Result as the matching texture format decodes it (I4/I8/IA4/IA8/RGB565/RGB5A3/RGBA8).
    switch (fmt) {
        case 0u: { let r = q(t.r, 15.0); return vec4<f32>(r, r, r, r); }
        case 1u, 8u: { return vec4<f32>(t.r, t.r, t.r, t.r); }
        case 2u: { let r = q(t.r, 15.0); return vec4<f32>(r, r, r, q(t.a, 15.0)); }
        case 3u: { return vec4<f32>(t.r, t.r, t.r, t.a); }
        case 4u: { return vec4<f32>(q(t.r, 31.0), q(t.g, 63.0), q(t.b, 31.0), 1.0); }
        case 5u: {
            if (t.a >= 1.0) { return vec4<f32>(q(t.r, 31.0), q(t.g, 31.0), q(t.b, 31.0), 1.0); }
            return vec4<f32>(q(t.r, 15.0), q(t.g, 15.0), q(t.b, 15.0), q(t.a, 7.0));
        }
        case 7u: { return vec4<f32>(t.a, t.a, t.a, t.a); }
        case 9u: { return vec4<f32>(t.g, t.g, t.g, t.g); }
        case 10u: { return vec4<f32>(t.b, t.b, t.b, t.b); }
        case 11u: { return vec4<f32>(t.g, t.g, t.g, t.r); }
        case 12u: { return vec4<f32>(t.b, t.b, t.b, t.g); }
        default: { return t; }
    }
}

