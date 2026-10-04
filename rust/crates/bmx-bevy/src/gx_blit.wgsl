// The copied XFB to the window (see gxrender.rs).

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

@group(0) @binding(0) var xfb: texture_2d<f32>;
@group(0) @binding(1) var xfb_samp: sampler;
// x = display gamma (1.0 = unchanged)
@group(0) @binding(2) var<uniform> disp: vec4<f32>;

fn graded(c: vec3<f32>) -> vec3<f32> {
    return pow(c, vec3<f32>(1.0 / disp.x));
}

// Downsampling the supersampled frame: four bilinear taps spread over the output pixel's
// footprint (a box filter over ~4x4 source pixels), which turns the higher internal
// resolution into smooth, anti-aliased edges.
fn resolve(uv: vec2<f32>) -> vec3<f32> {
    let q = fwidth(uv) * 0.25;
    return (textureSample(xfb, xfb_samp, uv + vec2<f32>(-q.x, -q.y)).rgb
          + textureSample(xfb, xfb_samp, uv + vec2<f32>(q.x, -q.y)).rgb
          + textureSample(xfb, xfb_samp, uv + vec2<f32>(-q.x, q.y)).rgb
          + textureSample(xfb, xfb_samp, uv + vec2<f32>(q.x, q.y)).rgb) * 0.25;
}

@fragment
fn fs_blit(i: FullOut) -> @location(0) vec4<f32> {
    return vec4<f32>(graded(resolve(i.uv)), 1.0);
}

// For sRGB targets: the XFB holds gamma-space values; the target re-encodes on write.
@fragment
fn fs_blit_srgb(i: FullOut) -> @location(0) vec4<f32> {
    return vec4<f32>(pow(graded(resolve(i.uv)), vec3<f32>(2.2)), 1.0);
}
