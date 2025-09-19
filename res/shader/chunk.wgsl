@group(0) @binding(0) var<uniform> view_projection: mat4x4<f32>;
@group(0) @binding(1) var<uniform> sun: vec3<f32>;
@group(0) @binding(2) var texture: texture_2d<f32>;
@group(0) @binding(3) var sampler_: sampler;

@group(1) @binding(0) var<uniform> model: mat4x4<f32>;
@group(1) @binding(1) var<uniform> normal: vec3<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
};

fn unpack_position(packed: u32) -> vec4<f32> {
    let n = packed & 65535;
    let x = n % 33;
    let y = (n / 33) % 33;
    let z = n / 1089;
    return vec4<f32>(f32(x), f32(y), f32(z), 1.0);
}

fn unpack_uv(packed: u32) -> vec2<f32> {
    let n = (packed & 536805376) >> 16;
    let u = f32(n % 64) / 64.0;
    let v = f32(n / 64) / 64.0;
    return vec2<f32>(u, v);
}

fn unpack_normal(packed: u32) -> vec3<f32> {
    return normal;
}

@vertex
fn vs_main(@location(0) packed: u32) -> VertexOutput {
    var output: VertexOutput;
    output.position = view_projection * model * unpack_position(packed);
    output.uv = unpack_uv(packed);
    output.normal = unpack_normal(packed);
    return output;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let sample = textureSample(texture, sampler_, vertex.uv);
    let cos_theta = clamp(dot(vertex.normal, sun), 0.0, 1.0);
    return vec4<f32>(sample.rgb * (1.0 - 0.3 * cos_theta), sample.a);
}
