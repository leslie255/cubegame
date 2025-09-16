@group(0) @binding(0) var<uniform> view_projection: mat4x4<f32>;
@group(0) @binding(1) var<uniform> sun: vec3<f32>;
@group(0) @binding(2) var texture: texture_2d<f32>;
@group(0) @binding(3) var sampler_: sampler;

@group(1) @binding(0) var<uniform> model: mat4x4<f32>;
@group(1) @binding(1) var<uniform> normal: vec3<f32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = view_projection * model * vec4<f32>(input.position, 1.0);
    output.uv = input.uv;
    output.normal = normal;
    return output;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let cos_theta = clamp(dot(vertex.normal, sun), 0.0, 1.0);
    let sample = textureSample(texture, sampler_, vertex.uv);
    return vec4<f32>(sample.rgb * (1.0 - 0.3 * cos_theta), sample.a);
}
