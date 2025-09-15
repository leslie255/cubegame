@group(0) @binding(0) var<uniform> model_view: mat4x4<f32>;
@group(0) @binding(1) var<uniform> projection: mat4x4<f32>;
@group(0) @binding(2) var<uniform> fg_color: vec4<f32>;
@group(0) @binding(3) var<uniform> bg_color: vec4<f32>;
@group(0) @binding(4) var texture: texture_2d<f32>;
@group(0) @binding(5) var sampler_: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
};

struct InstanceInput {
    @location(2) position_translation: vec2<f32>,
    @location(3) uv_translation: vec2<f32>,
};

struct VertexOutput {
    @location(0) uv: vec2<f32>,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput, instance: InstanceInput) -> VertexOutput {
    var result: VertexOutput;
    result.uv = input.uv + instance.uv_translation;
    let position_world = input.position.xy + instance.position_translation;
    result.position = projection * model_view * vec4<f32>(position_world.xy, 0.0, 1.0);
    return result;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let sample = textureSample(texture, sampler_, vertex.uv);
    return mix(bg_color, fg_color, sample.a);
}

