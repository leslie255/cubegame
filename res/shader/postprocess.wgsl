@group(0) @binding(0) var color_texture: texture_2d<f32>;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var sampler_: sampler;
@group(0) @binding(3) var<uniform> gamma: f32;

struct VertexInput {
    @builtin(vertex_index) i_vertex: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var result: VertexOutput;
    // First two is position, latter two is UV.
    const vertices = array<vec4<f32>, 6>(
        /* 0 */ vec4<f32>(-1., -1., 0., 1.), // bottom left
        /* 1 */ vec4<f32>( 1., -1., 1., 1.), // bottom right
        /* 2 */ vec4<f32>( 1.,  1., 1., 0.), // top    right
        /* 3 */ vec4<f32>(-1.,  1., 0., 0.), // top    left
        /* 4 */ vec4<f32>( 1.,  1., 1., 0.), // top    right
        /* 5 */ vec4<f32>(-1., -1., 0., 1.), // bottom left
    );
    let vertex_data = vertices[input.i_vertex];
    result.position = vec4<f32>(vertex_data.xy, 0.0, 1.0);
    result.uv = vertex_data.zw;
    return result;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let sample = textureSample(color_texture, sampler_, vertex.uv);
    let depth_sample = textureSample(depth_texture, sampler_, vertex.uv);
    if (depth_sample == 1.0) {
        return vec4<f32>(0.8, 0.95, 1.0, 1.0);
    }
    return vec4<f32>(
        pow(sample.r, gamma),
        pow(sample.g, gamma),
        pow(sample.b, gamma),
        sample.a);
}
