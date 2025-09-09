#version 330

in vec2 vert_uv;
in vec3 vert_normal;

out vec4 frag_color;

uniform sampler2D texture_atlas;

void main() {
    const vec3 sun = normalize(vec3(1., -1., 0.5));
    float cos_theta = clamp(dot(vert_normal, sun), 0.0, 1.0);
    frag_color = texture(texture_atlas, vert_uv) - cos_theta * 0.3;
}
