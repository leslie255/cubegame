#version 140

in vec2 vert_uv;
out vec4 frag_color;

uniform sampler2D texture_atlas;

void main() {
    frag_color = texture(texture_atlas, vert_uv) * 0.4 + vec4(0.6, 0.6, 0.6, 0.6);
}
