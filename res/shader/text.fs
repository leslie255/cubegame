#version 330

in vec2 vert_uv;
out vec4 color;

uniform sampler2D tex;
uniform vec4 fg_color;
uniform vec4 bg_color;

void main() {
    color = texture(tex, vert_uv);
    color = vec4(
        color.a * fg_color.r,
        color.a * fg_color.g,
        color.a * fg_color.b,
        color.a * fg_color.a);
    color += vec4(
        (1 - color.a) * bg_color.r,
        (1 - color.a) * bg_color.g,
        (1 - color.a) * bg_color.b,
        (1 - color.a) * bg_color.a);
}
