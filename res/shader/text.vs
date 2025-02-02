#version 140

in vec2 position;
in vec2 uv;
out vec2 vert_uv;

uniform mat4 model;
uniform mat4 projection;

void main() {
    gl_Position = projection * model * vec4(position.xy, 0.0, 1.0);
    vert_uv = uv;
}
