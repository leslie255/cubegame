#version 140

in vec2 position;
in vec2 uv;
out vec2 vert_uv;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 uv_matrix;

void main() {
    gl_Position = projection * view * model * vec4(position.xy, 0.0, 1.0);
    vert_uv = (uv_matrix * vec3(uv, 1.0)).xy;
}
