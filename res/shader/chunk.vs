#version 330

in vec3 position;
in vec2 uv;

uniform mat4 model_view;
uniform mat4 projection;
uniform vec3 normal;

out vec2 vert_uv;
out vec3 vert_normal;

void main() {
    gl_Position = projection * model_view * vec4(position, 1.0);
    vert_uv = uv;
    vert_normal = normal;
}
