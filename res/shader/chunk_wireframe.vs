#version 140

in vec3 position;
in vec2 uv;

uniform mat4 model_view;
uniform mat4 view;
uniform mat4 projection;

out vec2 vert_uv;

void main() {
    gl_Position = projection * model_view * vec4(position, 1.0);
    vert_uv = uv;
}
