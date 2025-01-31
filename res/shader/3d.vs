#version 140

in vec2 position;
in vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 vert_color;

void main() {
    gl_Position = projection * view * model * vec4(position.xy, 0.0, 1.0);
    vert_color = color;
}

