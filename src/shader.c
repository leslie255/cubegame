#include "shader.h"

/// Panic on compile failure.
static inline GLuint compile_shader(usize length, const char source[length], GLenum shader_type) {
  auto shader = glCreateShader(shader_type);
  glShaderSource(shader, 1, REF((const char *)&source[0]), REF((GLint)length));
  glCompileShader(shader);
  GLint success = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    GLsizei info_log_length = 0;
    glGetShaderInfoLog(shader, sizeof(shader_info_log), &info_log_length, shader_info_log);
    printf("Error compiling shader, reason:\n");
    fwrite(shader_info_log, 1, (usize)info_log_length, stderr);
    exit(255);
  }
  return shader;
}

/// Panic on link failure.
static inline bool link_shader_program(GLuint program) {
  glLinkProgram(program);
  GLint success = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
    GLsizei log_len = 0;
    glGetShaderInfoLog(program, sizeof(shader_info_log), &log_len, shader_info_log);
    printf("Error linking shader, reason:\n");
    fwrite(shader_info_log, 1, (usize)log_len, stderr);
    exit(255);
  }
  return success;
}

ShaderProgram
shader_init(usize vs_src_len, const char vs_src[vs_src_len], usize fs_src_len, const char fs_src[fs_src_len]) {
  GLuint vertex_shader = compile_shader(vs_src_len, vs_src, GL_VERTEX_SHADER);
  GLuint fragment_shader = compile_shader(fs_src_len, fs_src, GL_FRAGMENT_SHADER);
  GLuint gl_handle = glCreateProgram();
  glAttachShader(gl_handle, vertex_shader);
  glAttachShader(gl_handle, fragment_shader);
  link_shader_program(gl_handle);
  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);
  return (ShaderProgram){.gl_handle = gl_handle};
}

void shader_cleanup(ShaderProgram *shader) {
  glDeleteProgram(shader->gl_handle);
}

void shader_use(ShaderProgram shader) {
  glUseProgram(shader.gl_handle);
}
