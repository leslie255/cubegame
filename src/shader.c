#include "shader.h"

/// Panic on compile failure.
static inline GLuint compile_shader(ShaderSouce source) {
  auto shader = glCreateShader(source.type);
  glShaderSource(shader, 1, REF((const char *)&source.source[0]), REF((GLint)source.length));
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

ShaderProgram shader_init(usize n_sources, const ShaderSouce sources[n_sources]) {
  GLuint shader_program = glCreateProgram();
  GLuint *compiled_shaders = alloca(sizeof(GLuint[n_sources]));
  for (usize i = 0; i < n_sources; ++i) {
    GLuint shader = compile_shader(sources[i]);
    compiled_shaders[i] = shader;
    glAttachShader(shader_program, shader);
  }
  link_shader_program(shader_program);
  for (usize i = 0; i < n_sources; ++i)
    glDeleteShader(compiled_shaders[i]);
  return (ShaderProgram){.gl = shader_program};
}

void shader_cleanup(ShaderProgram *shader) {
  glDeleteProgram(shader->gl);
}

void shader_use(ShaderProgram shader) {
  glUseProgram(shader.gl);
}

