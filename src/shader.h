#pragma once

#include "common.h"

#include <glad/glad.h>
#include <cglm/cglm.h>

static char shader_info_log[512] = {};

typedef struct shader_source {
  GLenum type;
  usize length;
  const char *source;
} ShaderSouce;

typedef struct shader_program {
  GLuint gl;
} ShaderProgram;

ShaderProgram shader_init(usize n_sources, const ShaderSouce sources[n_sources]);

void shader_cleanup(ShaderProgram *shader);

void shader_use(ShaderProgram shader);

static inline void set_uniform1i(GLint location, int value) {
  glUniform1i(location, value);
}

static inline void set_uniform1f(GLint location, f32 value) {
  glUniform1f(location, value);
}

static inline void set_uniform_vec2(GLint location, vec2 value) {
  glUniform2fv(location, 1, value);
}

static inline void set_uniform_vec3(GLint location, vec3 value) {
  glUniform3fv(location, 1, value);
}

static inline void set_uniform_vec4(GLint location, vec4 value) {
  glUniform4fv(location, 1, value);
}

static inline void set_uniform_mat2(GLint location, mat2 value) {
  glUniformMatrix2fv(location, 1, GL_FALSE, (const f32 *)value);
}

static inline void set_uniform_mat3(GLint location, mat3 value) {
  glUniformMatrix3fv(location, 1, GL_FALSE, (const f32 *)value);
}

static inline void set_uniform_mat4(GLint location, mat4 value) {
  glUniformMatrix4fv(location, 1, GL_FALSE, (const f32 *)value);
}

#define shader_set_uniform(location, value)                                                                            \
  _Generic((value),                                                                                                    \
      int: set_uniform1i,                                                                                              \
      f32: set_uniform1f,                                                                                              \
      vec2: set_uniform_vec2,                                                                                          \
      vec3: set_uniform_vec3,                                                                                          \
      vec4: set_uniform_vec4,                                                                                          \
      mat2: set_uniform_mat2,                                                                                          \
      mat3: set_uniform_mat3,                                                                                          \
      mat4: set_uniform_mat4)(location, value)

#define SET_UNIFORM_FUNC(LOCATION_NAME, TY)                                                                            \
  static inline void set_uniform_##LOCATION_NAME(ShaderProgram shader, TY thing) {                                     \
    set_uniform_##TY(glGetUniformLocation(shader.gl, #LOCATION_NAME), thing);                                          \
  }
