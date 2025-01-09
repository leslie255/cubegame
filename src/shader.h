#pragma once

#include "common.h"

#include <glad/glad.h>

static char shader_info_log[512] = {};

typedef struct shader_program {
  GLuint gl_handle;
} ShaderProgram;

ShaderProgram
shader_init(usize vs_src_len, const char vs_src[vs_src_len], usize fs_src_len, const char fs_src[fs_src_len]);

void shader_cleanup(ShaderProgram *shader);

void shader_use(ShaderProgram shader);
