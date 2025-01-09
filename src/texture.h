#pragma once

#include <glad/glad.h>

#include "common.h"

typedef struct texture {
  GLuint gl;
  u32 width;
  u32 height;
  u32 n_channels;
} Texture;

Texture texture_load_from_file(char path[], bool log);

void texture_cleanup(Texture *texture);


