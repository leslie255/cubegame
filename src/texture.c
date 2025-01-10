#include "texture.h"

#include "debug_utils.h"

#include <stb/stb_image.h>

#define CHECK_OPENGL_ERROR()                                                                                           \
  ({                                                                                                                   \
    GLenum err = glGetError();                                                                                         \
    if (err != GL_NO_ERROR)                                                                                            \
      DBG_PRINTF("OpenGL error: %d\n", err);                                                                           \
  })

Texture texture_load_from_file(char path[], bool log) {
  Texture texture = {};
  auto texture_data = stbi_load(path, (i32 *)&texture.width, (i32 *)&texture.height, (i32 *)&texture.n_channels, 0);
  ASSERT(texture_data != nullptr);
  if (log)
    printf("Loaded texture `%s`, dimension: %dx%d, channels: %d\n", path, texture.width, texture.height,
           texture.n_channels);
  glGenTextures(1, &texture.gl);
  glBindTexture(GL_TEXTURE_2D, texture.gl);
  CHECK_OPENGL_ERROR();
  GLenum format;
  switch (texture.n_channels) {
  case 1: {
    format = GL_RED;
  } break;
  case 2: {
    format = GL_RG;
  } break;
  case 3: {
    format = GL_RGB;
  } break;
  case 4: {
    format = GL_RGBA;
  } break;
  }
  glTexImage2D(GL_TEXTURE_2D, 0, (GLint)format, (GLint)texture.width, (GLint)texture.height, 0, format,
               GL_UNSIGNED_BYTE, texture_data);
  stbi_image_free(texture_data);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  return texture;
}

void texture_cleanup(Texture *texture) {
  glDeleteTextures(1, &texture->gl);
}
