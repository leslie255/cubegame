#include "text.h"

#include "debug_utils.h"

#include <stb/stb_image.h>

FontData *init_pix_chicago_font() {
  static constexpr char path[] = "res/font/pix_chicago.png";
  FontData *font = xalloc(FontData, 1);
  memset(font, 0, sizeof(FontData));
  auto data = stbi_load(path, (i32 *)&font->image_width, (i32 *)&font->image_height, (i32 *)&font->image_n_channels, 0);
  ASSERT(data != nullptr);
  printf("Loaded texture `%s`, dimension: %dx%d, channels: %d\n", path, font->image_width, font->image_height,
         font->image_n_channels);
  GLenum format;
  switch (font->image_n_channels) {
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
  default: {
    format = GL_RGBA;
  } break;
  }
  glGenTextures(1, &font->gl_texture);
  glBindTexture(GL_TEXTURE_2D, font->gl_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, (GLint)format, (GLint)font->image_width, (GLint)font->image_height, 0, format,
               GL_UNSIGNED_BYTE, data);
  stbi_image_free(data);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
  glGenerateMipmap(GL_TEXTURE_2D);
  font->glyph_width = 10;
  font->glyph_height = 15;
  font->n_glyphs_per_line = 12;
  font->n_glyphs = 96;
  font->range_start = ' ';
  font->range_end = '~';
  memset(&font->present_map[' '], true, '~' - ' ');
  return font;
}

void font_cleanup(FontData **font) {
  stbi_image_free((*font)->texture_data);
  xfree(*font);
  if (IS_DEBUG_MODE)
    *font = nullptr;
}

bool font_has_char(const FontData *font, char ch) {
  return font->present_map[(usize)ch];
}

static inline void normalize_tex_coord(u32 total_width, u32 total_height, u32 x, u32 y, vec2 dest) {
  dest[0] = (f32)x / (f32)total_width;
  dest[1] = (f32)y / (f32)total_height;
}

void font_glyph_coord(const FontData *font, char ch, vec2 start, vec2 end) {
  DEBUG_ASSERT(font->n_glyphs_per_line != 0);
  u32 i = (u32)ch - (u32)font->range_start;
  DBG_PRINT(i);
  // Coord of the glyph.
  u32 x_glyph = i % font->n_glyphs_per_line;
  u32 y_glyph = i / font->n_glyphs_per_line;
  DBG_PRINT(x_glyph);
  DBG_PRINT(y_glyph);
  // Pixels that the glyph is on.
  u32 x_start = x_glyph * font->glyph_width;
  u32 y_start = y_glyph * font->glyph_height;
  u32 x_end = x_start + font->glyph_width;
  u32 y_end = y_start + font->glyph_height;
  DBG_PRINT(x_start);
  DBG_PRINT(y_start);
  DBG_PRINT(x_end);
  DBG_PRINT(y_end);
  // Normalize them.
  normalize_tex_coord(font->image_width, font->image_height, x_start, y_start, start);
  normalize_tex_coord(font->image_width, font->image_height, x_end, y_end, end);
}
