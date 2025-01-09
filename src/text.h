#pragma once

#include <glad/glad.h>
#include <cglm/cglm.h>

#include "common.h"
#include "shader.h"

constexpr f32 DEFAULT_FONT_SIZE = 24.f;

/// A font in texture atlas form.
typedef struct FontData {
  /// The OpenGL texture.
  GLuint gl_texture;
  /// Data of the image.
  u8 *texture_data;
  /// Width of the entire image.
  u32 image_width;
  /// Height of the entire image.
  u32 image_height;
  /// Number of channels of the image.
  u32 image_n_channels;
  /// Width of one single glyph.
  u32 glyph_width;
  /// Height of one single glyph.
  u32 glyph_height;
  /// Number of glyths per line.
  u32 n_glyphs_per_line;
  /// Total number of glyphs.
  u32 n_glyphs;
  /// Start of range of availible characters.
  char range_start;
  /// End of range of availible characters.
  char range_end;
  /// Map for whether a character is present (true) or missing (false).
  bool present_map[256];
} FontData;

/// Initializes the default font.
/// TODO: Load fonts from font descriptor file.
FontData *default_font();

void font_cleanup(FontData **font);

/// Whether a character is present (true) or missing (false) in a font.
bool font_has_char(const FontData *font, char ch);

/// Computes a matrix that crops the texture to just the glyph.
void font_sample(const FontData *font, char ch, mat3 dest);

typedef struct text_painter {
  const FontData *font;
  ShaderProgram shader;
  GLuint vao;
  GLuint ebo;
  GLuint vbo;
  vec4 bg_color;
  vec4 fg_color;
} TextPainter;

TextPainter text_painter_new(const FontData *font);

void text_painter_cleanup(TextPainter *tp);

void text_painter_set_fg_color(TextPainter *tp, const vec4 new_fg_color);

void text_painter_set_bg_color(TextPainter *tp, const vec4 new_bg_color);

/// Paint one character.
void text_paint(TextPainter tp, f32 frame_width, f32 frame_height, vec2 coord, char ch);
