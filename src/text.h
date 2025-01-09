#pragma once

#include <glad/glad.h>
#include <cglm/cglm.h>

#include "common.h"
#include "string.h"

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

/// Initializes the pixChicago font.
/// TODO: Load fonts from font descriptor file.
FontData *init_pix_chicago_font();

void font_cleanup(FontData **font);

/// Whether a character is present (true) or missing (false) in a font.
bool font_has_char(const FontData *font, char ch);

/// Position of a glyph in a font atlas, in normalized form.
/// Behavior is unstable (but not undefined compiler-wise) for characters that is not present in the font.
/// To check if a character is present or not, use `font_has_char`.
void font_glyph_coord(const FontData *font, char ch, vec2 start, vec2 end);

/// A line of text, and related OpenGL objects.
typedef struct text_line {
  String text;
  GLuint vao;
  GLuint ebo;
  GLuint vbo;
} TextLine;
