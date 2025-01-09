#include "text.h"

#include "debug_utils.h"

#include <stb/stb_image.h>

#define CHECK_OPENGL_ERROR()                                                                                           \
  ({                                                                                                                   \
    GLenum err = glGetError();                                                                                         \
    if (err != GL_NO_ERROR)                                                                                            \
      DBG_PRINTF("OpenGL error: %d\n", err);                                                                           \
  })

FontData *default_font() {
  static constexpr char path[] = "res/font/bigblueterminal.png";
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
  font->glyph_width = 8;
  font->glyph_height = 12;
  font->n_glyphs_per_line = 16;
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
  // Coord of the glyph.
  u32 x_glyph = i % font->n_glyphs_per_line;
  u32 y_glyph = i / font->n_glyphs_per_line;
  // Pixels that the glyph is on.
  u32 x_start = x_glyph * font->glyph_width;
  u32 y_start = y_glyph * font->glyph_height;
  u32 x_end = x_start + font->glyph_width;
  u32 y_end = y_start + font->glyph_height;
  // Normalize them.
  normalize_tex_coord(font->image_width, font->image_height, x_start, y_start, start);
  normalize_tex_coord(font->image_width, font->image_height, x_end, y_end, end);
}

/// Helper function used in `text_painter_new`.
static inline void tp_setup_shader(TextPainter *tp) {
  constexpr char VERTEX_SHADER[] = //
      "#version 330 core\n"
      "layout (location = 0) in vec2 the_pos;\n"
      "layout (location = 1) in vec2 the_tex_coord;\n"
      "out vec2 tex_coord;\n"
      "uniform mat4 model_proj;\n"
      "uniform mat3 tex_trans;\n"
      "void main() {\n"
      "  gl_Position = model_proj * vec4(the_pos, 0.f, 1.f);\n"
      "  vec3 tex_coord_transformed = tex_trans * vec3(the_tex_coord, 1.f);\n"
      "  tex_coord = tex_coord_transformed.xy;\n"
      "}\n";

  constexpr char FRAGMENT_SHADER[] = //
      "#version 330 core\n"
      "in vec2 tex_coord;\n"
      "out vec4 frag_color;\n"
      "uniform vec4 fg_color;\n"
      "uniform vec4 bg_color;\n"
      "uniform sampler2D the_texture;\n"
      "void main() {\n"
      "  vec4 sample = texture(the_texture, tex_coord);\n"
      "  frag_color = (sample.a * fg_color) + ((1.f - sample.a) * bg_color);\n"
      "}\n";

  tp->shader = shader_init(sizeof(VERTEX_SHADER), VERTEX_SHADER, sizeof(FRAGMENT_SHADER), FRAGMENT_SHADER);
  ASSERT(glGetUniformLocation(tp->shader.gl_handle, "model_proj") != -1);
  ASSERT(glGetUniformLocation(tp->shader.gl_handle, "tex_trans") != -1);
  ASSERT(glGetUniformLocation(tp->shader.gl_handle, "fg_color") != -1);
  ASSERT(glGetUniformLocation(tp->shader.gl_handle, "bg_color") != -1);
}

/// Helper function used in `text_painter_new`.
static inline void tp_setup_mesh(TextPainter *tp) {
  const f32 vertices[] = {
      //                 Coords     Texture
      /* Top left     */ 0.f, 1.f, 0.f, 0.f,
      /* Top right    */ 1.f, 1.f, 1.f, 0.f,
      /* Bottom left  */ 0.f, 0.f, 0.f, 1.f,
      /* Bottom right */ 1.f, 0.f, 1.f, 1.f,
  };

  constexpr u32 indices[] = {
      0, 1, 2, //
      1, 2, 3, //
  };

  // VAO.
  glGenVertexArrays(1, &tp->vao);
  glBindVertexArray(tp->vao);
  // EBO.
  glGenBuffers(1, &tp->ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tp->ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
  // VBO.
  glGenBuffers(1, &tp->vbo);
  glBindBuffer(GL_ARRAY_BUFFER, tp->vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer(
      /* location    */ 0,
      /* size (vec2) */ 2,
      /* type        */ GL_FLOAT,
      /* normalized? */ GL_FALSE,
      /* stride      */ sizeof(f32[4]),
      /* offset      */ nullptr);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(
      /* location    */ 1,
      /* size (vec2) */ 2,
      /* type        */ GL_FLOAT,
      /* normalized? */ GL_FALSE,
      /* stride      */ sizeof(f32[4]),
      /* offset      */ (void *)sizeof(f32[2]));
  glEnableVertexAttribArray(1);
}

TextPainter text_painter_new(const FontData *font) {
  TextPainter tp = {.font = font};
  tp_setup_shader(&tp);
  tp_setup_mesh(&tp);
  return tp;
}

void text_painter_cleanup(TextPainter *tp) {
  glDeleteVertexArrays(1, &tp->vao);
  glDeleteBuffers(1, &tp->ebo);
  glDeleteBuffers(1, &tp->vbo);
  shader_cleanup(&tp->shader);
}

void text_painter_set_fg_color(TextPainter *tp, vec4 new_fg_color) {
  auto fg_color = glGetUniformLocation(tp->shader.gl_handle, "fg_color");
  glUniform4f(fg_color, new_fg_color[0], new_fg_color[1], new_fg_color[2], new_fg_color[3]);
}

void text_painter_set_bg_color(TextPainter *tp, vec4 new_bg_color) {
  auto bg_color = glGetUniformLocation(tp->shader.gl_handle, "bg_color");
  glUniform4f(bg_color, new_bg_color[0], new_bg_color[1], new_bg_color[2], new_bg_color[3]);
}

/// Paint one character.
void text_paint(TextPainter tp, f32 frame_width, f32 frame_height, vec2 coord, char ch) {
  auto model_proj = glGetUniformLocation(tp.shader.gl_handle, "model_proj");
  auto tex_trans = glGetUniformLocation(tp.shader.gl_handle, "tex_trans");
  auto fg_color = glGetUniformLocation(tp.shader.gl_handle, "fg_color");
  auto bg_color = glGetUniformLocation(tp.shader.gl_handle, "bg_color");
  auto the_texture = glGetUniformLocation(tp.shader.gl_handle, "the_texture");

  shader_use(tp.shader);

  glUniform4f(fg_color, 1.f, 1.f, 1.f, 1.f);
  glUniform4f(bg_color, .2f, .2f, .2f, 1.f);

  f32 text_size = 50.f;
  mat4 model_mat = GLM_MAT4_IDENTITY;
  glm_translated(model_mat, (vec4){coord[0], coord[1], 0.f, 0.f});
  glm_scale(model_mat, (vec4){text_size * (f32)tp.font->glyph_width / (f32)tp.font->glyph_height, text_size});
  mat4 proj_mat = GLM_MAT4_IDENTITY;
  glm_ortho( //
      /* left   */ 0.0f,
      /* right  */ frame_width,
      /* bottom */ 0.0f,
      /* top    */ frame_height,
      /* nearZ  */ -1.f,
      /* farZ   */ +1.f,
      /* dest   */ proj_mat);
  mat4 model_proj_mat = GLM_MAT4_IDENTITY;
  glm_mat4_mul(proj_mat, model_mat, model_proj_mat);
  glUniformMatrix4fv(model_proj, 1, false, (f32 *)model_proj_mat);

  vec2 glyph_start;
  vec2 glyph_end;
  ASSERT(font_has_char(tp.font, ch));
  font_glyph_coord(tp.font, ch, glyph_start, glyph_end);
  mat3 tex_trans_mat = GLM_MAT3_IDENTITY;
  vec2 glyph_d;
  glm_vec2_sub(glyph_end, glyph_start, glyph_d);
  glm_translate2d(tex_trans_mat, glyph_start);
  glm_scale2d(tex_trans_mat, glyph_d);
  glUniformMatrix3fv(tex_trans, 1, false, (f32 *)tex_trans_mat);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, tp.font->gl_texture);
  glUniform1i(the_texture, 1);

  glBindVertexArray(tp.vao);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}
