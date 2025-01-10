#include "text.h"

#include <stb/stb_image.h>

#include "atlas.h"

FontData *default_font() {
  FontData *font = xalloc(FontData, 1);
  memset(font, 0, sizeof(FontData));
  font->texture = texture_load_from_file("res/font/bigblueterminal.png", true);
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
  texture_cleanup(&(*font)->texture);
  xfree(*font);
  if (IS_DEBUG_MODE)
    *font = nullptr;
}

bool font_has_char(const FontData *font, char ch) {
  return font->present_map[(usize)ch];
}

void font_sample(const FontData *font, char ch, mat3 dest) {
  DEBUG_ASSERT(font->n_glyphs_per_line != 0);
  u32 i = (u32)ch - (u32)font->range_start;
  // Coord of the glyph.
  u32 x = i % font->n_glyphs_per_line;
  u32 y = i / font->n_glyphs_per_line;
  // Pixels that the glyph is on.
  u32 x_min = x * font->glyph_width;
  u32 y_min = y * font->glyph_height;
  u32 x_max = x_min + font->glyph_width;
  u32 y_max = y_min + font->glyph_height;
  atlas_mat(font->texture.width, font->texture.height, x_min, y_min, x_max, y_max, dest);
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
      "  if (sample.a < 0.1f && bg_color.a < 0.1f)\n"
      "    discard;\n"
      "  frag_color = (sample.a * fg_color) + ((1.f - sample.a) * bg_color);\n"
      "}\n";

  tp->shader = shader_init(ARR_ARG(((ShaderSouce[]){
      {GL_VERTEX_SHADER, STRING_LITERAL_ARG(VERTEX_SHADER)},
      {GL_FRAGMENT_SHADER, STRING_LITERAL_ARG(FRAGMENT_SHADER)},
  })));

  ASSERT(glGetUniformLocation(tp->shader.gl, "model_proj") != -1);
  ASSERT(glGetUniformLocation(tp->shader.gl, "tex_trans") != -1);
  ASSERT(glGetUniformLocation(tp->shader.gl, "fg_color") != -1);
  ASSERT(glGetUniformLocation(tp->shader.gl, "bg_color") != -1);
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
      2, 1, 0, //
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

void text_painter_set_fg_color(TextPainter *tp, const vec4 new_fg_color) {
  memcpy(tp->fg_color, new_fg_color, sizeof(tp->fg_color));
}

void text_painter_set_bg_color(TextPainter *tp, const vec4 new_bg_color) {
  memcpy(tp->bg_color, new_bg_color, sizeof(tp->bg_color));
}

/// Paint one character.
void text_paint(TextPainter tp, f32 frame_width, f32 frame_height, vec2 coord, char ch) {
  auto model_proj = glGetUniformLocation(tp.shader.gl, "model_proj");
  auto tex_trans = glGetUniformLocation(tp.shader.gl, "tex_trans");
  auto fg_color = glGetUniformLocation(tp.shader.gl, "fg_color");
  auto bg_color = glGetUniformLocation(tp.shader.gl, "bg_color");
  auto the_texture = glGetUniformLocation(tp.shader.gl, "the_texture");

  shader_use(tp.shader);

  glUniform4f(fg_color, tp.fg_color[0], tp.fg_color[1], tp.fg_color[2], tp.fg_color[3]);
  glUniform4f(bg_color, tp.bg_color[0], tp.bg_color[1], tp.bg_color[2], tp.bg_color[3]);

  f32 text_size = DEFAULT_FONT_SIZE;
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

  mat3 tex_trans_mat;
  if (font_has_char(tp.font, ch))
    font_sample(tp.font, ch, tex_trans_mat);
  else
    glm_mat3_identity(tex_trans_mat);
  glUniformMatrix3fv(tex_trans, 1, false, (f32 *)tex_trans_mat);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, tp.font->texture.gl);
  glUniform1i(the_texture, 1);

  glBindVertexArray(tp.vao);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}
