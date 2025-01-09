#include "game.h"
#include "debug_utils.h"
#include "text.h"

constexpr f32 CAMERA_INIT_PITCH = 0.f;
constexpr f32 CAMERA_INIT_YAW = -90.f;

#define STRING_LITERAL_ARG(S) sizeof(S) - 1, S

#define CHECK_OPENGL_ERROR()                                                                                           \
  ({                                                                                                                   \
    GLenum err = glGetError();                                                                                         \
    if (err != GL_NO_ERROR)                                                                                            \
      DBG_PRINTF("OpenGL error: %d\n", err);                                                                           \
  })

/// Helper function used in `cube_painter_new`.
static GLuint cp_gen_ebo(u32 indices[6]) {
  GLuint ebo;
  glGenBuffers(1, &ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(u32[6]), indices, GL_STATIC_DRAW);
  return ebo;
}

/// Helper function used in `cube_painter_new`.
static GLuint cp_gen_vbo(f32 vertices[20]) {
  GLuint vbo;
  // VBO.
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(f32[20]), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer(
      /* location    */ 0,
      /* size (vec3) */ 3,
      /* type        */ GL_FLOAT,
      /* normalized? */ GL_FALSE,
      /* stride      */ sizeof(f32[5]),
      /* offset      */ nullptr);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(
      /* location    */ 1,
      /* size (vec3) */ 2,
      /* type        */ GL_FLOAT,
      /* normalized? */ GL_FALSE,
      /* stride      */ sizeof(f32[5]),
      /* offset      */ (void *)sizeof(f32[3]));
  glEnableVertexAttribArray(1);
  return vbo;
}

CubePainter cube_painter_new() {
  constexpr char VERTEX_SHADER[] = //
      "#version 330 core\n"
      "layout (location = 0) in vec3 the_pos;\n"
      "layout (location = 1) in vec2 the_tex_coord;\n"
      "out vec2 tex_coord;\n"
      "uniform mat4 model;\n"
      "uniform mat4 view;\n"
      "uniform mat4 proj;\n"
      "void main() {\n"
      "  gl_Position = proj * view * model * vec4(the_pos, 1.f);\n"
      "  tex_coord = the_tex_coord;\n"
      "}\n";

  constexpr char FRAGMENT_SHADER[] = //
      "#version 330 core\n"
      "in vec2 tex_coord;\n"
      "out vec4 frag_color;\n"
      "uniform sampler2D the_texture;\n"
      "void main() {\n"
      "  frag_color = texture(the_texture, tex_coord);\n"
      "}\n";

  CubePainter cp = {};
  cp.shader = shader_init(STRING_LITERAL_ARG(VERTEX_SHADER), STRING_LITERAL_ARG(FRAGMENT_SHADER));

  // VAO.
  glGenVertexArrays(1, &cp.vao);
  glBindVertexArray(cp.vao);

  // EBO.
  cp.ebo = cp_gen_ebo((u32[]){
      2, 1, 0, //
      1, 2, 3, //
  });
  cp.ebo_reversed = cp_gen_ebo((u32[]){
      0, 1, 2, //
      3, 2, 1, //
  });

  // VBO.
  cp.vbo_north_south = cp_gen_vbo((f32[]){
      //                 Coords              Texture
      /* Top left     */ 0.f, 1.f, 0.f, /**/ 0.f, 0.f,
      /* Top right    */ 1.f, 1.f, 0.f, /**/ 1.f, 0.f,
      /* Bottom left  */ 0.f, 0.f, 0.f, /**/ 0.f, 1.f,
      /* Bottom right */ 1.f, 0.f, 0.f, /**/ 1.f, 1.f,
  });

  return cp;
}

void cube_painter_cleanup(CubePainter *cp) {
  glDeleteVertexArrays(1, &cp->vao);
  glDeleteBuffers(1, &cp->vbo_north_south);
  glDeleteBuffers(1, &cp->ebo);
}

void paint_cube(CubePainter *cp, GameState *game, CubeFace faces, Texture texture) {
  MARK_USED(faces);

  mat4 model_mat = {};
  mat4 view_mat = {};
  camera_view_mat(game->camera, view_mat);
  mat4 proj_mat = {};
  camera_proj_mat(game->camera, game->frame_width / game->frame_height, proj_mat);

  shader_use(cp->shader);

  auto uniform_model = glGetUniformLocation(cp->shader.gl_handle, "model");
  auto uniform_view = glGetUniformLocation(cp->shader.gl_handle, "view");
  auto uniform_proj = glGetUniformLocation(cp->shader.gl_handle, "proj");
  glUniformMatrix4fv(uniform_view, 1, false, (f32 *)view_mat);
  glUniformMatrix4fv(uniform_proj, 1, false, (f32 *)proj_mat);
  glBindTexture(GL_TEXTURE_2D, texture.gl);
  glActiveTexture(GL_TEXTURE0);
  glUniform1i(glGetUniformLocation(cp->shader.gl_handle, "the_texture"), 0);

  if (faces & CubeFace_North) {
    glm_mat4_identity(model_mat);
    glUniformMatrix4fv(uniform_model, 1, false, (f32 *)model_mat);
    glBindVertexArray(cp->vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cp->ebo_reversed);
    glBindBuffer(GL_ARRAY_BUFFER, cp->vbo_north_south);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
  }

  if (faces & CubeFace_South) {
    glm_mat4_identity(model_mat);
    glm_translated(model_mat, (vec4){0.f, 0.f, 1.f, 0.f});
    glUniformMatrix4fv(uniform_model, 1, false, (f32 *)model_mat);
    glBindVertexArray(cp->vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cp->ebo);
    glBindBuffer(GL_ARRAY_BUFFER, cp->vbo_north_south);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
  }
}

GameState *game_init() {
  auto game = xalloc(GameState, 1);
  game->is_paused = false;
  game->camera_pitch = CAMERA_INIT_PITCH;
  game->camera_yaw = CAMERA_INIT_YAW;
  game->cursor_has_moved_before = false;
  game->previous_cursor_x = 0.f;
  game->previous_cursor_y = 0.f;
  game->is_wireframe_mode = false;
  game->disable_gl_face_culling = false;

  game->font = default_font();
  game->text_painter = text_painter_new(game->font);

  game->camera = (Camera){
      .position = {0.f, 0.f, 5.f},
      .direction = {0.f, 0.f, -1.f},
      .up = {0.f, 1.f, 0.f},
      .fov = glm_rad(90.f),
      .near_plane_dist = 0.1f,
      .far_plane_dist = 100.f,
  };
  glm_normalize(game->camera.up);
  glm_normalize(game->camera.direction);

  game->test_texture = texture_load_from_file("res/texture/test_texture.png", true);
  game->cube_painter = cube_painter_new();

  game->fps = NAN;
  game->overlap_text = EMPTY_STRING;

  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  return game;
}

void game_cleanup(GameState **game_) {
  auto game = *game_;
  text_painter_cleanup(&game->text_painter);
  font_cleanup(&game->font);
  xfree(game);
  if (IS_DEBUG_MODE)
    *game_ = nullptr;
}

void game_cursor_callback(void *game_, Window *window) {
  GameState *game = game_;

  if (!game->is_paused) {
    f32 sensitivity = 0.1f;
    f32 dx = (f32)(window->cursor_x - game->previous_cursor_x);
    f32 dy = (f32)(window->cursor_y - game->previous_cursor_y);
    game->previous_cursor_x = window->cursor_x;
    game->previous_cursor_y = window->cursor_y;
    if (!game->cursor_has_moved_before) {
      game->previous_cursor_x = window->cursor_x;
      game->previous_cursor_y = window->cursor_y;
      game->cursor_has_moved_before = true;
      return;
    }
    game->camera_pitch -= dy * sensitivity;
    game->camera_yaw += dx * sensitivity;
    if (game->camera_pitch > 89.9f)
      game->camera_pitch = 89.9f;
    if (game->camera_pitch < -89.9f)
      game->camera_pitch = -89.9f;
    camera_set_direction(&game->camera, game->camera_yaw, game->camera_pitch);
  }
}

void game_key_callback(void *game_, Window *window, int key, int scancode, int action, int mods) {
  MARK_USED(scancode);
  MARK_USED(mods);
  GameState *game = game_;
  if (glfwGetKey(window->glfw_handle, GLFW_KEY_F3) == GLFW_PRESS) {
    if (key == GLFW_KEY_L && action == GLFW_PRESS)
      game->is_wireframe_mode = !game->is_wireframe_mode;
    else if (key == GLFW_KEY_F && action == GLFW_PRESS)
      game->disable_gl_face_culling = !game->disable_gl_face_culling;

    glPolygonMode(GL_FRONT_AND_BACK, game->is_wireframe_mode ? GL_LINE : GL_FILL);
    if (game->is_wireframe_mode || game->disable_gl_face_culling)
      glDisable(GL_CULL_FACE);
    else
      glEnable(GL_CULL_FACE);

    return;
  }
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    game->is_paused = !game->is_paused;
    if (game->is_paused) {
      window_restore_cursor(window);
    } else {
      window_disable_cursor(window);
      game->cursor_has_moved_before = false;
    }
  }
}

void game_update_events(GameState *game, Window *window) {
  if (!game->is_paused) {
    vec3 camera_movement = {};
    if (glfwGetKey(window->glfw_handle, GLFW_KEY_W) == GLFW_PRESS)
      camera_movement[2] += 0.05f;
    if (glfwGetKey(window->glfw_handle, GLFW_KEY_S) == GLFW_PRESS)
      camera_movement[2] -= 0.05f;
    if (glfwGetKey(window->glfw_handle, GLFW_KEY_A) == GLFW_PRESS)
      camera_movement[0] -= 0.05f;
    if (glfwGetKey(window->glfw_handle, GLFW_KEY_D) == GLFW_PRESS)
      camera_movement[0] += 0.05f;
    if (glfwGetKey(window->glfw_handle, GLFW_KEY_R) == GLFW_PRESS)
      camera_movement[1] -= 0.05f;
    if (glfwGetKey(window->glfw_handle, GLFW_KEY_SPACE) == GLFW_PRESS)
      camera_movement[1] += 0.05f;
    if (glfwGetKey(window->glfw_handle, GLFW_KEY_C) == GLFW_PRESS)
      game->camera.fov = glm_rad(30.f);
    else
      game->camera.fov = glm_rad(90.f);
    if (camera_movement[0] != 0 || camera_movement[1] != 0 || camera_movement[2] != 0)
      camera_move(&game->camera, camera_movement);
  }
}

static inline f32 font_aspect_ratio(const GameState *game) {
  return (f32)game->font->glyph_width / (f32)game->font->glyph_height;
}

static inline u8 hex_digit(char digit) {
  switch (digit) {
  case '0' ... '9':
    return (u8)(digit - '0');
  case 'A' ... 'F':
    return (u8)(digit - 'A') + 10;
  case 'a' ... 'f':
    return (u8)(digit - 'a') + 10;
  default:
    return 0;
  }
}

static inline void print(GameState *game, vec2 pos, usize length, char s[length]) {
  if (length == 0)
    length = strlen(s);
  usize x_counter = 0;
  usize y_counter = 0;
  for (usize i = 0; i < length; ++i) {
    switch (s[i]) {
    case '\n': {
      ++y_counter;
      x_counter = 0;
    } break;
    case '\r': {
      x_counter = 0;
    } break;
    case '\a': {
      if (length - i < 7) {
        DBG_PRINTF("String is cut short after '\\a'\n");
        return;
      }
      u8 escape_kind = (u8)s[i + 1];
      u8 r = 0;
      r += hex_digit(s[i + 2]) * 16;
      r += hex_digit(s[i + 3]);
      u8 g = 0;
      g += hex_digit(s[i + 4]) * 16;
      g += hex_digit(s[i + 5]);
      u8 b = 0;
      b += hex_digit(s[i + 6]) * 16;
      b += hex_digit(s[i + 7]);
      i += 7;
      vec4 color = {(f32)r / 255.f, (f32)g / 255.f, (f32)b / 255.f};
      if (escape_kind == 0x01) {
        text_painter_set_fg_color(&game->text_painter, color);
      } else if (escape_kind == 0x02) {
        text_painter_set_bg_color(&game->text_painter, color);
      } else {
        DBG_PRINTF("Malformed escape code\n");
        return;
      }
    } break;
    default: {
      vec2 pos_ = {
          pos[0] + (f32)x_counter * DEFAULT_FONT_SIZE * font_aspect_ratio(game),
          pos[1] - (f32)y_counter * DEFAULT_FONT_SIZE,
      };
      text_paint(game->text_painter, game->frame_width, game->frame_height, pos_, s[i]);
      ++x_counter;
    } break;
    }
  }
}

static inline void draw_overlap_text(GameState *game) {
  if (game->is_paused) {
    string_append(&game->overlap_text,
                  STRING_LITERAL_ARG(
                      "\a\001000000\a\002E0E0E0[\a\001FF8000ESC\a\001000000] Game Paused\a\001FFFFFF\a\002303030\n"));
  } else {
    string_append(&game->overlap_text, STRING_LITERAL_ARG("\a\001FFFFFF\a\002303030Cube Game v0.0.0\n"));
  }

  if (isnan(game->fps))
    string_snprintf(&game->overlap_text, 64, "FPS ---.--\n");
  else
    string_snprintf(&game->overlap_text, 64, "FPS: %.2lf\n", game->fps);

  string_snprintf(&game->overlap_text, 64, "Camera XYZ: %f, %f, %f", game->camera.position[0], game->camera.position[1],
                  game->camera.position[2]);

  if (game->is_wireframe_mode) {
    string_append(&game->overlap_text,
                  STRING_LITERAL_ARG("\n\a\001FFFFFF\a\002303030[\a\001FF8000F3+L\a\001FFFFFF] DEBUG: Wireframe Mode"));
  }

  if (game->disable_gl_face_culling) {
    string_append(&game->overlap_text,
                  STRING_LITERAL_ARG(
                      "\n\a\001FFFFFF\a\002303030[\a\001FF8000F3+F\a\001FFFFFF] DEBUG: Disable OpenGL Face Culling"));
  }

  vec2 pos = {
      10.f,
      game->frame_height - DEFAULT_FONT_SIZE - 10.f,
  };

  glDisable(GL_DEPTH_TEST);
  if (!game->disable_gl_face_culling)
    glDisable(GL_CULL_FACE);
  if (game->is_wireframe_mode)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  print(game, pos, game->overlap_text.length, game->overlap_text.buffer);

  glEnable(GL_DEPTH_TEST);
  if (game->is_wireframe_mode)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  if (!game->disable_gl_face_culling)
    glEnable(GL_CULL_FACE);
}

void game_frame(GameState *game, f32 frame_width, f32 frame_height) {
  game->frame_width = frame_width;
  game->frame_height = frame_height;
  glClearColor(.1f, .1f, .1f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  paint_cube(&game->cube_painter, game, CubeFace_North | CubeFace_South, game->test_texture);
  string_clear(&game->overlap_text);
  draw_overlap_text(game);
  CHECK_OPENGL_ERROR();
}
