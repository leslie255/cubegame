#include "game.h"
#include "debug_utils.h"
#include "text.h"

constexpr f32 CAMERA_INIT_PITCH = 0.f;
constexpr f32 CAMERA_INIT_YAW = -90.f;

#define CHECK_OPENGL_ERROR()                                                                                           \
  ({                                                                                                                   \
    GLenum err = glGetError();                                                                                         \
    if (err != GL_NO_ERROR)                                                                                            \
      DBG_PRINTF("OpenGL error: %d\n", err);                                                                           \
  })

static inline void setup_the_3d_square(GameState *game) {
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

  constexpr f32 vertices[] = {
      //                 Coords              Texture
      /* Top left     */ 0.f, 1.f, 0.f, /**/ 0.f, 0.f,
      /* Top right    */ 1.f, 1.f, 0.f, /**/ 1.f, 0.f,
      /* Bottom left  */ 0.f, 0.f, 0.f, /**/ 0.f, 1.f,
      /* Bottom right */ 1.f, 0.f, 0.f, /**/ 1.f, 1.f,
  };

  constexpr u32 indices[] = {
      0, 1, 2, //
      1, 2, 3, //
  };

  // VAO.
  glGenVertexArrays(1, &game->vao1);
  glBindVertexArray(game->vao1);
  // EBO.
  glGenBuffers(1, &game->ebo1);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, game->ebo1);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
  // VBO.
  glGenBuffers(1, &game->vbo1);
  glBindBuffer(GL_ARRAY_BUFFER, game->vbo1);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
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

  // Texture.
  i32 texture_width;
  i32 texture_height;
  i32 texture_n_channels;
  constexpr char test_texture_path[] = "res/texture/test_texture.png";
  auto texture_data = stbi_load(test_texture_path, &texture_width, &texture_height, &texture_n_channels, 0);
  ASSERT(texture_data != nullptr);
  printf("Loaded texture, dimension: %dx%d, channels: %d\n", texture_width, texture_height, texture_n_channels);
  glGenTextures(1, &game->texture1);
  glBindTexture(GL_TEXTURE_2D, game->texture1);
  GLenum format;
  switch (texture_n_channels) {
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
  glTexImage2D(GL_TEXTURE_2D, 0, (GLint)format, texture_width, texture_height, 0, format, GL_UNSIGNED_BYTE,
               texture_data);
  stbi_image_free(texture_data);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // Shader.
  game->shader1 = shader_init(sizeof(VERTEX_SHADER), VERTEX_SHADER, sizeof(FRAGMENT_SHADER), FRAGMENT_SHADER);
  // Verify uniforms are ok.
  ASSERT(glGetUniformLocation(game->shader1.gl_handle, "model") != -1);
  ASSERT(glGetUniformLocation(game->shader1.gl_handle, "view") != -1);
  ASSERT(glGetUniformLocation(game->shader1.gl_handle, "proj") != -1);
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

  game->font = default_font();
  game->text_painter = text_painter_new(game->font);

  setup_the_3d_square(game);

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

  return game;
}

void game_cleanup(GameState **game_) {
  auto game = *game_;
  glDeleteVertexArrays(1, &game->vao1);
  glDeleteBuffers(1, &game->vbo1);
  glDeleteBuffers(1, &game->ebo1);
  glDeleteTextures(1, &game->texture1);
  shader_cleanup(&game->shader1);
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
    if (key == GLFW_KEY_L && action == GLFW_PRESS) {
      game->is_wireframe_mode = !game->is_wireframe_mode;
      glPolygonMode(GL_FRONT_AND_BACK, game->is_wireframe_mode ? GL_LINE : GL_FILL);
    }
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

static inline void draw_the_3d_square(GameState *game, f32 frame_width, f32 frame_height) {
  mat4 view_mat = {};
  camera_view_mat(game->camera, view_mat);
  mat4 proj_mat = {};
  camera_proj_mat(game->camera, frame_width / frame_height, proj_mat);

  shader_use(game->shader1);

  auto uniform_model = glGetUniformLocation(game->shader1.gl_handle, "model");
  auto uniform_view = glGetUniformLocation(game->shader1.gl_handle, "view");
  auto uniform_proj = glGetUniformLocation(game->shader1.gl_handle, "proj");
  mat4 model_mat = GLM_MAT4_IDENTITY;
  glUniformMatrix4fv(uniform_model, 1, false, (f32 *)model_mat);
  glUniformMatrix4fv(uniform_view, 1, false, (f32 *)view_mat);
  glUniformMatrix4fv(uniform_proj, 1, false, (f32 *)proj_mat);
  glBindTexture(GL_TEXTURE_2D, game->texture1);
  glActiveTexture(GL_TEXTURE0);
  glUniform1i(glGetUniformLocation(game->shader1.gl_handle, "the_texture"), 0);
  glBindVertexArray(game->vao1);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

static inline f32 font_aspect_ratio(const GameState *game) {
  return (f32)game->font->glyph_width / (f32)game->font->glyph_height;
}

/// Draw a line of text.
/// `length = 0` for C strings.
static inline void
draw_text_line(GameState *game, vec2 pos, f32 frame_width, f32 frame_height, usize length, char s[length]) {
  if (length == 0) {
    for (usize i = 0; s[i] != '\0'; ++i) {
      vec2 pos_ = {pos[0] + (f32)i * DEFAULT_FONT_SIZE * font_aspect_ratio(game), pos[1]};
      text_paint(game->text_painter, frame_width, frame_height, pos_, s[i]);
    }
  } else {
    for (usize i = 0; i < length; ++i) {
      vec2 pos_ = {pos[0] + (f32)i * DEFAULT_FONT_SIZE / font_aspect_ratio(game), pos[1]};
      text_paint(game->text_painter, frame_width, frame_height, pos_, s[i]);
    }
  }
}

void game_frame(GameState *game, f32 frame_width, f32 frame_height) {
  glClearColor(.1f, .1f, .1f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  draw_the_3d_square(game, frame_width, frame_height);
  glDisable(GL_DEPTH_TEST);
  // Testing missing characters:
  // text_painter_set_fg_color(&game->text_painter, (vec4){1.f, 1.f, 1.f, 1.f});
  // text_painter_set_bg_color(&game->text_painter, (vec4){0.f, 0.f, 0.f, 1.f});
  // draw_text_line(game, (vec2){10.f, 10.f}, frame_width, frame_height, 0, "测试");
  text_painter_set_bg_color(&game->text_painter, (vec4){1.f, 1.f, 1.f, 1.f});
  text_painter_set_fg_color(&game->text_painter, (vec4){0.f, 0.f, 0.f, 1.f});
  if (game->is_paused)
    draw_text_line(game, (vec2){10.f, frame_height - DEFAULT_FONT_SIZE - 10.f}, frame_width, frame_height, 0,
                   "Game Paused [ESC]");
  CHECK_OPENGL_ERROR();
}
