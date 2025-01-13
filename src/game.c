#include "game.h"
#include "debug_utils.h"
#include "text.h"
#include "mesh.h"

constexpr f32 CAMERA_INIT_PITCH = 0.f;
constexpr f32 CAMERA_INIT_YAW = -90.f;

constexpr char VERTEX_SHADER[] = //
    "#version 330 core\n"
    "layout (location = 0) in vec3 the_pos;\n"
    "layout (location = 1) in vec2 in_tex_coord;\n"
    "out vec2 vert_tex_coord;\n"
    "uniform mat4 model;\n"
    "uniform mat4 view;\n"
    "uniform mat4 proj;\n"
    "void main() {\n"
    "  gl_Position = proj * view * model * vec4(the_pos, 1.f);\n"
    "  vert_tex_coord = vec3(in_tex_coord, 1.f).xy;\n"
    "}\n";

constexpr char FRAGMENT_SHADER[] = //
    "#version 330 core\n"
    "in vec2 vert_tex_coord;\n"
    "out vec4 frag_color;\n"
    "uniform sampler2D the_texture;\n"
    "void main() {\n"
    "  frag_color = texture(the_texture, vert_tex_coord);\n"
    "}\n";

#define CHECK_OPENGL_ERROR()                                                                                           \
  ({                                                                                                                   \
    GLenum err = glGetError();                                                                                         \
    if (err != GL_NO_ERROR)                                                                                            \
      DBG_PRINTF("OpenGL error: %d\n", err);                                                                           \
  })

SET_UNIFORM_FUNC(model, mat4);
SET_UNIFORM_FUNC(view, mat4);
SET_UNIFORM_FUNC(proj, mat4);

GameState *game_init() {
  auto game = xalloc(GameState, 1);

  game->texture_atlas = texture_load_from_file("res/texture/texture_atlas.png", true);

  game->font = default_font();
  game->overlay_text = EMPTY_STRING;
  game->text_painter = text_painter_new(game->font);

  game->shader = shader_init(ARR_ARG(((ShaderSouce[]){
      {GL_VERTEX_SHADER, STRING_LITERAL_ARG(VERTEX_SHADER)},
      {GL_FRAGMENT_SHADER, STRING_LITERAL_ARG(FRAGMENT_SHADER)},
  })));

  game->world = world_alloc();
  memset(game->chunk_meshes, 0, sizeof(game->chunk_meshes));
  game->chunk_builder = chunk_builder_new(game->texture_atlas);
  world_gen(game->world);
  for (i32 y = -(WORLD_SIZE_Y / 2); y < (WORLD_SIZE_Y / 2); ++y) {
    for (i32 z = -(WORLD_SIZE_Z / 2); z < (WORLD_SIZE_Z / 2); ++z) {
      for (i32 x = -(WORLD_SIZE_X / 2); x < (WORLD_SIZE_X / 2); ++x) {
        ivec3 chunk_id = {x, y, z};
        auto chunk_data = *world_get_chunk(game->world, chunk_id);
        auto chunk_mesh = &game->chunk_meshes[y + WORLD_SIZE_Y / 2][z + WORLD_SIZE_Z / 2][x + WORLD_SIZE_X / 2];
        *chunk_mesh = mesh_init_empty();
        build_chunk(&game->chunk_builder, chunk_mesh, chunk_data, game->texture_atlas);
      }
    }
  }

  game->camera_pitch = CAMERA_INIT_PITCH;
  game->camera_yaw = CAMERA_INIT_YAW;
  game->camera = (Camera){
      .position = {0.f, 3.f, 0.f},
      .direction = {0.f, 0.f, -1.f},
      .up = {0.f, 1.f, 0.f},
      .fov = glm_rad(90.f),
      .near_plane_dist = 0.1f,
      .far_plane_dist = 500.f,
  };
  glm_normalize(game->camera.up);
  glm_normalize(game->camera.direction);

  game->is_paused = false;
  game->cursor_has_moved_before = false;
  game->previous_cursor_x = 0.f;
  game->previous_cursor_y = 0.f;
  game->is_wireframe_mode = false;
  game->disable_gl_face_culling = false;

  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  return game;
}

void game_cleanup(GameState **game_) {
  auto game = *game_;
  texture_cleanup(&game->texture_atlas);
  font_cleanup(&game->font);
  string_cleanup(&game->overlay_text);
  text_painter_cleanup(&game->text_painter);
  shader_cleanup(&game->shader);
  world_cleanup(&game->world);
  chunk_builder_cleanup(&game->chunk_builder);
  for (usize y = 0; y < WORLD_SIZE_Y; ++y)
    for (usize z = 0; z < WORLD_SIZE_Z; ++z)
      for (usize x = 0; x < WORLD_SIZE_X; ++x)
        mesh_cleanup(&game->chunk_meshes[y][z][x]);

  xfree(game);
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

void game_update_events(GameState *game, Window *window, f64 frame_time) {
  if (game->is_paused)
    return;

  // Movement.
  vec3 camera_movement = {};
  auto control_down = false;
  control_down |= glfwGetKey(window->glfw_handle, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS;
  control_down |= glfwGetKey(window->glfw_handle, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
  if (glfwGetKey(window->glfw_handle, GLFW_KEY_W) == GLFW_PRESS)
    camera_movement[2] += (control_down ? 15.f : 5.f) * (f32)frame_time;
  if (glfwGetKey(window->glfw_handle, GLFW_KEY_S) == GLFW_PRESS)
    camera_movement[2] -= 5.f * (f32)frame_time;
  if (glfwGetKey(window->glfw_handle, GLFW_KEY_A) == GLFW_PRESS)
    camera_movement[0] -= 5.f * (f32)frame_time;
  if (glfwGetKey(window->glfw_handle, GLFW_KEY_D) == GLFW_PRESS)
    camera_movement[0] += 5.f * (f32)frame_time;
  if (glfwGetKey(window->glfw_handle, GLFW_KEY_R) == GLFW_PRESS)
    camera_movement[1] -= 5.f * (f32)frame_time;
  if (glfwGetKey(window->glfw_handle, GLFW_KEY_SPACE) == GLFW_PRESS)
    camera_movement[1] += 5.f * (f32)frame_time;
  if (camera_movement[0] != 0.f || camera_movement[1] != 0.f || camera_movement[2] != 0.f)
    camera_move(&game->camera, camera_movement);

  // Zoom.
  auto is_sprinting = camera_movement[2] > 0 && control_down;
  if (glfwGetKey(window->glfw_handle, GLFW_KEY_C) == GLFW_PRESS)
    game->camera.fov = glm_rad(30.f * (is_sprinting ? 1.1f : 1.f));
  else
    game->camera.fov = glm_rad(90.f * (is_sprinting ? 1.1f : 1.f));
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
      vec4 color = {};
      u8 escape_kind = (u8)s[i + 1];
      if (s[i + 2] == 'X' || s[i + 2] == 'x') {
        i += 2;
      } else {
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
        color[0] = (f32)r / 255.f;
        color[1] = (f32)g / 255.f;
        color[2] = (f32)b / 255.f;
        color[3] = 1.f;
      }
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

static inline void draw_overlay_text(GameState *game) {
  string_clear(&game->overlay_text);
  if (game->is_paused) {
    string_append(
        &game->overlay_text,
        STRING_LITERAL_ARG("\a\001000000\a\002E0E0E0[\a\001FF8000ESC\a\001000000] Game Paused\a\001FFFFFF\a\002808080\n"));
  } else {
    string_append(&game->overlay_text, STRING_LITERAL_ARG("\a\001FFFFFF\a\002808080Cube Game v0.0.0"));
    if (IS_DEBUG_MODE)
      string_append(&game->overlay_text, STRING_LITERAL_ARG(" (DEBUG BUILD)"));
    string_push(&game->overlay_text, '\n');
  }

  if (isnan(game->display_fps) || isinf(game->display_fps))
    string_snprintf(&game->overlay_text, 64, "FPS ---.--\n");

  else
    string_snprintf(&game->overlay_text, 64, "FPS: %.2lf\n", game->display_fps);

  string_snprintf(&game->overlay_text, 64, "Camera XYZ: %f, %f, %f", game->camera.position[0], game->camera.position[1],
                  game->camera.position[2]);

  if (game->is_wireframe_mode) {
    string_append(&game->overlay_text,
                  STRING_LITERAL_ARG("\n\a\001FFFFFF\a\002X[\a\001FF8000F3+L\a\001FFFFFF] DEBUG: Wireframe Mode"));
  }

  if (game->disable_gl_face_culling) {
    string_append(
        &game->overlay_text,
        STRING_LITERAL_ARG("\n\a\001FFFFFF\a\002X[\a\001FF8000F3+F\a\001FFFFFF] DEBUG: Disable OpenGL Face Culling"));
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

  print(game, pos, game->overlay_text.length, game->overlay_text.buffer);

  glEnable(GL_DEPTH_TEST);
  if (game->is_wireframe_mode)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  if (!game->disable_gl_face_culling)
    glEnable(GL_CULL_FACE);
}

void game_frame(GameState *game, f32 frame_width, f32 frame_height) {
  game->frame_width = frame_width;
  game->frame_height = frame_height;

  glClearColor(.8f, .95f, 1.f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  shader_use(game->shader);

  glBindTexture(GL_TEXTURE_2D, game->texture_atlas.gl);
  glActiveTexture(GL_TEXTURE0);
  glUniform1i(glGetUniformLocation(game->shader.gl, "the_texture"), 0);

  mat4 view_mat = {};
  camera_view_mat(game->camera, view_mat);
  set_uniform_view(game->shader, view_mat);

  mat4 proj_mat = {};
  camera_proj_mat(game->camera, game->frame_width / game->frame_height, proj_mat);
  set_uniform_proj(game->shader, proj_mat);

  for (i32 y = -(WORLD_SIZE_Y / 2); y < (WORLD_SIZE_Y / 2); ++y) {
    for (i32 z = -(WORLD_SIZE_Z / 2); z < (WORLD_SIZE_Z / 2); ++z) {
      for (i32 x = -(WORLD_SIZE_X / 2); x < (WORLD_SIZE_X / 2); ++x) {
        mat4 model_mat = GLM_MAT4_IDENTITY;
        glm_translated(model_mat, (vec3){(f32)x * 32.f, (f32)y * 32.f, (f32)z * 32.f});
        set_uniform_model(game->shader, model_mat);
        auto chunk_mesh = game->chunk_meshes[y + WORLD_SIZE_Y / 2][z + WORLD_SIZE_Z / 2][x + WORLD_SIZE_X / 2];
        mesh_draw(chunk_mesh);
      }
    }
  }

  CHECK_OPENGL_ERROR();

  draw_overlay_text(game);
  CHECK_OPENGL_ERROR();
}
