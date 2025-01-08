#include "common.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cglm/cglm.h>
#include <stb/stb_image.h>

#include "camera.h"
#include "shader.h"
#include "window.h"

static constexpr char WINDOW_TITLE[] = "Cube Game";
constexpr f32 CAMERA_INIT_PITCH = 0.f;
constexpr f32 CAMERA_INIT_YAW = -90.f;

static constexpr char VERTEX_SHADER[] = //
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

static constexpr char FRAGMENT_SHADER[] = //
    "#version 330 core\n"
    "in vec2 tex_coord;\n"
    "out vec4 frag_color;\n"
    "uniform sampler2D the_texture;\n"
    "void main() {\n"
    "  frag_color = texture(the_texture, tex_coord);\n"
    "}\n";

const f32 vertices[] = {
    //                 Coords              Texture
    /* Top right    */ 1.f, 1.f, 0.f, /**/ 1.f, 0.f,
    /* Bottom right */ 1.f, 0.f, 0.f, /**/ 1.f, 1.f,
    /* Top left     */ 0.f, 0.f, 0.f, /**/ 0.f, 1.f,
    /* Bottom left  */ 0.f, 1.f, 0.f, /**/ 0.f, 0.f,
};

const u32 indices[] = {
    0, 1, 3, //
    1, 2, 3,
};

typedef struct game_state {
  ShaderProgram the_shader;
  GLuint the_vao;
  GLuint the_ebo;
  GLuint the_vbo;
  GLuint the_texture;
  Camera camera;
  bool is_paused;

  f32 camera_pitch;
  f32 camera_yaw;

  bool cursor_has_moved_before;
  f64 previous_cursor_x;
  f64 previous_cursor_y;

  bool is_wireframe_mode;
} GameState;

void game_init(GameState *game) {
  game->is_paused = false;
  game->camera_pitch = CAMERA_INIT_PITCH;
  game->camera_yaw = CAMERA_INIT_YAW;
  game->cursor_has_moved_before = false;
  game->previous_cursor_x = 0.f;
  game->previous_cursor_y = 0.f;
  game->is_wireframe_mode = false;

  // Shader.
  game->the_shader = shader_init(sizeof(VERTEX_SHADER), VERTEX_SHADER, sizeof(FRAGMENT_SHADER), FRAGMENT_SHADER);
  // Verify uniforms are ok.
  ASSERT(glGetUniformLocation(game->the_shader.gl_handle, "model") != -1);
  ASSERT(glGetUniformLocation(game->the_shader.gl_handle, "view") != -1);
  ASSERT(glGetUniformLocation(game->the_shader.gl_handle, "proj") != -1);

  // VAO.
  glGenVertexArrays(1, &game->the_vao);
  glBindVertexArray(game->the_vao);
  // EBO.
  glGenBuffers(1, &game->the_ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, game->the_ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
  // VBO.
  glGenBuffers(1, &game->the_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, game->the_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer(
      /* location    */ 0,
      /* size (vec3) */ 3,
      /* type        */ GL_FLOAT,
      /* normalized? */ GL_FALSE,
      /* stride      */ sizeof(float[5]),
      /* offset      */ nullptr);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(
      /* location    */ 1,
      /* size (vec3) */ 2,
      /* type        */ GL_FLOAT,
      /* normalized? */ GL_FALSE,
      /* stride      */ sizeof(float[5]),
      /* offset      */ (void *)sizeof(float[3]));
  glEnableVertexAttribArray(1);

  // Texture.
  i32 texture_width;
  i32 texture_height;
  i32 texture_n_channels;
  constexpr char test_texture_path[] = "res/texture/test_texture.png";
  auto texture_data = stbi_load(test_texture_path, &texture_width, &texture_height, &texture_n_channels, 0);
  ASSERT(texture_data != nullptr);
  printf("Loaded texture, dimension: %dx%d, channels: %d\n", texture_width, texture_height, texture_n_channels);
  glGenTextures(1, &game->the_texture);
  glBindTexture(GL_TEXTURE_2D, game->the_texture);
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
  USE_VARIABLE(scancode);
  USE_VARIABLE(mods);
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
      printf("Game is paused\n");
      window_restore_cursor(window);
    } else {
      printf("Game is unpaused\n");
      window_disable_cursor(window);
      game->cursor_has_moved_before = false;
    }
  }
}

void update_events(Window *window, GameState *game) {
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

void game_frame(GameState *game, f32 frame_width, f32 frame_height) {
  glClearColor(.1f, .1f, .1f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  mat4 view_mat = {};
  camera_view_mat(game->camera, view_mat);

  mat4 proj_mat = {};
  camera_proj_mat(game->camera, frame_width / frame_height, proj_mat);

  auto uniform_model = glGetUniformLocation(game->the_shader.gl_handle, "model");
  auto uniform_view = glGetUniformLocation(game->the_shader.gl_handle, "view");
  auto uniform_proj = glGetUniformLocation(game->the_shader.gl_handle, "proj");

  ({
    mat4 model_mat = GLM_MAT4_IDENTITY;
    glUniformMatrix4fv(uniform_model, 1, false, (f32 *)model_mat);
    glUniformMatrix4fv(uniform_view, 1, false, (f32 *)view_mat);
    glUniformMatrix4fv(uniform_proj, 1, false, (f32 *)proj_mat);
    shader_use(game->the_shader);
    glUniform1i(glGetUniformLocation(game->the_shader.gl_handle, "the_texture"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, game->the_texture);
    glBindVertexArray(game->the_vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, game->the_ebo);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
  });
}

i32 main() {
  [[gnu::cleanup(window_cleanup)]]
  Window *window = window_init(800, 600, WINDOW_TITLE);

  GameState *game = xalloc(GameState, 1);
  game_init(game);

  window->game_state = game;
  window->cursor_move_callback = game_cursor_callback;
  window->key_callback = game_key_callback;

  window_disable_cursor(window);

  while (!glfwWindowShouldClose(window->glfw_handle)) {
    glfwSwapBuffers(window->glfw_handle);
    glfwPollEvents();
    update_events(window, game);
    game_frame(game, (f32)window->width, (f32)window->height);
    window_update_fps(window);
  }

  glDeleteVertexArrays(1, &game->the_vao);
  glDeleteBuffers(1, &game->the_vbo);
  glDeleteBuffers(1, &game->the_ebo);
  glfwTerminate();
  return 0;
}
