#include "common.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cglm/cglm.h>
#include <stb/stb_image.h>

#include "camera.h"
#include "shader.h"
#include "window.h"

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

GameState *game_init();

void game_cleanup(GameState **game);

void game_cursor_callback(void *game_, Window *window);

void game_key_callback(void *game_, Window *window, int key, int scancode, int action, int mods);

void update_events(Window *window, GameState *game);

void game_frame(GameState *game, f32 frame_width, f32 frame_height);

