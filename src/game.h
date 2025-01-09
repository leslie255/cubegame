#include "common.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cglm/cglm.h>
#include <stb/stb_image.h>

#include "camera.h"
#include "shader.h"
#include "window.h"
#include "string.h"
#include "text.h"
#include "texture.h"

typedef struct cube_painter {
  ShaderProgram shader;
  GLuint vao;
  GLuint ebo;
  GLuint vbo;
} CubePainter;

typedef struct game_state {
  /// Shader used for the test square.
  ShaderProgram shader1;
  /// VAO used for the test square.
  GLuint vao1;
  /// EBO used for the test square.
  GLuint ebo1;
  /// VBO used for the test square.
  GLuint vbo1;
  /// Texture object used for the test square.
  Texture texture1;

  FontData *font;
  TextPainter text_painter;

  f32 camera_pitch;
  f32 camera_yaw;
  Camera camera;

  bool is_paused;

  bool cursor_has_moved_before;
  f64 previous_cursor_x;
  f64 previous_cursor_y;

  bool is_wireframe_mode;

  String overlay_text;

  /// FPS of the game.
  /// Passed in from outside.
  /// Leave NAN for unknown.
  f64 fps;

  String overlap_text;
} GameState;

GameState *game_init();

void game_cleanup(GameState **game);

void game_cursor_callback(void *game, Window *window);

void game_key_callback(void *gam_, Window *window, int key, int scancode, int action, int mods);

void game_update_events(GameState *game, Window *window);

void game_frame(GameState *game, f32 frame_width, f32 frame_height);
