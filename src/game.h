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
#include "chunk.h"
#include "mesh.h"
#include "world.h"

typedef enum cube_face : u8 {
  CubeFace_All = 0b11111111,
  /// +Y
  CubeFace_Top = 0b000001,
  /// -Y
  CubeFace_Bottom = 0b000010,
  /// +Z
  CubeFace_South = 0b001000,
  /// -Z
  CubeFace_North = 0b000100,
  /// +X
  CubeFace_East = 0b010000,
  /// -X
  CubeFace_West = 0b100000,
} CubeFace;

typedef struct game_state {
  Texture texture_atlas;
  FontData *font;
  String overlay_text;
  TextPainter text_painter;
  ShaderProgram shader;
  WorldData *world;
  ChunkBuilder chunk_builder;
  Mesh chunk_meshes[WORLD_SIZE_Y][WORLD_SIZE_Z][WORLD_SIZE_X];

  f32 camera_pitch;
  f32 camera_yaw;
  Camera camera;

  bool is_paused;
  bool cursor_has_moved_before;
  f64 previous_cursor_x;
  f64 previous_cursor_y;
  bool is_wireframe_mode;
  bool disable_gl_face_culling;

  /// Average FPS over the last period of time, for FPS display.
  /// Passed in from outside.
  /// Leave NAN for unknown.
  f64 display_fps;

  /// Updated when `game_frame` is called with `frame_width` being one of its parameter.
  f32 frame_width;
  /// Updated when `game_frame` is called with `frame_height` being one of its parameter.
  f32 frame_height;
} GameState;

GameState *game_init();

void game_cleanup(GameState **game);

void game_cursor_callback(void *game, Window *window);

void game_key_callback(void *gam_, Window *window, int key, int scancode, int action, int mods);

void game_update_events(GameState *game, Window *window, f64 frame_time);

void game_frame(GameState *game, f32 frame_width, f32 frame_height);
