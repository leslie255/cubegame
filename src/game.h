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

typedef struct cube_painter {
  ShaderProgram shader;
  LoadedMesh cube_mesh;
} CubePainter;

CubePainter cube_painter_new();

void cube_painter_cleanup(CubePainter *cp);

typedef struct game_state {
  /// Texture object used for the test square.
  Texture test_texture;
  Texture texture_atlas;

  FontData *font;
  TextPainter text_painter;

  f32 camera_pitch;
  f32 camera_yaw;
  Camera camera;

  CubePainter cube_painter;

  ChunkData *test_chunk;

  bool is_paused;

  bool cursor_has_moved_before;
  f64 previous_cursor_x;
  f64 previous_cursor_y;

  bool is_wireframe_mode;
  bool disable_gl_face_culling;

  String overlay_text;

  /// Average FPS over the last period of time, for FPS display.
  /// Passed in from outside.
  /// Leave NAN for unknown.
  f64 display_fps;

  String overlap_text;

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

void cube_paint( //
    CubePainter *cp,
    GameState *game,
    vec3 coord,
    CubeFace faces,
    Texture texture,
    u32 offset_x,
    u32 offset_y);
