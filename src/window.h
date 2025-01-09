#pragma once

#include "common.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

typedef struct window Window;

/// There is circular referencing here, but like nah idc.
typedef void WindowCallbackFn(void *game_state, Window *window);
typedef void WindowKeyCallbackFn(void *game_state, Window *window, int key, int scancode, int action, int mods);

/// `Window` have to always exist on the heap due to GLFW user pointers pointing to it.
/// In the future it may be re-made into an opaque struct for safety, but rn it's still an incomplete abstraction so
/// it's left open.
struct window {
  GLFWwindow *glfw_handle;
  const char *name;
  u32 width;
  u32 height;
  f64 fps;
  f64 cursor_x;
  f64 cursor_y;

  // Event callbacks.
  void *game_state;
  WindowCallbackFn *cursor_move_callback;
  WindowCallbackFn *frame_resize_callback;
  WindowKeyCallbackFn *key_callback;

  /// PRIVATE
  /// Used in FPS calculation.
  f64 previous_seconds;
  /// PRIVATE
  /// Used in FPS calculation.
  u32 frame_count;
  /// PRIVATE
  bool glfw_is_current_context;
};

Window *window_init(u32 width, u32 height, const char *title);

void window_update_fps(Window *window);

void window_cleanup(Window **window);

void window_disable_cursor(Window *window);

void window_restore_cursor(Window *window);

void window_poll_events(Window *window);
