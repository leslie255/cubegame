#pragma once

#include "common.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

/// `Window` have to always exist on the heap due to GLFW user pointers pointing to it.
/// In the future it may be re-made into an opaque struct for safety, but rn it's still an incomplete abstraction so
/// it's left open.
typedef struct window {
  GLFWwindow *glfw_handle;
  const char *name;
  u32 width;
  u32 height;
  f64 fps;
  f64 cursor_x;
  f64 cursor_y;

  /// PRIVATE
  /// Used in FPS calculation.
  f64 previous_seconds;
  /// PRIVATE
  /// Used in FPS calculation.
  u32 frame_count;
  /// PRIVATE
  /// For displaying the "(fps: ---.--)" part in window title.
  char *real_title;
} Window;

Window *window_init(u32 width, u32 height, const char *title);

void window_glfw_resize_callback(GLFWwindow *, int width, int height);

void cursor_position_callback(GLFWwindow *window, f64 xpos, f64 ypos);

void window_glfw_error_callback(i32 error, const char *description);

void window_update_fps(Window *window);

void window_cleanup(Window **window);

void window_disable_cursor(Window *window);

void window_restore_cursor(Window *window);
