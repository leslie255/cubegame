#pragma once

#include "common.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

typedef struct window {
  GLFWwindow *glfw_handle;
  const char *name;
  u32 width;
  u32 height;
  f64 fps;

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

void window_glfw_error_callback(i32 error, const char *description);

void window_update_fps(Window *window);

void window_cleanup(Window **window);
