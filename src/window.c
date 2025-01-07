#include "window.h"

static bool WINDOW_EXISTS = false;
static Window THE_ONLY_WINDOW = {};

constexpr usize TITLE_BUFFER_SIZE = 256;
static inline void update_title_with_fps(Window window) {
  if (isnan(window.fps))
    snprintf(window.real_title, TITLE_BUFFER_SIZE, "%s (FPS: ---.--)", window.name);
  else
    snprintf(window.real_title, TITLE_BUFFER_SIZE, "%s (FPS: %.2lf)", window.name, window.fps);
  glfwSetWindowTitle(window.glfw_handle, window.real_title);
}

/// Panics on error.
Window *window_init(u32 width, u32 height, const char *name) {
  ASSERT(!WINDOW_EXISTS);
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
  glfwSetErrorCallback(window_glfw_error_callback);
  GLFWwindow *glfw_window = glfwCreateWindow((int)width, (int)height, name, nullptr, nullptr);
  ASSERT(glfw_window != nullptr);
  glfwMakeContextCurrent(glfw_window);
  glfwSetFramebufferSizeCallback(glfw_window, window_glfw_resize_callback);
  glfwSetCursorPosCallback(glfw_window, cursor_position_callback);
  ASSERT(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) != 0);
  THE_ONLY_WINDOW = (Window){
      .glfw_handle = glfw_window,
      .width = width,
      .height = height,
      .name = name,
      .real_title = xalloc(char, TITLE_BUFFER_SIZE),
      .previous_seconds = glfwGetTime(),
      .fps = NAN,
  };
  update_title_with_fps(THE_ONLY_WINDOW);
  return &THE_ONLY_WINDOW;
}

void window_cleanup(Window **window_) {
  auto window = *window_;
  xfree(window->real_title);
}

void window_glfw_resize_callback(GLFWwindow *window, int width, int height) {
  if (window != THE_ONLY_WINDOW.glfw_handle)
    return;
  glViewport(0, 0, width, height);
  THE_ONLY_WINDOW.width = (u32)width;
  THE_ONLY_WINDOW.height = (u32)height;
}

void cursor_position_callback(GLFWwindow *window, f64 xpos, f64 ypos) {
  if (window != THE_ONLY_WINDOW.glfw_handle)
    return;
  THE_ONLY_WINDOW.cursor_x = xpos;
  THE_ONLY_WINDOW.cursor_y = ypos;
}

void window_glfw_error_callback(i32 error, const char *description) {
  fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

void window_update_fps(Window *window) {
  f64 current_seconds = glfwGetTime();
  f64 elapsed_seconds = current_seconds - window->previous_seconds;
  ++window->frame_count;
  if (elapsed_seconds > 1.) {
    window->previous_seconds = current_seconds;
    window->fps = (f64)window->frame_count / elapsed_seconds;
    window->frame_count = 0;
    update_title_with_fps(*window);
  }
}

void window_disable_cursor(Window *window) {
  glfwSetInputMode(window->glfw_handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwSetInputMode(window->glfw_handle, GLFW_RAW_MOUSE_MOTION, true);
}

void window_restore_cursor(Window *window) {
  glfwSetInputMode(window->glfw_handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}
