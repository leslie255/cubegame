#include "window.h"

/// Get the `Window` object from `GLFWwindow`.
static inline Window *get_window(GLFWwindow *glfw_window) {
  Window *window = glfwGetWindowUserPointer(glfw_window);
  DEBUG_ASSERT(window != nullptr);
  return window;
}

static inline void resize_callback(GLFWwindow *glfw_window, int width, int height) {
  auto window = get_window(glfw_window);
  glViewport(0, 0, width, height);
  window->width = (u32)width;
  window->height = (u32)height;
  if (window->cursor_move_callback != nullptr)
    (window->cursor_move_callback)(window->game_state, window);
}

static inline void cursor_position_callback(GLFWwindow *glfw_window, f64 xpos, f64 ypos) {
  auto window = get_window(glfw_window);
  window->cursor_x = xpos;
  window->cursor_y = ypos;
  if (window->cursor_move_callback != nullptr)
    (window->cursor_move_callback)(window->game_state, window);
}

static inline void error_callback(i32 error, const char *description) {
  fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

static inline void key_callback(GLFWwindow *glfw_window, int key, int scancode, int action, int mods) {
  auto window = get_window(glfw_window);
#ifdef __APPLE__
  if ((mods & GLFW_MOD_SUPER) != 0 && key == GLFW_KEY_W)
    glfwSetWindowShouldClose(glfw_window, true);
#endif
  if (window->key_callback != nullptr)
    (window->key_callback)(window->game_state, window, key, scancode, action, mods);
}

constexpr usize TITLE_BUFFER_SIZE = 256;
static inline void update_title_with_fps(Window *window) {
  if (isnan(window->fps))
    snprintf(window->real_title, TITLE_BUFFER_SIZE, "%s (FPS: ---.--)", window->name);
  else
    snprintf(window->real_title, TITLE_BUFFER_SIZE, "%s (FPS: %.2lf)", window->name, window->fps);
  glfwSetWindowTitle(window->glfw_handle, window->real_title);
}

/// Panics on error.
Window *window_init(u32 width, u32 height, const char *name) {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
  glfwSetErrorCallback(error_callback);
  GLFWwindow *glfw_window = glfwCreateWindow((int)width, (int)height, name, nullptr, nullptr);
  ASSERT(glfw_window != nullptr);
  glfwMakeContextCurrent(glfw_window);
  glfwSetFramebufferSizeCallback(glfw_window, resize_callback);
  glfwSetCursorPosCallback(glfw_window, cursor_position_callback);
  glfwSetKeyCallback(glfw_window, key_callback);
  ASSERT(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) != 0);
  Window *window = PUT_ON_HEAP(((Window){
      .glfw_handle = glfw_window,
      .width = width,
      .height = height,
      .name = name,
      .real_title = xalloc(char, TITLE_BUFFER_SIZE),
      .previous_seconds = glfwGetTime(),
      .fps = NAN,
  }));
  glfwSetWindowUserPointer(glfw_window, window);
  update_title_with_fps(window);
  return window;
}

void window_cleanup(Window **window_) {
  auto window = *window_;
  xfree(window->real_title);
  xfree(window);
  if (IS_DEBUG_MODE)
    window_ = nullptr;
}

void window_update_fps(Window *window) {
  f64 current_seconds = glfwGetTime();
  f64 elapsed_seconds = current_seconds - window->previous_seconds;
  ++window->frame_count;
  if (elapsed_seconds > 1.) {
    window->previous_seconds = current_seconds;
    window->fps = (f64)window->frame_count / elapsed_seconds;
    window->frame_count = 0;
    update_title_with_fps(window);
  }
}

void window_disable_cursor(Window *window) {
  glfwSetInputMode(window->glfw_handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  // glfwSetInputMode(window->glfw_handle, GLFW_RAW_MOUSE_MOTION, true);
}

void window_restore_cursor(Window *window) {
  glfwSetInputMode(window->glfw_handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

void window_poll_events(Window *window) {
  MARK_USED(window);
  glfwPollEvents();
}
