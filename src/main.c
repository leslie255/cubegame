#include "common.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cglm/cglm.h>
#include <stb/stb_image.h>

#include "game.h"
#include "window.h"

i32 main() {
  [[gnu::cleanup(window_cleanup)]]
  Window *window = window_init(800, 600, "Cube Game");

  [[gnu::cleanup(game_cleanup)]]
  GameState *game = game_init();

  window->game_state = game;
  window->cursor_move_callback = game_cursor_callback;
  window->key_callback = game_key_callback;

  window_disable_cursor(window);

  f64 previous_time = glfwGetTime();
  while (!glfwWindowShouldClose(window->glfw_handle)) {
    glfwSwapBuffers(window->glfw_handle);
    glfwPollEvents();
    game_frame(game, (f32)window->width, (f32)window->height);

    window_update_fps(window);
    game->display_fps = window->average_fps;

    f64 current_time = glfwGetTime();
    f64 frame_time = current_time - previous_time;
    previous_time = current_time;
    game_update_events(game, window, frame_time);
  }

  glfwTerminate();
  return 0;
}
