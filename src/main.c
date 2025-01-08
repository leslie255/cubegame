#include "common.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cglm/cglm.h>
#include <stb/stb_image.h>

#include "camera.h"
#include "shader.h"
#include "window.h"

constexpr f32 CAMERA_INIT_PITCH = 0.f;
constexpr f32 CAMERA_INIT_YAW = -90.f;

static const char VERTEX_SHADER[] = //
    "#version 330 core\n"
    "layout (location = 0) in vec3 the_pos;\n"
    "layout (location = 1) in vec2 the_tex_coord;\n"
    "out vec2 tex_coord;\n"
    "uniform mat4 model;\n"
    "uniform mat4 view;\n"
    "uniform mat4 proj;\n"
    "void main() {\n"
    "  gl_Position = proj * view * model * vec4(the_pos, 1.f);\n"
    "  tex_coord = the_tex_coord;\n"
    "}\n";

static const char FRAGMENT_SHADER[] = //
    "#version 330 core\n"
    "in vec2 tex_coord;\n"
    "out vec4 frag_color;\n"
    "uniform sampler2D the_texture;\n"
    "void main() {\n"
    "  frag_color = texture(the_texture, tex_coord);\n"
    "}\n";

static inline void handle_events(Window *window, Camera *camera) {

  if (glfwGetKey(window->glfw_handle, GLFW_KEY_F3) == GLFW_PRESS)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  else
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  vec3 camera_movement = {};
  if (glfwGetKey(window->glfw_handle, GLFW_KEY_W) == GLFW_PRESS)
    camera_movement[2] += 0.05f;
  if (glfwGetKey(window->glfw_handle, GLFW_KEY_S) == GLFW_PRESS)
    camera_movement[2] -= 0.05f;
  if (glfwGetKey(window->glfw_handle, GLFW_KEY_A) == GLFW_PRESS)
    camera_movement[0] -= 0.05f;
  if (glfwGetKey(window->glfw_handle, GLFW_KEY_D) == GLFW_PRESS)
    camera_movement[0] += 0.05f;
  if (glfwGetKey(window->glfw_handle, GLFW_KEY_R) == GLFW_PRESS)
    camera_movement[1] -= 0.05f;
  if (glfwGetKey(window->glfw_handle, GLFW_KEY_SPACE) == GLFW_PRESS)
    camera_movement[1] += 0.05f;
  if (glfwGetKey(window->glfw_handle, GLFW_KEY_C) == GLFW_PRESS)
    camera->fov = glm_rad(30.f);
  else
    camera->fov = glm_rad(90.f);

  if (camera_movement[0] != 0 || camera_movement[1] != 0 || camera_movement[2] != 0)
    camera_move(camera, camera_movement);

  // TODO: These are ugly, and shouldn't be here.
  static f32 camera_pitch = CAMERA_INIT_PITCH;
  static f32 camera_yaw = CAMERA_INIT_YAW;

  static f64 previous_cursor_x = 0.;
  static f64 previous_cursor_y = 0.;
  static bool is_first_time = true;

  f32 sensitivity = 0.1f;
  f32 dx = (f32)(window->cursor_x - previous_cursor_x);
  f32 dy = (f32)(window->cursor_y - previous_cursor_y);
  previous_cursor_x = window->cursor_x;
  previous_cursor_y = window->cursor_y;
  if (absf(dx) > 0.f || absf(dy) > 0.f) {
    if (is_first_time) {
      previous_cursor_x = window->cursor_x;
      previous_cursor_y = window->cursor_y;
      is_first_time = false;
      return;
    }
    camera_pitch -= dy * sensitivity;
    camera_yaw += dx * sensitivity;
    if (camera_pitch > 89.9f)
      camera_pitch = 89.9f;
    if (camera_pitch < -89.9f)
      camera_pitch = -89.9f;
    camera_set_direction(camera, camera_yaw, camera_pitch);
  }
}

i32 main() {
  [[gnu::cleanup(window_cleanup)]]
  Window *window = window_init(800, 600, "Cube Game");

  ShaderProgram shader = shader_init(sizeof(VERTEX_SHADER), VERTEX_SHADER, sizeof(FRAGMENT_SHADER), FRAGMENT_SHADER);
  GLint uniform_model = glGetUniformLocation(shader.gl_handle, "model");
  GLint uniform_view = glGetUniformLocation(shader.gl_handle, "view");
  GLint uniform_proj = glGetUniformLocation(shader.gl_handle, "proj");
  assert(uniform_model != -1);
  assert(uniform_view != -1);
  assert(uniform_proj != -1);

  const f32 vertices[] = {
      //                 Coords              Texture
      /* Top right    */ 1.f, 1.f, 0.f, /**/ 1.f, 0.f,
      /* Bottom right */ 1.f, 0.f, 0.f, /**/ 1.f, 1.f,
      /* Top left     */ 0.f, 0.f, 0.f, /**/ 0.f, 1.f,
      /* Bottom left  */ 0.f, 1.f, 0.f, /**/ 0.f, 0.f,
  };

  const u32 indices[] = {
      0, 1, 3, //
      1, 2, 3,
  };

  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glBindVertexArray(vao); // ?
  GLuint ebo;
  glGenBuffers(1, &ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
  GLuint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer(
      /* location    */ 0,
      /* size (vec3) */ 3,
      /* type        */ GL_FLOAT,
      /* normalized? */ GL_FALSE,
      /* stride      */ sizeof(float[5]),
      /* offset      */ nullptr);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(
      /* location    */ 1,
      /* size (vec3) */ 2,
      /* type        */ GL_FLOAT,
      /* normalized? */ GL_FALSE,
      /* stride      */ sizeof(float[5]),
      /* offset      */ (void *)sizeof(float[3]));
  glEnableVertexAttribArray(1);

  GLuint texture = ({
    i32 width;
    i32 height;
    i32 n_channels;
    constexpr char test_texture_path[] = "res/texture/test_texture.png";
    auto texture_data = stbi_load(test_texture_path, &width, &height, &n_channels, 0);
    ASSERT(texture_data != nullptr);
    printf("Loaded texture, dimension: %dx%d, channels: %d\n", width, height, n_channels);
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    GLenum format;
    switch (n_channels) {
    case 1: {
      format = GL_RED;
    } break;
    case 2: {
      format = GL_RG;
    } break;
    case 3: {
      format = GL_RGB;
    } break;
    case 4: {
      format = GL_RGBA;
    } break;
    }
    glTexImage2D(GL_TEXTURE_2D, 0, (GLint)format, width, height, 0, format, GL_UNSIGNED_BYTE, texture_data);
    stbi_image_free(texture_data);
    texture;
  });

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // glEnable(GL_CULL_FACE);
  // glCullFace(GL_BACK);
  // glFrontFace(GL_CW);

  glEnable(GL_DEPTH_TEST);

  Camera camera = {
      .position = {0.f, 0.f, 5.f},
      .direction = {0.f, 0.f, -1.f},
      .up = {0.f, 1.f, 0.f},
      .fov = glm_rad(90.f),
      .near_plane_dist = 0.1f,
      .far_plane_dist = 100.f,
  };
  glm_normalize(camera.up);
  glm_normalize(camera.direction);

  window_disable_cursor(window);

  while (!glfwWindowShouldClose(window->glfw_handle)) {
    glClearColor(.1f, .1f, .1f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    handle_events(window, &camera);

    mat4 view_mat = {};
    camera_view_mat(camera, view_mat);

    mat4 proj_mat = {};
    camera_proj_mat(camera, (f32)window->width / (f32)window->height, proj_mat);

    ({
      mat4 model_mat = GLM_MAT4_IDENTITY;
      glUniformMatrix4fv(uniform_model, 1, false, (f32 *)model_mat);
      glUniformMatrix4fv(uniform_view, 1, false, (f32 *)view_mat);
      glUniformMatrix4fv(uniform_proj, 1, false, (f32 *)proj_mat);
      shader_use(shader);
      glUniform1i(glGetUniformLocation(shader.gl_handle, "the_texture"), 0);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, texture);
      glBindVertexArray(vao);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
      glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    });

    glfwSwapBuffers(window->glfw_handle);
    glfwPollEvents();
    window_update_fps(window);
  }

  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo);
  glDeleteBuffers(1, &ebo);
  glfwTerminate();
  return 0;
}
