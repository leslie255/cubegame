#include "common.h"

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <cglm/cglm.h>

#include "camera.h"
#include "shader.h"
#include "window.h"

static const char VERTEX_SHADER[] = //
    "#version 330 core\n"
    "layout (location = 0) in vec3 the_pos;\n"
    "layout (location = 1) in vec3 the_color;\n"
    "out vec3 color;\n"
    "uniform mat4 model;\n"
    "uniform mat4 view;\n"
    "uniform mat4 proj;\n"
    "void main() {\n"
    "  gl_Position = proj * view * model * vec4(the_pos, 1.f);\n"
    "  color = the_color;\n"
    "}\n";

static const char FRAGMENT_SHADER[] = //
    "#version 330 core\n"
    "in vec3 color;\n"
    "out vec4 frag_color;\n"
    "void main() {\n"
    "  frag_color = vec4(color, 1.f);\n"
    "}\n";

static f32 camera_pitch = 0.f;
static f32 camera_yaw = -90.f;

void cursor_position_callback(GLFWwindow *window, f64 xpos, f64 ypos) {
  static f64 prev_x = 0.;
  static f64 prev_y = 0.;
  static bool is_first_frame = true;

  if (is_first_frame) {
    prev_x = xpos;
    prev_y = ypos;
    is_first_frame = false;
    return;
  }

  f32 sensitivity = 0.001f;
  f32 dx = (f32)(xpos - prev_x);
  f32 dy = (f32)(ypos - prev_y);
  prev_x = xpos;
  prev_y = ypos;

  camera_pitch -= dy * sensitivity;
  camera_yaw += dx * sensitivity;
  if (camera_pitch > 89.f)
    camera_pitch = 89.f;
  if (camera_pitch < -89.f)
    camera_pitch = -89.f;
}

static inline void handle_events(Window *window, Camera *camera) {
  camera->direction[0] = 1.f;
  camera->direction[1] = 0.f;
  camera->direction[2] = 0.f;
  camera_rotate_pitch(camera, camera_pitch);
  camera_rotate_yaw(camera, camera_yaw);

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

  if (camera_movement[0] != 0 || camera_movement[1] != 0 || camera_movement[2] != 0)
    camera_move(camera, camera_movement);
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
      // Coords               // Colors
      +.5f, +.5f, 0.f, /**/ 1.f, 0.f, 0.f, //
      +.5f, -.5f, 0.f, /**/ 1.f, 1.f, 0.f, //
      -.5f, -.5f, 0.f, /**/ 0.f, 1.f, 0.f, //
      -.5f, +.5f, 0.f, /**/ 0.f, 0.f, 1.f, //
  };

  const u32 indices[] = {
      0,
      1,
      3, //
      1,
      2,
      3, //
  };

  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  GLuint ebo;
  glGenBuffers(1, &ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
  GLuint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer(0,                // location
                        3,                // size (vec3)
                        GL_FLOAT,         // type
                        GL_FALSE,         // normalized?
                        sizeof(float[6]), // stride
                        nullptr           // offset
  );
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1,                       // location
                        3,                       // size (vec3)
                        GL_FLOAT,                // type
                        GL_FALSE,                // normalized?
                        sizeof(float[6]),        // stride
                        (void *)sizeof(float[3]) // offset
  );
  glEnableVertexAttribArray(1);

  // glEnable(GL_CULL_FACE);
  // glCullFace(GL_BACK);
  // glFrontFace(GL_CW);
  glEnable(GL_DEPTH_TEST);

  Camera camera = {
      .position = {0.f, 0.f, 5.f},
      .direction = {0.f, 0.f, 0.f},
      .up = {0.f, 1.f, 0.f},
      .fov = glm_rad(90.f),
      .near_plane_dist = 0.1f,
      .far_plane_dist = 100.f,
  };
  glm_normalize(camera.direction);
  glm_normalize(camera.up);

  window_raw_mouse_mode(window);
  glfwSetCursorPosCallback(window->glfw_handle, cursor_position_callback);

  while (!glfwWindowShouldClose(window->glfw_handle)) {
    glfwSwapBuffers(window->glfw_handle);
    glClearColor(.1f, .1f, .1f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    handle_events(window, &camera);

    mat4 view_mat = {};
    camera_view_mat(camera, view_mat);
    mat4 proj_mat = {};
    camera_proj_mat(camera, (f32)window->width / (f32)window->height, proj_mat);

    ({
      mat4 model_mat = GLM_MAT4_IDENTITY;
      glm_translate(model_mat, (vec3){0.f, 0.1f, 0.f});
      glm_rotate_x(model_mat, glm_rad(-90.f), model_mat);
      glm_rotate_z(model_mat, (f32)glfwGetTime(), model_mat);
      glUniformMatrix4fv(uniform_model, 1, false, (f32 *)model_mat);
      glUniformMatrix4fv(uniform_view, 1, false, (f32 *)view_mat);
      glUniformMatrix4fv(uniform_proj, 1, false, (f32 *)proj_mat);
      shader_use(shader);
      glBindVertexArray(vao);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
      glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    });

    ({
      mat4 model_mat = GLM_MAT4_IDENTITY;
      glm_translate(model_mat, (vec3){0.f, 0.f, -1.5f});
      glm_rotate_y(model_mat, glm_rad(45.f), model_mat);
      glUniformMatrix4fv(uniform_model, 1, false, (f32 *)model_mat);
      glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    });

    glfwPollEvents();
    window_update_fps(window);
  }

  glfwTerminate();
  return 0;
}
