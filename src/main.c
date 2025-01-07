#include "common.h"

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <cglm/cglm.h>

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

typedef struct camera {
  vec3 position;
  f32 fov;
  f32 near_plane_dist;
  f32 far_plane_dist;
} Camera;

void camera_view_mat(Camera camera, mat4 proj) {
  vec3 position_inverse = {
      -camera.position[0],
      -camera.position[1],
      -camera.position[2],
  };
  glm_translate(proj, position_inverse);
}

i32 main() {
  [[gnu::cleanup(window_cleanup)]]
  Window *window = window_init(800, 600, "Cube Game");

  ShaderProgram shader = shader_init(sizeof(VERTEX_SHADER), VERTEX_SHADER,
                                     sizeof(FRAGMENT_SHADER), FRAGMENT_SHADER);
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
      -.5f, -.5f, 0.f, /**/ 0.f, 0.f, 1.f, //
      -.5f, +.5f, 0.f, /**/ 0.f, 1.f, 1.f  //
  };

  const u32 indices[] = {
      0, 1, 3, //
      1, 2, 3, //
  };

  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  GLuint ebo;
  glGenBuffers(1, &ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);
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

  while (!glfwWindowShouldClose(window->glfw_handle)) {
    glfwSwapBuffers(window->glfw_handle);
    glClearColor(.1f, .1f, .1f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (glfwGetKey(window->glfw_handle, GLFW_KEY_ESCAPE) == GLFW_PRESS)
      break;

    if (glfwGetKey(window->glfw_handle, GLFW_KEY_F3) == GLFW_PRESS)
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    else
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    mat4 view_mat = GLM_MAT4_IDENTITY;
    glm_translate(view_mat, (vec3){
                                (f32)cos(glfwGetTime()) / 1.5f,
                                (f32)sin(glfwGetTime()) / 1.5f,
                                -2.5f,
                            });
    mat4 proj_mat = GLM_MAT4_IDENTITY;
    glm_perspective(glm_rad(45.f), 800.f / 600.f, .1f, 100.f, proj_mat);

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
      glm_translate(model_mat, (vec3){0.f, -0.1f, 0.f});
      glm_rotate_x(model_mat, glm_rad(-90.f), model_mat);
      glm_rotate_z(model_mat, (f32)glfwGetTime(), model_mat);
      glUniformMatrix4fv(uniform_model, 1, false, (f32 *)model_mat);
      glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    });

    glfwPollEvents();
    window_update_fps(window);
  }

  glfwTerminate();
  return 0;
}
