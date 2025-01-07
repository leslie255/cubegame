#include "camera.h"

void camera_view_mat(Camera camera, mat4 dest) {
  glm_look(camera.position, camera.direction, camera.up, dest);
}

void camera_proj_mat(Camera camera, f32 aspect_ratio, mat4 dest) {
  glm_perspective(camera.fov, aspect_ratio, camera.near_plane_dist, camera.far_plane_dist, dest);
}

void camera_rotate_pitch(Camera *camera, f32 d) {
  vec3 right;
  glm_vec3_cross(camera->direction, camera->up, right);
  glm_vec3_normalize(right);

  mat4 rotation;
  glm_rotate_make(rotation, d, right);
  glm_mat4_mulv3(rotation, camera->direction, 1.f, camera->direction);
  glm_vec3_normalize(camera->direction);

  glm_vec3_cross(right, camera->direction, camera->up);
  glm_vec3_normalize(camera->up);
}

void camera_rotate_yaw(Camera *camera, f32 d) {
  // Rotate the camera direction around the absolute up axis (Y axis)
  vec3 absolute_up = {0.0f, 1.0f, 0.0f};
  mat4 rotation;
  glm_rotate_make(rotation, d, absolute_up);
  glm_mat4_mulv3(rotation, camera->direction, 1.0f, camera->direction);
  glm_vec3_normalize(camera->direction);

  // Ensure the up vector remains orthogonal to the direction
  vec3 right;
  glm_vec3_cross(camera->direction, absolute_up, right);
  glm_vec3_cross(right, camera->direction, camera->up);
  glm_vec3_normalize(camera->up);
}

void camera_move(Camera *camera, vec3 v) {
  vec3 forward_xz;
  vec3 right;
  vec3 forward;

  // Extract the forward vector in the XZ plane
  glm_vec3_copy(camera->direction, forward);
  forward[1] = 0.0f;
  glm_vec3_normalize(forward);

  // Calculate the right vector
  glm_vec3_cross(forward, camera->up, right);
  glm_vec3_normalize(right);

  // Compute the movement components
  vec3 movement = GLM_VEC3_ZERO_INIT;
  glm_vec3_scale(forward, v[2], forward_xz);
  glm_vec3_add(movement, forward_xz, movement);

  vec3 right_scaled;
  glm_vec3_scale(right, v[0], right_scaled);
  glm_vec3_add(movement, right_scaled, movement);

  vec3 up_scaled;
  glm_vec3_scale(camera->up, v[1], up_scaled);
  glm_vec3_add(movement, up_scaled, movement);

  // Apply the movement to the camera's position
  glm_vec3_add(camera->position, movement, camera->position);
}
