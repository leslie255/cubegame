#include "camera.h"

void camera_view_mat(Camera camera, mat4 dest) {
  glm_look(camera.position, camera.direction, camera.up, dest);
}

void camera_proj_mat(Camera camera, f32 aspect_ratio, mat4 dest) {
  glm_perspective(camera.fov, aspect_ratio, camera.near_plane_dist, camera.far_plane_dist, dest);
}

void camera_set_direction(Camera *camera, float pitch, float yaw) {
  camera->direction[0] = cosf(glm_rad(pitch)) * cosf(glm_rad(yaw));
  camera->direction[1] = sinf(glm_rad(yaw));
  camera->direction[2] = sinf(glm_rad(pitch)) * cosf(glm_rad(yaw));
  glm_normalize(camera->direction);
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
