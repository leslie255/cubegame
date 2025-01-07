#pragma once

#include "common.h"

#include <cglm/cglm.h>

typedef struct camera {
  vec3 position;
  vec3 direction;
  vec3 up;
  f32 fov;
  f32 near_plane_dist;
  f32 far_plane_dist;
} Camera;

void camera_view_mat(Camera camera, mat4 dest);

void camera_proj_mat(Camera camera, f32 aspect_ratio, mat4 dest);

void camera_set_direction(Camera *camera, float pitch, float yaw);

void camera_move(Camera *camera, vec3 v);
