#include "atlas.h"

static inline void normalize_tex_coord(u32 total_width, u32 total_height, u32 x, u32 y, vec2 dest) {
  dest[0] = (f32)x / (f32)total_width;
  dest[1] = (f32)y / (f32)total_height;
}

void atlas_mat(u32 texture_width, u32 texture_height, u32 x_min, u32 y_min, u32 x_max, u32 y_max, mat3 dest) {
  vec2 min;
  vec2 max;
  normalize_tex_coord(texture_width, texture_height, x_min, y_min, min);
  normalize_tex_coord(texture_width, texture_height, x_max, y_max, max);
  vec2 d;
  glm_vec2_sub(max, min, d);
  glm_mat3_identity(dest);
  glm_translate2d(dest, min);
  glm_scale2d(dest, d);
}
