#pragma once

#include "common.h"

#include <cglm/cglm.h>

void atlas_normalize(u32 total_width, u32 total_height, u32 x, u32 y, vec2 dest);

void atlas_mat(u32 texture_width, u32 texture_height, u32 x_min, u32 y_min, u32 x_max, u32 y_max, mat3 dest);
