#pragma once

#include "common.h"

typedef enum block_id : u16 {
  BLOCKID_AIR = 0,
  BLOCKID_STONE,
  BLOCKID_DIRT,
  BLOCKID_GRASS,
  BLOCKID_TEST = 255,
} BlockId;

bool block_is_solid(BlockId block);

void block_texture_atlast_coord(BlockId block_id, u32 *dest_x, u32 *dest_y);
