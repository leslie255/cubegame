#pragma once

#include "common.h"

typedef enum block_id : u16 {
  BLOCKID_NULL = 0,
  BLOCKID_AIR = 1,
  BLOCKID_STONE,
  BLOCKID_DIRT,
  BLOCKID_GRASS,
} BlockId;

bool block_is_solid(BlockId block_id);
