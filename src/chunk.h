#pragma once

#include "common.h"

#include <cglm/cglm.h>

typedef enum block_id : u16 {
  BLOCKID_NULL = 0,
  BLOCKID_AIR = 1,
  BLOCKID_STONE,
  BLOCKID_DIRT,
  BLOCKID_GRASS,
} BlockId;

typedef struct block {
  BlockId id;
} Block;

typedef struct chunk {
  Block blocks[32 * 32 * 32];
} ChunkData;

ChunkData *chunk_alloc();
