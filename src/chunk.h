#pragma once

#include "common.h"
#include "block.h"

#include <cglm/cglm.h>

typedef struct [[gnu::packed]] tile {
  BlockId id;
} Tile;

typedef struct [[gnu::packed]] chunk {
  Tile blocks[32][32][32];
} ChunkData;

ChunkData *chunk_alloc();

void chunk_cleanup(ChunkData **chunk);
