#pragma once

#include "common.h"
#include "chunk.h"

#include <cglm/cglm.h>

/// In terms of chunks.
constexpr i32 WORLD_SIZE_Y = 8;
/// In terms of chunks.
constexpr i32 WORLD_SIZE_Z = 16;
/// In terms of chunks.
constexpr i32 WORLD_SIZE_X = 16;

typedef struct world {
  ChunkData *chunks[WORLD_SIZE_Y][WORLD_SIZE_Z][WORLD_SIZE_X];
} WorldData;

WorldData *world_alloc();

void world_cleanup(WorldData **world);

ChunkData **world_get_chunk(WorldData *world, ivec3 chunk_id);

[[nodiscard]]
bool world_to_chunk_coord(ivec3 world_coord, ivec3 dest_chunk_id, ivec3 dest_chunk_local_coord);

void chunk_to_world_coord(ivec3 chunk_id, ivec3 chunk_local_coord, ivec3 dest_world_coord);

void world_set_block(WorldData *world, ivec3 coord, BlockId block_id);

void world_gen(WorldData *world);
