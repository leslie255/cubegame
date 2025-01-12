#pragma once

#include "common.h"
#include "block.h"
#include "mesh.h"
#include "texture.h"

#include <cglm/cglm.h>

typedef struct [[gnu::packed]] tile {
  BlockId id;
} Tile;

typedef struct [[gnu::packed]] chunk {
  Tile blocks[32][32][32];
} ChunkData;

ChunkData *chunk_alloc();

void chunk_cleanup(ChunkData **chunk);

typedef struct quad {
  vec3 coord;
  vec2 size;
  u32 texture_x;
  u32 texture_y;
} Quad;

DEF_ARRAY(QuadArray, quad_array, Quad);

typedef enum cube_direction : usize {
  Direction_North = 0,
  Direction_South,
  Direction_East,
  Direction_West,
  Direction_Up,
  Direction_Down,
} CubeDirection;

/// State used for chunk building.
typedef struct chunk_builder {
  /// Indexed with `CubeDirection`.
  QuadArray quads[6];
  Texture texture_atlas;
} ChunkBuilder;

ChunkBuilder chunk_builder_new(Texture texture_atlas);

void chunk_builder_cleanup(ChunkBuilder *cb);

static inline void chunk_builder_clear_quads(ChunkBuilder *cb);

void build_chunk(ChunkBuilder *cb, Mesh *mesh, const ChunkData *chunk_data, Texture texture);
