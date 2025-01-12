#pragma once

#include "common.h"
#include "block.h"
#include "mesh.h"

#include <cglm/cglm.h>

typedef struct [[gnu::packed]] tile {
  BlockId id;
} Tile;

typedef struct [[gnu::packed]] chunk {
  Tile blocks[32][32][32];
} ChunkData;

ChunkData *chunk_alloc();

void chunk_cleanup(ChunkData **chunk);

typedef struct chunk_mesh {
  Mesh mesh;
} ChunkMesh;

ChunkMesh chunk_mesh_new();

void chunk_mesh_cleanup(ChunkMesh *chunk_mesh);

Mesh build_chunk(ChunkMesh *chunk_mesh, const ChunkData *chunk_data);
