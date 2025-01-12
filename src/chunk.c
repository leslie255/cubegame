#include "chunk.h"

ChunkData *chunk_alloc() {
  return xalloc(ChunkData, 1);
}

void chunk_cleanup(ChunkData **chunk) {
  xfree(*chunk);
  if (IS_DEBUG_MODE)
    *chunk = nullptr;
}

ChunkMesh chunk_mesh_new() {
  VerticesArray vertices = {};
  IndicesArray indices = {};
  MARK_USED(vertices);
  MARK_USED(indices);
  TODO_FUNCTION();
}

void chunk_mesh_cleanup(ChunkMesh *chunk_mesh) {
  MARK_USED(chunk_mesh);
  TODO_FUNCTION();
}

Mesh build_chunk(ChunkMesh *chunk_mesh, const ChunkData *chunk_data) {
  MARK_USED(chunk_mesh);
  MARK_USED(chunk_data);
  TODO_FUNCTION();
}
