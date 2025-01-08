#include "chunk.h"

ChunkData *chunk_alloc() {
  return xalloc(ChunkData, 1);
}

void chunk_cleanup(ChunkData **chunk) {
  xfree(*chunk);
  if (IS_DEBUG_MODE)
    *chunk = nullptr;
}
