#include "chunk.h"

ChunkData *chunk_alloc() {
  return xalloc(ChunkData, 1);
}
