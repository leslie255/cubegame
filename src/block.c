#include "block.h"

bool block_is_solid(BlockId block_id) {
  switch (block_id) {
  case BLOCKID_NULL:
    return false;
  case BLOCKID_AIR:
    return false;
  case BLOCKID_STONE:
    return true;
  case BLOCKID_DIRT:
    return true;
  case BLOCKID_GRASS:
    return true;
  }
}
