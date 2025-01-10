#include "block.h"

bool block_is_solid(BlockId block_id) {
  switch (block_id) {
  case BLOCKID_AIR:
    return false;
  case BLOCKID_STONE:
    return true;
  case BLOCKID_DIRT:
    return true;
  case BLOCKID_GRASS:
    return true;
  case BLOCKID_TEST:
    return true;
  }
}

void block_texture_atlast_coord(BlockId block_id, u32 *dest_x, u32 *dest_y) {
  u8 index = (block_id < 256) ? (u8)block_id : 0;
  u32 block_x = index % 16;
  u32 block_y = index / 16;
  *dest_x = block_x * 16;
  *dest_y = block_y * 16;
}
