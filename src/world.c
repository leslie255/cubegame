#include "world.h"

WorldData *world_alloc() {
  auto world = xalloc(WorldData, 1);
  for (i32 y = -(WORLD_SIZE_Y / 2); y < (WORLD_SIZE_Y / 2); ++y) {
    for (i32 z = -(WORLD_SIZE_Z / 2); z < (WORLD_SIZE_Z / 2); ++z) {
      for (i32 x = -(WORLD_SIZE_X / 2); x < (WORLD_SIZE_X / 2); ++x) {
        ivec3 chunk_id = {x, y, z};
        ChunkData *chunk = chunk_alloc();
        memset(chunk, 0, sizeof(ChunkData));
        *world_get_chunk(world, chunk_id) = chunk;
      }
    }
  }
  return world;
}

void world_cleanup(WorldData **world) {
  for (i32 y = -(WORLD_SIZE_Y / 2); y < (WORLD_SIZE_Y / 2); ++y)
    for (i32 z = -(WORLD_SIZE_Z / 2); z < (WORLD_SIZE_Z / 2); ++z)
      for (i32 x = -(WORLD_SIZE_X / 2); x < (WORLD_SIZE_X / 2); ++x)
        chunk_cleanup(world_get_chunk(*world, (ivec3){x, y, z}));
  xfree(*world);
}

ChunkData **world_get_chunk(WorldData *world, ivec3 chunk_id) {
  usize i_x = (usize)(chunk_id[0] + (WORLD_SIZE_X / 2));
  usize i_y = (usize)(chunk_id[1] + (WORLD_SIZE_Y / 2));
  usize i_z = (usize)(chunk_id[2] + (WORLD_SIZE_Z / 2));
  if (i_x > WORLD_SIZE_X && i_y > WORLD_SIZE_Y && i_z > WORLD_SIZE_Z)
    PANIC_PRINTF("Chunk {%d, %d, %d} does not exist\n", chunk_id[0], chunk_id[1], chunk_id[2]);
  return &world->chunks[i_y][i_z][i_x];
}

i32 mod_with_sign(i32 x, i32 y) {
  i32 result = x % y;
  return result < 0 ? result + y : result;
}

void world_to_chunk_coord(ivec3 world_coord, ivec3 dest_chunk_id, ivec3 dest_chunk_local_coord) {
  dest_chunk_id[0] = world_coord[0] / 32;
  dest_chunk_id[1] = world_coord[1] / 32;
  dest_chunk_id[2] = world_coord[2] / 32;

  if (world_coord[0] < 0 && world_coord[0] % 32 != 0)
    --dest_chunk_id[0];
  if (world_coord[1] < 0 && world_coord[1] % 32 != 0)
    --dest_chunk_id[1];
  if (world_coord[2] < 0 && world_coord[2] % 32 != 0)
    --dest_chunk_id[2];

  dest_chunk_local_coord[0] = (world_coord[0] % 32 + 32) % 32;
  dest_chunk_local_coord[1] = (world_coord[1] % 32 + 32) % 32;
  dest_chunk_local_coord[2] = (world_coord[2] % 32 + 32) % 32;
}

void chunk_to_world_coord(ivec3 chunk_id, ivec3 chunk_local_coord, ivec3 dest_world_coord) {
  dest_world_coord[0] = chunk_id[0] * 32 + chunk_local_coord[0];
  dest_world_coord[1] = chunk_id[1] * 32 + chunk_local_coord[1];
  dest_world_coord[2] = chunk_id[2] * 32 + chunk_local_coord[2];
}

void world_set_block(WorldData *world, ivec3 coord, BlockId block_id) {
  ivec3 chunk_id;
  ivec3 chunk_local_coord;
  world_to_chunk_coord(coord, chunk_id, chunk_local_coord);
  ChunkData *chunk = *world_get_chunk(world, chunk_id);
  Tile *tile = &chunk->blocks[chunk_local_coord[1]][chunk_local_coord[2]][chunk_local_coord[0]];
  tile->id = block_id;
}

void world_gen(WorldData *world) {
  // world_set_block(world, (ivec3){0, 0, 0}, BlockId_GRASS);
  // world_set_block(world, (ivec3){31, 0, 0}, BlockId_GRASS);
  // world_set_block(world, (ivec3){0, 0, 31}, BlockId_GRASS);
  // world_set_block(world, (ivec3){31, 0, 31}, BlockId_GRASS);
  // world_set_block(world, (ivec3){0, 31, 0}, BlockId_GRASS);
  // world_set_block(world, (ivec3){31, 31, 0}, BlockId_GRASS);
  // world_set_block(world, (ivec3){0, 31, 31}, BlockId_GRASS);
  // world_set_block(world, (ivec3){31, 31, 31}, BlockId_GRASS);
  // world_set_block(world, (ivec3){32, 0, 32}, BlockId_GRASS);
  // world_set_block(world, (ivec3){63, 0, 32}, BlockId_GRASS);
  // world_set_block(world, (ivec3){32, 0, 63}, BlockId_GRASS);
  // world_set_block(world, (ivec3){63, 0, 63}, BlockId_GRASS);
  // world_set_block(world, (ivec3){32, 31, 32}, BlockId_GRASS);
  // world_set_block(world, (ivec3){63, 31, 32}, BlockId_GRASS);
  // world_set_block(world, (ivec3){32, 31, 63}, BlockId_GRASS);
  // world_set_block(world, (ivec3){63, 31, 63}, BlockId_GRASS);
  for (i32 z = -256; z < 256; ++z) {
    for (i32 x = -265; x < 256; ++x) {
      world_set_block(world, (ivec3){x, 0, z}, BlockId_STONE);
      world_set_block(world, (ivec3){x, 1, z}, BlockId_STONE);
      world_set_block(world, (ivec3){x, 2, z}, BlockId_STONE);
      world_set_block(world, (ivec3){x, 3, z}, BlockId_DIRT);
      world_set_block(world, (ivec3){x, 4, z}, BlockId_DIRT);
      world_set_block(world, (ivec3){x, 5, z}, BlockId_DIRT);
      world_set_block(world, (ivec3){x, 6, z}, BlockId_GRASS);
    }
  }
}
