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

ChunkData *world_get_chunk_const(const WorldData *world, ivec3 chunk_id) {
  usize i_x = (usize)(chunk_id[0] + (WORLD_SIZE_X / 2));
  usize i_y = (usize)(chunk_id[1] + (WORLD_SIZE_Y / 2));
  usize i_z = (usize)(chunk_id[2] + (WORLD_SIZE_Z / 2));
  if (i_x > WORLD_SIZE_X && i_y > WORLD_SIZE_Y && i_z > WORLD_SIZE_Z)
    PANIC_PRINTF("Chunk {%d, %d, %d} does not exist\n", chunk_id[0], chunk_id[1], chunk_id[2]);
  return world->chunks[i_y][i_z][i_x];
}

i32 mod_with_sign(i32 x, i32 y) {
  i32 result = x % y;
  return result < 0 ? result + y : result;
}

/// Min is inclusive, max is exclusive.
static inline i32 is_between(i32 x, i32 min, i32 max) {
  return min <= x && x < max;
}

[[nodiscard]]
bool world_to_chunk_coord(ivec3 world_coord, ivec3 dest_chunk_id, ivec3 dest_chunk_local_coord) {
  if (!is_between(world_coord[0], -WORLD_SIZE_X * 32 / 2, WORLD_SIZE_X * 32 / 2) ||
      !is_between(world_coord[1], -WORLD_SIZE_Y * 32 / 2, WORLD_SIZE_Y * 32 / 2) ||
      !is_between(world_coord[1], -WORLD_SIZE_Z * 32 / 2, WORLD_SIZE_Z * 32 / 2)) {
    return false;
  }

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

  return true;
}

void chunk_to_world_coord(ivec3 chunk_id, ivec3 chunk_local_coord, ivec3 dest_world_coord) {
  dest_world_coord[0] = chunk_id[0] * 32 + chunk_local_coord[0];
  dest_world_coord[1] = chunk_id[1] * 32 + chunk_local_coord[1];
  dest_world_coord[2] = chunk_id[2] * 32 + chunk_local_coord[2];
}

void world_set_block(WorldData *world, ivec3 coord, BlockId block_id) {
  ivec3 chunk_id;
  ivec3 chunk_local_coord;
  if (!world_to_chunk_coord(coord, chunk_id, chunk_local_coord)) {
    printf(                                                           //
        "[DEBUG] `%s` called with out-of-world coord {%d, %d, %d}\n", //
        __func__, coord[0], coord[1], coord[2]);
    return;
  }
  ChunkData *chunk = *world_get_chunk(world, chunk_id);
  Tile *tile = &chunk->blocks[chunk_local_coord[1]][chunk_local_coord[2]][chunk_local_coord[0]];
  tile->id = block_id;
}

BlockId world_get_block(const WorldData *world, ivec3 coord, bool *is_in_world) {
  ivec3 chunk_id;
  ivec3 chunk_local_coord;
  if (world_to_chunk_coord(coord, chunk_id, chunk_local_coord)) {
    *is_in_world = true;
  } else {
    printf(                                                           //
        "[DEBUG] `%s` called with out-of-world coord {%d, %d, %d}\n", //
        __func__, coord[0], coord[1], coord[2]);
    *is_in_world = false;
    return BlockId_AIR;
  }
  ChunkData *chunk = world_get_chunk_const(world, chunk_id);
  return chunk->blocks[chunk_local_coord[1]][chunk_local_coord[2]][chunk_local_coord[0]].id;
}

static inline bool block_is(const WorldData *world, ivec3 pos, BlockId block_id) {
  bool is_in_range;
  auto block = world_get_block(world, pos, &is_in_range);
  if (is_in_range)
    return block == block_id;
  else
    return false;
}

static inline i32 rand_in(i32 min, i32 max) {
  return rand() % (max - min) + min;
}

static inline f32 wave_sampler(f32 amp, f32 scale, vec2 offset, ivec2 in) {
  f32 result = 0;
  result += sinf(((f32)in[0] + offset[0]) * (scale)) * (amp / 2.f);
  result += sinf(((f32)in[1] + offset[1]) * (scale)) * (amp / 2.f);
  return result;
}

static inline i32 terrain_height_at(ivec2 pos) {
  f32 out = 0.f;
  // Just some random values I cooked up.
  out += wave_sampler(1.3f, 0.51784f, (vec2){0.8f, 0.1f}, pos);
  out = MAX(out, wave_sampler(1.9f, 0.2472f, (vec2){2.8f, 7.4f}, pos));
  out -= wave_sampler(1.f, 0.397f, (vec2){6.3f, 1.4f}, pos);
  out += wave_sampler(2.f, 0.12f, (vec2){4.9f, 4.2f}, pos);
  out += wave_sampler(3.f, 0.1f, (vec2){3.1f, 6.4f}, pos);
  return (i32)(out) + 5;
}

static inline void gen_tree(WorldData *world, ivec2 pos_xz) {
  ivec3 pos = {pos_xz[0], terrain_height_at(pos_xz) + 1, pos_xz[1]};
  ivec3 pos_below = {pos[0], pos[1] - 1, pos[2]};
  if (!block_is(world, pos_below, BlockId_GRASS))
    return;

  i32 height = rand_in(2, 4);

  // Leaves.
  for (i32 y = pos[1] + height; y <= pos[1] + height + 3; ++y) {
    for (i32 z = pos[2] - 2; z <= pos[2] + 2; ++z) {
      for (i32 x = pos[0] - 2; x <= pos[0] + 2; ++x) {
        world_set_block(world, (ivec3){x, y, z}, BlockId_LEAVES);
      }
    }
  }
  for (i32 z = pos[2] - 1; z <= pos[2] + 1; ++z) {
    for (i32 x = pos[0] - 1; x <= pos[0] + 1; ++x) {
      world_set_block(world, (ivec3){x, pos[1] + height + 4, z}, BlockId_LEAVES);
    }
  }

  // Trunk.
  for (i32 y = pos[1]; y <= pos[1] + height + 2; ++y) {
    world_set_block(world, (ivec3){pos[0], y, pos[2]}, BlockId_LOG);
  }
}

static inline void gen_sand_patch(WorldData *world, ivec2 pos) {
  i32 radius = rand() % 20 + 6;
  for (i32 z = pos[1] - radius; z <= pos[1] + radius; ++z) {
    for (i32 x = pos[0] - radius; x <= pos[0] + radius; ++x) {
      auto y = terrain_height_at((ivec2){x, z});
      auto dz = z - pos[1];
      auto dx = x - pos[0];
      if (dz * dz + dx * dx < radius)
        world_set_block(world, (ivec3){x, y, z}, BlockId_SAND);
    }
  }
  if (rand() % 100 < 75) {
    pos[0] = pos[0] + rand() % radius * 2 - radius / 2;
    pos[1] = pos[1] + rand() % radius * 2 - radius / 2;
    gen_sand_patch(world, pos);
  }
}

constexpr i32 WORLD_SEED = 0;

constexpr ivec2 WORLDGEN_RANGE_X = {-WORLD_SIZE_X / 2 * 32, +WORLD_SIZE_X / 2 * 32};
constexpr ivec2 WORLDGEN_RANGE_Z = {-WORLD_SIZE_Z / 2 * 32, +WORLD_SIZE_Z / 2 * 32};

constexpr u32 N_TREES = 500;
constexpr u32 N_SAND_PATCHES = 70;

void world_gen(WorldData *world) {
  for (i32 z = WORLDGEN_RANGE_Z[0]; z < WORLDGEN_RANGE_Z[1]; ++z) {
    for (i32 x = WORLDGEN_RANGE_X[0]; x < WORLDGEN_RANGE_X[1]; ++x) {
      i32 terrain_height = terrain_height_at((ivec2){x, z});
      i32 y_stone_start = -32;
      i32 y_dirt_start = terrain_height - 3;
      i32 y_surface = terrain_height;
      for (i32 y = y_stone_start; y < y_dirt_start; ++y)
        world_set_block(world, (ivec3){x, y, z}, BlockId_STONE);
      for (i32 y = y_dirt_start; y < y_surface; ++y)
        world_set_block(world, (ivec3){x, y, z}, BlockId_DIRT);
      world_set_block(world, (ivec3){x, y_surface, z}, BlockId_GRASS);
    }
  }

  srand(WORLD_SEED);

  for (u32 i = 0; i < N_SAND_PATCHES; ++i) {
    ivec2 pos = {
        rand_in(WORLDGEN_RANGE_X[0], WORLDGEN_RANGE_X[1]),
        rand_in(WORLDGEN_RANGE_X[0], WORLDGEN_RANGE_X[1]),
    };
    gen_sand_patch(world, pos);
  }

  for (u32 i = 0; i < N_TREES; ++i) {
    ivec2 pos = {
        rand_in(WORLDGEN_RANGE_X[0], WORLDGEN_RANGE_X[1]),
        rand_in(WORLDGEN_RANGE_X[0], WORLDGEN_RANGE_X[1]),
    };
    gen_tree(world, pos);
  }
}
