#pragma once

#include "common.h"
#include "string.h"

typedef enum cube_direction : usize {
  CubeDirection_North = 0,
  CubeDirection_South,
  CubeDirection_East,
  CubeDirection_West,
  CubeDirection_Up,
  CubeDirection_Down,
} CubeDirection;

/// A block texture in the form of coordinate in the texture atlas.
typedef struct block_texture_coord {
  /// Coordinate that this texture is on in the texture atlas.
  u16 x;
  /// Coordinate that this texture is on in the texture atlas.
  u16 y;
} BlockTexture;

typedef struct block_model {
  /// Indexed by `CubeDirection`.
  BlockTexture faces[6];
} BlockModel;

typedef enum block_transparency : u8 {
  BlockTransparency_Solid,
  BlockTransparency_Transparent,
  BlockTransparency_Air,
} BlockTransparency;

typedef enum block_id : u16 {
  BlockId_AIR = 0,
  BlockId_STONE,
  BlockId_DIRT,
  BlockId_GRASS,
  BlockId_LOG,
  BlockId_LEAVES,
  BlockId_SAND,
  BlockId_CHERRY_LOG,
  BlockId_CHERRY_LEAVES,
  BlockId_WATER,
  BlockId_TEST_BLOCK = 255,
} BlockId;

typedef struct block_registry {
  StaticString name;
  BlockTransparency transparency;
  BlockModel model;
} BlockRegistry;

static BlockRegistry BLOCK_REGISTRIES[256] = {
    [BlockId_AIR] = {.name = STATIC_STRING("air"), .transparency = BlockTransparency_Air, .model = {}},
    [BlockId_STONE] =
        {
            .name = STATIC_STRING("stone"),
            .transparency = BlockTransparency_Solid,
            .model = {{
                [CubeDirection_North] = {1, 0},
                [CubeDirection_South] = {1, 0},
                [CubeDirection_East] = {1, 0},
                [CubeDirection_West] = {1, 0},
                [CubeDirection_Up] = {1, 0},
                [CubeDirection_Down] = {1, 0},
            }},
        },
    [BlockId_DIRT] = {.name = STATIC_STRING("dirt"),
                      .transparency = BlockTransparency_Solid,
                      .model = {{
                          [CubeDirection_North] = {2, 0},
                          [CubeDirection_South] = {2, 0},
                          [CubeDirection_East] = {2, 0},
                          [CubeDirection_West] = {2, 0},
                          [CubeDirection_Up] = {2, 0},
                          [CubeDirection_Down] = {2, 0},
                      }}},
    [BlockId_GRASS] = {.name = STATIC_STRING("grass"),
                       .transparency = BlockTransparency_Solid,
                       .model = {{
                           [CubeDirection_North] = {3, 0},
                           [CubeDirection_South] = {3, 0},
                           [CubeDirection_East] = {3, 0},
                           [CubeDirection_West] = {3, 0},
                           [CubeDirection_Up] = {4, 0},
                           [CubeDirection_Down] = {2, 0},
                       }}},
    [BlockId_LOG] = {.name = STATIC_STRING("log"),
                     .transparency = BlockTransparency_Solid,
                     .model = {{
                         [CubeDirection_North] = {5, 0},
                         [CubeDirection_South] = {5, 0},
                         [CubeDirection_East] = {5, 0},
                         [CubeDirection_West] = {5, 0},
                         [CubeDirection_Up] = {6, 0},
                         [CubeDirection_Down] = {6, 0},
                     }}},
    [BlockId_LEAVES] = {.name = STATIC_STRING("leaves"),
                        .transparency = BlockTransparency_Transparent,
                        .model = {{
                            [CubeDirection_North] = {7, 0},
                            [CubeDirection_South] = {7, 0},
                            [CubeDirection_East] = {7, 0},
                            [CubeDirection_West] = {7, 0},
                            [CubeDirection_Up] = {7, 0},
                            [CubeDirection_Down] = {7, 0},
                        }}},
    [BlockId_SAND] = {.name = STATIC_STRING("sand"),
                      .transparency = BlockTransparency_Solid,
                      .model = {{
                          [CubeDirection_North] = {8, 0},
                          [CubeDirection_South] = {8, 0},
                          [CubeDirection_East] = {8, 0},
                          [CubeDirection_West] = {8, 0},
                          [CubeDirection_Up] = {8, 0},
                          [CubeDirection_Down] = {8, 0},
                      }}},
    [BlockId_CHERRY_LOG] = {.name = STATIC_STRING("cherry_log"),
                     .transparency = BlockTransparency_Solid,
                     .model = {{
                         [CubeDirection_North] = {9, 0},
                         [CubeDirection_South] = {9, 0},
                         [CubeDirection_East] = {9, 0},
                         [CubeDirection_West] = {9, 0},
                         [CubeDirection_Up] = {10, 0},
                         [CubeDirection_Down] = {10, 0},
                     }}},
    [BlockId_CHERRY_LEAVES] = {.name = STATIC_STRING("cherry_leaves"),
                        .transparency = BlockTransparency_Transparent,
                        .model = {{
                            [CubeDirection_North] = {11, 0},
                            [CubeDirection_South] = {11, 0},
                            [CubeDirection_East] = {11, 0},
                            [CubeDirection_West] = {11, 0},
                            [CubeDirection_Up] = {11, 0},
                            [CubeDirection_Down] = {11, 0},
                        }}},
    [BlockId_WATER] = {.name = STATIC_STRING("water"),
                            .transparency = BlockTransparency_Solid,
                            .model = {{
                                [CubeDirection_North] = {12, 0},
                                [CubeDirection_South] = {12, 0},
                                [CubeDirection_East] = {12, 0},
                                [CubeDirection_West] = {12, 0},
                                [CubeDirection_Up] = {12, 0},
                                [CubeDirection_Down] = {12, 0},
                            }}},
    [BlockId_TEST_BLOCK] = {.name = STATIC_STRING("test_block"),
                            .transparency = BlockTransparency_Solid,
                            .model = {{
                                [CubeDirection_North] = {15, 15},
                                [CubeDirection_South] = {15, 15},
                                [CubeDirection_East] = {15, 15},
                                [CubeDirection_West] = {15, 15},
                                [CubeDirection_Up] = {15, 15},
                                [CubeDirection_Down] = {15, 15},
                            }}},
};
