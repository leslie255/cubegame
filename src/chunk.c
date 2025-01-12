#include "chunk.h"

static constexpr GLfloat CUBE_VERTICES[] = {
    // North
    0.f, 0.f, 0.f, 1.0f, 1.0f, // A 0
    1.f, 0.f, 0.f, 0.0f, 1.0f, // B 1
    1.f, 1.f, 0.f, 0.0f, 0.0f, // C 2
    0.f, 1.f, 0.f, 1.0f, 0.0f, // D 3
    // South
    0.f, 0.f, 1.f, 0.0f, 1.0f, // E 4
    1.f, 0.f, 1.f, 1.0f, 1.0f, // F 5
    1.f, 1.f, 1.f, 1.0f, 0.0f, // G 6
    0.f, 1.f, 1.f, 0.0f, 0.0f, // H 7
    // East
    0.f, 1.f, 0.f, 0.0f, 0.0f, // D 8
    0.f, 0.f, 0.f, 0.0f, 1.0f, // A 9
    0.f, 0.f, 1.f, 1.0f, 1.0f, // E 10
    0.f, 1.f, 1.f, 1.0f, 0.0f, // H 11
    // West
    1.f, 0.f, 0.f, 1.0f, 1.0f, // B 12
    1.f, 1.f, 0.f, 1.0f, 0.0f, // C 13
    1.f, 1.f, 1.f, 0.0f, 0.0f, // G 14
    1.f, 0.f, 1.f, 0.0f, 1.0f, // F 15
    // Up
    0.f, 0.f, 0.f, 0.0f, 1.0f, // A 16
    1.f, 0.f, 0.f, 1.0f, 1.0f, // B 17
    1.f, 0.f, 1.f, 1.0f, 0.0f, // F 18
    0.f, 0.f, 1.f, 0.0f, 0.0f, // E 19
    // Down
    1.f, 1.f, 0.f, 0.0f, 1.0f, // C 20
    0.f, 1.f, 0.f, 1.0f, 1.0f, // D 21
    0.f, 1.f, 1.f, 1.0f, 0.0f, // H 22
    1.f, 1.f, 1.f, 0.0f, 0.0f, // G 23
};

static constexpr VertexAttribFormat CUBE_VERTICES_FORMAT[] = {
    {.size = 3, .type = GL_FLOAT, .normalized = false, .stride = sizeof(f32[5]), .offset = 0},
    {.size = 2, .type = GL_FLOAT, .normalized = false, .stride = sizeof(f32[5]), .offset = sizeof(f32[3])},
};

static constexpr u32 CUBE_INDICES[6][6] = {
    // North
    {0, 3, 2, 2, 1, 0},
    // South
    {4, 5, 6, 6, 7, 4},
    // East
    {11, 8, 9, 9, 10, 11},
    // West
    {12, 13, 14, 14, 15, 12},
    // Down
    {20, 21, 22, 22, 23, 20},
    // Up
    {16, 17, 18, 18, 19, 16},
};

ChunkData *chunk_alloc() {
  return xalloc(ChunkData, 1);
}

void chunk_cleanup(ChunkData **chunk) {
  xfree(*chunk);
  if (IS_DEBUG_MODE)
    *chunk = nullptr;
}

ChunkBuilder chunk_builder_new(Texture texture_atlas) {
  return (ChunkBuilder){
      .quads = {},
      .texture_atlas = texture_atlas,
  };
}

void chunk_builder_cleanup(ChunkBuilder *cb) {
  for (usize i = 0; i < 6; ++i)
    quad_array_cleanup(&cb->quads[i]);
}

static inline void add_face(ChunkBuilder *chunk_builder, Mesh *mesh, Quad quad, const u32 face_indices[6]) {
  vec2 texture_coord = {
      (f32)quad.texture.x * 16.f / (f32)chunk_builder->texture_atlas.width,
      (f32)quad.texture.y * 16.f / (f32)chunk_builder->texture_atlas.height,
  };
  auto old_length = (u32)mesh->vertices.length / 5;
  for (usize i = 0; i < 6; ++i) {
    vertices_array_push(&mesh->vertices, CUBE_VERTICES[face_indices[i] * 5 + 0] + quad.coord[0]);
    vertices_array_push(&mesh->vertices, CUBE_VERTICES[face_indices[i] * 5 + 1] + quad.coord[1]);
    vertices_array_push(&mesh->vertices, CUBE_VERTICES[face_indices[i] * 5 + 2] + quad.coord[2]);
    vertices_array_push(&mesh->vertices, CUBE_VERTICES[face_indices[i] * 5 + 3] / 16.f + texture_coord[0]);
    vertices_array_push(&mesh->vertices, CUBE_VERTICES[face_indices[i] * 5 + 4] / 16.f + texture_coord[1]);
  }
  indices_array_push(&mesh->indices, old_length + 0);
  indices_array_push(&mesh->indices, old_length + 1);
  indices_array_push(&mesh->indices, old_length + 2);
  indices_array_push(&mesh->indices, old_length + 3);
  indices_array_push(&mesh->indices, old_length + 4);
  indices_array_push(&mesh->indices, old_length + 5);
}

/// Assembles quads of one direction into a mesh.
static inline void assemble_quads(ChunkBuilder *chunk_builder, Mesh *mesh, QuadArray quads, const u32 face_indices[6]) {
  for (usize i_quad = 0; i_quad < quads.length; ++i_quad) {
    add_face(chunk_builder, mesh, quads.items[i_quad], face_indices);
  }
}

static inline bool has_solid_neighbor(const ChunkData *chunk_data, u32 x, u32 y, u32 z, CubeDirection direction) {
  u32 neighbor_x = x;
  u32 neighbor_y = y;
  u32 neighbor_z = z;
  switch (direction) {
    // clang-format off
  case CubeDirection_North: { if (z == 0 ) { return false; } else { neighbor_z = z - 1; }} break;
  case CubeDirection_South: { if (z == 31) { return false; } else { neighbor_z = z + 1; }} break;
  case CubeDirection_East:  { if (x == 0 ) { return false; } else { neighbor_x = x - 1; }} break;
  case CubeDirection_West:  { if (x == 31) { return false; } else { neighbor_x = x + 1; }} break;
  case CubeDirection_Up:    { if (y == 31) { return false; } else { neighbor_y = y + 1; }} break;
  case CubeDirection_Down:  { if (y == 0 ) { return false; } else { neighbor_y = y - 1; }} break;
    // clang-format on
  }
  auto block_id = chunk_data->blocks[neighbor_y][neighbor_z][neighbor_x].id;
  auto block_transparency = BLOCK_REGISTRIES[block_id].transparency;
  return block_transparency == BlockTransparency_Solid;
}

void build_chunk(ChunkBuilder *chunk_builder, Mesh *mesh, const ChunkData *chunk_data, Texture texture) {
  chunk_builder->texture_atlas = texture;
  for (u32 y = 0; y < 32; ++y) {
    for (u32 z = 0; z < 32; ++z) {
      for (u32 x = 0; x < 32; ++x) {
        auto block_id = chunk_data->blocks[y][z][x].id;
        auto block_registry = BLOCK_REGISTRIES[block_id];
        if (block_registry.transparency == BlockTransparency_Air)
          continue;
        Quad quad = {
            .coord = {(f32)x, (f32)y, (f32)z},
            .size = {1.f, 1.f},
        };
        for (usize i = 0; i < 6; ++i) {
          auto face = (CubeDirection)i;
          if (!has_solid_neighbor(chunk_data, x, y, z, face)) {
            Quad quad_ = quad;
            quad_.texture = block_registry.model.faces[face];
            quad_array_push(&chunk_builder->quads[i], quad_);
          }
        }
      }
    }
  }

  vertices_array_clear(&mesh->vertices);
  indices_array_clear(&mesh->indices);
  vertex_attrib_format_array_clear(&mesh->vertex_attrib_pointers);
  vertex_attrib_format_array_append(&mesh->vertex_attrib_pointers, ARR_ARG(CUBE_VERTICES_FORMAT));

  for (usize i = 0; i < 6; ++i) {
    assemble_quads(chunk_builder, mesh, chunk_builder->quads[i], CUBE_INDICES[i]);
  }

  mesh_update(mesh);

  for (usize i = 0; i < 6; ++i) {
    quad_array_clear(&chunk_builder->quads[i]);
  }
}
