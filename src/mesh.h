#pragma once

#include "common.h"

#include "array.h"

#include <glad/glad.h>

DEF_ARRAY(VerticesArray, vertices_array, f32);
DEF_ARRAY(IndicesArray, indices_array, u32);

typedef struct mesh {
  IndicesArray indices;
  VerticesArray vertices;
} Mesh;

typedef struct loaded_mesh {
  Mesh mesh;
  GLuint vao;
  GLuint vbo;
  GLuint ebo;
  /// Number of vertex attribute pointers.
  usize n_attrib_pointers;
} LoadedMesh;

typedef struct vertex_attrib_format {
  i32 size;
  GLenum type;
  bool normalized;
  i32 stride;
  i32 offset;
} VertexAttribFormat;

LoadedMesh load_mesh(Mesh mesh, usize n_attrib_pointers, const VertexAttribFormat attrib_pointers[n_attrib_pointers]);

void mesh_draw(LoadedMesh mesh);

void reload_mesh(LoadedMesh *loaded_mesh);

void loaded_mesh_cleanup(LoadedMesh *loaded_mesh);
