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
  Mesh data;
  GLuint vao;
  GLuint vbo;
  GLuint ebo;
} LoadedMesh;

LoadedMesh load_mesh(Mesh mesh);

void reload_mesh(LoadedMesh *loaded_mesh);

void unload_mesh(LoadedMesh *loaded_mesh);
