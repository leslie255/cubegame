#pragma once

#include "common.h"

#include "array.h"

#include <glad/glad.h>

DEF_ARRAY(VerticesArray, vertices_array, f32);
DEF_ARRAY(IndicesArray, indices_array, u32);

typedef struct vertex_attrib_format {
  i32 size;
  GLenum type;
  bool normalized;
  i32 stride;
  i32 offset;
} VertexAttribFormat;

DEF_ARRAY(VertexAttribFormatArray, vertex_attrib_format_array, VertexAttribFormat);

typedef struct mesh {
  IndicesArray indices;
  VerticesArray vertices;
  VertexAttribFormatArray vertex_attrib_pointers;
  GLuint vao;
  GLuint vbo;
  GLuint ebo;
} Mesh;

void fprint_mesh_data(FILE *out, Mesh mesh);

Mesh mesh_init( //
    VerticesArray vertices,
    IndicesArray indices,
    VertexAttribFormatArray vertex_attrib_pointers);

/// Update the new data to GPU.
void mesh_update(Mesh *mesh);

void mesh_cleanup(Mesh *mesh);

void mesh_draw(Mesh mesh);
