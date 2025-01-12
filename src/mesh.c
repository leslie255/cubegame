#include "mesh.h"

Mesh mesh_init( //
    VerticesArray vertices,
    IndicesArray indices,
    VertexAttribFormatArray vertex_attrib_pointers) {
  Mesh mesh = {
      .vertices = vertices,
      .indices = indices,
      .vertex_attrib_pointers = vertex_attrib_pointers,
  };

  mesh_update(&mesh);

  return mesh;
}

void mesh_update(Mesh *mesh) {
  if (mesh->vao == 0) {
    DEBUG_ASSERT(mesh->vbo == 0 && mesh->ebo == 0);
    if (mesh->vertices.length != 0 && mesh->indices.length != 0) {
      glGenVertexArrays(1, &mesh->vao);
      glGenBuffers(1, &mesh->ebo);
      glGenBuffers(1, &mesh->vbo);
    } else {
      return;
    }
  }

  glBindVertexArray(mesh->vao);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->ebo);
  glBufferData(                                         //
      GL_ELEMENT_ARRAY_BUFFER,                          //
      (GLsizeiptr)(mesh->indices.length * sizeof(u32)), //
      mesh->indices.items,                              //
      GL_DYNAMIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, mesh->vbo);
  glBufferData(                                          //
      GL_ARRAY_BUFFER,                                   //
      (GLsizeiptr)(mesh->vertices.length * sizeof(f32)), //
      mesh->vertices.items,                              //
      GL_DYNAMIC_DRAW);

  for (usize i = 0; i < mesh->vertex_attrib_pointers.length; ++i) {
    auto p = mesh->vertex_attrib_pointers.items[i];
    glVertexAttribPointer( //
        (GLuint)i,         //
        p.size,            //
        p.type,            //
        p.normalized,      //
        p.stride,          //
        (void *)(usize)p.offset);
    glEnableVertexAttribArray((GLuint)i);
  }
}

void mesh_cleanup(Mesh *mesh) {
  glDeleteVertexArrays(1, &mesh->vao);
  glDeleteBuffers(1, &mesh->vbo);
  glDeleteBuffers(1, &mesh->ebo);
  vertices_array_cleanup(&mesh->vertices);
  indices_array_cleanup(&mesh->indices);
  vertex_attrib_format_array_cleanup(&mesh->vertex_attrib_pointers);
  if (IS_DEBUG_MODE) {
    mesh->vertices = (VerticesArray){};
    mesh->indices = (IndicesArray){};
    mesh->vertex_attrib_pointers = (VertexAttribFormatArray){};
    mesh->vao = 0;
    mesh->vbo = 0;
    mesh->ebo = 0;
  }
}

void mesh_draw(Mesh mesh) {
  glBindVertexArray(mesh.vao);
  glDrawElements(GL_TRIANGLES, (GLsizei)mesh.indices.length, GL_UNSIGNED_INT, nullptr);
}
