#include "mesh.h"

#include "debug_utils.h"

LoadedMesh load_mesh(Mesh mesh, usize n_attrib_pointers, const VertexAttribFormat attrib_pointers[n_attrib_pointers]) {
  LoadedMesh loaded_mesh = {.mesh = mesh};
  glGenVertexArrays(1, &loaded_mesh.vao);
  glBindVertexArray(loaded_mesh.vao);

  glGenBuffers(1, &loaded_mesh.ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, loaded_mesh.ebo);
  glBufferData(                                        //
      GL_ELEMENT_ARRAY_BUFFER,                         //
      (GLsizeiptr)(mesh.indices.length * sizeof(u32)), //
      mesh.indices.items,                              //
      GL_STATIC_DRAW);
  glGenBuffers(1, &loaded_mesh.vbo);
  glBindBuffer(GL_ARRAY_BUFFER, loaded_mesh.vbo);
  glBufferData(                                         //
      GL_ARRAY_BUFFER,                                  //
      (GLsizeiptr)(mesh.vertices.length * sizeof(f32)), //
      mesh.vertices.items,                              //
      GL_STATIC_DRAW);
  for (usize i = 0; i < n_attrib_pointers; ++i) {
    auto attrib_pointer = attrib_pointers[i];
    glVertexAttribPointer(                    //
        DBG_PRINT((GLuint)i),                 //
        DBG_PRINT(attrib_pointer.size),       //
        DBG_PRINT_HEX(attrib_pointer.type),   //
        DBG_PRINT(attrib_pointer.normalized), //
        DBG_PRINT(attrib_pointer.stride),     //
        (void *)DBG_PRINT((usize)attrib_pointer.offset));
    glEnableVertexAttribArray((GLuint)i);
  }

  return loaded_mesh;
}

void mesh_draw(LoadedMesh mesh) {
  glBindVertexArray(mesh.vao);
  glDrawElements(GL_TRIANGLES, (GLsizei)mesh.mesh.vertices.length, GL_UNSIGNED_INT, nullptr);
}

void reload_mesh(LoadedMesh *loaded_mesh) {
  MARK_USED(loaded_mesh);
  TODO_FUNCTION();
}

void loaded_mesh_cleanup(LoadedMesh *loaded_mesh) {
  glDeleteVertexArrays(1, &loaded_mesh->vao);
  glDeleteBuffers(1, &loaded_mesh->vbo);
  glDeleteBuffers(1, &loaded_mesh->ebo);
  vertices_array_cleanup(&loaded_mesh->mesh.vertices);
  indices_array_cleanup(&loaded_mesh->mesh.indices);
}
