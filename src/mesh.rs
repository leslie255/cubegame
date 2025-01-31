use std::ops::Add;

use glium::Surface as _;

pub fn default_3d_draw_parameters() -> glium::DrawParameters<'static> {
    glium::DrawParameters {
        depth: glium::Depth {
            test: glium::DepthTest::IfLess,
            write: true,
            ..Default::default()
        },
        ..Default::default()
    }
}

pub fn default_2d_draw_parameters() -> glium::DrawParameters<'static> {
    glium::DrawParameters {
        depth: glium::Depth {
            test: glium::DepthTest::Overwrite,
            write: false,
            ..Default::default()
        },
        blend: glium::draw_parameters::Blend::alpha_blending(),
        ..Default::default()
    }
}

#[derive(Debug)]
pub struct Mesh<V: Copy + glium::Vertex, I: Copy + glium::index::Index = u32> {
    vertex_buffer: Option<Box<glium::VertexBuffer<V>>>,
    index_buffer: Option<Box<glium::IndexBuffer<I>>>,
    vertices: Vec<V>,
    indices: Vec<I>,
}

impl<V: Copy + glium::Vertex, I: Copy + glium::index::Index> Mesh<V, I> {
    pub fn new() -> Self {
        Self {
            vertex_buffer: None,
            index_buffer: None,
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn with_data(
        display: &impl glium::backend::Facade,
        vertices: Vec<V>,
        indices: Vec<I>,
    ) -> Self {
        let mut self_ = Self::new();
        *self_.vertices_mut() = vertices;
        *self_.indices_mut() = indices;
        self_.update(display);
        self_
    }

    /// Send the vertices and indices to the GPU.
    #[track_caller]
    pub fn update(&mut self, display: &impl glium::backend::Facade) {
        if self.vertices.is_empty() || self.indices.is_empty() {
            self.vertex_buffer = None;
            self.index_buffer = None;
            return;
        }
        self.vertex_buffer = Some(
            glium::VertexBuffer::dynamic(display, self.vertices())
                .unwrap()
                .into(),
        );
        self.index_buffer = Some(
            glium::IndexBuffer::dynamic(
                display,
                glium::index::PrimitiveType::TrianglesList,
                self.indices(),
            )
            .unwrap()
            .into(),
        );
    }

    #[track_caller]
    pub fn draw<U: glium::uniforms::Uniforms>(
        &self,
        frame: &mut glium::Frame,
        uniforms: U,
        shader: &glium::Program,
        draw_parameters: &glium::DrawParameters,
    ) {
        let Some(vertex_buffer) = self.vertex_buffer.as_ref() else {
            return;
        };
        let Some(index_buffer) = self.index_buffer.as_ref() else {
            return;
        };
        frame
            .draw(
                vertex_buffer.as_ref(),
                index_buffer.as_ref(),
                shader,
                &uniforms,
                draw_parameters,
            )
            .unwrap();
    }

    pub fn vertices(&self) -> &[V] {
        &self.vertices
    }

    /// Remember to call `update`!
    pub fn vertices_mut(&mut self) -> &mut Vec<V> {
        &mut self.vertices
    }

    pub fn indices(&self) -> &[I] {
        &self.indices
    }

    /// Remember to call `update`!
    pub fn indices_mut(&mut self) -> &mut Vec<I> {
        &mut self.indices
    }

    /// Don't forget to `update`!
    pub fn append(&mut self, vertices: &[V], indices: &[I])
    where
        I: Add<u32, Output = I>,
    {
        let old_length = self.vertices().len() as u32;
        self.vertices_mut().extend_from_slice(vertices);
        self.indices_mut()
            .extend(indices.iter().map(|&i| i + old_length));
    }
}

impl<T: Copy + glium::Vertex> Default for Mesh<T> {
    fn default() -> Self {
        Self::new()
    }
}
