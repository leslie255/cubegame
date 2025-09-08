//! Utilities for drawing.

use std::ops::Add;

use glium::Surface as _;

use cgmath::*;

use crate::utils::MainThreadOnly;

pub fn matrix4_to_array<T>(matrix: Matrix4<T>) -> [[T; 4]; 4] {
    matrix.into()
}

pub fn matrix3_to_array<T>(matrix: Matrix3<T>) -> [[T; 3]; 3] {
    matrix.into()
}

pub fn texture_sampler(texture: &glium::Texture2d) -> glium::uniforms::Sampler<'_, glium::Texture2d> {
    texture
        .sampled()
        .minify_filter(glium::uniforms::MinifySamplerFilter::NearestMipmapLinear)
        .magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest)
        .wrap_function(glium::uniforms::SamplerWrapFunction::Clamp)
}

pub fn default_3d_draw_parameters() -> glium::DrawParameters<'static> {
    glium::DrawParameters {
        depth: glium::Depth {
            test: glium::DepthTest::IfLess,
            write: true,
            ..Default::default()
        },
        backface_culling: glium::BackfaceCullingMode::CullClockwise,
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
        backface_culling: glium::BackfaceCullingMode::CullCounterClockwise,
        blend: glium::draw_parameters::Blend::alpha_blending(),
        ..Default::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quad2 {
    pub left: f32,
    pub right: f32,
    pub bottom: f32,
    pub top: f32,
}

impl Quad2 {
    pub fn width(self) -> f32 {
        (self.right - self.left).abs()
    }

    pub fn height(self) -> f32 {
        (self.top - self.bottom).abs()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1. }
    }

    pub const fn into_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }

    pub const fn into_vec4(self) -> Vector4<f32> {
        vec4(self.r, self.g, self.b, self.a)
    }
}

impl From<Color> for [f32; 4] {
    fn from(color: Color) -> Self {
        color.into_array()
    }
}

impl From<Color> for Vector4<f32> {
    fn from(color: Color) -> Self {
        color.into_vec4()
    }
}

/// A single threaded mesh.
/// For `Send` and `Sync` mesh, use `SharedMesh`.
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
            glium::VertexBuffer::immutable(display, self.vertices())
                .unwrap()
                .into(),
        );
        self.index_buffer = Some(
            glium::IndexBuffer::immutable(
                display,
                glium::index::PrimitiveType::TrianglesList,
                self.indices(),
            )
            .unwrap()
            .into(),
        );
    }

    #[track_caller]
    pub fn draw(
        &self,
        frame: &mut glium::Frame,
        uniforms: impl glium::uniforms::Uniforms,
        shader: &glium::Program,
        draw_parameters: &glium::DrawParameters,
    ) {
        let Some(vertex_buffer) = self.vertex_buffer.as_deref() else {
            return;
        };
        let Some(index_buffer) = self.index_buffer.as_deref() else {
            return;
        };
        frame
            .draw(
                vertex_buffer,
                index_buffer,
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

impl<V: Copy + glium::Vertex, I: Copy + glium::index::Index> Default for Mesh<V, I> {
    fn default() -> Self {
        Self::new()
    }
}

/// A mesh that is `Send` and `Sync`.
/// However only the main thread can make draw calls.
pub struct SharedMesh<V: Copy + glium::Vertex, I: Copy + glium::index::Index = u32> {
    vertex_buffer: MainThreadOnly<Option<Box<glium::VertexBuffer<V>>>>,
    index_buffer: MainThreadOnly<Option<Box<glium::IndexBuffer<I>>>>,
    vertices: Vec<V>,
    indices: Vec<I>,
    /// Whether the GL vertex and index buffer reflects the up-to-date content of the vertices/indices.
    /// `true` for not updated.
    needs_update: bool,
}

impl<V: Copy + glium::Vertex + std::fmt::Debug, I: Copy + glium::index::Index + std::fmt::Debug>
    std::fmt::Debug for SharedMesh<V, I>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedMesh")
            .field("vertices", &self.vertices)
            .field("indices", &self.indices)
            .field("needs_update", &self.needs_update)
            .finish_non_exhaustive()
    }
}

impl<V: Copy + glium::Vertex, I: Copy + glium::index::Index> SharedMesh<V, I> {
    pub fn new() -> Self {
        Self {
            vertex_buffer: MainThreadOnly::new(None),
            index_buffer: MainThreadOnly::new(None),
            vertices: Vec::new(),
            indices: Vec::new(),
            needs_update: false,
        }
    }

    #[track_caller]
    pub fn vertices(&self) -> &[V] {
        &self.vertices
    }

    #[track_caller]
    pub fn indices(&self) -> &[I] {
        &self.indices
    }

    pub fn vertices_indices_mut(&mut self) -> (&mut Vec<V>, &mut Vec<I>) {
        self.needs_update = true;
        (&mut self.vertices, &mut self.indices)
    }

    pub fn vertices_mut(&mut self) -> &mut Vec<V> {
        self.vertices_indices_mut().0
    }

    pub fn indices_mut(&mut self) -> &mut Vec<I> {
        self.vertices_indices_mut().1
    }

    pub fn append(&mut self, vertices: &[V], indices: &[I])
    where
        I: Add<u32, Output = I>,
    {
        let old_length = self.vertices().len() as u32;
        let (self_vertices, self_indices) = self.vertices_indices_mut();
        self_vertices.extend_from_slice(vertices);
        self_indices.extend(indices.iter().map(|&i| i + old_length));
    }

    /// Must be called on main thread only.
    pub fn update_if_needed(&mut self, display: &impl glium::backend::Facade) -> bool {
        if !self.needs_update {
            return false;
        }
        self.needs_update = false;
        let vertex_buffer = self.vertex_buffer.get_mut();
        let index_buffer = self.index_buffer.get_mut();
        if self.vertices.is_empty() || self.indices.is_empty() {
            *vertex_buffer = None;
            *index_buffer = None;
            return true;
        }
        *vertex_buffer = Some(Box::new(
            glium::VertexBuffer::dynamic(display, &self.vertices).unwrap(),
        ));
        *index_buffer = Some(Box::new(
            glium::IndexBuffer::dynamic(
                display,
                glium::index::PrimitiveType::TrianglesList,
                &self.indices,
            )
            .unwrap(),
        ));
        true
    }

    /// Must be called on main thread only.
    pub fn draw(
        &self,
        frame: &mut glium::Frame,
        uniforms: impl glium::uniforms::Uniforms,
        shader: &glium::Program,
        draw_parameters: &glium::DrawParameters,
    ) {
        let Some(vertex_buffer) = self.vertex_buffer.get() else {
            return;
        };
        let Some(index_buffer) = self.index_buffer.get() else {
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
}

impl<V: Copy + glium::Vertex, I: Copy + glium::index::Index> Default for SharedMesh<V, I> {
    fn default() -> Self {
        Self::new()
    }
}
