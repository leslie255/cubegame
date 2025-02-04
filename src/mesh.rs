//! Utilities for drawing.

use std::{
    ops::Add,
    sync::{Arc, Mutex, MutexGuard},
};

use glium::Surface as _;

use cgmath::*;

pub fn matrix4_to_array<T>(matrix: Matrix4<T>) -> [[T; 4]; 4] {
    matrix.into()
}

pub fn matrix3_to_array<T>(matrix: Matrix3<T>) -> [[T; 3]; 3] {
    matrix.into()
}

pub fn texture_sampler(texture: &glium::Texture2d) -> glium::uniforms::Sampler<glium::Texture2d> {
    texture
        .sampled()
        .magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest)
        .minify_filter(glium::uniforms::MinifySamplerFilter::Nearest)
        .wrap_function(glium::uniforms::SamplerWrapFunction::Repeat)
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

/// A mesh whose data can be offloaded to other threads as `Arc<MeshData>`.
/// `MeshRef` can exist past the scope of its `SharedMesh`, at which point any mutations to it
/// would never get to the GPU.
#[derive(Debug)]
pub struct SharedMesh<V: Copy + glium::Vertex, I: Copy + glium::index::Index = u32> {
    vertex_buffer: Option<Box<glium::VertexBuffer<V>>>,
    index_buffer: Option<Box<glium::IndexBuffer<I>>>,
    ref_: MeshRef<V, I>,
}

impl<V: Copy + glium::Vertex, I: Copy + glium::index::Index> SharedMesh<V, I> {
    pub fn new() -> Self {
        Self {
            vertex_buffer: None,
            index_buffer: None,
            ref_: MeshRef::new(Arc::new(Mutex::new(MeshData::new()))),
        }
    }

    pub fn borrow(&self) -> MeshRef<V, I> {
        self.ref_.clone()
    }

    #[track_caller]
    pub fn lock_mesh_data(&self) -> MutexGuard<MeshData<V, I>> {
        self.ref_.lock_mesh_data()
    }

    pub fn update_if_needed(&mut self, display: &impl glium::backend::Facade) {
        let mut mesh_data = self.ref_.lock_mesh_data();
        if !mesh_data.needs_update {
            return;
        }
        mesh_data.needs_update = false;
        if mesh_data.vertices.is_empty() || mesh_data.indices.is_empty() {
            self.vertex_buffer = None;
            self.index_buffer = None;
        }
        self.vertex_buffer = Some(Box::new(
            glium::VertexBuffer::dynamic(display, &mesh_data.vertices).unwrap(),
        ));
        self.index_buffer = Some(Box::new(
            glium::IndexBuffer::dynamic(
                display,
                glium::index::PrimitiveType::TrianglesList,
                &mesh_data.indices,
            )
            .unwrap(),
        ));
    }

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
}

impl<V: Copy + glium::Vertex, I: Copy + glium::index::Index> Default for SharedMesh<V, I> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct MeshData<V: Copy + glium::Vertex, I: Copy + glium::index::Index = u32> {
    vertices: Vec<V>,
    indices: Vec<I>,
    /// Whether the GL vertex and index buffer reflects the up-to-date content of the vertices/indices.
    /// `true` for not updated.
    needs_update: bool,
}

impl<V: Copy + glium::Vertex, I: Copy + glium::index::Index> MeshData<V, I> {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            needs_update: false,
        }
    }

    pub fn vertices(&self) -> &[V] {
        &self.vertices
    }

    pub fn indices(&self) -> &[I] {
        &self.indices
    }

    pub fn vertices_mut(&mut self) -> &mut Vec<V> {
        self.vertices_indices_mut().0
    }

    pub fn indices_mut(&mut self) -> &mut Vec<I> {
        self.vertices_indices_mut().1
    }

    pub fn vertices_indices_mut(&mut self) -> (&mut Vec<V>, &mut Vec<I>) {
        self.needs_update = true;
        (&mut self.vertices, &mut self.indices)
    }

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

impl<V: Copy + glium::Vertex, I: Copy + glium::index::Index> Default for MeshData<V, I> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MeshRef<V: Copy + glium::Vertex, I: Copy + glium::index::Index = u32> {
    data: Arc<Mutex<MeshData<V, I>>>,
}

impl<V: Copy + glium::Vertex, I: Copy + glium::index::Index> MeshRef<V, I> {
    fn new(data: Arc<Mutex<MeshData<V, I>>>) -> Self {
        Self { data }
    }

    #[track_caller]
    pub fn lock_mesh_data(&self) -> MutexGuard<MeshData<V, I>> {
        self.data.lock().unwrap()
    }
}
