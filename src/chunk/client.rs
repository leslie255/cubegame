use std::array;

use bytemuck::{Pod, Zeroable};
use cgmath::*;

use wgpu::util::DeviceExt as _;

use crate::{
    block::{BlockFace, BlockId, BlockRegistry, BlockTextureId, BlockTransparency},
    chunk::{Chunk, ChunkData},
    game::GameResources,
    impl_as_bind_group,
    utils::Quad2d,
    wgpu_utils::{self, IndexBuffer, UniformBuffer, Vertex, Vertex3dUV, VertexBuffer},
    world::{ChunkId, ChunkManager, LocalCoordU8},
};

#[derive(Debug)]
pub struct ChunkRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_0: ChunkMeshBindGroup0,
    pub bind_group_0_wgpu: wgpu::BindGroup,
}

impl ChunkRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &GameResources,
        surface_color_format: wgpu::TextureFormat,
        depth_stencil_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        Self::new_with_pipeline_descriptor(
            device,
            queue,
            resources,
            surface_color_format,
            depth_stencil_format,
            |_| (),
        )
    }

    pub fn new_wireframe_mode(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &GameResources,
        surface_color_format: wgpu::TextureFormat,
        depth_stencil_format: Option<wgpu::TextureFormat>,
    ) -> Option<Self> {
        let supports_polygon_line_mode = device
            .features()
            .contains(wgpu::FeaturesWGPU::POLYGON_MODE_LINE.into());
        supports_polygon_line_mode.then(|| {
            Self::new_with_pipeline_descriptor(
                device,
                queue,
                resources,
                surface_color_format,
                depth_stencil_format,
                |pipeline_layout_descriptor| {
                    pipeline_layout_descriptor.primitive.polygon_mode = wgpu::PolygonMode::Line;
                    pipeline_layout_descriptor.primitive.cull_mode = None;
                },
            )
        })
    }

    fn new_with_pipeline_descriptor(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &GameResources,
        surface_color_format: wgpu::TextureFormat,
        depth_stencil_format: Option<wgpu::TextureFormat>,
        f: impl FnOnce(&mut wgpu::RenderPipelineDescriptor),
    ) -> Self {
        let (bind_group_0, bind_group_0_layout, bind_group_0_wgpu) =
            Self::create_bind_group_0(device, queue, resources);
        let bind_group_1_layout =
            wgpu_utils::create_bind_group_layout::<ChunkMeshBindGroup1>(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_0_layout, &bind_group_1_layout],
            push_constant_ranges: &[],
        });
        let mut pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &resources.shader_chunk,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[PackedChunkVertex::LAYOUT],
            },
            fragment: Some(wgpu::FragmentState {
                module: &resources.shader_chunk,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_color_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            operation: wgpu::BlendOperation::Add,
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                polygon_mode: wgpu::PolygonMode::Fill,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: depth_stencil_format.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        };
        f(&mut pipeline_descriptor);
        let pipeline = device.create_render_pipeline(&pipeline_descriptor);
        Self {
            pipeline,
            bind_group_0,
            bind_group_0_wgpu,
        }
    }

    fn create_bind_group_0(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &GameResources,
    ) -> (ChunkMeshBindGroup0, wgpu::BindGroupLayout, wgpu::BindGroup) {
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: resources.block_atlas_image.width(),
                    height: resources.block_atlas_image.height(),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
            },
            wgpu::wgt::TextureDataOrder::LayerMajor,
            &resources.block_atlas_image,
        );
        let texture_view = texture.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });
        let bind_group_0 = ChunkMeshBindGroup0 {
            projection: UniformBuffer::create_init(device, Matrix4::identity().into()),
            sun: UniformBuffer::create_init(device, Vector3::unit_x().into()),
            texture_view,
            sampler,
            gray_world: UniformBuffer::create_init(device, 0),
        };
        let bind_group_0_layout =
            wgpu_utils::create_bind_group_layout::<ChunkMeshBindGroup0>(device);
        let bind_group_0_wgpu =
            wgpu_utils::create_bind_group(device, &bind_group_0_layout, &bind_group_0);
        (bind_group_0, bind_group_0_layout, bind_group_0_wgpu)
    }

    pub fn set_projection(&self, queue: &wgpu::Queue, projection: Matrix4<f32>) {
        self.bind_group_0.projection.write(projection.into(), queue);
    }

    pub fn set_sun(&self, queue: &wgpu::Queue, sun: Vector3<f32>) {
        self.bind_group_0.sun.write(sun.into(), queue);
    }

    pub fn set_gray_world(&self, queue: &wgpu::Queue, gray_world: bool) {
        self.bind_group_0.gray_world.write(gray_world.into(), queue);
    }

    pub fn begin_drawing(&self, render_pass: &mut wgpu::RenderPass) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group_0_wgpu, &[]);
    }

    pub fn draw_chunk(&self, render_pass: &mut wgpu::RenderPass, mesh: &ChunkMesh) {
        render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        render_pass.set_index_buffer(
            mesh.index_buffer.slice(..),
            mesh.index_buffer.index_format(),
        );
        render_pass.set_bind_group(1, &mesh.bind_group_1_wgpu, &[]);
        render_pass.draw_indexed(0..mesh.index_buffer.length(), 0, 0..1);
    }
}

#[derive(Debug, Clone)]
pub struct ChunkMeshBindGroup0 {
    pub(super) projection: UniformBuffer<[[f32; 4]; 4]>,
    pub(super) sun: UniformBuffer<[f32; 3]>,
    pub(super) texture_view: wgpu::TextureView,
    pub(super) sampler: wgpu::Sampler,
    pub(super) gray_world: UniformBuffer<u32>,
}

#[derive(Debug, Clone)]
pub struct ChunkMeshBindGroup1 {
    pub(super) model_view: UniformBuffer<[[f32; 4]; 4]>,
    pub(super) _normal: UniformBuffer<[f32; 3]>,
}

impl_as_bind_group! {
    ChunkMeshBindGroup0 {
        0 => projection: UniformBuffer<[[f32; 4]; 4]>,
        1 => sun: UniformBuffer<[f32; 3]>,
        2 => texture_view: wgpu::TextureView,
        3 => sampler: wgpu::Sampler,
        4 => gray_world: UniformBuffer<u32>,
    }

    ChunkMeshBindGroup1 {
        0 => model_view: UniformBuffer<[[f32; 4]; 4]>,
        1 => _normal: UniformBuffer<[f32; 3]>,
    }
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct PackedChunkVertex(pub u32);

impl PackedChunkVertex {
    pub fn pack(position: Vector3<f32>, uv: Vector2<f32>, normal: Vector3<f32>) -> Self {
        let pos_packed = {
            let x_pos = position.x as u32;
            let y_pos = position.y as u32;
            let z_pos = position.z as u32;
            x_pos + 33 * y_pos + 33 * 33 * z_pos
        };
        let uv_packed = {
            let x_uv = (uv.x * 64.).round() as u32;
            let y_uv = (uv.y * 64.).round() as u32;
            x_uv + y_uv * 65
        };
        let x_normal = if normal.x > 0.0 { 1 } else { 0 };
        let y_normal = if normal.y > 0.0 { 1 } else { 0 };
        let z_normal = if normal.z > 0.0 { 1 } else { 0 };
        let packed =
            pos_packed | uv_packed << 16 | x_normal << 29 | y_normal << 30 | z_normal << 31;
        Self(packed)
    }
}

impl Vertex for PackedChunkVertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: size_of::<Self>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Uint32,
            offset: 0,
            shader_location: 0,
        }],
    };
}

#[derive(Debug, Clone)]
pub struct ChunkMesh {
    pub vertex_buffer: VertexBuffer<PackedChunkVertex>,
    pub index_buffer: IndexBuffer<u32>,
    pub bind_group_1: ChunkMeshBindGroup1,
    pub bind_group_1_wgpu: wgpu::BindGroup,
}

impl ChunkMesh {
    pub fn set_model_view(&self, queue: &wgpu::Queue, model_view: Matrix4<f64>) {
        let model_view_f32 = Matrix4 {
            x: model_view.x.map(|f| f as f32),
            y: model_view.y.map(|f| f as f32),
            z: model_view.z.map(|f| f as f32),
            w: model_view.w.map(|f| f as f32),
        };
        self.bind_group_1
            .model_view
            .write(model_view_f32.into(), queue);
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChunkMeshData {
    vertices: Vec<PackedChunkVertex>,
    indices: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct ChunkBuilder<'cx> {
    resources: &'cx GameResources,
    mesh_data_opaque: ChunkMeshData,
    mesh_data_transparent: ChunkMeshData,
    neighbor_chunk_surfaces: [Box<ChunkSurface>; 6],
}

impl<'cx> ChunkBuilder<'cx> {
    const CUBE_VERTICES: [[Vertex3dUV; 4]; 6] = [
        // South
        [
            Vertex3dUV::new([0., 0., 1.], [0., 1.]),
            Vertex3dUV::new([1., 0., 1.], [1., 1.]),
            Vertex3dUV::new([1., 1., 1.], [1., 0.]),
            Vertex3dUV::new([0., 1., 1.], [0., 0.]),
        ],
        // North
        [
            Vertex3dUV::new([0., 0., 0.], [1., 1.]),
            Vertex3dUV::new([0., 1., 0.], [1., 0.]),
            Vertex3dUV::new([1., 1., 0.], [0., 0.]),
            Vertex3dUV::new([1., 0., 0.], [0., 1.]),
        ],
        // East
        [
            Vertex3dUV::new([1., 0., 0.], [1., 1.]),
            Vertex3dUV::new([1., 1., 0.], [1., 0.]),
            Vertex3dUV::new([1., 1., 1.], [0., 0.]),
            Vertex3dUV::new([1., 0., 1.], [0., 1.]),
        ],
        // West
        [
            Vertex3dUV::new([0., 1., 0.], [0., 0.]),
            Vertex3dUV::new([0., 0., 0.], [0., 1.]),
            Vertex3dUV::new([0., 0., 1.], [1., 1.]),
            Vertex3dUV::new([0., 1., 1.], [1., 0.]),
        ],
        // Up
        [
            Vertex3dUV::new([1., 1., 0.], [0.0, 1.0]),
            Vertex3dUV::new([0., 1., 0.], [1.0, 1.0]),
            Vertex3dUV::new([0., 1., 1.], [1.0, 0.0]),
            Vertex3dUV::new([1., 1., 1.], [0.0, 0.0]),
        ],
        // Down
        [
            Vertex3dUV::new([0., 0., 0.], [0.0, 1.0]),
            Vertex3dUV::new([1., 0., 0.], [1.0, 1.0]),
            Vertex3dUV::new([1., 0., 1.], [1.0, 0.0]),
            Vertex3dUV::new([0., 0., 1.], [0.0, 0.0]),
        ],
    ];

    const FACE_INDICIES: [u32; 6] = [0, 1, 2, 2, 3, 0];

    pub fn new(resources: &'cx GameResources) -> Self {
        Self {
            resources,
            mesh_data_opaque: ChunkMeshData::default(),
            mesh_data_transparent: ChunkMeshData::default(),
            neighbor_chunk_surfaces: array::from_fn(|_| ChunkSurface::new_boxed()),
        }
    }

    pub fn build(&mut self, device: &wgpu::Device, chunk_id: ChunkId, chunks: &ChunkManager) {
        if !chunks.chunk_is_loaded(chunk_id) {
            return;
        }
        self.mesh_data_opaque.vertices.clear();
        self.mesh_data_opaque.indices.clear();
        self.mesh_data_transparent.vertices.clear();
        self.mesh_data_transparent.indices.clear();
        for (i_face, neighbor_surface) in self.neighbor_chunk_surfaces.iter_mut().enumerate() {
            let face = BlockFace::from_usize(i_face).unwrap();
            let neighbor_chunk_id = chunk_id + face.normal_vector().map(|f| f as i32);
            let has_neighbor_chunk = chunks
                .with_loaded_chunk(neighbor_chunk_id, |neighbor_chunk| {
                    neighbor_surface.initialize_for_chunk(
                        face.opposite(),
                        &self.resources.block_registry,
                        &neighbor_chunk.data,
                    );
                })
                .is_some();
            if !has_neighbor_chunk {
                neighbor_surface.clear();
            }
        }
        if !chunks.chunk_is_loaded(chunk_id) {
            return;
        }
        let bind_group_1 = ChunkMeshBindGroup1 {
            model_view: UniformBuffer::create_init(device, Matrix4::identity().into()),
            _normal: UniformBuffer::create_init(device, Vector3::unit_y().into()),
        };
        let bind_group_1_wgpu = {
            let layout = wgpu_utils::create_bind_group_layout::<ChunkMeshBindGroup1>(device);
            wgpu_utils::create_bind_group(device, &layout, &bind_group_1)
        };
        chunks.with_loaded_chunk(chunk_id, |chunk| {
            for y in 0..32 {
                for x in 0..32 {
                    for z in 0..32 {
                        let local_coord = LocalCoordU8::new(y, z, x);
                        let block_id = unsafe { chunk.data.get_block_unchecked(local_coord) };
                        self.build_block(chunk, local_coord, block_id);
                    }
                }
            }
            if !self.mesh_data_opaque.vertices.is_empty() {
                chunk.client.mesh_opaque = Some(ChunkMesh {
                    vertex_buffer: VertexBuffer::create_init(
                        device,
                        &self.mesh_data_opaque.vertices,
                    ),
                    index_buffer: IndexBuffer::create_init(device, &self.mesh_data_opaque.indices),
                    bind_group_1_wgpu: bind_group_1_wgpu.clone(),
                    bind_group_1: bind_group_1.clone(),
                });
            }
            if !self.mesh_data_transparent.vertices.is_empty() {
                chunk.client.mesh_transparent = Some(ChunkMesh {
                    vertex_buffer: VertexBuffer::create_init(
                        device,
                        &self.mesh_data_transparent.vertices,
                    ),
                    index_buffer: IndexBuffer::create_init(
                        device,
                        &self.mesh_data_transparent.indices,
                    ),
                    bind_group_1_wgpu,
                    bind_group_1,
                });
            }
            chunk.client.is_synced = true;
        });
    }

    fn build_block(&mut self, chunk: &mut Chunk, local_position: LocalCoordU8, block_id: BlockId) {
        let block_info = self.resources.block_registry.lookup(block_id).unwrap();
        if block_info.transparency == BlockTransparency::Air {
            return;
        }
        let block_transparency = block_info.transparency;
        for block_face in BlockFace::iter() {
            if let Some(neighbor_transparency) =
                self.neighbor_transparency(local_position, block_face, &chunk.data)
            {
                use BlockTransparency::*;
                match (block_transparency, neighbor_transparency) {
                    (_, Solid) | (Transparent, Transparent) => continue,
                    _ => (),
                }
            }
            let texture_id = block_info.model.texture_for_face(block_face);
            let texture_coord = self.texture_coord(texture_id);
            let vertices = Self::CUBE_VERTICES[block_face.to_usize()].map(|vertex| {
                let position = vec3(
                    vertex.position[0] + local_position.x as f32,
                    vertex.position[1] + local_position.y as f32,
                    vertex.position[2] + local_position.z as f32,
                );
                let uv = vec2(
                    vertex.uv[0] * texture_coord.right + (1. - vertex.uv[0]) * texture_coord.left,
                    vertex.uv[1] * texture_coord.bottom + (1. - vertex.uv[1]) * texture_coord.top,
                );
                let normal = block_face.normal_vector();
                PackedChunkVertex::pack(position, uv, normal)
            });
            if block_transparency.is_solid() {
                let indices =
                    Self::FACE_INDICIES.map(|i| i + self.mesh_data_opaque.vertices.len() as u32);
                self.mesh_data_opaque.vertices.extend(&vertices[..]);
                self.mesh_data_opaque.indices.extend(&indices[..]);
            } else {
                let indices = Self::FACE_INDICIES
                    .map(|i| i + self.mesh_data_transparent.vertices.len() as u32);
                self.mesh_data_transparent.vertices.extend(&vertices[..]);
                self.mesh_data_transparent.indices.extend(&indices[..]);
            }
        }
    }

    fn neighbor_local_coord(
        local_coord: LocalCoordU8,
        direction: BlockFace,
    ) -> Option<LocalCoordU8> {
        let mut result = local_coord;
        match direction {
            BlockFace::South => result.z = (result.z + 1).min(31),
            BlockFace::East => result.x = (result.x + 1).min(31),
            BlockFace::Top => result.y = (result.y + 1).min(31),
            BlockFace::North => result.z = result.z.saturating_sub(1),
            BlockFace::West => result.x = result.x.saturating_sub(1),
            BlockFace::Bottom => result.y = result.y.saturating_sub(1),
        }
        (result != local_coord).then_some(result)
    }

    fn neighbor_transparency(
        &self,
        local_coord: LocalCoordU8,
        direction: BlockFace,
        chunk: &ChunkData,
    ) -> Option<BlockTransparency> {
        if let Some(neighbor_coord) = Self::neighbor_local_coord(local_coord, direction) {
            chunk
                .try_get_block(neighbor_coord)
                .and_then(|block_id| self.resources.block_registry.lookup(block_id))
                .map(|block_info| block_info.transparency)
        } else {
            let neighbor_chunk_surface = &self.neighbor_chunk_surfaces[direction.to_usize()];
            if (local_coord.x == 31 && direction == BlockFace::East)
                || (local_coord.x == 0 && direction == BlockFace::West)
            {
                neighbor_chunk_surface.get(local_coord.y, local_coord.z)
            } else if (local_coord.y == 31 && direction == BlockFace::Top)
                || (local_coord.y == 0 && direction == BlockFace::Bottom)
            {
                neighbor_chunk_surface.get(local_coord.x, local_coord.z)
            } else if (local_coord.z == 31 && direction == BlockFace::South)
                || (local_coord.z == 0 && direction == BlockFace::North)
            {
                neighbor_chunk_surface.get(local_coord.x, local_coord.y)
            } else {
                None
            }
        }
    }

    /// The normalized texture coord for a block texture ID.
    fn texture_coord(&self, texture_id: BlockTextureId) -> Quad2d {
        let block_width = 16. / 256.;
        let block_height = 16. / 256.;
        let i_x = texture_id.0 % 16;
        let i_y = texture_id.0 / 16;
        Quad2d {
            left: i_x as f32 * block_width,
            right: (i_x + 1) as f32 * block_width,
            bottom: (i_y + 1) as f32 * block_height,
            top: i_y as f32 * block_height,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct ClientChunk {
    pub mesh_opaque: Option<ChunkMesh>,
    pub mesh_transparent: Option<ChunkMesh>,
    /// Whether the mesh is in-sync with the chunk data.
    pub is_synced: bool,
}

impl ClientChunk {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Zeroable)]
pub struct ChunkSurface {
    pub data: [BlockTransparency; 32 * 32],
}

impl ChunkSurface {
    pub fn new_boxed() -> Box<Self> {
        bytemuck::zeroed_box()
    }

    pub fn clear(&mut self) {
        bytemuck::write_zeroes(self);
    }

    pub fn initialize_for_chunk(
        &mut self,
        direction: BlockFace,
        block_registry: &BlockRegistry,
        chunk: &ChunkData,
    ) {
        for i in 0..32u8 {
            for j in 0..32u8 {
                let local_coord = match direction {
                    BlockFace::South => LocalCoordU8::new(i, j, 31),
                    BlockFace::North => LocalCoordU8::new(i, j, 0),
                    BlockFace::East => LocalCoordU8::new(31, i, j),
                    BlockFace::West => LocalCoordU8::new(0, i, j),
                    BlockFace::Top => LocalCoordU8::new(i, 31, j),
                    BlockFace::Bottom => LocalCoordU8::new(i, 0, j),
                };
                let block_id = unsafe { chunk.get_block_unchecked(local_coord) };
                let block_transparency = block_registry.lookup(block_id).unwrap().transparency;
                unsafe {
                    *self.get_unchecked_mut(i, j) = block_transparency;
                }
            }
        }
    }

    fn index_for(i: u8, j: u8) -> usize {
        (i as usize) * 32 + (j as usize)
    }

    pub fn get(&self, i: u8, j: u8) -> Option<BlockTransparency> {
        self.data.get(Self::index_for(i, j)).copied()
    }

    pub fn get_mut(&mut self, i: u8, j: u8) -> Option<&mut BlockTransparency> {
        self.data.get_mut(Self::index_for(i, j))
    }

    /// # Safety
    ///
    /// `i` and `j` must be in range
    pub unsafe fn get_unchecked(&self, i: u8, j: u8) -> BlockTransparency {
        unsafe { *self.data.get_unchecked(Self::index_for(i, j)) }
    }

    /// # Safety
    ///
    /// `i` and `j` must be in range
    pub unsafe fn get_unchecked_mut(&mut self, i: u8, j: u8) -> &mut BlockTransparency {
        unsafe { self.data.get_unchecked_mut(Self::index_for(i, j)) }
    }
}
