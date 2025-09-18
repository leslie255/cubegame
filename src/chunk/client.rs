use std::array;

use cgmath::*;

use wgpu::util::DeviceExt as _;

use crate::{
    block::{BlockFace, BlockId, BlockTextureId, BlockTransparency},
    chunk::{Chunk, ChunkData},
    game::GameResources,
    impl_as_bind_group,
    utils::Quad2d,
    wgpu_utils::{self, IndexBuffer, UniformBuffer, Vertex as _, Vertex3dUV, VertexBuffer},
    world::{ChunkId, LocalCoordU8},
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
        let (bind_group_0, bind_group_0_layout, bind_group_0_wgpu) =
            Self::create_bind_group_0(device, queue, resources);
        let bind_group_1_layout =
            wgpu_utils::create_bind_group_layout::<ChunkMeshBindGroup1>(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_0_layout, &bind_group_1_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &resources.shader_chunk,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Vertex3dUV::LAYOUT],
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
            primitive: Default::default(),
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
        });
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
            view_projection: UniformBuffer::create_init(device, Matrix4::identity().into()),
            sun: UniformBuffer::create_init(device, Vector3::unit_x().into()),
            texture_view,
            sampler,
        };
        let bind_group_0_layout =
            wgpu_utils::create_bind_group_layout::<ChunkMeshBindGroup0>(device);
        let bind_group_0_wgpu =
            wgpu_utils::create_bind_group(device, &bind_group_0_layout, &bind_group_0);
        (bind_group_0, bind_group_0_layout, bind_group_0_wgpu)
    }

    pub fn set_view_projection(&self, queue: &wgpu::Queue, view_projection: Matrix4<f32>) {
        self.bind_group_0
            .view_projection
            .write(view_projection.into(), queue);
    }

    pub fn set_sun(&self, queue: &wgpu::Queue, sun: Vector3<f32>) {
        self.bind_group_0.sun.write(sun.into(), queue);
    }
}

#[derive(Debug, Clone)]
pub struct ChunkMeshBindGroup0 {
    pub(super) view_projection: UniformBuffer<[[f32; 4]; 4]>,
    pub(super) sun: UniformBuffer<[f32; 3]>,
    pub(super) texture_view: wgpu::TextureView,
    pub(super) sampler: wgpu::Sampler,
}

#[derive(Debug, Clone)]
pub struct ChunkMeshBindGroup1 {
    pub(super) model: UniformBuffer<[[f32; 4]; 4]>,
    pub(super) normal: UniformBuffer<[f32; 3]>,
}

impl_as_bind_group! {
    ChunkMeshBindGroup0 {
        0 => view_projection: UniformBuffer<[[f32; 4]; 4]>,
        1 => sun: UniformBuffer<[f32; 3]>,
        2 => texture_view: wgpu::TextureView,
        3 => sampler: wgpu::Sampler,
    }

    ChunkMeshBindGroup1 {
        0 => model: UniformBuffer<[[f32; 4]; 4]>,
        1 => normal: UniformBuffer<[f32; 3]>,
    }
}

#[derive(Debug, Clone)]
pub struct ChunkMesh {
    pub vertex_buffer: VertexBuffer<Vertex3dUV>,
    pub index_buffer: IndexBuffer<u32>,
    pub bind_group_1: ChunkMeshBindGroup1,
    pub bind_group_1_wgpu: wgpu::BindGroup,
}

#[derive(Debug, Clone, Default)]
pub struct ChunkMeshData {
    vertices: Vec<Vertex3dUV>,
    indices: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct ChunkBuilder<'cx> {
    resources: &'cx GameResources,
    mesh_data: [ChunkMeshData; 6],
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
            mesh_data: array::from_fn(|_| ChunkMeshData::default()),
        }
    }

    pub fn build(&mut self, device: &wgpu::Device, chunk_id: ChunkId, chunk: &mut Chunk) {
        for mesh_data in &mut self.mesh_data {
            mesh_data.vertices.clear();
            mesh_data.indices.clear();
        }
        for y in 0..32 {
            for x in 0..32 {
                for z in 0..32 {
                    let local_coord = LocalCoordU8::new(y, z, x);
                    let block_id = unsafe { chunk.data.get_block_unchecked(local_coord) };
                    self.build_block(chunk, local_coord, block_id);
                }
            }
        }
        let model = Matrix4::from_translation(vec3(
            chunk_id.x as f32 * 32.0,
            chunk_id.y as f32 * 32.0,
            chunk_id.z as f32 * 32.0,
        ));
        for i_face in 0..6 {
            let mesh_data = &mut self.mesh_data[i_face];
            let mesh = &mut chunk.client.meshes[i_face];
            if mesh_data.vertices.is_empty() {
                continue;
            }
            let normal = BlockFace::from_usize(i_face).unwrap().normal_vector();
            let bind_group_1 = ChunkMeshBindGroup1 {
                model: UniformBuffer::create_init(device, model.into()),
                normal: UniformBuffer::create_init(device, normal.into()),
            };
            let bind_group_1_wgpu = {
                let layout = wgpu_utils::create_bind_group_layout::<ChunkMeshBindGroup1>(device);
                wgpu_utils::create_bind_group(device, &layout, &bind_group_1)
            };
            *mesh = Some(ChunkMesh {
                vertex_buffer: VertexBuffer::create_init(device, &mesh_data.vertices),
                index_buffer: IndexBuffer::create_init(device, &mesh_data.indices),
                bind_group_1_wgpu,
                bind_group_1,
            });
        }
    }

    fn build_block(&mut self, chunk: &mut Chunk, local_position: LocalCoordU8, block_id: BlockId) {
        let block_info = self.resources.block_registry.lookup(block_id).unwrap();
        if block_info.transparency == BlockTransparency::Air {
            return;
        }
        for block_face in BlockFace::iter() {
            if let Some(neighbor_block_id) =
                Self::neighbor_block(local_position, block_face, &chunk.data)
            {
                let block_transparency = block_info.transparency;
                let neighbor_transparency = self
                    .resources
                    .block_registry
                    .lookup(neighbor_block_id)
                    .unwrap()
                    .transparency;
                use BlockTransparency::*;
                match (block_transparency, neighbor_transparency) {
                    (_, Solid) | (Transparent, Transparent) => continue,
                    _ => (),
                }
            }
            let texture_id = block_info.model.texture_for_face(block_face);
            let texture_coord = self.texture_coord(texture_id);
            let vertices = Self::CUBE_VERTICES[block_face.to_usize()].map(|vertex| Vertex3dUV {
                position: [
                    vertex.position[0] + local_position.x as f32,
                    vertex.position[1] + local_position.y as f32,
                    vertex.position[2] + local_position.z as f32,
                ],
                uv: [
                    vertex.uv[0] * texture_coord.right + (1. - vertex.uv[0]) * texture_coord.left,
                    vertex.uv[1] * texture_coord.bottom + (1. - vertex.uv[1]) * texture_coord.top,
                ],
            });
            let mesh_data = &mut self.mesh_data[block_face.to_usize()];
            let indices = Self::FACE_INDICIES.map(|i| i + mesh_data.vertices.len() as u32);
            mesh_data.vertices.extend(&vertices[..]);
            mesh_data.indices.extend(&indices[..]);
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

    fn neighbor_block(
        local_coord: LocalCoordU8,
        direction: BlockFace,
        chunk: &ChunkData,
    ) -> Option<BlockId> {
        chunk.try_get_block(Self::neighbor_local_coord(local_coord, direction)?)
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

#[derive(Debug, Default)]
pub struct ClientChunk {
    pub meshes: [Option<ChunkMesh>; 6],
}

impl ClientChunk {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mesh(&self, face: BlockFace) -> Option<&ChunkMesh> {
        self.meshes[face.to_usize()].as_ref()
    }

    pub fn mesh_mut(&mut self, face: BlockFace) -> Option<&mut ChunkMesh> {
        self.meshes[face.to_usize()].as_mut()
    }
}
