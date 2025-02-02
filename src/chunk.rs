use std::alloc::Layout;

use cgmath::*;

use crate::{
    block::{BlockFace, BlockId, BlockRegistry, BlockTextureId, BlockTransparency},
    game::GameResources,
    mesh::{Mesh, Quad2},
};

/// A chunk-local coordinate.
/// They are `Ord` just for the sake of it. The ordering has no geometric implications.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalCoord {
    pub x: u8,
    pub y: u8,
    pub z: u8,
}

impl LocalCoord {
    pub fn new(x: u8, y: u8, z: u8) -> Self {
        Self { x, y, z }
    }

    pub fn as_array<S: From<u8>>(self) -> [S; 3] {
        [self.x.into(), self.y.into(), self.z.into()]
    }

    pub fn as_vec3<S: From<u8>>(self) -> Vector3<S> {
        vec3(self.x.into(), self.y.into(), self.z.into())
    }

    pub const fn index(self) -> usize {
        self.y as usize * 32 * 32 + self.z as usize * 32 + self.x as usize
    }
}

#[derive(Debug, Clone)]
pub struct ChunkData {
    blocks: [BlockId; 32 * 32 * 32],
}

impl ChunkData {
    /// A new chunk filled with block ID 0.
    pub fn new() -> Box<Self> {
        unsafe {
            let ptr = std::alloc::alloc_zeroed(Layout::new::<Self>()) as *mut Self;
            Box::from_raw(ptr)
        }
    }

    pub fn blocks(&self) -> &[BlockId] {
        self.blocks.as_slice()
    }

    pub fn blocks_mut(&mut self) -> &mut [BlockId] {
        self.blocks.as_mut_slice()
    }

    pub fn try_get_block(&self, local_coord: LocalCoord) -> Option<BlockId> {
        self.blocks.get(local_coord.index()).copied()
    }

    pub fn try_get_block_mut(&mut self, local_coord: LocalCoord) -> Option<&mut BlockId> {
        self.blocks.get_mut(local_coord.index())
    }

    #[track_caller]
    pub fn get_block(&self, local_coord: LocalCoord) -> BlockId {
        self.try_get_block(local_coord).unwrap()
    }

    #[track_caller]
    pub fn get_block_mut(&mut self, local_coord: LocalCoord) -> &mut BlockId {
        self.try_get_block_mut(local_coord).unwrap()
    }

    /// # Safety
    /// `local_coord` must be in range.
    pub unsafe fn get_block_unchecked(&self, local_coord: LocalCoord) -> BlockId {
        unsafe { *self.blocks.get_unchecked(local_coord.index()) }
    }

    /// # Safety
    /// `local_coord` must be in range.
    pub unsafe fn get_block_unchecked_mut(&mut self, local_coord: LocalCoord) -> &mut BlockId {
        unsafe { self.blocks.get_unchecked_mut(local_coord.index()) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

glium::implement_vertex!(BlockVertex, position, uv);

impl BlockVertex {
    pub const fn new(position: [f32; 3], uv: [f32; 2]) -> Self {
        Self { position, uv }
    }
}

#[derive(Debug)]
pub struct ChunkBuilder<'res> {
    block_registry: &'res BlockRegistry,
    block_atlas: &'res glium::Texture2d,
}

impl<'res> ChunkBuilder<'res> {
    const CUBE_VERTICES: [[BlockVertex; 4]; 6] = [
        // South
        [
            BlockVertex::new([0., 0., 0.], [1.0, 1.0]),
            BlockVertex::new([0., 1., 0.], [1.0, 0.0]),
            BlockVertex::new([1., 1., 0.], [0.0, 0.0]),
            BlockVertex::new([1., 0., 0.], [0.0, 1.0]),
        ],
        // North
        [
            BlockVertex::new([0., 0., 1.], [0.0, 1.0]),
            BlockVertex::new([1., 0., 1.], [1.0, 1.0]),
            BlockVertex::new([1., 1., 1.], [1.0, 0.0]),
            BlockVertex::new([0., 1., 1.], [0.0, 0.0]),
        ],
        // East
        [
            BlockVertex::new([1., 0., 0.], [1.0, 1.0]),
            BlockVertex::new([1., 1., 0.], [1.0, 0.0]),
            BlockVertex::new([1., 1., 1.], [0.0, 0.0]),
            BlockVertex::new([1., 0., 1.], [0.0, 1.0]),
        ],
        // West
        [
            BlockVertex::new([0., 1., 0.], [0.0, 0.0]),
            BlockVertex::new([0., 0., 0.], [0.0, 1.0]),
            BlockVertex::new([0., 0., 1.], [1.0, 1.0]),
            BlockVertex::new([0., 1., 1.], [1.0, 0.0]),
        ],
        // Up
        [
            BlockVertex::new([1., 1., 0.], [0.0, 1.0]),
            BlockVertex::new([0., 1., 0.], [1.0, 1.0]),
            BlockVertex::new([0., 1., 1.], [1.0, 0.0]),
            BlockVertex::new([1., 1., 1.], [0.0, 0.0]),
        ],
        // Down
        [
            BlockVertex::new([0., 0., 0.], [0.0, 1.0]),
            BlockVertex::new([1., 0., 0.], [1.0, 1.0]),
            BlockVertex::new([1., 0., 1.], [1.0, 0.0]),
            BlockVertex::new([0., 0., 1.], [0.0, 0.0]),
        ],
    ];

    const FACE_INDICIES: [u32; 6] = [0, 1, 2, 2, 3, 0];

    pub fn new(resources: &'res GameResources) -> Self {
        Self {
            block_registry: &resources.block_registry,
            block_atlas: &resources.block_atlas,
        }
    }

    pub fn build(
        &mut self,
        display: &impl glium::backend::Facade,
        chunk: &ChunkData,
        chunk_mesh: &mut ChunkMesh,
    ) {
        chunk_mesh.mesh.vertices_mut().clear();
        chunk_mesh.mesh.indices_mut().clear();
        for y in 0..32 {
            for x in 0..32 {
                for z in 0..32 {
                    let local_coord = LocalCoord::new(y, z, x);
                    let block_id = unsafe { chunk.get_block_unchecked(local_coord) };
                    self.build_block(local_coord, block_id, chunk_mesh);
                }
            }
        }
        chunk_mesh.mesh.update(display);
    }

    fn build_block(
        &mut self,
        local_position: LocalCoord,
        block_id: BlockId,
        chunk_mesh: &mut ChunkMesh,
    ) {
        let block_info = self.block_registry.lookup(block_id).unwrap();
        if block_info.transparency == BlockTransparency::Air {
            return;
        }
        for block_face in BlockFace::iter() {
            let texture_id = block_info.model.texture_for_face(block_face);
            let texture_coord = self.texture_coord(texture_id);
            let vertices = Self::CUBE_VERTICES[block_face.to_usize()].map(|vertex| BlockVertex {
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
            let indices = Self::FACE_INDICIES;
            chunk_mesh
                .mesh
                .append(vertices.as_slice(), indices.as_slice());
        }
    }

    /// The normalized texture coord for a block texture ID.
    fn texture_coord(&self, texture_id: BlockTextureId) -> Quad2 {
        let block_width: f32 = const { 16. / 256. };
        let block_height: f32 = const { 16. / 256. };
        let i_x = texture_id.0 % 16;
        let i_y = texture_id.0 / 16;
        Quad2 {
            left: i_x as f32 * block_width,
            right: (i_x + 1) as f32 * block_width,
            bottom: (i_y + 1) as f32 * block_height,
            top: i_y as f32 * block_height,
        }
    }
}

#[derive(Debug, Default)]
pub struct ChunkMesh {
    mesh: Mesh<BlockVertex>,
}

impl ChunkMesh {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(&mut self, _chunk: &ChunkData) {
        self.mesh.vertices_mut().clear();
        self.mesh.indices_mut().clear();
    }

    pub fn mesh(&self) -> &Mesh<BlockVertex> {
        &self.mesh
    }

    pub fn mesh_mut(&self) -> &Mesh<BlockVertex> {
        &self.mesh
    }
}
