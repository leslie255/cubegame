use std::alloc::Layout;

use cgmath::*;

use crate::{
    block::{BlockFace, BlockId, BlockRegistry, BlockTextureId, BlockTransparency},
    game::GameResources,
    mesh::{Quad2, SharedMesh},
};

pub type LocalCoord = Point3<u8>;

#[derive(Debug)]
pub struct Chunk {
    pub data: Box<ChunkData>,
    pub client: ClientChunk,
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

    fn index(local_coord: LocalCoord) -> usize {
        (local_coord.y as usize) * 32 * 32
            + (local_coord.z as usize) * 32
            + (local_coord.x as usize)
    }

    pub fn try_get_block(&self, local_coord: LocalCoord) -> Option<BlockId> {
        self.blocks.get(Self::index(local_coord)).copied()
    }

    pub fn try_get_block_mut(&mut self, local_coord: LocalCoord) -> Option<&mut BlockId> {
        self.blocks.get_mut(Self::index(local_coord))
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
        unsafe { *self.blocks.get_unchecked(Self::index(local_coord)) }
    }

    /// # Safety
    /// `local_coord` must be in range.
    pub unsafe fn get_block_unchecked_mut(&mut self, local_coord: LocalCoord) -> &mut BlockId {
        unsafe { self.blocks.get_unchecked_mut(Self::index(local_coord)) }
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

#[derive(Debug, Clone)]
pub struct ChunkBuilder<'res> {
    block_registry: &'res BlockRegistry,
}

impl<'res> ChunkBuilder<'res> {
    const CUBE_VERTICES: [[BlockVertex; 4]; 6] = [
        // South
        [
            BlockVertex::new([0., 0., 1.], [0.0, 1.0]),
            BlockVertex::new([1., 0., 1.], [1.0, 1.0]),
            BlockVertex::new([1., 1., 1.], [1.0, 0.0]),
            BlockVertex::new([0., 1., 1.], [0.0, 0.0]),
        ],
        // North
        [
            BlockVertex::new([0., 0., 0.], [1.0, 1.0]),
            BlockVertex::new([0., 1., 0.], [1.0, 0.0]),
            BlockVertex::new([1., 1., 0.], [0.0, 0.0]),
            BlockVertex::new([1., 0., 0.], [0.0, 1.0]),
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
        }
    }

    pub fn build(&mut self, chunk: &mut Chunk) {
        chunk.client.mesh.vertices_mut().clear();
        chunk.client.mesh.indices_mut().clear();
        for y in 0..32 {
            for x in 0..32 {
                for z in 0..32 {
                    let local_coord = LocalCoord::new(y, z, x);
                    let block_id = unsafe { chunk.data.get_block_unchecked(local_coord) };
                    self.build_block(chunk, local_coord, block_id);
                }
            }
        }
    }

    fn build_block(&mut self, chunk: &mut Chunk, local_position: LocalCoord, block_id: BlockId) {
        let block_info = self.block_registry.lookup(block_id).unwrap();
        if block_info.transparency == BlockTransparency::Air {
            return;
        }
        for block_face in BlockFace::iter() {
            if let Some(neighbor_block_id) =
                Self::neighbor_block(local_position, block_face, &chunk.data)
            {
                let neighbor_transparency = self
                    .block_registry
                    .lookup(neighbor_block_id)
                    .unwrap()
                    .transparency;
                if neighbor_transparency == BlockTransparency::Solid {
                    continue;
                }
            }
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
            chunk
                .client
                .mesh
                .append(vertices.as_slice(), indices.as_slice());
        }
    }

    fn neighbor_local_coord(local_coord: LocalCoord, direction: BlockFace) -> Option<LocalCoord> {
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
        local_coord: LocalCoord,
        direction: BlockFace,
        chunk: &ChunkData,
    ) -> Option<BlockId> {
        chunk.try_get_block(Self::neighbor_local_coord(local_coord, direction)?)
    }

    /// The normalized texture coord for a block texture ID.
    fn texture_coord(&self, texture_id: BlockTextureId) -> Quad2 {
        let block_width = 16. / 256.;
        let block_height = 16. / 256.;
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
pub struct ClientChunk {
    pub mesh: SharedMesh<BlockVertex>,
}

impl ClientChunk {
    pub fn new() -> Self {
        Self::default()
    }
}
