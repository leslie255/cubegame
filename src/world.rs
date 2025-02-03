use std::ops::Range;

use cgmath::*;

use crate::{
    block::BlockId,
    chunk::{ChunkBuilder, ChunkData, ChunkMesh, LocalCoord},
    game::GameResources,
    utils::vec_with,
};

pub type ChunkId = Point3<i32>;

pub type BlockCoord = Point3<i32>;

#[derive(Debug)]
pub struct World<'res> {
    chunks: Vec<Box<ChunkData>>,
    chunk_meshes: Vec<ChunkMesh>,
    chunk_builder: ChunkBuilder<'res>,
    needs_chunk_building: bool,
}

const WORLD_ASSERTIONS: () = {
    assert!(World::SIZE_X % 2 == 0);
    assert!(World::SIZE_Y % 2 == 0);
    assert!(World::SIZE_Z % 2 == 0);
};

impl<'res> World<'res> {
    pub const SIZE_X: i32 = 16;
    pub const SIZE_Y: i32 = 4;
    pub const SIZE_Z: i32 = 16;

    pub const N_CHUNKS: usize =
        (Self::SIZE_X as usize) * (Self::SIZE_Y as usize) * (Self::SIZE_Z as usize);

    pub const CHUNK_ID_X_RANGE: Range<i32> = (-Self::SIZE_X / 2)..(Self::SIZE_X / 2);
    pub const CHUNK_ID_Y_RANGE: Range<i32> = (-Self::SIZE_Y / 2)..(Self::SIZE_Y / 2);
    pub const CHUNK_ID_Z_RANGE: Range<i32> = (-Self::SIZE_Z / 2)..(Self::SIZE_Z / 2);

    pub const COORD_X_RANGE: Range<i32> =
        Self::CHUNK_ID_X_RANGE.start * 32..Self::CHUNK_ID_X_RANGE.end * 32;
    pub const COORD_Y_RANGE: Range<i32> =
        Self::CHUNK_ID_Y_RANGE.start * 32..Self::CHUNK_ID_Y_RANGE.end * 32;
    pub const COORD_Z_RANGE: Range<i32> =
        Self::CHUNK_ID_Z_RANGE.start * 32..Self::CHUNK_ID_Z_RANGE.end * 32;

    fn index_for_chunk_id(chunk_id: ChunkId) -> usize {
        (chunk_id.y - Self::CHUNK_ID_Y_RANGE.start) as usize
            * Self::SIZE_Z as usize
            * Self::SIZE_X as usize
            + (chunk_id.z - Self::CHUNK_ID_Z_RANGE.start) as usize * Self::SIZE_X as usize
            + (chunk_id.x - Self::CHUNK_ID_X_RANGE.start) as usize
    }

    pub fn new(resources: &'res GameResources) -> Self {
        Self {
            chunks: vec![ChunkData::new(); Self::N_CHUNKS],
            chunk_meshes: vec_with(Self::N_CHUNKS, ChunkMesh::new),
            needs_chunk_building: true,
            chunk_builder: ChunkBuilder::new(resources),
        }
    }

    pub fn chunk_is_loaded(&self, chunk_id: ChunkId) -> bool {
        Self::CHUNK_ID_X_RANGE.contains(&chunk_id.x)
            && Self::CHUNK_ID_Y_RANGE.contains(&chunk_id.y)
            && Self::CHUNK_ID_Z_RANGE.contains(&chunk_id.z)
    }

    /// # Safety
    /// `chunk_id` must be a loaded chunk.
    pub unsafe fn get_chunk_with_mesh_unchecked(
        &self,
        chunk_id: ChunkId,
    ) -> (&ChunkData, &ChunkMesh) {
        let index = Self::index_for_chunk_id(chunk_id);
        unsafe {
            (
                self.chunks.get_unchecked(index).as_ref(),
                self.chunk_meshes.get_unchecked(index),
            )
        }
    }

    /// # Safety
    /// `chunk_id` must be a loaded chunk.
    pub unsafe fn get_chunk_with_mesh_unchecked_mut(
        &mut self,
        chunk_id: ChunkId,
    ) -> (&mut ChunkData, &mut ChunkMesh) {
        let index = Self::index_for_chunk_id(chunk_id);
        unsafe {
            (
                self.chunks.get_unchecked_mut(index).as_mut(),
                self.chunk_meshes.get_unchecked_mut(index),
            )
        }
    }

    pub fn get_chunk(&self, chunk_id: ChunkId) -> Option<&ChunkData> {
        self.chunk_is_loaded(chunk_id)
            .then(|| unsafe { self.get_chunk_with_mesh_unchecked(chunk_id).0 })
    }

    pub fn get_chunk_mut(&mut self, chunk_id: ChunkId) -> Option<&mut ChunkData> {
        self.chunk_is_loaded(chunk_id)
            .then(|| unsafe { self.get_chunk_with_mesh_unchecked_mut(chunk_id).0 })
    }

    pub fn get_chunk_mesh(&self, chunk_id: ChunkId) -> Option<&ChunkMesh> {
        self.chunk_is_loaded(chunk_id)
            .then(|| unsafe { self.get_chunk_with_mesh_unchecked(chunk_id).1 })
    }

    pub fn get_chunk_mesh_mut(&mut self, chunk_id: ChunkId) -> Option<&mut ChunkMesh> {
        self.chunk_is_loaded(chunk_id)
            .then(|| unsafe { self.get_chunk_with_mesh_unchecked_mut(chunk_id).1 })
    }

    pub fn get_chunk_with_mesh(&self, chunk_id: ChunkId) -> Option<(&ChunkData, &ChunkMesh)> {
        self.chunk_is_loaded(chunk_id)
            .then(|| unsafe { self.get_chunk_with_mesh_unchecked(chunk_id) })
    }

    pub fn get_chunk_with_mesh_mut(
        &mut self,
        chunk_id: ChunkId,
    ) -> Option<(&mut ChunkData, &mut ChunkMesh)> {
        self.chunk_is_loaded(chunk_id)
            .then(|| unsafe { self.get_chunk_with_mesh_unchecked_mut(chunk_id) })
    }

    fn build_chunk(&mut self, display: &impl glium::backend::Facade, chunk_id: ChunkId) {
        assert!(self.chunk_is_loaded(chunk_id));
        // Fuck you borrow checker.
        let index = Self::index_for_chunk_id(chunk_id);
        let chunk = unsafe { self.chunks.get_unchecked(index).as_ref() };
        let chunk_mesh = unsafe { self.chunk_meshes.get_unchecked_mut(index) };
        self.chunk_builder.build(display, chunk, chunk_mesh);
    }

    pub fn do_chunk_building(&mut self, display: &impl glium::backend::Facade) {
        if !self.needs_chunk_building {
            return;
        }
        for y in Self::CHUNK_ID_Y_RANGE {
            for z in Self::CHUNK_ID_Z_RANGE {
                for x in Self::CHUNK_ID_X_RANGE {
                    self.build_chunk(display, ChunkId::new(x, y, z));
                }
            }
        }
        self.needs_chunk_building = false;
    }

    pub fn world_to_local_coord(&self, world_coord: BlockCoord) -> (ChunkId, LocalCoord) {
        let chunk_id = world_coord.map(|i| i.div_euclid(32));
        let local_coord = world_coord.map(|i| i.rem_euclid(32) as u8);
        (chunk_id, local_coord)
    }

    pub fn get_block(&self, world_coord: BlockCoord) -> Option<BlockId> {
        let (chunk_id, local_coord) = self.world_to_local_coord(world_coord);
        let chunk = self.get_chunk(chunk_id)?;
        chunk.try_get_block(local_coord)
    }

    pub fn get_block_mut(&mut self, world_coord: BlockCoord) -> Option<&mut BlockId> {
        let (chunk_id, local_coord) = self.world_to_local_coord(world_coord);
        let chunk = self.get_chunk_mut(chunk_id)?;
        chunk.try_get_block_mut(local_coord)
    }
}
