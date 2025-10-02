use std::alloc::Layout;

use bytemuck::Zeroable;

mod client;

pub use client::*;

use crate::{block::BlockId, world::LocalCoordU8};

#[derive(Debug)]
pub struct Chunk {
    /// Note that `data` is boxed here because `ChunkData` almost always exist boxed due to its
    /// size.
    pub data: Box<ChunkData>,
    pub client: ClientChunk,
}

#[derive(Debug, Clone, Zeroable)]
pub struct ChunkData {
    blocks: [BlockId; 32 * 32 * 32],
}

impl ChunkData {
    /// A new chunk filled with block ID 0.
    pub fn new_boxed() -> Box<Self> {
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

    fn index(local_coord: LocalCoordU8) -> usize {
        (local_coord.y as usize) * 32 * 32
            + (local_coord.z as usize) * 32
            + (local_coord.x as usize)
    }

    pub fn try_get_block(&self, local_coord: LocalCoordU8) -> Option<BlockId> {
        self.blocks.get(Self::index(local_coord)).copied()
    }

    pub fn try_get_block_mut(&mut self, local_coord: LocalCoordU8) -> Option<&mut BlockId> {
        self.blocks.get_mut(Self::index(local_coord))
    }

    #[track_caller]
    pub fn get_block(&self, local_coord: LocalCoordU8) -> BlockId {
        self.try_get_block(local_coord).unwrap()
    }

    #[track_caller]
    pub fn get_block_mut(&mut self, local_coord: LocalCoordU8) -> &mut BlockId {
        self.try_get_block_mut(local_coord).unwrap()
    }

    /// # Safety
    /// `local_coord` must be in range.
    pub unsafe fn get_block_unchecked(&self, local_coord: LocalCoordU8) -> BlockId {
        unsafe { *self.blocks.get_unchecked(Self::index(local_coord)) }
    }

    /// # Safety
    /// `local_coord` must be in range.
    pub unsafe fn get_block_unchecked_mut(&mut self, local_coord: LocalCoordU8) -> &mut BlockId {
        unsafe { self.blocks.get_unchecked_mut(Self::index(local_coord)) }
    }
}
