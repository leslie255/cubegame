use crate::{world::ChunkId, worldgen::WorldGenerator};

/// Feature that is smaller in horizontal area than 32x32.
pub trait SmallFeature {
    /// Number of rolls for a chunk.
    fn rolls(&self, chunk_id: ChunkId) -> u32;
    /// Chance per roll.
    fn chance(&self, chunk_id: ChunkId, worldgen: &WorldGenerator<'_>) -> f32;
}


