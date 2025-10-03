use std::{any::type_name, ops::Deref};

use derive_more::From;
use rand::{Rng as _, RngCore};

use crate::{
    block::{BlockId, GameBlocks},
    utils::WithY,
    world::{World, WorldCoordI32},
    worldgen::ColumnData,
};

pub type DynSurfaceFeatureGenerator<'cx> = Box<dyn SurfaceFeature + Send + Sync + 'cx>;

/// Feature that is smaller than one column.
pub trait SurfaceFeature {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    /// Number of rolls
    fn rolls(&self) -> u32;

    /// Chance per roll.
    /// Clamped between 0.0 and 1.0.
    fn chance(&self) -> f32;

    fn generate(&self, origin: WorldCoordI32, rng: &mut dyn RngCore, target: GenerationTarget);
}

impl<T: Deref> SurfaceFeature for T
where
    <T as Deref>::Target: SurfaceFeature,
{
    fn name(&self) -> &str {
        self.deref().name()
    }

    fn rolls(&self) -> u32 {
        self.deref().rolls()
    }

    fn chance(&self) -> f32 {
        self.deref().chance()
    }

    fn generate(&self, origin: WorldCoordI32, rng: &mut dyn RngCore, target: GenerationTarget) {
        self.deref().generate(origin, rng, target)
    }
}

/// Target of a feature generation.
#[derive(Debug, From)]
pub enum GenerationTarget<'a, 'scope, 'cx> {
    World(&'a World<'scope, 'cx>),
    Column(&'a mut ColumnData),
}

impl<'a, 'scope, 'cx> GenerationTarget<'a, 'scope, 'cx> {
    pub fn with_block<T>(
        &mut self,
        coord: WorldCoordI32,
        f: impl FnOnce(&mut BlockId) -> T,
    ) -> Option<T> {
        match self {
            GenerationTarget::World(world) => world.with_block(coord, f),
            GenerationTarget::Column(column) => {
                let (chunk_id, local_coord) = World::world_to_local_coord(coord);
                if (chunk_id.x != column.x) | (chunk_id.z != column.z) {
                    return None;
                }
                let chunk = column.get_mut(chunk_id.y)?;
                let block = chunk.try_get_block_mut(local_coord)?;
                Some(f(block))
            }
        }
    }

    pub fn get_block(&mut self, coord: WorldCoordI32) -> Option<BlockId> {
        self.with_block(coord, |block| *block)
    }

    pub fn set_block(&mut self, coord: WorldCoordI32, block: BlockId) -> Option<BlockId> {
        self.with_block(coord, |block_mut| {
            let old_block = *block_mut;
            *block_mut = block;
            old_block
        })
    }
}

#[derive(Debug, Clone)]
pub struct TreeFeature<'cx> {
    game_blocks: &'cx GameBlocks,
}

impl<'cx> TreeFeature<'cx> {
    pub fn new(game_blocks: &'cx GameBlocks) -> Self {
        Self { game_blocks }
    }
}

impl SurfaceFeature for TreeFeature<'_> {
    fn rolls(&self) -> u32 {
        30
    }

    fn chance(&self) -> f32 {
        0.2
    }

    fn generate(&self, origin: WorldCoordI32, rng: &mut dyn RngCore, mut target: GenerationTarget) {
        // The dirt block below.
        target.set_block(origin, self.game_blocks.dirt);

        let height = rng.random_range(1..=3i32);

        // Trunk.
        for y in (origin.y + 1)..=(origin.y + height + 2) {
            let coord = origin.with_y(y);
            target.with_block(coord, |block| {
                if *block == self.game_blocks.air
                    || *block == self.game_blocks.water
                    || *block == self.game_blocks.leaves
                    || *block == self.game_blocks.cherry_leaves
                {
                    *block = self.game_blocks.log;
                }
            });
        }

        let place_leave = |target: &mut GenerationTarget, coord: WorldCoordI32| {
            target.with_block(coord, |block| {
                if *block == self.game_blocks.air || *block == self.game_blocks.water {
                    *block = self.game_blocks.leaves;
                }
            });
        };

        // Leaves.
        for y in (origin.y + height + 1)..=(origin.y + height + 3) {
            for dz in (-2)..=2i32 {
                for dx in (-2)..=2i32 {
                    if dz.abs() >= 2 && dx.abs() >= 2 {
                        continue;
                    }
                    let coord = WorldCoordI32::new(origin.x + dx, y, origin.z + dz);
                    place_leave(&mut target, coord);
                }
            }
        }
        for y in (origin.y + height + 4)..=(origin.y + height + 5) {
            for dz in (-1)..=1i32 {
                for dx in (-1)..=1i32 {
                    if dz.abs() == 1 && dx.abs() == 1 {
                        continue;
                    }
                    let coord = WorldCoordI32::new(origin.x + dx, y, origin.z + dz);
                    place_leave(&mut target, coord);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct CherryTreeFeature<'cx> {
    game_blocks: &'cx GameBlocks,
}

impl<'cx> CherryTreeFeature<'cx> {
    pub fn new(game_blocks: &'cx GameBlocks) -> Self {
        Self { game_blocks }
    }
}

impl SurfaceFeature for CherryTreeFeature<'_> {
    fn rolls(&self) -> u32 {
        8
    }

    fn chance(&self) -> f32 {
        0.0125
    }

    fn generate(&self, origin: WorldCoordI32, rng: &mut dyn RngCore, mut target: GenerationTarget) {
        // The dirt block below.
        target.set_block(origin, self.game_blocks.dirt);

        let height = rng.random_range(1..=3i32);

        // Trunk.
        for y in (origin.y + 1)..=(origin.y + height + 2) {
            let coord = origin.with_y(y);
            target.with_block(coord, |block| {
                if *block == self.game_blocks.air
                    || *block == self.game_blocks.water
                    || *block == self.game_blocks.leaves
                    || *block == self.game_blocks.cherry_leaves
                {
                    *block = self.game_blocks.cherry_log;
                }
            });
        }

        let place_leave = |target: &mut GenerationTarget, coord: WorldCoordI32| {
            target.with_block(coord, |block| {
                if *block == self.game_blocks.air || *block == self.game_blocks.water {
                    *block = self.game_blocks.cherry_leaves;
                }
            });
        };

        // Leaves.
        for y in (origin.y + height + 1)..=(origin.y + height + 3) {
            for dz in (-2)..=2i32 {
                for dx in (-2)..=2i32 {
                    if dz.abs() >= 2 && dx.abs() >= 2 {
                        continue;
                    }
                    let coord = WorldCoordI32::new(origin.x + dx, y, origin.z + dz);
                    place_leave(&mut target, coord);
                }
            }
        }
        for y in (origin.y + height + 4)..=(origin.y + height + 5) {
            for dz in (-1)..=1i32 {
                for dx in (-1)..=1i32 {
                    if dz.abs() == 1 && dx.abs() == 1 {
                        continue;
                    }
                    let coord = WorldCoordI32::new(origin.x + dx, y, origin.z + dz);
                    place_leave(&mut target, coord);
                }
            }
        }
    }
}
