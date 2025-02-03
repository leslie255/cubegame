use cgmath::*;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256StarStar;

use crate::{
    block::GameBlocks,
    game::GameResources,
    world::{BlockCoord, World},
};

#[derive(Debug, Clone, Copy)]
struct WaveComponent {
    amplitude: f32,
    /// Bigger is more gentle waves.
    /// `1` is wave with period of 2π.
    scale: f32,
    /// Positive is right shift.
    offset: Vector2<f32>,
}

impl WaveComponent {
    fn random(rng: &mut impl Rng, scale: f32, amplitude: f32) -> Self {
        const PI: f32 = std::f32::consts::PI;
        Self {
            amplitude,
            scale,
            offset: vec2(rng.random_range((-PI)..PI), rng.random_range((-PI)..PI)),
        }
    }

    fn sample(self, xz: Point2<f32>) -> f32 {
        let xz = (xz - self.offset).map(|i| i / self.scale);
        let mut result = 0.;
        result += f32::sin(xz.x) * self.amplitude;
        result += f32::sin(xz.y) * self.amplitude;
        result /= 2.;
        result
    }
}

#[derive(Debug)]
pub struct WorldGenerator<'res> {
    seed: u64,
    wave_components: [WaveComponent; 5],
    rng: Xoshiro256StarStar,
    game_blocks: &'res GameBlocks,
}

impl<'res> WorldGenerator<'res> {
    pub fn new(seed: u64, resources: &'res GameResources) -> Self {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        Self {
            seed,
            wave_components: [
                WaveComponent::random(&mut rng, 1.8, 0.6),
                WaveComponent::random(&mut rng, 1.5, 0.7),
                WaveComponent::random(&mut rng, 3.0, 2.5),
                WaveComponent::random(&mut rng, 7.0, 3.0),
                WaveComponent::random(&mut rng, 8.5, 5.3),
            ],
            rng,
            game_blocks: &resources.game_blocks,
        }
    }

    pub fn terrain_height_at(&self, x: i32, z: i32) -> i32 {
        let mut height = 0.0f32;
        for wave_component in self.wave_components {
            height += wave_component.sample(point2(x as f32, z as f32));
        }
        height += 6.;
        height.floor() as i32
    }

    pub fn generate_world(&mut self, world: &mut World) {
        for z in World::COORD_Z_RANGE {
            for x in World::COORD_X_RANGE {
                let terrain_height = self.terrain_height_at(x, z);
                for y in (-32)..(terrain_height - 4) {
                    if let Some(block) = world.get_block_mut(BlockCoord::new(x, y, z)) {
                        *block = self.game_blocks.stone;
                    }
                }
                for y in (terrain_height - 4)..(terrain_height) {
                    if let Some(block) = world.get_block_mut(BlockCoord::new(x, y, z)) {
                        *block = self.game_blocks.dirt;
                    }
                }
                if let Some(block) = world.get_block_mut(BlockCoord::new(x, terrain_height, z)) {
                    *block = if terrain_height < 0 {
                        self.game_blocks.sand
                    } else {
                        self.game_blocks.grass
                    };
                }
            }
        }

        // Features.
        let n_trees = (1.92 * (World::SIZE_X * World::SIZE_Z) as f32) as u32;
        for _ in 0..n_trees {
            let x = self.rng.random_range(World::COORD_X_RANGE);
            let z = self.rng.random_range(World::COORD_X_RANGE);
            self.place_tree(x, z, world);
        }

        // Features.
        let n_cherry_trees = (0.3 * (World::SIZE_X * World::SIZE_Z) as f32) as u32;
        for _ in 0..n_cherry_trees {
            let x = self.rng.random_range(World::COORD_X_RANGE);
            let z = self.rng.random_range(World::COORD_X_RANGE);
            self.place_cherry_tree(x, z, world);
        }
    }

    pub fn place_tree(&mut self, x: i32, z: i32, world: &mut World) {
        let terrain_height = self.terrain_height_at(x, z);
        let height = self.rng.random_range(1..4); // The exposed part of the trunk.

        // The dirt block below.
        if let Some(block) = world.get_block_mut(BlockCoord::new(x, terrain_height, z)) {
            if *block != self.game_blocks.grass {
                return;
            }
            *block = self.game_blocks.dirt;
        }

        // Trunk.
        for y in (terrain_height + 1)..=(terrain_height + height + 3) {
            if let Some(block) = world.get_block_mut(BlockCoord::new(x, y, z)) {
                *block = self.game_blocks.log;
            }
        }

        // Leaves.
        for y in (terrain_height + height + 1)..=(terrain_height + height + 4) {
            for dz in (-2)..=2i32 {
                for dx in (-2)..=2i32 {
                    if dz.abs() == 2 && dx.abs() == 2 {
                        continue;
                    }
                    let z = z + dz;
                    let x = x + dx;
                    if let Some(block) = world.get_block_mut(BlockCoord::new(x, y, z)) {
                        if *block != self.game_blocks.log && *block != self.game_blocks.cherry_log {
                            *block = self.game_blocks.leaves;
                        }
                    }
                }
            }
        }
        for dz in (-1)..=1i32 {
            for dx in (-1)..=1i32 {
                if dz.abs() == 1 && dx.abs() == 1 {
                    continue;
                }
                let z = z + dz;
                let x = x + dx;
                if let Some(block) =
                    world.get_block_mut(BlockCoord::new(x, terrain_height + height + 5, z))
                {
                    if *block != self.game_blocks.log && *block != self.game_blocks.cherry_log {
                        *block = self.game_blocks.leaves;
                    }
                }
            }
        }
    }

    pub fn place_cherry_tree(&mut self, x: i32, z: i32, world: &mut World) {
        let terrain_height = self.terrain_height_at(x, z);
        let height = self.rng.random_range(2..6); // The exposed part of the trunk.

        // The dirt block below.
        if let Some(block) = world.get_block_mut(BlockCoord::new(x, terrain_height, z)) {
            if *block != self.game_blocks.grass {
                return;
            }
            *block = self.game_blocks.dirt;
        }

        // Trunk.
        for y in (terrain_height + 1)..=(terrain_height + height + 3) {
            if let Some(block) = world.get_block_mut(BlockCoord::new(x, y, z)) {
                *block = self.game_blocks.cherry_log;
            }
        }

        // Leaves.
        for y in (terrain_height + height + 2)..=(terrain_height + height + 4) {
            for dz in (-3)..=3i32 {
                for dx in (-3)..=3i32 {
                    if dz.abs() == 3 && dx.abs() == 3 {
                        continue;
                    }
                    let z = z + dz;
                    let x = x + dx;
                    if let Some(block) = world.get_block_mut(BlockCoord::new(x, y, z)) {
                        if *block != self.game_blocks.log && *block != self.game_blocks.cherry_log {
                            *block = self.game_blocks.cherry_leaves;
                        }
                    }
                }
            }
        }
        for dz in (-2)..=2i32 {
            for dx in (-2)..=2i32 {
                if dz.abs() == 2 && dx.abs() == 2 {
                    continue;
                }
                let z = z + dz;
                let x = x + dx;
                if let Some(block) =
                    world.get_block_mut(BlockCoord::new(x, terrain_height + height + 1, z))
                {
                    if *block != self.game_blocks.log && *block != self.game_blocks.cherry_log {
                        *block = self.game_blocks.cherry_leaves;
                    }
                }
                if let Some(block) =
                    world.get_block_mut(BlockCoord::new(x, terrain_height + height + 5, z))
                {
                    if *block != self.game_blocks.log && *block != self.game_blocks.cherry_log {
                        *block = self.game_blocks.cherry_leaves;
                    }
                }
            }
        }
    }
}
