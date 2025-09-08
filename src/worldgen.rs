use std::iter;

use cgmath::*;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256StarStar;

use crate::{
    block::GameBlocks,
    chunk::{ChunkData, LocalCoord},
    game::GameResources,
    world::{BlockCoord, ChunkId, World},
};

#[derive(Debug, Clone, Copy)]
struct WaveComponent {
    amplitude: f32,
    /// Bigger is more gentle waves.
    /// `1` is wave with period of 2.
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
        let xz = (xz - self.offset).div_element_wise(self.scale);
        (f32::sin(xz.x) + f32::sin(xz.y)) / 2. * self.amplitude
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

    pub fn generate_strip(
        &self,
        x_chunk: i32,
        z_chunk: i32,
        chunks_negative_y: &mut [Box<ChunkData>],
        chunks_positive_y: &mut [Box<ChunkData>],
    ) {
        for x_local in 0..32u8 {
            for z_local in 0..32u8 {
                let x_world = World::local_to_world_coord_axis(x_chunk, x_local);
                let z_world = World::local_to_world_coord_axis(z_chunk, z_local);
                let terrain_height = self.terrain_height_at(x_world, z_world);
                for y_world in World::COORD_Y_RANGE.start..=terrain_height {
                    let (y_chunk, y_local) = World::world_to_local_coord_axis(y_world);
                    let dist_to_surface = terrain_height - y_world;
                    let block = match dist_to_surface {
                        0 if terrain_height <= 0 => self.game_blocks.sand,
                        0 => self.game_blocks.grass,
                        1..=4 => self.game_blocks.dirt,
                        _ => self.game_blocks.stone,
                    };
                    let local_coord = LocalCoord::new(x_local, y_local, z_local);
                    let chunk_data = match y_chunk {
                        ..0 => &mut chunks_negative_y[(-y_chunk) as usize - 1],
                        0.. => &mut chunks_positive_y[y_chunk as usize],
                    };
                    *chunk_data.get_block_mut(local_coord) = block;
                }
            }
        }
    }

    /// Generates the world and builds the chunks into meshes.
    pub fn generate_world(&mut self, world: &World) {
        println!("[DEBUG] begining worldgen");
        for z_chunk in World::CHUNK_ID_Z_RANGE {
            for x_chunk in World::CHUNK_ID_X_RANGE {
                let mut chunks_positive_y: Vec<Box<ChunkData>> =
                    iter::repeat_with(ChunkData::new_boxed)
                        .take((World::SIZE_Y / 2) as usize)
                        .collect();
                let mut chunks_negative_y: Vec<Box<ChunkData>> =
                    iter::repeat_with(ChunkData::new_boxed)
                        .take((World::SIZE_Y / 2) as usize)
                        .collect();
                self.generate_strip(
                    x_chunk,
                    z_chunk,
                    &mut chunks_negative_y,
                    &mut chunks_positive_y,
                );
                for (i, chunk_data) in chunks_positive_y.into_iter().enumerate() {
                    let y_chunk = i as i32;
                    let chunk_id = ChunkId::new(x_chunk, y_chunk, z_chunk);
                    world.chunks().insert_chunk(chunk_id, chunk_data);
                }
                for (i, chunk_data) in chunks_negative_y.into_iter().enumerate() {
                    let y_chunk = -(i as i32 + 1);
                    let chunk_id = ChunkId::new(x_chunk, y_chunk, z_chunk);
                    world.chunks().insert_chunk(chunk_id, chunk_data);
                }
            }
        }
        println!("[DEBUG] finished worldgen");
        // let y_start = World::COORD_Y_RANGE.start;
        // for z in World::COORD_Z_RANGE {
        //     for x in World::COORD_X_RANGE {
        //         let terrain_height = self.terrain_height_at(x, z);
        //         for y in y_start..(terrain_height - 4) {
        //             world.set_block(BlockCoord::new(x, y, z), self.game_blocks.stone);
        //         }
        //         for y in (terrain_height - 4)..(terrain_height) {
        //             world.set_block(BlockCoord::new(x, y, z), self.game_blocks.dirt);
        //         }
        //         if terrain_height < 0 {
        //             world.set_block(BlockCoord::new(x, terrain_height, z), self.game_blocks.sand);
        //         } else {
        //             world.set_block(
        //                 BlockCoord::new(x, terrain_height, z),
        //                 self.game_blocks.grass,
        //             );
        //         }
        //     }
        // }

        // // Features.

        // let n_trees = (1.92 * (World::SIZE_X * World::SIZE_Z) as f32) as u32;
        // for _ in 0..n_trees {
        //     let x = self.rng.random_range(World::COORD_X_RANGE);
        //     let z = self.rng.random_range(World::COORD_X_RANGE);
        //     self.place_tree(x, z, world);
        // }

        // let n_cherry_trees = (0.3 * (World::SIZE_X * World::SIZE_Z) as f32) as u32;
        // for _ in 0..n_cherry_trees {
        //     let x = self.rng.random_range(World::COORD_X_RANGE);
        //     let z = self.rng.random_range(World::COORD_X_RANGE);
        //     self.place_cherry_tree(x, z, world);
        // }
    }

    #[allow(unused_variables)]
    pub fn place_tree(&mut self, x: i32, z: i32, world: &World) {
        let terrain_height = self.terrain_height_at(x, z);
        let height = self.rng.random_range(1..4); // The exposed part of the trunk.

        // The dirt block below.
        world.with_block(BlockCoord::new(x, terrain_height, z), |block| {
            if *block != self.game_blocks.grass {
                return;
            }
            *block = self.game_blocks.dirt;
        });

        // Trunk.
        for y in (terrain_height + 1)..=(terrain_height + height + 3) {
            world.set_block(BlockCoord::new(x, y, z), self.game_blocks.log);
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
                    world.with_block(BlockCoord::new(x, y, z), |block| {
                        if *block != self.game_blocks.log && *block != self.game_blocks.cherry_log {
                            *block = self.game_blocks.leaves;
                        }
                    });
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
                world.with_block(
                    BlockCoord::new(x, terrain_height + height + 5, z),
                    |block| {
                        if *block != self.game_blocks.log && *block != self.game_blocks.cherry_log {
                            *block = self.game_blocks.leaves;
                        }
                    },
                );
            }
        }
    }

    #[allow(unused_variables)]
    pub fn place_cherry_tree(&mut self, x: i32, z: i32, world: &World) {
        let terrain_height = self.terrain_height_at(x, z);
        let height = self.rng.random_range(2..6); // The exposed part of the trunk.

        // The dirt block below.
        world.with_block(BlockCoord::new(x, terrain_height, z), |block| {
            if *block != self.game_blocks.grass {
                return;
            }
            *block = self.game_blocks.dirt;
        });

        // Trunk.
        for y in (terrain_height + 1)..=(terrain_height + height + 3) {
            world.set_block(BlockCoord::new(x, y, z), self.game_blocks.cherry_log);
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
                    world.with_block(BlockCoord::new(x, y, z), |block| {
                        if *block != self.game_blocks.log && *block != self.game_blocks.cherry_log {
                            *block = self.game_blocks.cherry_leaves;
                        }
                    });
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
                world.with_block(
                    BlockCoord::new(x, terrain_height + height + 1, z),
                    |block| {
                        if *block != self.game_blocks.log && *block != self.game_blocks.cherry_log {
                            *block = self.game_blocks.cherry_leaves;
                        }
                    },
                );
                world.with_block(
                    BlockCoord::new(x, terrain_height + height + 5, z),
                    |block| {
                        if *block != self.game_blocks.log && *block != self.game_blocks.cherry_log {
                            *block = self.game_blocks.cherry_leaves;
                        }
                    },
                );
            }
        }
    }
}
