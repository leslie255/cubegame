use std::iter;

use noise::{Fbm, MultiFractal, NoiseFn, ScaleBias, ScalePoint, Simplex, SuperSimplex};
use rand::prelude::*;
use rand_xoshiro::Xoshiro128StarStar;

use crate::{
    block::GameBlocks,
    chunk::{ChunkData, LocalCoord},
    game::GameResources,
    world::{BlockCoord, ChunkId, World},
};

#[derive(Debug)]
pub struct NoiseComponent {}

pub struct WorldGenerator<'res> {
    seed: u64,
    terrain_noise_components: Vec<Box<dyn NoiseFn<f64, 2>>>,
    game_blocks: &'res GameBlocks,
}

impl<'res> std::fmt::Debug for WorldGenerator<'res> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorldGenerator")
            .field("seed", &self.seed)
            .field("game_blocks", &self.game_blocks)
            .finish_non_exhaustive()
    }
}

impl<'res> WorldGenerator<'res> {
    pub fn new(seed: u64, resources: &'res GameResources) -> Self {
        println!("[INFO] seed: {seed}");
        let mut rng_simplex_positive = Xoshiro128StarStar::seed_from_u64(seed);
        let mut rng_simplex_negative = Xoshiro128StarStar::seed_from_u64(seed);
        fn scale(
            noise: impl NoiseFn<f64, 2> + 'static,
            input_scale: f64,
            output_scale: f64,
        ) -> Box<dyn NoiseFn<f64, 2>> {
            let input_scaled = ScalePoint::new(noise).set_scale(input_scale);
            let input_output_scaled = ScaleBias::<_, _, 2>::new(input_scaled)
                .set_scale(output_scale)
                .set_bias(0.15 * output_scale);
            Box::new(input_output_scaled)
        }
        let mut simplex = |input_scale, output_scale| {
            let rng = match input_scale {
                ..=0.0 => &mut rng_simplex_negative,
                _ => &mut rng_simplex_positive,
            };
            let noise = SuperSimplex::new(rng.next_u32());
            scale(noise, input_scale, output_scale)
        };
        let mut rng_fbm = Xoshiro128StarStar::seed_from_u64(seed);
        let mut fbm = |input_scale, output_scale| {
            let sources = [
                Simplex::new(rng_fbm.next_u32()),
                Simplex::new(rng_fbm.next_u32()),
                Simplex::new(rng_fbm.next_u32()),
                Simplex::new(rng_fbm.next_u32()),
            ];
            let noise = Fbm::new(rng_fbm.next_u32())
                .set_octaves(sources.len())
                .set_sources(Vec::from(sources));
            scale(noise, input_scale, output_scale)
        };
        let si = 0.016;
        let so = 10.0;
        let terrain_noise_components: [Box<dyn NoiseFn<f64, 2>>; _] = [
            simplex(si * 1.5f64.powf(-4.5), so * 0.75f64.powf(-2.0)),
            simplex(si * 1.5f64.powf(-3.0), so * 0.75f64.powf(-1.25)),
            simplex(si * 1.5f64.powf(-1.5), so * 0.75f64.powf(-0.75)),
            simplex(si * 1.5f64.powf(0.0), so * 0.75f64.powf(0.0)),
            simplex(si * 1.5f64.powf(1.5), so * 0.75f64.powf(2.5)),
            simplex(si * 1.5f64.powf(3.0), so * 0.75f64.powf(5.0)),
            simplex(si * 1.5f64.powf(4.5), so * 0.75f64.powf(7.5)),
            fbm(si, so),
        ];
        Self {
            seed,
            terrain_noise_components: Vec::from_iter(terrain_noise_components),
            game_blocks: &resources.game_blocks,
        }
    }

    pub fn terrain_height_at(&self, x: i32, z: i32) -> i32 {
        let mut height = 0.;
        for noise_component in &self.terrain_noise_components {
            height += noise_component.get([x as f64, z as f64]);
        }
        height.round() as i32
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
                for y_world in World::COORD_Y_RANGE.start..=(terrain_height.max(-2)) {
                    if !World::COORD_Y_RANGE.contains(&y_world) {
                        break;
                    }
                    let (y_chunk, y_local) = World::world_to_local_coord_axis(y_world);
                    let dist_to_surface = terrain_height - y_world;
                    let block = match dist_to_surface {
                        ..0 => self.game_blocks.water,
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
                    unsafe {
                        // SAFETY:
                        // - X and Z is bounded in 0..32 from the for loop range.
                        // - Y is bounded by a check ealier.
                        *chunk_data.get_block_unchecked_mut(local_coord) = block;
                    }
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
    }

    #[allow(unused_variables)]
    pub fn place_tree(&mut self, rng: &mut impl Rng, x: i32, z: i32, world: &World) {
        let terrain_height = self.terrain_height_at(x, z);
        let height = rng.random_range(1..4); // The exposed part of the trunk.

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
    pub fn place_cherry_tree(&mut self, rng: &mut impl Rng, x: i32, z: i32, world: &World) {
        let terrain_height = self.terrain_height_at(x, z);
        let height = rng.random_range(2..6); // The exposed part of the trunk.

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
