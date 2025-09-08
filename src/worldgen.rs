use std::{iter, ops::Range};

use noise::{Fbm, MultiFractal, NoiseFn, ScaleBias, ScalePoint, Simplex, SuperSimplex};
use rand::prelude::*;
use rand_xoshiro::Xoshiro128StarStar;

use crate::{
    block::GameBlocks,
    chunk::ChunkData,
    game::GameResources,
    world::{ChunkId, LocalCoordU8, World, WorldCoordI32},
};

/// Storage for a "strip" of chunk data - chunks that have the same X and Z component in their ID's.
/// Used for world gen to pass out a generated strip.
#[derive(Debug, Clone)]
pub struct ChunkDataStrip {
    /// The starting Y level (in chunk ID, not block coordinate).
    x: i32,
    z: i32,
    y_min: i32,
    chunks: Vec<Box<ChunkData>>,
}

impl ChunkDataStrip {
    pub fn new(x: i32, z: i32, y_range: Range<i32>) -> Self {
        Self {
            x,
            z,
            y_min: y_range.start,
            chunks: iter::repeat_with(ChunkData::new_boxed)
                .take((y_range.end - y_range.start) as usize)
                .collect(),
        }
    }

    fn index_for(&self, y: i32) -> Option<usize> {
        usize::try_from(y - self.y_min).ok()
    }

    pub fn get(&self, y: i32) -> Option<&ChunkData> {
        self.chunks.get(self.index_for(y)?).map(Box::as_ref)
    }

    pub fn get_mut(&mut self, y: i32) -> Option<&mut ChunkData> {
        let index = self.index_for(y)?;
        self.chunks.get_mut(index).map(Box::as_mut)
    }

    pub fn into_chunks(self) -> impl Iterator<Item = (ChunkId, Box<ChunkData>)> {
        self.chunks
            .into_iter()
            .enumerate()
            .map(move |(i, chunk_data)| {
                let y = (i as i32) + self.y_min;
                let chunk_id = ChunkId::new(self.x, y, self.z);
                (chunk_id, chunk_data)
            })
    }
}

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
        y_chunks: Range<i32>,
    ) -> ChunkDataStrip {
        let mut strip = ChunkDataStrip::new(x_chunk, z_chunk, y_chunks.clone());
        let y_coord_min = World::local_to_world_coord_axis(y_chunks.start, 0);
        for x_local in 0..32u8 {
            for z_local in 0..32u8 {
                let x_world = World::local_to_world_coord_axis(x_chunk, x_local);
                let z_world = World::local_to_world_coord_axis(z_chunk, z_local);
                let terrain_height = self.terrain_height_at(x_world, z_world);
                for y_world in y_coord_min..=(terrain_height.max(-2)) {
                    let (y_chunk, y_local) = World::world_to_local_coord_axis(y_world);
                    let dist_to_surface = terrain_height - y_world;
                    let block = match dist_to_surface {
                        ..0 => self.game_blocks.water,
                        0 if terrain_height <= 0 => self.game_blocks.sand,
                        0 => self.game_blocks.grass,
                        1..=4 => self.game_blocks.dirt,
                        _ => self.game_blocks.stone,
                    };
                    let local_coord = LocalCoordU8::new(x_local, y_local, z_local);
                    if let Some(chunk_data) = strip.get_mut(y_chunk)
                        && let Some(block_) = chunk_data.try_get_block_mut(local_coord)
                    {
                        *block_ = block;
                    }
                }
            }
        }
        strip
    }

    #[allow(unused_variables)]
    pub fn place_tree(&mut self, rng: &mut impl Rng, x: i32, z: i32, world: &World) {
        let terrain_height = self.terrain_height_at(x, z);
        let height = rng.random_range(1..4); // The exposed part of the trunk.

        // The dirt block below.
        world.with_block(WorldCoordI32::new(x, terrain_height, z), |block| {
            if *block != self.game_blocks.grass {
                return;
            }
            *block = self.game_blocks.dirt;
        });

        // Trunk.
        for y in (terrain_height + 1)..=(terrain_height + height + 3) {
            world.set_block(WorldCoordI32::new(x, y, z), self.game_blocks.log);
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
                    world.with_block(WorldCoordI32::new(x, y, z), |block| {
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
                    WorldCoordI32::new(x, terrain_height + height + 5, z),
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
        world.with_block(WorldCoordI32::new(x, terrain_height, z), |block| {
            if *block != self.game_blocks.grass {
                return;
            }
            *block = self.game_blocks.dirt;
        });

        // Trunk.
        for y in (terrain_height + 1)..=(terrain_height + height + 3) {
            world.set_block(WorldCoordI32::new(x, y, z), self.game_blocks.cherry_log);
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
                    world.with_block(WorldCoordI32::new(x, y, z), |block| {
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
                    WorldCoordI32::new(x, terrain_height + height + 1, z),
                    |block| {
                        if *block != self.game_blocks.log && *block != self.game_blocks.cherry_log {
                            *block = self.game_blocks.cherry_leaves;
                        }
                    },
                );
                world.with_block(
                    WorldCoordI32::new(x, terrain_height + height + 5, z),
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
