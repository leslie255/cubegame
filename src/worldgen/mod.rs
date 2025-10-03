use std::{fmt::Debug, iter, ops::Range};

use noise::{Fbm, MultiFractal, NoiseFn, ScaleBias, ScalePoint, SuperSimplex};
use rand::prelude::*;
use rand_xoshiro::{Xoshiro128Plus, Xoshiro128StarStar, Xoshiro512Plus};

use crate::{
    block::GameBlocks,
    chunk::ChunkData,
    game::GameResources,
    world::{ChunkId, LocalCoordU8, World, WorldCoordI32},
};

mod feature;

pub use feature::*;

// fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
//     let x = f64::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
//     x * x * x * (x * (6.0 * x - 15.0) + 10.0)
// }

/// Storage for a "column" of chunk data - chunks that have the same X and Z component in their ID's.
/// Used for world gen to pass out a generated column.
#[derive(Debug, Clone)]
pub struct ColumnData {
    /// The starting Y level (in chunk ID, not block coordinate).
    x: i32,
    z: i32,
    y_min: i32,
    chunks: Vec<Box<ChunkData>>,
}

impl ColumnData {
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

type DynNoise = Box<dyn NoiseFn<f64, 2> + Send + Sync + 'static>;

#[derive(derive_more::Debug)]
pub struct WorldGenerator<'cx> {
    game_blocks: &'cx GameBlocks,
    seed: u64,
    #[debug(skip)]
    terrain_noises: Vec<DynNoise>,
    #[debug(skip)]
    surface_features: Vec<DynSurfaceFeatureGenerator<'cx>>,
}

impl<'cx> WorldGenerator<'cx> {
    pub fn new(seed: u64, resources: &'cx GameResources) -> Self {
        log::info!("seed: {seed}");
        fn scale(
            noise: impl NoiseFn<f64, 2> + Send + Sync + 'static,
            input_scale: f64,
            output_scale: f64,
            bias: f64,
        ) -> DynNoise {
            let input_scaled = ScalePoint::new(noise).set_scale(input_scale);
            let input_output_scaled = ScaleBias::<_, _, 2>::new(input_scaled)
                .set_scale(output_scale)
                .set_bias(bias);
            Box::new(input_output_scaled)
        }
        let mut rng_fbm = Xoshiro128StarStar::seed_from_u64(seed);
        let mut fbm = |input_scale, output_scale| {
            let sources = [
                SuperSimplex::new(rng_fbm.next_u32()),
                SuperSimplex::new(rng_fbm.next_u32()),
                SuperSimplex::new(rng_fbm.next_u32()),
                SuperSimplex::new(rng_fbm.next_u32()),
            ];
            let noise = Fbm::new(rng_fbm.next_u32())
                .set_octaves(sources.len())
                .set_sources(Vec::from(sources));
            scale(noise, input_scale, output_scale, 0.0)
        };
        let si = 0.016;
        let so = 10.0;
        let terrain_noises: [DynNoise; _] = [
            fbm(si * 1.5f64.powf(-4.5), so * 0.75f64.powf(-2.0)),
            fbm(si * 1.5f64.powf(-3.0), so * 0.75f64.powf(-1.25)),
            fbm(si * 1.5f64.powf(-1.5), so * 0.75f64.powf(-0.75)),
            fbm(si * 1.5f64.powf(0.0), so * 0.75f64.powf(0.0)),
        ];

        let game_blocks = &resources.game_blocks;

        let surface_features: [DynSurfaceFeatureGenerator<'cx>; _] = [
            Box::new(TreeFeature::new(game_blocks)),
            Box::new(CherryTreeFeature::new(game_blocks)),
        ];

        Self {
            seed,
            game_blocks,
            terrain_noises: Vec::from_iter(terrain_noises),
            surface_features: Vec::from_iter(surface_features),
        }
    }

    pub fn terrain_height_at(&self, x: i32, z: i32) -> i32 {
        let terrain: f64 = self
            .terrain_noises
            .iter()
            .map(|component| component.get([x as f64, z as f64]))
            .sum();
        terrain.floor() as i32
    }

    pub fn generate_column(&self, x_chunk: i32, z_chunk: i32, y_chunks: Range<i32>) -> ColumnData {
        let mut column = ColumnData::new(x_chunk, z_chunk, y_chunks.clone());

        // Terrain.
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
                    if let Some(chunk_data) = column.get_mut(y_chunk)
                        && let Some(block_) = chunk_data.try_get_block_mut(local_coord)
                    {
                        *block_ = block;
                    }
                }
            }
        }

        // Features.
        let mut rng_salt = Xoshiro512Plus::seed_from_u64(self.seed);
        for surface_feature in &self.surface_features {
            let salt = rng_salt.next_u64();
            for dx in (-1)..=1 {
                for dz in (-1)..=1 {
                    let x_chunk = x_chunk + dx;
                    let z_chunk = z_chunk + dz;
                    self.generate_feature(x_chunk, z_chunk, salt, &mut column, &surface_feature);
                }
            }
        }

        column
    }

    fn generate_feature(
        &self,
        x_chunk: i32,
        z_chunk: i32,
        salt: u64,
        column: &mut ColumnData,
        surface_feature: &dyn SurfaceFeature,
    ) {
        let local_seed = {
            let mut seed = [0u8; 16];
            let salt_higher: i32 = bytemuck::cast((salt & 0xFFFFFFFF) as u32);
            let salt_lower: i32 = bytemuck::cast(((salt >> 16) & 0xFFFFFFFF) as u32);
            let (ab, c) = seed.split_at_mut(8);
            let (a, b) = ab.split_at_mut(4);
            a.copy_from_slice(&(x_chunk + salt_higher).to_le_bytes());
            b.copy_from_slice(&(z_chunk + salt_lower).to_le_bytes());
            c.copy_from_slice(&self.seed.to_le_bytes());
            seed
        };
        let mut rng = Xoshiro128Plus::from_seed(local_seed);
        for _ in 0..20 {
            // Rinse the RNG a few times before using for good hygiene.
            // (xoshiro is incensitive to lower bits initially).
            rng.next_u64();
        }
        let chance = surface_feature.chance().clamp(0.0, 1.0);
        let count: u32 = (0..surface_feature.rolls())
            .map(|_| {
                let predicate: f32 = rng.random_range(0.0..1.0);
                u32::from(predicate < chance)
            })
            .sum();
        for _ in 0..count {
            let x_local = rng.random_range(0..32);
            let z_local = rng.random_range(0..32);
            let x_world = World::local_to_world_coord_axis(x_chunk, x_local);
            let z_world = World::local_to_world_coord_axis(z_chunk, z_local);
            let terrain_height = self.terrain_height_at(x_world, z_world);
            if terrain_height > 0 {
                let origin = WorldCoordI32::new(x_world, terrain_height, z_world);
                surface_feature.generate(origin, &mut rng, column.into());
            }
        }
    }
}
