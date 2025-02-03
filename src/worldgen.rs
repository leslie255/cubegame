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
    fn random(rng: &mut impl Rng, base_scale: f32, base_amplitude: f32) -> Self {
        const PI: f32 = std::f32::consts::PI;
        Self {
            amplitude: base_amplitude * rng.random_range(0.9..1.1),
            scale: base_scale * rng.random_range(0.9..1.1),
            offset: vec2(rng.random_range((-PI)..PI), rng.random_range((-PI)..PI)),
        }
    }

    fn sample(self, xz: Point2<f32>) -> f32 {
        let xz = xz - self.offset;
        let mut result = 0.;
        result += f32::sin(xz.x / self.scale) * self.amplitude;
        result += f32::sin(xz.y / self.scale) * self.amplitude;
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
                WaveComponent::random(&mut rng, 0.5, 0.7),
                WaveComponent::random(&mut rng, 1.5, 0.8),
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
                    *block = self.game_blocks.grass;
                }
            }
        }
    }
}
