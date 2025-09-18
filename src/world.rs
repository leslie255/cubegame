use std::{
    collections::HashMap,
    mem,
    ops::Range,
    sync::{Arc, Mutex, RwLock, mpmc},
    thread,
};

use cgmath::*;

use crate::{
    ProgramArgs,
    block::BlockId,
    chunk::{Chunk, ChunkBuilder, ClientChunk},
    game::GameResources,
    utils::*,
    worldgen::WorldGenerator,
};

pub type LocalCoordU8 = Point3<u8>;

pub type LocalCoordF32 = Point3<f32>;

pub type ChunkId = Point3<i32>;

pub type WorldCoordI32 = Point3<i32>;

pub type WorldCoordF32 = Point3<f32>;

/// A task for the chunk workers.
#[derive(Debug, Clone)]
pub enum ChunkBuildingTask {
    Rebuild { chunk_id: ChunkId },
    Generate { x: i32, z: i32, y_range: Range<i32> },
}

/// Manages loading and unloading chunks.
#[derive(Debug)]
pub struct ChunkManager {
    chunks: RwLock<HashMap<ChunkId, Arc<Mutex<Chunk>>>>,
}

impl ChunkManager {
    pub fn new() -> Self {
        let chunks = HashMap::new();
        Self {
            chunks: RwLock::new(chunks),
        }
    }

    /// Loop through each loaded chunk.
    pub fn for_each_loaded_chunk(&self, mut f: impl FnMut(ChunkId, &mut Chunk)) {
        self.for_each_loaded_chunk_id(|chunk_id| {
            let chunk = {
                let chunks = self.chunks.read().unwrap();
                Arc::clone(chunks.get(&chunk_id).unwrap())
            };
            f(chunk_id, &mut chunk.lock().unwrap());
        });
    }

    /// Loop through each loaded chunk's ID but not the chunk.
    pub fn for_each_loaded_chunk_id(&self, mut f: impl FnMut(ChunkId)) {
        let loaded_chunk_ids: Vec<ChunkId> = {
            let chunks = self.chunks.read().unwrap();
            chunks.keys().copied().collect()
        };
        for chunk_id in loaded_chunk_ids {
            if self.chunk_is_loaded(chunk_id) {
                f(chunk_id);
            }
        }
    }

    /// Whether a chunk of the provided ID is loaded.
    pub fn chunk_is_loaded(&self, chunk_id: ChunkId) -> bool {
        self.chunks.read().unwrap().contains_key(&chunk_id)
    }

    pub fn get_loaded_chunk(&self, chunk_id: ChunkId) -> Option<Arc<Mutex<Chunk>>> {
        let chunks = self.chunks.read().unwrap();
        chunks.get(&chunk_id).map(Arc::clone)
    }

    /// Get a mutable reference to a loaded chunk.
    pub fn with_loaded_chunk<T>(
        &self,
        chunk_id: ChunkId,
        f: impl FnOnce(&mut Chunk) -> T,
    ) -> Option<T> {
        let chunk = self.get_loaded_chunk(chunk_id)?;
        let mut chunk_locked = chunk.lock().unwrap();
        Some(f(&mut chunk_locked))
    }

    /// Insert a new chunk to the pool of loaded chunks.
    /// Replaces the old chunk if existed one.
    pub fn insert_chunk(&self, chunk_id: ChunkId, chunk: Chunk) {
        self.chunks
            .write()
            .unwrap()
            .insert(chunk_id, arc_mutex(chunk));
    }

    /// NOP if chunk already didn't exist
    pub fn remove_chunk(&self, chunk_id: ChunkId) {
        self.chunks.write().unwrap().remove(&chunk_id);
    }
}

impl Default for ChunkManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Manages chunk workers.
#[derive(Debug)]
pub struct ChunkWorkerPool<'scope, 'cx>
where
    'cx: 'scope,
{
    workers: Vec<thread::ScopedJoinHandle<'scope, ()>>,
    device: &'cx wgpu::Device,
    chunk_builder: ChunkBuilder<'cx>,
    thread_scope: &'scope thread::Scope<'scope, 'cx>,
    chunks: Arc<ChunkManager>,
    /// TX for sending chunk tasks to the chunk workers.
    tasks_tx: mpmc::Sender<ChunkBuildingTask>,
    /// RX for sending chunk workers to receive their tasks.
    /// Kept alive in the worker pool to create more chunk workers.
    tasks_rx: mpmc::Receiver<ChunkBuildingTask>,
}

impl<'scope, 'cx> ChunkWorkerPool<'scope, 'cx>
where
    'cx: 'scope,
{
    pub fn new(
        device: &'cx wgpu::Device,
        resources: &'cx GameResources,
        thread_scope: &'scope thread::Scope<'scope, 'cx>,
        chunks: Arc<ChunkManager>,
        worldgen: Arc<WorldGenerator<'cx>>,
    ) -> Self {
        let chunk_builder = ChunkBuilder::new(resources);
        let (tasks_tx, tasks_rx) = mpmc::sync_channel(128);
        Self {
            workers: {
                let n_threads = num_cpus::get();
                // let n_threads = 1;
                println!("[INFO] using {n_threads} chunk worker threads");
                vec_with(n_threads, || {
                    let worker = Self::new_worker(
                        device,
                        tasks_rx.clone(),
                        Arc::clone(&chunks),
                        chunk_builder.clone(),
                        Arc::clone(&worldgen),
                    );
                    thread_scope.spawn(worker)
                })
            },
            device,
            chunk_builder,
            thread_scope,
            tasks_tx,
            tasks_rx,
            chunks,
        }
    }

    /// Send a task to the chunk workers.
    pub fn send_task(&self, task: ChunkBuildingTask) {
        self.tasks_tx.send(task).unwrap();
    }

    fn new_worker(
        device: &'cx wgpu::Device,
        tasks_rx: mpmc::Receiver<ChunkBuildingTask>,
        chunk_manager: Arc<ChunkManager>,
        mut chunk_builder: ChunkBuilder<'cx>,
        worldgen: Arc<WorldGenerator<'cx>>,
    ) -> impl FnOnce() + 'cx {
        move || {
            loop {
                let task = match tasks_rx.recv() {
                    Ok(task) => task,
                    Err(mpmc::RecvError) => {
                        println!(
                            "[ERROR] chunk worker thread {:?} stopping because main thread seems to be no longer",
                            thread::current().id()
                        );
                        break;
                    }
                };
                match task {
                    ChunkBuildingTask::Rebuild { chunk_id } => {
                        Self::task_rebuild(device, &chunk_manager, &mut chunk_builder, chunk_id);
                    }
                    ChunkBuildingTask::Generate { x, z, y_range } => {
                        Self::task_generate(
                            device,
                            &chunk_manager,
                            &mut chunk_builder,
                            &worldgen,
                            x,
                            z,
                            y_range,
                        );
                    }
                }
            }
        }
    }

    fn task_generate(
        device: &wgpu::Device,
        chunks: &ChunkManager,
        chunk_builder: &mut ChunkBuilder,
        worldgen: &WorldGenerator,
        x: i32,
        z: i32,
        y_range: Range<i32>,
    ) {
        let generated = worldgen.generate_strip(x, z, y_range.clone());
        for (chunk_id, chunk_data) in generated.into_chunks() {
            let mut chunk = Chunk {
                data: chunk_data,
                client: ClientChunk::new(),
            };
            chunk_builder.build(device, chunk_id, &mut chunk);
            chunks.insert_chunk(chunk_id, chunk);
        }
    }

    fn task_rebuild(
        device: &wgpu::Device,
        chunks: &ChunkManager,
        chunk_builder: &mut ChunkBuilder,
        chunk_id: ChunkId,
    ) {
        if let Some(chunk) = chunks.get_loaded_chunk(chunk_id) {
            let mut chunk_lock = chunk.lock().unwrap();
            chunk_builder.build(device, chunk_id, &mut chunk_lock);
        } else {
            println!(
                "[WARNING] Chunk worker encountered a rebuild task referring to an unloaded chunk (chunk ID: {chunk_id:?})"
            );
        };
    }
}

#[derive(Debug)]
pub struct World<'scope, 'cx>
where
    'cx: 'scope,
{
    world_height: i32,
    view_distance: i32,
    chunks: Arc<ChunkManager>,
    pub worldgen: Arc<WorldGenerator<'cx>>,
    chunk_workers: ChunkWorkerPool<'scope, 'cx>,
}

impl<'scope, 'cx> World<'scope, 'cx> {
    pub fn new(
        device: &'cx wgpu::Device,
        resources: &'cx GameResources,
        thread_scope: &'scope thread::Scope<'scope, 'cx>,
        args: &ProgramArgs,
    ) -> Self {
        let chunks = Arc::new(ChunkManager::new());
        let world_seed = args.seed.unwrap_or_else(|| getrandom::u64().unwrap_or(255));
        let mut world_height = args.height as i32;
        if world_height % 2 != 0 {
            world_height += 1;
        }
        let worldgen = Arc::new(WorldGenerator::new(world_seed, resources));
        let chunk_workers = ChunkWorkerPool::new(
            device,
            resources,
            thread_scope,
            Arc::clone(&chunks),
            Arc::clone(&worldgen),
        );
        Self {
            world_height,
            view_distance: args.view as i32,
            chunks: Arc::clone(&chunks),
            chunk_workers,
            worldgen,
        }
    }

    pub fn y_chunk_range(&self) -> Range<i32> {
        (-self.world_height / 2)..(self.world_height / 2)
    }

    pub fn y_coord_range(&self) -> Range<i32> {
        let y_chunk_range = self.y_chunk_range();
        World::local_to_world_coord_axis(y_chunk_range.start, 0)
            ..World::local_to_world_coord_axis(y_chunk_range.end, 31)
    }

    /// Generates an area around (0, _, 0).
    pub fn generate_initial_area(&self) {
        self.load_chunks_in_view_distance(WorldCoordF32::new(0., 0., 0.));
    }

    /// Convert world coord to local coord.
    pub const fn world_to_local_coord(world_coord: WorldCoordI32) -> (ChunkId, LocalCoordU8) {
        let (x_chunk, x_local) = Self::world_to_local_coord_axis(world_coord.x);
        let (y_chunk, y_local) = Self::world_to_local_coord_axis(world_coord.y);
        let (z_chunk, z_local) = Self::world_to_local_coord_axis(world_coord.z);
        let chunk_id = ChunkId::new(x_chunk, y_chunk, z_chunk);
        let local_coord = LocalCoordU8::new(x_local, y_local, z_local);
        (chunk_id, local_coord)
    }

    /// Convert world coord to local coord for a single axis (any one of X, Y, or Z).
    pub const fn world_to_local_coord_axis(xyz: i32) -> (i32, u8) {
        (xyz.div_euclid(32), xyz.rem_euclid(32) as u8)
    }

    /// Convert world coord to local coord.
    pub fn world_to_local_coord_f32(world_coord: WorldCoordF32) -> (ChunkId, LocalCoordF32) {
        let (x_chunk, x_local) = Self::world_to_local_coord_f32_axis(world_coord.x);
        let (y_chunk, y_local) = Self::world_to_local_coord_f32_axis(world_coord.y);
        let (z_chunk, z_local) = Self::world_to_local_coord_f32_axis(world_coord.z);
        let chunk_id = ChunkId::new(x_chunk, y_chunk, z_chunk);
        let local_coord = LocalCoordF32::new(x_local, y_local, z_local);
        (chunk_id, local_coord)
    }

    /// Convert world coord to local coord for a single axis (any one of X, Y, or Z).
    pub fn world_to_local_coord_f32_axis(xyz: f32) -> (i32, f32) {
        (xyz.div_euclid(32.0).round() as i32, xyz.rem_euclid(32.0))
    }

    /// Convert local coord to world coord.
    pub const fn local_to_world_coord(
        chunk_id: ChunkId,
        local_coord: LocalCoordU8,
    ) -> WorldCoordI32 {
        WorldCoordI32::new(
            Self::local_to_world_coord_axis(chunk_id.x, local_coord.x),
            Self::local_to_world_coord_axis(chunk_id.y, local_coord.y),
            Self::local_to_world_coord_axis(chunk_id.z, local_coord.z),
        )
    }

    /// Convert local coord to world coord for a single axis (any one of X, Y, or Z).
    pub const fn local_to_world_coord_axis(xyz_chunk: i32, xyz_local: u8) -> i32 {
        xyz_chunk * 32 + xyz_local as i32
    }

    /// Convert local coord to world coord.
    pub fn local_to_world_coord_f32(
        chunk_id: ChunkId,
        local_coord: LocalCoordF32,
    ) -> WorldCoordF32 {
        WorldCoordF32::new(
            Self::local_to_world_coord_f32_axis(chunk_id.x, local_coord.x),
            Self::local_to_world_coord_f32_axis(chunk_id.y, local_coord.y),
            Self::local_to_world_coord_f32_axis(chunk_id.z, local_coord.z),
        )
    }

    /// Convert local coord to world coord for a single axis (any one of X, Y, or Z).
    pub fn local_to_world_coord_f32_axis(xyz_chunk: i32, xyz_local: f32) -> f32 {
        (xyz_chunk as f32) * 32.0 + xyz_local
    }

    pub fn with_block<T>(
        &self,
        world_coord: WorldCoordI32,
        f: impl FnOnce(&mut BlockId) -> T,
    ) -> Option<T> {
        let (chunk_id, local_coord) = Self::world_to_local_coord(world_coord);
        let chunk = self.chunks().get_loaded_chunk(chunk_id);
        chunk.map(|chunk| f(chunk.lock().unwrap().data.get_block_mut(local_coord)))
    }

    pub fn get_block(&self, world_coord: WorldCoordI32) -> Option<BlockId> {
        self.with_block(world_coord, |&mut block| block)
    }

    /// Returns the original block, or `None` if coordinate is in an unloaded chunk.
    pub fn set_block(&self, world_coord: WorldCoordI32, mut block: BlockId) -> Option<BlockId> {
        self.with_block(world_coord, |target_block| {
            mem::swap(target_block, &mut block);
            block
        })
    }

    pub fn chunks(&self) -> &ChunkManager {
        &self.chunks
    }

    pub fn rebuild_chunk(&self, chunk_id: ChunkId) {
        self.chunk_workers
            .send_task(ChunkBuildingTask::Rebuild { chunk_id });
    }

    pub fn generate_strip(&self, x: i32, z: i32) {
        self.chunk_workers.send_task(ChunkBuildingTask::Generate {
            x,
            z,
            y_range: self.y_chunk_range(),
        });
    }

    fn load_chunks_in_view_distance(&self, player_position: WorldCoordF32) {
        let (chunk_id, _) = World::world_to_local_coord_f32(player_position);
        // Generate new chunks.
        for z in (chunk_id.z - self.view_distance)..(chunk_id.z + self.view_distance) {
            for x in (chunk_id.x - self.view_distance)..(chunk_id.x + self.view_distance) {
                let distance2 = point2(x as f32, z as f32)
                    .distance2(point2(chunk_id.x as f32, chunk_id.z as f32));
                if distance2 > (self.view_distance as f32).powi(2) {
                    continue;
                }
                let is_loaded = self.chunks().chunk_is_loaded(ChunkId::new(x, 0, z));
                if !is_loaded {
                    self.generate_strip(x, z);
                }
            }
        }
    }

    pub fn player_moved(&self, old_position: WorldCoordF32, new_position: WorldCoordF32) {
        // Check for strip boundary crossing.
        let (old_chunk_id, _) = World::world_to_local_coord_f32(old_position);
        let (chunk_id, _) = World::world_to_local_coord_f32(new_position);
        let crossed_strip = (chunk_id.x != old_chunk_id.x) | (chunk_id.z != old_chunk_id.z);
        if !crossed_strip {
            return;
        }
        // Prone chunks far away.
        self.chunks().for_each_loaded_chunk_id(|chunk_id_| {
            let chunk_xz = point2(chunk_id_.x as f32, chunk_id_.z as f32);
            let current_chunk_xz = point2(chunk_id.x as f32, chunk_id.z as f32);
            let distance2 = chunk_xz.distance2(current_chunk_xz);
            if distance2 > (self.view_distance as f32).powi(2) + 1.0 {
                self.chunks().remove_chunk(chunk_id_);
            }
        });
        self.load_chunks_in_view_distance(new_position);
    }
}
