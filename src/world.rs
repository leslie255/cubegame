use std::{
    collections::VecDeque,
    mem,
    ops::Range,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use cgmath::*;

use crate::{
    block::BlockId,
    chunk::{Chunk, ChunkBuilder, ChunkData, ClientChunk, LocalCoord},
    game::GameResources,
    utils::vec_with,
};

pub type ChunkId = Point3<i32>;

pub type BlockCoord = Point3<i32>;

#[derive(Debug, Clone)]
pub enum ChunkBuildingTask {
    Rebuild { chunk_id: ChunkId },
}

#[derive(Debug)]
pub struct ChunkManager {
    chunks: Mutex<Vec<Chunk>>,
    chunk_builder_tasks: Mutex<VecDeque<ChunkBuildingTask>>,
}

impl ChunkManager {
    pub fn new() -> Self {
        Self {
            chunks: Mutex::new(vec_with(World::N_CHUNKS, || Chunk {
                data: ChunkData::new(),
                client: Some(Arc::new(Mutex::new(ClientChunk::new()))),
            })),
            chunk_builder_tasks: Mutex::new(VecDeque::new()),
        }
    }

    pub fn chunk_is_loaded(&self, chunk_id: ChunkId) -> bool {
        World::CHUNK_ID_X_RANGE.contains(&chunk_id.x)
            && World::CHUNK_ID_Y_RANGE.contains(&chunk_id.y)
            && World::CHUNK_ID_Z_RANGE.contains(&chunk_id.z)
    }

    pub fn with_chunk<T>(&self, chunk_id: ChunkId, f: impl FnOnce(&mut Chunk) -> T) -> Option<T> {
        if self.chunk_is_loaded(chunk_id) {
            let chunk_index = World::index_for_chunk_id(chunk_id);
            self.chunks.lock().unwrap().get_mut(chunk_index).map(f)
        } else {
            None
        }
    }

    pub fn push_task(&self, task: ChunkBuildingTask) {
        self.chunk_builder_tasks.lock().unwrap().push_back(task)
    }

    pub fn pop_task(&self) -> Option<ChunkBuildingTask> {
        self.chunk_builder_tasks.lock().unwrap().pop_front()
    }

    pub fn task_count(&self) -> usize {
        self.chunk_builder_tasks.lock().unwrap().len()
    }

    pub fn rebuild_all_chunks(&self) {
        println!("[DEBUG] adding chunk building tasks...");
        for z in World::CHUNK_ID_Z_RANGE {
            for x in World::CHUNK_ID_X_RANGE {
                for y in World::CHUNK_ID_Y_RANGE {
                    let chunk_id = ChunkId::new(x, y, z);
                    self.push_task(ChunkBuildingTask::Rebuild { chunk_id });
                }
            }
        }
        println!("[DEBUG] finished adding chunk building tasks");
    }
}

impl Default for ChunkManager {
    fn default() -> Self {
        Self::new()
    }
}

fn chunk_building_worker<'res>(
    chunk_manager: Arc<ChunkManager>,
    chunk_builder: &ChunkBuilder<'res>,
) -> impl FnOnce() + 'res {
    let mut chunk_builder = chunk_builder.clone();
    move || {
        loop {
            let Some(task) = chunk_manager.pop_task() else {
                thread::park_timeout(Duration::from_micros(500));
                continue;
            };
            match task {
                ChunkBuildingTask::Rebuild { chunk_id } => {
                    chunk_manager
                        .with_chunk(chunk_id, |chunk| {
                            let mut client_chunk = chunk.client.as_ref().unwrap().lock().unwrap();
                            chunk_builder.build(chunk.data.as_ref(), &mut client_chunk);
                        })
                        .unwrap();
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct ChunkBuildingWorkers<'scope, 'res>
where
    'res: 'scope,
{
    workers: Vec<thread::ScopedJoinHandle<'scope, ()>>,
    chunk_builder: ChunkBuilder<'res>,
    thread_scope: &'scope thread::Scope<'scope, 'res>,
    chunks: Arc<ChunkManager>,
}

impl<'scope, 'res> ChunkBuildingWorkers<'scope, 'res>
where
    'res: 'scope,
{
    pub fn new(
        resources: &'res GameResources,
        thread_scope: &'scope thread::Scope<'scope, 'res>,
        chunks: Arc<ChunkManager>,
    ) -> Self {
        let chunk_builder = ChunkBuilder::new(resources);
        Self {
            workers: {
                let n_threads = num_cpus::get() * 2;
                println!("[INFO] using {n_threads} chunk building threads");
                let make_worker = || {
                    let worker = chunk_building_worker(Arc::clone(&chunks), &chunk_builder);
                    thread_scope.spawn(worker)
                };
                std::iter::repeat_with(make_worker)
                    .take(n_threads)
                    .collect()
            },
            chunk_builder,
            thread_scope,
            chunks,
        }
    }
}

#[derive(Debug)]
pub struct World<'scope, 'res>
where
    'res: 'scope,
{
    chunks: Arc<ChunkManager>,
    chunk_building_workers: ChunkBuildingWorkers<'scope, 'res>,
}

const WORLD_ASSERTIONS: () = {
    assert!(World::SIZE_X % 2 == 0);
    assert!(World::SIZE_Y % 2 == 0);
    assert!(World::SIZE_Z % 2 == 0);
};

impl<'scope, 'res> World<'scope, 'res> {
    pub const SIZE_X: i32 = 32;
    pub const SIZE_Y: i32 = 8;
    pub const SIZE_Z: i32 = 32;

    pub const N_CHUNKS: usize =
        (Self::SIZE_X as usize) * (Self::SIZE_Y as usize) * (Self::SIZE_Z as usize);

    pub const CHUNK_ID_X_RANGE: Range<i32> = (-Self::SIZE_X / 2)..(Self::SIZE_X / 2);
    pub const CHUNK_ID_Y_RANGE: Range<i32> = (-Self::SIZE_Y / 2)..(Self::SIZE_Y / 2);
    pub const CHUNK_ID_Z_RANGE: Range<i32> = (-Self::SIZE_Z / 2)..(Self::SIZE_Z / 2);

    pub const COORD_X_RANGE: Range<i32> =
        Self::CHUNK_ID_X_RANGE.start * 32..Self::CHUNK_ID_X_RANGE.end * 32;
    pub const COORD_Y_RANGE: Range<i32> =
        Self::CHUNK_ID_Y_RANGE.start * 32..Self::CHUNK_ID_Y_RANGE.end * 32;
    pub const COORD_Z_RANGE: Range<i32> =
        Self::CHUNK_ID_Z_RANGE.start * 32..Self::CHUNK_ID_Z_RANGE.end * 32;

    pub fn new(
        resources: &'res GameResources,
        thread_scope: &'scope thread::Scope<'scope, 'res>,
    ) -> Self {
        let chunks = Arc::new(ChunkManager::new());
        Self {
            chunks: Arc::clone(&chunks),
            chunk_building_workers: ChunkBuildingWorkers::new(resources, thread_scope, chunks),
        }
    }

    fn index_for_chunk_id(chunk_id: ChunkId) -> usize {
        (chunk_id.y - Self::CHUNK_ID_Y_RANGE.start) as usize
            * Self::SIZE_Z as usize
            * Self::SIZE_X as usize
            + (chunk_id.z - Self::CHUNK_ID_Z_RANGE.start) as usize * Self::SIZE_X as usize
            + (chunk_id.x - Self::CHUNK_ID_X_RANGE.start) as usize
    }

    pub fn rebuild_all_chunks(&self) {
        self.chunks.rebuild_all_chunks();
    }

    pub fn world_to_local_coord(&self, world_coord: BlockCoord) -> (ChunkId, LocalCoord) {
        let chunk_id = world_coord.map(|i| i.div_euclid(32));
        let local_coord = world_coord.map(|i| i.rem_euclid(32) as u8);
        (chunk_id, local_coord)
    }

    pub fn with_block<T>(
        &self,
        world_coord: BlockCoord,
        f: impl FnOnce(&mut BlockId) -> T,
    ) -> Option<T> {
        let (chunk_id, local_coord) = self.world_to_local_coord(world_coord);
        self.chunks()
            .with_chunk(chunk_id, |chunk| f(chunk.data.get_block_mut(local_coord)))
    }

    pub fn get_block(&self, world_coord: BlockCoord) -> Option<BlockId> {
        self.with_block(world_coord, |&mut block| block)
    }

    /// Returns the original block, or `None` if coordinate is in an unloaded chunk.
    pub fn set_block(&self, world_coord: BlockCoord, mut block: BlockId) -> Option<BlockId> {
        self.with_block(world_coord, |target_block| {
            mem::swap(target_block, &mut block);
            block
        })
    }

    pub fn chunks(&self) -> &ChunkManager {
        &self.chunks
    }

    pub fn with_chunk<T>(&self, chunk_id: ChunkId, f: impl FnOnce(&mut Chunk) -> T) -> Option<T> {
        self.chunks().with_chunk(chunk_id, f)
    }
}
