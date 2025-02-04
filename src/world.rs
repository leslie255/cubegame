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
    chunk::{ChunkBuilder, ChunkData, ChunkMesh, ChunkMeshRef, LocalCoord},
    game::GameResources,
    utils::vec_with,
};

pub type ChunkId = Point3<i32>;

pub type BlockCoord = Point3<i32>;

#[derive(Debug, Clone)]
pub enum ChunkBuildingTask {
    Rebuild {
        chunk_id: ChunkId,
        target_mesh: ChunkMeshRef,
    },
}

#[derive(Debug)]
pub struct World<'scope, 'res>
where
    'res: 'scope,
{
    thread_scope: &'scope thread::Scope<'scope, 'res>,
    chunks: Arc<Mutex<Vec<Box<ChunkData>>>>,
    chunk_meshes: Vec<ChunkMesh>,
    chunk_building_tasks: Arc<Mutex<VecDeque<ChunkBuildingTask>>>,
    chunk_building_threads: Vec<thread::ScopedJoinHandle<'scope, ()>>,
    chunk_builder: ChunkBuilder<'res>,
    needs_chunk_building: bool,
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
        let mut self_ = Self {
            thread_scope,
            chunks: Arc::new(Mutex::new(vec![ChunkData::new(); Self::N_CHUNKS])),
            chunk_meshes: vec_with(Self::N_CHUNKS, ChunkMesh::new),
            needs_chunk_building: true,
            chunk_building_tasks: Default::default(),
            chunk_building_threads: Vec::new(),
            chunk_builder: ChunkBuilder::new(resources),
        };
        for _ in 0..4 {
            let thread = self_.init_chunk_building_thread();
            self_.chunk_building_threads.push(thread);
        }
        self_
    }

    fn init_chunk_building_thread(&mut self) -> thread::ScopedJoinHandle<'scope, ()> {
        let chunks = Arc::clone(&self.chunks);
        let chunk_builder = self.chunk_builder.clone();
        let tasks = Arc::clone(&self.chunk_building_tasks);
        self.thread_scope.spawn(move || {
            let chunks = chunks;
            let mut chunk_builder = chunk_builder;
            loop {
                let task = match tasks.lock().unwrap().pop_back() {
                    Some(task) => task,
                    None => {
                        thread::park_timeout(Duration::from_secs_f64(0.01));
                        continue;
                    }
                };
                match task {
                    ChunkBuildingTask::Rebuild {
                        chunk_id,
                        target_mesh,
                    } => {
                        let chunk_index = Self::index_for_chunk_id(chunk_id);
                        let chunk = chunks.lock().unwrap()[chunk_index].clone();
                        chunk_builder.build(chunk.as_ref(), target_mesh.clone());
                    }
                }
            }
        })
    }

    fn index_for_chunk_id(chunk_id: ChunkId) -> usize {
        (chunk_id.y - Self::CHUNK_ID_Y_RANGE.start) as usize
            * Self::SIZE_Z as usize
            * Self::SIZE_X as usize
            + (chunk_id.z - Self::CHUNK_ID_Z_RANGE.start) as usize * Self::SIZE_X as usize
            + (chunk_id.x - Self::CHUNK_ID_X_RANGE.start) as usize
    }

    pub fn chunk_is_loaded(&self, chunk_id: ChunkId) -> bool {
        Self::CHUNK_ID_X_RANGE.contains(&chunk_id.x)
            && Self::CHUNK_ID_Y_RANGE.contains(&chunk_id.y)
            && Self::CHUNK_ID_Z_RANGE.contains(&chunk_id.z)
    }

    pub fn with_chunk<T>(
        &self,
        chunk_id: ChunkId,
        f: impl FnOnce(&mut ChunkData) -> T,
    ) -> Option<T> {
        let index = Self::index_for_chunk_id(chunk_id);
        let mut chunks = self.chunks.lock().unwrap();
        let chunk = chunks.get_mut(index)?;
        Some(f(chunk))
    }

    pub fn get_chunk_mesh(&self, chunk_id: ChunkId) -> Option<&ChunkMesh> {
        self.chunk_meshes.get(Self::index_for_chunk_id(chunk_id))
    }

    pub fn get_chunk_mesh_mut(&mut self, chunk_id: ChunkId) -> Option<&mut ChunkMesh> {
        self.chunk_meshes
            .get_mut(Self::index_for_chunk_id(chunk_id))
    }

    fn build_chunk(&mut self, chunk_id: ChunkId) {
        let index = Self::index_for_chunk_id(chunk_id);
        let target_mesh = self.chunk_meshes.get_mut(index).unwrap().borrow();
        let mut chunk_building_tasks = self.chunk_building_tasks.lock().unwrap();
        chunk_building_tasks.push_front(ChunkBuildingTask::Rebuild {
            chunk_id,
            target_mesh,
        });
    }

    pub fn do_chunk_building(&mut self) {
        if !self.needs_chunk_building {
            return;
        }
        println!("[DEBUG] Adding chunk building tasks...");
        for y in Self::CHUNK_ID_Y_RANGE {
            for z in Self::CHUNK_ID_Z_RANGE {
                for x in Self::CHUNK_ID_X_RANGE {
                    self.build_chunk(ChunkId::new(x, y, z));
                }
            }
        }
        self.needs_chunk_building = false;
    }

    pub fn world_to_local_coord(&self, world_coord: BlockCoord) -> (ChunkId, LocalCoord) {
        let chunk_id = world_coord.map(|i| i.div_euclid(32));
        let local_coord = world_coord.map(|i| i.rem_euclid(32) as u8);
        (chunk_id, local_coord)
    }

    pub fn get_block(&self, world_coord: BlockCoord) -> Option<BlockId> {
        let (chunk_id, local_coord) = self.world_to_local_coord(world_coord);
        self.with_chunk(chunk_id, |chunk| chunk.try_get_block(local_coord))
            .flatten()
    }

    pub fn with_block<T>(
        &self,
        world_coord: BlockCoord,
        f: impl FnOnce(&mut BlockId) -> T,
    ) -> Option<T> {
        let (chunk_id, local_coord) = self.world_to_local_coord(world_coord);
        self.with_chunk(chunk_id, |chunk| f(chunk.get_block_mut(local_coord)))
    }

    /// Returns the original block, or `None` if block is in an unloaded chunk.
    pub fn try_set_block(&self, world_coord: BlockCoord, mut block: BlockId) -> Option<BlockId> {
        self.with_block(world_coord, |target_block| {
            mem::swap(target_block, &mut block);
            block
        })
    }
}
