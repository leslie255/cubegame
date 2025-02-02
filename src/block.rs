use index_vec::IndexVec;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockFace {
    /// +Z
    South,
    /// -Z
    North,
    /// +X
    East,
    /// -X
    West,
    /// -Y
    Top,
    /// +Y
    Bottom,
}

impl BlockFace {
    pub const fn from_usize(value: usize) -> Option<Self> {
        if value < 6 {
            Some(unsafe { Self::from_usize_unchecked(value) })
        } else {
            None
        }
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        (0..6).map(|i| unsafe { Self::from_usize_unchecked(i) })
    }

    /// # Safety
    /// `value` must be in `0..6`.
    pub const unsafe fn from_usize_unchecked(value: usize) -> Self {
        match value {
            0 => Self::South,
            1 => Self::North,
            2 => Self::East,
            3 => Self::West,
            4 => Self::Top,
            5 => Self::Bottom,
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    pub const fn to_usize(self) -> usize {
        match self {
            BlockFace::South => 0,
            BlockFace::North => 1,
            BlockFace::East => 2,
            BlockFace::West => 3,
            BlockFace::Top => 4,
            BlockFace::Bottom => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockTransparency {
    Solid,
    Transparent,
    Air,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockModel {
    pub faces: [BlockTextureId; 6],
}

impl BlockModel {
    pub const fn texture_for_face(self, face: BlockFace) -> BlockTextureId {
        self.faces[face.to_usize()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockTextureId(pub u32);

#[derive(Debug, Clone)]
pub struct BlockInfo {
    pub name: &'static str,
    pub transparency: BlockTransparency,
    pub model: BlockModel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From)]
#[repr(transparent)]
pub struct BlockId(pub usize);
impl index_vec::Idx for BlockId {
    fn from_usize(idx: usize) -> Self {
        Self(idx)
    }
    fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Default)]
pub struct BlockRegistry {
    blocks: IndexVec<BlockId, BlockInfo>,
}

impl BlockRegistry {
    pub fn register(&mut self, block_info: BlockInfo) -> BlockId {
        self.blocks.push(block_info)
    }

    pub fn lookup(&self, block_id: BlockId) -> Option<&BlockInfo> {
        self.blocks.get(block_id)
    }
}

#[derive(Debug, Clone)]
pub struct GameBlocks {
    pub air: BlockId,
    pub stone: BlockId,
    pub dirt: BlockId,
    pub grass: BlockId,
}

impl GameBlocks {
    pub fn new(block_registry: &mut BlockRegistry) -> Self {
        Self {
            air: block_registry.register(BlockInfo {
                name: "air",
                transparency: BlockTransparency::Air,
                model: BlockModel {
                    faces: [
                        BlockTextureId(0), // South
                        BlockTextureId(0), // North
                        BlockTextureId(0), // East
                        BlockTextureId(0), // West
                        BlockTextureId(0), // Top
                        BlockTextureId(0), // Bottom
                    ],
                },
            }),
            stone: block_registry.register(BlockInfo {
                name: "stone",
                transparency: BlockTransparency::Solid,
                model: BlockModel {
                    faces: [
                        BlockTextureId(1), // South
                        BlockTextureId(1), // North
                        BlockTextureId(1), // East
                        BlockTextureId(1), // West
                        BlockTextureId(1), // Top
                        BlockTextureId(1), // Bottom
                    ],
                },
            }),
            dirt: block_registry.register(BlockInfo {
                name: "dirt",
                transparency: BlockTransparency::Solid,
                model: BlockModel {
                    faces: [
                        BlockTextureId(2), // South
                        BlockTextureId(2), // North
                        BlockTextureId(2), // East
                        BlockTextureId(2), // West
                        BlockTextureId(2), // Top
                        BlockTextureId(2), // Bottom
                    ],
                },
            }),
            grass: block_registry.register(BlockInfo {
                name: "grass",
                transparency: BlockTransparency::Solid,
                model: BlockModel {
                    faces: [
                        BlockTextureId(3), // South
                        BlockTextureId(3), // North
                        BlockTextureId(3), // East
                        BlockTextureId(3), // West
                        BlockTextureId(4), // Top
                        BlockTextureId(2), // Bottom
                    ],
                },
            }),
        }
    }
}
