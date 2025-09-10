use cgmath::*;

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
            Self::South => 0,
            Self::North => 1,
            Self::East => 2,
            Self::West => 3,
            Self::Top => 4,
            Self::Bottom => 5,
        }
    }

    pub const fn normal_vector(self) -> Vector3<f32> {
        match self {
            Self::South => vec3(0., 0., 1.),
            Self::North => vec3(0., 0., -1.),
            Self::East => vec3(1., 0., 0.),
            Self::West => vec3(-1., 0., 0.),
            Self::Top => vec3(0., 1., 0.),
            Self::Bottom => vec3(0., -1., 0.),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockTransparency {
    Solid,
    Leaves,
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
pub struct BlockId(pub u16);
impl index_vec::Idx for BlockId {
    fn from_usize(idx: usize) -> Self {
        Self(idx as u16)
    }
    fn index(self) -> usize {
        self.0 as usize
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
    pub log: BlockId,
    pub leaves: BlockId,
    pub sand: BlockId,
    pub cherry_log: BlockId,
    pub cherry_leaves: BlockId,
    pub water: BlockId,
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
            log: block_registry.register(BlockInfo {
                name: "log",
                transparency: BlockTransparency::Solid,
                model: BlockModel {
                    faces: [
                        BlockTextureId(5), // South
                        BlockTextureId(5), // North
                        BlockTextureId(5), // East
                        BlockTextureId(5), // West
                        BlockTextureId(6), // Top
                        BlockTextureId(6), // Bottom
                    ],
                },
            }),
            leaves: block_registry.register(BlockInfo {
                name: "leaves",
                transparency: BlockTransparency::Leaves,
                model: BlockModel {
                    faces: [
                        BlockTextureId(7), // South
                        BlockTextureId(7), // North
                        BlockTextureId(7), // East
                        BlockTextureId(7), // West
                        BlockTextureId(7), // Top
                        BlockTextureId(7), // Bottom
                    ],
                },
            }),
            sand: block_registry.register(BlockInfo {
                name: "sand",
                transparency: BlockTransparency::Solid,
                model: BlockModel {
                    faces: [
                        BlockTextureId(8), // South
                        BlockTextureId(8), // North
                        BlockTextureId(8), // East
                        BlockTextureId(8), // West
                        BlockTextureId(8), // Top
                        BlockTextureId(8), // Bottom
                    ],
                },
            }),
            cherry_log: block_registry.register(BlockInfo {
                name: "cherry_log",
                transparency: BlockTransparency::Solid,
                model: BlockModel {
                    faces: [
                        BlockTextureId(9),  // South
                        BlockTextureId(9),  // North
                        BlockTextureId(9),  // East
                        BlockTextureId(9),  // West
                        BlockTextureId(10), // Top
                        BlockTextureId(10), // Bottom
                    ],
                },
            }),
            cherry_leaves: block_registry.register(BlockInfo {
                name: "cherry_leaves",
                transparency: BlockTransparency::Leaves,
                model: BlockModel {
                    faces: [
                        BlockTextureId(11), // South
                        BlockTextureId(11), // North
                        BlockTextureId(11), // East
                        BlockTextureId(11), // West
                        BlockTextureId(11), // Top
                        BlockTextureId(11), // Bottom
                    ],
                },
            }),
            water: block_registry.register(BlockInfo {
                name: "water",
                transparency: BlockTransparency::Transparent,
                model: BlockModel {
                    faces: [
                        BlockTextureId(12), // South
                        BlockTextureId(12), // North
                        BlockTextureId(12), // East
                        BlockTextureId(12), // West
                        BlockTextureId(12), // Top
                        BlockTextureId(12), // Bottom
                    ],
                },
            }),
        }
    }
}
