#![feature(mpmc_channel)]
#![allow(dead_code, linker_messages)]

use std::path::PathBuf;

use clap::Parser;

pub mod block;
pub mod chunk;
pub mod game;
pub mod input;
pub mod resource;
pub mod text;
pub mod utils;
pub mod wgpu_utils;
pub mod world;
pub mod worldgen;

use crate::game::App;

#[derive(Debug, Clone, clap::Parser)]
pub struct ProgramArgs {
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long)]
    pub res: Option<PathBuf>,
    /// The view radius, in chunks.
    #[arg(long, default_value_t = 8)]
    pub view: u16,
    /// The world height, in chunks.
    #[arg(long, default_value_t = 8)]
    pub height: u16,
    /// The window width on launch.
    #[arg(long, default_value_t = 800)]
    pub wwidth: u32,
    /// The window height on launch.
    #[arg(long, default_value_t = 600)]
    pub wheight: u32,
}

fn main() {
    unsafe {
        utils::this_thread_is_main_thread_pinky_promise();
    }

    let program_args = ProgramArgs::parse();

    let event_loop = winit::event_loop::EventLoop::builder().build().unwrap();

    let mut app = App::new(program_args);

    app.run(event_loop);

    // let resources = GameResources::load(&display, program_args.res.clone());
}
