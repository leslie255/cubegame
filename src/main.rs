#![feature(mpmc_channel)]
#![allow(dead_code, linker_messages)]

use std::{path::PathBuf, thread};

use clap::Parser;
use glium::{
    backend::glutin,
    winit::{self, dpi::LogicalSize, window::WindowAttributes},
};

pub mod block;
pub mod chunk;
pub mod game;
pub mod input;
pub mod mesh;
pub mod resource;
pub mod text;
pub mod utils;
pub mod world;
pub mod worldgen;

use game::{Game, GameResources};

#[derive(Debug, Clone, clap::Parser)]
pub struct ProgramArgs {
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long)]
    pub res: Option<PathBuf>,
}

fn main() {
    unsafe {
        utils::this_thread_is_main_thread_pinky_promise();
    }

    let program_args = ProgramArgs::parse();

    let event_loop = winit::event_loop::EventLoop::builder().build().unwrap();

    let window_attributes = WindowAttributes::default()
        .with_title("Cube Game")
        .with_inner_size(LogicalSize::new(800, 480));

    let (window, display) = glutin::SimpleWindowBuilder::new()
        .set_window_builder(window_attributes)
        .build(&event_loop);

    let resources = GameResources::load(&display, program_args.res);

    let world_seed = program_args
        .seed
        .unwrap_or_else(|| getrandom::u64().unwrap_or(255));

    thread::scope(|scope| {
        let mut game = Game::new(&resources, window, display, scope, world_seed);
        event_loop.run_app(&mut game).unwrap();
    });
}
