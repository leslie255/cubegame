#![feature(mpmc_channel)]

#![allow(dead_code, linker_messages)]

use std::thread;

use glium::{backend::glutin, winit};

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

fn main() {
    unsafe {
        utils::this_thread_is_main_thread_pinky_promise();
    }

    let event_loop = winit::event_loop::EventLoop::builder().build().unwrap();

    let (window, display) = glutin::SimpleWindowBuilder::new()
        .with_title("Cube Game")
        .with_inner_size(800, 480)
        .build(&event_loop);
    let scale_factor = window.scale_factor();
    println!("[INFO] UI scale factor: {scale_factor}");
    if scale_factor != 1. {
        let _ = window.request_inner_size(winit::dpi::LogicalSize::new(800, 480));
    }

    let resources = GameResources::load(&display);

    thread::scope(|scope| {
        let mut game = Game::new(&resources, window, display, scope);
        event_loop.run_app(&mut game).unwrap();
    });
}
