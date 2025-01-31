#![allow(dead_code)]

use glium::{backend::glutin, winit};

pub mod block;
pub mod game;
pub mod mesh;
pub mod resource;
pub mod text;

use game::{Game, GameResources};

fn main() {
    let event_loop = winit::event_loop::EventLoop::builder().build().unwrap();

    let (window, display) = glutin::SimpleWindowBuilder::new()
        .with_title("New Cube Game!")
        .with_inner_size(1600, 1200)
        .build(&event_loop);

    let resources = GameResources::load(&display);

    let mut game = Game::new(&resources, window, display);

    event_loop.run_app(&mut game).unwrap();
}
