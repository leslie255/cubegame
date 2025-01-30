#![allow(dead_code)]

use glium::winit;

pub mod game;
pub mod resource;
pub mod text;

use game::Game;

fn main() {
    let event_loop = winit::event_loop::EventLoop::builder().build().unwrap();

    let mut game = Game::new(&event_loop);

    event_loop.run_app(&mut game).unwrap();
}
