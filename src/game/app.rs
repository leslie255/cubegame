use std::sync::Arc;

use winit::{application::ApplicationHandler, dpi::LogicalSize, event::WindowEvent, event_loop::EventLoop, window::{Window, WindowAttributes}};

use crate::{game::{fps_counter::FpsCounter, Game}, input::InputHelper, ProgramArgs};

#[derive(Debug)]
pub struct App {
    program_args: ProgramArgs,
    fps_counter: FpsCounter,
    input_helper: InputHelper,
    window: Option<Arc<Window>>,
    game: Option<Game>,
}

impl App {
    pub fn new(program_args: ProgramArgs) -> Self {
        Self {
            program_args,
            fps_counter: FpsCounter::new(),
            input_helper: InputHelper::new(),
            window: None,
            game: None,
        }
    }

    pub fn run(&mut self, event_loop: EventLoop<()>) {
        event_loop.run_app(self).unwrap();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        match &mut self.game {
            Some(_) => {}
            None => {
                let window_size =
                    LogicalSize::new(self.program_args.wwidth, self.program_args.wheight);
                let window = event_loop
                    .create_window(
                        WindowAttributes::default()
                            .with_title("Cube Game")
                            .with_inner_size(window_size),
                    )
                    .unwrap();
                let arc_window = Arc::new(window);
                self.window = Some(arc_window.clone());
                self.game = Some(Game::new(arc_window.clone()));
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let game = self.game.as_mut().unwrap();
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                game.frame();
                self.window.as_ref().unwrap().request_redraw();
                let fps_update = self.fps_counter.frame();
                if let Some(fps) = fps_update {
                    game.update_fps(fps);
                }
            }
            WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => {
                self.input_helper.update_key_event(&event);
                game.keyboard_input(event, &self.input_helper);
            }
            WindowEvent::MouseInput {
                device_id: _,
                state,
                button,
            } => {
                game.mouse_input(state, button, &self.input_helper);
            }
            WindowEvent::Resized(_) => game.resized(),
            _ => (),
        }
    }
}


