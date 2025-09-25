use std::{mem::transmute, sync::Arc, thread, time::Instant};

use cgmath::*;

use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowAttributes},
};

use crate::{
    ProgramArgs,
    game::{self, Context, Game, GameResources, fps_counter::FpsCounter},
    input::InputHelper,
};

#[derive(Debug)]
pub struct App<'scope, 'self_> {
    program_args: ProgramArgs,
    fps_counter: FpsCounter,
    input_helper: InputHelper,
    window: Option<Arc<Window>>,
    context: Option<Context>,
    game: Option<Game<'scope, 'self_>>,
    thread_scope: &'scope thread::Scope<'scope, 'self_>,
    previous_window_event: Instant,
}

impl<'scope> App<'scope, 'static> {
    /// # Safety
    ///
    /// App is a self-referencing struct, it must not be moved after initialization.
    pub unsafe fn new(
        program_args: ProgramArgs,
        thread_scope: &'scope thread::Scope<'scope, 'static>,
    ) -> Self {
        // In reality self-referencing only happens after the initialization of the main window.
        // But we have off-loaded the unsafty here for sanity sake.
        Self {
            program_args,
            fps_counter: FpsCounter::new(),
            input_helper: InputHelper::new(),
            window: None,
            context: None,
            game: None,
            thread_scope,
            previous_window_event: Instant::now(),
        }
    }

    pub fn run(mut self, event_loop: EventLoop<()>) {
        event_loop.run_app(&mut self).unwrap();
    }
}

impl<'scope, 'self_> ApplicationHandler for App<'scope, 'self_> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        match &mut self.game {
            Some(_) => {}
            None => {
                log::info!("initializing window and game state");
                let window_size =
                    LogicalSize::new(self.program_args.wwidth, self.program_args.wheight);
                let window = event_loop
                    .create_window(
                        WindowAttributes::default()
                            .with_title("Cube Game")
                            .with_inner_size(window_size),
                    )
                    .unwrap();
                let window = Arc::new(window);
                self.window = Some(Arc::clone(&window));
                let (instance, adapter, device, queue) = game::initialize_wgpu();
                let resources =
                    GameResources::load(&device, self.program_args.res.as_ref().cloned());
                self.context = Some(Context {
                    device,
                    queue,
                    resources,
                });
                // SAEFTY: self would not be moved (guaranteed by `unsafe`ness of `App::new`), and context is never set to `None` afterwards.
                let context: &'self_ Context = unsafe { transmute(self.context.as_ref().unwrap()) };
                self.game = Some(Game::new(
                    &instance,
                    &adapter,
                    context,
                    window,
                    self.thread_scope,
                    &self.program_args,
                ));
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let now = Instant::now();
        let duration_since_last_event = now.duration_since(self.previous_window_event);
        self.previous_window_event = now;
        let game = self.game.as_mut().unwrap();
        game.before_handling_window_event(&self.input_helper, duration_since_last_event);
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
            WindowEvent::Resized(_) => game.frame_resized(),
            _ => (),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        let game = self.game.as_mut().unwrap();
        match event {
            DeviceEvent::MouseMotion { delta: (x, y) } => {
                game.cursor_moved(vec2(x as f32, y as f32), &self.input_helper);
            }
            DeviceEvent::MouseWheel { delta: _ } => (),
            _ => (),
        }
    }
}
