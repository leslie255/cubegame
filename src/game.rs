use std::{
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};

use cgmath::*;
use glium::{
    Surface,
    winit::{
        self,
        keyboard::{KeyCode, PhysicalKey},
    },
};

use crate::{
    ProgramArgs,
    block::{BlockRegistry, GameBlocks},
    input::InputHelper,
    mesh::{self, Color},
    resource::ResourceLoader,
    text::{Font, Line},
    utils::BoolToggle as _,
    world::World,
};

#[derive(Debug, Default, Clone, PartialEq)]
pub struct DebugOptions {
    pub wireframe_mode: bool,
    pub disable_gl_backface_culling: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub position: Point3<f32>,
    pub direction: Vector3<f32>,
    pub up: Vector3<f32>,
    /// In degrees.
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn view_matrix(self) -> Matrix4<f32> {
        Matrix4::look_to_rh(self.position, self.direction, self.up)
    }

    pub fn projection_matrix(self, frame_size: Vector2<f32>) -> Matrix4<f32> {
        cgmath::perspective(
            Deg(self.fov),
            frame_size.x / frame_size.y,
            self.near,
            self.far,
        )
    }
}

#[derive(Debug, Clone)]
pub struct PlayerCamera {
    pub camera: Camera,
    /// In degrees.
    pitch: f32,
    /// In degrees.
    yaw: f32,
}

impl PlayerCamera {
    pub fn new(position: Point3<f32>, pitch: f32, yaw: f32) -> Self {
        Self {
            camera: Camera {
                position,
                direction: Self::pitch_yaw_to_direction(pitch, yaw),
                up: Vector3::new(0., 1., 0.).normalize(),
                fov: 90.,
                near: 0.01,
                far: 10000.,
            },
            pitch,
            yaw,
        }
    }

    pub const fn pitch_yaw(&self) -> (f32, f32) {
        (self.pitch, self.yaw)
    }

    /// Move the camera according to cursor movement.
    /// Assuming up vector is directly up.
    pub fn cursor_moved(&mut self, sensitivity: f32, delta: Vector2<f32>) {
        self.yaw += delta.x * sensitivity;
        self.pitch -= delta.y * sensitivity;
        self.pitch = self.pitch.clamp(-89.99, 89.99);
        if self.yaw <= -180. {
            self.yaw += 360.;
        } else if self.yaw >= 180. {
            self.yaw -= 360.;
        }
        self.camera.direction = self.direction();
    }

    pub fn move_(&mut self, delta: Vector3<f32>) {
        let forward =
            Vector3::new(self.camera.direction.x, 0., self.camera.direction.z).normalize();
        let right = forward.cross(self.camera.up).normalize();
        let up = self.camera.up;

        let forward_scaled = delta.z * forward;
        let right_scaled = delta.x * right;
        let up_scaled = delta.y * up;

        self.camera.position += forward_scaled + right_scaled + up_scaled;
    }

    pub fn direction(&self) -> Vector3<f32> {
        Self::pitch_yaw_to_direction(self.pitch, self.yaw)
    }

    fn pitch_yaw_to_direction(pitch: f32, yaw: f32) -> Vector3<f32> {
        Vector3::new(
            yaw.to_radians().cos() * pitch.to_radians().cos(),
            pitch.to_radians().sin(),
            yaw.to_radians().sin() * pitch.to_radians().cos(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

glium::implement_vertex!(BlockVertex, position, uv);

impl BlockVertex {
    pub const fn new(position: [f32; 3], texture_coord: [f32; 2]) -> Self {
        Self {
            position,
            uv: texture_coord,
        }
    }
}

#[derive(Debug)]
pub struct InfoText<'res> {
    lines: Vec<Line<'res>>,
    font: &'res Font,
    shader: &'res glium::Program,
}

impl<'res> InfoText<'res> {
    pub fn new(
        display: &impl glium::backend::Facade,
        font: &'res Font,
        shader: &'res glium::Program,
    ) -> Self {
        Self {
            lines: vec![
                Line::with_string(
                    font,
                    shader,
                    display,
                    if cfg!(debug_assertions) {
                        "CUBE GAME v0.0.0 (Debug Build)".into()
                    } else {
                        "CUBE GAME v0.0.0".into()
                    },
                ),
                Line::with_string(font, shader, display, "FPS: ---.---".into()),
                Line::with_string(
                    font,
                    shader,
                    display,
                    "Camera XYZ: ---.---, ---.---, ---.---".into(),
                ),
                Line::with_string(
                    font,
                    shader,
                    display,
                    "Camera pitch/yaw: ---.---deg, ---.---deg".into(),
                ),
                Line::with_string(font, shader, display, "Facing: ----".into()),
                Line::with_string(font, shader, display, "Terrain height: ---".into()),
                Line::new(font, shader),
                Line::new(font, shader),
            ],
            font,
            shader,
        }
    }

    fn set_line(&mut self, display: &impl glium::backend::Facade, i_line: usize, text: &str) {
        let line = &mut self.lines[i_line];
        line.clear();
        line.push_str(text);
        line.update(display);
    }

    fn set_is_paused(&mut self, display: &impl glium::backend::Facade, is_paused: bool) {
        if is_paused {
            self.set_line(display, 0, "[ESC] GAME PAUSED");
        } else {
            self.set_line(display, 0, "CUBE GAME v0.0.0");
        }
    }

    fn set_fps(&mut self, display: &impl glium::backend::Facade, fps: f64) {
        if fps.is_nan() {
            self.set_line(display, 1, "FPS: ---.---");
        } else {
            self.set_line(display, 1, &format!("FPS: {fps:.3}"));
        }
    }

    fn set_camera_xyz(&mut self, display: &impl glium::backend::Facade, camera_xyz: Point3<f32>) {
        self.set_line(
            display,
            2,
            &format!(
                "Camera XYZ: {:.3}, {:.3}, {:.3}",
                camera_xyz.x, camera_xyz.y, camera_xyz.z
            ),
        );
    }

    fn set_camera_direction(
        &mut self,
        display: &glium::Display<glium::glutin::surface::WindowSurface>,
        pitch_yaw: (f32, f32),
        direction: Vector3<f32>,
    ) {
        let facing = if direction.x.abs() > direction.z.abs() {
            if direction.x.is_sign_positive() {
                "East (+X)"
            } else {
                "West (-X)"
            }
        } else if direction.z.is_sign_positive() {
            "South (+Z)"
        } else {
            "North (-Z)"
        };
        self.set_line(
            display,
            3,
            &format!(
                "Camera pitch/yaw: {:.3}deg, {:.3}deg",
                pitch_yaw.0, pitch_yaw.1
            ),
        );
        self.set_line(display, 4, &format!("Facing: {facing}"));
    }

    fn update_debug_options(
        &mut self,
        display: &impl glium::backend::Facade,
        debug_options: &DebugOptions,
    ) {
        if debug_options.wireframe_mode {
            self.set_line(display, 6, "[F3+L] DEBUG: Wire frame mode")
        } else {
            self.set_line(display, 6, "")
        }
        if debug_options.disable_gl_backface_culling {
            self.set_line(display, 7, "[F3+F] DEBUG: Disable OpenGL backface culling")
        } else {
            self.set_line(display, 7, "")
        }
    }

    fn set_terrain_height(&mut self, display: &impl glium::backend::Facade, terrain_height: i32) {
        self.set_line(display, 5, &format!("Terrain height: {terrain_height}"));
    }

    fn draw(&self, frame: &mut glium::Frame, content_scale: f32) {
        let font_size = 20. * content_scale;
        for (i, line) in self.lines.iter().enumerate() {
            line.draw(
                frame,
                /* position   */ Point2::new(10., 10. + font_size * i as f32),
                /* foreground */ Color::new(1., 1., 1., 1.),
                /* background */ Color::new(0., 0., 0., 0.),
                /* shadow     */ true,
                /* font size  */ font_size,
            );
        }
    }
}

#[derive(Debug)]
pub struct GameResources {
    pub loader: ResourceLoader,
    pub shader_text: glium::Program,
    pub shader_chunk: glium::Program,
    pub block_atlas: glium::Texture2d,
    pub font: Font,
    pub block_registry: BlockRegistry,
    pub game_blocks: GameBlocks,
}

impl GameResources {
    /// Uses the default resource directory if `res_directory` is `None`.
    pub fn load(display: &impl glium::backend::Facade, res_directory: Option<PathBuf>) -> Self {
        let res_directory = res_directory.unwrap_or_else(ResourceLoader::default_res_directory);
        let loader = ResourceLoader::with_res_directory(res_directory.clone())
            .unwrap_or_else(|| panic!("Cannot find resource directory {res_directory:?}"));
        let mut block_registry = BlockRegistry::default();
        Self {
            shader_text: Self::load_shader(display, &loader, "shader/text"),
            shader_chunk: Self::load_shader(display, &loader, "shader/chunk"),
            block_atlas: Self::load_texture(display, &loader, "texture/block_atlas.png"),
            font: Self::load_font(display, &loader, "font/big_blue_terminal.json"),
            loader,
            game_blocks: GameBlocks::new(&mut block_registry),
            block_registry,
        }
    }

    fn load_texture(
        display: &impl glium::backend::Facade,
        resource_loader: &ResourceLoader,
        name: &str,
    ) -> glium::Texture2d {
        let image = resource_loader.load_image(name);
        let image = glium::texture::RawImage2d::from_raw_rgba(
            image.to_rgba8().into_raw(),
            (image.width(), image.height()),
        );
        glium::Texture2d::with_mipmaps(
            display,
            image,
            glium::texture::MipmapsOption::AutoGeneratedMipmapsMax(4),
        )
        .unwrap()
    }

    /// `name` is in the format of `"shader/name"`, which would load `"res/shader/name.vs"` and `"res/shader/name.fs"`.
    fn load_shader(
        display: &impl glium::backend::Facade,
        resource_loader: &ResourceLoader,
        name: &str,
    ) -> glium::Program {
        let vs_source = resource_loader.read_to_string(format!("{name}.vs"));
        let fs_source = resource_loader.read_to_string(format!("{name}.fs"));
        glium::Program::from_source(display, &vs_source, &fs_source, None).unwrap()
    }

    fn load_font(
        display: &impl glium::backend::Facade,
        resource_loader: &ResourceLoader,
        name: &str,
    ) -> Font {
        let mut font = Font::load_from_path(resource_loader, name);
        let image = glium::texture::RawImage2d::from_raw_rgba(
            font.atlas().to_rgba8().into_raw(),
            (font.atlas().width(), font.atlas().height()),
        );
        font.gl_texture = Some(glium::Texture2d::new(display, image).unwrap());
        font
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FpsCounter {
    last_update: Instant,
    counter: u32,
    fps: f64,
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            last_update: Instant::now(),
            counter: 0,
            fps: f64::NAN,
        }
    }

    /// Should be called after every frame is drawn.
    /// Returns `Some(fps)` if FPS is updated.
    pub fn frame(&mut self) -> Option<f64> {
        self.counter += 1;
        let now = Instant::now();
        let seconds_since_last_update = now.duration_since(self.last_update).as_secs_f64();
        if seconds_since_last_update > 0.5 {
            self.fps = (self.counter as f64) / seconds_since_last_update;
            self.last_update = now;
            self.counter = 0;
            Some(self.fps)
        } else {
            None
        }
    }

    pub fn fps(&self) -> f64 {
        self.fps
    }
}

impl Default for FpsCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct Game<'scope, 'res>
where
    'res: 'scope,
{
    thread_scope: &'scope thread::Scope<'scope, 'res>,
    window: winit::window::Window,
    display: glium::Display<glium::glutin::surface::WindowSurface>,
    resources: &'res GameResources,
    input_helper: InputHelper,
    fps_counter: FpsCounter,
    debug_options: DebugOptions,
    player_camera: PlayerCamera,
    world: World<'scope, 'res>,
    info_text: InfoText<'res>,
    last_window_event: Instant,
    is_paused: bool,
}

impl<'scope, 'res> Game<'scope, 'res> {
    pub fn new(
        resources: &'res GameResources,
        window: winit::window::Window,
        display: glium::Display<glium::glutin::surface::WindowSurface>,
        thread_scope: &'scope thread::Scope<'scope, 'res>,
        args: ProgramArgs,
    ) -> Self {
        let world = World::new(resources, thread_scope, args);
        world.generate_initial_area();
        Self {
            thread_scope,
            window,
            input_helper: InputHelper::new(),
            fps_counter: FpsCounter::new(),
            debug_options: DebugOptions::default(),
            player_camera: PlayerCamera::new(Point3::new(0., 0., 2.), 0., -90.),
            world,
            info_text: InfoText::new(&display, &resources.font, &resources.shader_text),
            last_window_event: Instant::now(),
            is_paused: false,
            display,
            resources,
        }
    }

    pub fn blocks(&self) -> &GameBlocks {
        &self.resources.game_blocks
    }

    pub fn block_registry(&self) -> &BlockRegistry {
        &self.resources.block_registry
    }

    fn draw_render_test_chunk(&self) {}

    fn draw(&mut self) {
        let mut frame = self.display.draw();

        self.clear_frame(&mut frame);

        self.draw_chunks(&mut frame);
        self.draw_info_text(&mut frame);

        frame.finish().unwrap();

        self.update_fps();
    }

    fn update_fps(&mut self) {
        if let Some(fps) = self.fps_counter.frame() {
            self.info_text.set_fps(&self.display, fps);
        }
    }

    fn clear_frame(&mut self, frame: &mut glium::Frame) {
        let clear_color = if self.debug_options.wireframe_mode {
            (0.1, 0.1, 0.1, 1.)
        } else {
            (0.8, 0.95, 1.0, 1.0)
        };
        frame.clear_color_and_depth(clear_color, 1.);
    }

    fn draw_chunks(&mut self, frame: &mut glium::Frame) {
        let view_matrix = self.player_camera.camera.view_matrix();
        let frame_size = self.window.inner_size();
        let projection_matrix = self.player_camera.camera.projection_matrix(Vector2::new(
            frame_size.width as f32,
            frame_size.height as f32,
        ));
        let draw_parameters = &glium::DrawParameters {
            polygon_mode: if self.debug_options.wireframe_mode {
                glium::PolygonMode::Line
            } else {
                glium::PolygonMode::Fill
            },
            backface_culling: if self.debug_options.disable_gl_backface_culling {
                glium::BackfaceCullingMode::CullingDisabled
            } else {
                glium::BackfaceCullingMode::CullClockwise
            },
            ..mesh::default_3d_draw_parameters()
        };
        let shader = &self.resources.shader_chunk;
        self.world
            .chunks()
            .for_each_loaded_chunk(|chunk_id, chunk| {
                let model_matrix = Matrix4::from_translation(vec3(
                    (chunk_id.x * 32) as f32,
                    (chunk_id.y * 32) as f32,
                    (chunk_id.z * 32) as f32,
                ));
                let uniforms = glium::uniform! {
                    model_view: mesh::matrix4_to_array(view_matrix * model_matrix),
                    projection: mesh::matrix4_to_array(projection_matrix),
                    texture_atlas: mesh::texture_sampler(&self.resources.block_atlas),
                };
                chunk.client.mesh.update_if_needed(&self.display);
                chunk
                    .client
                    .mesh
                    .draw(frame, uniforms, shader, draw_parameters);
            });
    }

    fn draw_info_text(&mut self, frame: &mut glium::Frame) {
        self.info_text
            .set_camera_xyz(&self.display, self.player_camera.camera.position);
        self.info_text.set_camera_direction(
            &self.display,
            self.player_camera.pitch_yaw(),
            self.player_camera.camera.direction,
        );
        let terrain_height = {
            let x: i32 = self.player_camera.camera.position.x.round() as i32;
            let z: i32 = self.player_camera.camera.position.z.round() as i32;
            self.world.worldgen.terrain_height_at(x, z)
        };
        self.info_text
            .set_terrain_height(&self.display, terrain_height);
        self.info_text
            .draw(frame, self.window.scale_factor() as f32);
    }

    fn grab_cursor(&mut self) {
        let cursor_confined = self
            .window
            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
            .is_ok();
        if !cursor_confined {
            self.window
                .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                .unwrap_or_else(|_| eprintln!("[WARNING] Unable to grab cursor"))
        }
        self.window.set_cursor_visible(false);
    }

    fn ungrab_cursor(&mut self) {
        self.window
            .set_cursor_grab(winit::window::CursorGrabMode::None)
            .unwrap_or_else(|_| eprintln!("[WARNING] Unable to ungrab cursor"));
        self.window.set_cursor_visible(true);
    }

    fn toggle_paused(&mut self) {
        self.is_paused.toggle();
        self.info_text.set_is_paused(&self.display, self.is_paused);
        if self.is_paused {
            self.ungrab_cursor();
        } else {
            self.grab_cursor();
        }
    }

    fn before_window_event(&mut self, duration_since_last_window_event: Duration) {
        if !self.is_paused {
            let mut movement = vec3(0., 0., 0.);
            if self.input_helper.key_is_down(KeyCode::KeyW) {
                movement.z += 1.0;
            }
            if self.input_helper.key_is_down(KeyCode::KeyS) {
                movement.z -= 1.0;
            }
            if self.input_helper.key_is_down(KeyCode::KeyA) {
                movement.x -= 1.0;
            }
            if self.input_helper.key_is_down(KeyCode::KeyD) {
                movement.x += 1.0;
            }
            if self.input_helper.key_is_down(KeyCode::Space) {
                movement.y += 1.0;
            }
            if self.input_helper.key_is_down(KeyCode::KeyR) {
                movement.y -= 1.0;
            }
            movement.normalize();
            movement *= 4.;
            if self.input_helper.key_is_down(KeyCode::F3) {
                movement *= 32.;
            } else if self.input_helper.key_is_down(KeyCode::ControlLeft) {
                movement *= 4.;
            }
            movement *= duration_since_last_window_event.as_secs_f32();
            let old_position = self.player_camera.camera.position;
            self.player_camera.move_(movement);
            let new_position = self.player_camera.camera.position;
            self.world.player_moved(old_position, new_position)
        }
    }

    fn key_down(&mut self, key_code: KeyCode, _text: Option<&str>, is_repeat: bool) {
        if self.input_helper.key_is_down(KeyCode::F3) {
            match key_code {
                KeyCode::KeyL => self.debug_options.wireframe_mode.toggle(),
                KeyCode::KeyF => self.debug_options.disable_gl_backface_culling.toggle(),
                _ => (),
            }
            self.info_text
                .update_debug_options(&self.display, &self.debug_options);
            return;
        }
        match key_code {
            KeyCode::Escape if !is_repeat => self.toggle_paused(),
            KeyCode::KeyC if !is_repeat => self.player_camera.camera.fov = 30.,
            _ => (),
        }
    }

    fn key_up(&mut self, key_code: KeyCode) {
        #[allow(clippy::single_match)]
        match key_code {
            KeyCode::KeyC => self.player_camera.camera.fov = 90.,
            _ => (),
        }
    }

    fn cursor_moved(&mut self, delta: Vector2<f32>) {
        if !self.is_paused {
            self.player_camera.cursor_moved(0.1, delta);
        }
    }
}

impl winit::application::ApplicationHandler for Game<'_, '_> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        _ = event_loop;
        if !self.is_paused {
            self.grab_cursor();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let now = Instant::now();
        let duration_since_last_window_event = now.duration_since(self.last_window_event);
        self.last_window_event = now;
        self.before_window_event(duration_since_last_window_event);
        match event {
            winit::event::WindowEvent::CloseRequested => event_loop.exit(),
            winit::event::WindowEvent::RedrawRequested => {
                self.draw();
                self.window.request_redraw();
            }
            winit::event::WindowEvent::Resized(window_size) => {
                self.display.resize(window_size.into());
            }
            winit::event::WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => {
                self.input_helper.update_key_event(&event);
                match event.physical_key {
                    PhysicalKey::Code(key_code) => {
                        if event.state.is_pressed() {
                            self.key_down(key_code, event.text.as_deref(), event.repeat)
                        } else {
                            self.key_up(key_code)
                        }
                    }
                    PhysicalKey::Unidentified(_) => (),
                }
            }
            _ => (),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        match event {
            winit::event::DeviceEvent::MouseMotion { delta } => {
                self.cursor_moved(Vector2::new(delta.0 as f32, delta.1 as f32));
            }
            winit::event::DeviceEvent::MouseWheel { delta: _ } => (),
            winit::event::DeviceEvent::Motion { axis: _, value: _ } => (),
            winit::event::DeviceEvent::Button {
                button: _,
                state: _,
            } => (),
            _ => (),
        }
    }
}
