use std::{
    fmt::Write,
    time::{Duration, Instant, SystemTime},
};

use cgmath::*;
use glium::{
    Surface,
    backend::glutin,
    winit::{
        self,
        event::KeyEvent,
        keyboard::{KeyCode, PhysicalKey},
    },
};

use crate::{resource::ResourceLoader, text::TextPainter};

/// Keeps track of which keys are currently down.
#[derive(Debug, Clone)]
pub struct InputHelper {
    downed_keys: Vec<bool>,
}
impl InputHelper {
    pub fn new() -> Self {
        Self {
            downed_keys: vec![false; 256],
        }
    }

    fn index_for_key(key_code: KeyCode) -> usize {
        // This is technically unsafe lol due to KeyCode not being a stable API, but like nah.
        (key_code as u8).into()
    }

    pub fn key_is_down(&self, key_code: KeyCode) -> bool {
        self.downed_keys[Self::index_for_key(key_code)]
    }

    pub fn update_key_event(&mut self, key_event: &KeyEvent) {
        if key_event.repeat {
            return;
        }
        let key_code = match key_event.physical_key {
            PhysicalKey::Code(key_code) => key_code,
            PhysicalKey::Unidentified(_) => return,
        };
        let index = Self::index_for_key(key_code);
        self.downed_keys[index] = key_event.state.is_pressed();
    }
}

impl Default for InputHelper {
    fn default() -> Self {
        Self::new()
    }
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
                far: 1000.,
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
        self.pitch = self.pitch.clamp(-89., 89.);
        self.yaw %= 360.;
        self.camera.direction = self.direction();
    }

    pub fn move_(&mut self, delta: Vector3<f32>) {
        let forward =
            Vector3::new(self.camera.direction.x, 0., self.camera.direction.z).normalize();
        let right = forward.cross(self.camera.up).normalize();
        let up = self.camera.up;

        let forward_scaled = delta.y * forward;
        let right_scaled = delta.x * right;
        let up_scaled = delta.z * up;

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
pub struct TriangleVertex {
    pub position: [f32; 2],
    pub color: [f32; 3],
}

impl TriangleVertex {
    pub const fn new(position: [f32; 2], color: [f32; 3]) -> Self {
        Self { position, color }
    }
}
glium::implement_vertex!(TriangleVertex, position, color);

#[derive(Debug)]
pub struct Triangle {
    vertex_buffer: glium::VertexBuffer<TriangleVertex>,
    draw_parameters: glium::DrawParameters<'static>,
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    projection: Matrix4<f32>,
}

impl Triangle {
    const VERTICES: [TriangleVertex; 3] = [
        TriangleVertex::new([0.5, -0.5], [1.0, 0.0, 0.0]), // bottom right
        TriangleVertex::new([-0.5, -0.5], [0.0, 1.0, 0.0]), // bottom left
        TriangleVertex::new([0.0, 0.5], [0.0, 0.0, 1.0]),  // top
    ];

    pub fn set_model(&mut self, model: Matrix4<f32>) {
        self.model = model;
    }

    pub fn set_view(&mut self, view: Matrix4<f32>) {
        self.view = view;
    }

    pub fn set_projection(&mut self, projection: Matrix4<f32>) {
        self.projection = projection;
    }

    pub fn new(display: &impl glium::backend::Facade) -> Self {
        Self {
            vertex_buffer: glium::VertexBuffer::new(display, &Self::VERTICES[..]).unwrap(),
            draw_parameters: glium::DrawParameters {
                depth: glium::Depth {
                    test: glium::DepthTest::IfLess,
                    write: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            model: Matrix4::identity(),
            view: Matrix4::identity(),
            projection: Matrix4::identity(),
        }
    }

    pub fn draw(&self, frame: &mut glium::Frame, shader: &glium::Program) {
        let model: [[f32; 4]; 4] = self.model.into();
        let view: [[f32; 4]; 4] = self.view.into();
        let projection: [[f32; 4]; 4] = self.projection.into();
        let uniforms = glium::uniform! {
            model: model,
            view: view,
            projection: projection,
        };
        frame
            .draw(
                &self.vertex_buffer,
                glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
                shader,
                &uniforms,
                &self.draw_parameters,
            )
            .unwrap();
    }
}

#[derive(Debug)]
pub struct Game {
    window: winit::window::Window,
    display: glium::Display<glium::glutin::surface::WindowSurface>,
    resource_loader: ResourceLoader,
    shader: glium::Program,
    input_helper: InputHelper,
    player_camera: PlayerCamera,
    text_painter: TextPainter,
    triangle0: Triangle,
    triangle1: Triangle,
    last_window_event: SystemTime,
    fps_counter: u32,
    last_fps_update: Instant,
    fps: f64,
    is_paused: bool,
    overlay_text: String,
}

impl Game {
    const VERTEX_SHADER: &'static str = r#"
        #version 140

        in vec2 position;
        in vec3 color;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        out vec3 vert_color;

        void main() {
            gl_Position = projection * view * model * vec4(position.xy, 0.0, 1.0);
            vert_color = color;
        }
    "#;

    const FRAGMENT_SHADER: &'static str = r#"
        #version 140

        in vec3 vert_color;

        out vec4 color;

        void main() {
            color = vec4(vert_color.xyz, 1.0);
        }
    "#;

    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> Self {
        let (window, display) = glutin::SimpleWindowBuilder::new()
            .with_title("New Cube Game!")
            .with_inner_size(1600, 1200)
            .build(event_loop);
        let resource_loader = ResourceLoader::with_default_res_directory().unwrap();
        Self {
            window,
            shader: glium::program::Program::from_source(
                &display,
                Self::VERTEX_SHADER,
                Self::FRAGMENT_SHADER,
                None,
            )
            .unwrap(),
            input_helper: InputHelper::new(),
            player_camera: PlayerCamera::new(Point3::new(0., 0., 1.), 0., -90.),
            text_painter: TextPainter::new(&display, &resource_loader),
            triangle0: Triangle::new(&display),
            triangle1: Triangle::new(&display),
            last_window_event: SystemTime::now(),
            fps_counter: 0,
            last_fps_update: Instant::now(),
            fps: f64::NAN,
            overlay_text: String::new(),
            is_paused: false,
            display,
            resource_loader,
        }
    }

    fn draw(&mut self) {
        let mut frame = self.display.draw();
        frame.clear_color_and_depth((0.8, 0.95, 1.0, 1.0), 1.);

        let view_matrix = self.player_camera.camera.view_matrix();
        let frame_size = self.window.inner_size();
        let projection_matrix = self.player_camera.camera.projection_matrix(Vector2::new(
            frame_size.width as f32,
            frame_size.height as f32,
        ));
        self.triangle0
            .set_model(Matrix4::from_translation([0.5, 0., 0.].into()));
        self.triangle0.set_view(view_matrix);
        self.triangle0.set_projection(projection_matrix);
        self.triangle0.draw(&mut frame, &self.shader);

        let s = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos() as f32)
            / 1_000_000_000.
            * 360.;
        self.triangle1.set_model(
            Matrix4::from_translation([-0.5, 0.5, 0.].into()) * Matrix4::from_angle_y(Deg(s)),
        );
        self.triangle1.set_view(view_matrix);
        self.triangle1.set_projection(projection_matrix);
        self.triangle1.draw(&mut frame, &self.shader);

        self.overlay_text.clear();
        if self.is_paused {
            let _ = writeln!(&mut self.overlay_text, "[ESC] Game Paused");
        } else {
            let _ = writeln!(&mut self.overlay_text, "CUBE GAME v0.0.0");
        }
        if self.fps.is_nan() {
            let _ = writeln!(&mut self.overlay_text, "FPS: --.--");
        } else {
            let _ = writeln!(&mut self.overlay_text, "FPS: {:.2}", self.fps);
        }
        self.draw_overlay_text(&mut frame);

        frame.finish().unwrap();

        self.fps_counter += 1;
        let now = Instant::now();
        let seconds_last_fps_update = now.duration_since(self.last_fps_update).as_secs_f64();
        if seconds_last_fps_update > 0.5 {
            self.fps = (self.fps_counter as f64) / seconds_last_fps_update;
            self.last_fps_update = now;
            self.fps_counter = 0;
        }
    }

    fn draw_overlay_text(&self, frame: &mut glium::Frame) {
        let overlay_text_x = 10.0f32;
        let mut x = overlay_text_x;
        let mut y = 10.0f32;
        for char in self.overlay_text.chars() {
            if char == '\n' {
                x = overlay_text_x;
                y += 16. * self.window.scale_factor() as f32;
            } else {
                let char_width = self.text_painter.draw_char(
                    frame,
                    Vector2::new(x, y),
                    16. * self.window.scale_factor() as f32,
                    char,
                );
                x += char_width;
            }
        }
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
        self.is_paused = !self.is_paused;
        if self.is_paused {
            self.ungrab_cursor();
        } else {
            self.grab_cursor();
        }
    }

    fn before_window_event(&mut self, duration_since_last_window_event: Duration) {
        if !self.is_paused {
            let mut movement = <Vector3<f32>>::zero();
            if self.input_helper.key_is_down(KeyCode::KeyW) {
                movement.y += 1.0;
            }
            if self.input_helper.key_is_down(KeyCode::KeyS) {
                movement.y -= 1.0;
            }
            if self.input_helper.key_is_down(KeyCode::KeyA) {
                movement.x -= 1.0;
            }
            if self.input_helper.key_is_down(KeyCode::KeyD) {
                movement.x += 1.0;
            }
            if self.input_helper.key_is_down(KeyCode::Space) {
                movement.z += 1.0;
            }
            if self.input_helper.key_is_down(KeyCode::KeyR) {
                movement.z -= 1.0;
            }
            movement *= duration_since_last_window_event.as_secs_f32();
            self.player_camera.move_(movement);
        }
    }

    #[expect(unused_variables)]
    fn key_down(&mut self, key_code: KeyCode, text: Option<&str>, is_repeat: bool) {
        if key_code == KeyCode::Escape && !is_repeat {
            self.toggle_paused();
        }
    }

    #[expect(unused_variables)]
    fn key_up(&mut self, key_code: KeyCode) {}

    fn cursor_moved(&mut self, delta: Vector2<f32>) {
        if !self.is_paused {
            self.player_camera.cursor_moved(0.1, delta);
        }
    }
}

impl winit::application::ApplicationHandler for Game {
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
        let now = std::time::SystemTime::now();
        let duration_since_last_window_event = now.duration_since(self.last_window_event).unwrap();
        self.last_window_event = now;
        self.before_window_event(duration_since_last_window_event);

        match event {
            winit::event::WindowEvent::CloseRequested => event_loop.exit(),
            winit::event::WindowEvent::RedrawRequested => {
                self.draw();
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

        self.window.request_redraw();
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
