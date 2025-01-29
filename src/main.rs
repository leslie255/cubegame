#![allow(dead_code)]

use std::time::{Instant, SystemTime};

use cgmath::*;

use font::Font;
use glium::{
    Surface,
    backend::glutin,
    winit::{
        self,
        event::KeyEvent,
        keyboard::{KeyCode, PhysicalKey},
    },
};
use resource::ResourceLoader;

pub mod font;
pub mod resource;

pub fn matrix4_to_array<T>(matrix: Matrix4<T>) -> [[T; 4]; 4] {
    matrix.into()
}

pub fn matrix3_to_array<T>(matrix: Matrix3<T>) -> [[T; 3]; 3] {
    matrix.into()
}

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharacterMesh {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

glium::implement_vertex!(CharacterMesh, position, uv);

#[derive(Debug)]
pub struct TextPainter {
    vertices: glium::VertexBuffer<CharacterMesh>,
    indices: glium::IndexBuffer<u32>,
    shader: glium::Program,
    draw_parameters: glium::DrawParameters<'static>,
    font: Font,
}

impl TextPainter {
    const VERTEX_SHADER: &'static str = r#"
        #version 140

        in vec2 position;
        in vec2 uv;
        out vec2 vert_uv;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform mat3 uv_matrix;

        void main() {
            gl_Position = projection * view * model * vec4(position.xy, 0.0, 1.0);
            vert_uv = (uv_matrix * vec3(uv, 1.0)).xy;
        }
    "#;

    const FRAGMENT_SHADER: &'static str = r#"
        #version 140

        in vec2 vert_uv;
        out vec4 color;

        uniform sampler2D tex;
        uniform vec4 fg_color;
        uniform vec4 bg_color;

        void main() {
            color = texture(tex, vert_uv);
            color = vec4(
                color.a * fg_color.r,
                color.a * fg_color.g,
                color.a * fg_color.b,
                color.a * fg_color.a);
            color += vec4(
                (1 - color.a) * bg_color.r,
                (1 - color.a) * bg_color.g,
                (1 - color.a) * bg_color.b,
                (1 - color.a) * bg_color.a);
        }
    "#;

    #[rustfmt::skip]
    const VERTICES: [CharacterMesh; 4] = [
        CharacterMesh { position: [0., 0.], uv: [0., 0.] },
        CharacterMesh { position: [1., 0.], uv: [1., 0.] },
        CharacterMesh { position: [0., 1.], uv: [0., 1.] },
        CharacterMesh { position: [1., 1.], uv: [1., 1.] },
    ];

    #[rustfmt::skip]
    const INDICES: [u32; 6] = [
        2, 1, 0, //
        1, 2, 3, //
    ];

    pub fn new(display: &impl glium::backend::Facade, resource_loader: &ResourceLoader) -> Self {
        Self {
            vertices: glium::VertexBuffer::new(display, &Self::VERTICES[..]).unwrap(),
            indices: glium::IndexBuffer::new(
                display,
                glium::index::PrimitiveType::TrianglesList,
                &Self::INDICES[..],
            )
            .unwrap(),
            shader: glium::program::Program::from_source(
                display,
                Self::VERTEX_SHADER,
                Self::FRAGMENT_SHADER,
                None,
            )
            .unwrap(),
            draw_parameters: glium::DrawParameters {
                depth: glium::Depth {
                    test: glium::DepthTest::Overwrite,
                    write: false,
                    ..Default::default()
                },
                blend: glium::Blend {
                    alpha: glium::BlendingFunction::Addition {
                        source: glium::LinearBlendingFactor::One,
                        destination: glium::LinearBlendingFactor::One,
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
            font: {
                let mut font =
                    Font::load_from_path(resource_loader, "font/big_blue_terminal.json", false);
                let image = glium::texture::RawImage2d::from_raw_rgba(
                    font.atlas().to_rgba8().into_raw(),
                    (font.atlas().width(), font.atlas().height()),
                );
                font.gl_texture = Some(glium::Texture2d::new(display, image).unwrap());
                font
            },
        }
    }

    pub fn draw_char(
        &self,
        frame: &mut glium::Frame,
        position: Vector2<f32>,
        size: f32,
        char: char,
    ) {
        let (frame_width, frame_height) = frame.get_dimensions();
        let view: Matrix4<f32> = Matrix4::identity();
        let projection = cgmath::ortho(0., frame_width as f32, frame_height as f32, 0., -1., 1.);
        let model = Matrix4::from_translation(Vector3::new(position.x, position.y, 0.));
        let model = model * Matrix4::from_nonuniform_scale(size, size, 1.);
        let quad = self.font.sample(char);
        let uv_matrix = Matrix3::from_translation(vec2(quad.left, quad.top))
            * Matrix3::from_nonuniform_scale(quad.width(), quad.height());
        let texture_sampler = self
            .font
            .gl_texture
            .as_ref()
            .unwrap()
            .sampled()
            .magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest)
            .minify_filter(glium::uniforms::MinifySamplerFilter::Nearest);
        let uniforms = glium::uniform! {
            model: matrix4_to_array(model),
            view: matrix4_to_array(view),
            projection: matrix4_to_array(projection),
            uv_matrix: matrix3_to_array(uv_matrix),
            tex: texture_sampler,
            fg_color: [1., 1., 1., 1.0f32],
            bg_color: [0.5, 0.5, 0.5, 1.0f32],
        };
        frame
            .draw(
                &self.vertices,
                &self.indices,
                &self.shader,
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
    fps_counter_seconds: f64,
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
            fps_counter_seconds: 0.,
            display,
            resource_loader,
        }
    }

    fn draw(&mut self) {
        let before_drawing = Instant::now();

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

        self.text_painter
            .draw_char(&mut frame, Vector2::new(10., 10.), 100., 'a');

        frame.finish().unwrap();

        let after_drawing = Instant::now();
        let frame_time = after_drawing.duration_since(before_drawing);
        self.fps_counter_seconds += frame_time.as_secs_f64();
        self.fps_counter += 1;
        if self.fps_counter_seconds >= 0.5 {
            println!(
                "FPS: {}",
                (self.fps_counter as f64) / self.fps_counter_seconds
            );
            self.fps_counter_seconds = 0.;
            self.fps_counter = 0;
        }
    }

    #[expect(unused_variables)]
    fn key_down(&mut self, key_code: KeyCode, text: Option<&str>, is_repeat: bool) {}

    #[expect(unused_variables)]
    fn key_up(&mut self, key_code: KeyCode) {}
}

impl winit::application::ApplicationHandler for Game {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        _ = event_loop;
        self.window
            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
            .unwrap_or_else(|_| {
                self.window
                    .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                    .unwrap_or_else(|_| {
                        eprintln!("[WARNING] Unable to grab cursor");
                    })
            });
        self.window.set_cursor_visible(false);
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
                self.player_camera
                    .cursor_moved(0.1, Vector2::new(delta.0 as f32, delta.1 as f32));
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

fn main() {
    let event_loop = winit::event_loop::EventLoop::builder().build().unwrap();

    let mut game = Game::new(&event_loop);

    event_loop.run_app(&mut game).unwrap();
}
