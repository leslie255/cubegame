use std::time::{Duration, Instant, SystemTime};

use cgmath::*;
use glium::{
    Surface,
    winit::{
        self,
        keyboard::{KeyCode, PhysicalKey},
    },
};

use crate::{
    input::InputHelper,
    mesh::{self, Color},
    resource::ResourceLoader,
    text::{Font, Line},
};

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
        self.pitch = self.pitch.clamp(-89.99, 89.99);
        self.yaw = ((self.yaw + 180.) % 360.) - 180.;
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
pub struct Cube<'res> {
    vertex_buffer: glium::VertexBuffer<BlockVertex>,
    index_buffer: glium::IndexBuffer<u32>,
    shader: &'res glium::Program,
    texture_atlas: &'res glium::Texture2d,
    pub model: Matrix4<f32>,
    pub view: Matrix4<f32>,
    pub projection: Matrix4<f32>,
}

impl<'res> Cube<'res> {
    const VERTICES: &'static [BlockVertex] = &[
        // South
        BlockVertex::new([0., 0., 0.], [1.0, 1.0]), // A 0
        BlockVertex::new([1., 0., 0.], [0.0, 1.0]), // B 1
        BlockVertex::new([1., 1., 0.], [0.0, 0.0]), // C 2
        BlockVertex::new([0., 1., 0.], [1.0, 0.0]), // D 3
        // North
        BlockVertex::new([0., 0., 1.], [0.0, 1.0]), // E 4
        BlockVertex::new([1., 0., 1.], [1.0, 1.0]), // F 5
        BlockVertex::new([1., 1., 1.], [1.0, 0.0]), // G 6
        BlockVertex::new([0., 1., 1.], [0.0, 0.0]), // H 7
        // East
        BlockVertex::new([1., 0., 0.], [1.0, 1.0]), // B 8
        BlockVertex::new([1., 1., 0.], [1.0, 0.0]), // C 9
        BlockVertex::new([1., 1., 1.], [0.0, 0.0]), // G 10
        BlockVertex::new([1., 0., 1.], [0.0, 1.0]), // F 11
        // West
        BlockVertex::new([0., 1., 0.], [0.0, 0.0]), // D 12
        BlockVertex::new([0., 0., 0.], [0.0, 1.0]), // A 13
        BlockVertex::new([0., 0., 1.], [1.0, 1.0]), // E 14
        BlockVertex::new([0., 1., 1.], [1.0, 0.0]), // H 15
        // Up
        BlockVertex::new([1., 1., 0.], [0.0, 1.0]), // C 16
        BlockVertex::new([0., 1., 0.], [1.0, 1.0]), // D 17
        BlockVertex::new([0., 1., 1.], [1.0, 0.0]), // H 18
        BlockVertex::new([1., 1., 1.], [0.0, 0.0]), // G 19
        // Down
        BlockVertex::new([0., 0., 0.], [0.0, 1.0]), // A 20
        BlockVertex::new([1., 0., 0.], [1.0, 1.0]), // B 21
        BlockVertex::new([1., 0., 1.], [1.0, 0.0]), // F 22
        BlockVertex::new([0., 0., 1.], [0.0, 0.0]), // E 23
    ];

    const INDICES: &'static [u32] = &[
        /* South */ 0, 3, 2, 2, 1, 0, //
        /* North */ 4, 5, 6, 6, 7, 4, //
        /* East  */ 8, 9, 10, 10, 11, 8, //
        /* West  */ 12, 13, 14, 14, 15, 12, //
        /* Up    */ 16, 17, 18, 18, 19, 16, //
        /* Down  */ 20, 21, 22, 22, 23, 20, //
    ];

    pub fn new(display: &impl glium::backend::Facade, resources: &'res GameResources) -> Self {
        Self {
            vertex_buffer: glium::VertexBuffer::new(display, Self::VERTICES).unwrap(),
            index_buffer: glium::IndexBuffer::new(
                display,
                glium::index::PrimitiveType::TrianglesList,
                Self::INDICES,
            )
            .unwrap(),
            shader: &resources.shader_cube,
            texture_atlas: &resources.block_atlas,
            model: Matrix4::identity(),
            view: Matrix4::identity(),
            projection: Matrix4::identity(),
        }
    }

    pub fn draw(&self, frame: &mut glium::Frame) {
        frame
            .draw(
                &self.vertex_buffer,
                &self.index_buffer,
                self.shader,
                &glium::uniform! {
                    model: mesh::matrix4_to_array(self.model),
                    view: mesh::matrix4_to_array(self.view),
                    projection: mesh::matrix4_to_array(self.projection),
                    texture_atlas: mesh::texture_sampler(self.texture_atlas),
                },
                &mesh::default_3d_draw_parameters(),
            )
            .unwrap();
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangleVertex {
    pub position: [f32; 2],
    pub color: [f32; 3],
}

glium::implement_vertex!(TriangleVertex, position, color);

impl TriangleVertex {
    pub const fn new(position: [f32; 2], color: [f32; 3]) -> Self {
        Self { position, color }
    }
}

#[derive(Debug)]
pub struct Triangle {
    vertex_buffer: glium::VertexBuffer<TriangleVertex>,
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
                &glium::DrawParameters {
                    backface_culling: glium::BackfaceCullingMode::CullingDisabled,
                    ..mesh::default_3d_draw_parameters()
                },
            )
            .unwrap();
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
                Line::with_string(font, shader, display, "CUBE GAME v0.0.0".into()),
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
            self.set_line(display, 1, format!("FPS: {fps:.3}").as_str());
        }
    }

    fn set_camera_xyz(&mut self, display: &impl glium::backend::Facade, camera_xyz: Point3<f32>) {
        self.set_line(
            display,
            2,
            format!(
                "Camera XYZ: {:.3}, {:.3}, {:.3}",
                camera_xyz.x, camera_xyz.y, camera_xyz.z
            )
            .as_str(),
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
            "North (+Z)"
        } else {
            "South (-Z)"
        };
        self.set_line(
            display,
            3,
            format!(
                "Camera pitch/yaw: {:.3}deg, {:.3}deg",
                pitch_yaw.0, pitch_yaw.1
            )
            .as_str(),
        );
        self.set_line(display, 4, format!("Facing: {facing}").as_str());
    }

    fn draw(&self, frame: &mut glium::Frame, content_scale: f32) {
        let font_size = 16. * content_scale;
        for (i, line) in self.lines.iter().enumerate() {
            line.draw(
                frame,
                /* position   */ Point2::new(10., 10. + font_size * i as f32),
                /* foreground */ Color::new(1., 1., 1., 1.),
                /* background */ Color::new(0.5, 0.5, 0.5, 0.6),
                /* font size  */ font_size,
            );
        }
    }
}

#[derive(Debug)]
pub struct GameResources {
    loader: ResourceLoader,
    shader_3d: glium::Program,
    shader_text: glium::Program,
    shader_cube: glium::Program,
    block_atlas: glium::Texture2d,
    font: Font,
}

impl GameResources {
    pub fn load(display: &impl glium::backend::Facade) -> Self {
        let loader = ResourceLoader::with_default_res_directory().unwrap();
        Self {
            shader_3d: Self::load_shader(display, &loader, "shader/3d"),
            shader_text: Self::load_shader(display, &loader, "shader/text"),
            shader_cube: Self::load_shader(display, &loader, "shader/cube"),
            block_atlas: Self::load_texture(display, &loader, "texture/block_atlas.png"),
            font: Self::load_font(display, &loader, "font/big_blue_terminal.json"),
            loader,
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
        glium::Texture2d::new(display, image).unwrap()
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
pub struct Game<'res> {
    window: winit::window::Window,
    display: glium::Display<glium::glutin::surface::WindowSurface>,
    resources: &'res GameResources,
    input_helper: InputHelper,
    fps_counter: FpsCounter,
    player_camera: PlayerCamera,
    triangle0: Triangle,
    triangle1: Triangle,
    cube: Cube<'res>,
    info_text: InfoText<'res>,
    last_window_event: SystemTime,
    is_paused: bool,
}

impl<'res> Game<'res> {
    pub fn new(
        resources: &'res GameResources,
        window: winit::window::Window,
        display: glium::Display<glium::glutin::surface::WindowSurface>,
    ) -> Self {
        Self {
            window,
            input_helper: InputHelper::new(),
            fps_counter: FpsCounter::new(),
            player_camera: PlayerCamera::new(Point3::new(0., 0., 1.), 0., -90.),
            cube: Cube::new(&display, resources),
            triangle0: Triangle::new(&display),
            triangle1: Triangle::new(&display),
            info_text: InfoText::new(&display, &resources.font, &resources.shader_text),
            last_window_event: SystemTime::now(),
            is_paused: false,
            display,
            resources,
        }
    }

    fn draw(&mut self) {
        let mut frame = self.display.draw();
        // let clear_color = (0.8, 0.95, 1.0, 1.0);
        let clear_color = (0.1, 0.1, 0.1, 1.);
        frame.clear_color_and_depth(clear_color, 1.);

        let view_matrix = self.player_camera.camera.view_matrix();
        let frame_size = self.window.inner_size();
        let projection_matrix = self.player_camera.camera.projection_matrix(Vector2::new(
            frame_size.width as f32,
            frame_size.height as f32,
        ));
        self.triangle0
            .set_model(Matrix4::from_translation([2., 0., 0.].into()));
        self.triangle0.set_view(view_matrix);
        self.triangle0.set_projection(projection_matrix);
        self.triangle0.draw(&mut frame, &self.resources.shader_3d);

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
        self.triangle1.draw(&mut frame, &self.resources.shader_3d);

        self.cube.view = view_matrix;
        self.cube.projection = projection_matrix;
        self.cube.draw(&mut frame);

        self.info_text
            .set_camera_xyz(&self.display, self.player_camera.camera.position);
        self.info_text.set_camera_direction(
            &self.display,
            self.player_camera.pitch_yaw(),
            self.player_camera.camera.direction,
        );

        self.info_text
            .draw(&mut frame, self.window.scale_factor() as f32);

        frame.finish().unwrap();

        if let Some(fps) = self.fps_counter.frame() {
            self.info_text.set_fps(&self.display, fps);
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
        self.info_text.set_is_paused(&self.display, self.is_paused);
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

impl winit::application::ApplicationHandler for Game<'_> {
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
