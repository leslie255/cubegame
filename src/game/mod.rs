use std::{
    fmt::{self, Write as _},
    mem,
    sync::Arc,
    thread,
    time::Duration,
};

use cgmath::{num_traits::Float as _, *};

use pollster::FutureExt as _;
use winit::{
    event::{ElementState, KeyEvent, MouseButton},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Fullscreen, Window},
};

use crate::{
    ProgramArgs,
    chunk::ChunkRenderer,
    game::debug_toggles::DebugToggles,
    impl_as_bind_group,
    input::InputHelper,
    text::{Text, TextRenderer},
    utils::{BoolToggle, WithY as _},
    wgpu_utils::{self, DepthTextureView, UniformBuffer},
    world::World,
};

mod app;
mod debug_toggles;
mod fps_counter;
mod resource;

pub use app::*;
pub use resource::*;

pub fn initialize_wgpu() -> (wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .block_on()
        .unwrap();
    let features = wgpu::FeaturesWGPU::POLYGON_MODE_LINE;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: features.into(),
            ..Default::default()
        })
        .block_on()
        .unwrap();
    (instance, adapter, device, queue)
}

#[derive(Debug, Clone)]
struct PostprocessBindGroup {
    color_texture: wgpu::TextureView,
    depth_texture: DepthTextureView,
    sampler: wgpu::Sampler,
    gamma: UniformBuffer<f32>,
    fog_start: UniformBuffer<f32>,
}

impl_as_bind_group! {
    PostprocessBindGroup {
        0 => color_texture: wgpu::TextureView,
        1 => depth_texture: DepthTextureView,
        2 => sampler: wgpu::Sampler,
        3 => gamma: UniformBuffer<f32>,
        4 => fog_start: UniformBuffer<f32>,
    }
}

#[derive(Debug)]
struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    resources: GameResources,
}

#[derive(Debug)]
struct Game<'scope, 'cx>
where
    'cx: 'scope,
{
    device: &'cx wgpu::Device,
    queue: &'cx wgpu::Queue,
    device_name: String,
    backend: wgpu::Backend,
    window: Arc<Window>,
    resources: &'cx GameResources,
    frame_size_u: Vector2<u32>,
    frame_size: Vector2<f32>,
    window_surface: wgpu::Surface<'static>,
    window_surface_format: wgpu::TextureFormat,
    /// The scene before post-processing.
    scene_texture: wgpu::Texture,
    /// The scene before post-processing.
    scene_texture_view: wgpu::TextureView,
    depth_stencil_texture: wgpu::Texture,
    depth_stencil_texture_view: wgpu::TextureView,
    fps: f64,
    is_paused: bool,
    text_renderer: TextRenderer<'cx>,
    debug_toggles: DebugToggles,
    debug_text: Text,
    debug_text_string: String,
    debug_text_needs_updating: bool,
    chunk_renderer: ChunkRenderer,
    chunk_renderer_wireframe: Option<ChunkRenderer>,
    postprocess_pipeline: wgpu::RenderPipeline,
    postprocess_bind_group: PostprocessBindGroup,
    postprocess_bind_group_wgpu: wgpu::BindGroup,
    postprocess_bind_group_layout: wgpu::BindGroupLayout,
    world: Arc<World<'scope, 'cx>>,
    player_camera: PlayerCamera,
}

fn print_features(features: wgpu::Features) {
    let mut string = String::new();
    _ = write!(&mut string, "features: ");
    let mut is_first = true;
    for (name, _) in features.iter_names() {
        if !is_first {
            _ = write!(&mut string, ", ");
        }
        is_first = false;
        _ = write!(&mut string, "{name}");
    }
    log::info!("{string}");
}

impl<'scope, 'cx> Game<'scope, 'cx> {
    const DEPTH_STENCIL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    const SCENE_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

    fn new(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        context: &'cx Context,
        window: Arc<Window>,
        thread_scope: &'scope thread::Scope<'scope, 'cx>,
        program_args: &ProgramArgs,
    ) -> Self {
        let features = context.device.features();
        print_features(features);
        let adapter_info = adapter.get_info();
        let window_surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let surface_capabilities = window_surface.get_capabilities(adapter);
        let window_surface_format = surface_capabilities.formats[0];

        let text_renderer = TextRenderer::create(
            &context.device,
            &context.queue,
            &context.resources,
            Self::SCENE_TEXTURE_FORMAT,
            Some(Self::DEPTH_STENCIL_FORMAT),
        );
        let debug_text = text_renderer.create_text(&context.device, "CUBE GAME v0.0.0");
        debug_text.set_bg_color(&context.queue, vec4(0.2, 0.2, 0.2, 1.0));
        debug_text.set_fg_color(&context.queue, vec4(1.0, 1.0, 1.0, 1.0));

        let chunk_renderer = ChunkRenderer::new(
            &context.device,
            &context.queue,
            &context.resources,
            Self::SCENE_TEXTURE_FORMAT,
            Some(Self::DEPTH_STENCIL_FORMAT),
        );
        let world = World::new(
            &context.device,
            &context.resources,
            thread_scope,
            program_args,
        );

        let physical_size = window.inner_size();
        let frame_size_u = vec2(physical_size.width, physical_size.height);
        let frame_size = frame_size_u.map(|u| u as f32);

        let depth_stencil_texture =
            Self::create_depth_stencil_texture(&context.device, frame_size_u);
        let depth_stencil_texture_view = depth_stencil_texture.create_view(&Default::default());

        let scene_texture = Self::create_scene_texture(&context.device, frame_size_u);
        let scene_texture_view = scene_texture.create_view(&Default::default());

        let postprocess_bind_group_layout =
            wgpu_utils::create_bind_group_layout::<PostprocessBindGroup>(&context.device);
        let postprocess_pipeline = {
            let pipeline_layout =
                context
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&postprocess_bind_group_layout],
                        push_constant_ranges: &[],
                    });
            context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &context.resources.shader_postprocess,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &context.resources.shader_postprocess,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(window_surface_format.into())],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                })
        };

        let fog_start = 1. - 1. / (world.view_distance() * 2) as f32;
        log::info!("fog_start = {fog_start}");
        // let fog_start = 0.99;
        let postprocess_bind_group = PostprocessBindGroup {
            color_texture: scene_texture.create_view(&Default::default()),
            depth_texture: depth_stencil_texture
                .create_view(&Default::default())
                .into(),
            sampler: context.device.create_sampler(&Default::default()),
            gamma: UniformBuffer::create_init(&context.device, 2.2),
            fog_start: UniformBuffer::create_init(&context.device, fog_start),
        };
        let postprocess_bind_group_wgpu = wgpu_utils::create_bind_group(
            &context.device,
            &postprocess_bind_group_layout,
            &postprocess_bind_group,
        );

        let player_camera = PlayerCamera {
            position: point3(0., 0., 10000000.),
            pitch: 0.,
            yaw: -90.,
        };

        world.generate_initial_area(player_camera.position);

        let debug_toggles = DebugToggles::default();
        println!("Tips: Debug keys:");
        for (key, _enabled, description) in debug_toggles.keys() {
            let key = key.to_ascii_uppercase();
            println!("[F3+{key}] {description}");
        }

        let mut self_ = Self {
            device: &context.device,
            queue: &context.queue,
            device_name: adapter_info.name,
            backend: adapter_info.backend,
            window,
            resources: &context.resources,
            window_surface,
            window_surface_format,
            scene_texture,
            scene_texture_view,
            depth_stencil_texture,
            depth_stencil_texture_view,
            fps: f64::NAN,
            is_paused: false,
            text_renderer,
            debug_toggles,
            debug_text,
            debug_text_needs_updating: true,
            debug_text_string: String::new(),
            frame_size_u,
            frame_size,
            chunk_renderer,
            chunk_renderer_wireframe: None,
            postprocess_pipeline,
            postprocess_bind_group,
            postprocess_bind_group_wgpu,
            postprocess_bind_group_layout,
            world,
            player_camera,
        };
        self_.configure_surface();
        self_.pause_changed();
        self_
    }

    fn create_scene_texture(device: &wgpu::Device, size: Vector2<u32>) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::SCENE_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn create_depth_stencil_texture(device: &wgpu::Device, size: Vector2<u32>) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_STENCIL_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    pub fn keyboard_input(&mut self, event: KeyEvent, input_helper: &InputHelper) {
        match event.physical_key {
            PhysicalKey::Code(KeyCode::F3) if event.state.is_pressed() => {
                self.debug_toggles.f3_pressed();
            }
            PhysicalKey::Code(KeyCode::F11) if event.state.is_pressed() & !event.repeat => {
                let is_fullscreen = self
                    .window
                    .fullscreen()
                    .is_some_and(|fullscreen| matches!(fullscreen, Fullscreen::Borderless(_)));
                if is_fullscreen {
                    self.window.set_fullscreen(None);
                } else {
                    self.window
                        .set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                }
            }
            PhysicalKey::Code(KeyCode::Escape) if event.state.is_pressed() & !event.repeat => {
                self.is_paused.toggle();
                self.pause_changed();
            }
            key if event.state.is_pressed() & input_helper.key_is_down(KeyCode::F3) => {
                self.debug_toggles.key_pressed_with_f3(key);
            }
            _ => (),
        }
        self.update_debug_text();
    }

    #[expect(unused_variables)]
    pub fn mouse_input(
        &mut self,
        state: ElementState,
        button: MouseButton,
        input_helper: &InputHelper,
    ) {
    }

    #[expect(unused_variables)]
    pub fn cursor_moved(&mut self, delta: Vector2<f32>, input_helper: &InputHelper) {
        if !self.is_paused {
            self.player_camera.cursor_moved(0.1, delta);
        }
    }

    pub fn frame_resized(&mut self) {
        let physical_size = self.window.inner_size();
        self.frame_size_u = vec2(physical_size.width, physical_size.height);
        self.frame_size = self.frame_size_u.map(|u| u as f32);
        self.configure_surface();
        self.depth_stencil_texture =
            Self::create_depth_stencil_texture(self.device, self.frame_size_u);
        self.depth_stencil_texture_view =
            self.depth_stencil_texture.create_view(&Default::default());
        self.scene_texture = Self::create_scene_texture(self.device, self.frame_size_u);
        self.scene_texture_view = self.scene_texture.create_view(&Default::default());
        self.postprocess_bind_group.color_texture =
            self.scene_texture.create_view(&Default::default());
        self.postprocess_bind_group.depth_texture = self
            .depth_stencil_texture
            .create_view(&Default::default())
            .into();
        self.postprocess_bind_group_wgpu = wgpu_utils::create_bind_group(
            self.device,
            &self.postprocess_bind_group_layout,
            &self.postprocess_bind_group,
        );
    }

    fn configure_surface(&mut self) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.window_surface_format,
            view_formats: vec![self.window_surface_format.add_srgb_suffix()],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: self.frame_size_u.x,
            height: self.frame_size_u.y,
            desired_maximum_frame_latency: 3,
            present_mode: wgpu::PresentMode::AutoVsync,
        };
        self.window_surface.configure(self.device, &surface_config);
    }

    pub fn update_fps(&mut self, fps: f64) {
        self.fps = fps;
        self.debug_text_needs_updating = true;
    }

    fn debug_overlay_text(&self, out: &mut impl fmt::Write) -> fmt::Result {
        if !self.is_paused {
            writeln!(out, "CUBE GAME v0.0.0")?;
        } else {
            writeln!(out, "[ESC] PAUSED")?;
        }
        writeln!(out, "Backend: {}, {}", self.backend, self.device_name)?;
        writeln!(out, "FPS: {}", self.fps)?;
        let p = self.player_camera.position;
        writeln!(out, "XYZ: {:.04} {:.04} {:.04}", p.x, p.y, p.z)?;
        writeln!(
            out,
            "PITCH YAW: {:.04} {:.04}",
            self.player_camera.pitch, self.player_camera.yaw
        )?;
        Ok(())
    }

    fn update_debug_text(&mut self) {
        let mut debug_text = mem::take(&mut self.debug_text_string);
        debug_text.clear();
        if self.debug_toggles.show_debug_overlay {
            _ = self.debug_overlay_text(&mut debug_text);
        } else if self.is_paused {
            _ = writeln!(&mut debug_text, "[ESC] PAUSED");
        }
        _ = self.debug_toggles.prompt_text(&mut debug_text);
        self.debug_text_string = debug_text;
        self.text_renderer
            .update_text(self.device, &mut self.debug_text, &self.debug_text_string);
    }

    pub fn frame(&mut self) {
        if self.debug_text_needs_updating {
            self.update_debug_text();
        }

        // Scene.
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut render_pass = self.scene_render_pass(&mut encoder);

        self.draw_chunks(&mut render_pass);
        self.draw_debug_text(&mut render_pass);

        drop(render_pass);

        // Postprocess.
        let surface_texture = self
            .window_surface
            .get_current_texture()
            .expect("failed to acquire next swapchain texture");
        let surface_texture_view =
            surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor {
                    format: Some(self.window_surface_format.add_srgb_suffix()),
                    ..Default::default()
                });

        let mut render_pass = self.postprocess_render_pass(&mut encoder, &surface_texture_view);

        render_pass.set_pipeline(&self.postprocess_pipeline);
        render_pass.set_bind_group(0, &self.postprocess_bind_group_wgpu, &[]);
        render_pass.draw(0..6, 0..1);

        drop(render_pass);
        self.queue.submit([encoder.finish()]);

        self.window.pre_present_notify();
        surface_texture.present();
    }

    fn scene_render_pass<'a>(&self, encoder: &'a mut wgpu::CommandEncoder) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.scene_texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_stencil_texture_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0f32),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        })
    }

    fn postprocess_render_pass<'a>(
        &self,
        encoder: &'a mut wgpu::CommandEncoder,
        surface_texture_view: &wgpu::TextureView,
    ) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("postprocess"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface_texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        })
    }

    fn draw_debug_text(&self, render_pass: &mut wgpu::RenderPass) {
        let projection = cgmath::ortho(0., self.frame_size.x, self.frame_size.y, 0., -1.0, 1.0);
        self.debug_text.set_projection(self.queue, projection);

        let font_size = 17.0 * self.window.scale_factor() as f32;
        let model_view = Matrix4::from_scale(font_size);
        self.debug_text.set_model_view(self.queue, model_view);
        self.text_renderer.draw_text(render_pass, &self.debug_text);
    }

    fn draw_chunks(&mut self, render_pass: &mut wgpu::RenderPass) {
        let chunk_renderer = if self.debug_toggles.wireframe_mode {
            if let Some(chunk_renderer_wireframe) = &self.chunk_renderer_wireframe {
                chunk_renderer_wireframe
            } else if let renderer @ Some(_) = ChunkRenderer::new_wireframe_mode(
                self.device,
                self.queue,
                self.resources,
                Self::SCENE_TEXTURE_FORMAT,
                Some(Self::DEPTH_STENCIL_FORMAT),
            ) {
                self.chunk_renderer_wireframe = renderer;
                self.chunk_renderer_wireframe.as_ref().unwrap()
            } else {
                &self.chunk_renderer
            }
        } else {
            self.chunk_renderer_wireframe = None;
            &self.chunk_renderer
        };
        let far = self.world.view_distance() as f32 * 32.0;
        let projection = self.player_camera.projection_matrix(far, self.frame_size);
        chunk_renderer.set_projection(self.queue, projection);
        chunk_renderer.set_sun(self.queue, vec3(1., -2., 0.5).normalize());
        chunk_renderer.set_gray_world(self.queue, self.debug_toggles.gray_world);
        chunk_renderer.begin_drawing(render_pass);
        let (player_chunk_id, _) = World::world_to_local_coord_f32(self.player_camera.position);
        self.world
            .chunks()
            .for_each_loaded_chunk(|chunk_id, chunk| {
                let distance2 = point2(player_chunk_id.x as f32, player_chunk_id.z as f32)
                    .distance2(point2(chunk_id.x as f32, chunk_id.z as f32));
                if distance2 > (self.world.view_distance() as f32).powi(2) {
                    return;
                }
                let Some(mesh) = &chunk.client.mesh else {
                    return;
                };
                let translation = chunk_id.to_vec().map(|i| i as f64 * 32.);
                let model_view = self.player_camera.view_matrix(translation);
                mesh.set_model_view(self.queue, model_view);
                chunk_renderer.draw_chunk(render_pass, mesh);
            });
    }

    fn pause_changed(&mut self) {
        if !self.is_paused {
            _ = self.window.set_cursor_grab(CursorGrabMode::Locked);
            self.window.set_cursor_visible(false);
        } else {
            _ = self.window.set_cursor_grab(CursorGrabMode::None);
            self.window.set_cursor_visible(true);
        }
        self.debug_text_needs_updating = true;
    }

    pub fn before_handling_window_event(
        &mut self,
        input_helper: &InputHelper,
        duration_since_last_event: Duration,
    ) {
        if self.is_paused {
            return;
        }
        let mut movement = vec3(0., 0., 0.);
        if input_helper.key_is_down(KeyCode::KeyW) {
            movement.z += 1.;
        }
        if input_helper.key_is_down(KeyCode::KeyS) {
            movement.z -= 1.;
        }
        if input_helper.key_is_down(KeyCode::KeyA) {
            movement.x -= 1.;
        }
        if input_helper.key_is_down(KeyCode::KeyD) {
            movement.x += 1.;
        }
        if input_helper.key_is_down(KeyCode::Space) {
            movement.y += 1.;
        }
        if input_helper.key_is_down(KeyCode::KeyR) {
            movement.y -= 1.;
        }
        movement = movement.normalize_to(4.);
        if movement.x.is_nan() | movement.y.is_nan() | movement.z.is_nan() {
            return;
        }
        if input_helper.key_is_down(KeyCode::F3) {
            movement *= 64.;
        } else if input_helper.key_is_down(KeyCode::ControlLeft) {
            movement.z *= 4.;
        }
        movement *= duration_since_last_event.as_secs_f32();

        self.player_camera.move_(movement);
        self.world.player_moved(self.player_camera.position);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PlayerCamera {
    position: Point3<f32>,
    pitch: f32,
    yaw: f32,
}

impl PlayerCamera {
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
    }

    pub fn move_(&mut self, delta: Vector3<f32>) {
        let direction = self.direction();
        let forward = direction.with_y(0.).normalize();
        let up = Vector3::unit_y();
        let right = forward.cross(up).normalize();

        let forward_scaled = delta.z * forward;
        let right_scaled = delta.x * right;
        let up_scaled = delta.y * up;
        self.position += forward_scaled + right_scaled + up_scaled;
    }

    pub fn direction(&self) -> Vector3<f32> {
        Vector3::new(
            self.yaw.to_radians().cos() * self.pitch.to_radians().cos(),
            self.pitch.to_radians().sin(),
            self.yaw.to_radians().sin() * self.pitch.to_radians().cos(),
        )
    }

    pub fn direction_f64(self) -> Vector3<f64> {
        let pitch = self.pitch as f64;
        let yaw = self.yaw as f64;
        Vector3::new(
            yaw.to_radians().cos() * pitch.to_radians().cos(),
            pitch.to_radians().sin(),
            yaw.to_radians().sin() * pitch.to_radians().cos(),
        )
    }

    pub fn position_f64(self) -> Point3<f64> {
        self.position.map(|f| f as f64)
    }

    pub fn view_matrix(self, translation: Vector3<f64>) -> Matrix4<f64> {
        Matrix4::look_to_rh(
            self.position_f64() - translation,
            self.direction_f64(),
            Vector3::unit_y(),
        )
    }

    pub fn projection_matrix(self, far: f32, frame_size: Vector2<f32>) -> Matrix4<f32> {
        cgmath::perspective(Deg(90.), frame_size.x / frame_size.y, 0.1, far)
    }
}
