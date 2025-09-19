use std::{fmt::Write as _, sync::Arc, thread, time::Duration};

use cgmath::*;

use pollster::FutureExt as _;
use winit::{
    event::{ElementState, KeyEvent, MouseButton},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window},
};

use crate::{
    ProgramArgs,
    block::BlockFace,
    chunk::ChunkRenderer,
    impl_as_bind_group,
    input::InputHelper,
    text::{Text, TextRenderer},
    utils::{BoolToggle, WithY as _},
    wgpu_utils::{self, DepthTextureView, UniformBuffer},
    world::{ChunkId, World},
};

mod app;
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
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
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
    debug_text: Text,
    debug_text_string: String,
    debug_text_needs_updating: bool,
    chunk_renderer: ChunkRenderer,
    postprocess_pipeline: wgpu::RenderPipeline,
    postprocess_bind_group: PostprocessBindGroup,
    postprocess_bind_group_wgpu: wgpu::BindGroup,
    postprocess_bind_group_layout: wgpu::BindGroupLayout,
    world: World<'scope, 'cx>,
    player_camera: PlayerCamera,
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

        let fog_start = 1. - 1. / world.view_distance() as f32 / 80.0;
        log::info!("{fog_start}");
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
            position: point3(0., 0., 0.),
            pitch: 0.,
            yaw: -90.,
        };

        world.generate_initial_area();

        let mut self_ = Self {
            device: &context.device,
            queue: &context.queue,
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
            debug_text,
            debug_text_needs_updating: true,
            debug_text_string: String::new(),
            frame_size_u,
            frame_size,
            chunk_renderer,
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

    #[expect(unused_variables)]
    pub fn keyboard_input(&mut self, event: KeyEvent, input_helper: &InputHelper) {
        if (!event.repeat)
            & (event.physical_key == PhysicalKey::Code(KeyCode::Escape))
            & (event.state.is_pressed())
        {
            self.is_paused.toggle();
            self.pause_changed();
        }
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
        self.update_debug_text();
        self.debug_text_needs_updating = true;
    }

    fn update_debug_text(&mut self) {
        self.debug_text_string.clear();
        _ = writeln!(&mut self.debug_text_string, "CUBE GAME v0.0.0");
        _ = writeln!(&mut self.debug_text_string, "FPS: {}", self.fps);
        let p = self.player_camera.position;
        _ = writeln!(
            &mut self.debug_text_string,
            "XYZ: {:.04} {:.04} {:.04}",
            p.x, p.y, p.z
        );
    }

    pub fn frame(&mut self) {
        if self.debug_text_needs_updating {
            self.update_debug_text();
            self.text_renderer.update_text(
                self.device,
                &mut self.debug_text,
                &self.debug_text_string,
            );
        }

        // Scene.
        let mut encoder_scene = self.device.create_command_encoder(&Default::default());
        let mut render_pass = encoder_scene.begin_render_pass(&wgpu::RenderPassDescriptor {
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
        });

        self.draw_chunks(&mut render_pass);
        self.draw_debug_text(&mut render_pass);

        drop(render_pass);
        self.queue.submit([encoder_scene.finish()]);

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

        let mut encoder_postprocess = self.device.create_command_encoder(&Default::default());
        let mut render_pass = encoder_postprocess.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("postprocess"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &surface_texture_view,
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
        });

        render_pass.set_pipeline(&self.postprocess_pipeline);
        render_pass.set_bind_group(0, &self.postprocess_bind_group_wgpu, &[]);
        render_pass.draw(0..6, 0..1);

        drop(render_pass);
        self.queue.submit([encoder_postprocess.finish()]);

        self.window.pre_present_notify();
        surface_texture.present();
    }

    fn pause_changed(&self) {
        if !self.is_paused {
            _ = self.window.set_cursor_grab(CursorGrabMode::Locked);
            self.window.set_cursor_visible(false);
        } else {
            _ = self.window.set_cursor_grab(CursorGrabMode::None);
            self.window.set_cursor_visible(true);
        }
    }

    fn draw_debug_text(&self, render_pass: &mut wgpu::RenderPass) {
        let projection = cgmath::ortho(0., self.frame_size.x, self.frame_size.y, 0., -1.0, 1.0);
        self.debug_text.set_projection(self.queue, projection);

        let font_size = 17.0 * self.window.scale_factor() as f32;
        let model_view = Matrix4::from_scale(font_size);
        self.debug_text.set_model_view(self.queue, model_view);
        self.text_renderer.draw_text(render_pass, &self.debug_text);
    }

    fn draw_chunks(&self, render_pass: &mut wgpu::RenderPass) {
        render_pass.set_pipeline(&self.chunk_renderer.pipeline);
        render_pass.set_bind_group(0, &self.chunk_renderer.bind_group_0_wgpu, &[]);
        let far = self.world.view_distance() as f32 * 32.0;
        let projection = self.player_camera.projection_matrix(far, self.frame_size);
        let view = self.player_camera.view_matrix();
        self.chunk_renderer
            .set_view_projection(self.queue, projection * view);
        self.chunk_renderer
            .set_sun(self.queue, vec3(1., -2., 0.5).normalize());
        // self.world.chunks().with_loaded_chunk(ChunkId::new(0, 0, 0), |chunk| {
        //     for mesh in &chunk.client.meshes {
        //         let Some(mesh) = mesh else {
        //             continue;
        //         };
        //         render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        //         render_pass.set_index_buffer(
        //             mesh.index_buffer.slice(..),
        //             mesh.index_buffer.index_format(),
        //         );
        //         render_pass.set_bind_group(1, &mesh.bind_group_1_wgpu, &[]);
        //         render_pass.draw_indexed(0..mesh.index_buffer.length(), 0, 0..1);
        //     }
        // });
        let camera_chunk_id = World::world_to_local_coord_f32(self.player_camera.position).0;
        self.world
            .chunks()
            .for_each_loaded_chunk(|chunk_id, chunk| {
                let mut faces_mask = [
                    true, // south  (+z)
                    true, // north  (-z)
                    true, // east   (+x)
                    true, // west   (-x)
                    true, // top    (+y)
                    true, // bottom (-y)
                ];
                use BlockFace::*;
                if camera_chunk_id.z > chunk_id.z {
                    faces_mask[North.to_usize()] = false;
                } else if camera_chunk_id.z < chunk_id.z {
                    faces_mask[South.to_usize()] = false;
                } else if camera_chunk_id.x > chunk_id.x {
                    faces_mask[West.to_usize()] = false;
                } else if camera_chunk_id.x < chunk_id.x {
                    faces_mask[East.to_usize()] = false;
                } else if camera_chunk_id.y > chunk_id.y {
                    faces_mask[Bottom.to_usize()] = false;
                } else if camera_chunk_id.y < chunk_id.y {
                    faces_mask[Top.to_usize()] = false;
                }
                for (i_face, mesh) in chunk.client.meshes.iter().enumerate() {
                    if !faces_mask[i_face] {
                        continue;
                    }
                    let Some(mesh) = mesh else {
                        continue;
                    };
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        mesh.index_buffer.slice(..),
                        mesh.index_buffer.index_format(),
                    );
                    render_pass.set_bind_group(1, &mesh.bind_group_1_wgpu, &[]);
                    render_pass.draw_indexed(0..mesh.index_buffer.length(), 0, 0..1);
                }
            });
    }

    pub fn before_window_event(
        &mut self,
        input_helper: &InputHelper,
        duration_since_last_event: Duration,
    ) {
        if self.is_paused {
            return;
        }
        let mut movement = vec3(0., 0., 0.);
        if input_helper.key_is_down(KeyCode::KeyW) {
            movement.z += 1.0;
        }
        if input_helper.key_is_down(KeyCode::KeyS) {
            movement.z -= 1.0;
        }
        if input_helper.key_is_down(KeyCode::KeyA) {
            movement.x -= 1.0;
        }
        if input_helper.key_is_down(KeyCode::KeyD) {
            movement.x += 1.0;
        }
        if input_helper.key_is_down(KeyCode::Space) {
            movement.y += 1.0;
        }
        if input_helper.key_is_down(KeyCode::KeyR) {
            movement.y -= 1.0;
        }
        movement.normalize();
        movement *= 4.;
        if input_helper.key_is_down(KeyCode::F3) {
            movement *= 32.;
        } else if input_helper.key_is_down(KeyCode::ControlLeft) {
            movement *= 4.;
        }
        movement *= duration_since_last_event.as_secs_f32();

        let old_position = self.player_camera.position;
        self.player_camera.move_(movement);
        let new_position = self.player_camera.position;
        self.world.player_moved(old_position, new_position)
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

    pub fn view_matrix(self) -> Matrix4<f32> {
        Matrix4::look_to_rh(self.position, self.direction(), Vector3::unit_y())
    }

    pub fn projection_matrix(self, far: f32, frame_size: Vector2<f32>) -> Matrix4<f32> {
        cgmath::perspective(Deg(90.), frame_size.x / frame_size.y, 0.1, far)
    }
}
