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
    input::InputHelper,
    text::{Text, TextRenderer},
    utils::{BoolToggle, WithY as _},
    world::World,
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
    surface: wgpu::Surface<'static>,
    surface_color_format: wgpu::TextureFormat,
    depth_stencil_texture_view: wgpu::TextureView,
    fps: f64,
    is_paused: bool,
    text_renderer: TextRenderer<'cx>,
    debug_text: Text,
    debug_text_string: String,
    debug_text_needs_updating: bool,
    chunk_renderer: ChunkRenderer,
    world: World<'scope, 'cx>,
    player_camera: PlayerCamera,
}

impl<'scope, 'cx> Game<'scope, 'cx> {
    fn new(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        context: &'cx Context,
        window: Arc<Window>,
        thread_scope: &'scope thread::Scope<'scope, 'cx>,
        program_args: &ProgramArgs,
    ) -> Self {
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let surface_capabilities = surface.get_capabilities(adapter);
        let surface_color_format = surface_capabilities.formats[0];

        let text_renderer = TextRenderer::create(
            &context.device,
            &context.queue,
            &context.resources,
            surface_color_format,
            Some(Self::DEPTH_STENCIL_FORMAT),
        );
        let text = text_renderer.create_text(&context.device, "CUBE GAME v0.0.0");

        let chunk_renderer = ChunkRenderer::new(
            &context.device,
            &context.queue,
            &context.resources,
            surface_color_format,
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

        let depth_stencil_texture_view =
            Self::create_depth_stencil_texture(&context.device, frame_size_u);

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
            surface,
            surface_color_format,
            depth_stencil_texture_view,
            fps: f64::NAN,
            is_paused: false,
            text_renderer,
            debug_text: text,
            debug_text_needs_updating: true,
            debug_text_string: String::new(),
            // Would be updated later immediately in the `resized` call.
            frame_size_u,
            frame_size,
            chunk_renderer,
            world,
            player_camera,
        };
        self_.configure_surface();
        self_.pause_changed();
        self_
    }

    const DEPTH_STENCIL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    fn create_depth_stencil_texture(
        device: &wgpu::Device,
        size: Vector2<u32>,
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
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
        });
        texture.create_view(&Default::default())
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
        self.depth_stencil_texture_view =
            Self::create_depth_stencil_texture(self.device, self.frame_size_u);
    }

    fn configure_surface(&mut self) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.surface_color_format,
            view_formats: vec![self.surface_color_format.add_srgb_suffix()],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: self.frame_size_u.x,
            height: self.frame_size_u.y,
            desired_maximum_frame_latency: 3,
            present_mode: wgpu::PresentMode::AutoVsync,
        };
        self.surface.configure(self.device, &surface_config);
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
    }

    pub fn frame(&mut self) {
        let surface_texture = self
            .surface
            .get_current_texture()
            .expect("failed to acquire next swapchain texture");
        let texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.surface_color_format.add_srgb_suffix()),
                ..Default::default()
            });

        if self.debug_text_needs_updating {
            self.update_debug_text();
            self.text_renderer.update_text(
                self.device,
                &mut self.debug_text,
                &self.debug_text_string,
            );
        }

        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
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

        self.queue.submit([encoder.finish()]);
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
        let projection = self.player_camera.projection_matrix(self.frame_size);
        let view = self.player_camera.view_matrix();
        self.chunk_renderer
            .set_view_projection(self.queue, projection * view);
        self.chunk_renderer
            .set_sun(self.queue, vec3(1., -2., 0.5).normalize());
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

    pub fn projection_matrix(self, frame_size: Vector2<f32>) -> Matrix4<f32> {
        cgmath::perspective(Deg(90.), frame_size.x / frame_size.y, 0.1, 1000.0)
    }
}
