use std::{fmt::Write as _, sync::Arc};

use cgmath::*;

use pollster::FutureExt as _;
use winit::{
    event::{ElementState, KeyEvent, MouseButton},
    window::Window,
};

use crate::{
    input::InputHelper,
    resource::ResourceLoader,
    text::{Font, Text, TextRenderer},
};

mod app;
mod fps_counter;

pub use app::*;

#[derive(Debug)]
pub struct Game {
    device: wgpu::Device,
    queue: wgpu::Queue,
    window: Arc<Window>,
    frame_size_u: Vector2<u32>,
    frame_size: Vector2<f32>,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    fps: f64,
    text_renderer: TextRenderer,
    debug_text: Text,
    debug_text_string: String,
    debug_text_needs_updating: bool,
}

impl Game {
    pub fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .block_on()
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .block_on()
            .unwrap();

        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let surface_capabilities = surface.get_capabilities(&adapter);
        let surface_format = surface_capabilities.formats[0];

        let resource_loader = ResourceLoader::with_default_res_directory().unwrap();
        let font = Font::load_from_path(&resource_loader, "font/big_blue_terminal.json");
        let text_renderer =
            TextRenderer::create(&device, &queue, font, &resource_loader, surface_format);
        let text = text_renderer.create_text(&device, "CUBE GAME v0.0.0");

        let mut self_ = Self {
            device,
            queue,
            window,
            surface,
            surface_format,
            fps: f64::NAN,
            text_renderer,
            debug_text: text,
            debug_text_needs_updating: true,
            debug_text_string: String::new(),
            // Would be updated later immediately in the `resized` call.
            frame_size_u: vec2(0, 0),
            frame_size: vec2(0., 0.),
        };
        self_.resized();
        self_
    }

    #[expect(unused_variables)]
    pub fn keyboard_input(&mut self, event: KeyEvent, input_helper: &InputHelper) {}

    #[expect(unused_variables)]
    pub fn mouse_input(
        &mut self,
        state: ElementState,
        button: MouseButton,
        intput_helper: &InputHelper,
    ) {
    }

    pub fn resized(&mut self) {
        let physical_size = self.window.inner_size();
        self.frame_size_u = vec2(physical_size.width, physical_size.height);
        self.frame_size = self.frame_size_u.map(|u| u as f32);
        self.configure_surface();
    }

    fn configure_surface(&self) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.surface_format,
            view_formats: vec![self.surface_format.add_srgb_suffix()],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: self.frame_size_u.x,
            height: self.frame_size_u.y,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::AutoVsync,
        };
        self.surface.configure(&self.device, &surface_config);
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
                format: Some(self.surface_format.add_srgb_suffix()),
                ..Default::default()
            });

        if self.debug_text_needs_updating {
            self.update_debug_text();
            self.text_renderer.update_text(
                &self.device,
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
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        self.draw_debug_text(&mut render_pass);

        drop(render_pass);

        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        surface_texture.present();
    }

    fn draw_debug_text(&self, render_pass: &mut wgpu::RenderPass) {
        let projection = cgmath::ortho(0., self.frame_size.x, self.frame_size.y, 0., -1.0, 1.0);
        self.debug_text.set_projection(&self.queue, projection);

        let font_size = 17.0 * self.window.scale_factor() as f32;
        let model_view = Matrix4::from_scale(font_size);
        self.debug_text.set_model_view(&self.queue, model_view);
        self.text_renderer.draw_text(render_pass, &self.debug_text);
    }
}
