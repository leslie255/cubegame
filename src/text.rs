use std::{borrow::Cow, path::Path};

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use image::RgbaImage;
use serde::{Deserialize, Serialize};

use cgmath::*;
use wgpu::util::DeviceExt;

use crate::wgpu_utils::{IndexBuffer, Vertex, Vertex2dUV, VertexBuffer};
use crate::{
    impl_as_bind_group,
    resource::ResourceLoader,
    wgpu_utils::{self, UniformBuffer},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quad2d {
    pub left: f32,
    pub right: f32,
    pub bottom: f32,
    pub top: f32,
}

impl Quad2d {
    pub fn width(self) -> f32 {
        (self.right - self.left).abs()
    }

    pub fn height(self) -> f32 {
        (self.top - self.bottom).abs()
    }
}

fn normalize_coord_in_texture(texture_size: Vector2<u32>, coord: Vector2<u32>) -> Vector2<f32> {
    let texture_size_f = texture_size.map(|x| x as f32);
    let coord_f = coord.map(|x| x as f32);
    coord_f.div_element_wise(texture_size_f)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontMetaJson {
    pub path: String,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub glyph_width: u32,
    pub glyph_height: u32,
    pub present_start: u8,
    pub present_end: u8,
    pub glyphs_per_line: u32,
}

#[derive(Debug, Clone)]
pub struct Font {
    present_range: Range<u8>,
    glyphs_per_line: u32,
    glyph_size: Vector2<u32>,
    atlas_image: image::RgbaImage,
}

impl Font {
    pub fn load_from_path(
        resource_loader: &ResourceLoader,
        json_subpath: impl AsRef<Path>,
    ) -> Self {
        let json_subpath = json_subpath.as_ref();
        let font_meta = resource_loader.load_json_object::<FontMetaJson>(json_subpath);
        let atlas_image_subpath =
            resource_loader.solve_relative_subpath(json_subpath, &font_meta.path);
        let atlas_image = resource_loader.load_image(&atlas_image_subpath).to_rgba8();
        Self {
            atlas_image,
            present_range: font_meta.present_start..font_meta.present_end,
            glyphs_per_line: font_meta.glyphs_per_line,
            glyph_size: vec2(font_meta.glyph_width, font_meta.glyph_height),
        }
    }

    pub fn atlas_image(&self) -> &RgbaImage {
        &self.atlas_image
    }

    pub fn has_glyph(&self, char: char) -> bool {
        self.present_range.contains(&(char as u8))
    }

    fn position_for_glyph(&self, char: char) -> Vector2<u32> {
        assert!(self.has_glyph(char));
        let ith_glyph = ((char as u8) - self.present_range.start) as u32;
        let glyph_coord = vec2(
            ith_glyph % self.glyphs_per_line,
            ith_glyph / self.glyphs_per_line,
        );
        glyph_coord.mul_element_wise(self.glyph_size)
    }

    pub fn quad(&self, char: char) -> Quad2d {
        let top_left = self.position_for_glyph(char);
        let bottom_right = top_left.add_element_wise(self.glyph_size);
        let atlas_size = vec2(self.atlas_image.width(), self.atlas_image.height());
        let top_left = normalize_coord_in_texture(atlas_size, top_left);
        let bottom_right = normalize_coord_in_texture(atlas_size, bottom_right);
        Quad2d {
            left: top_left.x,
            right: bottom_right.x,
            bottom: bottom_right.y,
            top: top_left.y,
        }
    }

    pub fn glyph_aspect_ratio(&self) -> f32 {
        (self.glyph_size.x as f32) / (self.glyph_size.y as f32)
    }

    pub fn glyph_size(&self) -> Vector2<u32> {
        self.glyph_size
    }
}

#[derive(Debug, Clone)]
pub struct TextBindGroup {
    model_view: UniformBuffer<[[f32; 4]; 4]>,
    projection: UniformBuffer<[[f32; 4]; 4]>,
    fg_color: UniformBuffer<[f32; 4]>,
    bg_color: UniformBuffer<[f32; 4]>,
    texture_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

impl_as_bind_group! {
    TextBindGroup {
        0 => model_view: UniformBuffer<[[f32; 4]; 4]>,
        1 => projection: UniformBuffer<[[f32; 4]; 4]>,
        2 => fg_color: UniformBuffer<[f32; 4]>,
        3 => bg_color: UniformBuffer<[f32; 4]>,
        4 => texture_view: wgpu::TextureView,
        5 => sampler: wgpu::Sampler,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct TextInstance {
    pub position_offset: [f32; 2],
    pub uv_offset: [f32; 2],
}

impl TextInstance {
    pub fn new(position_offset: [f32; 2], uv_offset: [f32; 2]) -> Self {
        Self {
            position_offset,
            uv_offset,
        }
    }

    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: size_of::<Self>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: size_of::<[f32; 2]>() as u64,
                shader_location: 3,
            },
        ],
    };
}

#[derive(Debug, Clone)]
pub struct Text {
    bind_group: TextBindGroup,
    wgpu_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    n_instances: u32,
}

impl Text {
    pub fn set_fg_color(&self, queue: &wgpu::Queue, color: Vector4<f32>) {
        self.bind_group.fg_color.write(color.into(), queue);
    }

    pub fn set_model_view(&self, queue: &wgpu::Queue, model_view: Matrix4<f32>) {
        self.bind_group.model_view.write(model_view.into(), queue);
    }

    pub fn set_projection(&self, queue: &wgpu::Queue, projection: Matrix4<f32>) {
        self.bind_group.projection.write(projection.into(), queue);
    }
}

#[derive(Debug, Clone)]
pub struct TextRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub texture_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub font: Font,
    pub vertex_buffer: VertexBuffer<Vertex2dUV>,
    pub index_buffer: IndexBuffer<u16>,
}

impl TextRenderer {
    pub fn create(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        font: Font,
        resource_loader: &ResourceLoader,
        surface_color_format: wgpu::TextureFormat,
    ) -> Self {
        let shader_source = resource_loader.read_to_string("shader/text.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader_source)),
        });
        let bind_group_layout = wgpu_utils::create_bind_group_layout::<TextBindGroup>(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Vertex2dUV::LAYOUT, TextInstance::LAYOUT],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_color_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            operation: wgpu::BlendOperation::Add,
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let atlas_image = font.atlas_image();
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: atlas_image.width(),
                    height: atlas_image.height(),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
            },
            wgpu::wgt::TextureDataOrder::LayerMajor,
            atlas_image,
        );
        let texture_view = texture.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        // Vertex buffer.
        let (atlas_width, atlas_height) = font.atlas_image().dimensions();
        let glyph_size_pixels = font.glyph_size();
        let glyph_width = glyph_size_pixels.x as f32 / atlas_width as f32;
        let glyph_height = glyph_size_pixels.y as f32 / atlas_height as f32;
        let vertices_data = &[
            Vertex2dUV::new([0., 0.], [0., 0.]),
            Vertex2dUV::new([font.glyph_aspect_ratio(), 0.], [glyph_width, 0.]),
            Vertex2dUV::new([font.glyph_aspect_ratio(), 1.], [glyph_width, glyph_height]),
            Vertex2dUV::new([0., 1.], [0., glyph_height]),
        ];
        let vertex_buffer = VertexBuffer::create_init(device, vertices_data);

        // Index buffer.
        let indices_data = &[0u16, 1, 2, 2, 3, 0];
        let index_buffer = IndexBuffer::create_init(device, indices_data);

        Self {
            bind_group_layout,
            pipeline,
            texture_view,
            sampler,
            font,
            vertex_buffer,
            index_buffer,
        }
    }

    pub fn draw_text(&self, render_pass: &mut wgpu::RenderPass, text: &Text) {
        if text.n_instances == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &text.wgpu_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, text.instance_buffer.slice(..));
        render_pass.set_index_buffer(
            self.index_buffer.slice(..),
            self.index_buffer.index_format(),
        );
        render_pass.draw_indexed(0..self.index_buffer.length(), 0, 0..text.n_instances);
    }

    pub fn create_text(&self, device: &wgpu::Device, str: &str) -> Text {
        let bind_group = TextBindGroup {
            model_view: UniformBuffer::create_init(device, Matrix4::identity().into()),
            projection: UniformBuffer::create_init(device, Matrix4::identity().into()),
            fg_color: UniformBuffer::create_init(device, [1.; 4]),
            bg_color: UniformBuffer::create_init(device, [0.; 4]),
            texture_view: self.texture_view.clone(),
            sampler: self.sampler.clone(),
        };
        let wgpu_bind_group =
            wgpu_utils::create_bind_group(device, &self.bind_group_layout, &bind_group);
        let (n_instances, instance_buffer) = self.create_instance_buffer(device, str);
        Text {
            bind_group,
            wgpu_bind_group,
            instance_buffer,
            n_instances,
        }
    }

    pub fn update_text(&self, device: &wgpu::Device, text: &mut Text, str: &str) {
        (text.n_instances, text.instance_buffer) = self.create_instance_buffer(device, str);
    }

    fn create_instance_buffer(&self, device: &wgpu::Device, str: &str) -> (u32, wgpu::Buffer) {
        let mut instances: Vec<TextInstance> = Vec::new();
        let mut row = 0u32;
        let mut column = 0u32;
        for char in str.chars() {
            if char == '\n' {
                column = 0;
                row += 1;
                continue;
            } else if char == '\r' {
                column = 0;
                continue;
            } else if !self.font.has_glyph(char) {
                continue;
            }
            let quad = self.font.quad(char);
            instances.push(TextInstance {
                position_offset: [column as f32 * self.font.glyph_aspect_ratio(), row as f32],
                uv_offset: [quad.left, quad.top],
            });
            column += 1;
        }
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });
        (instances.len() as u32, instance_buffer)
    }
}
