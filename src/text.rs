use std::path::Path;

use std::ops::Range;

use glium::Surface as _;
use image::DynamicImage;
use serde::{Deserialize, Serialize};

use cgmath::*;

use crate::resource::ResourceLoader;

pub fn matrix4_to_array<T>(matrix: Matrix4<T>) -> [[T; 4]; 4] {
    matrix.into()
}

pub fn matrix3_to_array<T>(matrix: Matrix3<T>) -> [[T; 3]; 3] {
    matrix.into()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quad2 {
    pub left: f32,
    pub right: f32,
    pub bottom: f32,
    pub top: f32,
}

impl Quad2 {
    pub fn width(self) -> f32 {
        (self.right - self.left).abs()
    }

    pub fn height(self) -> f32 {
        (self.top - self.bottom).abs()
    }
}

fn uvec2_to_fvec2(uvec: Vector2<u32>) -> Vector2<f32> {
    uvec.map(|x| x as f32)
}

fn normalize_coord_in_texture(texture_size: Vector2<u32>, coord: Vector2<u32>) -> Vector2<f32> {
    uvec2_to_fvec2(coord).div_element_wise(uvec2_to_fvec2(texture_size))
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

pub struct Font {
    json_path: Option<String>,
    atlas_path: Option<String>,
    present_range: Range<u8>,
    glyphs_per_line: u32,
    glyph_size: Vector2<u32>,
    atlas: DynamicImage,
    pub gl_texture: Option<glium::Texture2d>,
}

impl std::fmt::Debug for Font {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Font")
            .field("json_path", &self.json_path)
            .field("atlas_path", &self.atlas_path)
            .field("present_range", &self.present_range)
            .field("glyphs_per_line", &self.glyphs_per_line)
            .field("glyph_size", &self.glyph_size)
            .finish_non_exhaustive()
    }
}

impl Font {
    pub fn load_from_path(
        resource_loader: &ResourceLoader,
        json_subpath: impl AsRef<Path>,
        log: bool,
    ) -> Self {
        let json_subpath = json_subpath.as_ref();
        let font_meta = resource_loader.load_json_object::<FontMetaJson>(json_subpath);
        if log {
            println!("[INFO] loaded resource {json_subpath:?}");
        }
        let atlas_subpath = resource_loader.solve_relative_subpath(json_subpath, &font_meta.path);
        let atlas = resource_loader.load_image(&atlas_subpath);
        Self {
            json_path: Some(json_subpath.as_os_str().to_string_lossy().into_owned()),
            atlas_path: Some(atlas_subpath.as_os_str().to_string_lossy().into_owned()),
            atlas,
            present_range: font_meta.present_start..font_meta.present_end,
            glyphs_per_line: font_meta.glyphs_per_line,
            glyph_size: vec2(font_meta.glyph_width, font_meta.glyph_height),
            gl_texture: None,
        }
    }

    pub fn path(&self) -> Option<&str> {
        self.atlas_path.as_deref()
    }

    pub fn atlas(&self) -> &DynamicImage {
        &self.atlas
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

    pub fn sample(&self, char: char) -> Quad2 {
        let top_left = self.position_for_glyph(char);
        let bottom_right = top_left.add_element_wise(self.glyph_size);
        let atlas_size = vec2(self.atlas.width(), self.atlas.height());
        let top_left = normalize_coord_in_texture(atlas_size, top_left);
        let bottom_right = normalize_coord_in_texture(atlas_size, bottom_right);
        Quad2 {
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharacterVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

glium::implement_vertex!(CharacterVertex, position, uv);

#[derive(Debug)]
pub struct TextPainter {
    vertices: glium::VertexBuffer<CharacterVertex>,
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
    const VERTICES: [CharacterVertex; 4] = [
        CharacterVertex { position: [0., 0.], uv: [0., 0.] },
        CharacterVertex { position: [1., 0.], uv: [1., 0.] },
        CharacterVertex { position: [0., 1.], uv: [0., 1.] },
        CharacterVertex { position: [1., 1.], uv: [1., 1.] },
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

    /// Returns the width of the character.
    pub fn draw_char(
        &self,
        frame: &mut glium::Frame,
        position: Vector2<f32>,
        size: f32,
        char: char,
    ) -> f32 {
        let char_width = size * self.font.glyph_aspect_ratio();
        let (frame_width, frame_height) = frame.get_dimensions();
        let view: Matrix4<f32> = Matrix4::identity();
        let projection = cgmath::ortho(0., frame_width as f32, frame_height as f32, 0., -1., 1.);
        let model = Matrix4::from_translation(Vector3::new(position.x, position.y, 0.));
        let model = model * Matrix4::from_nonuniform_scale(char_width, size, 1.);
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
        char_width
    }
}
