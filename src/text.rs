use std::path::Path;
use std::rc::Rc;

use std::ops::Range;

use glium::{Program, Surface as _};
use image::DynamicImage;
use serde::{Deserialize, Serialize};

use cgmath::*;

use crate::mesh::{self, Mesh};
use crate::resource::ResourceLoader;

pub fn default_font(
    resource_loader: &ResourceLoader,
    display: &impl glium::backend::Facade,
) -> Font {
    let mut font = Font::load_from_path(resource_loader, "font/big_blue_terminal.json");
    let image = glium::texture::RawImage2d::from_raw_rgba(
        font.atlas().to_rgba8().into_raw(),
        (font.atlas().width(), font.atlas().height()),
    );
    font.gl_texture = Some(glium::Texture2d::new(display, image).unwrap());
    font
}

pub fn text_shader(display: &impl glium::backend::Facade) -> glium::Program {
    const VERTEX_SHADER: &str = r#"
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

    const FRAGMENT_SHADER: &str = r#"
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

    glium::program::Program::from_source(display, VERTEX_SHADER, FRAGMENT_SHADER, None).unwrap()
}

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
    ) -> Self {
        let json_subpath = json_subpath.as_ref();
        let font_meta = resource_loader.load_json_object::<FontMetaJson>(json_subpath);
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
pub struct Line {
    mesh: Mesh<CharacterVertex>,
    string: String,
    font: Rc<Font>,
    shader: Rc<Program>,
}

impl Line {
    pub fn new(font: Rc<Font>, shader: Rc<Program>) -> Self {
        Self {
            mesh: Mesh::new().with_draw_parameters(mesh::default_2d_draw_parameters()),
            string: String::new(),
            font,
            shader,
        }
    }

    pub fn with_string(
        font: Rc<Font>,
        shader: Rc<Program>,
        display: &impl glium::backend::Facade,
        string: String,
    ) -> Self {
        let mut self_ = Self::new(font, shader);
        for char in string.chars() {
            self_.push_char(char);
        }
        self_.update(display);
        self_
    }

    pub fn push_char(&mut self, char: char) {
        let uv_quad = if self.font.has_glyph(char) {
            self.font.sample(char)
        } else {
            return;
        };
        // Width is just the aspect ratio because height is 1.
        let glyph_width = self.font.glyph_aspect_ratio();
        let quad = Quad2 {
            left: self.string.len() as f32 * glyph_width,
            right: (self.string.len() + 1) as f32 * glyph_width,
            bottom: 1.,
            top: 0.,
        };

        self.string.push(char);

        #[rustfmt::skip]
        let vertices = [
            CharacterVertex { position: [quad.left,  quad.top],    uv: [uv_quad.left,  uv_quad.top]    },
            CharacterVertex { position: [quad.left,  quad.bottom], uv: [uv_quad.left,  uv_quad.bottom] },
            CharacterVertex { position: [quad.right, quad.top],    uv: [uv_quad.right, uv_quad.top]    },
            CharacterVertex { position: [quad.right, quad.bottom], uv: [uv_quad.right, uv_quad.bottom] },
        ];
        #[rustfmt::skip]
        let indices = [
            2, 1, 0,
            1, 2, 3,
        ];

        self.mesh.append(&vertices, &indices);
    }

    /// Send the new mesh to the CPU.
    pub fn update(&mut self, display: &impl glium::backend::Facade) {
        self.mesh.update(display);
    }

    pub fn draw(
        &self,
        frame: &mut glium::Frame,
        position: Vector2<f32>,
        fg_color: Vector4<f32>,
        bg_color: Vector4<f32>,
        font_size: f32,
    ) {
        let (frame_width, frame_height) = frame.get_dimensions();
        let model = Matrix4::from_translation(Vector3::new(position.x, position.y, 0.));
        let model = model * Matrix4::from_nonuniform_scale(font_size, font_size, 1.);
        let view: Matrix4<f32> = Matrix4::identity();
        let projection = cgmath::ortho(0., frame_width as f32, frame_height as f32, 0., -1., 1.);
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
            uv_matrix: matrix3_to_array::<f32>(Matrix3::identity()),
            tex: texture_sampler,
            fg_color: Into::<[f32; 4]>::into(fg_color),
            bg_color: Into::<[f32; 4]>::into(bg_color),
        };
        self.mesh.draw(frame, uniforms, &self.shader);
    }
}
