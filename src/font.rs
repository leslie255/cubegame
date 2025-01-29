use std::path::Path;

use std::ops::Range;

use image::DynamicImage;
use serde::{Deserialize, Serialize};

use cgmath::*;

use crate::resource::ResourceLoader;

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
}
