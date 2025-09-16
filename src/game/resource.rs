use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use image::{DynamicImage, RgbaImage};
use serde::de::DeserializeOwned;

use crate::{
    block::{BlockRegistry, GameBlocks},
    text::Font,
};

#[derive(Debug)]
pub struct ResourceLoader {
    res_directory: PathBuf,
}

impl ResourceLoader {
    /// Find the default resource directory.
    pub fn default_res_directory() -> PathBuf {
        let path: PathBuf = match std::env::var("CARGO_MANIFEST_DIR") {
            Ok(path) => {
                let mut path = PathBuf::from(path);
                path.push("res/");
                path
            }
            Err(_) => "res/".into(),
        };
        path
    }

    /// Returns `None` if the resource directory does not exist.
    pub fn with_default_res_directory() -> Option<Self> {
        Self::with_res_directory(Self::default_res_directory())
    }

    /// Returns `None` if path directory does not exist.
    pub fn with_res_directory(res_directory: PathBuf) -> Option<Self> {
        if res_directory.is_dir() {
            Some(Self::with_res_directory_unchecked(res_directory))
        } else {
            None
        }
    }

    fn with_res_directory_unchecked(res_directory: PathBuf) -> Self {
        println!("[INFO] using resource directory {res_directory:?}");
        Self { res_directory }
    }

    pub fn res_directory(&self) -> &Path {
        self.res_directory.as_ref()
    }

    pub fn path_for(&self, subpath: impl AsRef<Path>) -> PathBuf {
        let mut path = self.res_directory.clone();
        path.push(subpath);
        path
    }

    /// Returns a new subpath.
    pub fn solve_relative_subpath(
        &self,
        origin_subpath: impl AsRef<Path>,
        relative_path: impl AsRef<Path>,
    ) -> PathBuf {
        let origin_subpath = PathBuf::from(origin_subpath.as_ref());
        match origin_subpath.parent() {
            Some(path) => path.join(relative_path),
            None => relative_path.as_ref().into(),
        }
    }

    #[cfg_attr(debug_assertions, track_caller)]
    fn handle_resource_not_exist(&self, subpath: impl AsRef<Path>) -> ! {
        let subpath = subpath.as_ref();
        panic!("Required resource does not exist: {subpath:?}")
    }

    #[cfg_attr(debug_assertions, track_caller)]
    fn handle_malformed_json(&self, subpath: impl AsRef<Path>) -> ! {
        let subpath = subpath.as_ref();
        panic!("JSON file is has malformed format: {subpath:?}")
    }

    #[cfg_attr(debug_assertions, track_caller)]
    fn handle_malformed_image_encoding(&self, subpath: impl AsRef<Path>) -> ! {
        let subpath = subpath.as_ref();
        panic!("Image file is has malformed encoding: {subpath:?}")
    }

    pub fn read_to_string(&self, subpath: impl AsRef<Path>) -> String {
        let path = self.path_for(subpath);
        println!("[INFO] loading {path:?}");
        std::fs::read_to_string(path).unwrap()
    }

    pub fn open_file(&self, subpath: impl AsRef<Path>) -> File {
        let path = self.path_for(&subpath);
        println!("[INFO] loading {path:?}");
        File::open(&path).unwrap_or_else(|_| self.handle_resource_not_exist(&path))
    }

    pub fn load_json_object<T: DeserializeOwned>(&self, subpath: impl AsRef<Path>) -> T {
        let file = self.open_file(&subpath);
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).unwrap_or_else(|_| self.handle_malformed_json(subpath))
    }

    pub fn load_image(&self, subpath: impl AsRef<Path>) -> DynamicImage {
        let subpath = subpath.as_ref();
        let path = self.path_for(subpath);
        if !path.is_file() {
            self.handle_resource_not_exist(subpath);
        }
        println!("[INFO] loading {path:?}");
        image::open(&path).unwrap_or_else(|_| self.handle_malformed_image_encoding(subpath))
    }

    pub fn load_shader(
        &self,
        device: &wgpu::Device,
        subpath: impl AsRef<Path>,
    ) -> wgpu::ShaderModule {
        let source = self.read_to_string(subpath);
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    }

    pub fn load_font(&self, subpath: impl AsRef<Path>) -> Font {
        Font::load_from_path(self, subpath)
    }
}

#[derive(Debug)]
pub struct GameResources {
    pub shader_text: wgpu::ShaderModule,
    pub shader_chunk: wgpu::ShaderModule,
    pub font: Font,
    pub block_atlas_image: RgbaImage,
    pub block_registry: BlockRegistry,
    pub game_blocks: GameBlocks,
    pub loader: ResourceLoader,
}

impl GameResources {
    /// Uses the default resource directory if `res_directory` is `None`.
    pub fn load(device: &wgpu::Device, res_directory: Option<PathBuf>) -> Self {
        let res_directory = res_directory.unwrap_or_else(ResourceLoader::default_res_directory);
        let loader = ResourceLoader::with_res_directory(res_directory.clone())
            .unwrap_or_else(|| panic!("Cannot find resource directory {res_directory:?}"));
        let mut block_registry = BlockRegistry::default();
        Self {
            shader_text: loader.load_shader(device, "shader/text.wgsl"),
            shader_chunk: loader.load_shader(device, "shader/chunk.wgsl"),
            font: loader.load_font("font/big_blue_terminal.json"),
            block_atlas_image: loader.load_image("texture/block_atlas.png").to_rgba8(),
            game_blocks: GameBlocks::new(&mut block_registry),
            block_registry,
            loader,
        }
    }
}
