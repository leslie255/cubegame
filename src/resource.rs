use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use image::DynamicImage;
use serde::de::DeserializeOwned;

#[derive(Debug)]
pub struct ResourceLoader {
    res_directory: PathBuf,
}

impl ResourceLoader {
    /// Find the default resource directory.
    pub fn default_res_directory() -> Option<PathBuf> {
        let path: PathBuf = match std::env::var("CARGO_MANIFEST_DIR") {
            Ok(path) => {
                let mut path = PathBuf::from(path);
                path.push("res/");
                path
            }
            Err(_) => "res/".into(),
        };
        assert!(path.is_dir());
        Some(path)
    }

    /// Returns `None` if the resource directory does not exist.
    pub fn with_default_res_directory() -> Option<Self> {
        Some(Self::with_res_directory_unchecked(
            Self::default_res_directory()?,
        ))
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
        image::open(&path)
            .unwrap_or_else(|_| self.handle_malformed_image_encoding(subpath))
    }
}
