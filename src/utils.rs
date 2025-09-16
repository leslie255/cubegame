use std::{
    cell::UnsafeCell,
    sync::{Arc, Mutex, RwLock},
};

use cgmath::*;

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

pub trait BoolToggle {
    fn toggle(&mut self);
}

impl BoolToggle for bool {
    fn toggle(&mut self) {
        *self = !*self;
    }
}

pub fn vec_with<T>(n: usize, f: impl FnMut() -> T) -> Vec<T> {
    std::iter::repeat_with(f).take(n).collect()
}

pub fn arc_mutex<T>(value: T) -> Arc<Mutex<T>> {
    Arc::new(Mutex::new(value))
}

pub fn arc_rwlock<T>(value: T) -> Arc<RwLock<T>> {
    Arc::new(RwLock::new(value))
}

pub fn array_ref<const N: usize, T>(array: &[T; N]) -> [&T; N] {
    std::array::from_fn(|i| &array[i])
}

thread_local! {
    static IS_MAIN_THREAD: UnsafeCell<bool> = const { UnsafeCell::new(false) };
}

pub trait WithX<S> {
    fn with_x(self, new_x: S) -> Self;
}

macro_rules! impl_with_x {
    ($t:tt $(,)?) => {
        impl<S> WithX<S> for $t<S> {
            #[allow(clippy::needless_update)]
            fn with_x(self, new_x: S) -> Self {
                Self { x: new_x, ..self }
            }
        }
    };
}

impl_with_x!(Point1);
impl_with_x!(Point2);
impl_with_x!(Point3);

pub trait WithY<S> {
    fn with_y(self, new_y: S) -> Self;
}

macro_rules! impl_with_y {
    ($t:tt $(,)?) => {
        impl<S> WithY<S> for $t<S> {
            fn with_y(self, new_y: S) -> Self {
                Self { y: new_y, ..self }
            }
        }
    };
}

impl_with_y!(Point2);
impl_with_y!(Point3);

pub trait WithZ<S> {
    fn with_z(self, new_z: S) -> Self;
}

impl<S> WithZ<S> for Point3<S> {
    fn with_z(self, new_z: S) -> Self {
        Self { z: new_z, ..self }
    }
}
