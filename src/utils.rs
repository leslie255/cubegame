use std::{
    cell::UnsafeCell,
    fmt::{self, Display},
    sync::{Arc, Mutex, RwLock},
};

use cgmath::*;

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

/// # Safety
/// Must only be called on main thread.
/// Necessary for `MainThreadOnly`'s main thread detection.
pub unsafe fn this_thread_is_main_thread_pinky_promise() {
    IS_MAIN_THREAD.with(|is_main_thread| unsafe {
        *is_main_thread.get() = true;
    });
}

/// Provided `this_thread_is_main_thread_pinky_promise` is called, this function returns whether
/// the current thread is the main thread.
pub fn is_main_thread() -> bool {
    IS_MAIN_THREAD.with(|is_main_thread| unsafe { *is_main_thread.get() })
}

#[derive(Clone, Copy)]
pub struct MainThreadOnly<T> {
    inner: T,
}

impl<T> Display for MainThreadOnly<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.inner.fmt(f)
    }
}

unsafe impl<T> Send for MainThreadOnly<T> {}
unsafe impl<T> Sync for MainThreadOnly<T> {}

impl<T> MainThreadOnly<T> {
    pub fn new(value: T) -> Self {
        Self { inner: value }
    }

    /// # Safety
    /// Must only be called in the main thread.
    pub unsafe fn get_unchecked(&self) -> &T {
        &self.inner
    }

    /// # Safety
    /// Must only be called in the main thread.
    pub unsafe fn get_unchecked_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    pub fn try_get(&self) -> Option<&T> {
        is_main_thread().then_some(unsafe { self.get_unchecked() })
    }

    pub fn try_get_mut(&mut self) -> Option<&mut T> {
        is_main_thread().then_some(unsafe { self.get_unchecked_mut() })
    }

    /// # Panics
    /// Panic if current thread is not the main thread.
    #[track_caller]
    pub fn get(&self) -> &T {
        self.try_get()
            .expect("Called `MainThreadOnly` on non-main thread")
    }

    /// # Panics
    /// Panic if current thread is not the main thread.
    #[track_caller]
    pub fn get_mut(&mut self) -> &mut T {
        self.try_get_mut()
            .expect("Called `MainThreadOnly` on non-main thread")
    }

    pub fn as_ref(&self) -> MainThreadOnly<&T> {
        MainThreadOnly::new(unsafe { self.get_unchecked() })
    }

    pub fn as_mut(&mut self) -> MainThreadOnly<&mut T> {
        MainThreadOnly::new(unsafe { self.get_unchecked_mut() })
    }
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
