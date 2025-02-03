pub trait BoolToggle {
    fn toggle(&mut self);
}

impl BoolToggle for bool {
    fn toggle(&mut self) {
        *self = !*self;
    }
}

pub fn vec_with<T>(n: usize, mut f: impl FnMut() -> T) -> Vec<T> {
    let mut vec = Vec::with_capacity(n);
    for _ in 0..n {
        vec.push(f());
    }
    vec
}
