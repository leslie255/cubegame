pub trait BoolToggle {
    fn toggle(&mut self);
}

impl BoolToggle for bool {
    fn toggle(&mut self) {
        *self = !*self;
    }
}

