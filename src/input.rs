use glium::winit::{event::KeyEvent, keyboard::{KeyCode, PhysicalKey}};

/// Keeps track of which keys are currently down.
#[derive(Debug, Clone)]
pub struct InputHelper {
    downed_keys: Vec<bool>,
}
impl InputHelper {
    pub fn new() -> Self {
        Self {
            downed_keys: vec![false; 256],
        }
    }

    fn index_for_key(key_code: KeyCode) -> usize {
        // This is technically unsafe lol due to KeyCode not being a stable API, but like nah.
        (key_code as u8).into()
    }

    pub fn key_is_down(&self, key_code: KeyCode) -> bool {
        self.downed_keys[Self::index_for_key(key_code)]
    }

    pub fn update_key_event(&mut self, key_event: &KeyEvent) {
        if key_event.repeat {
            return;
        }
        let key_code = match key_event.physical_key {
            PhysicalKey::Code(key_code) => key_code,
            PhysicalKey::Unidentified(_) => return,
        };
        let index = Self::index_for_key(key_code);
        self.downed_keys[index] = key_event.state.is_pressed();
    }
}

impl Default for InputHelper {
    fn default() -> Self {
        Self::new()
    }
}


