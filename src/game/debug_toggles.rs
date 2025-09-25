use std::fmt;

use winit::keyboard::{KeyCode, PhysicalKey};

use crate::utils::BoolToggle as _;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DebugToggles {
    pub wireframe_mode: bool,
    pub show_debug_overlay: bool,
}

impl DebugToggles {
    pub fn f3_pressed(&mut self) {}

    pub fn key_pressed_with_f3(&mut self, key: PhysicalKey) {
        #[allow(clippy::single_match)]
        match key {
            PhysicalKey::Code(KeyCode::KeyL) => self.wireframe_mode.toggle(),
            PhysicalKey::Code(KeyCode::KeyT) => self.show_debug_overlay.toggle(),
            _ => (),
        }
    }

    pub fn nothing_is_on(&self) -> bool {
        self == &Self::default()
    }

    /// Prompt currently activated debug toggles.
    pub fn prompt_text(&self, out: &mut impl fmt::Write) -> fmt::Result {
        if self.show_debug_overlay {
            writeln!(out, "[F3+T] Debug Text Overlay")?;
        }
        if self.wireframe_mode {
            writeln!(out, "[F3+L] Wireframe Mode")?;
        }
        Ok(())
    }
}
