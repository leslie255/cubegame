use std::fmt;

use winit::keyboard::{KeyCode, PhysicalKey};

use crate::utils::BoolToggle as _;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DebugToggles {
    pub wireframe_mode: bool,
    pub show_debug_overlay: bool,
    pub gray_world: bool,
    pub fog_disabled: bool,
}

impl DebugToggles {
    pub fn f3_pressed(&mut self) {}

    pub fn key_pressed_with_f3(&mut self, key: PhysicalKey) {
        #[allow(clippy::single_match)]
        match key {
            PhysicalKey::Code(KeyCode::KeyL) => self.wireframe_mode.toggle(),
            PhysicalKey::Code(KeyCode::KeyT) => self.show_debug_overlay.toggle(),
            PhysicalKey::Code(KeyCode::KeyU) => self.gray_world.toggle(),
            PhysicalKey::Code(KeyCode::KeyF) => self.fog_disabled.toggle(),
            _ => (),
        }
    }

    pub fn nothing_is_on(&self) -> bool {
        self == &Self::default()
    }

    /// Prompt currently activated debug toggles.
    pub fn prompt_text(&self, out: &mut impl fmt::Write) -> fmt::Result {
        for (key, enabled, description) in self.keys() {
            if enabled {
                let key = key.to_ascii_uppercase();
                writeln!(out, "[F3+{key}] {description}")?;
            }
        }
        Ok(())
    }

    pub fn keys(&self) -> impl Iterator<Item = (char, bool, &str)> {
        [
            ('t', self.show_debug_overlay, "Debug Text Overlay"),
            ('l', self.wireframe_mode, "Wireframe Mode"),
            ('u', self.gray_world, "Gray World"),
            ('f', self.fog_disabled, "Disable Fog"),
        ]
        .into_iter()
    }
}
