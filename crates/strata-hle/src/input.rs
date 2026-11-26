//! Minimal Xbox Gamepad / Keyboard-to-Pad input state for `GameInput` / XInput HLE.

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct GamepadState {
    pub buttons: u16,
    pub left_trigger: f32,
    pub right_trigger: f32,
    pub left_stick_x: f32,
    pub left_stick_y: f32,
    pub right_stick_x: f32,
    pub right_stick_y: f32,
}

pub const BTN_DPAD_UP: u16 = 0x0001;
pub const BTN_DPAD_DOWN: u16 = 0x0002;
pub const BTN_DPAD_LEFT: u16 = 0x0004;
pub const BTN_DPAD_RIGHT: u16 = 0x0008;
pub const BTN_MENU: u16 = 0x0010;
pub const BTN_VIEW: u16 = 0x0020;
pub const BTN_A: u16 = 0x1000;
pub const BTN_B: u16 = 0x2000;
pub const BTN_X: u16 = 0x4000;
pub const BTN_Y: u16 = 0x8000;

/// Map WASD + Enter/Esc keyboard virtual-key codes into a standard Xbox pad state.
pub fn keyboard_to_gamepad(pressed_vks: &[u16]) -> GamepadState {
    let mut s = GamepadState::default();
    for &vk in pressed_vks {
        match vk {
            0x57 => s.left_stick_y = 1.0,  // W
            0x53 => s.left_stick_y = -1.0, // S
            0x41 => s.left_stick_x = -1.0, // A
            0x44 => s.left_stick_x = 1.0,  // D
            0x0D => s.buttons |= BTN_A,    // Enter
            0x08 => s.buttons |= BTN_B,    // Backspace
            0x1B => s.buttons |= BTN_MENU, // Esc
            _ => {}
        }
    }
    s
}
