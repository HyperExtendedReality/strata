//! Real GameInput HLE (`IGameInput`, `IGameInputDevice`, `IGameInputReading`)
//! with physical controllers (XInput), keyboard and mouse polling, rumble,
//! reading history, and device enumeration.

use std::collections::VecDeque;
use std::time::Instant;

/// Bitflags matching GameInputKind.
pub const GAME_INPUT_KIND_GAMEPAD: u32 = 0x0002_0000;
pub const GAME_INPUT_KIND_KEYBOARD: u32 = 0x0000_0010;
pub const GAME_INPUT_KIND_MOUSE: u32 = 0x0000_0020;

/// Bitflags matching GameInputGamepadButtons.
pub const BTN_MENU: u32 = 0x0000_0001;
pub const BTN_VIEW: u32 = 0x0000_0002;
pub const BTN_A: u32 = 0x0000_0004;
pub const BTN_B: u32 = 0x0000_0008;
pub const BTN_X: u32 = 0x0000_0010;
pub const BTN_Y: u32 = 0x0000_0020;
pub const BTN_DPAD_UP: u32 = 0x0000_0040;
pub const BTN_DPAD_DOWN: u32 = 0x0000_0080;
pub const BTN_DPAD_LEFT: u32 = 0x0000_0100;
pub const BTN_DPAD_RIGHT: u32 = 0x0000_0200;
pub const BTN_LEFT_SHOULDER: u32 = 0x0000_0400;
pub const BTN_RIGHT_SHOULDER: u32 = 0x0000_0800;
pub const BTN_LEFT_THUMB: u32 = 0x0000_1000;
pub const BTN_RIGHT_THUMB: u32 = 0x0000_2000;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GamepadState {
    pub buttons: u32,
    pub left_trigger: f32,
    pub right_trigger: f32,
    pub left_thumbstick_x: f32,
    pub left_thumbstick_y: f32,
    pub right_thumbstick_x: f32,
    pub right_thumbstick_y: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MouseState {
    pub buttons: u32,
    pub position_x: i64,
    pub position_y: i64,
    pub wheel_x: i64,
    pub wheel_y: i64,
}

#[derive(Debug, Clone)]
pub struct InputReading {
    pub id: u64,
    pub timestamp_us: u64,
    pub kind: u32,
    pub device_id: u32,
    pub gamepad: GamepadState,
    pub mouse: MouseState,
    pub keys_down: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct InputDeviceInfo {
    pub id: u32,
    pub name: String,
    pub supported_kinds: u32,
    pub connected: bool,
}

#[cfg(windows)]
unsafe extern "system" {
    fn GetAsyncKeyState(vkey: i32) -> i16;
    fn GetCursorPos(point: *mut i32) -> i32;
}

/// XInput buttons → `GameInputGamepadButtons`.
pub fn gamepad_from_xinput(x: &crate::input::GamepadState) -> GamepadState {
    use crate::input as xi;
    let map = [
        (xi::BTN_MENU, BTN_MENU),
        (xi::BTN_VIEW, BTN_VIEW),
        (xi::BTN_A, BTN_A),
        (xi::BTN_B, BTN_B),
        (xi::BTN_X, BTN_X),
        (xi::BTN_Y, BTN_Y),
        (xi::BTN_DPAD_UP, BTN_DPAD_UP),
        (xi::BTN_DPAD_DOWN, BTN_DPAD_DOWN),
        (xi::BTN_DPAD_LEFT, BTN_DPAD_LEFT),
        (xi::BTN_DPAD_RIGHT, BTN_DPAD_RIGHT),
        (xi::BTN_LEFT_SHOULDER, BTN_LEFT_SHOULDER),
        (xi::BTN_RIGHT_SHOULDER, BTN_RIGHT_SHOULDER),
        (xi::BTN_LEFT_THUMB, BTN_LEFT_THUMB),
        (xi::BTN_RIGHT_THUMB, BTN_RIGHT_THUMB),
    ];
    GamepadState {
        buttons: map
            .iter()
            .filter(|(from, _)| x.buttons & from != 0)
            .fold(0, |acc, (_, to)| acc | to),
        left_trigger: x.left_trigger,
        right_trigger: x.right_trigger,
        left_thumbstick_x: x.left_stick_x,
        left_thumbstick_y: x.left_stick_y,
        right_thumbstick_x: x.right_stick_x,
        right_thumbstick_y: x.right_stick_y,
    }
}

/// Real GameInput subsystem managing virtual/host input devices and reading ring buffers.
pub struct GameInputManager {
    epoch: Instant,
    next_reading_id: u64,
    devices: Vec<InputDeviceInfo>,
    history: VecDeque<InputReading>,
    max_history: usize,
    current_gamepad: GamepadState,
    current_mouse: MouseState,
    current_keys: Vec<u8>,
}

impl Default for GameInputManager {
    fn default() -> Self {
        Self::new()
    }
}

impl GameInputManager {
    pub fn new() -> Self {
        let devices = vec![
            InputDeviceInfo {
                id: 1,
                name: "Strata Virtual Controller 0".into(),
                supported_kinds: GAME_INPUT_KIND_GAMEPAD,
                connected: true,
            },
            InputDeviceInfo {
                id: 2,
                name: "Strata Host Keyboard & Mouse".into(),
                supported_kinds: GAME_INPUT_KIND_KEYBOARD | GAME_INPUT_KIND_MOUSE,
                connected: true,
            },
        ];
        Self {
            epoch: Instant::now(),
            next_reading_id: 1,
            devices,
            history: VecDeque::with_capacity(64),
            max_history: 64,
            current_gamepad: GamepadState::default(),
            current_mouse: MouseState::default(),
            current_keys: Vec::new(),
        }
    }

    pub fn current_timestamp_us(&self) -> u64 {
        self.epoch.elapsed().as_micros() as u64
    }

    pub fn devices(&self) -> &[InputDeviceInfo] {
        &self.devices
    }

    /// Poll the host: the first connected XInput controller (a physical
    /// gamepad), else the keyboard mapped onto a virtual one; the whole
    /// keyboard; and the mouse.
    pub fn poll_host(&mut self) {
        let mut gp = GamepadState::default();
        let mut keys = Vec::new();
        let pad = crate::input::first_controller();
        if let Some((_, s)) = pad {
            gp = gamepad_from_xinput(&s.to_gamepad());
        }
        #[cfg(windows)]
        {
            let pressed =
                |vk: i32| -> bool { unsafe { (GetAsyncKeyState(vk) as u16 & 0x8000) != 0 } };
            for vk in 0x08..0xFF {
                if pressed(vk) {
                    keys.push(vk as u8);
                }
            }
            if pad.is_none() {
                // No controller: WASD / Space / Enter / Esc / E / Q / Shift.
                let k = |vk: u8| keys.contains(&vk);
                if k(0x57) {
                    gp.left_thumbstick_y += 1.0;
                }
                if k(0x53) {
                    gp.left_thumbstick_y -= 1.0;
                }
                if k(0x41) {
                    gp.left_thumbstick_x -= 1.0;
                }
                if k(0x44) {
                    gp.left_thumbstick_x += 1.0;
                }
                if k(0x20) || k(0x0D) {
                    gp.buttons |= BTN_A;
                }
                if k(0x1B) {
                    gp.buttons |= BTN_MENU;
                }
                if k(0x45) {
                    gp.buttons |= BTN_X;
                }
                if k(0x51) {
                    gp.buttons |= BTN_Y;
                }
                if k(0x10) {
                    gp.right_trigger = 1.0;
                }
            }
            let mut p = [0i32; 2];
            if unsafe { GetCursorPos(p.as_mut_ptr()) } != 0 {
                let mut buttons = 0;
                for (bit, vk) in [(1, 0x01), (2, 0x02), (4, 0x04), (8, 0x05), (16, 0x06)] {
                    if pressed(vk) {
                        buttons |= bit;
                    }
                }
                self.current_mouse = MouseState {
                    buttons,
                    position_x: i64::from(p[0]),
                    position_y: i64::from(p[1]),
                    ..self.current_mouse
                };
            }
        }
        self.devices[0].connected = true;
        self.devices[0].name = match pad {
            Some((u, _)) => format!("XInput Controller {u}"),
            None => "Strata Virtual Controller 0 (keyboard)".into(),
        };
        self.current_keys = keys;
        if gp != self.current_gamepad || self.history.is_empty() {
            self.push_gamepad(gp);
        }
        self.record(GAME_INPUT_KIND_KEYBOARD | GAME_INPUT_KIND_MOUSE, 2);
    }

    /// `IGameInputDevice::SetRumbleState`: forwarded to the controller.
    pub fn set_rumble(&self, low: f32, high: f32, left_trigger: f32, right_trigger: f32) {
        // XInput has two motors; trigger impulse motors fold into them.
        let user = crate::input::first_controller().map_or(0, |(u, _)| u);
        crate::input::set_rumble(user, low.max(left_trigger), high.max(right_trigger));
    }

    pub fn get_previous_reading(&self, before_id: u64, kind_mask: u32) -> Option<&InputReading> {
        self.history
            .iter()
            .rev()
            .find(|r| r.id < before_id && (r.kind & kind_mask) != 0)
    }

    /// The reading nearest `timestamp_us` (at or before it).
    pub fn get_temporal_reading(&self, timestamp_us: u64) -> Option<&InputReading> {
        self.history
            .iter()
            .rev()
            .find(|r| r.timestamp_us <= timestamp_us)
    }

    /// Inject a gamepad state snapshot (used by host adapters or tests).
    pub fn push_gamepad(&mut self, state: GamepadState) -> u64 {
        self.current_gamepad = state;
        self.record(GAME_INPUT_KIND_GAMEPAD, 1)
    }

    pub fn push_mouse(&mut self, state: MouseState) -> u64 {
        self.current_mouse = state;
        self.record(GAME_INPUT_KIND_MOUSE, 2)
    }

    fn record(&mut self, kind: u32, device_id: u32) -> u64 {
        let id = self.next_reading_id;
        self.next_reading_id += 1;
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(InputReading {
            id,
            timestamp_us: self.current_timestamp_us(),
            kind,
            device_id,
            gamepad: self.current_gamepad,
            mouse: self.current_mouse,
            keys_down: self.current_keys.clone(),
        });
        id
    }

    pub fn get_current_reading(&self, kind_mask: u32) -> Option<&InputReading> {
        self.history
            .iter()
            .rev()
            .find(|r| (r.kind & kind_mask) != 0)
    }

    pub fn get_next_reading(&self, after_id: u64, kind_mask: u32) -> Option<&InputReading> {
        self.history
            .iter()
            .find(|r| r.id > after_id && (r.kind & kind_mask) != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gameinput_tracks_readings_and_timestamps() {
        let mut gi = GameInputManager::new();
        let id1 = gi.push_gamepad(GamepadState {
            buttons: BTN_A,
            left_trigger: 0.75,
            ..Default::default()
        });
        let id2 = gi.push_gamepad(GamepadState {
            buttons: BTN_B,
            right_trigger: 1.0,
            ..Default::default()
        });

        let cur = gi.get_current_reading(GAME_INPUT_KIND_GAMEPAD).unwrap();
        assert_eq!(cur.id, id2);
        assert_eq!(cur.gamepad.buttons, BTN_B);

        let next = gi.get_next_reading(id1, GAME_INPUT_KIND_GAMEPAD).unwrap();
        assert_eq!(next.id, id2);
        let prev = gi
            .get_previous_reading(id2, GAME_INPUT_KIND_GAMEPAD)
            .unwrap();
        assert_eq!(prev.id, id1);
    }

    #[test]
    fn xinput_buttons_map_to_gameinput_bits() {
        use crate::input as xi;
        let g = gamepad_from_xinput(&xi::GamepadState {
            buttons: xi::BTN_A | xi::BTN_MENU | xi::BTN_DPAD_LEFT | xi::BTN_RIGHT_THUMB,
            left_trigger: 0.5,
            ..Default::default()
        });
        assert_eq!(
            g.buttons,
            BTN_A | BTN_MENU | BTN_DPAD_LEFT | BTN_RIGHT_THUMB
        );
        assert_eq!(g.left_trigger, 0.5);
    }
}
