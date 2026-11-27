//! Host controllers for `GameInput` / XInput HLE: physical gamepads through
//! XInput (`xinput1_4.dll`, loaded at run time; `xinput9_1_0.dll` as a
//! fallback), rumble, and the keyboard-to-pad mapping used when no
//! controller is connected.

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

/// XInput `wButtons` bits.
pub const BTN_DPAD_UP: u16 = 0x0001;
pub const BTN_DPAD_DOWN: u16 = 0x0002;
pub const BTN_DPAD_LEFT: u16 = 0x0004;
pub const BTN_DPAD_RIGHT: u16 = 0x0008;
pub const BTN_MENU: u16 = 0x0010;
pub const BTN_VIEW: u16 = 0x0020;
pub const BTN_LEFT_THUMB: u16 = 0x0040;
pub const BTN_RIGHT_THUMB: u16 = 0x0080;
pub const BTN_LEFT_SHOULDER: u16 = 0x0100;
pub const BTN_RIGHT_SHOULDER: u16 = 0x0200;
pub const BTN_A: u16 = 0x1000;
pub const BTN_B: u16 = 0x2000;
pub const BTN_X: u16 = 0x4000;
pub const BTN_Y: u16 = 0x8000;

/// XInput's recommended dead zones.
const LEFT_DEADZONE: f32 = 7849.0;
const RIGHT_DEADZONE: f32 = 8689.0;
const TRIGGER_THRESHOLD: f32 = 30.0;

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

/// A thumbstick axis pair with XInput's radial dead zone, scaled to [-1, 1].
pub fn stick(x: i16, y: i16, deadzone: f32) -> (f32, f32) {
    let (fx, fy) = (f32::from(x), f32::from(y));
    let mag = (fx * fx + fy * fy).sqrt();
    if mag <= deadzone {
        return (0.0, 0.0);
    }
    let scaled = ((mag - deadzone) / (32767.0 - deadzone)).min(1.0);
    (
        (fx / mag * scaled).clamp(-1.0, 1.0),
        (fy / mag * scaled).clamp(-1.0, 1.0),
    )
}

/// A trigger with XInput's threshold, scaled to [0, 1].
pub fn trigger(v: u8) -> f32 {
    let v = f32::from(v);
    if v <= TRIGGER_THRESHOLD {
        0.0
    } else {
        ((v - TRIGGER_THRESHOLD) / (255.0 - TRIGGER_THRESHOLD)).min(1.0)
    }
}

/// `XINPUT_STATE`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct XInputState {
    pub packet: u32,
    pub buttons: u16,
    pub left_trigger: u8,
    pub right_trigger: u8,
    pub thumb_lx: i16,
    pub thumb_ly: i16,
    pub thumb_rx: i16,
    pub thumb_ry: i16,
}

impl XInputState {
    pub fn to_gamepad(self) -> GamepadState {
        let (lx, ly) = stick(self.thumb_lx, self.thumb_ly, LEFT_DEADZONE);
        let (rx, ry) = stick(self.thumb_rx, self.thumb_ry, RIGHT_DEADZONE);
        GamepadState {
            buttons: self.buttons,
            left_trigger: trigger(self.left_trigger),
            right_trigger: trigger(self.right_trigger),
            left_stick_x: lx,
            left_stick_y: ly,
            right_stick_x: rx,
            right_stick_y: ry,
        }
    }
}

type GetStateFn = unsafe extern "system" fn(u32, *mut XInputState) -> u32;
type SetStateFn = unsafe extern "system" fn(u32, *const [u16; 2]) -> u32;

struct XInput {
    get_state: GetStateFn,
    set_state: Option<SetStateFn>,
}

#[cfg(windows)]
unsafe extern "system" {
    fn LoadLibraryA(name: *const i8) -> *mut core::ffi::c_void;
    fn GetProcAddress(m: *mut core::ffi::c_void, name: *const i8) -> *const core::ffi::c_void;
}

/// XInput from the system directory (never the title's or current folder:
/// the loader's search order puts System32 first for these known DLLs).
fn xinput() -> Option<&'static XInput> {
    static XI: std::sync::OnceLock<Option<XInput>> = std::sync::OnceLock::new();
    XI.get_or_init(|| {
        #[cfg(windows)]
        unsafe {
            for name in [c"xinput1_4.dll", c"xinput9_1_0.dll"] {
                let m = LoadLibraryA(name.as_ptr().cast());
                if m.is_null() {
                    continue;
                }
                let g = GetProcAddress(m, c"XInputGetState".as_ptr().cast());
                if g.is_null() {
                    continue;
                }
                let s = GetProcAddress(m, c"XInputSetState".as_ptr().cast());
                return Some(XInput {
                    get_state: std::mem::transmute::<*const core::ffi::c_void, GetStateFn>(g),
                    set_state: (!s.is_null())
                        .then(|| std::mem::transmute::<*const core::ffi::c_void, SetStateFn>(s)),
                });
            }
        }
        None
    })
    .as_ref()
}

/// The state of controller `user` (0..4), if one is connected.
pub fn read_controller(user: u32) -> Option<XInputState> {
    let xi = xinput()?;
    let mut s = XInputState::default();
    // SAFETY: XInputGetState writes one XINPUT_STATE.
    (unsafe { (xi.get_state)(user, &mut s) } == 0).then_some(s)
}

/// The first connected controller's index and state.
pub fn first_controller() -> Option<(u32, XInputState)> {
    (0..4).find_map(|u| read_controller(u).map(|s| (u, s)))
}

/// Sets controller `user`'s low- and high-frequency motor speeds (0..1).
pub fn set_rumble(user: u32, low: f32, high: f32) -> bool {
    let Some(set) = xinput().and_then(|x| x.set_state) else {
        return false;
    };
    let v = [
        (low.clamp(0.0, 1.0) * 65535.0) as u16,
        (high.clamp(0.0, 1.0) * 65535.0) as u16,
    ];
    // SAFETY: XInputSetState reads one XINPUT_VIBRATION.
    unsafe { set(user, &v) == 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sticks_and_triggers_use_xinput_dead_zones() {
        assert_eq!(stick(5000, 0, LEFT_DEADZONE), (0.0, 0.0));
        let (x, y) = stick(32767, 0, LEFT_DEADZONE);
        assert!((x - 1.0).abs() < 1e-4 && y == 0.0);
        let (x, y) = stick(-32768, -32768, LEFT_DEADZONE);
        assert!(x < -0.7 && y < -0.7 && (x * x + y * y).sqrt() <= 1.0001);
        assert_eq!(trigger(20), 0.0);
        assert_eq!(trigger(255), 1.0);
        let pad = XInputState {
            buttons: BTN_A | BTN_LEFT_SHOULDER,
            right_trigger: 255,
            ..Default::default()
        }
        .to_gamepad();
        assert_eq!(pad.buttons, BTN_A | BTN_LEFT_SHOULDER);
        assert_eq!(pad.right_trigger, 1.0);
    }

    #[test]
    fn polling_without_a_controller_is_harmless() {
        // Machines without controllers (CI) report none; with one, a state.
        let _ = first_controller();
        let _ = set_rumble(3, 0.0, 0.0);
    }
}
