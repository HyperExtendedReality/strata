//! The ACP channel ABI observed by XCL's retail-title probes. This models
//! command completion and the audio frame clock used by the title.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::Mutex;
use std::time::{Duration, Instant};

const AUDIO_FRAME: u32 = 2;
const COMMAND_DONE: u32 = 0x10;
const DISCONNECTED: u32 = 0x80;
const FRAME_PERIOD: Duration = Duration::from_nanos(2_666_667); // 128 samples at 48 kHz
const E_UNEXPECTED: i32 = 0x8000_ffff_u32 as i32;
const E_OUTOFMEMORY: i32 = 0x8007_000e_u32 as i32;
const E_INVALIDARG: i32 = 0x8007_0057_u32 as i32;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
struct Message {
    kind: u32,
    reserved: u32,
    value: u64,
    id: u64,
}

#[derive(Default)]
struct Channel {
    events: VecDeque<Message>,
    frame_at: Option<Instant>,
}

#[repr(C)]
struct AcpHal {
    vtable: &'static Vtable,
    channel: Mutex<Channel>,
}

#[repr(C)]
struct Vtable {
    connect: unsafe extern "system" fn(&AcpHal, u32, u32) -> i32,
    disconnect: unsafe extern "system" fn(&AcpHal) -> i32,
    submit: unsafe extern "system" fn(&AcpHal, u32, u64, u32, *const c_void, u32) -> i32,
    pop: unsafe extern "system" fn(&AcpHal, *mut Message) -> bool,
    count: unsafe extern "system" fn(&AcpHal) -> u32,
    release: unsafe extern "system" fn(*mut AcpHal),
}

static VTABLE: Vtable = Vtable {
    connect,
    disconnect,
    submit,
    pop,
    count,
    release,
};

unsafe extern "system" fn connect(hal: &AcpHal, _commands: u32, _messages: u32) -> i32 {
    let mut state = hal.channel.lock().unwrap_or_else(|e| e.into_inner());
    state.events.clear();
    state.frame_at = Some(Instant::now() + FRAME_PERIOD);
    0
}

unsafe extern "system" fn disconnect(hal: &AcpHal) -> i32 {
    let mut state = hal.channel.lock().unwrap_or_else(|e| e.into_inner());
    if state.frame_at.is_some() {
        if state.events.try_reserve(1).is_err() {
            return E_OUTOFMEMORY;
        }
        state.events.push_back(Message {
            kind: DISCONNECTED,
            ..Default::default()
        });
        state.frame_at = None;
    }
    0
}

unsafe extern "system" fn submit(
    hal: &AcpHal,
    _kind: u32,
    id: u64,
    _frame: u32,
    _data: *const c_void,
    _size: u32,
) -> i32 {
    let mut state = hal.channel.lock().unwrap_or_else(|e| e.into_inner());
    if state.frame_at.is_none() {
        return E_UNEXPECTED;
    }
    if id != 0 {
        if state.events.try_reserve(1).is_err() {
            return E_OUTOFMEMORY;
        }
        state.events.push_back(Message {
            kind: COMMAND_DONE,
            id,
            ..Default::default()
        });
    }
    0
}

unsafe extern "system" fn pop(hal: &AcpHal, output: *mut Message) -> bool {
    if output.is_null() {
        return false;
    }
    let mut state = hal.channel.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    let message = if let Some(message) = state.events.pop_front() {
        message
    } else if let Some(frame) = state.frame_at.filter(|&frame| frame <= now) {
        // A title returning from a pause gets a fresh tick without having
        // to drain all of the time that elapsed while it was paused.
        state.frame_at = Some((frame + FRAME_PERIOD).max(now));
        Message {
            kind: AUDIO_FRAME,
            ..Default::default()
        }
    } else {
        return false;
    };
    unsafe { output.write(message) };
    true
}

unsafe extern "system" fn count(hal: &AcpHal) -> u32 {
    let state = hal.channel.lock().unwrap_or_else(|e| e.into_inner());
    let due = state.frame_at.is_some_and(|frame| Instant::now() >= frame);
    u32::try_from(state.events.len())
        .unwrap_or(u32::MAX)
        .saturating_add(u32::from(due))
}

unsafe extern "system" fn release(hal: *mut AcpHal) {
    unsafe { drop(Box::from_raw(hal)) };
}

pub(crate) unsafe extern "system" fn create(out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let hal = Box::new(AcpHal {
        vtable: &VTABLE,
        channel: Mutex::new(Channel::default()),
    });
    unsafe { out.write(Box::into_raw(hal).cast()) };
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_acp_vtable_delivers_command_and_disconnect_messages() {
        let mut ptr = std::ptr::null_mut();
        let exported = crate::thunks::resolve_hle("acphal.dll", "AcpHalCreate").unwrap();
        unsafe {
            let create: unsafe extern "system" fn(*mut *mut c_void) -> i32 =
                std::mem::transmute(exported);
            assert_eq!(create(&mut ptr), 0);
            let hal = &*ptr.cast::<AcpHal>();
            let v = hal.vtable;
            let mut message = Message::default();
            assert_eq!((v.count)(hal), 0);
            assert_eq!((v.submit)(hal, 2, 99, 0, std::ptr::null(), 0), E_UNEXPECTED);
            assert_eq!((v.connect)(hal, 8, 8), 0);
            assert_eq!((v.submit)(hal, 2, 99, 0, std::ptr::null(), 0), 0);
            assert!((v.pop)(hal, &mut message));
            assert_eq!((message.kind, message.id), (COMMAND_DONE, 99));
            hal.channel.lock().unwrap().frame_at = Some(Instant::now());
            assert!((v.pop)(hal, &mut message));
            assert_eq!(message.kind, AUDIO_FRAME);
            assert_eq!((v.disconnect)(hal), 0);
            assert!((v.pop)(hal, &mut message));
            assert_eq!(message.kind, DISCONNECTED);
            assert_eq!((v.count)(hal), 0);
            (v.release)(ptr.cast());
        }
    }
}
