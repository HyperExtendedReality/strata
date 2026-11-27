//! Pure Win32 HWND window (zero external crates) the title's frames are
//! shown in: closing it ends the title, Alt+Enter toggles borderless
//! fullscreen on its monitor, and it can be sized to the back buffer.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetModuleHandleW(name: *const u16) -> *mut c_void;
}

#[cfg(windows)]
#[link(name = "user32")]
unsafe extern "system" {
    fn DefWindowProcW(hwnd: *mut c_void, msg: u32, wp: usize, lp: isize) -> isize;
    fn RegisterClassExW(wc: *const WndClassExW) -> u16;
    fn CreateWindowExW(
        ex_style: u32,
        class_name: *const u16,
        window_name: *const u16,
        style: u32,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        parent: *mut c_void,
        menu: *mut c_void,
        instance: *mut c_void,
        param: *mut c_void,
    ) -> *mut c_void;
    fn DestroyWindow(hwnd: *mut c_void) -> i32;
    fn ShowWindow(hwnd: *mut c_void, cmd: i32) -> i32;
    fn PeekMessageW(msg: *mut Msg, hwnd: *mut c_void, min: u32, max: u32, remove: u32) -> i32;
    fn TranslateMessage(msg: *const Msg) -> i32;
    fn DispatchMessageW(msg: *const Msg) -> isize;
    fn GetClientRect(hwnd: *mut c_void, rect: *mut [i32; 4]) -> i32;
    fn GetWindowRect(hwnd: *mut c_void, rect: *mut [i32; 4]) -> i32;
    fn AdjustWindowRectEx(rect: *mut [i32; 4], style: u32, menu: i32, ex_style: u32) -> i32;
    fn SetWindowPos(
        hwnd: *mut c_void,
        after: *mut c_void,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        flags: u32,
    ) -> i32;
    fn SetWindowLongPtrW(hwnd: *mut c_void, index: i32, value: isize) -> isize;
    fn MonitorFromWindow(hwnd: *mut c_void, flags: u32) -> *mut c_void;
    fn GetMonitorInfoW(monitor: *mut c_void, info: *mut MonitorInfo) -> i32;
    fn LoadCursorW(instance: *mut c_void, name: *const u16) -> *mut c_void;
}

#[repr(C)]
struct WndClassExW {
    cb_size: u32,
    style: u32,
    wnd_proc: unsafe extern "system" fn(*mut c_void, u32, usize, isize) -> isize,
    cls_extra: i32,
    wnd_extra: i32,
    instance: *mut c_void,
    icon: *mut c_void,
    cursor: *mut c_void,
    bg_brush: *mut c_void,
    menu_name: *const u16,
    class_name: *const u16,
    icon_sm: *mut c_void,
}

#[repr(C)]
#[derive(Default)]
struct Msg {
    hwnd: *mut c_void,
    message: u32,
    wparam: usize,
    lparam: isize,
    time: u32,
    pt_x: i32,
    pt_y: i32,
}

#[repr(C)]
struct MonitorInfo {
    size: u32,
    monitor: [i32; 4],
    work: [i32; 4],
    flags: u32,
}

const WS_OVERLAPPEDWINDOW: u32 = 0x00CF_0000;
const WS_POPUP: u32 = 0x8000_0000;
const WS_VISIBLE: u32 = 0x1000_0000;
const GWL_STYLE: i32 = -16;
const SWP_NOZORDER: u32 = 0x0004;
const SWP_FRAMECHANGED: u32 = 0x0020;
const SWP_NOMOVE: u32 = 0x0002;

/// The window was closed (WM_CLOSE / WM_DESTROY).
static CLOSED: AtomicBool = AtomicBool::new(false);
/// Alt+Enter was pressed since the last pump.
static TOGGLE_FULLSCREEN: AtomicBool = AtomicBool::new(false);

#[cfg(windows)]
unsafe extern "system" fn wnd_proc(hwnd: *mut c_void, msg: u32, wp: usize, lp: isize) -> isize {
    const WM_CLOSE: u32 = 0x0010;
    const WM_SYSKEYDOWN: u32 = 0x0104;
    const WM_SYSCHAR: u32 = 0x0106;
    const VK_RETURN: usize = 0x0D;
    match msg {
        WM_CLOSE => {
            CLOSED.store(true, Ordering::Release);
            0
        }
        // Alt+Enter (bit 29: Alt held).
        WM_SYSKEYDOWN if wp == VK_RETURN && (lp >> 29) & 1 == 1 => {
            TOGGLE_FULLSCREEN.store(true, Ordering::Release);
            0
        }
        // Swallow the Alt+Enter beep.
        WM_SYSCHAR if wp == VK_RETURN => 0,
        _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
    }
}

pub struct Win32Window {
    pub hinstance: *mut c_void,
    pub hwnd: *mut c_void,
    pub width: u32,
    pub height: u32,
    /// Borderless fullscreen, and the windowed rect to return to.
    pub fullscreen: bool,
    windowed_rect: [i32; 4],
}

impl Win32Window {
    #[cfg(windows)]
    pub fn create(title: &str, width: u32, height: u32) -> Option<Self> {
        let class_name: Vec<u16> = "StrataWindowClass\0".encode_utf16().collect();
        let win_title: Vec<u16> = format!("{title}\0").encode_utf16().collect();

        unsafe {
            let hinstance = GetModuleHandleW(std::ptr::null());
            let wc = WndClassExW {
                cb_size: std::mem::size_of::<WndClassExW>() as u32,
                style: 0,
                wnd_proc,
                cls_extra: 0,
                wnd_extra: 0,
                instance: hinstance,
                icon: std::ptr::null_mut(),
                cursor: LoadCursorW(std::ptr::null_mut(), 32512 as *const u16), // IDC_ARROW
                bg_brush: std::ptr::null_mut(),
                menu_name: std::ptr::null(),
                class_name: class_name.as_ptr(),
                icon_sm: std::ptr::null_mut(),
            };
            RegisterClassExW(&wc);

            // `width`×`height` is the client area: grow the window by its frame.
            let mut r = [0, 0, width as i32, height as i32];
            AdjustWindowRectEx(&mut r, WS_OVERLAPPEDWINDOW, 0, 0);
            const CW_USEDEFAULT: i32 = 0x8000_0000u32 as i32;
            let hwnd = CreateWindowExW(
                0,
                class_name.as_ptr(),
                win_title.as_ptr(),
                WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                r[2] - r[0],
                r[3] - r[1],
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                hinstance,
                std::ptr::null_mut(),
            );
            if hwnd.is_null() {
                return None;
            }
            ShowWindow(hwnd, 5); // SW_SHOW
            CLOSED.store(false, Ordering::Release);
            Some(Self {
                hinstance,
                hwnd,
                width,
                height,
                fullscreen: false,
                windowed_rect: [0; 4],
            })
        }
    }

    #[cfg(not(windows))]
    pub fn create(_title: &str, _width: u32, _height: u32) -> Option<Self> {
        None
    }

    /// Handles pending messages (and Alt+Enter). `false` once the window
    /// has been closed.
    pub fn pump_messages(&mut self) -> bool {
        #[cfg(windows)]
        unsafe {
            let mut msg = Msg::default();
            while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, 1) != 0 {
                if msg.message == 0x0012 {
                    CLOSED.store(true, Ordering::Release); // WM_QUIT
                }
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
        if TOGGLE_FULLSCREEN.swap(false, Ordering::AcqRel) {
            self.set_fullscreen(!self.fullscreen);
        }
        let (w, h) = self.client_size();
        self.width = w;
        self.height = h;
        !CLOSED.load(Ordering::Acquire)
    }

    /// The client area in pixels.
    pub fn client_size(&self) -> (u32, u32) {
        #[cfg(windows)]
        unsafe {
            let mut r = [0i32; 4];
            if GetClientRect(self.hwnd, &mut r) != 0 {
                return ((r[2] - r[0]).max(0) as u32, (r[3] - r[1]).max(0) as u32);
            }
        }
        (self.width, self.height)
    }

    /// Sizes the window so its client area is `width`×`height` (windowed only).
    pub fn set_client_size(&mut self, width: u32, height: u32) {
        if self.fullscreen {
            return;
        }
        #[cfg(windows)]
        unsafe {
            let mut r = [0, 0, width as i32, height as i32];
            AdjustWindowRectEx(&mut r, WS_OVERLAPPEDWINDOW, 0, 0);
            SetWindowPos(
                self.hwnd,
                std::ptr::null_mut(),
                0,
                0,
                r[2] - r[0],
                r[3] - r[1],
                SWP_NOZORDER | SWP_NOMOVE,
            );
        }
        self.width = width;
        self.height = height;
    }

    /// Borderless fullscreen over the window's monitor, or back to the
    /// windowed rect.
    pub fn set_fullscreen(&mut self, on: bool) {
        if on == self.fullscreen {
            return;
        }
        #[cfg(windows)]
        unsafe {
            if on {
                GetWindowRect(self.hwnd, &mut self.windowed_rect);
                let mut mi = MonitorInfo {
                    size: std::mem::size_of::<MonitorInfo>() as u32,
                    monitor: [0; 4],
                    work: [0; 4],
                    flags: 0,
                };
                GetMonitorInfoW(MonitorFromWindow(self.hwnd, 2), &mut mi);
                SetWindowLongPtrW(self.hwnd, GWL_STYLE, (WS_POPUP | WS_VISIBLE) as isize);
                let m = mi.monitor;
                SetWindowPos(
                    self.hwnd,
                    std::ptr::null_mut(),
                    m[0],
                    m[1],
                    m[2] - m[0],
                    m[3] - m[1],
                    SWP_NOZORDER | SWP_FRAMECHANGED,
                );
            } else {
                SetWindowLongPtrW(
                    self.hwnd,
                    GWL_STYLE,
                    (WS_OVERLAPPEDWINDOW | WS_VISIBLE) as isize,
                );
                let r = self.windowed_rect;
                SetWindowPos(
                    self.hwnd,
                    std::ptr::null_mut(),
                    r[0],
                    r[1],
                    r[2] - r[0],
                    r[3] - r[1],
                    SWP_NOZORDER | SWP_FRAMECHANGED,
                );
            }
        }
        self.fullscreen = on;
    }

    /// Whether the window has been closed.
    pub fn closed() -> bool {
        CLOSED.load(Ordering::Acquire)
    }
}

impl Drop for Win32Window {
    fn drop(&mut self) {
        #[cfg(windows)]
        if !self.hwnd.is_null() {
            unsafe {
                DestroyWindow(self.hwnd);
            }
        }
    }
}
