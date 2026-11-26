//! Pure Win32 HWND window wrapper (zero external crates) for presenting
//! `SwapchainPresenter` frames on Windows desktop.

use std::ffi::c_void;

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
    fn PeekMessageW(msg: *mut Msg, hwnd: *mut c_void, min: u32, max: u32, remove: u32) -> i32;
    fn TranslateMessage(msg: *const Msg) -> i32;
    fn DispatchMessageW(msg: *const Msg) -> isize;
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

pub struct Win32Window {
    pub hinstance: *mut c_void,
    pub hwnd: *mut c_void,
    pub width: u32,
    pub height: u32,
}

impl Win32Window {
    pub fn create(title: &str, width: u32, height: u32) -> Option<Self> {
        let class_name: Vec<u16> = "StrataWindowClass\0".encode_utf16().collect();
        let win_title: Vec<u16> = format!("{title}\0").encode_utf16().collect();

        unsafe {
            let hinstance = GetModuleHandleW(std::ptr::null());
            let wc = WndClassExW {
                cb_size: std::mem::size_of::<WndClassExW>() as u32,
                style: 0,
                wnd_proc: DefWindowProcW,
                cls_extra: 0,
                wnd_extra: 0,
                instance: hinstance,
                icon: std::ptr::null_mut(),
                cursor: std::ptr::null_mut(),
                bg_brush: std::ptr::null_mut(),
                menu_name: std::ptr::null(),
                class_name: class_name.as_ptr(),
                icon_sm: std::ptr::null_mut(),
            };
            RegisterClassExW(&wc);

            const WS_OVERLAPPEDWINDOW: u32 = 0x00CF_0000;
            const CW_USEDEFAULT: i32 = 0x8000_0000u32 as i32;
            let hwnd = CreateWindowExW(
                0,
                class_name.as_ptr(),
                win_title.as_ptr(),
                WS_OVERLAPPEDWINDOW,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                width as i32,
                height as i32,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                hinstance,
                std::ptr::null_mut(),
            );
            if hwnd.is_null() {
                return None;
            }
            Some(Self { hinstance, hwnd, width, height })
        }
    }

    pub fn pump_messages(&self) -> bool {
        let mut msg = Msg::default();
        unsafe {
            while PeekMessageW(&mut msg, self.hwnd, 0, 0, 1) != 0 {
                if msg.message == 0x0012 {
                    return false; // WM_QUIT
                }
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
        true
    }
}

impl Drop for Win32Window {
    fn drop(&mut self) {
        if !self.hwnd.is_null() {
            unsafe {
                DestroyWindow(self.hwnd);
            }
        }
    }
}
