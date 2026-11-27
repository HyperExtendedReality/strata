//! Strata High-Level Emulation (PE32+ loader analysis + `kernelx`/`d3d12_xs` bridge).

/// Looks a host export up once and caches it (0 if missing).
macro_rules! host_fn {
    ($dll:literal, $sym:literal) => {{
        static P: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *P.get_or_init(|| crate::loader::host_export($dll, $sym))
    }};
}

pub mod audio;
pub mod config;
pub mod crash;
pub mod d3d12_abi;
pub mod d3d12x;
pub mod dstorage;
pub mod files;
pub mod gameinput;
pub mod gdk;
pub mod hooks;
pub mod implib;
pub mod input;
pub mod loader;
pub mod pe;
pub mod profile;
pub mod runtime;
pub mod services;
pub mod stubs;
#[doc(hidden)]
pub mod testpe;
pub mod threads;
pub mod thunks;
pub mod title;
pub mod tls;
pub mod veh;
pub mod vfs;
pub mod xasync;
pub mod xg;
pub mod xgameruntime;
pub mod xmem_phys;
