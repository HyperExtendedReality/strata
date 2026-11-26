//! Strata High-Level Emulation (PE32+ loader analysis + `kernelx`/`d3d12_xs` bridge).

pub mod d3d12x;
pub mod gdk;
pub mod hooks;
pub mod input;
pub mod loader;
pub mod pe;
pub mod runtime;
pub mod thunks;
pub mod tls;
pub mod veh;
pub mod vfs;
pub mod xg;
pub mod xmem_phys;
