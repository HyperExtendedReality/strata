//! Runtime side of BMA traversal shortcuts ([`strata_shader::bma`]).
//!
//! The runner owns one device-local table. Every compute pipeline whose
//! SPIR-V carries the BMA spec constants is specialized to its address,
//! so instrumented shaders always read valid memory; with BMA off the table
//! is a small header with `FLAGS = 0` and the shaders behave exactly.
//!
//! Per frame: shaders sample the tile centres' BVH paths and take the
//! previous frame's shortcuts; [`ComputeRunner::bma_end_frame`] runs the
//! Mapping stage (`bma_mapping.slang`), then clears the sampled paths and
//! applies a root registered with [`ComputeRunner::bma_set_root`].

use crate::context::GpuError;
use crate::runner::{BdaBuffer, ComputeRunner};
use strata_shader::bma::{BmaConfig, table_header, table_layout};

/// Largest render target the table is sized for (tiles beyond it never
/// sample nor shortcut).
pub const MAX_WIDTH: u32 = 3840;
pub const MAX_HEIGHT: u32 = 2160;

/// The device-local BMA table.
pub struct BmaTable {
    pub(crate) buf: BdaBuffer,
    pub cfg: Option<BmaConfig>,
    pub tiles_x: u32,
    pub tiles_y: u32,
    /// Root to apply at the next frame boundary (`0` = keep).
    pending_root: u64,
    pub frames: u64,
}

// SAFETY: the buffer is device-local (its host pointer is never used) and
// the table is only reached through the runner's `Mutex`.
unsafe impl Send for BmaTable {}

impl BmaTable {
    pub fn address(&self) -> u64 {
        self.buf.bda
    }

    fn path_dwords(&self) -> u32 {
        self.cfg
            .map_or(0, |c| self.tiles_x * self.tiles_y * c.rays * c.depth)
    }

    fn entries(&self) -> u32 {
        self.cfg.map_or(0, |c| self.tiles_x * self.tiles_y * c.rays)
    }
}

/// Push constants of `bma_mapping.slang`.
fn mapping_push(
    table: u64,
    new_root: u64,
    cfg: &BmaConfig,
    mode: u32,
    flags: u32,
    count: u32,
) -> Vec<u8> {
    let mut p = Vec::with_capacity(64);
    p.extend_from_slice(&table.to_le_bytes());
    p.extend_from_slice(&new_root.to_le_bytes());
    p.extend_from_slice(&cfg.threshold.to_le_bytes());
    p.extend_from_slice(&mode.to_le_bytes());
    p.extend_from_slice(&flags.to_le_bytes());
    p.extend_from_slice(&count.to_le_bytes());
    for k in cfg.kernel_flat() {
        p.extend_from_slice(&k.to_le_bytes());
    }
    p
}

fn spirv_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Whether `words` declares the BMA table spec constants.
pub fn uses_bma_table(words: &[u32]) -> bool {
    // OpDecorate (71, 4 words) <target> SpecId(1) <id>
    words.windows(4).any(|w| {
        w[0] == (4 << 16) | 71 && w[2] == 1 && w[3] == strata_shader::bma::SPEC_ID_TABLE_LO
    })
}

impl ComputeRunner {
    /// The BMA table, created on first use from `STRATA_BVH_BMA` (a
    /// disabled header-only table when unset).
    pub fn bma_table_address(&self) -> Result<u64, GpuError> {
        let mut g = self.bma.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = g.as_ref() {
            return Ok(t.address());
        }
        let cfg = BmaConfig::from_env();
        let (tiles_x, tiles_y, size, header) = match cfg {
            Some(c) => {
                let tile = 1u32 << c.tile_shift();
                let (tx, ty) = (MAX_WIDTH.div_ceil(tile), MAX_HEIGHT.div_ceil(tile));
                let (size, _) = table_layout(tx, ty, c.rays, c.depth);
                (tx, ty, size, table_header(&c, tx, ty, 0, false))
            }
            None => {
                let c = BmaConfig::default();
                let (size, _) = table_layout(0, 0, c.rays, c.depth);
                (0, 0, size, table_header(&c, 0, 0, 0, false))
            }
        };
        // Header + zeroed entries; the paths are cleared by the reset pass.
        let buf = self.alloc_vram_bda_buffer(size.max(256), &header)?;
        let t = BmaTable {
            buf,
            cfg,
            tiles_x,
            tiles_y,
            pending_root: 0,
            frames: 0,
        };
        let addr = t.address();
        let fresh = t.cfg.is_some();
        *g = Some(t);
        drop(g);
        if fresh {
            // Clear entries and paths once (VRAM contents are undefined).
            self.bma_reset(0, true)?;
        }
        Ok(addr)
    }

    /// Registers the BVH (root box node address, i.e. structure base +
    /// header) whose traversals sample and shortcut; applied at the next
    /// [`Self::bma_end_frame`]. Only Strata-built structures qualify (the
    /// shaders read their build generation and size from the header).
    pub fn bma_set_root(&self, root: u64) {
        if !strata_shader::bma::shader_instrumentation_enabled()
            || self.bma_table_address().is_err()
        {
            return;
        }
        if let Some(t) = self.bma.lock().unwrap_or_else(|e| e.into_inner()).as_mut()
            && t.cfg.is_some()
        {
            t.pending_root = root;
        }
    }

    fn bma_reset(&self, new_root: u64, clear_entries: bool) -> Result<(), GpuError> {
        let (addr, cfg, paths, entries) = {
            let g = self.bma.lock().unwrap_or_else(|e| e.into_inner());
            let Some(t) = g.as_ref() else { return Ok(()) };
            let Some(cfg) = t.cfg else { return Ok(()) };
            (t.address(), cfg, t.path_dwords(), t.entries())
        };
        let words = spirv_words(strata_runtime_shaders::BMA_MAPPING_SPV);
        let flags = strata_shader::bma::FLAG_ENABLED
            | if cfg.primary {
                strata_shader::bma::FLAG_PRIMARY
            } else {
                0
            };
        if clear_entries {
            // Mode 0 over a zero sample generation maps nothing but borders,
            // and with empty paths every entry becomes 0.
            let pc = mapping_push(addr, 0, &cfg, 1, flags, paths);
            self.dispatch_spirv(&words, &pc, [paths.div_ceil(64).max(1), 1, 1])?;
            let pc = mapping_push(addr, 0, &cfg, 0, flags, entries);
            self.dispatch_spirv(&words, &pc, [entries.div_ceil(64).max(1), 1, 1])?;
        }
        let pc = mapping_push(addr, new_root, &cfg, 1, flags, paths);
        self.dispatch_spirv(&words, &pc, [paths.div_ceil(64).max(1), 1, 1])
    }

    /// The frame boundary: maps this frame's sampled paths into shortcuts
    /// for the next frame, clears the paths and applies a pending root.
    /// A no-op with BMA off.
    pub fn bma_end_frame(&self) -> Result<(), GpuError> {
        let (addr, cfg, entries, root) = {
            let mut g = self.bma.lock().unwrap_or_else(|e| e.into_inner());
            let Some(t) = g.as_mut() else { return Ok(()) };
            let Some(cfg) = t.cfg else { return Ok(()) };
            t.frames += 1;
            let root = std::mem::take(&mut t.pending_root);
            (t.address(), cfg, t.entries(), root)
        };
        let words = spirv_words(strata_runtime_shaders::BMA_MAPPING_SPV);
        let pc = mapping_push(addr, 0, &cfg, 0, 0, entries);
        self.dispatch_spirv(&words, &pc, [entries.div_ceil(64).max(1), 1, 1])?;
        self.bma_reset(root, false)
    }

    /// Frees the table (device teardown).
    pub(crate) fn bma_destroy(&self) {
        if let Some(t) = self.bma.lock().unwrap_or_else(|e| e.into_inner()).take() {
            self.free_bda_buffer(t.buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_push_layout_matches_the_shader() {
        let cfg = BmaConfig::default();
        let p = mapping_push(0x1122, 0x3344, &cfg, 1, 3, 9);
        // u64 table, u64 root, f32 threshold, u32 mode, u32 flags, u32 count, f32[9]
        assert_eq!(p.len(), 8 + 8 + 4 + 4 + 4 + 4 + 36);
        assert_eq!(&p[0..8], &0x1122u64.to_le_bytes());
        assert_eq!(&p[20..24], &1u32.to_le_bytes());
        assert_eq!(&p[28..32], &9u32.to_le_bytes());
    }

    #[test]
    fn detects_instrumented_spirv() {
        let lo = strata_shader::bma::SPEC_ID_TABLE_LO;
        assert!(uses_bma_table(&[0x0723_0203, (4 << 16) | 71, 7, 1, lo]));
        assert!(!uses_bma_table(&[
            0x0723_0203,
            (4 << 16) | 71,
            7,
            1,
            lo + 9
        ]));
    }
}
