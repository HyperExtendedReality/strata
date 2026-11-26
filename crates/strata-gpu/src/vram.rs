//! 64 KiB Sparse Tile Pool for Console-Aliased Render Targets, Depth & UAVs.
//!
//! Xbox Series X|S titles place transient render targets and UAV textures at
//! overlapping console virtual addresses and expect them to alias the same
//! physical memory. Backed by Vulkan sparse residency (`vkQueueBindSparse`),
//! each 64 KiB console page maps to exactly one 64 KiB VRAM tile shared by
//! every target covering that address—keeping VRAM ~3 GB instead of 15+ GB.

use std::collections::HashMap;

pub const TILE_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId(pub u32);

#[derive(Debug)]
struct TileEntry {
    tile: TileId,
    ref_count: u32,
}

/// Pure-logic 64 KiB page-to-VRAM-tile allocator (decoupled from Vulkan handles for unit testing).
#[derive(Debug, Default)]
pub struct SparseTilePool {
    next_tile: u32,
    free_tiles: Vec<TileId>,
    /// Console 64 KiB page index (`console_va >> 16`) -> active VRAM tile.
    mapped: HashMap<u64, TileEntry>,
}

impl SparseTilePool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Acquire (or alias) all 64 KiB tiles covering `[console_va, console_va + size_bytes)`.
    pub fn acquire_range(&mut self, console_va: u64, size_bytes: u64) -> Vec<(u64, TileId)> {
        let first_page = console_va / TILE_BYTES;
        let last_page = (console_va + size_bytes.max(1) - 1) / TILE_BYTES;
        let mut out = Vec::with_capacity((last_page - first_page + 1) as usize);

        for page in first_page..=last_page {
            let entry = self.mapped.entry(page).or_insert_with(|| {
                let tile = self.free_tiles.pop().unwrap_or_else(|| {
                    let id = TileId(self.next_tile);
                    self.next_tile += 1;
                    id
                });
                TileEntry { tile, ref_count: 0 }
            });
            entry.ref_count += 1;
            out.push((page, entry.tile));
        }

        out
    }

    /// Release a target's references to `[console_va, console_va + size_bytes)`.
    /// Tiles whose refcount hits 0 return to the free pool for reuse.
    pub fn release_range(&mut self, console_va: u64, size_bytes: u64) {
        let first_page = console_va / TILE_BYTES;
        let last_page = (console_va + size_bytes.max(1) - 1) / TILE_BYTES;

        for page in first_page..=last_page {
            if let Some(entry) = self.mapped.get_mut(&page) {
                entry.ref_count = entry.ref_count.saturating_sub(1);
                if entry.ref_count == 0 {
                    let tile = entry.tile;
                    self.mapped.remove(&page);
                    self.free_tiles.push(tile);
                }
            }
        }
    }

    pub fn active_tile_count(&self) -> usize {
        self.mapped.len()
    }

    pub fn active_vram_bytes(&self) -> u64 {
        self.mapped.len() as u64 * TILE_BYTES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliased_targets_share_exact_same_tiles() {
        let mut pool = SparseTilePool::new();
        let a = pool.acquire_range(0x1000_0000, 256 * 1024);
        let b = pool.acquire_range(0x1002_0000, 256 * 1024);

        // Pages 2 & 3 overlap -> 6 unique tiles total, not 8.
        assert_eq!(pool.active_tile_count(), 6);
        assert_eq!(a[2].1, b[0].1);
        assert_eq!(a[3].1, b[1].1);

        pool.release_range(0x1000_0000, 256 * 1024);
        assert_eq!(pool.active_tile_count(), 4);

        pool.release_range(0x1002_0000, 256 * 1024);
        assert_eq!(pool.active_tile_count(), 0);
    }
}
