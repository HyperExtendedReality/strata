//! Console Virtual Address -> Host 64-bit BDA (`VkDeviceAddress`) mapper
//! and per-page epoch hazard tracker for pipelined async submissions.

use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackingKind {
    /// Zero-copy host memory (`VK_EXT_external_memory_host`).
    ZeroCopyHost,
    /// GPU-resident VRAM mirror (updated via async copy queue / dirty hash).
    VramMirror,
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryRegion {
    pub console_base: u64,
    pub size: u64,
    pub host_bda: u64,
    pub kind: BackingKind,
    pub last_gpu_write_epoch: u64,
}

/// Maps contiguous console memory ranges to real 64-bit Vulkan BDA addresses,
/// and tracks aliased virtual views sharing the same physical console page
/// so write guards open/close all views atomically (Belobug / VIctory-7900XTX fix).
#[derive(Debug, Default)]
pub struct AddressSpace {
    regions: BTreeMap<u64, MemoryRegion>,
    /// Physical console page number -> all mapped virtual addresses aliasing that page.
    phys_views: BTreeMap<u64, Vec<u64>>,
}

impl AddressSpace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, region: MemoryRegion) {
        self.regions.insert(region.console_base, region);
    }

    /// Record that virtual address `view_va` maps console physical page `phys_page`.
    pub fn map_physical_view(&mut self, phys_page: u64, view_va: u64) {
        let views = self.phys_views.entry(phys_page).or_default();
        if !views.contains(&view_va) {
            views.push(view_va);
        }
    }

    /// Remove virtual address `view_va` from every physical page mapping.
    pub fn unmap_physical_view(&mut self, view_va: u64) -> bool {
        let mut removed = false;
        self.phys_views.retain(|_, views| {
            let before = views.len();
            views.retain(|&v| v != view_va);
            if views.len() != before {
                removed = true;
            }
            !views.is_empty()
        });
        removed
    }

    /// Return every virtual address aliasing `phys_page` so guards apply to all views atomically.
    pub fn views_of_physical_page(&self, phys_page: u64) -> &[u64] {
        self.phys_views.get(&phys_page).map_or(&[], Vec::as_slice)
    }

    /// Translate a console GPU/UMA address to its host 64-bit `VkDeviceAddress`.
    pub fn translate(&self, console_va: u64) -> Option<u64> {
        let (&base, reg) = self.regions.range(..=console_va).next_back()?;
        let off = console_va - base;
        if off < reg.size {
            Some(reg.host_bda + off)
        } else {
            None
        }
    }

    /// Record that submission `epoch` may write to `console_va`.
    pub fn mark_gpu_write(&mut self, console_va: u64, epoch: u64) {
        if let Some((&base, reg)) = self.regions.range_mut(..=console_va).next_back() {
            if console_va - base < reg.size {
                reg.last_gpu_write_epoch = reg.last_gpu_write_epoch.max(epoch);
            }
        }
    }

    /// Returns the fence epoch the CPU must wait on before reading `console_va`
    /// (0 if no pending GPU write hazard exists).
    pub fn cpu_read_hazard_epoch(&self, console_va: u64, completed_epoch: u64) -> u64 {
        if let Some((&base, reg)) = self.regions.range(..=console_va).next_back() {
            if console_va - base < reg.size && reg.last_gpu_write_epoch > completed_epoch {
                return reg.last_gpu_write_epoch;
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_console_va_to_host_bda_and_tracks_hazards() {
        let mut space = AddressSpace::new();
        space.register(MemoryRegion {
            console_base: 0x0000_0400_0000_0000,
            size: 16 * 1024 * 1024,
            host_bda: 0x0000_7F00_0000_0000,
            kind: BackingKind::ZeroCopyHost,
            last_gpu_write_epoch: 0,
        });

        assert_eq!(
            space.translate(0x0000_0400_0000_1234),
            Some(0x0000_7F00_0000_1234)
        );
        assert_eq!(space.translate(0x0000_0500_0000_0000), None);

        space.mark_gpu_write(0x0000_0400_0000_1234, 42);
        assert_eq!(space.cpu_read_hazard_epoch(0x0000_0400_0000_1234, 40), 42);
        assert_eq!(space.cpu_read_hazard_epoch(0x0000_0400_0000_1234, 42), 0);
    }
}
