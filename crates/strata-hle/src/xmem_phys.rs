//! Console `xmem.dll` 64 KiB Physical Page Pool & Scatter Mapper.
//!
//! Emulates `XMemAllocatePhysicalPages`, `XMemFreePhysicalPages`,
//! `XMemMapPhysicalPages`, and `XMemMapPhysicalPagesScatter` by tracking 64 KiB
//! physical page frames and registering every aliased virtual view inside
//! `strata_gpu::memory::AddressSpace`.

use std::collections::HashMap;

use strata_gpu::memory::AddressSpace;

pub const XMEM_PAGE_BYTES: usize = 64 * 1024;

#[derive(Debug, Default)]
pub struct PhysicalPagePool {
    next_pfn: u64,
    free_pfns: Vec<u64>,
    /// Physical Frame Number (PFN) -> allocated backing size in pages (always 1 per 64 KiB frame).
    allocated: HashMap<u64, bool>,
}

impl PhysicalPagePool {
    pub fn new() -> Self {
        Self {
            next_pfn: 1,
            free_pfns: Vec::new(),
            allocated: HashMap::new(),
        }
    }

    /// Allocate `count` 64 KiB physical page frame numbers (`XMemAllocatePhysicalPages`).
    pub fn alloc_pages(&mut self, count: usize) -> Vec<u64> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let pfn = self.free_pfns.pop().unwrap_or_else(|| {
                let id = self.next_pfn;
                self.next_pfn += 1;
                id
            });
            self.allocated.insert(pfn, true);
            out.push(pfn);
        }
        out
    }

    /// Free physical page frame numbers (`XMemFreePhysicalPages`).
    pub fn free_pages(&mut self, pfns: &[u64]) {
        for &pfn in pfns {
            if self.allocated.remove(&pfn).is_some() {
                self.free_pfns.push(pfn);
            }
        }
    }

    /// Map a list of PFNs into contiguous 64 KiB slots starting at `base_va`
    /// (`XMemMapPhysicalPagesScatter`), recording every aliased view in `AddressSpace`.
    pub fn map_scatter(
        &self,
        address_space: &mut AddressSpace,
        base_va: u64,
        pfns: &[u64],
    ) -> usize {
        let mut mapped = 0usize;
        for (i, &pfn) in pfns.iter().enumerate() {
            if self.allocated.contains_key(&pfn) {
                let va = base_va + (i as u64) * (XMEM_PAGE_BYTES as u64);
                address_space.map_physical_view(pfn, va);
                mapped += 1;
            }
        }
        mapped
    }

    /// Unmap `num_pages` contiguous 64 KiB virtual views starting at `base_va`
    /// (`XMemUnmapPhysicalPages`), returning how many views were removed.
    pub fn unmap_range(
        &self,
        address_space: &mut AddressSpace,
        base_va: u64,
        num_pages: usize,
    ) -> usize {
        let mut unmapped = 0usize;
        for i in 0..num_pages {
            let va = base_va + (i as u64) * (XMEM_PAGE_BYTES as u64);
            if address_space.unmap_physical_view(va) {
                unmapped += 1;
            }
        }
        unmapped
    }

    pub fn active_pages(&self) -> usize {
        self.allocated.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_maps_aliased_views_of_same_physical_page() {
        let mut pool = PhysicalPagePool::new();
        let mut space = AddressSpace::new();
        let pfns = pool.alloc_pages(2);

        assert_eq!(pool.map_scatter(&mut space, 0x1_0000_0000, &pfns), 2);
        assert_eq!(pool.map_scatter(&mut space, 0x2_0000_0000, &pfns), 2);

        // Both virtual views must be tracked under PFN 0 so write guards stay coherent.
        assert_eq!(
            space.views_of_physical_page(pfns[0]),
            &[0x1_0000_0000, 0x2_0000_0000]
        );
        assert_eq!(pool.unmap_range(&mut space, 0x2_0000_0000, 2), 2);
        assert_eq!(space.views_of_physical_page(pfns[0]), &[0x1_0000_0000]);
        pool.free_pages(&pfns);
        assert_eq!(pool.active_pages(), 0);
    }
}
