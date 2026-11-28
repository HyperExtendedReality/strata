//! Sparse tile mapping, GPU memory residency management, and hardware query
//! heaps (occlusion, binary occlusion, timestamp, pipeline statistics).

use std::collections::HashMap;

// ============================================================================
// Sparse Tile Mapping (`UpdateTileMappings` / `CopyTiles`)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    pub resource_id: u64,
    pub subresource: u32,
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileBinding {
    pub heap_id: u64,
    pub heap_tile_offset: u32,
}

#[derive(Debug, Default)]
pub struct SparseTileTable {
    pub bindings: HashMap<TileCoord, TileBinding>,
    pub update_count: u64,
}

impl SparseTileTable {
    pub fn update_tile(&mut self, coord: TileCoord, binding: Option<TileBinding>) {
        match binding {
            Some(b) => {
                self.bindings.insert(coord, b);
            }
            None => {
                self.bindings.remove(&coord);
            }
        }
        self.update_count += 1;
    }

    pub fn lookup(&self, coord: TileCoord) -> Option<TileBinding> {
        self.bindings.get(&coord).copied()
    }
}

// ============================================================================
// Residency Manager (`MakeResident` / `Evict` / Budget)
// ============================================================================

#[derive(Debug, Clone)]
pub struct ResidentObject {
    pub id: u64,
    pub size_bytes: u64,
    pub residency_count: u32,
    pub last_used_frame: u64,
}

#[derive(Debug)]
pub struct ResidencyManager {
    pub budget_bytes: u64,
    pub resident_bytes: u64,
    objects: HashMap<u64, ResidentObject>,
}

impl ResidencyManager {
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            resident_bytes: 0,
            objects: HashMap::new(),
        }
    }

    pub fn register(&mut self, id: u64, size_bytes: u64, frame: u64) {
        let entry = self.objects.entry(id).or_insert(ResidentObject {
            id,
            size_bytes,
            residency_count: 1,
            last_used_frame: frame,
        });
        if entry.residency_count == 1 {
            self.resident_bytes = self.resident_bytes.saturating_add(size_bytes);
        }
    }

    pub fn make_resident(&mut self, id: u64, frame: u64) -> bool {
        let Some(obj) = self.objects.get_mut(&id) else {
            return false;
        };
        if obj.residency_count == 0 {
            self.resident_bytes = self.resident_bytes.saturating_add(obj.size_bytes);
        }
        obj.residency_count += 1;
        obj.last_used_frame = frame;
        true
    }

    pub fn evict(&mut self, id: u64) -> bool {
        let Some(obj) = self.objects.get_mut(&id) else {
            return false;
        };
        if obj.residency_count > 0 {
            obj.residency_count -= 1;
            if obj.residency_count == 0 {
                self.resident_bytes = self.resident_bytes.saturating_sub(obj.size_bytes);
            }
            true
        } else {
            false
        }
    }

    /// Whether `id` was ever registered.
    pub fn is_known(&self, id: u64) -> bool {
        self.objects.contains_key(&id)
    }

    pub fn is_resident(&self, id: u64) -> bool {
        self.objects.get(&id).is_some_and(|o| o.residency_count > 0)
    }

    pub fn over_budget(&self) -> bool {
        self.resident_bytes > self.budget_bytes
    }
}

/// Host memory pressure tier modeled on `VIctory-Fast`'s commit-headroom rules:
/// distinguishes mild pressure (keep prefetch + 1 spare block alive) from
/// critical commit starvation (`< 1536 MiB` free commit or upload refusal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureTier {
    Normal,
    Mild,
    Critical,
}

/// Current Windows system commit headroom, in bytes.
pub fn available_commit_bytes() -> Option<u64> {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct PerformanceInformation {
            size: u32,
            commit_total: usize,
            commit_limit: usize,
            commit_peak: usize,
            physical_total: usize,
            physical_available: usize,
            system_cache: usize,
            kernel_total: usize,
            kernel_paged: usize,
            kernel_nonpaged: usize,
            page_size: usize,
            handle_count: u32,
            process_count: u32,
            thread_count: u32,
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn K32GetPerformanceInfo(info: *mut PerformanceInformation, size: u32) -> i32;
        }
        let mut info = std::mem::MaybeUninit::<PerformanceInformation>::zeroed();
        let size = std::mem::size_of::<PerformanceInformation>() as u32;
        if unsafe { K32GetPerformanceInfo(info.as_mut_ptr(), size) } != 0 {
            let info = unsafe { info.assume_init() };
            return Some(
                (info.commit_limit.saturating_sub(info.commit_total) as u64)
                    .saturating_mul(info.page_size as u64),
            );
        }
    }
    None
}

impl PressureTier {
    pub const CRITICAL_COMMIT_BYTES: u64 = 1536 * 1024 * 1024;
    pub const MILD_COMMIT_BYTES: u64 = 4096 * 1024 * 1024;

    pub fn from_available_commit(avail_commit_bytes: u64, upload_refused: bool) -> Self {
        if upload_refused || avail_commit_bytes < Self::CRITICAL_COMMIT_BYTES {
            Self::Critical
        } else if avail_commit_bytes < Self::MILD_COMMIT_BYTES {
            Self::Mild
        } else {
            Self::Normal
        }
    }

    /// Per-frame speculative staging admission cap (bytes). Required bake
    /// uploads bypass this cap (`None` = unlimited).
    pub fn speculative_staging_budget_bytes(self, is_required_bake: bool) -> Option<u64> {
        if is_required_bake {
            return None;
        }
        match self {
            Self::Normal => Some(64 * 1024 * 1024),
            Self::Mild => Some(32 * 1024 * 1024),
            Self::Critical => Some(16 * 1024 * 1024),
        }
    }
}

/// 500 ms pool-growth backoff after an allocation failure (`E_OUTOFMEMORY`),
/// preventing bursts of failing 64 MiB pool allocations while still allowing
/// reuse of already-allocated pool space.
#[derive(Debug)]
pub struct OomBackoff {
    cooldown: std::time::Duration,
    last_failure: Option<std::time::Instant>,
}

impl Default for OomBackoff {
    fn default() -> Self {
        Self::new(std::time::Duration::from_millis(500))
    }
}

impl OomBackoff {
    pub fn new(cooldown: std::time::Duration) -> Self {
        Self {
            cooldown,
            last_failure: None,
        }
    }

    pub fn record_failure(&mut self) {
        self.last_failure = Some(std::time::Instant::now());
    }

    pub fn can_grow(&self) -> bool {
        self.last_failure
            .is_none_or(|t| t.elapsed() >= self.cooldown)
    }
}

// ============================================================================
// Query Heaps (Occlusion, Timestamp, Pipeline Statistics)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryHeapKind {
    Occlusion,
    BinaryOcclusion,
    Timestamp,
    PipelineStatistics,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PipelineStats {
    pub ia_vertices: u64,
    pub ia_primitives: u64,
    pub vs_invocations: u64,
    pub gs_invocations: u64,
    pub gs_primitives: u64,
    pub ps_invocations: u64,
    pub cs_invocations: u64,
}

#[derive(Debug, Clone)]
pub struct QueryHeap {
    pub id: u64,
    pub kind: QueryHeapKind,
    pub values: Vec<u64>,
    pub stats: Vec<PipelineStats>,
    pub active: Vec<bool>,
}

impl QueryHeap {
    pub fn new(id: u64, kind: QueryHeapKind, count: usize) -> Self {
        Self {
            id,
            kind,
            values: vec![0; count],
            stats: if kind == QueryHeapKind::PipelineStatistics {
                vec![PipelineStats::default(); count]
            } else {
                Vec::new()
            },
            active: vec![false; count],
        }
    }

    pub fn begin(&mut self, index: usize) {
        if let Some(a) = self.active.get_mut(index) {
            *a = true;
        }
    }

    pub fn end(&mut self, index: usize, value: u64) {
        if let Some(a) = self.active.get_mut(index) {
            *a = false;
        }
        if let Some(v) = self.values.get_mut(index) {
            *v = match self.kind {
                QueryHeapKind::BinaryOcclusion => u64::from(value > 0),
                _ => value,
            };
        }
    }

    /// Resolve `[start..start+count]` query results into little-endian bytes for title readback.
    pub fn resolve_bytes(&self, start: usize, count: usize) -> Vec<u8> {
        let end = (start + count).min(self.values.len());
        let mut out = Vec::with_capacity((end.saturating_sub(start)) * 8);
        for &v in &self.values[start..end] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_tiles_residency_and_queries_work() {
        let mut tiles = SparseTileTable::default();
        let coord = TileCoord {
            resource_id: 1,
            subresource: 0,
            x: 2,
            y: 3,
            z: 0,
        };
        tiles.update_tile(
            coord,
            Some(TileBinding {
                heap_id: 9,
                heap_tile_offset: 12,
            }),
        );
        assert_eq!(tiles.lookup(coord).unwrap().heap_tile_offset, 12);

        let mut res = ResidencyManager::new(1024);
        res.register(1, 512, 1);
        assert!(res.is_resident(1));
        assert!(!res.over_budget());
        res.evict(1);
        assert!(!res.is_resident(1));

        let mut qh = QueryHeap::new(1, QueryHeapKind::Timestamp, 8);
        qh.end(2, 123_456);
        let resolved = qh.resolve_bytes(2, 1);
        assert_eq!(u64::from_le_bytes(resolved.try_into().unwrap()), 123_456);
    }
}
