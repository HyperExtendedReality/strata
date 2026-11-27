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

    pub fn is_resident(&self, id: u64) -> bool {
        self.objects.get(&id).is_some_and(|o| o.residency_count > 0)
    }

    pub fn over_budget(&self) -> bool {
        self.resident_bytes > self.budget_bytes
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
