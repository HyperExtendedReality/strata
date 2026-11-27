//! `ExecuteIndirect` command signatures, dedicated Copy/Upload staging ring
//! queue, and persistent `PipelineLibrary` cache.

use std::collections::{HashMap, VecDeque};
use std::path::Path;

// ============================================================================
// ExecuteIndirect Command Signature
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndirectArgKind {
    Draw,
    DrawIndexed,
    Dispatch,
    VertexBufferView {
        slot: u32,
    },
    IndexBufferView,
    Constant {
        root_param_idx: u32,
        dest_offset_dwords: u32,
        num_dwords: u32,
    },
    ConstantBufferView {
        root_param_idx: u32,
    },
    ShaderResourceView {
        root_param_idx: u32,
    },
    UnorderedAccessView {
        root_param_idx: u32,
    },
}

#[derive(Debug, Clone)]
pub struct CommandSignature {
    pub id: u64,
    pub byte_stride: u32,
    pub args: Vec<IndirectArgKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodedIndirectCommand {
    Draw {
        vertex_count: u32,
        instance_count: u32,
        start_vertex: u32,
        start_instance: u32,
    },
    DrawIndexed {
        index_count: u32,
        instance_count: u32,
        start_index: u32,
        base_vertex: i32,
        start_instance: u32,
    },
    Dispatch {
        x: u32,
        y: u32,
        z: u32,
    },
}

impl CommandSignature {
    /// Decode up to `max_count` indirect commands from title argument buffer bytes.
    pub fn decode_buffer(&self, arg_bytes: &[u8], max_count: usize) -> Vec<DecodedIndirectCommand> {
        let stride = self.byte_stride as usize;
        if stride == 0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        for i in 0..max_count {
            let base = i * stride;
            if base + stride > arg_bytes.len() {
                break;
            }
            let slice = &arg_bytes[base..base + stride];
            let mut off = 0usize;
            let read_u32 = |o: &mut usize| -> u32 {
                let v = slice
                    .get(*o..*o + 4)
                    .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                *o += 4;
                v
            };
            for &arg in &self.args {
                match arg {
                    IndirectArgKind::Draw => {
                        out.push(DecodedIndirectCommand::Draw {
                            vertex_count: read_u32(&mut off),
                            instance_count: read_u32(&mut off),
                            start_vertex: read_u32(&mut off),
                            start_instance: read_u32(&mut off),
                        });
                    }
                    IndirectArgKind::DrawIndexed => {
                        out.push(DecodedIndirectCommand::DrawIndexed {
                            index_count: read_u32(&mut off),
                            instance_count: read_u32(&mut off),
                            start_index: read_u32(&mut off),
                            base_vertex: read_u32(&mut off) as i32,
                            start_instance: read_u32(&mut off),
                        });
                    }
                    IndirectArgKind::Dispatch => {
                        out.push(DecodedIndirectCommand::Dispatch {
                            x: read_u32(&mut off),
                            y: read_u32(&mut off),
                            z: read_u32(&mut off),
                        });
                    }
                    IndirectArgKind::VertexBufferView { .. } | IndirectArgKind::IndexBufferView => {
                        off += 16
                    }
                    IndirectArgKind::Constant { num_dwords, .. } => {
                        off += (num_dwords as usize) * 4
                    }
                    IndirectArgKind::ConstantBufferView { .. }
                    | IndirectArgKind::ShaderResourceView { .. }
                    | IndirectArgKind::UnorderedAccessView { .. } => off += 8,
                }
            }
        }
        out
    }
}

// ============================================================================
// Dedicated Copy / Upload Queue & Staging Ring
// ============================================================================

#[derive(Debug, Clone)]
pub struct UploadJob {
    pub dst_va: u64,
    pub data: Vec<u8>,
    pub fence_value: u64,
}

#[derive(Debug, Default)]
pub struct UploadQueue {
    next_fence: u64,
    pub completed_fence: u64,
    pending: VecDeque<UploadJob>,
    pub total_bytes_uploaded: u64,
}

impl UploadQueue {
    pub fn enqueue_copy(&mut self, dst_va: u64, data: Vec<u8>) -> u64 {
        self.next_fence += 1;
        let fence = self.next_fence;
        self.pending.push_back(UploadJob {
            dst_va,
            data,
            fence_value: fence,
        });
        fence
    }

    /// Execute queued host-to-UMA/BDA copies and advance `completed_fence`.
    pub fn flush(&mut self, mut sink: impl FnMut(u64, &[u8])) -> u64 {
        while let Some(job) = self.pending.pop_front() {
            self.total_bytes_uploaded += job.data.len() as u64;
            sink(job.dst_va, &job.data);
            self.completed_fence = job.fence_value;
        }
        self.completed_fence
    }
}

// ============================================================================
// Persistent Pipeline Library Cache (`ID3D12PipelineLibrary`)
// ============================================================================

const LIB_MAGIC: &[u8; 4] = b"SPLB";

#[derive(Debug, Clone, Default)]
pub struct PipelineLibrary {
    entries: HashMap<String, Vec<u8>>,
}

impl PipelineLibrary {
    pub fn store(&mut self, name: impl Into<String>, blob: Vec<u8>) {
        self.entries.insert(name.into(), blob);
    }

    pub fn load(&self, name: &str) -> Option<&[u8]> {
        self.entries.get(name).map(Vec::as_slice)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(LIB_MAGIC);
        out.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for (name, blob) in &self.entries {
            let nb = name.as_bytes();
            out.extend_from_slice(&(nb.len() as u32).to_le_bytes());
            out.extend_from_slice(nb);
            out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
            out.extend_from_slice(blob);
        }
        out
    }

    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 8 || &bytes[0..4] != LIB_MAGIC {
            return None;
        }
        let count = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
        let mut cur = 8usize;
        let mut entries = HashMap::with_capacity(count);
        for _ in 0..count {
            let nlen = u32::from_le_bytes(bytes.get(cur..cur + 4)?.try_into().ok()?) as usize;
            cur += 4;
            let name = std::str::from_utf8(bytes.get(cur..cur + nlen)?)
                .ok()?
                .to_string();
            cur += nlen;
            let blen = u32::from_le_bytes(bytes.get(cur..cur + 4)?.try_into().ok()?) as usize;
            cur += 4;
            let blob = bytes.get(cur..cur + blen)?.to_vec();
            cur += blen;
            entries.insert(name, blob);
        }
        Some(Self { entries })
    }

    pub fn save_atomic(&self, path: &Path) -> std::io::Result<()> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, self.serialize())?;
        std::fs::rename(tmp, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_signature_decodes_indirect_dispatches() {
        let sig = CommandSignature {
            id: 1,
            byte_stride: 12,
            args: vec![IndirectArgKind::Dispatch],
        };
        let mut buf = Vec::new();
        for w in [8u32, 4, 2, 16, 1, 1] {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        let cmds = sig.decode_buffer(&buf, 4);
        assert_eq!(
            cmds,
            vec![
                DecodedIndirectCommand::Dispatch { x: 8, y: 4, z: 2 },
                DecodedIndirectCommand::Dispatch { x: 16, y: 1, z: 1 },
            ]
        );
    }

    #[test]
    fn pipeline_library_serializes_and_roundtrips() {
        let mut lib = PipelineLibrary::default();
        lib.store("pso_main", vec![1, 2, 3, 4]);
        let bytes = lib.serialize();
        let loaded = PipelineLibrary::deserialize(&bytes).unwrap();
        assert_eq!(loaded.load("pso_main"), Some(&[1, 2, 3, 4][..]));
    }
}
