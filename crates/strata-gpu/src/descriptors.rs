//! No-API Root Signatures, Descriptor Heaps, and Samplers.
//!
//! Translates D3D12.X root signatures, descriptor heaps (CBV/SRV/UAV, Sampler,
//! RTV, DSV), and sampler descriptors directly into Strata's 128-byte User SGPR
//! push-constant layout and Vulkan 1.3+ bindless / BDA tables.

use ash::vk;
use std::collections::HashMap;

/// Maximum User SGPR dwords packed into Strata's single global push-constant block (128 B).
pub const MAX_ROOT_DWORDS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootParamKind {
    Constants32Bit {
        num_values: u32,
        shader_register: u32,
    },
    Cbv {
        shader_register: u32,
    },
    Srv {
        shader_register: u32,
    },
    Uav {
        shader_register: u32,
    },
    DescriptorTable {
        range_count: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    Point,
    Linear,
    Anisotropic { max_anisotropy: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressMode {
    Wrap,
    Mirror,
    Clamp,
    Border,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerDesc {
    pub filter: FilterMode,
    pub address_u: AddressMode,
    pub address_v: AddressMode,
    pub address_w: AddressMode,
    pub mip_lod_bias: f32,
    pub min_lod: f32,
    pub max_lod: f32,
    pub compare_enable: bool,
}

impl Default for SamplerDesc {
    fn default() -> Self {
        Self {
            filter: FilterMode::Linear,
            address_u: AddressMode::Wrap,
            address_v: AddressMode::Wrap,
            address_w: AddressMode::Wrap,
            mip_lod_bias: 0.0,
            min_lod: 0.0,
            max_lod: 16.0,
            compare_enable: false,
        }
    }
}

impl SamplerDesc {
    /// Convert to a raw Vulkan `SamplerCreateInfo` for the No-API sampler table.
    pub fn to_vk_create_info(&self) -> vk::SamplerCreateInfo<'static> {
        let (min_f, mag_f, mip_f, aniso_en, max_aniso) = match self.filter {
            FilterMode::Point => (
                vk::Filter::NEAREST,
                vk::Filter::NEAREST,
                vk::SamplerMipmapMode::NEAREST,
                false,
                1.0,
            ),
            FilterMode::Linear => (
                vk::Filter::LINEAR,
                vk::Filter::LINEAR,
                vk::SamplerMipmapMode::LINEAR,
                false,
                1.0,
            ),
            FilterMode::Anisotropic { max_anisotropy } => (
                vk::Filter::LINEAR,
                vk::Filter::LINEAR,
                vk::SamplerMipmapMode::LINEAR,
                true,
                max_anisotropy.clamp(1, 16) as f32,
            ),
        };
        let addr = |m: AddressMode| match m {
            AddressMode::Wrap => vk::SamplerAddressMode::REPEAT,
            AddressMode::Mirror => vk::SamplerAddressMode::MIRRORED_REPEAT,
            AddressMode::Clamp => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            AddressMode::Border => vk::SamplerAddressMode::CLAMP_TO_BORDER,
        };
        vk::SamplerCreateInfo::default()
            .mag_filter(mag_f)
            .min_filter(min_f)
            .mipmap_mode(mip_f)
            .address_mode_u(addr(self.address_u))
            .address_mode_v(addr(self.address_v))
            .address_mode_w(addr(self.address_w))
            .mip_lod_bias(self.mip_lod_bias)
            .anisotropy_enable(aniso_en)
            .max_anisotropy(max_aniso)
            .compare_enable(self.compare_enable)
            .compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .min_lod(self.min_lod)
            .max_lod(self.max_lod)
    }
}

/// Root signature mapping D3D12 root parameters onto contiguous User SGPR slots.
#[derive(Debug, Clone)]
pub struct RootSignature {
    pub id: u64,
    pub params: Vec<RootParamKind>,
    pub param_sgpr_offsets: Vec<usize>,
    pub total_dwords: usize,
    pub static_samplers: Vec<SamplerDesc>,
}

impl RootSignature {
    pub fn build(id: u64, params: Vec<RootParamKind>, static_samplers: Vec<SamplerDesc>) -> Self {
        let mut offsets = Vec::with_capacity(params.len());
        let mut cursor = 0usize;
        for &p in &params {
            offsets.push(cursor);
            let width = match p {
                RootParamKind::Constants32Bit { num_values, .. } => num_values as usize,
                RootParamKind::Cbv { .. }
                | RootParamKind::Srv { .. }
                | RootParamKind::Uav { .. } => 2, // 64-bit BDA
                RootParamKind::DescriptorTable { .. } => 2, // 64-bit table base address
            };
            cursor = (cursor + width).min(MAX_ROOT_DWORDS);
        }
        Self {
            id,
            params,
            param_sgpr_offsets: offsets,
            total_dwords: cursor,
            static_samplers,
        }
    }

    /// Pack a 64-bit BDA pointer or descriptor table base into the User SGPR push-constant array.
    pub fn write_u64_param(
        &self,
        param_index: usize,
        value: u64,
        user_sgprs: &mut [u32; MAX_ROOT_DWORDS],
    ) {
        if let Some(&off) = self.param_sgpr_offsets.get(param_index)
            && off + 1 < MAX_ROOT_DWORDS
        {
            user_sgprs[off] = value as u32;
            user_sgprs[off + 1] = (value >> 32) as u32;
        }
    }
}

/// Descriptor heap type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DescriptorHeapKind {
    CbvSrvUav,
    Sampler,
    Rtv,
    Dsv,
}

/// One stored descriptor slot (32 bytes = 8 dwords, matching RDNA2 T#/V#/S# descriptors).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RawDescriptor {
    pub words: [u32; 8],
}

/// Virtual descriptor heap exposing contiguous CPU and GPU virtual addresses.
#[derive(Debug, Clone)]
pub struct DescriptorHeap {
    pub id: u64,
    pub kind: DescriptorHeapKind,
    pub base_va: u64,
    pub stride: u64,
    pub slots: Vec<RawDescriptor>,
}

impl DescriptorHeap {
    pub fn new(id: u64, kind: DescriptorHeapKind, base_va: u64, capacity: usize) -> Self {
        Self {
            id,
            kind,
            base_va,
            stride: 32,
            slots: vec![RawDescriptor::default(); capacity],
        }
    }

    pub fn handle_at(&self, index: usize) -> Option<u64> {
        (index < self.slots.len()).then_some(self.base_va + (index as u64) * self.stride)
    }

    pub fn index_of(&self, handle: u64) -> Option<usize> {
        let off = handle.checked_sub(self.base_va)?;
        if !off.is_multiple_of(self.stride) {
            return None;
        }
        let idx = (off / self.stride) as usize;
        (idx < self.slots.len()).then_some(idx)
    }

    pub fn write(&mut self, index: usize, desc: RawDescriptor) -> bool {
        if let Some(slot) = self.slots.get_mut(index) {
            *slot = desc;
            true
        } else {
            false
        }
    }

    pub fn read(&self, index: usize) -> Option<RawDescriptor> {
        self.slots.get(index).copied()
    }
}

/// Registry of active descriptor heaps, root signatures, and sampler descriptors.
#[derive(Debug, Default)]
pub struct DescriptorState {
    next_id: u64,
    next_va: u64,
    pub root_signatures: HashMap<u64, RootSignature>,
    pub heaps: HashMap<u64, DescriptorHeap>,
    pub samplers: HashMap<u64, SamplerDesc>,
}

impl DescriptorState {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            next_va: 0x0001_0000_0000,
            ..Default::default()
        }
    }

    pub fn create_root_signature(
        &mut self,
        params: Vec<RootParamKind>,
        static_samplers: Vec<SamplerDesc>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.root_signatures
            .insert(id, RootSignature::build(id, params, static_samplers));
        id
    }

    pub fn create_heap(&mut self, kind: DescriptorHeapKind, capacity: usize) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let va = self.next_va;
        self.next_va += ((capacity.max(1) as u64) * 32).next_multiple_of(65536);
        self.heaps
            .insert(id, DescriptorHeap::new(id, kind, va, capacity));
        id
    }

    pub fn create_sampler(&mut self, desc: SamplerDesc) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.samplers.insert(id, desc);
        id
    }

    /// Copy descriptors between heaps (`CopyDescriptorsSimple`).
    pub fn copy_descriptors(
        &mut self,
        src_heap: u64,
        src_idx: usize,
        dst_heap: u64,
        dst_idx: usize,
        count: usize,
    ) -> usize {
        let Some(src_slice) = self
            .heaps
            .get(&src_heap)
            .and_then(|h| h.slots.get(src_idx..src_idx + count))
            .map(|s| s.to_vec())
        else {
            return 0;
        };
        let Some(dst) = self.heaps.get_mut(&dst_heap) else {
            return 0;
        };
        let mut copied = 0;
        for (i, desc) in src_slice.into_iter().enumerate() {
            if dst.write(dst_idx + i, desc) {
                copied += 1;
            }
        }
        copied
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_signature_and_descriptor_heaps_pack_and_copy() {
        let mut ds = DescriptorState::new();
        let rs_id = ds.create_root_signature(
            vec![
                RootParamKind::Constants32Bit {
                    num_values: 4,
                    shader_register: 0,
                },
                RootParamKind::Cbv { shader_register: 1 },
            ],
            vec![SamplerDesc::default()],
        );
        let mut sgprs = [0u32; MAX_ROOT_DWORDS];
        ds.root_signatures[&rs_id].write_u64_param(1, 0xDEAD_BEEF_1234_5678, &mut sgprs);
        assert_eq!(sgprs[4], 0x1234_5678);
        assert_eq!(sgprs[5], 0xDEAD_BEEF);

        let h1 = ds.create_heap(DescriptorHeapKind::CbvSrvUav, 64);
        let h2 = ds.create_heap(DescriptorHeapKind::CbvSrvUav, 64);
        let raw = RawDescriptor {
            words: [1, 2, 3, 4, 5, 6, 7, 8],
        };
        ds.heaps.get_mut(&h1).unwrap().write(3, raw);
        assert_eq!(ds.copy_descriptors(h1, 3, h2, 10, 1), 1);
        assert_eq!(ds.heaps[&h2].read(10), Some(raw));
    }
}
