//! Descriptor tracing: where each resource instruction's descriptor comes from.
//!
//! RDNA2 resources are 4- or 8-dword descriptors (V#, T#, S#) held in SGPRs.
//! Before a resource can be bound on the host, Strata must know which root
//! parameter feeds it: a user SGPR pushed directly, a dword loaded from a
//! table whose address is a user SGPR pair (possibly through more tables), or
//! a per-lane value read with `v_readfirstlane_b32` (a waterfall loop over
//! non-uniform descriptors). This pass follows SGPR writes through the
//! program and names each descriptor's origin.
//!
//! The scan is linear (the last write before an instruction wins); branches
//! that define a descriptor differently on two paths are reported as
//! `Unknown` only if the definitions are computed, since tables are almost
//! always loaded once at the top of a shader.

use std::collections::HashMap;
use std::fmt;

use crate::rdna2::{DecodedInst, Inst, Operand};

/// Where an SGPR's value comes from.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Origin {
    /// User SGPR `n`, pushed by the host (Strata's push constants).
    User(u8),
    /// A dword at `base + offset`, `base` being the 64-bit address the SGPR
    /// pair it was loaded through holds.
    Load { base: Box<Origin>, offset: i32 },
    /// A 32-bit immediate.
    Const(u32),
    /// Read from a VGPR's first active lane: a per-lane (non-uniform) descriptor.
    Waterfall,
    /// Computed or otherwise untraceable.
    Unknown,
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Origin::User(n) => write!(f, "s{n}"),
            Origin::Load { base, offset } => write!(f, "[{base}]+0x{offset:x}"),
            Origin::Const(v) => write!(f, "#0x{v:x}"),
            Origin::Waterfall => f.write_str("readfirstlane"),
            Origin::Unknown => f.write_str("?"),
        }
    }
}

/// What kind of resource an instruction uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    /// V# (4 dwords): buffer load/store.
    Buffer,
    /// T# (8 dwords) sampled with an S#.
    SampledImage,
    /// T# read without a sampler (`image_load`).
    LoadedImage,
    /// T# written (`image_store`).
    StorageImage,
    /// BVH node descriptor (4 dwords).
    Bvh,
}

/// One resource access and where its descriptors come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorUse {
    pub pc: usize,
    pub kind: ResourceKind,
    /// Origin of the descriptor's first dword (the rest follow it).
    pub resource: Origin,
    /// Whether all its dwords come from one place, consecutively.
    pub contiguous: bool,
    pub sampler: Option<Origin>,
}

impl DescriptorUse {
    /// The table base and dword offset, when the descriptor is loaded from a
    /// table whose address is a user SGPR pair: what a host binding needs.
    pub fn table_slot(&self) -> Option<(u8, i32)> {
        match &self.resource {
            Origin::Load { base, offset } => match **base {
                Origin::User(n) => Some((n, *offset)),
                _ => None,
            },
            _ => None,
        }
    }
}

/// Which descriptor-set array an image use binds to, and at which index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageBinding {
    /// `textures[slot]` with `samplers[slot]` (set 0, bindings 0 and 1).
    Sampled(u32),
    /// `textures[slot]`, fetched without a sampler.
    Fetched(u32),
    /// `storage_images[slot]` (set 0, binding 2).
    Storage(u32),
}

/// Shaders bind at most this many images of each kind.
pub const MAX_IMAGE_SLOTS: u32 = 16;

/// The slot each image use binds to, in program order: the rule the SSA
/// builder numbers image instructions by, so the host binds what each
/// instruction reads. Non-image uses get `None`.
pub fn image_bindings(uses: &[DescriptorUse]) -> Vec<Option<ImageBinding>> {
    image_bindings_from(uses, 0, 0)
}

/// As [`image_bindings`], numbering from given first slots (a pixel
/// shader's images follow its vertex shader's in the shared set).
pub fn image_bindings_from(
    uses: &[DescriptorUse],
    texture_base: u32,
    storage_base: u32,
) -> Vec<Option<ImageBinding>> {
    let (mut textures, mut storage) = (texture_base, storage_base);
    uses.iter()
        .map(|u| match u.kind {
            ResourceKind::SampledImage => {
                textures += 1;
                Some(ImageBinding::Sampled(textures - 1))
            }
            ResourceKind::LoadedImage => {
                textures += 1;
                Some(ImageBinding::Fetched(textures - 1))
            }
            ResourceKind::StorageImage => {
                storage += 1;
                Some(ImageBinding::Storage(storage - 1))
            }
            ResourceKind::Buffer | ResourceKind::Bvh => None,
        })
        .collect()
}

/// Traces every resource instruction of `program`.
pub fn trace_descriptors(program: &[DecodedInst]) -> Vec<DescriptorUse> {
    let mut sgpr: HashMap<u8, Origin> = (0..32u8).map(|n| (n, Origin::User(n))).collect();
    let mut uses = Vec::new();

    let origin_of = |sgpr: &HashMap<u8, Origin>, op: &Operand| -> Origin {
        match op {
            Operand::Sgpr(s) => sgpr.get(s).cloned().unwrap_or(Origin::Unknown),
            Operand::InlineInt(i) => Origin::Const(*i as u32),
            Operand::Literal(v) | Operand::InlineFloat(v) => Origin::Const(*v),
            _ => Origin::Unknown,
        }
    };

    for di in program {
        match &di.inst {
            // --- writes that keep or create a traceable origin ---
            Inst::SMovB32 { dst, src0 } => {
                let o = origin_of(&sgpr, src0);
                sgpr.insert(*dst, o);
            }
            Inst::SMovkI32 { dst, simm16 } => {
                sgpr.insert(*dst, Origin::Const(*simm16 as i32 as u32));
            }
            Inst::SMovB64 { dst, src0 } => {
                let (lo, hi) = match src0 {
                    Operand::Sgpr(s) => (
                        sgpr.get(s).cloned().unwrap_or(Origin::Unknown),
                        sgpr.get(&(s + 1)).cloned().unwrap_or(Origin::Unknown),
                    ),
                    _ => (Origin::Unknown, Origin::Unknown),
                };
                sgpr.insert(*dst, lo);
                sgpr.insert(dst + 1, hi);
            }
            Inst::SLoadDword {
                dst,
                count,
                sbase,
                offset,
            } => {
                let base = pair_origin(&sgpr, *sbase);
                for k in 0..*count {
                    sgpr.insert(
                        dst + k,
                        Origin::Load {
                            base: Box::new(base.clone()),
                            offset: offset + 4 * k as i32,
                        },
                    );
                }
            }
            Inst::VReadFirstLaneB32 { dst, .. } => {
                sgpr.insert(*dst, Origin::Waterfall);
            }

            // --- other scalar writes: the result is computed ---
            Inst::SAddU32 { dst, .. }
            | Inst::SSubU32 { dst, .. }
            | Inst::SMulI32 { dst, .. }
            | Inst::SAndB32 { dst, .. }
            | Inst::SOrB32 { dst, .. }
            | Inst::SXorB32 { dst, .. }
            | Inst::SLshlB32 { dst, .. }
            | Inst::SLshrB32 { dst, .. }
            | Inst::SAddkI32 { dst, .. }
            | Inst::SMulkI32 { dst, .. } => {
                sgpr.insert(*dst, Origin::Unknown);
            }
            Inst::SAndB64 { dst, .. }
            | Inst::SOrB64 { dst, .. }
            | Inst::SXorB64 { dst, .. }
            | Inst::SAndN2B64 { dst, .. }
            | Inst::SOrN2B64 { dst, .. }
            | Inst::SNotB64 { dst, .. }
            | Inst::SAndSaveExecB64 { dst, .. }
            | Inst::SOrSaveExecB64 { dst, .. } => {
                sgpr.insert(*dst, Origin::Unknown);
                sgpr.insert(dst + 1, Origin::Unknown);
            }
            Inst::VCmp { dst: Some(d), .. } => {
                sgpr.insert(*d, Origin::Unknown);
                sgpr.insert(d + 1, Origin::Unknown);
            }

            // --- resource uses ---
            Inst::BufferLoad { srsrc, .. }
            | Inst::BufferStore { srsrc, .. }
            | Inst::BufferAtomic { srsrc, .. }
            | Inst::SBufferLoadDword { srsrc, .. } => {
                uses.push(describe(
                    &sgpr,
                    di.pc,
                    ResourceKind::Buffer,
                    *srsrc,
                    4,
                    None,
                ));
            }
            Inst::ImageSample {
                rsrc_sgpr,
                samp_sgpr,
                ..
            } => {
                uses.push(describe(
                    &sgpr,
                    di.pc,
                    ResourceKind::SampledImage,
                    *rsrc_sgpr,
                    8,
                    Some(*samp_sgpr),
                ));
            }
            Inst::ImageLoad { rsrc_sgpr, .. } | Inst::ImageGetResInfo { rsrc_sgpr, .. } => {
                uses.push(describe(
                    &sgpr,
                    di.pc,
                    ResourceKind::LoadedImage,
                    *rsrc_sgpr,
                    8,
                    None,
                ));
            }
            Inst::ImageStore { rsrc_sgpr, .. } => {
                uses.push(describe(
                    &sgpr,
                    di.pc,
                    ResourceKind::StorageImage,
                    *rsrc_sgpr,
                    8,
                    None,
                ));
            }
            Inst::ImageBvh64IntersectRay { rsrc_sgpr, .. } => {
                uses.push(describe(
                    &sgpr,
                    di.pc,
                    ResourceKind::Bvh,
                    *rsrc_sgpr,
                    4,
                    None,
                ));
            }
            _ => {}
        }
    }
    uses
}

/// The 64-bit address an SGPR pair holds, as an origin: a user pair names
/// its low register; a pair loaded from one table entry names that entry.
fn pair_origin(sgpr: &HashMap<u8, Origin>, base: u8) -> Origin {
    let lo = sgpr.get(&base).cloned().unwrap_or(Origin::Unknown);
    let hi = sgpr.get(&(base + 1)).cloned().unwrap_or(Origin::Unknown);
    match (&lo, &hi) {
        (Origin::User(a), Origin::User(b)) if *b == a + 1 => lo,
        (
            Origin::Load {
                base: b0,
                offset: o0,
            },
            Origin::Load {
                base: b1,
                offset: o1,
            },
        ) if b0 == b1 && *o1 == o0 + 4 => lo,
        _ => Origin::Unknown,
    }
}

fn describe(
    sgpr: &HashMap<u8, Origin>,
    pc: usize,
    kind: ResourceKind,
    first: u8,
    dwords: u8,
    sampler: Option<u8>,
) -> DescriptorUse {
    let get = |r: u8| sgpr.get(&r).cloned().unwrap_or(Origin::Unknown);
    let resource = get(first);
    let contiguous = (1..dwords).all(|k| match (&resource, get(first + k)) {
        (Origin::User(a), Origin::User(b)) => b == a + k,
        (
            Origin::Load { base, offset },
            Origin::Load {
                base: b2,
                offset: o2,
            },
        ) => *base == b2 && o2 == offset + 4 * k as i32,
        (Origin::Waterfall, Origin::Waterfall) => true,
        _ => false,
    });
    DescriptorUse {
        pc,
        kind,
        resource,
        contiguous,
        sampler: sampler.map(get),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna2::decode_program;
    use crate::rdna2::encode::*;

    #[test]
    fn buffer_descriptor_loaded_from_a_user_table() {
        // s_load_dwordx4 s[8:11], s[2:3], 0x20 ; buffer_load_dword v1, v0, s[8:11]
        let prog = [
            s_load_dwordx4(8, 2, 0x20)[0],
            s_load_dwordx4(8, 2, 0x20)[1],
            buffer_load_dword(1, 0, 8, 0)[0],
            buffer_load_dword(1, 0, 8, 0)[1],
            s_endpgm(),
        ];
        let uses = trace_descriptors(&decode_program(&prog).unwrap());
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].kind, ResourceKind::Buffer);
        assert_eq!(uses[0].table_slot(), Some((2, 0x20)));
        assert!(uses[0].contiguous);
        assert_eq!(uses[0].resource.to_string(), "[s2]+0x20");
    }

    #[test]
    fn descriptors_through_two_tables_and_direct_user_sgprs() {
        // s[4:5] = [s0:1]+0x10 (a table pointer), then a V# from [s4:5]+0x40;
        // a second buffer straight from user SGPRs s[12:15].
        let mut prog = Vec::new();
        prog.extend_from_slice(&s_load_dwordx2(4, 0, 0x10));
        prog.extend_from_slice(&s_load_dwordx4(8, 4, 0x40));
        prog.extend_from_slice(&buffer_load_dword(1, 0, 8, 0));
        prog.extend_from_slice(&buffer_store_dword(1, 0, 12, 0));
        prog.push(s_endpgm());
        let uses = trace_descriptors(&decode_program(&prog).unwrap());
        assert_eq!(uses[0].resource.to_string(), "[[s0]+0x10]+0x40");
        assert_eq!(
            uses[0].table_slot(),
            None,
            "nested: not a single root table"
        );
        assert_eq!(uses[1].resource, Origin::User(12));
        assert!(uses[1].contiguous);
    }

    #[test]
    fn a_computed_descriptor_is_unknown() {
        let mut prog = vec![s_add_u32(8, 0, 1)];
        prog.extend_from_slice(&buffer_load_dword(1, 0, 8, 0));
        prog.push(s_endpgm());
        let uses = trace_descriptors(&decode_program(&prog).unwrap());
        assert_eq!(uses[0].resource, Origin::Unknown);
        assert!(!uses[0].contiguous);
    }
}
