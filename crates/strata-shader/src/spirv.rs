//! Direct SPIR-V 1.6 binary writer targeting No-API Vulkan (`PhysicalStorageBuffer64`).
//!
//! Emits `spirv-val`-clean compute shader modules directly from Strata's SSA IR:
//! - 128-byte Push Constant block (`uint sgpr[32]`) for console User SGPRs
//! - 64-bit `PhysicalStorageBuffer` pointer loads/stores (scalar & 128-bit coalesced)
//! - Structured multi-block CFG (`OpSelectionMerge` / `OpLoopMerge` + `OpPhi`),
//!   blocks written in dominance order, loop phis patched once their back-edge
//!   values exist
//! - Subgroup-size-agnostic Wave64 ballots (`OpGroupNonUniformBallot` + Workgroup `OpAtomicOr`)
//! - Workgroup shared memory (LDS) + `OpControlBarrier`
//! - Inline watertight BVH Box32 ray-AABB intersection

use std::collections::HashMap;

use crate::rdna2::{AtomicOp, CmpOp, CmpType};
use crate::ssa::{
    ExportTarget, ImageDim, SsaModule, SsaOp, Stage, StageInput, Terminator, Ty, ValueId,
};

// SPIR-V header constants
const SPIRV_MAGIC: u32 = 0x0723_0203;
const SPIRV_VERSION_1_6: u32 = 0x0001_0600;

// SPIR-V opcodes
const OP_EXTENSION: u16 = 10;
const OP_EXT_INST_IMPORT: u16 = 11;
const OP_EXT_INST: u16 = 12;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_CAPABILITY: u16 = 17;

const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_BOOL: u16 = 20;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_IMAGE: u16 = 25;
const OP_TYPE_SAMPLER: u16 = 26;
const OP_TYPE_SAMPLED_IMAGE: u16 = 27;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_FUNCTION: u16 = 33;

const OP_CONSTANT_TRUE: u16 = 41;
const OP_CONSTANT_FALSE: u16 = 42;
const OP_CONSTANT: u16 = 43;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_COMPOSITE_CONSTRUCT: u16 = 80;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_SAMPLED_IMAGE: u16 = 86;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
const OP_IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
const OP_IMAGE_SAMPLE_DREF_IMPLICIT_LOD: u16 = 89;
const OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD: u16 = 90;
const OP_IMAGE_FETCH: u16 = 95;
const OP_IMAGE_GATHER: u16 = 96;
const OP_IMAGE_WRITE: u16 = 99;
const OP_IMAGE_QUERY_SIZE_LOD: u16 = 103;

const OP_CONVERT_F_TO_U: u16 = 109;
const OP_CONVERT_F_TO_S: u16 = 110;
const OP_CONVERT_S_TO_F: u16 = 111;
const OP_CONVERT_U_TO_F: u16 = 112;
const OP_U_CONVERT: u16 = 113;
const OP_CONVERT_U_TO_PTR: u16 = 120;
const OP_BITCAST: u16 = 124;

const OP_I_ADD: u16 = 128;
const OP_F_ADD: u16 = 129;
const OP_I_SUB: u16 = 130;
const OP_F_SUB: u16 = 131;
const OP_I_MUL: u16 = 132;
const OP_F_MUL: u16 = 133;
const OP_F_DIV: u16 = 136;

const OP_LOGICAL_NOT_EQUAL: u16 = 165;
const OP_LOGICAL_OR: u16 = 166;
const OP_LOGICAL_AND: u16 = 167;
const OP_LOGICAL_NOT: u16 = 168;
const OP_SELECT: u16 = 169;

const OP_I_EQUAL: u16 = 170;
const OP_I_NOT_EQUAL: u16 = 171;
const OP_U_GREATER_THAN: u16 = 172;
const OP_S_GREATER_THAN: u16 = 173;
const OP_U_GREATER_THAN_EQUAL: u16 = 174;
const OP_S_GREATER_THAN_EQUAL: u16 = 175;
const OP_U_LESS_THAN: u16 = 176;
const OP_S_LESS_THAN: u16 = 177;
const OP_U_LESS_THAN_EQUAL: u16 = 178;
const OP_S_LESS_THAN_EQUAL: u16 = 179;
const OP_F_ORD_EQUAL: u16 = 180;
const OP_F_ORD_NOT_EQUAL: u16 = 182;
const OP_F_ORD_LESS_THAN: u16 = 184;
const OP_F_ORD_GREATER_THAN: u16 = 186;
const OP_F_ORD_LESS_THAN_EQUAL: u16 = 188;
const OP_F_ORD_GREATER_THAN_EQUAL: u16 = 190;

const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
const OP_SHIFT_RIGHT_ARITHMETIC: u16 = 195;
const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
const OP_BITWISE_OR: u16 = 197;
const OP_BITWISE_XOR: u16 = 198;
const OP_BITWISE_AND: u16 = 199;
const OP_NOT: u16 = 200;
const OP_BIT_FIELD_S_EXTRACT: u16 = 202;
const OP_BIT_FIELD_U_EXTRACT: u16 = 203;

const OP_CONTROL_BARRIER: u16 = 224;
const OP_ATOMIC_EXCHANGE: u16 = 229;
const OP_ATOMIC_COMPARE_EXCHANGE: u16 = 230;
const OP_ATOMIC_I_ADD: u16 = 234;
const OP_ATOMIC_I_SUB: u16 = 235;
const OP_ATOMIC_S_MIN: u16 = 236;
const OP_ATOMIC_U_MIN: u16 = 237;
const OP_ATOMIC_S_MAX: u16 = 238;
const OP_ATOMIC_U_MAX: u16 = 239;
const OP_ATOMIC_AND: u16 = 240;
const OP_ATOMIC_OR: u16 = 241;
const OP_ATOMIC_XOR: u16 = 242;
const OP_PHI: u16 = 245;
const OP_LOOP_MERGE: u16 = 246;
const OP_SELECTION_MERGE: u16 = 247;
const OP_LABEL: u16 = 248;
const OP_BRANCH: u16 = 249;
const OP_BRANCH_CONDITIONAL: u16 = 250;
const OP_SWITCH: u16 = 251;
const OP_RETURN: u16 = 253;

const OP_GROUP_NON_UNIFORM_ANY: u16 = 335;
const OP_GROUP_NON_UNIFORM_BROADCAST_FIRST: u16 = 338;
const OP_GROUP_NON_UNIFORM_BALLOT: u16 = 339;
const OP_GROUP_NON_UNIFORM_SHUFFLE: u16 = 345;
const OP_GROUP_NON_UNIFORM_SHUFFLE_XOR: u16 = 346;
const OP_GROUP_NON_UNIFORM_QUAD_BROADCAST: u16 = 365;
const OP_GROUP_NON_UNIFORM_QUAD_SWAP: u16 = 366;

// Capabilities
const CAP_SHADER: u32 = 1;
const CAP_INT64: u32 = 11;
const CAP_IMAGE_QUERY: u32 = 50;
const CAP_GROUP_NON_UNIFORM: u32 = 61;
const CAP_GROUP_NON_UNIFORM_VOTE: u32 = 62;
const CAP_GROUP_NON_UNIFORM_BALLOT: u32 = 64;
const CAP_GROUP_NON_UNIFORM_SHUFFLE: u32 = 65;
const CAP_GROUP_NON_UNIFORM_QUAD: u32 = 68;
const CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES: u32 = 5347;
const CAP_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT: u32 = 56;
const CAP_QUAD_CONTROL_KHR: u32 = 5087;

// Storage classes
const SC_UNIFORM_CONSTANT: u32 = 0;
const SC_INPUT: u32 = 1;
const SC_OUTPUT: u32 = 3;
const SC_WORKGROUP: u32 = 4;
const SC_FUNCTION: u32 = 7;
const SC_PUSH_CONSTANT: u32 = 9;
const SC_PHYSICAL_STORAGE_BUFFER: u32 = 5349;

// Decorations
const DEC_BLOCK: u32 = 2;
const DEC_ARRAY_STRIDE: u32 = 6;
const DEC_BUILTIN: u32 = 11;
const DEC_BINDING: u32 = 33;
const DEC_LOCATION: u32 = 30;
const DEC_FLAT: u32 = 14;
const DEC_DESCRIPTOR_SET: u32 = 34;
const DEC_OFFSET: u32 = 35;

// BuiltIns
const BUILTIN_POSITION: u32 = 0;
const BUILTIN_FRAG_COORD: u32 = 15;
const BUILTIN_FRONT_FACING: u32 = 17;
const BUILTIN_FRAG_DEPTH: u32 = 22;
const BUILTIN_VERTEX_INDEX: u32 = 42;
const BUILTIN_INSTANCE_INDEX: u32 = 43;
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;
const BUILTIN_LOCAL_INVOCATION_INDEX: u32 = 29;
const BUILTIN_SUBGROUP_SIZE: u32 = 36;

// Memory operands
const MEM_ALIGNED: u32 = 0x2;

// Image operands
const IMG_OP_BIAS: u32 = 0x1;
const IMG_OP_LOD: u32 = 0x2;

/// Descriptor set 0 bindings images use (the runner's global layout).
/// Each image class is its own binding (a SPIR-V image type fixes its
/// dimensionality); a slot index means the same image in every class, and a
/// slot is only ever read through the class its instruction names.
pub const BINDING_TEXTURES: u32 = 0;
pub const BINDING_SAMPLERS: u32 = 1;
pub const BINDING_STORAGE_IMAGES: u32 = 2;
pub const BINDING_TEXTURES_3D: u32 = 3;
pub const BINDING_TEXTURES_CUBE: u32 = 4;
pub const BINDING_TEXTURES_ARRAY: u32 = 5;
pub const BINDING_STORAGE_3D: u32 = 6;
pub const BINDING_STORAGE_ARRAY: u32 = 7;

/// The sampled-image binding of an image class.
pub fn texture_binding(dim: ImageDim) -> u32 {
    match dim {
        ImageDim::D2 => BINDING_TEXTURES,
        ImageDim::D3 => BINDING_TEXTURES_3D,
        ImageDim::Cube => BINDING_TEXTURES_CUBE,
        ImageDim::D2Array => BINDING_TEXTURES_ARRAY,
    }
}

/// The storage-image binding of an image class (cubes are written as arrays).
pub fn storage_binding(dim: ImageDim) -> u32 {
    match dim.for_storage() {
        ImageDim::D3 => BINDING_STORAGE_3D,
        ImageDim::D2Array | ImageDim::Cube => BINDING_STORAGE_ARRAY,
        ImageDim::D2 => BINDING_STORAGE_IMAGES,
    }
}

/// All image classes, in the order the emitter indexes them.
pub const IMAGE_DIMS: [ImageDim; 4] = [
    ImageDim::D2,
    ImageDim::D3,
    ImageDim::Cube,
    ImageDim::D2Array,
];

fn dim_index(d: ImageDim) -> usize {
    match d {
        ImageDim::D2 => 0,
        ImageDim::D3 => 1,
        ImageDim::Cube => 2,
        ImageDim::D2Array => 3,
    }
}

/// `OpTypeImage` `Dim` and `Arrayed` operands of a class.
fn spirv_dim(d: ImageDim) -> (u32, u32) {
    match d {
        ImageDim::D2 => (1, 0),
        ImageDim::D3 => (2, 0),
        ImageDim::Cube => (3, 0),
        ImageDim::D2Array => (1, 1),
    }
}
/// Array length of each binding.
pub const IMAGE_ARRAY_LEN: u32 = crate::resource::MAX_IMAGE_SLOTS;

const OP_MODULE_PROCESSED: u16 = 330;
const IMAGE_SLOTS_TAG: &str = "strata-image-slots=";

/// Descriptor-array capacity required by a Strata-translated module.
pub fn required_image_slots(words: &[u32]) -> Option<u32> {
    let mut rest = words.get(5..)?;
    while let Some(&word) = rest.first() {
        let len = (word >> 16) as usize;
        if len == 0 {
            return None;
        }
        let (inst, tail) = rest.split_at_checked(len)?;
        if word as u16 == OP_FUNCTION {
            break;
        }
        if word as u16 == OP_MODULE_PROCESSED {
            let bytes: Vec<_> = inst[1..]
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .take_while(|&b| b != 0)
                .collect();
            if let Some(slots) = std::str::from_utf8(&bytes)
                .ok()
                .and_then(|s| s.strip_prefix(IMAGE_SLOTS_TAG))
            {
                return slots.parse().ok();
            }
        }
        rest = tail;
    }
    None
}

// GLSL.std.450 opcodes
const GLSL_ROUND_EVEN: u32 = 2;
const GLSL_FLOOR: u32 = 8;
const GLSL_FRACT: u32 = 10;
const GLSL_EXP2: u32 = 29;
const GLSL_LOG2: u32 = 30;
const GLSL_SQRT: u32 = 31;
const GLSL_INV_SQRT: u32 = 32;
const GLSL_FMIN: u32 = 37;
const GLSL_FMAX: u32 = 40;
const GLSL_FMA: u32 = 50;
const GLSL_PACK_HALF_2X16: u32 = 58;
const GLSL_UNPACK_HALF_2X16: u32 = 62;

pub fn emit_spirv(module: &SsaModule) -> Vec<u32> {
    let mut w = Writer::new();

    // Capabilities & Memory Model
    w.inst(OP_CAPABILITY, &[CAP_SHADER]);
    w.inst(OP_CAPABILITY, &[CAP_INT64]);
    w.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM]);
    w.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM_VOTE]);
    w.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM_BALLOT]);
    w.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM_SHUFFLE]);
    w.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM_QUAD]);
    w.inst(OP_CAPABILITY, &[CAP_IMAGE_QUERY]);
    w.inst(OP_CAPABILITY, &[CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES]);
    // Which image classes the live code reads (sampled) and writes (storage).
    let mut tex_dims = [false; 4];
    let mut stor_dims = [false; 4];
    for i in module
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| !i.dead)
    {
        match &i.op {
            SsaOp::ImageSample { dim, .. }
            | SsaOp::ImageSampleDref { dim, .. }
            | SsaOp::ImageGather { dim, .. }
            | SsaOp::ImageQuerySize { dim, .. }
            | SsaOp::ImageFetch { dim, .. } => tex_dims[dim_index(*dim)] = true,
            SsaOp::ImageStore { dim, .. } => stor_dims[dim_index(dim.for_storage())] = true,
            _ => {}
        }
    }
    let uses_textures = module.texture_slots > 0 || tex_dims.iter().any(|&b| b);
    let uses_storage = module.storage_slots > 0 || stor_dims.iter().any(|&b| b);
    if uses_textures && !tex_dims.iter().any(|&b| b) {
        tex_dims[0] = true;
    }
    if uses_storage && !stor_dims.iter().any(|&b| b) {
        stor_dims[0] = true;
    }
    if uses_storage {
        w.inst(OP_CAPABILITY, &[CAP_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT]);
    }

    // What the stage's interface needs.
    let stage = module.stage;
    let live = || {
        module
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| !i.dead)
    };
    let mut exports: Vec<ExportTarget> = live()
        .filter_map(|i| match i.op {
            SsaOp::Export { target, .. } => Some(target),
            _ => None,
        })
        .collect();
    exports.sort();
    exports.dedup();
    let uses_vid = live().any(|i| i.op == SsaOp::StageInput(StageInput::VertexIndex));
    let uses_iid = live().any(|i| i.op == SsaOp::StageInput(StageInput::InstanceIndex));
    let uses_frag_coord =
        live().any(|i| matches!(i.op, SsaOp::StageInput(StageInput::FragCoord(_))));
    let uses_front_facing = live().any(|i| i.op == SsaOp::StageInput(StageInput::FrontFacing));
    let full_quads = stage == Stage::Pixel
        && module.require_full_quads
        && live().any(|i| matches!(i.op, SsaOp::QuadSwap { .. } | SsaOp::QuadBroadcast { .. }));
    if full_quads {
        w.inst(OP_CAPABILITY, &[CAP_QUAD_CONTROL_KHR]);
        w.inst_str_plain(OP_EXTENSION, "SPV_KHR_quad_control");
    }

    let glsl_ext = w.id();
    w.inst_str(OP_EXT_INST_IMPORT, glsl_ext, "GLSL.std.450");

    // AddressingModel = PhysicalStorageBuffer64 (5348), MemoryModel = GLSL450 (1)
    w.inst(OP_MEMORY_MODEL, &[5348, 1]);

    let main_fn = w.id();
    let global_id_var = w.id(); // declared for compute only
    let push_var = w.id();
    let local_idx_var = if module.uses_wave64_ballot {
        Some(w.id())
    } else {
        None
    };
    let sg_size_var = if module.uses_wave64_ballot {
        Some(w.id())
    } else {
        None
    };
    let ballot_shared_var = if module.uses_wave64_ballot {
        Some(w.id())
    } else {
        None
    };
    let lds_var = if module.uses_lds { Some(w.id()) } else { None };
    let tex_vars: [Option<u32>; 4] = tex_dims.map(|u| u.then(|| w.id()));
    let samp_var = uses_textures.then(|| w.id());
    let simg_vars: [Option<u32>; 4] = stor_dims.map(|u| u.then(|| w.id()));

    let attr_vars: Vec<(u8, bool, u32)> = module
        .inputs
        .iter()
        .map(|(&loc, &flat)| (loc, flat, w.id()))
        .collect();
    let vid_var = uses_vid.then(|| w.id());
    let iid_var = uses_iid.then(|| w.id());
    let frag_coord_var = uses_frag_coord.then(|| w.id());
    let front_facing_var = uses_front_facing.then(|| w.id());
    let out_vars: Vec<(ExportTarget, u32)> = exports.iter().map(|&t| (t, w.id())).collect();

    // EntryPoint: GLCompute (5), Vertex (0) or Fragment (4).
    let model = match stage {
        Stage::Compute => 5,
        Stage::Vertex => 0,
        Stage::Pixel => 4,
    };
    let mut ep = vec![model, main_fn];
    encode_str(&mut ep, "main");
    if stage == Stage::Compute {
        ep.push(global_id_var);
    }
    ep.extend(attr_vars.iter().map(|a| a.2));
    ep.extend(vid_var.iter().chain(iid_var.iter()).copied());
    ep.extend(
        frag_coord_var
            .iter()
            .chain(front_facing_var.iter())
            .copied(),
    );
    ep.extend(out_vars.iter().map(|o| o.1));
    ep.push(push_var);
    if let Some(v) = local_idx_var {
        ep.push(v);
    }
    if let Some(v) = sg_size_var {
        ep.push(v);
    }
    if let Some(v) = ballot_shared_var {
        ep.push(v);
    }
    if let Some(v) = lds_var {
        ep.push(v);
    }
    ep.extend(
        tex_vars
            .iter()
            .flatten()
            .chain(samp_var.iter())
            .chain(simg_vars.iter().flatten())
            .copied(),
    );
    w.inst(OP_ENTRY_POINT, &ep);

    match stage {
        // ExecutionMode LocalSize (17)
        Stage::Compute => w.inst(
            OP_EXECUTION_MODE,
            &[
                main_fn,
                17,
                module.workgroup_size[0],
                module.workgroup_size[1],
                module.workgroup_size[2],
            ],
        ),
        Stage::Pixel => {
            w.inst(OP_EXECUTION_MODE, &[main_fn, 7]); // OriginUpperLeft
            if exports.contains(&ExportTarget::Depth) {
                w.inst(OP_EXECUTION_MODE, &[main_fn, 12]); // DepthReplacing
            }
            if full_quads {
                w.inst(OP_EXECUTION_MODE, &[main_fn, 5089]); // RequireFullQuadsKHR
            }
        }
        Stage::Vertex => {}
    }

    let image_array_len = module.texture_slots.max(module.storage_slots).max(1);
    if uses_textures || uses_storage {
        let mut tag = Vec::new();
        encode_str(&mut tag, &format!("{IMAGE_SLOTS_TAG}{image_array_len}"));
        w.inst(OP_MODULE_PROCESSED, &tag);
    }

    // Types
    let t_i32 = w.id();
    let t_ptr_in_i32 = w.id();
    let t_ptr_in_bool = w.id();
    let t_ptr_in_v4f = w.id();
    let t_ptr_out_v4f = w.id();
    let t_ptr_out_f32 = w.id();
    let t_void = w.id();
    let t_bool = w.id();
    let t_u32 = w.id();
    let t_u64 = w.id();
    let t_f32 = w.id();
    let t_uvec3 = w.id();
    let t_uvec4 = w.id();
    let t_fn = w.id();

    let c_32 = w.id();
    let t_sgpr_arr = w.id();
    let t_push_struct = w.id();
    let t_ptr_push_struct = w.id();
    let t_ptr_push_u32 = w.id();
    let t_ptr_in_uvec3 = w.id();
    let t_ptr_in_u32 = w.id();

    let c_2 = w.id();
    let t_ballot_arr = w.id();
    let t_ptr_wg_ballot_arr = w.id();
    let t_ptr_wg_u32 = w.id();

    let c_16384 = w.id();
    let t_lds_arr = w.id();
    let t_ptr_wg_lds_arr = w.id();

    let uses_spill = module.spill_dwords > 0;
    let c_spill_len = uses_spill.then(|| w.id());
    let t_spill_arr = uses_spill.then(|| w.id());
    let t_ptr_func_spill_arr = uses_spill.then(|| w.id());
    let t_ptr_func_u32 = uses_spill.then(|| w.id());
    let spill_var = uses_spill.then(|| w.id());

    // PhysicalStorageBuffer pointer types (scalar u32 & 128-bit uvec4)
    let t_bda_u32_struct = w.id();
    let t_bda_u32_ptr = w.id();
    let t_bda_u32_field_ptr = w.id();
    let t_bda_uvec4_struct = w.id();
    let t_bda_uvec4_ptr = w.id();
    let t_bda_uvec4_field_ptr = w.id();

    let t_v2f = w.id();
    let t_v3f = w.id();
    let t_v4f = w.id();
    let t_uv2 = w.id();
    // Per image class: image, sampled image, array, pointer to array, pointer.
    let t_img: [u32; 4] = std::array::from_fn(|_| w.id());
    let t_sampled: [u32; 4] = std::array::from_fn(|_| w.id());
    let t_img_arr: [u32; 4] = std::array::from_fn(|_| w.id());
    let t_ptr_img_arr: [u32; 4] = std::array::from_fn(|_| w.id());
    let t_ptr_img: [u32; 4] = std::array::from_fn(|_| w.id());
    let t_simg: [u32; 4] = std::array::from_fn(|_| w.id());
    let t_simg_arr: [u32; 4] = std::array::from_fn(|_| w.id());
    let t_ptr_simg_arr: [u32; 4] = std::array::from_fn(|_| w.id());
    let t_ptr_simg: [u32; 4] = std::array::from_fn(|_| w.id());
    let t_sampler = w.id();
    let c_img_len = w.id();
    let t_sampler_arr = w.id();
    let t_ptr_sampler_arr = w.id();
    let t_ptr_sampler = w.id();

    // Decorations
    if stage == Stage::Compute {
        w.inst(
            OP_DECORATE,
            &[global_id_var, DEC_BUILTIN, BUILTIN_GLOBAL_INVOCATION_ID],
        );
    }
    for &(loc, flat, var) in &attr_vars {
        w.inst(OP_DECORATE, &[var, DEC_LOCATION, u32::from(loc)]);
        if flat {
            w.inst(OP_DECORATE, &[var, DEC_FLAT]);
        }
    }
    if let Some(v) = vid_var {
        w.inst(OP_DECORATE, &[v, DEC_BUILTIN, BUILTIN_VERTEX_INDEX]);
    }
    if let Some(v) = iid_var {
        w.inst(OP_DECORATE, &[v, DEC_BUILTIN, BUILTIN_INSTANCE_INDEX]);
    }
    if let Some(v) = frag_coord_var {
        w.inst(OP_DECORATE, &[v, DEC_BUILTIN, BUILTIN_FRAG_COORD]);
    }
    if let Some(v) = front_facing_var {
        w.inst(OP_DECORATE, &[v, DEC_BUILTIN, BUILTIN_FRONT_FACING]);
    }
    for &(t, var) in &out_vars {
        match t {
            ExportTarget::Mrt(n) | ExportTarget::Param(n) => {
                w.inst(OP_DECORATE, &[var, DEC_LOCATION, u32::from(n)])
            }
            ExportTarget::Position => w.inst(OP_DECORATE, &[var, DEC_BUILTIN, BUILTIN_POSITION]),
            ExportTarget::Depth => w.inst(OP_DECORATE, &[var, DEC_BUILTIN, BUILTIN_FRAG_DEPTH]),
        }
    }
    if let Some(v) = local_idx_var {
        w.inst(
            OP_DECORATE,
            &[v, DEC_BUILTIN, BUILTIN_LOCAL_INVOCATION_INDEX],
        );
    }
    if let Some(v) = sg_size_var {
        w.inst(OP_DECORATE, &[v, DEC_BUILTIN, BUILTIN_SUBGROUP_SIZE]);
    }
    w.inst(OP_DECORATE, &[t_sgpr_arr, DEC_ARRAY_STRIDE, 4]);
    w.inst(OP_DECORATE, &[t_push_struct, DEC_BLOCK]);
    w.inst(OP_MEMBER_DECORATE, &[t_push_struct, 0, DEC_OFFSET, 0]);
    w.inst(OP_DECORATE, &[t_bda_u32_struct, DEC_BLOCK]);
    w.inst(OP_MEMBER_DECORATE, &[t_bda_u32_struct, 0, DEC_OFFSET, 0]);
    w.inst(OP_DECORATE, &[t_bda_uvec4_struct, DEC_BLOCK]);
    w.inst(OP_MEMBER_DECORATE, &[t_bda_uvec4_struct, 0, DEC_OFFSET, 0]);
    let image_vars = IMAGE_DIMS
        .iter()
        .enumerate()
        .map(|(i, &d)| (tex_vars[i], texture_binding(d)))
        .chain(std::iter::once((samp_var, BINDING_SAMPLERS)))
        .chain(
            IMAGE_DIMS
                .iter()
                .enumerate()
                .map(|(i, &d)| (simg_vars[i], storage_binding(d))),
        )
        .collect::<Vec<_>>();
    for (var, binding) in image_vars {
        if let Some(v) = var {
            w.inst(OP_DECORATE, &[v, DEC_DESCRIPTOR_SET, 0]);
            w.inst(OP_DECORATE, &[v, DEC_BINDING, binding]);
        }
    }

    // Type definitions
    w.inst(OP_TYPE_VOID, &[t_void]);
    w.inst(OP_TYPE_BOOL, &[t_bool]);
    w.inst(OP_TYPE_INT, &[t_u32, 32, 0]);
    w.inst(OP_TYPE_INT, &[t_u64, 64, 0]);
    w.inst(OP_TYPE_FLOAT, &[t_f32, 32]);
    w.inst(OP_TYPE_VECTOR, &[t_uvec3, t_u32, 3]);
    w.inst(OP_TYPE_VECTOR, &[t_uvec4, t_u32, 4]);
    w.inst(OP_TYPE_FUNCTION, &[t_fn, t_void]);

    w.inst(OP_CONSTANT, &[t_u32, c_32, 32]);
    w.inst(OP_TYPE_ARRAY, &[t_sgpr_arr, t_u32, c_32]);
    w.inst(OP_TYPE_STRUCT, &[t_push_struct, t_sgpr_arr]);
    w.inst(
        OP_TYPE_POINTER,
        &[t_ptr_push_struct, SC_PUSH_CONSTANT, t_push_struct],
    );
    w.inst(OP_TYPE_POINTER, &[t_ptr_push_u32, SC_PUSH_CONSTANT, t_u32]);

    w.inst(OP_TYPE_POINTER, &[t_ptr_in_uvec3, SC_INPUT, t_uvec3]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_in_u32, SC_INPUT, t_u32]);

    w.inst(OP_CONSTANT, &[t_u32, c_2, 2]);
    w.inst(OP_TYPE_ARRAY, &[t_ballot_arr, t_u32, c_2]);
    w.inst(
        OP_TYPE_POINTER,
        &[t_ptr_wg_ballot_arr, SC_WORKGROUP, t_ballot_arr],
    );
    w.inst(OP_TYPE_POINTER, &[t_ptr_wg_u32, SC_WORKGROUP, t_u32]);

    w.inst(OP_CONSTANT, &[t_u32, c_16384, 16384]);
    w.inst(OP_TYPE_ARRAY, &[t_lds_arr, t_u32, c_16384]);
    w.inst(
        OP_TYPE_POINTER,
        &[t_ptr_wg_lds_arr, SC_WORKGROUP, t_lds_arr],
    );

    if uses_spill {
        let c_len = c_spill_len.unwrap();
        let t_arr = t_spill_arr.unwrap();
        let t_ptr_arr = t_ptr_func_spill_arr.unwrap();
        let t_ptr_el = t_ptr_func_u32.unwrap();
        w.inst(OP_CONSTANT, &[t_u32, c_len, module.spill_dwords.max(1)]);
        w.inst(OP_TYPE_ARRAY, &[t_arr, t_u32, c_len]);
        w.inst(OP_TYPE_POINTER, &[t_ptr_arr, SC_FUNCTION, t_arr]);
        w.inst(OP_TYPE_POINTER, &[t_ptr_el, SC_FUNCTION, t_u32]);
    }

    w.inst(OP_TYPE_STRUCT, &[t_bda_u32_struct, t_u32]);
    w.inst(
        OP_TYPE_POINTER,
        &[t_bda_u32_ptr, SC_PHYSICAL_STORAGE_BUFFER, t_bda_u32_struct],
    );
    w.inst(
        OP_TYPE_POINTER,
        &[t_bda_u32_field_ptr, SC_PHYSICAL_STORAGE_BUFFER, t_u32],
    );

    w.inst(OP_TYPE_STRUCT, &[t_bda_uvec4_struct, t_uvec4]);
    w.inst(
        OP_TYPE_POINTER,
        &[
            t_bda_uvec4_ptr,
            SC_PHYSICAL_STORAGE_BUFFER,
            t_bda_uvec4_struct,
        ],
    );
    w.inst(
        OP_TYPE_POINTER,
        &[t_bda_uvec4_field_ptr, SC_PHYSICAL_STORAGE_BUFFER, t_uvec4],
    );

    w.inst(OP_TYPE_VECTOR, &[t_v2f, t_f32, 2]);
    w.inst(OP_TYPE_VECTOR, &[t_v4f, t_f32, 4]);
    w.inst(OP_TYPE_INT, &[t_i32, 32, 1]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_in_i32, SC_INPUT, t_i32]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_in_bool, SC_INPUT, t_bool]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_in_v4f, SC_INPUT, t_v4f]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_out_v4f, SC_OUTPUT, t_v4f]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_out_f32, SC_OUTPUT, t_f32]);
    w.inst(OP_TYPE_VECTOR, &[t_uv2, t_u32, 2]);
    w.inst(OP_TYPE_VECTOR, &[t_v3f, t_f32, 3]);
    if uses_textures || uses_storage {
        // Float images of each class in use; format Unknown (the view says
        // what it is).
        w.inst(OP_CONSTANT, &[t_u32, c_img_len, image_array_len]);
        for (i, &d) in IMAGE_DIMS.iter().enumerate() {
            let (sdim, arrayed) = spirv_dim(d);
            if tex_dims[i] {
                w.inst(OP_TYPE_IMAGE, &[t_img[i], t_f32, sdim, 0, arrayed, 0, 1, 0]);
                w.inst(OP_TYPE_SAMPLED_IMAGE, &[t_sampled[i], t_img[i]]);
                w.inst(OP_TYPE_ARRAY, &[t_img_arr[i], t_img[i], c_img_len]);
                w.inst(
                    OP_TYPE_POINTER,
                    &[t_ptr_img_arr[i], SC_UNIFORM_CONSTANT, t_img_arr[i]],
                );
                w.inst(
                    OP_TYPE_POINTER,
                    &[t_ptr_img[i], SC_UNIFORM_CONSTANT, t_img[i]],
                );
            }
            if stor_dims[i] {
                w.inst(
                    OP_TYPE_IMAGE,
                    &[t_simg[i], t_f32, sdim, 0, arrayed, 0, 2, 0],
                );
                w.inst(OP_TYPE_ARRAY, &[t_simg_arr[i], t_simg[i], c_img_len]);
                w.inst(
                    OP_TYPE_POINTER,
                    &[t_ptr_simg_arr[i], SC_UNIFORM_CONSTANT, t_simg_arr[i]],
                );
                w.inst(
                    OP_TYPE_POINTER,
                    &[t_ptr_simg[i], SC_UNIFORM_CONSTANT, t_simg[i]],
                );
            }
        }
        w.inst(OP_TYPE_SAMPLER, &[t_sampler]);
        w.inst(OP_TYPE_ARRAY, &[t_sampler_arr, t_sampler, c_img_len]);
        w.inst(
            OP_TYPE_POINTER,
            &[t_ptr_sampler_arr, SC_UNIFORM_CONSTANT, t_sampler_arr],
        );
        w.inst(
            OP_TYPE_POINTER,
            &[t_ptr_sampler, SC_UNIFORM_CONSTANT, t_sampler],
        );
    }

    // Constant pool
    let mut const_u32: HashMap<u32, u32> = HashMap::new();
    let mut const_u64: HashMap<u64, u32> = HashMap::new();
    let mut const_f32: HashMap<u32, u32> = HashMap::new();
    const_u32.insert(2, c_2);
    const_u32.insert(32, c_32);

    let c_true = w.id();
    let c_false = w.id();
    w.inst(OP_CONSTANT_TRUE, &[t_bool, c_true]);
    w.inst(OP_CONSTANT_FALSE, &[t_bool, c_false]);

    for val in [0u32, 1, 3, 5, 31, 64, 0x108, 0xffff_ffff] {
        const_u32.entry(val).or_insert_with(|| {
            let id = w.id();
            w.inst(OP_CONSTANT, &[t_u32, id, val]);
            id
        });
    }
    let c_zero = const_u32[&0];
    let c_one = const_u32[&1];
    let c_two = const_u32[&2];
    let c_scope_wg = c_two; // Scope::Workgroup = 2
    let c_scope_subgroup = const_u32[&3]; // Scope::Subgroup = 3
    let c_sem_wg = const_u32[&0x108]; // AcquireRelease | WorkgroupMemory
    let c_invalid_node = const_u32[&0xffff_ffff];

    let c_fzero = w.id();
    w.inst(OP_CONSTANT, &[t_f32, c_fzero, 0.0f32.to_bits()]);
    const_f32.insert(0.0f32.to_bits(), c_fzero);
    let c_finf = w.id();
    w.inst(OP_CONSTANT, &[t_f32, c_finf, f32::INFINITY.to_bits()]);
    const_f32.insert(f32::INFINITY.to_bits(), c_finf);

    let c_fone = w.id();
    w.inst(OP_CONSTANT, &[t_f32, c_fone, 1.0f32.to_bits()]);
    const_f32.insert(1.0f32.to_bits(), c_fone);

    let c_32_u64 = w.id();
    w.inst(OP_CONSTANT, &[t_u64, c_32_u64, 32, 0]);
    const_u64.insert(32, c_32_u64);

    // Pre-emit constants referenced by live SSA instructions
    let mut needs_bvh_offsets = false;
    for block in &module.blocks {
        for inst in &block.insts {
            if inst.dead {
                continue;
            }
            match &inst.op {
                SsaOp::ConstU32(v) => {
                    const_u32.entry(*v).or_insert_with(|| {
                        let id = w.id();
                        w.inst(OP_CONSTANT, &[t_u32, id, *v]);
                        id
                    });
                }
                SsaOp::ConstF32(bits) => {
                    const_f32.entry(*bits).or_insert_with(|| {
                        let id = w.id();
                        w.inst(OP_CONSTANT, &[t_f32, id, *bits]);
                        id
                    });
                }
                SsaOp::PushConstSgpr(idx) => {
                    const_u32.entry(*idx as u32).or_insert_with(|| {
                        let id = w.id();
                        w.inst(OP_CONSTANT, &[t_u32, id, *idx as u32]);
                        id
                    });
                }
                SsaOp::SpillLoad { slot } | SsaOp::SpillStore { slot, .. } => {
                    const_u32.entry(*slot).or_insert_with(|| {
                        let id = w.id();
                        w.inst(OP_CONSTANT, &[t_u32, id, *slot]);
                        id
                    });
                }
                SsaOp::BdaLoad { byte_offset, .. } | SsaOp::BdaStore { byte_offset, .. } => {
                    if *byte_offset != 0 {
                        let off = *byte_offset as u64;
                        const_u64.entry(off).or_insert_with(|| {
                            let id = w.id();
                            w.inst(OP_CONSTANT, &[t_u64, id, off as u32, (off >> 32) as u32]);
                            id
                        });
                    }
                }
                SsaOp::Bvh64IntersectBox32 { .. } => {
                    needs_bvh_offsets = true;
                }
                SsaOp::ImageSample { slot, .. }
                | SsaOp::ImageSampleDref { slot, .. }
                | SsaOp::ImageQuerySize { slot, .. }
                | SsaOp::ImageFetch { slot, .. }
                | SsaOp::ImageStore { slot, .. } => {
                    const_u32.entry(*slot).or_insert_with(|| {
                        let id = w.id();
                        w.inst(OP_CONSTANT, &[t_u32, id, *slot]);
                        id
                    });
                }
                SsaOp::ImageGather { slot, comp, .. } => {
                    for &k in &[*slot, *comp] {
                        const_u32.entry(k).or_insert_with(|| {
                            let id = w.id();
                            w.inst(OP_CONSTANT, &[t_u32, id, k]);
                            id
                        });
                    }
                }
                _ => {}
            }
        }
    }
    // Constants the image coordinate builders use (cube face decoding).
    for val in [4u32, 7] {
        const_u32.entry(val).or_insert_with(|| {
            let id = w.id();
            w.inst(OP_CONSTANT, &[t_u32, id, val]);
            id
        });
    }
    for val in [1.5f32, 2.0, -1.0] {
        const_f32.entry(val.to_bits()).or_insert_with(|| {
            let id = w.id();
            w.inst(OP_CONSTANT, &[t_f32, id, val.to_bits()]);
            id
        });
    }
    if needs_bvh_offsets {
        for off in (16u64..112).step_by(4) {
            const_u64.entry(off).or_insert_with(|| {
                let id = w.id();
                w.inst(OP_CONSTANT, &[t_u64, id, off as u32, 0]);
                id
            });
        }
    }

    // Interface variables
    if stage == Stage::Compute {
        w.inst(OP_VARIABLE, &[t_ptr_in_uvec3, global_id_var, SC_INPUT]);
    }
    for &(_, _, var) in &attr_vars {
        w.inst(OP_VARIABLE, &[t_ptr_in_v4f, var, SC_INPUT]);
    }
    for v in vid_var.iter().chain(iid_var.iter()) {
        w.inst(OP_VARIABLE, &[t_ptr_in_i32, *v, SC_INPUT]);
    }
    if let Some(v) = frag_coord_var {
        w.inst(OP_VARIABLE, &[t_ptr_in_v4f, v, SC_INPUT]);
    }
    if let Some(v) = front_facing_var {
        w.inst(OP_VARIABLE, &[t_ptr_in_bool, v, SC_INPUT]);
    }
    for &(t, var) in &out_vars {
        let ty = if t == ExportTarget::Depth {
            t_ptr_out_f32
        } else {
            t_ptr_out_v4f
        };
        w.inst(OP_VARIABLE, &[ty, var, SC_OUTPUT]);
    }
    w.inst(
        OP_VARIABLE,
        &[t_ptr_push_struct, push_var, SC_PUSH_CONSTANT],
    );
    if let Some(v) = local_idx_var {
        w.inst(OP_VARIABLE, &[t_ptr_in_u32, v, SC_INPUT]);
    }
    if let Some(v) = sg_size_var {
        w.inst(OP_VARIABLE, &[t_ptr_in_u32, v, SC_INPUT]);
    }
    if let Some(v) = ballot_shared_var {
        w.inst(OP_VARIABLE, &[t_ptr_wg_ballot_arr, v, SC_WORKGROUP]);
    }
    if let Some(v) = lds_var {
        w.inst(OP_VARIABLE, &[t_ptr_wg_lds_arr, v, SC_WORKGROUP]);
    }
    for i in 0..4 {
        if let Some(v) = tex_vars[i] {
            w.inst(OP_VARIABLE, &[t_ptr_img_arr[i], v, SC_UNIFORM_CONSTANT]);
        }
    }
    if let Some(v) = samp_var {
        w.inst(OP_VARIABLE, &[t_ptr_sampler_arr, v, SC_UNIFORM_CONSTANT]);
    }
    for i in 0..4 {
        if let Some(v) = simg_vars[i] {
            w.inst(OP_VARIABLE, &[t_ptr_simg_arr[i], v, SC_UNIFORM_CONSTANT]);
        }
    }

    let img_ids = ImgIds {
        t_img,
        t_ptr_img,
        t_sampled,
        tex_vars,
        t_sampler,
        t_ptr_sampler,
        samp_var,
    };
    let cube_ids = CoordIds {
        t_f32,
        t_u32,
        t_bool,
        t_v2f,
        t_v3f,
        c_half3: const_f32[&1.5f32.to_bits()],
        c_ftwo: const_f32[&2.0f32.to_bits()],
        c_fone,
        c_fmone: const_f32[&(-1.0f32).to_bits()],
        c_u: [0u32, 1, 2, 3, 4, 5, 7].map(|k| const_u32[&k]),
    };

    // Function body with multi-block labels
    w.inst(OP_FUNCTION, &[t_void, main_fn, 0, t_fn]);
    let block_labels: Vec<u32> = (0..module.blocks.len()).map(|_| w.id()).collect();
    let mut val_map: HashMap<ValueId, u32> = HashMap::new();
    // Phi operands defined later (loop back edges): `(word index, value)`.
    let mut phi_patches: Vec<(usize, ValueId)> = Vec::new();

    for bi in block_order(module) {
        let block = &module.blocks[bi];
        w.inst(OP_LABEL, &[block_labels[bi]]);

        // SPIR-V requires Function-storage-class OpVariable declarations at the very
        // start of the entry block (before any other instructions).
        if bi == 0 {
            if let (Some(sv), Some(t_ptr_arr)) = (spill_var, t_ptr_func_spill_arr) {
                w.inst(OP_VARIABLE, &[t_ptr_arr, sv, SC_FUNCTION]);
            }
        }

        // SPIR-V requires all OpPhi instructions at the very start of the block.
        for inst in &block.insts {
            if inst.dead {
                continue;
            }
            if let SsaOp::Phi { incoming } = &inst.op {
                let res_ty = match inst.ty {
                    Ty::Bool => t_bool,
                    Ty::U32 => t_u32,
                    Ty::F32 => t_f32,
                    Ty::U64 => t_u64,
                    Ty::U32x4 => t_uvec4,
                };
                let res = w.id();
                let mut ops = vec![res_ty, res];
                let first_word = w.body.len() + 1;
                for &(pred_blk, val_id) in incoming {
                    match val_map.get(&val_id) {
                        Some(&id) => ops.push(id),
                        None => {
                            phi_patches.push((first_word + ops.len(), val_id));
                            ops.push(0);
                        }
                    }
                    ops.push(block_labels[pred_blk]);
                }
                w.inst(OP_PHI, &ops);
                val_map.insert(inst.id, res);
            }
        }

        for inst in &block.insts {
            if inst.dead || matches!(inst.op, SsaOp::Phi { .. }) {
                continue;
            }
            match &inst.op {
                SsaOp::Phi { .. } => unreachable!(),
                SsaOp::ConstU32(v) => {
                    val_map.insert(inst.id, const_u32[v]);
                }
                SsaOp::ConstF32(bits) => {
                    val_map.insert(inst.id, const_f32[bits]);
                }
                SsaOp::ConstBool(b) => {
                    val_map.insert(inst.id, if *b { c_true } else { c_false });
                }
                SsaOp::PushConstSgpr(idx) => {
                    let ptr = w.id();
                    w.inst(
                        OP_ACCESS_CHAIN,
                        &[
                            t_ptr_push_u32,
                            ptr,
                            push_var,
                            c_zero,
                            const_u32[&(*idx as u32)],
                        ],
                    );
                    let res = w.id();
                    w.inst(OP_LOAD, &[t_u32, res, ptr]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::ThreadIdX => {
                    let ptr = w.id();
                    w.inst(OP_ACCESS_CHAIN, &[t_ptr_in_u32, ptr, global_id_var, c_zero]);
                    let res = w.id();
                    w.inst(OP_LOAD, &[t_u32, res, ptr]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::BitcastU32ToF32(a) => {
                    let res = w.id();
                    w.inst(OP_BITCAST, &[t_f32, res, val_map[a]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::BitcastF32ToU32(a) => {
                    let res = w.id();
                    w.inst(OP_BITCAST, &[t_u32, res, val_map[a]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::PackU64 { lo, hi } => {
                    let lo64 = w.id();
                    w.inst(OP_U_CONVERT, &[t_u64, lo64, val_map[lo]]);
                    let hi64 = w.id();
                    w.inst(OP_U_CONVERT, &[t_u64, hi64, val_map[hi]]);
                    let hi_sh = w.id();
                    w.inst(OP_SHIFT_LEFT_LOGICAL, &[t_u64, hi_sh, hi64, c_32_u64]);
                    let res = w.id();
                    w.inst(OP_BITWISE_OR, &[t_u64, res, lo64, hi_sh]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::UnpackLoU64(a) => {
                    let res = w.id();
                    w.inst(OP_U_CONVERT, &[t_u32, res, val_map[a]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::UnpackHiU64(a) => {
                    let shifted = w.id();
                    w.inst(
                        OP_SHIFT_RIGHT_LOGICAL,
                        &[t_u64, shifted, val_map[a], c_32_u64],
                    );
                    let res = w.id();
                    w.inst(OP_U_CONVERT, &[t_u32, res, shifted]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::AddU32(a, b) => {
                    let res = w.id();
                    w.inst(OP_I_ADD, &[t_u32, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::SubU32(a, b) => {
                    let res = w.id();
                    w.inst(OP_I_SUB, &[t_u32, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::MulU32(a, b) => {
                    let res = w.id();
                    w.inst(OP_I_MUL, &[t_u32, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::AddU64(a, b) => {
                    let res = w.id();
                    w.inst(OP_I_ADD, &[t_u64, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::AndU32(a, b) => {
                    let res = w.id();
                    w.inst(OP_BITWISE_AND, &[t_u32, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::OrU32(a, b) => {
                    let res = w.id();
                    w.inst(OP_BITWISE_OR, &[t_u32, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::XorU32(a, b) => {
                    let res = w.id();
                    w.inst(OP_BITWISE_XOR, &[t_u32, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::ShlU32(a, b) => {
                    let res = w.id();
                    w.inst(OP_SHIFT_LEFT_LOGICAL, &[t_u32, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::LshrU32(a, b) => {
                    let res = w.id();
                    w.inst(
                        OP_SHIFT_RIGHT_LOGICAL,
                        &[t_u32, res, val_map[a], val_map[b]],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::AshrI32(a, b) => {
                    let res = w.id();
                    w.inst(
                        OP_SHIFT_RIGHT_ARITHMETIC,
                        &[t_u32, res, val_map[a], val_map[b]],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::BfeU32 { src, offset, width } => {
                    let res = w.id();
                    w.inst(
                        OP_BIT_FIELD_U_EXTRACT,
                        &[t_u32, res, val_map[src], val_map[offset], val_map[width]],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::AddF32(a, b) => {
                    let res = w.id();
                    w.inst(OP_F_ADD, &[t_f32, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::SubF32(a, b) => {
                    let res = w.id();
                    w.inst(OP_F_SUB, &[t_f32, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::MulF32(a, b) => {
                    let res = w.id();
                    w.inst(OP_F_MUL, &[t_f32, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::MulLegacyF32(a, b) => {
                    let az = w.id();
                    w.inst(OP_F_ORD_EQUAL, &[t_bool, az, val_map[a], c_fzero]);
                    let bz = w.id();
                    w.inst(OP_F_ORD_EQUAL, &[t_bool, bz, val_map[b], c_fzero]);
                    let any_z = w.id();
                    w.inst(OP_LOGICAL_OR, &[t_bool, any_z, az, bz]);
                    let prod = w.id();
                    w.inst(OP_F_MUL, &[t_f32, prod, val_map[a], val_map[b]]);
                    let res = w.id();
                    w.inst(OP_SELECT, &[t_f32, res, any_z, c_fzero, prod]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::MinF32(a, b) => {
                    let res = w.id();
                    w.inst(
                        OP_EXT_INST,
                        &[t_f32, res, glsl_ext, GLSL_FMIN, val_map[a], val_map[b]],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::MaxF32(a, b) => {
                    let res = w.id();
                    w.inst(
                        OP_EXT_INST,
                        &[t_f32, res, glsl_ext, GLSL_FMAX, val_map[a], val_map[b]],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::FmaF32(a, b, c) => {
                    let res = w.id();
                    w.inst(
                        OP_EXT_INST,
                        &[
                            t_f32, res, glsl_ext, GLSL_FMA, val_map[a], val_map[b], val_map[c],
                        ],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::Log2F32(a) => {
                    let res = w.id();
                    w.inst(OP_EXT_INST, &[t_f32, res, glsl_ext, GLSL_LOG2, val_map[a]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::Exp2F32(a) => {
                    let res = w.id();
                    w.inst(OP_EXT_INST, &[t_f32, res, glsl_ext, GLSL_EXP2, val_map[a]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::RcpF32(a) => {
                    let res = w.id();
                    w.inst(OP_F_DIV, &[t_f32, res, c_fone, val_map[a]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::RsqF32(a) => {
                    let res = w.id();
                    w.inst(
                        OP_EXT_INST,
                        &[t_f32, res, glsl_ext, GLSL_INV_SQRT, val_map[a]],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::SqrtF32(a) => {
                    let res = w.id();
                    w.inst(OP_EXT_INST, &[t_f32, res, glsl_ext, GLSL_SQRT, val_map[a]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::FloorF32(a) => {
                    let res = w.id();
                    w.inst(OP_EXT_INST, &[t_f32, res, glsl_ext, GLSL_FLOOR, val_map[a]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::FractF32(a) => {
                    let res = w.id();
                    w.inst(OP_EXT_INST, &[t_f32, res, glsl_ext, GLSL_FRACT, val_map[a]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::CvtF32U32(a) => {
                    let res = w.id();
                    w.inst(OP_CONVERT_U_TO_F, &[t_f32, res, val_map[a]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::CvtU32F32(a) => {
                    let res = w.id();
                    w.inst(OP_CONVERT_F_TO_U, &[t_u32, res, val_map[a]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::CvtF32I32(a) => {
                    let res = w.id();
                    w.inst(OP_CONVERT_S_TO_F, &[t_f32, res, val_map[a]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::CvtI32F32(a) => {
                    let res = w.id();
                    w.inst(OP_CONVERT_F_TO_S, &[t_u32, res, val_map[a]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::Cmp { op, ty, a, b } => {
                    let spv_op = match (ty, op) {
                        (CmpType::F32, CmpOp::Eq) => OP_F_ORD_EQUAL,
                        (CmpType::F32, CmpOp::Ne) => OP_F_ORD_NOT_EQUAL,
                        (CmpType::F32, CmpOp::Lt) => OP_F_ORD_LESS_THAN,
                        (CmpType::F32, CmpOp::Le) => OP_F_ORD_LESS_THAN_EQUAL,
                        (CmpType::F32, CmpOp::Gt) => OP_F_ORD_GREATER_THAN,
                        (CmpType::F32, CmpOp::Ge) => OP_F_ORD_GREATER_THAN_EQUAL,
                        (CmpType::U32, CmpOp::Eq) | (CmpType::I32, CmpOp::Eq) => OP_I_EQUAL,
                        (CmpType::U32, CmpOp::Ne) | (CmpType::I32, CmpOp::Ne) => OP_I_NOT_EQUAL,
                        (CmpType::U32, CmpOp::Lt) => OP_U_LESS_THAN,
                        (CmpType::U32, CmpOp::Le) => OP_U_LESS_THAN_EQUAL,
                        (CmpType::U32, CmpOp::Gt) => OP_U_GREATER_THAN,
                        (CmpType::U32, CmpOp::Ge) => OP_U_GREATER_THAN_EQUAL,
                        (CmpType::I32, CmpOp::Lt) => OP_S_LESS_THAN,
                        (CmpType::I32, CmpOp::Le) => OP_S_LESS_THAN_EQUAL,
                        (CmpType::I32, CmpOp::Gt) => OP_S_GREATER_THAN,
                        (CmpType::I32, CmpOp::Ge) => OP_S_GREATER_THAN_EQUAL,
                    };
                    let res = w.id();
                    w.inst(spv_op, &[t_bool, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::LogicalAnd(a, b) => {
                    let res = w.id();
                    w.inst(OP_LOGICAL_AND, &[t_bool, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::LogicalOr(a, b) => {
                    let res = w.id();
                    w.inst(OP_LOGICAL_OR, &[t_bool, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::LogicalXor(a, b) => {
                    let res = w.id();
                    w.inst(OP_LOGICAL_NOT_EQUAL, &[t_bool, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::LogicalNot(a) => {
                    let res = w.id();
                    w.inst(OP_LOGICAL_NOT, &[t_bool, res, val_map[a]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::SelectU32 { cond, t, f } => {
                    let res = w.id();
                    w.inst(
                        OP_SELECT,
                        &[t_u32, res, val_map[cond], val_map[t], val_map[f]],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::WaveReadFirstLane(a) => {
                    let res = w.id();
                    w.inst(
                        OP_GROUP_NON_UNIFORM_BROADCAST_FIRST,
                        &[t_u32, res, c_scope_subgroup, val_map[a]],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::WaveBallot64(pred) => {
                    let lvar = local_idx_var.unwrap();
                    let sgvar = sg_size_var.unwrap();
                    let svar = ballot_shared_var.unwrap();
                    let tid = w.id();
                    w.inst(OP_LOAD, &[t_u32, tid, lvar]);
                    let sg_size = w.id();
                    w.inst(OP_LOAD, &[t_u32, sg_size, sgvar]);
                    let is_w64 = w.id();
                    w.inst(
                        OP_U_GREATER_THAN_EQUAL,
                        &[t_bool, is_w64, sg_size, const_u32[&64]],
                    );

                    let half_idx = w.id();
                    w.inst(
                        OP_SHIFT_RIGHT_LOGICAL,
                        &[t_u32, half_idx, tid, const_u32[&5]],
                    );
                    let p_slot = w.id();
                    w.inst(OP_ACCESS_CHAIN, &[t_ptr_wg_u32, p_slot, svar, half_idx]);
                    w.inst(OP_STORE, &[p_slot, c_zero]);
                    w.inst(OP_CONTROL_BARRIER, &[c_scope_wg, c_scope_wg, c_sem_wg]);

                    let ballot_v4 = w.id();
                    w.inst(
                        OP_GROUP_NON_UNIFORM_BALLOT,
                        &[t_uvec4, ballot_v4, c_scope_subgroup, val_map[pred]],
                    );
                    let ballot_lo = w.id();
                    w.inst(OP_COMPOSITE_EXTRACT, &[t_u32, ballot_lo, ballot_v4, 0]);
                    let ballot_hi = w.id();
                    w.inst(OP_COMPOSITE_EXTRACT, &[t_u32, ballot_hi, ballot_v4, 1]);
                    let is_hi_half = w.id();
                    w.inst(OP_I_NOT_EQUAL, &[t_bool, is_hi_half, half_idx, c_zero]);
                    let use_hi = w.id();
                    w.inst(OP_LOGICAL_AND, &[t_bool, use_hi, is_w64, is_hi_half]);
                    let half_mask = w.id();
                    w.inst(OP_SELECT, &[t_u32, half_mask, use_hi, ballot_hi, ballot_lo]);

                    let sg_m1 = w.id();
                    w.inst(OP_I_SUB, &[t_u32, sg_m1, sg_size, c_one]);
                    let sg_align_mask = w.id();
                    w.inst(OP_NOT, &[t_u32, sg_align_mask, sg_m1]);
                    let tid_in_half = w.id();
                    w.inst(OP_BITWISE_AND, &[t_u32, tid_in_half, tid, const_u32[&31]]);
                    let sub_shift = w.id();
                    w.inst(
                        OP_BITWISE_AND,
                        &[t_u32, sub_shift, tid_in_half, sg_align_mask],
                    );
                    let shifted_mask = w.id();
                    w.inst(
                        OP_SHIFT_LEFT_LOGICAL,
                        &[t_u32, shifted_mask, half_mask, sub_shift],
                    );
                    let atomic_old = w.id();
                    w.inst(
                        OP_ATOMIC_OR,
                        &[t_u32, atomic_old, p_slot, c_scope_wg, c_zero, shifted_mask],
                    );

                    w.inst(OP_CONTROL_BARRIER, &[c_scope_wg, c_scope_wg, c_sem_wg]);

                    let p_lo = w.id();
                    w.inst(OP_ACCESS_CHAIN, &[t_ptr_wg_u32, p_lo, svar, c_zero]);
                    let lo = w.id();
                    w.inst(OP_LOAD, &[t_u32, lo, p_lo]);
                    let p_hi = w.id();
                    w.inst(OP_ACCESS_CHAIN, &[t_ptr_wg_u32, p_hi, svar, c_one]);
                    let hi = w.id();
                    w.inst(OP_LOAD, &[t_u32, hi, p_hi]);

                    let lo64 = w.id();
                    w.inst(OP_U_CONVERT, &[t_u64, lo64, lo]);
                    let hi64 = w.id();
                    w.inst(OP_U_CONVERT, &[t_u64, hi64, hi]);
                    let hi_sh = w.id();
                    w.inst(OP_SHIFT_LEFT_LOGICAL, &[t_u64, hi_sh, hi64, c_32_u64]);
                    let full_mask = w.id();
                    w.inst(OP_BITWISE_OR, &[t_u64, full_mask, lo64, hi_sh]);
                    val_map.insert(inst.id, full_mask);
                }
                SsaOp::StageInput(input) => {
                    let res = w.id();
                    match *input {
                        StageInput::VertexIndex | StageInput::InstanceIndex => {
                            let var = if *input == StageInput::VertexIndex {
                                vid_var
                            } else {
                                iid_var
                            };
                            let i = w.id();
                            w.inst(OP_LOAD, &[t_i32, i, var.unwrap()]);
                            w.inst(OP_BITCAST, &[t_u32, res, i]);
                        }
                        StageInput::FragCoord(chan) => {
                            let v4 = w.id();
                            w.inst(OP_LOAD, &[t_v4f, v4, frag_coord_var.unwrap()]);
                            let f = w.id();
                            w.inst(OP_COMPOSITE_EXTRACT, &[t_f32, f, v4, u32::from(chan)]);
                            w.inst(OP_BITCAST, &[t_u32, res, f]);
                        }
                        StageInput::FrontFacing => {
                            let b = w.id();
                            w.inst(OP_LOAD, &[t_bool, b, front_facing_var.unwrap()]);
                            w.inst(OP_SELECT, &[t_u32, res, b, c_one, c_zero]);
                        }
                        StageInput::Attr { loc, chan, .. } => {
                            let var = attr_vars.iter().find(|a| a.0 == loc).unwrap().2;
                            let v4 = w.id();
                            w.inst(OP_LOAD, &[t_v4f, v4, var]);
                            let f = w.id();
                            w.inst(OP_COMPOSITE_EXTRACT, &[t_f32, f, v4, u32::from(chan)]);
                            w.inst(OP_BITCAST, &[t_u32, res, f]);
                        }
                    }
                    val_map.insert(inst.id, res);
                }
                SsaOp::Export { target, values } => {
                    let var = out_vars.iter().find(|o| o.0 == *target).unwrap().1;
                    let fs: Vec<u32> = values
                        .iter()
                        .map(|v| {
                            let f = w.id();
                            w.inst(OP_BITCAST, &[t_f32, f, val_map[v]]);
                            f
                        })
                        .collect();
                    if *target == ExportTarget::Depth {
                        w.inst(OP_STORE, &[var, fs[0]]);
                    } else {
                        let v4 = w.id();
                        w.inst(
                            OP_COMPOSITE_CONSTRUCT,
                            &[t_v4f, v4, fs[0], fs[1], fs[2], fs[3]],
                        );
                        w.inst(OP_STORE, &[var, v4]);
                    }
                    val_map.insert(inst.id, c_zero);
                }
                SsaOp::AndU64(a, b) => {
                    let res = w.id();
                    w.inst(OP_BITWISE_AND, &[t_u64, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::BfeI32 { src, offset, width } => {
                    let res = w.id();
                    w.inst(
                        OP_BIT_FIELD_S_EXTRACT,
                        &[t_u32, res, val_map[src], val_map[offset], val_map[width]],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::HalfToF32(a) => {
                    let pair = w.id();
                    w.inst(
                        OP_EXT_INST,
                        &[t_v2f, pair, glsl_ext, GLSL_UNPACK_HALF_2X16, val_map[a]],
                    );
                    let res = w.id();
                    w.inst(OP_COMPOSITE_EXTRACT, &[t_f32, res, pair, 0]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::PackHalf2x16(a, b) => {
                    let pair = w.id();
                    w.inst(
                        OP_COMPOSITE_CONSTRUCT,
                        &[t_v2f, pair, val_map[a], val_map[b]],
                    );
                    let res = w.id();
                    w.inst(
                        OP_EXT_INST,
                        &[t_u32, res, glsl_ext, GLSL_PACK_HALF_2X16, pair],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::RoundEvenF32(a) => {
                    let res = w.id();
                    w.inst(
                        OP_EXT_INST,
                        &[t_f32, res, glsl_ext, GLSL_ROUND_EVEN, val_map[a]],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::ImageSample {
                    slot,
                    dim,
                    u,
                    v,
                    w: wc,
                    lod,
                    bias,
                    implicit,
                } => {
                    let di = dim_index(*dim);
                    let si = sampled_image(&mut w, &img_ids, di, *slot, &const_u32);
                    let coord =
                        sample_coord(&mut w, &cube_ids, *dim, val_map[u], val_map[v], val_map[wc]);
                    let texel = w.id();
                    if *implicit {
                        let mut ops = vec![t_v4f, texel, si, coord];
                        if let Some(b) = bias {
                            ops.extend([IMG_OP_BIAS, val_map[b]]);
                        }
                        w.inst(OP_IMAGE_SAMPLE_IMPLICIT_LOD, &ops);
                    } else {
                        // Compute has no derivatives: implicit-LOD samples use level 0.
                        let lod_id = lod.map_or(c_fzero, |l| val_map[&l]);
                        let lod_id = match bias {
                            Some(b) => {
                                let sum = w.id();
                                w.inst(OP_F_ADD, &[t_f32, sum, lod_id, val_map[b]]);
                                sum
                            }
                            None => lod_id,
                        };
                        w.inst(
                            OP_IMAGE_SAMPLE_EXPLICIT_LOD,
                            &[t_v4f, texel, si, coord, IMG_OP_LOD, lod_id],
                        );
                    }
                    let res = w.id();
                    w.inst(OP_BITCAST, &[t_uvec4, res, texel]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::ImageSampleDref {
                    slot,
                    dim,
                    u,
                    v,
                    w: wc,
                    dref,
                    lod,
                    implicit,
                } => {
                    let di = dim_index(*dim);
                    let si = sampled_image(&mut w, &img_ids, di, *slot, &const_u32);
                    let coord =
                        sample_coord(&mut w, &cube_ids, *dim, val_map[u], val_map[v], val_map[wc]);
                    let res = w.id();
                    if *implicit {
                        w.inst(
                            OP_IMAGE_SAMPLE_DREF_IMPLICIT_LOD,
                            &[t_f32, res, si, coord, val_map[dref]],
                        );
                    } else {
                        let lod_id = lod.map_or(c_fzero, |l| val_map[&l]);
                        w.inst(
                            OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD,
                            &[t_f32, res, si, coord, val_map[dref], IMG_OP_LOD, lod_id],
                        );
                    }
                    val_map.insert(inst.id, res);
                }
                SsaOp::ImageGather {
                    slot,
                    dim,
                    u,
                    v,
                    w: wc,
                    comp,
                } => {
                    let di = dim_index(*dim);
                    let si = sampled_image(&mut w, &img_ids, di, *slot, &const_u32);
                    let coord =
                        sample_coord(&mut w, &cube_ids, *dim, val_map[u], val_map[v], val_map[wc]);
                    let texel = w.id();
                    w.inst(OP_IMAGE_GATHER, &[t_v4f, texel, si, coord, const_u32[comp]]);
                    let res = w.id();
                    w.inst(OP_BITCAST, &[t_uvec4, res, texel]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::ImageQuerySize { slot, dim, lod } => {
                    let di = dim_index(*dim);
                    let pi = w.id();
                    w.inst(
                        OP_ACCESS_CHAIN,
                        &[t_ptr_img[di], pi, tex_vars[di].unwrap(), const_u32[slot]],
                    );
                    let img = w.id();
                    w.inst(OP_LOAD, &[t_img[di], img, pi]);
                    let res = w.id();
                    if matches!(dim, ImageDim::D3 | ImageDim::D2Array) {
                        let sz = w.id();
                        w.inst(OP_IMAGE_QUERY_SIZE_LOD, &[t_uvec3, sz, img, val_map[lod]]);
                        let (sx, sy, sz3) = (w.id(), w.id(), w.id());
                        w.inst(OP_COMPOSITE_EXTRACT, &[t_u32, sx, sz, 0]);
                        w.inst(OP_COMPOSITE_EXTRACT, &[t_u32, sy, sz, 1]);
                        w.inst(OP_COMPOSITE_EXTRACT, &[t_u32, sz3, sz, 2]);
                        w.inst(OP_COMPOSITE_CONSTRUCT, &[t_uvec4, res, sx, sy, sz3, c_one]);
                    } else {
                        let sz2 = w.id();
                        w.inst(OP_IMAGE_QUERY_SIZE_LOD, &[t_uv2, sz2, img, val_map[lod]]);
                        let sx = w.id();
                        let sy = w.id();
                        w.inst(OP_COMPOSITE_EXTRACT, &[t_u32, sx, sz2, 0]);
                        w.inst(OP_COMPOSITE_EXTRACT, &[t_u32, sy, sz2, 1]);
                        w.inst(
                            OP_COMPOSITE_CONSTRUCT,
                            &[t_uvec4, res, sx, sy, c_one, c_one],
                        );
                    }
                    val_map.insert(inst.id, res);
                }
                SsaOp::BdaAtomic { op, addr, val, cmp } => {
                    let ptr_struct = w.id();
                    w.inst(
                        OP_CONVERT_U_TO_PTR,
                        &[t_bda_u32_ptr, ptr_struct, val_map[addr]],
                    );
                    let ptr_field = w.id();
                    w.inst(
                        OP_ACCESS_CHAIN,
                        &[t_bda_u32_field_ptr, ptr_field, ptr_struct, c_zero],
                    );
                    let res = w.id();
                    if *op == AtomicOp::CmpSwap {
                        let cmp_id = cmp.map_or(c_zero, |c| val_map[&c]);
                        w.inst(
                            OP_ATOMIC_COMPARE_EXCHANGE,
                            &[
                                t_u32,
                                res,
                                ptr_field,
                                c_one,
                                c_zero,
                                c_zero,
                                val_map[val],
                                cmp_id,
                            ],
                        );
                    } else {
                        let spv_op = match op {
                            AtomicOp::Swap => OP_ATOMIC_EXCHANGE,
                            AtomicOp::Add => OP_ATOMIC_I_ADD,
                            AtomicOp::Sub => OP_ATOMIC_I_SUB,
                            AtomicOp::SMin => OP_ATOMIC_S_MIN,
                            AtomicOp::UMin => OP_ATOMIC_U_MIN,
                            AtomicOp::SMax => OP_ATOMIC_S_MAX,
                            AtomicOp::UMax => OP_ATOMIC_U_MAX,
                            AtomicOp::And => OP_ATOMIC_AND,
                            AtomicOp::Or => OP_ATOMIC_OR,
                            AtomicOp::Xor => OP_ATOMIC_XOR,
                            AtomicOp::CmpSwap => unreachable!(),
                        };
                        w.inst(
                            spv_op,
                            &[t_u32, res, ptr_field, c_one, c_zero, val_map[val]],
                        );
                    }
                    val_map.insert(inst.id, res);
                }
                SsaOp::WaveShuffle { val, lane } => {
                    let res = w.id();
                    w.inst(
                        OP_GROUP_NON_UNIFORM_SHUFFLE,
                        &[t_u32, res, c_scope_subgroup, val_map[val], val_map[lane]],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::ImageFetch {
                    slot,
                    dim,
                    x,
                    y,
                    z,
                    lod,
                } => {
                    let di = dim_index(*dim);
                    let pi = w.id();
                    w.inst(
                        OP_ACCESS_CHAIN,
                        &[t_ptr_img[di], pi, tex_vars[di].unwrap(), const_u32[slot]],
                    );
                    let img = w.id();
                    w.inst(OP_LOAD, &[t_img[di], img, pi]);
                    let xy = w.id();
                    if *dim == ImageDim::D2 {
                        w.inst(OP_COMPOSITE_CONSTRUCT, &[t_uv2, xy, val_map[x], val_map[y]]);
                    } else {
                        w.inst(
                            OP_COMPOSITE_CONSTRUCT,
                            &[t_uvec3, xy, val_map[x], val_map[y], val_map[z]],
                        );
                    }
                    let texel = w.id();
                    let lod_id = lod.map_or(c_zero, |l| val_map[&l]);
                    w.inst(OP_IMAGE_FETCH, &[t_v4f, texel, img, xy, IMG_OP_LOD, lod_id]);
                    let res = w.id();
                    w.inst(OP_BITCAST, &[t_uvec4, res, texel]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::ImageStore {
                    slot,
                    dim,
                    x,
                    y,
                    z,
                    values,
                } => {
                    let di = dim_index(dim.for_storage());
                    let pi = w.id();
                    w.inst(
                        OP_ACCESS_CHAIN,
                        &[t_ptr_simg[di], pi, simg_vars[di].unwrap(), const_u32[slot]],
                    );
                    let img = w.id();
                    w.inst(OP_LOAD, &[t_simg[di], img, pi]);
                    let xy = w.id();
                    if dim.for_storage() == ImageDim::D2 {
                        w.inst(OP_COMPOSITE_CONSTRUCT, &[t_uv2, xy, val_map[x], val_map[y]]);
                    } else {
                        w.inst(
                            OP_COMPOSITE_CONSTRUCT,
                            &[t_uvec3, xy, val_map[x], val_map[y], val_map[z]],
                        );
                    }
                    let bits = w.id();
                    w.inst(
                        OP_COMPOSITE_CONSTRUCT,
                        &[
                            t_uvec4,
                            bits,
                            val_map[&values[0]],
                            val_map[&values[1]],
                            val_map[&values[2]],
                            val_map[&values[3]],
                        ],
                    );
                    let texel = w.id();
                    w.inst(OP_BITCAST, &[t_v4f, texel, bits]);
                    w.inst(OP_IMAGE_WRITE, &[img, xy, texel]);
                    val_map.insert(inst.id, c_zero);
                }
                SsaOp::WaveAny(pred) => {
                    let res = w.id();
                    w.inst(
                        OP_GROUP_NON_UNIFORM_ANY,
                        &[t_bool, res, c_scope_subgroup, val_map[pred]],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::LdsAtomicAddU32 { byte_addr, val } => {
                    let lds = lds_var.unwrap();
                    let dw_idx = w.id();
                    w.inst(
                        OP_SHIFT_RIGHT_LOGICAL,
                        &[t_u32, dw_idx, val_map[byte_addr], c_two],
                    );
                    let ptr = w.id();
                    w.inst(OP_ACCESS_CHAIN, &[t_ptr_wg_u32, ptr, lds, dw_idx]);
                    let res = w.id();
                    // Scope Workgroup, relaxed: DS atomics carry no ordering.
                    w.inst(
                        OP_ATOMIC_I_ADD,
                        &[t_u32, res, ptr, c_scope_wg, c_zero, val_map[val]],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::QuadBroadcast { val, lane } => {
                    let res = w.id();
                    let lane_id = const_u32[&u32::from(*lane)];
                    w.inst(
                        OP_GROUP_NON_UNIFORM_QUAD_BROADCAST,
                        &[t_u32, res, c_scope_subgroup, val_map[val], lane_id],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::QuadSwap { val, direction } => {
                    let dir_id = const_u32[&((*direction).min(2) as u32)];
                    let res = w.id();
                    w.inst(
                        OP_GROUP_NON_UNIFORM_QUAD_SWAP,
                        &[t_u32, res, c_scope_subgroup, val_map[val], dir_id],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::WorkgroupBarrier => {
                    w.inst(OP_CONTROL_BARRIER, &[c_scope_wg, c_scope_wg, c_sem_wg]);
                    val_map.insert(inst.id, c_zero);
                }
                SsaOp::LdsLoadU32 { byte_addr } => {
                    let lds = lds_var.unwrap();
                    let dw_idx = w.id();
                    w.inst(
                        OP_SHIFT_RIGHT_LOGICAL,
                        &[t_u32, dw_idx, val_map[byte_addr], c_two],
                    );
                    let ptr = w.id();
                    w.inst(OP_ACCESS_CHAIN, &[t_ptr_wg_u32, ptr, lds, dw_idx]);
                    let res = w.id();
                    w.inst(OP_LOAD, &[t_u32, res, ptr]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::LdsStoreU32 { byte_addr, val } => {
                    let lds = lds_var.unwrap();
                    let dw_idx = w.id();
                    w.inst(
                        OP_SHIFT_RIGHT_LOGICAL,
                        &[t_u32, dw_idx, val_map[byte_addr], c_two],
                    );
                    let ptr = w.id();
                    w.inst(OP_ACCESS_CHAIN, &[t_ptr_wg_u32, ptr, lds, dw_idx]);
                    w.inst(OP_STORE, &[ptr, val_map[val]]);
                    val_map.insert(inst.id, c_zero);
                }
                SsaOp::BdaLoad {
                    addr,
                    byte_offset,
                    count,
                } => {
                    let ea = if *byte_offset == 0 {
                        val_map[addr]
                    } else {
                        let sum = w.id();
                        w.inst(
                            OP_I_ADD,
                            &[t_u64, sum, val_map[addr], const_u64[&(*byte_offset as u64)]],
                        );
                        sum
                    };
                    if *count == 4 {
                        let ptr = w.id();
                        w.inst(OP_CONVERT_U_TO_PTR, &[t_bda_uvec4_ptr, ptr, ea]);
                        let field_ptr = w.id();
                        w.inst(
                            OP_ACCESS_CHAIN,
                            &[t_bda_uvec4_field_ptr, field_ptr, ptr, c_zero],
                        );
                        let res = w.id();
                        w.inst(OP_LOAD, &[t_uvec4, res, field_ptr, MEM_ALIGNED, 16]);
                        val_map.insert(inst.id, res);
                    } else {
                        let ptr = w.id();
                        w.inst(OP_CONVERT_U_TO_PTR, &[t_bda_u32_ptr, ptr, ea]);
                        let field_ptr = w.id();
                        w.inst(
                            OP_ACCESS_CHAIN,
                            &[t_bda_u32_field_ptr, field_ptr, ptr, c_zero],
                        );
                        let res = w.id();
                        w.inst(OP_LOAD, &[t_u32, res, field_ptr, MEM_ALIGNED, 4]);
                        val_map.insert(inst.id, res);
                    }
                }
                SsaOp::ExtractLane(v, lane) => {
                    let res = w.id();
                    w.inst(
                        OP_COMPOSITE_EXTRACT,
                        &[t_u32, res, val_map[v], *lane as u32],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::BdaStore {
                    addr,
                    byte_offset,
                    values,
                } => {
                    let ea = if *byte_offset == 0 {
                        val_map[addr]
                    } else {
                        let sum = w.id();
                        w.inst(
                            OP_I_ADD,
                            &[t_u64, sum, val_map[addr], const_u64[&(*byte_offset as u64)]],
                        );
                        sum
                    };
                    if values.len() == 4 {
                        let vec = w.id();
                        w.inst(
                            OP_COMPOSITE_CONSTRUCT,
                            &[
                                t_uvec4,
                                vec,
                                val_map[&values[0]],
                                val_map[&values[1]],
                                val_map[&values[2]],
                                val_map[&values[3]],
                            ],
                        );
                        let ptr = w.id();
                        w.inst(OP_CONVERT_U_TO_PTR, &[t_bda_uvec4_ptr, ptr, ea]);
                        let field_ptr = w.id();
                        w.inst(
                            OP_ACCESS_CHAIN,
                            &[t_bda_uvec4_field_ptr, field_ptr, ptr, c_zero],
                        );
                        w.inst(OP_STORE, &[field_ptr, vec, MEM_ALIGNED, 16]);
                    } else if let Some(first) = values.first() {
                        let ptr = w.id();
                        w.inst(OP_CONVERT_U_TO_PTR, &[t_bda_u32_ptr, ptr, ea]);
                        let field_ptr = w.id();
                        w.inst(
                            OP_ACCESS_CHAIN,
                            &[t_bda_u32_field_ptr, field_ptr, ptr, c_zero],
                        );
                        w.inst(OP_STORE, &[field_ptr, val_map[first], MEM_ALIGNED, 4]);
                    }
                }
                SsaOp::Bvh64IntersectBox32 {
                    node_addr,
                    t_max,
                    ox,
                    oy,
                    oz,
                    inv_dx,
                    inv_dy,
                    inv_dz,
                } => {
                    let base = val_map[node_addr];
                    let children_ptr = w.id();
                    w.inst(OP_CONVERT_U_TO_PTR, &[t_bda_uvec4_ptr, children_ptr, base]);
                    let children_fp = w.id();
                    w.inst(
                        OP_ACCESS_CHAIN,
                        &[t_bda_uvec4_field_ptr, children_fp, children_ptr, c_zero],
                    );
                    let children_vec = w.id();
                    w.inst(
                        OP_LOAD,
                        &[t_uvec4, children_vec, children_fp, MEM_ALIGNED, 16],
                    );

                    let mut out_ids = [0u32; 4];
                    let mut out_dists = [0u32; 4];
                    for child in 0..4u32 {
                        let cid = w.id();
                        w.inst(OP_COMPOSITE_EXTRACT, &[t_u32, cid, children_vec, child]);
                        let valid = w.id();
                        w.inst(OP_I_NOT_EQUAL, &[t_bool, valid, cid, c_invalid_node]);

                        let mut tmin = c_fzero;
                        let mut tmax = val_map[t_max];
                        let axes = [
                            (
                                val_map[ox],
                                val_map[inv_dx],
                                16u64 + (child as u64) * 12,
                                64u64 + (child as u64) * 12,
                            ),
                            (
                                val_map[oy],
                                val_map[inv_dy],
                                20u64 + (child as u64) * 12,
                                68u64 + (child as u64) * 12,
                            ),
                            (
                                val_map[oz],
                                val_map[inv_dz],
                                24u64 + (child as u64) * 12,
                                72u64 + (child as u64) * 12,
                            ),
                        ];
                        for (o, inv_d, min_off, max_off) in axes {
                            let load_f32 = |w: &mut Writer, off: u64| -> u32 {
                                let ea = w.id();
                                w.inst(OP_I_ADD, &[t_u64, ea, base, const_u64[&off]]);
                                let p = w.id();
                                w.inst(OP_CONVERT_U_TO_PTR, &[t_bda_u32_ptr, p, ea]);
                                let fp = w.id();
                                w.inst(OP_ACCESS_CHAIN, &[t_bda_u32_field_ptr, fp, p, c_zero]);
                                let u = w.id();
                                w.inst(OP_LOAD, &[t_u32, u, fp, MEM_ALIGNED, 4]);
                                let f = w.id();
                                w.inst(OP_BITCAST, &[t_f32, f, u]);
                                f
                            };
                            let bmin = load_f32(&mut w, min_off);
                            let bmax = load_f32(&mut w, max_off);
                            let d0 = w.id();
                            w.inst(OP_F_SUB, &[t_f32, d0, bmin, o]);
                            let t0 = w.id();
                            w.inst(OP_F_MUL, &[t_f32, t0, d0, inv_d]);
                            let d1 = w.id();
                            w.inst(OP_F_SUB, &[t_f32, d1, bmax, o]);
                            let t1 = w.id();
                            w.inst(OP_F_MUL, &[t_f32, t1, d1, inv_d]);
                            let lo = w.id();
                            w.inst(OP_EXT_INST, &[t_f32, lo, glsl_ext, GLSL_FMIN, t0, t1]);
                            let hi = w.id();
                            w.inst(OP_EXT_INST, &[t_f32, hi, glsl_ext, GLSL_FMAX, t0, t1]);
                            let next_min = w.id();
                            w.inst(
                                OP_EXT_INST,
                                &[t_f32, next_min, glsl_ext, GLSL_FMAX, tmin, lo],
                            );
                            let next_max = w.id();
                            w.inst(
                                OP_EXT_INST,
                                &[t_f32, next_max, glsl_ext, GLSL_FMIN, tmax, hi],
                            );
                            tmin = next_min;
                            tmax = next_max;
                        }
                        let slab_hit = w.id();
                        w.inst(OP_F_ORD_LESS_THAN_EQUAL, &[t_bool, slab_hit, tmin, tmax]);
                        let hit = w.id();
                        w.inst(OP_LOGICAL_AND, &[t_bool, hit, valid, slab_hit]);
                        let res_id = w.id();
                        w.inst(OP_SELECT, &[t_u32, res_id, hit, cid, c_invalid_node]);
                        let res_dist = w.id();
                        w.inst(OP_SELECT, &[t_f32, res_dist, hit, tmin, c_finf]);
                        out_ids[child as usize] = res_id;
                        out_dists[child as usize] = res_dist;
                    }
                    // 5-comparator sorting network (nearest hit first):
                    // (0,2), (1,3), (0,1), (2,3), (1,2)
                    for (a, b) in [(0usize, 2usize), (1, 3), (0, 1), (2, 3), (1, 2)] {
                        let swap = w.id();
                        w.inst(
                            OP_F_ORD_LESS_THAN,
                            &[t_bool, swap, out_dists[b], out_dists[a]],
                        );
                        let min_d = w.id();
                        w.inst(OP_SELECT, &[t_f32, min_d, swap, out_dists[b], out_dists[a]]);
                        let max_d = w.id();
                        w.inst(OP_SELECT, &[t_f32, max_d, swap, out_dists[a], out_dists[b]]);
                        let min_id = w.id();
                        w.inst(OP_SELECT, &[t_u32, min_id, swap, out_ids[b], out_ids[a]]);
                        let max_id = w.id();
                        w.inst(OP_SELECT, &[t_u32, max_id, swap, out_ids[a], out_ids[b]]);
                        out_dists[a] = min_d;
                        out_dists[b] = max_d;
                        out_ids[a] = min_id;
                        out_ids[b] = max_id;
                    }
                    let res = w.id();
                    w.inst(
                        OP_COMPOSITE_CONSTRUCT,
                        &[t_uvec4, res, out_ids[0], out_ids[1], out_ids[2], out_ids[3]],
                    );
                    val_map.insert(inst.id, res);
                }
                SsaOp::WavePermute { op, value, aux } => {
                    use crate::wave_ops::LanePermuteOp;
                    let v = val_map[value];
                    let res = w.id();
                    match op {
                        LanePermuteOp::ReadFirstLane => {
                            w.inst(
                                OP_GROUP_NON_UNIFORM_BROADCAST_FIRST,
                                &[t_u32, res, c_scope_subgroup, v],
                            );
                        }
                        LanePermuteOp::ReadLane { lane } => {
                            let l = const_u32.get(lane).copied().unwrap_or(c_zero);
                            w.inst(
                                OP_GROUP_NON_UNIFORM_SHUFFLE,
                                &[t_u32, res, c_scope_subgroup, v, l],
                            );
                        }
                        LanePermuteOp::DsSwizzle { offset16 } => {
                            let xor_mask = u32::from((offset16 >> 10) & 0x1F);
                            let m = const_u32.get(&xor_mask).copied().unwrap_or(c_zero);
                            w.inst(
                                OP_GROUP_NON_UNIFORM_SHUFFLE_XOR,
                                &[t_u32, res, c_scope_subgroup, v, m],
                            );
                        }
                        _ => {
                            let lane_id = aux.map_or(c_zero, |a| val_map[&a]);
                            w.inst(
                                OP_GROUP_NON_UNIFORM_SHUFFLE,
                                &[t_u32, res, c_scope_subgroup, v, lane_id],
                            );
                        }
                    }
                    val_map.insert(inst.id, res);
                }
                SsaOp::SpillLoad { slot } => {
                    let sv = spill_var.expect("spill_var must exist for SpillLoad");
                    let t_ptr_el = t_ptr_func_u32.expect("t_ptr_func_u32");
                    let ptr = w.id();
                    w.inst(OP_ACCESS_CHAIN, &[t_ptr_el, ptr, sv, const_u32[slot]]);
                    let res = w.id();
                    w.inst(OP_LOAD, &[t_u32, res, ptr]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::SpillStore { slot, val } => {
                    let sv = spill_var.expect("spill_var must exist for SpillStore");
                    let t_ptr_el = t_ptr_func_u32.expect("t_ptr_func_u32");
                    let ptr = w.id();
                    w.inst(OP_ACCESS_CHAIN, &[t_ptr_el, ptr, sv, const_u32[slot]]);
                    w.inst(OP_STORE, &[ptr, val_map[val]]);
                    val_map.insert(inst.id, c_zero);
                }
            }
        }

        // Emit block terminator
        match &block.term {
            Terminator::Return => w.inst(OP_RETURN, &[]),
            Terminator::Branch(target) => w.inst(OP_BRANCH, &[block_labels[*target]]),
            Terminator::BranchCond {
                cond,
                true_block,
                false_block,
                merge_block,
            } => {
                w.inst(OP_SELECTION_MERGE, &[block_labels[*merge_block], 0]);
                w.inst(
                    OP_BRANCH_CONDITIONAL,
                    &[
                        val_map[cond],
                        block_labels[*true_block],
                        block_labels[*false_block],
                    ],
                );
            }
            Terminator::Loop {
                cond,
                body,
                merge,
                continue_block,
            } => {
                w.inst(
                    OP_LOOP_MERGE,
                    &[block_labels[*merge], block_labels[*continue_block], 0],
                );
                match cond {
                    Some(c) => w.inst(
                        OP_BRANCH_CONDITIONAL,
                        &[val_map[c], block_labels[*body], block_labels[*merge]],
                    ),
                    None => w.inst(OP_BRANCH, &[block_labels[*body]]),
                }
            }
            Terminator::BackEdge { cond, header, exit } => match cond {
                Some(c) => w.inst(
                    OP_BRANCH_CONDITIONAL,
                    &[val_map[c], block_labels[*header], block_labels[*exit]],
                ),
                None => w.inst(OP_BRANCH, &[block_labels[*header]]),
            },
            Terminator::Switch {
                selector,
                default,
                targets,
                merge_block,
            } => {
                w.inst(OP_SELECTION_MERGE, &[block_labels[*merge_block], 0]);
                let mut ops = Vec::with_capacity(2 + targets.len() * 2);
                ops.push(val_map[selector]);
                ops.push(block_labels[*default]);
                for &(lit, target_blk) in targets {
                    ops.push(lit);
                    ops.push(block_labels[target_blk]);
                }
                w.inst(OP_SWITCH, &ops);
            }
        }
    }

    for (word, val_id) in phi_patches {
        w.body[word] = val_map[&val_id];
    }

    w.inst(OP_FUNCTION_END, &[]);
    w.finish()
}

/// Reverse post-order from the entry block: every block after the blocks
/// that dominate it, as SPIR-V requires (a loop's merge after its latch).
/// Unreachable blocks follow.
fn block_order(module: &SsaModule) -> Vec<usize> {
    let n = module.blocks.len();
    let mut seen = vec![false; n];
    let mut post = Vec::with_capacity(n);
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    seen[0] = true;
    while let Some((b, i)) = stack.pop() {
        let succ = module.blocks[b].term.successors();
        if let Some(&next) = succ.get(i) {
            stack.push((b, i + 1));
            if !seen[next] {
                seen[next] = true;
                stack.push((next, 0));
            }
        } else {
            post.push(b);
        }
    }
    post.reverse();
    post.extend((0..n).filter(|&b| !seen[b]));
    post
}

/// Type and variable ids the sampled-image accesses need.
struct ImgIds {
    t_img: [u32; 4],
    t_ptr_img: [u32; 4],
    t_sampled: [u32; 4],
    tex_vars: [Option<u32>; 4],
    t_sampler: u32,
    t_ptr_sampler: u32,
    samp_var: Option<u32>,
}

/// `textures*[slot]` of class `di` combined with `samplers[slot]`.
fn sampled_image(
    w: &mut Writer,
    ids: &ImgIds,
    di: usize,
    slot: u32,
    const_u32: &HashMap<u32, u32>,
) -> u32 {
    let pi = w.id();
    w.inst(
        OP_ACCESS_CHAIN,
        &[
            ids.t_ptr_img[di],
            pi,
            ids.tex_vars[di].unwrap(),
            const_u32[&slot],
        ],
    );
    let img = w.id();
    w.inst(OP_LOAD, &[ids.t_img[di], img, pi]);
    let ps = w.id();
    w.inst(
        OP_ACCESS_CHAIN,
        &[
            ids.t_ptr_sampler,
            ps,
            ids.samp_var.unwrap(),
            const_u32[&slot],
        ],
    );
    let smp = w.id();
    w.inst(OP_LOAD, &[ids.t_sampler, smp, ps]);
    let si = w.id();
    w.inst(OP_SAMPLED_IMAGE, &[ids.t_sampled[di], si, img, smp]);
    si
}

/// Types and constants the coordinate builders use.
struct CoordIds {
    t_f32: u32,
    t_u32: u32,
    t_bool: u32,
    t_v2f: u32,
    t_v3f: u32,
    c_half3: u32,
    c_ftwo: u32,
    c_fone: u32,
    c_fmone: u32,
    /// u32 constants 0, 1, 2, 3, 4, 5 and 7.
    c_u: [u32; 7],
}

/// The coordinate vector a sample of class `dim` takes from `(u, v, w)`.
///
/// A cube address on RDNA2 is face-relative: `s, t` in `[1, 2]` (the
/// `v_cubesc / |v_cubema| + 1.5` convention: `v_cubema` returns twice the
/// major axis) and the face id. SPIR-V samples cubes by direction, so the
/// direction is rebuilt from the face's (sc, tc, ma) assignment:
/// +X (-z, -y), -X (+z, -y), +Y (+x, +z), -Y (+x, -z), +Z (+x, -y), -Z (-x, -y).
fn sample_coord(w: &mut Writer, c: &CoordIds, dim: ImageDim, u: u32, v: u32, wc: u32) -> u32 {
    match dim {
        ImageDim::D2 => {
            let uv = w.id();
            w.inst(OP_COMPOSITE_CONSTRUCT, &[c.t_v2f, uv, u, v]);
            uv
        }
        ImageDim::D3 | ImageDim::D2Array => {
            let uvw = w.id();
            w.inst(OP_COMPOSITE_CONSTRUCT, &[c.t_v3f, uvw, u, v, wc]);
            uvw
        }
        ImageDim::Cube => {
            let unbias = |w: &mut Writer, x: u32| {
                let d = w.id();
                w.inst(OP_F_SUB, &[c.t_f32, d, x, c.c_half3]);
                let m = w.id();
                w.inst(OP_F_MUL, &[c.t_f32, m, d, c.c_ftwo]);
                m
            };
            let a = unbias(w, u);
            let b = unbias(w, v);
            let neg = |w: &mut Writer, x: u32| {
                let n = w.id();
                w.inst(OP_F_MUL, &[c.t_f32, n, x, c.c_fmone]);
                n
            };
            let na = neg(w, a);
            let nb = neg(w, b);
            let fi = w.id();
            w.inst(OP_CONVERT_F_TO_U, &[c.t_u32, fi, wc]);
            let face = w.id();
            w.inst(OP_BITWISE_AND, &[c.t_u32, face, fi, c.c_u[6]]);
            let is = |w: &mut Writer, k: usize| {
                let e = w.id();
                w.inst(OP_I_EQUAL, &[c.t_bool, e, face, c.c_u[k]]);
                e
            };
            let f: [u32; 6] = std::array::from_fn(|k| is(w, k));
            let sel = |w: &mut Writer, cond: u32, t: u32, e: u32| {
                let r = w.id();
                w.inst(OP_SELECT, &[c.t_f32, r, cond, t, e]);
                r
            };
            // x: +X 1, -X -1, -Z -a, else a.
            let x = sel(w, f[5], na, a);
            let x = sel(w, f[1], c.c_fmone, x);
            let x = sel(w, f[0], c.c_fone, x);
            // y: +Y 1, -Y -1, else -b.
            let y = sel(w, f[3], c.c_fmone, nb);
            let y = sel(w, f[2], c.c_fone, y);
            // z: +X -a, -X a, +Y b, -Y -b, +Z 1, -Z -1.
            let z = sel(w, f[4], c.c_fone, c.c_fmone);
            let z = sel(w, f[3], nb, z);
            let z = sel(w, f[2], b, z);
            let z = sel(w, f[1], a, z);
            let z = sel(w, f[0], na, z);
            let dir = w.id();
            w.inst(OP_COMPOSITE_CONSTRUCT, &[c.t_v3f, dir, x, y, z]);
            dir
        }
    }
}

struct Writer {
    next_id: u32,
    body: Vec<u32>,
}

impl Writer {
    fn new() -> Self {
        Self {
            next_id: 1,
            body: Vec::new(),
        }
    }

    fn id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn inst(&mut self, op: u16, operands: &[u32]) {
        let count = (1 + operands.len()) as u32;
        self.body.push((count << 16) | (op as u32));
        self.body.extend_from_slice(operands);
    }

    /// An instruction whose only operand is a literal string.
    fn inst_str_plain(&mut self, op: u16, s: &str) {
        let mut ops = Vec::new();
        encode_str(&mut ops, s);
        self.inst(op, &ops);
    }

    fn inst_str(&mut self, op: u16, res_id: u32, s: &str) {
        let mut ops = vec![res_id];
        encode_str(&mut ops, s);
        self.inst(op, &ops);
    }

    fn finish(self) -> Vec<u32> {
        let mut out = Vec::with_capacity(5 + self.body.len());
        out.push(SPIRV_MAGIC);
        out.push(SPIRV_VERSION_1_6);
        out.push(0); // Generator magic
        out.push(self.next_id);
        out.push(0); // Schema
        out.extend_from_slice(&self.body);
        out
    }
}

fn encode_str(dst: &mut Vec<u32>, s: &str) {
    let bytes = s.as_bytes();
    let total = bytes.len() + 1;
    let words = total.div_ceil(4);
    for wi in 0..words {
        let mut w = 0u32;
        for bi in 0..4 {
            let idx = wi * 4 + bi;
            if idx < bytes.len() {
                w |= (bytes[idx] as u32) << (bi * 8);
            }
        }
        dst.push(w);
    }
}
