//! Direct SPIR-V 1.6 binary writer targeting No-API Vulkan (`PhysicalStorageBuffer64`).
//!
//! Emits `spirv-val`-clean compute shader modules directly from Strata's SSA IR:
//! - 128-byte Push Constant block (`uint sgpr[32]`) for console User SGPRs
//! - 64-bit `PhysicalStorageBuffer` pointer loads/stores (scalar & 128-bit coalesced)
//! - Structured multi-block CFG (`OpSelectionMerge` + `OpBranchConditional` + `OpPhi`)
//! - Subgroup-size-agnostic Wave64 ballots (`OpGroupNonUniformBallot` + Workgroup `OpAtomicOr`)
//! - Workgroup shared memory (LDS) + `OpControlBarrier`
//! - Inline watertight BVH Box32 ray-AABB intersection

use std::collections::HashMap;

use crate::rdna2::{CmpOp, CmpType};
use crate::ssa::{SsaModule, SsaOp, Terminator, Ty, ValueId};

// SPIR-V header constants
const SPIRV_MAGIC: u32 = 0x0723_0203;
const SPIRV_VERSION_1_6: u32 = 0x0001_0600;

// SPIR-V opcodes
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
const OP_BIT_FIELD_U_EXTRACT: u16 = 202;

const OP_CONTROL_BARRIER: u16 = 224;
const OP_ATOMIC_OR: u16 = 241;
const OP_PHI: u16 = 245;
const OP_SELECTION_MERGE: u16 = 247;
const OP_LABEL: u16 = 248;
const OP_BRANCH: u16 = 249;
const OP_BRANCH_CONDITIONAL: u16 = 250;
const OP_RETURN: u16 = 253;

const OP_GROUP_NON_UNIFORM_BROADCAST_FIRST: u16 = 338;
const OP_GROUP_NON_UNIFORM_BALLOT: u16 = 339;
const OP_GROUP_NON_UNIFORM_QUAD_SWAP: u16 = 366;

// Capabilities
const CAP_SHADER: u32 = 1;
const CAP_INT64: u32 = 11;
const CAP_GROUP_NON_UNIFORM: u32 = 61;
const CAP_GROUP_NON_UNIFORM_BALLOT: u32 = 64;
const CAP_GROUP_NON_UNIFORM_QUAD: u32 = 68;
const CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES: u32 = 5347;

// Storage classes
const SC_INPUT: u32 = 1;
const SC_WORKGROUP: u32 = 4;
const SC_PUSH_CONSTANT: u32 = 9;
const SC_PHYSICAL_STORAGE_BUFFER: u32 = 5349;

// Decorations
const DEC_BLOCK: u32 = 2;
const DEC_ARRAY_STRIDE: u32 = 6;
const DEC_BUILTIN: u32 = 11;
const DEC_OFFSET: u32 = 35;

// BuiltIns
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;
const BUILTIN_LOCAL_INVOCATION_INDEX: u32 = 29;
const BUILTIN_SUBGROUP_SIZE: u32 = 36;

// Memory operands
const MEM_ALIGNED: u32 = 0x2;

// GLSL.std.450 opcodes
const GLSL_FLOOR: u32 = 8;
const GLSL_FRACT: u32 = 10;
const GLSL_EXP2: u32 = 29;
const GLSL_LOG2: u32 = 30;
const GLSL_SQRT: u32 = 31;
const GLSL_INV_SQRT: u32 = 32;
const GLSL_FMIN: u32 = 37;
const GLSL_FMAX: u32 = 40;
const GLSL_FMA: u32 = 50;

pub fn emit_spirv(module: &SsaModule) -> Vec<u32> {
    let mut w = Writer::new();

    // Capabilities & Memory Model
    w.inst(OP_CAPABILITY, &[CAP_SHADER]);
    w.inst(OP_CAPABILITY, &[CAP_INT64]);
    w.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM]);
    w.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM_BALLOT]);
    w.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM_QUAD]);
    w.inst(OP_CAPABILITY, &[CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES]);

    let glsl_ext = w.id();
    w.inst_str(OP_EXT_INST_IMPORT, glsl_ext, "GLSL.std.450");

    // AddressingModel = PhysicalStorageBuffer64 (5348), MemoryModel = GLSL450 (1)
    w.inst(OP_MEMORY_MODEL, &[5348, 1]);

    let main_fn = w.id();
    let global_id_var = w.id();
    let push_var = w.id();
    let local_idx_var = if module.uses_wave64_ballot { Some(w.id()) } else { None };
    let sg_size_var = if module.uses_wave64_ballot { Some(w.id()) } else { None };
    let ballot_shared_var = if module.uses_wave64_ballot { Some(w.id()) } else { None };
    let lds_var = if module.uses_lds { Some(w.id()) } else { None };

    // EntryPoint GLCompute (5) "main" ...
    let mut ep = vec![5, main_fn];
    encode_str(&mut ep, "main");
    ep.push(global_id_var);
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
    w.inst(OP_ENTRY_POINT, &ep);

    // ExecutionMode LocalSize (17)
    w.inst(
        OP_EXECUTION_MODE,
        &[
            main_fn,
            17,
            module.workgroup_size[0],
            module.workgroup_size[1],
            module.workgroup_size[2],
        ],
    );

    // Types
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

    // PhysicalStorageBuffer pointer types (scalar u32 & 128-bit uvec4)
    let t_bda_u32_struct = w.id();
    let t_bda_u32_ptr = w.id();
    let t_bda_u32_field_ptr = w.id();
    let t_bda_uvec4_struct = w.id();
    let t_bda_uvec4_ptr = w.id();
    let t_bda_uvec4_field_ptr = w.id();

    // Decorations
    w.inst(OP_DECORATE, &[global_id_var, DEC_BUILTIN, BUILTIN_GLOBAL_INVOCATION_ID]);
    if let Some(v) = local_idx_var {
        w.inst(OP_DECORATE, &[v, DEC_BUILTIN, BUILTIN_LOCAL_INVOCATION_INDEX]);
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
    w.inst(OP_TYPE_POINTER, &[t_ptr_push_struct, SC_PUSH_CONSTANT, t_push_struct]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_push_u32, SC_PUSH_CONSTANT, t_u32]);

    w.inst(OP_TYPE_POINTER, &[t_ptr_in_uvec3, SC_INPUT, t_uvec3]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_in_u32, SC_INPUT, t_u32]);

    w.inst(OP_CONSTANT, &[t_u32, c_2, 2]);
    w.inst(OP_TYPE_ARRAY, &[t_ballot_arr, t_u32, c_2]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_wg_ballot_arr, SC_WORKGROUP, t_ballot_arr]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_wg_u32, SC_WORKGROUP, t_u32]);

    w.inst(OP_CONSTANT, &[t_u32, c_16384, 16384]);
    w.inst(OP_TYPE_ARRAY, &[t_lds_arr, t_u32, c_16384]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_wg_lds_arr, SC_WORKGROUP, t_lds_arr]);

    w.inst(OP_TYPE_STRUCT, &[t_bda_u32_struct, t_u32]);
    w.inst(OP_TYPE_POINTER, &[t_bda_u32_ptr, SC_PHYSICAL_STORAGE_BUFFER, t_bda_u32_struct]);
    w.inst(OP_TYPE_POINTER, &[t_bda_u32_field_ptr, SC_PHYSICAL_STORAGE_BUFFER, t_u32]);

    w.inst(OP_TYPE_STRUCT, &[t_bda_uvec4_struct, t_uvec4]);
    w.inst(OP_TYPE_POINTER, &[t_bda_uvec4_ptr, SC_PHYSICAL_STORAGE_BUFFER, t_bda_uvec4_struct]);
    w.inst(OP_TYPE_POINTER, &[t_bda_uvec4_field_ptr, SC_PHYSICAL_STORAGE_BUFFER, t_uvec4]);

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
                _ => {}
            }
        }
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
    w.inst(OP_VARIABLE, &[t_ptr_in_uvec3, global_id_var, SC_INPUT]);
    w.inst(OP_VARIABLE, &[t_ptr_push_struct, push_var, SC_PUSH_CONSTANT]);
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

    // Function body with multi-block labels
    w.inst(OP_FUNCTION, &[t_void, main_fn, 0, t_fn]);
    let block_labels: Vec<u32> = (0..module.blocks.len()).map(|_| w.id()).collect();
    let mut val_map: HashMap<ValueId, u32> = HashMap::new();

    for (bi, block) in module.blocks.iter().enumerate() {
        w.inst(OP_LABEL, &[block_labels[bi]]);

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
                for &(pred_blk, val_id) in incoming {
                    ops.push(val_map[&val_id]);
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
                    w.inst(OP_ACCESS_CHAIN, &[t_ptr_push_u32, ptr, push_var, c_zero, const_u32[&(*idx as u32)]]);
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
                    w.inst(OP_SHIFT_RIGHT_LOGICAL, &[t_u64, shifted, val_map[a], c_32_u64]);
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
                    w.inst(OP_SHIFT_RIGHT_LOGICAL, &[t_u32, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::AshrI32(a, b) => {
                    let res = w.id();
                    w.inst(OP_SHIFT_RIGHT_ARITHMETIC, &[t_u32, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::BfeU32 { src, offset, width } => {
                    let res = w.id();
                    w.inst(OP_BIT_FIELD_U_EXTRACT, &[t_u32, res, val_map[src], val_map[offset], val_map[width]]);
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
                    w.inst(OP_EXT_INST, &[t_f32, res, glsl_ext, GLSL_FMIN, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::MaxF32(a, b) => {
                    let res = w.id();
                    w.inst(OP_EXT_INST, &[t_f32, res, glsl_ext, GLSL_FMAX, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::FmaF32(a, b, c) => {
                    let res = w.id();
                    w.inst(OP_EXT_INST, &[t_f32, res, glsl_ext, GLSL_FMA, val_map[a], val_map[b], val_map[c]]);
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
                    w.inst(OP_EXT_INST, &[t_f32, res, glsl_ext, GLSL_INV_SQRT, val_map[a]]);
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
                    w.inst(OP_SELECT, &[t_u32, res, val_map[cond], val_map[t], val_map[f]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::WaveReadFirstLane(a) => {
                    let res = w.id();
                    w.inst(OP_GROUP_NON_UNIFORM_BROADCAST_FIRST, &[t_u32, res, c_scope_subgroup, val_map[a]]);
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
                    w.inst(OP_U_GREATER_THAN_EQUAL, &[t_bool, is_w64, sg_size, const_u32[&64]]);

                    let half_idx = w.id();
                    w.inst(OP_SHIFT_RIGHT_LOGICAL, &[t_u32, half_idx, tid, const_u32[&5]]);
                    let p_slot = w.id();
                    w.inst(OP_ACCESS_CHAIN, &[t_ptr_wg_u32, p_slot, svar, half_idx]);
                    w.inst(OP_STORE, &[p_slot, c_zero]);
                    w.inst(OP_CONTROL_BARRIER, &[c_scope_wg, c_scope_wg, c_sem_wg]);

                    let ballot_v4 = w.id();
                    w.inst(OP_GROUP_NON_UNIFORM_BALLOT, &[t_uvec4, ballot_v4, c_scope_subgroup, val_map[pred]]);
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
                    w.inst(OP_BITWISE_AND, &[t_u32, sub_shift, tid_in_half, sg_align_mask]);
                    let shifted_mask = w.id();
                    w.inst(OP_SHIFT_LEFT_LOGICAL, &[t_u32, shifted_mask, half_mask, sub_shift]);
                    let atomic_old = w.id();
                    w.inst(OP_ATOMIC_OR, &[t_u32, atomic_old, p_slot, c_scope_wg, c_zero, shifted_mask]);

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
                SsaOp::QuadSwap { val, direction } => {
                    let dir_id = const_u32[&((*direction).min(2) as u32)];
                    let res = w.id();
                    w.inst(OP_GROUP_NON_UNIFORM_QUAD_SWAP, &[t_u32, res, c_scope_subgroup, val_map[val], dir_id]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::WorkgroupBarrier => {
                    w.inst(OP_CONTROL_BARRIER, &[c_scope_wg, c_scope_wg, c_sem_wg]);
                    val_map.insert(inst.id, c_zero);
                }
                SsaOp::LdsLoadU32 { byte_addr } => {
                    let lds = lds_var.unwrap();
                    let dw_idx = w.id();
                    w.inst(OP_SHIFT_RIGHT_LOGICAL, &[t_u32, dw_idx, val_map[byte_addr], c_two]);
                    let ptr = w.id();
                    w.inst(OP_ACCESS_CHAIN, &[t_ptr_wg_u32, ptr, lds, dw_idx]);
                    let res = w.id();
                    w.inst(OP_LOAD, &[t_u32, res, ptr]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::LdsStoreU32 { byte_addr, val } => {
                    let lds = lds_var.unwrap();
                    let dw_idx = w.id();
                    w.inst(OP_SHIFT_RIGHT_LOGICAL, &[t_u32, dw_idx, val_map[byte_addr], c_two]);
                    let ptr = w.id();
                    w.inst(OP_ACCESS_CHAIN, &[t_ptr_wg_u32, ptr, lds, dw_idx]);
                    w.inst(OP_STORE, &[ptr, val_map[val]]);
                    val_map.insert(inst.id, c_zero);
                }
                SsaOp::BdaLoad { addr, byte_offset, count } => {
                    let ea = if *byte_offset == 0 {
                        val_map[addr]
                    } else {
                        let sum = w.id();
                        w.inst(OP_I_ADD, &[t_u64, sum, val_map[addr], const_u64[&(*byte_offset as u64)]]);
                        sum
                    };
                    if *count == 4 {
                        let ptr = w.id();
                        w.inst(OP_CONVERT_U_TO_PTR, &[t_bda_uvec4_ptr, ptr, ea]);
                        let field_ptr = w.id();
                        w.inst(OP_ACCESS_CHAIN, &[t_bda_uvec4_field_ptr, field_ptr, ptr, c_zero]);
                        let res = w.id();
                        w.inst(OP_LOAD, &[t_uvec4, res, field_ptr, MEM_ALIGNED, 16]);
                        val_map.insert(inst.id, res);
                    } else {
                        let ptr = w.id();
                        w.inst(OP_CONVERT_U_TO_PTR, &[t_bda_u32_ptr, ptr, ea]);
                        let field_ptr = w.id();
                        w.inst(OP_ACCESS_CHAIN, &[t_bda_u32_field_ptr, field_ptr, ptr, c_zero]);
                        let res = w.id();
                        w.inst(OP_LOAD, &[t_u32, res, field_ptr, MEM_ALIGNED, 4]);
                        val_map.insert(inst.id, res);
                    }
                }
                SsaOp::ExtractLane(v, lane) => {
                    let res = w.id();
                    w.inst(OP_COMPOSITE_EXTRACT, &[t_u32, res, val_map[v], *lane as u32]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::BdaStore { addr, byte_offset, values } => {
                    let ea = if *byte_offset == 0 {
                        val_map[addr]
                    } else {
                        let sum = w.id();
                        w.inst(OP_I_ADD, &[t_u64, sum, val_map[addr], const_u64[&(*byte_offset as u64)]]);
                        sum
                    };
                    if values.len() == 4 {
                        let vec = w.id();
                        w.inst(
                            OP_COMPOSITE_CONSTRUCT,
                            &[t_uvec4, vec, val_map[&values[0]], val_map[&values[1]], val_map[&values[2]], val_map[&values[3]]],
                        );
                        let ptr = w.id();
                        w.inst(OP_CONVERT_U_TO_PTR, &[t_bda_uvec4_ptr, ptr, ea]);
                        let field_ptr = w.id();
                        w.inst(OP_ACCESS_CHAIN, &[t_bda_uvec4_field_ptr, field_ptr, ptr, c_zero]);
                        w.inst(OP_STORE, &[field_ptr, vec, MEM_ALIGNED, 16]);
                    } else if let Some(first) = values.first() {
                        let ptr = w.id();
                        w.inst(OP_CONVERT_U_TO_PTR, &[t_bda_u32_ptr, ptr, ea]);
                        let field_ptr = w.id();
                        w.inst(OP_ACCESS_CHAIN, &[t_bda_u32_field_ptr, field_ptr, ptr, c_zero]);
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
                    w.inst(OP_ACCESS_CHAIN, &[t_bda_uvec4_field_ptr, children_fp, children_ptr, c_zero]);
                    let children_vec = w.id();
                    w.inst(OP_LOAD, &[t_uvec4, children_vec, children_fp, MEM_ALIGNED, 16]);

                    let mut out_ids = [0u32; 4];
                    for child in 0..4u32 {
                        let cid = w.id();
                        w.inst(OP_COMPOSITE_EXTRACT, &[t_u32, cid, children_vec, child]);
                        let valid = w.id();
                        w.inst(OP_I_NOT_EQUAL, &[t_bool, valid, cid, c_invalid_node]);

                        let mut tmin = c_fzero;
                        let mut tmax = val_map[t_max];
                        let axes = [
                            (val_map[ox], val_map[inv_dx], 16u64 + (child as u64) * 12, 64u64 + (child as u64) * 12),
                            (val_map[oy], val_map[inv_dy], 20u64 + (child as u64) * 12, 68u64 + (child as u64) * 12),
                            (val_map[oz], val_map[inv_dz], 24u64 + (child as u64) * 12, 72u64 + (child as u64) * 12),
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
                            w.inst(OP_EXT_INST, &[t_f32, next_min, glsl_ext, GLSL_FMAX, tmin, lo]);
                            let next_max = w.id();
                            w.inst(OP_EXT_INST, &[t_f32, next_max, glsl_ext, GLSL_FMIN, tmax, hi]);
                            tmin = next_min;
                            tmax = next_max;
                        }
                        let slab_hit = w.id();
                        w.inst(OP_F_ORD_LESS_THAN_EQUAL, &[t_bool, slab_hit, tmin, tmax]);
                        let hit = w.id();
                        w.inst(OP_LOGICAL_AND, &[t_bool, hit, valid, slab_hit]);
                        let res_id = w.id();
                        w.inst(OP_SELECT, &[t_u32, res_id, hit, cid, c_invalid_node]);
                        out_ids[child as usize] = res_id;
                    }
                    let res = w.id();
                    w.inst(OP_COMPOSITE_CONSTRUCT, &[t_uvec4, res, out_ids[0], out_ids[1], out_ids[2], out_ids[3]]);
                    val_map.insert(inst.id, res);
                }
            }
        }

        // Emit block terminator
        match block.term {
            Terminator::Return => w.inst(OP_RETURN, &[]),
            Terminator::Branch(target) => w.inst(OP_BRANCH, &[block_labels[target]]),
            Terminator::BranchCond {
                cond,
                true_block,
                false_block,
                merge_block,
            } => {
                w.inst(OP_SELECTION_MERGE, &[block_labels[merge_block], 0]);
                w.inst(
                    OP_BRANCH_CONDITIONAL,
                    &[val_map[&cond], block_labels[true_block], block_labels[false_block]],
                );
            }
        }
    }

    w.inst(OP_FUNCTION_END, &[]);
    w.finish()
}

struct Writer {
    next_id: u32,
    body: Vec<u32>,
}

impl Writer {
    fn new() -> Self {
        Self { next_id: 1, body: Vec::new() }
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
