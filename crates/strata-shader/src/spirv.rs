//! Direct SPIR-V 1.6 binary writer targeting No-API Vulkan (`PhysicalStorageBuffer64`).
//!
//! Emits a self-contained, `spirv-val`-clean compute shader module with:
//! - 128-byte Push Constant block (`uint sgpr[32]`) for console User SGPRs
//! - 64-bit `PhysicalStorageBuffer` pointer loads/stores (scalar & 128-bit coalesced)
//! - `GroupNonUniformBroadcastFirst` for `v_readfirstlane_b32`
//! - Native `GLSL.std.450` `Fma` for `v_fmac_f32`

use std::collections::HashMap;

use crate::ssa::{SsaModule, SsaOp, ValueId};

// SPIR-V constants
const SPIRV_MAGIC: u32 = 0x0723_0203;
const SPIRV_VERSION_1_6: u32 = 0x0001_0600;

const OP_CAPABILITY: u16 = 17;
const OP_EXT_INST_IMPORT: u16 = 11;
const OP_EXT_INST: u16 = 12;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;

const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_BOOL: u16 = 20;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_FUNCTION: u16 = 33;

const OP_CONSTANT: u16 = 43;
const OP_CONSTANT_TRUE: u16 = 41;
const OP_CONSTANT_FALSE: u16 = 42;
const OP_VARIABLE: u16 = 59;

const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_LABEL: u16 = 248;
const OP_SELECTION_MERGE: u16 = 247;
const OP_BRANCH: u16 = 249;
const OP_BRANCH_CONDITIONAL: u16 = 250;
const OP_RETURN: u16 = 253;
const OP_PHI: u16 = 245;

const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_COMPOSITE_CONSTRUCT: u16 = 80;
const OP_COMPOSITE_EXTRACT: u16 = 81;

const OP_BITCAST: u16 = 124;
const OP_CONVERT_U_TO_PTR: u16 = 120;
const OP_U_CONVERT: u16 = 113;
const OP_I_ADD: u16 = 128;
const OP_F_ADD: u16 = 129;
const OP_F_MUL: u16 = 133;
const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
const OP_BITWISE_OR: u16 = 198;
const OP_SELECT: u16 = 169;
const OP_LOGICAL_OR: u16 = 166;
const OP_F_ORD_EQUAL: u16 = 180;
const OP_F_ORD_LESS_THAN: u16 = 184;
const OP_GROUP_NON_UNIFORM_BROADCAST_FIRST: u16 = 338;
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
const SC_PUSH_CONSTANT: u32 = 9;
const SC_PHYSICAL_STORAGE_BUFFER: u32 = 5349;

// Decorations
const DEC_BLOCK: u32 = 2;
const DEC_ARRAY_STRIDE: u32 = 6;
const DEC_BUILTIN: u32 = 11;
const DEC_OFFSET: u32 = 35;

// BuiltIns
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

// Memory operands
const MEM_ALIGNED: u32 = 0x2;

// GLSL.std.450 opcodes
const GLSL_LOG2: u32 = 30;
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

    // EntryPoint GLCompute (5) "main" %global_id_var %push_var
    let mut ep = vec![5, main_fn];
    encode_str(&mut ep, "main");
    ep.push(global_id_var);
    ep.push(push_var);
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

    // PhysicalStorageBuffer pointer types (scalar u32 & 128-bit uvec4)
    let t_bda_u32_struct = w.id();
    let t_bda_u32_ptr = w.id();
    let t_bda_u32_field_ptr = w.id();
    let t_bda_uvec4_struct = w.id();
    let t_bda_uvec4_ptr = w.id();
    let t_bda_uvec4_field_ptr = w.id();

    // Decorations
    w.inst(OP_DECORATE, &[global_id_var, DEC_BUILTIN, BUILTIN_GLOBAL_INVOCATION_ID]);
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
    let c_true = w.id();
    let c_false = w.id();
    w.inst(OP_CONSTANT_TRUE, &[t_bool, c_true]);
    w.inst(OP_CONSTANT_FALSE, &[t_bool, c_false]);

    let c_zero = w.id();
    w.inst(OP_CONSTANT, &[t_u32, c_zero, 0]);
    const_u32.insert(0, c_zero);

    let c_fzero = w.id();
    w.inst(OP_CONSTANT, &[t_f32, c_fzero, 0.0f32.to_bits()]);
    const_f32.insert(0.0f32.to_bits(), c_fzero);

    let c_scope_subgroup = w.id();
    w.inst(OP_CONSTANT, &[t_u32, c_scope_subgroup, 3]); // Scope::Subgroup = 3
    const_u32.insert(3, c_scope_subgroup);

    for dir in 0..=2u32 {
        const_u32.entry(dir).or_insert_with(|| {
            let id = w.id();
            w.inst(OP_CONSTANT, &[t_u32, id, dir]);
            id
        });
    }

    let c_32_u64 = w.id();
    w.inst(OP_CONSTANT, &[t_u64, c_32_u64, 32, 0]);
    const_u64.insert(32, c_32_u64);

    // Pre-emit any constants referenced by live SSA instructions (SPIR-V requires
    // OpConstant before OpFunction).
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
                _ => {}
            }
        }
    }

    // Interface variables
    w.inst(OP_VARIABLE, &[t_ptr_in_uvec3, global_id_var, SC_INPUT]);
    w.inst(OP_VARIABLE, &[t_ptr_push_struct, push_var, SC_PUSH_CONSTANT]);

    // Function body
    w.inst(OP_FUNCTION, &[t_void, main_fn, 0, t_fn]);
    let entry_lbl = w.id();
    w.inst(OP_LABEL, &[entry_lbl]);

    let mut val_map: HashMap<ValueId, u32> = HashMap::new();

    for block in &module.blocks {
        for inst in &block.insts {
            if inst.dead {
                continue;
            }
            match &inst.op {
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
                SsaOp::AddU32(a, b) => {
                    let res = w.id();
                    w.inst(OP_I_ADD, &[t_u32, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::AddU64(a, b) => {
                    let res = w.id();
                    w.inst(OP_I_ADD, &[t_u64, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::ShlU32(a, b) => {
                    let res = w.id();
                    w.inst(OP_SHIFT_LEFT_LOGICAL, &[t_u32, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::AddF32(a, b) => {
                    let res = w.id();
                    w.inst(OP_F_ADD, &[t_f32, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::MulF32(a, b) => {
                    let res = w.id();
                    w.inst(OP_F_MUL, &[t_f32, res, val_map[a], val_map[b]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::MulLegacyF32(a, b) => {
                    // Exact DX9/RDNA2 semantics: if either operand is ±0.0, result is 0.0 (even vs NaN/Inf).
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
                SsaOp::Log2F32(a) => {
                    let res = w.id();
                    w.inst(OP_EXT_INST, &[t_f32, res, glsl_ext, GLSL_LOG2, val_map[a]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::FmaF32(a, b, c) => {
                    let res = w.id();
                    w.inst(OP_EXT_INST, &[t_f32, res, glsl_ext, GLSL_FMA, val_map[a], val_map[b], val_map[c]]);
                    val_map.insert(inst.id, res);
                }
                SsaOp::CmpLtF32(a, b) => {
                    let res = w.id();
                    w.inst(OP_F_ORD_LESS_THAN, &[t_bool, res, val_map[a], val_map[b]]);
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
                SsaOp::QuadSwap { val, direction } => {
                    let dir_id = const_u32[&((*direction).min(2) as u32)];
                    let res = w.id();
                    w.inst(OP_GROUP_NON_UNIFORM_QUAD_SWAP, &[t_u32, res, c_scope_subgroup, val_map[val], dir_id]);
                    val_map.insert(inst.id, res);
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
            }
        }
    }

    w.inst(OP_RETURN, &[]);
    w.inst(OP_FUNCTION_END, &[]);
    w.finish()
}

/// Emit a structured SPIR-V 1.6 `OpSelectionMerge` + `OpBranchConditional` + `OpPhi`
/// test module to verify multi-block divergent control flow directly against `spirv-val`.
pub fn emit_structured_select_kernel(workgroup_size: [u32; 3]) -> Vec<u32> {
    let mut w = Writer::new();
    w.inst(OP_CAPABILITY, &[CAP_SHADER]);
    w.inst(OP_CAPABILITY, &[CAP_INT64]);
    w.inst(OP_CAPABILITY, &[CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES]);
    w.inst(OP_MEMORY_MODEL, &[5348, 1]);

    let main_fn = w.id();
    let gid_var = w.id();
    let push_var = w.id();
    let mut ep = vec![5, main_fn];
    encode_str(&mut ep, "main");
    ep.push(gid_var);
    ep.push(push_var);
    w.inst(OP_ENTRY_POINT, &ep);
    w.inst(OP_EXECUTION_MODE, &[main_fn, 17, workgroup_size[0], workgroup_size[1], workgroup_size[2]]);

    let t_void = w.id();
    let t_bool = w.id();
    let t_u32 = w.id();
    let t_u64 = w.id();
    let t_uvec3 = w.id();
    let t_fn = w.id();
    let t_push_struct = w.id();
    let t_ptr_push = w.id();
    let t_ptr_push_u64 = w.id();
    let t_ptr_in_uvec3 = w.id();
    let t_ptr_in_u32 = w.id();
    let t_bda_struct = w.id();
    let t_bda_ptr = w.id();
    let t_bda_field = w.id();

    w.inst(OP_DECORATE, &[gid_var, DEC_BUILTIN, BUILTIN_GLOBAL_INVOCATION_ID]);
    w.inst(OP_DECORATE, &[t_push_struct, DEC_BLOCK]);
    w.inst(OP_MEMBER_DECORATE, &[t_push_struct, 0, DEC_OFFSET, 0]);
    w.inst(OP_DECORATE, &[t_bda_struct, DEC_BLOCK]);
    w.inst(OP_MEMBER_DECORATE, &[t_bda_struct, 0, DEC_OFFSET, 0]);

    w.inst(OP_TYPE_VOID, &[t_void]);
    w.inst(OP_TYPE_BOOL, &[t_bool]);
    w.inst(OP_TYPE_INT, &[t_u32, 32, 0]);
    w.inst(OP_TYPE_INT, &[t_u64, 64, 0]);
    w.inst(OP_TYPE_VECTOR, &[t_uvec3, t_u32, 3]);
    w.inst(OP_TYPE_FUNCTION, &[t_fn, t_void]);
    w.inst(OP_TYPE_STRUCT, &[t_push_struct, t_u64]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_push, SC_PUSH_CONSTANT, t_push_struct]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_push_u64, SC_PUSH_CONSTANT, t_u64]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_in_uvec3, SC_INPUT, t_uvec3]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_in_u32, SC_INPUT, t_u32]);
    w.inst(OP_TYPE_STRUCT, &[t_bda_struct, t_u32]);
    w.inst(OP_TYPE_POINTER, &[t_bda_ptr, SC_PHYSICAL_STORAGE_BUFFER, t_bda_struct]);
    w.inst(OP_TYPE_POINTER, &[t_bda_field, SC_PHYSICAL_STORAGE_BUFFER, t_u32]);

    let c0 = w.id();
    let c10 = w.id();
    let c20 = w.id();
    let c32 = w.id();
    let c4_u64 = w.id();
    w.inst(OP_CONSTANT, &[t_u32, c0, 0]);
    w.inst(OP_CONSTANT, &[t_u32, c10, 10]);
    w.inst(OP_CONSTANT, &[t_u32, c20, 20]);
    w.inst(OP_CONSTANT, &[t_u32, c32, 32]);
    w.inst(OP_CONSTANT, &[t_u64, c4_u64, 4, 0]);

    w.inst(OP_VARIABLE, &[t_ptr_in_uvec3, gid_var, SC_INPUT]);
    w.inst(OP_VARIABLE, &[t_ptr_push, push_var, SC_PUSH_CONSTANT]);

    w.inst(OP_FUNCTION, &[t_void, main_fn, 0, t_fn]);
    let lbl_entry = w.id();
    let lbl_then = w.id();
    let lbl_else = w.id();
    let lbl_merge = w.id();

    w.inst(OP_LABEL, &[lbl_entry]);
    let p_tid = w.id();
    w.inst(OP_ACCESS_CHAIN, &[t_ptr_in_u32, p_tid, gid_var, c0]);
    let tid = w.id();
    w.inst(OP_LOAD, &[t_u32, tid, p_tid]);
    let cond = w.id();
    // OpULessThan = 176
    w.inst(176, &[t_bool, cond, tid, c32]);
    w.inst(OP_SELECTION_MERGE, &[lbl_merge, 0]);
    w.inst(OP_BRANCH_CONDITIONAL, &[cond, lbl_then, lbl_else]);

    w.inst(OP_LABEL, &[lbl_then]);
    let v_then = w.id();
    w.inst(OP_I_ADD, &[t_u32, v_then, tid, c10]);
    w.inst(OP_BRANCH, &[lbl_merge]);

    w.inst(OP_LABEL, &[lbl_else]);
    let v_else = w.id();
    w.inst(OP_I_ADD, &[t_u32, v_else, tid, c20]);
    w.inst(OP_BRANCH, &[lbl_merge]);

    w.inst(OP_LABEL, &[lbl_merge]);
    let phi = w.id();
    w.inst(OP_PHI, &[t_u32, phi, v_then, lbl_then, v_else, lbl_else]);

    let p_base = w.id();
    w.inst(OP_ACCESS_CHAIN, &[t_ptr_push_u64, p_base, push_var, c0]);
    let base = w.id();
    w.inst(OP_LOAD, &[t_u64, base, p_base]);
    let tid64 = w.id();
    w.inst(OP_U_CONVERT, &[t_u64, tid64, tid]);
    let off = w.id();
    // OpIMul = 132
    w.inst(132, &[t_u64, off, tid64, c4_u64]);
    let ea = w.id();
    w.inst(OP_I_ADD, &[t_u64, ea, base, off]);
    let bda_p = w.id();
    w.inst(OP_CONVERT_U_TO_PTR, &[t_bda_ptr, bda_p, ea]);
    let f_p = w.id();
    w.inst(OP_ACCESS_CHAIN, &[t_bda_field, f_p, bda_p, c0]);
    w.inst(OP_STORE, &[f_p, phi, MEM_ALIGNED, 4]);

    w.inst(OP_RETURN, &[]);
    w.inst(OP_FUNCTION_END, &[]);
    w.finish()
}

/// Emit a **Wave64-compatible** SPIR-V 1.6 compute shader that produces a 64-bit
/// console wave mask across any hardware subgroup size (Intel Arc SIMD8/16,
/// NVIDIA / AMD RDNA Wave32, and AMD GCN / RDNA native Wave64) using
/// `BuiltIn SubgroupSize`, `OpGroupNonUniformBallot`, `Workgroup` `OpAtomicOr`,
/// and `OpControlBarrier`.
pub fn emit_wave64_on_wave32_ballot_kernel() -> Vec<u32> {
    const OP_I_SUB: u16 = 130;
    const OP_LOGICAL_AND: u16 = 167;
    const OP_I_NOT_EQUAL: u16 = 171;
    const OP_U_GREATER_THAN_EQUAL: u16 = 175;
    const OP_BITWISE_AND: u16 = 199;
    const OP_NOT: u16 = 200;
    const OP_CONTROL_BARRIER: u16 = 224;
    const OP_ATOMIC_OR: u16 = 241;
    const OP_GROUP_NON_UNIFORM_BALLOT: u16 = 339;
    const SC_WORKGROUP: u32 = 4;
    const BUILTIN_SUBGROUP_SIZE: u32 = 36;

    let mut w = Writer::new();
    w.inst(OP_CAPABILITY, &[CAP_SHADER]);
    w.inst(OP_CAPABILITY, &[CAP_INT64]);
    w.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM]);
    w.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM_BALLOT]);
    w.inst(OP_CAPABILITY, &[CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES]);
    w.inst(OP_MEMORY_MODEL, &[5348, 1]);

    let main_fn = w.id();
    let local_idx_var = w.id();
    let sg_size_var = w.id();
    let push_var = w.id();
    let shared_var = w.id();

    let mut ep = vec![5, main_fn];
    encode_str(&mut ep, "main");
    ep.push(local_idx_var);
    ep.push(sg_size_var);
    ep.push(push_var);
    ep.push(shared_var);
    w.inst(OP_ENTRY_POINT, &ep);
    w.inst(OP_EXECUTION_MODE, &[main_fn, 17, 64, 1, 1]);

    let t_void = w.id();
    let t_bool = w.id();
    let t_u32 = w.id();
    let t_u64 = w.id();
    let t_uvec4 = w.id();
    let t_fn = w.id();

    let c2 = w.id();
    let t_shared_arr = w.id();
    let t_ptr_wg_arr = w.id();
    let t_ptr_wg_u32 = w.id();

    let t_push_struct = w.id();
    let t_ptr_push = w.id();
    let t_ptr_push_u64 = w.id();
    let t_ptr_in_u32 = w.id();

    let t_bda_u64_struct = w.id();
    let t_bda_u64_ptr = w.id();
    let t_bda_u64_field = w.id();

    // BuiltIn LocalInvocationIndex = 29, SubgroupSize = 36
    w.inst(OP_DECORATE, &[local_idx_var, DEC_BUILTIN, 29]);
    w.inst(OP_DECORATE, &[sg_size_var, DEC_BUILTIN, BUILTIN_SUBGROUP_SIZE]);
    w.inst(OP_DECORATE, &[t_push_struct, DEC_BLOCK]);
    w.inst(OP_MEMBER_DECORATE, &[t_push_struct, 0, DEC_OFFSET, 0]);
    w.inst(OP_DECORATE, &[t_bda_u64_struct, DEC_BLOCK]);
    w.inst(OP_MEMBER_DECORATE, &[t_bda_u64_struct, 0, DEC_OFFSET, 0]);

    w.inst(OP_TYPE_VOID, &[t_void]);
    w.inst(OP_TYPE_BOOL, &[t_bool]);
    w.inst(OP_TYPE_INT, &[t_u32, 32, 0]);
    w.inst(OP_TYPE_INT, &[t_u64, 64, 0]);
    w.inst(OP_TYPE_VECTOR, &[t_uvec4, t_u32, 4]);
    w.inst(OP_TYPE_FUNCTION, &[t_fn, t_void]);

    w.inst(OP_CONSTANT, &[t_u32, c2, 2]);
    w.inst(OP_TYPE_ARRAY, &[t_shared_arr, t_u32, c2]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_wg_arr, SC_WORKGROUP, t_shared_arr]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_wg_u32, SC_WORKGROUP, t_u32]);

    w.inst(OP_TYPE_STRUCT, &[t_push_struct, t_u64]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_push, SC_PUSH_CONSTANT, t_push_struct]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_push_u64, SC_PUSH_CONSTANT, t_u64]);
    w.inst(OP_TYPE_POINTER, &[t_ptr_in_u32, SC_INPUT, t_u32]);

    w.inst(OP_TYPE_STRUCT, &[t_bda_u64_struct, t_u64]);
    w.inst(OP_TYPE_POINTER, &[t_bda_u64_ptr, SC_PHYSICAL_STORAGE_BUFFER, t_bda_u64_struct]);
    w.inst(OP_TYPE_POINTER, &[t_bda_u64_field, SC_PHYSICAL_STORAGE_BUFFER, t_u64]);

    let c0 = w.id();
    let c1 = w.id();
    let c5 = w.id();
    let c31 = w.id();
    let c64 = w.id();
    let c_scope_wg = w.id();
    let c_scope_sg = w.id();
    let c_sem_wg = w.id();
    let c32_u64 = w.id();
    w.inst(OP_CONSTANT, &[t_u32, c0, 0]);
    w.inst(OP_CONSTANT, &[t_u32, c1, 1]);
    w.inst(OP_CONSTANT, &[t_u32, c5, 5]);
    w.inst(OP_CONSTANT, &[t_u32, c31, 31]);
    w.inst(OP_CONSTANT, &[t_u32, c64, 64]);
    w.inst(OP_CONSTANT, &[t_u32, c_scope_wg, 2]); // Scope::Workgroup = 2
    w.inst(OP_CONSTANT, &[t_u32, c_scope_sg, 3]); // Scope::Subgroup = 3
    w.inst(OP_CONSTANT, &[t_u32, c_sem_wg, 0x108]); // AcquireRelease | WorkgroupMemory
    w.inst(OP_CONSTANT, &[t_u64, c32_u64, 32, 0]);

    w.inst(OP_VARIABLE, &[t_ptr_in_u32, local_idx_var, SC_INPUT]);
    w.inst(OP_VARIABLE, &[t_ptr_in_u32, sg_size_var, SC_INPUT]);
    w.inst(OP_VARIABLE, &[t_ptr_push, push_var, SC_PUSH_CONSTANT]);
    w.inst(OP_VARIABLE, &[t_ptr_wg_arr, shared_var, SC_WORKGROUP]);

    w.inst(OP_FUNCTION, &[t_void, main_fn, 0, t_fn]);
    let lbl = w.id();
    w.inst(OP_LABEL, &[lbl]);

    // Predicate: even lanes are active (`(tid & 1) == 0`) -> expected 64-bit mask 0x5555_5555_5555_5555
    let tid = w.id();
    w.inst(OP_LOAD, &[t_u32, tid, local_idx_var]);
    let sg_size = w.id();
    w.inst(OP_LOAD, &[t_u32, sg_size, sg_size_var]);
    let is_w64 = w.id();
    w.inst(OP_U_GREATER_THAN_EQUAL, &[t_bool, is_w64, sg_size, c64]);

    let lsb = w.id();
    // OpBitwiseAnd = 199
    w.inst(199, &[t_u32, lsb, tid, c1]);
    let is_even = w.id();
    // OpIEqual = 170
    w.inst(170, &[t_bool, is_even, lsb, c0]);

    // Compute subgroup ballot for any subgroup size (SIMD8/16, Wave32, Wave64):
    // - Clear shared[0..2] before atomic merge.
    // - On Wave64, lanes 0..31 take ballot.x and lanes 32..63 take ballot.y (bit shift = 0).
    // - On SIMD8/16/32, each subgroup shifts ballot.x by ((tid & 31) & ~(sg_size - 1))
    //   and atomically ORs it into shared[tid >> 5].
    let half_idx = w.id();
    // OpShiftRightLogical = 194
    w.inst(194, &[t_u32, half_idx, tid, c5]);
    let p_slot = w.id();
    w.inst(OP_ACCESS_CHAIN, &[t_ptr_wg_u32, p_slot, shared_var, half_idx]);
    w.inst(OP_STORE, &[p_slot, c0]);
    w.inst(OP_CONTROL_BARRIER, &[c_scope_wg, c_scope_wg, c_sem_wg]);

    let ballot_v4 = w.id();
    w.inst(OP_GROUP_NON_UNIFORM_BALLOT, &[t_uvec4, ballot_v4, c_scope_sg, is_even]);
    let ballot_lo = w.id();
    w.inst(OP_COMPOSITE_EXTRACT, &[t_u32, ballot_lo, ballot_v4, 0]);
    let ballot_hi = w.id();
    w.inst(OP_COMPOSITE_EXTRACT, &[t_u32, ballot_hi, ballot_v4, 1]);
    let is_hi_half = w.id();
    w.inst(OP_I_NOT_EQUAL, &[t_bool, is_hi_half, half_idx, c0]);
    let use_hi = w.id();
    w.inst(OP_LOGICAL_AND, &[t_bool, use_hi, is_w64, is_hi_half]);
    let half_mask = w.id();
    w.inst(OP_SELECT, &[t_u32, half_mask, use_hi, ballot_hi, ballot_lo]);

    let sg_m1 = w.id();
    w.inst(OP_I_SUB, &[t_u32, sg_m1, sg_size, c1]);
    let sg_align_mask = w.id();
    w.inst(OP_NOT, &[t_u32, sg_align_mask, sg_m1]);
    let tid_in_half = w.id();
    w.inst(OP_BITWISE_AND, &[t_u32, tid_in_half, tid, c31]);
    let sub_shift = w.id();
    w.inst(OP_BITWISE_AND, &[t_u32, sub_shift, tid_in_half, sg_align_mask]);
    let shifted_mask = w.id();
    w.inst(OP_SHIFT_LEFT_LOGICAL, &[t_u32, shifted_mask, half_mask, sub_shift]);
    let atomic_old = w.id();
    w.inst(OP_ATOMIC_OR, &[t_u32, atomic_old, p_slot, c_scope_wg, c0, shifted_mask]);

    // Synchronize all subgroups before reading combined 64-bit mask
    w.inst(OP_CONTROL_BARRIER, &[c_scope_wg, c_scope_wg, c_sem_wg]);

    // Combine low32 | (high32 << 32) into the full 64-bit console wave mask and store to BDA
    let p_lo = w.id();
    w.inst(OP_ACCESS_CHAIN, &[t_ptr_wg_u32, p_lo, shared_var, c0]);
    let lo = w.id();
    w.inst(OP_LOAD, &[t_u32, lo, p_lo]);
    let p_hi = w.id();
    w.inst(OP_ACCESS_CHAIN, &[t_ptr_wg_u32, p_hi, shared_var, c1]);
    let hi = w.id();
    w.inst(OP_LOAD, &[t_u32, hi, p_hi]);

    let lo64 = w.id();
    w.inst(OP_U_CONVERT, &[t_u64, lo64, lo]);
    let hi64 = w.id();
    w.inst(OP_U_CONVERT, &[t_u64, hi64, hi]);
    let hi_sh = w.id();
    w.inst(OP_SHIFT_LEFT_LOGICAL, &[t_u64, hi_sh, hi64, c32_u64]);
    let full_mask = w.id();
    w.inst(OP_BITWISE_OR, &[t_u64, full_mask, lo64, hi_sh]);

    let p_dst = w.id();
    w.inst(OP_ACCESS_CHAIN, &[t_ptr_push_u64, p_dst, push_var, c0]);
    let dst_ea = w.id();
    w.inst(OP_LOAD, &[t_u64, dst_ea, p_dst]);
    let bda_p = w.id();
    w.inst(OP_CONVERT_U_TO_PTR, &[t_bda_u64_ptr, bda_p, dst_ea]);
    let f_p = w.id();
    w.inst(OP_ACCESS_CHAIN, &[t_bda_u64_field, f_p, bda_p, c0]);
    w.inst(OP_STORE, &[f_p, full_mask, MEM_ALIGNED, 8]);

    w.inst(OP_RETURN, &[]);
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
