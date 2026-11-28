//! Dynamic (non-static) geometry, task, mesh, hull and domain shaders emit
//! their optimized SSA body through `spirv.rs`, not a static placeholder:
//! each module must pass `spirv-val --target-env vulkan1.3` and carry the
//! body's memory loads and the stage's primitive instructions.

use std::process::Command;

use strata_shader::geometry::{
    GeometryShaderDesc, GsInputPrimitive, GsOutputPrimitive, MeshShaderDesc, TaskShaderDesc,
    translate_geometry_rdna2, translate_mesh_rdna2, translate_task_rdna2,
};
use strata_shader::rdna2::encode::*;
use strata_shader::tessellation::{
    DomainShaderDesc, HullShaderDesc, translate_domain_rdna2, translate_hull_rdna2,
};

const OP_LOAD: u32 = 61;
const OP_EMIT_VERTEX: u32 = 218;
const OP_END_PRIMITIVE: u32 = 219;
const OP_EMIT_MESH_TASKS: u32 = 5294;
const OP_SET_MESH_OUTPUTS: u32 = 5295;
const OP_RETURN: u32 = 253;

fn s_sendmsg(msg: u16) -> u32 {
    (0b1011_1111_1 << 23) | (0x10 << 16) | u32::from(msg)
}

/// Opcodes of the function bodies' instructions, in order.
fn opcodes(words: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let wc = (words[i] >> 16) as usize;
        out.push(words[i] & 0xffff);
        i += wc.max(1);
    }
    out
}

fn count(words: &[u32], op: u32) -> usize {
    opcodes(words).into_iter().filter(|&o| o == op).count()
}

/// Physical-storage-buffer loads (`OpLoad ... Aligned`): the body's
/// global loads, which a static placeholder would not have.
fn bda_loads(words: &[u32]) -> usize {
    let mut n = 0;
    let mut i = 5;
    while i < words.len() {
        let wc = (words[i] >> 16) as usize;
        if words[i] & 0xffff == OP_LOAD && wc == 6 && words[i + 4] == 2 {
            n += 1;
        }
        i += wc.max(1);
    }
    n
}

fn validate(words: &[u32], tag: &str) {
    let path = std::env::temp_dir().join(format!("strata_dyn_{tag}.spv"));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&path, bytes).unwrap();
    let out = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3"])
        .arg(&path)
        .output();
    if std::env::var_os("STRATA_KEEP_SPV").is_none() {
        let _ = std::fs::remove_file(&path);
    }
    match out {
        Ok(out) => assert!(
            out.status.success(),
            "spirv-val rejected {tag}: {}",
            String::from_utf8_lossy(&out.stderr)
        ),
        Err(_) => eprintln!("spirv-val not on PATH: {tag} not validated"),
    }
}

/// `v1 = *(s[0:1] + v0)`: a load the static evaluator cannot fold.
fn load_v1() -> Vec<u32> {
    global_load_dword(1, 0, 0, 0).to_vec()
}

#[test]
fn dynamic_geometry_shader_emits_its_body_and_primitives() {
    let mut w = load_v1();
    w.extend(v_mov_b32_literal(2, 0.0f32.to_bits()));
    w.extend(v_mov_b32_literal(3, 1.0f32.to_bits()));
    w.extend(exp(12, 0xf, [1, 1, 2, 3], false, false));
    w.push(s_sendmsg(0x22)); // GS emit, stream 0
    w.extend(exp(12, 0xf, [2, 1, 2, 3], false, false));
    w.extend(exp(32, 0xf, [1, 1, 1, 1], false, false)); // param0
    w.push(s_sendmsg(0x32)); // GS emit-cut
    w.push(s_sendmsg(0x03)); // GS_DONE
    w.push(s_endpgm());
    let desc = GeometryShaderDesc {
        input: GsInputPrimitive::Triangles,
        output: GsOutputPrimitive::TriangleStrip,
        max_vertices: 4,
        invocations: 1,
    };
    let spv = translate_geometry_rdna2(&w, &desc).unwrap();
    validate(&spv, "gs");
    assert_eq!(count(&spv, OP_EMIT_VERTEX), 2);
    assert_eq!(count(&spv, OP_END_PRIMITIVE), 1);
    assert!(bda_loads(&spv) >= 1, "the body's global load is emitted");
    // Emits follow their exports: the cut comes after the second emit.
    let ops = opcodes(&spv);
    let last_emit = ops.iter().rposition(|&o| o == OP_EMIT_VERTEX).unwrap();
    assert!(ops.iter().position(|&o| o == OP_END_PRIMITIVE).unwrap() > last_emit);

    // A non-zero stream is refused rather than dropped.
    let mut bad = load_v1();
    bad.push(s_sendmsg(0x122));
    bad.push(s_endpgm());
    assert!(translate_geometry_rdna2(&bad, &desc).is_err());
}

#[test]
fn dynamic_task_shader_launches_v0_v2_from_its_body() {
    let mut w = load_v1();
    w.push(v_mov_b32_vgpr(0, 1)); // gx = loaded value
    w.extend(v_mov_b32_literal(1, 2));
    w.extend(v_mov_b32_literal(2, 1));
    w.push(s_endpgm());
    let spv = translate_task_rdna2(&w, &TaskShaderDesc::default()).unwrap();
    validate(&spv, "task");
    assert_eq!(count(&spv, OP_EMIT_MESH_TASKS), 1);
    assert_eq!(
        count(&spv, OP_RETURN),
        0,
        "a task shader ends in OpEmitMeshTasksEXT"
    );
    assert!(bda_loads(&spv) >= 1);
}

#[test]
fn dynamic_mesh_shader_sets_counts_from_m0_and_writes_per_lane_outputs() {
    let mut w = vec![s_movk_i32(124, 0x1003)]; // 3 vertices, 1 primitive
    w.push(s_sendmsg(9)); // GS_ALLOC_REQ
    w.extend(load_v1());
    w.extend(v_mov_b32_literal(2, 0.0f32.to_bits()));
    w.extend(v_mov_b32_literal(3, 1.0f32.to_bits()));
    w.extend(exp(12, 0xf, [1, 2, 2, 3], false, false));
    w.extend(v_mov_b32_literal(4, 1 << 10 | 2 << 20));
    w.extend(exp(20, 1, [4, 4, 4, 4], true, false)); // prim
    w.push(s_endpgm());
    let desc = MeshShaderDesc {
        workgroup_size: [32, 1, 1],
        max_vertices: 32,
        max_primitives: 32,
        output_topology: GsOutputPrimitive::TriangleStrip,
    };
    let spv = translate_mesh_rdna2(&w, &desc).unwrap();
    validate(&spv, "mesh");
    assert_eq!(count(&spv, OP_SET_MESH_OUTPUTS), 1);
    assert!(bda_loads(&spv) >= 1);

    // Without a known M0 the counts are unknowable: refused.
    let mut bad = vec![s_sendmsg(9)];
    bad.extend(load_v1());
    bad.push(s_endpgm());
    assert!(translate_mesh_rdna2(&bad, &desc).is_err());
}

#[test]
fn dynamic_hull_shader_writes_tess_levels_from_its_body() {
    // Factors loaded from memory, stored to the factor block at 0, 4, 8, 12.
    let mut w = load_v1();
    for i in 0..4u16 {
        let [op, operands] = buffer_store_dword(1, 0, 4, i * 4);
        w.extend([op & !(1 << 12), operands]);
    }
    w.push(s_endpgm());
    let spv = translate_hull_rdna2(&w, &HullShaderDesc::default()).unwrap();
    validate(&spv, "hull");
    assert!(bda_loads(&spv) >= 1);
    // Triangles: three outer levels and one inner, each one store.
    let stores = count(&spv, 62);
    assert!(
        stores >= 5,
        "4 levels + the control-point pass-through: {stores}"
    );
}

#[test]
fn dynamic_domain_shader_reads_tess_coord_and_memory() {
    let mut w = load_v1();
    w.push(v_mul_f32(2, 0, 1)); // u * loaded
    w.extend(v_mov_b32_literal(3, 1.0f32.to_bits()));
    w.extend(exp(12, 0xf, [2, 1, 1, 3], true, false));
    w.push(s_endpgm());
    let spv = translate_domain_rdna2(&w, &DomainShaderDesc::default()).unwrap();
    validate(&spv, "domain");
    assert!(bda_loads(&spv) >= 1);
}
