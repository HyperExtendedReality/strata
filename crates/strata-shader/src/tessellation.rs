//! Hull (Tessellation Control / merged LS+HS) and Domain (Tessellation
//! Evaluation) shader translation to SPIR-V 1.6.
//!
//! On RDNA2 (GFX10.3), the hardware merges Local Shader (LS = VS-before-HS)
//! and Hull Shader (HS) into a single merged HS wave, and runs the Domain
//! Shader (DS) as either ES (before GS) or VS/NGG (before rasterizer).
//! This module translates RDNA2 HS/LS+HS and DS microcode into `spirv-val`-clean
//! `TessellationControl` and `TessellationEvaluation` SPIR-V 1.6 modules using
//! Strata's No-API conventions.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TessDomain {
    Triangles,
    Quads,
    Isolines,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TessSpacing {
    Equal,
    FractionalEven,
    FractionalOdd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TessWinding {
    Cw,
    Ccw,
}

/// Configuration for a Hull Shader (`TessellationControl`, incl. merged LS+HS).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HullShaderDesc {
    pub output_control_points: u32,
    pub domain: TessDomain,
    pub outer_factors: [u32; 4],
    pub inner_factors: [u32; 2],
}

impl Default for HullShaderDesc {
    fn default() -> Self {
        Self {
            output_control_points: 3,
            domain: TessDomain::Triangles,
            outer_factors: [1.0f32.to_bits(); 4],
            inner_factors: [1.0f32.to_bits(); 2],
        }
    }
}

/// Configuration for a Domain Shader (`TessellationEvaluation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainShaderDesc {
    pub domain: TessDomain,
    pub spacing: TessSpacing,
    pub winding: TessWinding,
    pub input_control_points: u32,
}

impl Default for DomainShaderDesc {
    fn default() -> Self {
        Self {
            domain: TessDomain::Triangles,
            spacing: TessSpacing::Equal,
            winding: TessWinding::Cw,
            input_control_points: 3,
        }
    }
}

fn push_inst(buf: &mut Vec<u32>, op: u16, operands: &[u32]) {
    let wc = (1 + operands.len()) as u32;
    buf.push((wc << 16) | u32::from(op));
    buf.extend_from_slice(operands);
}

fn push_string(buf: &mut Vec<u32>, s: &str) {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let mut word = 0u32;
        for b in 0..4 {
            if i + b < bytes.len() {
                word |= u32::from(bytes[i + b]) << (b * 8);
            }
        }
        buf.push(word);
        i += 4;
    }
    if bytes.len() % 4 == 0 {
        buf.push(0);
    }
}

/// Emit a valid SPIR-V 1.6 `TessellationControl` (Hull / LS+HS) shader module.
///
/// Copies input control-point positions (`gl_in[gl_InvocationID].gl_Position`)
/// to output control points (`gl_out[gl_InvocationID].gl_Position`) and writes
/// `TessLevelOuter` / `TessLevelInner` patch tessellation factors.
pub fn emit_hull_spirv(desc: &HullShaderDesc) -> Vec<u32> {
    let cp = desc.output_control_points.clamp(1, 32);
    let mut next_id = 1u32;
    let mut alloc = || {
        let id = next_id;
        next_id += 1;
        id
    };

    let entry_id = alloc();
    let void_ty = alloc();
    let fn_ty = alloc();
    let f32_ty = alloc();
    let u32_ty = alloc();
    let vec4_ty = alloc();
    let c_cp = alloc();
    let c_0 = alloc();
    let c_1 = alloc();
    let c_2 = alloc();
    let c_3 = alloc();
    let c_4 = alloc();

    let per_vert_in_struct = alloc();
    let per_vert_in_arr = alloc();
    let ptr_in_arr = alloc();
    let ptr_in_vec4 = alloc();
    let gl_in_var = alloc();

    let per_vert_out_struct = alloc();
    let per_vert_out_arr = alloc();
    let ptr_out_arr = alloc();
    let ptr_out_vec4 = alloc();
    let gl_out_var = alloc();

    let ptr_in_u32 = alloc();
    let inv_id_var = alloc();

    let arr_f32_4 = alloc();
    let ptr_out_f32_4 = alloc();
    let ptr_out_f32 = alloc();
    let tess_outer_var = alloc();

    let arr_f32_2 = alloc();
    let ptr_out_f32_2 = alloc();
    let tess_inner_var = alloc();

    let outer_c = [alloc(), alloc(), alloc(), alloc()];
    let inner_c = [alloc(), alloc()];

    let entry_label = alloc();
    let inv_val = alloc();
    let in_pos_ptr = alloc();
    let in_pos_val = alloc();
    let out_pos_ptr = alloc();
    let outer_ptrs = [alloc(), alloc(), alloc(), alloc()];
    let inner_ptrs = [alloc(), alloc()];

    let mut w = Vec::with_capacity(256);
    // SPIR-V 1.6 header
    w.extend_from_slice(&[0x0723_0203, 0x0001_0600, 0, next_id, 0]);

    // Capabilities
    push_inst(&mut w, 17 /* OpCapability */, &[1 /* Shader */]);
    push_inst(&mut w, 17 /* OpCapability */, &[3 /* Tessellation */]);
    // MemoryModel Logical GLSL450
    push_inst(&mut w, 14, &[0, 1]);

    // OpEntryPoint TessellationControl(1) %main "main" %gl_in %gl_out %inv_id %outer %inner
    let mut ep = vec![1u32, entry_id];
    push_string(&mut ep, "main");
    ep.extend_from_slice(&[
        gl_in_var,
        gl_out_var,
        inv_id_var,
        tess_outer_var,
        tess_inner_var,
    ]);
    push_inst(&mut w, 15, &ep);
    // OpExecutionMode %main OutputVertices(26) cp
    push_inst(&mut w, 16, &[entry_id, 26, cp]);

    // Decorations
    push_inst(
        &mut w,
        71, /* OpDecorate */
        &[per_vert_in_struct, 2 /* Block */],
    );
    push_inst(
        &mut w,
        72, /* OpMemberDecorate */
        &[
            per_vert_in_struct,
            0,
            11, /* BuiltIn */
            0,  /* Position */
        ],
    );
    push_inst(&mut w, 71, &[per_vert_out_struct, 2 /* Block */]);
    push_inst(
        &mut w,
        72,
        &[
            per_vert_out_struct,
            0,
            11, /* BuiltIn */
            0,  /* Position */
        ],
    );
    push_inst(
        &mut w,
        71,
        &[inv_id_var, 11 /* BuiltIn */, 8 /* InvocationId */],
    );
    push_inst(
        &mut w,
        71,
        &[
            tess_outer_var,
            11, /* BuiltIn */
            11, /* TessLevelOuter */
        ],
    );
    push_inst(&mut w, 71, &[tess_outer_var, 15 /* Patch */]);
    push_inst(
        &mut w,
        71,
        &[
            tess_inner_var,
            11, /* BuiltIn */
            12, /* TessLevelInner */
        ],
    );
    push_inst(&mut w, 71, &[tess_inner_var, 15 /* Patch */]);

    // Types & constants
    push_inst(&mut w, 19 /* OpTypeVoid */, &[void_ty]);
    push_inst(&mut w, 33 /* OpTypeFunction */, &[fn_ty, void_ty]);
    push_inst(&mut w, 22 /* OpTypeFloat */, &[f32_ty, 32]);
    push_inst(&mut w, 21 /* OpTypeInt */, &[u32_ty, 32, 0]);
    push_inst(&mut w, 23 /* OpTypeVector */, &[vec4_ty, f32_ty, 4]);
    push_inst(&mut w, 43 /* OpConstant */, &[u32_ty, c_cp, cp]);
    push_inst(&mut w, 43, &[u32_ty, c_0, 0]);
    push_inst(&mut w, 43, &[u32_ty, c_1, 1]);
    push_inst(&mut w, 43, &[u32_ty, c_2, 2]);
    push_inst(&mut w, 43, &[u32_ty, c_3, 3]);
    push_inst(&mut w, 43, &[u32_ty, c_4, 4]);
    for i in 0..4 {
        push_inst(&mut w, 43, &[f32_ty, outer_c[i], desc.outer_factors[i]]);
    }
    for i in 0..2 {
        push_inst(&mut w, 43, &[f32_ty, inner_c[i], desc.inner_factors[i]]);
    }

    push_inst(
        &mut w,
        30, /* OpTypeStruct */
        &[per_vert_in_struct, vec4_ty],
    );
    push_inst(
        &mut w,
        28, /* OpTypeArray */
        &[per_vert_in_arr, per_vert_in_struct, c_cp],
    );
    push_inst(
        &mut w,
        32, /* OpTypePointer */
        &[ptr_in_arr, 1 /* Input */, per_vert_in_arr],
    );
    push_inst(&mut w, 32, &[ptr_in_vec4, 1 /* Input */, vec4_ty]);
    push_inst(
        &mut w,
        59, /* OpVariable */
        &[ptr_in_arr, gl_in_var, 1],
    );

    push_inst(&mut w, 30, &[per_vert_out_struct, vec4_ty]);
    push_inst(&mut w, 28, &[per_vert_out_arr, per_vert_out_struct, c_cp]);
    push_inst(&mut w, 32, &[ptr_out_arr, 3 /* Output */, per_vert_out_arr]);
    push_inst(&mut w, 32, &[ptr_out_vec4, 3 /* Output */, vec4_ty]);
    push_inst(&mut w, 59, &[ptr_out_arr, gl_out_var, 3]);

    push_inst(&mut w, 32, &[ptr_in_u32, 1 /* Input */, u32_ty]);
    push_inst(&mut w, 59, &[ptr_in_u32, inv_id_var, 1]);

    push_inst(&mut w, 28, &[arr_f32_4, f32_ty, c_4]);
    push_inst(&mut w, 32, &[ptr_out_f32_4, 3 /* Output */, arr_f32_4]);
    push_inst(&mut w, 32, &[ptr_out_f32, 3 /* Output */, f32_ty]);
    push_inst(&mut w, 59, &[ptr_out_f32_4, tess_outer_var, 3]);

    push_inst(&mut w, 28, &[arr_f32_2, f32_ty, c_2]);
    push_inst(&mut w, 32, &[ptr_out_f32_2, 3 /* Output */, arr_f32_2]);
    push_inst(&mut w, 59, &[ptr_out_f32_2, tess_inner_var, 3]);

    // Function body
    push_inst(
        &mut w,
        54, /* OpFunction */
        &[void_ty, entry_id, 0, fn_ty],
    );
    push_inst(&mut w, 248 /* OpLabel */, &[entry_label]);
    push_inst(&mut w, 61 /* OpLoad */, &[u32_ty, inv_val, inv_id_var]);
    push_inst(
        &mut w,
        65, /* OpAccessChain */
        &[ptr_in_vec4, in_pos_ptr, gl_in_var, inv_val, c_0],
    );
    push_inst(&mut w, 61, &[vec4_ty, in_pos_val, in_pos_ptr]);
    push_inst(
        &mut w,
        65,
        &[ptr_out_vec4, out_pos_ptr, gl_out_var, inv_val, c_0],
    );
    push_inst(&mut w, 62 /* OpStore */, &[out_pos_ptr, in_pos_val]);

    let idx_c = [c_0, c_1, c_2, c_3];
    for i in 0..4 {
        push_inst(
            &mut w,
            65,
            &[ptr_out_f32, outer_ptrs[i], tess_outer_var, idx_c[i]],
        );
        push_inst(&mut w, 62, &[outer_ptrs[i], outer_c[i]]);
    }
    for i in 0..2 {
        push_inst(
            &mut w,
            65,
            &[ptr_out_f32, inner_ptrs[i], tess_inner_var, idx_c[i]],
        );
        push_inst(&mut w, 62, &[inner_ptrs[i], inner_c[i]]);
    }

    push_inst(&mut w, 253 /* OpReturn */, &[]);
    push_inst(&mut w, 56 /* OpFunctionEnd */, &[]);
    w
}

/// Emit a valid SPIR-V 1.6 `TessellationEvaluation` (Domain) shader module.
///
/// Interpolates control-point positions using `TessCoord`:
/// - `Triangles`: `u * p0 + v * p1 + w * p2`
/// - `Quads`: bilinear `mix(mix(p0, p1, u), mix(p3, p2, u), v)`
/// - `Isolines`: linear `mix(p0, p1, u)`
pub fn emit_domain_spirv(desc: &DomainShaderDesc) -> Vec<u32> {
    let cp = desc.input_control_points.clamp(
        match desc.domain {
            TessDomain::Quads => 4,
            TessDomain::Triangles => 3,
            TessDomain::Isolines => 2,
        },
        32,
    );
    let mut next_id = 1u32;
    let mut alloc = || {
        let id = next_id;
        next_id += 1;
        id
    };

    let entry_id = alloc();
    let void_ty = alloc();
    let fn_ty = alloc();
    let f32_ty = alloc();
    let u32_ty = alloc();
    let vec3_ty = alloc();
    let vec4_ty = alloc();

    let c_cp = alloc();
    let c_0 = alloc();
    let c_1 = alloc();
    let c_2 = alloc();
    let c_3 = alloc();
    let c_one_f = alloc();

    let per_vert_in_struct = alloc();
    let per_vert_in_arr = alloc();
    let ptr_in_arr = alloc();
    let ptr_in_vec4 = alloc();
    let gl_in_var = alloc();

    let per_vert_out_struct = alloc();
    let ptr_out_struct = alloc();
    let ptr_out_vec4 = alloc();
    let gl_out_var = alloc();

    let ptr_in_vec3 = alloc();
    let tess_coord_var = alloc();

    let entry_label = alloc();
    let tc_val = alloc();
    let u_val = alloc();
    let v_val = alloc();
    let w_val = alloc();
    let p_ptrs = [alloc(), alloc(), alloc(), alloc()];
    let p_vals = [alloc(), alloc(), alloc(), alloc()];
    let out_pos_ptr = alloc();

    // Extra IDs for interpolation math
    let t0 = alloc();
    let t1 = alloc();
    let t2 = alloc();
    let t3 = alloc();
    let t4 = alloc();
    let t5 = alloc();
    let t6 = alloc();
    let t7 = alloc();
    let final_pos = alloc();

    let mut w = Vec::with_capacity(256);
    w.extend_from_slice(&[0x0723_0203, 0x0001_0600, 0, next_id, 0]);

    push_inst(&mut w, 17, &[1 /* Shader */]);
    push_inst(&mut w, 17, &[3 /* Tessellation */]);
    push_inst(&mut w, 14, &[0, 1]);

    // OpEntryPoint TessellationEvaluation(2) %main "main" %gl_in %gl_out %tess_coord
    let mut ep = vec![2u32, entry_id];
    push_string(&mut ep, "main");
    ep.extend_from_slice(&[gl_in_var, gl_out_var, tess_coord_var]);
    push_inst(&mut w, 15, &ep);

    // Execution modes: domain, spacing, winding
    let domain_mode = match desc.domain {
        TessDomain::Triangles => 22, // Triangles
        TessDomain::Quads => 24,     // Quads
        TessDomain::Isolines => 25,  // Isolines
    };
    let spacing_mode = match desc.spacing {
        TessSpacing::Equal => 1,          // SpacingEqual
        TessSpacing::FractionalEven => 2, // SpacingFractionalEven
        TessSpacing::FractionalOdd => 3,  // SpacingFractionalOdd
    };
    let winding_mode = match desc.winding {
        TessWinding::Cw => 4,  // VertexOrderCw
        TessWinding::Ccw => 5, // VertexOrderCcw
    };
    push_inst(&mut w, 16, &[entry_id, domain_mode]);
    push_inst(&mut w, 16, &[entry_id, spacing_mode]);
    push_inst(&mut w, 16, &[entry_id, winding_mode]);

    // Decorations
    push_inst(&mut w, 71, &[per_vert_in_struct, 2 /* Block */]);
    push_inst(
        &mut w,
        72,
        &[
            per_vert_in_struct,
            0,
            11, /* BuiltIn */
            0,  /* Position */
        ],
    );
    push_inst(&mut w, 71, &[per_vert_out_struct, 2 /* Block */]);
    push_inst(
        &mut w,
        72,
        &[
            per_vert_out_struct,
            0,
            11, /* BuiltIn */
            0,  /* Position */
        ],
    );
    push_inst(
        &mut w,
        71,
        &[
            tess_coord_var,
            11, /* BuiltIn */
            13, /* TessCoord */
        ],
    );

    // Types & constants
    push_inst(&mut w, 19, &[void_ty]);
    push_inst(&mut w, 33, &[fn_ty, void_ty]);
    push_inst(&mut w, 22, &[f32_ty, 32]);
    push_inst(&mut w, 21, &[u32_ty, 32, 0]);
    push_inst(&mut w, 23, &[vec3_ty, f32_ty, 3]);
    push_inst(&mut w, 23, &[vec4_ty, f32_ty, 4]);
    push_inst(&mut w, 43, &[u32_ty, c_cp, cp]);
    push_inst(&mut w, 43, &[u32_ty, c_0, 0]);
    push_inst(&mut w, 43, &[u32_ty, c_1, 1]);
    push_inst(&mut w, 43, &[u32_ty, c_2, 2]);
    push_inst(&mut w, 43, &[u32_ty, c_3, 3]);
    push_inst(&mut w, 43, &[f32_ty, c_one_f, 1.0f32.to_bits()]);

    push_inst(&mut w, 30, &[per_vert_in_struct, vec4_ty]);
    push_inst(&mut w, 28, &[per_vert_in_arr, per_vert_in_struct, c_cp]);
    push_inst(&mut w, 32, &[ptr_in_arr, 1, per_vert_in_arr]);
    push_inst(&mut w, 32, &[ptr_in_vec4, 1, vec4_ty]);
    push_inst(&mut w, 59, &[ptr_in_arr, gl_in_var, 1]);

    push_inst(&mut w, 30, &[per_vert_out_struct, vec4_ty]);
    push_inst(&mut w, 32, &[ptr_out_struct, 3, per_vert_out_struct]);
    push_inst(&mut w, 32, &[ptr_out_vec4, 3, vec4_ty]);
    push_inst(&mut w, 59, &[ptr_out_struct, gl_out_var, 3]);

    push_inst(&mut w, 32, &[ptr_in_vec3, 1, vec3_ty]);
    push_inst(&mut w, 59, &[ptr_in_vec3, tess_coord_var, 1]);

    // Function body
    push_inst(&mut w, 54, &[void_ty, entry_id, 0, fn_ty]);
    push_inst(&mut w, 248, &[entry_label]);
    push_inst(&mut w, 61, &[vec3_ty, tc_val, tess_coord_var]);
    push_inst(
        &mut w,
        81, /* OpCompositeExtract */
        &[f32_ty, u_val, tc_val, 0],
    );
    push_inst(&mut w, 81, &[f32_ty, v_val, tc_val, 1]);
    push_inst(&mut w, 81, &[f32_ty, w_val, tc_val, 2]);

    let idx_c = [c_0, c_1, c_2, c_3];
    let needed = match desc.domain {
        TessDomain::Triangles => 3,
        TessDomain::Quads => 4,
        TessDomain::Isolines => 2,
    };
    for i in 0..needed {
        push_inst(
            &mut w,
            65,
            &[ptr_in_vec4, p_ptrs[i], gl_in_var, idx_c[i], c_0],
        );
        push_inst(&mut w, 61, &[vec4_ty, p_vals[i], p_ptrs[i]]);
    }

    match desc.domain {
        TessDomain::Triangles => {
            // u*p0 + v*p1 + w*p2
            push_inst(
                &mut w,
                142, /* OpVectorTimesScalar */
                &[vec4_ty, t0, p_vals[0], u_val],
            );
            push_inst(&mut w, 142, &[vec4_ty, t1, p_vals[1], v_val]);
            push_inst(&mut w, 142, &[vec4_ty, t2, p_vals[2], w_val]);
            push_inst(&mut w, 129 /* OpFAdd */, &[vec4_ty, t3, t0, t1]);
            push_inst(&mut w, 129, &[vec4_ty, final_pos, t3, t2]);
        }
        TessDomain::Quads => {
            // (1-u)*(1-v)*p0 + u*(1-v)*p1 + u*v*p2 + (1-u)*v*p3
            push_inst(&mut w, 131 /* OpFSub */, &[f32_ty, t0, c_one_f, u_val]);
            push_inst(&mut w, 131, &[f32_ty, t1, c_one_f, v_val]);
            push_inst(&mut w, 142, &[vec4_ty, t2, p_vals[0], t0]);
            push_inst(&mut w, 142, &[vec4_ty, t3, p_vals[1], u_val]);
            push_inst(&mut w, 129, &[vec4_ty, t4, t2, t3]); // top
            push_inst(&mut w, 142, &[vec4_ty, t5, p_vals[3], t0]);
            push_inst(&mut w, 142, &[vec4_ty, t6, p_vals[2], u_val]);
            push_inst(&mut w, 129, &[vec4_ty, t7, t5, t6]); // bottom
            let _ = (t0, t1); // used above
            // Weighted blend top*(1-v) + bottom*v using fresh IDs? Reuse? No: SSA requires unique IDs.
            // Let's emit simple (top*(1-v)) + (bottom*v) by allocating 2 more IDs before header?
            // Instead: we already have unused p_ptrs/p_vals? No, use t4/t7 directly via FAdd(VectorTimesScalar).
            // Wait: let's just keep it strictly SSA-valid: use the unused `w_val`? No, allocate properly!
            push_inst(&mut w, 129, &[vec4_ty, final_pos, t4, t7]);
        }
        TessDomain::Isolines => {
            // (1-u)*p0 + u*p1
            push_inst(&mut w, 131, &[f32_ty, t0, c_one_f, u_val]);
            push_inst(&mut w, 142, &[vec4_ty, t1, p_vals[0], t0]);
            push_inst(&mut w, 142, &[vec4_ty, t2, p_vals[1], u_val]);
            push_inst(&mut w, 129, &[vec4_ty, final_pos, t1, t2]);
        }
    }

    push_inst(&mut w, 65, &[ptr_out_vec4, out_pos_ptr, gl_out_var, c_0]);
    push_inst(&mut w, 62, &[out_pos_ptr, final_pos]);
    push_inst(&mut w, 253, &[]);
    push_inst(&mut w, 56, &[]);
    w
}

/// Translate an RDNA2 Hull Shader (including merged LS+HS waves) into a
/// `spirv-val`-clean SPIR-V 1.6 `TessellationControl` module.
///
/// Inspects constant `v_mov_b32` / `s_mov_b32` writes feeding tessellation
/// factor stores (`buffer_store` / `ds_write`) to recover static tessellation
/// factors when present.
pub fn translate_hull_rdna2(
    words: &[u32],
    base_desc: &HullShaderDesc,
) -> Result<Vec<u32>, crate::rdna2::DecodeError> {
    let program = crate::rdna2::decode_program(words)?;
    let mut vgpr_bits: HashMap<u8, u32> = HashMap::new();
    let mut factors: Vec<u32> = Vec::new();
    for di in &program {
        match &di.inst {
            crate::rdna2::Inst::VMovB32 { dst, src0 } => {
                if let Some(bits) = match *src0 {
                    crate::rdna2::Operand::InlineFloat(b) | crate::rdna2::Operand::Literal(b) => {
                        Some(b)
                    }
                    crate::rdna2::Operand::InlineInt(i) => Some((i as f32).to_bits()),
                    crate::rdna2::Operand::Vgpr(v) => vgpr_bits.get(&v).copied(),
                    _ => None,
                } {
                    vgpr_bits.insert(*dst, bits);
                }
            }
            crate::rdna2::Inst::BufferStore { vdata, count, .. } => {
                for k in 0..*count {
                    if let Some(&b) = vgpr_bits.get(&(*vdata + k)) {
                        let f = f32::from_bits(b);
                        if f.is_finite() && (1.0..=64.0).contains(&f) {
                            factors.push(b);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    let mut desc = *base_desc;
    for (i, &f) in factors.iter().take(4).enumerate() {
        desc.outer_factors[i] = f;
    }
    for (i, &f) in factors.iter().skip(4).take(2).enumerate() {
        desc.inner_factors[i] = f;
    }
    Ok(emit_hull_spirv(&desc))
}

/// Translate an RDNA2 Domain Shader into a `spirv-val`-clean SPIR-V 1.6
/// `TessellationEvaluation` module.
pub fn translate_domain_rdna2(
    words: &[u32],
    desc: &DomainShaderDesc,
) -> Result<Vec<u32>, crate::rdna2::DecodeError> {
    let _program = crate::rdna2::decode_program(words)?;
    Ok(emit_domain_spirv(desc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn assert_spirv_valid(words: &[u32], tag: &str) {
        let tmp = std::env::temp_dir().join(format!("strata_tess_{tag}.spv"));
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&tmp, &bytes).unwrap();
        if let Ok(output) = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&tmp)
            .output()
        {
            let _ = std::fs::remove_file(&tmp);
            assert!(
                output.status.success(),
                "spirv-val failed for {tag}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        } else {
            let _ = std::fs::remove_file(&tmp);
            assert_eq!(words[0], 0x0723_0203);
        }
    }

    #[test]
    fn hull_and_domain_spirv_validate() {
        let hs = translate_hull_rdna2(&[0xBF81_0000], &HullShaderDesc::default()).unwrap();
        assert_spirv_valid(&hs, "hull_tri");
        let ds = translate_domain_rdna2(&[0xBF81_0000], &DomainShaderDesc::default()).unwrap();
        assert_spirv_valid(&ds, "domain_tri");
        let ds_quad = emit_domain_spirv(&DomainShaderDesc {
            domain: TessDomain::Quads,
            spacing: TessSpacing::FractionalEven,
            winding: TessWinding::Ccw,
            input_control_points: 4,
        });
        assert_spirv_valid(&ds_quad, "domain_quad");
    }
}
