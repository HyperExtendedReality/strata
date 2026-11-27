//! Synthetic RDNA2 kernels with loops and LDS atomics, used by the compiler's
//! own tests (SPIR-V validation) and by `strata-cli`'s live GPU tests.
//!
//! Every kernel reads the thread id from `v0`, writes one dword per thread at
//! `s[0:1] + tid * 4`, and is meant for a 64-thread workgroup.

use crate::rdna2::encode::*;

/// SSRC code of inline integer `n` (0..=64).
const fn inline(n: u8) -> u8 {
    128 + n
}

const VCC_LO: u8 = 106;
const EXEC_LO: u8 = 126;

fn store_v1(prog: &mut Vec<u32>) {
    prog.push(v_lshlrev_b32_imm(2, 2, 0)); // v2 = tid * 4
    prog.extend_from_slice(&global_store_dword(2, 1, 0, 0));
    prog.push(s_endpgm());
}

/// `v1 = 0; for s4 in 0..4 { v1 += tid }` as a do-while: expects `4 * tid`.
pub fn uniform_do_while_sum() -> Vec<u32> {
    let mut p = vec![
        v_mov_b32_sgpr(1, inline(0)), // pc 0
        s_movk_i32(4, 0),             // pc 1
        v_add_nc_u32(1, 1, 0),        // pc 2: L
        s_addk_i32(4, 1),             // pc 3
        s_cmp_lt_u32(4, inline(4)),   // pc 4
        s_cbranch_scc1(-4),           // pc 5 -> pc 2
    ];
    store_v1(&mut p);
    p
}

/// Each lane loops `(tid & 3) + 1` times, leaving by its EXEC bit:
/// expects `(tid & 3) + 1`.
pub fn divergent_exec_loop() -> Vec<u32> {
    let mut p = vec![
        v_and_b32_imm(3, 3, 0),                // pc 0: v3 = tid & 3
        v_mov_b32_sgpr(1, inline(0)),          // pc 1: v1 = 0
        s_mov_b64(6, EXEC_LO),                 // pc 2: s[6:7] = exec
        v_add_nc_u32_imm(1, 1, 1),             // pc 3: L: v1 += 1
        v_add_nc_u32_imm(3, 65, 3),            // pc 4: v3 += -1 (inline 193)
        v_cmp_eq_u32_imm(3, 65),               // pc 5: vcc = v3 == -1
        s_andn2_b64(EXEC_LO, EXEC_LO, VCC_LO), // pc 6: done lanes leave
        s_cbranch_execnz(-5),                  // pc 7 -> pc 3
        s_mov_b64(EXEC_LO, 6),                 // pc 8: exec restored
    ];
    store_v1(&mut p);
    p
}

/// `while s4 < 4 { v1 += tid; s4++ }` with the test at the top:
/// expects `4 * tid`.
pub fn while_loop_sum() -> Vec<u32> {
    let mut p = vec![
        v_mov_b32_sgpr(1, inline(0)), // pc 0
        s_movk_i32(4, 0),             // pc 1
        s_cmp_lt_u32(4, inline(4)),   // pc 2
        s_cbranch_scc0(4),            // pc 3: L: leave for pc 8
        v_add_nc_u32(1, 1, 0),        // pc 4
        s_addk_i32(4, 1),             // pc 5
        s_cmp_lt_u32(4, inline(4)),   // pc 6
        s_branch(-5),                 // pc 7 -> pc 3
    ];
    store_v1(&mut p);
    p
}

/// Two nested uniform do-while loops of 3 and 2 trips: expects `6 * tid`.
pub fn nested_loops() -> Vec<u32> {
    let mut p = vec![
        v_mov_b32_sgpr(1, inline(0)), // pc 0
        s_movk_i32(4, 0),             // pc 1
        s_movk_i32(5, 0),             // pc 2: outer L1
        v_add_nc_u32(1, 1, 0),        // pc 3: inner L2
        s_addk_i32(5, 1),             // pc 4
        s_cmp_lt_u32(5, inline(2)),   // pc 5
        s_cbranch_scc1(-4),           // pc 6 -> pc 3
        s_addk_i32(4, 1),             // pc 7
        s_cmp_lt_u32(4, inline(3)),   // pc 8
        s_cbranch_scc1(-8),           // pc 9 -> pc 2
    ];
    store_v1(&mut p);
    p
}

/// Every lane adds 1 to one LDS dword (after zeroing it): expects 64.
pub fn lds_atomic_count() -> Vec<u32> {
    let mut p = vec![v_mov_b32_sgpr(4, inline(0)), v_mov_b32_sgpr(5, inline(1))];
    p.extend_from_slice(&ds_write_b32(4, 4, 0)); // lds[0] = 0
    p.push(s_barrier());
    p.extend_from_slice(&ds_add_u32(4, 5, 0)); // lds[0] += 1
    p.push(s_waitcnt());
    p.push(s_barrier());
    p.extend_from_slice(&ds_read_b32(1, 4, 0)); // v1 = lds[0]
    p.push(s_waitcnt());
    store_v1(&mut p);
    p
}

/// A V# (buffer descriptor) as user SGPRs: base address, stride, records.
pub fn buffer_descriptor(base: u64, stride: u32, records: u32) -> [u32; 4] {
    [
        base as u32,
        ((base >> 32) as u32 & 0xffff) | ((stride & 0x3fff) << 16),
        records,
        0,
    ]
}

fn store_v4_to_v7(p: &mut Vec<u32>) {
    p.push(s_waitcnt());
    p.push(v_lshlrev_b32_imm(2, 4, 0)); // v2 = tid * 16
    for k in 0..4u8 {
        p.extend_from_slice(&global_store_dword(2, 4 + k, 0, u16::from(k) * 4));
    }
    p.push(s_endpgm());
}

/// Thread `tid` reads RGBA8_UNORM element `tid` of the V# in s[4:7] and
/// writes its four floats to `s[0:1] + tid * 16`.
pub fn typed_load_rgba8_unorm() -> Vec<u32> {
    let mut p = tbuffer_load_format_idx(4, 4, 0, 4, 56, 0).to_vec();
    store_v4_to_v7(&mut p);
    p
}

/// Thread `tid` reads 16_16_SNORM element `tid` (2 components) as four
/// dwords: (x, y, 0, 1.0).
pub fn typed_load_rg16_snorm() -> Vec<u32> {
    let mut p = tbuffer_load_format_idx(4, 4, 0, 4, 24, 0).to_vec();
    store_v4_to_v7(&mut p);
    p
}

/// Thread `tid` writes (1.0, 0.5, 0.0, 1.0) as RGBA8_UNORM element `tid`
/// of the V# in s[4:7].
pub fn typed_store_rgba8_unorm() -> Vec<u32> {
    let mut p = vec![
        v_mov_b32_sgpr(4, 242),
        v_mov_b32_sgpr(5, 240),
        v_mov_b32_sgpr(6, 128),
        v_mov_b32_sgpr(7, 242),
    ];
    p.extend_from_slice(&tbuffer_store_format_idx(4, 4, 0, 4, 56, 0));
    p.push(s_endpgm());
    p
}

/// Thread `tid` reads `10_11_11_FLOAT` (R11G11B10) element `tid` of the V#
/// in s[4:7] as four dwords (r, g, b, 1.0) to `s[0:1] + tid * 16`.
pub fn typed_load_r11g11b10_float() -> Vec<u32> {
    let mut p = tbuffer_load_format_idx(4, 4, 0, 4, 36, 0).to_vec();
    store_v4_to_v7(&mut p);
    p
}

/// Thread `tid` writes (0.5, 1.0, ..) as `format` element `tid` of the V#
/// in s[4:7] (`count` components): a sub-dword store for 8/16-bit formats.
pub fn typed_store_half_one(format: u8, count: u8) -> Vec<u32> {
    let mut p = vec![v_mov_b32_sgpr(4, 240), v_mov_b32_sgpr(5, 242)];
    p.extend_from_slice(&tbuffer_store_format_idx(4, count, 0, 4, format, 0));
    p.push(s_endpgm());
    p
}

/// Thread `tid` fetches texel `(tid & 7, tid >> 3)` of the 2D T# in
/// s[8:15] and writes its four components to `s[0:1] + tid * 16`.
pub fn image_fetch_8x8() -> Vec<u32> {
    let mut p = vec![v_and_b32_imm(10, 7, 0), v_lshrrev_b32_imm(11, 3, 0)];
    p.extend_from_slice(&image_load_2d(4, 10, 8, 0xf));
    store_v4_to_v7(&mut p);
    p
}

/// Every thread samples the T# in s[8:15] with the S# in s[16:19] at
/// (0.3125, 0.3125), level 0, and writes the texel to `s[0:1] + tid * 16`.
pub fn image_sample_lz_fixed() -> Vec<u32> {
    let mut p = v_mov_b32_literal(10, 0.3125f32.to_bits()).to_vec();
    p.extend_from_slice(&v_mov_b32_literal(11, 0.3125f32.to_bits()));
    p.extend_from_slice(&image_sample_lz_2d(4, 10, 8, 16, 0xf));
    store_v4_to_v7(&mut p);
    p
}

/// Thread `tid` writes (1.0, 0.5, 0.0, 1.0) to texel `(tid & 7, tid >> 3)`
/// of the storage T# in s[8:15].
pub fn image_store_8x8() -> Vec<u32> {
    let mut p = vec![
        v_and_b32_imm(10, 7, 0),
        v_lshrrev_b32_imm(11, 3, 0),
        v_mov_b32_sgpr(4, 242),
        v_mov_b32_sgpr(5, 240),
        v_mov_b32_sgpr(6, 128),
        v_mov_b32_sgpr(7, 242),
    ];
    p.extend_from_slice(&image_store_2d(4, 10, 8, 0xf));
    p.push(s_endpgm());
    p
}

/// Vertex shader: a triangle covering the viewport from `VertexIndex`
/// ((-1,-1), (3,-1), (-1,3)), and param0 = (0.25, 0.5, 0.75, 1.0).
pub fn fullscreen_triangle_vs() -> Vec<u32> {
    let mut p = Vec::new();
    for (reg, shift) in [(1u8, 0u8), (2, 1)] {
        if shift == 0 {
            p.push(v_and_b32_imm(reg, 1, 0)); // x: id & 1
        } else {
            p.push(v_lshrrev_b32_imm(reg, 1, 0)); // y: id >> 1
        }
        p.push(v_cvt_f32_u32(reg, reg));
        p.push(v_mul_f32_code(reg, 246, reg)); // * 4.0
        p.push(v_add_f32_code(reg, 243, reg)); // + -1.0
    }
    p.push(v_mov_b32_sgpr(3, 128)); // z = 0
    p.push(v_mov_b32_sgpr(4, 242)); // w = 1
    p.extend_from_slice(&exp(12, 0xf, [1, 2, 3, 4], true, false)); // pos0
    p.extend_from_slice(&v_mov_b32_literal(5, 0.25f32.to_bits()));
    p.push(v_mov_b32_sgpr(6, 240)); // 0.5
    p.extend_from_slice(&v_mov_b32_literal(7, 0.75f32.to_bits()));
    p.push(v_mov_b32_sgpr(8, 242)); // 1.0
    p.extend_from_slice(&exp(32, 0xf, [5, 6, 7, 8], false, false)); // param0
    p.push(s_endpgm());
    p
}

/// Pixel shader: mrt0 = param0, interpolated (`v_interp_p1` + `p2`).
pub fn interpolated_color_ps() -> Vec<u32> {
    let mut p = Vec::new();
    for chan in 0..4u8 {
        p.push(v_interp(0, 5 + chan, 0, chan, 0)); // p1 with i in v0
        p.push(v_interp(1, 5 + chan, 0, chan, 1)); // p2 with j in v1
    }
    p.extend_from_slice(&exp(0, 0xf, [5, 6, 7, 8], true, false));
    p.push(s_endpgm());
    p
}

/// Pixel shader: mrt0 = param0, flat (`v_interp_mov`).
pub fn flat_color_ps() -> Vec<u32> {
    let mut p = Vec::new();
    for chan in 0..4u8 {
        p.push(v_interp(2, 5 + chan, 0, chan, 2));
    }
    p.extend_from_slice(&exp(0, 0xf, [5, 6, 7, 8], true, false));
    p.push(s_endpgm());
    p
}

/// Pixel shader: mrt0 = the T# in s[8:15] sampled (implicit LOD) with the
/// S# in s[16:19] at param0.xy.
pub fn sampled_color_ps() -> Vec<u32> {
    let mut p = vec![v_interp(2, 10, 0, 0, 2), v_interp(2, 11, 0, 1, 2)];
    let [w0, w1] = image_sample_lz_2d(5, 10, 8, 16, 0xf);
    p.extend_from_slice(&[(w0 & !(0xff << 18)) | (0x20 << 18), w1]); // image_sample (implicit)
    p.extend_from_slice(&exp(0, 0xf, [5, 6, 7, 8], true, false));
    p.push(s_endpgm());
    p
}

/// Pixel shader using a quad broadcast (helper lanes must take part).
pub fn quad_broadcast_ps() -> Vec<u32> {
    let mut p = vec![v_interp(2, 5, 0, 0, 2)];
    p.extend_from_slice(&ds_swizzle(6, 5, 0x8000)); // every lane reads quad lane 0
    p.push(v_mov_b32_sgpr(7, 128));
    p.push(v_mov_b32_sgpr(8, 242));
    p.extend_from_slice(&exp(0, 0xf, [6, 6, 7, 8], true, false));
    p.push(s_endpgm());
    p
}
