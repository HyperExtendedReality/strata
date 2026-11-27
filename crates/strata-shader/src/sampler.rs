//! Sampler variants (1D/2D/3D/Cube/CubeArray/2DArray, `_c`, `_d`, `_l`, `_b`,
//! `_o`, `_cl`, `_lz`, `gather4`, `get_lod`, `get_resinfo`) and GFX10 Cube-map
//! projection + Cube LOD math (`v_cubeid_f32`, `v_cubesc_f32`, `v_cubetc_f32`,
//! `v_cubema_f32`).

/// Full set of MIMG sampling / query variants in GFX10.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SampleModifiers {
    pub compare: bool,      // _c
    pub derivatives: bool,  // _d
    pub explicit_lod: bool, // _l
    pub bias: bool,         // _b
    pub offset: bool,       // _o
    pub clamp: bool,        // _cl
    pub lod_zero: bool,     // _lz
    pub gather4: bool,      // image_gather4*
    pub get_lod: bool,      // image_get_lod
    pub resinfo: bool,      // image_get_resinfo
}

impl SampleModifiers {
    /// Decode GFX10 MIMG opcode (`0x00..=0x7F`) into structured sampling modifiers.
    pub fn from_mimg_opcode(op: u8) -> Option<Self> {
        let mut m = Self {
            compare: false,
            derivatives: false,
            explicit_lod: false,
            bias: false,
            offset: false,
            clamp: false,
            lod_zero: false,
            gather4: false,
            get_lod: false,
            resinfo: false,
        };
        match op {
            0x0e => {
                m.resinfo = true;
                Some(m)
            }
            0x60 => {
                m.get_lod = true;
                Some(m)
            }
            0x20..=0x3f => {
                let sub = op - 0x20;
                m.offset = (sub & 0x10) != 0;
                m.compare = (sub & 0x08) != 0;
                match sub & 0x07 {
                    0 => {}
                    1 => m.clamp = true,
                    2 => m.derivatives = true,
                    3 => {
                        m.derivatives = true;
                        m.clamp = true;
                    }
                    4 => m.explicit_lod = true,
                    5 => m.bias = true,
                    6 => {
                        m.bias = true;
                        m.clamp = true;
                    }
                    7 => m.lod_zero = true,
                    _ => unreachable!(),
                }
                Some(m)
            }
            0x40..=0x5f => {
                m.gather4 = true;
                let sub = op - 0x40;
                m.offset = (sub & 0x10) != 0;
                m.compare = (sub & 0x08) != 0;
                match sub & 0x07 {
                    0 => {}
                    1 => m.clamp = true,
                    4 => m.explicit_lod = true,
                    5 => m.bias = true,
                    6 => {
                        m.bias = true;
                        m.clamp = true;
                    }
                    7 => m.lod_zero = true,
                    _ => {}
                }
                Some(m)
            }
            _ => None,
        }
    }
}

/// Result of GFX10 cube-map face selection (`v_cubeid`/`v_cubesc`/`v_cubetc`/`v_cubema`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CubeProjection {
    pub face_id: f32,
    pub sc: f32,
    pub tc: f32,
    pub ma: f32,
    pub uv: [f32; 2],
}

/// Evaluate AMD GFX10 cube-map instructions `(v_cubeid, v_cubesc, v_cubetc, v_cubema)`
/// and compute normalized face UV coordinates.
pub fn eval_cube_projection(x: f32, y: f32, z: f32) -> CubeProjection {
    let ax = x.abs();
    let ay = y.abs();
    let az = z.abs();
    let (face_id, sc, tc, ma) = if az >= ax && az >= ay {
        if z < 0.0 {
            (5.0, -x, -y, -2.0 * z)
        } else {
            (4.0, x, -y, 2.0 * z)
        }
    } else if ay >= ax {
        if y < 0.0 {
            (3.0, x, -z, -2.0 * y)
        } else {
            (2.0, x, z, 2.0 * y)
        }
    } else if x < 0.0 {
        (1.0, z, -y, -2.0 * x)
    } else {
        (0.0, -z, -y, 2.0 * x)
    };
    let inv_ma = if ma.abs() > 1e-20 {
        1.0 / ma.abs()
    } else {
        0.0
    };
    CubeProjection {
        face_id,
        sc,
        tc,
        ma,
        uv: [sc * inv_ma + 0.5, tc * inv_ma + 0.5],
    }
}

/// Compute cube-map LOD from face-space derivatives `(ddx, ddy)` and face resolution `face_size`.
pub fn compute_cube_lod(ddx_uv: [f32; 2], ddy_uv: [f32; 2], face_size: f32) -> f32 {
    let sx = (ddx_uv[0] * face_size).hypot(ddx_uv[1] * face_size);
    let sy = (ddy_uv[0] * face_size).hypot(ddy_uv[1] * face_size);
    let rho = sx.max(sy).max(1e-8);
    rho.log2().max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_mimg_sampler_variants_and_cube_lod() {
        let m = SampleModifiers::from_mimg_opcode(0x2c).unwrap(); // image_sample_c_l
        assert!(m.compare && m.explicit_lod && !m.gather4);
        let g = SampleModifiers::from_mimg_opcode(0x47).unwrap(); // image_gather4_lz
        assert!(g.gather4 && g.lod_zero);

        let proj = eval_cube_projection(0.0, 0.0, 2.0);
        assert_eq!(proj.face_id, 4.0);
        assert!((proj.uv[0] - 0.5).abs() < 1e-6);
        assert!((proj.uv[1] - 0.5).abs() < 1e-6);

        let lod = compute_cube_lod([0.0625, 0.0], [0.0, 0.0625], 256.0);
        assert!((lod - 4.0).abs() < 1e-5); // 0.0625 * 256 = 16 -> log2(16) = 4
    }
}
