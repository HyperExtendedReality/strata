//! Live GPU compute dispatch wrappers for all 7 internal Slang shaders
//! (`msaa_resolve`, `chunk_dirty_hash`, `scatter_upload`, `rect_expand`,
//! `ambient_clamp`, `overlay_hud`, `bcpack_decode`) over 64-bit BDA.

use crate::{context::GpuError, runner::ComputeRunner};

pub(crate) fn spirv_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn write_buf(buf: &mut crate::runner::BdaBuffer, data: &[u8]) {
    // SAFETY: BDA buffers are persistently mapped host-visible memory of at least uf.size bytes.
    let n = data.len().min(buf.size as usize);
    unsafe { buf.as_mut_slice::<u8>(n).copy_from_slice(&data[..n]) };
}

fn read_buf(buf: &crate::runner::BdaBuffer, out: &mut [u8]) {
    let n = out.len().min(buf.size as usize);
    // SAFETY: as in write_buf; the dispatch has completed before readback.
    out[..n].copy_from_slice(unsafe { buf.as_slice::<u8>(n) });
}

fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Push-constant payload for `msaa_resolve.slang`.
pub fn msaa_resolve_push(
    src_addr: u64,
    dst_addr: u64,
    width: u32,
    height: u32,
    sample_count: u32,
    format_mode: u32,
) -> Vec<u8> {
    let mut p = Vec::with_capacity(32);
    push_u64(&mut p, src_addr);
    push_u64(&mut p, dst_addr);
    push_u32(&mut p, width);
    push_u32(&mut p, height);
    push_u32(&mut p, sample_count);
    push_u32(&mut p, format_mode);
    p
}

/// Push-constant payload for `chunk_dirty_hash.slang`.
pub fn chunk_dirty_hash_push(
    src_addr: u64,
    prev_hash_addr: u64,
    dirty_flags_addr: u64,
    chunk_count: u32,
) -> Vec<u8> {
    let mut p = Vec::with_capacity(32);
    push_u64(&mut p, src_addr);
    push_u64(&mut p, prev_hash_addr);
    push_u64(&mut p, dirty_flags_addr);
    push_u32(&mut p, chunk_count);
    push_u32(&mut p, 0);
    p
}

/// Push-constant payload for `scatter_upload.slang`.
pub fn scatter_upload_push(
    dst_addr: u64,
    entries_addr: u64,
    count: u32,
    mode: u32,
    fill_value: u32,
) -> Vec<u8> {
    let mut p = Vec::with_capacity(32);
    push_u64(&mut p, dst_addr);
    push_u64(&mut p, entries_addr);
    push_u32(&mut p, count);
    push_u32(&mut p, mode);
    push_u32(&mut p, fill_value);
    push_u32(&mut p, 0);
    p
}

/// Push-constant payload for `rect_expand.slang`.
pub fn rect_expand_push(src_verts_addr: u64, dst_verts_addr: u64, rect_count: u32) -> Vec<u8> {
    let mut p = Vec::with_capacity(24);
    push_u64(&mut p, src_verts_addr);
    push_u64(&mut p, dst_verts_addr);
    push_u32(&mut p, rect_count);
    push_u32(&mut p, 0);
    p
}

/// Push-constant payload for `ambient_clamp.slang`.
pub fn ambient_clamp_push(
    surface_addr: u64,
    width: u32,
    height: u32,
    ambient_floor: f32,
    max_luminance: f32,
    scrub_nans: bool,
) -> Vec<u8> {
    let mut p = Vec::with_capacity(32);
    push_u64(&mut p, surface_addr);
    push_u32(&mut p, width);
    push_u32(&mut p, height);
    push_f32(&mut p, ambient_floor);
    push_f32(&mut p, max_luminance);
    push_u32(&mut p, u32::from(scrub_nans));
    push_u32(&mut p, 0);
    p
}

/// Push-constant payload for `overlay_hud.slang`.
#[allow(clippy::too_many_arguments)]
pub fn overlay_hud_push(
    fb_addr: u64,
    frametimes_addr: u64,
    fb_width: u32,
    fb_height: u32,
    box_rect: [u32; 4],
    vram_ratio: f32,
    ram_ratio: f32,
    backdrop_alpha: f32,
    sample_count: u32,
) -> Vec<u8> {
    let mut p = Vec::with_capacity(56);
    push_u64(&mut p, fb_addr);
    push_u64(&mut p, frametimes_addr);
    push_u32(&mut p, fb_width);
    push_u32(&mut p, fb_height);
    for v in box_rect {
        push_u32(&mut p, v);
    }
    push_f32(&mut p, vram_ratio);
    push_f32(&mut p, ram_ratio);
    push_f32(&mut p, backdrop_alpha);
    push_u32(&mut p, sample_count);
    p
}

/// Push-constant payload for `bcpack_decode.slang`.
pub fn bcpack_decode_push(
    endpoints_addr: u64,
    indices_addr: u64,
    dst_blocks_addr: u64,
    block_count: u32,
    block_bytes: u32,
    format_id: u32,
) -> Vec<u8> {
    let mut p = Vec::with_capacity(40);
    push_u64(&mut p, endpoints_addr);
    push_u64(&mut p, indices_addr);
    push_u64(&mut p, dst_blocks_addr);
    push_u32(&mut p, block_count);
    push_u32(&mut p, block_bytes);
    push_u32(&mut p, format_id);
    push_u32(&mut p, 0);
    p
}

impl ComputeRunner {
    /// Dispatch `msaa_resolve.slang` on GPU to resolve an RGBA8/RGBA32F/R32F MSAA surface.
    pub fn msaa_resolve_gpu(
        &self,
        src: &[u8],
        width: u32,
        height: u32,
        sample_count: u32,
        format_mode: u32,
    ) -> Result<Vec<u8>, GpuError> {
        let pixel_bytes = if format_mode == 1 { 16usize } else { 4usize };
        let dst_len = (width as usize) * (height as usize) * pixel_bytes;
        if dst_len == 0 || src.is_empty() {
            return Ok(vec![0u8; dst_len]);
        }
        let mut src_buf = self.alloc_bda_buffer(src.len() as u64)?;
        let dst_buf = self.alloc_bda_buffer(dst_len as u64)?;
        write_buf(&mut src_buf, src);
        let pc = msaa_resolve_push(
            src_buf.bda,
            dst_buf.bda,
            width,
            height,
            sample_count.max(1),
            format_mode,
        );
        let words = spirv_words(strata_runtime_shaders::MSAA_RESOLVE_SPV);
        let res = self.dispatch_spirv(&words, &pc, [width.div_ceil(8), height.div_ceil(8), 1]);
        let mut out = vec![0u8; dst_len];
        if res.is_ok() {
            read_buf(&dst_buf, &mut out);
        }
        self.free_bda_buffer(src_buf);
        self.free_bda_buffer(dst_buf);
        res.map(|()| out)
    }

    /// Dispatch `chunk_dirty_hash.slang` over 64-byte chunks, updating `prev_hashes`
    /// and returning per-chunk dirty flags (`1` = changed, `0` = clean).
    pub fn chunk_dirty_hash_gpu(
        &self,
        src: &[u8],
        prev_hashes: &mut [u32],
    ) -> Result<Vec<u32>, GpuError> {
        let chunks = (src.len() / 64).min(prev_hashes.len());
        if chunks == 0 {
            return Ok(Vec::new());
        }
        let mut src_buf = self.alloc_bda_buffer((chunks * 64) as u64)?;
        let mut hash_buf = self.alloc_bda_buffer((chunks * 4) as u64)?;
        let flag_buf = self.alloc_bda_buffer((chunks * 4) as u64)?;
        write_buf(&mut src_buf, &src[..chunks * 64]);
        let prev_bytes: Vec<u8> = prev_hashes[..chunks]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        write_buf(&mut hash_buf, &prev_bytes);
        let pc = chunk_dirty_hash_push(src_buf.bda, hash_buf.bda, flag_buf.bda, chunks as u32);
        let words = spirv_words(strata_runtime_shaders::CHUNK_DIRTY_HASH_SPV);
        let res = self.dispatch_spirv(&words, &pc, [(chunks as u32).div_ceil(64), 1, 1]);
        let mut flags_bytes = vec![0u8; chunks * 4];
        let mut new_hash_bytes = vec![0u8; chunks * 4];
        if res.is_ok() {
            read_buf(&flag_buf, &mut flags_bytes);
            read_buf(&hash_buf, &mut new_hash_bytes);
            for (dst, c) in prev_hashes[..chunks]
                .iter_mut()
                .zip(new_hash_bytes.chunks_exact(4))
            {
                *dst = u32::from_le_bytes(c.try_into().unwrap());
            }
        }
        self.free_bda_buffer(src_buf);
        self.free_bda_buffer(hash_buf);
        self.free_bda_buffer(flag_buf);
        res.map(|()| {
            flags_bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        })
    }

    /// Dispatch `scatter_upload.slang` to scatter `(dword_offset, value)` writes
    /// into a 64-bit BDA destination buffer.
    pub fn scatter_upload_gpu(
        &self,
        dst_addr: u64,
        entries: &[(u32, u32)],
    ) -> Result<(), GpuError> {
        if entries.is_empty() {
            return Ok(());
        }
        let bytes: Vec<u8> = entries
            .iter()
            .flat_map(|&(off, val)| [off.to_le_bytes(), val.to_le_bytes()].concat())
            .collect();
        let mut ent_buf = self.alloc_bda_buffer(bytes.len() as u64)?;
        write_buf(&mut ent_buf, &bytes);
        let pc = scatter_upload_push(dst_addr, ent_buf.bda, entries.len() as u32, 0, 0);
        let words = spirv_words(strata_runtime_shaders::SCATTER_UPLOAD_SPV);
        let res = self.dispatch_spirv(&words, &pc, [(entries.len() as u32).div_ceil(64), 1, 1]);
        self.free_bda_buffer(ent_buf);
        res
    }

    /// Dispatch `rect_expand.slang` to expand N 3-vertex RectList primitives into
    /// 6-vertex triangle-list pairs on GPU.
    pub fn rect_expand_gpu(&self, src_verts: &[[f32; 4]]) -> Result<Vec<[f32; 4]>, GpuError> {
        let rect_count = src_verts.len() / 3;
        if rect_count == 0 {
            return Ok(Vec::new());
        }
        let src_bytes: Vec<u8> = src_verts[..rect_count * 3]
            .iter()
            .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
            .collect();
        let dst_len = rect_count * 6 * 16;
        let mut src_buf = self.alloc_bda_buffer(src_bytes.len() as u64)?;
        let dst_buf = self.alloc_bda_buffer(dst_len as u64)?;
        write_buf(&mut src_buf, &src_bytes);
        let pc = rect_expand_push(src_buf.bda, dst_buf.bda, rect_count as u32);
        let words = spirv_words(strata_runtime_shaders::RECT_EXPAND_SPV);
        let res = self.dispatch_spirv(&words, &pc, [(rect_count as u32).div_ceil(64), 1, 1]);
        let mut out_bytes = vec![0u8; dst_len];
        if res.is_ok() {
            read_buf(&dst_buf, &mut out_bytes);
        }
        self.free_bda_buffer(src_buf);
        self.free_bda_buffer(dst_buf);
        res.map(|()| {
            out_bytes
                .chunks_exact(16)
                .map(|c| {
                    [
                        f32::from_le_bytes(c[0..4].try_into().unwrap()),
                        f32::from_le_bytes(c[4..8].try_into().unwrap()),
                        f32::from_le_bytes(c[8..12].try_into().unwrap()),
                        f32::from_le_bytes(c[12..16].try_into().unwrap()),
                    ]
                })
                .collect()
        })
    }

    /// Dispatch `ambient_clamp.slang` in-place on an RGBA32F BDA surface.
    pub fn ambient_clamp_gpu(
        &self,
        surface_addr: u64,
        width: u32,
        height: u32,
        ambient_floor: f32,
        max_luminance: f32,
        scrub_nans: bool,
    ) -> Result<(), GpuError> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        let pc = ambient_clamp_push(
            surface_addr,
            width,
            height,
            ambient_floor,
            max_luminance,
            scrub_nans,
        );
        let words = spirv_words(strata_runtime_shaders::AMBIENT_CLAMP_SPV);
        self.dispatch_spirv(&words, &pc, [width.div_ceil(8), height.div_ceil(8), 1])
    }

    /// Dispatch `overlay_hud.slang` to composite the performance HUD onto an RGBA8 BDA framebuffer.
    #[allow(clippy::too_many_arguments)]
    pub fn overlay_hud_gpu(
        &self,
        fb_addr: u64,
        frametimes_ms: &[f32],
        fb_width: u32,
        fb_height: u32,
        box_rect: [u32; 4],
        vram_ratio: f32,
        ram_ratio: f32,
        backdrop_alpha: f32,
    ) -> Result<(), GpuError> {
        if box_rect[2] == 0 || box_rect[3] == 0 {
            return Ok(());
        }
        let mut samples = [0f32; 120];
        let count = frametimes_ms.len().min(120);
        samples[..count].copy_from_slice(&frametimes_ms[..count]);
        let ft_bytes: Vec<u8> = samples.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut ft_buf = self.alloc_bda_buffer(ft_bytes.len() as u64)?;
        write_buf(&mut ft_buf, &ft_bytes);
        let pc = overlay_hud_push(
            fb_addr,
            ft_buf.bda,
            fb_width,
            fb_height,
            box_rect,
            vram_ratio,
            ram_ratio,
            backdrop_alpha,
            count as u32,
        );
        let words = spirv_words(strata_runtime_shaders::OVERLAY_HUD_SPV);
        let res = self.dispatch_spirv(
            &words,
            &pc,
            [box_rect[2].div_ceil(8), box_rect[3].div_ceil(8), 1],
        );
        self.free_bda_buffer(ft_buf);
        res
    }

    /// Dispatch `bcpack_decode.slang` to unpack parallel BC1-BC7 blocks on GPU from
    /// decoded endpoint and index streams.
    ///
    /// Not wired into DirectStorage on purpose: the shader only interleaves
    /// already-split endpoint and index words, and neither CPU decoder has
    /// that stage. `bcpack-rs` (BC1/3/4/5/7) emits finished blocks from its
    /// rANS lanes, and the table-driven path (BC2/BC6H) rebuilds each block
    /// from back-references to earlier ones, a serial dependency this shader
    /// does not model. Using it would add a split, two uploads, a dispatch
    /// and a readback into title memory to save no decode work.
    pub fn bcpack_decode_gpu(
        &self,
        endpoints: &[u8],
        indices: &[u8],
        block_count: u32,
        block_bytes: u32,
        format_id: u32,
    ) -> Result<Vec<u8>, GpuError> {
        let out_len = (block_count as usize) * (block_bytes as usize);
        if out_len == 0 || endpoints.is_empty() || indices.is_empty() {
            return Ok(vec![0u8; out_len]);
        }
        let mut ep_buf = self.alloc_bda_buffer(endpoints.len() as u64)?;
        let mut ix_buf = self.alloc_bda_buffer(indices.len() as u64)?;
        let dst_buf = self.alloc_bda_buffer(out_len as u64)?;
        write_buf(&mut ep_buf, endpoints);
        write_buf(&mut ix_buf, indices);
        let pc = bcpack_decode_push(
            ep_buf.bda,
            ix_buf.bda,
            dst_buf.bda,
            block_count,
            block_bytes,
            format_id,
        );
        let words = spirv_words(strata_runtime_shaders::BCPACK_DECODE_SPV);
        let res = self.dispatch_spirv(&words, &pc, [block_count.div_ceil(64), 1, 1]);
        let mut out = vec![0u8; out_len];
        if res.is_ok() {
            read_buf(&dst_buf, &mut out);
        }
        self.free_bda_buffer(ep_buf);
        self.free_bda_buffer(ix_buf);
        self.free_bda_buffer(dst_buf);
        res.map(|()| out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_constant_sizes_match_slang_structs() {
        assert_eq!(msaa_resolve_push(1, 2, 16, 16, 4, 0).len(), 32);
        assert_eq!(chunk_dirty_hash_push(1, 2, 3, 64).len(), 32);
        assert_eq!(scatter_upload_push(1, 2, 8, 0, 0).len(), 32);
        assert_eq!(rect_expand_push(1, 2, 4).len(), 24);
        assert_eq!(ambient_clamp_push(1, 16, 16, 0.02, 10.0, true).len(), 32);
        assert_eq!(
            overlay_hud_push(1, 2, 1920, 1080, [16, 16, 320, 120], 0.5, 0.4, 0.8, 60).len(),
            56
        );
        assert_eq!(bcpack_decode_push(1, 2, 3, 128, 16, 7).len(), 40);
    }
}
