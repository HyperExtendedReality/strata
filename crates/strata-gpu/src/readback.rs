//! Background dword-diff GPU→CPU readback relay.
//!
//! When shaders write to textures or buffers that the title's CPU reads back,
//! landing only the 32-bit dwords that actually changed avoids unnecessary
//! cache-line invalidation and write-guard churn.

/// Summary of one dword-diff readback landing pass.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LandingStats {
    pub dwords_scanned: usize,
    pub dwords_changed: usize,
    pub bytes_landed: usize,
}

/// Compare `gpu_shadow` against `title_dst` in 32-bit dword units and write
/// only the differing dwords into `title_dst`.
pub fn land_changed_dwords(gpu_shadow: &[u8], title_dst: &mut [u8]) -> LandingStats {
    let len = gpu_shadow.len().min(title_dst.len());
    let dword_count = len / 4;
    let mut stats = LandingStats {
        dwords_scanned: dword_count,
        dwords_changed: 0,
        bytes_landed: 0,
    };

    for i in 0..dword_count {
        let off = i * 4;
        let src_dw = u32::from_le_bytes(gpu_shadow[off..off + 4].try_into().unwrap());
        let dst_dw = u32::from_le_bytes(title_dst[off..off + 4].try_into().unwrap());
        if src_dw != dst_dw {
            title_dst[off..off + 4].copy_from_slice(&src_dw.to_le_bytes());
            stats.dwords_changed += 1;
            stats.bytes_landed += 4;
        }
    }

    // Handle any trailing 1..3 bytes.
    for i in (dword_count * 4)..len {
        if gpu_shadow[i] != title_dst[i] {
            title_dst[i] = gpu_shadow[i];
            stats.bytes_landed += 1;
        }
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lands_only_changed_dwords() {
        let mut dst = vec![0u8; 64];
        let mut shadow = vec![0u8; 64];
        shadow[8..12].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        shadow[40..44].copy_from_slice(&0x1234_5678u32.to_le_bytes());

        let stats = land_changed_dwords(&shadow, &mut dst);
        assert_eq!(stats.dwords_scanned, 16);
        assert_eq!(stats.dwords_changed, 2);
        assert_eq!(stats.bytes_landed, 8);
        assert_eq!(dst, shadow);

        // Second pass with identical contents lands zero dwords.
        let second = land_changed_dwords(&shadow, &mut dst);
        assert_eq!(second.dwords_changed, 0);
        assert_eq!(second.bytes_landed, 0);
    }
}
