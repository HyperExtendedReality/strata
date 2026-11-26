//! Content-addressed on-disk SPIR-V shader cache.
//!
//! Keys translated SPIR-V binaries by `(compiler_version, FNV-1a-64(rdna2_bytes))`
//! using atomic `.tmp` rename writes so concurrent/crashed runs never corrupt cache entries.

use std::path::{Path, PathBuf};

pub const COMPILER_VERSION: u32 = 1;

pub fn hash_rdna2(words: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ (COMPILER_VERSION as u64);
    for &w in words {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

pub struct ShaderDiskCache {
    dir: PathBuf,
}

impl ShaderDiskCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn entry_path(&self, key: u64) -> PathBuf {
        self.dir.join(format!("v{COMPILER_VERSION}_{key:016x}.spv"))
    }

    pub fn load(&self, rdna2_words: &[u32]) -> Option<Vec<u32>> {
        let path = self.entry_path(hash_rdna2(rdna2_words));
        let bytes = std::fs::read(path).ok()?;
        if bytes.len() < 20 || bytes.len() % 4 != 0 {
            return None;
        }
        Some(
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )
    }

    pub fn store(&self, rdna2_words: &[u32], spirv: &[u32]) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let path = self.entry_path(hash_rdna2(rdna2_words));
        let tmp = path.with_extension("spv.tmp");
        let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_spirv_through_atomic_disk_cache() {
        let dir = std::env::temp_dir().join("strata_shader_cache_test");
        let cache = ShaderDiskCache::new(&dir);
        let key_prog = [0xbf81_0000u32, 0x1234_5678];
        let spv = vec![0x0723_0203u32, 0x0001_0600, 0, 10, 0, 0x0002_0011];

        cache.store(&key_prog, &spv).unwrap();
        assert_eq!(cache.load(&key_prog), Some(spv));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
