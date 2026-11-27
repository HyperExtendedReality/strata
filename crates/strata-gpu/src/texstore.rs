//! Persistent content-addressed Texture Store kept between runs, plus CPU/GPU
//! texture conversion helpers paired with `tex_convert.slang` and `mip_gen.slang`.

use std::path::{Path, PathBuf};

/// FNV-1a 64-bit content hash for fast texture cache keying.
pub fn hash_texture_key(format_code: u32, width: u32, height: u32, raw_bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in [format_code, width, height] {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    for &b in raw_bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// Persistent on-disk store for converted/detiled textures kept between runs.
#[derive(Debug, Clone)]
pub struct TextureStore {
    root: PathBuf,
    pub hits: u64,
    pub misses: u64,
}

impl TextureStore {
    pub fn open(root: impl Into<PathBuf>) -> std::io::Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            hits: 0,
            misses: 0,
        })
    }

    fn path_for(&self, key: u64) -> PathBuf {
        self.root.join(format!("{key:016x}.tex"))
    }

    pub fn get(&mut self, key: u64) -> Option<Vec<u8>> {
        let path = self.path_for(key);
        match std::fs::read(&path) {
            Ok(bytes) => {
                self.hits += 1;
                Some(bytes)
            }
            Err(_) => {
                self.misses += 1;
                None
            }
        }
    }

    /// Atomically persist converted texture bytes to disk.
    pub fn put(&self, key: u64, data: &[u8]) -> std::io::Result<()> {
        let final_path = self.path_for(key);
        let tmp_path = self.root.join(format!("{key:016x}.tmp"));
        std::fs::write(&tmp_path, data)?;
        std::fs::rename(tmp_path, final_path)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

/// CPU reference matching `tex_convert.slang` mode 0 (BGRA8 -> RGBA8).
pub fn convert_bgra8_to_rgba8(src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
        d[3] = s[3];
    }
}

/// CPU reference matching `tex_convert.slang` mode 1 (R10G10B10A2_UNORM -> RGBA8_UNORM).
pub fn convert_r10g10b10a2_to_rgba8(src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let p = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        d[0] = ((p & 0x3FF) >> 2) as u8;
        d[1] = (((p >> 10) & 0x3FF) >> 2) as u8;
        d[2] = (((p >> 20) & 0x3FF) >> 2) as u8;
        d[3] = (((p >> 30) & 0x3) * 85) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_store_persists_between_opens() {
        let dir = std::env::temp_dir().join("strata_texstore_test");
        let _ = std::fs::remove_dir_all(&dir);
        let key = hash_texture_key(28, 4, 4, &[1, 2, 3, 4]);
        {
            let mut store = TextureStore::open(&dir).unwrap();
            assert!(store.get(key).is_none());
            store.put(key, &[9, 8, 7, 6]).unwrap();
        }
        {
            let mut store = TextureStore::open(&dir).unwrap();
            assert_eq!(store.get(key).as_deref(), Some(&[9, 8, 7, 6][..]));
            assert_eq!(store.hits, 1);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
