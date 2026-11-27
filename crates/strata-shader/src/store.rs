//! Persistent versioned shader store for ahead-of-time batch translation.
//!
//! Wraps [`crate::cache::ShaderDiskCache`] with a line-delimited manifest
//! (`manifest_v<N>.tsv`) recording each shader's content hash, stage, source
//! tag, and SPIR-V word count so interrupted batch runs can resume without
//! re-translating already-stored entries.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::cache::{COMPILER_VERSION, ShaderDiskCache, hash_rdna2};
use crate::container::ShaderStage;
use crate::coverage::{Outcome, Program, translate};

/// One persisted shader entry in the versioned store manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreRecord {
    pub hash: u64,
    pub stage: String,
    pub source: String,
    pub spirv_words: usize,
}

/// Statistics from a batch pre-translation pass.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BatchStats {
    pub total: usize,
    pub skipped_cached: usize,
    pub newly_translated: usize,
    pub failed: usize,
}

/// Versioned on-disk shader store with incremental manifest tracking.
pub struct ShaderStore {
    cache: ShaderDiskCache,
    records: BTreeMap<u64, StoreRecord>,
}

impl ShaderStore {
    pub fn open(dir: impl Into<PathBuf>) -> Self {
        let cache = ShaderDiskCache::new(dir);
        let records = Self::load_manifest(cache.dir());
        Self { cache, records }
    }

    fn manifest_path(dir: &Path) -> PathBuf {
        dir.join(format!("manifest_v{COMPILER_VERSION}.tsv"))
    }

    fn load_manifest(dir: &Path) -> BTreeMap<u64, StoreRecord> {
        let mut map = BTreeMap::new();
        let Ok(text) = std::fs::read_to_string(Self::manifest_path(dir)) else {
            return map;
        };
        for line in text.lines() {
            let mut cols = line.split('\t');
            let (Some(h), Some(stage), Some(words), Some(source)) =
                (cols.next(), cols.next(), cols.next(), cols.next())
            else {
                continue;
            };
            if let (Ok(hash), Ok(spirv_words)) =
                (u64::from_str_radix(h, 16), words.parse::<usize>())
            {
                map.insert(
                    hash,
                    StoreRecord {
                        hash,
                        stage: stage.to_string(),
                        source: source.to_string(),
                        spirv_words,
                    },
                );
            }
        }
        map
    }

    fn save_manifest(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(self.cache.dir())?;
        let mut buf = String::new();
        for rec in self.records.values() {
            buf.push_str(&format!(
                "{:016x}\t{}\t{}\t{}\n",
                rec.hash,
                rec.stage,
                rec.spirv_words,
                rec.source.replace(['\t', '\n', '\r'], " ")
            ));
        }
        let path = Self::manifest_path(self.cache.dir());
        let tmp = path.with_extension("tsv.tmp");
        std::fs::write(&tmp, buf)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn contains_words(&self, rdna2_words: &[u32]) -> bool {
        let key = hash_rdna2(rdna2_words);
        self.records.contains_key(&key) && self.cache.load(rdna2_words).is_some()
    }

    pub fn load(&self, rdna2_words: &[u32]) -> Option<Vec<u32>> {
        self.cache.load(rdna2_words)
    }

    pub fn records(&self) -> impl Iterator<Item = &StoreRecord> {
        self.records.values()
    }

    /// Translate and persist every program not already in the store.
    pub fn ingest_programs(&mut self, programs: &[Program]) -> std::io::Result<BatchStats> {
        let mut stats = BatchStats::default();
        for prog in programs {
            stats.total += 1;
            let key = hash_rdna2(&prog.words);
            if self.records.contains_key(&key) && self.cache.load(&prog.words).is_some() {
                stats.skipped_cached += 1;
                continue;
            }
            let report = translate(prog);
            match report.outcome {
                Outcome::Translated { spirv_words } => {
                    let stage_opts = match prog.stage {
                        Some(ShaderStage::Vertex) => crate::ssa::BuildOptions {
                            stage: crate::ssa::Stage::Vertex,
                            ..Default::default()
                        },
                        Some(ShaderStage::Pixel) => crate::ssa::BuildOptions {
                            stage: crate::ssa::Stage::Pixel,
                            ..Default::default()
                        },
                        _ => crate::ssa::BuildOptions::default(),
                    };
                    if let Ok(compiled) =
                        crate::compile_rdna2_with(&prog.words, [64, 1, 1], &stage_opts)
                    {
                        self.cache.store(&prog.words, &compiled.spirv)?;
                        let stage_name = match prog.stage {
                            Some(s) => format!("{s:?}"),
                            None => "Compute".to_string(),
                        };
                        self.records.insert(
                            key,
                            StoreRecord {
                                hash: key,
                                stage: stage_name,
                                source: prog.source.clone(),
                                spirv_words,
                            },
                        );
                        stats.newly_translated += 1;
                    } else {
                        stats.failed += 1;
                    }
                }
                Outcome::Failed(_) => {
                    stats.failed += 1;
                }
            }
        }
        self.save_manifest()?;
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna2::encode;

    #[test]
    fn persists_and_resumes_batch_translation() {
        let dir = std::env::temp_dir().join("strata_shader_store_test");
        let _ = std::fs::remove_dir_all(&dir);
        let prog = Program {
            source: "synthetic.sgd5@0x10".to_string(),
            stage: Some(ShaderStage::Compute),
            words: vec![encode::s_endpgm()],
        };

        let mut store = ShaderStore::open(&dir);
        let first = store.ingest_programs(std::slice::from_ref(&prog)).unwrap();
        assert_eq!(first.newly_translated, 1);
        assert_eq!(first.skipped_cached, 0);

        // Re-open and verify incremental resume skips already-stored entry.
        let mut reopened = ShaderStore::open(&dir);
        assert_eq!(reopened.len(), 1);
        let second = reopened.ingest_programs(&[prog]).unwrap();
        assert_eq!(second.newly_translated, 0);
        assert_eq!(second.skipped_cached, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
