//! Guest thread and stack management for console titles.
//!
//! Tracks guest-created threads, stack reserve/commit sizes, guard pages,
//! 8-core Zen 2 affinity masks, and thread names so the loader can cleanly
//! manage thread lifecycles and crash diagnostics.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

pub const DEFAULT_STACK_RESERVE: usize = 2 * 1024 * 1024; // 2 MiB
pub const DEFAULT_STACK_COMMIT: usize = 64 * 1024; // 64 KiB
pub const CONSOLE_CORE_MASK: u64 = 0xFF; // 8 Zen 2 logical cores

#[derive(Debug, Clone)]
pub struct GuestThreadInfo {
    pub id: u32,
    pub os_tid: u32,
    pub name: String,
    pub stack_reserve: usize,
    pub stack_commit: usize,
    pub affinity_mask: u64,
    pub priority: i32,
    pub alive: bool,
}

#[derive(Debug)]
pub struct ThreadManager {
    next_id: AtomicU32,
    threads: HashMap<u32, GuestThreadInfo>,
    pub max_cores: u32,
}

impl Default for ThreadManager {
    fn default() -> Self {
        Self::new(8)
    }
}

impl ThreadManager {
    pub fn new(max_cores: u32) -> Self {
        Self {
            next_id: AtomicU32::new(1),
            threads: HashMap::new(),
            max_cores: max_cores.clamp(1, 64),
        }
    }

    /// Normalize a guest-requested stack size to console page-aligned bounds.
    pub fn normalize_stack_size(requested: usize) -> usize {
        if requested == 0 {
            DEFAULT_STACK_RESERVE
        } else {
            requested
                .clamp(DEFAULT_STACK_COMMIT, 16 * 1024 * 1024)
                .next_multiple_of(4096)
        }
    }

    /// Clamp a guest affinity mask to the host/console core topology.
    pub fn clamp_affinity(&self, requested: u64) -> u64 {
        let valid_mask = if self.max_cores >= 64 {
            u64::MAX
        } else {
            (1u64 << self.max_cores) - 1
        };
        let masked = requested & valid_mask;
        if masked == 0 { valid_mask } else { masked }
    }

    pub fn register_thread(
        &mut self,
        os_tid: u32,
        name: impl Into<String>,
        stack_size: usize,
    ) -> u32 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let reserve = Self::normalize_stack_size(stack_size);
        self.threads.insert(
            id,
            GuestThreadInfo {
                id,
                os_tid,
                name: name.into(),
                stack_reserve: reserve,
                stack_commit: DEFAULT_STACK_COMMIT.min(reserve),
                affinity_mask: self.clamp_affinity(CONSOLE_CORE_MASK),
                priority: 0,
                alive: true,
            },
        );
        id
    }

    pub fn set_name(&mut self, id: u32, name: impl Into<String>) {
        if let Some(t) = self.threads.get_mut(&id) {
            t.name = name.into();
        }
    }

    pub fn set_affinity(&mut self, id: u32, mask: u64) -> Option<u64> {
        let clamped = self.clamp_affinity(mask);
        let t = self.threads.get_mut(&id)?;
        let prev = t.affinity_mask;
        t.affinity_mask = clamped;
        Some(prev)
    }

    pub fn mark_exited(&mut self, id: u32) {
        if let Some(t) = self.threads.get_mut(&id) {
            t.alive = false;
        }
    }

    pub fn active_count(&self) -> usize {
        self.threads.values().filter(|t| t.alive).count()
    }

    pub fn get(&self, id: u32) -> Option<&GuestThreadInfo> {
        self.threads.get(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_manager_normalizes_stacks_and_affinity() {
        let mut tm = ThreadManager::new(8);
        let id = tm.register_thread(1001, "RenderThread", 100_000);
        let info = tm.get(id).unwrap();
        assert_eq!(info.stack_reserve % 4096, 0);
        assert_eq!(tm.clamp_affinity(0xFFFF_FFFF), 0xFF);
        assert_eq!(tm.active_count(), 1);
        tm.mark_exited(id);
        assert_eq!(tm.active_count(), 0);
    }
}
