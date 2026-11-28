//! Process-wide log of title-memory writes, for caches that keep GPU copies
//! of title memory (textures) and want to skip re-reading clean data.
//!
//! Writers report every path that changes title memory behind a cache's
//! back: the write-watch page scan of shadowed mirrors (CPU writes by the
//! title), host writes the runtime makes (copies, acceleration structure
//! builds, image write-backs) and mirror readbacks (GPU writes landing in
//! host memory). Only ranges inside a *watched* region — memory whose CPU
//! writes the write-watch scan actually sees — can be trusted to be clean
//! when nothing was logged; everything else must still be verified.

use std::sync::{Arc, Mutex, Weak};

/// Pending ranges before a subscriber collapses to "everything dirty".
const MAX_PENDING: usize = 1 << 16;

/// One subscriber's pending ranges.
#[derive(Debug, Default)]
pub struct DirtyQueue {
    pending: Mutex<Pending>,
}

#[derive(Debug, Default)]
struct Pending {
    ranges: Vec<(u64, u64)>,
    overflow: bool,
}

/// What a subscriber drained.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Drained {
    pub ranges: Vec<(u64, u64)>,
    /// Too many writes were logged: treat everything as dirty.
    pub everything: bool,
}

impl DirtyQueue {
    pub fn drain(&self) -> Drained {
        let mut p = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        Drained {
            ranges: std::mem::take(&mut p.ranges),
            everything: std::mem::take(&mut p.overflow),
        }
    }
}

struct Log {
    subscribers: Vec<Weak<DirtyQueue>>,
    /// `(start, end)` of watched memory, sorted by start.
    watched: Vec<(u64, u64)>,
}

static LOG: Mutex<Log> = Mutex::new(Log {
    subscribers: Vec::new(),
    watched: Vec::new(),
});

fn log() -> std::sync::MutexGuard<'static, Log> {
    LOG.lock().unwrap_or_else(|e| e.into_inner())
}

/// A new subscriber; it sees every range noted from now on.
pub fn subscribe() -> Arc<DirtyQueue> {
    let q = Arc::new(DirtyQueue::default());
    let mut l = log();
    l.subscribers.retain(|w| w.strong_count() > 0);
    l.subscribers.push(Arc::downgrade(&q));
    q
}

/// Reports that `[addr, addr + len)` of title memory changed.
pub fn note(addr: u64, len: u64) {
    if len == 0 {
        return;
    }
    let l = log();
    for s in l.subscribers.iter().filter_map(Weak::upgrade) {
        let mut p = s.pending.lock().unwrap_or_else(|e| e.into_inner());
        if p.overflow {
            continue;
        }
        if p.ranges.len() >= MAX_PENDING {
            p.ranges.clear();
            p.overflow = true;
        } else {
            p.ranges.push((addr, len));
        }
    }
}

/// Marks `[addr, addr + len)` as write-watched (its CPU writes are noted).
/// The region's contents are new, so it is noted dirty too.
pub fn watch(addr: u64, len: u64) {
    {
        let mut l = log();
        let end = addr.saturating_add(len);
        let at = l.watched.partition_point(|r| r.0 < addr);
        l.watched.insert(at, (addr, end));
    }
    note(addr, len);
}

/// Stops trusting the watched region starting at `addr` (freed memory);
/// it is noted dirty, as whatever reuses the addresses holds other data.
pub fn unwatch(addr: u64) {
    let gone: Vec<(u64, u64)> = {
        let mut l = log();
        let (gone, keep) = l.watched.iter().partition(|r| r.0 == addr);
        l.watched = keep;
        gone
    };
    for (a, end) in gone {
        note(a, end - a);
    }
}

/// Whether `[addr, addr + len)` lies inside one watched region.
pub fn is_watched(addr: u64, len: u64) -> bool {
    let end = addr.saturating_add(len);
    let l = log();
    let at = l.watched.partition_point(|r| r.0 <= addr);
    at > 0 && l.watched[at - 1].1 >= end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subscribers_see_notes_and_watch_ranges_answer() {
        let a = subscribe();
        let b = subscribe();
        note(0x5000_0000, 16);
        // (Other tests note ranges concurrently: look for ours.)
        assert!(a.drain().ranges.contains(&(0x5000_0000, 16)));
        assert!(b.drain().ranges.contains(&(0x5000_0000, 16)));
        assert!(!a.drain().ranges.contains(&(0x5000_0000, 16)));

        watch(0x7000_0000, 0x1000);
        assert!(is_watched(0x7000_0100, 0x100));
        assert!(!is_watched(0x7000_0f00, 0x200));
        assert!(!is_watched(0x6fff_ff00, 0x10));
        unwatch(0x7000_0000);
        assert!(!is_watched(0x7000_0100, 0x100));
    }

    #[test]
    fn overflow_collapses_to_everything() {
        let a = subscribe();
        for i in 0..=MAX_PENDING as u64 {
            note(0x9000_0000 + i * 4, 4);
        }
        let d = a.drain();
        assert!(d.everything);
        assert!(d.ranges.is_empty());
    }
}
