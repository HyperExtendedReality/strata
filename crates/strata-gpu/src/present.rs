//! `PresentX` Frame-Pacing Clock, Atomic Disk `VkPipelineCache`, & No-API
//! `VK_KHR_dynamic_rendering` Presenter.
//!
//! Ensures every console frame-event type due at a deadline fires together
//! (preventing the black-screen hang seen in XCL when only one event type
//! fired per deadline), and persists Vulkan pipeline caches via atomic
//! temp-file rename (avoiding victory-fork's mapped-file device loss).

use std::path::Path;

/// Save raw `VkPipelineCache` bytes atomically (write `.tmp` then rename) so a
/// live mapped cache file is never overwritten in place.
pub fn save_pipeline_cache_atomic(path: &Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("vkcache.tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn load_pipeline_cache(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_default()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameEventKind {
    Origin = 0,
    Present = 1,
    CustomType2 = 2,
}

#[derive(Debug, Clone, Copy)]
pub struct ScheduledFrameEvent {
    pub kind: FrameEventKind,
    pub due_tick: u64,
    pub fired: bool,
}

#[derive(Debug)]
pub struct FramePacingClock {
    pub current_tick: u64,
    pub target_fps: u32,
    pub presented_frames: u64,
    events: Vec<ScheduledFrameEvent>,
}

impl FramePacingClock {
    pub fn new(target_fps: u32) -> Self {
        Self {
            current_tick: 0,
            target_fps: target_fps.max(1),
            presented_frames: 0,
            events: Vec::new(),
        }
    }

    pub fn schedule(&mut self, kind: FrameEventKind, due_tick: u64) {
        self.events.push(ScheduledFrameEvent {
            kind,
            due_tick,
            fired: false,
        });
    }

    /// Advance the clock to `tick` and fire **all** event kinds due at or before `tick`.
    pub fn advance_and_fire(&mut self, tick: u64) -> Vec<FrameEventKind> {
        self.current_tick = self.current_tick.max(tick);
        let mut fired = Vec::new();
        for ev in &mut self.events {
            if !ev.fired && ev.due_tick <= self.current_tick {
                ev.fired = true;
                fired.push(ev.kind);
            }
        }
        self.events.retain(|e| !e.fired);
        fired
    }

    pub fn record_present_x(&mut self) -> u64 {
        self.presented_frames += 1;
        self.presented_frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fires_all_event_types_sharing_the_same_deadline() {
        let mut clock = FramePacingClock::new(30);
        clock.schedule(FrameEventKind::Origin, 100);
        clock.schedule(FrameEventKind::CustomType2, 100);
        clock.schedule(FrameEventKind::Present, 120);

        let fired = clock.advance_and_fire(100);
        assert_eq!(
            fired,
            vec![FrameEventKind::Origin, FrameEventKind::CustomType2]
        );

        let fired2 = clock.advance_and_fire(120);
        assert_eq!(fired2, vec![FrameEventKind::Present]);
        assert_eq!(clock.record_present_x(), 1);
    }
}
