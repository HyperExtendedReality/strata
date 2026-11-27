//! GPU Timeout Detection and Recovery (TDR) & breadcrumb FaultBook.
//!
//! Tracks submitted command-list breadcrumbs so that if Vulkan returns
//! `VK_ERROR_DEVICE_LOST`, Strata identifies the exact offending pass/shader,
//! records it in the fault book, and coordinates clean device recovery.

use ash::vk;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceHealth {
    Healthy,
    DeviceLost,
    Recovering,
    Recovered,
}

#[derive(Debug, Clone)]
pub struct Breadcrumb {
    pub seq: u64,
    pub label: String,
    pub shader_hash: u64,
    pub completed: bool,
}

#[derive(Debug, Clone)]
pub struct FaultRecord {
    pub recovery_attempt: u32,
    pub last_completed_seq: u64,
    pub suspected_pass: Option<String>,
    pub suspected_shader: Option<u64>,
}

/// Watchdog and breadcrumb fault-book for GPU TDR recovery.
#[derive(Debug)]
pub struct TdrController {
    pub health: DeviceHealth,
    next_seq: u64,
    breadcrumbs: VecDeque<Breadcrumb>,
    max_breadcrumbs: usize,
    pub faults: Vec<FaultRecord>,
    pub recovery_count: u32,
}

impl Default for TdrController {
    fn default() -> Self {
        Self::new()
    }
}

impl TdrController {
    pub fn new() -> Self {
        Self {
            health: DeviceHealth::Healthy,
            next_seq: 1,
            breadcrumbs: VecDeque::with_capacity(256),
            max_breadcrumbs: 256,
            faults: Vec::new(),
            recovery_count: 0,
        }
    }

    pub fn push_breadcrumb(&mut self, label: impl Into<String>, shader_hash: u64) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        if self.breadcrumbs.len() >= self.max_breadcrumbs {
            self.breadcrumbs.pop_front();
        }
        self.breadcrumbs.push_back(Breadcrumb {
            seq,
            label: label.into(),
            shader_hash,
            completed: false,
        });
        seq
    }

    pub fn mark_completed_up_to(&mut self, seq: u64) {
        for b in &mut self.breadcrumbs {
            if b.seq <= seq {
                b.completed = true;
            }
        }
    }

    /// Inspect a Vulkan result; if `ERROR_DEVICE_LOST`, log a fault record and enter recovery.
    pub fn check_vk_result(&mut self, result: vk::Result) -> DeviceHealth {
        if result == vk::Result::ERROR_DEVICE_LOST {
            self.health = DeviceHealth::DeviceLost;
            let last_done = self
                .breadcrumbs
                .iter()
                .rev()
                .find(|b| b.completed)
                .map_or(0, |b| b.seq);
            let suspect = self.breadcrumbs.iter().find(|b| !b.completed);
            self.recovery_count += 1;
            self.faults.push(FaultRecord {
                recovery_attempt: self.recovery_count,
                last_completed_seq: last_done,
                suspected_pass: suspect.map(|b| b.label.clone()),
                suspected_shader: suspect.map(|b| b.shader_hash),
            });
        }
        self.health
    }

    pub fn begin_recovery(&mut self) {
        self.health = DeviceHealth::Recovering;
        self.breadcrumbs.clear();
    }

    pub fn finish_recovery(&mut self) {
        self.health = DeviceHealth::Recovered;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tdr_identifies_culprit_breadcrumb_and_recovers() {
        let mut tdr = TdrController::new();
        let s1 = tdr.push_breadcrumb("ShadowPass", 0x1111);
        let _s2 = tdr.push_breadcrumb("VolumetricFog", 0x2222);
        tdr.mark_completed_up_to(s1);

        assert_eq!(
            tdr.check_vk_result(vk::Result::ERROR_DEVICE_LOST),
            DeviceHealth::DeviceLost
        );
        assert_eq!(tdr.faults.len(), 1);
        assert_eq!(
            tdr.faults[0].suspected_pass.as_deref(),
            Some("VolumetricFog")
        );
        assert_eq!(tdr.faults[0].suspected_shader, Some(0x2222));

        tdr.begin_recovery();
        tdr.finish_recovery();
        assert_eq!(tdr.health, DeviceHealth::Recovered);
    }
}
