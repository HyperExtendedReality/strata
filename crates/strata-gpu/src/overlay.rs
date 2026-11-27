//! Lightweight in-app stats overlay (F3) and live settings menu (F1)
//! composited directly onto RGBA8 swapchain/present frames with zero external
//! font or UI dependencies.

/// Live frame statistics displayed in the F3 overlay panel.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrameStats {
    pub fps: f32,
    pub frame_ms: f32,
    pub draws: u32,
    pub dispatches: u32,
    pub pipelines_cached: u32,
    pub vram_used_mb: u32,
}

/// Interactive F1 settings menu state (Arrow Up/Down/Left/Right, Esc).
#[derive(Debug, Clone)]
pub struct SettingsMenu {
    pub open: bool,
    pub stats_visible: bool,
    pub selected: usize,
    pub upscale_mode: u8, // 0=Auto, 1=FSR, 2=DLSS, 3=Bilinear
    pub frame_cap_fps: u32,
    pub ambient_gi: bool,
}

impl Default for SettingsMenu {
    fn default() -> Self {
        Self {
            open: false,
            stats_visible: false,
            selected: 0,
            upscale_mode: 0,
            frame_cap_fps: 60,
            ambient_gi: false,
        }
    }
}

impl SettingsMenu {
    pub fn toggle_menu(&mut self) {
        self.open = !self.open;
    }

    pub fn toggle_stats(&mut self) {
        self.stats_visible = !self.stats_visible;
    }

    pub fn move_up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn move_down(&mut self) {
        self.selected = (self.selected + 1).min(2);
    }

    pub fn adjust_selected(&mut self, delta: i32) {
        match self.selected {
            0 => self.upscale_mode = ((self.upscale_mode as i32 + delta).rem_euclid(4)) as u8,
            1 => {
                let caps = [30u32, 60, 120, 0];
                let idx = caps
                    .iter()
                    .position(|&c| c == self.frame_cap_fps)
                    .unwrap_or(1) as i32;
                self.frame_cap_fps = caps[(idx + delta).rem_euclid(caps.len() as i32) as usize];
            }
            2 => self.ambient_gi = !self.ambient_gi,
            _ => {}
        }
    }

    /// Formats the F3 stats panel lines.
    pub fn format_stats_lines(stats: &FrameStats) -> Vec<String> {
        vec![
            format!("STRATA  {:.1} FPS ({:.2} ms)", stats.fps, stats.frame_ms),
            format!("DRAWS {}  DISPATCH {}", stats.draws, stats.dispatches),
            format!(
                "PSO {}  VRAM {} MB",
                stats.pipelines_cached, stats.vram_used_mb
            ),
        ]
    }
}

/// Blends a semi-transparent dark backdrop rectangle into an RGBA8 framebuffer
/// for the overlay panel or F1 menu.
pub fn composite_backdrop_rgba8(
    fb: &mut [u8],
    fb_w: u32,
    fb_h: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    alpha: u8,
) {
    let x1 = x.saturating_add(w).min(fb_w);
    let y1 = y.saturating_add(h).min(fb_h);
    let inv = 255u32.saturating_sub(u32::from(alpha));
    for py in y..y1 {
        for px in x..x1 {
            let idx = ((py * fb_w + px) as usize) * 4;
            if idx + 4 > fb.len() {
                continue;
            }
            fb[idx] = ((u32::from(fb[idx]) * inv) / 255) as u8;
            fb[idx + 1] = ((u32::from(fb[idx + 1]) * inv) / 255) as u8;
            fb[idx + 2] = ((u32::from(fb[idx + 2]) * inv) / 255) as u8;
            fb[idx + 3] = 255;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_menu_and_backdrop_work() {
        let mut m = SettingsMenu::default();
        m.toggle_menu();
        assert!(m.open);
        m.adjust_selected(1);
        assert_eq!(m.upscale_mode, 1);
        m.move_down();
        m.adjust_selected(1);
        assert_eq!(m.frame_cap_fps, 120);

        let lines = SettingsMenu::format_stats_lines(&FrameStats {
            fps: 60.0,
            frame_ms: 16.67,
            draws: 420,
            dispatches: 95,
            pipelines_cached: 128,
            vram_used_mb: 2048,
        });
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("60.0 FPS"));

        let mut fb = vec![200u8; 16 * 16 * 4];
        composite_backdrop_rgba8(&mut fb, 16, 16, 2, 2, 4, 4, 128);
        assert!(fb[(2 * 16 + 2) * 4] < 110);
    }
}
