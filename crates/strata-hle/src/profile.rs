//! Console hardware profiles (Series S / Series X) for resolution, target frame
//! rate, title memory budget, GPU compute-unit reporting, and `XSystem` queries.

/// Console hardware model reported to titles via `XSystemGetDeviceType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleModel {
    SeriesS,
    SeriesX,
}

impl ConsoleModel {
    /// Numeric `XSystemDeviceType` enum value.
    pub const fn xsystem_device_type(self) -> u32 {
        match self {
            Self::SeriesS => 5, // XboxScarlettLockhart
            Self::SeriesX => 6, // XboxScarlettAnaconda
        }
    }
}

/// Active hardware & presentation profile for running a console title.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConsoleProfile {
    pub model: ConsoleModel,
    pub output_width: u32,
    pub output_height: u32,
    pub target_fps: u32,
    pub title_memory_bytes: u64,
    pub gpu_compute_units: u32,
    pub cpu_logical_cores: u32,
}

impl ConsoleProfile {
    pub const fn series_x() -> Self {
        Self {
            model: ConsoleModel::SeriesX,
            output_width: 3840,
            output_height: 2160,
            target_fps: 60,
            title_memory_bytes: 13_500 * 1024 * 1024, // 13.5 GiB
            gpu_compute_units: 52,
            cpu_logical_cores: 8,
        }
    }

    pub const fn series_s() -> Self {
        Self {
            model: ConsoleModel::SeriesS,
            output_width: 2560,
            output_height: 1440,
            target_fps: 60,
            title_memory_bytes: 8_000 * 1024 * 1024, // 8 GiB
            gpu_compute_units: 20,
            cpu_logical_cores: 8,
        }
    }

    pub const fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.output_width = width;
        self.output_height = height;
        self
    }

    pub const fn with_target_fps(mut self, fps: u32) -> Self {
        self.target_fps = fps;
        self
    }

    /// Parse CLI profile name (`series-x`, `series-s`, `xsx`, `xss`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "series-x" | "seriesx" | "xsx" | "anaconda" => Some(Self::series_x()),
            "series-s" | "seriess" | "xss" | "lockhart" => Some(Self::series_s()),
            _ => None,
        }
    }
}

impl Default for ConsoleProfile {
    fn default() -> Self {
        Self::series_x()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_s_and_x_profiles_have_distinct_budgets() {
        let sx = ConsoleProfile::from_name("series-x").unwrap();
        let ss = ConsoleProfile::from_name("series-s").unwrap();
        assert_eq!(sx.model.xsystem_device_type(), 6);
        assert_eq!(ss.model.xsystem_device_type(), 5);
        assert!(sx.title_memory_bytes > ss.title_memory_bytes);
        assert_eq!(
            ss.with_resolution(1920, 1080)
                .with_target_fps(120)
                .target_fps,
            120
        );
    }
}
