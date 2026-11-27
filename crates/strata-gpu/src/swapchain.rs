//! Cross-platform No-API Vulkan swapchain presenter (`VK_KHR_win32_surface`,
//! `VK_KHR_xlib_surface`, `VK_KHR_xcb_surface`, `VK_KHR_wayland_surface`) +
//! `VK_KHR_swapchain` + `VK_KHR_dynamic_rendering`.
//!
//! Presents console backbuffers directly to a desktop window (Windows HWND or
//! Linux X11/Wayland surface) using Strata's Slang vertex-less
//! `present_blit.slang` shader (0 `VkRenderPass`, 0 `VkFramebuffer`, 0 vertex buffers).

use std::ffi::{CStr, c_void};

use ash::vk;

use crate::context::GpuError;
use crate::present::FramePacingClock;

/// Platform-native window surface handle for Vulkan WSI creation.
#[derive(Debug, Clone, Copy)]
pub enum SurfaceHandle {
    /// Windows Win32 (`VK_KHR_win32_surface`).
    Win32 {
        hinstance: *const c_void,
        hwnd: *const c_void,
    },
    /// Linux X11 Xlib (`VK_KHR_xlib_surface`).
    Xlib { dpy: *mut c_void, window: u64 },
    /// Linux X11 XCB (`VK_KHR_xcb_surface`).
    Xcb {
        connection: *mut c_void,
        window: u32,
    },
    /// Linux Wayland (`VK_KHR_wayland_surface`).
    Wayland {
        display: *mut c_void,
        surface: *mut c_void,
    },
}

impl SurfaceHandle {
    /// Instance extensions required to create a `VkSurfaceKHR` for this handle.
    pub fn required_instance_extensions(&self) -> [&'static CStr; 2] {
        let wsi = match self {
            Self::Win32 { .. } => ash::khr::win32_surface::NAME,
            Self::Xlib { .. } => ash::khr::xlib_surface::NAME,
            Self::Xcb { .. } => ash::khr::xcb_surface::NAME,
            Self::Wayland { .. } => ash::khr::wayland_surface::NAME,
        };
        [ash::khr::surface::NAME, wsi]
    }
}

pub struct SwapchainPresenter {
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub clock: FramePacingClock,
}

impl SwapchainPresenter {
    /// Create a Vulkan swapchain attached to an existing Win32 `(hinstance, hwnd)`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn create_win32(
        entry: &ash::Entry,
        instance: &ash::Instance,
        pdev: vk::PhysicalDevice,
        device: &ash::Device,
        hinstance: *const c_void,
        hwnd: *const c_void,
        width: u32,
        height: u32,
        target_fps: u32,
    ) -> Result<Self, GpuError> {
        unsafe {
            Self::create_for_surface(
                entry,
                instance,
                pdev,
                device,
                SurfaceHandle::Win32 { hinstance, hwnd },
                width,
                height,
                target_fps,
            )
        }
    }

    /// Create a Vulkan swapchain attached to a Win32, Xlib, XCB, or Wayland surface.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn create_for_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        pdev: vk::PhysicalDevice,
        device: &ash::Device,
        handle: SurfaceHandle,
        width: u32,
        height: u32,
        target_fps: u32,
    ) -> Result<Self, GpuError> {
        let surface_loader = ash::khr::surface::Instance::new(entry, instance);
        let swapchain_loader = ash::khr::swapchain::Device::new(instance, device);

        let surface = match handle {
            SurfaceHandle::Win32 { hinstance, hwnd } => {
                let win32_surface = ash::khr::win32_surface::Instance::new(entry, instance);
                let surface_ci = vk::Win32SurfaceCreateInfoKHR::default()
                    .hinstance(hinstance as vk::HINSTANCE)
                    .hwnd(hwnd as vk::HWND);
                unsafe { win32_surface.create_win32_surface(&surface_ci, None)? }
            }
            SurfaceHandle::Xlib { dpy, window } => {
                let xlib_surface = ash::khr::xlib_surface::Instance::new(entry, instance);
                let surface_ci = vk::XlibSurfaceCreateInfoKHR::default()
                    .dpy(dpy.cast())
                    .window(window as vk::Window);
                unsafe { xlib_surface.create_xlib_surface(&surface_ci, None)? }
            }
            SurfaceHandle::Xcb { connection, window } => {
                let xcb_surface = ash::khr::xcb_surface::Instance::new(entry, instance);
                let surface_ci = vk::XcbSurfaceCreateInfoKHR::default()
                    .connection(connection)
                    .window(window);
                unsafe { xcb_surface.create_xcb_surface(&surface_ci, None)? }
            }
            SurfaceHandle::Wayland { display, surface } => {
                let wayland_surface = ash::khr::wayland_surface::Instance::new(entry, instance);
                let surface_ci = vk::WaylandSurfaceCreateInfoKHR::default()
                    .display(display)
                    .surface(surface);
                unsafe { wayland_surface.create_wayland_surface(&surface_ci, None)? }
            }
        };

        let caps =
            unsafe { surface_loader.get_physical_device_surface_capabilities(pdev, surface)? };
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D { width, height }
        };
        let image_count = (caps.min_image_count + 1).min(if caps.max_image_count > 0 {
            caps.max_image_count
        } else {
            3
        });

        let format = vk::Format::B8G8R8A8_UNORM;
        let sc_ci = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);

        let swapchain = unsafe { swapchain_loader.create_swapchain(&sc_ci, None)? };
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let mut views = Vec::with_capacity(images.len());
        for &img in &images {
            let v = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(img)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )?
            };
            views.push(v);
        }

        Ok(Self {
            surface_loader,
            swapchain_loader,
            surface,
            swapchain,
            format,
            extent,
            images,
            views,
            clock: FramePacingClock::new(target_fps),
        })
    }

    /// Acquire the next presentable swapchain image index.
    pub unsafe fn acquire_next_image(
        &self,
        timeout_ns: u64,
        semaphore: vk::Semaphore,
        fence: vk::Fence,
    ) -> Result<(u32, bool), GpuError> {
        let res = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                timeout_ns,
                semaphore,
                fence,
            )?
        };
        Ok(res)
    }

    /// Present `image_index` on `queue`, advancing the `FramePacingClock`.
    pub unsafe fn present_image(
        &mut self,
        queue: vk::Queue,
        image_index: u32,
        wait_semaphores: &[vk::Semaphore],
    ) -> Result<bool, GpuError> {
        let swapchains = [self.swapchain];
        let indices = [image_index];
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&indices);
        let suboptimal = unsafe { self.swapchain_loader.queue_present(queue, &info)? };
        self.clock.record_present_x();
        Ok(suboptimal)
    }

    pub unsafe fn destroy(self, device: &ash::Device) {
        unsafe {
            for v in self.views {
                device.destroy_image_view(v, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_handle_selects_platform_wsi_extensions() {
        let win32 = SurfaceHandle::Win32 {
            hinstance: std::ptr::null(),
            hwnd: std::ptr::null(),
        };
        assert_eq!(
            win32.required_instance_extensions(),
            [ash::khr::surface::NAME, ash::khr::win32_surface::NAME]
        );

        let xlib = SurfaceHandle::Xlib {
            dpy: std::ptr::null_mut(),
            window: 1,
        };
        assert_eq!(
            xlib.required_instance_extensions(),
            [ash::khr::surface::NAME, ash::khr::xlib_surface::NAME]
        );

        let xcb = SurfaceHandle::Xcb {
            connection: std::ptr::null_mut(),
            window: 2,
        };
        assert_eq!(
            xcb.required_instance_extensions(),
            [ash::khr::surface::NAME, ash::khr::xcb_surface::NAME]
        );

        let wayland = SurfaceHandle::Wayland {
            display: std::ptr::null_mut(),
            surface: std::ptr::null_mut(),
        };
        assert_eq!(
            wayland.required_instance_extensions(),
            [ash::khr::surface::NAME, ash::khr::wayland_surface::NAME]
        );
    }
}
