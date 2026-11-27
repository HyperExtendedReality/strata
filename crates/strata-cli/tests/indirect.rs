//! Indirect dispatch through the D3D12.X command list and queue: the group
//! counts are read from title memory at execution, and counts past the
//! device limit are split, not clamped.

// The D3D12.X queue API takes `Arc<ComputeRunner>` (one thread uses it here).
#![allow(clippy::arc_with_non_send_sync)]

use std::sync::{Arc, Mutex};

use strata_gpu::runner::ComputeRunner;
use strata_gpu::thin_list::PipelineKey;
use strata_gpu::uma::UmaManager;
use strata_hle::d3d12x::{D3D12CommandListX, D3D12CommandQueueX};
use strata_shader::compile_rdna2_to_spirv;
use strata_shader::rdna2::encode::*;

#[test]
fn indirect_dispatch_reads_args_from_title_memory_and_runs_every_group() {
    let Ok(mut runner) = ComputeRunner::new() else {
        eprintln!("no Vulkan device: skipped");
        return;
    };
    runner.max_workgroup_count = [1_000, 1_000, 1_000]; // force a split
    let runner = Arc::new(runner);
    let uma = Arc::new(Mutex::new(UmaManager::new()));

    // Title memory holding the arguments: x = 3,000 groups.
    let args_va = {
        let mut u = uma.lock().unwrap();
        let Ok(a) = u.xmem_alloc_identity(&runner, 0, 4096) else {
            eprintln!("no VK_EXT_external_memory_host: skipped");
            return;
        };
        let p = a.cpu_ptr().unwrap() as *mut u32;
        unsafe { p.copy_from_nonoverlapping([3_000u32, 1, 1].as_ptr(), 3) };
        a.console_va
    };

    // One thread per group writes 1 at its global id.
    let mut prog = vec![v_mov_b32_sgpr(1, 129), v_lshlrev_b32_imm(2, 2, 0)];
    prog.extend_from_slice(&global_store_dword(2, 1, 0, 0));
    prog.push(s_endpgm());
    let spirv = compile_rdna2_to_spirv(&prog, [1, 1, 1]).unwrap().spirv;
    let mut out = runner.alloc_bda_buffer(3_000 * 4).unwrap();
    unsafe { out.as_mut_slice::<u32>(3_000).fill(0) };

    unsafe {
        let queue = D3D12CommandQueueX::new_boxed_with_memory(runner.clone(), uma.clone());
        (*queue).register_pipeline(PipelineKey(7), spirv);
        let list = D3D12CommandListX::new_boxed();
        let vt = (*list).header.vtable;
        let this = list as *mut std::ffi::c_void;
        (vt.set_pipeline_state)(this, 7);
        let sgprs = [out.bda as u32, (out.bda >> 32) as u32];
        (vt.set_compute_root_32bit_constants)(this, 0, 2, sgprs.as_ptr(), 0);
        (vt.dispatch_indirect)(this, args_va);
        (vt.close)(this);
        let lists = [this];
        ((*queue).header.vtable.execute_command_lists)(queue as *mut _, 1, lists.as_ptr());
        assert_eq!(
            (*queue)
                .executed_gpu_dispatches
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        (vt.release)(this);
        ((*queue).header.vtable.release)(queue as *mut _);
    }
    let got = unsafe { out.as_slice::<u32>(3_000).to_vec() };
    runner.free_bda_buffer(out);
    uma.lock().unwrap().destroy_all(&runner);
    assert_eq!(
        got.iter().filter(|&&v| v == 1).count(),
        3_000,
        "groups that ran"
    );
}
