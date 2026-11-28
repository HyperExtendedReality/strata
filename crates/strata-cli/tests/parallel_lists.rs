//! Multi-list `ExecuteCommandLists` batches of compute-only lists are
//! recorded into Vulkan command buffers in parallel and submitted in list
//! order: the result must equal serial execution, and a batch holding any
//! CPU-side command must fall back to the serial replay.

// The D3D12.X queue API takes `Arc<ComputeRunner>` (one thread uses it here).
#![allow(clippy::arc_with_non_send_sync)]

use std::ffi::c_void;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use strata_gpu::runner::ComputeRunner;
use strata_gpu::thin_list::PipelineKey;
use strata_gpu::uma::UmaManager;
use strata_hle::d3d12x::{D3D12CommandListX, D3D12CommandQueueX};
use strata_shader::compile_rdna2_to_spirv;
use strata_shader::rdna2::encode::*;

const N: usize = 64 * 64;

/// Every thread stores s2 at `s[0:1] + 4 * global_id`.
fn store_constant_spirv() -> Vec<u32> {
    let mut prog = vec![v_mov_b32_sgpr(1, 2), v_lshlrev_b32_imm(2, 2, 0)];
    prog.extend_from_slice(&global_store_dword(2, 1, 0, 0));
    prog.push(s_endpgm());
    compile_rdna2_to_spirv(&prog, [64, 1, 1]).unwrap().spirv
}

/// Records one list: a dispatch per value, each overwriting the buffer.
unsafe fn record(bda: u64, values: &[u32], copy: Option<(u64, u64)>) -> *mut c_void {
    unsafe {
        let list = D3D12CommandListX::new_boxed();
        let vt = (*list).header.vtable;
        let this = list as *mut c_void;
        (vt.set_pipeline_state)(this, 3);
        for &v in values {
            let sgprs = [bda as u32, (bda >> 32) as u32, v];
            (vt.set_compute_root_32bit_constants)(this, 0, 3, sgprs.as_ptr(), 0);
            (vt.dispatch)(this, 64, 1, 1);
            (vt.resource_barrier)(this, bda, 0x8, 0x8 | 0x1);
        }
        if let Some((dst, src)) = copy {
            (vt.copy_buffer_region)(this, dst, 0, src, 0, 16);
        }
        (vt.close)(this);
        this
    }
}

/// Runs `lists` as one batch; returns (buffer contents, lists run in parallel).
fn run(lists_of_values: &[&[u32]], with_copy: bool) -> Option<(Vec<u32>, u32)> {
    let Ok(runner) = ComputeRunner::new() else {
        eprintln!("no Vulkan device: skipped");
        return None;
    };
    let runner = Arc::new(runner);
    let uma = Arc::new(Mutex::new(UmaManager::new()));
    let scratch_va = {
        let mut u = uma.lock().unwrap();
        let Ok(a) = u.xmem_alloc_identity(&runner, 0, 4096) else {
            eprintln!("no VK_EXT_external_memory_host: skipped");
            return None;
        };
        a.console_va
    };
    let mut out = runner.alloc_bda_buffer((N * 4) as u64).unwrap();
    unsafe { out.as_mut_slice::<u32>(N).fill(0) };
    let parallel = unsafe {
        let queue = D3D12CommandQueueX::new_boxed_with_memory(runner.clone(), uma.clone());
        (*queue).register_pipeline(PipelineKey(3), store_constant_spirv());
        let lists: Vec<*mut c_void> = lists_of_values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let copy = (with_copy && i == 1).then_some((scratch_va + 64, scratch_va));
                record(out.bda, v, copy)
            })
            .collect();
        ((*queue).header.vtable.execute_command_lists)(
            queue as *mut _,
            lists.len() as u32,
            lists.as_ptr(),
        );
        let dispatches: usize = lists_of_values.iter().map(|v| v.len()).sum();
        assert_eq!(
            (*queue).executed_gpu_dispatches.load(Ordering::Relaxed) as usize,
            dispatches
        );
        assert_eq!(
            (*queue).submitted_lists.load(Ordering::Relaxed) as usize,
            lists.len()
        );
        let parallel = (*queue).parallel_lists.load(Ordering::Relaxed);
        for l in lists {
            ((*(l as *mut D3D12CommandListX)).header.vtable.release)(l);
        }
        ((*queue).header.vtable.release)(queue as *mut _);
        parallel
    };
    let got = unsafe { out.as_slice::<u32>(N).to_vec() };
    runner.free_bda_buffer(out);
    uma.lock().unwrap().destroy_all(&runner);
    Some((got, parallel))
}

#[test]
fn compute_only_batch_records_in_parallel_and_keeps_list_order() {
    // Every dispatch overwrites the whole buffer, so only list order (and
    // dispatch order inside each list) decides the final value.
    let lists: [&[u32]; 6] = [&[1, 2], &[3], &[4, 5, 6], &[7], &[8, 9], &[42]];
    let Some((got, parallel)) = run(&lists, false) else {
        return;
    };
    assert_eq!(parallel, 6, "every list recorded in parallel");
    assert!(
        got.iter().all(|&v| v == 42),
        "last list's last dispatch wins"
    );
}

#[test]
fn batch_with_a_cpu_side_command_falls_back_to_serial() {
    let lists: [&[u32]; 3] = [&[1], &[2], &[7]];
    let Some((got, parallel)) = run(&lists, true) else {
        return;
    };
    assert_eq!(parallel, 0, "a copy keeps the batch on the serial replay");
    assert!(got.iter().all(|&v| v == 7));
}
