//! Runs the compiler's loop and LDS-atomic kernels on the real GPU and checks
//! every thread's result. Skipped (with a note) when no Vulkan device exists.

use strata_gpu::runner::ComputeRunner;
use strata_shader::compile_rdna2_to_spirv;
use strata_shader::test_kernels::*;

fn run(runner: &ComputeRunner, prog: &[u32]) -> Vec<u32> {
    let compiled = compile_rdna2_to_spirv(prog, [64, 1, 1]).expect("compile");
    let mut out = runner.alloc_bda_buffer(64 * 4).unwrap();
    unsafe { out.as_mut_slice::<u32>(64).fill(0xDEAD_BEEF) };
    let mut push = [0u8; 128];
    push[0..8].copy_from_slice(&out.bda.to_le_bytes());
    runner
        .dispatch_spirv(&compiled.spirv, &push, [1, 1, 1])
        .expect("dispatch");
    let got = unsafe { out.as_slice::<u32>(64).to_vec() };
    runner.free_bda_buffer(out);
    got
}

#[test]
fn loop_kernels_compute_the_right_values_on_the_gpu() {
    let Ok(runner) = ComputeRunner::new() else {
        eprintln!("no Vulkan device: skipped");
        return;
    };
    let tids: Vec<u32> = (0..64).collect();

    let want: Vec<u32> = tids.iter().map(|t| 4 * t).collect();
    assert_eq!(run(&runner, &uniform_do_while_sum()), want, "do-while");
    assert_eq!(run(&runner, &while_loop_sum()), want, "while");

    let want: Vec<u32> = tids.iter().map(|t| 6 * t).collect();
    assert_eq!(run(&runner, &nested_loops()), want, "nested");

    let want: Vec<u32> = tids.iter().map(|t| (t & 3) + 1).collect();
    assert_eq!(
        run(&runner, &divergent_exec_loop()),
        want,
        "per-lane EXEC exit"
    );

    assert_eq!(
        run(&runner, &lds_atomic_count()),
        vec![64; 64],
        "ds_add_u32"
    );
}

#[test]
fn dispatches_beyond_the_group_limit_run_every_group() {
    let Ok(mut runner) = ComputeRunner::new() else {
        eprintln!("no Vulkan device: skipped");
        return;
    };
    // Pretend the device allows 1,000 groups per dimension, then dispatch
    // 2,500: every group must still run, each with its own id.
    runner.max_workgroup_count = [1_000, 1_000, 1_000];
    use strata_shader::rdna2::encode::*;
    let mut prog = vec![v_mov_b32_sgpr(1, 129), v_lshlrev_b32_imm(2, 2, 0)]; // v1 = 1
    prog.extend_from_slice(&global_store_dword(2, 1, 0, 0));
    prog.push(s_endpgm());
    let compiled = compile_rdna2_to_spirv(&prog, [1, 1, 1]).expect("compile");
    let total = 2_500u32;
    let mut out = runner.alloc_bda_buffer(u64::from(total) * 4).unwrap();
    unsafe { out.as_mut_slice::<u32>(total as usize).fill(0) };
    let mut push = [0u8; 128];
    push[0..8].copy_from_slice(&out.bda.to_le_bytes());
    runner
        .dispatch_spirv(&compiled.spirv, &push, [total, 1, 1])
        .expect("dispatch");
    let got = unsafe { out.as_slice::<u32>(total as usize).to_vec() };
    runner.free_bda_buffer(out);
    let ran = got.iter().filter(|&&v| v == 1).count();
    assert_eq!(ran, total as usize, "groups that ran");
}
