use std::{error::Error, io};

use strata_gpu::{
    memory::{AddressSpace, BackingKind, MemoryRegion},
    runner::ComputeRunner,
};
use strata_runtime_shaders::CLAMP_INDIRECT_SPV;

pub fn run() -> Result<(), Box<dyn Error>> {
    const CONSOLE_BASE: u64 = 0x10_0000_0000;
    // ClampEntry and VkDrawIndirectCommand both have a 16-byte stride.
    const STRIDE: usize = 16;
    const COMMANDS: [[u32; 4]; 2] = [[91, 8, 7, 11], [12, 1, 4, 2]];
    const LIMITS: [[u32; 2]; 2] = [[64, 2], [32, 2]];
    const EXPECTED: [[u32; 4]; 2] = [[64, 2, 7, 11], [12, 1, 4, 2]];
    const COMMAND_OFFSET: usize = COMMANDS.len() * STRIDE;
    const BUFFER_SIZE: usize = COMMAND_OFFSET + COMMANDS.len() * STRIDE;

    let runner = ComputeRunner::new()?;
    let mut buffer = runner.alloc_bda_buffer(BUFFER_SIZE as u64)?;
    let result = (|| {
        let mut addresses = AddressSpace::new();
        addresses.register(MemoryRegion {
            console_base: CONSOLE_BASE,
            size: buffer.size,
            host_bda: buffer.bda,
            kind: BackingKind::VramMirror,
            last_gpu_write_epoch: 0,
        });
        let translate = |offset| {
            addresses
                .translate(CONSOLE_BASE + offset as u64)
                .ok_or_else(|| {
                    io::Error::other(format!("unmapped synthetic console buffer offset {offset}"))
                })
        };

        // The allocation is host-coherent, and no GPU commands are pending yet.
        let bytes = unsafe { buffer.as_mut_slice::<u8>(BUFFER_SIZE) };
        for (index, (command, limits)) in COMMANDS.iter().zip(LIMITS).enumerate() {
            let entry = index * STRIDE;
            let args = COMMAND_OFFSET + index * STRIDE;
            bytes[entry..entry + 8].copy_from_slice(&translate(args)?.to_le_bytes());
            write_words(&mut bytes[entry + 8..entry + STRIDE], limits);
            write_words(&mut bytes[args..args + STRIDE], *command);
        }

        let mut push = [0; STRIDE];
        push[..8].copy_from_slice(&translate(0)?.to_le_bytes());
        push[8..12].copy_from_slice(&(COMMANDS.len() as u32).to_le_bytes());
        let shader: Vec<_> = CLAMP_INDIRECT_SPV
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect();
        runner.dispatch_spirv(&shader, &push, [1, 1, 1])?;

        // Dispatch waits for completion and makes compute writes visible to the host.
        let bytes = unsafe { buffer.as_slice::<u8>(BUFFER_SIZE) };
        for (index, expected) in EXPECTED.iter().enumerate() {
            let start = COMMAND_OFFSET + index * STRIDE;
            let mut expected_bytes = [0; STRIDE];
            write_words(&mut expected_bytes, *expected);
            if bytes[start..start + STRIDE] != expected_bytes {
                return Err(
                    io::Error::other(format!("command {index} differs from {expected:?}")).into(),
                );
            }
        }
        Ok::<_, Box<dyn Error>>(())
    })();
    runner.free_bda_buffer(buffer);
    result?;

    println!(
        "{}: translated two synthetic command addresses and verified Slang compute readback",
        runner.device_name
    );
    Ok(())
}

fn write_words(bytes: &mut [u8], words: impl IntoIterator<Item = u32>) {
    for (target, word) in bytes.chunks_exact_mut(4).zip(words) {
        target.copy_from_slice(&word.to_le_bytes());
    }
}
