#[cfg(target_arch = "x86")]
pub fn core_cycles() -> u64 {
    use core::arch::x86::{_mm_lfence, _rdtsc};
    unsafe {
        _mm_lfence();
        _rdtsc()
    }
}

#[cfg(target_arch = "x86_64")]
pub fn core_cycles() -> u64 {
    use core::arch::x86_64::{_mm_lfence, _rdtsc};
    unsafe {
        _mm_lfence();
        _rdtsc()
    }
}

#[cfg(target_arch = "aarch64")]
pub fn core_cycles() -> u64 {
    use core::arch::asm;
    let mut x: u64;
    unsafe {
        asm!("dsb sy", "mrs {}, pmccntr_el0", out(reg) x);
    }
    x
}

#[cfg(target_arch = "riscv64")]
pub fn core_cycles() -> u64 {
    use core::arch::asm;
    let mut x: u64;
    unsafe {
        asm!("rdcycle {}", out(reg) x);
    }
    x
}
