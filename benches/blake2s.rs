#![allow(non_snake_case)]
#![cfg(feature = "blake2s")]

mod util;
use util::core_cycles;

use crrl::blake2s::Blake2s256;

fn bench_blake2s_short() -> (f64, u8) {
    let z = core_cycles();
    let mut seed = [0u8; 32];
    seed[ 0.. 8].copy_from_slice(&z.to_le_bytes());
    seed[ 8..16].copy_from_slice(&z.to_le_bytes());
    seed[16..24].copy_from_slice(&z.to_le_bytes());
    seed[24..32].copy_from_slice(&z.to_le_bytes());
    let mut tt = [0; 100];
    let mut sh = Blake2s256::new();
    for i in 0..(tt.len() + 1000) {
        let begin = core_cycles();
        for _ in 0..100 {
            sh.update(&seed);
            sh.finalize_reset_write(&mut seed);
        }
        let end = core_cycles();
        if i >= 1000 {
            tt[i - 1000] = end.wrapping_sub(begin);
        }
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / 100.0, seed[0])
}

fn bench_blake2s_4096() -> (f64, u8) {
    let z = core_cycles();
    let mut sh = Blake2s256::new();
    let mut buf = [0u8; 4096];
    for i in 0..(buf.len() >> 5) {
        sh.update(&z.to_le_bytes());
        sh.update(&(i as u64).to_le_bytes());
        sh.finalize_reset_write(&mut buf[(i << 5)..]);
    }
    let mut tt = [0; 100];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for _ in 0..(buf.len() >> 5) {
            sh.update(&buf);
            sh.finalize_reset_write(&mut buf[(i << 5)..]);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / ((buf.len() >> 5) as f64), buf[0])
}

fn main() {
    let mut bx = 0u8;

    let (v, x) = bench_blake2s_short();
    bx ^= x;
    println!("BLAKE2s (short):               {:13.2}", v);
    let (v, x) = bench_blake2s_4096();
    bx ^= x;
    println!("BLAKE2s (4096 bytes):          {:13.2}", v);

    println!("{}", bx);
}
