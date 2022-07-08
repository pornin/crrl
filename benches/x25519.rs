#![allow(non_snake_case)]

mod util;
use util::core_cycles;

use crrl::x25519::{x25519, x25519_base};

fn bench_x25519() -> (f64, u8) {
    let z = core_cycles();
    let mut b = [0u8; 32];
    b[ 0.. 8].copy_from_slice(&z.to_le_bytes());
    b[ 8..16].copy_from_slice(&z.to_le_bytes());
    b[16..24].copy_from_slice(&z.to_le_bytes());
    b[24..32].copy_from_slice(&z.to_le_bytes());
    let mut tt = [0; 100];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for _ in 0..100 {
            b = x25519(&b, &b);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / 100.0, b[0])
}

fn bench_x25519_base() -> (f64, u8) {
    let z = core_cycles();
    let mut b = [0u8; 32];
    b[ 0.. 8].copy_from_slice(&z.to_le_bytes());
    b[ 8..16].copy_from_slice(&z.to_le_bytes());
    b[16..24].copy_from_slice(&z.to_le_bytes());
    b[24..32].copy_from_slice(&z.to_le_bytes());
    let mut tt = [0; 100];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for _ in 0..100 {
            b = x25519_base(&b);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / 100.0, b[0])
}

fn main() {
    let mut bx = 0u8;

    let (v, x) = bench_x25519();
    bx ^= x;
    println!("X25519 (generic):              {:13.2}", v);
    let (v, x) = bench_x25519_base();
    bx ^= x;
    println!("X25519 (base point):           {:13.2}", v);

    println!("{}", bx);
}
