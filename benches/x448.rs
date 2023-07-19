#![allow(non_snake_case)]

mod util;
use util::core_cycles;

use crrl::x448::{x448, x448_base};

fn bench_x448() -> (f64, u8) {
    let z = core_cycles();
    let mut b = [0u8; 56];
    b[ 0.. 8].copy_from_slice(&z.to_le_bytes());
    b[ 8..16].copy_from_slice(&z.to_le_bytes());
    b[16..24].copy_from_slice(&z.to_le_bytes());
    b[24..32].copy_from_slice(&z.to_le_bytes());
    b[32..40].copy_from_slice(&z.to_le_bytes());
    b[40..48].copy_from_slice(&z.to_le_bytes());
    b[48..56].copy_from_slice(&z.to_le_bytes());
    let mut tt = [0; 100];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for _ in 0..100 {
            b = x448(&b, &b);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / 100.0, b[0])
}

fn bench_x448_base() -> (f64, u8) {
    let z = core_cycles();
    let mut b = [0u8; 56];
    b[ 0.. 8].copy_from_slice(&z.to_le_bytes());
    b[ 8..16].copy_from_slice(&z.to_le_bytes());
    b[16..24].copy_from_slice(&z.to_le_bytes());
    b[24..32].copy_from_slice(&z.to_le_bytes());
    b[32..40].copy_from_slice(&z.to_le_bytes());
    b[40..48].copy_from_slice(&z.to_le_bytes());
    b[48..56].copy_from_slice(&z.to_le_bytes());
    let mut tt = [0; 100];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for _ in 0..100 {
            b = x448_base(&b);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / 100.0, b[0])
}

fn main() {
    let mut bx = 0u8;

    let (v, x) = bench_x448();
    bx ^= x;
    println!("X448 (generic):                {:13.2}", v);
    let (v, x) = bench_x448_base();
    bx ^= x;
    println!("X448 (base point):             {:13.2}", v);

    println!("{}", bx);
}
