#![allow(non_snake_case)]
#![cfg(feature = "ristretto255")]

mod util;
use util::core_cycles;

use crrl::ristretto255::Point;

fn bench_decode() -> (f64, u8) {
    let z = core_cycles();
    let mut buf = [0u8; 32];
    buf[ 0.. 8].copy_from_slice(&z.to_le_bytes());
    buf[ 8..16].copy_from_slice(&z.to_le_bytes());
    buf[16..24].copy_from_slice(&z.to_le_bytes());
    buf[24..32].copy_from_slice(&z.to_le_bytes());
    let mut tt = [0; 10];
    let mut P = Point::NEUTRAL;
    let Q = Point::BASE * z;
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..100 {
            let r = P.set_decode(&buf);
            buf[0] = buf[0].wrapping_add(1);
            buf[1] = buf[1].wrapping_add(r as u8);
            buf[2] = buf[2].wrapping_add(P.equals(Q) as u8);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[4] as f64) / 100.0, buf[0])
}

fn bench_encode() -> (f64, u8) {
    let z = core_cycles();
    let mut P = Point::BASE * z;
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..100 {
            let x = P.encode()[0];
            if x & 1 == 0 {
                P = -P;
            }
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[4] as f64) / 100.0, P.encode()[0])
}

fn main() {
    let mut bx = 0u8;

    let (v, x) = bench_decode();
    bx ^= x;
    println!("Ristretto255 decode:           {:13.2}", v);
    let (v, x) = bench_encode();
    bx ^= x;
    println!("Ristretto255 encode:           {:13.2}", v);

    println!("{}", bx);
}
