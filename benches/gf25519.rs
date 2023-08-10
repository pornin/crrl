#![cfg(feature = "gf25519")]

mod util;
use util::core_cycles;

use crrl::field::GF25519;

fn bench_gf25519_add() {
    let z = core_cycles();
    let mut x = GF25519::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut y = x + GF25519::ONE;
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            x += y;
            y += x;
            x += y;
            y += x;
            x += y;
            y += x;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GF25519 add:          {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode32()[0]);
}

fn bench_gf25519_sub() {
    let z = core_cycles();
    let mut x = GF25519::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut y = x + GF25519::ONE;
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            x -= y;
            y -= x;
            x -= y;
            y -= x;
            x -= y;
            y -= x;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GF25519 sub:          {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode32()[0]);
}

fn bench_gf25519_mul() {
    let z = core_cycles();
    let mut x = GF25519::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut y = x + GF25519::ONE;
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            x *= y;
            y *= x;
            x *= y;
            y *= x;
            x *= y;
            y *= x;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GF25519 mul:          {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode32()[0]);
}

fn bench_gf25519_square() {
    let z = core_cycles();
    let mut x = GF25519::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        x = x.xsquare(6000);
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GF25519 square:       {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode32()[0]);
}

fn bench_gf25519_div() {
    let z = core_cycles();
    let mut x = GF25519::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut y = x + GF25519::ONE;
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            x /= y;
            y /= x;
            x /= y;
            y /= x;
            x /= y;
            y /= x;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GF25519 div:          {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode32()[0]);
}

fn bench_gf25519_sqrt() {
    let z = core_cycles();
    let mut x = GF25519::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..6000 {
            let (x2, _) = x.sqrt();
            x = x2 + GF25519::ONE;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GF25519 sqrt:         {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode32()[0]);
}

fn bench_gf25519_legendre() {
    let z = core_cycles();
    let mut x = GF25519::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..6000 {
            let ls = x.legendre();
            x += GF25519::w64le(ls as u64, ls as u64, ls as u64, ls as u64);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GF25519 legendre:     {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode32()[0]);
}

fn main() {
    bench_gf25519_add();
    bench_gf25519_sub();
    bench_gf25519_mul();
    bench_gf25519_square();
    bench_gf25519_div();
    bench_gf25519_sqrt();
    bench_gf25519_legendre();
}
