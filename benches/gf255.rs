mod util;
use util::core_cycles;

use crrl::field::GF255e;

fn bench_gf255e_add() {
    let z = core_cycles();
    let mut x = GF255e::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut y = x + GF255e::ONE;
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
    println!("GF255e add:           {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode32()[0]);
}

fn bench_gf255e_sub() {
    let z = core_cycles();
    let mut x = GF255e::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut y = x + GF255e::ONE;
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
    println!("GF255e sub:           {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode32()[0]);
}

fn bench_gf255e_mul() {
    let z = core_cycles();
    let mut x = GF255e::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut y = x + GF255e::ONE;
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
    println!("GF255e mul:           {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode32()[0]);
}

fn bench_gf255e_square() {
    let z = core_cycles();
    let mut x = GF255e::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        x = x.xsquare(6000);
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GF255e square:        {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode32()[0]);
}

fn bench_gf255e_div() {
    let z = core_cycles();
    let mut x = GF255e::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut y = x + GF255e::ONE;
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
    println!("GF255e div:           {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode32()[0]);
}

fn bench_gf255e_sqrt() {
    let z = core_cycles();
    let mut x = GF255e::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..6000 {
            let (x2, _) = x.sqrt();
            x = x2 + GF255e::ONE;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GF255e sqrt:          {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode32()[0]);
}

fn bench_gf255e_legendre() {
    let z = core_cycles();
    let mut x = GF255e::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..6000 {
            let ls = x.legendre();
            x += GF255e::w64le(ls as u64, ls as u64, ls as u64, ls as u64);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GF255e legendre:      {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode32()[0]);
}

fn main() {
    bench_gf255e_add();
    bench_gf255e_sub();
    bench_gf255e_mul();
    bench_gf255e_square();
    bench_gf255e_div();
    bench_gf255e_sqrt();
    bench_gf255e_legendre();
}
