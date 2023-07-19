mod util;
use util::core_cycles;

use crrl::ed448::Scalar;

fn bench_sc448_add() {
    let z = core_cycles();
    let mut x = Scalar::from_w64le([ z, z.wrapping_mul(3), z.wrapping_mul(5),
        z.wrapping_mul(7), z.wrapping_mul(9), z.wrapping_mul(11),
        z.wrapping_mul(13) ]);
    let mut y = x + Scalar::ONE;
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
    println!("sc448 add:            {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode()[0]);
}

fn bench_sc448_sub() {
    let z = core_cycles();
    let mut x = Scalar::from_w64le([ z, z.wrapping_mul(3), z.wrapping_mul(5),
        z.wrapping_mul(7), z.wrapping_mul(9), z.wrapping_mul(11),
        z.wrapping_mul(13) ]);
    let mut y = x + Scalar::ONE;
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
    println!("sc448 sub:            {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode()[0]);
}

fn bench_sc448_mul() {
    let z = core_cycles();
    let mut x = Scalar::from_w64le([ z, z.wrapping_mul(3), z.wrapping_mul(5),
        z.wrapping_mul(7), z.wrapping_mul(9), z.wrapping_mul(11),
        z.wrapping_mul(13) ]);
    let mut y = x + Scalar::ONE;
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
    println!("sc448 mul:            {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode()[0]);
}

fn bench_sc448_square() {
    let z = core_cycles();
    let mut x = Scalar::from_w64le([ z, z.wrapping_mul(3), z.wrapping_mul(5),
        z.wrapping_mul(7), z.wrapping_mul(9), z.wrapping_mul(11),
        z.wrapping_mul(13) ]);
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        x = x.xsquare(6000);
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("sc448 square:         {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode()[0]);
}

fn bench_sc448_div() {
    let z = core_cycles();
    let mut x = Scalar::from_w64le([ z, z.wrapping_mul(3), z.wrapping_mul(5),
        z.wrapping_mul(7), z.wrapping_mul(9), z.wrapping_mul(11),
        z.wrapping_mul(13) ]);
    let mut y = x + Scalar::ONE;
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
    println!("sc448 div:            {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode()[0]);
}

fn bench_sc448_sqrt() {
    let z = core_cycles();
    let mut x = Scalar::from_w64le([ z, z.wrapping_mul(3), z.wrapping_mul(5),
        z.wrapping_mul(7), z.wrapping_mul(9), z.wrapping_mul(11),
        z.wrapping_mul(13) ]);
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..6000 {
            let (x2, _) = x.sqrt();
            x += x2 + Scalar::ONE;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("sc448 sqrt:           {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode()[0]);
}

fn bench_sc448_legendre() {
    let z = core_cycles();
    let mut x = Scalar::from_w64le([ z, z.wrapping_mul(3), z.wrapping_mul(5),
        z.wrapping_mul(7), z.wrapping_mul(9), z.wrapping_mul(11),
        z.wrapping_mul(13) ]);
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..6000 {
            let ls = x.legendre();
            x += Scalar::from_w64le([ ls as u64, ls as u64, ls as u64,
                ls as u64, ls as u64, ls as u64, ls as u64 ]);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("sc448 legendre:       {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode()[0]);
}

fn main() {
    bench_sc448_add();
    bench_sc448_sub();
    bench_sc448_mul();
    bench_sc448_square();
    bench_sc448_div();
    bench_sc448_sqrt();
    bench_sc448_legendre();
}
