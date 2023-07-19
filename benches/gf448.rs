mod util;
use util::core_cycles;

use crrl::field::GF448;

fn bench_gf448_add() {
    let z = core_cycles();
    let mut x = GF448::from_w64le([ z, z.wrapping_mul(3), z.wrapping_mul(5),
        z.wrapping_mul(7), z.wrapping_mul(9), z.wrapping_mul(11),
        z.wrapping_mul(13) ]);
    let mut y = x + GF448::ONE;
    let mut tt = [0; 10];
    for i in 0..30 {
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
        if i >= 20 {
            tt[i - 20] = end.wrapping_sub(begin);
        }
    }
    tt.sort();
    println!("GF448 add:            {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode()[0]);
}

fn bench_gf448_sub() {
    let z = core_cycles();
    let mut x = GF448::from_w64le([ z, z.wrapping_mul(3), z.wrapping_mul(5),
        z.wrapping_mul(7), z.wrapping_mul(9), z.wrapping_mul(11),
        z.wrapping_mul(13) ]);
    let mut y = x + GF448::ONE;
    let mut tt = [0; 10];
    for i in 0..30 {
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
        if i >= 20 {
            tt[i - 20] = end.wrapping_sub(begin);
        }
    }
    tt.sort();
    println!("GF448 sub:            {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode()[0]);
}

fn bench_gf448_mul() {
    let z = core_cycles();
    let mut x = GF448::from_w64le([ z, z.wrapping_mul(3), z.wrapping_mul(5),
        z.wrapping_mul(7), z.wrapping_mul(9), z.wrapping_mul(11),
        z.wrapping_mul(13) ]);
    let mut y = x + GF448::ONE;
    let mut tt = [0; 10];
    for i in 0..30 {
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
        if i >= 20 {
            tt[i - 20] = end.wrapping_sub(begin);
        }
    }
    tt.sort();
    println!("GF448 mul:            {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode()[0]);
}

fn bench_gf448_square() {
    let z = core_cycles();
    let mut x = GF448::from_w64le([ z, z.wrapping_mul(3), z.wrapping_mul(5),
        z.wrapping_mul(7), z.wrapping_mul(9), z.wrapping_mul(11),
        z.wrapping_mul(13) ]);
    let mut tt = [0; 10];
    for i in 0..30 {
        let begin = core_cycles();
        x = x.xsquare(6000);
        let end = core_cycles();
        if i >= 20 {
            tt[i - 20] = end.wrapping_sub(begin);
        }
    }
    tt.sort();
    println!("GF448 square:         {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode()[0]);
}

fn bench_gf448_div() {
    let z = core_cycles();
    let mut x = GF448::from_w64le([ z, z.wrapping_mul(3), z.wrapping_mul(5),
        z.wrapping_mul(7), z.wrapping_mul(9), z.wrapping_mul(11),
        z.wrapping_mul(13) ]);
    let mut y = x + GF448::ONE;
    let mut tt = [0; 10];
    for i in 0..30 {
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
        if i >= 20 {
            tt[i - 20] = end.wrapping_sub(begin);
        }
    }
    tt.sort();
    println!("GF448 div:            {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode()[0]);
}

fn bench_gf448_sqrt() {
    let z = core_cycles();
    let mut x = GF448::from_w64le([ z, z.wrapping_mul(3), z.wrapping_mul(5),
        z.wrapping_mul(7), z.wrapping_mul(9), z.wrapping_mul(11),
        z.wrapping_mul(13) ]);
    let mut tt = [0; 10];
    for i in 0..30 {
        let begin = core_cycles();
        for _ in 0..6000 {
            let (x2, _) = x.sqrt();
            x += x2 + GF448::ONE;
        }
        let end = core_cycles();
        if i >= 20 {
            tt[i - 20] = end.wrapping_sub(begin);
        }
    }
    tt.sort();
    println!("GF448 sqrt:           {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode()[0]);
}

fn bench_gf448_legendre() {
    let z = core_cycles();
    let mut x = GF448::from_w64le([ z, z.wrapping_mul(3), z.wrapping_mul(5),
        z.wrapping_mul(7), z.wrapping_mul(9), z.wrapping_mul(11),
        z.wrapping_mul(13) ]);
    let mut tt = [0; 10];
    for i in 0..30 {
        let begin = core_cycles();
        for _ in 0..6000 {
            let ls = x.legendre();
            let ls2 = ls as u64;
            x += GF448::w64le([ ls2, ls2, ls2, ls2, ls2, ls2, ls2 ]);
        }
        let end = core_cycles();
        if i >= 20 {
            tt[i - 20] = end.wrapping_sub(begin);
        }
    }
    tt.sort();
    println!("GF448 legendre:       {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.encode()[0]);
}

fn main() {
    bench_gf448_add();
    bench_gf448_sub();
    bench_gf448_mul();
    bench_gf448_square();
    bench_gf448_div();
    bench_gf448_sqrt();
    bench_gf448_legendre();

    /*
    bench_fiat_add();
    bench_fiat_sub();
    bench_fiat_mul();
    bench_fiat_square();
    */
}

/*
extern crate fiat_crypto;
use fiat_crypto::p448_solinas_64::*;

fn bench_fiat_add() {
    let z = core_cycles();
    let mut x: fiat_p448_tight_field_element = [
        z & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(3) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(5) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(7) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(9) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(11) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(13) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(15) & 0x00FFFFFFFFFFFFFF,
    ];
    let mut y = x;
    y[0] += 1;
    let mut tt = [0; 10];
    for i in 0..30 {
        let mut z: fiat_p448_loose_field_element = [0u64; 8];
        let begin = core_cycles();
        for _ in 0..1000 {
            fiat_p448_add(&mut z, &x, &y); fiat_p448_carry(&mut x, &z);
            fiat_p448_add(&mut z, &y, &x); fiat_p448_carry(&mut y, &z);
            fiat_p448_add(&mut z, &x, &y); fiat_p448_carry(&mut x, &z);
            fiat_p448_add(&mut z, &y, &x); fiat_p448_carry(&mut y, &z);
            fiat_p448_add(&mut z, &x, &y); fiat_p448_carry(&mut x, &z);
            fiat_p448_add(&mut z, &y, &x); fiat_p448_carry(&mut y, &z);
        }
        let end = core_cycles();
        if i >= 20 {
            tt[i - 20] = end.wrapping_sub(begin);
        }
    }
    tt.sort();
    println!("fc448 add:            {:11.2}  ({})", (tt[4] as f64) / 6000.0, x[0] as u8);
}

fn bench_fiat_sub() {
    let z = core_cycles();
    let mut x: fiat_p448_tight_field_element = [
        z & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(3) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(5) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(7) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(9) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(11) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(13) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(15) & 0x00FFFFFFFFFFFFFF,
    ];
    let mut y = x;
    y[0] += 1;
    let mut tt = [0; 10];
    for i in 0..30 {
        let mut z: fiat_p448_loose_field_element = [0u64; 8];
        let begin = core_cycles();
        for _ in 0..1000 {
            fiat_p448_sub(&mut z, &x, &y); fiat_p448_carry(&mut x, &z);
            fiat_p448_sub(&mut z, &y, &x); fiat_p448_carry(&mut y, &z);
            fiat_p448_sub(&mut z, &x, &y); fiat_p448_carry(&mut x, &z);
            fiat_p448_sub(&mut z, &y, &x); fiat_p448_carry(&mut y, &z);
            fiat_p448_sub(&mut z, &x, &y); fiat_p448_carry(&mut x, &z);
            fiat_p448_sub(&mut z, &y, &x); fiat_p448_carry(&mut y, &z);
        }
        let end = core_cycles();
        if i >= 20 {
            tt[i - 20] = end.wrapping_sub(begin);
        }
    }
    tt.sort();
    println!("fc448 sub:            {:11.2}  ({})", (tt[4] as f64) / 6000.0, x[0] as u8);
}

fn bench_fiat_mul() {
    let z = core_cycles();
    let mut x: fiat_p448_loose_field_element = [
        z & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(3) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(5) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(7) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(9) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(11) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(13) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(15) & 0x00FFFFFFFFFFFFFF,
    ];
    let mut y = x;
    y[0] += 1;
    let mut tt = [0; 10];
    for i in 0..30 {
        let mut z: fiat_p448_tight_field_element = [0u64; 8];
        let begin = core_cycles();
        for _ in 0..1000 {
            fiat_p448_carry_mul(&mut z, &x, &y); fiat_p448_relax(&mut x, &z);
            fiat_p448_carry_mul(&mut z, &y, &x); fiat_p448_relax(&mut y, &z);
            fiat_p448_carry_mul(&mut z, &x, &y); fiat_p448_relax(&mut x, &z);
            fiat_p448_carry_mul(&mut z, &y, &x); fiat_p448_relax(&mut y, &z);
            fiat_p448_carry_mul(&mut z, &x, &y); fiat_p448_relax(&mut x, &z);
            fiat_p448_carry_mul(&mut z, &y, &x); fiat_p448_relax(&mut y, &z);
        }
        let end = core_cycles();
        if i >= 20 {
            tt[i - 20] = end.wrapping_sub(begin);
        }
    }
    tt.sort();
    println!("fc448 mul:            {:11.2}  ({})", (tt[4] as f64) / 6000.0, x[0] as u8);
}

fn bench_fiat_square() {
    let z = core_cycles();
    let mut x: fiat_p448_loose_field_element = [
        z & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(3) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(5) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(7) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(9) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(11) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(13) & 0x00FFFFFFFFFFFFFF,
        z.wrapping_mul(15) & 0x00FFFFFFFFFFFFFF,
    ];
    let mut tt = [0; 10];
    for i in 0..30 {
        let mut z: fiat_p448_tight_field_element = [0u64; 8];
        let begin = core_cycles();
        for _ in 0..1000 {
            fiat_p448_carry_square(&mut z, &x); fiat_p448_relax(&mut x, &z);
            fiat_p448_carry_square(&mut z, &x); fiat_p448_relax(&mut x, &z);
            fiat_p448_carry_square(&mut z, &x); fiat_p448_relax(&mut x, &z);
            fiat_p448_carry_square(&mut z, &x); fiat_p448_relax(&mut x, &z);
            fiat_p448_carry_square(&mut z, &x); fiat_p448_relax(&mut x, &z);
            fiat_p448_carry_square(&mut z, &x); fiat_p448_relax(&mut x, &z);
        }
        let end = core_cycles();
        if i >= 20 {
            tt[i - 20] = end.wrapping_sub(begin);
        }
    }
    tt.sort();
    println!("fc448 square:         {:11.2}  ({})", (tt[4] as f64) / 6000.0, x[0] as u8);
}
*/
