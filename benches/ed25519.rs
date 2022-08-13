#![allow(non_snake_case)]

mod util;
use util::core_cycles;

use crrl::ed25519::{PrivateKey, Point, Scalar};
use sha2::{Sha256, Digest};

fn bench_mulgen() -> (f64, u8) {
    let z = core_cycles();
    let mut seed = [0u8; 32];
    seed[ 0.. 8].copy_from_slice(&z.to_le_bytes());
    seed[ 8..16].copy_from_slice(&z.to_le_bytes());
    seed[16..24].copy_from_slice(&z.to_le_bytes());
    seed[24..32].copy_from_slice(&z.to_le_bytes());
    let mut s = Scalar::decode_reduce(&seed);
    let mut tt = [0; 100];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for _ in 0..100 {
            let P = Point::mulgen(&s);
            if P.isneutral() != 0 {
                s += Scalar::ZERO;
            } else {
                s += Scalar::ONE;
            }
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / 100.0, s.encode32()[0])
}

fn bench_mul() -> (f64, u8) {
    let z = core_cycles();
    let mut seed = [0u8; 32];
    seed[ 0.. 8].copy_from_slice(&z.to_le_bytes());
    seed[ 8..16].copy_from_slice(&z.to_le_bytes());
    seed[16..24].copy_from_slice(&z.to_le_bytes());
    seed[24..32].copy_from_slice(&z.to_le_bytes());
    let mut s = Scalar::decode_reduce(&seed);
    let mut P = Point::mulgen(&s);
    let mut tt = [0; 100];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for _ in 0..100 {
            P *= s;
            if P.isneutral() != 0 {
                s += Scalar::ZERO;
            } else {
                s += Scalar::ONE;
            }
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / 100.0, s.encode32()[0])
}

fn bench_mul_add_mulgen() -> (f64, u8) {
    let z = core_cycles();
    let mut uu = [Scalar::ZERO; 128];
    let mut vv = [Scalar::ZERO; 128];
    let mut sh = Sha256::new();
    for i in 0..128 {
        sh.update(z.to_le_bytes());
        sh.update(((2 * i + 0) as u64).to_le_bytes());
        let b1 = sh.finalize_reset();
        sh.update(z.to_le_bytes());
        sh.update(((2 * i + 1) as u64).to_le_bytes());
        let b2 = sh.finalize_reset();
        uu[i] = Scalar::decode_reduce(&b1);
        vv[i] = Scalar::decode_reduce(&b2);
    }
    let mut tt = [0; 100];
    let mut P = Point::mulgen(&uu[127]);
    for i in 0..tt.len() {
        let begin = core_cycles();
        for j in 0..128 {
            let ku = (i + j) & 127;
            let kv = i.wrapping_sub(j) & 127;
            let Q = P.mul_add_mulgen_vartime(&uu[ku], &vv[kv]);
            P += Q;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / 128.0, P.encode()[0])
}

fn bench_skey_load() -> (f64, u8) {
    let z = core_cycles();
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&z.to_le_bytes());
    let mut tt = [0; 100];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for _ in 0..100 {
            let skey = PrivateKey::from_seed(&seed);
            seed[..].copy_from_slice(&skey.public_key.encode());
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / 100.0, seed[0])
}

fn bench_skey_sign() -> (f64, u8) {
    let z = core_cycles();
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&z.to_le_bytes());
    let skey = PrivateKey::from_seed(&seed);
    let mut tt = [0; 100];
    let mut msg = [0u8; 32];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for _ in 0..100 {
            let sig = skey.sign_raw(&msg);
            msg[..].copy_from_slice(&sig[0..32]);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / 100.0, msg[0])
}

fn bench_pkey_verify() -> (f64, u8) {
    let z = core_cycles();
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&z.to_le_bytes());
    let skey = PrivateKey::from_seed(&seed);
    let pkey = skey.public_key;
    let mut sigs = [[0u8; 64]; 128];
    for i in 0..128 {
        let msg = [i as u8; 32];
        let sig = skey.sign_raw(&msg);
        sigs[i][..].copy_from_slice(&sig);
    }
    let mut tt = [0; 100];
    let mut msg = [0u8; 32];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for j in 0..128 {
            let ff = pkey.verify_raw(&sigs[j], &msg);
            sigs[j][40] ^= 1u8.wrapping_add(ff as u8);
            msg[3] ^= 3u8.wrapping_sub(ff as u8);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / 128.0, msg[0])
}

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

fn bench_pkey_verify_trunc(rm: usize) -> (f64, f64, u8) {
    let z = core_cycles();
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&z.to_le_bytes());
    let skey = PrivateKey::from_seed(&seed);
    let pkey = skey.public_key;
    let mut sigs = [[0u8; 64]; 256];
    for i in 0..256 {
        let msg = [i as u8; 32];
        let sig = skey.sign_raw(&msg);
        sigs[i][..].copy_from_slice(&sig);
    }
    let mut x = 0;

    // Phase 1: all signatures are correct.
    let mut tt = [0; 2048];
    for i in 0..tt.len() {
        let msg = [i as u8; 32];
        let begin = core_cycles();
        x ^= (pkey.verify_trunc_raw(&sigs[i % 256], rm, &msg).is_some()) as u8;
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    // Remove 10% slowest and 10% fastest, make an average of the rest.
    let n10 = tt.len() / 10;
    let n80 = tt.len() - 2 * n10;
    let mut s = 0u64;
    for i in n10..(tt.len() - n10) {
        s += tt[i];
    }
    let res1 = (s as f64) / (n80 as f64);

    // Phase 2: all signatures are incorrect.
    // We expect much lower variance in that case.
    let mut tt = [0; 128];
    for i in 0..tt.len() {
        let msg = [(i + 1) as u8; 32];
        let begin = core_cycles();
        x ^= (pkey.verify_trunc_raw(&sigs[i % 256], rm, &msg).is_some()) as u8;
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    // Remove 10% slowest and 10% fastest, make an average of the rest.
    let n10 = tt.len() / 10;
    let n80 = tt.len() - 2 * n10;
    let mut s = 0u64;
    for i in n10..(tt.len() - n10) {
        s += tt[i];
    }
    let res2 = (s as f64) / (n80 as f64);

    (res1, res2, x)
}

fn main() {
    let mut bx = 0u8;

    let (v, x) = bench_mul();
    bx ^= x;
    println!("Ed25519 point mul:             {:13.2}", v);
    let (v, x) = bench_mulgen();
    bx ^= x;
    println!("Ed25519 point mulgen:          {:13.2}", v);
    let (v, x) = bench_mul_add_mulgen();
    bx ^= x;
    println!("Ed25519 point mul_add_mulgen:  {:13.2}", v);
    let (v, x) = bench_skey_load();
    bx ^= x;
    println!("Ed25519 skey_load:             {:13.2}", v);
    let (v, x) = bench_skey_sign();
    bx ^= x;
    println!("Ed25519 sign:                  {:13.2}", v);
    let (v, x) = bench_pkey_verify();
    bx ^= x;
    println!("Ed25519 verify:                {:13.2}", v);
    let (v, x) = bench_decode();
    bx ^= x;
    println!("Ed25519 decode:                {:13.2}", v);
    let (v, x) = bench_encode();
    bx ^= x;
    println!("Ed25519 encode:                {:13.2}", v);

    let (v1, v2, x) = bench_pkey_verify_trunc(8);
    bx ^= x;
    println!("Ed25519 verify_trunc8:         {:13.2}  {:13.2}", v1, v2);
    let (v1, v2, x) = bench_pkey_verify_trunc(16);
    bx ^= x;
    println!("Ed25519 verify_trunc16:        {:13.2}  {:13.2}", v1, v2);
    /*
    let (v1, v2, x) = bench_pkey_verify_trunc(24);
    bx ^= x;
    println!("Ed25519 verify_trunc24:        {:13.2}  {:13.2}", v1, v2);
    let (v1, v2, x) = bench_pkey_verify_trunc(28);
    bx ^= x;
    println!("Ed25519 verify_trunc28:        {:13.2}  {:13.2}", v1, v2);
    let (v1, v2, x) = bench_pkey_verify_trunc(32);
    bx ^= x;
    println!("Ed25519 verify_trunc32:        {:13.2}  {:13.2}", v1, v2);
    */

    println!("{}", bx);
}
