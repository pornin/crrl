#![allow(non_snake_case)]
#![cfg(feature = "jq255s")]

mod util;
use util::core_cycles;

use crrl::jq255s::{Point, Scalar, PrivateKey};
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
            let skey = PrivateKey::decode(&seed).unwrap();
            seed[..].copy_from_slice(&skey.public_key.encode());
            seed[31] &= 0x1Fu8;
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
    let mut sh = Sha256::new();
    sh.update(&seed);
    seed[..].copy_from_slice(&sh.finalize());
    seed[31] &= 0x1Fu8;
    let skey = PrivateKey::decode(&seed).unwrap();
    let mut tt = [0; 100];
    let mut msg = [0u8; 32];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for _ in 0..100 {
            let sig = skey.sign("", &msg);
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
    let mut sh = Sha256::new();
    sh.update(&seed);
    seed[..].copy_from_slice(&sh.finalize());
    seed[31] &= 0x1Fu8;
    let skey = PrivateKey::decode(&seed).unwrap();
    let pkey = skey.public_key;
    let mut sigs = [[0u8; 48]; 128];
    for i in 0..128 {
        let msg = [i as u8; 32];
        let sig = skey.sign("", &msg);
        sigs[i][..].copy_from_slice(&sig);
    }
    let mut tt = [0; 100];
    let mut msg = [0u8; 32];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for j in 0..128 {
            let ff = pkey.verify(&sigs[j], "", &msg);
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

fn main() {
    let mut bx = 0u8;

    let (v, x) = bench_mul();
    bx ^= x;
    println!("Jq255s point mul:              {:13.2}", v);
    let (v, x) = bench_mulgen();
    bx ^= x;
    println!("Jq255s point mulgen:           {:13.2}", v);
    let (v, x) = bench_mul_add_mulgen();
    bx ^= x;
    println!("Jq255s point mul_add_mulgen:   {:13.2}", v);
    let (v, x) = bench_skey_load();
    bx ^= x;
    println!("Jq255s skey_load:              {:13.2}", v);
    let (v, x) = bench_skey_sign();
    bx ^= x;
    println!("Jq255s sign:                   {:13.2}", v);
    let (v, x) = bench_pkey_verify();
    bx ^= x;
    println!("Jq255s verify:                 {:13.2}", v);
    let (v, x) = bench_decode();
    bx ^= x;
    println!("Jq255s decode:                 {:13.2}", v);
    let (v, x) = bench_encode();
    bx ^= x;
    println!("Jq255s encode:                 {:13.2}", v);

    println!("{}", bx);
}
