#![allow(non_snake_case)]
#![cfg(feature = "secp256k1")]

mod util;
use util::core_cycles;

use crrl::secp256k1::{Point, Scalar, PrivateKey};
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
    ((tt[tt.len() >> 1] as f64) / 128.0, P.encode_compressed()[0])
}

fn bench_skey_sign() -> (f64, u8) {
    let z = core_cycles();
    let mut sh = Sha256::new();
    sh.update(&z.to_le_bytes());
    sh.update(&[0x00u8]);
    let s1 = sh.finalize_reset();
    sh.update(&z.to_le_bytes());
    sh.update(&[0x01u8]);
    let s2 = sh.finalize_reset();
    let mut seed = [0u8; 48];
    seed[..32].copy_from_slice(&s1);
    seed[32..].copy_from_slice(&s2[..16]);
    let skey = PrivateKey::from_seed(&seed);
    let mut tt = [0; 100];
    let mut msg = [0u8; 32];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for _ in 0..100 {
            let sig = skey.sign_hash(&msg, &[]);
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
    let mut sh = Sha256::new();
    sh.update(&z.to_le_bytes());
    sh.update(&[0x00u8]);
    let s1 = sh.finalize_reset();
    sh.update(&z.to_le_bytes());
    sh.update(&[0x01u8]);
    let s2 = sh.finalize_reset();
    let mut seed = [0u8; 48];
    seed[..32].copy_from_slice(&s1);
    seed[32..].copy_from_slice(&s2[..16]);
    let skey = PrivateKey::from_seed(&seed);
    let pkey = skey.to_public_key();
    let mut sigs = [[0u8; 64]; 128];
    for i in 0..128 {
        let msg = [i as u8; 32];
        let sig = skey.sign_hash(&msg, &[]);
        sigs[i][..].copy_from_slice(&sig);
    }
    let mut tt = [0; 100];
    let mut msg = [0u8; 32];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for j in 0..128 {
            let ff = pkey.verify_hash(&sigs[j], &msg);
            sigs[j][40] ^= 1u8.wrapping_add(ff as u8);
            msg[3] ^= 3u8.wrapping_sub(ff as u8);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / 128.0, msg[0])
}

fn main() {
    let mut bx = 0u8;

    let (v, x) = bench_mul();
    bx ^= x;
    println!("secp256k1 point mul:           {:13.2}", v);
    let (v, x) = bench_mulgen();
    bx ^= x;
    println!("secp256k1 point mulgen:        {:13.2}", v);
    let (v, x) = bench_mul_add_mulgen();
    bx ^= x;
    println!("secp256k1 point mul_add_mulgen:{:13.2}", v);
    let (v, x) = bench_skey_sign();
    bx ^= x;
    println!("secp256k1 sign:                {:13.2}", v);
    let (v, x) = bench_pkey_verify();
    bx ^= x;
    println!("secp256k1 verify:              {:13.2}", v);

    println!("{}", bx);
}
