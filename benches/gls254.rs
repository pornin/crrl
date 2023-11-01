#![allow(non_snake_case)]
#![cfg(feature = "gls254")]

mod util;
use util::core_cycles;

use crrl::gls254::{Point, Scalar, PrivateKey};
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

fn bench_hash_to_curve() -> (f64, u8) {
    let mut buf = [0u8; 32];
    for i in 0..4 {
        let z = core_cycles();
        buf[(8 * i)..(8 * i + 8)].copy_from_slice(&z.to_le_bytes());
    }
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..100 {
            let P = Point::hash_to_curve("", &buf);
            buf[0] += P.isneutral() as u8;
            buf[1] += 3;
            buf[2] += 5;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[4] as f64) / 100.0, buf[0])
}

fn bench_split_mu() -> (f64, u8) {
    let z = core_cycles();
    let mut x = Scalar::from_u64(z);
    x.set_xsquare(5);
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            let (k0, s0, k1, s1) = Point::split_mu(&x);
            let mut buf = [0u8; 24];
            buf[..16].copy_from_slice(&(k0 ^ k1).to_le_bytes());
            buf[16..20].copy_from_slice(&s0.to_le_bytes());
            buf[20..24].copy_from_slice(&s1.to_le_bytes());
            x.set_decode_reduce(&buf);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[4] as f64) / 1000.0, x.encode()[0])
}

#[cfg(feature = "gls254bench")]
fn bench_raw_ecdh_1dt_3() -> (f64, u8) {
    let z = core_cycles();
    let mut seed = [0u8; 32];
    seed[ 0.. 8].copy_from_slice(&z.to_le_bytes());
    seed[ 8..16].copy_from_slice(&z.to_le_bytes());
    seed[16..24].copy_from_slice(&z.to_le_bytes());
    seed[24..32].copy_from_slice(&z.to_le_bytes());
    let mut sk = Scalar::decode_reduce(&seed).encode();
    let mut pp: [u8; 64] = [
        0x80, 0xAE, 0xB8, 0xED, 0x53, 0x59, 0xFF, 0x2D,
        0xD0, 0x77, 0x45, 0x61, 0xF9, 0x22, 0xE4, 0x63,
        0x9C, 0xEE, 0x3A, 0xF1, 0xE8, 0xF7, 0x23, 0x80,
        0x74, 0x5A, 0x57, 0x29, 0xC5, 0xAA, 0xF5, 0x02,
        0xA7, 0x52, 0x43, 0xDF, 0xCA, 0xE4, 0x13, 0x95,
        0xD8, 0x49, 0xE7, 0xC8, 0x52, 0x6E, 0x4D, 0x6E,
        0x03, 0x34, 0x21, 0x67, 0x21, 0x47, 0x37, 0xA4,
        0x0C, 0x67, 0x34, 0x13, 0xF3, 0x48, 0x4B, 0x7D,
    ];
    pp = Point::for_benchmarks_only_1dt_3(&pp, &sk).unwrap();
    let mut tt = [0; 100];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for _ in 0..100 {
            sk[..].copy_from_slice(&pp[..32]);
            sk[31] &= 0x1F;
            pp = Point::for_benchmarks_only_1dt_3(&pp, &sk).unwrap();
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / 100.0, pp[0])
}

#[cfg(feature = "gls254bench")]
fn bench_raw_ecdh_1dt_4() -> (f64, u8) {
    let z = core_cycles();
    let mut seed = [0u8; 32];
    seed[ 0.. 8].copy_from_slice(&z.to_le_bytes());
    seed[ 8..16].copy_from_slice(&z.to_le_bytes());
    seed[16..24].copy_from_slice(&z.to_le_bytes());
    seed[24..32].copy_from_slice(&z.to_le_bytes());
    let mut sk = Scalar::decode_reduce(&seed).encode();
    let mut pp: [u8; 64] = [
        0x80, 0xAE, 0xB8, 0xED, 0x53, 0x59, 0xFF, 0x2D,
        0xD0, 0x77, 0x45, 0x61, 0xF9, 0x22, 0xE4, 0x63,
        0x9C, 0xEE, 0x3A, 0xF1, 0xE8, 0xF7, 0x23, 0x80,
        0x74, 0x5A, 0x57, 0x29, 0xC5, 0xAA, 0xF5, 0x02,
        0xA7, 0x52, 0x43, 0xDF, 0xCA, 0xE4, 0x13, 0x95,
        0xD8, 0x49, 0xE7, 0xC8, 0x52, 0x6E, 0x4D, 0x6E,
        0x03, 0x34, 0x21, 0x67, 0x21, 0x47, 0x37, 0xA4,
        0x0C, 0x67, 0x34, 0x13, 0xF3, 0x48, 0x4B, 0x7D,
    ];
    pp = Point::for_benchmarks_only_1dt_4(&pp, &sk).unwrap();
    let mut tt = [0; 100];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for _ in 0..100 {
            sk[..].copy_from_slice(&pp[..32]);
            sk[31] &= 0x1F;
            pp = Point::for_benchmarks_only_1dt_4(&pp, &sk).unwrap();
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / 100.0, pp[0])
}

#[cfg(feature = "gls254bench")]
fn bench_raw_ecdh_1dt_5() -> (f64, u8) {
    let z = core_cycles();
    let mut seed = [0u8; 32];
    seed[ 0.. 8].copy_from_slice(&z.to_le_bytes());
    seed[ 8..16].copy_from_slice(&z.to_le_bytes());
    seed[16..24].copy_from_slice(&z.to_le_bytes());
    seed[24..32].copy_from_slice(&z.to_le_bytes());
    let mut sk = Scalar::decode_reduce(&seed).encode();
    let mut pp: [u8; 64] = [
        0x80, 0xAE, 0xB8, 0xED, 0x53, 0x59, 0xFF, 0x2D,
        0xD0, 0x77, 0x45, 0x61, 0xF9, 0x22, 0xE4, 0x63,
        0x9C, 0xEE, 0x3A, 0xF1, 0xE8, 0xF7, 0x23, 0x80,
        0x74, 0x5A, 0x57, 0x29, 0xC5, 0xAA, 0xF5, 0x02,
        0xA7, 0x52, 0x43, 0xDF, 0xCA, 0xE4, 0x13, 0x95,
        0xD8, 0x49, 0xE7, 0xC8, 0x52, 0x6E, 0x4D, 0x6E,
        0x03, 0x34, 0x21, 0x67, 0x21, 0x47, 0x37, 0xA4,
        0x0C, 0x67, 0x34, 0x13, 0xF3, 0x48, 0x4B, 0x7D,
    ];
    pp = Point::for_benchmarks_only_1dt_5(&pp, &sk).unwrap();
    let mut tt = [0; 100];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for _ in 0..100 {
            sk[..].copy_from_slice(&pp[..32]);
            sk[31] &= 0x1F;
            pp = Point::for_benchmarks_only_1dt_5(&pp, &sk).unwrap();
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / 100.0, pp[0])
}

#[cfg(feature = "gls254bench")]
fn bench_raw_ecdh_2dt_2() -> (f64, u8) {
    let z = core_cycles();
    let mut seed = [0u8; 32];
    seed[ 0.. 8].copy_from_slice(&z.to_le_bytes());
    seed[ 8..16].copy_from_slice(&z.to_le_bytes());
    seed[16..24].copy_from_slice(&z.to_le_bytes());
    seed[24..32].copy_from_slice(&z.to_le_bytes());
    let mut sk = Scalar::decode_reduce(&seed).encode();
    let mut pp: [u8; 64] = [
        0x80, 0xAE, 0xB8, 0xED, 0x53, 0x59, 0xFF, 0x2D,
        0xD0, 0x77, 0x45, 0x61, 0xF9, 0x22, 0xE4, 0x63,
        0x9C, 0xEE, 0x3A, 0xF1, 0xE8, 0xF7, 0x23, 0x80,
        0x74, 0x5A, 0x57, 0x29, 0xC5, 0xAA, 0xF5, 0x02,
        0xA7, 0x52, 0x43, 0xDF, 0xCA, 0xE4, 0x13, 0x95,
        0xD8, 0x49, 0xE7, 0xC8, 0x52, 0x6E, 0x4D, 0x6E,
        0x03, 0x34, 0x21, 0x67, 0x21, 0x47, 0x37, 0xA4,
        0x0C, 0x67, 0x34, 0x13, 0xF3, 0x48, 0x4B, 0x7D,
    ];
    pp = Point::for_benchmarks_only_2dt_2(&pp, &sk).unwrap();
    let mut tt = [0; 100];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for _ in 0..100 {
            sk[..].copy_from_slice(&pp[..32]);
            sk[31] &= 0x1F;
            pp = Point::for_benchmarks_only_2dt_2(&pp, &sk).unwrap();
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / 100.0, pp[0])
}

#[cfg(feature = "gls254bench")]
fn bench_raw_ecdh_2dt_3() -> (f64, u8) {
    let z = core_cycles();
    let mut seed = [0u8; 32];
    seed[ 0.. 8].copy_from_slice(&z.to_le_bytes());
    seed[ 8..16].copy_from_slice(&z.to_le_bytes());
    seed[16..24].copy_from_slice(&z.to_le_bytes());
    seed[24..32].copy_from_slice(&z.to_le_bytes());
    let mut sk = Scalar::decode_reduce(&seed).encode();
    let mut pp: [u8; 64] = [
        0x80, 0xAE, 0xB8, 0xED, 0x53, 0x59, 0xFF, 0x2D,
        0xD0, 0x77, 0x45, 0x61, 0xF9, 0x22, 0xE4, 0x63,
        0x9C, 0xEE, 0x3A, 0xF1, 0xE8, 0xF7, 0x23, 0x80,
        0x74, 0x5A, 0x57, 0x29, 0xC5, 0xAA, 0xF5, 0x02,
        0xA7, 0x52, 0x43, 0xDF, 0xCA, 0xE4, 0x13, 0x95,
        0xD8, 0x49, 0xE7, 0xC8, 0x52, 0x6E, 0x4D, 0x6E,
        0x03, 0x34, 0x21, 0x67, 0x21, 0x47, 0x37, 0xA4,
        0x0C, 0x67, 0x34, 0x13, 0xF3, 0x48, 0x4B, 0x7D,
    ];
    pp = Point::for_benchmarks_only_2dt_3(&pp, &sk).unwrap();
    let mut tt = [0; 100];
    for i in 0..tt.len() {
        let begin = core_cycles();
        for _ in 0..100 {
            sk[..].copy_from_slice(&pp[..32]);
            sk[31] &= 0x1F;
            pp = Point::for_benchmarks_only_2dt_3(&pp, &sk).unwrap();
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[tt.len() >> 1] as f64) / 100.0, pp[0])
}

fn main() {
    let mut bx = 0u8;

    let (v, x) = bench_mul();
    bx ^= x;
    println!("GLS254 point mul:              {:13.2}", v);
    #[cfg(feature = "gls254bench")]
    {
        let (v, x) = bench_raw_ecdh_1dt_3();
        bx ^= x;
        println!("GLS254 raw_ECDH 1DT-3:         {:13.2}", v);
        let (v, x) = bench_raw_ecdh_1dt_4();
        bx ^= x;
        println!("GLS254 raw_ECDH 1DT-4:         {:13.2}", v);
        let (v, x) = bench_raw_ecdh_1dt_5();
        bx ^= x;
        println!("GLS254 raw_ECDH 1DT-5:         {:13.2}", v);
        let (v, x) = bench_raw_ecdh_2dt_2();
        bx ^= x;
        println!("GLS254 raw_ECDH 2DT-2:         {:13.2}", v);
        let (v, x) = bench_raw_ecdh_2dt_3();
        bx ^= x;
        println!("GLS254 raw_ECDH 2DT-3:         {:13.2}", v);
    }
    let (v, x) = bench_mulgen();
    bx ^= x;
    println!("GLS254 point mulgen:           {:13.2}", v);
    let (v, x) = bench_skey_load();
    bx ^= x;
    println!("GLS254 skey_load:              {:13.2}", v);
    let (v, x) = bench_skey_sign();
    bx ^= x;
    println!("GLS254 sign:                   {:13.2}", v);
    let (v, x) = bench_pkey_verify();
    bx ^= x;
    println!("GLS254 verify:                 {:13.2}", v);
    let (v, x) = bench_decode();
    bx ^= x;
    println!("GLS254 decode:                 {:13.2}", v);
    let (v, x) = bench_encode();
    bx ^= x;
    println!("GLS254 encode:                 {:13.2}", v);
    let (v, x) = bench_hash_to_curve();
    bx ^= x;
    println!("GLS254 hash-to-curve:          {:13.2}", v);
    let (v, x) = bench_split_mu();
    bx ^= x;
    println!("GLS254 split_mu:               {:13.2}", v);

    println!("{}", bx);
}
