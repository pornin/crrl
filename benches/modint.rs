#![cfg(feature = "modint256")]

mod util;
use util::core_cycles;

use crrl::field::ModInt256;
use sha2::{Sha256, Digest};

fn bench_modint256_add<const M0: u64, const M1: u64,
                       const M2: u64, const M3: u64>() -> (f64, u8)
{
    let z = core_cycles();
    let mut x = ModInt256::<M0, M1, M2, M3>::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut y = x + ModInt256::<M0, M1, M2, M3>::ONE;
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..10000 {
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
    ((tt[4] as f64) / 60000.0, x.encode32()[0])
}

fn bench_modint256_sub<const M0: u64, const M1: u64,
                       const M2: u64, const M3: u64>() -> (f64, u8)
{
    let z = core_cycles();
    let mut x = ModInt256::<M0, M1, M2, M3>::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut y = x + ModInt256::<M0, M1, M2, M3>::ONE;
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..10000 {
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
    ((tt[4] as f64) / 60000.0, x.encode32()[0])
}

fn bench_modint256_mul<const M0: u64, const M1: u64,
                       const M2: u64, const M3: u64>() -> (f64, u8)
{
    let z = core_cycles();
    let mut x = ModInt256::<M0, M1, M2, M3>::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut y = x + ModInt256::<M0, M1, M2, M3>::ONE;
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..10000 {
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
    ((tt[4] as f64) / 60000.0, x.encode32()[0])
}

fn bench_modint256_square<const M0: u64, const M1: u64,
                          const M2: u64, const M3: u64>() -> (f64, u8)
{
    let z = core_cycles();
    let mut x = ModInt256::<M0, M1, M2, M3>::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        x = x.xsquare(60000);
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[4] as f64) / 60000.0, x.encode32()[0])
}

fn bench_modint256_div<const M0: u64, const M1: u64,
                       const M2: u64, const M3: u64>() -> (f64, u8)
{
    let z = core_cycles();
    let mut x = ModInt256::<M0, M1, M2, M3>::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut y = x + ModInt256::<M0, M1, M2, M3>::ONE;
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
    ((tt[4] as f64) / 6000.0, x.encode32()[0])
}

fn bench_modint256_sqrt<const M0: u64, const M1: u64,
                        const M2: u64, const M3: u64>() -> (f64, u8)
{
    let z = core_cycles();
    let mut x = ModInt256::<M0, M1, M2, M3>::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            let (x2, _) = x.sqrt();
            x = x2 + ModInt256::<M0, M1, M2, M3>::ONE;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[4] as f64) / 1000.0, x.encode32()[0])
}

fn bench_modint256_legendre<const M0: u64, const M1: u64,
                            const M2: u64, const M3: u64>() -> (f64, u8)
{
    let z = core_cycles();
    let mut x = ModInt256::<M0, M1, M2, M3>::w64le(z, z.wrapping_mul(3),
        z.wrapping_mul(5), z.wrapping_mul(7));
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            let ls = x.legendre();
            x += ModInt256::<M0, M1, M2, M3>::w64le(ls as u64, ls as u64, ls as u64, ls as u64);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[4] as f64) / 1000.0, x.encode32()[0])
}

fn bench_modint256_split<const M0: u64, const M1: u64,
                            const M2: u64, const M3: u64>() -> (f64, u8)
{
    let z = core_cycles();

    // Generate 512 pseudorandom elements. Number 512 was chosen so that
    // the total in-RAM size is 16 kB, which should fit in L1 cache with
    // enough room.
    let mut vv = [ModInt256::<M0, M1, M2, M3>::ZERO; 512];
    let mut sh = Sha256::new();
    for i in 0..512 {
        sh.update(z.to_le_bytes());
        sh.update((i as u64).to_le_bytes());
        let bb = sh.finalize_reset();
        vv[i] = ModInt256::<M0, M1, M2, M3>::decode_reduce(&bb);
    }

    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for j in 0..512 {
            let (c0, c1) = vv[j].split_vartime();
            let x = c0.wrapping_add(c1);
            vv[(j + 1) & 511] += ModInt256::<M0, M1, M2, M3>::from_i128(x);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    ((tt[4] as f64) / 512.0, vv[0].encode32()[0])
}

fn main() {
    let mut bx = 0u8;

    let (f1, b1) = bench_modint256_add::<0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFF, 0x0000000000000000, 0xFFFFFFFF00000001>();
    let (f2, b2) = bench_modint256_add::<0xFFFFFFFFFFFFFFED, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF>();
    let (f3, b3) = bench_modint256_add::<0xFFFFFFFFFFFFFF43, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF>();
    bx ^= b1 ^ b2 ^ b3;
    println!("ModInt256 add:         {:11.2} {:11.2} {:11.2}", f1, f2, f3);

    let (f1, b1) = bench_modint256_sub::<0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFF, 0x0000000000000000, 0xFFFFFFFF00000001>();
    let (f2, b2) = bench_modint256_sub::<0xFFFFFFFFFFFFFFED, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF>();
    let (f3, b3) = bench_modint256_sub::<0xFFFFFFFFFFFFFF43, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF>();
    bx ^= b1 ^ b2 ^ b3;
    println!("ModInt256 sub:         {:11.2} {:11.2} {:11.2}", f1, f2, f3);

    let (f1, b1) = bench_modint256_mul::<0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFF, 0x0000000000000000, 0xFFFFFFFF00000001>();
    let (f2, b2) = bench_modint256_mul::<0xFFFFFFFFFFFFFFED, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF>();
    let (f3, b3) = bench_modint256_mul::<0xFFFFFFFFFFFFFF43, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF>();
    bx ^= b1 ^ b2 ^ b3;
    println!("ModInt256 mul:         {:11.2} {:11.2} {:11.2}", f1, f2, f3);

    let (f1, b1) = bench_modint256_square::<0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFF, 0x0000000000000000, 0xFFFFFFFF00000001>();
    let (f2, b2) = bench_modint256_square::<0xFFFFFFFFFFFFFFED, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF>();
    let (f3, b3) = bench_modint256_square::<0xFFFFFFFFFFFFFF43, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF>();
    bx ^= b1 ^ b2 ^ b3;
    println!("ModInt256 square:      {:11.2} {:11.2} {:11.2}", f1, f2, f3);

    let (f1, b1) = bench_modint256_div::<0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFF, 0x0000000000000000, 0xFFFFFFFF00000001>();
    let (f2, b2) = bench_modint256_div::<0xFFFFFFFFFFFFFFED, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF>();
    let (f3, b3) = bench_modint256_div::<0xFFFFFFFFFFFFFF43, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF>();
    bx ^= b1 ^ b2 ^ b3;
    println!("ModInt256 div:         {:11.2} {:11.2} {:11.2}", f1, f2, f3);

    let (f1, b1) = bench_modint256_sqrt::<0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFF, 0x0000000000000000, 0xFFFFFFFF00000001>();
    let (f2, b2) = bench_modint256_sqrt::<0xFFFFFFFFFFFFFFED, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF>();
    let (f3, b3) = bench_modint256_sqrt::<0xFFFFFFFFFFFFFF43, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF>();
    bx ^= b1 ^ b2 ^ b3;
    println!("ModInt256 sqrt:        {:11.2} {:11.2} {:11.2}", f1, f2, f3);

    let (f1, b1) = bench_modint256_legendre::<0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFF, 0x0000000000000000, 0xFFFFFFFF00000001>();
    let (f2, b2) = bench_modint256_legendre::<0xFFFFFFFFFFFFFFED, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF>();
    let (f3, b3) = bench_modint256_legendre::<0xFFFFFFFFFFFFFF43, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF>();
    bx ^= b1 ^ b2 ^ b3;
    println!("ModInt256 legendre:    {:11.2} {:11.2} {:11.2}", f1, f2, f3);

    let (f1, b1) = bench_modint256_split::<0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFF, 0x0000000000000000, 0xFFFFFFFF00000001>();
    let (f2, b2) = bench_modint256_split::<0xFFFFFFFFFFFFFFED, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF>();
    let (f3, b3) = bench_modint256_split::<0xFFFFFFFFFFFFFF43, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF>();
    bx ^= b1 ^ b2 ^ b3;
    println!("ModInt256 split (var)  {:11.2} {:11.2} {:11.2}", f1, f2, f3);

    println!("{}", bx);
}
