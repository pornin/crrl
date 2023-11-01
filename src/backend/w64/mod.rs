// The zz module defines the Zu* type (custom non-modular integers with
// sizes of 128, 256 or 384 bits, with some constant-time operation to
// support scalar splitting in GLV and GLS curves). On aarch64 we use a
// 32-bit version, because the Arm Cortex-A55 has non-constant-time 64-bit
// multiplies (but 32-bit multiplies are constant-time).

#[cfg(any(
    feature = "zz32",
    all(
        not(feature = "zz64"),
        target_arch = "aarch64")))]
mod zz32;

#[cfg(any(
    feature = "zz32",
    all(
        not(feature = "zz64"),
        target_arch = "aarch64")))]
pub use zz32::{Zu128, Zu256, Zu384};

#[cfg(any(
    feature = "zz64",
    all(
        not(feature = "zz32"),
        not(target_arch = "aarch64"))))]
mod zz;

#[cfg(any(
    feature = "zz64",
    all(
        not(feature = "zz32"),
        not(target_arch = "aarch64"))))]
pub use zz::{Zu128, Zu256, Zu384};

// Module gf255 defines the generic GF255<MQ> type, with 64-bit limbs.
// It is used for GF255e and GF255s. For GF25519, an alternate implementation
// with 51-bit limbs is provided (in module gf25519) and used in some cases.
//  - If feature gf25519_m64 is set, then GF255<19> is used.
//  - If feature gf25519_m51 is set, then the alternate implementation is used.
//  - If neither gf25519_m64 nor gf25519_m51 is set, then the selected
//    implementation depends on the target architecture.
//  - Features gf25519_m51 and gf25519_m64 are mutually incompatible; they
//    cannot be both set at the same time.
#[cfg(all(
    feature = "gf255_m51",
    feature = "gf255_m64",
))]
compile_error!("cannot use m51 and m64 GF255 implementations simultaneously");

#[cfg(all(
    any(
        feature = "gf255",
        feature = "gf255e",
        feature = "gf255s",
        feature = "gf25519"),
    not(feature = "gf255_m51"),
    any(
        feature = "gf255_m64",
        not(target_arch = "riscv64")),
))]
pub mod gf255_m64;

#[cfg(all(
    any(
        feature = "gf255",
        feature = "gf255e",
        feature = "gf255s",
        feature = "gf25519"),
    not(feature = "gf255_m51"),
    any(
        feature = "gf255_m64",
        not(target_arch = "riscv64")),
))]
pub use gf255_m64::GF255;

#[cfg(all(
    any(
        feature = "gf255",
        feature = "gf255e",
        feature = "gf255s",
        feature = "gf25519"),
    not(feature = "gf255_m64"),
    any(
        feature = "gf255_m51",
        target_arch = "riscv64"),
))]
pub mod gf255_m51;

#[cfg(all(
    any(
        feature = "gf255",
        feature = "gf255e",
        feature = "gf255s",
        feature = "gf25519"),
    not(feature = "gf255_m64"),
    any(
        feature = "gf255_m51",
        target_arch = "riscv64"),
))]
pub use gf255_m51::GF255;

#[cfg(feature = "gf255e")]
pub type GF255e = GF255<18651>;

#[cfg(feature = "gf255s")]
pub type GF255s = GF255<3957>;

#[cfg(feature = "gf25519")]
pub type GF25519 = GF255<19>;

#[cfg(any(
    feature = "modint256",
    feature = "gfp256",
))]
pub mod modint;

#[cfg(feature = "modint256")]
pub use modint::ModInt256;

#[cfg(all(
    feature = "modint256",
    not(target_arch = "aarch64")))]
pub type ModInt256ct<const M0: u64, const M1: u64, const M2: u64, const M3: u64> = ModInt256<M0, M1, M2, M3>;

#[cfg(all(
    feature = "modint256",
    target_arch = "aarch64"))]
pub mod modint32;

#[cfg(all(
    feature = "modint256",
    target_arch = "aarch64"))]
pub use modint32::ModInt256ct;

/* disabled -- not faster than the generic code
#[cfg(feature = "gfp256")]
pub mod gfp256;

#[cfg(feature = "gfp256")]
pub use gfp256::GFp256;
*/

#[cfg(feature = "gfp256")]
pub type GFp256 = modint::ModInt256<
    0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFF,
    0x0000000000000000, 0xFFFFFFFF00000001>;

#[cfg(feature = "gfp256")]
impl GFp256 {
    /// Encodes a scalar element into bytes (little-endian).
    pub fn encode(self) -> [u8; 32] {
        self.encode32()
    }
}

#[cfg(feature = "secp256k1")]
pub mod gfsecp256k1;

#[cfg(feature = "secp256k1")]
pub use gfsecp256k1::GFsecp256k1;

#[cfg(feature = "gf448")]
pub mod gf448;

#[cfg(feature = "gf448")]
pub use gf448::GF448;

pub mod lagrange;

#[cfg(feature = "gfgen")]
pub mod gfgen;

#[cfg(all(
    feature = "gfb254",
    not(any(
        feature = "gfb254_m64",
        feature = "gfb254_arm64pmull")),
    any(
        feature = "gfb254_x86clmul",
        all(
            target_arch = "x86_64",
            target_feature = "sse4.1",
            target_feature = "pclmulqdq"))))]
pub mod gfb254_x86clmul;

#[cfg(all(
    feature = "gfb254",
    not(any(
        feature = "gfb254_m64",
        feature = "gfb254_arm64pmull")),
    any(
        feature = "gfb254_x86clmul",
        all(
            target_arch = "x86_64",
            target_feature = "sse4.1",
            target_feature = "pclmulqdq"))))]
pub use gfb254_x86clmul::{GFb127, GFb254};

#[cfg(all(
    feature = "gfb254",
    not(any(
        feature = "gfb254_x86clmul",
        feature = "gfb254_m64")),
    any(
        feature = "gfb254_arm64pmull",
        all(
            target_arch = "aarch64",
            target_feature = "aes"))))]
pub mod gfb254_arm64pmull;

#[cfg(all(
    feature = "gfb254",
    not(any(
        feature = "gfb254_x86clmul",
        feature = "gfb254_m64")),
    any(
        feature = "gfb254_arm64pmull",
        all(
            target_arch = "aarch64",
            target_feature = "aes"))))]
pub use gfb254_arm64pmull::{GFb127, GFb254};

#[cfg(all(
    feature = "gfb254",
    not(any(
        feature = "gfb254_x86clmul",
        feature = "gfb254_arm64pmull")),
    any(
        feature = "gfb254_m64",
        not(any(
            all(
                target_arch = "x86_64",
                target_feature = "sse4.1",
                target_feature = "pclmulqdq"),
            all(
                target_arch = "aarch64",
                target_feature = "aes"))))))]
pub mod gfb254_m64;

#[cfg(all(
    feature = "gfb254",
    not(any(
        feature = "gfb254_x86clmul",
        feature = "gfb254_arm64pmull")),
    any(
        feature = "gfb254_m64",
        not(any(
            all(
                target_arch = "x86_64",
                target_feature = "sse4.1",
                target_feature = "pclmulqdq"),
            all(
                target_arch = "aarch64",
                target_feature = "aes"))))))]
pub use gfb254_m64::{GFb127, GFb254};

// The 32-bit variants of the addcarry, umull,... functions.
pub(crate) mod util32;

// Carrying addition and subtraction should use u64::carrying_add()
// and u64::borrowing_sub(), but these functions are currently only
// experimental.

// Add with carry; carry is 0 or 1.
// (x, y, c_in) -> x + y + c_in mod 2^64, c_out

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn addcarry_u64(x: u64, y: u64, c: u8) -> (u64, u8) {
    use core::arch::x86_64::_addcarry_u64;
    unsafe {
        let mut d = 0u64;
        let cc = _addcarry_u64(c, x, y, &mut d);
        (d, cc)
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)]
#[inline(always)]
pub(crate) const fn addcarry_u64(x: u64, y: u64, c: u8) -> (u64, u8) {
    let z = (x as u128).wrapping_add(y as u128).wrapping_add(c as u128);
    (z as u64, (z >> 64) as u8)
}

// Subtract with borrow; borrow is 0 or 1.
// (x, y, c_in) -> x - y - c_in mod 2^64, c_out

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn subborrow_u64(x: u64, y: u64, c: u8) -> (u64, u8) {
    use core::arch::x86_64::_subborrow_u64;
    unsafe {
        let mut d = 0u64;
        let cc = _subborrow_u64(c, x, y, &mut d);
        (d, cc)
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)]
#[inline(always)]
pub(crate) const fn subborrow_u64(x: u64, y: u64, c: u8) -> (u64, u8) {
    let z = (x as u128).wrapping_sub(y as u128).wrapping_sub(c as u128);
    (z as u64, (z >> 127) as u8)
}

// Compute x*y over 128 bits, returned as two 64-bit words (lo, hi)
#[allow(dead_code)]
#[inline(always)]
pub(crate) const fn umull(x: u64, y: u64) -> (u64, u64) {
    let z = (x as u128) * (y as u128);
    (z as u64, (z >> 64) as u64)
}

// Compute x*y+z over 128 bits, returned as two 64-bit words (lo, hi)
#[allow(dead_code)]
#[inline(always)]
pub(crate) const fn umull_add(x: u64, y: u64, z: u64) -> (u64, u64) {
    let t = ((x as u128) * (y as u128)).wrapping_add(z as u128);
    (t as u64, (t >> 64) as u64)
}

// Compute x*y+z1+z2 over 128 bits, returned as two 64-bit words (lo, hi)
#[allow(dead_code)]
#[inline(always)]
pub(crate) const fn umull_add2(x: u64, y: u64, z1: u64, z2: u64) -> (u64, u64) {
    let t = ((x as u128) * (y as u128))
        .wrapping_add(z1 as u128).wrapping_add(z2 as u128);
    (t as u64, (t >> 64) as u64)
}

// Compute x1*y1+x2*y2 over 128 bits, returned as two 64-bit words (lo, hi)
#[allow(dead_code)]
#[inline(always)]
pub(crate) const fn umull_x2(x1: u64, y1: u64, x2: u64, y2: u64) -> (u64, u64) {
    let z1 = (x1 as u128) * (y1 as u128);
    let z2 = (x2 as u128) * (y2 as u128);
    let z = z1.wrapping_add(z2);
    (z as u64, (z >> 64) as u64)
}

// Compute x1*y1+x2*y2+z3 over 128 bits, returned as two 64-bit words (lo, hi)
#[allow(dead_code)]
#[inline(always)]
pub(crate) const fn umull_x2_add(x1: u64, y1: u64, x2: u64, y2: u64, z3: u64) -> (u64, u64) {
    let z1 = (x1 as u128) * (y1 as u128);
    let z2 = (x2 as u128) * (y2 as u128);
    let z = z1.wrapping_add(z2).wrapping_add(z3 as u128);
    (z as u64, (z >> 64) as u64)
}

// Return 0xFFFFFFFFFFFFFFFF if x >= 0x8000000000000000, 0 otherwise
// (i.e. take the sign bit of the signed interpretation, and expand it
// to 64 bits).
#[allow(dead_code)]
#[inline(always)]
pub(crate) const fn sgnw(x: u64) -> u64 {
    ((x as i64) >> 63) as u64
}

// Get the number of leading zeros in a 64-bit value.
// On some platforms, u64::leading_zeros() performs the computation with
// a code sequence that will be constant-time on most/all CPUs
// compatible with that platforms (e.g. any 64-bit x86 with support for
// the LZCNT opcode); on others, a non-constant-time sequence would be
// used, and we must instead rely on a safe (but slower) routine.
//
// On x86 without LZCNT, u64::leading_zeros() uses a BSR opcode, but since
// BSR yields an undefined result on an input of value 0, u64::leading_zeros()
// includes an explicit test and a conditional jump for that case, and that
// is not (in general) constant-time.
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "lzcnt"),
    target_arch = "aarch64",
    ))]
#[allow(dead_code)]
#[inline(always)]
pub(crate) const fn lzcnt(x: u64) -> u32 {
    x.leading_zeros()
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "lzcnt"),
    target_arch = "aarch64",
    )))]
#[allow(dead_code)]
pub(crate) const fn lzcnt(x: u64) -> u32 {
    let m = sgnw((x >> 32).wrapping_sub(1));
    let s = m & 32;
    let x = (x >> 32) ^ (m & (x ^ (x >> 32)));

    let m = sgnw((x >> 16).wrapping_sub(1));
    let s = s | (m & 16);
    let x = (x >> 16) ^ (m & (x ^ (x >> 16)));

    let m = sgnw((x >>  8).wrapping_sub(1));
    let s = s | (m &  8);
    let x = (x >>  8) ^ (m & (x ^ (x >>  8)));

    let m = sgnw((x >>  4).wrapping_sub(1));
    let s = s | (m &  4);
    let x = (x >>  4) ^ (m & (x ^ (x >>  4)));

    let m = sgnw((x >>  2).wrapping_sub(1));
    let s = s | (m &  2);
    let x = (x >>  2) ^ (m & (x ^ (x >>  2)));

    // At this point, x fits on 2 bits. Number of leading zeros is then:
    //   x = 0   -> 2
    //   x = 1   -> 1
    //   x = 2   -> 0
    //   x = 3   -> 0
    let s = s.wrapping_add(2u64.wrapping_sub(x) & ((x.wrapping_sub(3) >> 2)));

    s as u32
}
