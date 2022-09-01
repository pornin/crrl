pub mod gf255;
pub mod modint;
pub mod gfsecp256k1;
pub mod lagrange;

// Carrying addition and subtraction should use u64::carrying_add()
// and u64::borrowing_sub(), but these functions are currently only
// experimental.

// Add with carry; carry is 0 or 1.
// (x, y, c_in) -> x + y + c_in mod 2^64, c_out

#[cfg(target_arch = "x86_64")]
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
#[inline(always)]
pub(crate) const fn addcarry_u64(x: u64, y: u64, c: u8) -> (u64, u8) {
    let z = (x as u128).wrapping_add(y as u128).wrapping_add(c as u128);
    (z as u64, (z >> 64) as u8)
}

// Subtract with borrow; borrow is 0 or 1.
// (x, y, c_in) -> x - y - c_in mod 2^64, c_out

#[cfg(target_arch = "x86_64")]
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
#[inline(always)]
pub(crate) const fn subborrow_u64(x: u64, y: u64, c: u8) -> (u64, u8) {
    let z = (x as u128).wrapping_sub(y as u128).wrapping_sub(c as u128);
    (z as u64, (z >> 127) as u8)
}

// Compute x*y over 128 bits, returned as two 64-bit words (lo, hi)
#[inline(always)]
pub(crate) const fn umull(x: u64, y: u64) -> (u64, u64) {
    let z = (x as u128) * (y as u128);
    (z as u64, (z >> 64) as u64)
}

// Compute x*y+z over 128 bits, returned as two 64-bit words (lo, hi)
#[inline(always)]
pub(crate) const fn umull_add(x: u64, y: u64, z: u64) -> (u64, u64) {
    let t = ((x as u128) * (y as u128)).wrapping_add(z as u128);
    (t as u64, (t >> 64) as u64)
}

// Compute x*y+z1+z2 over 128 bits, returned as two 64-bit words (lo, hi)
#[inline(always)]
pub(crate) const fn umull_add2(x: u64, y: u64, z1: u64, z2: u64) -> (u64, u64) {
    let t = ((x as u128) * (y as u128))
        .wrapping_add(z1 as u128).wrapping_add(z2 as u128);
    (t as u64, (t >> 64) as u64)
}

// Compute x1*y1+x2*y2 over 128 bits, returned as two 64-bit words (lo, hi)
#[inline(always)]
pub(crate) const fn umull_x2(x1: u64, y1: u64, x2: u64, y2: u64) -> (u64, u64) {
    let z1 = (x1 as u128) * (y1 as u128);
    let z2 = (x2 as u128) * (y2 as u128);
    let z = z1.wrapping_add(z2);
    (z as u64, (z >> 64) as u64)
}

// Compute x1*y1+x2*y2+z3 over 128 bits, returned as two 64-bit words (lo, hi)
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
#[inline(always)]
pub(crate) const fn lzcnt(x: u64) -> u32 {
    x.leading_zeros()
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "lzcnt"),
    target_arch = "aarch64",
    )))]
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
