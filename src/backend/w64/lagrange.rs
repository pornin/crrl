use super::{addcarry_u64, subborrow_u64, umull, umull_add, umull_add2, sgnw};
use core::convert::TryFrom;

// Given integers k and n, with 0 <= k < n < Nmax (with n prime), return
// signed integers c0 and c1 such that k = c0/c1 mod n. Integers are provided
// as arrays of 64-bit limbs in little-endian convention (least significant
// limb comes first). This function is NOT constant-time and MUST NOT be
// used with secret inputs.
//
// Limit Nmax is such that the solution always exists; its value is:
//   Nmax = floor(2^254 / (2/sqrt(3)))
//        = 0x376CF5D0B09954E764AE85AE0F17077124BB06998A7B48F318E414C90DC8B4DC
//
// If a larger n is provided as parameter, then the algorithm still
// terminates, but the real (c0, c1) may be larger than 128 bits, and thus
// only truncated results are returned.
#[allow(dead_code)]
pub(crate) fn lagrange253_vartime(k: &[u64; 4], n: &[u64; 4]) -> (i128, i128) {
    let (v0, v1) = lagrange256_vartime(k, n, 254);
    let c0 = ((v0[0] as u128) | ((v0[1] as u128) << 64)) as i128;
    let c1 = ((v1[0] as u128) | ((v1[1] as u128) << 64)) as i128;
    (c0, c1)
}

// ========================================================================

/* unused
#[derive(Clone, Copy, Debug)]
struct ZInt64(u64);

#[allow(dead_code)]
impl ZInt64 {
    const BITLEN: usize = 64;
    const N: usize = 1;
    const ZERO: Self = Self(0);

    // Return true iff self < rhs (inputs must be nonnegative).
    fn lt(self, rhs: &Self) -> bool {
        self.0 < rhs.0
    }

    // Swap the contents of self with rhs.
    fn swap(&mut self, rhs: &mut Self) {
        let t = self.0;
        self.0 = rhs.0;
        rhs.0 = t;
    }

    // Get the length (in bits) of this value. If `unsigned` is true,
    // then the value is considered unsigned; otherwise, the top bit
    // is a sign bit.
    fn bitlength(self, unsigned: bool) -> u32 {
        let m = if unsigned { 0 } else { sgnw(self.0) };
        return 64 - (self.0 ^ m).leading_zeros();
    }

    // Return true if self is lower than 2^(64*s - 1). The value self
    // MUST be non-negative. The value s MUST be greater than 0, and
    // not greater than Self::N.
    fn ltnw(self, _s: usize) -> bool {
        // Parameter requirements imply that s == 1.
        (self.0 as i64) >= 0
    }

    // Return true for negative values.
    fn is_negative(self) -> bool {
        (self.0 as i64) < 0
    }

    // Add (2^s)*rhs to self.
    fn set_add_shifted(&mut self, rhs: &Self, s: u32) {
        if s < 64 {
            self.0 = self.0.wrapping_add(rhs.0 << s);
        }
    }

    // Subtract (2^s)*rhs from self.
    fn set_sub_shifted(&mut self, rhs: &Self, s: u32) {
        if s < 64 {
            self.0 = self.0.wrapping_sub(rhs.0 << s);
        }
    }
}
*/

macro_rules! define_bigint { ($typename:ident, $bitlen:expr) => {

    #[derive(Clone, Copy, Debug)]
    struct $typename([u64; $typename::N]);

    #[allow(dead_code)]
    impl $typename {
        const BITLEN: usize = $bitlen;
        const N: usize = (Self::BITLEN + 63) >> 6;
        const ZERO: Self = Self([0u64; Self::N]);

        // Return true iff self < rhs (inputs must be nonnegative).
        fn lt(self, rhs: &Self) -> bool {
            let (_, mut cc) = subborrow_u64(self.0[0], rhs.0[0], 0);
            for i in 1..Self::N {
                (_, cc) = subborrow_u64(self.0[i], rhs.0[i], cc);
            }
            cc != 0
        }

        // Swap the contents of self with rhs.
        fn swap(&mut self, rhs: &mut Self) {
            for i in 0..Self::N {
                let t = self.0[i];
                self.0[i] = rhs.0[i];
                rhs.0[i] = t;
            }
        }

        // Get the length (in bits) of this value. If `unsigned` is true,
        // then the value is considered unsigned; otherwise, the top bit
        // is a sign bit.
        fn bitlength(self, unsigned: bool) -> u32 {
            let m = if unsigned { 0 } else { sgnw(self.0[Self::N - 1]) };
            for i in (0..Self::N).rev() {
                let aw = self.0[i] ^ m;
                if aw != 0 {
                    return 64 * (i as u32) + 64 - aw.leading_zeros();
                }
            }
            0
        }

        // Return true if self is lower than 2^(64*s - 1). The value self
        // MUST be non-negative. The value s MUST be greater than 0, and
        // not greater than Self::N.
        fn ltnw(self, s: usize) -> bool {
            for i in s..Self::N {
                if self.0[i] != 0 {
                    return false;
                }
            }
            self.0[s - 1] < 0x8000000000000000
        }

        // Return true for negative values.
        fn is_negative(self) -> bool {
            self.0[Self::N - 1] >= 0x8000000000000000
        }

        // Add (2^s)*rhs to self.
        fn set_add_shifted(&mut self, rhs: &Self, s: u32) {
            if s < 64 {
                if s == 0 {
                    let (d0, mut cc) = addcarry_u64(self.0[0], rhs.0[0], 0);
                    self.0[0] = d0;
                    for i in 1..Self::N {
                        let (dx, ee) = addcarry_u64(self.0[i], rhs.0[i], cc);
                        self.0[i] = dx;
                        cc = ee;
                    }
                } else {
                    let (d0, mut cc) = addcarry_u64(
                        self.0[0], rhs.0[0] << s, 0);
                    self.0[0] = d0;
                    for i in 1..Self::N {
                        let bw = (rhs.0[i - 1] >> (64 - s)) | (rhs.0[i] << s);
                        let (dx, ee) = addcarry_u64(self.0[i], bw, cc);
                        self.0[i] = dx;
                        cc = ee;
                    }
                }
            } else {
                let j = (s >> 6) as usize;
                if j >= Self::N {
                    return;
                }
                let s = s & 63;
                if s == 0 {
                    let (dj, mut cc) = addcarry_u64(self.0[j], rhs.0[0], 0);
                    self.0[j] = dj;
                    for i in (j + 1)..Self::N {
                        let (dx, ee) = addcarry_u64(
                            self.0[i], rhs.0[i - j], cc);
                        self.0[i] = dx;
                        cc = ee;
                    }
                } else {
                    let (dj, mut cc) = addcarry_u64(
                        self.0[j], rhs.0[0] << s, 0);
                    self.0[j] = dj;
                    for i in (j + 1)..Self::N {
                        let bw = (rhs.0[i - j - 1] >> (64 - s))
                            | (rhs.0[i - j] << s);
                        let (dx, ee) = addcarry_u64(self.0[i], bw, cc);
                        self.0[i] = dx;
                        cc = ee;
                    }
                }
            }
        }

        // Subtract (2^s)*rhs from self.
        fn set_sub_shifted(&mut self, rhs: &Self, s: u32) {
            if s < 64 {
                if s == 0 {
                    let (d0, mut cc) = subborrow_u64(self.0[0], rhs.0[0], 0);
                    self.0[0] = d0;
                    for i in 1..Self::N {
                        let (dx, ee) = subborrow_u64(self.0[i], rhs.0[i], cc);
                        self.0[i] = dx;
                        cc = ee;
                    }
                } else {
                    let (d0, mut cc) = subborrow_u64(
                        self.0[0], rhs.0[0] << s, 0);
                    self.0[0] = d0;
                    for i in 1..Self::N {
                        let bw = (rhs.0[i - 1] >> (64 - s)) | (rhs.0[i] << s);
                        let (dx, ee) = subborrow_u64(self.0[i], bw, cc);
                        self.0[i] = dx;
                        cc = ee;
                    }
                }
            } else {
                let j = (s >> 6) as usize;
                if j >= Self::N {
                    return;
                }
                let s = s & 63;
                if s == 0 {
                    let (dj, mut cc) = subborrow_u64(self.0[j], rhs.0[0], 0);
                    self.0[j] = dj;
                    for i in (j + 1)..Self::N {
                        let (dx, ee) = subborrow_u64(
                            self.0[i], rhs.0[i - j], cc);
                        self.0[i] = dx;
                        cc = ee;
                    }
                } else {
                    let (dj, mut cc) = subborrow_u64(
                        self.0[j], rhs.0[0] << s, 0);
                    self.0[j] = dj;
                    for i in (j + 1)..Self::N {
                        let bw = (rhs.0[i - j - 1] >> (64 - s))
                            | (rhs.0[i - j] << s);
                        let (dx, ee) = subborrow_u64(self.0[i],bw, cc);
                        self.0[i] = dx;
                        cc = ee;
                    }
                }
            }
        }
    }

} } // End of macro: define_bigint

macro_rules! define_lagrange { ($name:ident, $n0:ident, $n1:ident, $n2:ident, $n3:ident) => {

    #[allow(dead_code)]
    pub(crate) fn $name(k: &[u64; $n1::N], n: &[u64; $n1::N], max_bitlen: u32)
        -> ([u64; $n0::N], [u64; $n0::N])
    {
        // Product of integers. Operands must be non-negative.
        fn umul(a: &[u64; $n1::N], b: &[u64; $n1::N]) -> $n3 {
            let mut d = $n3::ZERO;
            for i in 0..$n1::N {
                let (lo, mut cc) = umull_add(a[i], b[0], d.0[i]);
                d.0[i] = lo;
                for j in 1..$n1::N {
                    if (i + j) >= $n3::N {
                        break;
                    }
                    let (lo, hi) = umull_add2(a[i], b[j], d.0[i + j], cc);
                    d.0[i + j] = lo;
                    cc = hi;
                }
                if (i + $n1::N) < $n3::N {
                    d.0[i + $n1::N] = cc;
                }
            }
            d
        }

        // Initialization.
        // Coordinates of u and v are truncated (type $n0) since after
        // reduction, they should fit. Values nu (norm of u), nv (norm of v)
        // and sp (scalar product of u and v) are full-size.

        // u <- [n, 0]
        let mut u0 = $n0::ZERO;
        u0.0[..].copy_from_slice(&n[..$n0::N]);
        let mut u1 = $n0::ZERO;

        // v <- [k, 1]
        let mut v0 = $n0::ZERO;
        v0.0[..].copy_from_slice(&k[..$n0::N]);
        let mut v1 = $n0::ZERO;
        v1.0[0] = 1;

        // nu = u0^2 + u1^2 = n^2
        let mut nu = umul(n, n);

        // nv = v0^2 + v1^2 = k^2 + 1
        let mut nv = umul(k, k);
        let (dx, mut cc) = addcarry_u64(nv.0[0], 1, 0);
        nv.0[0] = dx;
        for i in 1..$n3::N {
            let (dx, ee) = addcarry_u64(nv.0[i], 0, cc);
            nv.0[i] = dx;
            cc = ee;
        }

        // sp = u0*v0 + u1*v1 = n*k
        let mut sp = umul(n, k);

        // We use a flag to indicate the first iteration, because at that
        // iteration, sp might lack a sign bit (it's 0, due to initial
        // conditions, but the unsigned value might fill the complete type).
        // After the first iteration, sp is necessarily lower than n/2 and
        // there is room for the sign bit.
        let mut first = true;

        // First algorithm loop, to shrink values enough to fit in type $n2.
        loop {
            // If u is smaller than v, then swap u and v.
            if nu.lt(&nv) {
                u0.swap(&mut v0);
                u1.swap(&mut v1);
                nu.swap(&mut nv);
            }

            // If nu has shrunk enough, then we can switch to the
            // second loop (since v is smaller than u at this point).
            if nu.ltnw($n2::N) {
                break;
            }

            // If v is small enough, return it.
            let bl_nv = nv.bitlength(true);
            if bl_nv <= max_bitlen {
                return (v0.0, v1.0);
            }

            // Compute this amount s = len(sp) - len(nv)
            // (if s < 0, it is replaced with 0).
            let bl_sp = sp.bitlength(first);
            let mut s = bl_sp.wrapping_sub(bl_nv);
            s &= !(((s as i32) >> 31) as u32);

            // Subtract or add v*2^s from/to u, depending on the sign of sp.
            if first || !sp.is_negative() {
                first = false;
                u0.set_sub_shifted(&v0, s);
                u1.set_sub_shifted(&v1, s);
                nu.set_add_shifted(&nv, 2 * s);
                nu.set_sub_shifted(&sp, s + 1);
                sp.set_sub_shifted(&nv, s);
            } else {
                u0.set_add_shifted(&v0, s);
                u1.set_add_shifted(&v1, s);
                nu.set_add_shifted(&nv, 2 * s);
                nu.set_add_shifted(&sp, s + 1);
                sp.set_add_shifted(&nv, s);
            }
        }

        // Shrink nu, nv and sp to the shorter size of $n2
        let mut new_nu = $n2::ZERO;
        let mut new_nv = $n2::ZERO;
        let mut new_sp = $n2::ZERO;
        new_nu.0[..].copy_from_slice(&nu.0[..$n2::N]);
        new_nv.0[..].copy_from_slice(&nv.0[..$n2::N]);
        new_sp.0[..].copy_from_slice(&sp.0[..$n2::N]);
        let mut nu = new_nu;
        let mut nv = new_nv;
        let mut sp = new_sp;

        // In the secondary loop, we need to check for the end condition,
        // which can be a "stuck" value of sp.
        let mut last_bl_sp = sp.bitlength(first);
        let mut stuck = 0u32;

        // Second algorithm loop, once values have shrunk enough to fit in $n2.
        loop {
            // If u is smaller than v, then swap u and v.
            if nu.lt(&nv) {
                u0.swap(&mut v0);
                u1.swap(&mut v1);
                nu.swap(&mut nv);
            }

            // If v is small enough, return it.
            let bl_nv = nv.bitlength(true);
            if bl_nv <= max_bitlen {
                return (v0.0, v1.0);
            }

            // sp normally decreases by 1 bit every two iterations. If it
            // appears to be "stuck" for too long, then this means that
            // we have reached the end of the algorithm, which implies that
            // the target bit length for nv, tested above, was not reached;
            // this means that the function was parameterized too eagerly.
            // It is up to the caller to handle all possible cases (some
            // callers can be made to tolerate truncated (v0,v1)).
            let bl_sp = sp.bitlength(first);
            if bl_sp >= last_bl_sp {
                stuck += 1;
                if bl_sp > last_bl_sp || stuck > 3 {
                    return (v0.0, v1.0);
                }
            } else {
                last_bl_sp = bl_sp;
                stuck = 0;
            }

            // s = len(sp) - len(nv)
            // (if s < 0, it is replaced with 0).
            let mut s = bl_sp.wrapping_sub(bl_nv);
            s &= !(((s as i32) >> 31) as u32);

            // Subtract or add v*2^s from/to u, depending on the sign of sp.
            if first || !sp.is_negative() {
                first = false;
                u0.set_sub_shifted(&v0, s);
                u1.set_sub_shifted(&v1, s);
                nu.set_add_shifted(&nv, 2 * s);
                nu.set_sub_shifted(&sp, s + 1);
                sp.set_sub_shifted(&nv, s);
            } else {
                u0.set_add_shifted(&v0, s);
                u1.set_add_shifted(&v1, s);
                nu.set_add_shifted(&nv, 2 * s);
                nu.set_add_shifted(&sp, s + 1);
                sp.set_add_shifted(&nv, s);
            }
        }
    }

} } // End of macro: define_lagrange

define_bigint!(ZInt128, 128);
define_bigint!(ZInt192, 192);
define_bigint!(ZInt256, 256);
define_bigint!(ZInt320, 320);
define_bigint!(ZInt384, 384);
define_bigint!(ZInt448, 448);
define_bigint!(ZInt512, 512);
define_bigint!(ZInt640, 640);
define_bigint!(ZInt768, 768);
define_bigint!(ZInt896, 896);
define_bigint!(ZInt1024, 1024);

define_lagrange!(lagrange256_vartime, ZInt128, ZInt256, ZInt384, ZInt512);
define_lagrange!(lagrange320_vartime, ZInt192, ZInt320, ZInt448, ZInt640);
define_lagrange!(lagrange384_vartime, ZInt192, ZInt384, ZInt512, ZInt768);
define_lagrange!(lagrange448_vartime, ZInt256, ZInt448, ZInt640, ZInt896);
define_lagrange!(lagrange512_vartime, ZInt256, ZInt512, ZInt768, ZInt1024);

//
// Rules:
//   k and n must have the same length, which is between 4 and 8 (inclusive)
//   k and n use unsigned little-endian notation
//   k < n (numerically)
//   c0 and c1 must have length at most ceil(n.len()/2)
// Processing ends when the minimal-size vector has been found, or when
// a vector v such that ||v||^2 < 2^max_bitlen has been found, whichever
// comes first.
// If the minimal-size vector does not fit in (c0,c1) then it is truncated.
// c0 and c1 use _signed_ little-endian notation.
#[allow(dead_code)]
pub(crate) fn lagrange_vartime(k: &[u64], n: &[u64], max_bitlen: u32,
    c0: &mut [u64], c1: &mut [u64])
{
    match n.len() {
        4 => {
            let (v0, v1) = lagrange256_vartime(
                <&[u64; 4]>::try_from(k).unwrap(),
                <&[u64; 4]>::try_from(n).unwrap(),
                max_bitlen);
            c0.copy_from_slice(&v0[..c0.len()]);
            c1.copy_from_slice(&v1[..c1.len()]);
        }
        5 => {
            let (v0, v1) = lagrange320_vartime(
                <&[u64; 5]>::try_from(k).unwrap(),
                <&[u64; 5]>::try_from(n).unwrap(),
                max_bitlen);
            c0.copy_from_slice(&v0[..c0.len()]);
            c1.copy_from_slice(&v1[..c1.len()]);
        }
        6 => {
            let (v0, v1) = lagrange384_vartime(
                <&[u64; 6]>::try_from(k).unwrap(),
                <&[u64; 6]>::try_from(n).unwrap(),
                max_bitlen);
            c0.copy_from_slice(&v0[..c0.len()]);
            c1.copy_from_slice(&v1[..c1.len()]);
        }
        7 => {
            let (v0, v1) = lagrange448_vartime(
                <&[u64; 7]>::try_from(k).unwrap(),
                <&[u64; 7]>::try_from(n).unwrap(),
                max_bitlen);
            c0.copy_from_slice(&v0[..c0.len()]);
            c1.copy_from_slice(&v1[..c1.len()]);
        }
        8 => {
            let (v0, v1) = lagrange512_vartime(
                <&[u64; 8]>::try_from(k).unwrap(),
                <&[u64; 8]>::try_from(n).unwrap(),
                max_bitlen);
            c0.copy_from_slice(&v0[..c0.len()]);
            c1.copy_from_slice(&v1[..c1.len()]);
        }
        _ => {
            unimplemented!();
        }
    }
}

macro_rules! define_lagrange_spec { ($name:ident, $n0:ident, $n1:ident, $n3:ident) => {

    #[allow(dead_code)]
    pub(crate) fn $name(
        a0: &[u64; $n1::N], a1: &[u64; $n1::N],
        b0: &[u64; $n1::N], b1: &[u64; $n1::N])
        -> ([u64; $n0::N], [u64; $n0::N], u32)
    {
        // Product of _signed_ integers.
        // Requirement: $n3::N <= 2*$n1::N
        fn smul(a: &[u64; $n1::N], b: &[u64; $n1::N]) -> $n3 {
            let mut d = $n3::ZERO;

            // Unsigned product.
            for i in 0..$n1::N {
                let (lo, mut cc) = umull_add(a[i], b[0], d.0[i]);
                d.0[i] = lo;
                for j in 1..$n1::N {
                    if (i + j) >= $n3::N {
                        break;
                    }
                    let (lo, hi) = umull_add2(a[i], b[j], d.0[i + j], cc);
                    d.0[i + j] = lo;
                    cc = hi;
                }
                if (i + $n1::N) < $n3::N {
                    d.0[i + $n1::N] = cc;
                }
            }

            // Adjustment for negative inputs.
            // If a < 0 then we must subtract b*2^(64*$n1::N).
            // If b < 0 then we must subtract a*2^(64*$n1::N).
            let sa = sgnw(a[$n1::N - 1]);
            let sb = sgnw(b[$n1::N - 1]);
            let mut cc = 0;
            for i in 0..($n3::N - $n1::N) {
                (d.0[i + $n1::N], cc) = subborrow_u64(
                    d.0[i + $n1::N], b[i] & sa, cc);
            }
            cc = 0;
            for i in 0..($n3::N - $n1::N) {
                (d.0[i + $n1::N], cc) = subborrow_u64(
                    d.0[i + $n1::N], a[i] & sb, cc);
            }

            d
        }

        // Initialization.
        // Coordinates of u and v are truncated (type $n0) since after
        // reduction, they should fit. Values nu (norm of u), nv (norm of v)
        // and sp (scalar product of u and v) are full-size.

        // u <- [a0, a1]
        // We only keep the second coordinate.
        let mut u1 = $n0::ZERO;
        u1.0[..].copy_from_slice(&a1[..$n0::N]);

        // v <- [b0, b1]
        // We only keep the second coordinate.
        let mut v1 = $n0::ZERO;
        v1.0[..].copy_from_slice(&b1[..$n0::N]);

        // nu = u0^2 + u1^2 = a0^2 + a1^2
        let mut nu = smul(a0, a0);
        nu.set_add_shifted(&smul(a1, a1), 0);

        // nv = v0^2 + v1^2 = b0^2 + b1^2
        let mut nv = smul(b0, b0);
        nv.set_add_shifted(&smul(b1, b1), 0);

        // sp = u0*v0 + u1*v1 = a0*b0 + a1*b1
        let mut sp = smul(a0, b0);
        sp.set_add_shifted(&smul(a1, b1), 0);

        // Algorithm loop.
        loop {
            // If u is smaller than v, then swap u and v.
            if nu.lt(&nv) {
                u1.swap(&mut v1);
                nu.swap(&mut nv);
            }

            // If 2*|sp| <= N_v, then the basis is size-reduced and we
            // can return.
            let bl_nv = nv.bitlength(true);
            let bl_sp = sp.bitlength(false);
            if bl_sp < bl_nv {
                let mut x = nv;
                if sp.is_negative() {
                    x.set_add_shifted(&sp, 1);
                } else {
                    x.set_sub_shifted(&sp, 1);
                }
                if !x.is_negative() {
                    return (v1.0, u1.0, nu.bitlength(true));
                }
            }

            // s = len(sp) - len(nv)
            // (if s < 0, it is replaced with 0).
            let mut s = bl_sp.wrapping_sub(bl_nv);
            s &= !(((s as i32) >> 31) as u32);

            // Subtract or add v*2^s from/to u, depending on the sign of sp.
            if !sp.is_negative() {
                u1.set_sub_shifted(&v1, s);
                nu.set_add_shifted(&nv, 2 * s);
                nu.set_sub_shifted(&sp, s + 1);
                sp.set_sub_shifted(&nv, s);
            } else {
                u1.set_add_shifted(&v1, s);
                nu.set_add_shifted(&nv, 2 * s);
                nu.set_add_shifted(&sp, s + 1);
                sp.set_add_shifted(&nv, s);
            }
        }
    }

} } // End of macro: define_lagrange_spec

define_lagrange_spec!(lagrange128_spec_vartime, ZInt128, ZInt128, ZInt256);
define_lagrange_spec!(lagrange192_spec_vartime, ZInt128, ZInt192, ZInt384);

// Given two unsigned integers a and b, with b >= 1 and a <= b < 2^127,
// reduce the lattice basis [[a, 1], [b, 0]] into [u, v] with:
//   u = e0*[a, 1] + e1*[b, 0]
//   v = f0*[a, 1] + f1*[b, 0]
// [u, v] is size-reduced, and u is lower than v. Returned values are
// (e0, e1, f0, f1, bl_nv), with bl_nv being the bit length of the squared
// norm of v. Values e0, e1, f0 or f1 may be truncated to fit their size;
// However, if bl_nv <= 124, then e0, e1, f0 and f1 are necessarily entire
// (not truncated).
//
// Values a and b are provided as two 64-bit words each (little-endian order).
pub(crate) fn lagrange128_basisconv_vartime(a: &[u64; 2], b: &[u64; 2])
    -> (i64, i64, i64, i64, u32)
{
    // Product of two 128-bit integers.
    fn umul(a: &[u64; 2], b: &[u64; 2]) -> ZInt256 {
        let a0 = a[0];
        let a1 = a[1];
        let b0 = b[0];
        let b1 = b[1];

        let (d0, d1) = umull(a0, b0);
        let (d1, d2) = umull_add(a0, b1, d1);
        let (d1, e2) = umull_add(a1, b0, d1);
        let (d2, d3) = umull_add2(a1, b1, d2, e2);

        ZInt256([d0, d1, d2, d3])
    }

    let mut e0 = 1i64;
    let mut e1 = 0i64;
    let mut f0 = 0i64;
    let mut f1 = 1i64;

    // nu = a^2 + 1
    let mut nu = umul(a, a);
    let mut cc;
    (nu.0[0], cc) = addcarry_u64(nu.0[0], 1, 0);
    (nu.0[1], cc) = addcarry_u64(nu.0[1], 0, cc);
    (nu.0[2], cc) = addcarry_u64(nu.0[2], 0, cc);
    (nu.0[3], _)  = addcarry_u64(nu.0[3], 0, cc);

    // nv = b^2
    let mut nv = umul(b, b);

    // sp = a*b
    let mut sp = umul(a, b);

    // Algorithm loop.
    loop {
        // If u is smaller than v, then swap u and v.
        if nu.lt(&nv) {
            (e0, e1, f0, f1) = (f0, f1, e0, e1);
            nu.swap(&mut nv);
        }

        // If 2*|sp| <= N_v, then the basis is size-reduced and we
        // can return.
        let bl_nv = nv.bitlength(true);
        let bl_sp = sp.bitlength(false);
        if bl_sp < bl_nv {
            let mut x = nv;
            if sp.is_negative() {
                x.set_add_shifted(&sp, 1);
            } else {
                x.set_sub_shifted(&sp, 1);
            }
            if !x.is_negative() {
                return (f0, f1, e0, e1, nu.bitlength(true));
            }
        }

        // s = len(sp) - len(nv)
        // (if s < 0, it is replaced with 0).
        let mut s = bl_sp.wrapping_sub(bl_nv);
        s &= !(((s as i32) >> 31) as u32);

        // Subtract or add v*2^s from/to u, depending on the sign of sp.
        if !sp.is_negative() {
            if s <= 63 {
                e0 = e0.wrapping_sub(f0 << s);
                e1 = e1.wrapping_sub(f1 << s);
            }
            nu.set_add_shifted(&nv, 2 * s);
            nu.set_sub_shifted(&sp, s + 1);
            sp.set_sub_shifted(&nv, s);
        } else {
            if s <= 63 {
                e0 = e0.wrapping_add(f0 << s);
                e1 = e1.wrapping_add(f1 << s);
            }
            nu.set_add_shifted(&nv, 2 * s);
            nu.set_add_shifted(&sp, s + 1);
            sp.set_add_shifted(&nv, s);
        }
    }
}
