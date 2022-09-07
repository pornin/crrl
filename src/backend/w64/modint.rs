use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::convert::TryFrom;

use super::{addcarry_u64, subborrow_u64, umull, umull_add, umull_add2, umull_x2, umull_x2_add, sgnw, lzcnt};
use super::lagrange::lagrange253_vartime;

#[derive(Clone, Copy, Debug)]
pub struct ModInt256<const M0: u64, const M1: u64, const M2: u64, const M3: u64>([u64; 4]);

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64> ModInt256<M0, M1, M2, M3> {

    // Modulus must be odd.
    // Top modulus word must not be zero (i.e. the modulus size must be at
    // least 193 bits).
    // If the modulus is not prime, then square root computations are
    // invalid. We cannot easily test primality at compile-time; moreover,
    // we want to be able to support a non-prime modulus.
    #[allow(dead_code)]
    const COMPILE_TIME_CHECKS: () = Self::compile_time_checks();
    const fn compile_time_checks() {
        static_assert!((M0 & 1) != 0);
        static_assert!(M3 != 0);
    }

    // Modulus, in base 2^64 (low-to-high order).
    pub const MODULUS: [u64; 4] = [ M0, M1, M2, M3 ];

    // floor(q / 4) + 1 (equal to (q+1)/4 if q = 3 mod 8).
    const QP1D4: [u64; 4] = Self::make_qp1d4();

    // floor(q / 8) (equal to (q-5)/8 if q = 5 mod 8).
    const QM5D8: [u64; 4] = Self::make_qm5d8();

    pub const ZERO: ModInt256<M0, M1, M2, M3> =
        ModInt256::<M0, M1, M2, M3>([ 0, 0, 0, 0 ]);
    pub const ONE: ModInt256<M0, M1, M2, M3> =
        ModInt256::<M0, M1, M2, M3>::w64le(1, 0, 0, 0);
    pub const MINUS_ONE: ModInt256<M0, M1, M2, M3> =
        ModInt256::<M0, M1, M2, M3>::w64le(M0 - 1, M1, M2, M3);

    const M0I: u64 = Self::make_m0i(M0);
    const HMP1: Self = Self::make_hmp1();
    const R2: Self = Self::make_r2();
    const T770: Self = Self::make_t770();
    const T64: Self = Self::w64le(0, 1, 0, 0);
    const T128: Self = Self::w64le(0, 0, 1, 0);

    // Create an element from its four 64-bit limbs. The limbs are
    // provided in little-endian order (least significant limb first).
    // This function computes the appropriate internal representation.
    // This function can be used in constant expressions (constant-time
    // evaluation). It is also safe to use at runtime, but from_w64le()
    // provides the same result and is potentially faster.
    pub const fn w64le(x0: u64, x1: u64, x2: u64, x3: u64) -> Self {
        Self::const_mmul(Self([ x0, x1, x2, x3 ]), Self::R2)
    }

    // Create an element from its four 64-bit limbs. The limbs are
    // provided in big-endian order (most significant limb first). This
    // function computes the appropriate internal representation.
    // This function can be used in constant expressions (constant-time
    // evaluation). It is also safe to use at runtime, but from_w64le()
    // provides the same result and is potentially faster.
    pub const fn w64be(x3: u64, x2: u64, x1: u64, x0: u64) -> Self {
        Self::const_mmul(Self([ x0, x1, x2, x3 ]), Self::R2)
    }

    // Create an element from its four 64-bit limbs. The limbs are
    // provided in little-endian order (least significant limb first).
    // This function computes the appropriate internal representation.
    // It is (potentially) faster than w64le(), but it can be only used
    // at runtime, not in const expressions.
    #[inline(always)]
    pub fn from_w64le(x0: u64, x1: u64, x2: u64, x3: u64) -> Self {
        let mut r = Self([ x0, x1, x2, x3 ]);
        r.set_mul(&Self::R2);
        r
    }

    // Create an element from its four 64-bit limbs. The limbs are
    // provided in big-endian order (most significant limb first).
    // This function computes the appropriate internal representation.
    // It is (potentially) faster than w64be(), but it can be only used
    // at runtime, not in const expressions.
    #[inline(always)]
    pub fn from_w64be(x3: u64, x2: u64, x1: u64, x0: u64) -> Self {
        let mut r = Self([ x0, x1, x2, x3 ]);
        r.set_mul(&Self::R2);
        r
    }

    // Create an element by converting the provided integer.
    // If the source value is negative, then it is implicitly reduced
    // modulo the ring order.
    #[inline(always)]
    pub fn from_i32(x: i32) -> Self {
        let mut r = Self::from_w64le(x as u64, 0, 0, 0);
        r.set_cond(&(r - Self::T64), (x >> 31) as u32);
        r
    }

    // Create an element by converting the provided integer.
    #[inline(always)]
    pub fn from_u32(x: u32) -> Self {
        Self::from_w64le(x as u64, 0, 0, 0)
    }

    // Create an element by converting the provided integer.
    // If the source value is negative, then it is implicitly reduced
    // modulo the ring order.
    #[inline(always)]
    pub fn from_i64(x: i64) -> Self {
        let mut r = Self::from_w64le(x as u64, 0, 0, 0);
        r.set_cond(&(r - Self::T64), (x >> 63) as u32);
        r
    }

    // Create an element by converting the provided integer.
    #[inline(always)]
    pub fn from_u64(x: u64) -> Self {
        Self::from_w64le(x, 0, 0, 0)
    }

    // Create an element by converting the provided integer.
    // If the source value is negative, then it is implicitly reduced
    // modulo the ring order.
    #[inline(always)]
    pub fn from_i128(x: i128) -> Self {
        let mut r = Self::from_w64le(x as u64, (x >> 64) as u64, 0, 0);
        r.set_cond(&(r - Self::T128), (x >> 127) as u32);
        r
    }

    // Create an element by converting the provided integer.
    #[inline(always)]
    pub fn from_u128(x: u128) -> Self {
        Self::from_w64le(x as u64, (x >> 64) as u64, 0, 0)
    }

    #[inline(always)]
    fn set_add(&mut self, rhs: &Self) {
        // If we have room for the carry in the top word, we can use a
        // specialized function.
        if M3 < 0x7FFFFFFFFFFFFFFF {
            // Carry fits in top limb.

            // Raw addition.
            let (d0, cc) = addcarry_u64(self.0[0], rhs.0[0], 0);
            let (d1, cc) = addcarry_u64(self.0[1], rhs.0[1], cc);
            let (d2, cc) = addcarry_u64(self.0[2], rhs.0[2], cc);
            let (d3, _)  = addcarry_u64(self.0[3], rhs.0[3], cc);

            // Subtract the modulus.
            let (e0, cc) = subborrow_u64(d0, M0, 0);
            let (e1, cc) = subborrow_u64(d1, M1, cc);
            let (e2, cc) = subborrow_u64(d2, M2, cc);
            let (e3, cc) = subborrow_u64(d3, M3, cc);

            // Add back the modulus in case the result was negative.
            let w = (cc as u64).wrapping_neg();
            let (d0, cc) = addcarry_u64(e0, w & M0, 0);
            let (d1, cc) = addcarry_u64(e1, w & M1, cc);
            let (d2, cc) = addcarry_u64(e2, w & M2, cc);
            let (d3, _)  = addcarry_u64(e3, w & M3, cc);

            self.0[0] = d0;
            self.0[1] = d1;
            self.0[2] = d2;
            self.0[3] = d3;

        } else {
            // Carry does not fit in top limb, we use an extra word.

            // Raw addition; final carry in d4.
            let (d0, cc) = addcarry_u64(self.0[0], rhs.0[0], 0);
            let (d1, cc) = addcarry_u64(self.0[1], rhs.0[1], cc);
            let (d2, cc) = addcarry_u64(self.0[2], rhs.0[2], cc);
            let (d3, cc) = addcarry_u64(self.0[3], rhs.0[3], cc);
            let d4 = cc as u64;

            // Subtract the modulus.
            let (e0, cc) = subborrow_u64(d0, M0, 0);
            let (e1, cc) = subborrow_u64(d1, M1, cc);
            let (e2, cc) = subborrow_u64(d2, M2, cc);
            let (e3, cc) = subborrow_u64(d3, M3, cc);
            let (e4, _)  = subborrow_u64(d4, 0, cc);

            // Add back the modulus in case the result was negative.
            let (d0, cc) = addcarry_u64(e0, e4 & M0, 0);
            let (d1, cc) = addcarry_u64(e1, e4 & M1, cc);
            let (d2, cc) = addcarry_u64(e2, e4 & M2, cc);
            let (d3, _)  = addcarry_u64(e3, e4 & M3, cc);

            self.0[0] = d0;
            self.0[1] = d1;
            self.0[2] = d2;
            self.0[3] = d3;
        }
    }

    #[inline(always)]
    fn set_sub(&mut self, rhs: &Self) {
        // Raw subtraction.
        let (d0, cc) = subborrow_u64(self.0[0], rhs.0[0], 0);
        let (d1, cc) = subborrow_u64(self.0[1], rhs.0[1], cc);
        let (d2, cc) = subborrow_u64(self.0[2], rhs.0[2], cc);
        let (d3, cc) = subborrow_u64(self.0[3], rhs.0[3], cc);

        // Add back the modulus if there was a borrow.
        let w = (cc as u64).wrapping_neg();
        let (d0, cc) = addcarry_u64(d0, w & M0, 0);
        let (d1, cc) = addcarry_u64(d1, w & M1, cc);
        let (d2, cc) = addcarry_u64(d2, w & M2, cc);
        let (d3, _)  = addcarry_u64(d3, w & M3, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    // Negate this value (in place).
    #[inline(always)]
    pub fn set_neg(&mut self) {
        let (d0, cc) = subborrow_u64(0, self.0[0], 0);
        let (d1, cc) = subborrow_u64(0, self.0[1], cc);
        let (d2, cc) = subborrow_u64(0, self.0[2], cc);
        let (d3, cc) = subborrow_u64(0, self.0[3], cc);

        // Add back the modulus if there was a borrow.
        let w = (cc as u64).wrapping_neg();
        let (d0, cc) = addcarry_u64(d0, w & M0, 0);
        let (d1, cc) = addcarry_u64(d1, w & M1, cc);
        let (d2, cc) = addcarry_u64(d2, w & M2, cc);
        let (d3, _)  = addcarry_u64(d3, w & M3, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    // Conditionally copy the provided value ('a') into self:
    //  - If ctl == 0xFFFFFFFF, then the value of 'a' is copied into self.
    //  - If ctl == 0, then the value of self is unchanged.
    // clt MUST be equal to 0 or 0xFFFFFFFF.
    #[inline(always)]
    pub fn set_cond(&mut self, a: &Self, ctl: u32) {
        let cw = ((ctl as i32) as i64) as u64;
        self.0[0] ^= cw & (self.0[0] ^ a.0[0]);
        self.0[1] ^= cw & (self.0[1] ^ a.0[1]);
        self.0[2] ^= cw & (self.0[2] ^ a.0[2]);
        self.0[3] ^= cw & (self.0[3] ^ a.0[3]);
    }

    // Return a value equal to either a0 (if ctl == 0) or a1 (if
    // ctl == 0xFFFFFFFF). Value ctl MUST be either 0 or 0xFFFFFFFF.
    #[inline(always)]
    pub fn select(a0: &Self, a1: &Self, ctl: u32) -> Self {
        let mut r = *a0;
        r.set_cond(a1, ctl);
        r
    }

    // Conditionally swap two elements: values a and b are exchanged if
    // ctl == 0xFFFFFFFF, or not exchanged if ctl == 0x00000000. Value
    // ctl MUST be either 0x00000000 or 0xFFFFFFFF.
    #[inline(always)]
    pub fn cswap(a: &mut Self, b: &mut Self, ctl: u32) {
        let cw = ((ctl as i32) as i64) as u64;
        let t = cw & (a.0[0] ^ b.0[0]); a.0[0] ^= t; b.0[0] ^= t;
        let t = cw & (a.0[1] ^ b.0[1]); a.0[1] ^= t; b.0[1] ^= t;
        let t = cw & (a.0[2] ^ b.0[2]); a.0[2] ^= t; b.0[2] ^= t;
        let t = cw & (a.0[3] ^ b.0[3]); a.0[3] ^= t; b.0[3] ^= t;
    }

    // Montgomery reduction (division by 2^256). Input must be normalized;
    // output is normalized.
    #[inline(always)]
    fn set_montyred(&mut self) {
        let (d0, d1, d2, d3) = (self.0[0], self.0[1], self.0[2], self.0[3]);

        // At each round:
        //    d <- (d + f*m) / 2^64
        // Since f <= 2^64 - 1, m <= 2^256 - 1 and d <= 2^256 - 1, the
        // new value d' is such that:
        //    d' <= (2^256 - 1 + (2^64 - 1)*(2^256 - 1)) / 2^64
        //       <= 2^256 - 1
        // i.e. the output of each round must also fit on four limbs.

        let f = d0.wrapping_mul(Self::M0I);
        let (_, hi)  = umull_add(f, M0, d0);
        let (d0, hi) = umull_add2(f, M1, d1, hi);
        let (d1, hi) = umull_add2(f, M2, d2, hi);
        let (d2, d3) = umull_add2(f, M3, d3, hi);

        let f = d0.wrapping_mul(Self::M0I);
        let (_, hi)  = umull_add(f, M0, d0);
        let (d0, hi) = umull_add2(f, M1, d1, hi);
        let (d1, hi) = umull_add2(f, M2, d2, hi);
        let (d2, d3) = umull_add2(f, M3, d3, hi);

        let f = d0.wrapping_mul(Self::M0I);
        let (_, hi)  = umull_add(f, M0, d0);
        let (d0, hi) = umull_add2(f, M1, d1, hi);
        let (d1, hi) = umull_add2(f, M2, d2, hi);
        let (d2, d3) = umull_add2(f, M3, d3, hi);

        let f = d0.wrapping_mul(Self::M0I);
        let (_, hi)  = umull_add(f, M0, d0);
        let (d0, hi) = umull_add2(f, M1, d1, hi);
        let (d1, hi) = umull_add2(f, M2, d2, hi);
        let (d2, d3) = umull_add2(f, M3, d3, hi);

        // In total, from the original value x, we computed
        // (x + f*m) / 2^256, for some value f which is lower than 2^256.
        // Since x < m, the result must be such that:
        //    (x + f*m) / 2^256 < (m + (2^256 - 1)*m) / 2^256
        //                      < m
        // Hence, the output is already reduced.
        //
        // Note: if the input was greater not normalized, and ranged up
        // to 2^256 - 1, then an output value _equal_ to m is feasible.
        // This is outside of the allowed range for this function.

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    // Internal multiplication routine. This is a Montgomery multiplication:
    //    self <- (self * rhs) / 2^256 mod m
    // This computes a multiplication as long as operands and result are
    // in Montgomery representation.
    // The right operand (rhs) must be properly normalized on entry (in
    // the 0..m-1 range) but this value can range up to 2^256-1. Output
    // is properly normalized.
    #[inline(always)]
    fn set_mul(&mut self, rhs: &Self) {
        let (a0, a1, a2, a3) = (self.0[0], self.0[1], self.0[2], self.0[3]);
        let (b0, b1, b2, b3) = (rhs.0[0], rhs.0[1], rhs.0[2], rhs.0[3]);

        // We start with d = 0. At each round, we compute:
        //   d <- (d + aj*b + f*m) / 2^64
        // with aj being the next limb of a, and f being set to the proper
        // value that makes the division exact.
        //
        // If at the start of a round we have d <= 2*m-1, then we have:
        //   d + aj*b + f*m <= 2*m - 1 + (2^64 - 1)*(m - 1) + (2^64 - 1)*m
        //                  <= 2*m - 1 + 2^64*m - 2^64 - m + 1 + 2^64*m - m
        //                  <= 2^64*(2*m - 1)
        // Thus, the output is lower than 2*m - 1 as well. This property is
        // therefore maintained through all rounds. A single conditional
        // subtraction at the end normalizes the result.
        //
        // Value ranges (in limbs):
        // ------------------------
        //
        // In the first round, the initial d is zero, and the first
        // computation of d + aj*b fits on five limbs; this is not
        // necessarily the case for the subsequent rounds. The intermediate
        // value d + aj*b may range up to:
        //   d + aj*b <= 2*m - 1 + (2^64 - 1)*m
        //            <= (2^64 + 1)*m - 1
        // Thus, this is guaranteed to fit in five limb if:
        //   m <= floor(2^320 / (2^64 + 1))
        // Maximum allowed m for this property is 2^256 - 2^192 + 2^128 - 2^64.
        // We are in this case if M3 is not 2^64-1, or if M3 is equal to
        // 2^64-1 and M2 is zero and M1 is not 2^64-1.
        //
        // The output of each round will fit on four limbs if m < 2^255,
        // i.e. if the top limb of m is less than 2^63. Otherwise, it may
        // need five limbs.
        //
        // Thus, we have three cases to handle:
        //   1. m < 2^255
        //       d + aj*b fits on five limbs, output d fits on four
        //   2. 2^255 <= m < 2^256 - 2^192 + 2^128 - 2^64
        //       d + aj*b fits on five limbs, output d needs four + carry
        //   3. m >= 2^256 - 2^192 + 2^128 - 2^64
        //       d + aj*b needs five limbs + carry, output d needs four + carry

        if M3 < 0x8000000000000000 {

            // m < 2^255, d fits on four limbs, d + aj*b fits on five limbs

            let (d0, hi) = umull(a0, b0);
            let (d1, hi) = umull_add(a0, b1, hi);
            let (d2, hi) = umull_add(a0, b2, hi);
            let (d3, d4) = umull_add(a0, b3, hi);
            let f = d0.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, d0);
            let (d0, hi) = umull_add2(f, M1, d1, hi);
            let (d1, hi) = umull_add2(f, M2, d2, hi);
            let (d2, hi) = umull_add2(f, M3, d3, hi);
            let d3 = d4.wrapping_add(hi);

            let (d0, hi) = umull_add(a1, b0, d0);
            let (d1, hi) = umull_add2(a1, b1, d1, hi);
            let (d2, hi) = umull_add2(a1, b2, d2, hi);
            let (d3, d4) = umull_add2(a1, b3, d3, hi);
            let f = d0.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, d0);
            let (d0, hi) = umull_add2(f, M1, d1, hi);
            let (d1, hi) = umull_add2(f, M2, d2, hi);
            let (d2, hi) = umull_add2(f, M3, d3, hi);
            let d3 = d4.wrapping_add(hi);

            let (d0, hi) = umull_add(a2, b0, d0);
            let (d1, hi) = umull_add2(a2, b1, d1, hi);
            let (d2, hi) = umull_add2(a2, b2, d2, hi);
            let (d3, d4) = umull_add2(a2, b3, d3, hi);
            let f = d0.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, d0);
            let (d0, hi) = umull_add2(f, M1, d1, hi);
            let (d1, hi) = umull_add2(f, M2, d2, hi);
            let (d2, hi) = umull_add2(f, M3, d3, hi);
            let d3 = d4.wrapping_add(hi);

            let (d0, hi) = umull_add(a3, b0, d0);
            let (d1, hi) = umull_add2(a3, b1, d1, hi);
            let (d2, hi) = umull_add2(a3, b2, d2, hi);
            let (d3, d4) = umull_add2(a3, b3, d3, hi);
            let f = d0.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, d0);
            let (d0, hi) = umull_add2(f, M1, d1, hi);
            let (d1, hi) = umull_add2(f, M2, d2, hi);
            let (d2, hi) = umull_add2(f, M3, d3, hi);
            let d3 = d4.wrapping_add(hi);

            // Subtract m if needed.
            let (_, cc) = subborrow_u64(d0, M0, 0);
            let (_, cc) = subborrow_u64(d1, M1, cc);
            let (_, cc) = subborrow_u64(d2, M2, cc);
            let (_, cc) = subborrow_u64(d3, M3, cc);
            let w = (cc as u64).wrapping_sub(1);
            let (d0, cc) = subborrow_u64(d0, w & M0, 0);
            let (d1, cc) = subborrow_u64(d1, w & M1, cc);
            let (d2, cc) = subborrow_u64(d2, w & M2, cc);
            let (d3, _)  = subborrow_u64(d3, w & M3, cc);

            self.0[0] = d0;
            self.0[1] = d1;
            self.0[2] = d2;
            self.0[3] = d3;

        } else if (M3 < 0xFFFFFFFFFFFFFFFF)
            || (M3 == 0xFFFFFFFFFFFFFFFF && M2 == 0 && M1 < 0xFFFFFFFFFFFFFFFF)
        {
            // m < 2^256 - 2^192 + 2^128 - 2^64
            // d + aj*b fits on five limbs
            // Output d of each round needs four limbs + carry.

            let (d0, hi) = umull(a0, b0);
            let (d1, hi) = umull_add(a0, b1, hi);
            let (d2, hi) = umull_add(a0, b2, hi);
            let (d3, d4) = umull_add(a0, b3, hi);
            let f = d0.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, d0);
            let (d0, hi) = umull_add2(f, M1, d1, hi);
            let (d1, hi) = umull_add2(f, M2, d2, hi);
            let (d2, hi) = umull_add2(f, M3, d3, hi);
            let (d3, d4) = addcarry_u64(d4, hi, 0);

            let (d0, hi) = umull_add(a1, b0, d0);
            let (d1, hi) = umull_add2(a1, b1, d1, hi);
            let (d2, hi) = umull_add2(a1, b2, d2, hi);
            let (d3, hi) = umull_add2(a1, b3, d3, hi);
            let (d4, _)  = addcarry_u64(hi, 0, d4);
            let f = d0.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, d0);
            let (d0, hi) = umull_add2(f, M1, d1, hi);
            let (d1, hi) = umull_add2(f, M2, d2, hi);
            let (d2, hi) = umull_add2(f, M3, d3, hi);
            let (d3, d4) = addcarry_u64(d4, hi, 0);

            let (d0, hi) = umull_add(a2, b0, d0);
            let (d1, hi) = umull_add2(a2, b1, d1, hi);
            let (d2, hi) = umull_add2(a2, b2, d2, hi);
            let (d3, hi) = umull_add2(a2, b3, d3, hi);
            let (d4, _)  = addcarry_u64(hi, 0, d4);
            let f = d0.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, d0);
            let (d0, hi) = umull_add2(f, M1, d1, hi);
            let (d1, hi) = umull_add2(f, M2, d2, hi);
            let (d2, hi) = umull_add2(f, M3, d3, hi);
            let (d3, d4) = addcarry_u64(d4, hi, 0);

            let (d0, hi) = umull_add(a3, b0, d0);
            let (d1, hi) = umull_add2(a3, b1, d1, hi);
            let (d2, hi) = umull_add2(a3, b2, d2, hi);
            let (d3, hi) = umull_add2(a3, b3, d3, hi);
            let (d4, _)  = addcarry_u64(hi, 0, d4);
            let f = d0.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, d0);
            let (d0, hi) = umull_add2(f, M1, d1, hi);
            let (d1, hi) = umull_add2(f, M2, d2, hi);
            let (d2, hi) = umull_add2(f, M3, d3, hi);
            let (d3, d4) = addcarry_u64(d4, hi, 0);

            // Subtract m if needed.
            let (_, cc) = subborrow_u64(d0, M0, 0);
            let (_, cc) = subborrow_u64(d1, M1, cc);
            let (_, cc) = subborrow_u64(d2, M2, cc);
            let (_, cc) = subborrow_u64(d3, M3, cc);
            let w = ((d4 ^ cc) as u64).wrapping_sub(1);
            let (d0, cc) = subborrow_u64(d0, w & M0, 0);
            let (d1, cc) = subborrow_u64(d1, w & M1, cc);
            let (d2, cc) = subborrow_u64(d2, w & M2, cc);
            let (d3, _)  = subborrow_u64(d3, w & M3, cc);

            self.0[0] = d0;
            self.0[1] = d1;
            self.0[2] = d2;
            self.0[3] = d3;

        } else {
            // m > 2^256 - 2^192 + 2^128 - 2^64
            // d + aj*b fits on five limbs + carry (except on first round)
            // Output d of each round needs four limbs + carry

            let (d0, hi) = umull(a0, b0);
            let (d1, hi) = umull_add(a0, b1, hi);
            let (d2, hi) = umull_add(a0, b2, hi);
            let (d3, d4) = umull_add(a0, b3, hi);
            let f = d0.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, d0);
            let (d0, hi) = umull_add2(f, M1, d1, hi);
            let (d1, hi) = umull_add2(f, M2, d2, hi);
            let (d2, hi) = umull_add2(f, M3, d3, hi);
            let (d3, d4) = addcarry_u64(d4, hi, 0);

            let (d0, hi) = umull_add(a1, b0, d0);
            let (d1, hi) = umull_add2(a1, b1, d1, hi);
            let (d2, hi) = umull_add2(a1, b2, d2, hi);
            let (d3, hi) = umull_add2(a1, b3, d3, hi);
            let (d4, d5) = addcarry_u64(hi, 0, d4);
            let f = d0.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, d0);
            let (d0, hi) = umull_add2(f, M1, d1, hi);
            let (d1, hi) = umull_add2(f, M2, d2, hi);
            let (d2, hi) = umull_add2(f, M3, d3, hi);
            let (d3, cc) = addcarry_u64(d4, hi, 0);
            let (d4, _) = addcarry_u64(d5 as u64, 0, cc);

            let (d0, hi) = umull_add(a2, b0, d0);
            let (d1, hi) = umull_add2(a2, b1, d1, hi);
            let (d2, hi) = umull_add2(a2, b2, d2, hi);
            let (d3, hi) = umull_add2(a2, b3, d3, hi);
            let (d4, d5) = addcarry_u64(hi, d4, 0);
            let f = d0.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, d0);
            let (d0, hi) = umull_add2(f, M1, d1, hi);
            let (d1, hi) = umull_add2(f, M2, d2, hi);
            let (d2, hi) = umull_add2(f, M3, d3, hi);
            let (d3, cc) = addcarry_u64(d4, hi, 0);
            let (d4, _) = addcarry_u64(d5 as u64, 0, cc);

            let (d0, hi) = umull_add(a3, b0, d0);
            let (d1, hi) = umull_add2(a3, b1, d1, hi);
            let (d2, hi) = umull_add2(a3, b2, d2, hi);
            let (d3, hi) = umull_add2(a3, b3, d3, hi);
            let (d4, d5) = addcarry_u64(hi, d4, 0);
            let f = d0.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, d0);
            let (d0, hi) = umull_add2(f, M1, d1, hi);
            let (d1, hi) = umull_add2(f, M2, d2, hi);
            let (d2, hi) = umull_add2(f, M3, d3, hi);
            let (d3, cc) = addcarry_u64(d4, hi, 0);
            let (d4, _) = addcarry_u64(d5 as u64, 0, cc);

            // Subtract m if needed.
            let (_, cc) = subborrow_u64(d0, M0, 0);
            let (_, cc) = subborrow_u64(d1, M1, cc);
            let (_, cc) = subborrow_u64(d2, M2, cc);
            let (_, cc) = subborrow_u64(d3, M3, cc);
            let (_, cc) = subborrow_u64(d4, 0, cc);
            let w = (cc as u64).wrapping_sub(1);
            let (d0, cc) = subborrow_u64(d0, w & M0, 0);
            let (d1, cc) = subborrow_u64(d1, w & M1, cc);
            let (d2, cc) = subborrow_u64(d2, w & M2, cc);
            let (d3, _)  = subborrow_u64(d3, w & M3, cc);

            self.0[0] = d0;
            self.0[1] = d1;
            self.0[2] = d2;
            self.0[3] = d3;
        }
    }

    // Square this value (in place).
    #[inline(always)]
    pub fn set_square(&mut self) {
        let (a0, a1, a2, a3) = (self.0[0], self.0[1], self.0[2], self.0[3]);

        // 1. Non-square products. Max intermediate value:
        //   a0*a1            * 2^64
        //   a0*a2            * 2^128
        //   (a0*a3 + a1*a2)  * 2^192
        //   a1*a3            * 2^256
        //   a2*a3            * 2^320
        // for a total which is stlightly below 2^448, which means that
        // the value fits on e1..e6 (no possible carry into e7).
        let (e1, e2) = umull(a0, a1);
        let (e3, e4) = umull(a0, a3);
        let (e5, e6) = umull(a2, a3);
        let (lo, hi) = umull(a0, a2);
        let (e2, cc) = addcarry_u64(e2, lo, 0);
        let (e3, cc) = addcarry_u64(e3, hi, cc);
        let (lo, hi) = umull(a1, a3);
        let (e4, cc) = addcarry_u64(e4, lo, cc);
        let (e5, cc) = addcarry_u64(e5, hi, cc);
        let (e6, _)  = addcarry_u64(e6, 0, cc);
        let (lo, hi) = umull(a1, a2);
        let (e3, cc) = addcarry_u64(e3, lo, 0);
        let (e4, cc) = addcarry_u64(e4, hi, cc);
        let (e5, cc) = addcarry_u64(e5, 0, cc);
        let (e6, _)  = addcarry_u64(e6, 0, cc);

        // 2. Double the intermediate value, then add the squares.
        let e7 = e6 >> 63;
        let e6 = (e6 << 1) | (e5 >> 63);
        let e5 = (e5 << 1) | (e4 >> 63);
        let e4 = (e4 << 1) | (e3 >> 63);
        let e3 = (e3 << 1) | (e2 >> 63);
        let e2 = (e2 << 1) | (e1 >> 63);
        let e1 = e1 << 1;

        let (e0, hi) = umull(a0, a0);
        let (e1, cc) = addcarry_u64(e1, hi, 0);
        let (lo, hi) = umull(a1, a1);
        let (e2, cc) = addcarry_u64(e2, lo, cc);
        let (e3, cc) = addcarry_u64(e3, hi, cc);
        let (lo, hi) = umull(a2, a2);
        let (e4, cc) = addcarry_u64(e4, lo, cc);
        let (e5, cc) = addcarry_u64(e5, hi, cc);
        let (lo, hi) = umull(a3, a3);
        let (e6, cc) = addcarry_u64(e6, lo, cc);
        let (e7, _)  = addcarry_u64(e7, hi, cc);

        // 3. Montgomery reduction.
        // We add f*m to the current value. In the general case, this
        // may require an extra carry bit; however, if m is small enough,
        // then this can be avoided. Maximum input value x is m-1; thus,
        // we compute at most:
        //    (m - 1)^2 + (2^256 - 1)*m
        // This value will fit on 512 bits if m is lower than:
        // 0x9E3779B97F4A7C15F39CC0605CEDC8341082276BF3A27251F86C6A11D0C18E95
        // We round that down (for testing convenience) and select a slightly
        // faster code path if M3 < 0x9E3779B97F4A7C15.
        if M3 < 0x9E3779B97F4A7C15 {

            let f = e0.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, e0);
            let (e1, hi) = umull_add2(f, M1, e1, hi);
            let (e2, hi) = umull_add2(f, M2, e2, hi);
            let (e3, hi) = umull_add2(f, M3, e3, hi);
            let (e4, cc) = addcarry_u64(e4, hi, 0);
            let (e5, cc) = addcarry_u64(e5, 0, cc);
            let (e6, cc) = addcarry_u64(e6, 0, cc);
            let (e7, _)  = addcarry_u64(e7, 0, cc);
            let f = e1.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, e1);
            let (e2, hi) = umull_add2(f, M1, e2, hi);
            let (e3, hi) = umull_add2(f, M2, e3, hi);
            let (e4, hi) = umull_add2(f, M3, e4, hi);
            let (e5, cc) = addcarry_u64(e5, hi, 0);
            let (e6, cc) = addcarry_u64(e6, 0, cc);
            let (e7, _)  = addcarry_u64(e7, 0, cc);
            let f = e2.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, e2);
            let (e3, hi) = umull_add2(f, M1, e3, hi);
            let (e4, hi) = umull_add2(f, M2, e4, hi);
            let (e5, hi) = umull_add2(f, M3, e5, hi);
            let (e6, cc) = addcarry_u64(e6, hi, 0);
            let (e7, _)  = addcarry_u64(e7, 0, cc);
            let f = e3.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, e3);
            let (e4, hi) = umull_add2(f, M1, e4, hi);
            let (e5, hi) = umull_add2(f, M2, e5, hi);
            let (e6, hi) = umull_add2(f, M3, e6, hi);
            let e7 = e7.wrapping_add(hi);

            // 4. Conditional subtraction.
            let (_, cc) = subborrow_u64(e4, M0, 0);
            let (_, cc) = subborrow_u64(e5, M1, cc);
            let (_, cc) = subborrow_u64(e6, M2, cc);
            let (_, cc) = subborrow_u64(e7, M3, cc);
            let w = (cc as u64).wrapping_sub(1);
            let (d0, cc) = subborrow_u64(e4, w & M0, 0);
            let (d1, cc) = subborrow_u64(e5, w & M1, cc);
            let (d2, cc) = subborrow_u64(e6, w & M2, cc);
            let (d3, _)  = subborrow_u64(e7, w & M3, cc);

            self.0[0] = d0;
            self.0[1] = d1;
            self.0[2] = d2;
            self.0[3] = d3;

        } else {

            let f = e0.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, e0);
            let (e1, hi) = umull_add2(f, M1, e1, hi);
            let (e2, hi) = umull_add2(f, M2, e2, hi);
            let (e3, hi) = umull_add2(f, M3, e3, hi);
            let (e4, cc) = addcarry_u64(e4, hi, 0);
            let (e5, cc) = addcarry_u64(e5, 0, cc);
            let (e6, cc) = addcarry_u64(e6, 0, cc);
            let (e7, e8) = addcarry_u64(e7, 0, cc);
            let f = e1.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, e1);
            let (e2, hi) = umull_add2(f, M1, e2, hi);
            let (e3, hi) = umull_add2(f, M2, e3, hi);
            let (e4, hi) = umull_add2(f, M3, e4, hi);
            let (e5, cc) = addcarry_u64(e5, hi, 0);
            let (e6, cc) = addcarry_u64(e6, 0, cc);
            let (e7, cc) = addcarry_u64(e7, 0, cc);
            let (e8, _)  = addcarry_u64(e8 as u64, 0, cc);
            let f = e2.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, e2);
            let (e3, hi) = umull_add2(f, M1, e3, hi);
            let (e4, hi) = umull_add2(f, M2, e4, hi);
            let (e5, hi) = umull_add2(f, M3, e5, hi);
            let (e6, cc) = addcarry_u64(e6, hi, cc);
            let (e7, cc) = addcarry_u64(e7, 0, cc);
            let (e8, _)  = addcarry_u64(e8 as u64, 0, cc);
            let f = e3.wrapping_mul(Self::M0I);
            let (_, hi)  = umull_add(f, M0, e3);
            let (e4, hi) = umull_add2(f, M1, e4, hi);
            let (e5, hi) = umull_add2(f, M2, e5, hi);
            let (e6, hi) = umull_add2(f, M3, e6, hi);
            let (e7, cc) = addcarry_u64(e7, hi, cc);
            let (e8, _)  = addcarry_u64(e8 as u64, 0, cc);

            // 4. Conditional subtraction.
            let (_, cc) = subborrow_u64(e4, M0, 0);
            let (_, cc) = subborrow_u64(e5, M1, cc);
            let (_, cc) = subborrow_u64(e6, M2, cc);
            let (_, cc) = subborrow_u64(e7, M3, cc);
            let (_, cc) = subborrow_u64(e8, 0, cc);
            let w = (cc as u64).wrapping_sub(1);
            let (d0, cc) = subborrow_u64(e4, w & M0, 0);
            let (d1, cc) = subborrow_u64(e5, w & M1, cc);
            let (d2, cc) = subborrow_u64(e6, w & M2, cc);
            let (d3, _)  = subborrow_u64(e7, w & M3, cc);

            self.0[0] = d0;
            self.0[1] = d1;
            self.0[2] = d2;
            self.0[3] = d3;
        }
    }

    // Square this value.
    #[inline(always)]
    pub fn square(self) -> Self {
        let mut r = self;
        r.set_square();
        r
    }

    // Square this value n times (in place).
    #[inline]
    fn set_xsquare(&mut self, n: u32) {
        for _ in 0..n {
            self.set_square();
        }
    }

    // Square this value n times.
    #[inline(always)]
    pub fn xsquare(self, n: u32) -> Self {
        let mut r = self;
        r.set_xsquare(n);
        r
    }

    #[inline(always)]
    fn set_half(&mut self) {
        let (a0, a1, a2, a3) = (self.0[0], self.0[1], self.0[2], self.0[3]);

        let d0 = (a0 >> 1) | (a1 << 63);
        let d1 = (a1 >> 1) | (a2 << 63);
        let d2 = (a2 >> 1) | (a3 << 63);
        let d3 = a3 >> 1;
        let w = (a0 & 1).wrapping_neg();
        let (d0, cc) = addcarry_u64(d0, w & Self::HMP1.0[0], 0);
        let (d1, cc) = addcarry_u64(d1, w & Self::HMP1.0[1], cc);
        let (d2, cc) = addcarry_u64(d2, w & Self::HMP1.0[2], cc);
        let (d3, _)  = addcarry_u64(d3, w & Self::HMP1.0[3], cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    #[inline(always)]
    pub fn half(self) -> Self {
        let mut r = self;
        r.set_half();
        r
    }

    #[inline(always)]
    fn set_mul2(&mut self) {
        let r = *self;
        self.set_add(&r);
    }

    #[inline(always)]
    pub fn mul2(self) -> Self {
        let mut r = self;
        r.set_mul2();
        r
    }

    #[inline(always)]
    fn set_mul3(&mut self) {
        let r = *self;
        self.set_add(&r);
        self.set_add(&r);
    }

    #[inline(always)]
    pub fn mul3(self) -> Self {
        let mut r = self;
        r.set_mul3();
        r
    }

    #[inline(always)]
    fn set_mul4(&mut self) {
        let r = *self;
        self.set_add(&r);
        let r = *self;
        self.set_add(&r);
    }

    #[inline(always)]
    pub fn mul4(self) -> Self {
        let mut r = self;
        r.set_mul4();
        r
    }

    #[inline(always)]
    fn set_mul8(&mut self) {
        let r = *self;
        self.set_add(&r);
        let r = *self;
        self.set_add(&r);
        let r = *self;
        self.set_add(&r);
    }

    #[inline(always)]
    pub fn mul8(self) -> Self {
        let mut r = self;
        r.set_mul8();
        r
    }

    #[inline(always)]
    fn set_mul16(&mut self) {
        let r = *self;
        self.set_add(&r);
        let r = *self;
        self.set_add(&r);
        let r = *self;
        self.set_add(&r);
        let r = *self;
        self.set_add(&r);
    }

    #[inline(always)]
    pub fn mul16(self) -> Self {
        let mut r = self;
        r.set_mul16();
        r
    }

    #[inline(always)]
    fn set_mul32(&mut self) {
        let r = *self;
        self.set_add(&r);
        let r = *self;
        self.set_add(&r);
        let r = *self;
        self.set_add(&r);
        let r = *self;
        self.set_add(&r);
        let r = *self;
        self.set_add(&r);
    }

    #[inline(always)]
    pub fn mul32(self) -> Self {
        let mut r = self;
        r.set_mul32();
        r
    }

    // TODO: find out if there is a possible set_mul_small() which is
    // faster than a normal Montgomery multiplication.

    // Set this value to (u*f+v*g)/2^64 (with 'u' being self). Parameters f
    // and g are provided as u64, but they are signed integers in the
    // -2^62..+2^62 range.
    #[inline]
    fn set_montylin(&mut self, u: &Self, v: &Self, f: u64, g: u64) {
        // Make sure f is nonnegative, by negating it if necessary, and
        // also negating u in that case to keep u*f unchanged.
        let sf = sgnw(f);
        let f = (f ^ sf).wrapping_sub(sf);
        let tu = Self::select(u, &-u, sf as u32);

        // Same treatment for g and v.
        let sg = sgnw(g);
        let g = (g ^ sg).wrapping_sub(sg);
        let tv = Self::select(v, &-v, sg as u32);

        // Compute the linear combination on plain integers. Since f and
        // g are at most 2^62 each, intermediate 128-bit products cannot
        // overflow.
        let (d0, t) = umull_x2(tu.0[0], f, tv.0[0], g);
        let (d1, t) = umull_x2_add(tu.0[1], f, tv.0[1], g, t);
        let (d2, t) = umull_x2_add(tu.0[2], f, tv.0[2], g, t);
        let (d3, d4) = umull_x2_add(tu.0[3], f, tv.0[3], g, t);

        // Perform one round of Montgomery reduction.
        // Since u <= m - 1, v <= m - 1, f <= 2^62 and g <= 2^62, this
        // outputs a value d such that:
        //    d <= ((m - 1)*2^62 + (m - 1)*2^62 + k*m) / 2^64
        // for some integer k <= 2^64 - 1. This leads to:
        //    d <= ((2^64 + 2^63)*m - 2^63) / 2^64
        // which means that d is lower than 2*m. A single conditional
        // subtraction will ensure that the value is normalized to 0..m-1.

        let k = d0.wrapping_mul(Self::M0I);
        let (_, hi)  = umull_add(k, M0, d0);
        let (d0, hi) = umull_add2(k, M1, d1, hi);
        let (d1, hi) = umull_add2(k, M2, d2, hi);
        let (d2, hi) = umull_add2(k, M3, d3, hi);
        let (d3, d4) = addcarry_u64(d4, hi, 0);

        let (_, cc) = subborrow_u64(d0, M0, 0);
        let (_, cc) = subborrow_u64(d1, M1, cc);
        let (_, cc) = subborrow_u64(d2, M2, cc);
        let (_, cc) = subborrow_u64(d3, M3, cc);
        let w = ((d4 ^ cc) as u64).wrapping_sub(1);
        let (d0, cc) = subborrow_u64(d0, w & M0, 0);
        let (d1, cc) = subborrow_u64(d1, w & M1, cc);
        let (d2, cc) = subborrow_u64(d2, w & M2, cc);
        let (d3, _)  = subborrow_u64(d3, w & M3, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    #[inline(always)]
    fn montylin(a: &Self, b: &Self, f: u64, g: u64) -> Self {
        let mut r = Self::ZERO;
        r.set_montylin(a, b, f, g);
        r
    }

    // Set this value to abs((a*f+b*g)/2^31). Values a and b are
    // interpreted as 256-bit integers (not modular). Coefficients f and
    // g are provided as u64, but they really are signed integers in the
    // -2^31..+2^31 range (inclusive). The low 31 bits are dropped (i.e.
    // the division is assumed to be exact). The result is assumed to
    // fit in 256 bits (otherwise, truncation occurs). The absolute
    // value of (a*f+b*g)/2^31 is computed. Returned value is -1 (u64)
    // if (a*f+b*g) was negative, 0 otherwise.
    #[inline]
    fn set_lindiv31abs(&mut self, a: &Self, b: &Self, f: u64, g: u64) -> u64 {
        // Replace f and g with abs(f) and abs(g), but remember the
        // original signs.
        let sf = sgnw(f);
        let f = (f ^ sf).wrapping_sub(sf);
        let sg = sgnw(g);
        let g = (g ^ sg).wrapping_sub(sg);

        // Apply the signs of f and g to the source operands. We extend
        // the sources with an extra word to have room for the sign bit.
        let (a0, cc) = subborrow_u64(a.0[0] ^ sf, sf, 0);
        let (a1, cc) = subborrow_u64(a.0[1] ^ sf, sf, cc);
        let (a2, cc) = subborrow_u64(a.0[2] ^ sf, sf, cc);
        let (a3, cc) = subborrow_u64(a.0[3] ^ sf, sf, cc);
        let a4 = (cc as u64).wrapping_neg();
        let (b0, cc) = subborrow_u64(b.0[0] ^ sg, sg, 0);
        let (b1, cc) = subborrow_u64(b.0[1] ^ sg, sg, cc);
        let (b2, cc) = subborrow_u64(b.0[2] ^ sg, sg, cc);
        let (b3, cc) = subborrow_u64(b.0[3] ^ sg, sg, cc);
        let b4 = (cc as u64).wrapping_neg();

        // Compute a*f+b*g into d0:d1:d2:d3:t. Since f and g are at
        // most 2^31, we can add two 128-bit products with no overflow.
        // The value a*f+b*g necessarily fits on 5 limbs.
        // Also, a4 and b4 must be either 0 or -1 at this point.
        let (d0, t) = umull_x2(a0, f, b0, g);
        let (d1, t) = umull_x2_add(a1, f, b1, g, t);
        let (d2, t) = umull_x2_add(a2, f, b2, g, t);
        let (d3, t) = umull_x2_add(a3, f, b3, g, t);
        // d4 <- a4*f + b4*g + t; a4 and b4 can be only 0 or -1
        let d4 = t.wrapping_sub(a4 & f).wrapping_sub(b4 & g);

        // Right-shift the value by 31 bits.
        let d0 = (d0 >> 31) | (d1 << 33);
        let d1 = (d1 >> 31) | (d2 << 33);
        let d2 = (d2 >> 31) | (d3 << 33);
        let d3 = (d3 >> 31) | (d4 << 33);

        // If the result is negative, then negate it.
        let t = sgnw(d4);
        let (d0, cc) = subborrow_u64(d0 ^ t, t, 0);
        let (d1, cc) = subborrow_u64(d1 ^ t, t, cc);
        let (d2, cc) = subborrow_u64(d2 ^ t, t, cc);
        let (d3, _)  = subborrow_u64(d3 ^ t, t, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
        t
    }

    #[inline(always)]
    fn lindiv31abs(a: &Self, b: &Self, f: u64, g: u64) -> (Self, u64) {
        let mut r = Self::ZERO;
        let ng = r.set_lindiv31abs(a, b, f, g);
        (r, ng)
    }

    fn set_div(&mut self, y: &Self) {
        // Extended binary GCD:
        //
        //   a <- y
        //   b <- m (modulus)
        //   u <- x (self)
        //   v <- 0
        //
        // Value a is normalized (in the 0..m-1 range). Values a and b are
        // then considered as (signed) integers. Values u and v are field
        // elements.
        //
        // Invariants:
        //    a*x = y*u mod m
        //    b*x = y*v mod m
        //    b is always odd
        //
        // At each step:
        //    if a is even, then:
        //        a <- a/2, u <- u/2 mod m
        //    else:
        //        if a < b:
        //            (a, u, b, v) <- (b, v, a, u)
        //        a <- (a-b)/2, u <- (u-v)/2 mod m
        //
        // What we implement below is the optimized version of this
        // algorithm, as described in https://eprint.iacr.org/2020/972

        let mut a = *y;
        let mut b = Self([ M0, M1, M2, M3 ]);
        let mut u = *self;
        let mut v = Self::ZERO;

        // Generic loop does 15*31 = 465 inner iterations.
        for _ in 0..15 {
            // Get approximations of a and b over 64 bits:
            //  - If len(a) <= 64 and len(b) <= 64, then we just use
            //    their values (low limbs).
            //  - Otherwise, with n = max(len(a), len(b)), we use:
            //       (a mod 2^31) + 2^31*floor(a / 2^(n - 33))
            //       (b mod 2^31) + 2^31*floor(b / 2^(n - 33))

            let m3 = a.0[3] | b.0[3];
            let m2 = a.0[2] | b.0[2];
            let m1 = a.0[1] | b.0[1];
            let tnz3 = sgnw(m3 | m3.wrapping_neg());
            let tnz2 = sgnw(m2 | m2.wrapping_neg()) & !tnz3;
            let tnz1 = sgnw(m1 | m1.wrapping_neg()) & !tnz3 & !tnz2;
            let tnzm = (m3 & tnz3) | (m2 & tnz2) | (m1 & tnz1);
            let tnza = (a.0[3] & tnz3) | (a.0[2] & tnz2) | (a.0[1] & tnz1);
            let tnzb = (b.0[3] & tnz3) | (b.0[2] & tnz2) | (b.0[1] & tnz1);
            let snza = (a.0[2] & tnz3) | (a.0[1] & tnz2) | (a.0[0] & tnz1);
            let snzb = (b.0[2] & tnz3) | (b.0[1] & tnz2) | (b.0[0] & tnz1);

            // If both len(a) <= 64 and len(b) <= 64, then:
            //    tnzm = 0
            //    tnza = 0, snza = 0, tnzb = 0, snzb = 0
            // Otherwise:
            //    tnzm != 0
            //    tnza contains the top non-zero limb of a
            //    snza contains the limb right below tnza
            //    tnzb contains the top non-zero limb of a
            //    snzb contains the limb right below tnzb
            //
            // We count the number of leading zero bits in tnzm:
            //  - If s <= 31, then the top 31 bits can be extracted from
            //    tnza and tnzb alone.
            //  - If 32 <= s <= 63, then we need some bits from snza and
            //    snzb as well.
            let s = lzcnt(tnzm);
            let sm = (31_i32.wrapping_sub(s as i32) >> 31) as u64;
            let tnza = tnza ^ (sm & (tnza ^ ((tnza << 32) | (snza >> 32))));
            let tnzb = tnzb ^ (sm & (tnzb ^ ((tnzb << 32) | (snzb >> 32))));
            let s = s - (32 & (sm as u32));
            let tnza = tnza << s;
            let tnzb = tnzb << s;

            // At this point:
            //  - If len(a) <= 64 and len(b) <= 64, then:
            //       tnza = 0
            //       tnzb = 0
            //       tnz1 = tnz2 = tnz3 = 0
            //       we want to use the entire low words of a and b
            //  - Otherwise, we want to use the top 33 bits of tnza and
            //    tnzb, and the low 31 bits of the low words of a and b.
            let tzx = !(tnz1 | tnz2 | tnz3);
            let tnza = tnza | (a.0[0] & tzx);
            let tnzb = tnzb | (b.0[0] & tzx);
            let mut xa = (a.0[0] & 0x7FFFFFFF) | (tnza & 0xFFFFFFFF80000000);
            let mut xb = (b.0[0] & 0x7FFFFFFF) | (tnzb & 0xFFFFFFFF80000000);

            // Compute the 31 inner iterations on xa and xb.
            let mut fg0 = 1u64;
            let mut fg1 = 1u64 << 32;
            for _ in 0..31 {
                let a_odd = (xa & 1).wrapping_neg();
                let (_, cc) = subborrow_u64(xa, xb, 0);
                let swap = a_odd & (cc as u64).wrapping_neg();
                let t1 = swap & (xa ^ xb);
                xa ^= t1;
                xb ^= t1;
                let t2 = swap & (fg0 ^ fg1);
                fg0 ^= t2;
                fg1 ^= t2;
                xa = xa.wrapping_sub(a_odd & xb);
                fg0 = fg0.wrapping_sub(a_odd & fg1);
                xa >>= 1;
                fg1 <<= 1;
            }
            fg0 = fg0.wrapping_add(0x7FFFFFFF7FFFFFFF);
            fg1 = fg1.wrapping_add(0x7FFFFFFF7FFFFFFF);
            let f0 = (fg0 & 0xFFFFFFFF).wrapping_sub(0x7FFFFFFF);
            let g0 = (fg0 >> 32).wrapping_sub(0x7FFFFFFF);
            let f1 = (fg1 & 0xFFFFFFFF).wrapping_sub(0x7FFFFFFF);
            let g1 = (fg1 >> 32).wrapping_sub(0x7FFFFFFF);

            // Propagate updates to a, b, u and v.
            let (na, nega) = Self::lindiv31abs(&a, &b, f0, g0);
            let (nb, negb) = Self::lindiv31abs(&a, &b, f1, g1);
            let f0 = (f0 ^ nega).wrapping_sub(nega);
            let g0 = (g0 ^ nega).wrapping_sub(nega);
            let f1 = (f1 ^ negb).wrapping_sub(negb);
            let g1 = (g1 ^ negb).wrapping_sub(negb);
            let nu = Self::montylin(&u, &v, f0, g0);
            let nv = Self::montylin(&u, &v, f1, g1);
            a = na;
            b = nb;
            u = nu;
            v = nv;
        }

        // If y is invertible, then the final GCD is 1, and
        // len(a) + len(b) <= 47, so we can end the computation with
        // the low words directly. We only need 45 iterations to reach
        // the point where b = 1.
        let mut xa = a.0[0];
        let mut xb = b.0[0];
        let mut f0 = 1u64;
        let mut g0 = 0u64;
        let mut f1 = 0u64;
        let mut g1 = 1u64;
        for _ in 0..45 {
            let a_odd = (xa & 1).wrapping_neg();
            let (_, cc) = subborrow_u64(xa, xb, 0);
            let swap = a_odd & (cc as u64).wrapping_neg();
            let t1 = swap & (xa ^ xb);
            xa ^= t1;
            xb ^= t1;
            let t2 = swap & (f0 ^ f1);
            f0 ^= t2;
            f1 ^= t2;
            let t3 = swap & (g0 ^ g1);
            g0 ^= t3;
            g1 ^= t3;
            xa = xa.wrapping_sub(a_odd & xb);
            f0 = f0.wrapping_sub(a_odd & f1);
            g0 = g0.wrapping_sub(a_odd & g1);
            xa >>= 1;
            f1 <<= 1;
            g1 <<= 1;
        }

        self.set_montylin(&u, &v, f1, g1);

        // If y is invertible then b = 1 at this point. If y is not
        // invertible, then b != 1. We clear the result in the latter
        // case (by convention, we want to return 0 in that case).
        let w = b.0[1] | b.0[2] | b.0[3] | (xb ^ 1);
        let w = !sgnw(w | w.wrapping_neg());
        self.0[0] &= w;
        self.0[1] &= w;
        self.0[2] &= w;
        self.0[3] &= w;

        // At this point, each outer iteration injected 31 extra doublings,
        // except for the last one which injected 43, for a total of
        // 31*15 + 45 = 510. But each call to montylin() also implied a
        // division by 2^64, and there were 16 calls; thus, we really
        // divided the result by 2^(16*64-510) = 2^514.
        //
        // Moreover, both divisor and dividend were in Montgomery
        // representation; we thus computed in total:
        //   ((x*R)/(y*R))/2^514 = (x/y)/2^514
        // We want to Montgomery representation of the result, i.e.:
        //   (x/y)*2^256
        // We thus need to multiply by 2^(514+256) = 2^770, which we
        // do with a Montgomery multiplication with the precomputed
        // Montgomery representation of 2^770.
        self.set_mul(&Self::T770);
    }

    // Perform a batch inversion of some elements. All elements of
    // the slice are replaced with their respective inverse (elements
    // of value zero are "inverted" into themselves).
    pub fn batch_invert(xx: &mut [Self]) {
        // We use Montgomery's trick:
        //   1/u = v*(1/(u*v))
        //   1/v = u*(1/(u*v))
        // Applied recursively on n elements, this computes an inversion
        // with a single inversion in the field, and 3*(n-1) multiplications.
        // We use batches of 200 elements; larger batches only yield
        // moderate improvements, while sticking to a fixed moderate batch
        // size allows stack-based allocation.
        let n = xx.len();
        let mut i = 0;
        while i < n {
            let blen = if (n - i) > 200 { 200 } else { n - i };
            let mut tt = [Self::ZERO; 200];
            tt[0] = xx[i];
            let zz0 = tt[0].iszero();
            tt[0].set_cond(&Self::ONE, zz0);
            for j in 1..blen {
                tt[j] = xx[i + j];
                tt[j].set_cond(&Self::ONE, tt[j].iszero());
                tt[j] *= tt[j - 1];
            }
            let mut k = Self::ONE / tt[blen - 1];
            for j in (1..blen).rev() {
                let mut x = xx[i + j];
                let zz = x.iszero();
                x.set_cond(&Self::ONE, zz);
                xx[i + j].set_cond(&(k * tt[j - 1]), !zz);
                k *= x;
            }
            xx[i].set_cond(&k, !zz0);
            i += blen;
        }
    }

    // Compute the Legendre symbol on this value. Return value is:
    //   0   if this value is zero
    //  +1   if this value is a non-zero quadratic residue
    //  -1   if this value is not a quadratic residue
    pub fn legendre(self) -> i32 {
        // The algorithm is very similar to the optimized binary GCD that
        // is implemented in set_div(), with the following differences:
        //  - We do not keep track of the 'u' and 'v' values.
        //  - In each inner iteration, the running symbol value is
        //    adjusted, taking into account the low 2 or 3 bits of the
        //    involved values.
        //  - Since we need a couple of bits of look-ahead, we can only
        //    run 29 iterations in the inner loop, and we need an extra
        //    recomputation step for the next 2.
        // Otherwise, the 'a' and 'b' values are modified exactly as in
        // the binary GCD, so that we get the same guaranteed convergence
        // in a total of 510 iterations.

        let mut a = self;
        let mut b = Self([ M0, M1, M2, M3 ]);
        let mut ls = 0u64;  // running symbol information in the low bit

        // Outer loop
        for _ in 0..15 {
            // Get approximations of a and b over 64 bits.
            let m3 = a.0[3] | b.0[3];
            let m2 = a.0[2] | b.0[2];
            let m1 = a.0[1] | b.0[1];
            let tnz3 = sgnw(m3 | m3.wrapping_neg());
            let tnz2 = sgnw(m2 | m2.wrapping_neg()) & !tnz3;
            let tnz1 = sgnw(m1 | m1.wrapping_neg()) & !tnz3 & !tnz2;
            let tnzm = (m3 & tnz3) | (m2 & tnz2) | (m1 & tnz1);
            let tnza = (a.0[3] & tnz3) | (a.0[2] & tnz2) | (a.0[1] & tnz1);
            let tnzb = (b.0[3] & tnz3) | (b.0[2] & tnz2) | (b.0[1] & tnz1);
            let snza = (a.0[2] & tnz3) | (a.0[1] & tnz2) | (a.0[0] & tnz1);
            let snzb = (b.0[2] & tnz3) | (b.0[1] & tnz2) | (b.0[0] & tnz1);

            let s = lzcnt(tnzm);
            let sm = (31_i32.wrapping_sub(s as i32) >> 31) as u64;
            let tnza = tnza ^ (sm & (tnza ^ ((tnza << 32) | (snza >> 32))));
            let tnzb = tnzb ^ (sm & (tnzb ^ ((tnzb << 32) | (snzb >> 32))));
            let s = s - (32 & (sm as u32));
            let tnza = tnza << s;
            let tnzb = tnzb << s;

            let tzx = !(tnz1 | tnz2 | tnz3);
            let tnza = tnza | (a.0[0] & tzx);
            let tnzb = tnzb | (b.0[0] & tzx);
            let mut xa = (a.0[0] & 0x7FFFFFFF) | (tnza & 0xFFFFFFFF80000000);
            let mut xb = (b.0[0] & 0x7FFFFFFF) | (tnzb & 0xFFFFFFFF80000000);

            // First 29 inner iterations.
            let mut fg0 = 1u64;
            let mut fg1 = 1u64 << 32;
            for _ in 0..29 {
                let a_odd = (xa & 1).wrapping_neg();
                let (_, cc) = subborrow_u64(xa, xb, 0);
                let swap = a_odd & (cc as u64).wrapping_neg();
                ls ^= swap & ((xa & xb) >> 1);
                let t1 = swap & (xa ^ xb);
                xa ^= t1;
                xb ^= t1;
                let t2 = swap & (fg0 ^ fg1);
                fg0 ^= t2;
                fg1 ^= t2;
                xa = xa.wrapping_sub(a_odd & xb);
                fg0 = fg0.wrapping_sub(a_odd & fg1);
                xa >>= 1;
                fg1 <<= 1;
                ls ^= xb.wrapping_add(2) >> 2;
            }

            // Compute the updated a and b (low words only) to get enough
            // bits for the next two iterations.
            let fg0z = fg0.wrapping_add(0x7FFFFFFF7FFFFFFF);
            let fg1z = fg1.wrapping_add(0x7FFFFFFF7FFFFFFF);
            let f0 = (fg0z & 0xFFFFFFFF).wrapping_sub(0x7FFFFFFF);
            let g0 = (fg0z >> 32).wrapping_sub(0x7FFFFFFF);
            let f1 = (fg1z & 0xFFFFFFFF).wrapping_sub(0x7FFFFFFF);
            let g1 = (fg1z >> 32).wrapping_sub(0x7FFFFFFF);
            let mut a0 = a.0[0].wrapping_mul(f0)
                .wrapping_add(b.0[0].wrapping_mul(g0)) >> 29;
            let mut b0 = a.0[0].wrapping_mul(f1)
                .wrapping_add(b.0[0].wrapping_mul(g1)) >> 29;
            for _ in 0..2 {
                let a_odd = (xa & 1).wrapping_neg();
                let (_, cc) = subborrow_u64(xa, xb, 0);
                let swap = a_odd & (cc as u64).wrapping_neg();
                ls ^= swap & ((a0 & b0) >> 1);
                let t1 = swap & (xa ^ xb);
                xa ^= t1;
                xb ^= t1;
                let t2 = swap & (fg0 ^ fg1);
                fg0 ^= t2;
                fg1 ^= t2;
                let t3 = swap & (a0 ^ b0);
                a0 ^= t3;
                b0 ^= t3;
                xa = xa.wrapping_sub(a_odd & xb);
                fg0 = fg0.wrapping_sub(a_odd & fg1);
                a0 = a0.wrapping_sub(a_odd & b0);
                xa >>= 1;
                fg1 <<= 1;
                a0 >>= 1;
                ls ^= b0.wrapping_add(2) >> 2;
            }

            // Propagate updates to a and b.
            fg0 = fg0.wrapping_add(0x7FFFFFFF7FFFFFFF);
            fg1 = fg1.wrapping_add(0x7FFFFFFF7FFFFFFF);
            let f0 = (fg0 & 0xFFFFFFFF).wrapping_sub(0x7FFFFFFF);
            let g0 = (fg0 >> 32).wrapping_sub(0x7FFFFFFF);
            let f1 = (fg1 & 0xFFFFFFFF).wrapping_sub(0x7FFFFFFF);
            let g1 = (fg1 >> 32).wrapping_sub(0x7FFFFFFF);

            let (na, nega) = Self::lindiv31abs(&a, &b, f0, g0);
            let (nb, _)    = Self::lindiv31abs(&a, &b, f1, g1);
            ls ^= nega & (nb.0[0] >> 1);
            a = na;
            b = nb;
        }

        // Final iterations: values are at most 47 bits now. We do not
        // need to keep track of update coefficients. Just like the GCD,
        // we need only 45 iterations, because after 45 iterations,
        // value a is 0 or 1, and b is 1, and no further modification to
        // the Legendre symbol may happen.
        let mut xa = a.0[0];
        let mut xb = b.0[0];
        for _ in 0..45 {
            let a_odd = (xa & 1).wrapping_neg();
            let (_, cc) = subborrow_u64(xa, xb, 0);
            let swap = a_odd & (cc as u64).wrapping_neg();
            ls ^= swap & ((xa & xb) >> 1);
            let t1 = swap & (xa ^ xb);
            xa ^= t1;
            xb ^= t1;
            xa = xa.wrapping_sub(a_odd & xb);
            xa >>= 1;
            ls ^= xb.wrapping_add(2) >> 2;
        }

        // At this point, if the source value was not zero, then the low
        // bit of ls contains the QR status (0 = square, 1 = non-square),
        // which we need to convert to the expected value (+1 or -1).
        // If y == 0, then we return 0, per the API.
        let r = 1u32.wrapping_sub(((ls as u32) & 1) << 1);
        (r & !(self.iszero() as u32)) as i32
    }

    // Raise this value to the provided exponent. The exponent is non-zero
    // and is public. The exponent is encoded over four 64-bit limbs.
    fn set_modpow_pubexp(&mut self, e: &[u64; 4]) {
        // Make a 4-bit window; win[i] contains x^(i+1)
        let mut win = [Self::ZERO; 15];
        win[0] = *self;
        for i in 1..8 {
            let j = i * 2;
            win[j - 1] = win[i - 1].square();
            win[j] = win[j - 1] * win[0];
        }

        // Explore 4-bit chunks of the exponent, high to low. Skip leading
        // chunks of value 0.
        let mut z = false;
        for i in (0..4).rev() {
            let ew = e[i];
            for j in (0..16).rev() {
                if z {
                    self.set_xsquare(4);
                }
                let c = ((ew >> (j << 2)) & 0x0F) as usize;
                if c != 0 {
                    if z {
                        self.set_mul(&win[c - 1]);
                    } else {
                        z = true;
                        *self = win[c - 1];
                    }
                }
            }
        }
        if !z {
            *self = Self::ONE;
        }
    }

    // Set this value to its square root. Returned value is 0xFFFFFFFF
    // if the operation succeeded (value was indeed a quadratic residue),
    // 0 otherwise (value was not a quadratic residue). In the latter case,
    // this value is set to zero as well.
    // When this operation succeeds, the returned square root is the one
    // whose least significant bit is 0 (when normalized in 0..q-1).
    //
    // This operation returns unspecified results if the modulus is not
    // prime. If the modulus q is prime but is equal to 1 modulo 8, then
    // the method is not implemented (which triggers a panic).
    fn set_sqrt(&mut self) -> u32 {
        // Keep a copy of the source value, to check the square root
        // afterwards.
        let x = *self;

        if (M0 & 3) == 3 {
            // q = 3 mod 4
            // The candidate square root is x^((q+1)/4)
            self.set_modpow_pubexp(&Self::QP1D4);
        } else if (M0 & 7) == 5 {
            // q = 5 mod 8; we use Atkin's algorithm:
            //   b <- (2*x)^((q-5)/8)
            //   c <- 2*x*b^2
            //   y <- x*b*(c - 1)
            let mut b = self.mul2();
            b.set_modpow_pubexp(&Self::QM5D8);
            *self *= b;
            let c = ((self as &Self) * b).mul2();
            *self *= c - &Self::ONE;
        } else {
            // General case is Tonelli-Shanks but it requires knowledge
            // of a non-QR in the field, which we don't provide in the
            // type parameters.
            unimplemented!();
        }

        // Choose the square root whose least significant bit is 0.
        self.set_cond(&-(self as &Self),
            ((self.encode32()[0] as u32) & 1).wrapping_neg());

        // Check computed square root; clear this value on mismatch.
        let r = self.square().equals(x);
        self.set_cond(&Self::ZERO, !r);
        r
    }

    #[inline(always)]
    pub fn sqrt(self) -> (Self, u32) {
        let mut x = self;
        let r = x.set_sqrt();
        (x, r)
    }

    // Compute two signed integers (c0, c1) such that this self = c0/c1 in
    // the ring. If the modulus is less than Nmax = floor(2^254 / (2/sqrt(3)))
    // (approximately 1.73*2^253), then a solution is guaranteed to exist
    // with c0 and c1 both fitting in signed 128-bit integers; in that case,
    // this function returns such c0 and c1.
    //
    // If the modulus is larger than Nmax, then the returned c0 and c1
    // are potentially truncated to 128 bits. It can be shown that the
    // smallest vector [c0, c1] such that c0/c1 is equal to a given ring
    // element is such that |c0| and |c1| are both lower than about
    // 1.075*2^128; thus, given the truncated c0 and c1 returned by this
    // function, one can find the real values by trying all combinations
    // (c0 + a*2^128) / (c1 + b*2^128) for a and b both ranging from -1
    // to +1.
    //
    // If this element is zero, then this function returns (0, 1). Otherwise,
    // neither c0 nor c1 can be zero (though the _truncated_ version of c0
    // or c1 may be zero, if the modulus is large enough).
    //
    // THIS FUNCTION IS NOT CONSTANT-TIME. It shall be used only for a
    // public source element.
    pub fn split_vartime(self) -> (i128, i128) {
        let mut k = self;
        k.set_montyred();
        lagrange253_vartime(&k.0, &Self::MODULUS)
    }

    // Equality check between two elements (constant-time); returned value
    // is 0xFFFFFFFF on equality, 0 otherwise.
    #[inline]
    pub fn equals(self, rhs: Self) -> u32 {
        let r = (self.0[0] ^ rhs.0[0])
              | (self.0[1] ^ rhs.0[1])
              | (self.0[2] ^ rhs.0[2])
              | (self.0[3] ^ rhs.0[3]);
        ((r | r.wrapping_neg()) >> 63).wrapping_sub(1) as u32
    }

    // Compare this value with zero (constant-time); returned value
    // is 0xFFFFFFFF if this element is zero, 0 otherwise.
    #[inline]
    pub fn iszero(self) -> u32 {
        let r = self.0[0] | self.0[1] | self.0[2] | self.0[3];
        ((r | r.wrapping_neg()) >> 63).wrapping_sub(1) as u32
    }

    // Decoding exactly 32 bytes in little-endian convention; the value is
    // implicitly reduced modulo the ring order.
    #[inline(always)]
    fn decode32_reduce(buf: &[u8]) -> Self {
        let mut r = Self::ZERO;
        if buf.len() == 32 {
            r.set_decode32_reduce(buf);
        }
        r
    }

    // Set the value by decoding exactly 32 bytes in little-endian
    // convention; the value is implicitly reduced modulo the ring order.
    #[inline]
    fn set_decode32_reduce(&mut self, buf: &[u8]) {
        debug_assert!(buf.len() == 32);
        self.0[0] = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[ 0.. 8]).unwrap());
        self.0[1] = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[ 8..16]).unwrap());
        self.0[2] = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[16..24]).unwrap());
        self.0[3] = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[24..32]).unwrap());
        // Montgomery multiplication implies automatic reduction.
        self.set_mul(&Self::R2);
    }

    // Encode this value onto exactly 32 bytes. The normalized value (in
    // the 0..m-1 range) is written in little-endian order over exactly
    // 32 bytes. If the modulus is shorter than 256 bits then the top bits
    // (or bytes) are set to zero.
    #[inline]
    pub fn encode32(self) -> [u8; 32] {
        let mut r = self;
        r.set_montyred();
        let mut d = [0u8; 32];
        d[ 0.. 8].copy_from_slice(&r.0[0].to_le_bytes());
        d[ 8..16].copy_from_slice(&r.0[1].to_le_bytes());
        d[16..24].copy_from_slice(&r.0[2].to_le_bytes());
        d[24..32].copy_from_slice(&r.0[3].to_le_bytes());
        d
    }

    // Decode a value from exactly 32 bytes. The value is interpreted in
    // little-endian convention. If the provided slice does not have length
    // exactly 32 bytes, or if the value is not strictly lower than the
    // modulus, then the decoding fails.
    //
    // Returned value are (r, cc). On success, r is the decoded value, and
    // cc == 0xFFFFFFFF. On failure, r is zero, and cc == 0. If the slice
    // length is 32 bytes, then whether the value was in the correct range
    // or not is a constant-time information.
    #[inline]
    pub fn decode32(buf: &[u8]) -> (Self, u32) {
        // If the source slice length is not correct then we cannot hide
        // it from timning-based attackers, so we may as well return right
        // away.
        if buf.len() != 32 {
            return (Self::ZERO, 0);
        }

        let mut r = Self::ZERO;
        r.0[0] = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[ 0.. 8]).unwrap());
        r.0[1] = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[ 8..16]).unwrap());
        r.0[2] = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[16..24]).unwrap());
        r.0[3] = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[24..32]).unwrap());

        // Clear the value if not canonical.
        let (_, cc) = subborrow_u64(r.0[0], M0, 0);
        let (_, cc) = subborrow_u64(r.0[1], M1, cc);
        let (_, cc) = subborrow_u64(r.0[2], M2, cc);
        let (_, cc) = subborrow_u64(r.0[3], M3, cc);
        let cc = (cc as u64).wrapping_neg();
        r.0[0] &= cc;
        r.0[1] &= cc;
        r.0[2] &= cc;
        r.0[3] &= cc;

        r.set_mul(&Self::R2);
        (r, cc as u32)
    }

    pub fn decode(buf: &[u8]) -> Option<Self> {
        let n = 24 + (if M3 < 0x100000000 {
            if M3 < 0x10000 {
                if M3 < 0x100 { 1 } else { 2 }
            } else {
                if M3 < 0x1000000 { 3 } else { 4 }
            }
        } else {
            if M3 < 0x1000000000000 {
                if M3 < 0x10000000000 { 5 } else { 6 }
            } else {
                if M3 < 0x100000000000000 { 7 } else { 8 }
            }
        });
        if n != buf.len() {
            return None;
        }
        let mut bb = [0u8; 32];
        bb[0..n].copy_from_slice(buf);
        let (r, cc) = Self::decode32(&bb);
        if cc != 0 {
            Some(r)
        } else {
            None
        }
    }

    // Decode an element from some bytes. The bytes are interpreted in
    // unsigned little-endian convention, and the resulting integer is
    // reduced modulo m. This process never fails.
    pub fn decode_reduce(buf: &[u8]) -> Self {
        let mut r = Self::ZERO;
        let mut n = buf.len();
        if n == 0 {
            return r;
        }
        if (n & 31) != 0 {
            let k = n & !(31 as usize);
            let mut tmp = [0u8; 32];
            tmp[..(n - k)].copy_from_slice(&buf[k..]);
            n = k;
            r.set_decode32_reduce(&tmp);
        } else {
            n -= 32;
            r.set_decode32_reduce(&buf[n..]);
        }

        while n > 0 {
            n -= 32;
            let d = Self::decode32_reduce(&buf[n..n + 32]);
            r.set_mul(&Self::R2);
            r.set_add(&d);
        }

        r
    }

    // Given m0 (odd), compute -1/m0 mod 2^64.
    // This is used to initialize the M0I constant.
    const fn make_m0i(m0: u64) -> u64 {
        let y = 2u64.wrapping_sub(m0);
        let y = y.wrapping_mul(2u64.wrapping_sub(y.wrapping_mul(m0)));
        let y = y.wrapping_mul(2u64.wrapping_sub(y.wrapping_mul(m0)));
        let y = y.wrapping_mul(2u64.wrapping_sub(y.wrapping_mul(m0)));
        let y = y.wrapping_mul(2u64.wrapping_sub(y.wrapping_mul(m0)));
        let y = y.wrapping_mul(2u64.wrapping_sub(y.wrapping_mul(m0)));
        y.wrapping_neg()
    }

    // Compute (m+1)/2.
    const fn make_hmp1() -> Self {

        // Custom add-with-carry.
        const fn adc(x: u64, y: u64, cc: u64) -> (u64, u64) {
            let z = (x as u128)
                .wrapping_add(y as u128)
                .wrapping_add(cc as u128);
            (z as u64, (z >> 64) as u64)
        }

        let d0 = (M0 >> 1) | (M1 << 63);
        let d1 = (M1 >> 1) | (M2 << 63);
        let d2 = (M2 >> 1) | (M3 << 63);
        let d3 = M3 >> 1;
        let (d0, cc) = adc(d0, 1, 0);
        let (d1, cc) = adc(d1, 0, cc);
        let (d2, cc) = adc(d2, 0, cc);
        let d3 = d3.wrapping_add(cc);

        Self([ d0, d1, d2, d3 ])
    }

    // Montgomery multiplication of a[] by b[]. Value a must be lower
    // than m; value a may be arbitrary (up to 2^256-1). Returned value
    // is a*b/2^256 mod m, fully reduced. This function is meant for
    // evaluation in constant contexts (e.g. compile-time evaluation); it
    // may be somewhat slower than the runtime conversion functions
    // (it is still constant-time, thus safe to use at runtime).
    const fn const_mmul(a: Self, b: Self) -> Self {

        // Custom add-with-carry.
        const fn adc(x: u64, y: u64, cc: u64) -> (u64, u64) {
            let z = (x as u128)
                .wrapping_add(y as u128)
                .wrapping_add(cc as u128);
            (z as u64, (z >> 64) as u64)
        }

        // Compute x*y + a + b, returned over two words (lo, hi).
        const fn umaal(x: u64, y: u64, a: u64, b: u64) -> (u64, u64) {
            let z = (x as u128) * (y as u128) + (a as u128) + (b as u128);
            (z as u64, (z >> 64) as u64)
        }

        // Given d0..d4 (with d <= 2*m-1), operand b[] (b <= m-1) and
        // multiplier aj, return ((d + aj*b) / 2^64) mod m, partially
        // reduced (output is at most 2*m-1).
        const fn mmul1<const M0: u64, const M1: u64,
                       const M2: u64, const M3: u64>
                      (aj: u64, b: [u64; 4],
                       d0: u64, d1: u64, d2: u64, d3: u64, d4: u64, m0i: u64)
                      -> (u64, u64, u64, u64, u64)
        {
            // d <- d + a*bj (may range up to (2^64+1)*m, needs 6 words)
            let (d0, hi) = umaal(aj, b[0], d0, 0);
            let (d1, hi) = umaal(aj, b[1], d1, hi);
            let (d2, hi) = umaal(aj, b[2], d2, hi);
            let (d3, hi) = umaal(aj, b[3], d3, hi);
            let (d4, d5) = adc(d4, hi, 0);
            let f = d0.wrapping_mul(m0i);
            let (_, hi)  = umaal(f, M0, d0, 0);
            let (d0, hi) = umaal(f, M1, d1, hi);
            let (d1, hi) = umaal(f, M2, d2, hi);
            let (d2, hi) = umaal(f, M3, d3, hi);
            let (d3, cc) = adc(d4, hi, 0);
            let (d4, _)  = adc(d5, 0, cc);
            (d0, d1, d2, d3, d4)
        }

        let m0i = Self::M0I;
        let (d0, d1, d2, d3, d4) = (0u64, 0u64, 0u64, 0u64, 0u64);
        let (d0, d1, d2, d3, d4) =
            mmul1::<M0, M1, M2, M3>(a.0[0], b.0, d0, d1, d2, d3, d4, m0i);
        let (d0, d1, d2, d3, d4) =
            mmul1::<M0, M1, M2, M3>(a.0[1], b.0, d0, d1, d2, d3, d4, m0i);
        let (d0, d1, d2, d3, d4) =
            mmul1::<M0, M1, M2, M3>(a.0[2], b.0, d0, d1, d2, d3, d4, m0i);
        let (d0, d1, d2, d3, d4) =
            mmul1::<M0, M1, M2, M3>(a.0[3], b.0, d0, d1, d2, d3, d4, m0i);
        Self(Self::const_mred1(d0, d1, d2, d3, d4))
    }

    // Given d = d0..d4 of value at most 2*m-1, return d mod m
    // (i.e. subtract m once if needed). This is a support function for
    // operations in constant contexts. It is constant-time and safe to
    // use at runtime.
    const fn const_mred1(a0: u64, a1: u64, a2: u64, a3: u64, a4: u64)
        -> [u64; 4]
    {
        // Custom subtract-with-borrow.
        const fn sbb(x: u64, y: u64, cc: u64) -> (u64, u64) {
            let z = (x as u128)
                .wrapping_sub(y as u128)
                .wrapping_sub(cc as u128);
            (z as u64, (z >> 127) as u64)
        }

        // Subtract the modulus; since the input is supposed to be
        // at most 2*m-1, the result must fit in four words, and the
        // top word (e4) will be zero. However, if the input was less
        // than m initially, then the top word (e4) will be -1; we thus
        // use e4 to select the correct result.
        let (e0, cc) = sbb(a0, M0, 0);
        let (e1, cc) = sbb(a1, M1, cc);
        let (e2, cc) = sbb(a2, M2, cc);
        let (e3, cc) = sbb(a3, M3, cc);
        let e4 = a4.wrapping_sub(cc);

        [ e0 ^ (e4 & (e0 ^ a0)),
          e1 ^ (e4 & (e1 ^ a1)),
          e2 ^ (e4 & (e2 ^ a2)),
          e3 ^ (e4 & (e3 ^ a3)) ]
    }

    // Compute R2 = 2^512 mod m. This function is meant for compile-time
    // use, not runtime, hence it defines its own primitives which are
    // compatible with const evaluation. This function has no requirement
    // for constant-time processing.
    const fn make_r2() -> Self {
        // R2 = 2^512 mod m = Montgomery representation of 2^256 mod m.
        // We first compute 2^257 mod m, which is the Montgomery
        // representation of 2 modulo m. We then perform 8 successive
        // Montgomery squarings to get the result.

        // Given a (modulo m), return 2*a mod m. Input must be lower than m.
        const fn mdbl<const M0: u64, const M1: u64,
                      const M2: u64, const M3: u64>(a: [u64; 4]) -> [u64; 4]
        {
            let d0 = a[0] << 1;
            let d1 = (a[0] >> 63) | (a[1] << 1);
            let d2 = (a[1] >> 63) | (a[2] << 1);
            let d3 = (a[2] >> 63) | (a[3] << 1);
            let d4 = a[3] >> 63;
            ModInt256::<M0, M1, M2, M3>::const_mred1(d0, d1, d2, d3, d4)
        }

        // Given a (modulo m), return 256*a mod m. Input must be lower than m.
        const fn mmul256<const M0: u64, const M1: u64,
                         const M2: u64, const M3: u64>(a: [u64; 4]) -> [u64; 4]
        {
            let a = mdbl::<M0, M1, M2, M3>(a);
            let a = mdbl::<M0, M1, M2, M3>(a);
            let a = mdbl::<M0, M1, M2, M3>(a);
            let a = mdbl::<M0, M1, M2, M3>(a);
            let a = mdbl::<M0, M1, M2, M3>(a);
            let a = mdbl::<M0, M1, M2, M3>(a);
            let a = mdbl::<M0, M1, M2, M3>(a);
            mdbl::<M0, M1, M2, M3>(a)
        }

        // Since m3 != 0 and m0 is odd, we know that 2^192 < m.
        // We then multiply it by 256 eight times, to get R = 2^256 mod m.
        let a: [u64; 4] = [ 0, 0, 0, 1 ];
        let a = mmul256::<M0, M1, M2, M3>(a);
        let a = mmul256::<M0, M1, M2, M3>(a);
        let a = mmul256::<M0, M1, M2, M3>(a);
        let a = mmul256::<M0, M1, M2, M3>(a);
        let a = mmul256::<M0, M1, M2, M3>(a);
        let a = mmul256::<M0, M1, M2, M3>(a);
        let a = mmul256::<M0, M1, M2, M3>(a);
        let a = mmul256::<M0, M1, M2, M3>(a);

        // Double it again to get 2^257 mod m, which is the Montgomery
        // representation of 2.
        let a = mdbl::<M0, M1, M2, M3>(a);

        // Apply 8 successive Montgomery squarings to get the Montgomery
        // representation of 2^256, i.e. the value R2.
        let r = Self(a);
        let r = Self::const_mmul(r, r);
        let r = Self::const_mmul(r, r);
        let r = Self::const_mmul(r, r);
        let r = Self::const_mmul(r, r);
        let r = Self::const_mmul(r, r);
        let r = Self::const_mmul(r, r);
        let r = Self::const_mmul(r, r);
        let r = Self::const_mmul(r, r);

        r
    }

    // Compute the Montgomery representation of 2^770 (compile-time).
    const fn make_t770() -> Self {
        let r = Self::const_mmul(Self([ 4, 0, 0, 0 ]), Self::R2);
        let r = Self::const_mmul(r, Self::R2);
        let r = Self::const_mmul(r, Self::R2);
        let r = Self::const_mmul(r, Self::R2);
        r
    }

    // Compute floor(q / 4) + 1 (this is equal to (q + 1)/4 if q = 3 mod 4).
    const fn make_qp1d4() -> [u64; 4] {

        // Custom add-with-carry.
        const fn adc(x: u64, y: u64, cc: u64) -> (u64, u64) {
            let z = (x as u128)
                .wrapping_add(y as u128)
                .wrapping_add(cc as u128);
            (z as u64, (z >> 64) as u64)
        }

        let d0 = (M0 >> 2) | (M1 << 62);
        let d1 = (M1 >> 2) | (M2 << 62);
        let d2 = (M2 >> 2) | (M3 << 62);
        let d3 = M3 >> 2;
        let (d0, cc) = adc(d0, 1, 0);
        let (d1, cc) = adc(d1, 0, cc);
        let (d2, cc) = adc(d2, 0, cc);
        let (d3, _)  = adc(d3, 0, cc);

        [ d0, d1, d2, d3 ]
    }

    // Compute floor(q / 8) (this is equal to (q - 5)/8 if q = 5 mod 8).
    const fn make_qm5d8() -> [u64; 4] {
        let d0 = (M0 >> 3) | (M1 << 61);
        let d1 = (M1 >> 3) | (M2 << 61);
        let d2 = (M2 >> 3) | (M3 << 61);
        let d3 = M3 >> 3;
        [ d0, d1, d2, d3 ]
    }
}

// ========================================================================
// Implementations of all the traits needed to use the simple operators
// (+, *, /...) on field element instances, with or without references.

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Add<ModInt256<M0, M1, M2, M3>> for ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn add(self, other: ModInt256<M0, M1, M2, M3>)
        -> ModInt256<M0, M1, M2, M3>
    {
        let mut r = self;
        r.set_add(&other);
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Add<&ModInt256<M0, M1, M2, M3>> for ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn add(self, other: &ModInt256<M0, M1, M2, M3>)
        -> ModInt256<M0, M1, M2, M3>
    {
        let mut r = self;
        r.set_add(other);
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Add<ModInt256<M0, M1, M2, M3>> for &ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn add(self, other: ModInt256<M0, M1, M2, M3>)
        -> ModInt256<M0, M1, M2, M3>
    {
        let mut r = *self;
        r.set_add(&other);
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Add<&ModInt256<M0, M1, M2, M3>> for &ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn add(self, other: &ModInt256<M0, M1, M2, M3>)
        -> ModInt256<M0, M1, M2, M3>
    {
        let mut r = *self;
        r.set_add(other);
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    AddAssign<ModInt256<M0, M1, M2, M3>> for ModInt256<M0, M1, M2, M3>
{
    #[inline(always)]
    fn add_assign(&mut self, other: ModInt256<M0, M1, M2, M3>) {
        self.set_add(&other);
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    AddAssign<&ModInt256<M0, M1, M2, M3>> for ModInt256<M0, M1, M2, M3>
{
    #[inline(always)]
    fn add_assign(&mut self, other: &ModInt256<M0, M1, M2, M3>) {
        self.set_add(other);
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Div<ModInt256<M0, M1, M2, M3>> for ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn div(self, other: ModInt256<M0, M1, M2, M3>)
        -> ModInt256<M0, M1, M2, M3>
    {
        let mut r = self;
        r.set_div(&other);
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Div<&ModInt256<M0, M1, M2, M3>> for ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn div(self, other: &ModInt256<M0, M1, M2, M3>)
        -> ModInt256<M0, M1, M2, M3>
    {
        let mut r = self;
        r.set_div(other);
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Div<ModInt256<M0, M1, M2, M3>> for &ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn div(self, other: ModInt256<M0, M1, M2, M3>)
        -> ModInt256<M0, M1, M2, M3>
    {
        let mut r = *self;
        r.set_div(&other);
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Div<&ModInt256<M0, M1, M2, M3>> for &ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn div(self, other: &ModInt256<M0, M1, M2, M3>)
        -> ModInt256<M0, M1, M2, M3>
    {
        let mut r = *self;
        r.set_div(other);
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    DivAssign<ModInt256<M0, M1, M2, M3>> for ModInt256<M0, M1, M2, M3>
{
    #[inline(always)]
    fn div_assign(&mut self, other: ModInt256<M0, M1, M2, M3>) {
        self.set_div(&other);
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    DivAssign<&ModInt256<M0, M1, M2, M3>> for ModInt256<M0, M1, M2, M3>
{
    #[inline(always)]
    fn div_assign(&mut self, other: &ModInt256<M0, M1, M2, M3>) {
        self.set_div(other);
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Mul<ModInt256<M0, M1, M2, M3>> for ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn mul(self, other: ModInt256<M0, M1, M2, M3>)
        -> ModInt256<M0, M1, M2, M3>
    {
        let mut r = self;
        r.set_mul(&other);
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Mul<&ModInt256<M0, M1, M2, M3>> for ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn mul(self, other: &ModInt256<M0, M1, M2, M3>)
        -> ModInt256<M0, M1, M2, M3>
    {
        let mut r = self;
        r.set_mul(other);
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Mul<ModInt256<M0, M1, M2, M3>> for &ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn mul(self, other: ModInt256<M0, M1, M2, M3>)
        -> ModInt256<M0, M1, M2, M3>
    {
        let mut r = *self;
        r.set_mul(&other);
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Mul<&ModInt256<M0, M1, M2, M3>> for &ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn mul(self, other: &ModInt256<M0, M1, M2, M3>)
        -> ModInt256<M0, M1, M2, M3>
    {
        let mut r = *self;
        r.set_mul(other);
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    MulAssign<ModInt256<M0, M1, M2, M3>> for ModInt256<M0, M1, M2, M3>
{
    #[inline(always)]
    fn mul_assign(&mut self, other: ModInt256<M0, M1, M2, M3>) {
        self.set_mul(&other);
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    MulAssign<&ModInt256<M0, M1, M2, M3>> for ModInt256<M0, M1, M2, M3>
{
    #[inline(always)]
    fn mul_assign(&mut self, other: &ModInt256<M0, M1, M2, M3>) {
        self.set_mul(other);
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Neg for ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn neg(self) -> ModInt256<M0, M1, M2, M3> {
        let mut r = self;
        r.set_neg();
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Neg for &ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn neg(self) -> ModInt256<M0, M1, M2, M3> {
        let mut r = *self;
        r.set_neg();
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Sub<ModInt256<M0, M1, M2, M3>> for ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn sub(self, other: ModInt256<M0, M1, M2, M3>)
        -> ModInt256<M0, M1, M2, M3>
    {
        let mut r = self;
        r.set_sub(&other);
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Sub<&ModInt256<M0, M1, M2, M3>> for ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn sub(self, other: &ModInt256<M0, M1, M2, M3>)
        -> ModInt256<M0, M1, M2, M3>
    {
        let mut r = self;
        r.set_sub(other);
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Sub<ModInt256<M0, M1, M2, M3>> for &ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn sub(self, other: ModInt256<M0, M1, M2, M3>)
        -> ModInt256<M0, M1, M2, M3>
    {
        let mut r = *self;
        r.set_sub(&other);
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    Sub<&ModInt256<M0, M1, M2, M3>> for &ModInt256<M0, M1, M2, M3>
{
    type Output = ModInt256<M0, M1, M2, M3>;

    #[inline(always)]
    fn sub(self, other: &ModInt256<M0, M1, M2, M3>)
        -> ModInt256<M0, M1, M2, M3>
    {
        let mut r = *self;
        r.set_sub(other);
        r
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    SubAssign<ModInt256<M0, M1, M2, M3>> for ModInt256<M0, M1, M2, M3>
{
    #[inline(always)]
    fn sub_assign(&mut self, other: ModInt256<M0, M1, M2, M3>) {
        self.set_sub(&other);
    }
}

impl<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
    SubAssign<&ModInt256<M0, M1, M2, M3>> for ModInt256<M0, M1, M2, M3>
{
    #[inline(always)]
    fn sub_assign(&mut self, other: &ModInt256<M0, M1, M2, M3>) {
        self.set_sub(other);
    }
}

// ========================================================================

#[cfg(test)]
mod tests {

    use super::ModInt256;
    use num_bigint::{BigInt, Sign};
    use sha2::{Sha256, Digest};

    /* unused
    use std::fmt;

    fn print<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
            (name: &str, v: ModInt256<M0, M1, M2, M3>)
    {
        println!("{} = 0x{:016X}{:016X}{:016X}{:016X}",
            name, v.0[3], v.0[2], v.0[1], v.0[0]);
    }
    */

    // va, vb and vx must be 32 bytes each in length
    fn check_gf_ops<const M0: u64, const M1: u64,
                    const M2: u64, const M3: u64>
                   (va: &[u8], vb: &[u8], vx: &[u8])
    {
        let zp = BigInt::from_slice(Sign::Plus, &[
            M0 as u32, (M0 >> 32) as u32,
            M1 as u32, (M1 >> 32) as u32,
            M2 as u32, (M2 >> 32) as u32,
            M3 as u32, (M3 >> 32) as u32,
        ]);
        let zpz = &zp << 64;

        let a = ModInt256::<M0, M1, M2, M3>::decode32_reduce(va);
        let b = ModInt256::<M0, M1, M2, M3>::decode32_reduce(vb);
        let za = BigInt::from_bytes_le(Sign::Plus, va);
        let zb = BigInt::from_bytes_le(Sign::Plus, vb);

        let vc = a.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = &za % &zp;
        assert!(zc == zd);

        let c = a + b;
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za + &zb) % &zp;
        assert!(zc == zd);

        let c = a - b;
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = ((&zpz + &za) - &zb) % &zp;
        assert!(zc == zd);

        let c = -a;
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&zpz - &za) % &zp;
        assert!(zc == zd);

        let c = a * b;
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za * &zb) % &zp;
        assert!(zc == zd);

        let c = a.half();
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd: BigInt = ((&zpz + (&zc << 1)) - &za) % &zp;
        assert!(zd.sign() == Sign::NoSign);

        let c = a.mul2();
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za << 1) % &zp;
        assert!(zc == zd);

        let c = a.mul4();
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za << 2) % &zp;
        assert!(zc == zd);

        let c = a.mul8();
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za << 3) % &zp;
        assert!(zc == zd);

        let c = a.mul16();
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za << 4) % &zp;
        assert!(zc == zd);

        let c = a.mul32();
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za << 5) % &zp;
        assert!(zc == zd);

        /*
         * No mul_small() defined on this structure.
         *
        let x = b.0[1] as u32;
        let c = a.mul_small(x);
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za * x) % &zp;
        assert!(zc == zd);
         */

        let c = a.square();
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za * &za) % &zp;
        assert!(zc == zd);

        let (e, cc) = ModInt256::<M0, M1, M2, M3>::decode32(va);
        if cc != 0 {
            assert!(cc == 0xFFFFFFFF);
            assert!(e.encode32() == va);
        } else {
            assert!(e.encode32() == [0u8; 32]);
        }

        let mut tmp = [0u8; 96];
        tmp[0..32].copy_from_slice(va);
        tmp[32..64].copy_from_slice(vb);
        tmp[64..96].copy_from_slice(vx);
        for k in 0..97 {
            let c = ModInt256::<M0, M1, M2, M3>::decode_reduce(&tmp[0..k]);
            let vc = c.encode32();
            let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
            let zd = BigInt::from_bytes_le(Sign::Plus, &tmp[0..k]) % &zp;
            assert!(zc == zd);
        }

        let c = a / b;
        let d = c * b;
        if b.iszero() != 0 {
            assert!(c.iszero() != 0);
        } else {
            assert!(a.equals(d) != 0);
        }
    }

    // This tests ring operations. If nqr is non-zero, then the function
    // assumes that the ring is a field (i.e. modulus is prime) and that
    // nqr is a non-quadratic-residue in that ring.
    fn test_ring<const M0: u64, const M1: u64, const M2: u64, const M3: u64>
                (nqr: u32)
    {
        let mut va = [0u8; 32];
        let mut vb = [0u8; 32];
        let mut vx = [0u8; 32];
        check_gf_ops::<M0, M1, M2, M3>(&va, &vb, &vx);
        assert!(ModInt256::<M0, M1, M2, M3>::decode_reduce(&va).iszero() == 0xFFFFFFFF);
        assert!(ModInt256::<M0, M1, M2, M3>::decode_reduce(&va).equals(ModInt256::<M0, M1, M2, M3>::decode_reduce(&vb)) == 0xFFFFFFFF);
        assert!(ModInt256::<M0, M1, M2, M3>::decode_reduce(&va).legendre() == 0);
        for i in 0..32 {
            va[i] = 0xFFu8;
            vb[i] = 0xFFu8;
            vx[i] = 0xFFu8;
        }
        check_gf_ops::<M0, M1, M2, M3>(&va, &vb, &vx);
        if M0 == 0xFFFFFFFFFFFFFFFF
           && M1 == 0xFFFFFFFFFFFFFFFF
           && M2 == 0xFFFFFFFFFFFFFFFF
           && M3 == 0xFFFFFFFFFFFFFFFF
        {
            assert!(ModInt256::<M0, M1, M2, M3>::decode_reduce(&va).iszero() == 0xFFFFFFFF);
        } else {
            assert!(ModInt256::<M0, M1, M2, M3>::decode_reduce(&va).iszero() == 0);
        }
        assert!(ModInt256::<M0, M1, M2, M3>::decode_reduce(&va).equals(ModInt256::<M0, M1, M2, M3>::decode_reduce(&vb)) == 0xFFFFFFFF);
        va[ 0.. 8].copy_from_slice(&M0.to_le_bytes());
        va[ 8..16].copy_from_slice(&M1.to_le_bytes());
        va[16..24].copy_from_slice(&M2.to_le_bytes());
        va[24..32].copy_from_slice(&M3.to_le_bytes());
        assert!(ModInt256::<M0, M1, M2, M3>::decode_reduce(&va).iszero() == 0xFFFFFFFF);
        let mut sh = Sha256::new();
        let xnqr = ModInt256::<M0, M1, M2, M3>::w64le(nqr as u64, 0, 0, 0);
        let tt = ModInt256::<M0, M1, M2, M3>::w64le(0, 0, 1, 0);
        let corr128 = [
            -tt,
            ModInt256::<M0, M1, M2, M3>::ZERO,
            tt,
        ];
        for i in 0..300 {
            sh.update(((3 * i + 0) as u64).to_le_bytes());
            let va = sh.finalize_reset();
            sh.update(((3 * i + 1) as u64).to_le_bytes());
            let vb = sh.finalize_reset();
            sh.update(((3 * i + 2) as u64).to_le_bytes());
            let vx = sh.finalize_reset();
            check_gf_ops::<M0, M1, M2, M3>(&va, &vb, &vx);
            assert!(ModInt256::<M0, M1, M2, M3>::decode_reduce(&va).iszero() == 0);
            assert!(ModInt256::<M0, M1, M2, M3>::decode_reduce(&va).equals(ModInt256::<M0, M1, M2, M3>::decode_reduce(&vb)) == 0);
            if nqr != 0 {
                let s = ModInt256::<M0, M1, M2, M3>::decode_reduce(&va).square();
                let s2 = s * xnqr;
                assert!(s.legendre() == 1);
                assert!(s2.legendre() == -1);
                let (t, r) = s.sqrt();
                assert!(r == 0xFFFFFFFF);
                assert!(t.square().equals(s) == 0xFFFFFFFF);
                assert!((t.encode32()[0] & 1) == 0);
                let (t2, r) = s2.sqrt();
                assert!(r == 0);
                assert!(t2.iszero() == 0xFFFFFFFF);
            }

            let a = ModInt256::<M0, M1, M2, M3>::decode_reduce(&va);
            let (c0, c1) = a.split_vartime();
            let b0 = ModInt256::<M0, M1, M2, M3>::from_i128(c0);
            let b1 = ModInt256::<M0, M1, M2, M3>::from_i128(c1);
            let mut ok = false;
            for k1 in 0..3 {
                let ah = a * (b1 + corr128[k1]);
                for k0 in 0..3 {
                    if ah.equals(b0 + corr128[k0]) == 0xFFFFFFFF {
                        ok = true;
                    }
                }
            }
            assert!(ok);
        }
    }

    #[test]
    fn gfp256_ops() {
        // Modulus from curve P-256.
        test_ring::< 0xFFFFFFFFFFFFFFFF,
                     0x00000000FFFFFFFF,
                     0x0000000000000000,
                     0xFFFFFFFF00000001 >(3);
    }

    #[test]
    fn gf25519_ops() {
        // 2^255 - 19
        test_ring::< 0xFFFFFFFFFFFFFFED,
                     0xFFFFFFFFFFFFFFFF,
                     0xFFFFFFFFFFFFFFFF,
                     0x7FFFFFFFFFFFFFFF >(2);
    }

    #[test]
    fn gfsdo255e_ops() {
        // Order of the do255e group (prime, 254 bits).
        test_ring::< 0x1F52C8AE74D84525,
                     0x9D0C930F54078C53,
                     0xFFFFFFFFFFFFFFFF,
                     0x3FFFFFFFFFFFFFFF >(2);
    }

    #[test]
    fn gfspec1_ops() {
        // Largest modulus that exercises the "middle case" of
        // Montgomery multiplication.
        test_ring::< 0xFFFFFFFFFFFFFF27,
                     0xFFFFFFFFFFFFFFFE,
                     0x0000000000000000,
                     0xFFFFFFFFFFFFFFFF >(5);
    }

    #[test]
    fn gfspec2_ops() {
        // Largest 256-bit prime.
        test_ring::< 0xFFFFFFFFFFFFFF43,
                     0xFFFFFFFFFFFFFFFF,
                     0xFFFFFFFFFFFFFFFF,
                     0xFFFFFFFFFFFFFFFF >(2);
    }

    #[test]
    fn gfp256_batch_invert() {
        type GF = ModInt256<0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFF,
                            0x0000000000000000, 0xFFFFFFFF00000001>;
        let mut xx = [GF::ZERO; 300];
        let mut sh = Sha256::new();
        for i in 0..300 {
            sh.update((i as u64).to_le_bytes());
            let v = sh.finalize_reset();
            xx[i] = GF::decode_reduce(&v);
        }
        xx[120] = GF::ZERO;
        let mut yy = xx;
        GF::batch_invert(&mut yy[..]);
        for i in 0..300 {
            if xx[i].iszero() != 0 {
                assert!(yy[i].iszero() == 0xFFFFFFFF);
            } else {
                assert!((xx[i] * yy[i]).equals(GF::ONE) == 0xFFFFFFFF);
            }
        }
    }

    #[test]
    fn ttinv() {
        type GF = ModInt256<0xF3B9CAC2FC632551, 0xBCE6FAADA7179E84,
                            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFF00000000>;
        let num = GF::from_u32(1);
        let den = -GF::from_u32(3);
        let r = num / den;
        assert!(r.iszero() == 0);
        assert!((-r * GF::w64be(0, 0, 0, 3)).equals(GF::ONE) != 0);
    }
}
