use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::convert::TryFrom;

use super::{addcarry_u64, subborrow_u64, umull, umull_x2, umull_x2_add, sgnw, lzcnt};
use super::lagrange::{lagrange253_vartime};

#[derive(Clone, Copy, Debug)]
#[repr(align(32))]
pub struct GF255<const MQ: u64>([u64; 4]);

/// Special container for "not reduced" values returned by `add_noreduce()`
/// and `sub_noreduce()`; for this code, this is an alias on `GF255<MQ>`
/// and the "not reduced" values are normal values.
pub type GF255NotReduced<const MQ: u64> = GF255<MQ>;

impl<const MQ: u64> GF255<MQ> {

    // Parameter restrictions:
    //   MQ is odd
    //   MQ <= 32767
    //   q = 2^255 - MQ is prime
    // Moreover, if MQ == 7 mod 8 (i.e. q = 1 mod 8), then square root
    // computations are not implemented.
    //
    // Primality cannot easily be tested at compile-time, but we check
    // the other properties.
    //
    // Tightest restriction on MQ is from set_sqrt_ext(), which assumes that
    // only the lowest 15 bits of q may be non-zero. Other arithmetic
    // functions have looser requirements (set_mul() and set_square() need
    // MQ <= 2^31 - 1).
    #[allow(dead_code)]
    const COMPILE_TIME_CHECKS: () = Self::compile_time_checks();
    const fn compile_time_checks() {
        static_assert!((MQ & 1) != 0);
        static_assert!(MQ <= 32767);
    }

    // Element encoding length (in bytes); always 32 bytes.
    pub const ENC_LEN: usize = 32;

    // Modulus is q = 2^255 - T255_MINUS_Q.
    // (this is the type parameter MQ, as a 32-bit integer)
    pub const T255_MINUS_Q: u32 = MQ as u32;

    // Modulus q in base 2^64 (low-to-high order).
    pub const MODULUS: [u64; 4] = [
        MQ.wrapping_neg(),
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x7FFFFFFFFFFFFFFF
    ];

    pub const ZERO: GF255<MQ> = GF255::<MQ>([ 0, 0, 0, 0 ]);
    pub const ONE: GF255<MQ> = GF255::<MQ>([ 1, 0, 0, 0 ]);
    pub const MINUS_ONE: GF255<MQ> = GF255::<MQ>([
        (MQ + 1).wrapping_neg(),
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x7FFFFFFFFFFFFFFF,
    ]);

    // 1/2^508 in the field, as a constant; this is used when computing
    // divisions in the field. The value is computed at compile-time.
    const INVT508: GF255<MQ> = GF255::<MQ>::make_invt508();

    // Create an element from a 256-bit value (implicitly reduced modulo
    // the field order) provided as four 64-bit limbs (in low-to-high order).
    pub const fn w64le(x0: u64, x1: u64, x2: u64, x3: u64) -> Self {
        Self([ x0, x1, x2, x3 ])
    }

    // Create an element from a 256-bit value (implicitly reduced modulo
    // the field order) provided as four 64-bit limbs (in high-to-low order).
    pub const fn w64be(x3: u64, x2: u64, x1: u64, x0: u64) -> Self {
        Self([ x0, x1, x2, x3 ])
    }

    // Create an element from a 256-bit value (implicitly reduced modulo
    // the field order) provided as four 64-bit limbs (in low-to-high order).
    pub fn from_w64le(x0: u64, x1: u64, x2: u64, x3: u64) -> Self {
        Self([ x0, x1, x2, x3 ])
    }

    // Create an element from a 256-bit value (implicitly reduced modulo
    // the field order) provided as four 64-bit limbs (in high-to-low order).
    pub fn from_w64be(x3: u64, x2: u64, x1: u64, x0: u64) -> Self {
        Self([ x0, x1, x2, x3 ])
    }

    // Create an element by converting the provided integer.
    // If the source value is negative, then it is implicitly reduced
    // modulo the ring order.
    #[inline(always)]
    pub fn from_i32(x: i32) -> Self {
        // We add q to ensure a nonnegative integer.
        let x0 = (x as i64) as u64;
        let xh = ((x as i64) >> 63) as u64;
        let (d0, cc) = addcarry_u64(x0, MQ.wrapping_neg(), 0);
        let (d1, cc) = addcarry_u64(xh, 0xFFFFFFFFFFFFFFFF, cc);
        let (d2, cc) = addcarry_u64(xh, 0xFFFFFFFFFFFFFFFF, cc);
        let (d3, _)  = addcarry_u64(xh, 0x7FFFFFFFFFFFFFFF, cc);
        Self([ d0, d1, d2, d3 ])
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
        // We add q to ensure a nonnegative integer.
        let x0 = x as u64;
        let xh = (x >> 63) as u64;
        let (d0, cc) = addcarry_u64(x0, MQ.wrapping_neg(), 0);
        let (d1, cc) = addcarry_u64(xh, 0xFFFFFFFFFFFFFFFF, cc);
        let (d2, cc) = addcarry_u64(xh, 0xFFFFFFFFFFFFFFFF, cc);
        let (d3, _)  = addcarry_u64(xh, 0x7FFFFFFFFFFFFFFF, cc);
        Self([ d0, d1, d2, d3 ])
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
        // We add q to ensure a nonnegative integer.
        let x0 = x as u64;
        let x1 = (x >> 64) as u64;
        let xh = (x >> 127) as u64;
        let (d0, cc) = addcarry_u64(x0, MQ.wrapping_neg(), 0);
        let (d1, cc) = addcarry_u64(x1, 0xFFFFFFFFFFFFFFFF, cc);
        let (d2, cc) = addcarry_u64(xh, 0xFFFFFFFFFFFFFFFF, cc);
        let (d3, _)  = addcarry_u64(xh, 0x7FFFFFFFFFFFFFFF, cc);
        Self([ d0, d1, d2, d3 ])
    }

    // Create an element by converting the provided integer.
    #[inline(always)]
    pub fn from_u128(x: u128) -> Self {
        Self::from_w64le(x as u64, (x >> 64) as u64, 0, 0)
    }

    #[inline]
    fn set_add(&mut self, rhs: &Self) {
        // 1. Addition with carry
        let (d0, cc) = addcarry_u64(self.0[0], rhs.0[0], 0);
        let (d1, cc) = addcarry_u64(self.0[1], rhs.0[1], cc);
        let (d2, cc) = addcarry_u64(self.0[2], rhs.0[2], cc);
        let (d3, cc) = addcarry_u64(self.0[3], rhs.0[3], cc);

        // 2. In case of an output carry, subtract 2*q, i.e. add 2*MQ.
        let (d0, cc) = addcarry_u64(d0,
            (cc as u64).wrapping_neg() & (2 * MQ), 0);
        let (d1, cc) = addcarry_u64(d1, 0, cc);
        let (d2, cc) = addcarry_u64(d2, 0, cc);
        let (d3, cc) = addcarry_u64(d3, 0, cc);

        // 3. If there is again an extra carry, then we have to subtract 2*q
        // again. In that case, original sum was at least 2^257 - 2*MQ, and
        // the low word is now lower than 2*MQ, so adding 2*MQ to it will
        // not overflow.
        let d0 = d0.wrapping_add((cc as u64).wrapping_neg() & (2 * MQ));

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    /// Return self + rhs (no reduction).
    #[inline(always)]
    pub fn add_noreduce(self, rhs: &Self) -> GF255NotReduced<MQ> {
        self + rhs
    }

    /// Return 2*self (no reduction).
    #[inline(always)]
    pub fn mul2_noreduce(self) -> GF255NotReduced<MQ> {
        self.mul2()
    }

    /// Return self - rhs (no reduction).
    #[inline(always)]
    pub fn sub_noreduce(self, rhs: &Self) -> GF255NotReduced<MQ> {
        self - rhs
    }

    /// Return 2*self + b and 2*self - b (no reduction).
    #[inline(always)]
    pub fn mul2add_mul2sub_noreduce(self, b: &Self)
        -> (GF255NotReduced<MQ>, GF255NotReduced<MQ>)
    {
        let d = self.mul2();
        let e = d + b;
        let f = d - b;
        (e, f)
    }

    /// Return self + b and self + b - c (no reduction).
    #[inline(always)]
    pub fn add_addsub_noreduce(self, b: &Self, c: &Self)
        -> (GF255NotReduced<MQ>, GF255NotReduced<MQ>)
    {
        let d = self + b;
        let e = d - c;
        (d, e)
    }

    /// Return self - b and self - b + 2*c (no reduction).
    #[inline(always)]
    pub fn sub_subadd2_noreduce(self, b: &Self, c: &Self)
        -> (GF255NotReduced<MQ>, GF255NotReduced<MQ>)
    {
        let d = self - b;
        let e = d + c.mul2();
        (d, e)
    }

    #[inline]
    fn set_sub(&mut self, rhs: &Self) {
        // 1. Subtraction with borrow
        let (d0, cc) = subborrow_u64(self.0[0], rhs.0[0], 0);
        let (d1, cc) = subborrow_u64(self.0[1], rhs.0[1], cc);
        let (d2, cc) = subborrow_u64(self.0[2], rhs.0[2], cc);
        let (d3, cc) = subborrow_u64(self.0[3], rhs.0[3], cc);

        // 2. In case of an output borrow, add 2*q, i.e. subtract 2*MQ.
        let (d0, cc) = subborrow_u64(d0,
            (cc as u64).wrapping_neg() & (2 * MQ), 0);
        let (d1, cc) = subborrow_u64(d1, 0, cc);
        let (d2, cc) = subborrow_u64(d2, 0, cc);
        let (d3, cc) = subborrow_u64(d3, 0, cc);

        // 3. If there is again a borrow, then add 2*q again. In that case,
        // the low word must be at least 2^64 - 2*MQ, and the extra
        // subtraction won't trigger a new carry.
        let d0 = d0.wrapping_sub((cc as u64).wrapping_neg() & (2 * MQ));

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    // Negate this value (in place).
    #[inline]
    pub fn set_neg(&mut self) {
        // 1. Compute 2*q - self over 256 bits.
        let (d0, cc) = subborrow_u64((2 * MQ).wrapping_neg(), self.0[0], 0);
        let (d1, cc) = subborrow_u64(1u64.wrapping_neg(), self.0[1], cc);
        let (d2, cc) = subborrow_u64(1u64.wrapping_neg(), self.0[2], cc);
        let (d3, cc) = subborrow_u64(1u64.wrapping_neg(), self.0[3], cc);

        // 2. If the result is negative, add back q = 2^255 - MQ.
        let e = (cc as u64).wrapping_neg();
        let (d0, cc) = addcarry_u64(d0, e & MQ.wrapping_neg(), 0);
        let (d1, cc) = addcarry_u64(d1, e, cc);
        let (d2, cc) = addcarry_u64(d2, e, cc);
        let (d3, _) = addcarry_u64(d3, e >> 1, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    // Conditionally copy the provided value ('a') into self:
    //  - If ctl == 0xFFFFFFFF, then the value of 'a' is copied into self.
    //  - If ctl == 0, then the value of self is unchanged.
    // ctl MUST be equal to 0 or 0xFFFFFFFF.
    #[inline]
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
    #[inline]
    pub fn cswap(a: &mut Self, b: &mut Self, ctl: u32) {
        let cw = ((ctl as i32) as i64) as u64;
        let t = cw & (a.0[0] ^ b.0[0]); a.0[0] ^= t; b.0[0] ^= t;
        let t = cw & (a.0[1] ^ b.0[1]); a.0[1] ^= t; b.0[1] ^= t;
        let t = cw & (a.0[2] ^ b.0[2]); a.0[2] ^= t; b.0[2] ^= t;
        let t = cw & (a.0[3] ^ b.0[3]); a.0[3] ^= t; b.0[3] ^= t;
    }

    #[inline]
    fn set_half(&mut self) {
        // 1. Right-shift by 1 bit; keep dropped bit in 'tt' (expanded)
        let d0 = (self.0[0] >> 1) | (self.0[1] << 63);
        let d1 = (self.0[1] >> 1) | (self.0[2] << 63);
        let d2 = (self.0[2] >> 1) | (self.0[3] << 63);
        let d3 = self.0[3] >> 1;
        let tt = (self.0[0] & 1).wrapping_neg();

        // 2. If the dropped bit was 1, add back (q+1)/2. Since the value
        // was right-shifted, and (q+1)/2 < 2^255, this cannot overflow.
        let (d0, cc) = addcarry_u64(d0, tt & ((MQ - 1) >> 1).wrapping_neg(), 0);
        let (d1, cc) = addcarry_u64(d1, tt, cc);
        let (d2, cc) = addcarry_u64(d2, tt, cc);
        let (d3, _) = addcarry_u64(d3, tt >> 2, cc);

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

    // Multiply this value by 2.
    #[inline]
    pub fn set_mul2(&mut self) {
        // 1. Extract top bits.
        let tt = self.0[3] >> 62;

        // 2. Left-shift (also clearing the extracted bits).
        let d0 = self.0[0] << 1;
        let d1 = (self.0[0] >> 63) | (self.0[1] << 1);
        let d2 = (self.0[1] >> 63) | (self.0[2] << 1);
        let d3 = (self.0[2] >> 63) | ((self.0[3] << 1) & 0x7FFFFFFFFFFFFFFF);

        // 3. Add back the top bits with reduction. Since we extracted
        // one more bit than needed, this cannot overflow.
        let (d0, cc) = addcarry_u64(d0, tt * MQ, 0);
        let (d1, cc) = addcarry_u64(d1, 0, cc);
        let (d2, cc) = addcarry_u64(d2, 0, cc);
        let (d3, _)  = addcarry_u64(d3, 0, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    #[inline(always)]
    pub fn mul2(self) -> Self {
        let mut r = self;
        r.set_mul2();
        r
    }

    // Multiply this value by 4.
    #[inline]
    pub fn set_mul4(&mut self) {
        // 1. Extract top bits.
        let tt = self.0[3] >> 61;

        // 2. Left-shift (also clearing the extracted bits).
        let d0 = self.0[0] << 2;
        let d1 = (self.0[0] >> 62) | (self.0[1] << 2);
        let d2 = (self.0[1] >> 62) | (self.0[2] << 2);
        let d3 = (self.0[2] >> 62) | ((self.0[3] << 2) & 0x7FFFFFFFFFFFFFFF);

        // 3. Add back the top bits with reduction. Since we extracted
        // one more bit than needed, this cannot overflow.
        let (d0, cc) = addcarry_u64(d0, tt * MQ, 0);
        let (d1, cc) = addcarry_u64(d1, 0, cc);
        let (d2, cc) = addcarry_u64(d2, 0, cc);
        let (d3, _)  = addcarry_u64(d3, 0, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    #[inline(always)]
    pub fn mul4(self) -> Self {
        let mut r = self;
        r.set_mul4();
        r
    }

    // Multiply this value by 8.
    #[inline]
    pub fn set_mul8(&mut self) {
        // 1. Extract top bits.
        let tt = self.0[3] >> 60;

        // 2. Left-shift (also clearing the extracted bits).
        let d0 = self.0[0] << 3;
        let d1 = (self.0[0] >> 61) | (self.0[1] << 3);
        let d2 = (self.0[1] >> 61) | (self.0[2] << 3);
        let d3 = (self.0[2] >> 61) | ((self.0[3] << 3) & 0x7FFFFFFFFFFFFFFF);

        // 3. Add back the top bits with reduction. Since we extracted
        // one more bit than needed, this cannot overflow.
        let (d0, cc) = addcarry_u64(d0, tt * MQ, 0);
        let (d1, cc) = addcarry_u64(d1, 0, cc);
        let (d2, cc) = addcarry_u64(d2, 0, cc);
        let (d3, _)  = addcarry_u64(d3, 0, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    #[inline(always)]
    pub fn mul8(self) -> Self {
        let mut r = self;
        r.set_mul8();
        r
    }

    // Multiply this value by 16.
    #[inline]
    pub fn set_mul16(&mut self) {
        // 1. Extract top bits.
        let tt = self.0[3] >> 59;

        // 2. Left-shift (also clearing the extracted bits).
        let d0 = self.0[0] << 4;
        let d1 = (self.0[0] >> 60) | (self.0[1] << 4);
        let d2 = (self.0[1] >> 60) | (self.0[2] << 4);
        let d3 = (self.0[2] >> 60) | ((self.0[3] << 4) & 0x7FFFFFFFFFFFFFFF);

        // 3. Add back the top bits with reduction. Since we extracted
        // one more bit than needed, this cannot overflow.
        let (d0, cc) = addcarry_u64(d0, tt * MQ, 0);
        let (d1, cc) = addcarry_u64(d1, 0, cc);
        let (d2, cc) = addcarry_u64(d2, 0, cc);
        let (d3, _)  = addcarry_u64(d3, 0, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    #[inline(always)]
    pub fn mul16(self) -> Self {
        let mut r = self;
        r.set_mul16();
        r
    }

    // Multiply this value by 32.
    #[inline]
    pub fn set_mul32(&mut self) {
        // 1. Extract top bits.
        let tt = self.0[3] >> 58;

        // 2. Left-shift (also clearing the extracted bits).
        let d0 = self.0[0] << 5;
        let d1 = (self.0[0] >> 59) | (self.0[1] << 5);
        let d2 = (self.0[1] >> 59) | (self.0[2] << 5);
        let d3 = (self.0[2] >> 59) | ((self.0[3] << 5) & 0x7FFFFFFFFFFFFFFF);

        // 3. Add back the top bits with reduction. Since we extracted
        // one more bit than needed, this cannot overflow.
        let (d0, cc) = addcarry_u64(d0, tt * MQ, 0);
        let (d1, cc) = addcarry_u64(d1, 0, cc);
        let (d2, cc) = addcarry_u64(d2, 0, cc);
        let (d3, _)  = addcarry_u64(d3, 0, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    #[inline(always)]
    pub fn mul32(self) -> Self {
        let mut r = self;
        r.set_mul32();
        r
    }

    // Multiply this value by a small integer.
    #[inline]
    pub fn set_mul_small(&mut self, x: u32) {
        let b = x as u64;

        // Compute the product as an integer over five words.
        // Max value is (2^32 - 1)*(2^256 - 1), so the top word (d4) is
        // at most 2^32 - 2.
        let (d0, d1) = umull(self.0[0], b);
        let (d2, d3) = umull(self.0[2], b);
        let (lo, hi) = umull(self.0[1], b);
        let (d1, cc) = addcarry_u64(d1, lo, 0);
        let (d2, cc) = addcarry_u64(d2, hi, cc);
        let (lo, d4) = umull(self.0[3], b);
        let (d3, cc) = addcarry_u64(d3, lo, cc);
        let (d4, _)  = addcarry_u64(d4, 0, cc);

        // Do the reduction by folding the top word (d4) _and_ the top bit
        // of the previous word (d3). Since that frees up the top bit, only
        // one pass is needed.
        // Maximum fold value is (2^33 - 3)*MQ, which fits on 64 bits as
        // long as MQ <= 2^31.
        let d4 = ((d4 << 1) | (d3 >> 63)) * MQ;
        let (d0, cc) = addcarry_u64(d0, d4, 0);
        let (d1, cc) = addcarry_u64(d1, 0, cc);
        let (d2, cc) = addcarry_u64(d2, 0, cc);
        let (d3, _)  = addcarry_u64(d3 & 0x7FFFFFFFFFFFFFFF, 0, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    #[inline(always)]
    pub fn mul_small(self, x: u32) -> Self {
        let mut r = self;
        r.set_mul_small(x);
        r
    }

    #[inline(always)]
    fn set_mul(&mut self, rhs: &Self) {
        let (a0, a1, a2, a3) = (self.0[0], self.0[1], self.0[2], self.0[3]);
        let (b0, b1, b2, b3) = (rhs.0[0], rhs.0[1], rhs.0[2], rhs.0[3]);

        // 1. Product -> 512 bits
        let (e0, e1) = umull(a0, b0);
        let (e2, e3) = umull(a1, b1);
        let (e4, e5) = umull(a2, b2);
        let (e6, e7) = umull(a3, b3);

        let (lo, hi) = umull(a0, b1);
        let (e1, cc) = addcarry_u64(e1, lo, 0);
        let (e2, cc) = addcarry_u64(e2, hi, cc);
        let (lo, hi) = umull(a0, b3);
        let (e3, cc) = addcarry_u64(e3, lo, cc);
        let (e4, cc) = addcarry_u64(e4, hi, cc);
        let (lo, hi) = umull(a2, b3);
        let (e5, cc) = addcarry_u64(e5, lo, cc);
        let (e6, cc) = addcarry_u64(e6, hi, cc);
        let (e7, _)  = addcarry_u64(e7, 0, cc);

        let (lo, hi) = umull(a1, b0);
        let (e1, cc) = addcarry_u64(e1, lo, 0);
        let (e2, cc) = addcarry_u64(e2, hi, cc);
        let (lo, hi) = umull(a3, b0);
        let (e3, cc) = addcarry_u64(e3, lo, cc);
        let (e4, cc) = addcarry_u64(e4, hi, cc);
        let (lo, hi) = umull(a3, b2);
        let (e5, cc) = addcarry_u64(e5, lo, cc);
        let (e6, cc) = addcarry_u64(e6, hi, cc);
        let (e7, _)  = addcarry_u64(e7, 0, cc);

        let (lo, hi) = umull(a0, b2);
        let (e2, cc) = addcarry_u64(e2, lo, 0);
        let (e3, cc) = addcarry_u64(e3, hi, cc);
        let (lo, hi) = umull(a1, b3);
        let (e4, cc) = addcarry_u64(e4, lo, cc);
        let (e5, cc) = addcarry_u64(e5, hi, cc);
        let (e6, cc) = addcarry_u64(e6, 0, cc);
        let (e7, _)  = addcarry_u64(e7, 0, cc);

        let (lo, hi) = umull(a2, b0);
        let (e2, cc) = addcarry_u64(e2, lo, 0);
        let (e3, cc) = addcarry_u64(e3, hi, cc);
        let (lo, hi) = umull(a3, b1);
        let (e4, cc) = addcarry_u64(e4, lo, cc);
        let (e5, cc) = addcarry_u64(e5, hi, cc);
        let (e6, cc) = addcarry_u64(e6, 0, cc);
        let (e7, _)  = addcarry_u64(e7, 0, cc);

        let (lo, hi) = umull(a1, b2);
        let (lo2, hi2) = umull(a2, b1);
        let (lo, cc) = addcarry_u64(lo, lo2, 0);
        let (hi, tt) = addcarry_u64(hi, hi2, cc);
        let (e3, cc) = addcarry_u64(e3, lo, 0);
        let (e4, cc) = addcarry_u64(e4, hi, cc);
        let (e5, cc) = addcarry_u64(e5, tt as u64, cc);
        let (e6, cc) = addcarry_u64(e6, 0, cc);
        let (e7, _)  = addcarry_u64(e7, 0, cc);

        // 2. Reduction
        // We fold the upper words in two steps; first step adds the
        // low words of the multiplication by 2*MQ, while high words
        // of these products are kept in h0..h3.
        let (lo, h0) = umull(e4, 2 * MQ);
        let (e0, cc) = addcarry_u64(e0, lo, 0);
        let (lo, h1) = umull(e5, 2 * MQ);
        let (e1, cc) = addcarry_u64(e1, lo, cc);
        let (lo, h2) = umull(e6, 2 * MQ);
        let (e2, cc) = addcarry_u64(e2, lo, cc);
        let (lo, h3) = umull(e7, 2 * MQ);
        let (e3, cc) = addcarry_u64(e3, lo, cc);
        let (h3, _)  = addcarry_u64(h3, 0, cc);

        // Max value for h3 is 1 + floor(2*MQ*(2^64 - 1) / 2^64).
        // We then compute (2*h3 + b)*MQ, with b being the top bit of e3
        // (i.e. b = 0 or 1). This value fits on 64 bits as long as
        // MQ <= 2^31 - 1.

        let h3 = (h3 << 1) | (e3 >> 63);
        let e3 = e3 & 0x7FFFFFFFFFFFFFFF;
        let (e0, cc) = addcarry_u64(e0, h3 * MQ, 0);
        let (e1, cc) = addcarry_u64(e1, h0, cc);
        let (e2, cc) = addcarry_u64(e2, h1, cc);
        let (e3, _)  = addcarry_u64(e3, h2, cc);

        self.0[0] = e0;
        self.0[1] = e1;
        self.0[2] = e2;
        self.0[3] = e3;
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

        // 3. Reduction.
        // See set_mul() for comments on the range; this is the same
        // reduction.
        let (lo, h0) = umull(e4, 2 * MQ);
        let (e0, cc) = addcarry_u64(e0, lo, 0);
        let (lo, h1) = umull(e5, 2 * MQ);
        let (e1, cc) = addcarry_u64(e1, lo, cc);
        let (lo, h2) = umull(e6, 2 * MQ);
        let (e2, cc) = addcarry_u64(e2, lo, cc);
        let (lo, h3) = umull(e7, 2 * MQ);
        let (e3, cc) = addcarry_u64(e3, lo, cc);
        let (h3, _)  = addcarry_u64(h3, 0, cc);

        let h3 = (h3 << 1) | (e3 >> 63);
        let e3 = e3 & 0x7FFFFFFFFFFFFFFF;
        let (e0, cc) = addcarry_u64(e0, h3 * MQ, 0);
        let (e1, cc) = addcarry_u64(e1, h0, cc);
        let (e2, cc) = addcarry_u64(e2, h1, cc);
        let (e3, _)  = addcarry_u64(e3, h2, cc);

        self.0[0] = e0;
        self.0[1] = e1;
        self.0[2] = e2;
        self.0[3] = e3;
    }

    // Square this value.
    #[inline(always)]
    pub fn square(self) -> Self {
        let mut r = self;
        r.set_square();
        r
    }

    // Square this value n times (in place).
    #[inline(always)]
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

    // Ensure that the internal encoding of this value is in the 0..q-1
    // range.
    #[inline]
    fn set_normalized(&mut self) {
        // Propagate top bit if set.
        let e = (self.0[3] >> 63).wrapping_neg();
        let (d0, cc) = addcarry_u64(self.0[0], e & MQ, 0);
        let (d1, cc) = addcarry_u64(self.0[1], 0, cc);
        let (d2, cc) = addcarry_u64(self.0[2], 0, cc);
        let (d3, _)  = addcarry_u64(self.0[3] & 0x7FFFFFFFFFFFFFFF, 0, cc);

        // Value is now at most 2^255 + MQ - 1. Subtract q, then add it
        // back in case the result would be negative.
        let (d0, cc) = subborrow_u64(d0, MQ.wrapping_neg(), 0);
        let (d1, cc) = subborrow_u64(d1, !0u64, cc);
        let (d2, cc) = subborrow_u64(d2, !0u64, cc);
        let (d3, cc) = subborrow_u64(d3, (!0u64) >> 1, cc);

        let e = (cc as u64).wrapping_neg();
        let (d0, cc) = addcarry_u64(d0, e & MQ.wrapping_neg(), 0);
        let (d1, cc) = addcarry_u64(d1, e, cc);
        let (d2, cc) = addcarry_u64(d2, e, cc);
        let (d3, _)  = addcarry_u64(d3, e >> 1, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    // Set this value to u*f+v*g (with 'u' being self). Parameters f and g
    // are provided as u64, but they are signed integers in the -2^62..+2^62
    // range.
    #[inline]
    fn set_lin(&mut self, u: &Self, v: &Self, f: u64, g: u64) {
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
        let (d3, t) = umull_x2_add(tu.0[3], f, tv.0[3], g, t);

        // Upper word t can be up to 63 bits.
        let (lo, hi) = umull(t, 2 * MQ);
        let (d0, cc) = addcarry_u64(d0, lo, 0);
        let (d1, cc) = addcarry_u64(d1, hi, cc);
        let (d2, cc) = addcarry_u64(d2, 0, cc);
        let (d3, cc) = addcarry_u64(d3, 0, cc);

        // If there is a carry, then current value is lower than
        // 2 * MQ * 2^63, and the folding cannot propagate beyond the
        // second limb.
        let (d0, cc) = addcarry_u64(d0,
            (cc as u64).wrapping_neg() & (2 * MQ), 0);
        let (d1, _)  = addcarry_u64(d1, 0, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    #[inline(always)]
    fn lin(a: &Self, b: &Self, f: u64, g: u64) -> Self {
        let mut r = Self::ZERO;
        r.set_lin(a, b, f, g);
        r
    }

    // Set this value to abs((a*f+b*g)/2^31). Values a and b are interpreted
    // as signed 256-bit integers. Coefficients f and g are provided as u64,
    // but they really are signed integers in the -2^31..+2^31 range
    // (inclusive). The low 31 bits are dropped (i.e. the division is assumed
    // to be exact). The result is assumed to fit in 256 bits (including the
    // sign bit) (otherwise, truncation occurs).
    //
    // Returned value is -1 (u64) if (a*f+b*g) was negative, 0 otherwise.
    #[inline]
    fn set_lindiv31abs(&mut self, a: &Self, b: &Self, f: u64, g: u64) -> u64 {
        // Replace f and g with abs(f) and abs(g), but remember the
        // original signs.
        let sf = sgnw(f);
        let f = (f ^ sf).wrapping_sub(sf);
        let sg = sgnw(g);
        let g = (g ^ sg).wrapping_sub(sg);

        // Apply the signs of f and g to the source operands.
        let (a0, cc) = subborrow_u64(a.0[0] ^ sf, sf, 0);
        let (a1, cc) = subborrow_u64(a.0[1] ^ sf, sf, cc);
        let (a2, cc) = subborrow_u64(a.0[2] ^ sf, sf, cc);
        let (a3, _)  = subborrow_u64(a.0[3] ^ sf, sf, cc);
        let (b0, cc) = subborrow_u64(b.0[0] ^ sg, sg, 0);
        let (b1, cc) = subborrow_u64(b.0[1] ^ sg, sg, cc);
        let (b2, cc) = subborrow_u64(b.0[2] ^ sg, sg, cc);
        let (b3, _)  = subborrow_u64(b.0[3] ^ sg, sg, cc);

        // Compute a*f+b*g into d0:d1:d2:d3:t. Since f and g are at
        // most 2^31, we can add two 128-bit products with no overflow.
        let (d0, t) = umull_x2(a0, f, b0, g);
        let (d1, t) = umull_x2_add(a1, f, b1, g, t);
        let (d2, t) = umull_x2_add(a2, f, b2, g, t);
        let (d3, t) = umull_x2_add(a3, f, b3, g, t);

        // If a < 0, then the result is overestimated by f*2^256;
        // similarly, if b < 0 then the result is overestimated by g*2^256.
        // We must thus subtract 2^256*(sa*f+sb*g), with sa and sb being
        // the signs of a and b, respectively (1 for negative, 0 otherwise).
        let t = t.wrapping_sub(f & sgnw(a3));
        let t = t.wrapping_sub(g & sgnw(b3));

        // Shift-right the value by 31 bits.
        let d0 = (d0 >> 31) | (d1 << 33);
        let d1 = (d1 >> 31) | (d2 << 33);
        let d2 = (d2 >> 31) | (d3 << 33);
        let d3 = (d3 >> 31) | (t << 33);

        // If the result is negative, then negate it.
        let t = (t >> 63).wrapping_neg();
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
        //   b <- q (modulus)
        //   u <- x (self)
        //   v <- 0
        //
        // Value a is normalized (in the 0..q-1 range). Values a and b are
        // then considered as (signed) integers. Values u and v are field
        // elements.
        //
        // Invariants:
        //    a*x = y*u mod q
        //    b*x = y*v mod q
        //    b is always odd
        //
        // At each step:
        //    if a is even, then:
        //        a <- a/2, u <- u/2 mod q
        //    else:
        //        if a < b:
        //            (a, u, b, v) <- (b, v, a, u)
        //        a <- (a-b)/2, u <- (u-v)/2 mod q
        //
        // What we implement below is the optimized version of this
        // algorithm, as described in https://eprint.iacr.org/2020/972

        let mut a = *y;
        a.set_normalized();
        let mut b = Self([
            MQ.wrapping_neg(),
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF,
        ]);
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
            let nu = Self::lin(&u, &v, f0, g0);
            let nv = Self::lin(&u, &v, f1, g1);
            a = na;
            b = nb;
            u = nu;
            v = nv;
        }

        // If y is invertible, then the final GCD is 1, and
        // len(a) + len(b) <= 45, so we can end the computation with
        // the low words directly. We only need 43 iterations to reach
        // the point where b = 1.
        //
        // If y is zero, then v is unchanged (hence zero) and none of
        // the subsequent iterations will change it either, so we get
        // 0 on output, which is what we want.
        let mut xa = a.0[0];
        let mut xb = b.0[0];
        let mut f0 = 1u64;
        let mut g0 = 0u64;
        let mut f1 = 0u64;
        let mut g1 = 1u64;
        for _ in 0..43 {
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

        self.set_lin(&u, &v, f1, g1);

        // At this point, we have injected extra factors of 2, one for
        // each of the 31*15+43 = 508 iterations, so we must divide by
        // 2^508 (mod q). This is done with a multiplication by the
        // appropriate constant.
        self.set_mul(&Self::INVT508);
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
        // in a total of 508 iterations.

        let mut a = self;
        a.set_normalized();
        let mut b = Self([
            MQ.wrapping_neg(),
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF,
        ]);
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

        // Final iterations: values are at most 45 bits now. We do not
        // need to keep track of update coefficients. Just like the GCD,
        // we need only 43 iterations, because after 43 iterations,
        // value a is 0 or 1, and b is 1, and no further modification to
        // the Legendre symbol may happen.
        let mut xa = a.0[0];
        let mut xb = b.0[0];
        for _ in 0..43 {
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

    // Set this value to its square root. Returned value is 0xFFFFFFFF
    // if the operation succeeded (value was indeed a quadratic
    // residue), 0 otherwise (value was not a quadratic residue). In the
    // latter case, this value is set to the square root of -self (if
    // q = 3 mod 4) or of either 2*self or -2*self (if q = 5 mod 8). In
    // all cases, the returned root is the one whose least significant
    // bit is 0 (when normalized in 0..q-1).
    //
    // This operation returns unspecified results if the modulus is not
    // prime. If the modulus q is prime but is equal to 1 modulo 8, then
    // the method is not implemented (which triggers a panic).
    fn set_sqrt_ext(&mut self) -> u32 {
        // We can support only q = 3, 5 or 7 mod 8, not q = 1 mod 8.
        // See compile_time_checks() for the compile-time verification
        // that MQ matches that restriction.

        // Input is denoted x in code comments.

        // In both cases (q = 3 mod 4 and q = 5 mod 8), we need to compute
        // a modular exponentiation, and the exponent's top 240 bits are
        // all equal to one, so we have a common part to compute
        // z^(2^240-1), with z being the exponentiated value. We also
        // obtain z^2 and z^3 in the process, which we store in a 2-bit
        // window for the end of the exponentiation.

        // Base value is x if q = 3 mod 4, 2*x if q = 5 mod 8.
        let z = if (MQ & 3) == 1 { *self } else { (*self).mul2() };
        let z2 = z.square();
        let z3 = z2 * z;
        let zp4 = z3.xsquare(2) * z3;
        let zp5 = zp4.square() * z;
        let zp15 = (zp5.xsquare(5) * zp5).xsquare(5) * zp5;
        let zp30 = zp15.xsquare(15) * zp15;
        let zp60 = zp30.xsquare(30) * zp30;
        let zp120 = zp60.xsquare(60) * zp60;
        let zp240 = zp120.xsquare(120) * zp120;
        let win: [Self; 3] = [ z, z2, z3 ];

        // Candidate square root goes in y.
        let mut y = zp240;

        if (MQ & 3) == 1 {
            // q = 3 mod 4; square root candidate is computed as:
            //   y <- x^((q+1)/4)
            // We need to process 13 extra exponent bits.
            let e = MQ.wrapping_neg().wrapping_add(1) >> 2;
            for i in 0..6 {
                y.set_xsquare(2);
                let k = ((e >> (11 - (2 * i))) & 3) as usize;
                if k != 0 {
                    y.set_mul(&win[k - 1]);
                }
            }
            y.set_square();
            if (e & 1) != 0 {
                y.set_mul(&z);
            }
        } else if (MQ & 7) == 3 {
            // q = 5 mod 8; we use Atkin's algorithm:
            //   b <- (2*x)^((q-5)/8)
            //   c <- 2*x*b^2
            //   y <- x*b*(c - 1)
            let e = MQ.wrapping_neg().wrapping_sub(5) >> 3;
            let mut b = y;
            for i in 0..6 {
                b.set_xsquare(2);
                let k = ((e >> (10 - (2 * i))) & 3) as usize;
                if k != 0 {
                    b.set_mul(&win[k - 1]);
                }
            }

            // Compute c = 2*x*b^2.
            let c = self.mul2() * b.square();

            // We really computed c = (2*x)^((q-1)/4), which is a square
            // root of the Legendre symbol of 2*x. With q = 5 mod 8, 2 is
            // not a square. Thus, if the square root of x exists, then c is
            // a square root of -1 (except if x = 0, in which case c = 0).
            // Otherwise, c = 1 or -1.
            // We compute y = x*b*(c' - 1); then:
            //   y^2 = x*c*(c' - 1)^2/2
            // If c = i or -i, then using c = c' (as mandated by Atkin's
            // formulas) yields c*(c - 1)^2/2 = 1, i.e. y^2 = x, which is
            // the expected result.
            // If c = 1 or -1, then we set c' = 3, so that c*(c' - 1)^2/2
            // is equal to 2 or -2, and y^2 = 2*x or -2*x.
            let mut cp = c;
            let ff = c.equals(Self::ONE) | c.equals(Self::MINUS_ONE);
            cp.set_cond(&Self::w64le(3, 0, 0, 0), ff);
            y = (*self) * b * (cp - Self::ONE);
        } else {
            // General case is Tonelli-Shanks but it requires knowledge
            // of a non-QR in the field, which we don't provide in the
            // type parameters.
            unimplemented!();
        }

        // Normalize y and negate it if necessary to set the low bit to 0.
        y.set_normalized();
        y.set_cond(&-y, ((y.0[0] as u32) & 1).wrapping_neg());

        // Check that the candidate is indeed a square root.
        let r = y.square().equals(*self);
        *self = y;
        r
    }

    // Set this value to its square root. Returned value is 0xFFFFFFFF
    // if the operation succeeded (value was indeed a quadratic
    // residue), 0 otherwise (value was not a quadratic residue). This
    // differs from set_sqrt_ext() in that this function sets the value
    // to zero if there is no square root.
    fn set_sqrt(&mut self) -> u32 {
        let r = self.set_sqrt_ext();
        self.set_cond(&Self::ZERO, !r);
        r
    }

    // Compute the square root of this value. Returned value are (y, r):
    //  - If this value is indeed a quadratic residue, then y is the
    //    square root whose least significant bit (when normalized in 0..q-1)
    //    is 0, and r is equal to 0xFFFFFFFF.
    //  - If this value is not a quadratic residue, then y is zero, and
    //    r is equal to 0.
    #[inline(always)]
    pub fn sqrt(self) -> (Self, u32) {
        let mut x = self;
        let r = x.set_sqrt();
        (x, r)
    }

    // Compute the square root of this value. Returned value are (y, r):
    //  - If this value is indeed a quadratic residue, then y is a
    //    square root of this value, and r is 0xFFFFFFFF.
    //  - If this value is not a quadratic residue, then y is set to
    //    a square root of -x (if modulus q = 3 mod 4), or to a square
    //    root of either 2*x or -2*x (if modulus q = 5 mod 8); morever,
    //    r is set to 0x00000000.
    // In all cases, the returned root is normalized: the lest significant
    // bit of its integer representation (in the 0..q-1 range) is 0.
    #[inline(always)]
    pub fn sqrt_ext(self) -> (Self, u32) {
        let mut x = self;
        let r = x.set_sqrt_ext();
        (x, r)
    }

    // Compute two signed integers (c0, c1) such that this self = c0/c1 in
    // the ring. WARNING: since the modulus is close to 2^255, and larger
    // than about 1.73*2^253, the returned values may be truncated. Indeed,
    // it can be shown with modulus q close to 2^255, then the coordinates
    // c0 and c1 of the minimal-sized solution will be lower than 1.52*2^127
    // in absolute value. Thus, if this function returns two signed integers
    // c0 and c1, then it must be that:
    //    self = (c0 + a*2^128) / (c1 + b*2^128)
    // for two integers a and b which are both in { -1, 0, +1 }. It is
    // up to the caller to enumerate and test the possible solutions.
    //
    // If this element is zero, then this function returns (0, 1). Otherwise,
    // neither c0 nor c1 can be zero.
    //
    // THIS FUNCTION IS NOT CONSTANT-TIME. It shall be used only for a
    // public source element.
    pub fn split_vartime(self) -> (i128, i128) {
        let mut k = self;
        k.set_normalized();
        lagrange253_vartime(&k.0, &Self::MODULUS)
    }

    // Equality check between two field elements (constant-time);
    // returned value is 0xFFFFFFFF on equality, 0 otherwise.
    #[inline(always)]
    pub fn equals(self, rhs: Self) -> u32 {
        (self - rhs).iszero()
    }

    // Compare this value with zero (constant-time); returned value
    // is 0xFFFFFFFF if this element is zero, 0 otherwise.
    #[inline]
    pub fn iszero(self) -> u32 {
        // Since values are over 256 bits, there are three possible
        // representations for 0: 0, q amnd 2*q.
        let a0 = self.0[0];
        let a1 = self.0[1];
        let a2 = self.0[2];
        let a3 = self.0[3];
        let t0 = a0 | a1 | a2 | a3;
        let t1 = a0.wrapping_add(MQ) | !a1 | !a2 | (a3 ^ 0x7FFFFFFFFFFFFFFF);
        let t2 = a0.wrapping_add(2 * MQ) | !a1 | !a2 | !a3;

        // Top bit of r is 0 if and only if one of t0, t1 or t2 is zero.
        let r = (t0 | t0.wrapping_neg())
              & (t1 | t1.wrapping_neg())
              & (t2 | t2.wrapping_neg());
        ((r >> 63) as u32).wrapping_sub(1)
    }

    /* unused
    #[inline(always)]
    fn decode32_reduce(buf: &[u8]) -> Self {
        let mut r = Self::ZERO;
        if buf.len() == 32 {
            r.set_decode32_reduce(buf);
        }
        r
    }
    */

    #[inline(always)]
    fn set_decode32_reduce(&mut self, buf: &[u8]) {
        debug_assert!(buf.len() == 32);
        self.0[0] = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[ 0.. 8]).unwrap());
        self.0[1] = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[ 8..16]).unwrap());
        self.0[2] = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[16..24]).unwrap());
        self.0[3] = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[24..32]).unwrap());
    }

    // Encode this value over exactly 32 bytes. Encoding is always canonical
    // (little-endian encoding of the value in the 0..q-1 range, top bit
    // of the last byte is always 0).
    #[inline(always)]
    pub fn encode32(self) -> [u8; 32] {
        let mut r = self;
        r.set_normalized();
        let mut d = [0u8; 32];
        d[ 0.. 8].copy_from_slice(&r.0[0].to_le_bytes());
        d[ 8..16].copy_from_slice(&r.0[1].to_le_bytes());
        d[16..24].copy_from_slice(&r.0[2].to_le_bytes());
        d[24..32].copy_from_slice(&r.0[3].to_le_bytes());
        d
    }

    // Encode this value over exactly 32 bytes. Encoding is always canonical
    // (little-endian encoding of the value in the 0..q-1 range, top bit
    // of the last byte is always 0).
    #[inline(always)]
    pub fn encode(self) -> [u8; 32] {
        self.encode32()
    }

    // Decode the field element from the provided bytes. If the source
    // slice does not have length exactly 32 bytes, or if the encoding
    // is non-canonical (i.e. does not represent an integer in the 0
    // to q-1 range), then this element is set to zero, and 0 is returned.
    // Otherwise, this element is set to the decoded value, and 0xFFFFFFFF
    // is returned.
    #[inline]
    pub fn set_decode_ct(&mut self, buf: &[u8]) -> u32 {
        if buf.len() != 32 {
            *self = Self::ZERO;
            return 0;
        }

        self.set_decode32_reduce(buf);

        // Try to subtract q from the value; if that does not yield a
        // borrow, then the encoding was not canonical.
        let (_, cc) = subborrow_u64(self.0[0], MQ.wrapping_neg(), 0);
        let (_, cc) = subborrow_u64(self.0[1], !0u64, cc);
        let (_, cc) = subborrow_u64(self.0[2], !0u64, cc);
        let (_, cc) = subborrow_u64(self.0[3], (!0u64) >> 1, cc);

        // Clear the value if not canonical.
        let cc = (cc as u64).wrapping_neg();
        self.0[0] &= cc;
        self.0[1] &= cc;
        self.0[2] &= cc;
        self.0[3] &= cc;

        cc as u32
    }

    // Decode a field element from 32 bytes. On success, this returns
    // (r, cc), where cc has value 0xFFFFFFFF. If the source encoding is not
    // canonical (i.e. the unsigned little-endian interpretation of the
    // 32 bytes yields an integer with is not lower than q), then this
    // returns (0, 0).
    #[inline(always)]
    pub fn decode_ct(buf: &[u8]) -> (Self, u32) {
        let mut r = Self::ZERO;
        let cc = r.set_decode_ct(buf);
        (r, cc)
    }

    // Decode a field element from 32 bytes. On success, this returns
    // (r, cc), where cc has value 0xFFFFFFFF. If the source encoding is not
    // canonical (i.e. the unsigned little-endian interpretation of the
    // 32 bytes yields an integer with is not lower than q), then this
    // returns (0, 0).
    #[inline]
    pub fn decode32(buf: &[u8]) -> (Self, u32) {
        Self::decode_ct(buf)
    }

    // Decode a field element from 32 bytes. If the source slice has length
    // exactly 32 bytes and contains a valid canonical encoding of a field
    // element, then that element is returned. Otherwise, `None` is
    // returned. Side-channel analysis may reveal to outsiders whether the
    // decoding succeeded.
    #[inline(always)]
    pub fn decode(buf: &[u8]) -> Option<Self> {
        let (r, cc) = Self::decode32(buf);
        if cc != 0 {
            Some(r)
        } else {
            None
        }
    }

    // Decode a field element from some bytes. The bytes are interpreted
    // in unsigned little-endian convention, and the resulting integer
    // is reduced modulo q. This process never fails.
    pub fn set_decode_reduce(&mut self, buf: &[u8]) {
        *self = Self::ZERO;
        let mut n = buf.len();
        if n == 0 {
            return;
        }
        if (n & 31) != 0 {
            let k = n & !(31 as usize);
            let mut tmp = [0u8; 32];
            tmp[..(n - k)].copy_from_slice(&buf[k..]);
            n = k;
            self.set_decode32_reduce(&tmp);
        } else {
            n -= 32;
            self.set_decode32_reduce(&buf[n..]);
        }

        while n > 0 {
            let k = n - 32;
            let e0 = u64::from_le_bytes(*<&[u8; 8]>
                ::try_from(&buf[k     ..k +  8]).unwrap());
            let e1 = u64::from_le_bytes(*<&[u8; 8]>
                ::try_from(&buf[k +  8..k + 16]).unwrap());
            let e2 = u64::from_le_bytes(*<&[u8; 8]>
                ::try_from(&buf[k + 16..k + 24]).unwrap());
            let e3 = u64::from_le_bytes(*<&[u8; 8]>
                ::try_from(&buf[k + 24..k + 32]).unwrap());
            let (d0, h0) = umull(self.0[0], 2 * MQ);
            let (d0, cc) = addcarry_u64(d0, e0, 0);
            let (d1, h1) = umull(self.0[1], 2 * MQ);
            let (d1, cc) = addcarry_u64(d1, e1, cc);
            let (d2, h2) = umull(self.0[2], 2 * MQ);
            let (d2, cc) = addcarry_u64(d2, e2, cc);
            let (d3, h3) = umull(self.0[3], 2 * MQ);
            let (d3, cc) = addcarry_u64(d3, e3, cc);
            let (h3, _)  = addcarry_u64(h3, 0, cc);

            let h3 = (h3 << 1) | (d3 >> 63);
            let (d0, cc) = addcarry_u64(d0, h3 * MQ, 0);
            let (d1, cc) = addcarry_u64(d1, h0, cc);
            let (d2, cc) = addcarry_u64(d2, h1, cc);
            let (d3, _)  = addcarry_u64(d3 & 0x7FFFFFFFFFFFFFFF, h2, cc);

            self.0[0] = d0;
            self.0[1] = d1;
            self.0[2] = d2;
            self.0[3] = d3;

            n = k;
        }
    }

    // Decode a field element from some bytes. The bytes are interpreted
    // in unsigned little-endian convention, and the resulting integer
    // is reduced modulo q. This process never fails.
    #[inline(always)]
    pub fn decode_reduce(buf: &[u8]) -> Self {
        let mut r = Self::ZERO;
        r.set_decode_reduce(buf);
        r
    }

    // This function computes a representation of 1/2^508 at compile time.
    // It is not meant to be used at runtime and has no constant-time
    // requirement.
    const fn make_invt508() -> Self {

        const fn adc(x: u64, y: u64, c: u64) -> (u64, u64) {
            let z = (x as u128).wrapping_add(y as u128).wrapping_add(c as u128);
            (z as u64, (z >> 64) as u64)
        }

        const fn sqr<const MQ: u64>(a: GF255<MQ>) -> GF255<MQ> {
            // This follows the same steps as the runtime set_square().
            let (a0, a1, a2, a3) = (a.0[0], a.0[1], a.0[2], a.0[3]);

            // 1. Non-square products.
            let (e1, e2) = umull(a0, a1);
            let (e3, e4) = umull(a0, a3);
            let (e5, e6) = umull(a2, a3);
            let (lo, hi) = umull(a0, a2);
            let (e2, cc) = adc(e2, lo, 0);
            let (e3, cc) = adc(e3, hi, cc);
            let (lo, hi) = umull(a1, a3);
            let (e4, cc) = adc(e4, lo, cc);
            let (e5, cc) = adc(e5, hi, cc);
            let (e6, _)  = adc(e6, 0, cc);
            let (lo, hi) = umull(a1, a2);
            let (e3, cc) = adc(e3, lo, 0);
            let (e4, cc) = adc(e4, hi, cc);
            let (e5, cc) = adc(e5, 0, cc);
            let (e6, _)  = adc(e6, 0, cc);

            // 2. Double the intermediate value, then add the squares.
            let e7 = e6 >> 63;
            let e6 = (e6 << 1) | (e5 >> 63);
            let e5 = (e5 << 1) | (e4 >> 63);
            let e4 = (e4 << 1) | (e3 >> 63);
            let e3 = (e3 << 1) | (e2 >> 63);
            let e2 = (e2 << 1) | (e1 >> 63);
            let e1 = e1 << 1;

            let (e0, hi) = umull(a0, a0);
            let (e1, cc) = adc(e1, hi, 0);
            let (lo, hi) = umull(a1, a1);
            let (e2, cc) = adc(e2, lo, cc);
            let (e3, cc) = adc(e3, hi, cc);
            let (lo, hi) = umull(a2, a2);
            let (e4, cc) = adc(e4, lo, cc);
            let (e5, cc) = adc(e5, hi, cc);
            let (lo, hi) = umull(a3, a3);
            let (e6, cc) = adc(e6, lo, cc);
            let (e7, _)  = adc(e7, hi, cc);

            // 3. Reduction.
            let (lo, h0) = umull(e4, 2 * MQ);
            let (e0, cc) = adc(e0, lo, 0);
            let (lo, h1) = umull(e5, 2 * MQ);
            let (e1, cc) = adc(e1, lo, cc);
            let (lo, h2) = umull(e6, 2 * MQ);
            let (e2, cc) = adc(e2, lo, cc);
            let (lo, h3) = umull(e7, 2 * MQ);
            let (e3, cc) = adc(e3, lo, cc);
            let (h3, _)  = adc(h3, 0, cc);

            let h3 = (h3 << 1) | (e3 >> 63);
            let e3 = e3 & 0x7FFFFFFFFFFFFFFF;
            let (e0, cc) = adc(e0, h3 * MQ, 0);
            let (e1, cc) = adc(e1, h0, cc);
            let (e2, cc) = adc(e2, h1, cc);
            let (e3, _)  = adc(e3, h2, cc);

            GF255::<MQ>([ e0, e1, e2, e3 ])
        }

        // 1/2 = (q + 1)/2 mod q
        let a = Self([
            ((MQ - 1) >> 1).wrapping_neg(),
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0x3FFFFFFFFFFFFFFF,
        ]);

        // square 9 times to get 1/2^512 mod q
        let a = sqr::<MQ>(a);
        let a = sqr::<MQ>(a);
        let a = sqr::<MQ>(a);
        let a = sqr::<MQ>(a);
        let a = sqr::<MQ>(a);
        let a = sqr::<MQ>(a);
        let a = sqr::<MQ>(a);
        let a = sqr::<MQ>(a);
        let a = sqr::<MQ>(a);

        // multiply by 16 to get the result (see set_mul16()).
        let (a0, a1, a2, a3) = (a.0[0], a.0[1], a.0[2], a.0[3]);
        let tt = a3 >> 59;
        let d0 = a0 << 4;
        let d1 = (a0 >> 60) | (a1 << 4);
        let d2 = (a1 >> 60) | (a2 << 4);
        let d3 = (a2 >> 60) | ((a3 << 4) & 0x7FFFFFFFFFFFFFFF);
        let (d0, cc) = adc(d0, tt * MQ, 0);
        let (d1, cc) = adc(d1, 0, cc);
        let (d2, cc) = adc(d2, 0, cc);
        let (d3, _)  = adc(d3, 0, cc);

        Self([ d0, d1, d2, d3 ])
    }

    /// Constant-time table lookup: given a table of 48 field elements,
    /// and an index `j` in the 0 to 15 range, return the elements of
    /// index `j*3` to `j*3+2`. If `j` is not in the 0 to 15 range
    /// (inclusive), then this returns three zeros.
    pub fn lookup16_x3(tab: &[Self; 48], j: u32) -> [Self; 3] {
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            let mut d = [Self::ZERO; 3];
            for i in 0..16 {
                let w = ((j.wrapping_sub(i as u32)
                    | (i as u32).wrapping_sub(j)) >> 31).wrapping_sub(1);
                d[0].set_cond(&tab[3 * i + 0], w);
                d[1].set_cond(&tab[3 * i + 1], w);
                d[2].set_cond(&tab[3 * i + 2], w);
            }
            d
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            use core::arch::x86_64::*;

            let xj = _mm256_set1_epi32(j as i32);
            let mut xi = _mm256_setzero_si256();
            let mut a0 = _mm256_setzero_si256();
            let mut a1 = _mm256_setzero_si256();
            let mut a2 = _mm256_setzero_si256();
            for i in 0..16 {
                let m = _mm256_cmpeq_epi32(xi, xj);
                xi = _mm256_add_epi32(xi, _mm256_set1_epi32(1));
                a0 = _mm256_blendv_epi8(a0,
                    _mm256_loadu_si256(core::mem::transmute(
                        core::ptr::addr_of!(tab[3 * i + 0]))), m);
                a1 = _mm256_blendv_epi8(a1,
                    _mm256_loadu_si256(core::mem::transmute(
                        core::ptr::addr_of!(tab[3 * i + 1]))), m);
                a2 = _mm256_blendv_epi8(a2,
                    _mm256_loadu_si256(core::mem::transmute(
                        core::ptr::addr_of!(tab[3 * i + 2]))), m);
            }
            [
                core::mem::transmute(a0),
                core::mem::transmute(a1),
                core::mem::transmute(a2),
            ]
        }
    }

    /// Constant-time table lookup: given a table of 64 field elements,
    /// and an index `j` in the 0 to 15 range, return the elements of
    /// index `j*4` to `j*4+3`. If `j` is not in the 0 to 15 range
    /// (inclusive), then this returns four zeros.
    pub fn lookup16_x4(tab: &[Self; 64], j: u32) -> [Self; 4] {
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            let mut d = [Self::ZERO; 4];
            for i in 0..16 {
                let w = ((j.wrapping_sub(i as u32)
                    | (i as u32).wrapping_sub(j)) >> 31).wrapping_sub(1);
                d[0].set_cond(&tab[4 * i + 0], w);
                d[1].set_cond(&tab[4 * i + 1], w);
                d[2].set_cond(&tab[4 * i + 2], w);
                d[3].set_cond(&tab[4 * i + 3], w);
            }
            d
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            use core::arch::x86_64::*;

            let xj = _mm256_set1_epi32(j as i32);
            let mut xi = _mm256_setzero_si256();
            let mut a0 = _mm256_setzero_si256();
            let mut a1 = _mm256_setzero_si256();
            let mut a2 = _mm256_setzero_si256();
            let mut a3 = _mm256_setzero_si256();
            for i in 0..16 {
                let m = _mm256_cmpeq_epi32(xi, xj);
                xi = _mm256_add_epi32(xi, _mm256_set1_epi32(1));
                a0 = _mm256_blendv_epi8(a0,
                    _mm256_loadu_si256(core::mem::transmute(
                        core::ptr::addr_of!(tab[4 * i + 0]))), m);
                a1 = _mm256_blendv_epi8(a1,
                    _mm256_loadu_si256(core::mem::transmute(
                        core::ptr::addr_of!(tab[4 * i + 1]))), m);
                a2 = _mm256_blendv_epi8(a2,
                    _mm256_loadu_si256(core::mem::transmute(
                        core::ptr::addr_of!(tab[4 * i + 2]))), m);
                a3 = _mm256_blendv_epi8(a3,
                    _mm256_loadu_si256(core::mem::transmute(
                        core::ptr::addr_of!(tab[4 * i + 3]))), m);
            }
            [
                core::mem::transmute(a0),
                core::mem::transmute(a1),
                core::mem::transmute(a2),
                core::mem::transmute(a3),
            ]
        }
    }
}

// ========================================================================
// Implementations of all the traits needed to use the simple operators
// (+, *, /...) on field element instances, with or without references.

impl<const MQ: u64> Add<GF255<MQ>> for GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn add(self, other: GF255<MQ>) -> GF255<MQ> {
        let mut r = self;
        r.set_add(&other);
        r
    }
}

impl<const MQ: u64> Add<&GF255<MQ>> for GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn add(self, other: &GF255<MQ>) -> GF255<MQ> {
        let mut r = self;
        r.set_add(other);
        r
    }
}

impl<const MQ: u64> Add<GF255<MQ>> for &GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn add(self, other: GF255<MQ>) -> GF255<MQ> {
        let mut r = *self;
        r.set_add(&other);
        r
    }
}

impl<const MQ: u64> Add<&GF255<MQ>> for &GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn add(self, other: &GF255<MQ>) -> GF255<MQ> {
        let mut r = *self;
        r.set_add(other);
        r
    }
}

impl<const MQ: u64> AddAssign<GF255<MQ>> for GF255<MQ> {
    #[inline(always)]
    fn add_assign(&mut self, other: GF255<MQ>) {
        self.set_add(&other);
    }
}

impl<const MQ: u64> AddAssign<&GF255<MQ>> for GF255<MQ> {
    #[inline(always)]
    fn add_assign(&mut self, other: &GF255<MQ>) {
        self.set_add(other);
    }
}

impl<const MQ: u64> Div<GF255<MQ>> for GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn div(self, other: GF255<MQ>) -> GF255<MQ> {
        let mut r = self;
        r.set_div(&other);
        r
    }
}

impl<const MQ: u64> Div<&GF255<MQ>> for GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn div(self, other: &GF255<MQ>) -> GF255<MQ> {
        let mut r = self;
        r.set_div(other);
        r
    }
}

impl<const MQ: u64> Div<GF255<MQ>> for &GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn div(self, other: GF255<MQ>) -> GF255<MQ> {
        let mut r = *self;
        r.set_div(&other);
        r
    }
}

impl<const MQ: u64> Div<&GF255<MQ>> for &GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn div(self, other: &GF255<MQ>) -> GF255<MQ> {
        let mut r = *self;
        r.set_div(other);
        r
    }
}

impl<const MQ: u64> DivAssign<GF255<MQ>> for GF255<MQ> {
    #[inline(always)]
    fn div_assign(&mut self, other: GF255<MQ>) {
        self.set_div(&other);
    }
}

impl<const MQ: u64> DivAssign<&GF255<MQ>> for GF255<MQ> {
    #[inline(always)]
    fn div_assign(&mut self, other: &GF255<MQ>) {
        self.set_div(other);
    }
}

impl<const MQ: u64> Mul<GF255<MQ>> for GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn mul(self, other: GF255<MQ>) -> GF255<MQ> {
        let mut r = self;
        r.set_mul(&other);
        r
    }
}

impl<const MQ: u64> Mul<&GF255<MQ>> for GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn mul(self, other: &GF255<MQ>) -> GF255<MQ> {
        let mut r = self;
        r.set_mul(other);
        r
    }
}

impl<const MQ: u64> Mul<GF255<MQ>> for &GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn mul(self, other: GF255<MQ>) -> GF255<MQ> {
        let mut r = *self;
        r.set_mul(&other);
        r
    }
}

impl<const MQ: u64> Mul<&GF255<MQ>> for &GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn mul(self, other: &GF255<MQ>) -> GF255<MQ> {
        let mut r = *self;
        r.set_mul(other);
        r
    }
}

impl<const MQ: u64> MulAssign<GF255<MQ>> for GF255<MQ> {
    #[inline(always)]
    fn mul_assign(&mut self, other: GF255<MQ>) {
        self.set_mul(&other);
    }
}

impl<const MQ: u64> MulAssign<&GF255<MQ>> for GF255<MQ> {
    #[inline(always)]
    fn mul_assign(&mut self, other: &GF255<MQ>) {
        self.set_mul(other);
    }
}

impl<const MQ: u64> Neg for GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn neg(self) -> GF255<MQ> {
        let mut r = self;
        r.set_neg();
        r
    }
}

impl<const MQ: u64> Neg for &GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn neg(self) -> GF255<MQ> {
        let mut r = *self;
        r.set_neg();
        r
    }
}

impl<const MQ: u64> Sub<GF255<MQ>> for GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn sub(self, other: GF255<MQ>) -> GF255<MQ> {
        let mut r = self;
        r.set_sub(&other);
        r
    }
}

impl<const MQ: u64> Sub<&GF255<MQ>> for GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn sub(self, other: &GF255<MQ>) -> GF255<MQ> {
        let mut r = self;
        r.set_sub(other);
        r
    }
}

impl<const MQ: u64> Sub<GF255<MQ>> for &GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn sub(self, other: GF255<MQ>) -> GF255<MQ> {
        let mut r = *self;
        r.set_sub(&other);
        r
    }
}

impl<const MQ: u64> Sub<&GF255<MQ>> for &GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn sub(self, other: &GF255<MQ>) -> GF255<MQ> {
        let mut r = *self;
        r.set_sub(other);
        r
    }
}

impl<const MQ: u64> SubAssign<GF255<MQ>> for GF255<MQ> {
    #[inline(always)]
    fn sub_assign(&mut self, other: GF255<MQ>) {
        self.set_sub(&other);
    }
}

impl<const MQ: u64> SubAssign<&GF255<MQ>> for GF255<MQ> {
    #[inline(always)]
    fn sub_assign(&mut self, other: &GF255<MQ>) {
        self.set_sub(other);
    }
}

// ========================================================================

#[cfg(test)]
mod tests {

    use super::{GF255};
    use num_bigint::{BigInt, Sign};
    use sha2::{Sha256, Digest};

    /* unused
    fn print<const MQ: u64>(name: &str, v: GF255<MQ>) {
        println!("{} = 0x{:016X}{:016X}{:016X}{:016X}",
            name, v.0[3], v.0[2], v.0[1], v.0[0]);
    }
    */

    // va, vb and vx must be 32 bytes each in length
    fn check_gf_ops<const MQ: u64>(va: &[u8], vb: &[u8], vx: &[u8]) {
        let zp = BigInt::from_slice(Sign::Plus, &[
            (MQ as u32).wrapping_neg(),
            0xFFFFFFFFu32, 0xFFFFFFFFu32, 0xFFFFFFFFu32,
            0xFFFFFFFFu32, 0xFFFFFFFFu32, 0xFFFFFFFFu32, 0x7FFFFFFFu32,
        ]);
        let zp4 = &zp << 2;

        let mut a = GF255::<MQ>::ZERO;
        a.set_decode32_reduce(va);
        let mut b = GF255::<MQ>::ZERO;
        b.set_decode32_reduce(vb);
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
        let zd = ((&zp4 + &za) - &zb) % &zp;
        assert!(zc == zd);

        let c = -a;
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&zp4 - &za) % &zp;
        assert!(zc == zd);

        let c = a * b;
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za * &zb) % &zp;
        assert!(zc == zd);

        let c = a.half();
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd: BigInt = ((&zp4 + (&zc << 1)) - &za) % &zp;
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

        let x = b.0[1] as u32;
        let c = a.mul_small(x);
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za * x) % &zp;
        assert!(zc == zd);

        let c = a.square();
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za * &za) % &zp;
        assert!(zc == zd);

        let (e, cc) = GF255::<MQ>::decode32(va);
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
            let c = GF255::<MQ>::decode_reduce(&tmp[0..k]);
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

    fn test_gf<const MQ: u64>(nqr: u32) {
        let mut va = [0u8; 32];
        let mut vb = [0u8; 32];
        let mut vx = [0u8; 32];
        check_gf_ops::<MQ>(&va, &vb, &vx);
        assert!(GF255::<MQ>::decode_reduce(&va).iszero() == 0xFFFFFFFF);
        assert!(GF255::<MQ>::decode_reduce(&va).equals(GF255::<MQ>::decode_reduce(&vb)) == 0xFFFFFFFF);
        assert!(GF255::<MQ>::decode_reduce(&va).legendre() == 0);
        for i in 0..32 {
            va[i] = 0xFFu8;
            vb[i] = 0xFFu8;
            vx[i] = 0xFFu8;
        }
        check_gf_ops::<MQ>(&va, &vb, &vx);
        assert!(GF255::<MQ>::decode_reduce(&va).iszero() == 0);
        assert!(GF255::<MQ>::decode_reduce(&va).equals(GF255::<MQ>::decode_reduce(&vb)) == 0xFFFFFFFF);
        va[0..8].copy_from_slice(&MQ.wrapping_neg().to_le_bytes());
        va[31] = 0x7F;
        assert!(GF255::<MQ>::decode_reduce(&va).iszero() == 0xFFFFFFFF);
        va[0..8].copy_from_slice(&(2 * MQ).wrapping_neg().to_le_bytes());
        va[31] = 0xFF;
        assert!(GF255::<MQ>::decode_reduce(&va).iszero() == 0xFFFFFFFF);
        let mut sh = Sha256::new();
        let tt = GF255::<MQ>::w64le(0, 0, 1, 0);
        let corr128 = [ -tt, GF255::<MQ>::ZERO, tt ];
        for i in 0..300 {
            sh.update(((3 * i + 0) as u64).to_le_bytes());
            let va = sh.finalize_reset();
            sh.update(((3 * i + 1) as u64).to_le_bytes());
            let vb = sh.finalize_reset();
            sh.update(((3 * i + 2) as u64).to_le_bytes());
            let vx = sh.finalize_reset();
            check_gf_ops::<MQ>(&va, &vb, &vx);
            assert!(GF255::<MQ>::decode_reduce(&va).iszero() == 0);
            assert!(GF255::<MQ>::decode_reduce(&va).equals(GF255::<MQ>::decode_reduce(&vb)) == 0);
            let s = GF255::<MQ>::decode_reduce(&va).square();
            let s2 = s.mul_small(nqr);
            assert!(s.legendre() == 1);
            assert!(s2.legendre() == -1);
            let (t, r) = s.sqrt();
            assert!(r == 0xFFFFFFFF);
            assert!(t.square().equals(s) == 0xFFFFFFFF);
            assert!((t.encode32()[0] & 1) == 0);
            let (t, r) = s.sqrt_ext();
            assert!(r == 0xFFFFFFFF);
            assert!(t.square().equals(s) == 0xFFFFFFFF);
            assert!((t.encode32()[0] & 1) == 0);
            let (t2, r) = s2.sqrt();
            assert!(r == 0);
            assert!(t2.iszero() == 0xFFFFFFFF);
            let (t2, r) = s2.sqrt_ext();
            assert!(r == 0);
            if (MQ & 3) == 1 {
                // q = 3 mod 4, we are supposed to get a square root of -s2
                assert!(t2.square().equals(-s2) == 0xFFFFFFFF);
            } else if (MQ & 7) == 3 {
                // q = 5 mod 8, we are supposed to get a square root of
                // 2*s2 or -2*s2
                let y = t2.square();
                let z = s2.mul2();
                assert!((y.equals(z) | y.equals(-z)) == 0xFFFFFFFF);
            } else {
                unimplemented!();
            }

            let a = GF255::<MQ>::decode_reduce(&va);
            let (c0, c1) = a.split_vartime();
            let b0 = GF255::<MQ>::from_i128(c0);
            let b1 = GF255::<MQ>::from_i128(c1);
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
    fn gf255e_ops() {
        test_gf::<18651>(2);
    }

    #[test]
    fn gf255s_ops() {
        test_gf::<3957>(2);
    }

    #[test]
    fn gf25519_ops() {
        test_gf::<19>(2);
    }

    #[test]
    fn gf25519_batch_invert() {
        let mut xx = [GF255::<19>::ZERO; 300];
        let mut sh = Sha256::new();
        for i in 0..300 {
            sh.update((i as u64).to_le_bytes());
            let v = sh.finalize_reset();
            xx[i] = GF255::<19>::decode_reduce(&v);
        }
        xx[120] = GF255::<19>::ZERO;
        let mut yy = xx;
        GF255::<19>::batch_invert(&mut yy[..]);
        for i in 0..300 {
            if xx[i].iszero() != 0 {
                assert!(yy[i].iszero() == 0xFFFFFFFF);
            } else {
                assert!((xx[i] * yy[i]).equals(GF255::<19>::ONE) == 0xFFFFFFFF);
            }
        }
    }
}
