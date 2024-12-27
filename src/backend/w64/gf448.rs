use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::convert::TryFrom;

use super::{addcarry_u64, subborrow_u64, umull, umull_add, umull_add2, umull_x2, umull_x2_add, sgnw, lzcnt};

#[derive(Clone, Copy, Debug)]
pub struct GF448([u64; 7]);

impl GF448 {

    // Internal element representation: a 448-bit integer, in base 2^64.
    // Representation is slightly redundant: all 448-bit values are
    // allowed.

    // Modulus p in base 2^64 (low-to-high order).
    pub const MODULUS: [u64; 7] = [
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFEFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
    ];

    // Element encoding length: 56 bytes.
    pub const ENC_LEN: usize = 56;

    pub const ZERO: GF448 = GF448([ 0, 0, 0, 0, 0, 0, 0 ]);
    pub const ONE: GF448 = GF448([ 1, 0, 0, 0, 0, 0, 0 ]);
    pub const MINUS_ONE: GF448 = GF448([
        0xFFFFFFFFFFFFFFFE,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFEFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
    ]);

    // 1/2^894 (in the field)
    const INVT894: GF448 = GF448([
        0x0000000000000013,
        0x0000000000000000,
        0x0000000000000000,
        0xFFFFFFF300000000,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
    ]);

    // Create an element from a 448-bit value (implicitly reduced modulo
    // the field order) provided as seven 64-bit limbs (in low-to-high order).
    pub const fn w64le(x: [u64; 7]) -> Self {
        Self(x)
    }

    // Create an element from a 448-bit value (implicitly reduced modulo
    // the field order) provided as seven 64-bit limbs (in high-to-low order).
    pub const fn w64be(x: [u64; 7]) -> Self {
        Self([ x[6], x[5], x[4], x[3], x[2], x[1], x[0] ])
    }

    // Create an element from a 448-bit value (implicitly reduced modulo
    // the field order) provided as seven 64-bit limbs (in low-to-high order).
    #[inline(always)]
    pub fn from_w64le(x: [u64; 7]) -> Self {
        Self(x)
    }

    // Create an element from a 448-bit value (implicitly reduced modulo
    // the field order) provided as seven 64-bit limbs (in high-to-low order).
    #[inline(always)]
    pub fn from_w64be(x: [u64; 7]) -> Self {
        Self([ x[6], x[5], x[4], x[3], x[2], x[1], x[0] ])
    }

    // Create an element by converting to provided integer (implicitly
    // reduced modulo the field order).
    #[inline(always)]
    pub fn from_i32(x: i32) -> Self {
        let x0 = (x as i64) as u64;
        let xh = ((x as i64) >> 63) as u64;
        let (d0, cc) = addcarry_u64(x0, Self::MODULUS[0], 0);
        let (d1, cc) = addcarry_u64(xh, Self::MODULUS[1], cc);
        let (d2, cc) = addcarry_u64(xh, Self::MODULUS[2], cc);
        let (d3, cc) = addcarry_u64(xh, Self::MODULUS[3], cc);
        let (d4, cc) = addcarry_u64(xh, Self::MODULUS[4], cc);
        let (d5, cc) = addcarry_u64(xh, Self::MODULUS[5], cc);
        let (d6, _) = addcarry_u64(xh, Self::MODULUS[6], cc);
        Self([ d0, d1, d2, d3, d4, d5, d6 ])
    }

    // Create an element by converting to provided integer (implicitly
    // reduced modulo the field order).
    #[inline(always)]
    pub fn from_u32(x: u32) -> Self {
        Self([ x as u64, 0, 0, 0, 0, 0, 0 ])
    }

    // Create an element by converting to provided integer (implicitly
    // reduced modulo the field order).
    #[inline(always)]
    pub fn from_i64(x: i64) -> Self {
        let x0 = x as u64;
        let xh = (x >> 63) as u64;
        let (d0, cc) = addcarry_u64(x0, Self::MODULUS[0], 0);
        let (d1, cc) = addcarry_u64(xh, Self::MODULUS[1], cc);
        let (d2, cc) = addcarry_u64(xh, Self::MODULUS[2], cc);
        let (d3, cc) = addcarry_u64(xh, Self::MODULUS[3], cc);
        let (d4, cc) = addcarry_u64(xh, Self::MODULUS[4], cc);
        let (d5, cc) = addcarry_u64(xh, Self::MODULUS[5], cc);
        let (d6, _) = addcarry_u64(xh, Self::MODULUS[6], cc);
        Self([ d0, d1, d2, d3, d4, d5, d6 ])
    }

    // Create an element by converting to provided integer (implicitly
    // reduced modulo the field order).
    #[inline(always)]
    pub fn from_u64(x: u64) -> Self {
        Self([ x, 0, 0, 0, 0, 0, 0 ])
    }

    // Create an element by converting to provided integer (implicitly
    // reduced modulo the field order).
    #[inline(always)]
    pub fn from_i128(x: i128) -> Self {
        let x0 = x as u64;
        let x1 = (x >> 64) as u64;
        let xh = (x >> 127) as u64;
        let (d0, cc) = addcarry_u64(x0, Self::MODULUS[0], 0);
        let (d1, cc) = addcarry_u64(x1, Self::MODULUS[1], cc);
        let (d2, cc) = addcarry_u64(xh, Self::MODULUS[2], cc);
        let (d3, cc) = addcarry_u64(xh, Self::MODULUS[3], cc);
        let (d4, cc) = addcarry_u64(xh, Self::MODULUS[4], cc);
        let (d5, cc) = addcarry_u64(xh, Self::MODULUS[5], cc);
        let (d6, _) = addcarry_u64(xh, Self::MODULUS[6], cc);
        Self([ d0, d1, d2, d3, d4, d5, d6 ])
    }

    // Create an element by converting to provided integer (implicitly
    // reduced modulo the field order).
    #[inline(always)]
    pub fn from_u128(x: u128) -> Self {
        Self([ x as u64, (x >> 64) as u64, 0, 0, 0, 0, 0 ])
    }

    #[inline]
    fn set_add(&mut self, rhs: &Self) {
        // 1. Addition with carry
        let (d0, cc) = addcarry_u64(self.0[0], rhs.0[0], 0);
        let (d1, cc) = addcarry_u64(self.0[1], rhs.0[1], cc);
        let (d2, cc) = addcarry_u64(self.0[2], rhs.0[2], cc);
        let (d3, cc) = addcarry_u64(self.0[3], rhs.0[3], cc);
        let (d4, cc) = addcarry_u64(self.0[4], rhs.0[4], cc);
        let (d5, cc) = addcarry_u64(self.0[5], rhs.0[5], cc);
        let (d6, cc) = addcarry_u64(self.0[6], rhs.0[6], cc);

        // 2. In case of an output carry, reduce.
        // Reduction is: 2^448 = 2^224 + 1
        let e = cc;
        let (d0, cc) = addcarry_u64(d0, 0, e);
        let (d1, cc) = addcarry_u64(d1, 0, cc);
        let (d2, cc) = addcarry_u64(d2, 0, cc);
        let (d3, cc) = addcarry_u64(d3, (e as u64) << 32, cc);
        let (d4, cc) = addcarry_u64(d4, 0, cc);
        let (d5, cc) = addcarry_u64(d5, 0, cc);
        let (d6, cc) = addcarry_u64(d6, 0, cc);

        // 3. At that point we have:
        //    d <= 2*(2^448 - 1) - p = 2^448 + 2^224 - 1
        // Thus, if we have a carry (d >= 2^448), then wrapping it again
        // can impact at most d0..3, but not upper words, and there won't
        // be any extra carry.
        let e = cc;
        let (d0, cc) = addcarry_u64(d0, 0, e);
        let (d1, cc) = addcarry_u64(d1, 0, cc);
        let (d2, cc) = addcarry_u64(d2, 0, cc);
        let (d3, _) = addcarry_u64(d3, (e as u64) << 32, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
        self.0[4] = d4;
        self.0[5] = d5;
        self.0[6] = d6;
    }

    #[inline]
    fn set_sub(&mut self, rhs: &Self) {
        // 1. Subtraction with borrow
        let (d0, cc) = subborrow_u64(self.0[0], rhs.0[0], 0);
        let (d1, cc) = subborrow_u64(self.0[1], rhs.0[1], cc);
        let (d2, cc) = subborrow_u64(self.0[2], rhs.0[2], cc);
        let (d3, cc) = subborrow_u64(self.0[3], rhs.0[3], cc);
        let (d4, cc) = subborrow_u64(self.0[4], rhs.0[4], cc);
        let (d5, cc) = subborrow_u64(self.0[5], rhs.0[5], cc);
        let (d6, cc) = subborrow_u64(self.0[6], rhs.0[6], cc);

        // 2. In case of an output borrow, add p = 2^448 - 2^224 - 1,
        // i.e. add 2^448 (this is implicit, it cancels the borrow)
        // and also subtract 2^224 + 1.
        let e = cc;
        let (d0, cc) = subborrow_u64(d0, 0, e);
        let (d1, cc) = subborrow_u64(d1, 0, cc);
        let (d2, cc) = subborrow_u64(d2, 0, cc);
        let (d3, cc) = subborrow_u64(d3, (e as u64) << 32, cc);
        let (d4, cc) = subborrow_u64(d4, 0, cc);
        let (d5, cc) = subborrow_u64(d5, 0, cc);
        let (d6, cc) = subborrow_u64(d6, 0, cc);

        // 3. If we still got a borrow, then we need to add p again, i.e.
        // subtract 2^224 + 1 once more. We know that, at this point:
        //    d >= -(2^448 - 1) + p = -(2^224)
        // Thus, d3 >= 2^64 - 2^32, and there cannot be any further
        // modification of d4..d6.
        let e = cc;
        let (d0, cc) = subborrow_u64(d0, 0, e);
        let (d1, cc) = subborrow_u64(d1, 0, cc);
        let (d2, cc) = subborrow_u64(d2, 0, cc);
        let (d3, _) = subborrow_u64(d3, (e as u64) << 32, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
        self.0[4] = d4;
        self.0[5] = d5;
        self.0[6] = d6;
    }

    // Negate this value (in place).
    #[inline]
    pub fn set_neg(&mut self) {
        // 1. Compute p - self over 448 bits.
        let (d0, cc) = subborrow_u64(Self::MODULUS[0], self.0[0], 0);
        let (d1, cc) = subborrow_u64(Self::MODULUS[1], self.0[1], cc);
        let (d2, cc) = subborrow_u64(Self::MODULUS[2], self.0[2], cc);
        let (d3, cc) = subborrow_u64(Self::MODULUS[3], self.0[3], cc);
        let (d4, cc) = subborrow_u64(Self::MODULUS[4], self.0[4], cc);
        let (d5, cc) = subborrow_u64(Self::MODULUS[5], self.0[5], cc);
        let (d6, cc) = subborrow_u64(Self::MODULUS[6], self.0[6], cc);

        // 2. If there is a borrow, then d >= p - (2^448 - 1) = -(2^224)
        // and adding back p (i.e. subtracting 2^224 + 1) cannot modify
        // words d4..d7.
        let e = cc;
        let (d0, cc) = subborrow_u64(d0, 0, e);
        let (d1, cc) = subborrow_u64(d1, 0, cc);
        let (d2, cc) = subborrow_u64(d2, 0, cc);
        let (d3, _) = subborrow_u64(d3, (e as u64) << 32, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
        self.0[4] = d4;
        self.0[5] = d5;
        self.0[6] = d6;
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
        self.0[4] ^= cw & (self.0[4] ^ a.0[4]);
        self.0[5] ^= cw & (self.0[5] ^ a.0[5]);
        self.0[6] ^= cw & (self.0[6] ^ a.0[6]);
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
        let t = cw & (a.0[4] ^ b.0[4]); a.0[4] ^= t; b.0[4] ^= t;
        let t = cw & (a.0[5] ^ b.0[5]); a.0[5] ^= t; b.0[5] ^= t;
        let t = cw & (a.0[6] ^ b.0[6]); a.0[6] ^= t; b.0[6] ^= t;
    }

    #[inline]
    fn set_half(&mut self) {
        // 1. Right-shift by 1 bit; keep dropped bit in 'tt' (expanded)
        let d0 = (self.0[0] >> 1) | (self.0[1] << 63);
        let d1 = (self.0[1] >> 1) | (self.0[2] << 63);
        let d2 = (self.0[2] >> 1) | (self.0[3] << 63);
        let d3 = (self.0[3] >> 1) | (self.0[4] << 63);
        let d4 = (self.0[4] >> 1) | (self.0[5] << 63);
        let d5 = (self.0[5] >> 1) | (self.0[6] << 63);
        let d6 = self.0[6] >> 1;
        let tt = (self.0[0] & 1).wrapping_neg();

        // 2. If the dropped bit was 1, add back (p+1)/2. Since the value
        // was right-shifted, and (p+1)/2 < 2^447, this cannot overflow.
        // Also, (p+1)/2 = 2^447 - 2^223, so the three low words are
        // unmodified.
        let (d3, cc) = addcarry_u64(d3, tt << 31, 0);
        let (d4, cc) = addcarry_u64(d4, tt, cc);
        let (d5, cc) = addcarry_u64(d5, tt, cc);
        let (d6, _) = addcarry_u64(d6, tt >> 1, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
        self.0[4] = d4;
        self.0[5] = d5;
        self.0[6] = d6;
    }

    #[inline(always)]
    pub fn half(self) -> Self {
        let mut r = self;
        r.set_half();
        r
    }

    // Add hi*2^448 to this value; this is equivalent to performing an
    // appropriate reduction when the intermediate value has extra bits of
    // value `hi`.
    #[inline(always)]
    fn reduce_small(&mut self, hi: u64) {
        // Add hi*2^448 = hi*p + hi*(1 + 2^224) = hi + hi*2^224 mod p
        let (d0, cc) = addcarry_u64(self.0[0], hi, 0);
        let (d1, cc) = addcarry_u64(self.0[1], 0, cc);
        let (d2, cc) = addcarry_u64(self.0[2], 0, cc);
        let (d3, cc) = addcarry_u64(self.0[3], hi << 32, cc);
        let (d4, cc) = addcarry_u64(self.0[4], hi >> 32, cc);
        let (d5, cc) = addcarry_u64(self.0[5], 0, cc);
        let (d6, cc) = addcarry_u64(self.0[6], 0, cc);

        // If there is a carry, then wrap it. In that case, current value
        // d (with the extra carry) is such that:
        //    d <= 2^448 - 1 + (2^64 - 1)*(1 + 2^224) < 2^448 + 2^288
        // Thus, this wrap operation cannot impact d6 or d7, nor yield
        // another final carry.
        let e = cc;
        let (d0, cc) = addcarry_u64(d0, 0, e);
        let (d1, cc) = addcarry_u64(d1, 0, cc);
        let (d2, cc) = addcarry_u64(d2, 0, cc);
        let (d3, cc) = addcarry_u64(d3, (e as u64) << 32, cc);
        let (d4, _) = addcarry_u64(d4, 0, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
        self.0[4] = d4;
        self.0[5] = d5;
        self.0[6] = d6;
    }

    // Multiply this value by 2.
    #[inline]
    pub fn set_mul2(&mut self) {
        // 1. Extract top bit.
        let tt = self.0[6] >> 63;

        // 2. Left-shift.
        let d0 = self.0[0] << 1;
        let d1 = (self.0[0] >> 63) | (self.0[1] << 1);
        let d2 = (self.0[1] >> 63) | (self.0[2] << 1);
        let d3 = (self.0[2] >> 63) | (self.0[3] << 1);
        let d4 = (self.0[3] >> 63) | (self.0[4] << 1);
        let d5 = (self.0[4] >> 63) | (self.0[5] << 1);
        let d6 = (self.0[5] >> 63) | (self.0[6] << 1);

        // 3. Wrap back the extracted bit.
        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
        self.0[4] = d4;
        self.0[5] = d5;
        self.0[6] = d6;
        self.reduce_small(tt);
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
        let tt = self.0[6] >> 62;

        // 2. Left-shift.
        let d0 = self.0[0] << 2;
        let d1 = (self.0[0] >> 62) | (self.0[1] << 2);
        let d2 = (self.0[1] >> 62) | (self.0[2] << 2);
        let d3 = (self.0[2] >> 62) | (self.0[3] << 2);
        let d4 = (self.0[3] >> 62) | (self.0[4] << 2);
        let d5 = (self.0[4] >> 62) | (self.0[5] << 2);
        let d6 = (self.0[5] >> 62) | (self.0[6] << 2);

        // 3. Wrap back the extracted bits.
        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
        self.0[4] = d4;
        self.0[5] = d5;
        self.0[6] = d6;
        self.reduce_small(tt);
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
        let tt = self.0[6] >> 61;

        // 2. Left-shift.
        let d0 = self.0[0] << 3;
        let d1 = (self.0[0] >> 61) | (self.0[1] << 3);
        let d2 = (self.0[1] >> 61) | (self.0[2] << 3);
        let d3 = (self.0[2] >> 61) | (self.0[3] << 3);
        let d4 = (self.0[3] >> 61) | (self.0[4] << 3);
        let d5 = (self.0[4] >> 61) | (self.0[5] << 3);
        let d6 = (self.0[5] >> 61) | (self.0[6] << 3);

        // 3. Wrap back the extracted bits.
        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
        self.0[4] = d4;
        self.0[5] = d5;
        self.0[6] = d6;
        self.reduce_small(tt);
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
        let tt = self.0[6] >> 60;

        // 2. Left-shift.
        let d0 = self.0[0] << 4;
        let d1 = (self.0[0] >> 60) | (self.0[1] << 4);
        let d2 = (self.0[1] >> 60) | (self.0[2] << 4);
        let d3 = (self.0[2] >> 60) | (self.0[3] << 4);
        let d4 = (self.0[3] >> 60) | (self.0[4] << 4);
        let d5 = (self.0[4] >> 60) | (self.0[5] << 4);
        let d6 = (self.0[5] >> 60) | (self.0[6] << 4);

        // 3. Wrap back the extracted bits.
        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
        self.0[4] = d4;
        self.0[5] = d5;
        self.0[6] = d6;
        self.reduce_small(tt);
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
        let tt = self.0[6] >> 59;

        // 2. Left-shift.
        let d0 = self.0[0] << 5;
        let d1 = (self.0[0] >> 59) | (self.0[1] << 5);
        let d2 = (self.0[1] >> 59) | (self.0[2] << 5);
        let d3 = (self.0[2] >> 59) | (self.0[3] << 5);
        let d4 = (self.0[3] >> 59) | (self.0[4] << 5);
        let d5 = (self.0[4] >> 59) | (self.0[5] << 5);
        let d6 = (self.0[5] >> 59) | (self.0[6] << 5);

        // 3. Wrap back the extracted bits.
        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
        self.0[4] = d4;
        self.0[5] = d5;
        self.0[6] = d6;
        self.reduce_small(tt);
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

        // Compute the product as an integer over eight words.
        // Max value is (2^32 - 1)*(2^448 - 1), so the top word (d4) is
        // at most 2^32 - 2.
        let (d0, d1) = umull(self.0[0], b);
        let (d2, d3) = umull(self.0[2], b);
        let (d4, d5) = umull(self.0[4], b);
        let (d6, d7) = umull(self.0[6], b);
        let (lo, hi) = umull(self.0[1], b);
        let (d1, cc) = addcarry_u64(d1, lo, 0);
        let (d2, cc) = addcarry_u64(d2, hi, cc);
        let (lo, hi) = umull(self.0[3], b);
        let (d3, cc) = addcarry_u64(d3, lo, cc);
        let (d4, cc) = addcarry_u64(d4, hi, cc);
        let (lo, hi) = umull(self.0[5], b);
        let (d5, cc) = addcarry_u64(d5, lo, cc);
        let (d6, cc) = addcarry_u64(d6, hi, cc);
        let (d7, _) = addcarry_u64(d7, 0, cc);

        // Wrap the extra bits.
        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
        self.0[4] = d4;
        self.0[5] = d5;
        self.0[6] = d6;
        self.reduce_small(d7);
    }

    #[inline(always)]
    pub fn mul_small(self, x: u32) -> Self {
        let mut r = self;
        r.set_mul_small(x);
        r
    }

    #[inline(always)]
    fn set_mul(&mut self, rhs: &Self) {
        // 1. Product -> 896 bits.
        let mut d = [0u64; 14];
        (d[0], d[1]) = umull(self.0[0], rhs.0[0]);
        (d[2], d[3]) = umull(self.0[2], rhs.0[0]);
        (d[4], d[5]) = umull(self.0[4], rhs.0[0]);
        (d[6], d[7]) = umull(self.0[6], rhs.0[0]);
        let (lo, hi) = umull(self.0[1], rhs.0[0]);
        let mut cc;
        (d[1], cc) = addcarry_u64(d[1], lo, 0);
        (d[2], cc) = addcarry_u64(d[2], hi, cc);
        let (lo, hi) = umull(self.0[3], rhs.0[0]);
        (d[3], cc) = addcarry_u64(d[3], lo, cc);
        (d[4], cc) = addcarry_u64(d[4], hi, cc);
        let (lo, hi) = umull(self.0[5], rhs.0[0]);
        (d[5], cc) = addcarry_u64(d[5], lo, cc);
        (d[6], cc) = addcarry_u64(d[6], hi, cc);
        (d[7], _) = addcarry_u64(d[7], 0, cc);
        for i in 1..7 {
            let (lo, hi) = umull_add(self.0[0], rhs.0[i], d[i + 0]);
            d[i + 0] = lo;
            let (lo, hi) = umull_add2(self.0[1], rhs.0[i], d[i + 1], hi);
            d[i + 1] = lo;
            let (lo, hi) = umull_add2(self.0[2], rhs.0[i], d[i + 2], hi);
            d[i + 2] = lo;
            let (lo, hi) = umull_add2(self.0[3], rhs.0[i], d[i + 3], hi);
            d[i + 3] = lo;
            let (lo, hi) = umull_add2(self.0[4], rhs.0[i], d[i + 4], hi);
            d[i + 4] = lo;
            let (lo, hi) = umull_add2(self.0[5], rhs.0[i], d[i + 5], hi);
            d[i + 5] = lo;
            let (lo, hi) = umull_add2(self.0[6], rhs.0[i], d[i + 6], hi);
            d[i + 6] = lo;
            d[i + 7] = hi;
        }

        // 2. Reduction.
        // If we write the current value as:
        //   d = dlow + e*2^448 + f*2^(448+224)
        // with e and f both lower than 2^224, then the reduction is:
        //   d = dlow + e + e*2^224 + f*2^224 + f*2^448  mod p
        //     = dlow + (e + f) + (e + 2*f)*2^224  mod p
        // We can thus compute:
        //   g = e + f
        //   h = g + f
        // and then add g + h*2^224 to dlow.
        // If we write g = gl + gh*2^224 (with gh = 0 or 1),
        // then:
        //   g + h*2^224 = gl + (h + gh)*2^224
        let e0 = d[7];
        let e1 = d[8];
        let e2 = d[9];
        let e3 = d[10] & 0x00000000FFFFFFFF;
        let f0 = (d[10] >> 32) | (d[11] << 32);
        let f1 = (d[11] >> 32) | (d[12] << 32);
        let f2 = (d[12] >> 32) | (d[13] << 32);
        let f3 = d[13] >> 32;
        let (g0, cc) = addcarry_u64(e0, f0, 0);
        let (g1, cc) = addcarry_u64(e1, f1, cc);
        let (g2, cc) = addcarry_u64(e2, f2, cc);
        let (g3, _) = addcarry_u64(e3, f3, cc);
        let gh = g3 >> 32;
        let (h0, cc) = addcarry_u64(g0, f0, gh as u8);
        let (h1, cc) = addcarry_u64(g1, f1, cc);
        let (h2, cc) = addcarry_u64(g2, f2, cc);
        let (h3, _) = addcarry_u64(g3, f3, cc);
        let g3 = g3 & 0x00000000FFFFFFFF;

        // gl is in g0..g3 (224 bits), and h+gh is in h0..h3 (at most 226 bits)
        let mut cc;
        (self.0[0], cc) = addcarry_u64(d[0], g0, 0);
        (self.0[1], cc) = addcarry_u64(d[1], g1, cc);
        (self.0[2], cc) = addcarry_u64(d[2], g2, cc);
        (self.0[3], cc) = addcarry_u64(d[3], g3 | (h0 << 32), cc);
        (self.0[4], cc) = addcarry_u64(d[4], (h0 >> 32) | (h1 << 32), cc);
        (self.0[5], cc) = addcarry_u64(d[5], (h1 >> 32) | (h2 << 32), cc);
        (self.0[6], cc) = addcarry_u64(d[6], (h2 >> 32) | (h3 << 32), cc);
        let (x, _) = addcarry_u64(h3 >> 32, 0, cc);
        self.reduce_small(x);

        /*
         * Below is an alternate version, which uses Karatsuba and integrates
         * the value reassembly into the modular reduction. It has lower
         * latency, as measured on a sequence of dependent multiplications
         * (115.5 cycles, vs 127.3 cycles for the code above, on an x86
         * Intel i5-8259U "Coffee Lake"), but it decreases overall performance
         * with Ed448 and X448; presumably, the higher-latency code above
         * also offers enough "holes" that the CPU can schedule instructions
         * from other operations "for free".

        // Split first input into two 224-bit halves.
        let a0 = self.0[0];
        let a1 = self.0[1];
        let a2 = self.0[2];
        let a3 = self.0[3] & 0x00000000FFFFFFFF;
        let a4 = (self.0[3] >> 32) | (self.0[4] << 32);
        let a5 = (self.0[4] >> 32) | (self.0[5] << 32);
        let a6 = (self.0[5] >> 32) | (self.0[6] << 32);
        let a7 = self.0[6] >> 32;

        // Split second input into two 224-bit halves.
        let b0 = rhs.0[0];
        let b1 = rhs.0[1];
        let b2 = rhs.0[2];
        let b3 = rhs.0[3] & 0x00000000FFFFFFFF;
        let b4 = (rhs.0[3] >> 32) | (rhs.0[4] << 32);
        let b5 = (rhs.0[4] >> 32) | (rhs.0[5] << 32);
        let b6 = (rhs.0[5] >> 32) | (rhs.0[6] << 32);
        let b7 = rhs.0[6] >> 32;

        #[inline(always)]
        fn mul224(
            x0: u64, x1: u64, x2: u64, x3: u64,
            y0: u64, y1: u64, y2: u64, y3: u64)
            -> (u64, u64, u64, u64, u64, u64, u64)
        {
            let (e0, e1) = umull(x0, y0);
            let (e2, e3) = umull(x1, y1);
            let (e4, e5) = umull(x2, y2);
            let e6 = x3 * y3;

            let (lo, hi) = umull(x0, y1);
            let (e1, cc) = addcarry_u64(e1, lo, 0);
            let (e2, cc) = addcarry_u64(e2, hi, cc);
            let (lo, hi) = umull(x0, y3);
            let (e3, cc) = addcarry_u64(e3, lo, cc);
            let (e4, cc) = addcarry_u64(e4, hi, cc);
            let (lo, hi) = umull(x2, y3);
            let (e5, cc) = addcarry_u64(e5, lo, cc);
            let (e6, _) = addcarry_u64(e6, hi, cc);

            let (lo, hi) = umull(x1, y0);
            let (e1, cc) = addcarry_u64(e1, lo, 0);
            let (e2, cc) = addcarry_u64(e2, hi, cc);
            let (lo, hi) = umull(x3, y0);
            let (e3, cc) = addcarry_u64(e3, lo, cc);
            let (e4, cc) = addcarry_u64(e4, hi, cc);
            let (lo, hi) = umull(x3, y2);
            let (e5, cc) = addcarry_u64(e5, lo, cc);
            let (e6, _) = addcarry_u64(e6, hi, cc);

            let (lo, hi) = umull(x0, y2);
            let (e2, cc) = addcarry_u64(e2, lo, 0);
            let (e3, cc) = addcarry_u64(e3, hi, cc);
            let (lo, hi) = umull(x1, y3);
            let (e4, cc) = addcarry_u64(e4, lo, cc);
            let (e5, cc) = addcarry_u64(e5, hi, cc);
            let (e6, _) = addcarry_u64(e6, 0, cc);

            let (lo, hi) = umull(x2, y0);
            let (e2, cc) = addcarry_u64(e2, lo, 0);
            let (e3, cc) = addcarry_u64(e3, hi, cc);
            let (lo, hi) = umull(x3, y1);
            let (e4, cc) = addcarry_u64(e4, lo, cc);
            let (e5, cc) = addcarry_u64(e5, hi, cc);
            let (e6, _) = addcarry_u64(e6, 0, cc);

            let (lo, hi) = umull(x1, y2);
            let (lo2, hi2) = umull(x2, y1);
            let (lo, cc) = addcarry_u64(lo, lo2, 0);
            let (hi, tt) = addcarry_u64(hi, hi2, cc);
            let (e3, cc) = addcarry_u64(e3, lo, 0);
            let (e4, cc) = addcarry_u64(e4, hi, cc);
            let (e5, cc) = addcarry_u64(e5, tt as u64, cc);
            let (e6, _) = addcarry_u64(e6, 0, cc);

            (e0, e1, e2, e3, e4, e5, e6)
        }

        #[inline(always)]
        fn mul256(
            x0: u64, x1: u64, x2: u64, x3: u64,
            y0: u64, y1: u64, y2: u64, y3: u64)
            -> (u64, u64, u64, u64, u64, u64, u64, u64)
        {
            let (e0, e1) = umull(x0, y0);
            let (e2, e3) = umull(x1, y1);
            let (e4, e5) = umull(x2, y2);
            let (e6, e7) = umull(x3, y3);

            let (lo, hi) = umull(x0, y1);
            let (e1, cc) = addcarry_u64(e1, lo, 0);
            let (e2, cc) = addcarry_u64(e2, hi, cc);
            let (lo, hi) = umull(x0, y3);
            let (e3, cc) = addcarry_u64(e3, lo, cc);
            let (e4, cc) = addcarry_u64(e4, hi, cc);
            let (lo, hi) = umull(x2, y3);
            let (e5, cc) = addcarry_u64(e5, lo, cc);
            let (e6, cc) = addcarry_u64(e6, hi, cc);
            let (e7, _)  = addcarry_u64(e7, 0, cc);

            let (lo, hi) = umull(x1, y0);
            let (e1, cc) = addcarry_u64(e1, lo, 0);
            let (e2, cc) = addcarry_u64(e2, hi, cc);
            let (lo, hi) = umull(x3, y0);
            let (e3, cc) = addcarry_u64(e3, lo, cc);
            let (e4, cc) = addcarry_u64(e4, hi, cc);
            let (lo, hi) = umull(x3, y2);
            let (e5, cc) = addcarry_u64(e5, lo, cc);
            let (e6, cc) = addcarry_u64(e6, hi, cc);
            let (e7, _)  = addcarry_u64(e7, 0, cc);

            let (lo, hi) = umull(x0, y2);
            let (e2, cc) = addcarry_u64(e2, lo, 0);
            let (e3, cc) = addcarry_u64(e3, hi, cc);
            let (lo, hi) = umull(x1, y3);
            let (e4, cc) = addcarry_u64(e4, lo, cc);
            let (e5, cc) = addcarry_u64(e5, hi, cc);
            let (e6, cc) = addcarry_u64(e6, 0, cc);
            let (e7, _)  = addcarry_u64(e7, 0, cc);

            let (lo, hi) = umull(x2, y0);
            let (e2, cc) = addcarry_u64(e2, lo, 0);
            let (e3, cc) = addcarry_u64(e3, hi, cc);
            let (lo, hi) = umull(x3, y1);
            let (e4, cc) = addcarry_u64(e4, lo, cc);
            let (e5, cc) = addcarry_u64(e5, hi, cc);
            let (e6, cc) = addcarry_u64(e6, 0, cc);
            let (e7, _)  = addcarry_u64(e7, 0, cc);

            let (lo, hi) = umull(x1, y2);
            let (lo2, hi2) = umull(x2, y1);
            let (lo, cc) = addcarry_u64(lo, lo2, 0);
            let (hi, tt) = addcarry_u64(hi, hi2, cc);
            let (e3, cc) = addcarry_u64(e3, lo, 0);
            let (e4, cc) = addcarry_u64(e4, hi, cc);
            let (e5, cc) = addcarry_u64(e5, tt as u64, cc);
            let (e6, cc) = addcarry_u64(e6, 0, cc);
            let (e7, _)  = addcarry_u64(e7, 0, cc);

            (e0, e1, e2, e3, e4, e5, e6, e7)
        }

        // f <- a0..3 * b0..3
        let (f0, f1, f2, f3, f4, f5, f6) =
            mul224(a0, a1, a2, a3, b0, b1, b2, b3);
        // h <- a4..7 * b4..7
        let (h0, h1, h2, h3, h4, h5, h6) =
            mul224(a4, a5, a6, a7, b4, b5, b6, b7);
        // g <- (a0..3 + a4..7) * (b0..3 + b4..7)
        let (c0, r) = addcarry_u64(a0, a4, 0);
        let (c1, r) = addcarry_u64(a1, a5, r);
        let (c2, r) = addcarry_u64(a2, a6, r);
        let (c3, _) = addcarry_u64(a3, a7, r);
        let (e0, r) = addcarry_u64(b0, b4, 0);
        let (e1, r) = addcarry_u64(b1, b5, r);
        let (e2, r) = addcarry_u64(b2, b6, r);
        let (e3, _) = addcarry_u64(b3, b7, r);
        let (g0, g1, g2, g3, g4, g5, g6, g7) =
            mul256(c0, c1, c2, c3, e0, e1, e2, e3);

        // We have:
        //    a*b = f + (2^224)*(g - f - h) + (2^448)*h
        // Since 2^448 = 2^224 + 1 in the field, this simplifies into:
        //    a*b = f + (2^224)*(g - f - h) + h + (2^224)*h  mod p
        //        = (f + h) + (2^224)*(g - f)  mod p
        // f and h have size 448 bits. g-f is non-negative and can use up
        // to 450 bits.

        // j <- g - f  (450 bits)
        let (j0, r) = subborrow_u64(g0, f0, 0);
        let (j1, r) = subborrow_u64(g1, f1, r);
        let (j2, r) = subborrow_u64(g2, f2, r);
        let (j3, r) = subborrow_u64(g3, f3, r);
        let (j4, r) = subborrow_u64(g4, f4, r);
        let (j5, r) = subborrow_u64(g5, f5, r);
        let (j6, r) = subborrow_u64(g6, f6, r);
        let (j7, _) = subborrow_u64(g7, 0, r);

        // d <- f + h  (449 bits)
        let (d0, r) = addcarry_u64(f0, h0, 0);
        let (d1, r) = addcarry_u64(f1, h1, r);
        let (d2, r) = addcarry_u64(f2, h2, r);
        let (d3, r) = addcarry_u64(f3, h3, r);
        let (d4, r) = addcarry_u64(f4, h4, r);
        let (d5, r) = addcarry_u64(f5, h5, r);
        let (d6, r) = addcarry_u64(f6, h6, r);
        let d7 = r as u64;

        // Split j into low and high halves (k) and add them together (m).
        let k0 = j0;
        let k1 = j1;
        let k2 = j2;
        let k3 = j3 & 0x00000000FFFFFFFF;
        let k4 = (j3 >> 32) | (j4 << 32);
        let k5 = (j4 >> 32) | (j5 << 32);
        let k6 = (j5 >> 32) | (j6 << 32);
        let k7 = (j6 >> 32) | (j7 << 32);
        let (m0, r) = addcarry_u64(k0, k4, 0);
        let (m1, r) = addcarry_u64(k1, k5, r);
        let (m2, r) = addcarry_u64(k2, k6, r);
        let (m3, _) = addcarry_u64(k3, k7, r);

        // a*b = d0..7 + (2^224)*k0..3 + (2^448)*k4..7
        //     = d0..7 + k4..7 + (2^224)*(k0..3 + k4..7)
        //     = d0..7 + k4..7 + (2^224)*m0..3
        // len(d0..7) <= 449
        // len(k4..7) <= 226
        // len(m0..3) <= 227

        // Add k4..7 (truncated to 224 bits) + (2^224)*m0..3 to d0..7.
        let (d0, r) = addcarry_u64(d0, k4, 0);
        let (d1, r) = addcarry_u64(d1, k5, r);
        let (d2, r) = addcarry_u64(d2, k6, r);
        let (d3, r) = addcarry_u64(d3,
            (k7 & 0x00000000FFFFFFFF) | (m0 << 32), r);
        let (d4, r) = addcarry_u64(d4, (m0 >> 32) | (m1 << 32), r);
        let (d5, r) = addcarry_u64(d5, (m1 >> 32) | (m2 << 32), r);
        let (d6, r) = addcarry_u64(d6, (m2 >> 32) | (m3 << 32), r);
        let (d7, _) = addcarry_u64(d7, m3 >> 32, r);

        // d7 wraps around at positions 0 and 224. We also have two
        // bits from k7 to inject at position 224.
        let (d0, r) = addcarry_u64(d0, d7, 0);
        let (d1, r) = addcarry_u64(d1, 0, r);
        let (d2, r) = addcarry_u64(d2, 0, r);
        let (d3, r) = addcarry_u64(d3,
            (d7 << 32) + (k7 & 0xFFFFFFFF00000000), r);
        let (d4, r) = addcarry_u64(d4, 0, r);
        let (d5, r) = addcarry_u64(d5, 0, r);
        let (d6, r) = addcarry_u64(d6, 0, r);
        let hi = r as u64;

        // If there is a carry, then it means that the addition on d3
        // overflowed, so the current value of d3 is small-ish (at most
        // 36 bits) and the carry won't propagate beyond d3 (even though
        // it reinjects at bit 224 too).
        let (d0, r) = addcarry_u64(d0, hi, 0);
        let (d1, r) = addcarry_u64(d1, 0, r);
        let (d2, r) = addcarry_u64(d2, 0, r);
        let (d3, _) = addcarry_u64(d3, hi << 32, r);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
        self.0[4] = d4;
        self.0[5] = d5;
        self.0[6] = d6;
        */
    }

    // Square this value (in place).
    #[inline(always)]
    pub fn set_square(&mut self) {
        // 1. Square over integers -> 896 bits.
        // We first compute cross-products.
        let mut d = [0u64; 14];
        let mut cc;
        // a0*a1
        (d[1], d[2]) = umull(self.0[0], self.0[1]);
        // a0*a3
        (d[3], d[4]) = umull(self.0[0], self.0[3]);
        // a0*a5
        (d[5], d[6]) = umull(self.0[0], self.0[5]);
        // a0*a2
        let (lo, hi) = umull(self.0[0], self.0[2]);
        (d[2], cc) = addcarry_u64(d[2], lo, 0);
        (d[3], cc) = addcarry_u64(d[3], hi, cc);
        // a0*a4
        let (lo, hi) = umull(self.0[0], self.0[4]);
        (d[4], cc) = addcarry_u64(d[4], lo, cc);
        (d[5], cc) = addcarry_u64(d[5], hi, cc);
        // a0*a6
        let (lo, hi) = umull(self.0[0], self.0[6]);
        (d[6], cc) = addcarry_u64(d[6], lo, cc);
        (d[7], _) = addcarry_u64(0, hi, cc);

        // a1*a2
        let (lo, hi) = umull(self.0[1], self.0[2]);
        (d[3], cc) = addcarry_u64(d[3], lo, 0);
        (d[4], cc) = addcarry_u64(d[4], hi, cc);
        // a1*a4
        let (lo, hi) = umull(self.0[1], self.0[4]);
        (d[5], cc) = addcarry_u64(d[5], lo, cc);
        (d[6], cc) = addcarry_u64(d[6], hi, cc);
        // a1*a6
        let (lo, hi) = umull(self.0[1], self.0[6]);
        (d[7], cc) = addcarry_u64(d[7], lo, cc);
        (d[8], _) = addcarry_u64(0, hi, cc);
        // a1*a3
        let (lo, hi) = umull(self.0[1], self.0[3]);
        (d[4], cc) = addcarry_u64(d[4], lo, 0);
        (d[5], cc) = addcarry_u64(d[5], hi, cc);
        // a1*a5
        let (lo, hi) = umull(self.0[1], self.0[5]);
        (d[6], cc) = addcarry_u64(d[6], lo, cc);
        (d[7], cc) = addcarry_u64(d[7], hi, cc);
        (d[8], _) = addcarry_u64(d[8], 0, cc);

        // a2*a3
        let (lo, hi) = umull(self.0[2], self.0[3]);
        (d[5], cc) = addcarry_u64(d[5], lo, 0);
        (d[6], cc) = addcarry_u64(d[6], hi, cc);
        // a2*a5
        let (lo, hi) = umull(self.0[2], self.0[5]);
        (d[7], cc) = addcarry_u64(d[7], lo, cc);
        (d[8], cc) = addcarry_u64(d[8], hi, cc);
        d[9] = cc as u64;
        // a2*a4
        let (lo, hi) = umull(self.0[2], self.0[4]);
        (d[6], cc) = addcarry_u64(d[6], lo, 0);
        (d[7], cc) = addcarry_u64(d[7], hi, cc);
        // a2*a6
        let (lo, hi) = umull(self.0[2], self.0[6]);
        (d[8], cc) = addcarry_u64(d[8], lo, cc);
        (d[9], _) = addcarry_u64(d[9], hi, cc);

        // a3*a4
        let (lo, hi) = umull(self.0[3], self.0[4]);
        (d[7], cc) = addcarry_u64(d[7], lo, 0);
        (d[8], cc) = addcarry_u64(d[8], hi, cc);
        // a3*a6
        let (lo, hi) = umull(self.0[3], self.0[6]);
        (d[9], cc) = addcarry_u64(d[9], lo, cc);
        (d[10], _) = addcarry_u64(0, hi, cc);
        // a3*a5
        let (lo, hi) = umull(self.0[3], self.0[5]);
        (d[8], cc) = addcarry_u64(d[8], lo, 0);
        (d[9], cc) = addcarry_u64(d[9], hi, cc);
        (d[10], _) = addcarry_u64(d[10], 0, cc);

        // a4*a5
        let (lo, hi) = umull(self.0[4], self.0[5]);
        (d[9], cc) = addcarry_u64(d[9], lo, 0);
        (d[10], cc) = addcarry_u64(d[10], hi, cc);
        d[11] = cc as u64;
        // a4*a6
        let (lo, hi) = umull(self.0[4], self.0[6]);
        (d[10], cc) = addcarry_u64(d[10], lo, 0);
        (d[11], _) = addcarry_u64(d[11], hi, cc);

        // a5*a6
        let (lo, hi) = umull(self.0[5], self.0[6]);
        (d[11], cc) = addcarry_u64(d[11], lo, 0);
        (d[12], _) = addcarry_u64(0, hi, cc);

        // Double all cross-products.
        d[13] = d[12] >> 63;
        d[12] = (d[11] >> 63) | (d[12] << 1);
        d[11] = (d[10] >> 63) | (d[11] << 1);
        d[10] = (d[ 9] >> 63) | (d[10] << 1);
        d[ 9] = (d[ 8] >> 63) | (d[ 9] << 1);
        d[ 8] = (d[ 7] >> 63) | (d[ 8] << 1);
        d[ 7] = (d[ 6] >> 63) | (d[ 7] << 1);
        d[ 6] = (d[ 5] >> 63) | (d[ 6] << 1);
        d[ 5] = (d[ 4] >> 63) | (d[ 5] << 1);
        d[ 4] = (d[ 3] >> 63) | (d[ 4] << 1);
        d[ 3] = (d[ 2] >> 63) | (d[ 3] << 1);
        d[ 2] = (d[ 1] >> 63) | (d[ 2] << 1);
        d[ 1] = d[ 1] << 1;

        // Add squares.
        let (lo, hi) = umull(self.0[0], self.0[0]);
        d[0] = lo;
        (d[1], cc) = addcarry_u64(d[1], hi, 0);
        let (lo, hi) = umull(self.0[1], self.0[1]);
        (d[2], cc) = addcarry_u64(d[2], lo, cc);
        (d[3], cc) = addcarry_u64(d[3], hi, cc);
        let (lo, hi) = umull(self.0[2], self.0[2]);
        (d[4], cc) = addcarry_u64(d[4], lo, cc);
        (d[5], cc) = addcarry_u64(d[5], hi, cc);
        let (lo, hi) = umull(self.0[3], self.0[3]);
        (d[6], cc) = addcarry_u64(d[6], lo, cc);
        (d[7], cc) = addcarry_u64(d[7], hi, cc);
        let (lo, hi) = umull(self.0[4], self.0[4]);
        (d[8], cc) = addcarry_u64(d[8], lo, cc);
        (d[9], cc) = addcarry_u64(d[9], hi, cc);
        let (lo, hi) = umull(self.0[5], self.0[5]);
        (d[10], cc) = addcarry_u64(d[10], lo, cc);
        (d[11], cc) = addcarry_u64(d[11], hi, cc);
        let (lo, hi) = umull(self.0[6], self.0[6]);
        (d[12], cc) = addcarry_u64(d[12], lo, cc);
        (d[13], _) = addcarry_u64(d[13], hi, cc);

        // 2. Reduction.
        // If we write the current value as:
        //   d = dlow + e*2^448 + f*2^(448+224)
        // with e and f both lower than 2^224, then the reduction is:
        //   d = dlow + e + e*2^224 + f*2^224 + f*2^448  mod p
        //     = dlow + (e + f) + (e + 2*f)*2^224  mod p
        // We can thus compute:
        //   g = e + f
        //   h = g + f
        // and then add g + h*2^224 to dlow.
        // If we write g = gl + gh*2^224 (with gh = 0 or 1),
        // then:
        //   g + h*2^224 = gl + (h + gh)*2^224
        let e0 = d[7];
        let e1 = d[8];
        let e2 = d[9];
        let e3 = d[10] & 0x00000000FFFFFFFF;
        let f0 = (d[10] >> 32) | (d[11] << 32);
        let f1 = (d[11] >> 32) | (d[12] << 32);
        let f2 = (d[12] >> 32) | (d[13] << 32);
        let f3 = d[13] >> 32;
        let (g0, cc) = addcarry_u64(e0, f0, 0);
        let (g1, cc) = addcarry_u64(e1, f1, cc);
        let (g2, cc) = addcarry_u64(e2, f2, cc);
        let (g3, _) = addcarry_u64(e3, f3, cc);
        let gh = g3 >> 32;
        let (h0, cc) = addcarry_u64(g0, f0, gh as u8);
        let (h1, cc) = addcarry_u64(g1, f1, cc);
        let (h2, cc) = addcarry_u64(g2, f2, cc);
        let (h3, _) = addcarry_u64(g3, f3, cc);
        let g3 = g3 & 0x00000000FFFFFFFF;

        // gl is in g0..g3 (224 bits), and h+gh is in h0..h3 (at most 226 bits)
        let mut cc;
        (self.0[0], cc) = addcarry_u64(d[0], g0, 0);
        (self.0[1], cc) = addcarry_u64(d[1], g1, cc);
        (self.0[2], cc) = addcarry_u64(d[2], g2, cc);
        (self.0[3], cc) = addcarry_u64(d[3], g3 | (h0 << 32), cc);
        (self.0[4], cc) = addcarry_u64(d[4], (h0 >> 32) | (h1 << 32), cc);
        (self.0[5], cc) = addcarry_u64(d[5], (h1 >> 32) | (h2 << 32), cc);
        (self.0[6], cc) = addcarry_u64(d[6], (h2 >> 32) | (h3 << 32), cc);
        let (x, _) = addcarry_u64(h3 >> 32, 0, cc);
        self.reduce_small(x);
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

    // Ensure that the internal value is in the 0..p-1 range.
    #[inline]
    fn set_normalized(&mut self) {
        // Add 1 + 2^224.
        let (d0, cc) = addcarry_u64(self.0[0], 1, 0);
        let (d1, cc) = addcarry_u64(self.0[1], 0, cc);
        let (d2, cc) = addcarry_u64(self.0[2], 0, cc);
        let (d3, cc) = addcarry_u64(self.0[3], 1u64 << 32, cc);
        let (d4, cc) = addcarry_u64(self.0[4], 0, cc);
        let (d5, cc) = addcarry_u64(self.0[5], 0, cc);
        let (d6, cc) = addcarry_u64(self.0[6], 0, cc);

        // If there is no carry, then the original value was less
        // than 2^448 - 2^224 - 1 = p, i.e. already normalized. If there
        // is a carry, then we must subtract p from the value, i.e.
        // subtract 2^448 and add 1 + 2^224, which is exactly what we
        // just computed. Thus, we only need to replace the current value
        // with d if and only if there is a carry.
        let m = (cc as u64).wrapping_neg();
        self.0[0] ^= m & (self.0[0] ^ d0);
        self.0[1] ^= m & (self.0[1] ^ d1);
        self.0[2] ^= m & (self.0[2] ^ d2);
        self.0[3] ^= m & (self.0[3] ^ d3);
        self.0[4] ^= m & (self.0[4] ^ d4);
        self.0[5] ^= m & (self.0[5] ^ d5);
        self.0[6] ^= m & (self.0[6] ^ d6);
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
        let (lo, mut cc) = umull_x2(tu.0[0], f, tv.0[0], g);
        self.0[0] = lo;
        for i in 1..7 {
            let (lo, hi) = umull_x2_add(tu.0[i], f, tv.0[i], g, cc);
            self.0[i] = lo;
            cc = hi;
        }

        // Upper word cc can be up to 63 bits.
        self.reduce_small(cc);
    }

    #[inline(always)]
    fn lin(a: &Self, b: &Self, f: u64, g: u64) -> Self {
        let mut r = Self::ZERO;
        r.set_lin(a, b, f, g);
        r
    }

    // Set this value to abs((a*f+b*g)/2^31). Values a and b are interpreted
    // as unsigned 448-bit integers. Coefficients f and g are provided as u64,
    // but they really are signed integers in the -2^31..+2^31 range
    // (inclusive). The low 31 bits are dropped (i.e. the division is assumed
    // to be exact). The result is assumed to fit in 448 bits (otherwise,
    // truncation occurs).
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
        let mut aa = [0u64; 8];
        let (d, mut cc) = subborrow_u64(a.0[0] ^ sf, sf, 0);
        aa[0] = d;
        for i in 1..7 {
            let (d, ee) = subborrow_u64(a.0[i] ^ sf, sf, cc);
            aa[i] = d;
            cc = ee;
        }
        aa[7] = (cc as u64).wrapping_neg();
        let mut bb = [0u64; 8];
        let (d, mut cc) = subborrow_u64(b.0[0] ^ sg, sg, 0);
        bb[0] = d;
        for i in 1..7 {
            let (d, ee) = subborrow_u64(b.0[i] ^ sg, sg, cc);
            bb[i] = d;
            cc = ee;
        }
        bb[7] = (cc as u64).wrapping_neg();

        // Compute a*f+b*g into xx. Since f and g are at most 2^31, we can
        // add two 128-bit products with no overflow.
        let mut xx = [0u64; 8];
        let (lo, mut t) = umull_x2(aa[0], f, bb[0], g);
        xx[0] = lo;
        for i in 1..8 {
            let (lo, hi) = umull_x2_add(aa[i], f, bb[i], g, t);
            xx[i] = lo;
            t = hi;
        }

        // Negate the result if it is negative.
        let m = sgnw(xx[7]);
        (xx[0], cc) = subborrow_u64(xx[0] ^ m, m, 0);
        for i in 1..8 {
            (xx[i], cc) = subborrow_u64(xx[i] ^ m, m, cc);
        }

        // Right-shift the value by 31 bits and write it into self.
        for i in 0..7 {
            self.0[i] = (xx[i] >> 31) | (xx[i + 1] << 33);
        }

        // Returned value is -1 if the intermediate a*f+b*g was negative.
        m
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
        //   b <- p (modulus)
        //   u <- x (self)
        //   v <- 0
        //
        // Value a is normalized (in the 0..p-1 range). Values a and b are
        // then considered as (signed) integers. Values u and v are field
        // elements.
        //
        // Invariants:
        //    a*x = y*u mod p
        //    b*x = y*v mod p
        //    b is always odd
        //
        // At each step:
        //    if a is even, then:
        //        a <- a/2, u <- u/2 mod p
        //    else:
        //        if a < b:
        //            (a, u, b, v) <- (b, v, a, u)
        //        a <- (a-b)/2, u <- (u-v)/2 mod p
        //
        // What we implement below is the optimized version of this
        // algorithm, as described in https://eprint.iacr.org/2020/972

        let mut a = *y;
        a.set_normalized();
        let mut b = Self(Self::MODULUS);
        let mut u = *self;
        let mut v = Self::ZERO;

        // Generic loop does 27*31 = 837 inner iterations.
        for _ in 0..27 {
            // Get approximations of a and b over 64 bits:
            //  - If len(a) <= 64 and len(b) <= 64, then we just use
            //    their values (low limbs).
            //  - Otherwise, with n = max(len(a), len(b)), we use:
            //       (a mod 2^31) + 2^31*floor(a / 2^(n - 33))
            //       (b mod 2^31) + 2^31*floor(b / 2^(n - 33))
            let mut c_hi = 0xFFFFFFFFFFFFFFFFu64;
            let mut c_lo = 0xFFFFFFFFFFFFFFFFu64;
            let mut a_hi = 0u64;
            let mut a_lo = 0u64;
            let mut b_hi = 0u64;
            let mut b_lo = 0u64;
            for j in (0..7).rev() {
                let aw = a.0[j];
                let bw = b.0[j];
                a_hi ^= (a_hi ^ aw) & c_hi;
                a_lo ^= (a_lo ^ aw) & c_lo;
                b_hi ^= (b_hi ^ bw) & c_hi;
                b_lo ^= (b_lo ^ bw) & c_lo;
                c_lo = c_hi;
                let mw = aw | bw;
                c_hi &= ((mw | mw.wrapping_neg()) >> 63).wrapping_sub(1);
            }

            // If c_lo = 0, then we grabbed two words for a and b.
            // If c_lo != 0 but c_hi = 0, then we grabbed one word
            // (in a_hi / b_hi), which means that both values are at
            // most 64 bits.
            // It is not possible that c_hi != 0 because b != 0 (i.e.
            // we must have encountered at least one non-zero word).
            let s = lzcnt(a_hi | b_hi);
            let mut xa = (a_hi << s) | ((a_lo >> 1) >> (63 - s));
            let mut xb = (b_hi << s) | ((b_lo >> 1) >> (63 - s));
            xa = (xa & 0xFFFFFFFF80000000) | (a.0[0] & 0x000000007FFFFFFF);
            xb = (xb & 0xFFFFFFFF80000000) | (b.0[0] & 0x000000007FFFFFFF);

            // If c_lo != 0, then the computed values for xa and xb should
            // be ignored, since both a and b fit in a single word each.
            xa ^= c_lo & (xa ^ a.0[0]);
            xb ^= c_lo & (xb ^ b.0[0]);

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
        // len(a) + len(b) <= 59, so we can end the computation with
        // the low words directly. We only need 57 iterations to reach
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
        for _ in 0..57 {
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
        // each of the 27*31+57 = 894 iterations, so we must divide by
        // 2^894 (mod p). This is done with a multiplication by the
        // appropriate constant.
        self.set_mul(&Self::INVT894);
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
        // We use batches of 100 elements; larger batches only yield
        // moderate improvements, while sticking to a fixed moderate batch
        // size allows stack-based allocation.
        let n = xx.len();
        let mut i = 0;
        while i < n {
            let blen = if (n - i) > 100 { 100 } else { n - i };
            let mut tt = [Self::ZERO; 100];
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
        // in a total of 894 iterations.

        let mut a = self;
        a.set_normalized();
        let mut b = Self(Self::MODULUS);
        let mut ls = 0u64;  // running symbol information in the low bit

        // Outer loop
        for _ in 0..27 {
            // Get approximations of a and b over 64 bits.
            let mut c_hi = 0xFFFFFFFFFFFFFFFFu64;
            let mut c_lo = 0xFFFFFFFFFFFFFFFFu64;
            let mut a_hi = 0u64;
            let mut a_lo = 0u64;
            let mut b_hi = 0u64;
            let mut b_lo = 0u64;
            for j in (0..7).rev() {
                let aw = a.0[j];
                let bw = b.0[j];
                a_hi ^= (a_hi ^ aw) & c_hi;
                a_lo ^= (a_lo ^ aw) & c_lo;
                b_hi ^= (b_hi ^ bw) & c_hi;
                b_lo ^= (b_lo ^ bw) & c_lo;
                c_lo = c_hi;
                let mw = aw | bw;
                c_hi &= ((mw | mw.wrapping_neg()) >> 63).wrapping_sub(1);
            }

            let s = lzcnt(a_hi | b_hi);
            let mut xa = (a_hi << s) | ((a_lo >> 1) >> (63 - s));
            let mut xb = (b_hi << s) | ((b_lo >> 1) >> (63 - s));
            xa = (xa & 0xFFFFFFFF80000000) | (a.0[0] & 0x000000007FFFFFFF);
            xb = (xb & 0xFFFFFFFF80000000) | (b.0[0] & 0x000000007FFFFFFF);

            xa ^= c_lo & (xa ^ a.0[0]);
            xb ^= c_lo & (xb ^ b.0[0]);

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

        // Final iterations: values are at most 59 bits now. We do not
        // need to keep track of update coefficients. Just like the GCD,
        // we need only 57 iterations, because after 57 iterations,
        // value a is 0 or 1, and b is 1, and no further modification to
        // the Legendre symbol may happen.
        let mut xa = a.0[0];
        let mut xb = b.0[0];
        for _ in 0..57 {
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
        (r & !self.iszero()) as i32
    }

    // Set this value to its square root. Returned value is 0xFFFFFFFF
    // if the operation succeeded (value was indeed a quadratic
    // residue), 0 otherwise (value was not a quadratic residue). In the
    // latter case, this value is set to the square root of -self.
    // In all cases, the returned root is the one whose least significant
    // bit is 0 (when normalized in 0..p-1).
    fn set_sqrt_ext(&mut self) -> u32 {
        // Candidate root is y = x^((p+1)/4).
        // We have: (p+1)/4 = 2^446 - 2^222
        let z = *self;
        let zp2 = z.square() * z;
        let zp3 = zp2.square() * z;
        let zp4 = zp3.square() * z;
        let zp7 = zp4.xsquare(3) * zp3;
        let zp14 = zp7.xsquare(7) * zp7;
        let zp28 = zp14.xsquare(14) * zp14;
        let zp56 = zp28.xsquare(28) * zp28;
        let zp112 = zp56.xsquare(56) * zp56;
        let zp224 = zp112.xsquare(112) * zp112;
        let mut y = zp224.xsquare(222);

        // Normalize y and negate it if necessary to set the low bit to 0.
        y.set_normalized();
        y.set_cond(&-y, ((y.0[0] as u32) & 1).wrapping_neg());

        // Check that the candidate is indeed a square root.
        // Note that: y^2 = x^((p+1)/2) = x*(x^((p-1)/2))
        // If x is a non-zero square, then x^((p-1)/2) = 1, and y is
        // a square root of x; if x is not a square, then x^((p-1)/2) = -1,
        // and we indeed have y^2 = -x, as expected by the API.
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

    // Compute the square root of this value. Returned values are (y, r):
    //  - If this value is indeed a quadratic residue, then y is the
    //    square root whose least significant bit (when normalized in 0..p-1)
    //    is 0, and r is equal to 0xFFFFFFFF.
    //  - If this value is not a quadratic residue, then y is zero, and
    //    r is equal to 0.
    #[inline(always)]
    pub fn sqrt(self) -> (Self, u32) {
        let mut x = self;
        let r = x.set_sqrt();
        (x, r)
    }

    // Compute the square root of this value. Returned values are (y, r):
    //  - If this value is indeed a quadratic residue, then y is a
    //    square root of this value, and r is 0xFFFFFFFF.
    //  - If this value is not a quadratic residue, then y is set to
    //    a square root of -x, and r is 0x00000000.
    // In all cases, the returned root is normalized: the least significant
    // bit of its integer representation (in the 0..p-1 range) is 0.
    #[inline(always)]
    pub fn sqrt_ext(self) -> (Self, u32) {
        let mut x = self;
        let r = x.set_sqrt_ext();
        (x, r)
    }

    // TODO: split_vartime()?
    // This is not really useful for this field; split_vartime() is mostly
    // to help with ECDSA/EdDSA/Schnorr signature verification, and works
    // in the scalar field, not in the base curve field.

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
        // There are two possible representations of zero: 0 and p.
        let a0 = self.0[0];
        let a1 = self.0[1];
        let a2 = self.0[2];
        let a3 = self.0[3];
        let a4 = self.0[4];
        let a5 = self.0[5];
        let a6 = self.0[6];

        let t1 = a0 | a1 | a2 | a3 | a4 | a5 | a6;
        let t2 = !(a0 & a1 & a2 & (a3.wrapping_add(1u64 << 32)) & a4 & a5 & a6);

        // t1 == 0 if and only if the integer is 0
        // t2 == 0 if and only if the integer is p
        let r = (t1 | t1.wrapping_neg()) & (t2 | t2.wrapping_neg());
        ((r >> 63) as u32).wrapping_sub(1)
    }

    /* unused
    #[inline(always)]
    fn decode56_reduce(buf: &[u8]) -> Self {
        let mut r = Self::ZERO;
        if buf.len() == 56 {
            r.set_decode56_reduce(buf);
        }
        r
    }
    */

    #[inline(always)]
    fn set_decode56_reduce(&mut self, buf: &[u8]) {
        debug_assert!(buf.len() == 56);
        for i in 0..7 {
            self.0[i] = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[(i * 8)..(i * 8 + 8)]).unwrap());
        }
    }

    // Encode this value over exactly 56 bytes. Encoding is always canonical
    // (little-endian encoding of the value in the 0..p-1 range, top bit
    // of the last byte is always 0).
    #[inline(always)]
    pub fn encode(self) -> [u8; 56] {
        let mut r = self;
        r.set_normalized();
        let mut d = [0u8; 56];
        for i in 0..7 {
            d[(i * 8)..(i * 8 + 8)].copy_from_slice(&r.0[i].to_le_bytes());
        }
        d
    }

    // Decode a field element from 56 bytes. On success, this sets this
    // element to the decoded value, and returns 0xFFFFFFFF. If the source
    // encoding is not canonical (i.e. the unsigned little-endian
    // interpretation of the 56 bytes is not lower than the field modulus p,
    // or if the source slice has not length exactly 56 bytes), then this
    // sets this element to zero, and returns 0.
    #[inline]
    pub fn set_decode_ct(&mut self, buf: &[u8]) -> u32 {
        *self = Self::ZERO;
        if buf.len() != 56 {
            return 0;
        }

        self.set_decode56_reduce(buf);

        // If adding 1 + 2^224 yields a carry, then the input value was
        // not canonical and must be zeroed.
        let (_, cc) = addcarry_u64(self.0[0], 1, 0);
        let (_, cc) = addcarry_u64(self.0[1], 0, cc);
        let (_, cc) = addcarry_u64(self.0[2], 0, cc);
        let (_, cc) = addcarry_u64(self.0[3], 1u64 << 32, cc);
        let (_, cc) = addcarry_u64(self.0[4], 0, cc);
        let (_, cc) = addcarry_u64(self.0[5], 0, cc);
        let (_, cc) = addcarry_u64(self.0[6], 0, cc);
        let m = !(cc as u64).wrapping_neg();
        for i in 0..7 {
            self.0[i] &= m;
        }

        m as u32
    }

    // Decode a field element from 56 bytes. On success, this returns the
    // new element, and a status 0xFFFFFFFF. If the source encoding is not
    // canonical (i.e. the unsigned little-endian interpretation of the 56
    // bytes is not lower than the field modulus p, or if the source slice
    // has not length exactly 56 bytes), then this sets this returns the
    // element zero and the status 0x00000000.
    #[inline(always)]
    pub fn decode_ct(buf: &[u8]) -> (Self, u32) {
        let mut r = Self::ZERO;
        let cc = r.set_decode_ct(buf);
        (r, cc)
    }

    // Decode a field element from 56 bytes. On success, this returns the
    // new element. If the source encoding is not canonical (i.e. the
    // unsigned little-endian interpretation of the 56 bytes is not lower
    // than the field modulus p, or if the source slice has not length
    // exactly 56 bytes), then this sets this returns `None`.
    #[inline(always)]
    pub fn decode(buf: &[u8]) -> Option<Self> {
        let (r, cc) = Self::decode_ct(buf);
        if cc != 0 {
            Some(r)
        } else {
            None
        }
    }

    // Decode a field element from some bytes. The bytes are interpreted
    // in unsigned little-endian convention, and the resulting integer is
    // reduced modulo p. This process never fails.
    pub fn set_decode_reduce(&mut self, buf: &[u8]) {
        *self = Self::ZERO;
        let mut n = buf.len();
        if n == 0 {
            return;
        }

        // Get high bytes so that:
        //  - we get at most 56 bytes
        //  - remaining number of bytes will be multiple of 28
        let mut n1 = n % 28;
        if n1 == 0 {
            n1 = 28;
        }
        if n >= 28 + n1 {
            n1 += 28;
        }
        n -= n1;
        let mut tmp = [0u8; 56];
        tmp[..n1].copy_from_slice(&buf[n..]);
        self.set_decode56_reduce(&tmp);

        // Process remaining 28-byte chunks in high to low order.
        // For each chunk, the current value is multiplied by 2^224,
        // and the new chunk is added in. Namely:
        // current value: r = rl + rh*2^224
        // new chunk: t (with 0 <= t < 2^224)
        //   t + r*2^224 = t + rl*2^224 + rh*2^448
        //               = t + rl*2^224 + rh + rh*2^224  mod p
        while n > 0 {
            n -= 28;
            let t0 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[n..(n + 8)]).unwrap());
            let t1 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[(n + 8)..(n + 16)]).unwrap());
            let t2 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[(n + 16)..(n + 24)]).unwrap());
            let t3 = u32::from_le_bytes(*<&[u8; 4]>::try_from(&buf[(n + 24)..(n + 28)]).unwrap()) as u64;

            // s <- t + rl*2^224
            let s = Self::from_w64le([
                t0, t1, t2, t3 | (self.0[0] << 32),
                (self.0[0] >> 32) | (self.0[1] << 32),
                (self.0[1] >> 32) | (self.0[2] << 32),
                (self.0[2] >> 32) | (self.0[3] << 32) ]);

            // r <- rh + rh*2^224
            self.0[0] = (self.0[3] >> 32) | (self.0[4] << 32);
            self.0[1] = (self.0[4] >> 32) | (self.0[5] << 32);
            self.0[2] = (self.0[5] >> 32) | (self.0[6] << 32);
            self.0[3] = (self.0[6] >> 32) | (self.0[3] & 0xFFFFFFFF00000000);

            *self += s;
        }
    }

    // Decode a field element from some bytes. The bytes are interpreted
    // in unsigned little-endian convention, and the resulting integer is
    // reduced modulo p. This process never fails.
    pub fn decode_reduce(buf: &[u8]) -> Self {
        let mut r = Self::ZERO;
        r.set_decode_reduce(buf);
        r
    }
}

// ========================================================================
// Implementations of all the traits needed to use the simple operators
// (+, *, /...) on field element instances, with or without references.

impl Add<GF448> for GF448 {
    type Output = GF448;

    #[inline(always)]
    fn add(self, other: GF448) -> GF448 {
        let mut r = self;
        r.set_add(&other);
        r
    }
}

impl Add<&GF448> for GF448 {
    type Output = GF448;

    #[inline(always)]
    fn add(self, other: &GF448) -> GF448 {
        let mut r = self;
        r.set_add(other);
        r
    }
}

impl Add<GF448> for &GF448 {
    type Output = GF448;

    #[inline(always)]
    fn add(self, other: GF448) -> GF448 {
        let mut r = *self;
        r.set_add(&other);
        r
    }
}

impl Add<&GF448> for &GF448 {
    type Output = GF448;

    #[inline(always)]
    fn add(self, other: &GF448) -> GF448 {
        let mut r = *self;
        r.set_add(other);
        r
    }
}

impl AddAssign<GF448> for GF448 {
    #[inline(always)]
    fn add_assign(&mut self, other: GF448) {
        self.set_add(&other);
    }
}

impl AddAssign<&GF448> for GF448 {
    #[inline(always)]
    fn add_assign(&mut self, other: &GF448) {
        self.set_add(other);
    }
}

impl Div<GF448> for GF448 {
    type Output = GF448;

    #[inline(always)]
    fn div(self, other: GF448) -> GF448 {
        let mut r = self;
        r.set_div(&other);
        r
    }
}

impl Div<&GF448> for GF448 {
    type Output = GF448;

    #[inline(always)]
    fn div(self, other: &GF448) -> GF448 {
        let mut r = self;
        r.set_div(other);
        r
    }
}

impl Div<GF448> for &GF448 {
    type Output = GF448;

    #[inline(always)]
    fn div(self, other: GF448) -> GF448 {
        let mut r = *self;
        r.set_div(&other);
        r
    }
}

impl Div<&GF448> for &GF448 {
    type Output = GF448;

    #[inline(always)]
    fn div(self, other: &GF448) -> GF448 {
        let mut r = *self;
        r.set_div(other);
        r
    }
}

impl DivAssign<GF448> for GF448 {
    #[inline(always)]
    fn div_assign(&mut self, other: GF448) {
        self.set_div(&other);
    }
}

impl DivAssign<&GF448> for GF448 {
    #[inline(always)]
    fn div_assign(&mut self, other: &GF448) {
        self.set_div(other);
    }
}

impl Mul<GF448> for GF448 {
    type Output = GF448;

    #[inline(always)]
    fn mul(self, other: GF448) -> GF448 {
        let mut r = self;
        r.set_mul(&other);
        r
    }
}

impl Mul<&GF448> for GF448 {
    type Output = GF448;

    #[inline(always)]
    fn mul(self, other: &GF448) -> GF448 {
        let mut r = self;
        r.set_mul(other);
        r
    }
}

impl Mul<GF448> for &GF448 {
    type Output = GF448;

    #[inline(always)]
    fn mul(self, other: GF448) -> GF448 {
        let mut r = *self;
        r.set_mul(&other);
        r
    }
}

impl Mul<&GF448> for &GF448 {
    type Output = GF448;

    #[inline(always)]
    fn mul(self, other: &GF448) -> GF448 {
        let mut r = *self;
        r.set_mul(other);
        r
    }
}

impl MulAssign<GF448> for GF448 {
    #[inline(always)]
    fn mul_assign(&mut self, other: GF448) {
        self.set_mul(&other);
    }
}

impl MulAssign<&GF448> for GF448 {
    #[inline(always)]
    fn mul_assign(&mut self, other: &GF448) {
        self.set_mul(other);
    }
}

impl Neg for GF448 {
    type Output = GF448;

    #[inline(always)]
    fn neg(self) -> GF448 {
        let mut r = self;
        r.set_neg();
        r
    }
}

impl Neg for &GF448 {
    type Output = GF448;

    #[inline(always)]
    fn neg(self) -> GF448 {
        let mut r = *self;
        r.set_neg();
        r
    }
}

impl Sub<GF448> for GF448 {
    type Output = GF448;

    #[inline(always)]
    fn sub(self, other: GF448) -> GF448 {
        let mut r = self;
        r.set_sub(&other);
        r
    }
}

impl Sub<&GF448> for GF448 {
    type Output = GF448;

    #[inline(always)]
    fn sub(self, other: &GF448) -> GF448 {
        let mut r = self;
        r.set_sub(other);
        r
    }
}

impl Sub<GF448> for &GF448 {
    type Output = GF448;

    #[inline(always)]
    fn sub(self, other: GF448) -> GF448 {
        let mut r = *self;
        r.set_sub(&other);
        r
    }
}

impl Sub<&GF448> for &GF448 {
    type Output = GF448;

    #[inline(always)]
    fn sub(self, other: &GF448) -> GF448 {
        let mut r = *self;
        r.set_sub(other);
        r
    }
}

impl SubAssign<GF448> for GF448 {
    #[inline(always)]
    fn sub_assign(&mut self, other: GF448) {
        self.set_sub(&other);
    }
}

impl SubAssign<&GF448> for GF448 {
    #[inline(always)]
    fn sub_assign(&mut self, other: &GF448) {
        self.set_sub(other);
    }
}

// ========================================================================

#[cfg(test)]
mod tests {

    use super::{GF448};
    use num_bigint::{BigInt, Sign};
    use crate::sha2::Sha512;

    /*
    fn print(name: &str, v: GF448) {
        println!("{} = 0x{:016X}{:016X}{:016X}{:016X}{:016X}{:016X}{:016X}",
            name, v.0[6], v.0[5], v.0[4], v.0[3], v.0[2], v.0[1], v.0[0]);
    }
    */

    // va, vb and vx must be 56 bytes each in length
    fn check_gf_ops(va: &[u8], vb: &[u8], vx: &[u8]) {
        let zp = BigInt::from_slice(Sign::Plus, &[
            0xFFFFFFFFu32, 0xFFFFFFFFu32, 0xFFFFFFFFu32, 0xFFFFFFFFu32,
            0xFFFFFFFFu32, 0xFFFFFFFFu32, 0xFFFFFFFFu32, 0xFFFFFFFEu32,
            0xFFFFFFFFu32, 0xFFFFFFFFu32, 0xFFFFFFFFu32, 0xFFFFFFFFu32,
            0xFFFFFFFFu32, 0xFFFFFFFFu32,
        ]);
        let zp4 = &zp << 2;

        let mut a = GF448::ZERO;
        a.set_decode56_reduce(va);
        let mut b = GF448::ZERO;
        b.set_decode56_reduce(vb);
        let za = BigInt::from_bytes_le(Sign::Plus, va);
        let zb = BigInt::from_bytes_le(Sign::Plus, vb);

        let vc = a.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = &za % &zp;
        assert!(zc == zd);

        let c = a + b;
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za + &zb) % &zp;
        assert!(zc == zd);

        let c = a - b;
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = ((&zp4 + &za) - &zb) % &zp;
        assert!(zc == zd);

        let c = -a;
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&zp4 - &za) % &zp;
        assert!(zc == zd);

        let c = a * b;
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za * &zb) % &zp;
        assert!(zc == zd);

        let c = a.half();
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd: BigInt = ((&zp4 + (&zc << 1)) - &za) % &zp;
        assert!(zd.sign() == Sign::NoSign);

        let c = a.mul2();
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za << 1) % &zp;
        assert!(zc == zd);

        let c = a.mul4();
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za << 2) % &zp;
        assert!(zc == zd);

        let c = a.mul8();
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za << 3) % &zp;
        assert!(zc == zd);

        let c = a.mul16();
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za << 4) % &zp;
        assert!(zc == zd);

        let c = a.mul32();
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za << 5) % &zp;
        assert!(zc == zd);

        let x = b.0[1] as u32;
        let c = a.mul_small(x);
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za * x) % &zp;
        assert!(zc == zd);

        let c = a.square();
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za * &za) % &zp;
        assert!(zc == zd);

        let (e, cc) = GF448::decode_ct(va);
        if cc != 0 {
            assert!(cc == 0xFFFFFFFF);
            assert!(e.encode() == va);
        } else {
            assert!(e.encode() == [0u8; 56]);
        }

        let mut tmp = [0u8; 168];
        tmp[0..56].copy_from_slice(va);
        tmp[56..112].copy_from_slice(vb);
        tmp[112..168].copy_from_slice(vx);
        for k in 0..169 {
            let c = GF448::decode_reduce(&tmp[0..k]);
            let vc = c.encode();
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

    #[test]
    fn gf448_ops() {
        let mut va = [0u8; 56];
        let mut vb = [0u8; 56];
        let mut vx = [0u8; 56];
        check_gf_ops(&va, &vb, &vx);
        assert!(GF448::decode_reduce(&va).iszero() == 0xFFFFFFFF);
        assert!(GF448::decode_reduce(&va).equals(GF448::decode_reduce(&vb)) == 0xFFFFFFFF);
        assert!(GF448::decode_reduce(&va).legendre() == 0);
        for i in 0..56 {
            va[i] = 0xFFu8;
            vb[i] = 0xFFu8;
            vx[i] = 0xFFu8;
        }
        check_gf_ops(&va, &vb, &vx);
        assert!(GF448::decode_reduce(&va).iszero() == 0);
        assert!(GF448::decode_reduce(&va).equals(GF448::decode_reduce(&vb)) == 0xFFFFFFFF);
        va[28] = 0xFEu8;
        assert!(GF448::decode_reduce(&va).iszero() == 0xFFFFFFFF);
        let mut sh = Sha512::new();
        for i in 0..300 {
            sh.update(((3 * i + 0) as u64).to_le_bytes());
            let va = &sh.finalize_reset()[..56];
            sh.update(((3 * i + 1) as u64).to_le_bytes());
            let vb = &sh.finalize_reset()[..56];
            sh.update(((3 * i + 2) as u64).to_le_bytes());
            let vx = &sh.finalize_reset()[..56];
            check_gf_ops(va, vb, vx);
            assert!(GF448::decode_reduce(&va).iszero() == 0);
            assert!(GF448::decode_reduce(&va).equals(GF448::decode_reduce(&vb)) == 0);
            let nqr = 7u32;
            let s = GF448::decode_reduce(&va).square();
            let s2 = s.mul_small(nqr);
            assert!(s.legendre() == 1);
            assert!(s2.legendre() == -1);
            let (t, r) = s.sqrt();
            assert!(r == 0xFFFFFFFF);
            assert!(t.square().equals(s) == 0xFFFFFFFF);
            assert!((t.encode()[0] & 1) == 0);
            let (t, r) = s.sqrt_ext();
            assert!(r == 0xFFFFFFFF);
            assert!(t.square().equals(s) == 0xFFFFFFFF);
            assert!((t.encode()[0] & 1) == 0);
            let (t2, r) = s2.sqrt();
            assert!(r == 0);
            assert!(t2.iszero() == 0xFFFFFFFF);
            let (t2, r) = s2.sqrt_ext();
            assert!(r == 0);
            assert!(t2.square().equals(-s2) == 0xFFFFFFFF);
        }
    }

    #[test]
    fn gf448_batch_invert() {
        let mut xx = [GF448::ZERO; 300];
        let mut sh = Sha512::new();
        for i in 0..300 {
            sh.update((i as u64).to_le_bytes());
            let v = sh.finalize_reset();
            xx[i] = GF448::decode_reduce(&v[0..56]);
        }
        xx[120] = GF448::ZERO;
        let mut yy = xx;
        GF448::batch_invert(&mut yy[..]);
        for i in 0..300 {
            if xx[i].iszero() != 0 {
                assert!(yy[i].iszero() == 0xFFFFFFFF);
            } else {
                assert!((xx[i] * yy[i]).equals(GF448::ONE) == 0xFFFFFFFF);
            }
        }
    }
}
