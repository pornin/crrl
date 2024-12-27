use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::convert::TryFrom;

use super::{addcarry_u64, subborrow_u64, umull, umull_x2, umull_x2_add, sgnw, lzcnt};

#[derive(Clone, Copy, Debug)]
pub struct GFp256([u64; 4]);

impl GFp256 {

    // Element encoding length (in bytes); always 32 bytes.
    pub const ENC_LEN: usize = 32;

    // Modulus q in base 2^64 (low-to-high order).
    pub const MODULUS: [u64; 4] = [
        0xFFFFFFFFFFFFFFFF,
        0x00000000FFFFFFFF,
        0x0000000000000000,
        0xFFFFFFFF00000001,
    ];

    pub const ZERO: Self = Self([ 0, 0, 0, 0 ]);
    pub const ONE: Self = Self([ 1, 0, 0, 0 ]);
    pub const MINUS_ONE: Self = Self([
        0xFFFFFFFFFFFFFFFE,
        0x00000000FFFFFFFF,
        0x0000000000000000,
        0xFFFFFFFF00000001,
    ]);

    // (q+1)/2
    const HALFMOD: [u64; 4] = [
        0x0000000000000000,
        0x0000000080000000,
        0x8000000000000000,
        0x7FFFFFFF80000000,
    ];

    // q*2^34
    const MODT34: [u64; 5] = [
        0xFFFFFFFC00000000,
        0xFFFFFFFFFFFFFFFF,
        0x0000000000000003,
        0x0000000400000000,
        0x00000003FFFFFFFC,
    ];

    // 1/2^510 in the field, as a constant; this is used when computing
    // divisions in the field. The value is computed at compile-time.
    const INVT510: GFp256 = Self([
        0x00000033FFFFFFDC,
        0xFFFFFFE00000001C,
        0xFFFFFFEC0000003F,
        0x00000017FFFFFFFF,
    ]);

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
        let (d0, cc) = addcarry_u64(x0, Self::MODULUS[0], 0);
        let (d1, cc) = addcarry_u64(xh, Self::MODULUS[1], cc);
        let (d2, cc) = addcarry_u64(xh, Self::MODULUS[2], cc);
        let (d3, _)  = addcarry_u64(xh, Self::MODULUS[3], cc);
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
        let (d0, cc) = addcarry_u64(x0, Self::MODULUS[0], 0);
        let (d1, cc) = addcarry_u64(xh, Self::MODULUS[1], cc);
        let (d2, cc) = addcarry_u64(xh, Self::MODULUS[2], cc);
        let (d3, _)  = addcarry_u64(xh, Self::MODULUS[3], cc);
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
        let (d0, cc) = addcarry_u64(x0, Self::MODULUS[0], 0);
        let (d1, cc) = addcarry_u64(x1, Self::MODULUS[1], cc);
        let (d2, cc) = addcarry_u64(xh, Self::MODULUS[2], cc);
        let (d3, _)  = addcarry_u64(xh, Self::MODULUS[3], cc);
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

        // 2. In case of an output carry, subtract q.
        let m = (cc as u64).wrapping_neg();
        let (d0, cc) = subborrow_u64(d0, Self::MODULUS[0] & m, 0);
        let (d1, cc) = subborrow_u64(d1, Self::MODULUS[1] & m, cc);
        let (d2, cc) = subborrow_u64(d2, Self::MODULUS[2] & m, cc);
        let (d3, cc) = subborrow_u64(d3, Self::MODULUS[3] & m, cc);

        // 3. If there was a carry, but no borrow, then subtracting q
        // once was not enough, we have to do it again.
        let m = m & (cc as u64).wrapping_sub(1);
        let (d0, cc) = subborrow_u64(d0, Self::MODULUS[0] & m, 0);
        let (d1, cc) = subborrow_u64(d1, Self::MODULUS[1] & m, cc);
        let (d2, cc) = subborrow_u64(d2, Self::MODULUS[2] & m, cc);
        let (d3, _)  = subborrow_u64(d3, Self::MODULUS[3] & m, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    #[inline]
    fn set_sub(&mut self, rhs: &Self) {
        // 1. Subtraction with borrow
        let (d0, cc) = subborrow_u64(self.0[0], rhs.0[0], 0);
        let (d1, cc) = subborrow_u64(self.0[1], rhs.0[1], cc);
        let (d2, cc) = subborrow_u64(self.0[2], rhs.0[2], cc);
        let (d3, cc) = subborrow_u64(self.0[3], rhs.0[3], cc);

        // 2. In case of an output borrow, add q.
        let m = (cc as u64).wrapping_neg();
        let (d0, cc) = addcarry_u64(d0, Self::MODULUS[0] & m, 0);
        let (d1, cc) = addcarry_u64(d1, Self::MODULUS[1] & m, cc);
        let (d2, cc) = addcarry_u64(d2, Self::MODULUS[2] & m, cc);
        let (d3, cc) = addcarry_u64(d3, Self::MODULUS[3] & m, cc);

        // 3. If there was a borrow, but no carry was produced to
        // counteract it, then we must add q again.
        let m = m & (cc as u64).wrapping_sub(1);
        let (d0, cc) = addcarry_u64(d0, Self::MODULUS[0] & m, 0);
        let (d1, cc) = addcarry_u64(d1, Self::MODULUS[1] & m, cc);
        let (d2, cc) = addcarry_u64(d2, Self::MODULUS[2] & m, cc);
        let (d3, _)  = addcarry_u64(d3, Self::MODULUS[3] & m, cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    // Negate this value (in place).
    #[inline]
    pub fn set_neg(&mut self) {
        // 1. Compute q - self.
        let (d0, cc) = subborrow_u64(Self::MODULUS[0], self.0[0], 0);
        let (d1, cc) = subborrow_u64(Self::MODULUS[1], self.0[1], cc);
        let (d2, cc) = subborrow_u64(Self::MODULUS[2], self.0[2], cc);
        let (d3, cc) = subborrow_u64(Self::MODULUS[3], self.0[3], cc);

        // 2. In case of an output borrow, add q.
        let m = (cc as u64).wrapping_neg();
        let (d0, cc) = addcarry_u64(d0, Self::MODULUS[0] & m, 0);
        let (d1, cc) = addcarry_u64(d1, Self::MODULUS[1] & m, cc);
        let (d2, cc) = addcarry_u64(d2, Self::MODULUS[2] & m, cc);
        let (d3, _)  = addcarry_u64(d3, Self::MODULUS[3] & m, cc);

        // No further action needed since 2*q - x cannot be negative for
        // x < 2^256.
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
        // Optimization note: lowest limb of (q+1)/2 is zero; we expect the
        // compiler to optimize it out when inlining these calls.
        let (d0, cc) = addcarry_u64(d0, tt & Self::HALFMOD[0], 0);
        let (d1, cc) = addcarry_u64(d1, tt & Self::HALFMOD[1], cc);
        let (d2, cc) = addcarry_u64(d2, tt & Self::HALFMOD[2], cc);
        let (d3, _)  = addcarry_u64(d3, tt & Self::HALFMOD[3], cc);

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

    // Reduction by wrapping top bits (d4).
    // Requirement: d4 <= 2^32
    #[inline(always)]
    fn set_wrap_small(&mut self, d0: u64, d1: u64, d2: u64, d3: u64, d4: u64) {
        // 2^256 = 2^224 - 2^192 - 2^96 + 1 mod q
        // We compute in w0..w3 the product d4*(2^224 - 2^192 - 2^96 + 1)
        // (since d4 <= 2^32, that product fits on 256 bits).
        let w0 = d4;
        let (w1, cc) = subborrow_u64(0, d4 << 32, 0);
        let (w2, cc) = subborrow_u64(0, d4 >> 32, cc);
        let (w3, _)  = subborrow_u64(d4 << 32, d4, cc);

        // Add w to d.
        let (d0, cc) = addcarry_u64(d0, w0, 0);
        let (d1, cc) = addcarry_u64(d1, w1, cc);
        let (d2, cc) = addcarry_u64(d2, w2, cc);
        let (d3, cc) = addcarry_u64(d3, w3, cc);

        // 4. On carry, subtract q. Since we added at most:
        //    (2^32)*(2^224 - 2^192 - 2^96 + 1)
        // which is lower than q, subtracting q once is enough.
        let m = (cc as u64).wrapping_neg();
        let (d0, cc) = subborrow_u64(d0, m & Self::MODULUS[0], 0);
        let (d1, cc) = subborrow_u64(d1, m & Self::MODULUS[1], cc);
        let (d2, cc) = subborrow_u64(d2, m & Self::MODULUS[2], cc);
        let (d3, _)  = subborrow_u64(d3, m & Self::MODULUS[3], cc);

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    // Multiply this value by 2.
    #[inline]
    pub fn set_mul2(&mut self) {
        // 1. Extract top bit.
        let tt = self.0[3] >> 63;

        // 2. Left-shift.
        let d0 = self.0[0] << 1;
        let d1 = (self.0[0] >> 63) | (self.0[1] << 1);
        let d2 = (self.0[1] >> 63) | (self.0[2] << 1);
        let d3 = (self.0[2] >> 63) | (self.0[3] << 1);

        // 3. Wrap top bit.
        self.set_wrap_small(d0, d1, d2, d3, tt);
    }

    #[inline(always)]
    pub fn mul2(self) -> Self {
        let mut r = self;
        r.set_mul2();
        r
    }

    // Multiply this value by 3.
    #[inline]
    pub fn set_mul3(&mut self) {
        // 1. Extract top bit.
        let tt = self.0[3] >> 63;

        // 2. Left-shift and add.
        let d0 = self.0[0] << 1;
        let d1 = (self.0[0] >> 63) | (self.0[1] << 1);
        let d2 = (self.0[1] >> 63) | (self.0[2] << 1);
        let d3 = (self.0[2] >> 63) | (self.0[3] << 1);
        let (d0, cc) = addcarry_u64(d0, self.0[0], 0);
        let (d1, cc) = addcarry_u64(d1, self.0[1], cc);
        let (d2, cc) = addcarry_u64(d2, self.0[2], cc);
        let (d3, cc) = addcarry_u64(d3, self.0[3], cc);
        let (tt, _)  = addcarry_u64(tt, 0, cc);

        // 3. Wrap top bits.
        self.set_wrap_small(d0, d1, d2, d3, tt);
    }

    #[inline(always)]
    pub fn mul3(self) -> Self {
        let mut r = self;
        r.set_mul3();
        r
    }


    // Multiply this value by 4.
    #[inline]
    pub fn set_mul4(&mut self) {
        // 1. Extract top bits.
        let tt = self.0[3] >> 62;

        // 2. Left-shift.
        let d0 = self.0[0] << 2;
        let d1 = (self.0[0] >> 62) | (self.0[1] << 2);
        let d2 = (self.0[1] >> 62) | (self.0[2] << 2);
        let d3 = (self.0[2] >> 62) | (self.0[3] << 2);

        // 3. Wrap top bits.
        self.set_wrap_small(d0, d1, d2, d3, tt);
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
        let tt = self.0[3] >> 61;

        // 2. Left-shift.
        let d0 = self.0[0] << 3;
        let d1 = (self.0[0] >> 61) | (self.0[1] << 3);
        let d2 = (self.0[1] >> 61) | (self.0[2] << 3);
        let d3 = (self.0[2] >> 61) | (self.0[3] << 3);

        // 3. Wrap top bits.
        self.set_wrap_small(d0, d1, d2, d3, tt);
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
        let tt = self.0[3] >> 60;

        // 2. Left-shift.
        let d0 = self.0[0] << 4;
        let d1 = (self.0[0] >> 60) | (self.0[1] << 4);
        let d2 = (self.0[1] >> 60) | (self.0[2] << 4);
        let d3 = (self.0[2] >> 60) | (self.0[3] << 4);

        // 3. Wrap top bits.
        self.set_wrap_small(d0, d1, d2, d3, tt);
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
        let tt = self.0[3] >> 59;

        // 2. Left-shift.
        let d0 = self.0[0] << 5;
        let d1 = (self.0[0] >> 59) | (self.0[1] << 5);
        let d2 = (self.0[1] >> 59) | (self.0[2] << 5);
        let d3 = (self.0[2] >> 59) | (self.0[3] << 5);

        // 3. Wrap top bits.
        self.set_wrap_small(d0, d1, d2, d3, tt);
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

        // Fold top word.
        self.set_wrap_small(d0, d1, d2, d3, d4);
    }

    #[inline(always)]
    pub fn mul_small(self, x: u32) -> Self {
        let mut r = self;
        r.set_mul_small(x);
        r
    }

    // Reduction of a 320-bit value.
    #[inline(always)]
    fn set_wrap_medium(&mut self, d0: u64, d1: u64, d2: u64, d3: u64, d4: u64) {
        // w <- d4*(2^224 - 2^192 - 2^96 + 1)
        let w0 = d4;
        let (w1, cc) = subborrow_u64(0, d4 << 32, 0);
        let (w2, cc) = subborrow_u64(0, d4 >> 32, cc);
        let (w3, cc) = subborrow_u64(d4 << 32, d4, cc);
        let (w4, _)  = subborrow_u64(d4 >> 32, 0, cc);

        // d4 < 2^64, hence w < 2^288 and w4 <= 2^32 - 1.

        // Add w to d0..d3, with extra bits into w4.
        let (d0, cc) = addcarry_u64(d0, w0, 0);
        let (d1, cc) = addcarry_u64(d1, w1, cc);
        let (d2, cc) = addcarry_u64(d2, w2, cc);
        let (d3, cc) = addcarry_u64(d3, w3, cc);
        let (w4, _)  = addcarry_u64(0, w4, cc);

        // Now w4 <= 2^32; we can use set_wrap_small() to finish the reduction.
        self.set_wrap_small(d0, d1, d2, d3, w4);
    }

    // Reduction of a 512-bit value.
    #[inline(always)]
    fn set_wrap_large(&mut self, e0: u64, e1: u64, e2: u64, e3: u64,
        e4: u64, e5: u64, e6: u64, e7: u64)
    {
        // The value can be partially reduced as follows:
        //   - take the low half e0..e3
        //   - add the following:
        //      q*2^34
        //      e4 + (e4 << 224)
        //      (e5 << 32) + (e5 << 64)
        //      2*(e6 << 96) + 2*(e6 << 128)
        //      (e7 << 128) + 2*(e7 << 160) + 3*(e7 << 192)
        //   - subtract the following:
        //      (e4 << 96) + (e4 << 192)
        //      (e5 << 128) + (e5 << 160) + (e5 << 224)
        //      e6 + (e6 << 32) + (e6 << 224)
        //      e7 + (e7 << 32) + (e7 << 64)
        // The addition of q*2^34 ensures that the result is non-negative;
        // it will also be lower than 2^(256+35), so the remaining word to
        // fold fits on 35 bits.
        //
        // We try to minimize the number of carry propagation passes
        // to perform.

        // add e4 + (e5 << 64) + (e7 << 128) + (e7 << 192)
        let (e0, cc) = addcarry_u64(e0, e4, 0);
        let (e1, cc) = addcarry_u64(e1, e5, cc);
        let (e2, cc) = addcarry_u64(e2, e7, cc);
        let (e3, cc) = addcarry_u64(e3, e7, cc);
        let eh = cc as u64;

        // add 2*(e6 << 128) + 2*(e7 << 192)
        let (e2, cc) = addcarry_u64(e2, e6 << 1, 0);
        let (e3, cc) = addcarry_u64(e3, (e6 >> 63) | (e7 << 1), cc);
        let (eh, _)  = addcarry_u64(eh, e7 >> 63, cc);

        // add (e5 << 32) + 2*(e6 << 96) + 2*(e7 << 160)
        let (e0, cc) = addcarry_u64(e0, e5 << 32, 0);
        let (e1, cc) = addcarry_u64(e1, (e5 >> 32) | (e6 << 33), cc);
        let (e2, cc) = addcarry_u64(e2, (e6 >> 31) | (e7 << 33), cc);
        let (e3, cc) = addcarry_u64(e3, e7 >> 31, cc);
        let (eh, _)  = addcarry_u64(eh, 0, cc);

        // add (e4 << 224)
        let (e3, cc) = addcarry_u64(e3, e4 << 32, 0);
        let (eh, _)  = addcarry_u64(eh, e4 >> 32, cc);

        // add q*2^34
        let (e0, cc) = addcarry_u64(e0, Self::MODT34[0], 0);
        let (e1, cc) = addcarry_u64(e1, Self::MODT34[1], cc);
        let (e2, cc) = addcarry_u64(e2, Self::MODT34[2], cc);
        let (e3, cc) = addcarry_u64(e3, Self::MODT34[3], cc);
        let (eh, _)  = addcarry_u64(eh, Self::MODT34[4], cc);

        // subtract e7 + (e7 << 64) + (e5 << 128) + (e4 << 192)
        let (e0, cc) = subborrow_u64(e0, e7, 0);
        let (e1, cc) = subborrow_u64(e1, e7, cc);
        let (e2, cc) = subborrow_u64(e2, e5, cc);
        let (e3, cc) = subborrow_u64(e3, e4, cc);
        let (eh, _)  = subborrow_u64(eh, 0, cc);

        // subtract (e6 << 32) + (e4 << 96) + (e5 << 160) + (e5 << 224)
        let (e0, cc) = subborrow_u64(e0, e6 << 32, 0);
        let (e1, cc) = subborrow_u64(e1, (e6 >> 32) | (e4 << 32), cc);
        let (e2, cc) = subborrow_u64(e2, (e4 >> 32) | (e5 << 32), cc);
        let (e3, cc) = subborrow_u64(e3, (e5 >> 32) | (e5 << 32), cc);
        let (eh, _)  = subborrow_u64(eh, e5 >> 32, cc);

        // subtract e6 + (e7 << 32) + (e6 << 224)
        let (t6, cc) = addcarry_u64(e6, e7 << 32, 0);
        let (t7, _)  = addcarry_u64(0, e7 >> 32, cc);
        let (e0, cc) = subborrow_u64(e0, t6, 0);
        let (e1, cc) = subborrow_u64(e1, t7, cc);
        let (e2, cc) = subborrow_u64(e2, 0, cc);
        let (e3, cc) = subborrow_u64(e3, e6 << 32, cc);
        let (eh, _)  = subborrow_u64(eh, e6 >> 32, cc);

        // We have a non-negative result over 5 words (e0:e1:e2:e3:eh);
        // top word might be up to 35 bits.
        self.set_wrap_medium(e0, e1, e2, e3, eh);
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

        // 2. Reduction.
        self.set_wrap_large(e0, e1, e2, e3, e4, e5, e6, e7);
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
        self.set_wrap_large(e0, e1, e2, e3, e4, e5, e6, e7);
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
        // Subtract q.
        let (d0, cc) = subborrow_u64(self.0[0], Self::MODULUS[0], 0);
        let (d1, cc) = subborrow_u64(self.0[1], Self::MODULUS[1], cc);
        let (d2, cc) = subborrow_u64(self.0[2], Self::MODULUS[2], cc);
        let (d3, cc) = subborrow_u64(self.0[3], Self::MODULUS[3], cc);

        // Add back q if there was a borrow.
        let m = (cc as u64).wrapping_neg();
        let (d0, cc) = addcarry_u64(d0, m & Self::MODULUS[0], 0);
        let (d1, cc) = addcarry_u64(d1, m & Self::MODULUS[1], cc);
        let (d2, cc) = addcarry_u64(d2, m & Self::MODULUS[2], cc);
        let (d3, _)  = addcarry_u64(d3, m & Self::MODULUS[3], cc);

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
        self.set_wrap_medium(d0, d1, d2, d3, t);
    }

    #[inline(always)]
    fn lin(a: &Self, b: &Self, f: u64, g: u64) -> Self {
        let mut r = Self::ZERO;
        r.set_lin(a, b, f, g);
        r
    }

    // Set this value to abs((a*f+b*g)/2^31). Values a and b are interpreted
    // as unsigned 256-bit integers. Coefficients f and g are provided as u64,
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
        let (a3, cc) = subborrow_u64(a.0[3] ^ sf, sf, cc);
        let (a4, _)  = subborrow_u64(0, 0, cc);
        let (b0, cc) = subborrow_u64(b.0[0] ^ sg, sg, 0);
        let (b1, cc) = subborrow_u64(b.0[1] ^ sg, sg, cc);
        let (b2, cc) = subborrow_u64(b.0[2] ^ sg, sg, cc);
        let (b3, cc) = subborrow_u64(b.0[3] ^ sg, sg, cc);
        let (b4, _)  = subborrow_u64(0, 0, cc);

        // Compute a*f+b*g into d0:d1:d2:d3:t. Since f and g are at
        // most 2^31, we can add two 128-bit products with no overflow.
        let (d0, t) = umull_x2(a0, f, b0, g);
        let (d1, t) = umull_x2_add(a1, f, b1, g, t);
        let (d2, t) = umull_x2_add(a2, f, b2, g, t);
        let (d3, t) = umull_x2_add(a3, f, b3, g, t);
        // d4 <- a4*f + b4*g + t; a4 and b4 can be only 0 or -1
        let d4 = t.wrapping_sub(a4 & f).wrapping_sub(b4 & g);

        // Shift-right the value by 31 bits.
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
        let mut b = Self(Self::MODULUS);
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
        // len(a) + len(b) <= 47, so we can end the computation with
        // the low words directly. We only need 45 iterations to reach
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

        self.set_lin(&u, &v, f1, g1);

        // At this point, we have injected extra factors of 2, one for
        // each of the 31*15+45 = 510 iterations, so we must divide by
        // 2^510 (mod q). This is done with a multiplication by the
        // appropriate constant.
        self.set_mul(&Self::INVT510);
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
        a.set_normalized();
        let mut b = Self(Self::MODULUS);
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

    // Set this value to its square root. Returned value is 0xFFFFFFFF
    // if the operation succeeded (value was indeed a quadratic
    // residue), 0 otherwise (value was not a quadratic residue). In the
    // latter case, this value is set to the square root of -self. In
    // all cases, the returned root is the one whose least significant
    // bit is 0 (when normalized in 0..q-1).
    fn set_sqrt_ext(&mut self) -> u32 {
        // Candidate root is self^((q+1)/4).
        // (q+1)/4 = 2^254 - 2^222 + 2^190 + 2^94
        let z = *self;
        let z2 = z.square() * z;
        let z4 = z2.xsquare(2) * z2;
        let z8 = z4.xsquare(4) * z4;
        let z16 = z8.xsquare(8) * z8;
        let z32 = z16.xsquare(16) * z16;
        let mut y = ((z32.xsquare(32) * z).xsquare(96) * z).xsquare(94);

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
    //    a square root of -x, and r is 0x00000000.
    // In all cases, the returned root is normalized: the lest significant
    // bit of its integer representation (in the 0..q-1 range) is 0.
    #[inline(always)]
    pub fn sqrt_ext(self) -> (Self, u32) {
        let mut x = self;
        let r = x.set_sqrt_ext();
        (x, r)
    }

    /* TODO? Not very useful on this field.
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
    */

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
        // There are two possible representations for 0: 0 and q.
        let a0 = self.0[0];
        let a1 = self.0[1];
        let a2 = self.0[2];
        let a3 = self.0[3];
        let t0 = a0 | a1 | a2 | a3;
        let t1 = (a0 ^ Self::MODULUS[0]) | (a1 ^ Self::MODULUS[1])
            | (a2 ^ Self::MODULUS[2]) | (a3 ^ Self::MODULUS[3]);

        // Top bit of r is 0 if and only if one of t0 or t1 is zero.
        let r = (t0 | t0.wrapping_neg()) & (t1 | t1.wrapping_neg());
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
    // (little-endian encoding of the value in the 0..q-1 range).
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
    // (little-endian encoding of the value in the 0..q-1 range).
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
        let (_, cc) = subborrow_u64(self.0[0], Self::MODULUS[0], 0);
        let (_, cc) = subborrow_u64(self.0[1], Self::MODULUS[1], cc);
        let (_, cc) = subborrow_u64(self.0[2], Self::MODULUS[2], cc);
        let (_, cc) = subborrow_u64(self.0[3], Self::MODULUS[3], cc);

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
            self.set_wrap_large(e0, e1, e2, e3,
                self.0[0], self.0[1], self.0[2], self.0[3]);
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
}

// ========================================================================
// Implementations of all the traits needed to use the simple operators
// (+, *, /...) on field element instances, with or without references.

impl Add<GFp256> for GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn add(self, other: GFp256) -> GFp256 {
        let mut r = self;
        r.set_add(&other);
        r
    }
}

impl Add<&GFp256> for GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn add(self, other: &GFp256) -> GFp256 {
        let mut r = self;
        r.set_add(other);
        r
    }
}

impl Add<GFp256> for &GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn add(self, other: GFp256) -> GFp256 {
        let mut r = *self;
        r.set_add(&other);
        r
    }
}

impl Add<&GFp256> for &GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn add(self, other: &GFp256) -> GFp256 {
        let mut r = *self;
        r.set_add(other);
        r
    }
}

impl AddAssign<GFp256> for GFp256 {
    #[inline(always)]
    fn add_assign(&mut self, other: GFp256) {
        self.set_add(&other);
    }
}

impl AddAssign<&GFp256> for GFp256 {
    #[inline(always)]
    fn add_assign(&mut self, other: &GFp256) {
        self.set_add(other);
    }
}

impl Div<GFp256> for GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn div(self, other: GFp256) -> GFp256 {
        let mut r = self;
        r.set_div(&other);
        r
    }
}

impl Div<&GFp256> for GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn div(self, other: &GFp256) -> GFp256 {
        let mut r = self;
        r.set_div(other);
        r
    }
}

impl Div<GFp256> for &GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn div(self, other: GFp256) -> GFp256 {
        let mut r = *self;
        r.set_div(&other);
        r
    }
}

impl Div<&GFp256> for &GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn div(self, other: &GFp256) -> GFp256 {
        let mut r = *self;
        r.set_div(other);
        r
    }
}

impl DivAssign<GFp256> for GFp256 {
    #[inline(always)]
    fn div_assign(&mut self, other: GFp256) {
        self.set_div(&other);
    }
}

impl DivAssign<&GFp256> for GFp256 {
    #[inline(always)]
    fn div_assign(&mut self, other: &GFp256) {
        self.set_div(other);
    }
}

impl Mul<GFp256> for GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn mul(self, other: GFp256) -> GFp256 {
        let mut r = self;
        r.set_mul(&other);
        r
    }
}

impl Mul<&GFp256> for GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn mul(self, other: &GFp256) -> GFp256 {
        let mut r = self;
        r.set_mul(other);
        r
    }
}

impl Mul<GFp256> for &GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn mul(self, other: GFp256) -> GFp256 {
        let mut r = *self;
        r.set_mul(&other);
        r
    }
}

impl Mul<&GFp256> for &GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn mul(self, other: &GFp256) -> GFp256 {
        let mut r = *self;
        r.set_mul(other);
        r
    }
}

impl MulAssign<GFp256> for GFp256 {
    #[inline(always)]
    fn mul_assign(&mut self, other: GFp256) {
        self.set_mul(&other);
    }
}

impl MulAssign<&GFp256> for GFp256 {
    #[inline(always)]
    fn mul_assign(&mut self, other: &GFp256) {
        self.set_mul(other);
    }
}

impl Neg for GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn neg(self) -> GFp256 {
        let mut r = self;
        r.set_neg();
        r
    }
}

impl Neg for &GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn neg(self) -> GFp256 {
        let mut r = *self;
        r.set_neg();
        r
    }
}

impl Sub<GFp256> for GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn sub(self, other: GFp256) -> GFp256 {
        let mut r = self;
        r.set_sub(&other);
        r
    }
}

impl Sub<&GFp256> for GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn sub(self, other: &GFp256) -> GFp256 {
        let mut r = self;
        r.set_sub(other);
        r
    }
}

impl Sub<GFp256> for &GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn sub(self, other: GFp256) -> GFp256 {
        let mut r = *self;
        r.set_sub(&other);
        r
    }
}

impl Sub<&GFp256> for &GFp256 {
    type Output = GFp256;

    #[inline(always)]
    fn sub(self, other: &GFp256) -> GFp256 {
        let mut r = *self;
        r.set_sub(other);
        r
    }
}

impl SubAssign<GFp256> for GFp256 {
    #[inline(always)]
    fn sub_assign(&mut self, other: GFp256) {
        self.set_sub(&other);
    }
}

impl SubAssign<&GFp256> for GFp256 {
    #[inline(always)]
    fn sub_assign(&mut self, other: &GFp256) {
        self.set_sub(other);
    }
}

// ========================================================================

#[cfg(test)]
mod tests {

    use super::{GFp256};
    use num_bigint::{BigInt, Sign};
    use crate::sha2::Sha256;

    /*
    fn print(name: &str, v: GFp256) {
        println!("{} = 0x{:016X}{:016X}{:016X}{:016X}",
            name, v.0[3], v.0[2], v.0[1], v.0[0]);
    }
    */

    // va, vb and vx must be 32 bytes each in length
    fn check_gf_ops(va: &[u8], vb: &[u8], vx: &[u8]) {
        let zp = BigInt::from_slice(Sign::Plus, &[
            0xFFFFFFFFu32, 0xFFFFFFFFu32, 0xFFFFFFFFu32, 0x00000000u32,
            0x00000000u32, 0x00000000u32, 0x00000001u32, 0xFFFFFFFFu32,
        ]);
        let zp4 = &zp << 2;

        let mut a = GFp256::ZERO;
        a.set_decode32_reduce(va);
        let mut b = GFp256::ZERO;
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

        let c = a.mul3();
        let vc = c.encode32();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za + (&za << 1)) % &zp;
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

        let (e, cc) = GFp256::decode32(va);
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
            let c = GFp256::decode_reduce(&tmp[0..k]);
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

    #[test]
    fn gfp256_ops() {
        let mut va = [0u8; 32];
        let mut vb = [0u8; 32];
        let mut vx = [0u8; 32];
        check_gf_ops(&va, &vb, &vx);
        assert!(GFp256::decode_reduce(&va).iszero() == 0xFFFFFFFF);
        assert!(GFp256::decode_reduce(&va).equals(GFp256::decode_reduce(&vb)) == 0xFFFFFFFF);
        assert!(GFp256::decode_reduce(&va).legendre() == 0);
        for i in 0..32 {
            va[i] = 0xFFu8;
            vb[i] = 0xFFu8;
            vx[i] = 0xFFu8;
        }
        check_gf_ops(&va, &vb, &vx);
        assert!(GFp256::decode_reduce(&va).iszero() == 0);
        assert!(GFp256::decode_reduce(&va).equals(GFp256::decode_reduce(&vb)) == 0xFFFFFFFF);
        for i in 0..4 {
            va[8*i..8*i + 8].copy_from_slice(&GFp256::MODULUS[i].to_le_bytes());
        }
        assert!(GFp256::decode_reduce(&va).iszero() == 0xFFFFFFFF);
        let mut sh = Sha256::new();
        for i in 0..300 {
            sh.update(((3 * i + 0) as u64).to_le_bytes());
            let va = sh.finalize_reset();
            sh.update(((3 * i + 1) as u64).to_le_bytes());
            let vb = sh.finalize_reset();
            sh.update(((3 * i + 2) as u64).to_le_bytes());
            let vx = sh.finalize_reset();
            check_gf_ops(&va, &vb, &vx);
            assert!(GFp256::decode_reduce(&va).iszero() == 0);
            assert!(GFp256::decode_reduce(&va).equals(GFp256::decode_reduce(&vb)) == 0);
            let s = GFp256::decode_reduce(&va).square();
            let s2 = -s;
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
            assert!(t2.square().equals(-s2) == 0xFFFFFFFF);
        }
    }

    #[test]
    fn gfp256_batch_invert() {
        let mut xx = [GFp256::ZERO; 300];
        let mut sh = Sha256::new();
        for i in 0..300 {
            sh.update((i as u64).to_le_bytes());
            let v = sh.finalize_reset();
            xx[i] = GFp256::decode_reduce(&v);
        }
        xx[120] = GFp256::ZERO;
        let mut yy = xx;
        GFp256::batch_invert(&mut yy[..]);
        for i in 0..300 {
            if xx[i].iszero() != 0 {
                assert!(yy[i].iszero() == 0xFFFFFFFF);
            } else {
                assert!((xx[i] * yy[i]).equals(GFp256::ONE) == 0xFFFFFFFF);
            }
        }
    }
}
