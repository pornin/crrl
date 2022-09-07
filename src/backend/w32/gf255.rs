use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::convert::TryFrom;

use super::{addcarry_u32, subborrow_u32, umull, umull_add, umull_add2, umull_x2, umull_x2_add, sgnw, lzcnt};
use super::lagrange::lagrange253_vartime;

#[derive(Clone, Copy, Debug)]
pub struct GF255<const MQ: u64>([u32; 8]);

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
    // Tighest restriction on MQ is from set_sqrt(), which assumes that
    // only the lowest 15 bits of q may be non-zero. Other arithmetic
    // functions have looser requirements (set_mul() and set_square() need
    // MQ <= 2^31 - 1).
    #[allow(dead_code)]
    const COMPILE_TIME_CHECKS: () = Self::compile_time_checks();
    const fn compile_time_checks() {
        static_assert!((MQ & 1) != 0);
        static_assert!(MQ <= 32767);
    }

    // Modulus is q = 2^255 - T255_MINUS_Q.
    // (this is the type parameter MQ, as a 32-bit integer)
    pub const T255_MINUS_Q: u32 = MQ as u32;

    // Modulus q in base 2^32 (low-to-high order).
    pub const MODULUS: [u32; 8] = [
        (MQ as u32).wrapping_neg(),
        0xFFFFFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0x7FFFFFFF
    ];

    pub const ZERO: GF255<MQ> = GF255::<MQ>([ 0, 0, 0, 0, 0, 0, 0, 0 ]);
    pub const ONE: GF255<MQ> = GF255::<MQ>([ 1, 0, 0, 0, 0, 0, 0, 0 ]);
    pub const MINUS_ONE: GF255<MQ> = GF255::<MQ>([
        ((MQ + 1) as u32).wrapping_neg(),
        0xFFFFFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0x7FFFFFFF
    ]);

    // 1/2^508 in the field, as a constant; this is used when computing
    // divisions in the field. The value is computed at compile-time.
    const INVT508: GF255<MQ> = GF255::<MQ>::make_invt508();

    // Create an element from a 256-bit value (implicitly reduced modulo
    // the field order) provided as four 64-bit limbs (in low-to-high order).
    pub const fn w64le(x0: u64, x1: u64, x2: u64, x3: u64) -> Self {
        Self([
            x0 as u32, (x0 >> 32) as u32,
            x1 as u32, (x1 >> 32) as u32,
            x2 as u32, (x2 >> 32) as u32,
            x3 as u32, (x3 >> 32) as u32,
        ])
    }

    // Create an element from a 256-bit value (implicitly reduced modulo
    // the field order) provided as four 64-bit limbs (in high-to-low order).
    pub const fn w64be(x3: u64, x2: u64, x1: u64, x0: u64) -> Self {
        Self([
            x0 as u32, (x0 >> 32) as u32,
            x1 as u32, (x1 >> 32) as u32,
            x2 as u32, (x2 >> 32) as u32,
            x3 as u32, (x3 >> 32) as u32,
        ])
    }

    // Create an element from a 256-bit value (implicitly reduced modulo
    // the field order) provided as four 64-bit limbs (in low-to-high order).
    pub fn from_w64le(x0: u64, x1: u64, x2: u64, x3: u64) -> Self {
        Self([
            x0 as u32, (x0 >> 32) as u32,
            x1 as u32, (x1 >> 32) as u32,
            x2 as u32, (x2 >> 32) as u32,
            x3 as u32, (x3 >> 32) as u32,
        ])
    }

    // Create an element from a 256-bit value (implicitly reduced modulo
    // the field order) provided as four 64-bit limbs (in high-to-low order).
    pub fn from_w64be(x3: u64, x2: u64, x1: u64, x0: u64) -> Self {
        Self([
            x0 as u32, (x0 >> 32) as u32,
            x1 as u32, (x1 >> 32) as u32,
            x2 as u32, (x2 >> 32) as u32,
            x3 as u32, (x3 >> 32) as u32,
        ])
    }

    // Create an element by converting the provided integer.
    // If the source value is negative, then it is implicitly reduced
    // modulo the ring order.
    #[inline(always)]
    pub fn from_i32(x: i32) -> Self {
        // We add q to ensure a nonnegative integer.
        let x0 = x as u32;
        let xh = (x >> 31) as u32;
        let (d0, cc) = addcarry_u32(x0, (MQ as u32).wrapping_neg(), 0);
        let (d1, cc) = addcarry_u32(xh, 0xFFFFFFFF, cc);
        let (d2, cc) = addcarry_u32(xh, 0xFFFFFFFF, cc);
        let (d3, cc) = addcarry_u32(xh, 0xFFFFFFFF, cc);
        let (d4, cc) = addcarry_u32(xh, 0xFFFFFFFF, cc);
        let (d5, cc) = addcarry_u32(xh, 0xFFFFFFFF, cc);
        let (d6, cc) = addcarry_u32(xh, 0xFFFFFFFF, cc);
        let (d7, _)  = addcarry_u32(xh, 0x7FFFFFFF, cc);
        Self([ d0, d1, d2, d3, d4, d5, d6, d7 ])
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
        let x0 = x as u32;
        let x1 = (x >> 32) as u32;
        let xh = (x >> 63) as u32;
        let (d0, cc) = addcarry_u32(x0, (MQ as u32).wrapping_neg(), 0);
        let (d1, cc) = addcarry_u32(x1, 0xFFFFFFFF, cc);
        let (d2, cc) = addcarry_u32(xh, 0xFFFFFFFF, cc);
        let (d3, cc) = addcarry_u32(xh, 0xFFFFFFFF, cc);
        let (d4, cc) = addcarry_u32(xh, 0xFFFFFFFF, cc);
        let (d5, cc) = addcarry_u32(xh, 0xFFFFFFFF, cc);
        let (d6, cc) = addcarry_u32(xh, 0xFFFFFFFF, cc);
        let (d7, _)  = addcarry_u32(xh, 0x7FFFFFFF, cc);
        Self([ d0, d1, d2, d3, d4, d5, d6, d7 ])
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
        let x0 = x as u32;
        let x1 = (x >> 32) as u32;
        let x2 = (x >> 64) as u32;
        let x3 = (x >> 96) as u32;
        let xh = (x >> 127) as u32;
        let (d0, cc) = addcarry_u32(x0, (MQ as u32).wrapping_neg(), 0);
        let (d1, cc) = addcarry_u32(x1, 0xFFFFFFFF, cc);
        let (d2, cc) = addcarry_u32(x2, 0xFFFFFFFF, cc);
        let (d3, cc) = addcarry_u32(x3, 0xFFFFFFFF, cc);
        let (d4, cc) = addcarry_u32(xh, 0xFFFFFFFF, cc);
        let (d5, cc) = addcarry_u32(xh, 0xFFFFFFFF, cc);
        let (d6, cc) = addcarry_u32(xh, 0xFFFFFFFF, cc);
        let (d7, _)  = addcarry_u32(xh, 0x7FFFFFFF, cc);
        Self([ d0, d1, d2, d3, d4, d5, d6, d7 ])
    }

    // Create an element by converting the provided integer.
    #[inline(always)]
    pub fn from_u128(x: u128) -> Self {
        Self::from_w64le(x as u64, (x >> 64) as u64, 0, 0)
    }

    #[inline]
    fn set_add(&mut self, rhs: &Self) {
        // 1. Addition with carry
        let (d, mut cc) = addcarry_u32(self.0[0], rhs.0[0], 0);
        self.0[0] = d;
        for i in 1..8 {
            let (d, ee) = addcarry_u32(self.0[i], rhs.0[i], cc);
            self.0[i] = d;
            cc = ee;
        }

        // 2. In case of an output carry, subtract 2*q, i.e. add 2*MQ.
        let (d, mut cc) = addcarry_u32(self.0[0],
            (cc as u32).wrapping_neg() & (2 * (MQ as u32)), 0);
        self.0[0] = d;
        for i in 1..8 {
            let (d, ee) = addcarry_u32(self.0[i], 0, cc);
            self.0[i] = d;
            cc = ee;
        }

        // 3. If there is again an extra carry, then we have to subtract 2*q
        // again. In that case, original sum was at least 2^257 - 2*MQ, and
        // the low word is now lower than 2*MQ, so adding 2*MQ to it will
        // not overflow.
        self.0[0] = self.0[0].wrapping_add(
            (cc as u32).wrapping_neg() & (2 * (MQ as u32)));
    }

    #[inline]
    fn set_sub(&mut self, rhs: &Self) {
        // 1. Subtraction with borrow
        let (d, mut cc) = subborrow_u32(self.0[0], rhs.0[0], 0);
        self.0[0] = d;
        for i in 1..8 {
            let (d, ee) = subborrow_u32(self.0[i], rhs.0[i], cc);
            self.0[i] = d;
            cc = ee;
        }

        // 2. In case of an output borrow, add 2*q, i.e. subtract 2*MQ.
        let (d, mut cc) = subborrow_u32(self.0[0],
            (cc as u32).wrapping_neg() & (2 * (MQ as u32)), 0);
        self.0[0] = d;
        for i in 1..8 {
            let (d, ee) = subborrow_u32(self.0[i], 0, cc);
            self.0[i] = d;
            cc = ee;
        }

        // 3. If there is again a borrow, then add 2*q again. In that case,
        // the low word must be at least 2^32 - 2*MQ, and the extra
        // subtraction won't trigger a new carry.
        self.0[0] = self.0[0].wrapping_sub(
            (cc as u32).wrapping_neg() & (2 * (MQ as u32)));
    }

    // Negate this value (in place).
    #[inline]
    pub fn set_neg(&mut self) {
        // 1. Compute 2*q - self over 256 bits.
        let (d, mut cc) = subborrow_u32(
            (2 * (MQ as u32)).wrapping_neg(), self.0[0], 0);
        self.0[0] = d;
        for i in 1..8 {
            let (d, ee) = subborrow_u32(0xFFFFFFFF, self.0[i], cc);
            self.0[i] = d;
            cc = ee;
        }

        // 2. If the result is negative, add back q = 2^255 - MQ.
        let w = (cc as u32).wrapping_neg();
        let (d, mut cc) = addcarry_u32(self.0[0],
            w & (MQ as u32).wrapping_neg(), 0);
        self.0[0] = d;
        for i in 1..7 {
            let (d, ee) = addcarry_u32(w, self.0[i], cc);
            self.0[i] = d;
            cc = ee;
        }
        let (d, _) = addcarry_u32(self.0[7], w >> 1, cc);
        self.0[7] = d;
    }

    // Conditionally copy the provided value ('a') into self:
    //  - If ctl == 0xFFFFFFFF, then the value of 'a' is copied into self.
    //  - If ctl == 0, then the value of self is unchanged.
    // ctl MUST be equal to 0 or 0xFFFFFFFF.
    #[inline]
    pub fn set_cond(&mut self, a: &Self, ctl: u32) {
        for i in 0..8 {
            self.0[i] ^= ctl & (self.0[i] ^ a.0[i]);
        }
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
        for i in 0..8 {
            let t = ctl & (a.0[i] ^ b.0[i]);
            a.0[i] ^= t;
            b.0[i] ^= t;
        }
    }

    #[inline]
    fn set_half(&mut self) {
        // 1. Right-shift by 1 bit; keep dropped bit in 'tt' (expanded)
        let tt = (self.0[0] & 1).wrapping_neg();
        for i in 0..7 {
            self.0[i] = (self.0[i] >> 1) | (self.0[i + 1] << 31);
        }
        self.0[7] = self.0[7] >> 1;

        // 2. If the dropped bit was 1, add back (q+1)/2. Since the value
        // was right-shifted, and (q+1)/2 < 2^255, this cannot overflow.
        let (d, mut cc) = addcarry_u32(self.0[0],
            tt & (((MQ as u32) - 1) >> 1).wrapping_neg(), 0);
        self.0[0] = d;
        for i in 1..7 {
            let (d, ee) = addcarry_u32(self.0[i], tt, cc);
            self.0[i] = d;
            cc = ee;
        }
        let (d, _) = addcarry_u32(self.0[7], tt >> 2, cc);
        self.0[7] = d;
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
        let tt = self.0[7] >> 30;

        // 2. Left-shift (also clearing the extracted bits).
        self.0[7] = ((self.0[7] << 1) & 0x7FFFFFFF) | (self.0[6] >> 31);
        for i in (1..7).rev() {
            self.0[i] = (self.0[i] << 1) | (self.0[i - 1] >> 31);
        }
        self.0[0] = self.0[0] << 1;

        // 3. Add back the top bits with reduction. Since we extracted
        // one more bit than needed, this cannot overflow.
        let (d, mut cc) = addcarry_u32(self.0[0], tt * (MQ as u32), 0);
        self.0[0] = d;
        for i in 1..8 {
            let (d, ee) = addcarry_u32(self.0[i], 0, cc);
            self.0[i] = d;
            cc = ee;
        }
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
        let tt = self.0[7] >> 29;

        // 2. Left-shift (also clearing the extracted bits).
        self.0[7] = ((self.0[7] << 2) & 0x7FFFFFFF) | (self.0[6] >> 30);
        for i in (1..7).rev() {
            self.0[i] = (self.0[i] << 2) | (self.0[i - 1] >> 30);
        }
        self.0[0] = self.0[0] << 2;

        // 3. Add back the top bits with reduction. Since we extracted
        // one more bit than needed, this cannot overflow.
        let (d, mut cc) = addcarry_u32(self.0[0], tt * (MQ as u32), 0);
        self.0[0] = d;
        for i in 1..8 {
            let (d, ee) = addcarry_u32(self.0[i], 0, cc);
            self.0[i] = d;
            cc = ee;
        }
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
        let tt = self.0[7] >> 28;

        // 2. Left-shift (also clearing the extracted bits).
        self.0[7] = ((self.0[7] << 3) & 0x7FFFFFFF) | (self.0[6] >> 29);
        for i in (1..7).rev() {
            self.0[i] = (self.0[i] << 3) | (self.0[i - 1] >> 29);
        }
        self.0[0] = self.0[0] << 3;

        // 3. Add back the top bits with reduction. Since we extracted
        // one more bit than needed, this cannot overflow.
        let (d, mut cc) = addcarry_u32(self.0[0], tt * (MQ as u32), 0);
        self.0[0] = d;
        for i in 1..8 {
            let (d, ee) = addcarry_u32(self.0[i], 0, cc);
            self.0[i] = d;
            cc = ee;
        }
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
        let tt = self.0[7] >> 27;

        // 2. Left-shift (also clearing the extracted bits).
        self.0[7] = ((self.0[7] << 4) & 0x7FFFFFFF) | (self.0[6] >> 28);
        for i in (1..7).rev() {
            self.0[i] = (self.0[i] << 4) | (self.0[i - 1] >> 28);
        }
        self.0[0] = self.0[0] << 4;

        // 3. Add back the top bits with reduction. Since we extracted
        // one more bit than needed, this cannot overflow.
        let (d, mut cc) = addcarry_u32(self.0[0], tt * (MQ as u32), 0);
        self.0[0] = d;
        for i in 1..8 {
            let (d, ee) = addcarry_u32(self.0[i], 0, cc);
            self.0[i] = d;
            cc = ee;
        }
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
        let tt = self.0[7] >> 26;

        // 2. Left-shift (also clearing the extracted bits).
        self.0[7] = ((self.0[7] << 5) & 0x7FFFFFFF) | (self.0[6] >> 27);
        for i in (1..7).rev() {
            self.0[i] = (self.0[i] << 5) | (self.0[i - 1] >> 27);
        }
        self.0[0] = self.0[0] << 5;

        // 3. Add back the top bits with reduction. Since we extracted
        // one more bit than needed, this cannot overflow.
        let (d, mut cc) = addcarry_u32(self.0[0], tt * (MQ as u32), 0);
        self.0[0] = d;
        for i in 1..8 {
            let (d, ee) = addcarry_u32(self.0[i], 0, cc);
            self.0[i] = d;
            cc = ee;
        }
    }

    #[inline(always)]
    pub fn mul32(self) -> Self {
        let mut r = self;
        r.set_mul32();
        r
    }

    // Multiplies this value by a small integer (in place).
    #[inline]
    pub fn set_mul_small(&mut self, x: u32) {
        // Compute the product as an integer over nine words.
        // Max value is (2^32 - 1)*(2^256 - 1), so the top word (cc) is
        // at most 2^32 - 2.
        let (lo, mut cc) = umull(self.0[0], x);
        self.0[0] = lo;
        for i in 1..8 {
            let (lo, hi) = umull_add(self.0[i], x, cc);
            self.0[i] = lo;
            cc = hi;
        }

        // Do the reduction by folding the top word (cc) _and_ the top
        // bit of the previous word (self.0[7]). Since that clears the top
        // bit, only one pass is needed (folding won't overflow).
        // We want to compute:
        //   (2*cc + (self.0[7] >> 31)) * MQ
        // Since 2*cc might not fit in 32 bits, we expand this into:
        //   cc * (2*MQ) + (self.0[7] >> 31) * MQ
        // The second multiplication can be done with a bitwise AND.
        let (c0, c1) = umull_add(cc, 2 * (MQ as u32),
            sgnw(self.0[7]) & (MQ as u32));
        let (d, cc) = addcarry_u32(self.0[0], c0, 0);
        self.0[0] = d;
        let (d, mut cc) = addcarry_u32(self.0[1], c1, cc);
        self.0[1] = d;
        for i in 2..7 {
            let (d, ee) = addcarry_u32(self.0[i], 0, cc);
            self.0[i] = d;
            cc = ee;
        }
        let (d, _) = addcarry_u32(self.0[7] & 0x7FFFFFFF, 0, cc);
        self.0[7] = d;
    }

    #[inline(always)]
    pub fn mul_small(self, x: u32) -> Self {
        let mut r = self;
        r.set_mul_small(x);
        r
    }

    #[inline(always)]
    fn set_mul(&mut self, rhs: &Self) {
        // 1. Product -> 512 bits.
        let mut c = [0u32; 16];
        let (lo, mut hi) = umull(self.0[0], rhs.0[0]);
        c[0] = lo;
        for i in 1..8 {
            let (lo, ee) = umull_add(self.0[0], rhs.0[i], hi);
            c[i] = lo;
            hi = ee;
        }
        c[8] = hi;
        for j in 1..8 {
            let (lo, mut hi) = umull_add(self.0[j], rhs.0[0], c[j]);
            c[j] = lo;
            for i in 1..8 {
                let (lo, ee) = umull_add2(self.0[j], rhs.0[i], c[i + j], hi);
                c[i + j] = lo;
                hi = ee;
            }
            c[j + 8] = hi;
        }

        // 2. Reduction
        // We fold the upper words in two steps; first step adds the
        // low words of the multiplication by 2*MQ, while high words
        // of these products are kept in c[8]..c[15]
        for i in 0..8 {
            let (lo, hi) = umull_add(c[i + 8], 2 * (MQ as u32), c[i]);
            c[i] = lo;
            c[i + 8] = hi;
        }

        // Max value for c[15] is 1 + floor(2*MQ*(2^32 - 1) / 2^32).
        // We then compute (2*c[15] + b)*MQ, with b being the top bit of c[7]
        // (i.e. b = 0 or 1). This value fits on 32 bits as long as
        // MQ <= 2^15 - 1 (hence the restriction on the MQ parameter).
        let g = (c[15] << 1) | (c[7] >> 31);
        c[7] &= 0x7FFFFFFF;
        let (d, mut cc) = addcarry_u32(c[0], g * (MQ as u32), 0);
        self.0[0] = d;
        for i in 1..8 {
            let (d, ee) = addcarry_u32(c[i], c[i + 7], cc);
            self.0[i] = d;
            cc = ee;
        }
    }

    // Square this value (in place).
    #[inline(always)]
    pub fn set_square(&mut self) {
        // 1. Square over integers -> 512 bits.
        // We first compute the non-square products.
        let mut c = [0u32; 16];
        let (lo, mut hi) = umull(self.0[0], self.0[1]);
        c[1] = lo;
        for i in 2..8 {
            let (lo, ee) = umull_add(self.0[0], self.0[i], hi);
            c[i] = lo;
            hi = ee;
        }
        c[8] = hi;
        for j in 1..7 {
            let (lo, mut hi) = umull_add(
                self.0[j], self.0[j + 1], c[2 * j + 1]);
            c[2 * j + 1] = lo;
            for i in (j + 2)..8 {
                let (lo, ee) = umull_add2(self.0[j], self.0[i], c[i + j], hi);
                c[i + j] = lo;
                hi = ee;
            }
            c[j + 8] = hi;
        }

        // 2. Double all non-square products.
        c[15] = c[14] >> 31;
        for i in (2..15).rev() {
            c[i] = (c[i] << 1) | (c[i - 1] >> 31);
        }
        c[1] = c[1] << 1;

        // 3. Add all squares.
        let (lo, hi) = umull(self.0[0], self.0[0]);
        c[0] = lo;
        let (d, mut cc) = addcarry_u32(c[1], hi, 0);
        c[1] = d;
        for i in 1..8 {
            let (lo, hi) = umull(self.0[i], self.0[i]);
            let (d, ee) = addcarry_u32(c[2 * i], lo, cc);
            c[2 * i] = d;
            let (d, ee) = addcarry_u32(c[2 * i + 1], hi, ee);
            c[2 * i + 1] = d;
            cc = ee;
        }

        // 4. Reduction
        // This is identical to the reduction in set_mul().
        for i in 0..8 {
            let (lo, hi) = umull_add(c[i + 8], 2 * (MQ as u32), c[i]);
            c[i] = lo;
            c[i + 8] = hi;
        }
        let g = (c[15] << 1) | (c[7] >> 31);
        c[7] &= 0x7FFFFFFF;
        let (d, mut cc) = addcarry_u32(c[0], g * (MQ as u32), 0);
        self.0[0] = d;
        for i in 1..8 {
            let (d, ee) = addcarry_u32(c[i], c[i + 7], cc);
            self.0[i] = d;
            cc = ee;
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
        let w = (self.0[7] >> 31).wrapping_neg();
        let (d, mut cc) = addcarry_u32(self.0[0], w & (MQ as u32), 0);
        self.0[0] = d;
        for i in 1..7 {
            let (d, ee) = addcarry_u32(self.0[i], 0, cc);
            self.0[i] = d;
            cc = ee;
        }
        let (d, _) = addcarry_u32(self.0[7] & 0x7FFFFFFF, 0, cc);
        self.0[7] = d;

        // Value is now at most 2^255 + MQ - 1. Subtract q, then add it
        // back in case the result would be negative.
        let (d, mut cc) = subborrow_u32(self.0[0],
            (MQ as u32).wrapping_neg(), 0);
        self.0[0] = d;
        for i in 1..7 {
            let (d, ee) = subborrow_u32(self.0[i], 0xFFFFFFFF, cc);
            self.0[i] = d;
            cc = ee;
        }
        let (d, cc) = subborrow_u32(self.0[7], 0x7FFFFFFF, cc);
        self.0[7] = d;

        let w = (cc as u32).wrapping_neg();
        let (d, mut cc) = addcarry_u32(self.0[0],
            w & (MQ as u32).wrapping_neg(), 0);
        self.0[0] = d;
        for i in 1..7 {
            let (d, ee) = addcarry_u32(self.0[i], w, cc);
            self.0[i] = d;
            cc = ee;
        }
        let (d, _) = addcarry_u32(self.0[7], w >> 1, cc);
        self.0[7] = d;
    }

    // Set this value to u*f+v*g (with 'u' being self). Parameters f and g
    // are provided as u32, but they are signed integers in the -2^30..+2^30
    // range.
    #[inline]
    fn set_lin(&mut self, u: &Self, v: &Self, f: u32, g: u32) {
        // Make sure f is nonnegative, by negating it if necessary, and
        // also negating u in that case to keep u*f unchanged.
        let sf = sgnw(f);
        let f = (f ^ sf).wrapping_sub(sf);
        let tu = Self::select(u, &-u, sf);

        // Same treatment for g and v.
        let sg = sgnw(g);
        let g = (g ^ sg).wrapping_sub(sg);
        let tv = Self::select(v, &-v, sg);

        // Compute the linear combination on plain integers. Since f and
        // g are at most 2^30 each, intermediate 64-bit products cannot
        // overflow.
        let (lo, mut cc) = umull_x2(tu.0[0], f, tv.0[0], g);
        self.0[0] = lo;
        for i in 1..8 {
            let (lo, hi) = umull_x2_add(tu.0[i], f, tv.0[i], g, cc);
            self.0[i] = lo;
            cc = hi;
        }

        // Upper word cc can be up to 31 bits.
        let (lo, hi) = umull(cc, 2 * (MQ as u32));
        let (d, cc) = addcarry_u32(self.0[0], lo, 0);
        self.0[0] = d;
        let (d, mut cc) = addcarry_u32(self.0[1], hi, cc);
        self.0[1] = d;
        for i in 2..8 {
            let (d, ee) = addcarry_u32(self.0[i], 0, cc);
            self.0[i] = d;
            cc = ee;
        }

        // If there is a carry, then current value is lower than
        // 2 * MQ * 2^31, and the folding cannot propagate beyond the
        // second limb.
        let (d, cc) = addcarry_u32(self.0[0],
            (cc as u32).wrapping_neg() & (2 * (MQ as u32)), 0);
        self.0[0] = d;
        let (d, _) = addcarry_u32(self.0[1], 0, cc);
        self.0[1] = d;
    }

    #[inline(always)]
    fn lin(a: &Self, b: &Self, f: u32, g: u32) -> Self {
        let mut r = Self::ZERO;
        r.set_lin(a, b, f, g);
        r
    }

    // Set this value to abs((a*f+b*g)/2^15). Values a and b are interpreted
    // as signed 256-bit integers. Coefficients f and g are provided as u32,
    // but they really are signed integers in the -2^15..+2^15 range
    // (inclusive). The low 15 bits are dropped (i.e. the division is assumed
    // to be exact). The result is assumed to fit in 256 bits (including the
    // sign bit) (otherwise, truncation occurs).
    //
    // Returned value is -1 (u32) if (a*f+b*g) was negative, 0 otherwise.
    #[inline]
    fn set_lindiv15abs(&mut self, a: &Self, b: &Self, f: u32, g: u32) -> u32 {
        // Replace f and g with abs(f) and abs(g), but remember the
        // original signs.
        let sf = sgnw(f);
        let f = (f ^ sf).wrapping_sub(sf);
        let sg = sgnw(g);
        let g = (g ^ sg).wrapping_sub(sg);

        // Apply the signs of f and g to the source operands.
        let mut aa = [0u32; 8];
        let (d, mut cc) = subborrow_u32(a.0[0] ^ sf, sf, 0);
        aa[0] = d;
        for i in 1..8 {
            let (d, ee) = subborrow_u32(a.0[i] ^ sf, sf, cc);
            aa[i] = d;
            cc = ee;
        }
        let mut bb = [0u32; 8];
        let (d, mut cc) = subborrow_u32(b.0[0] ^ sg, sg, 0);
        bb[0] = d;
        for i in 1..8 {
            let (d, ee) = subborrow_u32(b.0[i] ^ sg, sg, cc);
            bb[i] = d;
            cc = ee;
        }

        // Compute a*f+b*g into self (high word in t). Since f and g are at
        // most 2^31, we can add two 64-bit products with no overflow.
        let (lo, mut t) = umull_x2(aa[0], f, bb[0], g);
        self.0[0] = lo;
        for i in 1..8 {
            let (lo, hi) = umull_x2_add(aa[i], f, bb[i], g, t);
            self.0[i] = lo;
            t = hi;
        }

        // If a < 0, then the result is overestimated by f*2^256;
        // similarly, if b < 0 then the result is overestimated by g*2^256.
        // We must thus subtract 2^256*(sa*f+sb*g), with sa and sb being
        // the signs of a and b, respectively (1 for negative, 0 otherwise).
        let t = t.wrapping_sub(f & sgnw(aa[7]));
        let t = t.wrapping_sub(g & sgnw(bb[7]));

        // Shift-right the value by 15 bits.
        for i in 0..7 {
            self.0[i] = (self.0[i] >> 15) | (self.0[i + 1] << 17);
        }
        self.0[7] = (self.0[7] >> 15) | (t << 17);

        // If the result is negative, then negate it.
        let t = sgnw(t);
        let (d, mut cc) = subborrow_u32(self.0[0] ^ t, t, 0);
        self.0[0] = d;
        for i in 1..8 {
            let (d, ee) = subborrow_u32(self.0[i] ^ t, t, cc);
            self.0[i] = d;
            cc = ee;
        }

        t
    }

    #[inline(always)]
    fn lindiv15abs(a: &Self, b: &Self, f: u32, g: u32) -> (Self, u32) {
        let mut r = Self::ZERO;
        let ng = r.set_lindiv15abs(a, b, f, g);
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
            (MQ as u32).wrapping_neg(),
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0x7FFFFFFF,
        ]);
        let mut u = *self;
        let mut v = Self::ZERO;

        // Generic loop does 32*15 = 480 inner iterations.
        for _ in 0..32 {
            // Get approximations of a and b over 32 bits:
            //  - If len(a) <= 32 and len(b) <= 32, then we just use
            //    their values (low limbs).
            //  - Otherwise, with n = max(len(a), len(b)), we use:
            //       (a mod 2^15) + 2^15*floor(a / 2^(n - 17))
            //       (b mod 2^15) + 2^15*floor(b / 2^(n - 17))
            let mut c_hi = 0xFFFFFFFFu32;
            let mut c_lo = 0xFFFFFFFFu32;
            let mut a_hi = 0u32;
            let mut a_lo = 0u32;
            let mut b_hi = 0u32;
            let mut b_lo = 0u32;
            for j in (0..8).rev() {
                let aw = a.0[j];
                let bw = b.0[j];
                a_hi ^= (a_hi ^ aw) & c_hi;
                a_lo ^= (a_lo ^ aw) & c_lo;
                b_hi ^= (b_hi ^ bw) & c_hi;
                b_lo ^= (b_lo ^ bw) & c_lo;
                c_lo = c_hi;
                let mw = aw | bw;
                c_hi &= ((mw | mw.wrapping_neg()) >> 31).wrapping_sub(1);
            }

            // If c_lo = 0, then we grabbed two words for a and b.
            // If c_lo != 0 but c_hi = 0, then we grabbed one word
            // (in a_hi / b_hi), which means that both values are at
            // most 32 bits.
            // It is not possible that c_hi != 0 because b != 0 (i.e.
            // we must have encountered at least one non-zero word).
            let s = lzcnt(a_hi | b_hi);
            let mut xa = (a_hi << s) | ((a_lo >> 1) >> (31 - s));
            let mut xb = (b_hi << s) | ((b_lo >> 1) >> (31 - s));
            xa = (xa & 0xFFFF8000) | (a.0[0] & 0x00007FFF);
            xb = (xb & 0xFFFF8000) | (b.0[0] & 0x00007FFF);

            // If c_lo != 0, then the computed values for xa and xb should
            // be ignored, since both a and b fit in a single word each.
            xa ^= c_lo & (xa ^ a.0[0]);
            xb ^= c_lo & (xb ^ b.0[0]);

            // Compute the 15 inner iterations on xa and xb.
            let mut fg0 = 1u32;
            let mut fg1 = 1u32 << 16;
            for _ in 0..15 {
                let a_odd = (xa & 1).wrapping_neg();
                let (_, cc) = subborrow_u32(xa, xb, 0);
                let swap = a_odd & (cc as u32).wrapping_neg();
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
            fg0 = fg0.wrapping_add(0x7FFF7FFF);
            fg1 = fg1.wrapping_add(0x7FFF7FFF);
            let f0 = (fg0 & 0xFFFF).wrapping_sub(0x7FFF);
            let g0 = (fg0 >> 16).wrapping_sub(0x7FFF);
            let f1 = (fg1 & 0xFFFF).wrapping_sub(0x7FFF);
            let g1 = (fg1 >> 16).wrapping_sub(0x7FFF);

            // Propagate updates to a, b, u and v.
            let (na, nega) = Self::lindiv15abs(&a, &b, f0, g0);
            let (nb, negb) = Self::lindiv15abs(&a, &b, f1, g1);
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
        // len(a) + len(b) <= 30, so we can end the computation with
        // the low words directly. We only need 28 iterations to reach
        // the point where b = 1.
        //
        // If y is zero, then v is unchanged (hence zero) and none of
        // the subsequent iterations will change it either, so we get
        // 0 on output, which is what we want.
        let mut xa = a.0[0];
        let mut xb = b.0[0];
        let mut f0 = 1u32;
        let mut g0 = 0u32;
        let mut f1 = 0u32;
        let mut g1 = 1u32;
        for _ in 0..28 {
            let a_odd = (xa & 1).wrapping_neg();
            let (_, cc) = subborrow_u32(xa, xb, 0);
            let swap = a_odd & (cc as u32).wrapping_neg();
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
        // each of the 15*32+28 = 508 iterations, so we must divide by
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
            (MQ as u32).wrapping_neg(),
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0x7FFFFFFF,
        ]);
        let mut ls = 0u32;  // running symbol information in the low bit

        // Outer loop
        for _ in 0..32 {
            // Get approximations of a and b over 64 bits.
            let mut c_hi = 0xFFFFFFFFu32;
            let mut c_lo = 0xFFFFFFFFu32;
            let mut a_hi = 0u32;
            let mut a_lo = 0u32;
            let mut b_hi = 0u32;
            let mut b_lo = 0u32;
            for j in (0..8).rev() {
                let aw = a.0[j];
                let bw = b.0[j];
                a_hi ^= (a_hi ^ aw) & c_hi;
                a_lo ^= (a_lo ^ aw) & c_lo;
                b_hi ^= (b_hi ^ bw) & c_hi;
                b_lo ^= (b_lo ^ bw) & c_lo;
                c_lo = c_hi;
                let mw = aw | bw;
                c_hi &= ((mw | mw.wrapping_neg()) >> 31).wrapping_sub(1);
            }

            let s = lzcnt(a_hi | b_hi);
            let mut xa = (a_hi << s) | ((a_lo >> 1) >> (31 - s));
            let mut xb = (b_hi << s) | ((b_lo >> 1) >> (31 - s));
            xa = (xa & 0xFFFF8000) | (a.0[0] & 0x00007FFF);
            xb = (xb & 0xFFFF8000) | (b.0[0] & 0x00007FFF);

            xa ^= c_lo & (xa ^ a.0[0]);
            xb ^= c_lo & (xb ^ b.0[0]);

            // First 13 inner iterations.
            let mut fg0 = 1u32;
            let mut fg1 = 1u32 << 16;
            for _ in 0..13 {
                let a_odd = (xa & 1).wrapping_neg();
                let (_, cc) = subborrow_u32(xa, xb, 0);
                let swap = a_odd & (cc as u32).wrapping_neg();
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
            let fg0z = fg0.wrapping_add(0x7FFF7FFF);
            let fg1z = fg1.wrapping_add(0x7FFF7FFF);
            let f0 = (fg0z & 0xFFFF).wrapping_sub(0x7FFF);
            let g0 = (fg0z >> 16).wrapping_sub(0x7FFF);
            let f1 = (fg1z & 0xFFFF).wrapping_sub(0x7FFF);
            let g1 = (fg1z >> 16).wrapping_sub(0x7FFF);
            let mut a0 = a.0[0].wrapping_mul(f0)
                .wrapping_add(b.0[0].wrapping_mul(g0)) >> 13;
            let mut b0 = a.0[0].wrapping_mul(f1)
                .wrapping_add(b.0[0].wrapping_mul(g1)) >> 13;
            for _ in 0..2 {
                let a_odd = (xa & 1).wrapping_neg();
                let (_, cc) = subborrow_u32(xa, xb, 0);
                let swap = a_odd & (cc as u32).wrapping_neg();
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
            fg0 = fg0.wrapping_add(0x7FFF7FFF);
            fg1 = fg1.wrapping_add(0x7FFF7FFF);
            let f0 = (fg0 & 0xFFFF).wrapping_sub(0x7FFF);
            let g0 = (fg0 >> 16).wrapping_sub(0x7FFF);
            let f1 = (fg1 & 0xFFFF).wrapping_sub(0x7FFF);
            let g1 = (fg1 >> 16).wrapping_sub(0x7FFF);

            let (na, nega) = Self::lindiv15abs(&a, &b, f0, g0);
            let (nb, _)    = Self::lindiv15abs(&a, &b, f1, g1);
            ls ^= nega & (nb.0[0] >> 1);
            a = na;
            b = nb;
        }

        // Final iterations: values are at most 30 bits now. We do not
        // need to keep track of update coefficients. Just like the GCD,
        // we need only 28 iterations, because after 28 iterations,
        // value a is 0 or 1, and b is 1, and no further modification to
        // the Legendre symbol may happen.
        let mut xa = a.0[0];
        let mut xb = b.0[0];
        for _ in 0..28 {
            let a_odd = (xa & 1).wrapping_neg();
            let (_, cc) = subborrow_u32(xa, xb, 0);
            let swap = a_odd & (cc as u32).wrapping_neg();
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
        let r = 1u32.wrapping_sub((ls & 1) << 1);
        (r & !self.iszero()) as i32
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
        let mut t0 = self.0[0];
        let mut t1 = self.0[0].wrapping_add(MQ as u32);
        let mut t2 = self.0[0].wrapping_add(2 * (MQ as u32));
        for i in 1..7 {
            t0 |= self.0[i];
            t1 |= !self.0[i];
            t2 |= !self.0[i];
        }
        t0 |= self.0[7];
        t1 |= self.0[7] ^ 0x7FFFFFFF;
        t2 |= !self.0[7];

        // Top bit of r is 0 if and only if one of t0, t1 or t2 is zero.
        let r = (t0 | t0.wrapping_neg())
              & (t1 | t1.wrapping_neg())
              & (t2 | t2.wrapping_neg());
        (r >> 31).wrapping_sub(1)
    }

    pub fn decode(buf: &[u8]) -> Option<Self> {
        let (r, cc) = Self::decode32(buf);
        if cc != 0 {
            Some(r)
        } else {
            None
        }
    }

    #[inline(always)]
    fn decode32_reduce(buf: &[u8]) -> Self {
        let mut r = Self::ZERO;
        if buf.len() == 32 {
            r.set_decode32_reduce(buf);
        }
        r
    }

    #[inline(always)]
    fn set_decode32_reduce(&mut self, buf: &[u8]) {
        debug_assert!(buf.len() == 32);
        for i in 0..8 {
            self.0[i] = u32::from_le_bytes(*<&[u8; 4]>::try_from(
                &buf[(4 * i)..(4 * i + 4)]).unwrap());
        }
    }

    // Encode this value over exactly 32 bytes. Encoding is always canonical
    // (little-endian encoding of the value in the 0..q-1 range, top bit
    // of the last byte is always 0).
    #[inline(always)]
    pub fn encode32(self) -> [u8; 32] {
        let mut r = self;
        r.set_normalized();
        let mut d = [0u8; 32];
        for i in 0..8 {
            d[(4 * i)..(4 * i + 4)].copy_from_slice(&r.0[i].to_le_bytes());
        }
        d
    }

    // Decode a field element from 32 bytes. On success, this returns
    // (r, cc), where cc has value 0xFFFFFFFF. If the source encoding is not
    // canonical (i.e. the unsigned little-endian interpretation of the
    // 32 bytes yields an integer with is not lower than q), then this
    // returns (0, 0).
    #[inline]
    pub fn decode32(buf: &[u8]) -> (Self, u32) {
        if buf.len() != 32 {
            return (Self::ZERO, 0);
        }

        let mut r = Self::decode32_reduce(buf);

        // Try to subtract q from the value; if that does not yield a
        // borrow, then the encoding was not canonical.
        let (_, mut cc) = subborrow_u32(r.0[0], (MQ as u32).wrapping_neg(), 0);
        for i in 1..7 {
            let (_, ee) = subborrow_u32(r.0[i], 0xFFFFFFFF, cc);
            cc = ee;
        }
        let (_, cc) = subborrow_u32(r.0[7], 0x7FFFFFFF, cc);

        // Clear the value if not canonical.
        let w = (cc as u32).wrapping_neg();
        for i in 0..8 {
            r.0[i] &= w;
        }

        (r, w)
    }

    // Decode a field element from some bytes. The bytes are interpreted
    // in unsigned little-endian convention, and the resulting integer
    // is reduced modulo q. This process never fails.
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
            // Multiply the current value by 2^256 (i.e. 2*MQ, modulo q)
            // and add the new chunk.
            let k = n - 32;
            let bw = u32::from_le_bytes(*<&[u8; 4]>::try_from(
                    &buf[k..(k + 4)]).unwrap());
            let (lo, mut cc) = umull_add(r.0[0], 2 * (MQ as u32), bw);
            r.0[0] = lo;
            for i in 1..8 {
                let bw = u32::from_le_bytes(*<&[u8; 4]>::try_from(
                    &buf[(k + 4 * i)..(k + 4 * i + 4)]).unwrap());
                let (lo, hi) = umull_add2(r.0[i], 2 * (MQ as u32), bw, cc);
                r.0[i] = lo;
                cc = hi;
            }

            // We have some high bits in cc. Max value:
            //   floor(((2^256 - 1) * (2*MQ) + (2^256 - 1)) / 2^256)
            //   = floor((2^256 - 1) * (2*MQ + 1) / 2^256)
            //   = 2*MQ
            // We can do the folding with an extra bit from the value,
            // because (2 * cc + 1) * MQ <= (4*MQ + 1)*MQ < 2^32.
            let h = ((cc << 1) | r.0[7] >> 31) * (MQ as u32);
            let (d, mut cc) = addcarry_u32(r.0[0], h, 0);
            r.0[0] = d;
            for i in 1..7 {
                let (d, ee) = addcarry_u32(r.0[i], 0, cc);
                r.0[i] = d;
                cc = ee;
            }
            let (d, _) = addcarry_u32(r.0[7] & 0x7FFFFFFF, 0, cc);
            r.0[7] = d;

            n = k;
        }

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

        const fn mm(x: u64, y: u64) -> (u64, u64) {
            let z = (x as u128) * (y as u128);
            (z as u64, (z >> 64) as u64)
        }

        const fn sqr<const MQ: u64>(a: GF255<MQ>) -> GF255<MQ> {
            // This follows the same steps as the runtime set_square()
            // in the 64-bit backend.
            let a0 = (a.0[0] as u64) | ((a.0[1] as u64) << 32);
            let a1 = (a.0[2] as u64) | ((a.0[3] as u64) << 32);
            let a2 = (a.0[4] as u64) | ((a.0[5] as u64) << 32);
            let a3 = (a.0[6] as u64) | ((a.0[7] as u64) << 32);

            // 1. Non-square products. Max intermediate value:
            let (e1, e2) = mm(a0, a1);
            let (e3, e4) = mm(a0, a3);
            let (e5, e6) = mm(a2, a3);
            let (lo, hi) = mm(a0, a2);
            let (e2, cc) = adc(e2, lo, 0);
            let (e3, cc) = adc(e3, hi, cc);
            let (lo, hi) = mm(a1, a3);
            let (e4, cc) = adc(e4, lo, cc);
            let (e5, cc) = adc(e5, hi, cc);
            let (e6, _)  = adc(e6, 0, cc);
            let (lo, hi) = mm(a1, a2);
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

            let (e0, hi) = mm(a0, a0);
            let (e1, cc) = adc(e1, hi, 0);
            let (lo, hi) = mm(a1, a1);
            let (e2, cc) = adc(e2, lo, cc);
            let (e3, cc) = adc(e3, hi, cc);
            let (lo, hi) = mm(a2, a2);
            let (e4, cc) = adc(e4, lo, cc);
            let (e5, cc) = adc(e5, hi, cc);
            let (lo, hi) = mm(a3, a3);
            let (e6, cc) = adc(e6, lo, cc);
            let (e7, _)  = adc(e7, hi, cc);

            // 3. Reduction.
            let (lo, h0) = mm(e4, 2 * MQ);
            let (e0, cc) = adc(e0, lo, 0);
            let (lo, h1) = mm(e5, 2 * MQ);
            let (e1, cc) = adc(e1, lo, cc);
            let (lo, h2) = mm(e6, 2 * MQ);
            let (e2, cc) = adc(e2, lo, cc);
            let (lo, h3) = mm(e7, 2 * MQ);
            let (e3, cc) = adc(e3, lo, cc);
            let (h3, _)  = adc(h3, 0, cc);

            let h3 = (h3 << 1) | (e3 >> 63);
            let e3 = e3 & 0x7FFFFFFFFFFFFFFF;
            let (e0, cc) = adc(e0, h3 * MQ, 0);
            let (e1, cc) = adc(e1, h0, cc);
            let (e2, cc) = adc(e2, h1, cc);
            let (e3, _)  = adc(e3, h2, cc);

            GF255::<MQ>([
                e0 as u32, (e0 >> 32) as u32,
                e1 as u32, (e1 >> 32) as u32,
                e2 as u32, (e2 >> 32) as u32,
                e3 as u32, (e3 >> 32) as u32,
            ])
        }

        // 1/2 = (q + 1)/2 mod q
        let a = Self([
            (((MQ as u32) - 1) >> 1).wrapping_neg(),
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0x3FFFFFFF,
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
        let a0 = (a.0[0] as u64) | ((a.0[1] as u64) << 32);
        let a1 = (a.0[2] as u64) | ((a.0[3] as u64) << 32);
        let a2 = (a.0[4] as u64) | ((a.0[5] as u64) << 32);
        let a3 = (a.0[6] as u64) | ((a.0[7] as u64) << 32);
        let tt = a3 >> 59;
        let d0 = a0 << 4;
        let d1 = (a0 >> 60) | (a1 << 4);
        let d2 = (a1 >> 60) | (a2 << 4);
        let d3 = (a2 >> 60) | ((a3 << 4) & 0x7FFFFFFFFFFFFFFF);
        let (d0, cc) = adc(d0, tt * MQ, 0);
        let (d1, cc) = adc(d1, 0, cc);
        let (d2, cc) = adc(d2, 0, cc);
        let (d3, _)  = adc(d3, 0, cc);

        Self([
            d0 as u32, (d0 >> 32) as u32,
            d1 as u32, (d1 >> 32) as u32,
            d2 as u32, (d2 >> 32) as u32,
            d3 as u32, (d3 >> 32) as u32,
        ])
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
        println!("{} = 0x{:08X}{:08X}{:08X}{:08X}{:08X}{:08X}{:08X}{:08X}",
            name, v.0[7], v.0[6], v.0[5], v.0[4],
            v.0[3], v.0[2], v.0[1], v.0[0]);
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

        let a = GF255::<MQ>::decode32_reduce(va);
        let b = GF255::<MQ>::decode32_reduce(vb);
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

        let x = b.0[1];
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
