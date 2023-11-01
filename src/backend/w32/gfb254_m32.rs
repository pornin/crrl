use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::convert::TryFrom;

use super::{umull, sgnw};

/// Element of GF(2^127), using modulus 1 + z^63 + z^127.
#[derive(Clone, Copy, Debug)]
pub struct GFb127([u32; 4]);

impl GFb127 {

    // IMPLEMENTATION NOTES
    // --------------------
    //
    // We tolerate internal values up to 128 bits. All computations are
    // performed modulo z + z^64 + z^128, which makes reductions easier
    // (z^64 and z^128 are 64-bit aligned).

    pub const ZERO: Self = Self([ 0, 0, 0, 0 ]);
    pub const ONE: Self = Self([ 1, 0, 0, 0 ]);

    pub const fn w64le(x0: u64, x1: u64) -> Self {
        Self([ x0 as u32, (x0 >> 32) as u32, x1 as u32, (x1 >> 32) as u32 ])
    }

    // Get the bit at the specified index. The index `k` MUST be between
    // 0 and 126 (inclusive). Side-channel attacks may reveal the value of
    // the index (bit not the value of the read bit). Returned value is
    // 0 or 1.
    #[inline(always)]
    pub fn get_bit(self, k: usize) -> u32 {
        // Normalize the value first.
        let mut x = self;
        x.set_normalized();
        (x.0[k >> 5] >> (k & 31)) & 1
    }

    // Set the bit at the specified index. The index `k` MUST be between
    // 0 and 126 (inclusive). Side-channel attacks may reveal the value of
    // the index (bit not the value of the written bit). Only the least
    // significant bit of `val` is used; the over bits are ignored.
    #[inline(always)]
    pub fn set_bit(&mut self, k: usize, val: u32) {
        // We need to normalize the value, otherwise we can get the wrong
        // outcome.
        self.set_normalized();
        let ki = k >> 5;
        let kj = k & 31;
        self.0[ki] &= !(1u32 << kj);
        self.0[ki] |= (val & 1) << kj;
    }

    // XOR (add) a one-bit value at the specified index. The index `k`
    // MUST be between 0 and 126 (inclusive). Side-channel attacks may
    // reveal the value of the index (bit not the value of the added bit).
    // Only the least significant bit of `val` is used; the over bits
    // are ignored.
    #[inline(always)]
    pub fn xor_bit(&mut self, k: usize, val: u32) {
        self.0[k >> 5] ^= (val & 1) << (k & 31);
    }

    #[inline(always)]
    fn set_add(&mut self, rhs: &Self) {
        self.0[0] ^= rhs.0[0];
        self.0[1] ^= rhs.0[1];
        self.0[2] ^= rhs.0[2];
        self.0[3] ^= rhs.0[3];
    }

    // Subtraction is the same thing as addition in binary fields.

    #[inline(always)]
    pub fn set_cond(&mut self, a: &Self, ctl: u32) {
        self.0[0] ^= ctl & (self.0[0] ^ a.0[0]);
        self.0[1] ^= ctl & (self.0[1] ^ a.0[1]);
        self.0[2] ^= ctl & (self.0[2] ^ a.0[2]);
        self.0[3] ^= ctl & (self.0[3] ^ a.0[3]);
    }

    #[inline(always)]
    pub fn select(a0: &Self, a1: &Self, ctl: u32) -> Self {
        let mut r = *a0;
        r.set_cond(a1, ctl);
        r
    }

    #[inline(always)]
    pub fn cswap(a: &mut Self, b: &mut Self, ctl: u32) {
        let t = ctl & (a.0[0] ^ b.0[0]); a.0[0] ^= t; b.0[0] ^= t;
        let t = ctl & (a.0[1] ^ b.0[1]); a.0[1] ^= t; b.0[1] ^= t;
        let t = ctl & (a.0[2] ^ b.0[2]); a.0[2] ^= t; b.0[2] ^= t;
        let t = ctl & (a.0[3] ^ b.0[3]); a.0[3] ^= t; b.0[3] ^= t;
    }

    // Multiply this value by sb = 1 + z^27.
    #[inline(always)]
    pub fn set_mul_sb(&mut self) {
        let (a0, a1, a2, a3) = (self.0[0], self.0[1], self.0[2], self.0[3]);
        let c0 = a0 ^ (a0 << 27);
        let c1 = a1 ^ (a0 >> 5) ^ (a1 << 27);
        let c2 = a2 ^ (a1 >> 5) ^ (a2 << 27);
        let c3 = a3 ^ (a2 >> 5) ^ (a3 << 27);
        let c4 = a3 >> 5;
        self.0[0] = c0 ^ (c4 << 1);
        self.0[1] = c1;
        self.0[2] = c2 ^ c4;
        self.0[3] = c3;
    }

    // Multiply this value by sb = 1 + z^27.
    #[inline(always)]
    pub fn mul_sb(self) -> Self {
        let mut x = self;
        x.set_mul_sb();
        x
    }

    // Multiply this value by b = 1 + z^54.
    #[inline(always)]
    pub fn set_mul_b(&mut self) {
        let (a0, a1, a2, a3) = (self.0[0], self.0[1], self.0[2], self.0[3]);
        let c0 = a0;
        let c1 = a1 ^ (a0 << 22);
        let c2 = a2 ^ (a0 >> 10) ^ (a1 << 22);
        let c3 = a3 ^ (a1 >> 10) ^ (a2 << 22);
        let c4 = (a2 >> 10) ^ (a3 << 22);
        let c5 = a3 >> 10;
        self.0[0] = c0 ^ (c4 << 1);
        self.0[1] = c1 ^ (c4 >> 31) ^ (c5 << 1);
        self.0[2] = c2 ^ c4;
        self.0[3] = c3 ^ c5;
    }

    // Multiply this value by b = 1 + z^54.
    #[inline(always)]
    pub fn mul_b(self) -> Self {
        let mut x = self;
        x.set_mul_b();
        x
    }

    // Multiply this value by bb = 1 + z^108.
    #[inline(always)]
    pub fn set_mul_bb(&mut self) {
        let (a0, a1, a2, a3) = (self.0[0], self.0[1], self.0[2], self.0[3]);
        let c3 = a0 << 12;
        let c4 = (a0 >> 20) | (a1 << 12);
        let c5 = (a1 >> 20) | (a2 << 12);
        let c6 = (a2 >> 20) | (a3 << 12);
        let c7 = a3 >> 20;

        let s0 = c4 ^ c6;
        let s1 = c5 ^ c7;
        let d0 = a0 ^ (s0 << 1);
        let d1 = a1 ^ (s0 >> 31) ^ (s1 << 1);
        let d2 = a2 ^ (s1 >> 31) ^ (c6 << 1) ^ s0;
        let d3 = a3 ^ c3 ^ (c6 >> 31) ^ (c7 << 1) ^ s1;

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    // Multiply this value by bb = 1 + z^108.
    #[inline(always)]
    pub fn mul_bb(self) -> Self {
        let mut x = self;
        x.set_mul_bb();
        x
    }

    // Divide this value by z.
    #[inline(always)]
    pub fn set_div_z(&mut self) {
        let (a0, a1, a2, a3) = (self.0[0], self.0[1], self.0[2], self.0[3]);
        let b = (a0 & 1) << 30;
        self.0[0] = (a0 >> 1) | (a1 << 31);
        self.0[1] = ((a1 >> 1) | (a2 << 31)) ^ b;
        self.0[2] = (a2 >> 1) | (a3 << 31);
        self.0[3] = (a3 >> 1) ^ b;
    }

    // Divide this value by z.
    #[inline(always)]
    pub fn div_z(self) -> Self {
        let mut x = self;
        x.set_div_z();
        x
    }

    // Divide this value by z^2.
    #[inline(always)]
    pub fn set_div_z2(&mut self) {
        let (a0, a1, a2, a3) = (self.0[0], self.0[1], self.0[2], self.0[3]);
        let bb = (a0 & 3) << 29;
        self.0[0] = (a0 >> 2) | (a1 << 30);
        self.0[1] = ((a1 >> 2) | (a2 << 30)) ^ bb;
        self.0[2] = (a2 >> 2) | (a3 << 30);
        self.0[3] = (a3 >> 2) ^ bb;
    }

    // Divide this value by z.
    #[inline(always)]
    pub fn div_z2(self) -> Self {
        let mut x = self;
        x.set_div_z2();
        x
    }

    // Binary polynomial multiplication (32x32->64).
    #[inline]
    fn mm(x: u32, y: u32) -> (u32, u32) {
        let x0 = x & 0x11111111;
        let x1 = x & 0x22222222;
        let x2 = x & 0x44444444;
        let x3 = x & 0x88888888;
        let y0 = y & 0x11111111;
        let y1 = y & 0x22222222;
        let y2 = y & 0x44444444;
        let y3 = y & 0x88888888;

        let (u000, u001) = umull(x0, y0);
        let (u010, u011) = umull(x1, y3);
        let (u020, u021) = umull(x2, y2);
        let (u030, u031) = umull(x3, y1);
        let (u100, u101) = umull(x0, y1);
        let (u110, u111) = umull(x1, y0);
        let (u120, u121) = umull(x2, y3);
        let (u130, u131) = umull(x3, y2);
        let (u200, u201) = umull(x0, y2);
        let (u210, u211) = umull(x1, y1);
        let (u220, u221) = umull(x2, y0);
        let (u230, u231) = umull(x3, y3);
        let (u300, u301) = umull(x0, y3);
        let (u310, u311) = umull(x1, y2);
        let (u320, u321) = umull(x2, y1);
        let (u330, u331) = umull(x3, y0);

        let z00 = (u000 ^ u010 ^ u020 ^ u030) & 0x11111111;
        let z01 = (u001 ^ u011 ^ u021 ^ u031) & 0x11111111;
        let z10 = (u100 ^ u110 ^ u120 ^ u130) & 0x22222222;
        let z11 = (u101 ^ u111 ^ u121 ^ u131) & 0x22222222;
        let z20 = (u200 ^ u210 ^ u220 ^ u230) & 0x44444444;
        let z21 = (u201 ^ u211 ^ u221 ^ u231) & 0x44444444;
        let z30 = (u300 ^ u310 ^ u320 ^ u330) & 0x88888888;
        let z31 = (u301 ^ u311 ^ u321 ^ u331) & 0x88888888;

        (z00 | z10 | z20 | z30, z01 | z11 | z21 | z31)
    }

    fn set_mul(&mut self, rhs: &Self) {
        let (a0, a1, a2, a3) = (self.0[0], self.0[1], self.0[2], self.0[3]);
        let (b0, b1, b2, b3) = (rhs.0[0], rhs.0[1], rhs.0[2], rhs.0[3]);

        // Karatsuba-Ofman:
        //   (c0 + c1*z^64)*(d0 + d1*z^64)
        //    = c0*d0 + (c0*d1 + c1*d0)*z^64 + c1*d1*z^128
        // We use:
        //   (c0 + c1)*(d0 + d1) + c0*d0 + c1*d1 = c0*d1 + c1*d0
        // This transform is applied recursively on two levels, so that
        // the operations is reduced to 9 invocations of mm().

        // c0:c1 <- a0:a1 + a2:a3
        let c0 = a0 ^ a2;
        let c1 = a1 ^ a3;

        // d0:d1 <- b0:b1 + b2:b3
        let d0 = b0 ^ b2;
        let d1 = b1 ^ b3;

        // e0:e1:e2:e3 <- a0:a1 * b0:b1
        let (tl0, tl1) = Self::mm(a0, b0);
        let (th0, th1) = Self::mm(a1, b1);
        let (tm0, tm1) = Self::mm(a0 ^ a1, b0 ^ b1);
        let (tm0, tm1) = (tm0 ^ tl0 ^ th0, tm1 ^ tl1 ^ th1);
        let e0 = tl0;
        let e1 = tl1 ^ tm0;
        let e2 = th0 ^ tm1;
        let e3 = th1;

        // f0:f1:f2:f3 <- a2:a3 * b2:b3
        let (tl0, tl1) = Self::mm(a2, b2);
        let (th0, th1) = Self::mm(a3, b3);
        let (tm0, tm1) = Self::mm(a2 ^ a3, b2 ^ b3);
        let (tm0, tm1) = (tm0 ^ tl0 ^ th0, tm1 ^ tl1 ^ th1);
        let f0 = tl0;
        let f1 = tl1 ^ tm0;
        let f2 = th0 ^ tm1;
        let f3 = th1;

        // g0:g1:g2:g3 <- c0:c1 * d0:d1
        let (tl0, tl1) = Self::mm(c0, d0);
        let (th0, th1) = Self::mm(c1, d1);
        let (tm0, tm1) = Self::mm(c0 ^ c1, d0 ^ d1);
        let (tm0, tm1) = (tm0 ^ tl0 ^ th0, tm1 ^ tl1 ^ th1);
        let g0 = tl0;
        let g1 = tl1 ^ tm0;
        let g2 = th0 ^ tm1;
        let g3 = th1;

        // Assemble the unreduced result in r0..r7
        let r0 = e0;
        let r1 = e1;
        let r2 = e2 ^ e0 ^ f0 ^ g0;
        let r3 = e3 ^ e1 ^ f1 ^ g1;
        let r4 = f0 ^ e2 ^ f2 ^ g2;
        let r5 = f1 ^ e3 ^ f3 ^ g3;
        let r6 = f2;
        let r7 = f3;

        // Reduction: z^128 = z + z^64
        // Note: r0..7 has length at most 255 bits (not 256), so r7 fits
        // on 31 bits.
        //
        // (t2 + t3*z^64)*z^128
        //  = (t2 + t3*z^64)*z + t2*z^64 + t3*(z + z^64)
        //  = (t2 + t3 + t3*z^64)*z + (t2 + t3)*z^64
        // with:
        //    t2 = r4:r5
        //    t3 = r6:r7
        // We set s0:s1 = t2 + t3
        let s0 = r4 ^ r6;
        let s1 = r5 ^ r7;

        let d0 = r0 ^ (s0 << 1);
        let d1 = r1 ^ (s0 >> 31) ^ (s1 << 1);
        let d2 = r2 ^ (s1 >> 31) ^ (r6 << 1) ^ s0;
        let d3 = r3 ^ (r6 >> 31) ^ (r7 << 1) ^ s1;

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    // Expand the low 16 bits of the input into a 32-bit word by
    // inserting zeros between any two consecutive input bits
    // (this is the squaring of a binary polynomial).
    #[inline(always)]
    fn expand_lo(x: u32) -> u32 {
        let x = (x & 0x000000FF) | ((x & 0x0000FF00) <<  8);
        let x = (x & 0x000F000F) | ((x & 0x00F000F0) <<  4);
        let x = (x & 0x03030303) | ((x & 0x0C0C0C0C) <<  2);
        let x = (x & 0x11111111) | ((x & 0x22222222) <<  1);
        x
    }

    // Extract all even-indexed bits from the input and push them into
    // the low 16 bits of the result; the high 16 bits are set to 0.
    #[inline(always)]
    fn squeeze(x: u32) -> u32 {
        let x = (x & 0x11111111) | ((x & 0x44444444) >> 1);
        let x = (x & 0x03030303) | ((x & 0x30303030) >> 2);
        let x = (x & 0x000F000F) | ((x & 0x0F000F00) >> 4);
        let x = (x & 0x000000FF) | ((x & 0x00FF0000) >> 8);
        x
    }

    // Square this value (in place).
    #[inline(always)]
    pub fn set_square(&mut self) {
        let (a0, a1, a2, a3) = (self.0[0], self.0[1], self.0[2], self.0[3]);

        // Square the polynomial.
        let r0 = Self::expand_lo(a0);
        let r1 = Self::expand_lo(a0 >> 16);
        let r2 = Self::expand_lo(a1);
        let r3 = Self::expand_lo(a1 >> 16);
        let r4 = Self::expand_lo(a2);
        let r5 = Self::expand_lo(a2 >> 16);
        let r6 = Self::expand_lo(a3);
        let r7 = Self::expand_lo(a3 >> 16);

        // Reduce.
        let s0 = r4 ^ r6;
        let s1 = r5 ^ r7;

        let d0 = r0 ^ (s0 << 1);
        let d1 = r1 ^ (s0 >> 31) ^ (s1 << 1);
        let d2 = r2 ^ (s1 >> 31) ^ (r6 << 1) ^ s0;
        let d3 = r3 ^ (r6 >> 31) ^ (r7 << 1) ^ s1;

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    // Square this value.
    #[inline(always)]
    pub fn square(self) -> Self {
        let mut r = self;
        r.set_square();
        r
    }

    // Square this value n times (in place).
    // Note: for large values of n, this can be done more efficiently with
    // a precomputed table, since squaring is linear.
    #[inline(always)]
    fn set_xsquare(&mut self, n: u32) {
        for _ in 0..n {
            self.set_square();
        }
    }

    // Square this value n times.
    // Note: for large values of n, this can be done more efficiently with
    // a precomputed table, since squaring is linear.
    #[inline(always)]
    pub fn xsquare(self, n: u32) -> Self {
        let mut r = self;
        r.set_xsquare(n);
        r
    }

    // Ensure that the internal encoding is reduced to 127 bits.
    #[inline]
    fn set_normalized(&mut self) {
        let h = self.0[3] & 0x80000000;
        self.0[0] ^= h >> 31;
        self.0[1] ^= h;
        self.0[3] ^= h;
    }

    // For divisions, we use an optimized binary GCD. Depending on the
    // target platforms, this might be faster or slower than the
    // Itoh-Tsujii algorithm (which uses Fermat's Little Theorem,
    // optimized with tables for computing sequences of squarings), but
    // it does not need large tables.

    // Update polynomials a and b with the provided factors:
    //   a' = (a*f0 + b*g0)/z^32
    //   b' = (a*f1 + b*g1)/z^32
    // Divisions by z^32 are assumed to be exact (low bits are dropped).
    //
    // Coefficients f1 and g1 are always even; parameters f1z and g1z
    // are equal to f1/z and g1/z, respectively (the full f1 and g1 may
    // need 33 bits each).
    fn lin_div32(a: &mut Self, b: &mut Self,
        f0: u32, g0: u32, f1z: u32, g1z: u32)
    {
        // a*f0 + b*g0, keeping only the relevant parts.
        let (_, hi) = Self::mm(a.0[0], f0);
        let (mut a0, mut a1) = Self::mm(a.0[1], f0);
        a0 ^= hi;
        let (lo, mut a2) = Self::mm(a.0[2], f0);
        a1 ^= lo;
        let (lo, mut a3) = Self::mm(a.0[3], f0);
        a2 ^= lo;
        let (_, hi) = Self::mm(b.0[0], g0);
        a0 ^= hi;
        let (lo, hi) = Self::mm(b.0[1], g0);
        a0 ^= lo;
        a1 ^= hi;
        let (lo, hi) = Self::mm(b.0[2], g0);
        a1 ^= lo;
        a2 ^= hi;
        let (lo, hi) = Self::mm(b.0[3], g0);
        a2 ^= lo;
        a3 ^= hi;

        // z*(a*f1z + b*g1z), keeping only the relevant parts.
        let (mut bf, hi) = Self::mm(a.0[0], f1z);
        let (mut b0, mut b1) = Self::mm(a.0[1], f1z);
        b0 ^= hi;
        let (lo, mut b2) = Self::mm(a.0[2], f1z);
        b1 ^= lo;
        let (lo, mut b3) = Self::mm(a.0[3], f1z);
        b2 ^= lo;
        let (lo, hi) = Self::mm(b.0[0], g1z);
        bf ^= lo;
        b0 ^= hi;
        let (lo, hi) = Self::mm(b.0[1], g1z);
        b0 ^= lo;
        b1 ^= hi;
        let (lo, hi) = Self::mm(b.0[2], g1z);
        b1 ^= lo;
        b2 ^= hi;
        let (lo, hi) = Self::mm(b.0[3], g1z);
        b2 ^= lo;
        b3 ^= hi;
        let (b0, b1, b2, b3) = (
            (bf >> 31) | (b0 << 1),
            (b0 >> 31) | (b1 << 1),
            (b1 >> 31) | (b2 << 1),
            (b2 >> 31) | (b3 << 1));

        a.0[0] = a0;
        a.0[1] = a1;
        a.0[2] = a2;
        a.0[3] = a3;
        b.0[0] = b0;
        b.0[1] = b1;
        b.0[2] = b2;
        b.0[3] = b3;
    }

    // Update field elements a and b with the provided factors:
    //   a' = a*f0 + b*g0
    //   b' = a*f1 + b*g1
    //
    // Coefficients f1 and g1 are always even; parameters f1z and g1z
    // are equal to f1/z and g1/z, respectively (the full f1 and g1 may
    // need 33 bits each).
    fn lin(a: &mut Self, b: &mut Self,
        f0: u32, g0: u32, f1z: u32, g1z: u32)
    {
        // a*f0 + b*g0
        let (mut a0, mut a1) = Self::mm(a.0[0], f0);
        let (lo, mut a2) = Self::mm(a.0[1], f0);
        a1 ^= lo;
        let (lo, mut a3) = Self::mm(a.0[2], f0);
        a2 ^= lo;
        let (lo, mut a4) = Self::mm(a.0[3], f0);
        a3 ^= lo;
        let (lo, hi) = Self::mm(b.0[0], g0);
        a0 ^= lo;
        a1 ^= hi;
        let (lo, hi) = Self::mm(b.0[1], g0);
        a1 ^= lo;
        a2 ^= hi;
        let (lo, hi) = Self::mm(b.0[2], g0);
        a2 ^= lo;
        a3 ^= hi;
        let (lo, hi) = Self::mm(b.0[3], g0);
        a3 ^= lo;
        a4 ^= hi;

        // Reduce a0..a4
        // Note: a4 fits on 31 bits.
        a0 ^= a4 << 1;
        a2 ^= a4;

        // (a*f1 + b*g1)/z
        let (mut b0, mut b1) = Self::mm(a.0[0], f1z);
        let (lo, mut b2) = Self::mm(a.0[1], f1z);
        b1 ^= lo;
        let (lo, mut b3) = Self::mm(a.0[2], f1z);
        b2 ^= lo;
        let (lo, mut b4) = Self::mm(a.0[3], f1z);
        b3 ^= lo;
        let (lo, hi) = Self::mm(b.0[0], g1z);
        b0 ^= lo;
        b1 ^= hi;
        let (lo, hi) = Self::mm(b.0[1], g1z);
        b1 ^= lo;
        b2 ^= hi;
        let (lo, hi) = Self::mm(b.0[2], g1z);
        b2 ^= lo;
        b3 ^= hi;
        let (lo, hi) = Self::mm(b.0[3], g1z);
        b3 ^= lo;
        b4 ^= hi;

        // Multiply (a*f1 + b*g1)/z by z.
        // Note that b4 fits on 31 bits.
        b4 = (b4 << 1) | (b3 >> 31);
        b3 = (b3 << 1) | (b2 >> 31);
        b2 = (b2 << 1) | (b1 >> 31);
        b1 = (b1 << 1) | (b0 >> 31);
        b0 <<= 1;

        // Reduce b0:b1:b2.
        b0 ^= b4 << 1;
        b1 ^= b4 >> 31;
        b2 ^= b4;

        a.0[0] = a0;
        a.0[1] = a1;
        a.0[2] = a2;
        a.0[3] = a3;
        b.0[0] = b0;
        b.0[1] = b1;
        b.0[2] = b2;
        b.0[3] = b3;
    }

    // Invert this value; if this value is zero, then it stays at zero.
    #[inline(always)]
    pub fn set_invert(&mut self) {
        let r = self.invert();
        *self = r;
    }

    // Get the inverse of this value; the inverse of zero is formally
    // defined to be zero.
    #[inline(always)]
    pub fn invert(self) -> Self {
        Self::ONE / self
    }

    // Set this value x to x/y. If the divisor y is zero, then this sets
    // this value to zero, regardless of its initial value.
    fn set_div(&mut self, y: &Self) {
        // Binary GCD variant, from Brunner, Curiger and Hofstetter ("On
        // computing multiplicative inverses in GF(2^m)", IEEE Trans. on
        // Computers, vol. 48, issue 8, pp. 1010-1015, 1993). We can
        // perform iterations in chunks of 32, working only on low
        // words, and propagating the changes only once every 32 inner
        // iterations. Compared to the binary GCD on integers (as in
        // eprint.iacr.org/2020/972), the carryless nature of
        // multiplications in GF(2)[z] simplifies things: we do not need
        // to keep track of the high bits precisely; we only need a
        // counter that remembers the balance between the maximum bounds
        // of the value degrees. We can thus run 32 inner iterations
        // instead of 15; we also do not need any sign-based corrective
        // step.
        //
        //   a <- y  (normalized)
        //   b <- m  (field modulus = z^127 + z^63 + 1)
        //   u <- x
        //   v <- 0
        //   n <- -1
        //   invariants:
        //      a*x = u*y mod m
        //      b*x = v*y mod m
        //      n = maxsize(a) - maxsize(b)
        //      b_0 = 1  (least significant bit of polynomial b)
        //   (maxsize(t) is the proven bound on the size of t; we use
        //   size(t) = 1 + degree(t))
        //
        // At each iteration:
        //   if a_0 != 0:
        //       if n < 0:
        //           (a, u, b, v) <- (b, v, a, u)
        //           n = -n
        //       a <- a + b
        //       u <- u + v
        //   a <- a/z
        //   n <- n - 1
        //
        // maxsize(a) + maxsize(b) starts at 255. Each iteration decreases
        // that sum by 1; hence, 253 iterations are sufficient to ensure
        // that size(a) + size(b) <= 2, assuming that y != 0: modulus m is
        // irreducible, so the algorithm cannot reach a = 0 unless b = 1,
        // since b then contains the GCD of m and y. For the same reason,
        // at that point, we cannot have size(b) = 2, since that would imply
        // that a = 0. Since b_0 = 1, we necessarily have b = 1 after 253
        // iterations (still under the assumption that y != 0), and v then
        // contains the result (y/x).
        //
        // If y = 0, then v and b are unchanged throughout the algorithm,
        // and v remains at zero; this is the result we want to return in
        // that case.

        // We multiply the divisor by z^256, because the implementation
        // below will leave the result in v scaled up by z^256; thus,
        // the multiplication on the divisor counteracts that scaling.
        // In the field, z^256 = z^64 + z^2 + z.
        let (y0, y1, y2, y3) = (y.0[0], y.0[1], y.0[2], y.0[3]);
        let c0 = (y0 << 1) ^ (y0 << 2);
        let c1 = (y0 >> 31) ^ (y1 << 1) ^ (y0 >> 30) ^ (y1 << 2);
        let c2 = (y1 >> 31) ^ (y2 << 1) ^ (y1 >> 30) ^ (y2 << 2) ^ y0;
        let c3 = (y2 >> 31) ^ (y3 << 1) ^ (y2 >> 30) ^ (y3 << 2) ^ y1;
        let c4 = (y3 >> 31) ^ (y3 >> 30) ^ y2;
        let c5 = y3;
        let a0 = c0 ^ (c4 << 1);
        let a1 = c1 ^ (c4 >> 31) ^ (c5 << 1);
        let a2 = c2 ^ (c5 >> 31) ^ c4;
        let a3 = c3 ^ c5;
        let mut a = Self([ a0, a1, a2, a3, ]);
        a.set_normalized();
        let mut b = Self([ 1, 0x80000000, 0, 0x80000000, ]);
        let mut u = *self;
        let mut v = Self::ZERO;
        let mut n = 0xFFFFFFFFu32;

        // Total number of iterations: 7*32 + 29 = 253.
        for i in 0..8 {
            let mut xa = a.0[0];
            let mut xb = b.0[0];
            let mut f0 = 1u32;
            let mut g0 = 0u32;
            let mut f1 = 0u32;
            let mut g1 = 1u32;
            let num_inner;
            if i == 7 {
                f0 <<= 3;
                g1 <<= 3;
                num_inner = 29;
            } else {
                num_inner = 32;
            }

            for j in 1..(num_inner + 1) {
                // a_odd = -1 if a_0 = 1
                let a_odd = (xa & 1).wrapping_neg();
                // n_neg = -1 if n < 0
                let n_neg = sgnw(n);
                // swap = -1 if we must swap the values
                let swap = a_odd & n_neg;
                let t = swap & (xa ^ xb);
                xa ^= t;
                xb ^= t;
                let t = swap & (f0 ^ f1);
                f0 ^= t;
                f1 ^= t;
                let t = swap & (g0 ^ g1);
                g0 ^= t;
                g1 ^= t;
                n = n.wrapping_sub(swap & (n << 1));
                // XOR b into a if a is odd
                xa ^= a_odd & xb;
                f0 ^= a_odd & f1;
                g0 ^= a_odd & g1;
                // xa is now even, divide by z
                xa >>= 1;
                n = n.wrapping_sub(1);
                // We exit the loop before the shift on f1 and g1 because
                // we need to handle the case of f1 or g1 overflowing the
                // 64-bit variable.
                if j == num_inner {
                    break;
                }
                f1 <<= 1;
                g1 <<= 1;
            }

            // Propagate changes to a and b.
            Self::lin_div32(&mut a, &mut b, f0, g0, f1, g1);
            Self::lin(&mut u, &mut v, f0, g0, f1, g1);
        }

        // Result is in v.
        *self = v;
    }

    // Set this value to its square root. In a binary field, all values
    // have a square root, and it is unique.
    #[inline(always)]
    pub fn set_sqrt(&mut self) {
        // We split the input into "odd" and "even" parts:
        //    a = ae + z*ao
        // with:
        //    ae = \sum_{i=0}^{63} a_{2*i}*z^{2*i}
        //    ao = \sum_{i=0}^{62} a_{2*i+1}*z^{2*i}
        // Then:
        //    sqrt(a) = sqrt(ae) + sqrt(z)*sqrt(ao)
        // Square roots of ae and ao are obtained by "squeezing" words
        // (odd-numbered digits are removed). In GF(2^127) with our
        // defined modulus, sqrt(z) = z^64 + z^32, so the multiplication
        // by sqrt(z) is done easily; in fact, no reduction is necessary
        // since sqrt(ae) and sqrt(ao) both fit on 64 bits.

        let (a0, a1, a2, a3) = (self.0[0], self.0[1], self.0[2], self.0[3]);

        // sqrt(ae) = c0:c1
        let c0 = Self::squeeze(a0) | (Self::squeeze(a1) << 16);
        let c1 = Self::squeeze(a2) | (Self::squeeze(a3) << 16);

        // sqrt(ao) = d0:d1
        let d0 = Self::squeeze(a0 >> 1) | (Self::squeeze(a1 >> 1) << 16);
        let d1 = Self::squeeze(a2 >> 1) | (Self::squeeze(a3 >> 1) << 16);

        // sqrt(a) = c0:c1 + (z^32 + z^64)*(d0:d1)
        self.0[0] = c0;
        self.0[1] = c1 ^ d0;
        self.0[2] = d1 ^ d0;
        self.0[3] = d1;
    }

    // Compute the square root of this value. In a binary field, all values
    // have a square root, and it is unique.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        let mut x = self;
        x.set_sqrt();
        x
    }

    // Get the trace for this value (in GF(2^127)). This is 0 or 1.
    #[inline(always)]
    pub fn trace(self) -> u32 {
        // For i = 0 to 126, only z^0 = 1 has trace 1. However, we must
        // also take into account z^127 (our internal format is not
        // entirely reduced).
        (self.0[0] ^ (self.0[3] >> 31)) & 1
    }

    // Set this value to its halftrace.
    #[inline]
    pub fn set_halftrace(&mut self) {
        // We split the input into "odd" and "even" parts:
        //    a = ae + z*ao
        // with:
        //    ae = \sum_{i=0}^{63} a_{2*i}*z^{2*i}
        //    ao = \sum_{i=0}^{62} a_{2*i+1}*z^{2*i}
        // We then have:
        //    H(a) = H(ae) + H(z*ao)
        // Since H(x) = H(sqrt(x)) + sqrt(x) for all x, we can replace H(ae):
        //    H(a) = H(sqrt(ae)) + H(z*ao) + sqrt(ae)
        // sqrt(ae) is obtained through squeezing and has half-size, so it
        // can be split again, recursively. We thus remove all even-indexed
        // bits from the computation, which allows use of a half-size table
        // for the matrix that processes the odd-indexed bit.

        // We accumulate the odd-indexed bits in ao. We will ignore the
        // even-indexed bits in this variable, so we do not care what values
        // are written there.
        let mut ao = *self;

        // We accumulate the extra values (square roots) into e0:e1.
        let x0 = Self::squeeze(self.0[0]) | (Self::squeeze(self.0[1]) << 16);
        let x1 = Self::squeeze(self.0[2]) | (Self::squeeze(self.0[3]) << 16);
        let mut e0 = x0;
        let e1 = x1;

        // Shrink x from 64 to 32 bits.
        ao.0[0] ^= x0;
        ao.0[1] ^= x1;
        let mut x = Self::squeeze(x0) | (Self::squeeze(x1) << 16);
        e0 ^= x;

        // At this point, we have:
        //    H(a) = H(x) + H(z*a0) + e
        // and x has length 32 bits. We apply the even/odd split
        // repeatedly until x is a 1-bit value, thus equal to its halftrace.
        for _ in 0..5 {
            ao.0[0] ^= x;
            x = Self::squeeze(x);
            e0 ^= x;
        }

        // We now get the halftrace of the odd-indexed bits in ao.
        let (mut d0, mut d1, mut d2, mut d3) = (e0 ^ x, e1, 0, 0);
        for i in 0..4 {
            let mut mw = ao.0[i];
            for j in (0..16).rev() {
                let m = sgnw(mw);
                mw <<= 2;
                d0 ^= m & Self::HALFTRACE[(i << 4) + j].0[0];
                d1 ^= m & Self::HALFTRACE[(i << 4) + j].0[1];
                d2 ^= m & Self::HALFTRACE[(i << 4) + j].0[2];
                d3 ^= m & Self::HALFTRACE[(i << 4) + j].0[3];
            }
        }

        self.0[0] = d0;
        self.0[1] = d1;
        self.0[2] = d2;
        self.0[3] = d3;
    }

    // Get the halftrace of this value (in GF(2^127)).
    #[inline(always)]
    pub fn halftrace(self) -> Self {
        let mut x = self;
        x.set_halftrace();
        x
    }

    // Halftrace of z^(2*i+1) for i = 0 to 63.
    const HALFTRACE: [Self; 64] = [
        GFb127::w64le(0x0000000000000000, 0x0000000000000001),
        GFb127::w64le(0x0001011201141668, 0x0000000000010014),
        GFb127::w64le(0x000100110105135E, 0x0000000100000016),
        GFb127::w64le(0x01031401116159DE, 0x0000000501000426),
        GFb127::w64le(0x000101150117177E, 0x0000000100000106),
        GFb127::w64le(0x0010017C041E2620, 0x0000011400060260),
        GFb127::w64le(0x01010472112C52C8, 0x0001001200040648),
        GFb127::w64le(0x1204585042CC8A00, 0x0004043010241E00),
        GFb127::w64le(0x0000001400060200, 0x0000000000000010),
        GFb127::w64le(0x0000043000240200, 0x0000001000040600),
        GFb127::w64le(0x0105135E135E5EE8, 0x0001011600121628),
        GFb127::w64le(0x04506EC02CA82000, 0x0010064000686000),
        GFb127::w64le(0x0010150C04722C20, 0x0000010400021460),
        GFb127::w64le(0x055D5EE23FE878C8, 0x0015162202284848),
        GFb127::w64le(0x15522EC87C28E080, 0x0112064800682080),
        GFb127::w64le(0x75E2E808F880C080, 0x0562280848804080),
        GFb127::w64le(0x000100030101115E, 0x0000000100000002),
        GFb127::w64le(0x0101000A110050C8, 0x0001000200000008),
        GFb127::w64le(0x0000042000200000, 0x0000000000000400),
        GFb127::w64le(0x110200885000C080, 0x0102000800000080),
        GFb127::w64le(0x0014132C06522C20, 0x0000010400061060),
        GFb127::w64le(0x0000040000200000, 0x0000040000200000),
        GFb127::w64le(0x051D52E237C878C8, 0x0015122202484848),
        GFb127::w64le(0x52088080C0008000, 0x1208008000008000),
        GFb127::w64le(0x0013057D053F377E, 0x0000011100060646),
        GFb127::w64le(0x01492C02192050C8, 0x0001044200602048),
        GFb127::w64le(0x144F5C2A6BE09848, 0x0107146A062068C8),
        GFb127::w64le(0x0040200008000000, 0x0040200008000000),
        GFb127::w64le(0x065476902EE42A00, 0x00140270004C7E00),
        GFb127::w64le(0x5628C880E0808000, 0x1628488020808000),
        GFb127::w64le(0x67EAE888B8804080, 0x176A28880880C080),
        GFb127::w64le(0x6880800080000000, 0x6880800080000000),
        GFb127::w64le(0x0000011300150736, 0x0000000100010014),
        GFb127::w64le(0x0002140300610916, 0x0001000701000426),
        GFb127::w64le(0x0010057C043E2620, 0x0000011400060640),
        GFb127::w64le(0x0306585812CC4A80, 0x0106043810241E00),
        GFb127::w64le(0x0014151C06762E20, 0x0000011400021460),
        GFb127::w64le(0x045062C02C882000, 0x0010024000486800),
        GFb127::w64le(0x00402C0008200000, 0x0000040000602000),
        GFb127::w64le(0x27EAE88838804080, 0x176A288848804080),
        GFb127::w64le(0x01100577143F67B6, 0x000101130004064E),
        GFb127::w64le(0x10432C8A49209048, 0x0103044A006820C8),
        GFb127::w64le(0x146F582A6BC09848, 0x0107106A062068C8),
        GFb127::w64le(0x52C8A080C8008000, 0x1248208008808000),
        GFb127::w64le(0x051D5A9237C47AC8, 0x00150632022C5E48),
        GFb127::w64le(0x5E68E880E8808000, 0x1668688020808000),
        GFb127::w64le(0x11C220085800C080, 0x0142600808004080),
        GFb127::w64le(0x8000000000000000, 0x0000000080000000),
        GFb127::w64le(0x0002151000740E20, 0x0001000401010430),
        GFb127::w64le(0x03044C5B12AD4396, 0x0107043711241A2E),
        GFb127::w64le(0x044067BC28B60620, 0x00100374004E6E60),
        GFb127::w64le(0x24ECB0D02A4C0A00, 0x166C2C3058A45E00),
        GFb127::w64le(0x105739964F56BE68, 0x0103075E006A36A8),
        GFb127::w64le(0x5698C240E488A000, 0x12582AC008C8E000),
        GFb127::w64le(0x5E28C480E0A08000, 0x16684C8020E08000),
        GFb127::w64le(0xA7EAE88838804080, 0x176AA888C880C080),
        GFb127::w64le(0x0214492C06922420, 0x0104052411221C60),
        GFb127::w64le(0x34AF9C5A636C9A48, 0x1767287A58C47EC8),
        GFb127::w64le(0x42F79A6A8F483848, 0x137F3AAA0EC888C8),
        GFb127::w64le(0xF5224808F080C080, 0x05A28808C0804080),
        GFb127::w64le(0x31B2C6C854A8E080, 0x15722E4858E82080),
        GFb127::w64le(0xAB4AA08818004080, 0x1BCAE088E800C080),
        GFb127::w64le(0xBA88808040008000, 0x3A888080C0008000),
        GFb127::w64le(0x6880800080000000, 0x6880800080000000),
    ];

    // Equality check between two field elements (constant-time);
    // returned value is 0xFFFFFFFF on equality, 0x00000000 otherwise.
    #[inline(always)]
    pub fn equals(self, rhs: Self) -> u32 {
        (self + rhs).iszero()
    }

    // Compare this value with zero (constant-time); returned value
    // is 0xFFFFFFFF if this element is zero, 0x00000000 otherwise.
    #[inline]
    pub fn iszero(self) -> u32 {
        let (a0, a1, a2, a3) = (self.0[0], self.0[1], self.0[2], self.0[3]);

        // Normalize the value.
        let h = a3 & 0x80000000;
        let a0 = a0 ^ (h >> 31);
        let a1 = a1 ^ h;
        let a3 = a3 ^ h;

        // Check that we got a full zero.
        let t = a0 | a1 | a2 | a3;
        ((t | t.wrapping_neg()) >> 31).wrapping_sub(1)
    }

    #[inline(always)]
    pub fn encode(self) -> [u8; 16] {
        let mut r = self;
        r.set_normalized();
        let mut d = [0u8; 16];
        d[ 0.. 4].copy_from_slice(&r.0[0].to_le_bytes());
        d[ 4.. 8].copy_from_slice(&r.0[1].to_le_bytes());
        d[ 8..12].copy_from_slice(&r.0[2].to_le_bytes());
        d[12..16].copy_from_slice(&r.0[3].to_le_bytes());
        d
    }

    // Decode the value from bytes with implicit reduction modulo
    // z^127 + z^63 + 1. Input MUST be of length 16 bytes exactly.
    #[inline]
    fn set_decode16_reduce(&mut self, buf: &[u8]) {
        debug_assert!(buf.len() == 16);
        self.0[0] = u32::from_le_bytes(*<&[u8; 4]>::try_from(&buf[ 0.. 4]).unwrap());
        self.0[1] = u32::from_le_bytes(*<&[u8; 4]>::try_from(&buf[ 4.. 8]).unwrap());
        self.0[2] = u32::from_le_bytes(*<&[u8; 4]>::try_from(&buf[ 8..12]).unwrap());
        self.0[3] = u32::from_le_bytes(*<&[u8; 4]>::try_from(&buf[12..16]).unwrap());
    }

    // Decode the value from bytes. If the input is invalid (i.e. the
    // input length is not exactly 16 bytes, or the top bit of the last
    // byte is not zero), then this value is set to zero and 0x00000000
    // is returned. Otherwise, the decoding succeeds, and 0xFFFFFFFF is
    // returned.
    #[inline]
    pub fn set_decode_ct(&mut self, buf: &[u8]) -> u32 {
        if buf.len() != 16 {
            *self = Self::ZERO;
            return 0;
        }
        self.set_decode16_reduce(buf);
        let m = !sgnw(self.0[3]);
        self.0[0] &= m;
        self.0[1] &= m;
        self.0[2] &= m;
        self.0[3] &= m;
        m
    }

    // Decode a value from bytes. If the input is invalid (i.e. the
    // input length is not exactly 16 bytes, or the top bit of the last
    // byte is not zero), then this returns zero and 0x00000000.
    // Otherwise, the decoded value and 0xFFFFFFFF are returned.
    #[inline]
    pub fn decode_ct(buf: &[u8]) -> (Self, u32) {
        let mut x = Self::ZERO;
        let r = x.set_decode_ct(buf);
        (x, r)
    }

    // Decode a value from bytes. If the input is invalid (i.e. the
    // input length is not exactly 16 bytes, or the top bit of the last
    // byte is not zero), then this returns `None`; otherwise, the decoded
    // value is returned. Side-channel analysis may reveal to outsiders
    // whether the decoding succeeded.
    #[inline]
    pub fn decode(buf: &[u8]) -> Option<Self> {
        let (x, r) = Self::decode_ct(buf);
        if r != 0 {
            Some(x)
        } else {
            None
        }
    }
}

// ========================================================================
// Implementations of all the traits needed to use the simple operators
// (+, *, /...) on field element instances, with or without references.

impl Add<GFb127> for GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn add(self, other: GFb127) -> GFb127 {
        let mut r = self;
        r.set_add(&other);
        r
    }
}

impl Add<&GFb127> for GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn add(self, other: &GFb127) -> GFb127 {
        let mut r = self;
        r.set_add(other);
        r
    }
}

impl Add<GFb127> for &GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn add(self, other: GFb127) -> GFb127 {
        let mut r = *self;
        r.set_add(&other);
        r
    }
}

impl Add<&GFb127> for &GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn add(self, other: &GFb127) -> GFb127 {
        let mut r = *self;
        r.set_add(other);
        r
    }
}

impl AddAssign<GFb127> for GFb127 {
    #[inline(always)]
    fn add_assign(&mut self, other: GFb127) {
        self.set_add(&other);
    }
}

impl AddAssign<&GFb127> for GFb127 {
    #[inline(always)]
    fn add_assign(&mut self, other: &GFb127) {
        self.set_add(other);
    }
}

impl Div<GFb127> for GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn div(self, other: GFb127) -> GFb127 {
        let mut r = self;
        r.set_div(&other);
        r
    }
}

impl Div<&GFb127> for GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn div(self, other: &GFb127) -> GFb127 {
        let mut r = self;
        r.set_div(other);
        r
    }
}

impl Div<GFb127> for &GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn div(self, other: GFb127) -> GFb127 {
        let mut r = *self;
        r.set_div(&other);
        r
    }
}

impl Div<&GFb127> for &GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn div(self, other: &GFb127) -> GFb127 {
        let mut r = *self;
        r.set_div(other);
        r
    }
}

impl DivAssign<GFb127> for GFb127 {
    #[inline(always)]
    fn div_assign(&mut self, other: GFb127) {
        self.set_div(&other);
    }
}

impl DivAssign<&GFb127> for GFb127 {
    #[inline(always)]
    fn div_assign(&mut self, other: &GFb127) {
        self.set_div(other);
    }
}

impl Mul<GFb127> for GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn mul(self, other: GFb127) -> GFb127 {
        let mut r = self;
        r.set_mul(&other);
        r
    }
}

impl Mul<&GFb127> for GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn mul(self, other: &GFb127) -> GFb127 {
        let mut r = self;
        r.set_mul(other);
        r
    }
}

impl Mul<GFb127> for &GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn mul(self, other: GFb127) -> GFb127 {
        let mut r = *self;
        r.set_mul(&other);
        r
    }
}

impl Mul<&GFb127> for &GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn mul(self, other: &GFb127) -> GFb127 {
        let mut r = *self;
        r.set_mul(other);
        r
    }
}

impl MulAssign<GFb127> for GFb127 {
    #[inline(always)]
    fn mul_assign(&mut self, other: GFb127) {
        self.set_mul(&other);
    }
}

impl MulAssign<&GFb127> for GFb127 {
    #[inline(always)]
    fn mul_assign(&mut self, other: &GFb127) {
        self.set_mul(other);
    }
}

impl Neg for GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn neg(self) -> GFb127 {
        self
    }
}

impl Neg for &GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn neg(self) -> GFb127 {
        *self
    }
}

impl Sub<GFb127> for GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn sub(self, other: GFb127) -> GFb127 {
        let mut r = self;
        r.set_add(&other);
        r
    }
}

impl Sub<&GFb127> for GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn sub(self, other: &GFb127) -> GFb127 {
        let mut r = self;
        r.set_add(other);
        r
    }
}

impl Sub<GFb127> for &GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn sub(self, other: GFb127) -> GFb127 {
        let mut r = *self;
        r.set_add(&other);
        r
    }
}

impl Sub<&GFb127> for &GFb127 {
    type Output = GFb127;

    #[inline(always)]
    fn sub(self, other: &GFb127) -> GFb127 {
        let mut r = *self;
        r.set_add(other);
        r
    }
}

impl SubAssign<GFb127> for GFb127 {
    #[inline(always)]
    fn sub_assign(&mut self, other: GFb127) {
        self.set_add(&other);
    }
}

impl SubAssign<&GFb127> for GFb127 {
    #[inline(always)]
    fn sub_assign(&mut self, other: &GFb127) {
        self.set_add(other);
    }
}

// ========================================================================

/// Element of GF(2^254), defined over GF(2^127)\[u\] with modulus 1 + u + u^2.
#[derive(Clone, Copy, Debug)]
pub struct GFb254([GFb127; 2]);

impl GFb254 {

    pub const ZERO: Self = Self([ GFb127::ZERO, GFb127::ZERO ]);
    pub const ONE: Self = Self([ GFb127::ONE, GFb127::ZERO ]);
    pub const U: Self = Self([ GFb127::ZERO, GFb127::ONE ]);

    pub const fn w64le(x0: u64, x1: u64, x2: u64, x3: u64) -> Self {
        Self([ GFb127::w64le(x0, x1), GFb127::w64le(x2, x3) ])
    }

    pub const fn b127(x0: GFb127, x1: GFb127) -> Self {
        Self([ x0, x1 ])
    }

    pub fn from_b127(x0: GFb127, x1: GFb127) -> Self {
        Self([ x0, x1 ])
    }

    // Get x0 and x1 (both in GFb127) such that self = x0 + x1*u
    #[inline(always)]
    pub fn to_components(self) -> (GFb127, GFb127) {
        (self.0[0], self.0[1])
    }

    #[inline(always)]
    fn set_add(&mut self, rhs: &Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
    }

    // Subtraction is the same thing as addition in binary fields.

    #[inline(always)]
    pub fn set_cond(&mut self, a: &Self, ctl: u32) {
        self.0[0].set_cond(&a.0[0], ctl);
        self.0[1].set_cond(&a.0[1], ctl);
    }

    #[inline(always)]
    pub fn select(a0: &Self, a1: &Self, ctl: u32) -> Self {
        let mut r = *a0;
        r.set_cond(a1, ctl);
        r
    }

    #[inline(always)]
    pub fn cswap(a: &mut Self, b: &mut Self, ctl: u32) {
        GFb127::cswap(&mut a.0[0], &mut b.0[0], ctl);
        GFb127::cswap(&mut a.0[1], &mut b.0[1], ctl);
    }

    #[inline]
    fn set_mul(&mut self, rhs: &Self) {
        // (a0 + a1*u)*(b0 + b1*u)
        //  = a0*b0 + (a0*b1 + a1*b0)*u + a1*b1*(u + 1)
        //  = (a0*b0 + a1*b1) + u*((a0 + a1)*(b0 + b1) + a0*b0)
        let (a0, a1) = (self.0[0], self.0[1]);
        let (b0, b1) = (rhs.0[0], rhs.0[1]);
        let a0b0 = a0 * b0;
        let a1b1 = a1 * b1;
        self.0[0] = a0b0 + a1b1;
        self.0[1] = (a0 + a1) * (b0 + b1) + a0b0;
    }

    // Multiply this value by an element in GF(2^127).
    #[inline]
    pub fn set_mul_b127(&mut self, rhs: &GFb127) {
        self.0[0] *= rhs;
        self.0[1] *= rhs;
    }

    // Multiply this value by an element in GF(2^127).
    #[inline]
    pub fn mul_b127(self, rhs: &GFb127) -> Self {
        Self([ self.0[0] * rhs, self.0[1] * rhs ])
    }

    // Multiply this value by u.
    #[inline(always)]
    pub fn set_mul_u(&mut self) {
        // (a0 + a1*u)*u = a1 + (a0 + a1)*u
        let (a0, a1) = (self.0[0], self.0[1]);
        self.0[0] = a1;
        self.0[1] = a0 + a1;
    }

    // Multiply this value by u.
    #[inline(always)]
    pub fn mul_u(self) -> Self {
        let mut x = self;
        x.set_mul_u();
        x
    }

    // Multiply this value by u + 1.
    #[inline(always)]
    pub fn set_mul_u1(&mut self) {
        // (a0 + a1*u)*(u + 1) = (a0 + a1) + a0*u
        let (a0, a1) = (self.0[0], self.0[1]);
        self.0[0] = a0 + a1;
        self.0[1] = a0;
    }

    // Multiply this value by u + 1.
    #[inline(always)]
    pub fn mul_u1(self) -> Self {
        let mut x = self;
        x.set_mul_u1();
        x
    }

    // Multiply this value by phi(self) = self^(2^127). This yields an
    // element of GF(2^127).
    #[inline(always)]
    pub fn mul_selfphi(self) -> GFb127 {
        let (x0, x1) = (self.0[0], self.0[1]);
        (x0 + x1).square() + x0 * x1
    }

    // Multiply this value by sb = 1 + z^27 (an element of GF(2^127)).
    #[inline(always)]
    pub fn set_mul_sb(&mut self) {
        self.0[0].set_mul_sb();
        self.0[1].set_mul_sb();
    }

    // Multiply this value by sb = 1 + z^27 (an element of GF(2^127)).
    #[inline(always)]
    pub fn mul_sb(self) -> Self {
        Self([ self.0[0].mul_sb(), self.0[1].mul_sb() ])
    }

    // Multiply this value by b = 1 + z^54 (an element of GF(2^127)).
    #[inline(always)]
    pub fn set_mul_b(&mut self) {
        self.0[0].set_mul_b();
        self.0[1].set_mul_b();
    }

    // Multiply this value by b = 1 + z^54 (an element of GF(2^127)).
    #[inline(always)]
    pub fn mul_b(self) -> Self {
        Self([ self.0[0].mul_b(), self.0[1].mul_b() ])
    }

    // Multiply this value by bb = 1 + z^108 (an element of GF(2^127)).
    #[inline(always)]
    pub fn set_mul_bb(&mut self) {
        self.0[0].set_mul_bb();
        self.0[1].set_mul_bb();
    }

    // Multiply this value by bb = 1 + z^108 (an element of GF(2^127)).
    #[inline(always)]
    pub fn mul_bb(self) -> Self {
        Self([ self.0[0].mul_bb(), self.0[1].mul_bb() ])
    }

    // Divide this value by z.
    #[inline(always)]
    pub fn set_div_z(&mut self) {
        self.0[0].set_div_z();
        self.0[1].set_div_z();
    }

    // Divide this value by z.
    #[inline(always)]
    pub fn div_z(self) -> Self {
        Self([ self.0[0].div_z(), self.0[1].div_z() ])
    }

    // Divide this value by z^2.
    #[inline(always)]
    pub fn set_div_z2(&mut self) {
        self.0[0].set_div_z2();
        self.0[1].set_div_z2();
    }

    // Divide this value by z^2.
    #[inline(always)]
    pub fn div_z2(self) -> Self {
        Self([ self.0[0].div_z2(), self.0[1].div_z2() ])
    }

    // Square this value (in place).
    #[inline(always)]
    pub fn set_square(&mut self) {
        // (a0 + a1*u)^2 = a0^2 + (u + 1)*a1^2
        //               = (a0^2 + a1^2) + u*a1^2
        let (a0, a1) = (self.0[0], self.0[1]);
        let t = a1.square();
        self.0[0] = a0.square() + t;
        self.0[1] = t;
    }

    // Square this value.
    #[inline(always)]
    pub fn square(self) -> Self {
        let mut r = self;
        r.set_square();
        r
    }

    // Square this value n times (in place).
    // Note: for large values of n, this can be done more efficiently with
    // a precomputed table, since squaring is linear.
    #[inline(always)]
    fn set_xsquare(&mut self, n: u32) {
        for _ in 0..n {
            self.set_square();
        }
    }

    // Square this value n times.
    // Note: for large values of n, this can be done more efficiently with
    // a precomputed table, since squaring is linear.
    #[inline(always)]
    pub fn xsquare(self, n: u32) -> Self {
        let mut r = self;
        r.set_xsquare(n);
        r
    }

    /// Invert this value; if this value is zero, then it stays at zero.
    pub fn set_invert(&mut self) {
        // We can reduce the inversion to an inversion over GF(2^127):
        //    1/(y0 + u*y1) = (y0 + y1 + u*y1)/(y0^2 + y0*y1 + y1^2)
        // This is equivalent to Itoh-Tsujii, because:
        //    y0 + y1 + u*y1 = (y0 + u*y1)^(2^127)
        // and indeed:
        //    (y0 + y1*u)*(y0 + y1 + u*y1)
        //     = y0^2 + y0*y1 + u*y0*y1 + u*y0*y1 + (u + u^2)*y1^2
        //     = y0^2 + y0*y1 + y1^2
        // Note that y0 + y1 + u*y1 != 0 if y0 + y1*u != 0, and vice-versa.
        let (y0, y1) = (self.0[0], self.0[1]);
        let t = (y0 + y1).square() + (y0 * y1);
        let ti = t.invert();
        self.0[0] = (y0 + y1) * ti;
        self.0[1] = y1 * ti;
    }

    /// Invert this value; if this value is zero, then zero is returned.
    #[inline(always)]
    pub fn invert(self) -> Self {
        let mut x = self;
        x.set_invert();
        x
    }

    #[inline(always)]
    fn set_div(&mut self, y: &Self) {
        self.set_mul(&y.invert());
    }

    // Set this value to its square root. In a binary field, all values
    // have a square root, and it is unique.
    #[inline(always)]
    pub fn set_sqrt(&mut self) {
        // sqrt() is a field automorphism:
        //    sqrt(a0 + u*a1) = sqrt(a0) + sqrt(u)*sqrt(a1)
        // We have u = 1 + u^2 in the field, hence sqrt(u) = u + 1.
        let d0 = self.0[0].sqrt();
        let d1 = self.0[1].sqrt();
        self.0[0] = d0 + d1;
        self.0[1] = d1;
    }

    // Compute the square root of this value. In a binary field, all values
    // have a square root, and it is unique.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        let mut x = self;
        x.set_sqrt();
        x
    }

    // Get the trace for this value (in GF(2^254)). This is 0 or 1.
    #[inline(always)]
    pub fn trace(self) -> u32 {
        // The trace of a0 + a1*u is equal to the trace of a1 in GF(2^127).
        // Indeed:
        //    Tr(a0 + u*a1) = Tr(a0) + Tr(u*a1)
        // We have Tr(a0) = 0, so we can concentrate on Tr(u*a1).
        //    Tr(x) = \sum_{i=0}^{253} x^(2^i)
        // We have:
        //    u^2 = u + 1
        //    u^4 = u^2 + 1 = u
        // Thus:
        //    u^(2^i) = u      if i is even
        //    u^(2^i) = u + 1  if i odd
        // We then get:
        //    Tr(a) = \sum_{i=0}^{253} (u^(2^i))*(a1^(2^i))
        //          =   \sum_{i=0}^{126} (u^(2^(2*i)))*(a1^(2^(2*i)))
        //            + \sum_{i=0}^{126} (u^(2^(2*i+1)))*(a1^(2^(2*i+1)))
        //          =   u*\sum_{i=0}^{126} a1^(2^(2*i))
        //            + (u+1)*\sum_{i=0}^{126} a1^(2^(2*i+1))
        // If we write:
        //    e = \sum_{i=0}^{126} a1^(2^(2*i))
        // then:
        //    Tr(a) = e^2 + u*(e + e^2)
        // Since a1 \in GF(2^127), we have a1^(2^127) = a1. We can write:
        //    e =   \sum_{i=0}^{63} a1^(2^(2*i))
        //        + \sum_{i=64}^{126} a1^(2^(2*i))
        //      =   \sum_{i=0}^{63} a1^(2^(2*i))
        //        + \sum_{i=0}^{62} a1^(2^(2*i+1+127))
        //      =   \sum_{i=0}^{63} a1^(2^(2*i))
        //        + \sum_{i=0}^{62} a1^((2^(2*i+1))*(2^127))
        //      =   \sum_{i=0}^{63} a1^(2^(2*i))
        //        + \sum_{i=0}^{62} (a1^(2^(2*i+1)))^(2^127)
        //      =   \sum_{i=0}^{63} a1^(2^(2*i))
        //        + \sum_{i=0}^{62} a1^(2^(2*i+1))
        //      =   \sum_{i=0}^{126} a1^(2^i)
        //      = Tr(a1)   (trace of a1 in GF(2^127))
        //
        // In total, we get that the trace of a in GF(2^254) is equal to the
        // trace of a1 in GF(2^127).
        self.0[1].trace()
    }

    // For an input a, set this value to a solution x of the equation
    // x^2 + x = a + u*Tr(a). This equation always has exactly two
    // solutions, x and x+1; it is unspecified which of the two equations
    // is returned.
    #[inline]
    pub fn set_qsolve(&mut self) {
        // We write:
        //    x^2 + x = (x0 + x1*u)^2 + x0 + x1*u
        //            = (x0^2 + x1^2 + x0) + (x1^2 + x1)*u
        // Tr(a) = Tr_127(a1), thus we are looking for x1 as a solution
        // of:
        //    x1^2 + x1 = a1 + Tr_127(a1)
        // The halftrace of a1 (in GF(2^127)) is exactly a solution to
        // that equation. This yields two possible values for x1, which
        // are H(a1) and H(a1)+1. For a solution x1, we then need to
        // solve:
        //    x0^2 + x0 = a0 + x1^2
        // That equation has solutions only if Tr_127(a0 + x1^2) = 0;
        // we can thus select the right solution for x1 by adding 1
        // to H(a1) if that value has a trace (over GF(2^127)) distinct
        // from that of a0.
        let (a0, a1) = (self.0[0], self.0[1]);
        let mut x1 = a1.halftrace();
        x1.xor_bit(0, x1.trace() ^ a0.trace());
        let x0 = (a0 + x1.square()).halftrace();
        self.0[0] = x0;
        self.0[1] = x1;
    }

    // Get the halftrace of this value (in GF(2^127)).
    #[inline(always)]
    pub fn qsolve(self) -> Self {
        let mut x = self;
        x.set_qsolve();
        x
    }

    // Equality check between two field elements (constant-time);
    // returned value is 0xFFFFFFFF on equality, 0x00000000 otherwise.
    #[inline(always)]
    pub fn equals(self, rhs: Self) -> u32 {
        (self + rhs).iszero()
    }

    // Compare this value with zero (constant-time); returned value
    // is 0xFFFFFFFF if this element is zero, 0x00000000 otherwise.
    #[inline]
    pub fn iszero(self) -> u32 {
        self.0[0].iszero() & self.0[1].iszero()
    }

    #[inline(always)]
    pub fn encode(self) -> [u8; 32] {
        let mut d = [0u8; 32];
        d[..16].copy_from_slice(&self.0[0].encode());
        d[16..].copy_from_slice(&self.0[1].encode());
        d
    }

    // Decode the value from bytes with implicit reduction modulo
    // z^127 + z^63 + 1 for both components. Input MUST be of length
    // 32 bytes exactly.
    #[allow(dead_code)]
    #[inline]
    fn set_decode32_reduce(&mut self, buf: &[u8]) {
        debug_assert!(buf.len() == 32);
        self.0[0].set_decode16_reduce(&buf[..16]);
        self.0[1].set_decode16_reduce(&buf[16..]);
    }

    // Decode the value from bytes. If the input is invalid (i.e. the
    // input length is not exactly 32 bytes, or the top bit of either
    // component is not zero), then this value is set to zero and 0x00000000
    // is returned. Otherwise, the decoding succeeds, and 0xFFFFFFFF is
    // returned.
    #[inline]
    pub fn set_decode_ct(&mut self, buf: &[u8]) -> u32 {
        if buf.len() != 32 {
            *self = Self::ZERO;
            return 0;
        }
        let r0 = self.0[0].set_decode_ct(&buf[..16]);
        let r1 = self.0[1].set_decode_ct(&buf[16..]);
        let r = r0 & r1;
        self.set_cond(&Self::ZERO, !r);
        r
    }

    // Decode a value from bytes. If the input is invalid (i.e. the
    // input length is not exactly 16 bytes, or the top bit of either
    // component is not zero), then this returns zero and 0x00000000.
    // Otherwise, the decoded value and 0xFFFFFFFF are returned.
    #[inline]
    pub fn decode_ct(buf: &[u8]) -> (Self, u32) {
        let mut x = Self::ZERO;
        let r = x.set_decode_ct(buf);
        (x, r)
    }

    // Decode a value from bytes. If the input is invalid (i.e. the
    // input length is not exactly 16 bytes, or the top bit of either
    // component is not zero), then this returns `None`; otherwise, the
    // decoded value is returned. Side-channel analysis may reveal to
    // outsiders whether the decoding succeeded.
    #[inline]
    pub fn decode(buf: &[u8]) -> Option<Self> {
        let (x, r) = Self::decode_ct(buf);
        if r != 0 {
            Some(x)
        } else {
            None
        }
    }

    // Constant-time table lookup: given a table of 32 field elements, and
    // an index `j` in the 0 to 15 range, return the elements of index
    // `j*2` and `j*2+1`. If `j` is not in the 0 to 15 range (inclusive),
    // then this returns two zeros.
    #[inline]
    pub fn lookup16_x2(tab: &[Self; 32], j: u32) -> [Self; 2] {
        let mut r = [Self::ZERO; 2];
        for i in 0..16 {
            let m = (i as u32) ^ j;
            let m = ((m | m.wrapping_neg()) >> 31).wrapping_sub(1);
            r[0].set_cond(&tab[(i * 2) + 0], m);
            r[1].set_cond(&tab[(i * 2) + 1], m);
        }
        r
    }

    // Constant-time table lookup: given a table of 16 field elements, and
    // an index `j` in the 0 to 7 range, return the elements of index
    // `j*2` and `j*2+1`. If `j` is not in the 0 to 7 range (inclusive),
    // then this returns two zeros.
    #[inline]
    pub fn lookup8_x2(tab: &[Self; 16], j: u32) -> [Self; 2] {
        let mut r = [Self::ZERO; 2];
        for i in 0..8 {
            let m = (i as u32) ^ j;
            let m = ((m | m.wrapping_neg()) >> 31).wrapping_sub(1);
            r[0].set_cond(&tab[(i * 2) + 0], m);
            r[1].set_cond(&tab[(i * 2) + 1], m);
        }
        r
    }

    // Constant-time table lookup: given a table of 8 field elements, and
    // an index `j` in the 0 to 3 range, return the elements of index
    // `j*2` and `j*2+1`. If `j` is not in the 0 to 3 range (inclusive),
    // then this returns two zeros.
    #[inline]
    pub fn lookup4_x2(tab: &[Self; 8], j: u32) -> [Self; 2] {
        let mut r = [Self::ZERO; 2];
        for i in 0..4 {
            let m = (i as u32) ^ j;
            let m = ((m | m.wrapping_neg()) >> 31).wrapping_sub(1);
            r[0].set_cond(&tab[(i * 2) + 0], m);
            r[1].set_cond(&tab[(i * 2) + 1], m);
        }
        r
    }

    /// Constant-time table lookup, short table. This is similar to
    /// `lookup16_x2()`, except that there are only four pairs of values
    /// (8 elements of GF(2^254)), and the pair index MUST be in the
    /// proper range (if the index is not in the range, an unpredictable
    /// value is returned).
    #[inline]
    pub fn lookup4_x2_nocheck(tab: &[Self; 8], j: u32) -> [Self; 2] {
        let mut r = [Self::ZERO; 2];
        for i in 0..4 {
            let m = (i as u32) ^ j;
            let m = ((m | m.wrapping_neg()) >> 31).wrapping_sub(1);
            r[0].set_cond(&tab[(i * 2) + 0], m);
            r[1].set_cond(&tab[(i * 2) + 1], m);
        }
        r
    }
}

// ========================================================================
// Implementations of all the traits needed to use the simple operators
// (+, *, /...) on field element instances, with or without references.

impl Add<GFb254> for GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn add(self, other: GFb254) -> GFb254 {
        let mut r = self;
        r.set_add(&other);
        r
    }
}

impl Add<&GFb254> for GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn add(self, other: &GFb254) -> GFb254 {
        let mut r = self;
        r.set_add(other);
        r
    }
}

impl Add<GFb254> for &GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn add(self, other: GFb254) -> GFb254 {
        let mut r = *self;
        r.set_add(&other);
        r
    }
}

impl Add<&GFb254> for &GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn add(self, other: &GFb254) -> GFb254 {
        let mut r = *self;
        r.set_add(other);
        r
    }
}

impl AddAssign<GFb254> for GFb254 {
    #[inline(always)]
    fn add_assign(&mut self, other: GFb254) {
        self.set_add(&other);
    }
}

impl AddAssign<&GFb254> for GFb254 {
    #[inline(always)]
    fn add_assign(&mut self, other: &GFb254) {
        self.set_add(other);
    }
}

impl Div<GFb254> for GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn div(self, other: GFb254) -> GFb254 {
        let mut r = self;
        r.set_div(&other);
        r
    }
}

impl Div<&GFb254> for GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn div(self, other: &GFb254) -> GFb254 {
        let mut r = self;
        r.set_div(other);
        r
    }
}

impl Div<GFb254> for &GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn div(self, other: GFb254) -> GFb254 {
        let mut r = *self;
        r.set_div(&other);
        r
    }
}

impl Div<&GFb254> for &GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn div(self, other: &GFb254) -> GFb254 {
        let mut r = *self;
        r.set_div(other);
        r
    }
}

impl DivAssign<GFb254> for GFb254 {
    #[inline(always)]
    fn div_assign(&mut self, other: GFb254) {
        self.set_div(&other);
    }
}

impl DivAssign<&GFb254> for GFb254 {
    #[inline(always)]
    fn div_assign(&mut self, other: &GFb254) {
        self.set_div(other);
    }
}

impl Mul<GFb254> for GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn mul(self, other: GFb254) -> GFb254 {
        let mut r = self;
        r.set_mul(&other);
        r
    }
}

impl Mul<&GFb254> for GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn mul(self, other: &GFb254) -> GFb254 {
        let mut r = self;
        r.set_mul(other);
        r
    }
}

impl Mul<GFb254> for &GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn mul(self, other: GFb254) -> GFb254 {
        let mut r = *self;
        r.set_mul(&other);
        r
    }
}

impl Mul<&GFb254> for &GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn mul(self, other: &GFb254) -> GFb254 {
        let mut r = *self;
        r.set_mul(other);
        r
    }
}

impl MulAssign<GFb254> for GFb254 {
    #[inline(always)]
    fn mul_assign(&mut self, other: GFb254) {
        self.set_mul(&other);
    }
}

impl MulAssign<&GFb254> for GFb254 {
    #[inline(always)]
    fn mul_assign(&mut self, other: &GFb254) {
        self.set_mul(other);
    }
}

impl Neg for GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn neg(self) -> GFb254 {
        self
    }
}

impl Neg for &GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn neg(self) -> GFb254 {
        *self
    }
}

impl Sub<GFb254> for GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn sub(self, other: GFb254) -> GFb254 {
        let mut r = self;
        r.set_add(&other);
        r
    }
}

impl Sub<&GFb254> for GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn sub(self, other: &GFb254) -> GFb254 {
        let mut r = self;
        r.set_add(other);
        r
    }
}

impl Sub<GFb254> for &GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn sub(self, other: GFb254) -> GFb254 {
        let mut r = *self;
        r.set_add(&other);
        r
    }
}

impl Sub<&GFb254> for &GFb254 {
    type Output = GFb254;

    #[inline(always)]
    fn sub(self, other: &GFb254) -> GFb254 {
        let mut r = *self;
        r.set_add(other);
        r
    }
}

impl SubAssign<GFb254> for GFb254 {
    #[inline(always)]
    fn sub_assign(&mut self, other: GFb254) {
        self.set_add(&other);
    }
}

impl SubAssign<&GFb254> for GFb254 {
    #[inline(always)]
    fn sub_assign(&mut self, other: &GFb254) {
        self.set_add(other);
    }
}

// ========================================================================

#[cfg(test)]
mod tests {

    use super::{GFb127, GFb254};
    use sha2::{Sha256, Digest};

    /*
    fn print(name: &str, v: GFb127) {
        print!("{} = K(0)", name);
        for i in 0..128 {
            if ((v.0[i >> 5] >> (i & 31)) & 1) != 0 {
                print!(" + z**{}", i);
            }
        }
        println!();
    }
    */

    // va and vb must be 16 bytes each in length
    fn check_gfb127_ops(va: &[u8], vb: &[u8]) {
        let mut a = GFb127::ZERO;
        a.set_decode16_reduce(va);
        let mut b = GFb127::ZERO;
        b.set_decode16_reduce(vb);

        fn norm(v: &[u8]) -> [u8; 16] {
            let mut w = [0u8; 16];
            w[..].copy_from_slice(v);
            let hw = (w[15] >> 7) & 1;
            w[0] ^= hw;
            w[7] ^= hw << 7;
            w[15] ^= hw << 7;
            w
        }

        fn add(wa: &[u8], wb: &[u8]) -> [u8; 16] {
            let mut wc = [0u8; 16];
            for i in 0..16 {
                wc[i] = wa[i] ^ wb[i];
            }
            norm(&wc)
        }

        fn mul(wa: &[u8], wb: &[u8]) -> [u8; 16] {
            let mut zd = [0u8; 32];
            for i in 0..128 {
                for j in 0..128 {
                    let ta = (wa[i >> 3] >> (i & 7)) & 1;
                    let tb = (wb[j >> 3] >> (j & 7)) & 1;
                    zd[(i + j) >> 3] ^= (ta & tb) << ((i + j) & 7);
                }
            }
            for i in (127..256).rev() {
                let td = (zd[i >> 3] >> (i & 7)) & 1;
                zd[i >> 3] ^= td << (i & 7);
                zd[(i - 64) >> 3] ^= td << ((i - 64) & 7);
                zd[(i - 127) >> 3] ^= td << ((i - 127) & 7);
            }
            let mut wc = [0u8; 16];
            wc[..].copy_from_slice(&zd[..16]);
            wc
        }

        let vc = a.encode();
        assert!(vc == norm(va));
        let vc = b.encode();
        assert!(vc == norm(vb));
        let mut bz = true;
        for i in 0..16 {
            if vc[i] != 0 {
                bz = false;
            }
        }

        let c = a + b;
        let vc = c.encode();
        assert!(vc == add(va, vb));

        let c = a - b;
        let vc = c.encode();
        assert!(vc == add(va, vb));

        let c = a.mul_sb();
        let vc = c.encode();
        let mut vx = [0u8; 16];
        vx[0] = 1;
        vx[3] = 8;
        assert!(vc == mul(va, &vx));

        let c = a.mul_b();
        let vc = c.encode();
        let mut vx = [0u8; 16];
        vx[0] = 1;
        vx[6] = 64;
        assert!(vc == mul(va, &vx));

        let c = a.mul_bb();
        let vc = c.encode();
        let mut vx = [0u8; 16];
        vx[0] = 1;
        vx[13] = 16;
        assert!(vc == mul(va, &vx));

        let c = a * b;
        let vc = c.encode();
        assert!(vc == mul(va, vb));

        let c = a.square();
        let vc = c.encode();
        assert!(vc == mul(va, va));

        let c = a / b;
        if bz {
            assert!(b.iszero() == 0xFFFFFFFF);
            assert!(c.iszero() == 0xFFFFFFFF);
        } else {
            assert!(b.iszero() == 0x00000000);
            let d = c * b;
            let vd = d.encode();
            assert!(vd == norm(va));
            assert!(d.equals(a) == 0xFFFFFFFF);
        }

        let c = a.sqrt();
        let d = c.square();
        assert!(d.equals(a) == 0xFFFFFFFF);

        let tra = a.trace();
        assert!(tra == ((norm(va)[0] & 1) as u32));
        let c = a.halftrace();
        let d = c.square() + c;
        if tra == 0 {
            assert!(d.equals(a) == 0xFFFFFFFF);
        } else {
            assert!((d + a + GFb127::ONE).iszero() == 0xFFFFFFFF);
        }
    }

    #[test]
    fn gfb127_ops() {
        let mut va = [0u8; 16];
        let mut vb = [0u8; 16];
        check_gfb127_ops(&va, &vb);
        let mut a = GFb127::ZERO;
        let mut b = GFb127::ZERO;
        a.set_decode16_reduce(&va);
        b.set_decode16_reduce(&vb);
        assert!(a.iszero() == 0xFFFFFFFF);
        assert!(b.iszero() == 0xFFFFFFFF);
        va[0] = 0x01;
        va[7] = 0x80;
        va[15] = 0x80;
        check_gfb127_ops(&va, &vb);
        a.set_decode16_reduce(&va);
        assert!(a.iszero() == 0xFFFFFFFF);
        assert!(a.equals(b) == 0xFFFFFFFF);
        vb[15] = 0x80;
        check_gfb127_ops(&va, &vb);
        b.set_decode16_reduce(&vb);
        assert!(b.iszero() == 0x00000000);
        assert!(a.equals(b) == 0x00000000);

        let mut sh = Sha256::new();
        for i in 0..300 {
            sh.update((i as u64).to_le_bytes());
            let vh = sh.finalize_reset();
            check_gfb127_ops(&vh[0..16], &vh[16..32]);
        }
    }

    // va and vb must be 32 bytes each in length
    fn check_gfb254_ops(va: &[u8], vb: &[u8]) {
        let mut a = GFb254::ZERO;
        a.set_decode32_reduce(va);
        let mut b = GFb254::ZERO;
        b.set_decode32_reduce(vb);

        fn norm(v: &[u8]) -> [u8; 32] {
            let mut w = [0u8; 32];
            w[..].copy_from_slice(v);
            let hw = (w[15] >> 7) & 1;
            w[0] ^= hw;
            w[7] ^= hw << 7;
            w[15] ^= hw << 7;
            let hw = (w[31] >> 7) & 1;
            w[16] ^= hw;
            w[23] ^= hw << 7;
            w[31] ^= hw << 7;
            w
        }

        fn add(wa: &[u8], wb: &[u8]) -> [u8; 32] {
            let mut a0 = GFb127::ZERO;
            let mut a1 = GFb127::ZERO;
            a0.set_decode16_reduce(&wa[..16]);
            a1.set_decode16_reduce(&wa[16..]);
            let mut b0 = GFb127::ZERO;
            let mut b1 = GFb127::ZERO;
            b0.set_decode16_reduce(&wb[..16]);
            b1.set_decode16_reduce(&wb[16..]);
            let mut wc = [0u8; 32];
            wc[..16].copy_from_slice(&(a0 + b0).encode());
            wc[16..].copy_from_slice(&(a1 + b1).encode());
            wc
        }

        fn mul(wa: &[u8], wb: &[u8]) -> [u8; 32] {
            let mut a0 = GFb127::ZERO;
            let mut a1 = GFb127::ZERO;
            a0.set_decode16_reduce(&wa[..16]);
            a1.set_decode16_reduce(&wa[16..]);
            let mut b0 = GFb127::ZERO;
            let mut b1 = GFb127::ZERO;
            b0.set_decode16_reduce(&wb[..16]);
            b1.set_decode16_reduce(&wb[16..]);
            let mut wc = [0u8; 32];
            let c0 = a0 * b0 + a1 * b1;
            let c1 = a0 * b1 + a1 * b0 + a1 * b1;
            wc[..16].copy_from_slice(&c0.encode());
            wc[16..].copy_from_slice(&c1.encode());
            wc
        }

        let vc = a.encode();
        assert!(vc == norm(va));
        let vc = b.encode();
        assert!(vc == norm(vb));
        let mut bz = true;
        for i in 0..32 {
            if vc[i] != 0 {
                bz = false;
            }
        }

        let c = a + b;
        let vc = c.encode();
        assert!(vc == add(va, vb));

        let c = a - b;
        let vc = c.encode();
        assert!(vc == add(va, vb));

        let c = a * b;
        let vc = c.encode();
        assert!(vc == mul(va, vb));

        let c = a.square();
        let vc = c.encode();
        assert!(vc == mul(va, va));

        let c = a / b;
        if bz {
            assert!(b.iszero() == 0xFFFFFFFF);
            assert!(c.iszero() == 0xFFFFFFFF);
        } else {
            assert!(b.iszero() == 0x00000000);
            let d = c * b;
            let vd = d.encode();
            assert!(vd == norm(va));
            assert!(d.equals(a) == 0xFFFFFFFF);
        }

        let c = a.sqrt();
        let d = c.square();
        assert!(d.equals(a) == 0xFFFFFFFF);

        let tra = a.trace();
        assert!(tra == ((norm(va)[16] & 1) as u32));
        let c = a.qsolve();
        let d = c.square() + c;
        if tra == 0 {
            assert!(d.equals(a) == 0xFFFFFFFF);
        } else {
            assert!((d + a + GFb254::U).iszero() == 0xFFFFFFFF);
        }

        let c = a.div_z();
        let d = a / GFb254::w64le(2, 0, 0, 0);
        assert!(c.equals(d) == 0xFFFFFFFF);

        let c = a.div_z2();
        let d = a / GFb254::w64le(4, 0, 0, 0);
        assert!(c.equals(d) == 0xFFFFFFFF);
    }

    #[test]
    fn gfb254_ops() {
        let mut va = [0u8; 32];
        let mut vb = [0u8; 32];
        check_gfb254_ops(&va, &vb);
        let mut a = GFb254::ZERO;
        let mut b = GFb254::ZERO;
        a.set_decode32_reduce(&va);
        b.set_decode32_reduce(&vb);
        assert!(a.iszero() == 0xFFFFFFFF);
        assert!(b.iszero() == 0xFFFFFFFF);
        va[16] = 1;
        check_gfb254_ops(&va, &vb);
        a.set_decode32_reduce(&va);
        assert!(a.iszero() == 0x00000000);
        assert!(a.equals(b) == 0x00000000);
        vb[23] = 0x80;
        vb[31] = 0x80;
        check_gfb254_ops(&va, &vb);
        b.set_decode32_reduce(&vb);
        assert!(b.iszero() == 0x00000000);
        assert!(a.equals(b) == 0xFFFFFFFF);

        let mut sh = Sha256::new();
        for i in 0..300 {
            sh.update(((2 * i + 0) as u64).to_le_bytes());
            let va = sh.finalize_reset();
            sh.update(((2 * i + 1) as u64).to_le_bytes());
            let vb = sh.finalize_reset();
            check_gfb254_ops(&va, &vb);
        }
    }
}
