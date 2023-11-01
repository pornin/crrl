use core::convert::TryFrom;

use super::{addcarry_u64, subborrow_u64, umull, umull_add, umull_add2, sgnw};

/// A custom 128-bit integer with some constant-time operations.
#[derive(Clone, Copy, Debug)]
pub struct Zu128([u64; 2]);

impl Zu128 {

    pub const ZERO: Self = Self([0; 2]);

    #[inline(always)]
    pub const fn w64le(x0: u64, x1: u64) -> Self {
        Self([ x0, x1 ])
    }

    #[inline(always)]
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() != 16 {
            None
        } else {
            let mut x = Self::ZERO;
            for i in 0..2 {
                x.0[i] = u64::from_le_bytes(*<&[u8; 8]>::try_from(
                    &buf[(8 * i)..(8 * i + 8)]).unwrap());
            }
            Some(x)
        }
    }

    #[inline(always)]
    pub fn mul128x128(self, b: &Self) -> Zu256 {
        let (a0, a1) = (self.0[0], self.0[1]);
        let (b0, b1) = (b.0[0], b.0[1]);
        let mut d = [0u64; 4];
        let mut hi;
        (d[0], hi) = umull(a0, b0);
        (d[1], d[2]) = umull_add(a1, b0, hi);
        (d[1], hi) = umull_add(a0, b1, d[1]);
        (d[2], d[3]) = umull_add2(a1, b1, d[2], hi);
        Zu256(d)
    }

    #[inline(always)]
    pub fn mul128x128trunc(self, b: &Self) -> Self {
        let (a0, a1) = (self.0[0], self.0[1]);
        let (b0, b1) = (b.0[0], b.0[1]);
        let (d0, hi) = umull(a0, b0);
        let d1 = a0.wrapping_mul(b1)
            .wrapping_add(a1.wrapping_mul(b0))
            .wrapping_add(hi);
        Self([ d0, d1 ])
    }

    /// Interpreting this value as a signed 128-bit integer, return its
    /// absolute value (in a `u128` type) and the original sign (0xFFFFFFFF
    /// for negative, 0x00000000 for non-negative).
    #[inline(always)]
    pub fn abs(self) -> (u128, u32) {
        let (a0, a1) = (self.0[0], self.0[1]);
        let s = sgnw(a1);
        let (d0, cc) = subborrow_u64(a0 ^ s, s, 0);
        let (d1, _)  = subborrow_u64(a1 ^ s, s, cc);
        ((d0 as u128) | ((d1 as u128) << 64), s as u32)
    }

    /// Interpreting this value as a signed 128-bit integer `x`, return
    /// the absolute value of `2*x+1` (as a `u128` type) and the original
    /// sign (0xFFFFFFFF for negative, 0x00000000 for non-negative).
    #[inline(always)]
    pub fn double_inc_abs(self) -> (u128, u32) {
        let (a0, a1) = (self.0[0], self.0[1]);
        let s = sgnw(a1);
        let b0 = (a0 << 1) | 1;
        let b1 = (a0 >> 63) | (a1 << 1);
        let (d0, cc) = subborrow_u64(b0 ^ s, s, 0);
        let (d1, _)  = subborrow_u64(b1 ^ s, s, cc);
        ((d0 as u128) | ((d1 as u128) << 64), s as u32)
    }

    #[inline(always)]
    pub fn set_sub(&mut self, b: &Self) {
        let cc;
        (self.0[0], cc) = subborrow_u64(self.0[0], b.0[0], 0);
        (self.0[1], _)  = subborrow_u64(self.0[1], b.0[1], cc);
    }

    #[inline(always)]
    pub fn set_sub_u32(&mut self, b: u32) {
        let cc;
        (self.0[0], cc) = subborrow_u64(self.0[0], b as u64, 0);
        (self.0[1], _)  = subborrow_u64(self.0[1], 0, cc);
    }
}

/// A custom 256-bit integer with some constant-time operations.
#[derive(Clone, Copy, Debug)]
pub struct Zu256([u64; 4]);

impl Zu256 {

    pub const ZERO: Self = Self([0; 4]);

    #[inline(always)]
    pub const fn w64le(x0: u64, x1: u64, x2: u64, x3: u64) -> Self {
        Self([ x0, x1, x2, x3 ])
    }

    #[inline(always)]
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() != 32 {
            None
        } else {
            let mut x = Self::ZERO;
            for i in 0..4 {
                x.0[i] = u64::from_le_bytes(*<&[u8; 8]>::try_from(
                    &buf[(8 * i)..(8 * i + 8)]).unwrap());
            }
            Some(x)
        }
    }

    #[inline(always)]
    pub fn trunc128(self) -> Zu128 {
        Zu128([ self.0[0], self.0[1] ])
    }

    #[inline(always)]
    pub fn mul256x128(self, b: &Zu128) -> Zu384 {
        let (a0, a1, a2, a3) = (self.0[0], self.0[1], self.0[2], self.0[3]);
        let (b0, b1) = (b.0[0], b.0[1]);
        let mut d = [0u64; 6];
        let mut hi;
        (d[0], hi) = umull(a0, b0);
        (d[1], hi) = umull_add(a1, b0, hi);
        (d[2], hi) = umull_add(a2, b0, hi);
        (d[3], d[4]) = umull_add(a3, b0, hi);
        (d[1], hi) = umull_add(a0, b1, d[1]);
        (d[2], hi) = umull_add2(a1, b1, d[2], hi);
        (d[3], hi) = umull_add2(a2, b1, d[3], hi);
        (d[4], d[5]) = umull_add2(a3, b1, d[4], hi);
        Zu384(d)
    }

    /// Return `floor((self + b)/2^224) mod 2^32` (i.e. addition truncated
    /// to 256 bits, then return the high 32 bits of the 256-bit result).
    #[inline(always)]
    pub fn add_rsh224(self, b: &Self) -> u32 {
        let mut cc;
        (_, cc) = addcarry_u64(self.0[0], b.0[0], 0);
        for i in 1..3 {
            (_, cc) = addcarry_u64(self.0[i], b.0[i], cc);
        }
        let (w, _) = addcarry_u64(self.0[3], b.0[3], cc);
        (w >> 32) as u32
    }

    /// Return the borrow resulting from the subtraction of `b` from `self`;
    /// returned value is 1 in case of borrow, 0 otherwise. The subtraction
    /// result itself is discarded.
    #[inline(always)]
    pub fn borrow(self, b: &Self) -> u32 {
        let mut cc;
        (_, cc) = subborrow_u64(self.0[0], b.0[0], 0);
        for i in 1..4 {
            (_, cc) = subborrow_u64(self.0[i], b.0[i], cc);
        }
        cc as u32
    }
}

/// A custom 384-bit integer with some constant-time operations.
#[derive(Clone, Copy, Debug)]
pub struct Zu384([u64; 6]);

impl Zu384 {

    pub const ZERO: Self = Self([0; 6]);

    #[inline(always)]
    pub const fn w64le(x0: u64, x1: u64, x2: u64, x3: u64, x4: u64, x5: u64)
        -> Self
    {
        Self([ x0, x1, x2, x3, x4, x5 ])
    }

    #[inline(always)]
    pub fn set_add(&mut self, b: &Self) {
        let mut cc = 0;
        for i in 0..6 {
            (self.0[i], cc) = addcarry_u64(self.0[i], b.0[i], cc);
        }
    }

    /// Returns `self mod 2^n` and `(floor(self/2^n) + cc) mod 2^128`.
    /// Shift count `n` MUST be between 225 and 255 (inclusive).
    #[inline(always)]
    pub fn trunc_and_rsh_cc(&mut self, b: u32, n: u32) -> (Zu256, Zu128) {
        let n1 = n - 192;
        let n2 = 64 - n1;
        let (d0, cc) = addcarry_u64(
            (self.0[3] >> n1) | (self.0[4] << n2), b as u64, 0);
        let (d1, _)  = addcarry_u64(
            (self.0[4] >> n1) | (self.0[5] << n2), 0, cc);
        let c0 = self.0[0];
        let c1 = self.0[1];
        let c2 = self.0[2];
        let c3 = self.0[3] & ((!0u64) >> n2);
        (Zu256([ c0, c1, c2, c3 ]), Zu128([ d0, d1 ]))
    }
}
