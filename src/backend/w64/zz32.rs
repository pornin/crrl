// This file is used for the Zu* types (used in splitting scalars for some
// GLV and GLS curves) on architectures where 32x32->64 multiplications are
// constant-time, but 64x64->128 multiplications are not (e.g. the ARM
// Cortex-A55).

use core::convert::TryFrom;

use super::util32::{addcarry_u32, subborrow_u32, umull, umull_add, umull_add2, sgnw};

/// A custom 128-bit integer with some constant-time operations.
#[derive(Clone, Copy, Debug)]
pub struct Zu128([u32; 4]);

impl Zu128 {

    pub const ZERO: Self = Self([0; 4]);

    #[inline(always)]
    pub const fn w64le(x0: u64, x1: u64) -> Self {
        Self([ x0 as u32, (x0 >> 32) as u32, x1 as u32, (x1 >> 32) as u32 ])
    }

    #[inline(always)]
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() != 16 {
            None
        } else {
            let mut x = Self::ZERO;
            for i in 0..4 {
                x.0[i] = u32::from_le_bytes(*<&[u8; 4]>::try_from(
                    &buf[(4 * i)..(4 * i + 4)]).unwrap());
            }
            Some(x)
        }
    }

    #[inline(always)]
    pub fn mul128x128(self, b: &Self) -> Zu256 {
        let mut d = [0u32; 8];
        for i in 0..4 {
            let f = self.0[i];
            let mut hi = 0;
            for j in 0..4 {
                (d[i + j], hi) = umull_add2(f, b.0[j], d[i + j], hi);
            }
            d[i + 4] = hi;
        }
        Zu256(d)
    }

    #[inline(always)]
    pub fn mul128x128trunc(self, b: &Self) -> Self {
        let f = self.0[0];
        let (d0, hi) = umull(f, b.0[0]);
        let (d1, hi) = umull_add(f, b.0[1], hi);
        let (d2, hi) = umull_add(f, b.0[2], hi);
        let d3 = f.wrapping_mul(b.0[3]).wrapping_add(hi);
        let f = self.0[1];
        let (d1, hi) = umull_add(f, b.0[0], d1);
        let (d2, hi) = umull_add2(f, b.0[1], d2, hi);
        let d3 = f.wrapping_mul(b.0[2]).wrapping_add(d3).wrapping_add(hi);
        let f = self.0[2];
        let (d2, hi) = umull_add(f, b.0[0], d2);
        let d3 = f.wrapping_mul(b.0[1]).wrapping_add(d3).wrapping_add(hi);
        let f = self.0[3];
        let d3 = f.wrapping_mul(b.0[0]).wrapping_add(d3);
        Self([ d0, d1, d2, d3 ])
    }

    /// Interpreting this value as a signed 128-bit integer, return its
    /// absolute value (in a `u128` type) and the original sign (0xFFFFFFFF
    /// for negative, 0x00000000 for non-negative).
    #[inline(always)]
    pub fn abs(self) -> (u128, u32) {
        let s = sgnw(self.0[3]);
        let (d0, cc) = subborrow_u32(self.0[0] ^ s, s, 0);
        let (d1, cc) = subborrow_u32(self.0[1] ^ s, s, cc);
        let (d2, cc) = subborrow_u32(self.0[2] ^ s, s, cc);
        let (d3, _)  = subborrow_u32(self.0[3] ^ s, s, cc);
        ((d0 as u128) | ((d1 as u128) << 32)
         | ((d2 as u128) << 64) | ((d3 as u128) << 96), s)
    }

    /// Interpreting this value as a signed 128-bit integer `x`, return
    /// the absolute value of `2*x+1` (as a `u128` type) and the original
    /// sign (0xFFFFFFFF for negative, 0x00000000 for non-negative).
    #[inline(always)]
    pub fn double_inc_abs(self) -> (u128, u32) {
        let s = sgnw(self.0[3]);
        let b0 = (self.0[0] << 1) | 1;
        let b1 = (self.0[0] >> 31) | (self.0[1] << 1);
        let b2 = (self.0[1] >> 31) | (self.0[2] << 1);
        let b3 = (self.0[2] >> 31) | (self.0[3] << 1);
        let (d0, cc) = subborrow_u32(b0 ^ s, s, 0);
        let (d1, cc) = subborrow_u32(b1 ^ s, s, cc);
        let (d2, cc) = subborrow_u32(b2 ^ s, s, cc);
        let (d3, _)  = subborrow_u32(b3 ^ s, s, cc);
        ((d0 as u128) | ((d1 as u128) << 32)
         | ((d2 as u128) << 64) | ((d3 as u128) << 96), s)
    }

    #[inline(always)]
    pub fn set_sub(&mut self, b: &Self) {
        let mut cc = 0;
        for i in 0..4 {
            (self.0[i], cc) = subborrow_u32(self.0[i], b.0[i], cc);
        }
    }

    #[inline(always)]
    pub fn set_sub_u32(&mut self, b: u32) {
        let mut cc;
        (self.0[0], cc) = subborrow_u32(self.0[0], b, 0);
        for i in 1..4 {
            (self.0[i], cc) = subborrow_u32(self.0[i], 0, cc);
        }
    }
}

/// A custom 256-bit integer with some constant-time operations.
#[derive(Clone, Copy, Debug)]
pub struct Zu256([u32; 8]);

impl Zu256 {

    pub const ZERO: Self = Self([0; 8]);

    #[inline(always)]
    pub const fn w64le(x0: u64, x1: u64, x2: u64, x3: u64) -> Self {
        Self([
            x0 as u32, (x0 >> 32) as u32,
            x1 as u32, (x1 >> 32) as u32,
            x2 as u32, (x2 >> 32) as u32,
            x3 as u32, (x3 >> 32) as u32,
        ])
    }

    #[inline(always)]
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() != 32 {
            None
        } else {
            let mut x = Self::ZERO;
            for i in 0..8 {
                x.0[i] = u32::from_le_bytes(*<&[u8; 4]>::try_from(
                    &buf[(4 * i)..(4 * i + 4)]).unwrap());
            }
            Some(x)
        }
    }

    #[inline(always)]
    pub fn trunc128(self) -> Zu128 {
        Zu128([ self.0[0], self.0[1], self.0[2], self.0[3] ])
    }

    #[inline(always)]
    pub fn mul256x128(self, b: &Zu128) -> Zu384 {
        let mut d = [0u32; 12];
        for i in 0..8 {
            let f = self.0[i];
            let mut hi = 0;
            for j in 0..4 {
                (d[i + j], hi) = umull_add2(f, b.0[j], d[i + j], hi);
            }
            d[i + 4] = hi;
        }
        Zu384(d)
    }

    /// Return `floor((self + b)/2^224) mod 2^32` (i.e. addition truncated
    /// to 256 bits, then return the high 32 bits of the 256-bit result).
    #[inline(always)]
    pub fn add_rsh224(self, b: &Self) -> u32 {
        let mut cc;
        (_, cc) = addcarry_u32(self.0[0], b.0[0], 0);
        for i in 1..7 {
            (_, cc) = addcarry_u32(self.0[i], b.0[i], cc);
        }
        let (w, _) = addcarry_u32(self.0[7], b.0[7], cc);
        w
    }

    /// Return the borrow resulting from the subtraction of `b` from `self`;
    /// returned value is 1 in case of borrow, 0 otherwise. The subtraction
    /// result itself is discarded.
    #[inline(always)]
    pub fn borrow(self, b: &Self) -> u32 {
        let mut cc;
        (_, cc) = subborrow_u32(self.0[0], b.0[0], 0);
        for i in 1..8 {
            (_, cc) = subborrow_u32(self.0[i], b.0[i], cc);
        }
        cc as u32
    }
}

/// A custom 384-bit integer with some constant-time operations.
#[derive(Clone, Copy, Debug)]
pub struct Zu384([u32; 12]);

impl Zu384 {

    pub const ZERO: Self = Self([0; 12]);

    #[inline(always)]
    pub const fn w64le(x0: u64, x1: u64, x2: u64, x3: u64, x4: u64, x5: u64)
        -> Self
    {
        Self([
            x0 as u32, (x0 >> 32) as u32,
            x1 as u32, (x1 >> 32) as u32,
            x2 as u32, (x2 >> 32) as u32,
            x3 as u32, (x3 >> 32) as u32,
            x4 as u32, (x4 >> 32) as u32,
            x5 as u32, (x5 >> 32) as u32,
        ])
    }

    #[inline(always)]
    pub fn set_add(&mut self, b: &Self) {
        let mut cc = 0;
        for i in 0..12 {
            (self.0[i], cc) = addcarry_u32(self.0[i], b.0[i], cc);
        }
    }

    /// Returns `self mod 2^n` and `(floor(self/2^n) + cc) mod 2^128`.
    /// Shift count `n` MUST be between 225 and 255 (inclusive).
    #[inline(always)]
    pub fn trunc_and_rsh_cc(&mut self, b: u32, n: u32) -> (Zu256, Zu128) {
        let n1 = n - 224;
        let n2 = 32 - n1;
        let (d0, cc) = addcarry_u32(
            (self.0[7] >> n1) | (self.0[8] << n2), b, 0);
        let (d1, cc) = addcarry_u32(
            (self.0[8] >> n1) | (self.0[9] << n2), 0, cc);
        let (d2, cc) = addcarry_u32(
            (self.0[9] >> n1) | (self.0[10] << n2), 0, cc);
        let (d3, _)  = addcarry_u32(
            (self.0[10] >> n1) | (self.0[11] << n2), 0, cc);
        let mut e = [0u32; 8];
        e[..].copy_from_slice(&self.0[..8]);
        e[7] &= (!0u32) >> n2;
        (Zu256(e), Zu128([ d0, d1, d2, d3 ]))
    }
}
