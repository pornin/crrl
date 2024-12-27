use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::convert::TryFrom;

use super::{umull, sgnw, lzcnt};
use super::lagrange::lagrange253_vartime;

#[derive(Clone, Copy, Debug)]
pub struct GF255<const MQ: u64>([u64; 5]);

/// Special container for "not reduced" values returned by `add_noreduce()`
/// and `sub_noreduce()`; these values can only be used as operands in
/// multiplications and squarings.
#[derive(Clone, Copy, Debug)]
pub struct GF255NotReduced<const MQ: u64>([u64; 5]);

// 2^51 - 1
const M51: u64 = 0x0007FFFFFFFFFFFF;

impl<const MQ: u64> GF255<MQ> {

    // Parameter restrictions:
    //   MQ is odd
    //   MQ <= 32765
    //   q = 2^255 - MQ is prime
    // Moreover, if MQ == 7 mod 8 (i.e. q = 1 mod 8), then square root
    // computations are not implemented.
    //
    // Primality cannot easily be tested at compile-time, but we check
    // the other properties.
    //
    // Restrictions on MQ come from set_sqrt_ext() and the range/reduction
    // analysis for multiplications and squarings.
    #[allow(dead_code)]
    const COMPILE_TIME_CHECKS: () = Self::compile_time_checks();
    const fn compile_time_checks() {
        static_assert!((MQ & 1) != 0);
        static_assert!(MQ <= 32767);
    }

    // INTERNAL FORMAT
    // ===============
    //
    // Value is split over five unsigned limbs, in base 2^51. If the limbs
    // are y0 to y4, then the value is:
    //   y0 + y1*2^51 + y2*2^102 + y3*2^153 + y4*2^204
    // The value is implicitly considered modulo q = 2^255 - MQ.
    //
    // There are two defined ranges for limb values:
    //  - In GF255 instances, limbs have value less than n*2^51, for some
    //    real value n > 1. We normally keep n < 1.3; in fact, we can
    //    enforce use of n = 1.07.
    //  - In GF255NotReduced instances, limbs have value less than m*2^51,
    //    for some other real value m > n.
    //
    // "Not reduced" values are obtain through additions and subtractions of
    // up to four field elements. In all generality, this requires that
    // m >= 3*n + ceil(n). Multiplications and squarings accept input
    // operands which are not reduced, while all other operations expect
    // reduced operands. All operations produce reduced outputs, except
    // the functions whose name includes "noreduce"; the "not reduced"
    // values use a different Rust type so that they can be used only
    // with multiplications and squarings.
    //
    // For multiplications (and squarings), we compute intermediate 115-bit
    // products:
    //     c_{i,j} = a_i * b_j
    // which we split into low and high parts:
    //     d_{i,j} = c_{i,j} mod 2^51
    //     h_{i,j} = floor(c_{i,j} / 2^51)
    // To perform the split efficiently on RISC-V systems (which do not
    // have a combined shift opcode such as the shld opcode on x86), we
    // pre-shift the operands (by 6 bits for a, 7 bits for b), so that the
    // split falls on a register boundary.
    //
    // We then add together the d_{i,j}:
    //     d0 = d_{0,0}
    //     d1 = d_{0,1} + d_{1,0}
    //     d2 = d_{0,2} + d_{1,1} + d_{2,0}
    //     d3 = d_{0,3} + d_{1,2} + d_{2,1} + d_{3,0}
    //     d4 = d_{0,4} + d_{1,3} + d_{2,2} + d_{3,1} + d_{4,0}
    //     d5 = d_{1,4} + d_{2,3} + d_{3,2} + d_{4,1}
    //     d6 = d_{2,4} + d_{3,3} + d_{4,2}
    //     d7 = d_{3,4} + d_{4,3}
    //     d8 = d_{4,4}
    // and similarly for h_{i,j} (into h0..h8).
    //
    // The first reduction step computes:
    //     e0 = d0 + MQ * (h4 + d5);
    //     e1 = d1 + h0 + MQ * (h5 + d6);
    //     e2 = d2 + h1 + MQ * (h6 + d7);
    //     e3 = d3 + h2 + MQ * (h7 + d8);
    //     e4 = d4 + h3 + MQ * h8;
    // which yields 5 limbs than can then be further reduced with some
    // simple propagation (function set_carry_propagate()).
    //
    // Ideally, all operations are performed on 64-bit integers (with
    // products c_{i,j} yielding two 64-bit values with umull()). This
    // has some requirements on the maximum limb value (m*2^51):
    //  - Shifts on a_i and b_j must not lose some bits; since b_j values
    //    are shifted by 7 bits, this implies m <= 64.
    //  - Values d0..d8 cannot overflow:
    //     max(d0) = 2^51 - 1
    //     max(d1) = (2^51 - 1) * 2
    //     max(d2) = (2^51 - 1) * 3
    //     max(d3) = (2^51 - 1) * 4
    //     max(d4) = (2^51 - 1) * 5
    //     max(d5) = (2^51 - 1) * 4
    //     max(d6) = (2^51 - 1) * 3
    //     max(d7) = (2^51 - 1) * 2
    //     max(d8) = 2^51 - 1
    //  - Maximum values for d0..d8 are:
    //     max(h0) = m^2 * 2^51
    //     max(h1) = m^2 * 2^51 * 2
    //     max(h2) = m^2 * 2^51 * 3
    //     max(h3) = m^2 * 2^51 * 4
    //     max(h4) = m^2 * 2^51 * 5
    //     max(h5) = m^2 * 2^51 * 4
    //     max(h6) = m^2 * 2^51 * 3
    //     max(h7) = m^2 * 2^51 * 2
    //     max(h8) = m^2 * 2^51
    //    Largest h* value is h4; for that value to fit in 64 bits, we need
    //    5*m^2 <= 2^13, hence m <= sqrt(2^13/5) =~ 40.48.
    //  - Maximum e* values are:
    //     max(e0) = (4*MQ + 1)*(2^51 - 1) + (5*MQ)*m^2*2^51
    //     max(e1) = (3*MQ + 2)*(2^51 - 1) + (4*MQ + 1)*m^2*2^51
    //     max(e2) = (2*MQ + 3)*(2^51 - 1) + (3*MQ + 2)*m^2*2^51
    //     max(e3) = (MQ + 4)*(2^51 - 1) + (2*MQ + 3)*m^2*2^51
    //     max(e4) = 5*(2^51 - 1) + (MQ + 4)*m^2*2^51
    //
    // Ideally, all e* values fit in 64 bits; for the smallest possible
    // MQ (such that 2^255 - MQ is prime), this yields the following limit
    // on m:
    //    MQ     max(m)
    //    19    9.24235...
    //    31    7.21423...
    //   475    1.62752...
    //   735    1.19534...
    //   765    1.15820...
    // For the next possible MQ (921), we get m < 0.98930... which is not
    // tenable (we need m >= 1, otherwise not all field element values
    // can be represented). Since we need m >= 3*n + ceil(n), which is
    // always greater than 5, this means that we can keep the computations
    // to 64-bit types only if MQ = 19 or 31.
    //
    // For larger MQ values, the multiplications must be done with a
    // 128-bit result, i.e. umull() calls. We again apply a 13-bit shift,
    // i.e. multiply by MQ*2^13 (note that MQ < 2^15), so that the
    // split of the result in low and high parts remains simple. In that
    // case, the computations are as follows:
    //     f0 = MQ*(h4 + d5)
    //     f1 = MQ*(h5 + d6)
    //     f2 = MQ*(h6 + d7)
    //     f3 = MQ*(h7 + d8)
    //     f4 = MQ*h8
    //     e0 = d0 + (f0 mod 2^51) + MQ*floor(f4 / 2^51)
    //     e1 = d1 + h0 + (f1 mod 2^51) + floor(f0 / 2^51)
    //     e2 = d2 + h1 + (f2 mod 2^51) + floor(f1 / 2^51)
    //     e3 = d3 + h2 + (f3 mod 2^51) + floor(f2 / 2^51)
    //     e4 = d4 + h3 + (f4 mod 2^51) + floor(f3 / 2^51)
    // We assume that h4 + d5, h5 + d6, h6 + d7, h7 + d8 and h8 all fit
    // on 64 bits. Note that floor(f_i / 2^51) < MQ*2^13 < 2^28, and
    // MQ*floor(f4 / 2^51) < 2^43. Thus, no overflow occurs as long as
    //     d5 + h4 < 2^64
    //     d6 + h5 < 2^64
    //     d7 + h6 < 2^64
    //     d8 + h7 < 2^64
    //     h8 < 2^64
    //     d1 + h0 < 2^64 - 2^51
    //     d2 + h1 < 2^64 - 2^51
    //     d3 + h2 < 2^64 - 2^51
    //     d4 + h3 < 2^64 - 2^51
    // Apply the expressions of the maximum values of d* and h* (as
    // described previously), we get that the critical value here is
    // d5 + h4, which may go up to 4*(2^51 - 1) + 5*m^2*2^51, thus
    // implying an upper limit of:
    //     m <= 40.46727...
    //
    // In total, we get the following acceptable limb limits:
    //
    //    GF255<MQ>               1.07*2^51
    //    GF255NotReduced<MQ>     9.24*2^51 if MQ = 19
    //                            7.21*2^51 if MQ = 31
    //                            40.46*2^51 if MQ > 31

    // Element encoding length (in bytes); always 32 bytes.
    pub const ENC_LEN: usize = 32;

    // Modulus is q = 2^255 - T255_MINUS_Q.
    pub const T255_MINUS_Q: u32 = MQ as u32;

    // Modulus q in base 2^64 (low-to-high order).
    pub const MODULUS: [u64; 4] = [
        MQ.wrapping_neg(),
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x7FFFFFFFFFFFFFFF
    ];

    pub const ZERO: GF255<MQ> = GF255::<MQ>([ 0, 0, 0, 0, 0 ]);
    pub const ONE: GF255<MQ> = GF255::<MQ>([ 1, 0, 0, 0, 0 ]);
    pub const MINUS_ONE: GF255<MQ> = GF255::<MQ>([
        0x0007FFFFFFFFFFFF - MQ,
        0x0007FFFFFFFFFFFF,
        0x0007FFFFFFFFFFFF,
        0x0007FFFFFFFFFFFF,
        0x0007FFFFFFFFFFFF,
    ]);

    // Modulus q, over 51-bit limbs.
    const MOD_M51: [u64; 5] = [
        0x0008000000000000 - MQ,
        0x0007FFFFFFFFFFFF,
        0x0007FFFFFFFFFFFF,
        0x0007FFFFFFFFFFFF,
        0x0007FFFFFFFFFFFF,
    ];

    // 2*q, with limb values up to 52 bits.
    const DMOD_M51: [u64; 5] = [
        (0x0008000000000000 - MQ) << 1,
        0x0007FFFFFFFFFFFF << 1,
        0x0007FFFFFFFFFFFF << 1,
        0x0007FFFFFFFFFFFF << 1,
        0x0007FFFFFFFFFFFF << 1,
    ];

    // 9*q, with limb values up to 55 bits.
    const NMOD_M51: [u64; 5] = [
        (0x0008000000000000 - MQ) * 9,
        0x0007FFFFFFFFFFFF * 9,
        0x0007FFFFFFFFFFFF * 9,
        0x0007FFFFFFFFFFFF * 9,
        0x0007FFFFFFFFFFFF * 9,
    ];

    // 1/2^508 in the field, as a constant; this is used when computing
    // divisions in the field. The value is computed at compile-time.
    const INVT508: GF255<MQ> = GF255::<MQ>::make_invt508();

    // Create an element from a 256-bit value (implicitly reduced modulo
    // the field order) provided as four 64-bit limbs (in low-to-high order).
    pub const fn w64le(x0: u64, x1: u64, x2: u64, x3: u64) -> Self {
        // We wrap around the top bit; y0 may be up to 2^51 - 1 + MQ
        // (which is less than 1.00000000002*2^51, thus fine), while the
        // five other limbs fit on 51 bits.
        let y0 = (x0 & M51) + (MQ & sgnw(x3));
        let y1 = (x0 >> 51) | ((x1 << 13) & M51);
        let y2 = (x1 >> 38) | ((x2 << 26) & M51);
        let y3 = (x2 >> 25) | ((x3 << 39) & M51);
        let y4 = (x3 >> 12) & M51;
        Self([ y0, y1, y2, y3, y4 ])
    }

    // Create an element from a 256-bit value (implicitly reduced modulo
    // the field order) provided as four 64-bit limbs (in high-to-low order).
    pub const fn w64be(x3: u64, x2: u64, x1: u64, x0: u64) -> Self {
        Self::w64le(x0, x1, x2, x3)
    }

    // Create an element from a 256-bit value (implicitly reduced modulo
    // the field order) provided as four 64-bit limbs (in low-to-high order).
    pub fn from_w64le(x0: u64, x1: u64, x2: u64, x3: u64) -> Self {
        Self::w64le(x0, x1, x2, x3)
    }

    // Create an element from a 256-bit value (implicitly reduced modulo
    // the field order) provided as four 64-bit limbs (in high-to-low order).
    pub fn from_w64be(x3: u64, x2: u64, x1: u64, x0: u64) -> Self {
        Self::w64le(x0, x1, x2, x3)
    }

    // Create an element by converting the provided integer.
    // If the source value is negative, then it is implicitly reduced
    // modulo the ring order.
    #[inline(always)]
    pub fn from_i32(x: i32) -> Self {
        // Max limb 0: 2^51 + 2^31 - 1
        Self([
            ((x as i64) + (Self::MOD_M51[0] as i64)) as u64,
            Self::MOD_M51[1],
            Self::MOD_M51[2],
            Self::MOD_M51[3],
            Self::MOD_M51[4],
        ])
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
        // We add q to ensure a non-negative integer.
        let y0 = Self::MOD_M51[0] + ((x as u64) & M51);
        let y1 = (((x >> 51) + (Self::MOD_M51[1] as i64)) as u64) + (y0 >> 51);
        Self([
            y0 & M51,
            y1,
            Self::MOD_M51[2],
            Self::MOD_M51[3],
            Self::MOD_M51[4],
        ])
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
        let y0 = Self::MOD_M51[0] + ((x as u64) & M51);
        let y1 = Self::MOD_M51[1] + (((x >> 51) as u64) & M51) + (y0 >> 51);
        let y2 = ((((x >> 102) as i64) + (Self::MOD_M51[2] as i64)) as u64)
            + (y1 >> 51);
        Self([
            y0 & M51,
            y1 & M51,
            y2,
            Self::MOD_M51[3],
            Self::MOD_M51[4],
        ])
    }

    // Create an element by converting the provided integer.
    #[inline(always)]
    pub fn from_u128(x: u128) -> Self {
        Self::from_w64le(x as u64, (x >> 64) as u64, 0, 0)
    }

    // Set this value to the provided limbs, with additional carry
    // propagation.
    #[inline(always)]
    fn set_carry_propagate(&mut self,
        d0: u64, d1: u64, d2: u64, d3: u64, d4: u64)
    {
        // Max output limb value: 2^51 - 1 + MQ*(2^13 - 1) < 1.00000012*2^51
        let h0 = d0 >> 51;
        let h1 = d1 >> 51;
        let h2 = d2 >> 51;
        let h3 = d3 >> 51;
        let h4 = d4 >> 51;
        self.0[0] = (d0 & M51) + (h4 * MQ);
        self.0[1] = (d1 & M51) + h0;
        self.0[2] = (d2 & M51) + h1;
        self.0[3] = (d3 & M51) + h2;
        self.0[4] = (d4 & M51) + h3;
    }

    #[inline]
    fn set_add(&mut self, rhs: &Self) {
        let d0 = self.0[0] + rhs.0[0];
        let d1 = self.0[1] + rhs.0[1];
        let d2 = self.0[2] + rhs.0[2];
        let d3 = self.0[3] + rhs.0[3];
        let d4 = self.0[4] + rhs.0[4];
        self.set_carry_propagate(d0, d1, d2, d3, d4);
    }

    #[inline]
    fn set_sub(&mut self, rhs: &Self) {
        // We add 2*q, with limbs close to 2*2^51, to avoid negative values.
        let d0 = (self.0[0] + Self::DMOD_M51[0]) - rhs.0[0];
        let d1 = (self.0[1] + Self::DMOD_M51[1]) - rhs.0[1];
        let d2 = (self.0[2] + Self::DMOD_M51[2]) - rhs.0[2];
        let d3 = (self.0[3] + Self::DMOD_M51[3]) - rhs.0[3];
        let d4 = (self.0[4] + Self::DMOD_M51[4]) - rhs.0[4];
        self.set_carry_propagate(d0, d1, d2, d3, d4);
    }

    // Negate this value (in place).
    #[inline]
    pub fn set_neg(&mut self) {
        // We add 2*q, with limbs close to 2*2^51, to avoid negative values.
        let d0 = Self::DMOD_M51[0] - self.0[0];
        let d1 = Self::DMOD_M51[1] - self.0[1];
        let d2 = Self::DMOD_M51[2] - self.0[2];
        let d3 = Self::DMOD_M51[3] - self.0[3];
        let d4 = Self::DMOD_M51[4] - self.0[4];
        self.set_carry_propagate(d0, d1, d2, d3, d4);
    }

    /// Return self + rhs (no reduction).
    #[inline(always)]
    pub fn add_noreduce(self, rhs: &Self) -> GF255NotReduced<MQ> {
        GF255NotReduced::<MQ>([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4],
        ])
    }

    /// Return 2*self (no reduction).
    #[inline(always)]
    pub fn mul2_noreduce(self) -> GF255NotReduced<MQ> {
        GF255NotReduced::<MQ>([
            self.0[0] << 1,
            self.0[1] << 1,
            self.0[2] << 1,
            self.0[3] << 1,
            self.0[4] << 1,
        ])
    }

    /// Return self - rhs (no reduction).
    #[inline(always)]
    pub fn sub_noreduce(self, rhs: &Self) -> GF255NotReduced<MQ> {
        GF255NotReduced::<MQ>([
            (self.0[0] + Self::DMOD_M51[0]) - rhs.0[0],
            (self.0[1] + Self::DMOD_M51[1]) - rhs.0[1],
            (self.0[2] + Self::DMOD_M51[2]) - rhs.0[2],
            (self.0[3] + Self::DMOD_M51[3]) - rhs.0[3],
            (self.0[4] + Self::DMOD_M51[4]) - rhs.0[4],
        ])
    }

    /// Return 2*self + b and 2*self - b (no reduction).
    #[inline(always)]
    pub fn mul2add_mul2sub_noreduce(self, b: &Self)
        -> (GF255NotReduced<MQ>, GF255NotReduced<MQ>)
    {
        let d0 = self.0[0] << 1;
        let d1 = self.0[1] << 1;
        let d2 = self.0[2] << 1;
        let d3 = self.0[3] << 1;
        let d4 = self.0[4] << 1;
        let e0 = d0 + b.0[0];
        let e1 = d1 + b.0[1];
        let e2 = d2 + b.0[2];
        let e3 = d3 + b.0[3];
        let e4 = d4 + b.0[4];
        let f0 = (d0 + Self::DMOD_M51[0]) - b.0[0];
        let f1 = (d1 + Self::DMOD_M51[1]) - b.0[1];
        let f2 = (d2 + Self::DMOD_M51[2]) - b.0[2];
        let f3 = (d3 + Self::DMOD_M51[3]) - b.0[3];
        let f4 = (d4 + Self::DMOD_M51[4]) - b.0[4];
        (GF255NotReduced::<MQ>([ e0, e1, e2, e3, e4 ]),
         GF255NotReduced::<MQ>([ f0, f1, f2, f3, f4 ]))
    }

    /// Return self + b and self + b - c (no reduction).
    #[inline(always)]
    pub fn add_addsub_noreduce(self, b: &Self, c: &Self)
        -> (GF255NotReduced<MQ>, GF255NotReduced<MQ>)
    {
        let d0 = self.0[0] + b.0[0];
        let d1 = self.0[1] + b.0[1];
        let d2 = self.0[2] + b.0[2];
        let d3 = self.0[3] + b.0[3];
        let d4 = self.0[4] + b.0[4];
        let e0 = (d0 + Self::DMOD_M51[0]) - c.0[0];
        let e1 = (d1 + Self::DMOD_M51[1]) - c.0[1];
        let e2 = (d2 + Self::DMOD_M51[2]) - c.0[2];
        let e3 = (d3 + Self::DMOD_M51[3]) - c.0[3];
        let e4 = (d4 + Self::DMOD_M51[4]) - c.0[4];
        (GF255NotReduced::<MQ>([ d0, d1, d2, d3, d4 ]),
         GF255NotReduced::<MQ>([ e0, e1, e2, e3, e4 ]))
    }

    /// Return self - b and self - b + 2*c (no reduction).
    #[inline(always)]
    pub fn sub_subadd2_noreduce(self, b: &Self, c: &Self)
        -> (GF255NotReduced<MQ>, GF255NotReduced<MQ>)
    {
        let d0 = (self.0[0] + Self::DMOD_M51[0]) - b.0[0];
        let d1 = (self.0[1] + Self::DMOD_M51[1]) - b.0[1];
        let d2 = (self.0[2] + Self::DMOD_M51[2]) - b.0[2];
        let d3 = (self.0[3] + Self::DMOD_M51[3]) - b.0[3];
        let d4 = (self.0[4] + Self::DMOD_M51[4]) - b.0[4];
        let e0 = d0 + (c.0[0] << 1);
        let e1 = d1 + (c.0[1] << 1);
        let e2 = d2 + (c.0[2] << 1);
        let e3 = d3 + (c.0[3] << 1);
        let e4 = d4 + (c.0[4] << 1);
        (GF255NotReduced::<MQ>([ d0, d1, d2, d3, d4 ]),
         GF255NotReduced::<MQ>([ e0, e1, e2, e3, e4 ]))
    }

    /// For inputs self and rhs, compute self + 8*rhs and self - 8*rhs,
    /// both values being returned in the special "not reduced" format.
    #[inline(always)]
    pub fn add8_sub8_noreduce(self, rhs: &Self)
        -> (GF255NotReduced<MQ>, GF255NotReduced<MQ>)
    {
        if MQ <= 31 {
            let b = rhs.mul8();
            (self.add_noreduce(&b), self.sub_noreduce(&b))
        } else {
            let b0 = rhs.0[0] << 3;
            let b1 = rhs.0[1] << 3;
            let b2 = rhs.0[2] << 3;
            let b3 = rhs.0[3] << 3;
            let b4 = rhs.0[4] << 3;
            let d0 = self.0[0] + b0;
            let d1 = self.0[1] + b1;
            let d2 = self.0[2] + b2;
            let d3 = self.0[3] + b3;
            let d4 = self.0[4] + b4;
            let e0 = (self.0[0] + Self::NMOD_M51[0]) - b0;
            let e1 = (self.0[1] + Self::NMOD_M51[1]) - b1;
            let e2 = (self.0[2] + Self::NMOD_M51[2]) - b2;
            let e3 = (self.0[3] + Self::NMOD_M51[3]) - b3;
            let e4 = (self.0[4] + Self::NMOD_M51[4]) - b4;
            (GF255NotReduced::<MQ>([ d0, d1, d2, d3, d4 ]),
             GF255NotReduced::<MQ>([ e0, e1, e2, e3, e4 ]))
        }
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
    }

    #[inline]
    fn set_half(&mut self) {
        // 1. Right-shift by 1 bit; keep dropped bit in 'tt' (expanded)
        let d0 = (self.0[0] >> 1) + ((self.0[1] & 1) << 50);
        let d1 = (self.0[1] >> 1) + ((self.0[2] & 1) << 50);
        let d2 = (self.0[2] >> 1) + ((self.0[3] & 1) << 50);
        let d3 = (self.0[3] >> 1) + ((self.0[4] & 1) << 50);
        let d4 = self.0[4] >> 1;
        let tt = (self.0[0] & 1).wrapping_neg();

        // 2. If the dropped bit was 1, add back (q+1)/2.
        let d0 = d0 + (tt & (0x0008000000000000 - (MQ >> 1)));
        let d1 = d1 + (tt & 0x0007FFFFFFFFFFFF);
        let d2 = d2 + (tt & 0x0007FFFFFFFFFFFF);
        let d3 = d3 + (tt & 0x0007FFFFFFFFFFFF);
        let d4 = d4 + (tt & 0x0003FFFFFFFFFFFF);

        // 3. Reduce.
        self.set_carry_propagate(d0, d1, d2, d3, d4);
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
        let d0 = self.0[0] << 1;
        let d1 = self.0[1] << 1;
        let d2 = self.0[2] << 1;
        let d3 = self.0[3] << 1;
        let d4 = self.0[4] << 1;
        self.set_carry_propagate(d0, d1, d2, d3, d4);
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
        let d0 = self.0[0] * 3;
        let d1 = self.0[1] * 3;
        let d2 = self.0[2] * 3;
        let d3 = self.0[3] * 3;
        let d4 = self.0[4] * 3;
        self.set_carry_propagate(d0, d1, d2, d3, d4);
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
        let d0 = self.0[0] << 2;
        let d1 = self.0[1] << 2;
        let d2 = self.0[2] << 2;
        let d3 = self.0[3] << 2;
        let d4 = self.0[4] << 2;
        self.set_carry_propagate(d0, d1, d2, d3, d4);
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
        let d0 = self.0[0] << 3;
        let d1 = self.0[1] << 3;
        let d2 = self.0[2] << 3;
        let d3 = self.0[3] << 3;
        let d4 = self.0[4] << 3;
        self.set_carry_propagate(d0, d1, d2, d3, d4);
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
        let d0 = self.0[0] << 4;
        let d1 = self.0[1] << 4;
        let d2 = self.0[2] << 4;
        let d3 = self.0[3] << 4;
        let d4 = self.0[4] << 4;
        self.set_carry_propagate(d0, d1, d2, d3, d4);
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
        let d0 = self.0[0] << 5;
        let d1 = self.0[1] << 5;
        let d2 = self.0[2] << 5;
        let d3 = self.0[3] << 5;
        let d4 = self.0[4] << 5;
        self.set_carry_propagate(d0, d1, d2, d3, d4);
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
        // Input limbs are up to 1.07*2^51; if 1.07*x < 2^13, then we can
        // perform the multiplications over 64 bits.
        if x <= 7656 {
            let x = x as u64;
            let d0 = self.0[0] * x;
            let d1 = self.0[1] * x;
            let d2 = self.0[2] * x;
            let d3 = self.0[3] * x;
            let d4 = self.0[4] * x;
            self.set_carry_propagate(d0, d1, d2, d3, d4);
            return;
        }

        // Multiply limbs. Each high part is lower than x, which is lower
        // than 2^32; thus, MQ*h4 < (2^15)*(2^32) = 2^47.
        // We have: 2^51 + 2^47 = 1.0625*2^51 < 1.07*2^51
        let xs = (x as u64) << 13;
        let (d0, h0) = umull(self.0[0], xs);
        let (d1, h1) = umull(self.0[1], xs);
        let (d2, h2) = umull(self.0[2], xs);
        let (d3, h3) = umull(self.0[3], xs);
        let (d4, h4) = umull(self.0[4], xs);
        self.0[0] = (d0 >> 13) + (MQ * h4);
        self.0[1] = (d1 >> 13) + h0;
        self.0[2] = (d2 >> 13) + h1;
        self.0[3] = (d3 >> 13) + h2;
        self.0[4] = (d4 >> 13) + h3;
    }

    #[inline(always)]
    pub fn mul_small(self, x: u32) -> Self {
        let mut r = self;
        r.set_mul_small(x);
        r
    }

    #[inline(always)]
    fn set_mul(&mut self, rhs: &Self) {
        let (a0, a1, a2, a3, a4) =
            (self.0[0], self.0[1], self.0[2], self.0[3], self.0[4]);
        let (b0, b1, b2, b3, b4) =
            (rhs.0[0], rhs.0[1], rhs.0[2], rhs.0[3], rhs.0[4]);

        // See comments at the start for range analysis.
        let a0 = a0 << 6;
        let a1 = a1 << 6;
        let a2 = a2 << 6;
        let a3 = a3 << 6;
        let a4 = a4 << 6;
        let b0 = b0 << 7;
        let b1 = b1 << 7;
        let b2 = b2 << 7;
        let b3 = b3 << 7;
        let b4 = b4 << 7;

        let (c00, h00) = umull(a0, b0);
        let (c01, h01) = umull(a0, b1);
        let (c02, h02) = umull(a0, b2);
        let (c03, h03) = umull(a0, b3);
        let (c04, h04) = umull(a0, b4);
        let (c10, h10) = umull(a1, b0);
        let (c11, h11) = umull(a1, b1);
        let (c12, h12) = umull(a1, b2);
        let (c13, h13) = umull(a1, b3);
        let (c14, h14) = umull(a1, b4);
        let (c20, h20) = umull(a2, b0);
        let (c21, h21) = umull(a2, b1);
        let (c22, h22) = umull(a2, b2);
        let (c23, h23) = umull(a2, b3);
        let (c24, h24) = umull(a2, b4);
        let (c30, h30) = umull(a3, b0);
        let (c31, h31) = umull(a3, b1);
        let (c32, h32) = umull(a3, b2);
        let (c33, h33) = umull(a3, b3);
        let (c34, h34) = umull(a3, b4);
        let (c40, h40) = umull(a4, b0);
        let (c41, h41) = umull(a4, b1);
        let (c42, h42) = umull(a4, b2);
        let (c43, h43) = umull(a4, b3);
        let (c44, h44) = umull(a4, b4);

        let d0 = c00 >> 13;
        let d1 = (c01 >> 13)
               + (c10 >> 13);
        let d2 = (c02 >> 13)
               + (c11 >> 13)
               + (c20 >> 13);
        let d3 = (c03 >> 13)
               + (c12 >> 13)
               + (c21 >> 13)
               + (c30 >> 13);
        let d4 = (c04 >> 13)
               + (c13 >> 13)
               + (c22 >> 13)
               + (c31 >> 13)
               + (c40 >> 13);
        let d5 = (c14 >> 13)
               + (c23 >> 13)
               + (c32 >> 13)
               + (c41 >> 13);
        let d6 = (c24 >> 13)
               + (c33 >> 13)
               + (c42 >> 13);
        let d7 = (c34 >> 13)
               + (c43 >> 13);
        let d8 = c44 >> 13;

        let h0 = h00;
        let h1 = h01 + h10;
        let h2 = h02 + h11 + h20;
        let h3 = h03 + h12 + h21 + h30;
        let h4 = h04 + h13 + h22 + h31 + h40;
        let h5 = h14 + h23 + h32 + h41;
        let h6 = h24 + h33 + h42;
        let h7 = h34 + h43;
        let h8 = h44;

        if MQ <= 31 {
            let e0 = d0 + MQ * (h4 + d5);
            let e1 = d1 + h0 + MQ * (h5 + d6);
            let e2 = d2 + h1 + MQ * (h6 + d7);
            let e3 = d3 + h2 + MQ * (h7 + d8);
            let e4 = d4 + h3 + MQ * h8;
            self.set_carry_propagate(e0, e1, e2, e3, e4);
        } else {
            let (f0, g0) = umull(h4 + d5, MQ << 13);
            let (f1, g1) = umull(h5 + d6, MQ << 13);
            let (f2, g2) = umull(h6 + d7, MQ << 13);
            let (f3, g3) = umull(h7 + d8, MQ << 13);
            let (f4, g4) = umull(h8, MQ << 13);
            let e0 = d0 + (f0 >> 13) + MQ * g4;
            let e1 = d1 + h0 + (f1 >> 13) + g0;
            let e2 = d2 + h1 + (f2 >> 13) + g1;
            let e3 = d3 + h2 + (f3 >> 13) + g2;
            let e4 = d4 + h3 + (f4 >> 13) + g3;
            self.set_carry_propagate(e0, e1, e2, e3, e4);
        }
    }

    // Square this value (in place).
    #[inline(always)]
    pub fn set_square(&mut self) {
        let (a0, a1, a2, a3, a4) =
            (self.0[0], self.0[1], self.0[2], self.0[3], self.0[4]);

        // Similar to set_mul(), but we merge double-products togther
        // by using the appropriate operand words (the z* words are twice
        // the corresponding s* words).

        let s0 = a0 << 6;
        let s1 = a1 << 6;
        let s2 = a2 << 6;
        let s3 = a3 << 6;
        let s4 = a4 << 6;
        let z0 = a0 << 7;
        let z1 = a1 << 7;
        let z2 = a2 << 7;
        let z3 = a3 << 7;
        let z4 = a4 << 7;

        let (c00, h00) = umull(s0, z0);
        let (c01, h01) = umull(z0, z1);
        let (c02, h02) = umull(z0, z2);
        let (c03, h03) = umull(z0, z3);
        let (c04, h04) = umull(z0, z4);
        let (c11, h11) = umull(s1, z1);
        let (c12, h12) = umull(z1, z2);
        let (c13, h13) = umull(z1, z3);
        let (c14, h14) = umull(z1, z4);
        let (c22, h22) = umull(s2, z2);
        let (c23, h23) = umull(z2, z3);
        let (c24, h24) = umull(z2, z4);
        let (c33, h33) = umull(s3, z3);
        let (c34, h34) = umull(z3, z4);
        let (c44, h44) = umull(s4, z4);

        let d0 = c00 >> 13;
        let d1 = c01 >> 13;
        let d2 = (c02 >> 13)
               + (c11 >> 13);
        let d3 = (c03 >> 13)
               + (c12 >> 13);
        let d4 = (c04 >> 13)
               + (c13 >> 13)
               + (c22 >> 13);
        let d5 = (c14 >> 13)
               + (c23 >> 13);
        let d6 = (c24 >> 13)
               + (c33 >> 13);
        let d7 = c34 >> 13;
        let d8 = c44 >> 13;

        let h0 = h00;
        let h1 = h01;
        let h2 = h02 + h11;
        let h3 = h03 + h12;
        let h4 = h04 + h13 + h22;
        let h5 = h14 + h23;
        let h6 = h24 + h33;
        let h7 = h34;
        let h8 = h44;

        if MQ <= 31 {
            let e0 = d0 + MQ * (h4 + d5);
            let e1 = d1 + h0 + MQ * (h5 + d6);
            let e2 = d2 + h1 + MQ * (h6 + d7);
            let e3 = d3 + h2 + MQ * (h7 + d8);
            let e4 = d4 + h3 + MQ * h8;
            self.set_carry_propagate(e0, e1, e2, e3, e4);
        } else {
            let (f0, g0) = umull(h4 + d5, MQ << 13);
            let (f1, g1) = umull(h5 + d6, MQ << 13);
            let (f2, g2) = umull(h6 + d7, MQ << 13);
            let (f3, g3) = umull(h7 + d8, MQ << 13);
            let (f4, g4) = umull(h8, MQ << 13);
            let e0 = d0 + (f0 >> 13) + MQ * g4;
            let e1 = d1 + h0 + (f1 >> 13) + g0;
            let e2 = d2 + h1 + (f2 >> 13) + g1;
            let e3 = d3 + h2 + (f3 >> 13) + g2;
            let e4 = d4 + h3 + (f4 >> 13) + g3;
            self.set_carry_propagate(e0, e1, e2, e3, e4);
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

    // Fully reduce the value.
    #[inline]
    fn set_normalized(&mut self) {
        // Add MQ, and propagate carries.
        let a0 = self.0[0] + MQ;
        let a1 = self.0[1] + (a0 >> 51);
        let a2 = self.0[2] + (a1 >> 51);
        let a3 = self.0[3] + (a2 >> 51);
        let a4 = self.0[4] + (a3 >> 51);
        let b0 = (a0 & M51) + MQ * (a4 >> 51);
        // MQ*(a4 >> 51) < MQ*2^13 < 2^28, so the carry into b1 can only
        // be 0 or 1.
        let b1 = (a1 & M51) + (b0 >> 51);
        let b2 = (a2 & M51) + (b1 >> 51);
        let b3 = (a3 & M51) + (b2 >> 51);
        let b4 = (a4 & M51) + (b3 >> 51);
        // b4 may exceed 2^51-1 only if b0 produced a carry, in which case
        // b0 % 2^51 must be small, and the carry propagation will stop there.
        let c0 = (b0 & M51) + (MQ & (b4 >> 51).wrapping_neg());
        let c1 = b1 & M51;
        let c2 = b2 & M51;
        let c3 = b3 & M51;
        let c4 = b4 & M51;

        // Subtract MQ; propagate the borrow.
        let d0 = c0.wrapping_sub(MQ);
        let d1 = c1.wrapping_sub(d0 >> 63);
        let d2 = c2.wrapping_sub(d1 >> 63);
        let d3 = c3.wrapping_sub(d2 >> 63);
        let d4 = c4.wrapping_sub(d3 >> 63);

        // If there is a borrow, then we must add back the modulus. In such
        // a case, limb d0 is between -1 and -MQ, and limbs d1..d4 are -1,
        // so the addition of the modulus will simply yield 0x0007FFFFFFFFFFFF
        // for limbs d1 to d4.
        // If there is no borrow, some of the limbs (but not d4) may be
        // "negative" and the truncation to 51 bits is required.
        let w = sgnw(d4);
        let e0 = d0.wrapping_add(w & Self::MOD_M51[0]) & M51;
        let e1 = (d1 | w) & M51;
        let e2 = (d2 | w) & M51;
        let e3 = (d3 | w) & M51;
        let e4 = (d4 | w) & M51;

        self.0[0] = e0;
        self.0[1] = e1;
        self.0[2] = e2;
        self.0[3] = e3;
        self.0[4] = e4;
    }

    // Encode this value into four 64-bit limbs in little-endian order
    // (fully normalized).
    fn to_limbs64(self) -> [u64; 4] {
        let mut x = self;
        x.set_normalized();
        let x0 = x.0[0] | (x.0[1] << 51);
        let x1 = (x.0[1] >> 13) | (x.0[2] << 38);
        let x2 = (x.0[2] >> 26) | (x.0[3] << 25);
        let x3 = (x.0[3] >> 39) | (x.0[4] << 12);
        [x0, x1, x2, x3]
    }

    // Set this value to u*f+v*g (with 'u' being self). Parameters f and g
    // are provided as u64, but they are signed integers in the -2^47..+2^47
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

        // Multipliers can be at most 2^47, so we can left-shift them
        // by 13 bits.
        let f = f << 13;
        let g = g << 13;

        let (c0, h0) = umull(tu.0[0], f);
        let (c1, h1) = umull(tu.0[1], f);
        let (c2, h2) = umull(tu.0[2], f);
        let (c3, h3) = umull(tu.0[3], f);
        let (c4, h4) = umull(tu.0[4], f);
        let (d0, j0) = umull(tv.0[0], g);
        let (d1, j1) = umull(tv.0[1], g);
        let (d2, j2) = umull(tv.0[2], g);
        let (d3, j3) = umull(tv.0[3], g);
        let (d4, j4) = umull(tv.0[4], g);
        // With |f| and |g| at most 2^47, and the original operand limbs
        // at most 1.07*2^51, h4 and j4 cannot be larger than 1.07*2^47
        // each, so that e0 < 2*(2^51 - 1) + 2*2^15*1.07*2^47 < 2^64.
        let e0 = (c0 >> 13) + (d0 >> 13) + MQ * (h4 + j4);
        let e1 = (c1 >> 13) + (d1 >> 13) + (h0 + j0);
        let e2 = (c2 >> 13) + (d2 >> 13) + (h1 + j1);
        let e3 = (c3 >> 13) + (d3 >> 13) + (h2 + j2);
        let e4 = (c4 >> 13) + (d4 >> 13) + (h3 + j3);
        self.set_carry_propagate(e0, e1, e2, e3, e4);
    }

    #[inline(always)]
    fn lin(a: &Self, b: &Self, f: u64, g: u64) -> Self {
        let mut r = Self::ZERO;
        r.set_lin(a, b, f, g);
        r
    }

    // Set this value to abs((a*f+b*g)/2^31). Values a and b are interpreted
    // as unsigned 255-bit integers; limbs must be normalized (less than
    // 2^51 each). Coefficients f and g are provided as u64, but they really
    // are signed integers in the -2^31..+2^31 range (inclusive). The low
    // 31 bits are dropped (i.e. the division is assumed to be exact). The
    // result is assumed to fit in 255 bits.
    //
    // Returned value is -1 (u64) if (a*f+b*g) was negative, 0 otherwise.
    #[inline]
    fn set_lindiv31abs(&mut self, a: &Self, b: &Self, f: u64, g: u64) -> u64 {
        // Replace f and g with abs(f) and abs(g), but remember the
        // original signs.
        let sf = sgnw(f);
        let f = (f ^ sf).wrapping_sub(sf);
        let sf51 = sf & M51;
        let sg = sgnw(g);
        let g = (g ^ sg).wrapping_sub(sg);
        let sg51 = sg & M51;

        // Apply the signs of f and g to the source operands; get sign into
        // a separate variable. The top 13 bits of each produced limb is to
        // be ignored.
        let a0 = (a.0[0] ^ sf51).wrapping_sub(sf);
        let a1 = (a.0[1] ^ sf51).wrapping_add(a0 >> 63);
        let a2 = (a.0[2] ^ sf51).wrapping_add(a1 >> 63);
        let a3 = (a.0[3] ^ sf51).wrapping_add(a2 >> 63);
        let a4 = (a.0[4] ^ sf51).wrapping_add(a3 >> 63);
        let sa = sf.wrapping_add(a4 >> 51);
        let b0 = (b.0[0] ^ sg51).wrapping_sub(sg);
        let b1 = (b.0[1] ^ sg51).wrapping_add(b0 >> 63);
        let b2 = (b.0[2] ^ sg51).wrapping_add(b1 >> 63);
        let b3 = (b.0[3] ^ sg51).wrapping_add(b2 >> 63);
        let b4 = (b.0[4] ^ sg51).wrapping_add(b3 >> 63);
        let sb = sg.wrapping_add(a4 >> 51);

        // Compute a*f+b*g into e0..e4 + high word in e5. The shift of
        // source limbs by 13 is needed to remove spurious bits left in
        // the high word positions; it also helps with splitting each
        // multiplication output into low and high parts.
        let (c0, h0) = umull(a0 << 13, f);
        let (c1, h1) = umull(a1 << 13, f);
        let (c2, h2) = umull(a2 << 13, f);
        let (c3, h3) = umull(a3 << 13, f);
        let (c4, h4) = umull(a4 << 13, f);
        let (d0, j0) = umull(b0 << 13, g);
        let (d1, j1) = umull(b1 << 13, g);
        let (d2, j2) = umull(b2 << 13, g);
        let (d3, j3) = umull(b3 << 13, g);
        let (d4, j4) = umull(b4 << 13, g);
        let mut e0 = (c0 >> 13) + (d0 >> 13);
        let mut e1 = (c1 >> 13) + (d1 >> 13) + h0 + j0;
        let mut e2 = (c2 >> 13) + (d2 >> 13) + h1 + j1;
        let mut e3 = (c3 >> 13) + (d3 >> 13) + h2 + j2;
        let mut e4 = (c4 >> 13) + (d4 >> 13) + h3 + j3;
        let mut e5 = h4 + j4;
        // Some carry propagation.
        e1 += e0 >> 51;
        e0 &= M51;
        e2 += e1 >> 51;
        e1 &= M51;
        e3 += e2 >> 51;
        e2 &= M51;
        e4 += e3 >> 51;
        e3 &= M51;
        e5 += e4 >> 51;
        e4 &= M51;

        // f and g are at most 2^31 each, so the word e5 fits on 32 bits.

        // If a < 0, then the result is overestimated by f*2^255;
        // similarly, if b < 0 then the result is overestimated by g*2^255.
        // We must thus subtract 2^255*(sa*f+sb*g), with sa and sb being
        // the signs of a and b, respectively (1 for negative, 0 otherwise).
        e5 = e5.wrapping_sub(f & sa);
        e5 = e5.wrapping_sub(g & sb);

        // Shift-right the value by 31 bits.
        let v0 = ((e0 >> 31) | (e1 << 20)) & M51;
        let v1 = ((e1 >> 31) | (e2 << 20)) & M51;
        let v2 = ((e2 >> 31) | (e3 << 20)) & M51;
        let v3 = ((e3 >> 31) | (e4 << 20)) & M51;
        let v4 = ((e4 >> 31) | (e5 << 20)) & M51;

        // If the result is negative, then negate it.
        let t = sgnw(e5);
        let sv = t & M51;
        let r0 = (v0 ^ sv).wrapping_add(sv & 1);
        let r1 = (v1 ^ sv).wrapping_add(r0 >> 63);
        let r2 = (v2 ^ sv).wrapping_add(r1 >> 63);
        let r3 = (v3 ^ sv).wrapping_add(r2 >> 63);
        let r4 = (v4 ^ sv).wrapping_add(r3 >> 63);
        self.0[0] = r0 & M51;
        self.0[1] = r1 & M51;
        self.0[2] = r2 & M51;
        self.0[3] = r3 & M51;
        self.0[4] = r4 & M51;
        t
    }

    #[inline(always)]
    fn lindiv31abs(a: &Self, b: &Self, f: u64, g: u64) -> (Self, u64) {
        let mut r = Self::ZERO;
        let ng = r.set_lindiv31abs(a, b, f, g);
        (r, ng)
    }

    fn set_div(&mut self, y: &Self) {
        // We use the optimized binary GCD from:
        //    https://eprint.iacr.org/2020/972
        // with a slight adaptation, in that we use only 63 bits for the
        // "approximate" values, and still use 31 inner iterations. In
        // the paper, registers of size 2*k bits are assumed, with k-1
        // inner iterations; we can show, though, that the algorithm still
        // works with 2*k-1 bits in each register. Following the steps
        // of the proof in annex A of the paper:
        //
        // Annex A.1:
        //
        //   These inequalities still hold (they don't depend on k).
        //      |f0| + |g0| <= 2^t
        //      |f1| + |g1| <= 2^t
        //
        //   Let a' and b' be the approximate values. Their top k bits match
        //   the top k bits of the n-bit values a and b:
        //      a = a_top*2^(n - k) + a_rest      (0 <= a_rest < 2^(n-k))
        //      a' = a'_top*2^(k - 1) + a'_rest   (0 <= a'_rest < 2^(k - 1))
        //   Thus:
        //      a - a'*2^(n - 2*k + 1) = a_rest - a'rest*2^(n - 2*k + 1)
        //      -2^(n-k) < a - a'*2^(n - 2*k + 1) < +2^(n-k)
        //      |a - a'*2^(n - 2*k + 1)| < 2^(n - k)
        //   and similarly:
        //      |b - b'*2^(n - 2*k + 1)| < 2^(n - k)
        //   We get the same expressions as in annex A.1, with a replacement
        //   of n with n+1. This propagates to the limits on a_t and a'_t:
        //      |a_t - a'_t*2^(n - 2*k + 1)| <= 2^(n - k)
        //      |b_t - b'_t*2^(n - 2*k + 1)| <= 2^(n - k)
        //
        // Annex A.2:
        //
        //   With d being the "divergence point", we get that:
        //      |2^(n - 2*k + 1)*(a'_d - b'_d) - (a_d - b_d)|
        //        <= |a'_d*2^(n - 2*k + 1) - a_d| + |b_d - b'd*2^(n - 2*k + 1)|
        //        <= 2^(n + 1 - k)
        //   Since (a'_d - b'_d)*2^(n - 2*k + 1) >= 0 and (a_d - b_d) < 0
        //   (by assumption of d being the divergence point), and their
        //   difference is less than 2^(n + 1 - k), then both must be
        //   lower (in absolute value) than 2^(n + 1 - k):
        //      0 <= (a'_d - b'_d)*2^(n - 2*k + 1) < 2^(n + 1 - k)
        //   Thus:
        //      0 <= a'_d - b'_d < 2^k
        //   and also:
        //      0 > a_d - b_d >= -2^(n + 1 - k)
        //   which implies that |a_{d+1}| <= 2^(n-k). The inequality can
        //   be exact only if a'_d - b'_d = 0, which implies that a'_{d+1} = 0
        //   and all subsequent iterations just divide a_d by 2, with no
        //   further subtraction.
        //
        //   These expressions match the original A.2, still with the n -> n+1
        //   replacement.
        //
        // Annex A.3:
        //
        //   n is the current maximum of the lengths of a and b; we suppose
        //   that n >= 2*k (otherwise, a' and b' are exact values and there
        //   cannot be any divergence). The k-1 inner iterations reach a
        //   divergence point d.
        //   Still following the n -> n+1 replacement, we get that after
        //   the divergence point, no value a_t or b_t greater than 2^(n-k)
        //   (in absolute value) may grow further, and a value lower than
        //   2^(n-k) cannot regrow beyond n-k bits.
        //
        //   Case 1: len(a) = n and len(b) <= n - k
        //     We have b' < 2^(k - 1) and a' >= 2^(2*k - 2). First iterations
        //     do not swap. After t <= k - 2 iterations, minimal value for
        //     a'_t is (a' - (2^t - 1)*b')/2^t. We have:
        //        a' - (2^t - 1)*b' >= 2^(2*k - 2) - (2^t - 1)*(2^(k - 1) - 1)
        //                          > 2^(2*k - 2) - 2^(2*k - 3)
        //                          > 2^(2*k - 3)
        //     Thus, a'_t > 2^(2*k - 3 - t). With t <= k - 2, we have
        //     2*k - 3 - t >= k - 1, thus a'_t > 2^(k - 1) > b'_t. As in
        //     the paper, in this case there can be no swap and no divergence
        //     point.
        //
        //   Case 2: len(a) <= n - k and len(b) = n
        //     Same treatment as in the paper: after possibly a few halvings
        //     of a (and a'), we get a swap, then fall back to the situation
        //     of case 1.
        //
        //   Case 3: len(a) = n and len(b) >= n - k + 1 (or vice versa).
        //     If, at the divergence point, a_d < b_d < 2^(n-k), then
        //     we already have obtained a reduction by at least k+1 bits,
        //     and (as pointed out above) neither value will be allowed
        //     to regrow beyond 2^(n-k), so the total size reduction will
        //     be at least k+1 bits.
        //
        //     Write len(b) = n - k + h, for some integer h >= 1, and,
        //     at the divergence point d:
        //        len(a_d) = n - k + i
        //        len(b_d) = n - k + j
        //     a_d < b_d at the divergence point, so i <= j. We suppose that
        //     we are not in the previous subcase, i.e. j >= 1. Moreover, up
        //     to the divergence point, values have only shrunk, so we know
        //     that one of a_d and b_d, at least, is no greater than b.
        //     Thus, i <= h.
        //
        //     At the divergence point, we know that:
        //        -2^(n - k + 1) <= a_d - b_d < 0
        //     This implies that:
        //        2^(n - k + j - 1) - 2^(n - k + 1) <= a_d < 2^(n - k + i)
        //     Thus:
        //        2^(j - 1) - 2 < 2^i
        //     We may have j = i + 2 only if i = 0; otherwise, j <= i + 1.
        //     Note that h >= 1, so that in all cases we have j <= h + 1.
        //
        //     After the divergence point, |a_{d+1}| <= 2^(n-k). If
        //     |a_{d+1}| < 2^(n-k), then, as described above, a
        //     cannot regrow beyond n-k bits, while b cannot grow while
        //     it is larger than n-k bits, and cannot regrow back above
        //     n-k bits. In total, the subsequent operations cannot make
        //     values go back to more than (n - k + j) + (n - k) bits,
        //     which is at most 2*n + h + 1 - 2*k, while the initial total
        //     length was 2*n - k + h, so the total was reduced by at least
        //     k - 1 bits.
        //
        //     The remaining subcase is when a_{d+1} = -2^(n-k), exactly.
        //     This may happen only if b_d = a_d + 2^(n-k+1); all remaining
        //     iterations will divide a by 2 and leave b untouched. After
        //     iteration d, len(a_{d+1}) = n - k + 1 and
        //     len(b_{d+1}) = n - k + j <= n - k + h + 1, for a total of
        //     at most 2*n - 2*k + h + 2, so a reduction of at least k-2
        //     bits. If iteration d was not the last one
        //     (i.e. d + 1 < k - 1), then a will be further shrunk at least
        //     once (no subtraction, just division by 2), and the reduction
        //     will be at least k-1 bits.
        //
        //     The final subcase to explore is thus when the divergent
        //     point is exactly the last iteration. This case may fail
        //     to bring a total reduction of at least k-1 bits only if
        //     both following (in)equalities hold:
        //        (n - k + i) + (n - k + j) = n + (n - k + h) - (k - 2)
        //        n - k + i <= len(|a_{d+1}|)
        //     Since len(|a_{d+1}|) = n - k + 1, we get that i <= 1.
        //     We established that j <= i + 1, hence j <= 2. The first
        //     equality implies that h = i + j - 2. In all this situation,
        //     h >= 1, so the only possible combination here is i = 1,
        //     j = 2, and h = 1. In that case, using the inequalities
        //     from annex A.1 (adapted above to our situation), we have:
        //        |a_d - a'_d*2^(n - 2*k + 1)| <= 2^(n - k)
        //        |b_d - b'_d*2^(n - 2*k + 1)| <= 2^(n - k)
        //     which implies that:
        //        a'_d*2^(n - 2*k + 1) <= a_d + 2^(n - k)
        //        b'_d*2^(n - 2*k + 1) >= b_d - 2^(n - k)
        //     But we also have:
        //        b_d = a_d + 2^(n - k + 1)
        //     thus:
        //        a_d + 2^(n - k) = b_d - 2^(n - k)
        //     Since a'_d = b'_d in that situation (it is a consequence
        //     of a'_{d+1} = -2^(n - k)), we obtain that:
        //        a'_d*2^(n - 2*k + 1) = a_d + 2^(n - k) = b_d - 2^(n - k)
        //     In all this analysis, we considered that n > 2*k - 1
        //     (otherwise, a' and b' are not approximations, they are exact
        //     values, and we are applying the normal binary GCD). Thus,
        //     2^(n - 2*k + 1) is even, which means that both a_d and b_d
        //     are even, which is not possible (an invariant of the algorithm
        //     is that b is always odd). It is thus not possible that the
        //     divergence point yields a_{d+1} = -2^(n - k) at the last
        //     iteration.
        //
        // We have explored all cases; this proves the overall
        // conclusion: we can extract approximations of a and b over
        // 2*k-1 bits, and run k-1 inner iterations; we still obtain the
        // expected total size reduction of k-1 bits. Here, We use
        // 63-bit approximations and 31 inner iterations; the extra room
        // bit in values allows for easy constant-time comparisons
        // without support for carries.

        let mut a = *y;
        a.set_normalized();
        let mut b = Self(Self::MOD_M51);
        let mut u = *self;
        let mut v = Self::ZERO;

        // Generic loop does 15*31 = 465 inner iterations.
        for _ in 0..15 {
            // Get approximations of a and b over 63 bits:
            //  - If len(a) <= 63 and len(b) <= 63, then we just use
            //    their values.
            //  - Otherwise, with n = max(len(a), len(b)), we use:
            //       (a mod 2^31) + 2^31*floor(a / 2^(n - 32))
            //       (b mod 2^31) + 2^31*floor(b / 2^(n - 32))
            // We must first find the top non-zero limb.
            let (a0, a1, a2, a3, a4) = (a.0[0], a.0[1], a.0[2], a.0[3], a.0[4]);
            let (b0, b1, b2, b3, b4) = (b.0[0], b.0[1], b.0[2], b.0[3], b.0[4]);
            let m4 = a4 | b4;
            let m3 = a3 | b3;
            let m2 = a2 | b2;
            let m1 = a1 | b1;
            let tz4 = sgnw(m4.wrapping_sub(1));
            let tz3 = sgnw(m3.wrapping_sub(1));
            let tz2 = sgnw(m2.wrapping_sub(1));
            let tz1 = sgnw(m1.wrapping_sub(1));
            let cc4 = !tz4;
            let cc3 = tz4 & !tz3;
            let cc2 = tz4 & tz3 & !tz2;
            let cc1 = tz4 & tz3 & tz2;
            let one_limb = tz1 & tz2 & tz3 & tz4;
            let tnz = (cc4 & m4) | (cc3 & m3) | (cc2 & m2) | (cc1 & m1);
            let s = 64 - lzcnt(tnz);
            let ta1 = (cc4 & a4) | (cc3 & a3) | (cc2 & a2) | (cc1 & a1);
            let ta0 = (cc4 & a3) | (cc3 & a2) | (cc2 & a1) | (cc1 & a0);
            let tb1 = (cc4 & b4) | (cc3 & b3) | (cc2 & b2) | (cc1 & b1);
            let tb0 = (cc4 & b3) | (cc3 & b2) | (cc2 & b1) | (cc1 & b0);

            // If len(a) and/or len(b) is at least 52, then:
            //    s contains the length in bits of the top combined limb
            //    1 <= s <= 51
            //    ta1 and ta0 are the two top a limbs to consider
            //    tb1 and tb0 are the two top b limbs to consider
            //    one_limb is zero
            // Otherwise, len(a) <= 51 and len(b) <= 51:
            //    s = 0
            //    ta0, ta1, tb0 and tb1 are all zero
            //    one_limb is -1 (as u64)
            let wa = (ta1 << (51 - s)) | (ta0 >> s);
            let wb = (tb1 << (51 - s)) | (tb0 >> s);
            let xa = ((wa & 0x0007FFFFFFF80000) << 12) | (a0 & 0x7FFFFFFF);
            let xb = ((wb & 0x0007FFFFFFF80000) << 12) | (b0 & 0x7FFFFFFF);
            let mut xa = xa ^ (one_limb & (xa ^ a0));
            let mut xb = xb ^ (one_limb & (xb ^ b0));

            // Compute the 31 inner iterations on xa and xb.
            let mut fg0 = 1u64;
            let mut fg1 = 1u64 << 32;
            for _ in 0..31 {
                let a_odd = (xa & 1).wrapping_neg();
                let swap = a_odd & sgnw(xa.wrapping_sub(xb));
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
            let swap = a_odd & sgnw(xa.wrapping_sub(xb));
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

    // Return self^2, self^3 and self^(2**240 - 1)
    fn pow2_240(self) -> (Self, Self, Self) {
        let x = self;
        let x2 = x.square();
        let x3 = x2 * x;
        let xp4 = x3.xsquare(2) * x3;
        let xp5 = xp4.square() * x;
        let xp15 = (xp5.xsquare(5) * xp5).xsquare(5) * xp5;
        let xp30 = xp15.xsquare(15) * xp15;
        let xp60 = xp30.xsquare(30) * xp30;
        let xp120 = xp60.xsquare(60) * xp60;
        let xp240 = xp120.xsquare(120) * xp120;
        (x2, x3, xp240)
    }

    /*
    /// Invert this value. If this value is 0, then it remains equal to 0.
    pub fn set_invert(&mut self) {
        // TODO: optimize with binary GCD
        let (x2, x3, mut y) = self.pow2_240();
        let win: [Self; 3] = [ *self, x2, x3 ];
        let e = MQ.wrapping_neg() - 2;
        for i in 0..7 {
            y.set_xsquare(2);
            let k = ((e >> (13 - (2 * i))) & 3) as usize;
            if k != 0 {
                y.set_mul(&win[k - 1]);
            }
        }
        y.set_square();
        self.set_mul(&y);
    }

    #[inline(always)]
    pub fn invert(self) -> Self {
        let mut r = self;
        r.set_invert();
        r
    }

    fn set_div(&mut self, y: &Self) {
        self.set_mul(&y.invert());
    }
    */

    pub fn set_invert(&mut self) {
        let r = self.invert();
        *self = r;
    }

    #[inline(always)]
    pub fn invert(self) -> Self {
        let mut r = Self::ONE;
        r.set_div(&self);
        r
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
            let mut k = tt[blen - 1].invert();
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
        // We use the same optimized binary GCD as in set_div(), though
        // without keeping track of the (u,v) values; we also keep a
        // running result value that is adjusted by each swap according
        // to the law of quadratic reciprocity.
        // Since that adjustment requires a look-ahead of 2 bits, we need
        // a special adjustment for the last 2 iterations of the inner
        // loop.

        let mut a = self;
        a.set_normalized();
        let mut b = Self(Self::MOD_M51);
        let mut ls = 0u64;  // running symbol information in bit 1

        // Generic loop does 15*31 = 465 inner iterations.
        for _ in 0..15 {
            // Get approximations of a and b over 63 bits.
            let (a0, a1, a2, a3, a4) = (a.0[0], a.0[1], a.0[2], a.0[3], a.0[4]);
            let (b0, b1, b2, b3, b4) = (b.0[0], b.0[1], b.0[2], b.0[3], b.0[4]);
            let m4 = a4 | b4;
            let m3 = a3 | b3;
            let m2 = a2 | b2;
            let m1 = a1 | b1;
            let tz4 = sgnw(m4.wrapping_sub(1));
            let tz3 = sgnw(m3.wrapping_sub(1));
            let tz2 = sgnw(m2.wrapping_sub(1));
            let tz1 = sgnw(m1.wrapping_sub(1));
            let cc4 = !tz4;
            let cc3 = tz4 & !tz3;
            let cc2 = tz4 & tz3 & !tz2;
            let cc1 = tz4 & tz3 & tz2;
            let one_limb = tz1 & tz2 & tz3 & tz4;
            let tnz = (cc4 & m4) | (cc3 & m3) | (cc2 & m2) | (cc1 & m1);
            let s = 64 - lzcnt(tnz);
            let ta1 = (cc4 & a4) | (cc3 & a3) | (cc2 & a2) | (cc1 & a1);
            let ta0 = (cc4 & a3) | (cc3 & a2) | (cc2 & a1) | (cc1 & a0);
            let tb1 = (cc4 & b4) | (cc3 & b3) | (cc2 & b2) | (cc1 & b1);
            let tb0 = (cc4 & b3) | (cc3 & b2) | (cc2 & b1) | (cc1 & b0);

            let wa = (ta1 << (51 - s)) | (ta0 >> s);
            let wb = (tb1 << (51 - s)) | (tb0 >> s);
            let xa = ((wa & 0x0007FFFFFFF80000) << 12) | (a0 & 0x7FFFFFFF);
            let xb = ((wb & 0x0007FFFFFFF80000) << 12) | (b0 & 0x7FFFFFFF);
            let mut xa = xa ^ (one_limb & (xa ^ a0));
            let mut xb = xb ^ (one_limb & (xb ^ b0));

            // First 29 inner iterations.
            let mut fg0 = 1u64;
            let mut fg1 = 1u64 << 32;
            for _ in 0..29 {
                let a_odd = (xa & 1).wrapping_neg();
                let swap = a_odd & sgnw(xa.wrapping_sub(xb));
                ls ^= swap & (xa & xb);
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
                ls ^= xb.wrapping_add(2) >> 1;
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
                let swap = a_odd & sgnw(xa.wrapping_sub(xb));
                ls ^= swap & (a0 & b0);
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
                ls ^= b0.wrapping_add(2) >> 1;
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
            ls ^= nega & nb.0[0];
            a = na;
            b = nb;
        }

        // If y is invertible, then the final GCD is 1, and
        // len(a) + len(b) <= 45, so we can end the computation with
        // the low words directly. We only need 43 iterations to reach
        // the point where b = 1.
        let mut xa = a.0[0];
        let mut xb = b.0[0];
        for _ in 0..43 {
            let a_odd = (xa & 1).wrapping_neg();
            let swap = a_odd & sgnw(xa.wrapping_sub(xb));
            let t1 = swap & (xa ^ xb);
            xa ^= t1;
            xb ^= t1;
            xa = xa.wrapping_sub(a_odd & xb);
            xa >>= 1;
            ls ^= xb.wrapping_add(2) >> 1;
        }

        // At this point, if the source value was not zero, then bit 1
        // of ls contains the QR status (0 = square, 1 = non-square),
        // which we need to convert to the expected value (+1 or -1).
        // If y == 0, then we return 0, per the API.
        let r = 1u32.wrapping_sub((ls as u32) & 2);
        (r & !(self.iszero() as u32)) as i32
    }

    // Set this value to its square root. Returned value is 0xFFFFFFFF
    // if the operation succeeded (value was indeed a quadratic
    // residue), 0 otherwise (value was not a quadratic residue). In the
    // latter case, this value is set to:
    //  - A square root of -self, if q = 3 mod 4
    //  - A square root of either 2*self or -2*self, if q = 5 mod 8.
    // The case q = 1 mod 8 is not supported; a panic is triggered if this
    // function is called for such a field.
    //
    // In all cases, the returned root is the one whose least significant
    // bit is 0 (when normalized in 0..q-1).
    fn set_sqrt_ext(&mut self) -> u32 {
        // Let x denote the operand (self).
        //
        // If q = 3 mod 4, then the candidate root is y = x^((q+1)/4).
        // If q = 5 mod 8, then we use Atkin's algorithm:
        //   a <- 2*x
        //   b <- a^((q-5)/8)
        //   c <- a*b^2
        //   y <- x*b*(c - 1)
        //
        // Both cases start with computing a^(2^240 - 1), with a = x or
        // a = 2*x.

        let a = if (MQ & 3) == 1 { *self } else { (*self).mul2() };
        let (a2, a3, mut y) = a.pow2_240();
        let win: [Self; 3] = [ a, a2, a3 ];

        if (MQ & 3) == 1 {
            // q = 3 mod 4; square root candidate is:
            //   y = x^((q + 1)/4)
            let e = (MQ.wrapping_neg() + 1) >> 2;
            for i in 0..6 {
                y.set_xsquare(2);
                let k = ((e >> (11 - (2 * i))) & 3) as usize;
                if k != 0 {
                    y.set_mul(&win[k - 1]);
                }
            }
            y.set_square();
            if (e & 1) != 0 {
                y.set_mul(&win[0]);
            }
        } else if (MQ & 7) == 3 {
            // q = 5 mod 8:
            //   a <- 2*x
            //   b <- a^((q-5)/8)
            //   c <- a*b^2
            //   y <- x*b*(c - 1)
            let e = (MQ.wrapping_neg() - 5) >> 3;
            for i in 0..6 {
                y.set_xsquare(2);
                let k = ((e >> (10 - (2 * i))) & 3) as usize;
                if k != 0 {
                    y.set_mul(&win[k - 1]);
                }
            }
            let b = y;

            // c <- 2*x*b^2
            let c = a * b.square();

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
            // Case q = 1 mod 8 is not implemented. A general algorithm
            // is Tonelli-Shanks, which can be optimized a bit, e.g.
            // with: https://eprint.iacr.org/2023/828
            // However, this requires knowledge of a non-QR in the field,
            // which we do not provide in the type parameters.
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
    // The case q = 1 mod 8 is not supported; a panic is triggered if this
    // function is called for such a field.
    #[inline(always)]
    pub fn sqrt(self) -> (Self, u32) {
        let mut x = self;
        let r = x.set_sqrt();
        (x, r)
    }

    // Compute the square root of this value. Returned value are (y, r):
    //  - If this value is indeed a quadratic residue, then y is a
    //    square root of this value, and r is 0xFFFFFFFF.
    //  - If this value is not a quadratic residue, then y is set to a
    //    square root of -self (if field modulus q = 3 mod 4), or to a
    //    square root of either 2*self or -2*self (if q = 5 mod 8);
    //    morever, r is set to 0x00000000.
    // The case q = 1 mod 8 is not supported; a panic is triggered if this
    // function is called for such a field.
    //
    // In all cases, the returned root is normalized: the least significant
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
        lagrange253_vartime(&self.to_limbs64(), &Self::MODULUS)
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
        // Carry propagation.
        let a0 = self.0[0];
        let a1 = self.0[1] + (a0 >> 51);
        let a2 = self.0[2] + (a1 >> 51);
        let a3 = self.0[3] + (a2 >> 51);
        let a4 = self.0[4] + (a3 >> 51);
        let b0 = (a0 & M51) + MQ * (a4 >> 51);
        // MQ*(a4 >> 51) < MQ*2^13 < 2^28, so the carry into b1 can only
        // be 0 or 1.
        let b1 = (a1 & M51) + (b0 >> 51);
        let b2 = (a2 & M51) + (b1 >> 51);
        let b3 = (a3 & M51) + (b2 >> 51);
        let b4 = (a4 & M51) + (b3 >> 51);
        // b4 may exceed 2^51-1 only if b0 produced a carry, in which case
        // b0 % 2^51 must be small, and the carry propagation will stop there.
        let c0 = (b0 & M51) + MQ * (b4 >> 51);
        let c1 = b1 & M51;
        let c2 = b2 & M51;
        let c3 = b3 & M51;
        let c4 = b4 & M51;

        // Limbs in c0..c4 fit on 51 bits each. There are two possible
        // representations for zero.
        let t1 = c0 | c1 | c2 | c3 | c4;
        let t2 = ((c0 ^ (MQ - 1)) & c1 & c2 & c3 & c4) ^ M51;

        // Value is zero only if t1 == 0 or t2 == 0.
        let t1z = t1 | t1.wrapping_neg();
        let t2z = t2 | t2.wrapping_neg();
        (((t1z & t2z) >> 63) as u32).wrapping_sub(1)
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

    // Decode 32 bytes (unsigned little-endian) with implicit reduction.
    #[inline(always)]
    fn set_decode32_reduce(&mut self, buf: &[u8]) {
        debug_assert!(buf.len() == 32);
        let d0 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[ 0.. 8]).unwrap());
        let d1 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[ 8..16]).unwrap());
        let d2 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[16..24]).unwrap());
        let d3 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[24..32]).unwrap());
        *self = Self::from_w64le(d0, d1, d2, d3)
    }

    // Encode this value over exactly 32 bytes. Encoding is always canonical
    // (little-endian encoding of the value in the 0..q-1 range, top bit
    // of the last byte is always 0).
    #[inline(always)]
    pub fn encode32(self) -> [u8; 32] {
        let k = self.to_limbs64();
        let mut d = [0u8; 32];
        d[ 0.. 8].copy_from_slice(&k[0].to_le_bytes());
        d[ 8..16].copy_from_slice(&k[1].to_le_bytes());
        d[16..24].copy_from_slice(&k[2].to_le_bytes());
        d[24..32].copy_from_slice(&k[3].to_le_bytes());
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

        // Decode the input bytes without any reduction; top limb may thus
        // use 52 bits.
        let d0 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[ 0.. 8]).unwrap());
        let d1 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[ 8..16]).unwrap());
        let d2 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[16..24]).unwrap());
        let d3 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[24..32]).unwrap());
        self.0[0] = d0 & M51;
        self.0[1] = (d0 >> 51) | ((d1 << 13) & M51);
        self.0[2] = (d1 >> 38) | ((d2 << 26) & M51);
        self.0[3] = (d2 >> 25) | ((d3 << 39) & M51);
        self.0[4] = d3 >> 12;

        // Try to subtract q from the value; if that does not yield a
        // borrow, then the encoding was not canonical.
        let cc = self.0[0].wrapping_sub(Self::MOD_M51[0]) >> 63;
        let cc = self.0[1].wrapping_sub(Self::MOD_M51[1] + cc) >> 63;
        let cc = self.0[2].wrapping_sub(Self::MOD_M51[2] + cc) >> 63;
        let cc = self.0[3].wrapping_sub(Self::MOD_M51[3] + cc) >> 63;
        let cc = self.0[4].wrapping_sub(Self::MOD_M51[4] + cc) >> 63;

        // Clear the value if not canonical.
        let cc = (cc as u64).wrapping_neg();
        self.0[0] &= cc;
        self.0[1] &= cc;
        self.0[2] &= cc;
        self.0[3] &= cc;
        self.0[4] &= cc;

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
            let mut x = Self::ZERO;
            x.set_decode32_reduce(&buf[k..k + 32]);
            if MQ <= 3827 {
                let d0 = self.0[0] * (2 * MQ) + x.0[0];
                let d1 = self.0[1] * (2 * MQ) + x.0[1];
                let d2 = self.0[2] * (2 * MQ) + x.0[2];
                let d3 = self.0[3] * (2 * MQ) + x.0[3];
                let d4 = self.0[4] * (2 * MQ) + x.0[4];
                self.set_carry_propagate(d0, d1, d2, d3, d4);
            } else {
                let (d0, h0) = umull(self.0[0], (2 * MQ) << 13);
                let (d1, h1) = umull(self.0[1], (2 * MQ) << 13);
                let (d2, h2) = umull(self.0[2], (2 * MQ) << 13);
                let (d3, h3) = umull(self.0[3], (2 * MQ) << 13);
                let (d4, h4) = umull(self.0[4], (2 * MQ) << 13);
                let e0 = (d0 >> 13) + MQ * h4 + x.0[0];
                let e1 = (d1 >> 13) + h0 + x.0[1];
                let e2 = (d2 >> 13) + h1 + x.0[2];
                let e3 = (d3 >> 13) + h2 + x.0[3];
                let e4 = (d4 >> 13) + h3 + x.0[4];
                self.set_carry_propagate(e0, e1, e2, e3, e4);
            }
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
        // We internally use 64-bit limbs and a custom add-with-carry
        // function.

        const fn adc(x: u64, y: u64, c: u64) -> (u64, u64) {
            let z = (x as u128).wrapping_add(y as u128).wrapping_add(c as u128);
            (z as u64, (z >> 64) as u64)
        }

        const fn sqr<const MQ: u64>(a: [u64; 4]) -> [u64; 4] {
            // This follows the same steps as the runtime set_square()
            // in the m64 implementation.
            let (a0, a1, a2, a3) = (a[0], a[1], a[2], a[3]);

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

            [ e0, e1, e2, e3 ]
        }

        // 1/2 = (q + 1)/2 mod q
        let a = [
            ((MQ - 1) >> 1).wrapping_neg(),
            0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0x3FFFFFFFFFFFFFFF,
        ];

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

        // multiply by 16 to get the result.
        let (a0, a1, a2, a3) = (a[0], a[1], a[2], a[3]);
        let tt = a3 >> 59;
        let d0 = a0 << 4;
        let d1 = (a0 >> 60) | (a1 << 4);
        let d2 = (a1 >> 60) | (a2 << 4);
        let d3 = (a2 >> 60) | ((a3 << 4) & 0x7FFFFFFFFFFFFFFF);
        let (d0, cc) = adc(d0, tt * MQ, 0);
        let (d1, cc) = adc(d1, 0, cc);
        let (d2, cc) = adc(d2, 0, cc);
        let (d3, _)  = adc(d3, 0, cc);

        Self::w64le(d0, d1, d2, d3)
    }

    /// Constant-time table lookup: given a table of 48 field elements,
    /// and an index `j` in the 0 to 15 range, return the elements of
    /// index `j*3` to `j*3+2`. If `j` is not in the 0 to 15 range
    /// (inclusive), then this returns three zeros.
    pub fn lookup16_x3(tab: &[Self; 48], j: u32) -> [Self; 3] {
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

    /// Constant-time table lookup: given a table of 64 field elements,
    /// and an index `j` in the 0 to 15 range, return the elements of
    /// index `j*4` to `j*4+3`. If `j` is not in the 0 to 15 range
    /// (inclusive), then this returns four zeros.
    pub fn lookup16_x4(tab: &[Self; 64], j: u32) -> [Self; 4] {
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
}

impl<const MQ: u64> GF255NotReduced<MQ> {

    pub fn square(self) -> GF255<MQ> {
        // GF255::<MQ>::set_square() tolerates unreduced limbs
        let mut r = GF255::<MQ>(self.0);
        r.set_square();
        r
    }

    pub fn xsquare(self, n: u32) -> GF255<MQ> {
        // GF255::<MQ>::set_xsquare() tolerates unreduced limbs
        let mut r = GF255::<MQ>(self.0);
        r.set_xsquare(n);
        r
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

// Traits for multiplications involving "not-reduced" values

impl<const MQ: u64> Mul<GF255NotReduced<MQ>> for GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn mul(self, other: GF255NotReduced<MQ>) -> GF255<MQ> {
        let mut r = self;
        r.set_mul(&GF255::<MQ>(other.0));
        r
    }
}

impl<const MQ: u64> Mul<&GF255NotReduced<MQ>> for GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn mul(self, other: &GF255NotReduced<MQ>) -> GF255<MQ> {
        let mut r = self;
        r.set_mul(&GF255::<MQ>(other.0));
        r
    }
}

impl<const MQ: u64> Mul<GF255NotReduced<MQ>> for &GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn mul(self, other: GF255NotReduced<MQ>) -> GF255<MQ> {
        let mut r = *self;
        r.set_mul(&GF255::<MQ>(other.0));
        r
    }
}

impl<const MQ: u64> Mul<&GF255NotReduced<MQ>> for &GF255<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn mul(self, other: &GF255NotReduced<MQ>) -> GF255<MQ> {
        let mut r = *self;
        r.set_mul(&GF255::<MQ>(other.0));
        r
    }
}

impl<const MQ: u64> MulAssign<GF255NotReduced<MQ>> for GF255<MQ> {
    #[inline(always)]
    fn mul_assign(&mut self, other: GF255NotReduced<MQ>) {
        self.set_mul(&GF255::<MQ>(other.0));
    }
}

impl<const MQ: u64> MulAssign<&GF255NotReduced<MQ>> for GF255<MQ> {
    #[inline(always)]
    fn mul_assign(&mut self, other: &GF255NotReduced<MQ>) {
        self.set_mul(&GF255::<MQ>(other.0));
    }
}

impl<const MQ: u64> Mul<GF255<MQ>> for GF255NotReduced<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn mul(self, other: GF255<MQ>) -> GF255<MQ> {
        let mut r = GF255::<MQ>(self.0);
        r.set_mul(&other);
        r
    }
}

impl<const MQ: u64> Mul<&GF255<MQ>> for GF255NotReduced<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn mul(self, other: &GF255<MQ>) -> GF255<MQ> {
        let mut r = GF255::<MQ>(self.0);
        r.set_mul(other);
        r
    }
}

impl<const MQ: u64> Mul<GF255<MQ>> for &GF255NotReduced<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn mul(self, other: GF255<MQ>) -> GF255<MQ> {
        let mut r = GF255::<MQ>(self.0);
        r.set_mul(&other);
        r
    }
}

impl<const MQ: u64> Mul<&GF255<MQ>> for &GF255NotReduced<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn mul(self, other: &GF255<MQ>) -> GF255<MQ> {
        let mut r = GF255::<MQ>(self.0);
        r.set_mul(other);
        r
    }
}

impl<const MQ: u64> Mul<GF255NotReduced<MQ>> for GF255NotReduced<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn mul(self, other: GF255NotReduced<MQ>) -> GF255<MQ> {
        let mut r = GF255::<MQ>(self.0);
        r.set_mul(&GF255::<MQ>(other.0));
        r
    }
}

impl<const MQ: u64> Mul<&GF255NotReduced<MQ>> for GF255NotReduced<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn mul(self, other: &GF255NotReduced<MQ>) -> GF255<MQ> {
        let mut r = GF255::<MQ>(self.0);
        r.set_mul(&GF255::<MQ>(other.0));
        r
    }
}

impl<const MQ: u64> Mul<GF255NotReduced<MQ>> for &GF255NotReduced<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn mul(self, other: GF255NotReduced<MQ>) -> GF255<MQ> {
        let mut r = GF255::<MQ>(self.0);
        r.set_mul(&GF255::<MQ>(other.0));
        r
    }
}

impl<const MQ: u64> Mul<&GF255NotReduced<MQ>> for &GF255NotReduced<MQ> {
    type Output = GF255<MQ>;

    #[inline(always)]
    fn mul(self, other: &GF255NotReduced<MQ>) -> GF255<MQ> {
        let mut r = GF255::<MQ>(self.0);
        r.set_mul(&GF255::<MQ>(other.0));
        r
    }
}

// ========================================================================

#[cfg(test)]
mod tests {

    use super::{GF255};
    use num_bigint::{BigInt, Sign};
    use crate::sha2::Sha256;

    /*
    fn print<const MQ: u64>(name: &str, v: GF255<MQ>) {
        println!("{} = 0x{:016X} + 0x{:016X}*2**51 + 0x{:016X}*2**102 + 0x{:016X}*2**153 + 0x{:016X}*2**204",
            name, v.0[0], v.0[1], v.0[2], v.0[3], v.0[4]);
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
