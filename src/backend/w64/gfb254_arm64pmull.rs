use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::convert::TryFrom;

use core::arch::aarch64::*;
use core::mem::transmute;

/// Element of GF(2^127), using modulus 1 + z^63 + z^127.
#[derive(Clone, Copy, Debug)]
pub struct GFb127(uint64x2_t);

impl GFb127 {

    // IMPLEMENTATION NOTES
    // --------------------
    //
    // We tolerate internal values up to 128 bits. All computations are
    // performed modulo z + z^64 + z^128, which makes reductions easier
    // (z^64 and z^128 are 64-bit aligned).

    pub const ZERO: Self = Self::w64le(0, 0);
    pub const ONE: Self = Self::w64le(1, 0);

    /// Create a constant GF(2^127) value from its 128-bit representation
    /// (x0 is the low 64 bits, x1 the high 64 bits). The value is
    /// implicitly reduced to 127 bits. This is for hardcoding constants
    /// evaluated at compile-time.
    pub const fn w64le(x0: u64, x1: u64) -> Self {
        unsafe {
            Self(transmute([ x0, x1 ]))
        }
    }

    /// Make a value out of two 64-bit limbs (least significant limb first).
    /// The value is implicitly reduced to 127 bits.
    #[inline(always)]
    pub fn from_w64le(x0: u64, x1: u64) -> Self {
        unsafe {
            // Apparently the compiler is smart enough to avoid going
            // through RAM in that case.
            Self(transmute([ x0, x1 ]))
        }
    }

    // Split this value into two 64-bit limbs.
    #[inline(always)]
    fn to_limbs(self) -> [u64; 2] {
        unsafe {
            transmute(self.0)
        }
    }

    // Normalize this value and split it into two 64-bit limbs.
    #[inline(always)]
    fn normalize_limbs(self) -> [u64; 2] {
        unsafe {
            let xa = vandq_u64(self.0, transmute([0, 1u64 << 63]));
            let xb = vsriq_n_u64(xa, xa, 63);
            let xc = vcopyq_laneq_u64(xa, 0, xb, 1);
            transmute(veorq_u64(self.0, xc))
        }
    }

    // Get the bit at the specified index. The index `k` MUST be between
    // 0 and 126 (inclusive). Side-channel attacks may reveal the value of
    // the index (bit not the value of the read bit). Returned value is
    // 0 or 1.
    #[inline(always)]
    pub fn get_bit(self, k: usize) -> u32 {
        // Normalize the value.
        let x = self.normalize_limbs();
        ((x[k >> 6] >> (k & 63)) as u32) & 1
    }

    // Set the bit at the specified index. The index `k` MUST be between
    // 0 and 126 (inclusive). Side-channel attacks may reveal the value of
    // the index (bit not the value of the written bit). Only the least
    // significant bit of `val` is used; the over bits are ignored.
    #[inline(always)]
    pub fn set_bit(&mut self, k: usize, val: u32) {
        // We need to normalize the value, otherwise we can get the wrong
        // outcome.
        let mut x = self.normalize_limbs();
        let ki = k >> 6;
        let kj = k & 63;
        x[ki] &= !(1u64 << kj);
        x[ki] |= ((val & 1) as u64) << kj;
        *self = Self::from_w64le(x[0], x[1]);
    }

    // XOR (add) a one-bit value at the specified index. The index `k`
    // MUST be between 0 and 126 (inclusive). Side-channel attacks may
    // reveal the value of the index (bit not the value of the added bit).
    // Only the least significant bit of `val` is used; the over bits
    // are ignored.
    #[inline(always)]
    pub fn xor_bit(&mut self, k: usize, val: u32) {
        let mut x = self.to_limbs();
        x[k >> 6] ^= ((val & 1) as u64) << (k & 64);
        *self = Self::from_w64le(x[0], x[1]);
    }

    #[inline(always)]
    fn set_add(&mut self, rhs: &Self) {
        unsafe {
            self.0 = veorq_u64(self.0, rhs.0);
        }
    }

    // Subtraction is the same thing as addition in binary fields.

    #[inline(always)]
    pub fn set_cond(&mut self, a: &Self, ctl: u32) {
        unsafe {
            let cw = vreinterpretq_u64_u32(vdupq_n_u32(ctl));
            self.0 = vbslq_u64(cw, a.0, self.0);
        }
    }

    #[inline(always)]
    pub fn select(a0: &Self, a1: &Self, ctl: u32) -> Self {
        let mut r = *a0;
        r.set_cond(a1, ctl);
        r
    }

    #[inline(always)]
    pub fn cswap(a: &mut Self, b: &mut Self, ctl: u32) {
        unsafe {
            let xa = a.0;
            let xb = b.0;
            let cw = vreinterpretq_u64_u32(vdupq_n_u32(ctl));
            a.0 = vbslq_u64(cw, xb, xa);
            b.0 = vbslq_u64(cw, xa, xb);
        }
    }

    // Multiply this value by sb = 1 + z^27.
    #[inline(always)]
    pub fn set_mul_sb(&mut self) {
        unsafe {
            let a = self.0;
            let f = vshlq_n_u64(a, 27);
            let g = vshrq_n_u64(a, 37);

            // g = g0 + g1*z^64
            // r = a + f + g*z^64
            //   = a + f + g0*z^64 + g1*z + g1*z^64
            let g0 = vget_low_u64(g);
            let g1 = vget_high_u64(g);
            let h = vcombine_u64(vshl_n_u64(g1, 1), veor_u64(g0, g1));
            self.0 = veorq_u64(veorq_u64(a, f), h);
        }
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
        unsafe {
            let a = self.0;
            let f = vshlq_n_u64(a, 54);
            let g = vshrq_n_u64(a, 10);

            // g = g0 + g1*z^64
            // r = a + f + g*z^64
            //   = a + f + g0*z^64 + g1*z + g1*z^64
            let g0 = vget_low_u64(g);
            let g1 = vget_high_u64(g);
            let h = vcombine_u64(vshl_n_u64(g1, 1), veor_u64(g0, g1));
            self.0 = veorq_u64(veorq_u64(a, f), h);
        }
    }

    // Multiply this value by sb = 1 + z^54.
    #[inline(always)]
    pub fn mul_b(self) -> Self {
        let mut x = self;
        x.set_mul_b();
        x
    }

    /* unused
    // Multiply this value by bb = 1 + z^108.
    #[inline(always)]
    pub fn set_mul_bb(&mut self) {
        unsafe {
            let a = self.0;
            let e0 = vshlq_n_u64(a, 44);
            let e1 = vshrq_n_u64(a, 20);

            // e0 = f0:f1
            // e1 = f2:f3     note: len(f2) <= 44, len(f3) <= 44
            // r = a + f0*z^64 + f1*z^128 + f2*z^128 + f3*z^192
            //   = a + f0*z^64 + (f1 + f2)*z + (f1 + f2)*z^64
            //       + (f3*z)*z^64 + f3*z + f3*z^64
            //   = a + f0*z^64 + (f1 + f2 + f3)*z
            //       + (f1 + f2 + f3)*z^64 + (f3*z)*z^64

            // g = (f1 + f2 + f3) + f3*z^64
            let g = vzip2q_u64(
                veorq_u64(veorq_u64(e0, e1), vdupq_laneq_u64(e1, 0)), e1);

            // h = (f1 + f2 + f3)*z + (f3*z)*z^64
            let j = vextq_u64(g, g, 1);
            let h = vsriq_n_u64(vshlq_n_u64(g, 1), j, 63);

            // r = a + f0*z^64 + lo(g)*z^64 + h
            let r = veorq_u64(
                veorq_u64(a, h),
                vextq_u64(veorq_u64(e0, e0), veorq_u64(e0, g), 1));
            self.0 = r;
        }
    }

    // Multiply this value by sb = 1 + z^108.
    #[inline(always)]
    pub fn mul_bb(self) -> Self {
        let mut x = self;
        x.set_mul_bb();
        x
    }
    */

    // Divide this value by z.
    #[inline(always)]
    pub fn set_div_z(&mut self) {
        unsafe {
            let a = self.0;

            // Move the least significant bit upwards (reverse reduction).
            let b = vshlq_n_u64(a, 63);
            let a = veorq_u64(a, vdupq_laneq_u64(b, 0));
            // Simple shift (lsb is now implicitly zero).
            self.0 = vorrq_u64(
                vshrq_n_u64(a, 1),
                vextq_u64(b, veorq_u64(a, a), 1));
        }
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
        self.set_div_z();
        self.set_div_z();
    }

    // Divide this value by z.
    #[inline(always)]
    pub fn div_z2(self) -> Self {
        let mut x = self;
        x.set_div_z2();
        x
    }

    #[inline]
    fn set_mul(&mut self, rhs: &Self) {
        unsafe {
            let a = self.0;
            let b = rhs.0;

            // a*b = c0 + c1*z^64 + c2*z^128
            let c0 = vreinterpretq_u64_p128(vmull_p64(
                vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0)));
            let c2 = vreinterpretq_u64_p128(vmull_high_p64(
                vreinterpretq_p64_u64(a), vreinterpretq_p64_u64(b)));
            let bx = vextq_u64(b, b, 1);
            let c4 = vreinterpretq_u64_p128(vmull_p64(
                vgetq_lane_u64(a, 0), vgetq_lane_u64(bx, 0)));
            let c5 = vreinterpretq_u64_p128(vmull_high_p64(
                vreinterpretq_p64_u64(a), vreinterpretq_p64_u64(bx)));
            let c1 = veorq_u64(c4, c5);

            // a*b = d0 + d1*z^128
            let z = vreinterpretq_u64_p128(0);
            let d0 = veorq_u64(c0, vextq_u64(z, c1, 1));
            let d1 = veorq_u64(c2, vextq_u64(c1, z, 1));

            // Reduction: z^128 = z^64 + z
            // We write:
            //   d0 = e0 + e1*z^64
            //   d1 = e2 + e3*z^64
            // We note that len(e3) <= 63.
            //   (e2 + e3*z^64)*z^128
            //    = (e2 + e3 + e3*z^64)*z + (e2 + e3)*z^64
            // Since len(e3) <= 63, the most significant bit of e2 + e3 is
            // equal to the most significant bit of e2.
            //
            // We split values into 64-bit halves: the Cortex-A55 can
            // execute two 64-bit SIMD operations in the same cycle,
            // but only one 128-bit operation per cycle.

            let e2 = vget_low_u64(d1);
            let e3 = vget_high_u64(d1);
            let f = veor_u64(e2, e3);
            let g = vshl_n_u64(e3, 1);
            let h = vshl_n_u64(f, 1);
            let j = vsri_n_u64(g, e2, 63);
            let k = veor_u64(f, j);
            self.0 = veorq_u64(d0, vcombine_u64(h, k));
        }
    }

    // Square this value (in place).
    #[inline(always)]
    pub fn set_square(&mut self) {
        unsafe {
            let a = self.0;

            // a^2 = d0 + d1*z^128
            let d0 = vreinterpretq_u64_p128(vmull_p64(
                vgetq_lane_u64(a, 0), vgetq_lane_u64(a, 0)));
            let d1 = vreinterpretq_u64_p128(vmull_high_p64(
                vreinterpretq_p64_u64(a), vreinterpretq_p64_u64(a)));

            // Reduction: similar to set_mul(), but we know that the
            // top bit of e2 is 0, so we can skip some operations.
            let e2 = vget_low_u64(d1);
            let e3 = vget_high_u64(d1);
            let f = veor_u64(e2, e3);
            let g = vshl_n_u64(e3, 1);
            let h = vshl_n_u64(f, 1);
            let k = veor_u64(f, g);
            self.0 = veorq_u64(d0, vcombine_u64(h, k));
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
        unsafe {
            let xa = vandq_u64(self.0, transmute([0, 1u64 << 63]));
            let xb = vsriq_n_u64(xa, xa, 63);
            let xc = vcopyq_laneq_u64(xa, 0, xb, 1);
            self.0 = veorq_u64(self.0, xc);
        }
    }

    // Invert this value; if this value is zero, then it stays at zero.
    pub fn set_invert(&mut self) {
        // We use Itoh-Tsujii, with optimized sequences of squarings.
        // We have:
        //   1/a = a^(2^127 - 2)
        //       = (a^2)^(2^126 - 1)
        // We use an addition chain for the exponent:
        //   1 -> 2 -> 3 -> 6 -> 7 -> 14 -> 21 -> 42 -> 63 -> 126

        let a1 = self.square();
        let a2 = a1 * a1.square();
        let a3 = a1 * a2.square();
        let a6 = a3 * a3.xsquare(3);
        let a7 = a1 * a6.square();
        let a14 = a7 * a7.xsquare(7);
        let a21 = a7 * a14.xsquare(7);
        let a42 = a21 * a21.xsquare(21);
        let a63 = a21 * a42.xsquare(21);
        let a126 = a63 * a63.xsquare(63);

        /* Unused -- alternate computations for the last three steps,
           using a table. The memory bandwidth is low on the Cortex A55
           and the table method happens to be slower.
        let a42 = a21 * a21.frob(&Self::FROB21);
        let a63 = a21 * a42.frob(&Self::FROB21);
        let a126 = a63 * a63.frob(&Self::FROB63);
        */

        *self = a126;
    }

    /* unused
    #[inline]
    fn frob(self, tab: &[GFb127; 128]) -> Self {
        unsafe {
            let mut a = self.0;
            let mut d0 = vdup_n_u64(0);
            let mut d1 = vdup_n_u64(0);
            let t: &[uint64x1_t; 256] = transmute(tab);
            for i in (0..2).rev() {
                let mut mw = vget_high_u64(a);
                a = vextq_u64(a, a, 1);
                for j in (0..8).rev() {
                    let m = vcltz_s64(vreinterpret_s64_u64(mw));
                    mw = vshl_n_u64(mw, 1);
                    d0 = veor_u64(d0, vand_u64(m, t[(i << 7) + (j << 4) + 14]));
                    d1 = veor_u64(d1, vand_u64(m, t[(i << 7) + (j << 4) + 15]));
                    let m = vcltz_s64(vreinterpret_s64_u64(mw));
                    mw = vshl_n_u64(mw, 1);
                    d0 = veor_u64(d0, vand_u64(m, t[(i << 7) + (j << 4) + 12]));
                    d1 = veor_u64(d1, vand_u64(m, t[(i << 7) + (j << 4) + 13]));
                    let m = vcltz_s64(vreinterpret_s64_u64(mw));
                    mw = vshl_n_u64(mw, 1);
                    d0 = veor_u64(d0, vand_u64(m, t[(i << 7) + (j << 4) + 10]));
                    d1 = veor_u64(d1, vand_u64(m, t[(i << 7) + (j << 4) + 11]));
                    let m = vcltz_s64(vreinterpret_s64_u64(mw));
                    mw = vshl_n_u64(mw, 1);
                    d0 = veor_u64(d0, vand_u64(m, t[(i << 7) + (j << 4) +  8]));
                    d1 = veor_u64(d1, vand_u64(m, t[(i << 7) + (j << 4) +  9]));
                    let m = vcltz_s64(vreinterpret_s64_u64(mw));
                    mw = vshl_n_u64(mw, 1);
                    d0 = veor_u64(d0, vand_u64(m, t[(i << 7) + (j << 4) +  6]));
                    d1 = veor_u64(d1, vand_u64(m, t[(i << 7) + (j << 4) +  7]));
                    let m = vcltz_s64(vreinterpret_s64_u64(mw));
                    mw = vshl_n_u64(mw, 1);
                    d0 = veor_u64(d0, vand_u64(m, t[(i << 7) + (j << 4) +  4]));
                    d1 = veor_u64(d1, vand_u64(m, t[(i << 7) + (j << 4) +  5]));
                    let m = vcltz_s64(vreinterpret_s64_u64(mw));
                    mw = vshl_n_u64(mw, 1);
                    d0 = veor_u64(d0, vand_u64(m, t[(i << 7) + (j << 4) +  2]));
                    d1 = veor_u64(d1, vand_u64(m, t[(i << 7) + (j << 4) +  3]));
                    let m = vcltz_s64(vreinterpret_s64_u64(mw));
                    mw = vshl_n_u64(mw, 1);
                    d0 = veor_u64(d0, vand_u64(m, t[(i << 7) + (j << 4) +  0]));
                    d1 = veor_u64(d1, vand_u64(m, t[(i << 7) + (j << 4) +  1]));
                }
            }
            Self(vcombine_u64(d0, d1))
        }
    }
    */

    /* unused
    // z^(i*2^21) for i = 0 to 127.
    const FROB21: [GFb127; 128] = [
        GFb127::w64le(0x0000000000000001, 0x0000000000000000),
        GFb127::w64le(0x0000000000010010, 0x0000000000000001),
        GFb127::w64le(0x0000000100000102, 0x0000000000000001),
        GFb127::w64le(0x0001001001021022, 0x0000000100010113),
        GFb127::w64le(0x0000000000010006, 0x0000000000000000),
        GFb127::w64le(0x0000000100160060, 0x0000000000010006),
        GFb127::w64le(0x000100060102060C, 0x0000000000010006),
        GFb127::w64le(0x00160162162E60CC, 0x000100070115066B),
        GFb127::w64le(0x0000000100000014, 0x0000000000000000),
        GFb127::w64le(0x0001001000140140, 0x0000000100000014),
        GFb127::w64le(0x0000011600001428, 0x0000000100000015),
        GFb127::w64le(0x01161162142942AA, 0x000101070015156D),
        GFb127::w64le(0x0001000600140078, 0x0000000000000000),
        GFb127::w64le(0x0016007401380780, 0x0001000600140079),
        GFb127::w64le(0x01160674142878F0, 0x000100060015007E),
        GFb127::w64le(0x171673653A5D8FFE, 0x0101060715137E79),
        GFb127::w64le(0x0000000000000110, 0x0000000000000001),
        GFb127::w64le(0x0000000001101102, 0x0000000000010101),
        GFb127::w64le(0x0000011000011222, 0x0000000100000013),
        GFb127::w64le(0x0110110312332206, 0x0001010100121301),
        GFb127::w64le(0x0000000001100660, 0x0000000000010006),
        GFb127::w64le(0x000001101762660C, 0x0000000101070606),
        GFb127::w64le(0x0110066112246CCC, 0x000100060013006A),
        GFb127::w64le(0x176374394EACCC16, 0x01070614136D6B17),
        GFb127::w64le(0x0000011000001540, 0x0000000100000014),
        GFb127::w64le(0x0110110215415428, 0x0001010100141414),
        GFb127::w64le(0x0001076200156AAA, 0x000000070000006D),
        GFb127::w64le(0x0772763B6BFCAA7A, 0x0006071500786C16),
        GFb127::w64le(0x0110066015407F80, 0x0001000600140078),
        GFb127::w64le(0x1762734D2BAFF8F2, 0x01070612146C7969),
        GFb127::w64le(0x076413596AD57FFC, 0x00070012006D016F),
        GFb127::w64le(0x65165F67D271FD10, 0x070112066D076F00),
        GFb127::w64le(0x0000000000010102, 0x0000000000000001),
        GFb127::w64le(0x0000000101121022, 0x0000000000000113),
        GFb127::w64le(0x0001010201030006, 0x0000000100010001),
        GFb127::w64le(0x0112112110340062, 0x0000011201130016),
        GFb127::w64le(0x000000010104060C, 0x0000000000010006),
        GFb127::w64le(0x00010114164E60CC, 0x000000000113066A),
        GFb127::w64le(0x0104070F060C0014, 0x0001000700070007),
        GFb127::w64le(0x174D76F260DA014C, 0x0112077F067C0166),
        GFb127::w64le(0x0001010200141428, 0x0000000100000014),
        GFb127::w64le(0x01121036156942A8, 0x000001130000157D),
        GFb127::w64le(0x0117142E143C007A, 0x0001001500150117),
        GFb127::w64le(0x055D56F74390058C, 0x0113157E146E110B),
        GFb127::w64le(0x01040618145078F0, 0x0001000600140079),
        GFb127::w64le(0x165A75DD3DDF8FF0, 0x0113066A157D7E1C),
        GFb127::w64le(0x125C6CD878F2011E, 0x0013006B01690764),
        GFb127::w64le(0x4938B5A38CEC1F0E, 0x13146B6A686F6274),
        GFb127::w64le(0x0000000001111222, 0x0000000000010013),
        GFb127::w64le(0x0000011103312206, 0x0000000101131301),
        GFb127::w64le(0x0111122313320662, 0x0001001300120117),
        GFb127::w64le(0x033031253566640C, 0x0113131213171715),
        GFb127::w64le(0x0000011114446CCC, 0x000000010015006A),
        GFb127::w64le(0x0111055728A0CC14, 0x00010115156B6A06),
        GFb127::w64le(0x14457FF86CCE154E, 0x00150078017B0762),
        GFb127::w64le(0x3B8593B8DB595A0E, 0x1578797B7D67705D),
        GFb127::w64le(0x0111122215556AA8, 0x000100130014017C),
        GFb127::w64le(0x033137523FD6A87A, 0x01131315157D7D04),
        GFb127::w64le(0x06676CDF7FEA7F8E, 0x0006006B0078071C),
        GFb127::w64le(0x0AA5B2CB81D9F6D4, 0x066A6A7C7F0E0F33),
        GFb127::w64le(0x1444799915577FF2, 0x0015007E01040618),
        GFb127::w64le(0x3DF48C3A288DF33A, 0x157F7F03020B0C3A),
        GFb127::w64le(0x798C15297FF30128, 0x007F0102060C1429),
        GFb127::w64le(0x8D162F60F2003A2C, 0x7F0102070D172E65),
        GFb127::w64le(0x0000000100010006, 0x0000000000000001),
        GFb127::w64le(0x0001001100160062, 0x0000000100000017),
        GFb127::w64le(0x000101040102060E, 0x0000000000010104),
        GFb127::w64le(0x01141142162C62E8, 0x000101050017174B),
        GFb127::w64le(0x0001000700000014, 0x0000000000010006),
        GFb127::w64le(0x001700700016014C, 0x0001000600170073),
        GFb127::w64le(0x0102071A00021424, 0x0000000101020619),
        GFb127::w64le(0x173A71A016014E72, 0x01030609173972AF),
        GFb127::w64le(0x0001001200140078, 0x0000000100000015),
        GFb127::w64le(0x00020136013807AA, 0x000000030001013C),
        GFb127::w64le(0x0116125E142878D8, 0x0001010400151554),
        GFb127::w64le(0x033D33C13A75A72A, 0x0003030F003839DB),
        GFb127::w64le(0x0014007800000110, 0x000100060015007F),
        GFb127::w64le(0x013A078C013A11FC, 0x0003000B013A068A),
        GFb127::w64le(0x142A79EC002912D2, 0x0102060D152A7EEF),
        GFb127::w64le(0x394F92F33A17D2FA, 0x03050A1A394B97E4),
        GFb127::w64le(0x0000011001100662, 0x0000000100010117),
        GFb127::w64le(0x011010121760640E, 0x0001010000161605),
        GFb127::w64le(0x0111144112246CE8, 0x000101040012134A),
        GFb127::w64le(0x0552543D4E8AE816, 0x00040517005D4A12),
        GFb127::w64le(0x011007700002154C, 0x0001000701110672),
        GFb127::w64le(0x1672770C174F5826, 0x010606161671750F),
        GFb127::w64le(0x12276BA200316A72, 0x0102060A13266AAC),
        GFb127::w64le(0x4BD0B6054F28707C, 0x050F1E2F4BDDB93A),
        GFb127::w64le(0x0110132215407FAA, 0x000101030014143D),
        GFb127::w64le(0x022125672B85D2D8, 0x0002020500282956),
        GFb127::w64le(0x077179FD6AD57F28, 0x0006071A00796CCD),
        GFb127::w64le(0x0EE6FB37D28D2B16, 0x000D0F3E01F7D842),
        GFb127::w64le(0x15427F8C002B01FE, 0x0105061E1445799F),
        GFb127::w64le(0x29A1F4D72BC4EED4, 0x02090C3629A6F5D7),
        GFb127::w64le(0x6ADA7EDA01D702FC, 0x070E12256DDA6DD9),
        GFb127::w64le(0xDD61C83DC438FA6E, 0x0F102373DC72DF67),
        GFb127::w64le(0x000101030104060E, 0x0000000100000105),
        GFb127::w64le(0x01131136164E62EA, 0x000001120001175A),
        GFb127::w64le(0x0007070B060E0014, 0x0001010501040103),
        GFb127::w64le(0x077974B462FC0344, 0x0113175A16491121),
        GFb127::w64le(0x0105070E00161424, 0x000100060105061F),
        GFb127::w64le(0x175C70FA174F4E7C, 0x0112066D175C72CF),
        GFb127::w64le(0x071914341430007A, 0x0103071A071B060C),
        GFb127::w64le(0x67A359454D4C09BE, 0x11306595649760AC),
        GFb127::w64le(0x01101232145078DA, 0x0000011100011546),
        GFb127::w64le(0x033335533DDFA72C, 0x00010232010638EC),
        GFb127::w64le(0x06626C8878DA031A, 0x0110154714561232),
        GFb127::w64le(0x088E9AD3A61617E4, 0x033439E83BDF357A),
        GFb127::w64le(0x145278FC013B12DC, 0x0111066715407E84),
        GFb127::w64le(0x3FF982352BEDD2EA, 0x02340DAA3EF8915A),
        GFb127::w64le(0x79C513EB11C6087C, 0x13276BC46BC66BDE),
        GFb127::w64le(0xA9F4F8FFC3907630, 0x3350AFAFADB8B4A6),
        GFb127::w64le(0x0111133314466CEA, 0x000100120105125B),
        GFb127::w64le(0x022027532A86EA14, 0x0112130417495A11),
        GFb127::w64le(0x077579BC6CE81746, 0x0117135F134B1227),
        GFb127::w64le(0x0CC5D190FF515200, 0x044C4C5959352173),
        GFb127::w64le(0x15557EEC157F6A7E, 0x0014016914456CCA),
        GFb127::w64le(0x2B93F96C15027E5C, 0x15687D5129A6DF54),
        GFb127::w64le(0x6A8379617D3671BA, 0x152D7889799D6AB0),
        GFb127::w64le(0xFB0C1A3351E7E498, 0x55F0F0E2F7CDCFA3),
        GFb127::w64le(0x01131317157D7D2C, 0x0111133315557FFD),
        GFb127::w64le(0x0004026A00512F18, 0x022026402A82FD03),
        GFb127::w64le(0x066A6A717F0F0FC6, 0x06666AAA7FFF000F),
        GFb127::w64le(0x001B0D4C00C8F0B2, 0x0CC0D583FF0F0F35),
        GFb127::w64le(0x157D7F0F02230CCA, 0x15557FFF0002000C),
        GFb127::w64le(0x02720D2D2EFEE610, 0x2A80FF02020D0C2E),
        GFb127::w64le(0x7F0C02280DE42C58, 0x7FFF0002000D002E),
        GFb127::w64le(0x8D162F60F2003A2D, 0x7F0102070D172E65),
    ];
    */

    /* unused
    // z^(i*2^63) for i = 0 to 127.
    const FROB63: [GFb127; 128] = [
        GFb127::w64le(0x0000000000000001, 0x0000000000000000),
        GFb127::w64le(0x0000000100000110, 0x0000000000000001),
        GFb127::w64le(0x0000000000010102, 0x0000000000000000),
        GFb127::w64le(0x0001010201111220, 0x0000000000010102),
        GFb127::w64le(0x0000000100010004, 0x0000000000000000),
        GFb127::w64le(0x0001011401100440, 0x0000000100010005),
        GFb127::w64le(0x0001010301060408, 0x0000000000000000),
        GFb127::w64le(0x0017173916644880, 0x000101030107050B),
        GFb127::w64le(0x0000000100000010, 0x0000000000000001),
        GFb127::w64le(0x0000010000001102, 0x0000000000000100),
        GFb127::w64le(0x0001010200101020, 0x0000000000010102),
        GFb127::w64le(0x0101020011132004, 0x0000000001010200),
        GFb127::w64le(0x0001001400100040, 0x0000000100010005),
        GFb127::w64le(0x0100150211024408, 0x0000010001000500),
        GFb127::w64le(0x0116143810604080, 0x000101030107050B),
        GFb127::w64le(0x1717391764488010, 0x0101030107050B01),
        GFb127::w64le(0x0000000000000102, 0x0000000000000000),
        GFb127::w64le(0x0000010200011220, 0x0000000000000102),
        GFb127::w64le(0x0000000001030004, 0x0000000000000000),
        GFb127::w64le(0x0103000513300440, 0x0000000001030004),
        GFb127::w64le(0x0000010201020408, 0x0000000000000000),
        GFb127::w64le(0x0103162912244880, 0x000001020102050A),
        GFb127::w64le(0x0103010704080010, 0x0000000000000000),
        GFb127::w64le(0x1739176448801100, 0x01030107050B0116),
        GFb127::w64le(0x0000010200001020, 0x0000000000000102),
        GFb127::w64le(0x0001020000112004, 0x0000000000010200),
        GFb127::w64le(0x0103000410300040, 0x0000000001030004),
        GFb127::w64le(0x0300041131064408, 0x0000000103000401),
        GFb127::w64le(0x0102142810204080, 0x000001020102050A),
        GFb127::w64le(0x0215281520408010, 0x0001020102050A01),
        GFb127::w64le(0x1438106040800100, 0x01030107050B0117),
        GFb127::w64le(0x3917654A80111022, 0x030107050B011714),
        GFb127::w64le(0x0000000000010004, 0x0000000000000000),
        GFb127::w64le(0x0001000401100440, 0x0000000000010004),
        GFb127::w64le(0x0000000101060408, 0x0000000000000000),
        GFb127::w64le(0x0106051916644880, 0x0000000101060409),
        GFb127::w64le(0x0001000500000010, 0x0000000000000000),
        GFb127::w64le(0x0110054000001100, 0x0001000500010015),
        GFb127::w64le(0x0107050A00101020, 0x0000000000000001),
        GFb127::w64le(0x17654A8011112202, 0x0107050B0117143A),
        GFb127::w64le(0x0001000400100040, 0x0000000000010004),
        GFb127::w64le(0x0100040011024408, 0x0000000001000400),
        GFb127::w64le(0x0106041810604080, 0x0000000101060409),
        GFb127::w64le(0x0604191364488010, 0x0000010106040901),
        GFb127::w64le(0x0010004000000100, 0x0001000500010015),
        GFb127::w64le(0x1102450A00011020, 0x0100050001001500),
        GFb127::w64le(0x1060408001010202, 0x0107050B0117153B),
        GFb127::w64le(0x654B801511320242, 0x07050B0117153A12),
        GFb127::w64le(0x0000000001020408, 0x0000000000000000),
        GFb127::w64le(0x0102040912244880, 0x0000000001020408),
        GFb127::w64le(0x0000010304080010, 0x0000000000000000),
        GFb127::w64le(0x0409132448801100, 0x0000010304080113),
        GFb127::w64le(0x0102050A00001020, 0x0000000000000000),
        GFb127::w64le(0x12254A8000112200, 0x0102050A0102152B),
        GFb127::w64le(0x050B001410300040, 0x0000000000000103),
        GFb127::w64le(0x4B80151133004606, 0x050B0117153A1262),
        GFb127::w64le(0x0102040810204080, 0x0000000001020408),
        GFb127::w64le(0x0204081120408010, 0x0000000102040801),
        GFb127::w64le(0x0408102040800100, 0x0000010304080113),
        GFb127::w64le(0x0811214280111020, 0x0001030408011304),
        GFb127::w64le(0x1020408000010200, 0x0102050A0102152A),
        GFb127::w64le(0x2041801401120042, 0x02050A0102152A10),
        GFb127::w64le(0x4080010103000606, 0x050B0117153B1167),
        GFb127::w64le(0x8117153B1066468A, 0x0B0117153B106646),
        GFb127::w64le(0x0000000100000010, 0x0000000000000000),
        GFb127::w64le(0x0000010000001100, 0x0000000100000011),
        GFb127::w64le(0x0001010200101020, 0x0000000000000000),
        GFb127::w64le(0x0101020011112200, 0x0001010200111122),
        GFb127::w64le(0x0001001400100040, 0x0000000000000001),
        GFb127::w64le(0x0100150011004402, 0x0001001500110145),
        GFb127::w64le(0x0116143810604080, 0x0000000000010103),
        GFb127::w64le(0x17153B1166468A06, 0x0117153B1066468A),
        GFb127::w64le(0x0000000000000100, 0x0000000100000011),
        GFb127::w64le(0x0000010200011020, 0x0000010000001100),
        GFb127::w64le(0x0000000001010200, 0x0001010200111122),
        GFb127::w64le(0x0103000511320040, 0x0101020011112200),
        GFb127::w64le(0x0000010001000402, 0x0001001500110045),
        GFb127::w64le(0x0103142910244280, 0x0100150011004402),
        GFb127::w64le(0x0101030106060A06, 0x0117153B1167458B),
        GFb127::w64le(0x153B1166468A0702, 0x17153B1166468A07),
        GFb127::w64le(0x0000010200001020, 0x0000000000000000),
        GFb127::w64le(0x0001020000112200, 0x0000010200001122),
        GFb127::w64le(0x0103000410300040, 0x0000000000000000),
        GFb127::w64le(0x0300041133004400, 0x0103000411330045),
        GFb127::w64le(0x0102142810204080, 0x0000000000000102),
        GFb127::w64le(0x02152A1122448A04, 0x0102152A1123478B),
        GFb127::w64le(0x1438106040800100, 0x0000000001030107),
        GFb127::w64le(0x3B1167448A07120E, 0x153B1166468A0702),
        GFb127::w64le(0x0000000000010200, 0x0000010200001122),
        GFb127::w64le(0x0001000401120040, 0x0001020000112200),
        GFb127::w64le(0x0000000103000400, 0x0103000411330044),
        GFb127::w64le(0x0106051B10644082, 0x0300041133004400),
        GFb127::w64le(0x0001020102040A04, 0x0102152A1122458A),
        GFb127::w64le(0x01120142040A0502, 0x02152A1122448A04),
        GFb127::w64le(0x030107040A06120E, 0x153B1167458B0016),
        GFb127::w64le(0x1167448A07130C2A, 0x3B1167448A07130C),
        GFb127::w64le(0x0001000400100040, 0x0000000000000000),
        GFb127::w64le(0x0100040011004400, 0x0001000400110044),
        GFb127::w64le(0x0106041810604080, 0x0000000000000001),
        GFb127::w64le(0x0604191166448802, 0x0106041911664588),
        GFb127::w64le(0x0010004000000100, 0x0000000000010005),
        GFb127::w64le(0x110045000003100A, 0x0011004501010415),
        GFb127::w64le(0x1060408001010200, 0x000000010107051A),
        GFb127::w64le(0x67458A03131C2A36, 0x1167448A07130C2A),
        GFb127::w64le(0x0000000001000400, 0x0001000400110044),
        GFb127::w64le(0x0102040910244080, 0x0100040011004400),
        GFb127::w64le(0x0000010106040802, 0x0106041911664489),
        GFb127::w64le(0x0409112644880302, 0x0604191166448802),
        GFb127::w64le(0x010005000002100A, 0x0011004500010115),
        GFb127::w64le(0x1025408002110800, 0x110045000003100B),
        GFb127::w64le(0x07050A02121E2A36, 0x1167458B0016163A),
        GFb127::w64le(0x458A03131D2A3222, 0x67458A03131D2A32),
        GFb127::w64le(0x0102040810204080, 0x0000000000000000),
        GFb127::w64le(0x0204081122448800, 0x0102040811224489),
        GFb127::w64le(0x0408102040800100, 0x0000000000000103),
        GFb127::w64le(0x0811234488011206, 0x0408112344890317),
        GFb127::w64le(0x1020408000010200, 0x000000000102050A),
        GFb127::w64le(0x22458A0003162A14, 0x1122458B03061D3B),
        GFb127::w64le(0x4080010103000400, 0x00000103050B1024),
        GFb127::w64le(0x8B0117153A12624E, 0x458A03131D2A3222),
        GFb127::w64le(0x0000000102040800, 0x0102040811224488),
        GFb127::w64le(0x0000010204080102, 0x0204081122448800),
        GFb127::w64le(0x0001030408001206, 0x0408112344880013),
        GFb127::w64le(0x0103040801130408, 0x0811234488011206),
        GFb127::w64le(0x02050A0002142A14, 0x1122458A0103172B),
        GFb127::w64le(0x050A0102152A1022, 0x22458A0003162B17),
        GFb127::w64le(0x0B0016163A16624E, 0x458B0016163A1662),
        GFb127::w64le(0x8117153B1066468B, 0x0B0117153B106646),
    ];
    */

    // Get the inverse of this value; the inverse of zero is formally
    // defined to be zero.
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

        unsafe {
            let m1 = vreinterpretq_u64_u32(vdupq_n_u32(0x55555555));
            let m2 = vreinterpretq_u64_u32(vdupq_n_u32(0x33333333));
            let m3 = vreinterpretq_u64_u32(vdupq_n_u32(0x0F0F0F0F));

            // Split a into ae and ao, then "squeeze" ae and ao:
            //   a = ae + ao*z
            //   sqrt(a) = sqrt(ae) + sqrt(ao)*sqrt(z)
            let a = self.0;
            let mut ae = vandq_u64(a, m1);
            let mut ao = vandq_u64(vshrq_n_u64(a, 1), m1);
            ae = vandq_u64(veorq_u64(ae, vshrq_n_u64(ae, 1)), m2);
            ao = vandq_u64(veorq_u64(ao, vshrq_n_u64(ao, 1)), m2);
            ae = vandq_u64(veorq_u64(ae, vshrq_n_u64(ae, 2)), m3);
            ao = vandq_u64(veorq_u64(ao, vshrq_n_u64(ao, 2)), m3);
            let z8 = vreinterpretq_u8_p128(0);
            ae = vreinterpretq_u64_u8(vuzp1q_u8(
                vreinterpretq_u8_u64(veorq_u64(ae, vshrq_n_u64(ae, 4))), z8));
            ao = vreinterpretq_u64_u8(vuzp1q_u8(
                vreinterpretq_u8_u64(veorq_u64(ao, vshrq_n_u64(ao, 4))), z8));

            // sqrt(ae) and sqrt(ao) have length 64 bits each.
            // We need to multiply sqrt(ao) by sqrt(z) = z^32 + z^64; no
            // reduction will be necessary. We currently have sqrt(ao)*z^64
            // in the 'ao' variable.
            let z32 = vreinterpretq_u32_p128(0);
            let ao32 = vreinterpretq_u32_u64(ao);
            let ao32 = veorq_u32(
                vextq_u32(z32, ao32, 2),
                vextq_u32(z32, ao32, 3));
            self.0 = veorq_u64(ae, vreinterpretq_u64_u32(ao32));
        }
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
        unsafe {
            // For i = 0 to 126, only z^0 = 1 has trace 1. However, we must
            // also take into account z^127 (our internal format is not
            // entirely reduced).
            let a = vreinterpretq_u32_u64(self.0);
            (vgetq_lane_u32(a, 0) & 1) ^ (vgetq_lane_u32(a, 3) >> 31)
        }
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

        unsafe {
            let m1 = vreinterpretq_u64_u32(vdupq_n_u32(0x55555555));
            let m2 = vreinterpretq_u64_u32(vdupq_n_u32(0x33333333));
            let m3 = vreinterpretq_u64_u32(vdupq_n_u32(0x0F0F0F0F));

            // We accumulate the odd-indexed bits in ao. We will ignore the
            // even-indexed bits in this variable, so we do not care what
            // values are written there.
            let z = vreinterpretq_u64_p128(0);
            let z8 = vreinterpretq_u8_p128(0);
            let mut ao = z;

            // We accumulate the extra values (square roots) ino e.
            let mut x = self.0;
            let mut e = z;

            // Do the split-and-squeeze 7 times, so that x is reduced to
            // a single bit.
            for _ in 0..7 {
                ao = veorq_u64(ao, x);
                x = vandq_u64(x, m1);
                x = vandq_u64(veorq_u64(x, vshrq_n_u64(x, 1)), m2);
                x = vandq_u64(veorq_u64(x, vshrq_n_u64(x, 2)), m3);
                x = vreinterpretq_u64_u8(vuzp1q_u8(
                    vreinterpretq_u8_u64(veorq_u64(x, vshrq_n_u64(x, 4))), z8));
                e = veorq_u64(e, x);
            }

            // len(x) = 1, hence H(x) = x. We now apply the halftrace of the
            // odd-indexed bits in ao.
            let mut d = veorq_u64(e, x);
            for i in (0..2).rev() {
                let mut mw = vcopyq_laneq_u64(ao, 0, ao, 1);
                ao = vextq_u64(ao, ao, 1);
                for j in (0..32).rev() {
                    let m = vcltzq_s64(vreinterpretq_s64_u64(mw));
                    mw = vshlq_n_u64(mw, 2);
                    d = veorq_u64(d, vandq_u64(m,
                        Self::HALFTRACE[(i << 5) + j].0));
                }
            }
            self.0 = d;
        }
    }

    // Get the halftrace of this value (in GF(2^127)).
    #[inline(always)]
    pub fn halftrace(self) -> Self {
        let mut x = self;
        x.set_halftrace();
        x
    }

    // Halftrace of z^(2*i+1) for i = 0 to 63.
    const HALFTRACE: [GFb127; 64] = [
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
        unsafe {
            // There are two possible internal representations of zero:
            // the full-zero value, or the modulus 1 + z^63 + z^127
            let c = vceqzq_u64(self.0);
            let d = vceqq_u64(self.0,
                transmute([1u64 + (1u64 << 63), 1u64 << 63]));
            let c = vandq_u64(c, vextq_u64(c, c, 1));
            let d = vandq_u64(d, vextq_u64(d, d, 1));
            vgetq_lane_u32::<0>(vreinterpretq_u32_u64(vorrq_u64(c, d)))
        }
    }

    #[inline(always)]
    pub fn encode(self) -> [u8; 16] {
        unsafe {
            let mut r = self;
            r.set_normalized();
            transmute(r.0)
        }
    }

    // Decode the value from bytes with implicit reduction modulo
    // z^127 + z^63 + 1. Input MUST be of length 16 bytes exactly.
    #[inline]
    fn set_decode16_reduce(&mut self, buf: &[u8]) {
        debug_assert!(buf.len() == 16);
        unsafe {
            self.0 = transmute(*<&[u8; 16]>::try_from(buf).unwrap());
        }
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
        let m = ((buf[15] >> 7) as u32).wrapping_sub(1);
        unsafe {
            self.0 = vandq_u64(self.0, vreinterpretq_u64_u32(vdupq_n_u32(m)));
        }
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
#[repr(align(32))]
pub struct GFb254([GFb127; 2]);

// Note: here we declared GFb254 to be 32-byte aligned; this helps with the
// AVX2 implementation of the lookup*_x2() functions. However it is not
// strictly necessary, since with AVX2, the lookup*_x2() functions use
// vmovdqu opcodes to read the table elements, and they tolerate unaligned
// accesses. The 16-byte alignment of the internal GFb127 elements must
// still be preserved.

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

        unsafe {
            // We access the inner uint64x2_t representations.
            let (a0, a1) = (self.0[0].0, self.0[1].0);
            let (b0, b1) = (rhs.0[0].0, rhs.0[1].0);

            let z = vreinterpretq_u64_p128(0);

            // c = a0*b0
            let ct0 = vreinterpretq_u64_p128(vmull_p64(
                vgetq_lane_u64(a0, 0), vgetq_lane_u64(b0, 0)));
            let ct2 = vreinterpretq_u64_p128(vmull_high_p64(
                vreinterpretq_p64_u64(a0), vreinterpretq_p64_u64(b0)));
            let ctx = vextq_u64(b0, b0, 1);
            let ct4 = vreinterpretq_u64_p128(vmull_p64(
                vgetq_lane_u64(a0, 0), vgetq_lane_u64(ctx, 0)));
            let ct5 = vreinterpretq_u64_p128(vmull_high_p64(
                vreinterpretq_p64_u64(a0), vreinterpretq_p64_u64(ctx)));
            let ct1 = veorq_u64(ct4, ct5);
            let c0 = veorq_u64(ct0, vextq_u64(z, ct1, 1));
            let c1 = veorq_u64(ct2, vextq_u64(ct1, z, 1));

            // d = a1*b1
            let dt0 = vreinterpretq_u64_p128(vmull_p64(
                vgetq_lane_u64(a1, 0), vgetq_lane_u64(b1, 0)));
            let dt2 = vreinterpretq_u64_p128(vmull_high_p64(
                vreinterpretq_p64_u64(a1), vreinterpretq_p64_u64(b1)));
            let dtx = vextq_u64(b1, b1, 1);
            let dt4 = vreinterpretq_u64_p128(vmull_p64(
                vgetq_lane_u64(a1, 0), vgetq_lane_u64(dtx, 0)));
            let dt5 = vreinterpretq_u64_p128(vmull_high_p64(
                vreinterpretq_p64_u64(a1), vreinterpretq_p64_u64(dtx)));
            let dt1 = veorq_u64(dt4, dt5);
            let d0 = veorq_u64(dt0, vextq_u64(z, dt1, 1));
            let d1 = veorq_u64(dt2, vextq_u64(dt1, z, 1));

            // e = (a0 + a1)*(b0 + b1)
            let a2 = veorq_u64(a0, a1);
            let b2 = veorq_u64(b0, b1);
            let et0 = vreinterpretq_u64_p128(vmull_p64(
                vgetq_lane_u64(a2, 0), vgetq_lane_u64(b2, 0)));
            let et2 = vreinterpretq_u64_p128(vmull_high_p64(
                vreinterpretq_p64_u64(a2), vreinterpretq_p64_u64(b2)));
            let etx = vextq_u64(b2, b2, 1);
            let et4 = vreinterpretq_u64_p128(vmull_p64(
                vgetq_lane_u64(a2, 0), vgetq_lane_u64(etx, 0)));
            let et5 = vreinterpretq_u64_p128(vmull_high_p64(
                vreinterpretq_p64_u64(a2), vreinterpretq_p64_u64(etx)));
            let et1 = veorq_u64(et4, et5);
            let e0 = veorq_u64(et0, vextq_u64(z, et1, 1));
            let e1 = veorq_u64(et2, vextq_u64(et1, z, 1));

            // r = a0*b0 + a1*b1 = c + d
            // s = (a0 + a1)*(b0 + 1) + a0*b0 = c + e
            let r0 = veorq_u64(c0, d0);
            let r1 = veorq_u64(c1, d1);
            let s0 = veorq_u64(c0, e0);
            let s1 = veorq_u64(c1, e1);

            // Reduce r and s simultaneously (interleaved).
            let x2 = vget_low_u64(r1);
            let y2 = vget_low_u64(s1);
            let x3 = vget_high_u64(r1);
            let y3 = vget_high_u64(s1);
            let f0 = veor_u64(x2, x3);
            let f1 = veor_u64(y2, y3);
            let g0 = vshl_n_u64(x3, 1);
            let g1 = vshl_n_u64(y3, 1);
            let h0 = vshl_n_u64(f0, 1);
            let h1 = vshl_n_u64(f1, 1);
            let j0 = vsri_n_u64(g0, x2, 63);
            let j1 = vsri_n_u64(g1, y2, 63);
            let k0 = veor_u64(f0, j0);
            let k1 = veor_u64(f1, j1);
            self.0[0].0 = veorq_u64(r0, vcombine_u64(h0, k0));
            self.0[1].0 = veorq_u64(s0, vcombine_u64(h1, k1));
        }
    }

    // Multiply this value by an element in GF(2^127).
    #[inline(always)]
    pub fn set_mul_b127(&mut self, rhs: &GFb127) {
        self.0[0] *= rhs;
        self.0[1] *= rhs;
    }

    // Multiply this value by an element in GF(2^127).
    #[inline(always)]
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

    // Multiply this value by sb = 1 + z^54 (an element of GF(2^127)).
    #[inline(always)]
    pub fn mul_b(self) -> Self {
        Self([ self.0[0].mul_b(), self.0[1].mul_b() ])
    }

    /* unused
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
    */

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
        unsafe {
            // We access the inner uint64x2_t representations.
            let (a0, a1) = (self.0[0].0, self.0[1].0);

            // Do all squarings on polynomials.
            let d0 = vreinterpretq_u64_p128(vmull_p64(
                vgetq_lane_u64(a0, 0), vgetq_lane_u64(a0, 0)));
            let d1 = vreinterpretq_u64_p128(vmull_high_p64(
                vreinterpretq_p64_u64(a0), vreinterpretq_p64_u64(a0)));
            let d2 = vreinterpretq_u64_p128(vmull_p64(
                vgetq_lane_u64(a1, 0), vgetq_lane_u64(a1, 0)));
            let d3 = vreinterpretq_u64_p128(vmull_high_p64(
                vreinterpretq_p64_u64(a1), vreinterpretq_p64_u64(a1)));

            // Do the two reductions in parallel.
            // Thanks to the interleaving, latency of 128-bit ops is
            // absorbed, so we can use the 128-bit version.
            let z = vreinterpretq_u64_p128(0);
            let d1z = vshlq_n_u64(d1, 1);           // e2z : e3z
            let d3z = vshlq_n_u64(d3, 1);           // e2z : e3z
            let f0 = vcopyq_laneq_u64(z, 1, d1, 1);  // 0   : e3
            let f1 = vcopyq_laneq_u64(z, 1, d3, 1);  // 0   : e3
            let g0 = vextq_u64(d1z, d1, 1);          // e3z : e2
            let g1 = vextq_u64(d3z, d3, 1);          // e3z : e2
            let r0 = veorq_u64(veorq_u64(d0, f0), veorq_u64(d1z, g0));
            let r1 = veorq_u64(veorq_u64(d2, f1), veorq_u64(d3z, g1));
            self.0[0] = GFb127(veorq_u64(r0, r1));
            self.0[1] = GFb127(r1);
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
        unsafe {
            let z = vreinterpretq_u64_p128(0);
            let one32 = vdupq_n_u32(1);
            let mut xj = vdupq_n_u32(j);
            let mut a0 = z;
            let mut a1 = z;
            let mut a2 = z;
            let mut a3 = z;
            for i in 0..16 {
                let m = vreinterpretq_u64_u32(vceqzq_u32(xj));
                xj = vsubq_u32(xj, one32);
                a0 = vbslq_u64(m, tab[(i * 2) + 0].0[0].0, a0);
                a1 = vbslq_u64(m, tab[(i * 2) + 0].0[1].0, a1);
                a2 = vbslq_u64(m, tab[(i * 2) + 1].0[0].0, a2);
                a3 = vbslq_u64(m, tab[(i * 2) + 1].0[1].0, a3);
            }
            [
                Self([ GFb127(a0), GFb127(a1) ]),
                Self([ GFb127(a2), GFb127(a3) ]),
            ]
        }
    }

    // Constant-time table lookup: given a table of 16 field elements, and
    // an index `j` in the 0 to 7 range, return the elements of index
    // `j*2` and `j*2+1`. If `j` is not in the 0 to 7 range (inclusive),
    // then this returns two zeros.
    #[inline(always)]
    pub fn lookup8_x2(tab: &[Self; 16], j: u32) -> [Self; 2] {
        unsafe {
            // Saturate out-of-range j values to 8-15.
            let k = j >> 3;
            let j2 = (j & 7) | ((k | k.wrapping_neg()) >> 28);
            let xj = vreinterpret_u8_u64(
                vshl_u64(vdup_n_u64(0xFF), vdup_n_s64((j2 as i64) << 3)));

            let z = vreinterpretq_u64_p128(0);
            let mut a0 = z;
            let mut a1 = z;
            let mut a2 = z;
            let mut a3 = z;

            let m = vreinterpretq_u64_u8(vdupq_lane_u8(xj, 0));
            a0 = vbslq_u64(m, tab[(0 * 2) + 0].0[0].0, a0);
            a1 = vbslq_u64(m, tab[(0 * 2) + 0].0[1].0, a1);
            a2 = vbslq_u64(m, tab[(0 * 2) + 1].0[0].0, a2);
            a3 = vbslq_u64(m, tab[(0 * 2) + 1].0[1].0, a3);
            let m = vreinterpretq_u64_u8(vdupq_lane_u8(xj, 1));
            a0 = vbslq_u64(m, tab[(1 * 2) + 0].0[0].0, a0);
            a1 = vbslq_u64(m, tab[(1 * 2) + 0].0[1].0, a1);
            a2 = vbslq_u64(m, tab[(1 * 2) + 1].0[0].0, a2);
            a3 = vbslq_u64(m, tab[(1 * 2) + 1].0[1].0, a3);
            let m = vreinterpretq_u64_u8(vdupq_lane_u8(xj, 2));
            a0 = vbslq_u64(m, tab[(2 * 2) + 0].0[0].0, a0);
            a1 = vbslq_u64(m, tab[(2 * 2) + 0].0[1].0, a1);
            a2 = vbslq_u64(m, tab[(2 * 2) + 1].0[0].0, a2);
            a3 = vbslq_u64(m, tab[(2 * 2) + 1].0[1].0, a3);
            let m = vreinterpretq_u64_u8(vdupq_lane_u8(xj, 3));
            a0 = vbslq_u64(m, tab[(3 * 2) + 0].0[0].0, a0);
            a1 = vbslq_u64(m, tab[(3 * 2) + 0].0[1].0, a1);
            a2 = vbslq_u64(m, tab[(3 * 2) + 1].0[0].0, a2);
            a3 = vbslq_u64(m, tab[(3 * 2) + 1].0[1].0, a3);
            let m = vreinterpretq_u64_u8(vdupq_lane_u8(xj, 4));
            a0 = vbslq_u64(m, tab[(4 * 2) + 0].0[0].0, a0);
            a1 = vbslq_u64(m, tab[(4 * 2) + 0].0[1].0, a1);
            a2 = vbslq_u64(m, tab[(4 * 2) + 1].0[0].0, a2);
            a3 = vbslq_u64(m, tab[(4 * 2) + 1].0[1].0, a3);
            let m = vreinterpretq_u64_u8(vdupq_lane_u8(xj, 5));
            a0 = vbslq_u64(m, tab[(5 * 2) + 0].0[0].0, a0);
            a1 = vbslq_u64(m, tab[(5 * 2) + 0].0[1].0, a1);
            a2 = vbslq_u64(m, tab[(5 * 2) + 1].0[0].0, a2);
            a3 = vbslq_u64(m, tab[(5 * 2) + 1].0[1].0, a3);
            let m = vreinterpretq_u64_u8(vdupq_lane_u8(xj, 6));
            a0 = vbslq_u64(m, tab[(6 * 2) + 0].0[0].0, a0);
            a1 = vbslq_u64(m, tab[(6 * 2) + 0].0[1].0, a1);
            a2 = vbslq_u64(m, tab[(6 * 2) + 1].0[0].0, a2);
            a3 = vbslq_u64(m, tab[(6 * 2) + 1].0[1].0, a3);
            let m = vreinterpretq_u64_u8(vdupq_lane_u8(xj, 7));
            a0 = vbslq_u64(m, tab[(7 * 2) + 0].0[0].0, a0);
            a1 = vbslq_u64(m, tab[(7 * 2) + 0].0[1].0, a1);
            a2 = vbslq_u64(m, tab[(7 * 2) + 1].0[0].0, a2);
            a3 = vbslq_u64(m, tab[(7 * 2) + 1].0[1].0, a3);

            [
                Self([ GFb127(a0), GFb127(a1) ]),
                Self([ GFb127(a2), GFb127(a3) ]),
            ]
        }
    }

    // Constant-time table lookup: given a table of 8 field elements, and
    // an index `j` in the 0 to 3 range, return the elements of index
    // `j*2` and `j*2+1`. If `j` is not in the 0 to 3 range (inclusive),
    // then this returns two zeros.
    #[inline]
    pub fn lookup4_x2(tab: &[Self; 8], j: u32) -> [Self; 2] {
        unsafe {
            let z = vreinterpretq_u64_p128(0);
            let z32 = vreinterpretq_u32_p128(0);
            let one32 = vdupq_n_u32(1);
            let xj = vdupq_n_u32(j);
            let mut xi = z32;
            let mut a0 = z;
            let mut a1 = z;
            let mut a2 = z;
            let mut a3 = z;
            for i in 0..4 {
                let m = vreinterpretq_u64_u32(vceqq_u32(xi, xj));
                xi = vaddq_u32(xi, one32);
                a0 = vbslq_u64(m, tab[(i * 2) + 0].0[0].0, a0);
                a1 = vbslq_u64(m, tab[(i * 2) + 0].0[1].0, a1);
                a2 = vbslq_u64(m, tab[(i * 2) + 1].0[0].0, a2);
                a3 = vbslq_u64(m, tab[(i * 2) + 1].0[1].0, a3);
            }
            [
                Self([ GFb127(a0), GFb127(a1) ]),
                Self([ GFb127(a2), GFb127(a3) ]),
            ]
        }
    }

    /// Constant-time table lookup, short table. This is similar to
    /// `lookup16_x2()`, except that there are only four pairs of values
    /// (8 elements of GF(2^254)), and the pair index MUST be in the
    /// proper range (if the index is not in the range, an unpredictable
    /// value is returned).
    #[inline]
    pub fn lookup4_x2_nocheck(tab: &[Self; 8], j: u32) -> [Self; 2] {
        unsafe {
            let one32 = vdupq_n_u32(1);
            let two32 = vdupq_n_u32(2);
            let xj = vdupq_n_u32(j);
            let xm0 = vreinterpretq_u64_u32(
                vceqq_u32(vandq_u32(xj, one32), one32));
            let xm1 = vreinterpretq_u64_u32(
                vceqq_u32(vandq_u32(xj, two32), two32));
            let a0 = vbslq_u64(xm1,
                vbslq_u64(xm0, tab[6].0[0].0, tab[4].0[0].0),
                vbslq_u64(xm0, tab[2].0[0].0, tab[0].0[0].0));
            let a1 = vbslq_u64(xm1,
                vbslq_u64(xm0, tab[6].0[1].0, tab[4].0[1].0),
                vbslq_u64(xm0, tab[2].0[1].0, tab[0].0[1].0));
            let a2 = vbslq_u64(xm1,
                vbslq_u64(xm0, tab[7].0[0].0, tab[5].0[0].0),
                vbslq_u64(xm0, tab[3].0[0].0, tab[1].0[0].0));
            let a3 = vbslq_u64(xm1,
                vbslq_u64(xm0, tab[7].0[1].0, tab[5].0[1].0),
                vbslq_u64(xm0, tab[3].0[1].0, tab[1].0[1].0));
            [
                Self([ GFb127(a0), GFb127(a1) ]),
                Self([ GFb127(a2), GFb127(a3) ]),
            ]
        }
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
    use crate::sha2::Sha256;

    /*
    fn print(name: &str, v: GFb127) {
        print!("{} = K(0)", name);
        for i in 0..128 {
            if ((v.0[i >> 6] >> (i & 63)) & 1) != 0 {
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
        for i in 0..16 {
            va[i] = 0xFF;
            vb[i] = 0xFF;
        }
        check_gfb127_ops(&va, &vb);
        va[15] &= 0x7F;
        vb[15] &= 0x7F;
        check_gfb127_ops(&va, &vb);

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
