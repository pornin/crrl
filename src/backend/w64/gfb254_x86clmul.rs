use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::convert::TryFrom;

use core::arch::x86_64::*;

/// Element of GF(2^127), using modulus 1 + z^63 + z^127.
#[derive(Clone, Copy, Debug)]
pub struct GFb127(__m128i);

// A structure that contains two elements of GF(2^127); this is defined so
// that we may make 32-byte aligned arrays of `GFb127`.
#[repr(align(32))]
struct GFb127x2([GFb127; 2]);

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
        unsafe { Self(core::mem::transmute([ x0, x1 ])) }
    }

    /// Make a value out of two 64-bit limbs (least significant limb first).
    /// The value is implicitly reduced to 127 bits.
    pub fn from_w64le(x0: u64, x1: u64) -> Self {
        unsafe { Self(_mm_set_epi64x(x1 as i64, x0 as i64)) }
    }

    // Split this value into two 64-bit limbs.
    fn to_limbs(self) -> [u64; 2] {
        unsafe {
            [
                _mm_cvtsi128_si64(self.0) as u64,
                _mm_cvtsi128_si64(_mm_bsrli_si128(self.0, 8)) as u64,
            ]
        }
    }

    // Normalize this value and split it into two 64-bit limbs.
    #[inline(always)]
    fn normalize_limbs(self) -> [u64; 2] {
        let mut x = self.to_limbs();
        let h = x[1] & 0x8000000000000000;
        x[0] ^= h ^ (x[1] >> 63);
        x[1] ^= h;
        x
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
            self.0 = _mm_xor_si128(self.0, rhs.0);
        }
    }

    // Subtraction is the same thing as addition in binary fields.

    #[inline(always)]
    pub fn set_cond(&mut self, a: &Self, ctl: u32) {
        unsafe {
            let cw = _mm_set1_epi32(ctl as i32);
            self.0 = _mm_blendv_epi8(self.0, a.0, cw);
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
            let cw = _mm_set1_epi32(ctl as i32);
            a.0 = _mm_blendv_epi8(xa, xb, cw);
            b.0 = _mm_blendv_epi8(xb, xa, cw);
        }
    }

    // Multiply this value by sb = 1 + z^27.
    #[inline(always)]
    pub fn set_mul_sb(&mut self) {
        unsafe {
            let a = self.0;
            let e0 = _mm_slli_epi64(a, 27);
            let e1 = _mm_srli_epi64(a, 37);
            let c0 = _mm_xor_si128(
                _mm_xor_si128(a, e0),
                _mm_bslli_si128(e1, 8));
            let c1 = _mm_slli_epi64(_mm_bsrli_si128(e1, 8), 1);
            self.0 = _mm_xor_si128(c0, _mm_blend_epi16(e1, c1, 0x0F));
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
            let e0 = _mm_slli_epi64(a, 54);
            let e1 = _mm_srli_epi64(a, 10);
            let c0 = _mm_xor_si128(
                _mm_xor_si128(a, e0),
                _mm_bslli_si128(e1, 8));
            let c1 = _mm_slli_epi64(_mm_bsrli_si128(e1, 8), 1);
            self.0 = _mm_xor_si128(c0, _mm_blend_epi16(e1, c1, 0x0F));
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
            let e0 = _mm_slli_epi64(a, 44);
            let e1 = _mm_srli_epi64(a, 20);

            // e0 = f0:f1
            // e1 = f2:f3     note: len(f2) <= 44, len(f3) <= 44
            // r = a + f0*z^64 + f1*z^128 + f2*z^128 + f3*z^192
            //   = a + f0*z^64 + (f1 + f2)*z + (f1 + f2)*z^64
            //       + (f3*z)*z^64 + f3*z + f3*z^64
            //   = a + f0*z^64 + (f1 + f2 + f3)*z
            //       + (f1 + f2 + f3)*z^64 + (f3*z)*z^64

            // g = (f1 + f2 + f3) + f3*z^64
            let g = _mm_xor_si128(e1,
                _mm_bsrli_si128(_mm_xor_si128(e0, e1), 8));

            // h = (f1 + f2 + f3)*z + (f3*z)*z^64
            let h = _mm_xor_si128(
                _mm_slli_epi64(g, 1),
                _mm_bslli_si128(_mm_srli_epi64(g, 63), 8));

            // r = a + f0*z^64 + lo(g)*z^64 + h
            let r = _mm_xor_si128(
                _mm_xor_si128(a, h),
                _mm_bslli_si128(_mm_xor_si128(e0, g), 8));
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
            let a = _mm_xor_si128(
                a, _mm_shuffle_epi32(_mm_slli_epi64(a, 63), 0x44));
            // Simple shift (lsb is now implicitly zero).
            self.0 = _mm_or_si128(
                _mm_srli_epi64(a, 1),
                _mm_bsrli_si128(_mm_slli_epi64(a, 63), 8));
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
            let c0 = _mm_clmulepi64_si128(a, b, 0x00);
            let c1 = _mm_xor_si128(
                _mm_clmulepi64_si128(a, b, 0x01),
                _mm_clmulepi64_si128(a, b, 0x10));
            let c2 = _mm_clmulepi64_si128(a, b, 0x11);

            // a*b = d0 + d1*z^128
            let d0 = _mm_xor_si128(c0, _mm_bslli_si128(c1, 8));
            let d1 = _mm_xor_si128(_mm_bsrli_si128(c1, 8), c2);

            // Reduction: z^128 = z^64 + z
            // We write:
            //   d0 = e0 + e1*z^64
            //   d1 = e2 + e3*z^64
            // We note that len(e3) <= 63.
            //   (e2 + e3*z^64)*z^128
            //    = (e2 + e3 + e3*z^64)*z + (e2 + e3)*z^64

            // f = e2 + e3 + e3*z^64
            // g = (e2 + e3)*z^64
            let f = _mm_xor_si128(d1, _mm_bsrli_si128(d1, 8));
            let g = _mm_bslli_si128(f, 8);

            // h = z*f
            let h = _mm_or_si128(
                _mm_slli_epi64(f, 1),
                _mm_bslli_si128(_mm_srli_epi64(f, 63), 8));

            self.0 = _mm_xor_si128(d0, _mm_xor_si128(g, h));
        }
    }

    // Square this value (in place).
    #[inline(always)]
    pub fn set_square(&mut self) {
        unsafe {
            let a = self.0;

            // a^2 = d0 + d1*z^128
            let d0 = _mm_clmulepi64_si128(a, a, 0x00);
            let d1 = _mm_clmulepi64_si128(a, a, 0x11);

            // Reduction: z^128 = z^64 + z
            // We write:
            //   d0 = e0 + e1*z^64
            //   d1 = e2 + e3*z^64
            // We note that len(e3) <= 63.
            //   (e2 + e3*z^64)*z^128
            //    = (e2 + e3 + e3*z^64)*z + (e2 + e3)*z^64
            // Since d1 is a square in GF(2)[z], its odd-indexed bits are
            // all zero, and (e2 + e3)*z cannot "bleed" a non-zero bit into
            // the z^64 half.

            // f = e2 + e3 + e3*z^64
            // g = (e2 + e3)*z^64
            let f = _mm_xor_si128(d1, _mm_bsrli_si128(d1, 8));
            let g = _mm_bslli_si128(f, 8);

            // h = z*f
            let h = _mm_slli_epi64(f, 1);

            self.0 = _mm_xor_si128(d0, _mm_xor_si128(g, h));

            /* unused: alternate implementation that uses pshufb. It seems
               slower than the code that uses pclmulqdq in general, but it
               is slightly faster (on Intel Skylake CPUs) in long sequences
               of successive squarings, because it has a lower latency.
               The GFb254::set_square() function uses a variant of this code
               when AVX2 is enabled, because it allows computation of two
               GF(2^127) squarings in parallel.

            let a = self.0;

            // Square the polynomial by "expanding" the bits.
            let m16 = _mm_set1_epi8(0x0F);
            let shk = _mm_setr_epi8(
                0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15,
                0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55);
            let t0 = _mm_shuffle_epi8(shk,
                _mm_and_si128(a, m16));
            let t1 = _mm_shuffle_epi8(shk,
                _mm_and_si128(_mm_srli_epi16(a, 4), m16));
            let d0 = _mm_unpacklo_epi8(t0, t1);
            let d1 = _mm_unpackhi_epi8(t0, t1);

            // Reduction.
            let f = _mm_xor_si128(d1, _mm_bsrli_si128(d1, 8));
            let g = _mm_bslli_si128(f, 8);
            let h = _mm_slli_epi64(f, 1);

            self.0 = _mm_xor_si128(d0, _mm_xor_si128(g, h));
            */
        }
    }

    // Square this value.
    #[inline(always)]
    pub fn square(self) -> Self {
        let mut r = self;
        r.set_square();
        r
    }

    // Square this value (in place) (alternate code with a lower
    // throughput but also a lower latency than the usual code).
    #[inline(always)]
    fn set_square_alt(&mut self) {
        unsafe {
            let a = self.0;

            // Square the polynomial by "expanding" the bits.
            let m16 = _mm_set1_epi8(0x0F);
            let shk = _mm_setr_epi8(
                0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15,
                0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55);
            let t0 = _mm_shuffle_epi8(shk,
                _mm_and_si128(a, m16));
            let t1 = _mm_shuffle_epi8(shk,
                _mm_and_si128(_mm_srli_epi16(a, 4), m16));
            let d0 = _mm_unpacklo_epi8(t0, t1);
            let d1 = _mm_unpackhi_epi8(t0, t1);

            // Reduction.
            let f = _mm_xor_si128(d1, _mm_bsrli_si128(d1, 8));
            let g = _mm_bslli_si128(f, 8);
            let h = _mm_slli_epi64(f, 1);

            self.0 = _mm_xor_si128(d0, _mm_xor_si128(g, h));
        }
    }

    // Square this value (alternate code with a lower throughput but also
    // a lower latency than the usual code).
    #[allow(dead_code)]
    #[inline(always)]
    fn square_alt(self) -> Self {
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
            self.set_square_alt();
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
            let a = self.0;
            let h = _mm_and_si128(a, _mm_set_epi64x((1u64 << 63) as i64, 0));
            let v = _mm_shuffle_epi32(h, 0x4E);
            self.0 = _mm_xor_si128(
                _mm_xor_si128(a, h),
                _mm_xor_si128(v, _mm_srli_epi64(v, 63)));
        }
    }

    // Invert this value; if this value is zero, then it stays at zero.
    pub fn set_invert(&mut self) {
        // We use Itoh-Tsujii, with optimized sequences of squarings.
        // We have:
        //   1/a = a^(2^127 - 2)
        //       = (a^2)^(2^126 - 1)
        // We use an addition chain for the exponent:
        //   1 -> 2 -> 3 -> 6 -> 7 -> 14 -> 28 -> 42 -> 84 -> 126
        // Since raising to the power 2^m is linear in GF(2^127), for
        // any integer m >= 1, we can optimize that operation with a
        // table of 128 values. Each such table has size 2 kB. We
        // apply this optimization for m = 14 and 42.

        let a1 = self.square();
        let a2 = a1 * a1.square();
        let a3 = a1 * a2.square();
        let a6 = a3 * a3.xsquare(3);
        let a7 = a1 * a6.square();
        let a14 = a7 * a7.xsquare(7);
        let a28 = a14 * a14.frob(&Self::FROB14);
        let a42 = a14 * a28.frob(&Self::FROB14);
        let a84 = a42 * a42.frob(&Self::FROB42);
        let a126 = a42 * a84.frob(&Self::FROB42);
        *self = a126;
    }

    #[inline]
    fn frob(self, tab: &[GFb127x2; 64]) -> Self {
        #[cfg(not(target_feature = "avx2"))]
        unsafe {
            let mut a = self.0;
            let mut d = _mm_setzero_si128();
            for i in (0..4).rev() {
                let mut mw = _mm_shuffle_epi32(a, 0xFF);
                a = _mm_bslli_si128(a, 4);
                for j in (0..16).rev() {
                    let m = _mm_srai_epi32(mw, 31);
                    mw = _mm_slli_epi32(mw, 1);
                    d = _mm_xor_si128(d,
                        _mm_and_si128(m, tab[(i << 4) + j].0[1].0));
                    let m = _mm_srai_epi32(mw, 31);
                    mw = _mm_slli_epi32(mw, 1);
                    d = _mm_xor_si128(d,
                        _mm_and_si128(m, tab[(i << 4) + j].0[0].0));
                }
            }
            Self(d)
        }
        #[cfg(target_feature = "avx2")]
        unsafe {
            // With the AVX2 implementation, we handle even-indexed bits
            // in the low lane, and odd-indexed bits in the high lane.
            // The two lanes are XORed together at the end.
            let a = self.0;
            let mut ya = _mm256_setr_m128i(_mm_slli_epi32(a, 1), a);
            let mut yd = _mm256_setzero_si256();
            for i in (0..4).rev() {
                let mut ymw = _mm256_shuffle_epi32(ya, 0xFF);
                ya = _mm256_bslli_epi128(ya, 4);
                for j in (0..16).rev() {
                    yd = _mm256_xor_si256(yd,
                        _mm256_maskload_epi32(
                            core::mem::transmute(&tab[(i << 4) + j].0), ymw));
                    ymw = _mm256_slli_epi32(ymw, 2);
                }
            }
            let d = _mm_xor_si128(
                _mm256_castsi256_si128(yd), _mm256_extracti128_si256(yd, 1));
            Self(d)
        }
    }

    // z^(i*2^14) for i = 0 to 127.
    const FROB14: [GFb127x2; 64] = [
        GFb127x2([ GFb127::w64le(0x0000000000000001, 0x0000000000000000),
                   GFb127::w64le(0x0000000100010114, 0x0000000000000001) ]),
        GFb127x2([ GFb127::w64le(0x0000000100010112, 0x0000000000000000),
                   GFb127::w64le(0x0000000700070768, 0x0000000100010113) ]),
        GFb127x2([ GFb127::w64le(0x0000000100010104, 0x0000000000000001),
                   GFb127::w64le(0x0000001100111052, 0x0000000000000010) ]),
        GFb127x2([ GFb127::w64le(0x0000001700171648, 0x0000000100010113),
                   GFb127::w64le(0x0000007100717784, 0x0000001000101131) ]),
        GFb127x2([ GFb127::w64le(0x0000000100010012, 0x0000000000000000),
                   GFb127::w64le(0x0000010701061368, 0x0000000100010013) ]),
        GFb127x2([ GFb127::w64le(0x0000010101001304, 0x0000000000000001),
                   GFb127::w64le(0x0000071107167852, 0x0000010001011310) ]),
        GFb127x2([ GFb127::w64le(0x0000011701161248, 0x0000000100010013),
                   GFb127::w64le(0x0000117111612584, 0x0000001000100131) ]),
        GFb127x2([ GFb127::w64le(0x0000170117174812, 0x0000010001011300),
                   GFb127::w64le(0x0000700770719768, 0x0000100110103113) ]),
        GFb127x2([ GFb127::w64le(0x0000000100000104, 0x0000000000000001),
                   GFb127::w64le(0x0001001001051052, 0x0000000000010010) ]),
        GFb127x2([ GFb127::w64le(0x0001001601051648, 0x0000000100010113),
                   GFb127::w64le(0x0007007607197784, 0x0001001101031131) ]),
        GFb127x2([ GFb127::w64le(0x0001000001050012, 0x0000000000010000),
                   GFb127::w64le(0x0011011611541368, 0x0000000100110013) ]),
        GFb127x2([ GFb127::w64le(0x0017011617481304, 0x0001000101130001),
                   GFb127::w64le(0x0071076070927852, 0x0010011010301310) ]),
        GFb127x2([ GFb127::w64le(0x0001011601041248, 0x0000000100010013),
                   GFb127::w64le(0x0107107702092584, 0x0001001100030131) ]),
        GFb127x2([ GFb127::w64le(0x0101160104134812, 0x0000010001001300),
                   GFb127::w64le(0x0711771108239768, 0x0100110003003113) ]),
        GFb127x2([ GFb127::w64le(0x0117011712480104, 0x0001000100130001),
                   GFb127::w64le(0x1170117124811052, 0x0010001001300010) ]),
        GFb127x2([ GFb127::w64le(0x1700170149171648, 0x0100010013010113),
                   GFb127::w64le(0x7000700790717784, 0x1000100130101131) ]),
        GFb127x2([ GFb127::w64le(0x0000000000010012, 0x0000000000000000),
                   GFb127::w64le(0x0001001301061368, 0x0000000000010012) ]),
        GFb127x2([ GFb127::w64le(0x0001001301001304, 0x0000000000000000),
                   GFb127::w64le(0x0007007907167850, 0x0001001301011316) ]),
        GFb127x2([ GFb127::w64le(0x0001001301161248, 0x0000000000010012),
                   GFb127::w64le(0x0011012311612584, 0x0000000000100120) ]),
        GFb127x2([ GFb127::w64le(0x0017014917174810, 0x0001001301011316),
                   GFb127::w64le(0x0071078370719748, 0x0010013010103172) ]),
        GFb127x2([ GFb127::w64le(0x0001001300000104, 0x0000000000000000),
                   GFb127::w64le(0x0107137801051050, 0x0001001300010116) ]),
        GFb127x2([ GFb127::w64le(0x0101131201051648, 0x0000000000010012),
                   GFb127::w64le(0x0711782407197584, 0x0100130101031720) ]),
        GFb127x2([ GFb127::w64le(0x0117124801050010, 0x0001001300010116),
                   GFb127::w64le(0x1170249211541348, 0x0010013000111172) ]),
        GFb127x2([ GFb127::w64le(0x1700490417481104, 0x0100130101131600),
                   GFb127::w64le(0x7000900870925850, 0x1000300310307316) ]),
        GFb127x2([ GFb127::w64le(0x0001001201041248, 0x0000000000010012),
                   GFb127::w64le(0x0002002502092584, 0x0000000100020121) ]),
        GFb127x2([ GFb127::w64le(0x0004004904134810, 0x0001001301011317),
                   GFb127::w64le(0x000800950821974A, 0x0003003103063174) ]),
        GFb127x2([ GFb127::w64le(0x0012010512480104, 0x0000000100120001),
                   GFb127::w64le(0x0024021924811050, 0x0001000301210107) ]),
        GFb127x2([ GFb127::w64le(0x004804054915164A, 0x0013010113170004),
                   GFb127::w64le(0x00920855905175A4, 0x0030031130711741) ]),
        GFb127x2([ GFb127::w64le(0x0104124800010010, 0x0001001300010117),
                   GFb127::w64le(0x020825970104134A, 0x0003013101071074) ]),
        GFb127x2([ GFb127::w64le(0x0412480101001104, 0x0100130001011701),
                   GFb127::w64le(0x0824971105165A50, 0x0301310007107507) ]),
        GFb127x2([ GFb127::w64le(0x124901170114124A, 0x0013000101170104),
                   GFb127::w64le(0x24901171114125A4, 0x0130001011701041) ]),
        GFb127x2([ GFb127::w64le(0x4900170115174A10, 0x1300010017010517),
                   GFb127::w64le(0x900070075071B74A, 0x3000100170105174) ]),
        GFb127x2([ GFb127::w64le(0x0000000100000104, 0x0000000000000000),
                   GFb127::w64le(0x0001001001051050, 0x0000000100000105) ]),
        GFb127x2([ GFb127::w64le(0x0001001601051648, 0x0000000000000001),
                   GFb127::w64le(0x00070074071B75A2, 0x000100170105174A) ]),
        GFb127x2([ GFb127::w64le(0x0001000001050010, 0x0000000100000105),
                   GFb127::w64le(0x0011011611541348, 0x0000001000001051) ]),
        GFb127x2([ GFb127::w64le(0x00170114174A1122, 0x000100170105175A),
                   GFb127::w64le(0x0071074070B25A30, 0x00100171105175A5) ]),
        GFb127x2([ GFb127::w64le(0x0001011601041248, 0x0000000000000001),
                   GFb127::w64le(0x01071075020B25A2, 0x000101170104124A) ]),
        GFb127x2([ GFb127::w64le(0x0101160104134810, 0x0000000100000005),
                   GFb127::w64le(0x071175110A21B148, 0x0100171105175A51) ]),
        GFb127x2([ GFb127::w64le(0x01170115124A0122, 0x000101170104125A),
                   GFb127::w64le(0x1170115124A11230, 0x00101171104124A5) ]),
        GFb127x2([ GFb127::w64le(0x170015014B153048, 0x0100170105175A01),
                   GFb127::w64le(0x70005005B05115A2, 0x100070075071B74A) ]),
        GFb127x2([ GFb127::w64le(0x0000000000010010, 0x0000000100000105),
                   GFb127::w64le(0x0001001301041348, 0x0001001001051050) ]),
        GFb127x2([ GFb127::w64le(0x0001001101021122, 0x000100170104175B),
                   GFb127::w64le(0x0005005B05105A32, 0x00070074071B75A3) ]),
        GFb127x2([ GFb127::w64le(0x0001001301141248, 0x0001000001050000),
                   GFb127::w64le(0x00110121114325A2, 0x001101171155125B) ]),
        GFb127x2([ GFb127::w64le(0x0015014B15314812, 0x00170104175A0013),
                   GFb127::w64le(0x005105A35011B168, 0x0071074070B25A30) ]),
        GFb127x2([ GFb127::w64le(0x0001001100020122, 0x000101170105125B),
                   GFb127::w64le(0x0105135A01031232, 0x01071075020B25A3) ]),
        GFb127x2([ GFb127::w64le(0x0101111203053048, 0x0101170105125B00),
                   GFb127::w64le(0x05115A24011917A2, 0x071175100A20B15B) ]),
        GFb127x2([ GFb127::w64le(0x0115124A01230012, 0x01170105125A0013),
                   GFb127::w64le(0x115024B213341368, 0x1170105125A00130) ]),
        GFb127x2([ GFb127::w64le(0x15004B04314A1322, 0x170005005B05015B),
                   GFb127::w64le(0x5000B00A10B27A32, 0x70005005B05115A3) ]),
        GFb127x2([ GFb127::w64le(0x0001001201041248, 0x0000000000000000),
                   GFb127::w64le(0x00020025020B25A0, 0x000100120105125B) ]),
        GFb127x2([ GFb127::w64le(0x0004004904134810, 0x0000000000010013),
                   GFb127::w64le(0x000A00B30A23B166, 0x0005005B05115A32) ]),
        GFb127x2([ GFb127::w64le(0x00120105124A0120, 0x000100120105125B),
                   GFb127::w64le(0x0024021924A11210, 0x00100120105025A3) ]),
        GFb127x2([ GFb127::w64le(0x004A04234B173066, 0x0005005B05015B02),
                   GFb127::w64le(0x00B20A35B0711740, 0x005105A25010B17B) ]),
        GFb127x2([ GFb127::w64le(0x0104124800010010, 0x0000000000010013),
                   GFb127::w64le(0x020A25B101061166, 0x0105125A00030132) ]),
        GFb127x2([ GFb127::w64le(0x0412480101021120, 0x000100120005015B),
                   GFb127::w64le(0x0A24B11307107410, 0x05105A25010A17A3) ]),
        GFb127x2([ GFb127::w64le(0x124B013101161066, 0x0105125A00130002),
                   GFb127::w64le(0x24B0131111630740, 0x105025B20035127B) ]),
        GFb127x2([ GFb127::w64le(0x4B00310317316610, 0x05005B05015A0213),
                   GFb127::w64le(0xB000100170115166, 0x5000B00A10B27A32) ]),
        GFb127x2([ GFb127::w64le(0x0000000100020120, 0x000100120105125A),
                   GFb127::w64le(0x0001001201011212, 0x00020025020B25A0) ]),
        GFb127x2([ GFb127::w64le(0x0003003003073066, 0x0005005A05125B06),
                   GFb127::w64le(0x000100160117174A, 0x000A00B30A22B174) ]),
        GFb127x2([ GFb127::w64le(0x0001000201210012, 0x00120105125A0000),
                   GFb127::w64le(0x0013011013161146, 0x0025020A25A00106) ]),
        GFb127x2([ GFb127::w64le(0x003103163164112A, 0x005A05135B070114),
                   GFb127::w64le(0x00110162105074B2, 0x00B20A35B0711740) ]),
        GFb127x2([ GFb127::w64le(0x0003013001061066, 0x0105125B00000106),
                   GFb127::w64le(0x010112170005054A, 0x020A25B101071174) ]),
        GFb127x2([ GFb127::w64le(0x0301300106116612, 0x05125B0000010600),
                   GFb127::w64le(0x0113171104015B46, 0x0A25B10007117506) ]),
        GFb127x2([ GFb127::w64le(0x013101171064032A, 0x125B000101070114),
                   GFb127::w64le(0x13101171064132B2, 0x25B0001010701140) ]),
        GFb127x2([ GFb127::w64le(0x3100170165173A66, 0x5B00010007011506),
                   GFb127::w64le(0x900070075071B74B, 0x3000100170105174) ]),
    ];

    // z^(i*2^42) for i = 0 to 127.
    const FROB42: [GFb127x2; 64] = [
        GFb127x2([ GFb127::w64le(0x0000000000000001, 0x0000000000000000),
                   GFb127::w64le(0x0000000100000110, 0x0000000000000000) ]),
        GFb127x2([ GFb127::w64le(0x0000000000010100, 0x0000000000000001),
                   GFb127::w64le(0x0001010001111000, 0x0000000100000110) ]),
        GFb127x2([ GFb127::w64le(0x0000000100010002, 0x0000000000000001),
                   GFb127::w64le(0x0001011201100220, 0x0000000100000111) ]),
        GFb127x2([ GFb127::w64le(0x0001010101020202, 0x0000000100000103),
                   GFb127::w64le(0x0013131312222222, 0x0000001300001230) ]),
        GFb127x2([ GFb127::w64le(0x0000000100000006, 0x0000000000000000),
                   GFb127::w64le(0x0000011600000660, 0x0000000000000001) ]),
        GFb127x2([ GFb127::w64le(0x0001010000060600, 0x0000000100000006),
                   GFb127::w64le(0x0117160006666002, 0x0000011600010761) ]),
        GFb127x2([ GFb127::w64le(0x000100040006000C, 0x0000000100000007),
                   GFb127::w64le(0x0116044C06600CC2, 0x0000011700010775) ]),
        GFb127x2([ GFb127::w64le(0x01040404060C0C0E, 0x000001050001070A),
                   GFb127::w64le(0x124848486CCCCCEA, 0x0000125A00137FA0) ]),
        GFb127x2([ GFb127::w64le(0x0000000000000014, 0x0000000000000001),
                   GFb127::w64le(0x0000001400001540, 0x0000000100000110) ]),
        GFb127x2([ GFb127::w64le(0x0000000000141402, 0x0000000000010115),
                   GFb127::w64le(0x0014140215554220, 0x0001011501110450) ]),
        GFb127x2([ GFb127::w64le(0x000000140014002A, 0x0000000100010017),
                   GFb127::w64le(0x0014156A154028A2, 0x0001010701101665) ]),
        GFb127x2([ GFb127::w64le(0x0014141614282A2E, 0x000101140102173D),
                   GFb127::w64le(0x017D7D5B6AAA8EC8, 0x0013127C12235BD2) ]),
        GFb127x2([ GFb127::w64le(0x0000001400000078, 0x0000000100000006),
                   GFb127::w64le(0x0000153800007F82, 0x0000011600000675) ]),
        GFb127x2([ GFb127::w64le(0x001414020078780C, 0x000101150006067E),
                   GFb127::w64le(0x152D3A2C7FFD8EEA, 0x0117022E06730CF7) ]),
        GFb127x2([ GFb127::w64le(0x00140052007800FE, 0x0001001100060067),
                   GFb127::w64le(0x153857DE7F82F1C2, 0x0116107706756133) ]),
        GFb127x2([ GFb127::w64le(0x1450525A78F2FECC, 0x010411450619678C),
                   GFb127::w64le(0x6DA581137FD90248, 0x124936DA6DA5B7CB) ]),
        GFb127x2([ GFb127::w64le(0x0000000000000112, 0x0000000000000001),
                   GFb127::w64le(0x0000011200010320, 0x0000000100000110) ]),
        GFb127x2([ GFb127::w64le(0x0000000001131202, 0x0000000000010013),
                   GFb127::w64le(0x0113120302232220, 0x0001001301101230) ]),
        GFb127x2([ GFb127::w64le(0x0000011201120226, 0x0000000100010111),
                   GFb127::w64le(0x0113010703220462, 0x0001000101110103) ]),
        GFb127x2([ GFb127::w64le(0x0113131110262422, 0x0001001201031237),
                   GFb127::w64le(0x1204042324446004, 0x0013011612310772) ]),
        GFb127x2([ GFb127::w64le(0x000001120000066C, 0x0000000100000006),
                   GFb127::w64le(0x0001054C00060AC2, 0x0000011600000773) ]),
        GFb127x2([ GFb127::w64le(0x01131202066A6C0C, 0x000100130006006A),
                   GFb127::w64le(0x04494E2A0CC8CCE6, 0x0116125A07727EB0) ]),
        GFb127x2([ GFb127::w64le(0x0112044A066C0CD6, 0x0001011700060775),
                   GFb127::w64le(0x054802700ACE194E, 0x011701050774070C) ]),
        GFb127x2([ GFb127::w64le(0x164C4E4460D6D8E8, 0x0105125B07187FB1),
                   GFb127::w64le(0x485C78CED9BF4234, 0x125B01067EB11619) ]),
        GFb127x2([ GFb127::w64le(0x000000000000156A, 0x0000000000000107),
                   GFb127::w64le(0x0000156A00143CA0, 0x0000010700011770) ]),
        GFb127x2([ GFb127::w64le(0x00000000157F680E, 0x000000000106136D),
                   GFb127::w64le(0x157F681A289E8EE0, 0x0106136C16725BD0) ]),
        GFb127x2([ GFb127::w64le(0x0000156A156A28DA, 0x0000010701071663),
                   GFb127::w64le(0x157E146E3C8A55AE, 0x010601121767115D) ]),
        GFb127x2([ GFb127::w64le(0x157F7D7140FCF6C6, 0x0106126B05185DB9),
                   GFb127::w64le(0x68765092F1358EB4, 0x136B100C5DA10D9F) ]),
        GFb127x2([ GFb127::w64le(0x0000156A00007F7C, 0x0000010700000612),
                   GFb127::w64le(0x001443DC007889CE, 0x000111620006674D) ]),
        GFb127x2([ GFb127::w64le(0x157F680E7F037024, 0x0106136D06146B6E),
                   GFb127::w64le(0x579DFEBCF14B0098, 0x106630B86154A396) ]),
        GFb127x2([ GFb127::w64le(0x156A57A67F7CF0D2, 0x0107107106126127),
                   GFb127::w64le(0x438E2CCA8931FFC0, 0x11731731672A72B2) ]),
        GFb127x2([ GFb127::w64le(0x3FFFF9E180061242, 0x030C30C30A28A28C),
                   GFb127::w64le(0x80006DDA006B07A0, 0x36DB6DB6B6DB6DDD) ]),
        GFb127x2([ GFb127::w64le(0x0000000000010106, 0x0000000000000001),
                   GFb127::w64le(0x0001010601111660, 0x0000000100000110) ]),
        GFb127x2([ GFb127::w64le(0x0000000100070602, 0x0000000000000007),
                   GFb127::w64le(0x0007071207766220, 0x0000000700000771) ]),
        GFb127x2([ GFb127::w64le(0x000101070104020E, 0x0000000100000105),
                   GFb127::w64le(0x0015157F14422EE2, 0x0000001500001456) ]),
        GFb127x2([ GFb127::w64le(0x00060607060C0C0A, 0x000000060000060A),
                   GFb127::w64le(0x006A6B7C6CCCCAAC, 0x0000006A00006CA1) ]),
        GFb127x2([ GFb127::w64le(0x0001010600060614, 0x0000000100000006),
                   GFb127::w64le(0x0117107406667542, 0x0000011600010767) ]),
        GFb127x2([ GFb127::w64le(0x000706040012140C, 0x0000000700000013),
                   GFb127::w64le(0x0764704C13354CCE, 0x0000076300071433) ]),
        GFb127x2([ GFb127::w64le(0x0102041C06180C26, 0x0000010300010718),
                   GFb127::w64le(0x143C51E0798CE666, 0x0000142800156C9E) ]),
        GFb127x2([ GFb127::w64le(0x0618181814282830, 0x0000061E0006123D),
                   GFb127::w64le(0x6DB1B1A56AAABF3C, 0x00006DDD006B00D0) ]),
        GFb127x2([ GFb127::w64le(0x000000000014147A, 0x0000000000010113),
                   GFb127::w64le(0x0014147A15553DA0, 0x0001011301110230) ]),
        GFb127x2([ GFb127::w64le(0x00000014006C7826, 0x0000000100070669),
                   GFb127::w64le(0x006C6D666ABFA462, 0x0007077907760F85) ]),
        GFb127x2([ GFb127::w64le(0x0014146E14502AD2, 0x000101120104174F),
                   GFb127::w64le(0x01050227152A7D04, 0x0015146E14432E8C) ]),
        GFb127x2([ GFb127::w64le(0x0078786078F0FC9C, 0x00060679060C7288),
                   GFb127::w64le(0x070F1AE37FFF5932, 0x006A6C1E6CCBDE99) ]),
        GFb127x2([ GFb127::w64le(0x0014147A0078791C, 0x000101130006066A),
                   GFb127::w64le(0x152D44BC7FFC8FE6, 0x0117045A067319C9) ]),
        GFb127x2([ GFb127::w64le(0x006C785E016910D6, 0x0007066F00121563),
                   GFb127::w64le(0x6BD6CB377F8FD7BE, 0x07641C93135F4B01) ]),
        GFb127x2([ GFb127::w64le(0x142853B679E2FCC8, 0x01021123060D66DE),
                   GFb127::w64le(0x133471D67ED726C4, 0x143D57E8789AF161) ]),
        GFb127x2([ GFb127::w64le(0x79E1EDDD122E07BA, 0x0618679E14575129),
                   GFb127::w64le(0x6DDD077900D70E90, 0x6DB6B6DC6DDDB1AB) ]),
        GFb127x2([ GFb127::w64le(0x000000000113146E, 0x0000000000010015),
                   GFb127::w64le(0x0113146F022528E0, 0x0001001501101450) ]),
        GFb127x2([ GFb127::w64le(0x0000011207786E2A, 0x000000010007017B),
                   GFb127::w64le(0x07796D0D0FE8C8A2, 0x0007006B07716DA3) ]),
        GFb127x2([ GFb127::w64le(0x0113157D164A28F6, 0x0001001401051451),
                   GFb127::w64le(0x146E02312E887948, 0x0015011014570178) ]),
        GFb127x2([ GFb127::w64le(0x066A6B7460D4DEA0, 0x0006006D060A6CB4),
                   GFb127::w64le(0x6C191D86D99F4ADA, 0x006A07626CA6145F) ]),
        GFb127x2([ GFb127::w64le(0x0113146E066A7964, 0x000100150006007E),
                   GFb127::w64le(0x044F51820CDCF26A, 0x0116142E07726D9A) ]),
        GFb127x2([ GFb127::w64le(0x07786846131164FE, 0x0007017D00120609),
                   GFb127::w64le(0x1CFFA68C207CB31A, 0x07636CD9145900AC) ]),
        GFb127x2([ GFb127::w64le(0x102057F875BEF21C, 0x01031429070C6C8F),
                   GFb127::w64le(0x57EC75EEE71B1590, 0x142907186D890431) ]),
        GFb127x2([ GFb127::w64le(0x75A9A59942F6C71A, 0x061E6DDA125100A1),
                   GFb127::w64le(0xB1C907CCD597B018, 0x6DDA071207A76327) ]),
        GFb127x2([ GFb127::w64le(0x00000000157F1772, 0x000000000106157F),
                   GFb127::w64le(0x157F176628E60520, 0x0106157E167428F0) ]),
        GFb127x2([ GFb127::w64le(0x0000156A6A6958FE, 0x0000010707137D0D),
                   GFb127::w64le(0x6A7D6432CFCD73EE, 0x07126A7A624AC9BD) ]),
        GFb127x2([ GFb127::w64le(0x157F020D3F80041A, 0x01061479030A28F3),
                   GFb127::w64le(0x177229F67A087350, 0x157F16602EF36A51) ]),
        GFb127x2([ GFb127::w64le(0x7F031A4D820A49E8, 0x06146C7D1E51CB84),
                   GFb127::w64le(0x7121A0B226C5AE76, 0x6B7B714BCDC04A0E) ]),
        GFb127x2([ GFb127::w64le(0x157F17727F02732C, 0x0106157F06147F02),
                   GFb127::w64le(0x57E47674F058343C, 0x106057F46141F038) ]),
        GFb127x2([ GFb127::w64le(0x6A6927837D77D00A, 0x07137B1F126B1A43),
                   GFb127::w64le(0xB2C22B40AE8BFC90, 0x7027B4A020D1BBC7) ]),
        GFb127x2([ GFb127::w64le(0x40820834830C30AE, 0x051E51E51E45E45E),
                   GFb127::w64le(0x0924876536CF0720, 0x51F11F11E4264270) ]),
        GFb127x2([ GFb127::w64le(0x8000144500156C8A, 0x0A28A28A3CF3CF29),
                   GFb127::w64le(0x80006DDA006B07A1, 0x36DB6DB6B6DB6DDD) ]),
    ];

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
            let m1 = _mm_set1_epi32(0x55555555);
            let m2 = _mm_set1_epi32(0x33333333);
            let m3 = _mm_set1_epi32(0x0F0F0F0F);
            let sklo = _mm_set_epi8(
                -1, -1, -1, -1, -1, -1, -1, -1,
                14, 12, 10, 8, 6, 4, 2, 0);
            let skhi = _mm_set_epi8(
                14, 12, 10, 8, 6, 4, 2, 0,
                -1, -1, -1, -1, -1, -1, -1, -1);

            // Split a into ae and ao, then "squeeze" ae and ao:
            //   a = ae + ao*z
            //   sqrt(a) = sqrt(ae) + sqrt(ao)*sqrt(z)
            let a = self.0;
            let mut ae = _mm_and_si128(a, m1);
            let mut ao = _mm_and_si128(_mm_srli_epi64(a, 1), m1);
            ae = _mm_and_si128(_mm_xor_si128(ae, _mm_srli_epi64(ae, 1)), m2);
            ao = _mm_and_si128(_mm_xor_si128(ao, _mm_srli_epi64(ao, 1)), m2);
            ae = _mm_and_si128(_mm_xor_si128(ae, _mm_srli_epi64(ae, 2)), m3);
            ao = _mm_and_si128(_mm_xor_si128(ao, _mm_srli_epi64(ao, 2)), m3);
            ae = _mm_xor_si128(ae, _mm_srli_epi64(ae, 4));
            ao = _mm_xor_si128(ao, _mm_srli_epi64(ao, 4));
            ae = _mm_shuffle_epi8(ae, sklo);
            ao = _mm_shuffle_epi8(ao, skhi);

            // sqrt(ae) and sqrt(ao) have length 64 bits each.
            // We need to multiply sqrt(ao) by sqrt(z) = z^32 + z^64; no
            // reduction will be necessary. We currently have sqrt(ao)*z^64
            // in the 'ao' variable.
            self.0 = _mm_xor_si128(ae,
                _mm_xor_si128(ao, _mm_bsrli_si128(ao, 4)));
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
            let a = self.0;
            let b = _mm_xor_si128(a, _mm_bsrli_si128(_mm_srli_epi64(a, 63), 8));
            (_mm_cvtsi128_si32(b) as u32) & 1
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
            let m1 = _mm_set1_epi32(0x55555555);
            let m2 = _mm_set1_epi32(0x33333333);
            let m3 = _mm_set1_epi32(0x0F0F0F0F);
            let sklo = _mm_set_epi8(
                -1, -1, -1, -1, -1, -1, -1, -1,
                14, 12, 10, 8, 6, 4, 2, 0);

            // We accumulate the odd-indexed bits in ao. We will ignore the
            // even-indexed bits in this variable, so we do not care what
            // values are written there.
            let mut ao = _mm_setzero_si128();

            // We accumulate the extra values (square roots) ino e.
            let mut x = self.0;
            let mut e = _mm_setzero_si128();

            // Do the split-and-squeeze 7 times, so that x is reduced to
            // a single bit.
            for _ in 0..7 {
                ao = _mm_xor_si128(ao, x);
                x = _mm_and_si128(x, m1);
                x = _mm_and_si128(_mm_xor_si128(x, _mm_srli_epi64(x, 1)), m2);
                x = _mm_and_si128(_mm_xor_si128(x, _mm_srli_epi64(x, 2)), m3);
                x = _mm_xor_si128(x, _mm_srli_epi64(x, 4));
                x = _mm_shuffle_epi8(x, sklo);
                e = _mm_xor_si128(e, x);
            }

            // len(x) = 1, hence H(x) = x. We now apply the halftrace of the
            // odd-indexed bits in ao.
            let mut d = _mm_xor_si128(e, x);
            #[cfg(not(target_feature = "avx2"))]
            {
                for i in (0..4).rev() {
                    let mut mw = _mm_shuffle_epi32(ao, 0xFF);
                    ao = _mm_bslli_si128(ao, 4);
                    for j in (0..8).rev() {
                        let m = _mm_srai_epi32(mw, 31);
                        mw = _mm_slli_epi32(mw, 2);
                        d = _mm_xor_si128(d, _mm_and_si128(m,
                            Self::HALFTRACE[(i << 3) + j].0[1].0));
                        let m = _mm_srai_epi32(mw, 31);
                        mw = _mm_slli_epi32(mw, 2);
                        d = _mm_xor_si128(d, _mm_and_si128(m,
                            Self::HALFTRACE[(i << 3) + j].0[0].0));
                    }
                }
            }
            #[cfg(target_feature = "avx2")]
            {
                let mut yao = _mm256_setr_m128i(_mm_slli_epi32(ao, 2), ao);
                let mut yd = _mm256_setzero_si256();
                for i in (0..4).rev() {
                    let mut ymw = _mm256_shuffle_epi32(yao, 0xFF);
                    yao = _mm256_bslli_epi128(yao, 4);
                    for j in (0..8).rev() {
                        yd = _mm256_xor_si256(yd,
                            _mm256_maskload_epi32(core::mem::transmute(
                                &Self::HALFTRACE[(i << 3) + j].0), ymw));
                        ymw = _mm256_slli_epi32(ymw, 4);
                    }
                }
                d = _mm_xor_si128(d, _mm_xor_si128(
                    _mm256_castsi256_si128(yd),
                    _mm256_extracti128_si256(yd, 1)));
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
    const HALFTRACE: [GFb127x2; 32] = [
        GFb127x2([ GFb127::w64le(0x0000000000000000, 0x0000000000000001),
                   GFb127::w64le(0x0001011201141668, 0x0000000000010014) ]),
        GFb127x2([ GFb127::w64le(0x000100110105135E, 0x0000000100000016),
                   GFb127::w64le(0x01031401116159DE, 0x0000000501000426) ]),
        GFb127x2([ GFb127::w64le(0x000101150117177E, 0x0000000100000106),
                   GFb127::w64le(0x0010017C041E2620, 0x0000011400060260) ]),
        GFb127x2([ GFb127::w64le(0x01010472112C52C8, 0x0001001200040648),
                   GFb127::w64le(0x1204585042CC8A00, 0x0004043010241E00) ]),
        GFb127x2([ GFb127::w64le(0x0000001400060200, 0x0000000000000010),
                   GFb127::w64le(0x0000043000240200, 0x0000001000040600) ]),
        GFb127x2([ GFb127::w64le(0x0105135E135E5EE8, 0x0001011600121628),
                   GFb127::w64le(0x04506EC02CA82000, 0x0010064000686000) ]),
        GFb127x2([ GFb127::w64le(0x0010150C04722C20, 0x0000010400021460),
                   GFb127::w64le(0x055D5EE23FE878C8, 0x0015162202284848) ]),
        GFb127x2([ GFb127::w64le(0x15522EC87C28E080, 0x0112064800682080),
                   GFb127::w64le(0x75E2E808F880C080, 0x0562280848804080) ]),
        GFb127x2([ GFb127::w64le(0x000100030101115E, 0x0000000100000002),
                   GFb127::w64le(0x0101000A110050C8, 0x0001000200000008) ]),
        GFb127x2([ GFb127::w64le(0x0000042000200000, 0x0000000000000400),
                   GFb127::w64le(0x110200885000C080, 0x0102000800000080) ]),
        GFb127x2([ GFb127::w64le(0x0014132C06522C20, 0x0000010400061060),
                   GFb127::w64le(0x0000040000200000, 0x0000040000200000) ]),
        GFb127x2([ GFb127::w64le(0x051D52E237C878C8, 0x0015122202484848),
                   GFb127::w64le(0x52088080C0008000, 0x1208008000008000) ]),
        GFb127x2([ GFb127::w64le(0x0013057D053F377E, 0x0000011100060646),
                   GFb127::w64le(0x01492C02192050C8, 0x0001044200602048) ]),
        GFb127x2([ GFb127::w64le(0x144F5C2A6BE09848, 0x0107146A062068C8),
                   GFb127::w64le(0x0040200008000000, 0x0040200008000000) ]),
        GFb127x2([ GFb127::w64le(0x065476902EE42A00, 0x00140270004C7E00),
                   GFb127::w64le(0x5628C880E0808000, 0x1628488020808000) ]),
        GFb127x2([ GFb127::w64le(0x67EAE888B8804080, 0x176A28880880C080),
                   GFb127::w64le(0x6880800080000000, 0x6880800080000000) ]),
        GFb127x2([ GFb127::w64le(0x0000011300150736, 0x0000000100010014),
                   GFb127::w64le(0x0002140300610916, 0x0001000701000426) ]),
        GFb127x2([ GFb127::w64le(0x0010057C043E2620, 0x0000011400060640),
                   GFb127::w64le(0x0306585812CC4A80, 0x0106043810241E00) ]),
        GFb127x2([ GFb127::w64le(0x0014151C06762E20, 0x0000011400021460),
                   GFb127::w64le(0x045062C02C882000, 0x0010024000486800) ]),
        GFb127x2([ GFb127::w64le(0x00402C0008200000, 0x0000040000602000),
                   GFb127::w64le(0x27EAE88838804080, 0x176A288848804080) ]),
        GFb127x2([ GFb127::w64le(0x01100577143F67B6, 0x000101130004064E),
                   GFb127::w64le(0x10432C8A49209048, 0x0103044A006820C8) ]),
        GFb127x2([ GFb127::w64le(0x146F582A6BC09848, 0x0107106A062068C8),
                   GFb127::w64le(0x52C8A080C8008000, 0x1248208008808000) ]),
        GFb127x2([ GFb127::w64le(0x051D5A9237C47AC8, 0x00150632022C5E48),
                   GFb127::w64le(0x5E68E880E8808000, 0x1668688020808000) ]),
        GFb127x2([ GFb127::w64le(0x11C220085800C080, 0x0142600808004080),
                   GFb127::w64le(0x8000000000000000, 0x0000000080000000) ]),
        GFb127x2([ GFb127::w64le(0x0002151000740E20, 0x0001000401010430),
                   GFb127::w64le(0x03044C5B12AD4396, 0x0107043711241A2E) ]),
        GFb127x2([ GFb127::w64le(0x044067BC28B60620, 0x00100374004E6E60),
                   GFb127::w64le(0x24ECB0D02A4C0A00, 0x166C2C3058A45E00) ]),
        GFb127x2([ GFb127::w64le(0x105739964F56BE68, 0x0103075E006A36A8),
                   GFb127::w64le(0x5698C240E488A000, 0x12582AC008C8E000) ]),
        GFb127x2([ GFb127::w64le(0x5E28C480E0A08000, 0x16684C8020E08000),
                   GFb127::w64le(0xA7EAE88838804080, 0x176AA888C880C080) ]),
        GFb127x2([ GFb127::w64le(0x0214492C06922420, 0x0104052411221C60),
                   GFb127::w64le(0x34AF9C5A636C9A48, 0x1767287A58C47EC8) ]),
        GFb127x2([ GFb127::w64le(0x42F79A6A8F483848, 0x137F3AAA0EC888C8),
                   GFb127::w64le(0xF5224808F080C080, 0x05A28808C0804080) ]),
        GFb127x2([ GFb127::w64le(0x31B2C6C854A8E080, 0x15722E4858E82080),
                   GFb127::w64le(0xAB4AA08818004080, 0x1BCAE088E800C080) ]),
        GFb127x2([ GFb127::w64le(0xBA88808040008000, 0x3A888080C0008000),
                   GFb127::w64le(0x6880800080000000, 0x6880800080000000) ]),
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
            let c = _mm_cmpeq_epi64(self.0, _mm_setzero_si128());
            let d = _mm_cmpeq_epi64(self.0,
                _mm_set_epi32((1u32 << 31) as i32, 0, (1u32 << 31) as i32, 1));
            let c = _mm_and_si128(c, _mm_bsrli_si128(c, 8));
            let d = _mm_and_si128(d, _mm_bsrli_si128(d, 8));
            _mm_cvtsi128_si32(_mm_or_si128(c, d)) as u32
        }
    }

    #[inline(always)]
    pub fn encode(self) -> [u8; 16] {
        unsafe {
            let mut r = self;
            r.set_normalized();
            core::mem::transmute(r.0)
        }
    }

    // Decode the value from bytes with implicit reduction modulo
    // z^127 + z^63 + 1. Input MUST be of length 16 bytes exactly.
    #[inline]
    fn set_decode16_reduce(&mut self, buf: &[u8]) {
        debug_assert!(buf.len() == 16);
        unsafe {
            self.0 = core::mem::transmute(*<&[u8; 16]>::try_from(buf).unwrap());
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
            self.0 = _mm_and_si128(self.0, _mm_set1_epi32(m as i32));
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

        // We bypass the GFb127 abstraction so that we may mutualize
        // the reductions.
        #[cfg(not(target_feature = "avx2"))]
        unsafe {
            #[inline(always)]
            fn mm(a: __m128i, b: __m128i) -> (__m128i, __m128i) {
                unsafe {
                    let c0 = _mm_clmulepi64_si128(a, b, 0x00);
                    let c1 = _mm_xor_si128(
                        _mm_clmulepi64_si128(a, b, 0x01),
                        _mm_clmulepi64_si128(a, b, 0x10));
                    let c2 = _mm_clmulepi64_si128(a, b, 0x11);

                    let d0 = _mm_xor_si128(c0, _mm_bslli_si128(c1, 8));
                    let d1 = _mm_xor_si128(_mm_bsrli_si128(c1, 8), c2);

                    (d0, d1)
                }
            }

            #[inline(always)]
            fn red(d0: __m128i, d1: __m128i) -> __m128i {
                unsafe {
                    // f = e2 + e3 + e3*z^64
                    // g = (e2 + e3)*z^64
                    let f = _mm_xor_si128(d1, _mm_bsrli_si128(d1, 8));
                    let g = _mm_bslli_si128(f, 8);

                    // h = z*f
                    let h = _mm_or_si128(
                        _mm_slli_epi64(f, 1),
                        _mm_bslli_si128(_mm_srli_epi64(f, 63), 8));

                    _mm_xor_si128(d0, _mm_xor_si128(g, h))
                }
            }

            let (a0, a1) = (self.0[0].0, self.0[1].0);
            let (b0, b1) = (rhs.0[0].0, rhs.0[1].0);

            let (a0b0l, a0b0h) = mm(a0, b0);
            let (a1b1l, a1b1h) = mm(a1, b1);
            let (cl, ch) = mm(_mm_xor_si128(a0, a1), _mm_xor_si128(b0, b1));

            self.0[0].0 = red(
                _mm_xor_si128(a0b0l, a1b1l),
                _mm_xor_si128(a0b0h, a1b1h));
            self.0[1].0 = red(
                _mm_xor_si128(a0b0l, cl),
                _mm_xor_si128(a0b0h, ch));
        }

        #[cfg(target_feature = "avx2")]
        unsafe {
            // There is no AVX2 parallel variant of pclmulqdq, we must
            // do the multiplications over __m128i values.
            let (a0, a1) = (self.0[0].0, self.0[1].0);
            let (b0, b1) = (rhs.0[0].0, rhs.0[1].0);

            // a0*b0
            let c0 = _mm_clmulepi64_si128(a0, b0, 0x00);
            let c1 = _mm_clmulepi64_si128(a0, b0, 0x11);
            let c2 = _mm_xor_si128(
                _mm_clmulepi64_si128(a0, b0, 0x01),
                _mm_clmulepi64_si128(a0, b0, 0x10));

            // a1*b1
            let d0 = _mm_clmulepi64_si128(a1, b1, 0x00);
            let d1 = _mm_clmulepi64_si128(a1, b1, 0x11);
            let d2 = _mm_xor_si128(
                _mm_clmulepi64_si128(a1, b1, 0x01),
                _mm_clmulepi64_si128(a1, b1, 0x10));

            // (a0 + a1)*(b0 + b1)
            let a2 = _mm_xor_si128(a0, a1);
            let b2 = _mm_xor_si128(b0, b1);
            let e0 = _mm_clmulepi64_si128(a2, b2, 0x00);
            let e1 = _mm_clmulepi64_si128(a2, b2, 0x11);
            let e2 = _mm_xor_si128(
                    _mm_clmulepi64_si128(a2, b2, 0x01),
                    _mm_clmulepi64_si128(a2, b2, 0x10));

            // Move to the 256-bit space to make both reductions in parallel.
            // (a0 + a1*u)*(b0 + b1*u)
            //  = a0*b0 + (a0*b1 + a1*b0)*u + a1*b1*(u + 1)
            //  = (a0*b0 + a1*b1) + u*((a0 + a1)*(b0 + b1) + a0*b0)
            // We store a0*b0 + a1*b1 in the low lanes, and
            // (a0 + a1)*(b0 + b1) + a0*b0 in the high lanes. Each value
            // consists in three 128-bit chunks right now (the "middle chunk"
            // must be added to the two others).
            let y0 = _mm256_xor_si256(
                _mm256_setr_m128i(c0, c0),
                _mm256_setr_m128i(d0, e0));
            let y1 = _mm256_xor_si256(
                _mm256_setr_m128i(c1, c1),
                _mm256_setr_m128i(d1, e1));
            let y2 = _mm256_xor_si256(
                _mm256_setr_m128i(c2, c2),
                _mm256_setr_m128i(d2, e2));

            // Add middle word to get the two unreduced 255-bit values.
            let y0 = _mm256_xor_si256(y0, _mm256_bslli_epi128(y2, 8));
            let y1 = _mm256_xor_si256(y1, _mm256_bsrli_epi128(y2, 8));

            // Reduce:
            //   f = e2 + e3 + e3*z^64
            //   g = (e2 + e3)*z^64
            let yf = _mm256_xor_si256(y1, _mm256_bsrli_epi128(y1, 8));
            let yg = _mm256_bslli_epi128(yf, 8);

            // h = z*f
            let yh = _mm256_xor_si256(
                _mm256_slli_epi64(yf, 1),
                _mm256_bslli_epi128(_mm256_srli_epi64(yf, 63), 8));

            let yd = _mm256_xor_si256(y0, _mm256_xor_si256(yg, yh));

            // Write the result.
            *self = core::mem::transmute(yd);
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
        #[cfg(target_feature = "avx2")]
        unsafe {
            let a0 = self.0[0].0;
            let a1 = self.0[1].0;
            let a = _mm256_setr_m128i(a0, a1);
            let f = _mm256_slli_epi64(a, 27);
            let g = _mm256_srli_epi64(a, 37);

            // g = g0 + g1*z^64
            // r = a + f + g1*z + (g0 + g1)*z^64
            let h = _mm256_shuffle_epi32(g, 0x4E);
            let k = _mm256_blend_epi32(
                _mm256_slli_epi64(h, 1), _mm256_xor_si256(g, h), 0xCC);
            let r = _mm256_xor_si256(_mm256_xor_si256(a, f), k);
            self.0[0].0 = _mm256_castsi256_si128(r);
            self.0[1].0 = _mm256_extracti128_si256(r, 1);
        }

        #[cfg(not(target_feature = "avx2"))]
        {
            self.0[0].set_mul_sb();
            self.0[1].set_mul_sb();
        }
    }

    // Multiply this value by sb = 1 + z^27 (an element of GF(2^127)).
    #[inline(always)]
    pub fn mul_sb(self) -> Self {
        let mut r = self;
        r.set_mul_sb();
        r
    }

    // Multiply this value by b = 1 + z^54 (an element of GF(2^127)).
    #[inline(always)]
    pub fn set_mul_b(&mut self) {
        #[cfg(target_feature = "avx2")]
        unsafe {
            let a0 = self.0[0].0;
            let a1 = self.0[1].0;
            let a = _mm256_setr_m128i(a0, a1);
            let f = _mm256_slli_epi64(a, 54);
            let g = _mm256_srli_epi64(a, 10);

            // g = g0 + g1*z^64
            // r = a + f + g1*z + (g0 + g1)*z^64
            let h = _mm256_shuffle_epi32(g, 0x4E);
            let k = _mm256_blend_epi32(
                _mm256_slli_epi64(h, 1), _mm256_xor_si256(g, h), 0xCC);
            let r = _mm256_xor_si256(_mm256_xor_si256(a, f), k);
            self.0[0].0 = _mm256_castsi256_si128(r);
            self.0[1].0 = _mm256_extracti128_si256(r, 1);
        }

        #[cfg(not(target_feature = "avx2"))]
        {
            self.0[0].set_mul_b();
            self.0[1].set_mul_b();
        }
    }

    // Multiply this value by sb = 1 + z^54 (an element of GF(2^127)).
    #[inline(always)]
    pub fn mul_b(self) -> Self {
        let mut r = self;
        r.set_mul_b();
        r
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

        #[cfg(not(target_feature = "avx2"))]
        {
            let (a0, a1) = (self.0[0], self.0[1]);
            let t = a1.square();
            self.0[0] = a0.square() + t;
            self.0[1] = t;
        }

        #[cfg(target_feature = "avx2")]
        unsafe {
            // Use AVX2 to square both sub-elements simultaneously.
            let a = _mm256_setr_m128i(self.0[0].0, self.0[1].0);

            // Square the polynomial by "expanding" the bits.
            let m16 = _mm256_set1_epi8(0x0F);
            let shk = _mm256_setr_epi8(
                0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15,
                0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55,
                0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15,
                0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55);
            let t0 = _mm256_shuffle_epi8(shk,
                _mm256_and_si256(a, m16));
            let t1 = _mm256_shuffle_epi8(shk,
                _mm256_and_si256(_mm256_srli_epi16(a, 4), m16));
            let d0 = _mm256_unpacklo_epi8(t0, t1);
            let d1 = _mm256_unpackhi_epi8(t0, t1);

            // Reduction.
            let f = _mm256_xor_si256(d1, _mm256_bsrli_epi128(d1, 8));
            let g = _mm256_bslli_epi128(f, 8);
            let h = _mm256_slli_epi64(f, 1);

            let b = _mm256_xor_si256(d0, _mm256_xor_si256(g, h));

            // Resplit b into the two individual squares and assemble.
            let b0 = _mm256_castsi256_si128(b);
            let b1 = _mm256_extracti128_si256(b, 1);

            self.0[0] = GFb127(_mm_xor_si128(b0, b1));
            self.0[1] = GFb127(b1);
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
        #[cfg(not(target_feature = "avx2"))]
        unsafe {
            let xj = _mm_set1_epi32(j as i32);
            let mut xi = _mm_setzero_si128();
            let mut a0 = _mm_setzero_si128();
            let mut a1 = _mm_setzero_si128();
            let mut a2 = _mm_setzero_si128();
            let mut a3 = _mm_setzero_si128();
            for i in 0..16 {
                let m = _mm_cmpeq_epi32(xi, xj);
                xi = _mm_add_epi32(xi, _mm_set1_epi32(1));
                a0 = _mm_blendv_epi8(a0, tab[(i * 2) + 0].0[0].0, m);
                a1 = _mm_blendv_epi8(a1, tab[(i * 2) + 0].0[1].0, m);
                a2 = _mm_blendv_epi8(a2, tab[(i * 2) + 1].0[0].0, m);
                a3 = _mm_blendv_epi8(a3, tab[(i * 2) + 1].0[1].0, m);
            }
            [
                Self([ GFb127(a0), GFb127(a1) ]),
                Self([ GFb127(a2), GFb127(a3) ]),
            ]
        }
        #[cfg(target_feature = "avx2")]
        unsafe {
            // With AVX2 we can read a full GF(2^254) element in a
            // single access. The GFb254 structure has been tagged
            // with 32-byte alignment so that the access as __m256i
            // is valid.
            let xj = _mm256_set1_epi32(j as i32);
            let mut xi = _mm256_setzero_si256();
            let mut a0 = _mm256_setzero_si256();
            let mut a1 = _mm256_setzero_si256();
            for i in 0..16 {
                // We use explicit unaligned loads (vmovdqu opcodes)
                // as an extra protection in case the caller did not
                // honour the 32-byte alignment of GFb254.
                let m = _mm256_cmpeq_epi32(xi, xj);
                xi = _mm256_add_epi32(xi, _mm256_set1_epi32(1));
                a0 = _mm256_blendv_epi8(a0,
                    _mm256_loadu_si256(core::mem::transmute(
                        core::ptr::addr_of!(tab[i * 2 + 0].0))), m);
                a1 = _mm256_blendv_epi8(a1,
                    _mm256_loadu_si256(core::mem::transmute(
                        core::ptr::addr_of!(tab[i * 2 + 1].0))), m);
            }
            [ core::mem::transmute(a0), core::mem::transmute(a1) ]
        }
    }

    // Constant-time table lookup: given a table of 16 field elements, and
    // an index `j` in the 0 to 7 range, return the elements of index
    // `j*2` and `j*2+1`. If `j` is not in the 0 to 7 range (inclusive),
    // then this returns two zeros.
    #[inline]
    pub fn lookup8_x2(tab: &[Self; 16], j: u32) -> [Self; 2] {
        #[cfg(not(target_feature = "avx2"))]
        unsafe {
            let xj = _mm_set1_epi32(j as i32);
            let mut xi = _mm_setzero_si128();
            let mut a0 = _mm_setzero_si128();
            let mut a1 = _mm_setzero_si128();
            let mut a2 = _mm_setzero_si128();
            let mut a3 = _mm_setzero_si128();
            for i in 0..8 {
                let m = _mm_cmpeq_epi32(xi, xj);
                xi = _mm_add_epi32(xi, _mm_set1_epi32(1));
                a0 = _mm_blendv_epi8(a0, tab[(i * 2) + 0].0[0].0, m);
                a1 = _mm_blendv_epi8(a1, tab[(i * 2) + 0].0[1].0, m);
                a2 = _mm_blendv_epi8(a2, tab[(i * 2) + 1].0[0].0, m);
                a3 = _mm_blendv_epi8(a3, tab[(i * 2) + 1].0[1].0, m);
            }
            [
                Self([ GFb127(a0), GFb127(a1) ]),
                Self([ GFb127(a2), GFb127(a3) ]),
            ]
        }
        #[cfg(target_feature = "avx2")]
        unsafe {
            // With AVX2 we can read a full GF(2^254) element in a
            // single access. The GFb254 structure has been tagged
            // with 32-byte alignment so that the access as __m256i
            // is valid.
            let xj = _mm256_set1_epi32(j as i32);
            let mut xi = _mm256_setzero_si256();
            let mut a0 = _mm256_setzero_si256();
            let mut a1 = _mm256_setzero_si256();
            for i in 0..8 {
                // We use explicit unaligned loads (vmovdqu opcodes)
                // as an extra protection in case the caller did not
                // honour the 32-byte alignment of GFb254.
                let m = _mm256_cmpeq_epi32(xi, xj);
                xi = _mm256_add_epi32(xi, _mm256_set1_epi32(1));
                a0 = _mm256_blendv_epi8(a0,
                    _mm256_loadu_si256(core::mem::transmute(
                        core::ptr::addr_of!(tab[i * 2 + 0].0))), m);
                a1 = _mm256_blendv_epi8(a1,
                    _mm256_loadu_si256(core::mem::transmute(
                        core::ptr::addr_of!(tab[i * 2 + 1].0))), m);
            }
            [ core::mem::transmute(a0), core::mem::transmute(a1) ]
        }
    }

    // Constant-time table lookup: given a table of 8 field elements, and
    // an index `j` in the 0 to 3 range, return the elements of index
    // `j*2` and `j*2+1`. If `j` is not in the 0 to 3 range (inclusive),
    // then this returns two zeros.
    #[inline]
    pub fn lookup4_x2(tab: &[Self; 8], j: u32) -> [Self; 2] {
        #[cfg(not(target_feature = "avx2"))]
        unsafe {
            let xj = _mm_set1_epi32(j as i32);
            let mut xi = _mm_setzero_si128();
            let mut a0 = _mm_setzero_si128();
            let mut a1 = _mm_setzero_si128();
            let mut a2 = _mm_setzero_si128();
            let mut a3 = _mm_setzero_si128();
            for i in 0..4 {
                let m = _mm_cmpeq_epi32(xi, xj);
                xi = _mm_add_epi32(xi, _mm_set1_epi32(1));
                a0 = _mm_blendv_epi8(a0, tab[(i * 2) + 0].0[0].0, m);
                a1 = _mm_blendv_epi8(a1, tab[(i * 2) + 0].0[1].0, m);
                a2 = _mm_blendv_epi8(a2, tab[(i * 2) + 1].0[0].0, m);
                a3 = _mm_blendv_epi8(a3, tab[(i * 2) + 1].0[1].0, m);
            }
            [
                Self([ GFb127(a0), GFb127(a1) ]),
                Self([ GFb127(a2), GFb127(a3) ]),
            ]
        }
        #[cfg(target_feature = "avx2")]
        unsafe {
            // With AVX2 we can read a full GF(2^254) element in a
            // single access. The GFb254 structure has been tagged
            // with 32-byte alignment so that the access as __m256i
            // is valid.
            let xj = _mm256_set1_epi32(j as i32);
            let mut xi = _mm256_setzero_si256();
            let mut a0 = _mm256_setzero_si256();
            let mut a1 = _mm256_setzero_si256();
            for i in 0..4 {
                // We use explicit unaligned loads (vmovdqu opcodes)
                // as an extra protection in case the caller did not
                // honour the 32-byte alignment of GFb254.
                let m = _mm256_cmpeq_epi32(xi, xj);
                xi = _mm256_add_epi32(xi, _mm256_set1_epi32(1));
                a0 = _mm256_blendv_epi8(a0,
                    _mm256_loadu_si256(core::mem::transmute(
                        core::ptr::addr_of!(tab[i * 2 + 0].0))), m);
                a1 = _mm256_blendv_epi8(a1,
                    _mm256_loadu_si256(core::mem::transmute(
                        core::ptr::addr_of!(tab[i * 2 + 1].0))), m);
            }
            [ core::mem::transmute(a0), core::mem::transmute(a1) ]
        }
    }

    /// Constant-time table lookup, short table. This is similar to
    /// `lookup16_x2()`, except that there are only four pairs of values
    /// (8 elements of GF(2^254)), and the pair index MUST be in the
    /// proper range (if the index is not in the range, an unpredictable
    /// value is returned).
    #[inline]
    pub fn lookup4_x2_nocheck(tab: &[Self; 8], j: u32) -> [Self; 2] {
        #[cfg(not(target_feature = "avx2"))]
        unsafe {
            let xj = _mm_set1_epi32(j as i32);
            let xm0 = _mm_cmpeq_epi32(
                _mm_and_si128(xj, _mm_set1_epi32(1)),
                _mm_set1_epi32(1));
            let xm1 = _mm_cmpeq_epi32(
                _mm_and_si128(xj, _mm_set1_epi32(2)),
                _mm_set1_epi32(2));
            let a0 = _mm_blendv_epi8(
                    _mm_blendv_epi8(tab[0].0[0].0, tab[2].0[0].0, xm0),
                    _mm_blendv_epi8(tab[4].0[0].0, tab[6].0[0].0, xm0),
                    xm1);
            let a1 = _mm_blendv_epi8(
                    _mm_blendv_epi8(tab[0].0[1].0, tab[2].0[1].0, xm0),
                    _mm_blendv_epi8(tab[4].0[1].0, tab[6].0[1].0, xm0),
                    xm1);
            let a2 = _mm_blendv_epi8(
                    _mm_blendv_epi8(tab[1].0[0].0, tab[3].0[0].0, xm0),
                    _mm_blendv_epi8(tab[5].0[0].0, tab[7].0[0].0, xm0),
                    xm1);
            let a3 = _mm_blendv_epi8(
                    _mm_blendv_epi8(tab[1].0[1].0, tab[3].0[1].0, xm0),
                    _mm_blendv_epi8(tab[5].0[1].0, tab[7].0[1].0, xm0),
                    xm1);
            [
                Self([ GFb127(a0), GFb127(a1) ]),
                Self([ GFb127(a2), GFb127(a3) ]),
            ]
        }
        #[cfg(target_feature = "avx2")]
        unsafe {
            // With AVX2 we can read a full GF(2^254) element in a
            // single access. The GFb254 structure has been tagged
            // with 32-byte alignment so that the access as __m256i
            // is valid.
            let xj = _mm256_set1_epi32(j as i32);
            let xm0 = _mm256_cmpeq_epi32(
                _mm256_and_si256(xj, _mm256_set1_epi32(1)),
                _mm256_set1_epi32(1));
            let xm1 = _mm256_cmpeq_epi32(
                _mm256_and_si256(xj, _mm256_set1_epi32(2)),
                _mm256_set1_epi32(2));

            let v0 = _mm256_loadu_si256(
                core::mem::transmute(core::ptr::addr_of!(tab[0].0)));
            let v1 = _mm256_loadu_si256(
                core::mem::transmute(core::ptr::addr_of!(tab[1].0)));
            let v2 = _mm256_loadu_si256(
                core::mem::transmute(core::ptr::addr_of!(tab[2].0)));
            let v3 = _mm256_loadu_si256(
                core::mem::transmute(core::ptr::addr_of!(tab[3].0)));
            let v4 = _mm256_loadu_si256(
                core::mem::transmute(core::ptr::addr_of!(tab[4].0)));
            let v5 = _mm256_loadu_si256(
                core::mem::transmute(core::ptr::addr_of!(tab[5].0)));
            let v6 = _mm256_loadu_si256(
                core::mem::transmute(core::ptr::addr_of!(tab[6].0)));
            let v7 = _mm256_loadu_si256(
                core::mem::transmute(core::ptr::addr_of!(tab[7].0)));
            let a0 = _mm256_blendv_epi8(
                    _mm256_blendv_epi8(v0, v2, xm0),
                    _mm256_blendv_epi8(v4, v6, xm0), xm1);
            let a1 = _mm256_blendv_epi8(
                    _mm256_blendv_epi8(v1, v3, xm0),
                    _mm256_blendv_epi8(v5, v7, xm0), xm1);
            [ core::mem::transmute(a0), core::mem::transmute(a1) ]
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

        let c = a.mul_sb();
        let vc = c.encode();
        let mut vx = [0u8; 32];
        vx[0] = 1;
        vx[3] = 8;
        assert!(vc == mul(va, &vx));

        let c = a.mul_b();
        let vc = c.encode();
        let mut vx = [0u8; 32];
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
