#[macro_export]
macro_rules! define_gfgen { ($typename:ident, $fieldparams:ident, $submod:ident, $squarespec:expr) => {
    // We define a sub-module so that the 'use' clauses do not spill over
    // the caller.
    pub use $submod::$typename;
    mod $submod {

    use $crate::backend::w64::{addcarry_u64, subborrow_u64, umull, umull_add, umull_add2, umull_x2, umull_x2_add, sgnw, lzcnt};
    use $crate::backend::w64::lagrange::lagrange_vartime;
    use super::$fieldparams;
    use core::convert::TryFrom;

    #[derive(Clone, Copy, Debug)]
    pub struct $typename([u64; $typename::N]);

    impl $typename {
        const N: usize = Self::top_word_index() + 1;
        const BITLEN: usize = Self::mod_bitlen();
        const M0I: u64 = Self::ninv64($typename::MODULUS[0]);
        const R: Self = Self::pow2mod(Self::N * 64);
        const R2: Self = Self::pow2mod(Self::N * 128);
        const T64: Self = Self::pow2mod(Self::N * 64 + 64);
        const T128: Self = Self::pow2mod(Self::N * 64 + 128);

        // Element encoded length, in bytes.
        pub const ENC_LEN: usize = (Self::BITLEN + 7) >> 3;

        // Modulus (little-endian order, 64-bit limbs).
        pub const MODULUS: [u64; Self::N] = Self::make_modulus();

        pub const ZERO: Self = Self([0u64; Self::N]);
        pub const ONE: Self = Self::R;
        pub const TWO: Self = Self::const_small(2);
        pub const THREE: Self = Self::const_small(3);
        pub const MINUS_ONE: Self = Self::const_neg(Self::R);

        // Maximum length (in bits) of split values, including the
        // sign bit (see `Self::split_vartime()`).
        pub const SPLIT_BITLEN: usize = (Self::BITLEN >> 1) + 2;

        // Encoding length (in bytes) of split values (see
        // `Self::split_vartime()`).
        pub const SPLIT_LEN: usize = (Self::SPLIT_BITLEN + 7) >> 3;

        const P1: u64 = Self::top_u32();
        const P1DIV_M: u64 = 1 + ((((((1u64 << 32) - Self::P1) as u128) << 64)
            / (Self::P1 as u128)) as u64);
        const NUM1: usize = (2 * Self::BITLEN - 34) / 31;
        const NUM2: usize = 2 * Self::BITLEN - 31 * Self::NUM1 - 2;
        const TFIXDIV: Self = Self::const_mmul(Self::const_mmul(
            Self::pow2mod(Self::NUM1 * 33 + 64 - Self::NUM2),
            Self::R2), Self::R2);
        const SQRT_EXP: [u64; Self::N] = Self::const_sqrt_exp();

        // Create an element from its 64-bit limbs, provided in little-endian
        // order (least significant limb first). This function is meant to be
        // used in constant expressions (constant-time evaluation). It is
        // also safe to use at runtime, but slower than the (non-const)
        // from_w64le() function.
        //
        // Note: if the value is numerically larger than the modulus, then
        // it is implicitly reduced.
        pub const fn w64le(x: [u64; Self::N]) -> Self {
            Self::const_mmul(Self(x), Self::R2)
        }

        // Create an element from its 64-bit limbs, provided in big-endian
        // order (most significant limb first). This function is meant to be
        // used in constant expressions (constant-time evaluation). It is
        // also safe to use at runtime, but slower than the (non-const)
        // from_w64le() function.
        //
        // Note: if the value is numerically larger than the modulus, then
        // it is implicitly reduced.
        pub const fn w64be(x: [u64; Self::N]) -> Self {
            Self::w64le(Self::const_rev(x))
        }

        // Create an element from its 64-bit limbs, provided in little-endian
        // order (least significant limb first). This function is faster than
        // w64le(), but can be used only at runtime, not in const expressions.
        //
        // Note: if the value is numerically larger than the modulus, then
        // it is implicitly reduced.
        #[inline(always)]
        pub fn from_w64le(x: [u64; Self::N]) -> Self {
            let mut r = Self(x);
            r.set_mul(&Self::R2);
            r
        }

        // Create an element from its 64-bit limbs, provided in big-endian
        // order (most significant limb first). This function is faster than
        // w64be(), but can be used only at runtime, not in const expressions.
        //
        // Note: if the value is numerically larger than the modulus, then
        // it is implicitly reduced.
        #[inline(always)]
        pub fn from_w64be(x: [u64; Self::N]) -> Self {
            let mut y = [0u64; Self::N];
            for i in 0..Self::N {
                y[i] = x[Self::N - 1 - i];
            }
            Self::from_w64le(y)
        }

        // Create an element by converting the provided integer (implicitly
        // reduced modulo the field order).
        #[inline(always)]
        pub fn from_i32(x: i32) -> Self {
            let mut d = [0u64; Self::N];
            d[0] = x as u64;
            let mut r = Self::from_w64le(d);
            r.set_cond(&(r - Self::T64), (x >> 31) as u32);
            r
        }

        // Create an element by converting the provided integer (implicitly
        // reduced modulo the field order).
        #[inline(always)]
        pub fn from_u32(x: u32) -> Self {
            let mut d = [0u64; Self::N];
            d[0] = x as u64;
            Self::from_w64le(d)
        }

        // Create an element by converting the provided integer (implicitly
        // reduced modulo the field order).
        #[inline(always)]
        pub fn from_i64(x: i64) -> Self {
            let mut d = [0u64; Self::N];
            d[0] = x as u64;
            let mut r = Self::from_w64le(d);
            r.set_cond(&(r - Self::T64), (x >> 63) as u32);
            r
        }

        // Create an element by converting the provided integer (implicitly
        // reduced modulo the field order).
        #[inline(always)]
        pub fn from_u64(x: u64) -> Self {
            let mut d = [0u64; Self::N];
            d[0] = x;
            Self::from_w64le(d)
        }

        // Create an element by converting the provided integer (implicitly
        // reduced modulo the field order).
        #[inline(always)]
        pub fn from_i128(x: i128) -> Self {
            let mut r = Self::from_u128(x as u128);
            r.set_cond(&(r - Self::T128), (x >> 127) as u32);
            r
        }

        // Create an element by converting the provided integer (implicitly
        // reduced modulo the field order).
        #[inline(always)]
        pub fn from_u128(x: u128) -> Self {
            if Self::N == 1 {
                let r = Self::from_u64((x >> 64) as u64);
                (r * Self::R2) + Self::from_u64(x as u64)
            } else {
                let mut d = [0u64; Self::N];
                d[0] = x as u64;
                d[1] = (x >> 64) as u64;
                Self::from_w64le(d)
            }
        }

        #[inline]
        fn set_add(&mut self, rhs: &Self) {
            let mut cc1 = 0;
            for i in 0..Self::N {
                (self.0[i], cc1) = addcarry_u64(
                    self.0[i], rhs.0[i], cc1);
            }
            let mut cc2 = 0;
            for i in 0..Self::N {
                (self.0[i], cc2) = subborrow_u64(
                    self.0[i], Self::MODULUS[i], cc2);
            }
            let cc1 = (cc1 as u64).wrapping_neg();
            let cc2 = (cc2 as u64).wrapping_neg();
            let m = cc2 & !cc1;
            let mut cc3 = 0;
            for i in 0..Self::N {
                (self.0[i], cc3) = addcarry_u64(
                    self.0[i], m & Self::MODULUS[i], cc3);
            }
        }

        #[inline]
        fn set_sub(&mut self, rhs: &Self) {
            let mut cc1 = 0;
            for i in 0..Self::N {
                (self.0[i], cc1) = subborrow_u64(
                    self.0[i], rhs.0[i], cc1);
            }
            let m = (cc1 as u64).wrapping_neg();
            let mut cc2 = 0;
            for i in 0..Self::N {
                (self.0[i], cc2) = addcarry_u64(    
                    self.0[i], m & Self::MODULUS[i], cc2);
            }
        }

        // Negate this element.
        #[inline]
        pub fn set_neg(&mut self) {
            let mut cc1 = 0;
            for i in 0..Self::N {
                (self.0[i], cc1) = subborrow_u64(
                    0, self.0[i], cc1);
            }
            let m = (cc1 as u64).wrapping_neg();
            let mut cc2 = 0;
            for i in 0..Self::N {
                (self.0[i], cc2) = addcarry_u64(
                    self.0[i], m & Self::MODULUS[i], cc2);
            }
        }

        #[inline]
        pub fn set_cond(&mut self, a: &Self, ctl: u32) {
            let cw = ((ctl as i32) as i64) as u64;
            for i in 0..Self::N {
                self.0[i] ^= cw & (self.0[i] ^ a.0[i]);
            }
        }

        #[inline]
        pub fn select(a0: &Self, a1: &Self, ctl: u32) -> Self {
            let mut r = *a0;
            r.set_cond(a1, ctl);
            r
        }

        #[inline]
        pub fn cswap(a: &mut Self, b: &mut Self, ctl: u32) {
            let cw = ((ctl as i32) as i64) as u64;
            for i in 0..Self::N {
                let t = cw & (a.0[i] ^ b.0[i]);
                a.0[i] ^= t;
                b.0[i] ^= t;
            }
        }

        fn set_montyred(&mut self) {
            for _ in 0..Self::N {
                let f = self.0[0].wrapping_mul(Self::M0I);
                let (_, mut cc) = umull_add(f, Self::MODULUS[0], self.0[0]);
                for i in 1..Self::N {
                    let (d, hi) = umull_add2(
                        f, Self::MODULUS[i], self.0[i], cc);
                    self.0[i - 1] = d;
                    cc = hi;
                }
                self.0[Self::N - 1] = cc;
            }
        }

        fn set_mul(&mut self, rhs: &Self) {
            let mut t = Self::ZERO;

            // combined muls + reduction
            let mut cch = 0;
            for i in 0..Self::N {
                let f = rhs.0[i];
                let (lo, mut cc1) = umull_add(f, self.0[0], t.0[0]);
                let g = lo.wrapping_mul(Self::M0I);
                let (_, mut cc2) = umull_add(g, Self::MODULUS[0], lo);
                for j in 1..Self::N {
                    let (d, hi1) = umull_add2(f, self.0[j], t.0[j], cc1);
                    cc1 = hi1;
                    let (d, hi2) = umull_add2(g, Self::MODULUS[j], d, cc2);
                    cc2 = hi2;
                    t.0[j - 1] = d;
                }
                let (d, ee) = addcarry_u64(cc1, cc2, cch);
                t.0[Self::N - 1] = d;
                cch = ee;
            }

            // final reduction: subtract modulus if necessary
            let mut cc = 0;
            for i in 0..Self::N {
                let (d, ee) = subborrow_u64(t.0[i], Self::MODULUS[i], cc);
                t.0[i] = d;
                cc = ee;
            }
            let mm = (cch as u64).wrapping_sub(cc as u64);
            cc = 0;
            for i in 0..Self::N {
                let (d, ee) = addcarry_u64(t.0[i], mm & Self::MODULUS[i], cc);
                self.0[i] = d;
                cc = ee;
            }
        }

        pub fn set_square(&mut self) {
            // If instructed so, we use the generic multiplication support
            // for squarings. The special code may or may not be faster,
            // depending on the modulus size, value, and target architecture.
            if !($squarespec) {
                let r = *self;
                self.set_mul(&r);
                return;
            }

            // Compute the square over integers.
            let mut t = [0u64; Self::N << 1];

            // sum_{i<j} a_i*a_j*2^(64*(i+j)) < 2^(64*(2*N-1))
            // -> t[2*N-1] remains at zero
            let f = self.0[0];
            let (d, mut cc) = umull(f, self.0[1]);
            t[1] = d;
            for j in 2..Self::N {
                let (d, hi) = umull_add(f, self.0[j], cc);
                t[j] = d;
                cc = hi;
            }
            t[Self::N] = cc;
            for i in 1..(Self::N - 1) {
                let f = self.0[i];
                let (d, mut cc) = umull_add(f, self.0[i + 1], t[(i << 1) + 1]);
                t[(i << 1) + 1] = d;
                for j in (i + 2)..Self::N {
                    let (d, hi) = umull_add2(f, self.0[j], t[i + j], cc);
                    t[i + j] = d;
                    cc = hi;
                }
                t[i + Self::N] = cc;
            }

            // Double the partial sum.
            // -> t contains sum_{i!=j} a_i*a_j*2^(64*(i+j))
            let mut cc = 0;
            for i in 1..((Self::N << 1) - 1) {
                let w = t[i];
                let ee = w >> 63;
                t[i] = (w << 1) | cc;
                cc = ee;
            }
            t[(Self::N << 1) - 1] = cc;

            // Add the squares a_i*a_i*w^(64*2*i).
            let mut cc = 0;
            for i in 0..Self::N {
                let (lo, hi) = umull(self.0[i], self.0[i]);
                let (d0, ee) = addcarry_u64(lo, t[i << 1], cc);
                let (d1, ee) = addcarry_u64(hi, t[(i << 1) + 1], ee);
                t[i << 1] = d0;
                t[(i << 1) + 1] = d1;
                cc = ee;
            }

            // Apply Montgomery reduction. We use the following facts:
            //  - upper half is necessarily less than p
            //  - set_montyred() accepts a full-limbs input and outputs a
            //    value of at most p
            //  - set_add() tolerates an input operand equal to p provided
            //    that the sum is less than 2*p
            self.0.copy_from_slice(&t[..Self::N]);
            self.set_montyred();
            let mut y = Self([0u64; Self::N]);
            y.0.copy_from_slice(&t[Self::N..]);
            self.set_add(&y);
        }

        /// Compute the square of this value.
        #[inline(always)]
        pub fn square(self) -> Self {
            let mut r = self;
            r.set_square();
            r
        }

        /// Compute the square of this value.
        #[inline(always)]
        pub fn set_xsquare(&mut self, n: u32) {
            for _ in 0..n {
                self.set_square();
            }
        }

        /// Compute the square of this value.
        #[inline(always)]
        pub fn xsquare(self, n: u32) -> Self {
            let mut r = self;
            r.set_xsquare(n);
            r
        }

        /// Halve this value.
        #[inline]
        pub fn set_half(&mut self) {
            let m = (self.0[0] & 1).wrapping_neg();
            let (mut dd, mut cc) = addcarry_u64(
                self.0[0], m & Self::MODULUS[0], 0);
            dd >>= 1;
            for i in 1..Self::N {
                let (x, ee) = addcarry_u64(self.0[i], m & Self::MODULUS[i], cc);
                cc = ee;
                self.0[i - 1] = dd | (x << 63);
                dd = x >> 1;
            }
            self.0[Self::N - 1] = dd | ((cc as u64) << 63);
        }

        /// Compute the half of this value.
        #[inline(always)]
        pub fn half(self) -> Self {
            let mut r = self;
            r.set_half();
            r
        }

        #[inline(always)]
        fn set_mul2(&mut self) {
            let r = *self;
            self.set_add(&r);
        }

        #[inline(always)]
        pub fn mul2(self) -> Self {
            let mut r = self;
            r.set_mul2();
            r
        }

        #[inline(always)]
        fn set_mul3(&mut self) {
            let r = *self;
            self.set_add(&r);
            self.set_add(&r);
        }

        #[inline(always)]
        pub fn mul3(self) -> Self {
            let mut r = self;
            r.set_mul3();
            r
        }

        #[inline(always)]
        fn set_mul4(&mut self) {
            let r = *self;
            self.set_add(&r);
            let r = *self;
            self.set_add(&r);
        }

        #[inline(always)]
        pub fn mul4(self) -> Self {
            let mut r = self;
            r.set_mul4();
            r
        }

        #[inline(always)]
        fn set_mul8(&mut self) {
            let r = *self;
            self.set_add(&r);
            let r = *self;
            self.set_add(&r);
            let r = *self;
            self.set_add(&r);
        }

        #[inline(always)]
        pub fn mul8(self) -> Self {
            let mut r = self;
            r.set_mul8();
            r
        }

        #[inline(always)]
        fn set_mul16(&mut self) {
            let r = *self;
            self.set_add(&r);
            let r = *self;
            self.set_add(&r);
            let r = *self;
            self.set_add(&r);
            let r = *self;
            self.set_add(&r);
        }

        #[inline(always)]
        pub fn mul16(self) -> Self {
            let mut r = self;
            r.set_mul16();
            r
        }

        #[inline(always)]
        fn set_mul32(&mut self) {
            let r = *self;
            self.set_add(&r);
            let r = *self;
            self.set_add(&r);
            let r = *self;
            self.set_add(&r);
            let r = *self;
            self.set_add(&r);
            let r = *self;
            self.set_add(&r);
        }

        #[inline(always)]
        pub fn mul32(self) -> Self {
            let mut r = self;
            r.set_mul32();
            r
        }

        /// Multiply this value by a small integer k.
        pub fn set_mul_small(&mut self, k: u32) {
            // Special case: if the modulus fits on one limb only,
            // use normal conversion.
            if Self::N == 1 {
                *self *= Self::from_u32(k);
                return;
            }

            // Do the product over integers.
            let (d, mut hi) = umull(self.0[0], k as u64);
            self.0[0] = d;
            for i in 1..Self::N {
                let (d, ee) = umull_add(self.0[i], k as u64, hi);
                self.0[i] = d;
                hi = ee;
            }

            // We write:
            //    p = p1*2^m + p0   (modulus)
            //    x = x1*2^m + x0   (unreduced product)
            // with:
            //    2^31 <= p1 < 2^32
            //    0 <= p0 < 2^m
            //    0 <= x0 < 2^m
            // Since the current value x is the product of the input (less
            // than p) by a multiplier of at most 2^31, we know that:
            //    0 <= x < p*2^31 < 2^(63+m)
            //    0 <= x1 < 2^63.
            // We compute:
            //    b = floor(x1/p1)
            // Analysis shows that floor(x/p) = b, b-1 or b+1.
            //
            // We thus obtain b, then increment it (unless b == p1), then
            // subtract b*p from x; we then add back p repeatedly until a
            // non-negative result is obtained. At most two conditional
            // additions are needed to achieve that result.
            //
            // Division by p1 can be done with the Granlund-Montgomery method:
            //    https://dl.acm.org/doi/10.1145/773473.178249
            // (LLVM usually applies that method, but may fail to do so if for
            // instance optimizing for code size on some platforms, thus it is
            // best to apply the method explicitly so that constant-time code
            // is more reliably achieved.)

            // Extract top word of x.
            let bl = Self::BITLEN & 63;
            let x1 = if bl == 0 {
                    (self.0[Self::N - 1] >> 32) | (hi << 32)
                } else if bl < 32 {
                    (self.0[Self::N - 1] << (32 - bl))
                        | (self.0[Self::N - 2] >> (32 + bl))
                } else if bl == 32 {
                    self.0[Self::N - 1]
                } else {
                    (hi << (96 - bl)) | (self.0[Self::N - 1] >> (bl - 32))
                };

            // Compute b = floor(x1/p1).
            let (_, t) = umull(x1, Self::P1DIV_M);
            let b = (x1.wrapping_sub(t) >> 1).wrapping_add(t) >> 31;

            // Add 1 to b, unless b == p1 (we cannot have b > p1).
            let b = b + (Self::P1.wrapping_sub(b) >> 63);

            // Subtract b*p from x.
            let mut cc1 = 0;
            let mut cc2 = 0;
            for i in 0..Self::N {
                let (d, ee) = umull_add(b, Self::MODULUS[i], cc1);
                cc1 = ee;
                let (d, ee) = subborrow_u64(self.0[i], d, cc2);
                self.0[i] = d;
                cc2 = ee;
            }
            let (mut hi, _) = subborrow_u64(hi, cc1, cc2);

            // Add p (at most twice) as long as the value is negative.
            for _ in 0..2 {
                let m = sgnw(hi);
                let mut cc = 0;
                for i in 0..Self::N {
                    let (d, ee) = addcarry_u64(
                        self.0[i], m & Self::MODULUS[i], cc);
                    self.0[i] = d;
                    cc = ee;
                }
                hi = hi.wrapping_add(cc as u64);
            }
        }

        #[inline(always)]
        pub fn mul_small(self, k: u32) -> Self {
            let mut r = self;
            r.set_mul_small(k);
            r
        }

        // Set this value to (u*f+v*g)/2^64. Coefficients f
        // and g are provided as u64, but they are signed integers in the
        // [-2^62..+2^62] range.
        fn set_montylin(&mut self, u: &Self, v: &Self, f: u64, g: u64) {
            // Make sure f and g are non-negative.
            let sf = sgnw(f);
            let f = (f ^ sf).wrapping_sub(sf);
            let tu = Self::select(u, &-u, sf as u32);
            let sg = sgnw(g);
            let g = (g ^ sg).wrapping_sub(sg);
            let tv = Self::select(v, &-v, sg as u32);

            let (d, mut cc) = umull_x2(tu.0[0], f, tv.0[0], g);
            self.0[0] = d;
            for i in 1..Self::N {
                let (d, hi) = umull_x2_add(tu.0[i], f, tv.0[i], g, cc);
                self.0[i] = d;
                cc = hi;
            }
            let up = cc;

            // Montgomery reduction (one round)
            let k = self.0[0].wrapping_mul(Self::M0I);
            let (_, mut cc) = umull_add(k, Self::MODULUS[0], self.0[0]);
            for i in 1..Self::N {
                let (d, hi) = umull_add2(k, Self::MODULUS[i], self.0[i], cc);
                self.0[i - 1] = d;
                cc = hi;
            }
            let (d, cc1) = addcarry_u64(up, cc, 0);
            self.0[Self::N - 1] = d;

            // |f| <= 2^62 and |g| <= 2^62, therefore |u*f + v*g| < p*2^63
            // We added less than p*2^64, and divided by 2^64, so the result
            // is less than 2*p and a single conditional subtraction is enough.
            let mut cc2 = 0;
            for i in 0..Self::N {
                let (d, ee) = subborrow_u64(
                    self.0[i], Self::MODULUS[i], cc2);
                self.0[i] = d;
                cc2 = ee;
            }
            let mm = (cc1 as u64).wrapping_sub(cc2 as u64);
            let mut cc = 0;
            for i in 0..Self::N {
                let (d, ee) = addcarry_u64(
                    self.0[i], mm & Self::MODULUS[i], cc);
                self.0[i] = d;
                cc = ee;
            }
        }

        #[inline(always)]
        fn montylin(a: &Self, b: &Self, f: u64, g: u64) -> Self {
            let mut r = Self::ZERO;
            r.set_montylin(a, b, f, g);
            r
        }

        // Set this value to abs((a*f + b*g)/2^31). Values a and b are
        // interpreted as plain unsigned integers (not modular).
        // Coefficients f and g are provided as u64 but they really are
        // signed integers in the [-2^31..+2^31] range (inclusive). The
        // low 31 bits of a*f + b*g are dropped (i.e. the division is
        // assumed to be exact). The result is assumed to fit in N limbs
        // (extra high bits, if any, are dropped). The absolute value of
        // (a*f + b*g)/2^31 is computed. Returned value is -1 (as a u64)
        // if a*f + b*g was negative, 0 otherwise.
        fn set_lindiv31abs(&mut self, a: &Self, b: &Self, f: u64, g: u64)
            -> u64
        {
            // Replace f and g with abs(f) and abs(g), but remember the
            // original signs.
            let sf = sgnw(f);
            let f = (f ^ sf).wrapping_sub(sf);
            let sg = sgnw(g);
            let g = (g ^ sg).wrapping_sub(sg);

            // Compute a*f + b*g (upper word in 'up')
            let mut cc1 = 0;
            let mut cc2 = 0;
            let mut cc3 = 0;
            for i in 0..Self::N {
                let (d1, ee1) = subborrow_u64(a.0[i] ^ sf, sf, cc1);
                cc1 = ee1;
                let (d2, ee2) = subborrow_u64(b.0[i] ^ sg, sg, cc2);
                cc2 = ee2;
                let (d3, hi3) = umull_x2_add(d1, f, d2, g, cc3);
                self.0[i] = d3;
                cc3 = hi3;
            }
            let up = cc3.wrapping_sub((cc1 as u64).wrapping_neg() & f)
                .wrapping_sub((cc2 as u64).wrapping_neg() & g);

            // Right-shift the result by 31 bits.
            for i in 0..(Self::N - 1) {
                self.0[i] = (self.0[i] >> 31) | (self.0[i + 1] << 33);
            }
            self.0[Self::N - 1] = (self.0[Self::N - 1] >> 31) | (up << 33);

            // Negate the result if (a*f + b*g) was negative.
            let w = sgnw(up);
            let mut cc = 0;
            for i in 0..Self::N {
                let (d, ee) = subborrow_u64(self.0[i] ^ w, w, cc);
                self.0[i] = d;
                cc = ee;
            }

            w
        }

        #[inline(always)]
        fn lindiv31abs(a: &Self, b: &Self, f: u64, g: u64) -> (Self, u64) {
            let mut r = Self::ZERO;
            let ng = r.set_lindiv31abs(a, b, f, g);
            (r, ng)
        }

        /// Divide this value by `y`. If `y` is zero, then this sets this
        /// value to zero.
        fn set_div(&mut self, y: &Self) {
            // a <- y
            // b <- p (modulus)
            // u <- x (self)
            // v <- 0
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
            //        a <- (a - b)/2
            //        u <- (u - v)/2 mod p
            //
            // We optimize this algorithm following:
            //    https://eprint.iacr.org/2020/972

            let mut a = *y;
            let mut b = Self(Self::MODULUS);
            let mut u = *self;
            let mut v = Self::ZERO;

            // Generic loop; each iteration reduces the sum of the sizes
            // of a and b by at least 31, and that sum starts at 2*BITLEN
            // (at most). We need to run it until the sum of the two lengths
            // is at most 64 (i.e. NUM1 outer iterations).
            for _ in 0..Self::NUM1 {
                // Get approximations of a and b over 64 bits:
                //  - If len(a) <= 64 and len(b) <= 64, then we just
                //    use their values (low limbs).
                //  - Otherwise, with n = max(len(a), len(b)), we use:
                //       (a mod 2^31) + 2^31*floor(a / 2^(n - 33))
                //       (b mod 2^31) + 2^31*floor(b / 2^(n - 33))
                let mut c_hi = 0xFFFFFFFFFFFFFFFFu64;
                let mut c_lo = 0xFFFFFFFFFFFFFFFFu64;
                let mut a_hi = 0u64;
                let mut a_lo = 0u64;
                let mut b_hi = 0u64;
                let mut b_lo = 0u64;
                for j in (0..Self::N).rev() {
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
                // If c_lo != 0, then c_hi = 0 (they cannot be both non-zero
                // since that would mean that a = b = 0, but b is odd). In that
                // case, we grabbed one word (in a_hi and b_hi) and both values
                // fit in 64 bits.
                let s = lzcnt(a_hi | b_hi);
                let mut xa = (a_hi << s) | ((a_lo >> 1) >> (63 - s));
                let mut xb = (b_hi << s) | ((b_lo >> 1) >> (63 - s));
                xa = (xa & 0xFFFFFFFF80000000) | (a.0[0] & 0x000000007FFFFFFF);
                xb = (xb & 0xFFFFFFFF80000000) | (b.0[0] & 0x000000007FFFFFFF);

                // If c_lo != 0, then we should ignore the computed xa and xb,
                // and instead use the low limbs directly.
                xa ^= c_lo & (xa ^ a.0[0]);
                xb ^= c_lo & (xb ^ b.0[0]);

                // Compute the 31 inner iterations.
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
                let nu = Self::montylin(&u, &v, f0, g0);
                let nv = Self::montylin(&u, &v, f1, g1);
                a = na;
                b = nb;
                u = nu;
                v = nv;
            }

            // If y is non-zero, then the final GCD is 1, and
            // len(a) + len(b) <= NUM2 + 2 at this point (initially,
            // len(a) + len(b) <= 2*BITLEN, and each outer iteration reduces
            // the total by at least 31). Thus, the two values fit in one word
            // and we can finish the computation that way. We only need NUM2
            // iterations to reach the point where b = 1.
            let mut xa = a.0[0];
            let mut xb = b.0[0];
            let mut f0 = 1u64;
            let mut g0 = 0u64;
            let mut f1 = 0u64;
            let mut g1 = 1u64;
            for _ in 0..Self::NUM2 {
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

            self.set_montylin(&u, &v, f1, g1);

            // If y != 0 then b = 1 at this point. If y == 0, then we
            // force the result to zero.
            let w = !y.iszero();
            let w = ((w as u64) << 32) | (w as u64);
            for i in 0..Self::N {
                self.0[i] &= w;
            }

            // At this point, each outer iteration injected 31 extra doublings,
            // plus NUM2 for the last loop, for a total of NUM1*31 + NUM2.
            // Each montylin() call divided by 2^64, so in total we really
            // divided the value by 2^(64*(NUM1+1) - 31*NUM1 - NUM2).
            //
            // Moreover, both divisor and dividend were in Montgomery
            // representation, so the result is not in Montgomery representation
            // (the two R factors canceled each other). We want the result
            // in Montgomery representation, i.e. multiplied by 2^(64*N).
            // Therefore, we must multiply by 2^(33*NUM1 + 64 - NUM2 + 64*N),
            // which we need in Montgomery representation.
            self.set_mul(&Self::TFIXDIV);
        }

        #[inline(always)]
        pub fn set_invert(&mut self) {
            let r = *self;
            *self = Self::ONE;
            self.set_div(&r);
        }

        #[inline(always)]
        pub fn invert(self) -> Self {
            let mut r = Self::ONE;
            r.set_div(&self);
            r
        }

        // Perform a batch inversion of some elements. All elements of the
        // slice are replaced with their respective inverses (elements of
        // value zero are kept unchanged).
        pub fn batch_invert(xx: &mut [Self]) {
            // We use Montgomery's trick:
            //   1/u = v*(1/(u*v))
            //   1/v = u*(1/(u*v))
            // Applied recursively on n elements, this computes an
            // inversion with a single inversion in the field, and
            // 3*(n-1) multiplications. We use fixed-size sub-batches of
            // elements so that we may use stack allocation.
            const SUBLEN: usize = if $typename::N > 100 {
                10
            } else {
                1024 / $typename::N
            };

            let n = xx.len();
            let mut i = 0;
            while i < n {
                let blen = if (n - i) > SUBLEN { SUBLEN } else { n - i };
                let mut tt = [Self::ZERO; SUBLEN];
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

        /// Legendre symbol on this value. Return value is:
        ///   0   if this value is zero
        ///  +1   if this value is a non-zero quadratic residue
        ///  -1   if this value is not a quadratic residue
        pub fn legendre(self) -> i32 {
            // This is the same optimized binary GCD as in division, except
            // that we do not need to keep track of u and v. We can also
            // work directly on the Montgomery representation because
            // R = 2^(64*N) is a square.
            let mut a = self;
            let mut b = Self(Self::MODULUS);
            let mut ls = 0u64;

            // Generic loop; each iteration reduces the sum of the sizes
            // of a and b by at least 31, and that sum starts at 2*BITLEN
            // (at most). We need to run it until the sum of the two lengths
            // is at most 64.
            for _ in 0..Self::NUM1 {
                // Get approximations of a and b over 64 bits:
                //  - If len(a) <= 64 and len(b) <= 64, then we just
                //    use their values (low limbs).
                //  - Otherwise, with n = max(len(a), len(b)), we use:
                //       (a mod 2^31) + 2^31*floor(a / 2^(n - 33))
                //       (b mod 2^31) + 2^31*floor(b / 2^(n - 33))
                let mut c_hi = 0xFFFFFFFFFFFFFFFFu64;
                let mut c_lo = 0xFFFFFFFFFFFFFFFFu64;
                let mut a_hi = 0u64;
                let mut a_lo = 0u64;
                let mut b_hi = 0u64;
                let mut b_lo = 0u64;
                for j in (0..Self::N).rev() {
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
                // If c_lo != 0, then c_hi = 0 (they cannot be both non-zero
                // since that would mean that a = b = 0, but b is odd). In that
                // case, we grabbed one word (in a_hi and b_hi) and both values
                // fit in 64 bits.
                let s = lzcnt(a_hi | b_hi);
                let mut xa = (a_hi << s) | ((a_lo >> 1) >> (63 - s));
                let mut xb = (b_hi << s) | ((b_lo >> 1) >> (63 - s));
                xa = (xa & 0xFFFFFFFF80000000) | (a.0[0] & 0x000000007FFFFFFF);
                xb = (xb & 0xFFFFFFFF80000000) | (b.0[0] & 0x000000007FFFFFFF);

                // If c_lo != 0, then we should ignore the computed xa and xb,
                // and instead use the low limbs directly.
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

                // Propagate updates to a, b, u and v.
                let (na, nega) = Self::lindiv31abs(&a, &b, f0, g0);
                let (nb, _)    = Self::lindiv31abs(&a, &b, f1, g1);
                ls ^= nega & (nb.0[0] >> 1);
                a = na;
                b = nb;
            }

            // If y is non-zero, then the final GCD is 1, and
            // len(a) + len(b) <= NUM2 + 2 at this point (initially,
            // len(a) + len(b) <= 2*BITLEN, and each outer iteration reduces
            // the total by at least 31). Thus, the two values fit in one word
            // and we can finish the computation that way. We only need NUM2
            // iterations to reach the point where b = 1.
            let mut xa = a.0[0];
            let mut xb = b.0[0];
            for _ in 0..Self::NUM2 {
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

        // Raise this value to the provided exponent. The exponent is non-zero
        // and is public. The exponent is encoded over N 64-bit limbs.
        fn set_modpow_pubexp(&mut self, e: &[u64; Self::N]) {
            // Make a 4-bit window; win[i] contains x^(i+1)
            let mut win = [Self::ZERO; 15];
            win[0] = *self;
            for i in 1..8 {
                let j = i * 2;
                win[j - 1] = win[i - 1].square();
                win[j] = win[j - 1] * win[0];
            }

            // Explore 4-bit chunks of the exponent, high to low. Skip leading
            // chunks of value 0.
            let mut z = false;
            for i in (0..Self::N).rev() {
                let ew = e[i];
                for j in (0..16).rev() {
                    if z {
                        self.set_xsquare(4);
                    }
                    let c = ((ew >> (j << 2)) & 0x0F) as usize;
                    if c != 0 {
                        if z {
                            self.set_mul(&win[c - 1]);
                        } else {
                            z = true;
                            *self = win[c - 1];
                        }
                    }
                }
            }
            if !z {
                *self = Self::ONE;
            }
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
            // Keep a copy of the source value, to check the square root
            // afterwards.
            let x = *self;

            let plo = Self::MODULUS[0];
            if (plo & 3) == 3 {
                // p = 3 mod 4
                // The candidate square root is x^((p+1)/4)
                self.set_modpow_pubexp(&Self::SQRT_EXP);
            } else if (plo & 7) == 5 {
                // p = 5 mod 8; we use Atkin's algorithm:
                //   b <- (2*x)^((p-5)/8)
                //   c <- 2*x*b^2
                //   y <- x*b*(c - 1)
                let mut b = self.mul2();
                b.set_modpow_pubexp(&Self::SQRT_EXP);
                let mut c = self.mul2() * b.square();

                // We really computed c = (2*x)^((p-1)/4), which is a square
                // root of the Legendre symbol of 2*x. With p = 5 mod 8,
                // 2 is not a square. Thus, if the square root of x exists,
                // then c is a square root of -1 (except if x = 0, in which
                // case c = 0). Otherwise, c = 1 or -1 (and not a square root
                // of -1).
                // We compute y = x*b*(c' - 1); then:
                //   y^2 = x*c*(c' - 1)^2/2
                // If c = i or -i, then using c' = c (as mandated by Atkin's
                // formulas) yields c*(c - 1)^2/2 = 1, i.e. y^2 = x, which is
                // the expected result.
                // If c = 1 or -1, then we set c' = 3, so that c*(c' - 1)^2/2
                // is equal to 2 or -2, and y^2 = 2*x or -2*x.
                let ff = c.equals(Self::ONE) | c.equals(Self::MINUS_ONE);
                c.set_cond(&Self::THREE, ff);
                *self *= b * (c - Self::ONE);
            } else {
                // General case is Tonelli-Shanks but it requires knowledge
                // of a non-QR in the field, which we don't provide in the
                // type parameters.
                // TODO: implement that case, preferably along the lines of:
                //    https://eprint.iacr.org/2023/828
                unimplemented!();
            }

            // Normalize square root so that its least significant bit is 0.
            self.set_cond(&-(self as &Self),
                ((self.encode()[0] as u32) & 1).wrapping_neg());

            // Check computed square root to set the result status. The
            // original value is in x.
            self.square().equals(x)
        }

        #[inline(always)]
        pub fn sqrt_ext(self) -> (Self, u32) {
            let mut x = self;
            let r = x.set_sqrt_ext();
            (x, r)
        }

        // Set this value to its square root. Returned value is 0xFFFFFFFF
        // if the operation succeeded (value was indeed a quadratic residue),
        // 0 otherwise (value was not a quadratic residue). In the latter case,
        // this value is set to zero as well.
        // When this operation succeeds, the returned square root is the one
        // whose least significant bit is 0 (when normalized in 0..q-1).
        //
        // This operation returns unspecified results if the modulus is not
        // prime. If the modulus q is prime but is equal to 1 modulo 8, then
        // the method is not implemented (which triggers a panic).
        #[inline(always)]
        fn set_sqrt(&mut self) -> u32 {
            let r = self.set_sqrt_ext();
            self.set_cond(&Self::ZERO, !r);
            r
        }

        #[inline(always)]
        pub fn sqrt(self) -> (Self, u32) {
            let mut x = self;
            let r = x.set_sqrt();
            (x, r)
        }

        // Split this scalar into two signed integers (c0, c1) such that
        // self = c0/c1. The two signed integers are returned in _signed_
        // little-endian encoding. If the modulus is p, then it is
        // guaranteed that c0^2 and c1^2 are lower than p*2/sqrt(3). If
        // the modulus bit length is n bits, then the solutions must fit
        // in cl = floor(n/2)+2 bits (including the sign bit), and this
        // function returns c0 and c1 as arrays of ceil(cl/8) bytes.
        // The constants `Self::SPLIT_BITLEN` and `Self::SPLIT_LEN` contain,
        // respectively,  the lengths in bits and in bytes of c0 and c1.
        //
        // WARNING: this function is not implemented for moduli larger than
        // 2^512 (TODO).
        //
        // This function is NOT constant-time; it must be used only on
        // public scalar values (e.g. when verifying signatures).
        pub fn split_vartime(self)
            -> ([u8; Self::SPLIT_LEN], [u8; Self::SPLIT_LEN])
        {
            // Get the value in normal representation (not Montgomery).
            let mut k = self;
            k.set_montyred();

            // If the modulus is too short, then we "expand" it to 4 words.
            // We use as target bit length half the size of the modulus,
            // rounded low, so that we never get that way a solution
            // greater than the announced maximum. The output arrays must
            // have length at least half of the modulus.
            const DX: usize = ($typename::SPLIT_BITLEN + 63) >> 6;
            const DLEN: usize = if DX > 2 && DX > (($typename::N + 1) >> 1) {
                ($typename::N + 1) >> 1
            } else {
                DX
            };
            let mut d0 = [0u64; DLEN];
            let mut d1 = [0u64; DLEN];
            if Self::N < 4 {
                let mut nt = [0u64; 4];
                let mut kt = [0u64; 4];
                nt[..Self::N].copy_from_slice(&Self::MODULUS);
                kt[..Self::N].copy_from_slice(&k.0);
                lagrange_vartime(&kt, &nt,
                    (Self::BITLEN >> 1) as u32, &mut d0, &mut d1);
            } else {
                lagrange_vartime(&k.0, &Self::MODULUS,
                    (Self::BITLEN >> 1) as u32, &mut d0, &mut d1);
            }

            // If this modulus is too close to some limit, then the obtained
            // values may have been truncated, which requires a corrective
            // action here. This may happen only if the number of limbs N is
            // even. If the real values are c0 and c1, and we obtained d0
            // and d1, then:
            //
            //    k = c0/c1
            //    c0 = d0 + a*2^(32*N)   with a \in { -1, 0, +1 }
            //    c1 = d1 + b*2^(32*N)   with b \in { -1, 0, +1 }
            //
            // Thus:
            //    (d1*k - d0)/2^(32*N) = a - b*k
            //
            // We compute e = (d1*k - d0)/2^(32*N); if it is not equal
            // to 0, 1 or -1, then we try adding or subtracting k.
            //
            // The compiler should optimize away the code below when the
            // modulus makes it so that it is not necessary.
            let mut c0h = sgnw(d0[DLEN - 1]) as u8;
            let mut c1h = sgnw(d1[DLEN - 1]) as u8;
            if Self::SPLIT_LEN > (DLEN << 3) {
                // thalf is 2^(32*N) in normal (non-Montgomery) representation.
                let mut thalf = Self::ZERO;
                thalf.0[DLEN] = 1;

                // Let (s0, s1) = (d0, d1) as field elements in normal
                // (non-Montgomery) representation.
                let mut s0 = Self::ZERO;
                let mut s1 = Self::ZERO;
                s0.0[..DLEN].copy_from_slice(&d0);
                s1.0[..DLEN].copy_from_slice(&d1);
                if (d0[DLEN - 1] >> 63) != 0 {
                    s0 -= thalf;
                }
                if (d1[DLEN - 1] >> 63) != 0 {
                    s1 -= thalf;
                }

                // Since s0 is in non-Montgomery representation,
                // multiplying by self (which is k in Montgomery
                // representation) yield d0*k in normal representation.
                // Moreover, thalf = 2^(32*N), which is the Montgomery
                // representation of 1/2^(32*N).
                let mut e = (s1 * self - s0) * thalf - k;
                let mut one = Self::ZERO;
                one.0[0] = 1;
                let mut minus_one = Self(Self::MODULUS);
                minus_one.0[0] &= !1u64;
                let mut a = -100i32;
                let mut b = -1i32;
                for _ in 0..3 {
                    if e.iszero() != 0 {
                        a = 0;
                        break;
                    }
                    if e.equals(one) != 0 {
                        a = 1;
                        break;
                    }
                    if e.equals(minus_one) != 0 {
                        a = -1;
                        break;
                    }
                    e += k;
                    b += 1;
                }
                assert!(a != -100);

                c0h = c0h.wrapping_add(a as u8);
                c1h = c1h.wrapping_add(b as u8);
            }

            // We can encode the two outputs; if we don't have enough bytes
            // is d0/d1, then we use d0h and d1h for the top byte.
            let mut c0 = [0u8; Self::SPLIT_LEN];
            let mut c1 = [0u8; Self::SPLIT_LEN];
            for i in 0..Self::SPLIT_LEN {
                let j = i >> 3;
                if j < DLEN {
                    c0[i] = (d0[j] >> ((i & 7) << 3)) as u8;
                    c1[i] = (d1[j] >> ((i & 7) << 3)) as u8;
                } else {
                    c0[i] = c0h;
                    c1[i] = c1h;
                }
            }

            (c0, c1)
        }

        // Equality check (constant-time): returned value is 0xFFFFFFFF on
        // equality, 0 otherwise.
        #[inline]
        pub fn equals(self, rhs: Self) -> u32 {
            // Values have a single valid internal representation, so we
            // can do a simple comparison.
            let mut r = 0;
            for i in 0..Self::N {
                r |= self.0[i] ^ rhs.0[i];
            }
            ((r | r.wrapping_neg()) >> 63).wrapping_sub(1) as u32
        }

        // Compare this value with zero (constant-time): returned value
        // is 0xFFFFFFFF if this element is zero, 0 otherwise.
        #[inline]
        pub fn iszero(self) -> u32 {
            // Values have a single valid internal representation, so we
            // can do a simple comparison.
            let mut r = 0;
            for i in 0..Self::N {
                r |= self.0[i];
            }
            ((r | r.wrapping_neg()) >> 63).wrapping_sub(1) as u32
        }

        // Encode this value into bytes (unsigned little-endian encoding
        // of the value, normalized to [0..p-1], with the same size as
        // the modulus).
        pub fn encode(self) -> [u8; Self::ENC_LEN] {
            let mut x = self;
            x.set_montyred();
            let mut d = [0u8; Self::ENC_LEN];
            let mut j = 0;
            for i in 0..Self::N {
                if (j + 8) <= Self::ENC_LEN {
                    d[j..(j + 8)].copy_from_slice(&x.0[i].to_le_bytes());
                    j += 8;
                } else {
                    let k = Self::ENC_LEN - j;
                    d[j..].copy_from_slice(&x.0[i].to_le_bytes()[..k]);
                }
            }
            d
        }

        // Decode up to 8*N bytes from the provided slice, into an integer
        // value, with unsigned little-endian convention. The value is stored
        // as-is, unreduced and not converted to Montgomery representation.
        fn set_decode_raw(&mut self, buf: &[u8]) {
            let n = buf.len();
            let mut j = 0;
            for i in 0..Self::N {
                self.0[i] = if j + 8 < n {
                        u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[j..(j + 8)]).unwrap())
                    } else if j < n {
                        let k = n - j;
                        let mut tmp = [0u8; 8];
                        tmp[..k].copy_from_slice(&buf[j..]);
                        u64::from_le_bytes(tmp)
                    } else {
                        0
                    };
                j += 8;
            }
        }

        // Decode a value from bytes. If the provided slice length
        // matches the modulus length exactly (`Self::ENC_LEN`), _and_
        // the unsigned little-endian interpretation of these bytes is
        // an integer in the 0 to p-1 range, then this value is set to
        // that integer, and 0xFFFFFFFF is returned. Otherwise (wrong length,
        // or value not lower than the modulus), this value is set to zero,
        // and 0x00000000 is returned.
        #[inline]
        pub fn set_decode_ct(&mut self, buf: &[u8]) -> u32 {
            if buf.len() != Self::ENC_LEN {
                // We cannot hide from side-channels the length of the
                // input slice, so we can return early here.
                *self = Self::ZERO;
                return 0;
            }
            self.set_decode_raw(buf);

            // Subtracting the modulus must yield a borrow; otherwise, this
            // is a non-canonical input.
            let mut cc = 0;
            for i in 0..Self::N {
                (_, cc) = subborrow_u64(self.0[i], Self::MODULUS[i], cc);
            }
            let r = (cc as u32).wrapping_neg();
            self.set_cond(&Self::ZERO, !r);

            // Convert to Montgomery representation.
            self.set_mul(&Self::R2);
            r
        }

        // Decode a value from bytes. If the provided slice length
        // matches the modulus length exactly (`Self::ENC_LEN`), _and_
        // the unsigned little-endian interpretation of these bytes is
        // an integer in the 0 to p-1 range, then the corresponding field
        // element is returned, along with the 0xFFFFFFFF status. Otherwise
        // (wrong length, or value not lower than the modulus), the field
        // element zero and the 0x00000000 status are returned.
        #[inline(always)]
        pub fn decode_ct(buf: &[u8]) -> (Self, u32) {
            let mut x = Self::ZERO;
            let r = x.set_decode_ct(buf);
            (x, r)
        }

        // Decode a value from bytes. If the provided slice length
        // matches the modulus length exactly (`Self::ENC_LEN`), _and_
        // the unsigned little-endian interpretation of these bytes is
        // an integer in the 0 to p-1 range, then the corresponding field
        // element is returned. Otherwise (wrong length, or value not lower
        // than the modulus), `None` is returned. Side-channel analysis
        // may reveal to outsiders whether the decoding succeeded or not.
        #[inline(always)]
        pub fn decode(buf: &[u8]) -> Option<Self> {
            let (x, r) = Self::decode_ct(buf);
            if r != 0 {
                Some(x)
            } else {
                None
            }
        }

        // Decode a value from bytes. The source bytes are interpreted as
        // the unsigned little-endian representation of an integer, which
        // is then reduced modulo the field modulus. This function cannot
        // fail.
        pub fn set_decode_reduce(&mut self, buf: &[u8]) {
            let n = buf.len();
            if n == 0 {
                *self = Self::ZERO;
                return;
            }

            // We process the bytes by chunks, in high-to-low order.
            // The current value is in Montgomery representation. For
            // each new chunk, we add the raw chunk value, _then_
            // multiply by R (i.e. Montgomery-multiply by R2), which
            // combines the shift of the current value and the
            // conversion to Montgomery representation. That addition
            // must be a custom routine because the raw value is not
            // properly reduced; on the other hand, the Montgomery
            // multiplication can accept an unreduced input, and outputs
            // a reduced value. Thus, it suffices to make a raw addition
            // on integers, and subtract the modulus only on a non-zero
            // carry.

            // Get top chunk and convert to Montgomery representation.
            const CHUNK_LEN: usize = 8 * $typename::N;
            let mut j = n - (n % CHUNK_LEN);
            if j == n {
                j -= CHUNK_LEN;
            }
            self.set_decode_raw(&buf[j..]);
            self.set_mul(&Self::R2);

            // Process subsequent chunks.
            while j >= CHUNK_LEN {
                j -= CHUNK_LEN;

                // Decode next chunk into x.
                let mut x = Self::ZERO;
                x.set_decode_raw(&buf[j..]);

                // Add x to current value; if necessary, subtract the
                // modulus (one subtraction is enough, since the current
                // value is reduced).
                let mut cc = 0;
                for i in 0..Self::N {
                    let (w, ee) = addcarry_u64(
                        self.0[i], x.0[i], cc);
                    self.0[i] = w;
                    cc = ee;
                }
                let m = (cc as u64).wrapping_neg();
                cc = 0;
                for i in 0..Self::N {
                    let (w, ee) = subborrow_u64(
                        self.0[i], m & Self::MODULUS[i], cc);
                    self.0[i] = w;
                    cc = ee;
                }

                // Apply Montgomery conversion / chunk shift.
                self.set_mul(&Self::R2);
            }
        }

        // Decode a value from bytes. The source bytes are interpreted as
        // the unsigned little-endian representation of an integer, which
        // is then reduced modulo the field modulus. This function cannot
        // fail.
        #[inline(always)]
        pub fn decode_reduce(buf: &[u8]) -> Self {
            let mut r = Self::ZERO;
            r.set_decode_reduce(buf);
            r
        }

        // =================================================================
        // Below are support functions for compile-time computation of
        // constants.

        // Return -1/x mod 2^64. It is assumed that x is odd.
        const fn ninv64(x: u64) -> u64 {
            let y = 2u64.wrapping_sub(x);
            let y = y.wrapping_mul(2u64.wrapping_sub(y.wrapping_mul(x)));
            let y = y.wrapping_mul(2u64.wrapping_sub(y.wrapping_mul(x)));
            let y = y.wrapping_mul(2u64.wrapping_sub(y.wrapping_mul(x)));
            let y = y.wrapping_mul(2u64.wrapping_sub(y.wrapping_mul(x)));
            let y = y.wrapping_mul(2u64.wrapping_sub(y.wrapping_mul(x)));
            y.wrapping_neg()
        }

        // Custom add-with-carry, for use in const (compile-time) contexts.
        const fn adc(x: u64, y: u64, cc: u64) -> (u64, u64) {
            let z = (x as u128) + (y as u128) + (cc as u128);
            (z as u64, (z >> 64) as u64)
        }

        // Custom sub-with-borrow, for use in const (compile-time) contexts.
        const fn sbb(x: u64, y: u64, cc: u64) -> (u64, u64) {
            let z = (x as u128)
                .wrapping_sub(y as u128)
                .wrapping_sub(cc as u128);
            (z as u64, ((z >> 64) as u64).wrapping_neg())
        }

        // Subtract the modulus, return borrow (compile-time).
        const fn subm(a: Self) -> (Self, u64) {
            const fn subm_inner(a: $typename, cc: u64, i: usize)
                -> ($typename, u64)
            {
                if i == a.0.len() {
                    (a, cc)
                } else {
                    let (d, cc) = $typename::sbb(
                        a.0[i], $typename::MODULUS[i], cc);
                    let mut aa = a;
                    aa.0[i] = d;
                    subm_inner(aa, cc, i + 1)
                }
            }

            subm_inner(a, 0, 0)
        }

        // Add the modulus if mm == -1; return a unchanged with mm == 0
        // (compile-time).
        const fn addm_cond(a: $typename, mm: u64) -> $typename {
            const fn addm_cond_inner(a: $typename, mm: u64,
                cc: u64, i: usize) -> $typename
            {
                if i == a.0.len() {
                    a
                } else {
                    let (d, cc) = $typename::adc(
                        a.0[i], $typename::MODULUS[i] & mm, cc);
                    let mut aa = a;
                    aa.0[i] = d;
                    addm_cond_inner(aa, mm, cc, i + 1)
                }
            }

            addm_cond_inner(a, mm, 0, 0)
        }

        // Get index of the top non-zero word of the modulus
        // (from the parameters).
        const fn top_word_index() -> usize {
            const fn top_word_index_inner(j: usize) -> usize {
                if $fieldparams::MODULUS[j] != 0 {
                    j
                } else {
                    top_word_index_inner(j - 1)
                }
            }
            top_word_index_inner($fieldparams::MODULUS.len() - 1)
        }

        // Get the modulus (normalized).
        const fn make_modulus() -> [u64; Self::N] {
            const fn make_modulus_inner(d: [u64; $typename::N], j: usize)
                -> [u64; $typename::N]
            {
                if j == $typename::N {
                    d
                } else {
                    let mut dd = d;
                    dd[j] = $fieldparams::MODULUS[j];
                    make_modulus_inner(dd, j + 1)
                }
            }
            make_modulus_inner([0u64; Self::N], 0)
        }

        // Compute the modulus exact bit length (compile-time).
        const fn mod_bitlen() -> usize {
            const fn bitlen(x: u64, max: usize) -> usize {
                if max == 1 {
                    x as usize
                } else {
                    let hm = max >> 1;
                    let y = x >> hm;
                    if y == 0 {
                        bitlen(x, hm)
                    } else {
                        bitlen(y, max - hm) + hm
                    }
                }
            }
            (Self::N - 1) * 64 + bitlen($typename::MODULUS[Self::N - 1], 64)
        }

        // Get the top 32 bits of the actual modulus value (if the modulus
        // is less than 32 bits in length, then this returns the modulus).
        const fn top_u32() -> u64 {
            if Self::BITLEN < 32 {
                Self::MODULUS[0]
            } else {
                let hi = Self::MODULUS[Self::N - 1];
                let bl = Self::BITLEN & 63;
                if bl == 0 {
                    hi >> 32
                } else if bl < 32 {
                    let lo = Self::MODULUS[Self::N - 2];
                    (hi << (32 - bl)) | (lo >> (bl + 32))
                } else {
                    hi >> (bl - 32)
                }
            }
        }

        // Compute 2^n in the field, using repeated additions. This is
        // used only at compile-time.
        const fn pow2mod(n: usize) -> Self {

            const fn lsh_inner(d: $typename, cc: u64, i: usize)
                -> ($typename, u64)
            {
                if i == $typename::N {
                    (d, cc)
                } else {
                    let mut dd = d;
                    dd.0[i] = (d.0[i] << 1) | cc;
                    let cc = d.0[i] >> 63;
                    lsh_inner(dd, cc, i + 1)
                }
            }

            const fn pow2mod_inner(d: $typename, n: usize) -> $typename {
                if n == 0 {
                    d
                } else {
                    let (d, dh) = lsh_inner(d, 0, 0);
                    let (d, cc) = $typename::subm(d);
                    let d = $typename::addm_cond(d, (cc & !dh).wrapping_neg());
                    pow2mod_inner(d, n - 1)
                }
            }

            const fn pow2mod_inner2(d: $typename, n: usize) -> $typename {
                if n <= 25 {
                    pow2mod_inner(d, n)
                } else {
                    pow2mod_inner2(pow2mod_inner(d, 25), n - 25)
                }
            }

            let bl = Self::mod_bitlen();
            let mut d = Self([0u64; Self::N]);
            if n < bl {
                d.0[n >> 6] = 1u64 << (n & 63);
                d
            } else {
                d.0[(bl - 1) >> 6] = 1u64 << ((bl - 1) & 63);
                pow2mod_inner2(d, n - (bl - 1))
            }
        }

        // Const implementation of modular negation. This MUST NOT be
        // applied on zero.
        const fn const_neg(a: Self) -> Self {
            const fn const_neg_inner(d: $typename, a: $typename,
                cc: u64, j: usize) -> $typename
            {
                if j == $typename::N {
                    d
                } else {
                    let mut dd = d;
                    let (x, cc) = $typename::sbb(
                        $typename::MODULUS[j], a.0[j], cc);
                    dd.0[j] = x;
                    const_neg_inner(dd, a, cc, j + 1)
                }
            }
            const_neg_inner(Self::ZERO, a, 0, 0)
        }

        // Const implementation of Montgomery multiplication. It uses
        // recursion in order to be compatible with the constraints of
        // const code; at runtime, it would be slower than the normal
        // implementation, but still constant-time (in case it gets
        // mistakenly used).
        const fn const_mmul(a: Self, b: Self) -> Self {

            const fn umaal(x: u64, y: u64, a: u64, b: u64) -> (u64, u64) {
                let z = (x as u128) * (y as u128) + (a as u128) + (b as u128);
                (z as u64, (z >> 64) as u64)
            }

            const fn mmul1(d: $typename, dh: u64, a: $typename, bj: u64)
                -> ($typename, u64)
            {
                const fn mmul1_inner(d: $typename, dh: u64, a: $typename,
                    bj: u64, fm: u64, cc1: u64, cc2: u64, i: usize)
                    -> ($typename, u64)
                {
                    if i == d.0.len() {
                        let mut dd = d;
                        let (z, zh1) = $typename::adc(dh, cc1, 0);
                        let (z, zh2) = $typename::adc(z, cc2, 0);
                        dd.0[i - 1] = z;
                        (dd, zh1 + zh2)
                    } else {
                        let (z, cc1) = umaal(a.0[i], bj, d.0[i], cc1);
                        let (z, cc2) = umaal($typename::MODULUS[i], fm, z, cc2);
                        let mut dd = d;
                        if i > 0 {
                            dd.0[i - 1] = z;
                        }
                        mmul1_inner(dd, dh, a, bj, fm, cc1, cc2, i + 1)
                    }
                }

                let fm = a.0[0].wrapping_mul(bj).wrapping_add(d.0[0])
                    .wrapping_mul($typename::M0I);
                mmul1_inner(d, dh, a, bj, fm, 0, 0, 0)
            }

            const fn mmul_inner(d: $typename, dh: u64,
                a: $typename, b: $typename, j: usize) -> ($typename, u64)
            {
                if j == d.0.len() {
                    (d, dh)
                } else {
                    let (d, dh) = mmul1(d, dh, a, b.0[j]);
                    mmul_inner(d, dh, a, b, j + 1)
                }
            }

            let (d, dh) = mmul_inner(Self([0u64; $typename::N]), 0, a, b, 0);
            let (d, cc) = $typename::subm(d);
            $typename::addm_cond(d, (cc & !dh).wrapping_neg())
        }

        const fn const_rev(x: [u64; Self::N]) -> [u64; Self::N] {
            const fn const_rev_inner(x: [u64; $typename::N], i: usize)
                -> [u64; $typename::N]
            {
                let j = $typename::N - 1 - i;
                if j <= i {
                    x
                } else {
                    let mut y = x;
                    y[i] = x[j];
                    y[j] = x[i];
                    const_rev_inner(y, i + 1)
                }
            }
            const_rev_inner(x, 0)
        }

        const fn const_small(x: u64) -> Self {
            let mut d = [0u64; Self::N];
            d[0] = x;
            Self::const_mmul(Self(d), Self::R2)
        }

        const fn const_sqrt_exp() -> [u64; Self::N] {
            const fn const_sqrt_exp_3mod4(d: [u64; $typename::N],
                cc: u64, dd: u64, i: usize) -> [u64; $typename::N]
            {
                if i == $typename::N {
                    let mut d2 = d;
                    d2[$typename::N - 1] = dd;
                    d2
                } else {
                    let (x, cc) = $typename::adc($typename::MODULUS[i], 0, cc);
                    let mut d2 = d;
                    if i > 0 {
                        d2[i - 1] = dd | (x << 62);
                    }
                    const_sqrt_exp_3mod4(d2, cc, x >> 2, i + 1)
                }
            }

            const fn const_sqrt_exp_5mod8(d: [u64; $typename::N], i: usize)
                -> [u64; $typename::N]
            {
                let mut d2 = d;
                d2[i] = $typename::MODULUS[i] >> 3;
                if i < ($typename::N - 1) {
                    d2[i] |= $typename::MODULUS[i + 1] << 61;
                    const_sqrt_exp_5mod8(d2, i + 1)
                } else {
                    d2
                }
            }

            if ($typename::MODULUS[0] & 3) == 3 {
                const_sqrt_exp_3mod4([0u64; Self::N], 1, 0, 0)
            } else if ($typename::MODULUS[0] & 7) == 5 {
                const_sqrt_exp_5mod8([0u64; Self::N], 0)
            } else {
                [0u64; Self::N]
            }
        }
    }

    // ========================================================================
    // Implementations of all the traits needed to use the simple operators
    // (+, *, /...) on field element instances, with or without references.

    impl core::ops::Add<$typename> for $typename {
        type Output = $typename;

        #[inline(always)]
        fn add(self, other: $typename) -> $typename {
            let mut r = self;
            r.set_add(&other);
            r
        }
    }

    impl core::ops::Add<&$typename> for $typename {
        type Output = $typename;

        #[inline(always)]
        fn add(self, other: &$typename) -> $typename {
            let mut r = self;
            r.set_add(other);
            r
        }
    }

    impl core::ops::Add<$typename> for &$typename {
        type Output = $typename;

        #[inline(always)]
        fn add(self, other: $typename) -> $typename {
            let mut r = *self;
            r.set_add(&other);
            r
        }
    }

    impl core::ops::Add<&$typename> for &$typename {
        type Output = $typename;

        #[inline(always)]
        fn add(self, other: &$typename) -> $typename {
            let mut r = *self;
            r.set_add(other);
            r
        }
    }

    impl core::ops::AddAssign<$typename> for $typename {
        #[inline(always)]
        fn add_assign(&mut self, other: $typename) {
            self.set_add(&other);
        }
    }

    impl core::ops::AddAssign<&$typename> for $typename {
        #[inline(always)]
        fn add_assign(&mut self, other: &$typename) {
            self.set_add(other);
        }
    }

    impl core::ops::Div<$typename> for $typename {
        type Output = $typename;

        #[inline(always)]
        fn div(self, other: $typename) -> $typename {
            let mut r = self;
            r.set_div(&other);
            r
        }
    }

    impl core::ops::Div<&$typename> for $typename {
        type Output = $typename;

        #[inline(always)]
        fn div(self, other: &$typename) -> $typename {
            let mut r = self;
            r.set_div(other);
            r
        }
    }

    impl core::ops::Div<$typename> for &$typename {
        type Output = $typename;

        #[inline(always)]
        fn div(self, other: $typename) -> $typename {
            let mut r = *self;
            r.set_div(&other);
            r
        }
    }

    impl core::ops::Div<&$typename> for &$typename {
        type Output = $typename;

        #[inline(always)]
        fn div(self, other: &$typename) -> $typename {
            let mut r = *self;
            r.set_div(other);
            r
        }
    }

    impl core::ops::DivAssign<$typename> for $typename {
        #[inline(always)]
        fn div_assign(&mut self, other: $typename) {
            self.set_div(&other);
        }
    }

    impl core::ops::DivAssign<&$typename> for $typename {
        #[inline(always)]
        fn div_assign(&mut self, other: &$typename) {
            self.set_div(other);
        }
    }

    impl core::ops::Mul<$typename> for $typename {
        type Output = $typename;

        #[inline(always)]
        fn mul(self, other: $typename) -> $typename {
            let mut r = self;
            r.set_mul(&other);
            r
        }
    }

    impl core::ops::Mul<&$typename> for $typename {
        type Output = $typename;

        #[inline(always)]
        fn mul(self, other: &$typename) -> $typename {
            let mut r = self;
            r.set_mul(other);
            r
        }
    }

    impl core::ops::Mul<$typename> for &$typename {
        type Output = $typename;

        #[inline(always)]
        fn mul(self, other: $typename) -> $typename {
            let mut r = *self;
            r.set_mul(&other);
            r
        }
    }

    impl core::ops::Mul<&$typename> for &$typename {
        type Output = $typename;

        #[inline(always)]
        fn mul(self, other: &$typename) -> $typename {
            let mut r = *self;
            r.set_mul(other);
            r
        }
    }

    impl core::ops::MulAssign<$typename> for $typename {
        #[inline(always)]
        fn mul_assign(&mut self, other: $typename) {
            self.set_mul(&other);
        }
    }

    impl core::ops::MulAssign<&$typename> for $typename {
        #[inline(always)]
        fn mul_assign(&mut self, other: &$typename) {
            self.set_mul(other);
        }
    }

    impl core::ops::Neg for $typename {
        type Output = $typename;

        #[inline(always)]
        fn neg(self) -> $typename {
            let mut r = self;
            r.set_neg();
            r
        }
    }

    impl core::ops::Neg for &$typename {
        type Output = $typename;

        #[inline(always)]
        fn neg(self) -> $typename {
            let mut r = *self;
            r.set_neg();
            r
        }
    }

    impl core::ops::Sub<$typename> for $typename {
        type Output = $typename;

        #[inline(always)]
        fn sub(self, other: $typename) -> $typename {
            let mut r = self;
            r.set_sub(&other);
            r
        }
    }

    impl core::ops::Sub<&$typename> for $typename {
        type Output = $typename;

        #[inline(always)]
        fn sub(self, other: &$typename) -> $typename {
            let mut r = self;
            r.set_sub(other);
            r
        }
    }

    impl core::ops::Sub<$typename> for &$typename {
        type Output = $typename;

        #[inline(always)]
        fn sub(self, other: $typename) -> $typename {
            let mut r = *self;
            r.set_sub(&other);
            r
        }
    }

    impl core::ops::Sub<&$typename> for &$typename {
        type Output = $typename;

        #[inline(always)]
        fn sub(self, other: &$typename) -> $typename {
            let mut r = *self;
            r.set_sub(other);
            r
        }
    }

    impl core::ops::SubAssign<$typename> for $typename {
        #[inline(always)]
        fn sub_assign(&mut self, other: $typename) {
            self.set_sub(&other);
        }
    }

    impl core::ops::SubAssign<&$typename> for $typename {
        #[inline(always)]
        fn sub_assign(&mut self, other: &$typename) {
            self.set_sub(other);
        }
    }

    } // sub-module

} } // End of macro: define_gfgen

pub use define_gfgen;

// ========================================================================

#[macro_export]
macro_rules! define_gfgen_tests { ($typename:ident, $nqr:expr, $submod:ident) => {

    #[cfg(test)]
    mod $submod {

    use super::$typename;

    /*
    fn print(name: &str, v: $typename) {
        print!("{} = 0x", name);
        for i in (0..$typename::MODULUS.len()).rev() {
            print!("{:016X}", v.0[i]);
        }
        println!();
    }
    */

    // va, vb and vx must have length equal to that of the encoding length
    // of the field.
    fn check_gf_ops(va: &[u8], vb: &[u8], vx: &[u8]) {
        use num_bigint::{BigInt, Sign};
        use core::convert::TryFrom;

        let mut zpmw = [0u32; $typename::MODULUS.len() * 2];
        for i in 0..$typename::MODULUS.len() {
            zpmw[2 * i] = $typename::MODULUS[i] as u32;
            zpmw[2 * i + 1] = ($typename::MODULUS[i] >> 32) as u32;
        }
        let zp = BigInt::from_slice(Sign::Plus, &zpmw);
        let zpz = &zp << 64;

        let x = u32::from_le_bytes(*<&[u8; 4]>::try_from(&vb[..4]).unwrap());
        let a = $typename::decode_reduce(va);
        let b = $typename::decode_reduce(vb);
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
        let zd = ((&zpz + &za) - &zb) % &zp;
        assert!(zc == zd);

        let c = -a;
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&zpz - &za) % &zp;
        assert!(zc == zd);

        let c = a * b;
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za * &zb) % &zp;
        assert!(zc == zd);

        let c = a.half();
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd: BigInt = ((&zpz + (&zc << 1)) - &za) % &zp;
        assert!(zd.sign() == Sign::NoSign);

        let c = a.mul2();
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za << 1) % &zp;
        assert!(zc == zd);

        let c = a.mul3();
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za + (&za << 1)) % &zp;
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

        let c = a.mul_small(x);
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = ((&za * x) + &zpz) % &zp;
        assert!(zc == zd);

        let c = a.square();
        let vc = c.encode();
        let zc = BigInt::from_bytes_le(Sign::Plus, &vc);
        let zd = (&za * &za) % &zp;
        assert!(zc == zd);

        let (e, cc) = $typename::decode_ct(va);
        if cc != 0 {
            assert!(cc == 0xFFFFFFFF);
            assert!(e.encode() == va);
        } else {
            assert!(e.encode() == [0u8; $typename::ENC_LEN]);
        }

        let mut tmp = [0u8; 3 * $typename::ENC_LEN];
        tmp[0..$typename::ENC_LEN].copy_from_slice(va);
        tmp[$typename::ENC_LEN..(2 * $typename::ENC_LEN)].copy_from_slice(vb);
        tmp[(2 * $typename::ENC_LEN)..].copy_from_slice(vx);
        for k in 0..(tmp.len() + 1) {
            let c = $typename::decode_reduce(&tmp[0..k]);
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

    fn mkrndv(vv: &mut [u8; $typename::ENC_LEN], bx: u64) {
        use sha2::{Sha512, Digest};

        let mut sh = Sha512::new();
        let mut j = 0;
        while j < $typename::ENC_LEN {
            sh.update((bx + ((j as u64) << 40)).to_le_bytes());
            if (j + 64) < $typename::ENC_LEN {
                vv[j..(j + 64)].copy_from_slice(&sh.finalize_reset()[..]);
            } else {
                vv[j..].copy_from_slice(&sh.finalize_reset()[..($typename::ENC_LEN - j)]);
            };
            j += 64;
        }
    }

    fn mkrnd(bx: u64) -> $typename {
        let mut vv = [0u8; $typename::ENC_LEN];
        mkrndv(&mut vv, bx);
        $typename::decode_reduce(&vv)
    }

    #[test]
    fn field_ops() {
        let mut va = [0u8; $typename::ENC_LEN];
        let mut vb = [0u8; $typename::ENC_LEN];
        let mut vx = [0u8; $typename::ENC_LEN];
        check_gf_ops(&va, &vb, &vx);
        assert!($typename::decode_reduce(&va).iszero() == 0xFFFFFFFF);
        assert!($typename::decode_reduce(&va).equals($typename::decode_reduce(&vb)) == 0xFFFFFFFF);
        assert!($typename::decode_reduce(&va).legendre() == 0);
        for i in 0..$typename::ENC_LEN {
            va[i] = 0xFFu8;
            vb[i] = 0xFFu8;
            vx[i] = 0xFFu8;
        }
        check_gf_ops(&va, &vb, &vx);
        let mut vcorr = [0u8; $typename::SPLIT_LEN + 1];
        vcorr[vcorr.len() - 1] = 1;
        let corr = $typename::decode_reduce(&vcorr);
        for i in 0..300 {
            mkrndv(&mut va, 3 * i + 0);
            mkrndv(&mut vb, 3 * i + 1);
            mkrndv(&mut vx, 3 * i + 2);
            check_gf_ops(&va, &vb, &vx);
            if $typename::MODULUS.len() > 1 {
                assert!($typename::decode_reduce(&va).iszero() == 0);
                assert!($typename::decode_reduce(&va).equals($typename::decode_reduce(&vb)) == 0);
            }
            let nqr = ($nqr) as u32;
            let s = $typename::decode_reduce(&va).square();
            let s2 = s.mul_small(nqr);
            assert!(s.legendre() == 1);
            assert!(s2.legendre() == -1);

            let plo = $typename::MODULUS[0];
            if (plo & 7) != 1 {
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
                if (plo & 3) == 3 {
                    assert!(t2.square().equals(-s2) == 0xFFFFFFFF);
                } else {
                    let y = t2.square();
                    let z = s2.mul2();
                    assert!((y.equals(z) | y.equals(-z)) == 0xFFFFFFFF);
                }
            }

            let a = $typename::decode_reduce(&va);
            let (c0, c1) = a.split_vartime();
            let mut b0 = $typename::decode_reduce(&c0);
            if c0[c0.len() - 1] >= 0x80 {
                b0 -= corr;
            }
            let mut b1 = $typename::decode_reduce(&c1);
            if c1[c1.len() - 1] >= 0x80 {
                b1 -= corr;
            }
            assert!((a * b1).equals(b0) == 0xFFFFFFFF);
        }
    }

    #[test]
    fn batch_invert() {
        let mut xx = [$typename::ZERO; 300];
        for i in 0..300 {
            xx[i] = mkrnd((10000 + i) as u64);
        }
        xx[120] = $typename::ZERO;
        let mut yy = xx;
        $typename::batch_invert(&mut yy[..]);
        for i in 0..300 {
            if xx[i].iszero() != 0 {
                assert!(yy[i].iszero() == 0xFFFFFFFF);
            } else {
                assert!((xx[i] * yy[i]).equals($typename::ONE) == 0xFFFFFFFF);
            }
        }
    }

    } // end of module

} } // End of macro: define_gfgen_tests

pub use define_gfgen_tests;
