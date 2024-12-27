use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::convert::TryFrom;

use super::{sgnw};

/// Element of GF(2^127), using modulus 1 + z^63 + z^127.
#[derive(Clone, Copy, Debug)]
pub struct GFb127([u64; 2]);

impl GFb127 {

    // IMPLEMENTATION NOTES
    // --------------------
    //
    // We tolerate internal values up to 128 bits. All computations are
    // performed modulo z + z^64 + z^128, which makes reductions easier
    // (z^64 and z^128 are 64-bit aligned).

    pub const ZERO: Self = Self([ 0, 0 ]);
    pub const ONE: Self = Self([ 1, 0 ]);

    pub const fn w64le(x0: u64, x1: u64) -> Self {
        Self([ x0, x1 ])
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
        ((x.0[k >> 6] >> (k & 63)) as u32) & 1
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
        let ki = k >> 6;
        let kj = k & 63;
        self.0[ki] &= !(1u64 << kj);
        self.0[ki] |= ((val & 1) as u64) << kj;
    }

    // XOR (add) a one-bit value at the specified index. The index `k`
    // MUST be between 0 and 126 (inclusive). Side-channel attacks may
    // reveal the value of the index (bit not the value of the added bit).
    // Only the least significant bit of `val` is used; the over bits
    // are ignored.
    #[inline(always)]
    pub fn xor_bit(&mut self, k: usize, val: u32) {
        self.0[k >> 6] ^= ((val & 1) as u64) << (k & 63);
    }

    #[inline(always)]
    fn set_add(&mut self, rhs: &Self) {
        self.0[0] ^= rhs.0[0];
        self.0[1] ^= rhs.0[1];
    }

    // Subtraction is the same thing as addition in binary fields.

    #[inline(always)]
    pub fn set_cond(&mut self, a: &Self, ctl: u32) {
        let cw = ((ctl as i32) as i64) as u64;
        self.0[0] ^= cw & (self.0[0] ^ a.0[0]);
        self.0[1] ^= cw & (self.0[1] ^ a.0[1]);
    }

    #[inline(always)]
    pub fn select(a0: &Self, a1: &Self, ctl: u32) -> Self {
        let mut r = *a0;
        r.set_cond(a1, ctl);
        r
    }

    #[inline(always)]
    pub fn cswap(a: &mut Self, b: &mut Self, ctl: u32) {
        let cw = ((ctl as i32) as i64) as u64;
        let t = cw & (a.0[0] ^ b.0[0]); a.0[0] ^= t; b.0[0] ^= t;
        let t = cw & (a.0[1] ^ b.0[1]); a.0[1] ^= t; b.0[1] ^= t;
    }

    // Multiply this value by sb = 1 + z^27.
    #[inline(always)]
    pub fn set_mul_sb(&mut self) {
        let (a0, a1) = (self.0[0], self.0[1]);
        let c0 = a0 ^ (a0 << 27);
        let c1 = a1 ^ (a0 >> 37) ^ (a1 << 27);
        let c2 = a1 >> 37;
        self.0[0] = c0 ^ (c2 << 1);
        self.0[1] = c1 ^ c2;
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
        let (a0, a1) = (self.0[0], self.0[1]);
        let c0 = a0 ^ (a0 << 54);
        let c1 = a1 ^ (a0 >> 10) ^ (a1 << 54);
        let c2 = a1 >> 10;
        self.0[0] = c0 ^ (c2 << 1);
        self.0[1] = c1 ^ c2;
    }

    // Multiply this value by bb = 1 + z^54.
    #[inline(always)]
    pub fn mul_b(self) -> Self {
        let mut x = self;
        x.set_mul_b();
        x
    }

    // Divide this value by z.
    #[inline(always)]
    pub fn set_div_z(&mut self) {
        let (a0, a1) = (self.0[0], self.0[1]);
        let b = (a0 & 1) << 62;
        self.0[0] = ((a0 >> 1) | (a1 << 63)) ^ b;
        self.0[1] = (a1 >> 1) ^ b;
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
        let (a0, a1) = (self.0[0], self.0[1]);
        let bb = (a0 & 3) << 61;
        self.0[0] = ((a0 >> 2) | (a1 << 62)) ^ bb;
        self.0[1] = (a1 >> 2) ^ bb;
    }

    // Divide this value by z.
    #[inline(always)]
    pub fn div_z2(self) -> Self {
        let mut x = self;
        x.set_div_z2();
        x
    }

    #[inline(always)]
    fn set_mul(&mut self, rhs: &Self) {
        // We cannot do full 64x64->128 multiplications with 4-bit spacing,
        // because that means that up to 16 individual bits may accumulate
        // in a given position, leading to a carry spill (the value 16
        // requires 5 bits). This may impact only one of the low bits of
        // the high output half; the low 64-bit output is still correct.
        // We thus use bit-reversal: if rev_n() is the bit-reversal function
        // over n bits, and trunc_n is the truncation of a value to its
        // n least significant bits, then, given two inputs x and y over
        // 64 bits each, we have:
        //    clmul(rev_64(x), rev_64(y)) = rev_127(clmul(x, y))
        // Thus, we can get the high half of a 64x64->127 carryless product
        // by getting the low half of the carryless product of the
        // bit-reversed operands.
        //
        // Our target platform for this code is a RISC-V SiFive U74 core;
        // it supports the Zbb extension, hence the rev8 opcode which
        // performs byte-level reversal of a 64-bit value; we must still
        // reverse the individual bits in each value. We can couple that
        // operation with the 4-way split needed for our use of integer
        // multiplications (4-bit spacing of data bits to absorbate carries).

        // Carryless product of two 64-bit values; output is truncated to
        // its low 64 bits.
        #[inline(always)]
        fn mm64low(x: u64, y: u64) -> u64 {
            // We do one level of Karatsuba-Ofman, splitting values between
            // even-indexed and odd-indexed. Compared to a plain cross-mul
            // implementation, we do fewer mul opcodes (12 instead of 16)
            // but we have a few extra shifts and XORs. It seems that the
            // biggest savings come from reducing the register pressure
            // by having fewer "in flight" values, and also using only two
            // masking constants instead of four.

            // xe*ye
            let xe0 = x & 0x1111111111111111;
            let xe1 = x & 0x4444444444444444;
            let ye0 = y & 0x1111111111111111;
            let ye1 = y & 0x4444444444444444;
            let ze0 = (xe0.wrapping_mul(ye0)
                     ^ xe1.wrapping_mul(ye1)) & 0x1111111111111111;
            let ze1 = (xe0.wrapping_mul(ye1)
                     ^ xe1.wrapping_mul(ye0)) & 0x4444444444444444;
            let ze = ze0 ^ ze1;

            // xo*yo
            let xo0 = (x >> 1) & 0x1111111111111111;
            let xo1 = (x >> 1) & 0x4444444444444444;
            let yo0 = (y >> 1) & 0x1111111111111111;
            let yo1 = (y >> 1) & 0x4444444444444444;
            let zo0 = (xo0.wrapping_mul(yo0)
                     ^ xo1.wrapping_mul(yo1)) & 0x1111111111111111;
            let zo1 = (xo0.wrapping_mul(yo1)
                     ^ xo1.wrapping_mul(yo0)) & 0x4444444444444444;
            let zo = zo0 ^ zo1;

            // xt*yt
            let xt0 = xe0 ^ xo0;
            let xt1 = xe1 ^ xo1;
            let yt0 = ye0 ^ yo0;
            let yt1 = ye1 ^ yo1;
            let zt0 = (xt0.wrapping_mul(yt0)
                     ^ xt1.wrapping_mul(yt1)) & 0x1111111111111111;
            let zt1 = (xt0.wrapping_mul(yt1)
                     ^ xt1.wrapping_mul(yt0)) & 0x4444444444444444;
            let zt = zt0 ^ zt1;

            // We can use an addition here to put together the
            // even-indexed and odd-indexed bits. The operation
            // 'c + 2*d' can be done with a single sh1add opcode.
            (ze ^ (zo << 2)) + ((ze ^ zo ^ zt) << 1)
        }

        // Carryless product of two 64-bit values; output is truncated to
        // its low 64 bits. The two inputs are provided in already 4-way
        // split format (`x1`, `x3`, `y1` and `y3` must also be pre-shifted
        // by 1 bit to the right).
        #[inline(always)]
        fn mm64low_presplit(x0: u64, x1: u64, x2: u64, x3: u64,
                            y0: u64, y1: u64, y2: u64, y3: u64) -> u64
        {
            // See `mm64low()` (above) for comments.

            // xe*ye
            let xe0 = x0;
            let xe1 = x2;
            let ye0 = y0;
            let ye1 = y2;
            let ze0 = (xe0.wrapping_mul(ye0)
                     ^ xe1.wrapping_mul(ye1)) & 0x1111111111111111;
            let ze1 = (xe0.wrapping_mul(ye1)
                     ^ xe1.wrapping_mul(ye0)) & 0x4444444444444444;
            let ze = ze0 ^ ze1;

            // xo*yo
            let xo0 = x1;
            let xo1 = x3;
            let yo0 = y1;
            let yo1 = y3;
            let zo0 = (xo0.wrapping_mul(yo0)
                     ^ xo1.wrapping_mul(yo1)) & 0x1111111111111111;
            let zo1 = (xo0.wrapping_mul(yo1)
                     ^ xo1.wrapping_mul(yo0)) & 0x4444444444444444;
            let zo = zo0 ^ zo1;

            // xt*yt
            let xt0 = xe0 ^ xo0;
            let xt1 = xe1 ^ xo1;
            let yt0 = ye0 ^ yo0;
            let yt1 = ye1 ^ yo1;
            let zt0 = (xt0.wrapping_mul(yt0)
                     ^ xt1.wrapping_mul(yt1)) & 0x1111111111111111;
            let zt1 = (xt0.wrapping_mul(yt1)
                     ^ xt1.wrapping_mul(yt0)) & 0x4444444444444444;
            let zt = zt0 ^ zt1;

            (ze ^ (zo << 2)) + ((ze ^ zo ^ zt) << 1)
        }

        // Bit-reversal of a 64-bit value, and 4-way split of the result.
        // Output values 1 and 3 are furthermore shifted by 1 bit to the
        // right, because that's what `mm64low_presplit()` expects.
        #[inline(always)]
        fn rev64_split(x: u64) -> (u64, u64, u64, u64) {
            // If the platform has an efficient bit-reversal opcode, then
            // it would be cheaper to use x.reverse_bits() then four ANDs.
            // However, our target platform (RISC-V SiFive U74) does not
            // have such an opcode (the Zbkb extension offers brev8, which
            // bit-reverses all bytes within a value, but that extension
            // is not implemented by the U74).
            let t = x.swap_bytes();

            // Swapping 4-bit nibbles within each byte. We use an expression
            // inspired from Strachey, as described in "Hacker's Delight",
            // 2nd edition, section 7-1. Strachey's algorithm step below
            // would entail a rotation of t to the right by 4 bits, which
            // we merge into the shifts used for the 4-way split.
            let v = t & 0x0F0F0F0F0F0F0F0F;
            let t = v.rotate_left(8) | (t ^ v);
            ((t.rotate_right(7)) & 0x1111111111111111,
             (t & 0x4444444444444444).rotate_right(6),
             (t.rotate_right(3)) & 0x4444444444444444,
             (t & 0x1111111111111111).rotate_right(2))
        }

        #[inline(always)]
        fn rev64(x: u64) -> u64 {
            // Using Strachey here does not seem to improve performance.
            // On platforms with an actual "reverse all bits" opcode, the
            // plain `reverse_bits()` function will be much faster than
            // anything else we could do here.
            x.reverse_bits()
        }

        let (a0, a1) = (self.0[0], self.0[1]);
        let (b0, b1) = (rhs.0[0], rhs.0[1]);

        // Low half of each product.
        let z0l = mm64low(a0, b0);
        let z1l = mm64low(a1, b1);
        let z2l = mm64low(a0 ^ a1, b0 ^ b1);

        // Bit-reversal and 4-way split.
        let (x00, x01, x02, x03) = rev64_split(a0);
        let (x10, x11, x12, x13) = rev64_split(a1);
        let (x20, x21, x22, x23) = (x00 ^ x10, x01 ^ x11, x02 ^ x12, x03 ^ x13);
        let (y00, y01, y02, y03) = rev64_split(b0);
        let (y10, y11, y12, y13) = rev64_split(b1);
        let (y20, y21, y22, y23) = (y00 ^ y10, y01 ^ y11, y02 ^ y12, y03 ^ y13);

        // High half of each product, using the bit-reversed operands.
        let z0h = mm64low_presplit(x00, x01, x02, x03, y00, y01, y02, y03);
        let z1h = mm64low_presplit(x10, x11, x12, x13, y10, y11, y12, y13);
        let z2h = mm64low_presplit(x20, x21, x22, x23, y20, y21, y22, y23);
        let z0h = rev64(z0h) >> 1;
        let z1h = rev64(z1h) >> 1;
        let z2h = rev64(z2h) >> 1;

        // Reassemble the complete carryless product (Karatsuba-Ofman).
        let h0 = z0l;
        let h1 = z0h ^ z2l ^ z0l ^ z1l;
        let h2 = z1l ^ z2h ^ z0h ^ z1h;
        let h3 = z1h;

        // Reduction.
        let k = h2 ^ h3;
        let m0 = h0 ^ (k << 1);
        let m1 = h1 ^ (k >> 63) ^ (h3 << 1) ^ k;

        self.0[0] = m0;
        self.0[1] = m1;
    }

    /* unused -- alternate implementation of set_mul(); values are split
       into three chunks each, leading to 12 21x44 multiplications, which
       can be done with 3-bit spacing and keeping to 64-bit values. In
       practice it is slightly slower than the other version, though it
       can be faster if byte/bit reversal of a 64-bit value is slow on
       a given system.

    fn set_mul(&mut self, rhs: &Self) {

        // Multiplication of binary polynomials. Size limits:
        //    len(x) <= 44
        //    len(y) <= 42
        // Returned values are w0 and w1 such that x*y = w0 + w1*z^21.
        // Note that w0 and w1 can both be up to 64 bits in length.
        #[inline]
        fn mm42(x: u64, y: u64) -> (u64, u64) {

            #[inline(always)]
            fn mm21(x: u64, y: u64) -> u64 {
                let x0 = x & 0x9249249249249249;
                let x1 = x & 0x2492492492492492;
                let x2 = x & 0x4924924924924924;
                let y0 = y & 0x9249249249249249;
                let y1 = y & 0x2492492492492492;
                let y2 = y & 0x4924924924924924;

                let z0 = (x0.wrapping_mul(y0)
                    ^ x1.wrapping_mul(y2)
                    ^ x2.wrapping_mul(y1)) & 0x9249249249249249;
                let z1 = (x0.wrapping_mul(y1)
                    ^ x1.wrapping_mul(y0)
                    ^ x2.wrapping_mul(y2)) & 0x2492492492492492;
                let z2 = (x0.wrapping_mul(y2)
                    ^ x1.wrapping_mul(y1)
                    ^ x2.wrapping_mul(y0)) & 0x4924924924924924;

                z0 ^ z1 ^ z2
            }

            let z0 = mm21(x, y & 0x00000000001FFFFF);
            let z1 = mm21(x, y >> 21);

            (z0, z1)
        }

        let (a0, a1) = (self.0[0], self.0[1]);
        let (mut b0, mut b1) = (rhs.0[0], rhs.0[1]);

        // Normalize b to make it fit in 127 bits.
        let bh = b1 & 0x8000000000000000;
        b0 ^= bh ^ (b1 >> 63);
        b1 ^= bh;

        // Split in basis z^42:
        //   a = c0 + c1*z^42 + c2*z^84
        //   b = d0 + d1*z^42 + d2*z^84 + dh*z^126
        // dh \in {0, 1} (handled separately)
        // All other values are over 42 bits, except c2 which uses 44 bits.
        let c0 = a0 & 0x000003FFFFFFFFFF;
        let c1 = (a0 >> 42) | ((a1 << 22) & 0x000003FFFFFFFFFF);
        let c2 = a1 >> 20;

        let dhm = sgnw(b1 << 1);
        let d0 = b0 & 0x000003FFFFFFFFFF;
        let d1 = (b0 >> 42) | ((b1 << 22) & 0x000003FFFFFFFFFF);
        let d2 = (b1 >> 20) & 0x000003FFFFFFFFFF;

        // Compute:
        //   e0 = c0*d0
        //   e1 = c1*d1
        //   e2 = c2*d2
        //   e3 = (c0 + c1)*(d0 + d1)
        //   e4 = (c0 + c2)*(d0 + d2)
        //   e5 = (c1 + c2)*(d1 + d2)
        // Then:
        //   (c0 + c1*z^42 + c2*z^84)*(d0 + d1*z^42 + d2*z^84)
        //    =   c0*d0
        //      + (c0*d1 + c1*d0)*z^42
        //      + (c0*d2 + c1*d1 + c2*d0)*z^84
        //      + (c1*d2 + c2*d1)*z^126
        //      + c2*d2*z^168
        //    =   e0
        //      + (e3 + e0 + e1)*z^42
        //      + (e4 + e0 + e1 + e2)*z^84
        //      + (e5 + e1 + e2)*z^126
        //      + e2*z^168
        // This is a reduction of a 3x3 multiplication into 6 smaller
        // multiplications. A reduction to 5 smaller multiplications is
        // possible with Toom-Cook, but it implies a division by an
        // unpleasant constant (i.e. not z^j for some integer j), which can
        // be ignored for asymptotic complexity analysis, but is, in
        // practical terms, a bother.
        //
        // Note that all right operands of the small multiplications fit in
        // 42 bits, while the left operands never exceeds 44 bits, which
        // fits within the range supported by mm42().
        let (e0l, e0h) = mm42(c0, d0);
        let (e1l, e1h) = mm42(c1, d1);
        let (e2l, e2h) = mm42(c2, d2);
        let (e3l, e3h) = mm42(c0 ^ c1, d0 ^ d1);
        let (e4l, e4h) = mm42(c0 ^ c2, d0 ^ d2);
        let (e5l, e5h) = mm42(c1 ^ c2, d1 ^ d2);

        // Add the e* values together, still in basis z^21 with limbs of
        // 64 bits. We get ten values.
        let (f0, f1) = (e0l,                    e0h);
        let (f2, f3) = (e3l ^ e0l ^ e1l,        e3h ^ e0h ^ e1h);
        let (f4, f5) = (e4l ^ e0l ^ e1l ^ e2l,  e4h ^ e0h ^ e1h ^ e2h);
        let (f6, f7) = (e5l ^ e1l ^ e2l,        e5h ^ e1h ^ e2h);
        let (f8, f9) = (e2l,                    e2h);

        // We reassemble the chunks into basis z^64.
        let g0 = f0 ^ (f1 << 21) ^ (f2 << 42) ^ (f3 << 63);
        let g1 = (f1 >> 43) ^ (f2 >> 22) ^ (f3 >> 1) ^ (f4 << 20) ^ (f5 << 41) ^ (f6 << 62);
        let g2 = (f4 >> 44) ^ (f5 >> 23) ^ (f6 >> 2) ^ (f7 << 19) ^ (f8 << 40) ^ (f9 << 61);
        let g3 = (f7 >> 45) ^ (f8 >> 24) ^ (f9 >> 3);

        // We also plug in the value a*dh*z^126 (we had to exclude dh so
        // that all d* could fit in 42 bits).
        let h0 = g0;
        let h1 = g1 ^ ((dhm & a0) << 62);
        let h2 = g2 ^ ((dhm & a0) >> 2) ^ ((dhm & a1) << 62);
        let h3 = g3 ^ ((dhm & a1) >> 2);

        // Reduction: z^128 = z + z^64
        // Note: h0..3 has length at most 254 bits (not 256), so h3 fits
        // on 62 bits.
        //
        // (h2 + h3*z^64)*z^128
        //  = (h2 + h3*z^64)*z + h2*z^64 + h3*(z + z^64)
        //  = (h2 + h3 + h3*z^64)*z + (h2 + h3)*z^64
        let k = h2 ^ h3;
        let m0 = h0 ^ (k << 1);
        let m1 = h1 ^ (k >> 63) ^ (h3 << 1) ^ k;

        self.0[0] = m0;
        self.0[1] = m1;
    }
    */

    // Extract all even-indexed bits from the input and push them into
    // the low 32 bits of the result; the high 32 bits are set to 0.
    #[inline(always)]
    fn squeeze(x: u64) -> u64 {
        let x = (x & 0x1111111111111111) | ((x & 0x4444444444444444) >> 1);
        let x = (x & 0x0303030303030303) | ((x & 0x3030303030303030) >> 2);
        let x = (x & 0x000F000F000F000F) | ((x & 0x0F000F000F000F00) >> 4);
        let x = (x & 0x000000FF000000FF) | ((x & 0x00FF000000FF0000) >> 8);
        let x = (x & 0x000000000000FFFF) | ((x & 0x0000FFFF00000000) >> 16);
        x
    }

    // Square this value (in place).
    #[inline(always)]
    pub fn set_square(&mut self) {
        // Squaring of a 32-bit value.
        #[inline(always)]
        fn expand_32(x: u64) -> u64 {
            // On a RISC-V SiFive U74 CPU, the use of multiplications appears
            // to be about 1.5x faster than doing the expansion by moving
            // the bits.
            let x0 = x & 0x1111111111111111;
            let x1 = x & 0x2222222222222222;
            let x2 = x & 0x4444444444444444;
            let x3 = x & 0x8888888888888888;

            #[inline(always)]
            fn sq_lo(x: u64) -> u64 {
                x.wrapping_mul(x)
            }

            let y0 = (sq_lo(x0) ^ sq_lo(x2)) & 0x1111111111111111;
            let y1 = (sq_lo(x1) ^ sq_lo(x3)) & 0x4444444444444444;

            y0 ^ y1
        }
        let (a0, a1) = (self.0[0], self.0[1]);

        // Square the polynomial.
        // We can omit the masking of the high halves of the values since
        // `expand_32()` uses wrapping multiplications.
        let c0 = expand_32(a0);
        let c1 = expand_32(a0 >> 32);
        let c2 = expand_32(a1);
        let c3 = expand_32(a1 >> 32);

        // Reduce.
        // Note that in a squared polynomial, all odd-indexed bits are zero;
        // we can thus skip one bit propagation operations.
        let e = c2 ^ c3;
        let d0 = c0 ^ (e << 1);
        let d1 = c1 ^ (c3 << 1) ^ e;

        self.0[0] = d0;
        self.0[1] = d1;
    }

    /* unused -- alternate implementation of set_square(), using
       multiplications that return the high output word to avoid some
       shifts. In practice it makes the overall EC code slower when
       used in context, even though it has lower latency.

    #[inline(always)]
    pub fn set_square(&mut self) {
        #[inline(always)]
        fn sq64(x: u64) -> (u64, u64) {
            #[inline(always)]
            fn sq_lo(x: u64) -> u64 {
                x.wrapping_mul(x)
            }

            #[inline(always)]
            fn sq_hi(x: u64) -> u64 {
                (((x as u128) * (x as u128)) >> 64) as u64
            }

            let x0 = x & 0x1111111111111111;
            let x1 = x & 0x2222222222222222;
            let x2 = x & 0x4444444444444444;
            let x3 = x & 0x8888888888888888;
            let lo = ((sq_lo(x0) ^ sq_lo(x2)) & 0x1111111111111111)
                   ^ ((sq_lo(x1) ^ sq_lo(x3)) & 0x4444444444444444);

            let y0 = x0 & 0xFFFFFFFF00000000;
            let y1 = x1 & 0xFFFFFFFF00000000;
            let y2 = x2 & 0xFFFFFFFF00000000;
            let y3 = x3 & 0xFFFFFFFF00000000;
            let hi = ((sq_hi(y0) ^ sq_hi(y2)) & 0x1111111111111111)
                   ^ ((sq_hi(y1) ^ sq_hi(y3)) & 0x4444444444444444);

            (lo, hi)
        }

        let (a0, a1) = (self.0[0], self.0[1]);

        // Square the polynomial.
        let (c0, c1) = sq64(a0);
        let (c2, c3) = sq64(a1);

        // Reduce.
        // Note that in a squared polynomial, all odd-indexed bits are zero;
        // we can thus skip one bit propagation operations.
        let e = c2 ^ c3;
        let d0 = c0 ^ (e << 1);
        let d1 = c1 ^ (c3 << 1) ^ e;

        self.0[0] = d0;
        self.0[1] = d1;
    }
    */

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
        let h = self.0[1] & 0x8000000000000000;
        self.0[0] ^= h ^ (h >> 63);
        self.0[1] ^= h;
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
    fn frob(self, tab: &[GFb127; 128]) -> Self {
        let mut d = Self::ZERO;
        let mut a = self.0[1];
        for i in (64..128).rev() {
            let m = sgnw(a);
            a <<= 1;
            d.0[0] ^= tab[i].0[0] & m;
            d.0[1] ^= tab[i].0[1] & m;
        }
        let mut a = self.0[0];
        for i in (0..64).rev() {
            let m = sgnw(a);
            a <<= 1;
            d.0[0] ^= tab[i].0[0] & m;
            d.0[1] ^= tab[i].0[1] & m;
        }
        d
    }

    // z^(i*2^14) for i = 0 to 127.
    const FROB14: [Self; 128] = [
        GFb127::w64le(0x0000000000000001, 0x0000000000000000),
        GFb127::w64le(0x0000000100010114, 0x0000000000000001),
        GFb127::w64le(0x0000000100010112, 0x0000000000000000),
        GFb127::w64le(0x0000000700070768, 0x0000000100010113),
        GFb127::w64le(0x0000000100010104, 0x0000000000000001),
        GFb127::w64le(0x0000001100111052, 0x0000000000000010),
        GFb127::w64le(0x0000001700171648, 0x0000000100010113),
        GFb127::w64le(0x0000007100717784, 0x0000001000101131),
        GFb127::w64le(0x0000000100010012, 0x0000000000000000),
        GFb127::w64le(0x0000010701061368, 0x0000000100010013),
        GFb127::w64le(0x0000010101001304, 0x0000000000000001),
        GFb127::w64le(0x0000071107167852, 0x0000010001011310),
        GFb127::w64le(0x0000011701161248, 0x0000000100010013),
        GFb127::w64le(0x0000117111612584, 0x0000001000100131),
        GFb127::w64le(0x0000170117174812, 0x0000010001011300),
        GFb127::w64le(0x0000700770719768, 0x0000100110103113),
        GFb127::w64le(0x0000000100000104, 0x0000000000000001),
        GFb127::w64le(0x0001001001051052, 0x0000000000010010),
        GFb127::w64le(0x0001001601051648, 0x0000000100010113),
        GFb127::w64le(0x0007007607197784, 0x0001001101031131),
        GFb127::w64le(0x0001000001050012, 0x0000000000010000),
        GFb127::w64le(0x0011011611541368, 0x0000000100110013),
        GFb127::w64le(0x0017011617481304, 0x0001000101130001),
        GFb127::w64le(0x0071076070927852, 0x0010011010301310),
        GFb127::w64le(0x0001011601041248, 0x0000000100010013),
        GFb127::w64le(0x0107107702092584, 0x0001001100030131),
        GFb127::w64le(0x0101160104134812, 0x0000010001001300),
        GFb127::w64le(0x0711771108239768, 0x0100110003003113),
        GFb127::w64le(0x0117011712480104, 0x0001000100130001),
        GFb127::w64le(0x1170117124811052, 0x0010001001300010),
        GFb127::w64le(0x1700170149171648, 0x0100010013010113),
        GFb127::w64le(0x7000700790717784, 0x1000100130101131),
        GFb127::w64le(0x0000000000010012, 0x0000000000000000),
        GFb127::w64le(0x0001001301061368, 0x0000000000010012),
        GFb127::w64le(0x0001001301001304, 0x0000000000000000),
        GFb127::w64le(0x0007007907167850, 0x0001001301011316),
        GFb127::w64le(0x0001001301161248, 0x0000000000010012),
        GFb127::w64le(0x0011012311612584, 0x0000000000100120),
        GFb127::w64le(0x0017014917174810, 0x0001001301011316),
        GFb127::w64le(0x0071078370719748, 0x0010013010103172),
        GFb127::w64le(0x0001001300000104, 0x0000000000000000),
        GFb127::w64le(0x0107137801051050, 0x0001001300010116),
        GFb127::w64le(0x0101131201051648, 0x0000000000010012),
        GFb127::w64le(0x0711782407197584, 0x0100130101031720),
        GFb127::w64le(0x0117124801050010, 0x0001001300010116),
        GFb127::w64le(0x1170249211541348, 0x0010013000111172),
        GFb127::w64le(0x1700490417481104, 0x0100130101131600),
        GFb127::w64le(0x7000900870925850, 0x1000300310307316),
        GFb127::w64le(0x0001001201041248, 0x0000000000010012),
        GFb127::w64le(0x0002002502092584, 0x0000000100020121),
        GFb127::w64le(0x0004004904134810, 0x0001001301011317),
        GFb127::w64le(0x000800950821974A, 0x0003003103063174),
        GFb127::w64le(0x0012010512480104, 0x0000000100120001),
        GFb127::w64le(0x0024021924811050, 0x0001000301210107),
        GFb127::w64le(0x004804054915164A, 0x0013010113170004),
        GFb127::w64le(0x00920855905175A4, 0x0030031130711741),
        GFb127::w64le(0x0104124800010010, 0x0001001300010117),
        GFb127::w64le(0x020825970104134A, 0x0003013101071074),
        GFb127::w64le(0x0412480101001104, 0x0100130001011701),
        GFb127::w64le(0x0824971105165A50, 0x0301310007107507),
        GFb127::w64le(0x124901170114124A, 0x0013000101170104),
        GFb127::w64le(0x24901171114125A4, 0x0130001011701041),
        GFb127::w64le(0x4900170115174A10, 0x1300010017010517),
        GFb127::w64le(0x900070075071B74A, 0x3000100170105174),
        GFb127::w64le(0x0000000100000104, 0x0000000000000000),
        GFb127::w64le(0x0001001001051050, 0x0000000100000105),
        GFb127::w64le(0x0001001601051648, 0x0000000000000001),
        GFb127::w64le(0x00070074071B75A2, 0x000100170105174A),
        GFb127::w64le(0x0001000001050010, 0x0000000100000105),
        GFb127::w64le(0x0011011611541348, 0x0000001000001051),
        GFb127::w64le(0x00170114174A1122, 0x000100170105175A),
        GFb127::w64le(0x0071074070B25A30, 0x00100171105175A5),
        GFb127::w64le(0x0001011601041248, 0x0000000000000001),
        GFb127::w64le(0x01071075020B25A2, 0x000101170104124A),
        GFb127::w64le(0x0101160104134810, 0x0000000100000005),
        GFb127::w64le(0x071175110A21B148, 0x0100171105175A51),
        GFb127::w64le(0x01170115124A0122, 0x000101170104125A),
        GFb127::w64le(0x1170115124A11230, 0x00101171104124A5),
        GFb127::w64le(0x170015014B153048, 0x0100170105175A01),
        GFb127::w64le(0x70005005B05115A2, 0x100070075071B74A),
        GFb127::w64le(0x0000000000010010, 0x0000000100000105),
        GFb127::w64le(0x0001001301041348, 0x0001001001051050),
        GFb127::w64le(0x0001001101021122, 0x000100170104175B),
        GFb127::w64le(0x0005005B05105A32, 0x00070074071B75A3),
        GFb127::w64le(0x0001001301141248, 0x0001000001050000),
        GFb127::w64le(0x00110121114325A2, 0x001101171155125B),
        GFb127::w64le(0x0015014B15314812, 0x00170104175A0013),
        GFb127::w64le(0x005105A35011B168, 0x0071074070B25A30),
        GFb127::w64le(0x0001001100020122, 0x000101170105125B),
        GFb127::w64le(0x0105135A01031232, 0x01071075020B25A3),
        GFb127::w64le(0x0101111203053048, 0x0101170105125B00),
        GFb127::w64le(0x05115A24011917A2, 0x071175100A20B15B),
        GFb127::w64le(0x0115124A01230012, 0x01170105125A0013),
        GFb127::w64le(0x115024B213341368, 0x1170105125A00130),
        GFb127::w64le(0x15004B04314A1322, 0x170005005B05015B),
        GFb127::w64le(0x5000B00A10B27A32, 0x70005005B05115A3),
        GFb127::w64le(0x0001001201041248, 0x0000000000000000),
        GFb127::w64le(0x00020025020B25A0, 0x000100120105125B),
        GFb127::w64le(0x0004004904134810, 0x0000000000010013),
        GFb127::w64le(0x000A00B30A23B166, 0x0005005B05115A32),
        GFb127::w64le(0x00120105124A0120, 0x000100120105125B),
        GFb127::w64le(0x0024021924A11210, 0x00100120105025A3),
        GFb127::w64le(0x004A04234B173066, 0x0005005B05015B02),
        GFb127::w64le(0x00B20A35B0711740, 0x005105A25010B17B),
        GFb127::w64le(0x0104124800010010, 0x0000000000010013),
        GFb127::w64le(0x020A25B101061166, 0x0105125A00030132),
        GFb127::w64le(0x0412480101021120, 0x000100120005015B),
        GFb127::w64le(0x0A24B11307107410, 0x05105A25010A17A3),
        GFb127::w64le(0x124B013101161066, 0x0105125A00130002),
        GFb127::w64le(0x24B0131111630740, 0x105025B20035127B),
        GFb127::w64le(0x4B00310317316610, 0x05005B05015A0213),
        GFb127::w64le(0xB000100170115166, 0x5000B00A10B27A32),
        GFb127::w64le(0x0000000100020120, 0x000100120105125A),
        GFb127::w64le(0x0001001201011212, 0x00020025020B25A0),
        GFb127::w64le(0x0003003003073066, 0x0005005A05125B06),
        GFb127::w64le(0x000100160117174A, 0x000A00B30A22B174),
        GFb127::w64le(0x0001000201210012, 0x00120105125A0000),
        GFb127::w64le(0x0013011013161146, 0x0025020A25A00106),
        GFb127::w64le(0x003103163164112A, 0x005A05135B070114),
        GFb127::w64le(0x00110162105074B2, 0x00B20A35B0711740),
        GFb127::w64le(0x0003013001061066, 0x0105125B00000106),
        GFb127::w64le(0x010112170005054A, 0x020A25B101071174),
        GFb127::w64le(0x0301300106116612, 0x05125B0000010600),
        GFb127::w64le(0x0113171104015B46, 0x0A25B10007117506),
        GFb127::w64le(0x013101171064032A, 0x125B000101070114),
        GFb127::w64le(0x13101171064132B2, 0x25B0001010701140),
        GFb127::w64le(0x3100170165173A66, 0x5B00010007011506),
        GFb127::w64le(0x900070075071B74B, 0x3000100170105174),
    ];

    // z^(i*2^42) for i = 0 to 127.
    const FROB42: [Self; 128] = [
        GFb127::w64le(0x0000000000000001, 0x0000000000000000),
        GFb127::w64le(0x0000000100000110, 0x0000000000000000),
        GFb127::w64le(0x0000000000010100, 0x0000000000000001),
        GFb127::w64le(0x0001010001111000, 0x0000000100000110),
        GFb127::w64le(0x0000000100010002, 0x0000000000000001),
        GFb127::w64le(0x0001011201100220, 0x0000000100000111),
        GFb127::w64le(0x0001010101020202, 0x0000000100000103),
        GFb127::w64le(0x0013131312222222, 0x0000001300001230),
        GFb127::w64le(0x0000000100000006, 0x0000000000000000),
        GFb127::w64le(0x0000011600000660, 0x0000000000000001),
        GFb127::w64le(0x0001010000060600, 0x0000000100000006),
        GFb127::w64le(0x0117160006666002, 0x0000011600010761),
        GFb127::w64le(0x000100040006000C, 0x0000000100000007),
        GFb127::w64le(0x0116044C06600CC2, 0x0000011700010775),
        GFb127::w64le(0x01040404060C0C0E, 0x000001050001070A),
        GFb127::w64le(0x124848486CCCCCEA, 0x0000125A00137FA0),
        GFb127::w64le(0x0000000000000014, 0x0000000000000001),
        GFb127::w64le(0x0000001400001540, 0x0000000100000110),
        GFb127::w64le(0x0000000000141402, 0x0000000000010115),
        GFb127::w64le(0x0014140215554220, 0x0001011501110450),
        GFb127::w64le(0x000000140014002A, 0x0000000100010017),
        GFb127::w64le(0x0014156A154028A2, 0x0001010701101665),
        GFb127::w64le(0x0014141614282A2E, 0x000101140102173D),
        GFb127::w64le(0x017D7D5B6AAA8EC8, 0x0013127C12235BD2),
        GFb127::w64le(0x0000001400000078, 0x0000000100000006),
        GFb127::w64le(0x0000153800007F82, 0x0000011600000675),
        GFb127::w64le(0x001414020078780C, 0x000101150006067E),
        GFb127::w64le(0x152D3A2C7FFD8EEA, 0x0117022E06730CF7),
        GFb127::w64le(0x00140052007800FE, 0x0001001100060067),
        GFb127::w64le(0x153857DE7F82F1C2, 0x0116107706756133),
        GFb127::w64le(0x1450525A78F2FECC, 0x010411450619678C),
        GFb127::w64le(0x6DA581137FD90248, 0x124936DA6DA5B7CB),
        GFb127::w64le(0x0000000000000112, 0x0000000000000001),
        GFb127::w64le(0x0000011200010320, 0x0000000100000110),
        GFb127::w64le(0x0000000001131202, 0x0000000000010013),
        GFb127::w64le(0x0113120302232220, 0x0001001301101230),
        GFb127::w64le(0x0000011201120226, 0x0000000100010111),
        GFb127::w64le(0x0113010703220462, 0x0001000101110103),
        GFb127::w64le(0x0113131110262422, 0x0001001201031237),
        GFb127::w64le(0x1204042324446004, 0x0013011612310772),
        GFb127::w64le(0x000001120000066C, 0x0000000100000006),
        GFb127::w64le(0x0001054C00060AC2, 0x0000011600000773),
        GFb127::w64le(0x01131202066A6C0C, 0x000100130006006A),
        GFb127::w64le(0x04494E2A0CC8CCE6, 0x0116125A07727EB0),
        GFb127::w64le(0x0112044A066C0CD6, 0x0001011700060775),
        GFb127::w64le(0x054802700ACE194E, 0x011701050774070C),
        GFb127::w64le(0x164C4E4460D6D8E8, 0x0105125B07187FB1),
        GFb127::w64le(0x485C78CED9BF4234, 0x125B01067EB11619),
        GFb127::w64le(0x000000000000156A, 0x0000000000000107),
        GFb127::w64le(0x0000156A00143CA0, 0x0000010700011770),
        GFb127::w64le(0x00000000157F680E, 0x000000000106136D),
        GFb127::w64le(0x157F681A289E8EE0, 0x0106136C16725BD0),
        GFb127::w64le(0x0000156A156A28DA, 0x0000010701071663),
        GFb127::w64le(0x157E146E3C8A55AE, 0x010601121767115D),
        GFb127::w64le(0x157F7D7140FCF6C6, 0x0106126B05185DB9),
        GFb127::w64le(0x68765092F1358EB4, 0x136B100C5DA10D9F),
        GFb127::w64le(0x0000156A00007F7C, 0x0000010700000612),
        GFb127::w64le(0x001443DC007889CE, 0x000111620006674D),
        GFb127::w64le(0x157F680E7F037024, 0x0106136D06146B6E),
        GFb127::w64le(0x579DFEBCF14B0098, 0x106630B86154A396),
        GFb127::w64le(0x156A57A67F7CF0D2, 0x0107107106126127),
        GFb127::w64le(0x438E2CCA8931FFC0, 0x11731731672A72B2),
        GFb127::w64le(0x3FFFF9E180061242, 0x030C30C30A28A28C),
        GFb127::w64le(0x80006DDA006B07A0, 0x36DB6DB6B6DB6DDD),
        GFb127::w64le(0x0000000000010106, 0x0000000000000001),
        GFb127::w64le(0x0001010601111660, 0x0000000100000110),
        GFb127::w64le(0x0000000100070602, 0x0000000000000007),
        GFb127::w64le(0x0007071207766220, 0x0000000700000771),
        GFb127::w64le(0x000101070104020E, 0x0000000100000105),
        GFb127::w64le(0x0015157F14422EE2, 0x0000001500001456),
        GFb127::w64le(0x00060607060C0C0A, 0x000000060000060A),
        GFb127::w64le(0x006A6B7C6CCCCAAC, 0x0000006A00006CA1),
        GFb127::w64le(0x0001010600060614, 0x0000000100000006),
        GFb127::w64le(0x0117107406667542, 0x0000011600010767),
        GFb127::w64le(0x000706040012140C, 0x0000000700000013),
        GFb127::w64le(0x0764704C13354CCE, 0x0000076300071433),
        GFb127::w64le(0x0102041C06180C26, 0x0000010300010718),
        GFb127::w64le(0x143C51E0798CE666, 0x0000142800156C9E),
        GFb127::w64le(0x0618181814282830, 0x0000061E0006123D),
        GFb127::w64le(0x6DB1B1A56AAABF3C, 0x00006DDD006B00D0),
        GFb127::w64le(0x000000000014147A, 0x0000000000010113),
        GFb127::w64le(0x0014147A15553DA0, 0x0001011301110230),
        GFb127::w64le(0x00000014006C7826, 0x0000000100070669),
        GFb127::w64le(0x006C6D666ABFA462, 0x0007077907760F85),
        GFb127::w64le(0x0014146E14502AD2, 0x000101120104174F),
        GFb127::w64le(0x01050227152A7D04, 0x0015146E14432E8C),
        GFb127::w64le(0x0078786078F0FC9C, 0x00060679060C7288),
        GFb127::w64le(0x070F1AE37FFF5932, 0x006A6C1E6CCBDE99),
        GFb127::w64le(0x0014147A0078791C, 0x000101130006066A),
        GFb127::w64le(0x152D44BC7FFC8FE6, 0x0117045A067319C9),
        GFb127::w64le(0x006C785E016910D6, 0x0007066F00121563),
        GFb127::w64le(0x6BD6CB377F8FD7BE, 0x07641C93135F4B01),
        GFb127::w64le(0x142853B679E2FCC8, 0x01021123060D66DE),
        GFb127::w64le(0x133471D67ED726C4, 0x143D57E8789AF161),
        GFb127::w64le(0x79E1EDDD122E07BA, 0x0618679E14575129),
        GFb127::w64le(0x6DDD077900D70E90, 0x6DB6B6DC6DDDB1AB),
        GFb127::w64le(0x000000000113146E, 0x0000000000010015),
        GFb127::w64le(0x0113146F022528E0, 0x0001001501101450),
        GFb127::w64le(0x0000011207786E2A, 0x000000010007017B),
        GFb127::w64le(0x07796D0D0FE8C8A2, 0x0007006B07716DA3),
        GFb127::w64le(0x0113157D164A28F6, 0x0001001401051451),
        GFb127::w64le(0x146E02312E887948, 0x0015011014570178),
        GFb127::w64le(0x066A6B7460D4DEA0, 0x0006006D060A6CB4),
        GFb127::w64le(0x6C191D86D99F4ADA, 0x006A07626CA6145F),
        GFb127::w64le(0x0113146E066A7964, 0x000100150006007E),
        GFb127::w64le(0x044F51820CDCF26A, 0x0116142E07726D9A),
        GFb127::w64le(0x07786846131164FE, 0x0007017D00120609),
        GFb127::w64le(0x1CFFA68C207CB31A, 0x07636CD9145900AC),
        GFb127::w64le(0x102057F875BEF21C, 0x01031429070C6C8F),
        GFb127::w64le(0x57EC75EEE71B1590, 0x142907186D890431),
        GFb127::w64le(0x75A9A59942F6C71A, 0x061E6DDA125100A1),
        GFb127::w64le(0xB1C907CCD597B018, 0x6DDA071207A76327),
        GFb127::w64le(0x00000000157F1772, 0x000000000106157F),
        GFb127::w64le(0x157F176628E60520, 0x0106157E167428F0),
        GFb127::w64le(0x0000156A6A6958FE, 0x0000010707137D0D),
        GFb127::w64le(0x6A7D6432CFCD73EE, 0x07126A7A624AC9BD),
        GFb127::w64le(0x157F020D3F80041A, 0x01061479030A28F3),
        GFb127::w64le(0x177229F67A087350, 0x157F16602EF36A51),
        GFb127::w64le(0x7F031A4D820A49E8, 0x06146C7D1E51CB84),
        GFb127::w64le(0x7121A0B226C5AE76, 0x6B7B714BCDC04A0E),
        GFb127::w64le(0x157F17727F02732C, 0x0106157F06147F02),
        GFb127::w64le(0x57E47674F058343C, 0x106057F46141F038),
        GFb127::w64le(0x6A6927837D77D00A, 0x07137B1F126B1A43),
        GFb127::w64le(0xB2C22B40AE8BFC90, 0x7027B4A020D1BBC7),
        GFb127::w64le(0x40820834830C30AE, 0x051E51E51E45E45E),
        GFb127::w64le(0x0924876536CF0720, 0x51F11F11E4264270),
        GFb127::w64le(0x8000144500156C8A, 0x0A28A28A3CF3CF29),
        GFb127::w64le(0x80006DDA006B07A1, 0x36DB6DB6B6DB6DDD),
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

        let (a0, a1) = (self.0[0], self.0[1]);

        // sqrt(ae) = c0:c1 (32 bits each)
        let c0 = Self::squeeze(a0);
        let c1 = Self::squeeze(a1);

        // sqrt(ao) = d0:d1 (32 bits each)
        let d0 = Self::squeeze(a0 >> 1);
        let d1 = Self::squeeze(a1 >> 1);

        // sqrt(a) = (c0 + c1*z^32) + (d0 + d1*z^32)*(z^32 + z^64)
        //         = c0 + (c1 + d0)*z^32 + (d0 + d1)*z^64 + d1*z^96
        self.0[0] = c0 | ((c1 ^ d0) << 32);
        self.0[1] = (d0 ^ d1) | (d1 << 32);
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
        ((self.0[0] ^ (self.0[1] >> 63)) as u32) & 1
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

        // We accumulate the extra values (square roots) into e.
        let mut x = Self::squeeze(self.0[0]) | (Self::squeeze(self.0[1]) << 32);
        let mut e = x;

        // At this point, we have:
        //    H(a) = H(x) + H(z*a0) + e
        // and x has length 64 bits. We apply the even/odd split
        // repeatedly until x is a 1-bit value, thus equal to its halftrace.
        for _ in 0..6 {
            ao.0[0] ^= x;
            x = Self::squeeze(x);
            e ^= x;
        }

        // We now get the halftrace of the odd-indexed bits in ao.
        let (mut d0, mut d1) = (e ^ x, 0);
        for i in 0..2 {
            let mut mw = ao.0[i];
            for j in (0..32).rev() {
                let m = sgnw(mw);
                mw <<= 2;
                d0 ^= m & Self::HALFTRACE[(i << 5) + j].0[0];
                d1 ^= m & Self::HALFTRACE[(i << 5) + j].0[1];
            }
        }

        self.0[0] = d0;
        self.0[1] = d1;
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
        let a0 = self.0[0];
        let a1 = self.0[1];

        // Normalize the value.
        let h = a1 & 0x8000000000000000;
        let a0 = a0 ^ h ^ (h >> 63);
        let a1 = a1 ^ h;

        // Check that we got a full zero.
        let t = a0 | a1;
        (((t | t.wrapping_neg()) >> 63) as u32).wrapping_sub(1)
    }

    #[inline(always)]
    pub fn encode(self) -> [u8; 16] {
        let mut r = self;
        r.set_normalized();
        let mut d = [0u8; 16];
        d[..8].copy_from_slice(&r.0[0].to_le_bytes());
        d[8..].copy_from_slice(&r.0[1].to_le_bytes());
        d
    }

    // Decode the value from bytes with implicit reduction modulo
    // z^127 + z^63 + 1. Input MUST be of length 16 bytes exactly.
    #[inline]
    fn set_decode16_reduce(&mut self, buf: &[u8]) {
        debug_assert!(buf.len() == 16);
        self.0[0] = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[..8]).unwrap());
        self.0[1] = u64::from_le_bytes(*<&[u8; 8]>::try_from(&buf[8..]).unwrap());
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
        let m = !sgnw(self.0[1]);
        self.0[0] &= m;
        self.0[1] &= m;
        m as u32
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

    // Multiply this value by sb = 1 + z^54 (an element of GF(2^127)).
    #[inline(always)]
    pub fn mul_b(self) -> Self {
        Self([ self.0[0].mul_b(), self.0[1].mul_b() ])
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
        let sj = (j as i64) - 1;
        let mut nf = !0u64;
        for i in 0..16 {
            let t = ((sj - (i as i64)) >> 32) as u64;
            let m = t & nf;
            nf &= !t;
            r[0].0[0].0[0] |= m & tab[(i * 2) + 0].0[0].0[0];
            r[0].0[0].0[1] |= m & tab[(i * 2) + 0].0[0].0[1];
            r[0].0[1].0[0] |= m & tab[(i * 2) + 0].0[1].0[0];
            r[0].0[1].0[1] |= m & tab[(i * 2) + 0].0[1].0[1];
            r[1].0[0].0[0] |= m & tab[(i * 2) + 1].0[0].0[0];
            r[1].0[0].0[1] |= m & tab[(i * 2) + 1].0[0].0[1];
            r[1].0[1].0[0] |= m & tab[(i * 2) + 1].0[1].0[0];
            r[1].0[1].0[1] |= m & tab[(i * 2) + 1].0[1].0[1];
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
        let sj = (j as i64) - 1;
        let mut nf = !0u64;
        for i in 0..8 {
            let t = ((sj - (i as i64)) >> 32) as u64;
            let m = t & nf;
            nf &= !t;
            r[0].0[0].0[0] |= m & tab[(i * 2) + 0].0[0].0[0];
            r[0].0[0].0[1] |= m & tab[(i * 2) + 0].0[0].0[1];
            r[0].0[1].0[0] |= m & tab[(i * 2) + 0].0[1].0[0];
            r[0].0[1].0[1] |= m & tab[(i * 2) + 0].0[1].0[1];
            r[1].0[0].0[0] |= m & tab[(i * 2) + 1].0[0].0[0];
            r[1].0[0].0[1] |= m & tab[(i * 2) + 1].0[0].0[1];
            r[1].0[1].0[0] |= m & tab[(i * 2) + 1].0[1].0[0];
            r[1].0[1].0[1] |= m & tab[(i * 2) + 1].0[1].0[1];
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
        let sj = (j as i64) - 1;
        let mut nf = !0u64;
        for i in 0..4 {
            let t = ((sj - (i as i64)) >> 32) as u64;
            let m = t & nf;
            nf &= !t;
            r[0].0[0].0[0] |= m & tab[(i * 2) + 0].0[0].0[0];
            r[0].0[0].0[1] |= m & tab[(i * 2) + 0].0[0].0[1];
            r[0].0[1].0[0] |= m & tab[(i * 2) + 0].0[1].0[0];
            r[0].0[1].0[1] |= m & tab[(i * 2) + 0].0[1].0[1];
            r[1].0[0].0[0] |= m & tab[(i * 2) + 1].0[0].0[0];
            r[1].0[0].0[1] |= m & tab[(i * 2) + 1].0[0].0[1];
            r[1].0[1].0[0] |= m & tab[(i * 2) + 1].0[1].0[0];
            r[1].0[1].0[1] |= m & tab[(i * 2) + 1].0[1].0[1];
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
        #[inline(always)]
        fn select(x0: u64, x1: u64, x2: u64, x3: u64, m0: u64, m1: u64) -> u64 {
            let y0 = x0 ^ (m0 & (x0 ^ x1));
            let y1 = x2 ^ (m0 & (x2 ^ x3));
            y0 ^ (m1 & (y0 ^ y1))
        }

        let m0 = ((j & 1) as u64).wrapping_neg();
        let m1 = ((j >> 1) as u64).wrapping_neg();
        let mut r = [Self::ZERO; 2];
        r[0].0[0].0[0] = select(tab[0].0[0].0[0], tab[2].0[0].0[0],
                                tab[4].0[0].0[0], tab[6].0[0].0[0], m0, m1);
        r[0].0[0].0[1] = select(tab[0].0[0].0[1], tab[2].0[0].0[1],
                                tab[4].0[0].0[1], tab[6].0[0].0[1], m0, m1);
        r[0].0[1].0[0] = select(tab[0].0[1].0[0], tab[2].0[1].0[0],
                                tab[4].0[1].0[0], tab[6].0[1].0[0], m0, m1);
        r[0].0[1].0[1] = select(tab[0].0[1].0[1], tab[2].0[1].0[1],
                                tab[4].0[1].0[1], tab[6].0[1].0[1], m0, m1);
        r[1].0[0].0[0] = select(tab[1].0[0].0[0], tab[3].0[0].0[0],
                                tab[5].0[0].0[0], tab[7].0[0].0[0], m0, m1);
        r[1].0[0].0[1] = select(tab[1].0[0].0[1], tab[3].0[0].0[1],
                                tab[5].0[0].0[1], tab[7].0[0].0[1], m0, m1);
        r[1].0[1].0[0] = select(tab[1].0[1].0[0], tab[3].0[1].0[0],
                                tab[5].0[1].0[0], tab[7].0[1].0[0], m0, m1);
        r[1].0[1].0[1] = select(tab[1].0[1].0[1], tab[3].0[1].0[1],
                                tab[5].0[1].0[1], tab[7].0[1].0[1], m0, m1);
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
