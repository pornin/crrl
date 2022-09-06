//! Ristreto255 implementation.
//!
//! The Ristretto255 group is a prime order group currently specified in
//! [draft-irtf-cfrg-ristretto255-decaf448-03]. It is internally defined
//! over the curve Edwards25519, which is a twisted Edwards curve. Users
//! of Ristretto255 should not, in general, think about the underlying
//! curve points; the group has prime order and that is the abstraction
//! that is convenient for building cryptographic protocols.
//!
//! The `Point` structure represents a Ristretto255 point. Such points
//! can be encoded into 32 bytes, and decoded back; encoding is always
//! canonical, and this is enforced upon decoding. This implementation
//! strictly follows the draft; in particular, when decoding a point from
//! 32 bytes, it verifies that the top bit (most significant bit of the
//! last byte) is zero (this bit is always zero for a valid encoding, but
//! the code does not ignore it when decoding).
//!
//! The `Scalar` type is an alias for the `ed25519::Scalar` type, which
//! represents integers modulo the Ristretto255 order `L` (in the twisted
//! Edwards curve, this is the order of a specific subgroup of the
//! curve).
//!
//! [draft-irtf-cfrg-ristretto255-decaf448-03]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-ristretto255-decaf448

// Projective/fractional coordinates traditionally use uppercase letters,
// using lowercase only for affine coordinates.
#![allow(non_snake_case)]

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use super::field::GF25519;
use super::ed25519::{Point as Ed25519Point, Scalar as Ed25519Scalar};

/// A Ristretto255 point.
#[derive(Clone, Copy, Debug)]
pub struct Point(Ed25519Point);

/// A Ristretto255 scalar (integer modulo the group prime order `L`).
pub type Scalar = Ed25519Scalar;

impl Point {

    /// The neutral element (identity point) in the group.
    pub const NEUTRAL: Self = Self(Ed25519Point::NEUTRAL);

    /// The conventional base point in the group.
    pub const BASE: Self = Self(Ed25519Point::BASE);

    /// The d constant from the twisted Edwards curve equation.
    const D: GF25519 = Ed25519Point::D;

    // Some constants defined in the draft.

    const SQRT_M1: GF25519 = Ed25519Point::SQRT_M1;

    const SQRT_AD_MINUS_ONE: GF25519 = GF25519::w64be(
        0x376931BF2B8348AC,
        0x0F3CFCC931F5D1FD,
        0xAF9D8E0C1B7854BD,
        0x7E97F6A0497B2E1B,
    );
    const INVSQRT_A_MINUS_D: GF25519 = GF25519::w64be(
        0x786C8905CFAFFCA2,
        0x16C27B91FE01D840,
        0x9D2F16175A4172BE,
        0x99C8FDAA805D40EA,
    );
    const ONE_MINUS_D_SQ: GF25519 = GF25519::w64be(
        0x029072A8B2B3E0D7,
        0x9994ABDDBE70DFE4,
        0x2C81A138CD5E350F,
        0xE27C09C1945FC176,
    );
    const D_MINUS_ONE_SQ: GF25519 = GF25519::w64be(
        0x5968B37AF66C2241,
        0x4CDCD32F529B4EEB,
        0xD29E4A2CB01E1999,
        0x31AD5AAA44ED4D20,
    );

    /// Tests whether a field element is "negative".
    ///
    /// A field element is considered "negative" if its least significant
    /// bit, when represented as an integer in the 0 to L-1 range, is 1.
    /// This is returned here as a `u32` value with the usual pattern
    /// (0xFFFFFFFF for negative, 0x00000000 for nonnegative).
    fn is_negative(x: GF25519) -> u32 {
        ((x.encode()[0] & 1) as u32).wrapping_neg()
    }

    /// Gets the "absolute value" of a field element.
    ///
    /// If x is negative (as per the `is_negative()` function), then
    /// this returns -x. Otherwise, this returns x.
    fn abs(x: GF25519) -> GF25519 {
        GF25519::select(&x, &-x, Self::is_negative(x))
    }

    /// Square root of a ratio.
    ///
    /// This returns (r, x) such that:
    ///
    ///  - If u and v are non-zero, and u/v is a square, then
    ///    r = 0xFFFFFFFF and x = sqrt(u/v).
    ///
    ///  - If u is zero, then r = 0xFFFFFFFF and x = 0 (regardless of
    ///    the value of v).
    ///
    ///  - If u is non-zero but v is zero, then r = 0x00000000 and x = 0.
    ///
    ///  - If u and v are non-zero, and u/v is not a square, then
    ///    r = 0x00000000 and x = sqrt(SQRT_M1*(u/v)).
    ///
    /// The sqrt() function returns the nonnegative square root of its
    /// operand (as per `is_negative()`).
    fn sqrt_ratio_m1(u: GF25519, v: GF25519) -> (u32, GF25519) {
        let v3 = v.square() * v;
        let v7 = v3.square() * v;
        let x = u * v7;

        // Raise x to the power (p-5)/8 = 2^252 - 3.
        let x2 = x.square() * x;
        let x4 = x2.xsquare(2) * x2;
        let x5 = x4.square() * x;
        let x10 = x5.xsquare(5) * x5;
        let x20 = x10.xsquare(10) * x10;
        let x25 = x20.xsquare(5) * x5;
        let x50 = x25.xsquare(25) * x25;
        let x100 = x50.xsquare(50) * x50;
        let x125 = x100.xsquare(25) * x25;
        let x250 = x125.xsquare(125) * x125;
        let x = x250.xsquare(2) * x;

        let r = (u * v3) * x;
        let c = v * r.square();
        let correct_sign_sqrt   = c.equals(u);
        let flipped_sign_sqrt   = c.equals(-u);
        let flipped_sign_sqrt_i = c.equals(-u * Self::SQRT_M1);

        let r_prime = r * Self::SQRT_M1;
        let r = GF25519::select(&r, &r_prime,
            flipped_sign_sqrt | flipped_sign_sqrt_i);
        let r = Self::abs(r);

        (correct_sign_sqrt | flipped_sign_sqrt, r)
    }

    /// Sets this element by decoding its binary representation.
    ///
    /// If the input does not have length exactly 32 bytes, or if the
    /// input has length 32 bytes but is not the valid, canonical encoding
    /// of a Ristretto255 point, then this function sets `self` to the
    /// neutral element and returns 0x00000000; otherwise, it sets `self`
    /// to the decoded element and returns 0xFFFFFFFF.
    pub fn set_decode(&mut self, buf: &[u8]) -> u32 {
        *self = Self::NEUTRAL;
        if buf.len() != 32 {
            return 0;
        }

        let (s, mut r) = GF25519::decode32(buf);
        r &= !Self::is_negative(s);

        let ss = s.square();
        let u1 = GF25519::ONE - ss;
        let u2 = GF25519::ONE + ss;
        let u2_sqr = u2.square();

        let v = -(Self::D * u1.square()) - u2_sqr;

        let (was_square, invsqrt) =
            Self::sqrt_ratio_m1(GF25519::ONE, v * u2_sqr);

        let den_x = invsqrt * u2;
        let den_y = invsqrt * den_x * v;

        let x = Self::abs((s * den_x).mul2());
        let y = u1 * den_y;
        let t = x * y;

        r &= was_square & !(Self::is_negative(t) | y.iszero());

        self.0.set_cond(&Ed25519Point { X: x, Y: y, Z: GF25519::ONE, T: t }, r);
        r
    }

    /// Decodes an element from its binary representation.
    ///
    /// If the input does not have length exactly 32 bytes, or if the
    /// input has length 32 bytes but is not the valid, canonical encoding
    /// of a Ristretto255 point, then this function returns `None`.
    /// Otherwise, it returns the decoded element.
    ///
    /// Since this function uses an option type, outsiders may detect
    /// through side-channels whether decoding succeeded or failed;
    /// however, the decoded value should not leak.
    pub fn decode(buf: &[u8]) -> Option<Self> {
        let mut r = Self::NEUTRAL;
        if r.set_decode(buf) != 0 {
            Some(r)
        } else {
            None
        }
    }

    /// Encodes this element into bytes.
    ///
    /// Encoding is always canonical.
    pub fn encode(self) -> [u8; 32] {
        let (x0, y0, z0, t0) = (&self.0.X, &self.0.Y, &self.0.Z, &self.0.T);

        let u1 = (z0 + y0) * (z0 - y0);
        let u2 = x0 * y0;

        let (_, invsqrt) = Self::sqrt_ratio_m1(GF25519::ONE, u1 * u2.square());

        let den1 = invsqrt * u1;
        let den2 = invsqrt * u2;
        let z_inv = den1 * den2 * t0;

        let ix0 = x0 * Self::SQRT_M1;
        let iy0 = y0 * Self::SQRT_M1;
        let enchanted_denominator = den1 * Self::INVSQRT_A_MINUS_D;

        let rotate = Self::is_negative(t0 * z_inv);

        let x = GF25519::select(&x0, &iy0, rotate);
        let y = GF25519::select(&y0, &ix0, rotate);
        let z = z0;
        let den_inv = GF25519::select(&den2, &enchanted_denominator, rotate);

        let y = GF25519::select(&y, &-y, Self::is_negative(x * z_inv));

        let s = Self::abs(den_inv * (z - y));

        s.encode()
    }

    /// Compares two points for equality.
    ///
    /// Returned value is 0xFFFFFFFF if the two points are equal,
    /// 0x00000000 otherwise.
    ///
    /// Note: this function is vastly faster than encoding the two elements
    /// and comparing the two encodings.
    #[inline]
    pub fn equals(self, rhs: Self) -> u32 {
        let (x1, y1) = (&self.0.X, &self.0.Y);
        let (x2, y2) = (&rhs.0.X, &rhs.0.Y);
        (x1 * y2).equals(y1 * x2) | (y1 * y2).equals(x1 * x2)
    }

    /// Tests whether this element is the neutral (identity point).
    ///
    /// Returned value is 0xFFFFFFFF for the neutral, 0x00000000 for
    /// all other elements.
    #[inline(always)]
    pub fn isneutral(self) -> u32 {
        // We use the equals() formula against the Ed25519 neutral (0,1)
        // (which is a valid representation of the Ristretto255 neutral).
        let (x1, y1) = (&self.0.X, &self.0.Y);
        x1.iszero() | y1.iszero()
    }

    /// Conditionally copies the provided element (`P`) into `self`.
    ///
    ///  - If `ctl` = 0xFFFFFFFF, then the value of `P` is copied into `self`.
    ///
    ///  - If `ctl` = 0x00000000, then the value of `self` is unchanged.
    ///
    /// Value `ctl` MUST be equal to either 0x00000000 or 0xFFFFFFFF.
    #[inline(always)]
    pub fn set_cond(&mut self, P: &Self, ctl: u32) {
        self.0.set_cond(&P.0, ctl);
    }

    /// Returns an element equal to `P0` (if `ctl` = 0x00000000) or to
    /// `P1` (if `ctl` = 0xFFFFFFFF).
    ///
    /// Value `ctl` MUST be equal to either 0x00000000 or 0xFFFFFFFF.
    #[inline(always)]
    pub fn select(P0: &Self, P1: &Self, ctl: u32) -> Self {
        let mut P = *P0;
        P.set_cond(P1, ctl);
        P
    }

    /// Conditionally negates this point.
    ///
    /// This point is negated if `ctl` = 0xFFFFFFFF, but kept unchanged if
    /// `ctl` = 0x00000000.
    ///
    /// Value `ctl` MUST be equal to either 0x00000000 or 0xFFFFFFFF.
    #[inline(always)]
    pub fn set_condneg(&mut self, ctl: u32) {
        self.0.set_condneg(ctl);
    }

    /// Ristretto255 map (bytes to point).
    ///
    /// This is the map described in section 4.3.4 of the draft under the
    /// name "MAP". Its output is not uniformly distributed; in general,
    /// it should not be used directly, but only through `one_way_map()`,
    /// which invokes it twice (on distinct inputs) and adds the two
    /// results.
    fn map(buf: &[u8]) -> Self {
        assert!(buf.len() == 32);
        let mut tmp = [0u8; 32];
        tmp[..].copy_from_slice(buf);
        tmp[31] &= 0x7F;
        let t = GF25519::decode_reduce(&tmp);

        let r = Self::SQRT_M1 * t.square();
        let u = (r + GF25519::ONE) * Self::ONE_MINUS_D_SQ;
        let v = (-GF25519::ONE - r * Self::D) * (r + Self::D);

        let (was_square, s) = Self::sqrt_ratio_m1(u, v);
        let s_prime = -Self::abs(s * t);
        let s = GF25519::select(&s_prime, &s, was_square);
        let c = GF25519::select(&r, &-GF25519::ONE, was_square);

        let N = c * (r - GF25519::ONE) * Self::D_MINUS_ONE_SQ - v;

        let w0 = (s * v).mul2();
        let w1 = N * Self::SQRT_AD_MINUS_ONE;
        let w2 = GF25519::ONE - s.square();
        let w3 = GF25519::ONE + s.square();

        Self(Ed25519Point { X: w0 * w3, Y: w2 * w1, Z: w1 * w3, T: w0 * w2 })
    }

    /// The one-way map of bytes to Ristretto255 elements.
    ///
    /// This is the map described in the draft, section 4.3.4. The input
    /// MUST have length exactly 64 bytes (a panic is triggered otherwise).
    /// If the input is itself a 64-byte output of a secure hash function
    /// (e.g. SHA-512) then this constitutes a hash function with output
    /// in Ristretto255 (output is then indistinguishable from random
    /// uniform selection).
    pub fn one_way_map(buf: &[u8]) -> Self {
        assert!(buf.len() == 64);
        let mut b1 = [0u8; 32];
        let mut b2 = [0u8; 32];
        b1[..].copy_from_slice(&buf[..32]);
        b2[..].copy_from_slice(&buf[32..]);
        Self::map(&b1) + Self::map(&b2)
    }

    /// Adds `rhs` to `self`.
    #[inline(always)]
    fn set_add(&mut self, rhs: &Self) {
        self.0 += &rhs.0;
    }

    /// Subtracts `rhs` from `self`.
    #[inline(always)]
    fn set_sub(&mut self, rhs: &Self) {
        self.0 -= &rhs.0;
    }

    /// Negates this element.
    #[inline(always)]
    pub fn set_neg(&mut self) {
        self.0.set_neg();
    }

    /// Multiplies this element by the provided integer.
    ///
    /// The function is constant-time with regard to the Ristretto255
    /// element, but NOT to the multiplier `n`, which is assumed to be
    /// public.
    #[inline(always)]
    fn set_mul_small(&mut self, n: u64) {
        self.0.set_mul_small(n);

    }

    /// Multiplies this element by a scalar.
    #[inline(always)]
    fn set_mul(&mut self, n: &Scalar) {
        self.0 *= n;
    }

    /// Sets this element to n times the conventional base (`Self::BASE`).
    #[inline(always)]
    pub fn set_mulgen(&mut self, n: &Scalar) {
        self.0.set_mulgen(n);
    }

    /// Returns the product of the conventional base (`Self::BASE`) by
    /// the provided scalar.
    #[inline(always)]
    pub fn mulgen(n: &Scalar) -> Self {
        Self(Ed25519Point::mulgen(n))
    }

    /// Doubles this element (in place).
    #[inline(always)]
    pub fn set_double(&mut self) {
        self.0.set_double();
    }

    /// Doubles this element.
    #[inline(always)]
    pub fn double(self) -> Self {
        Self(self.0.double())
    }

    /// Doubles this element n times (in place).
    #[inline(always)]
    pub fn set_xdouble(&mut self, n: u32) {
        self.0.set_xdouble(n);
    }

    /// Doubles this element n times.
    #[inline(always)]
    pub fn xdouble(self, n: u32) -> Self {
        Self(self.0.xdouble(n))
    }

    /// Given scalars `u` and `v`, returns `u*self + v*B` (with `B` being
    /// the conventional generator of the prime order subgroup).
    //
    // This can be used to support EdDSA-style signature verification, though
    // for that task `verify_helper_vartime()` is faster.
    ///
    /// THIS FUNCTION IS NOT CONSTANT-TIME; it shall be used only with
    /// public data.
    #[inline(always)]
    pub fn mul_add_mulgen_vartime(self, u: &Scalar, v: &Scalar) -> Self {
	    Self(self.0.mul_add_mulgen_vartime(u, v))
    }

    /// Check whether `s*B = R + k*A`, for the provided scalars `s`
    /// and `k`, provided points `A` (`self`) and `R`, and conventional
    /// generator `B`.
    ///
    /// Returned value is true on match, false otherwise. This function
    /// is meant to support EdDSA-style signature verification.
    ///
    /// THIS FUNCTION IS NOT CONSTANT-TIME; it shall be used only with
    /// public data.
    pub fn verify_helper_vartime(self,
        R: &Point, s: &Scalar, k: &Scalar) -> bool
    {
        self.0.verify_helper_vartime(&R.0, s, k)
    }
}

impl Add<Point> for Point {
    type Output = Point;

    #[inline(always)]
    fn add(self, other: Point) -> Point {
        let mut r = self;
        r.set_add(&other);
        r
    }
}

impl Add<&Point> for Point {
    type Output = Point;

    #[inline(always)]
    fn add(self, other: &Point) -> Point {
        let mut r = self;
        r.set_add(other);
        r
    }
}

impl Add<Point> for &Point {
    type Output = Point;

    #[inline(always)]
    fn add(self, other: Point) -> Point {
        let mut r = *self;
        r.set_add(&other);
        r
    }
}

impl Add<&Point> for &Point {
    type Output = Point;

    #[inline(always)]
    fn add(self, other: &Point) -> Point {
        let mut r = *self;
        r.set_add(other);
        r
    }
}

impl AddAssign<Point> for Point {
    #[inline(always)]
    fn add_assign(&mut self, other: Point) {
        self.set_add(&other);
    }
}

impl AddAssign<&Point> for Point {
    #[inline(always)]
    fn add_assign(&mut self, other: &Point) {
        self.set_add(other);
    }
}

impl Mul<Scalar> for Point {
    type Output = Point;

    #[inline(always)]
    fn mul(self, other: Scalar) -> Point {
        let mut r = self;
        r.set_mul(&other);
        r
    }
}

impl Mul<&Scalar> for Point {
    type Output = Point;

    #[inline(always)]
    fn mul(self, other: &Scalar) -> Point {
        let mut r = self;
        r.set_mul(other);
        r
    }
}

impl Mul<Scalar> for &Point {
    type Output = Point;

    #[inline(always)]
    fn mul(self, other: Scalar) -> Point {
        let mut r = *self;
        r.set_mul(&other);
        r
    }
}

impl Mul<&Scalar> for &Point {
    type Output = Point;

    #[inline(always)]
    fn mul(self, other: &Scalar) -> Point {
        let mut r = *self;
        r.set_mul(other);
        r
    }
}

impl MulAssign<Scalar> for Point {
    #[inline(always)]
    fn mul_assign(&mut self, other: Scalar) {
        self.set_mul(&other);
    }
}

impl MulAssign<&Scalar> for Point {
    #[inline(always)]
    fn mul_assign(&mut self, other: &Scalar) {
        self.set_mul(other);
    }
}

impl Mul<Point> for Scalar {
    type Output = Point;

    #[inline(always)]
    fn mul(self, other: Point) -> Point {
        let mut r = other;
        r.set_mul(&self);
        r
    }
}

impl Mul<&Point> for Scalar {
    type Output = Point;

    #[inline(always)]
    fn mul(self, other: &Point) -> Point {
        let mut r = *other;
        r.set_mul(&self);
        r
    }
}

impl Mul<Point> for &Scalar {
    type Output = Point;

    #[inline(always)]
    fn mul(self, other: Point) -> Point {
        let mut r = other;
        r.set_mul(self);
        r
    }
}

impl Mul<&Point> for &Scalar {
    type Output = Point;

    #[inline(always)]
    fn mul(self, other: &Point) -> Point {
        let mut r = *other;
        r.set_mul(self);
        r
    }
}

impl Mul<u64> for Point {
    type Output = Point;

    #[inline(always)]
    fn mul(self, other: u64) -> Point {
        let mut r = self;
        r.set_mul_small(other);
        r
    }
}

impl Mul<u64> for &Point {
    type Output = Point;

    #[inline(always)]
    fn mul(self, other: u64) -> Point {
        let mut r = *self;
        r.set_mul_small(other);
        r
    }
}

impl MulAssign<u64> for Point {
    #[inline(always)]
    fn mul_assign(&mut self, other: u64) {
        self.set_mul_small(other);
    }
}

impl Mul<Point> for u64 {
    type Output = Point;

    #[inline(always)]
    fn mul(self, other: Point) -> Point {
        let mut r = other;
        r.set_mul_small(self);
        r
    }
}

impl Mul<&Point> for u64 {
    type Output = Point;

    #[inline(always)]
    fn mul(self, other: &Point) -> Point {
        let mut r = *other;
        r.set_mul_small(self);
        r
    }
}

impl Neg for Point {
    type Output = Point;

    #[inline(always)]
    fn neg(self) -> Point {
        let mut r = self;
        r.set_neg();
        r
    }
}

impl Neg for &Point {
    type Output = Point;

    #[inline(always)]
    fn neg(self) -> Point {
        let mut r = *self;
        r.set_neg();
        r
    }
}

impl Sub<Point> for Point {
    type Output = Point;

    #[inline(always)]
    fn sub(self, other: Point) -> Point {
        let mut r = self;
        r.set_sub(&other);
        r
    }
}

impl Sub<&Point> for Point {
    type Output = Point;

    #[inline(always)]
    fn sub(self, other: &Point) -> Point {
        let mut r = self;
        r.set_sub(other);
        r
    }
}

impl Sub<Point> for &Point {
    type Output = Point;

    #[inline(always)]
    fn sub(self, other: Point) -> Point {
        let mut r = *self;
        r.set_sub(&other);
        r
    }
}

impl Sub<&Point> for &Point {
    type Output = Point;

    #[inline(always)]
    fn sub(self, other: &Point) -> Point {
        let mut r = *self;
        r.set_sub(other);
        r
    }
}

impl SubAssign<Point> for Point {
    #[inline(always)]
    fn sub_assign(&mut self, other: Point) {
        self.set_sub(&other);
    }
}

impl SubAssign<&Point> for Point {
    #[inline(always)]
    fn sub_assign(&mut self, other: &Point) {
        self.set_sub(other);
    }
}

// ========================================================================

#[cfg(test)]
mod tests {

    use super::{Point, Scalar};
    use sha2::{Sha256, Digest};

    /*
    use std::fmt;
    use crate::field::GF25519;

    fn print_gf(name: &str, x: GF25519) {
        print!("{} = 0x", name);
        let bb = x.encode();
        for i in (0..32).rev() {
            print!("{:02X}", bb[i]);
        }
        println!();
    }

    fn print(name: &str, P: Point) {
        println!("{}:", name);
        print_gf("  X", P.0.X);
        print_gf("  Y", P.0.Y);
        print_gf("  Z", P.0.Z);
        print_gf("  T", P.0.T);
    }
    */

    // Test vectors from draft-irtf-cfrg-ristretto255-decaf448-03,
    // section A.1.
    const VEC_MULGEN: [&str; 16] = [
        "0000000000000000000000000000000000000000000000000000000000000000",
        "e2f2ae0a6abc4e71a884a961c500515f58e30b6aa582dd8db6a65945e08d2d76",
        "6a493210f7499cd17fecb510ae0cea23a110e8d5b901f8acadd3095c73a3b919",
        "94741f5d5d52755ece4f23f044ee27d5d1ea1e2bd196b462166b16152a9d0259",
        "da80862773358b466ffadfe0b3293ab3d9fd53c5ea6c955358f568322daf6a57",
        "e882b131016b52c1d3337080187cf768423efccbb517bb495ab812c4160ff44e",
        "f64746d3c92b13050ed8d80236a7f0007c3b3f962f5ba793d19a601ebb1df403",
        "44f53520926ec81fbd5a387845beb7df85a96a24ece18738bdcfa6a7822a176d",
        "903293d8f2287ebe10e2374dc1a53e0bc887e592699f02d077d5263cdd55601c",
        "02622ace8f7303a31cafc63f8fc48fdc16e1c8c8d234b2f0d6685282a9076031",
        "20706fd788b2720a1ed2a5dad4952b01f413bcf0e7564de8cdc816689e2db95f",
        "bce83f8ba5dd2fa572864c24ba1810f9522bc6004afe95877ac73241cafdab42",
        "e4549ee16b9aa03099ca208c67adafcafa4c3f3e4e5303de6026e3ca8ff84460",
        "aa52e000df2e16f55fb1032fc33bc42742dad6bd5a8fc0be0167436c5948501f",
        "46376b80f409b29dc2b5f6f0c52591990896e5716f41477cd30085ab7f10301e",
        "e0c418f7c8d9c4cdd7395b93ea124f3ad99021bb681dfc3302a9d99a2e53e64e",
    ];

    #[test]
    fn mulgen() {
        let mut P = Point::NEUTRAL;
        for i in 0..16 {
            let buf = hex::decode(VEC_MULGEN[i]).unwrap();
            let Q = Point::decode(&buf[..]).unwrap();
            assert!(P.equals(Q) == 0xFFFFFFFF);
            assert!(Q.equals(P) == 0xFFFFFFFF);
            assert!(P.encode() == &buf[..]);
            assert!(Q.encode() == &buf[..]);
            let R = Point::mulgen(&Scalar::from_u32(i as u32));
            assert!(P.equals(R) == 0xFFFFFFFF);
            assert!(R.equals(P) == 0xFFFFFFFF);
            assert!(R.encode() == &buf[..]);
            P += Point::BASE;
        }
    }

    // Test vectors from draft-irtf-cfrg-ristretto255-decaf448-03,
    // section A.2.
    const VEC_INVALID: [&str; 29] = [
        // Non-canonical field encodings.
        "00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
        "f3ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
        "edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",

        // Negative field elements.
        "0100000000000000000000000000000000000000000000000000000000000000",
        "01ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
        "ed57ffd8c914fb201471d1c3d245ce3c746fcbe63a3679d51b6a516ebebe0e20",
        "c34c4e1826e5d403b78e246e88aa051c36ccf0aafebffe137d148a2bf9104562",
        "c940e5a4404157cfb1628b108db051a8d439e1a421394ec4ebccb9ec92a8ac78",
        "47cfc5497c53dc8e61c91d17fd626ffb1c49e2bca94eed052281b510b1117a24",
        "f1c6165d33367351b0da8f6e4511010c68174a03b6581212c71c0e1d026c3c72",
        "87260f7a2f12495118360f02c26a470f450dadf34a413d21042b43b9d93e1309",

        // Non-square x^2.
        "26948d35ca62e643e26a83177332e6b6afeb9d08e4268b650f1f5bbd8d81d371",
        "4eac077a713c57b4f4397629a4145982c661f48044dd3f96427d40b147d9742f",
        "de6a7b00deadc788eb6b6c8d20c0ae96c2f2019078fa604fee5b87d6e989ad7b",
        "bcab477be20861e01e4a0e295284146a510150d9817763caf1a6f4b422d67042",
        "2a292df7e32cababbd9de088d1d1abec9fc0440f637ed2fba145094dc14bea08",
        "f4a9e534fc0d216c44b218fa0c42d99635a0127ee2e53c712f70609649fdff22",
        "8268436f8c4126196cf64b3c7ddbda90746a378625f9813dd9b8457077256731",
        "2810e5cbc2cc4d4eece54f61c6f69758e289aa7ab440b3cbeaa21995c2f4232b",

        // Negative xy value.
        "3eb858e78f5a7254d8c9731174a94f76755fd3941c0ac93735c07ba14579630e",
        "a45fdc55c76448c049a1ab33f17023edfb2be3581e9c7aade8a6125215e04220",
        "d483fe813c6ba647ebbfd3ec41adca1c6130c2beeee9d9bf065c8d151c5f396e",
        "8a2e1d30050198c65a54483123960ccc38aef6848e1ec8f5f780e8523769ba32",
        "32888462f8b486c68ad7dd9610be5192bbeaf3b443951ac1a8118419d9fa097b",
        "227142501b9d4355ccba290404bde41575b037693cef1f438c47f8fbf35d1165",
        "5c37cc491da847cfeb9281d407efc41e15144c876e0170b499a96a22ed31e01e",
        "445425117cb8c90edcbc7c1cc0e74f747f2c1efa5630a967c64f287792a48a4b",

        // s = -1, which causes y = 0.
        "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
    ];

    #[test]
    fn invalid() {
        for s in VEC_INVALID.iter() {
            let buf = hex::decode(s).unwrap();
            assert!(Point::decode(&buf[..]).is_none());
        }
    }

    // Test vectors from draft-irtf-cfrg-ristretto255-decaf448-03,
    // section A.3.
    struct Ristretto255MapTestVector<'a> {
        I: &'a str,
        O: &'a str,
    }
    const VEC_MAP: [Ristretto255MapTestVector; 11] = [
        Ristretto255MapTestVector {
            I: "5d1be09e3d0c82fc538112490e35701979d99e06ca3e2b5b54bffe8b4dc772c14d98b696a1bbfb5ca32c436cc61c16563790306c79eaca7705668b47dffe5bb6",
            O: "3066f82a1a747d45120d1740f14358531a8f04bbffe6a819f86dfe50f44a0a46",
        },
        Ristretto255MapTestVector {
            I: "f116b34b8f17ceb56e8732a60d913dd10cce47a6d53bee9204be8b44f6678b270102a56902e2488c46120e9276cfe54638286b9e4b3cdb470b542d46c2068d38",
            O: "f26e5b6f7d362d2d2a94c5d0e7602cb4773c95a2e5c31a64f133189fa76ed61b",
        },
        Ristretto255MapTestVector {
            I: "8422e1bbdaab52938b81fd602effb6f89110e1e57208ad12d9ad767e2e25510c27140775f9337088b982d83d7fcf0b2fa1edffe51952cbe7365e95c86eaf325c",
            O: "006ccd2a9e6867e6a2c5cea83d3302cc9de128dd2a9a57dd8ee7b9d7ffe02826",
        },
        Ristretto255MapTestVector {
            I: "ac22415129b61427bf464e17baee8db65940c233b98afce8d17c57beeb7876c2150d15af1cb1fb824bbd14955f2b57d08d388aab431a391cfc33d5bafb5dbbaf",
            O: "f8f0c87cf237953c5890aec3998169005dae3eca1fbb04548c635953c817f92a",
        },
        Ristretto255MapTestVector {
            I: "165d697a1ef3d5cf3c38565beefcf88c0f282b8e7dbd28544c483432f1cec7675debea8ebb4e5fe7d6f6e5db15f15587ac4d4d4a1de7191e0c1ca6664abcc413",
            O: "ae81e7dedf20a497e10c304a765c1767a42d6e06029758d2d7e8ef7cc4c41179",
        },
        Ristretto255MapTestVector {
            I: "a836e6c9a9ca9f1e8d486273ad56a78c70cf18f0ce10abb1c7172ddd605d7fd2979854f47ae1ccf204a33102095b4200e5befc0465accc263175485f0e17ea5c",
            O: "e2705652ff9f5e44d3e841bf1c251cf7dddb77d140870d1ab2ed64f1a9ce8628",
        },
        Ristretto255MapTestVector {
            I: "2cdc11eaeb95daf01189417cdddbf95952993aa9cb9c640eb5058d09702c74622c9965a697a3b345ec24ee56335b556e677b30e6f90ac77d781064f866a3c982",
            O: "80bd07262511cdde4863f8a7434cef696750681cb9510eea557088f76d9e5065",
        },
        Ristretto255MapTestVector {
            I: "edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1200000000000000000000000000000000000000000000000000000000000000",
            O: "304282791023b73128d277bdcb5c7746ef2eac08dde9f2983379cb8e5ef0517f",
        },
        Ristretto255MapTestVector {
            I: "edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            O: "304282791023b73128d277bdcb5c7746ef2eac08dde9f2983379cb8e5ef0517f",
        },
        Ristretto255MapTestVector {
            I: "0000000000000000000000000000000000000000000000000000000000000080ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
            O: "304282791023b73128d277bdcb5c7746ef2eac08dde9f2983379cb8e5ef0517f",
        },
        Ristretto255MapTestVector {
            I: "00000000000000000000000000000000000000000000000000000000000000001200000000000000000000000000000000000000000000000000000000000080",
            O: "304282791023b73128d277bdcb5c7746ef2eac08dde9f2983379cb8e5ef0517f",
        },
    ];

    #[test]
    fn one_way_map() {
        for tv in VEC_MAP.iter() {
            let input = hex::decode(tv.I).unwrap();
            let output = hex::decode(tv.O).unwrap();
            assert!(Point::one_way_map(&input[..]).encode() == &output[..]);
        }
    }

    #[test]
    fn mul_add_mulgen_vartime() {
        let mut sh = Sha256::new();
        for i in 0..20 {
            // Build pseudorandom A, u and v
            sh.update(((3 * i + 0) as u64).to_le_bytes());
            let v1 = sh.finalize_reset();
            sh.update(((3 * i + 1) as u64).to_le_bytes());
            let v2 = sh.finalize_reset();
            sh.update(((3 * i + 2) as u64).to_le_bytes());
            let v3 = sh.finalize_reset();
            let A = Point::mulgen(&Scalar::decode_reduce(&v1));
            let u = Scalar::decode_reduce(&v2);
            let v = Scalar::decode_reduce(&v3);

            // Compute u*A + v*B in two different ways; check that they
            // match.
            let R1 = u * A + Point::mulgen(&v);
            let R2 = A.mul_add_mulgen_vartime(&u, &v);
            assert!(R1.equals(R2) == 0xFFFFFFFF);
        }
    }
}
