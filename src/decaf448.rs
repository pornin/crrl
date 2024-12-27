//! Decaf448 implementation.
//!
//! The decaf448 group is a prime order group currently specified in
//! [draft-irtf-cfrg-ristretto255-decaf448-07]. It is internally defined
//! over the curve edwards448, which is an Edwards curve. Users
//! of decaf448 should not, in general, think about the underlying
//! curve points; the group has prime order and that is the abstraction
//! that is convenient for building cryptographic protocols.
//!
//! The `Point` structure represents a decaf448 point. Such points
//! can be encoded into 56 bytes, and decoded back; encoding is always
//! canonical, and this is enforced upon decoding. This implementation
//! strictly follows the draft; in particular, when decoding a point from
//! 56 bytes, it verifies that the least significant bit of the first byte
//! is zero (this bit is always zero for a valid encoding, but the code
//! does not ignore it when decoding).
//!
//! The `Scalar` type is an alias for the `ed448::Scalar` type, which
//! represents integers modulo the decaf448 order `L` (in the Edwards
//! curve, this is the order of a specific subgroup of the curve).
//!
//! [draft-irtf-cfrg-ristretto255-decaf448-07]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-ristretto255-decaf448

// Projective/fractional coordinates traditionally use uppercase letters,
// using lowercase only for affine coordinates.
#![allow(non_snake_case)]

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use super::field::GF448;
use super::ed448::{Point as Ed448Point, Scalar as Ed448Scalar};

/// A decaf448 point.
#[derive(Clone, Copy, Debug)]
pub struct Point(Ed448Point);

/// A decaf448 scalar (integer modulo the group prime order `L`).
pub type Scalar = Ed448Scalar;

impl Point {

    /// The neutral element (identity point) in the group.
    pub const NEUTRAL: Self = Self(Ed448Point::NEUTRAL);

    /// The conventional base point in the group.
    pub const BASE: Self = Self(Ed448Point {
        // If B is the conventional generator of edwards448, then the
        // two possible representations of the decaf448 generator are
        // 2*B and 2*B+N (with N = (0,-1) being the order-2 point on
        // edwards448). { B, B+N } are not used for historical reasons.
        // The coordinates below are for 2*B+N, which are exactly the
        // coordinates one would get from decode() on the canonical
        // representation of the decaf448 generator.
        X: GF448::w64be([
            0x5555555555555555,
            0x5555555555555555,
            0x5555555555555555,
            0x55555555AAAAAAAA,
            0xAAAAAAAAAAAAAAAA,
            0xAAAAAAAAAAAAAAAA,
            0xAAAAAAAAAAAAAAAA,
        ]),
        Y: GF448::w64be([
            0x51FA169CB528FB72,
            0x4CA629DFAF793D4F,
            0xFC91285FCA77B228,
            0x481C928C75273B47,
            0xF29A9A7CC5D5CF67,
            0x44434D412E325F94,
            0x25150432156C7912,
        ]),
        Z: GF448::ONE,
    });
    // pub const BASE: Self = Self(Ed448Point::BASE);

    /// -d, for d = constant from the Edwards curve equation (d = -39081).
    const MINUS_D: u32 = Ed448Point::MINUS_D;

    // Some constants defined in the draft.

    // 1 - d
    const ONE_MINUS_D: u32 = Self::MINUS_D + 1;
    // 1 - 2*d
    const ONE_MINUS_TWO_D: u32 = 2 * Self::MINUS_D + 1;
    const ONE_MINUS_TWO_D_FULL: GF448 = GF448::w64be([
        0, 0, 0, 0, 0, 0, Self::ONE_MINUS_TWO_D as u64 ]);
    // -4*d
    const MINUS_FOUR_D: u32 = 4 * Self::MINUS_D;
    // sqrt(-d)
    const SQRT_MINUS_D: GF448 = GF448::w64be([
        0x22D962FBEB24F768,
        0x3BF68D722FA26AA0,
        0xA1F1A7B8A5B8D54B,
        0x64A2D780968C14BA,
        0x839A66F4FD6EDED2,
        0x60337BF6AA20CE52,
        0x9642EF0F45572736,
    ]);
    // 1/sqrt(-d)
    const INVSQRT_MINUS_D: GF448 = GF448::w64be([
        0x6EF40652E222C057,
        0x902BE35A0BCAC807,
        0x5A90950C3A5B27A7,
        0xD6BA56F128A6521A,
        0xBE707EE2C21FBA15,
        0xEFBB2479F19E94F3,
        0x53AFBB5EB878682C,
    ]);

    /// Tests whether a field element is "negative".
    ///
    /// A field element is considered "negative" if its least significant
    /// bit, when represented as an integer in the 0 to L-1 range, is 1.
    /// This is returned here as a `u32` value with the usual pattern
    /// (0xFFFFFFFF for negative, 0x00000000 for nonnegative).
    fn is_negative(x: GF448) -> u32 {
        ((x.encode()[0] & 1) as u32).wrapping_neg()
    }

    /// Gets the "absolute value" of a field element.
    ///
    /// If x is negative (as per the `is_negative()` function), then
    /// this returns -x. Otherwise, this returns x.
    fn abs(x: GF448) -> GF448 {
        GF448::select(&x, &-x, Self::is_negative(x))
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
    ///    r = 0x00000000 and x = sqrt(-u/v).
    ///
    /// The sqrt() function returns the nonnegative square root of its
    /// operand (as per `is_negative()`).
    fn sqrt_ratio_m1(u: GF448, v: GF448) -> (u32, GF448) {
        // Let s = u*(u*v)^((p-3)/4)
        // If u != 0, v != 0, and u/v is a sqquare, then:
        //     u*v = (u/v)*(v^2) is also a square
        //     (u*v)^((p-3)/4) = (u*v)^((p+1)/4) / (u*v)
        //                     = sqrt(u*v) / (u*v)
        //                     = sqrt(1/(u*v))
        //     s = u*sqrt(1/(u*v)) = sqrt(u/v)
        // If u != 0, v != 0, and u/v is not a square, then:
        //     (u*v)^((p-3)/4) = (u*v)^((p+1)/4) / (u*v)
        //                     = sqrt(-u*v) / (u*v)
        //                     = sqrt(-1/(u*v))
        //     s = u*sqrt(1/(u*v)) = sqrt(-u/v)
        // We check whether u/v was a square by verifying whether u == v*s^2.
        //
        // If u = 0, then s = 0 (regardless of v) and u == v*s^2, so we
        // get an output of (true, 0), as expected.
        //
        // If u != 0 but v = 0, then s = 0, but v*s^2 = 0 which is distinct
        // from u, hence we get (false, 0), again as expected.
        //
        // We always return the non-negative s. Zero is non-negative.

        // t = u*v
        let t = u * v;

        // s = u*(u*v)^((p-3)/4)
        // (p-3)/4 = 2^446 - 2^222 - 1
        let t2 = t.square() * t;
        let t3 = t2.square() * t;
        let t6 = t3.xsquare(3) * t3;
        let t12 = t6.xsquare(6) * t6;
        let t24 = t12.xsquare(12) * t12;
        let t48 = t24.xsquare(24) * t24;
        let t54 = t48.xsquare(6) * t6;
        let t108 = t54.xsquare(54) * t54;
        let t216 = t108.xsquare(108) * t108;
        let t222 = t216.xsquare(6) * t6;
        let t223 = t222.square() * t;
        let s = t223.xsquare(223) * t222 * u;

        // check whether u == v*s^2
        let r = (v * s.square()).equals(u);

        (r, Self::abs(s))
    }

    /// Sets this element by decoding its binary representation.
    ///
    /// If the input does not have length exactly 56 bytes, or if the
    /// input has length 56 bytes but is not the valid, canonical encoding
    /// of a decaf448 point, then this function sets `self` to the
    /// neutral element and returns 0x00000000; otherwise, it sets `self`
    /// to the decoded element and returns 0xFFFFFFFF.
    pub fn set_decode(&mut self, buf: &[u8]) -> u32 {
        // We follow the draft spec, section 5.3.1.
        *self = Self::NEUTRAL;

        // Input must have length 56 bytes. We return early if that is not
        // the case: we cannot hide the different length from attackers
        // observing timing-based side-channels.
        if buf.len() != 56 {
            return 0;
        }

        // Decode the bytes as value s. The encoding must be canonical,
        // and s must be non-negative.
        let (s, mut r) = GF448::decode_ct(buf);
        r &= !Self::is_negative(s);

        // Draft spec formulas:
        //   ss = s^2
        //   u1 = 1 + ss
        //   u2 = u1^2 - 4 * D * ss
        //   (was_square, invsqrt) = SQRT_RATIO_M1(1, u2 * u1^2)
        //   u3 = CT_ABS(2 * s * invsqrt * u1 * SQRT_MINUS_D)
        //   x = u3 * invsqrt * u2 * INVSQRT_MINUS_D
        //   y = (1 - ss) * invsqrt * u1
        //   t = x * y
        let ss = s.square();
        let u1 = GF448::ONE + ss;
        let u1_sqr = u1.square();
        let u2 = u1_sqr + ss.mul_small(Self::MINUS_FOUR_D);
        let (was_square, isqrt) = Self::sqrt_ratio_m1(GF448::ONE, u2 * u1_sqr);
        let u3 = Self::abs(s.mul2() * isqrt * u1 * Self::SQRT_MINUS_D);
        let x = u3 * isqrt * u2 * Self::INVSQRT_MINUS_D;
        let y = (GF448::ONE - ss) * isqrt * u1;
        // We do not compute t because ed448::Point uses projective (X:Y:Z)
        // coordinates, not extended coordinates.
        // let t = x * y;

        // If u2*u1^2 was not square, then the point is invalid.
        r &= was_square;

        self.0.set_cond(&Ed448Point { X: x, Y: y, Z: GF448::ONE }, r);
        r
    }

    /// Decodes an element from its binary representation.
    ///
    /// If the input does not have length exactly 56 bytes, or if the
    /// input has length 56 bytes but is not the valid, canonical encoding
    /// of a decaf448 point, then this function returns `None`.
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
    pub fn encode(self) -> [u8; 56] {
        // Draft spec, section 5.3.2, uses the extended coordinates
        // (X:Y:Z:T), with x = X/Z, y = Y/Z, and x*y = T/Z. The ed448::Point
        // structure only has X, Y and Z, so we must first convert it to
        // extended coordinates, with a few multiplications. Note that Z
        // is always non-zero.
        let x0 = self.0.X * self.0.Z;
        // let y0 = self.0.Y * self.0.Z;
        let z0 = self.0.Z.square();
        let t0 = self.0.X * self.0.Y;

        // u1 = (x0 + t0) * (x0 - t0) 
        // (_, invsqrt) = SQRT_RATIO_M1(1, u1 * ONE_MINUS_D * x0^2) 
        // ratio = CT_ABS(invsqrt * u1 * SQRT_MINUS_D) 
        // u2 = INVSQRT_MINUS_D * ratio * z0 - t0 
        // s = CT_ABS(ONE_MINUS_D * invsqrt * x0 * u2)
        let u1 = (x0 + t0) * (x0 - t0);
        let (_, invsqrt) = Self::sqrt_ratio_m1(GF448::ONE,
            u1.mul_small(Self::ONE_MINUS_D) * x0.square());
        let ratio = Self::abs(invsqrt * u1 * Self::SQRT_MINUS_D);
        let u2 = Self::INVSQRT_MINUS_D * ratio * z0 - t0;
        let s = Self::abs(invsqrt.mul_small(Self::ONE_MINUS_D) * x0 * u2);

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
        // Draft spec, section 5.3.3.
        let (x1, y1) = (&self.0.X, &self.0.Y);
        let (x2, y2) = (&rhs.0.X, &rhs.0.Y);
        (x1 * y2).equals(y1 * x2)
    }

    /// Tests whether this element is the neutral (identity point).
    ///
    /// Returned value is 0xFFFFFFFF for the neutral, 0x00000000 for
    /// all other elements.
    #[inline(always)]
    pub fn isneutral(self) -> u32 {
        // We use the equals() formula against the edwards448 neutral (0,1)
        // (which is a valid representation of the decaf448 neutral).
        self.0.X.iszero()
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

    /// Decaf448 map (bytes to point).
    ///
    /// This is the map described in section 5.3.4 of the draft under the
    /// name "MAP". Its output is not uniformly distributed; in general,
    /// it should not be used directly, but only through `one_way_map()`,
    /// which invokes it twice (on distinct inputs) and adds the two
    /// results.
    fn map(buf: &[u8]) -> Self {
        assert!(buf.len() == 56);

        // Decode the 56 bytes into a field element (with reduction).
        let t = GF448::decode_reduce(buf);

        let r = -t.square();
        let rm1 = r - GF448::ONE;
        let rp1 = r + GF448::ONE;
        let u0 = -rm1.mul_small(Self::MINUS_D);
        let u1 = (u0 + GF448::ONE) * (u0 - r);
        let (was_square, v) = Self::sqrt_ratio_m1(
            Self::ONE_MINUS_TWO_D_FULL, rp1 * u1);
        let v_prime = GF448::select(&(t * v), &v, was_square);
        let sgn = GF448::select(&GF448::MINUS_ONE, &GF448::ONE, was_square);
        let s = v_prime * rp1;

        let ss = s.square();
        let w0 = Self::abs(s).mul2();
        let w1 = ss + GF448::ONE;
        let w2 = ss - GF448::ONE;
        let w3 = v_prime * s * rm1.mul_small(Self::ONE_MINUS_TWO_D) + sgn;

        // Extended coordinates are:
        //   X = w0*w3
        //   Y = w2*w1
        //   Z = w1*w3
        //   T = w0*w2
        // We do not compute T because ed448::Point uses projective
        // coordinates.
        Self(Ed448Point { X: w0 * w3, Y: w2 * w1, Z: w1 * w3 })
    }

    /// The one-way map of bytes to decaf448 elements.
    ///
    /// This is the map described in the draft, section 5.3.4. The input
    /// MUST have length exactly 112 bytes (a panic is triggered otherwise).
    /// If the input is itself a 112-byte output of a secure hash function
    /// (e.g. SHAKE256) then this constitutes a hash function with output
    /// in decaf448 (output is then indistinguishable from random
    /// uniform selection).
    pub fn one_way_map(buf: &[u8]) -> Self {
        assert!(buf.len() == 112);
        Self::map(&buf[..56]) + Self::map(&buf[56..])
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
    /// The function is constant-time with regard to the decaf448
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
        // n is multiplied by 2 to account for the base point discrepancy.
        self.0.set_mulgen(&n.mul2());
    }

    /// Returns the product of the conventional base (`Self::BASE`) by
    /// the provided scalar.
    #[inline(always)]
    pub fn mulgen(n: &Scalar) -> Self {
        // n is multiplied by 2 to account for the base point discrepancy.
        Self(Ed448Point::mulgen(&n.mul2()))
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
        // v is multiplied by 2 to account for the base point discrepancy.
	    Self(self.0.mul_add_mulgen_vartime(u, &v.mul2()))
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
        // s is multiplied by 2 to account for the base point discrepancy.
        self.0.verify_helper_vartime(&R.0, &s.mul2(), k)
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
    use crate::sha2::Sha512;

    /*
    use std::fmt;
    use crate::field::GF448;

    fn print_gf(name: &str, x: GF448) {
        print!("{} = 0x", name);
        let bb = x.encode();
        for i in (0..56).rev() {
            print!("{:02X}", bb[i]);
        }
        println!();
    }

    fn print(name: &str, P: Point) {
        println!("{}:", name);
        print_gf("  X", P.0.X);
        print_gf("  Y", P.0.Y);
        print_gf("  Z", P.0.Z);
    }
    */

    // Test vectors from draft-irtf-cfrg-ristretto255-decaf448-07,
    // section B.1.
    const VEC_MULGEN: [&str; 16] = [
        "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "6666666666666666666666666666666666666666666666666666666633333333333333333333333333333333333333333333333333333333",
        "c898eb4f87f97c564c6fd61fc7e49689314a1f818ec85eeb3bd5514ac816d38778f69ef347a89fca817e66defdedce178c7cc709b2116e75",
        "a0c09bf2ba7208fda0f4bfe3d0f5b29a543012306d43831b5adc6fe7f8596fa308763db15468323b11cf6e4aeb8c18fe44678f44545a69bc",
        "b46f1836aa287c0a5a5653f0ec5ef9e903f436e21c1570c29ad9e5f596da97eeaf17150ae30bcb3174d04bc2d712c8c7789d7cb4fda138f4",
        "1c5bbecf4741dfaae79db72dface00eaaac502c2060934b6eaaeca6a20bd3da9e0be8777f7d02033d1b15884232281a41fc7f80eed04af5e",
        "86ff0182d40f7f9edb7862515821bd67bfd6165a3c44de95d7df79b8779ccf6460e3c68b70c16aaa280f2d7b3f22d745b97a89906cfc476c",
        "502bcb6842eb06f0e49032bae87c554c031d6d4d2d7694efbf9c468d48220c50f8ca28843364d70cee92d6fe246e61448f9db9808b3b2408",
        "0c9810f1e2ebd389caa789374d78007974ef4d17227316f40e578b336827da3f6b482a4794eb6a3975b971b5e1388f52e91ea2f1bcb0f912",
        "20d41d85a18d5657a29640321563bbd04c2ffbd0a37a7ba43a4f7d263ce26faf4e1f74f9f4b590c69229ae571fe37fa639b5b8eb48bd9a55",
        "e6b4b8f408c7010d0601e7eda0c309a1a42720d6d06b5759fdc4e1efe22d076d6c44d42f508d67be462914d28b8edce32e7094305164af17",
        "be88bbb86c59c13d8e9d09ab98105f69c2d1dd134dbcd3b0863658f53159db64c0e139d180f3c89b8296d0ae324419c06fa87fc7daaf34c1",
        "a456f9369769e8f08902124a0314c7a06537a06e32411f4f93415950a17badfa7442b6217434a3a05ef45be5f10bd7b2ef8ea00c431edec5",
        "186e452c4466aa4383b4c00210d52e7922dbf9771e8b47e229a9b7b73c8d10fd7ef0b6e41530f91f24a3ed9ab71fa38b98b2fe4746d51d68",
        "4ae7fdcae9453f195a8ead5cbe1a7b9699673b52c40ab27927464887be53237f7f3a21b938d40d0ec9e15b1d5130b13ffed81373a53e2b43",
        "841981c3bfeec3f60cfeca75d9d8dc17f46cf0106f2422b59aec580a58f342272e3a5e575a055ddb051390c54c24c6ecb1e0aceb075f6056",
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

    // Test vectors from draft-irtf-cfrg-ristretto255-decaf448-07,
    // section B.2.
    const VEC_INVALID: [&str; 21] = [
        // Non-canonical field encodings.
        "8e24f838059ee9fef1e209126defe53dcd74ef9b6304601c6966099effffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "86fcc7212bd4a0b980928666dc28c444a605ef38e09fb569e28d4443ffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "866d54bd4c4ff41a55d4eefdbeca73cbd653c7bd3135b383708ec0bdffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "4a380ccdab9c86364a89e77a464d64f9157538cfdfa686adc0d5ece4ffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "f22d9d4c945dd44d11e0b1d3d3d358d959b4844d83b08c44e659d79fffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "8cdffc681aa99e9c818c8ef4c3808b58e86acdef1ab68c8477af185bffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "0e1c12ac7b5920effbd044e897c57634e2d05b5c27f8fa3df8a086a1ffffffffffffffffffffffffffffffffffffffffffffffffffffffff",

        // Negative field elements.
        "15141bd2121837ef71a0016bd11be757507221c26542244f23806f3fd3496b7d4c36826276f3bf5deea2c60c4fa4cec69946876da497e795",
        "455d380238434ab740a56267f4f46b7d2eb2dd8ee905e51d7b0ae8a6cb2bae501e67df34ab21fa45946068c9f233939b1d9521a998b7cb93",
        "810b1d8e8bf3a9c023294bbfd3d905a97531709bdc0f42390feedd7010f77e98686d400c9c86ed250ceecd9de0a18888ffecda0f4ea1c60d",
        "d3af9cc41be0e5de83c0c6273bedcb9351970110044a9a41c7b9b2267cdb9d7bf4dc9c2fdb8bed32878184604f1d9944305a8df4274ce301",
        "9312bcab09009e4330ff89c4bc1e9e000d863efc3c863d3b6c507a40fd2cdefde1bf0892b4b5ed9780b91ed1398fb4a7344c605aa5efda74",
        "53d11bce9e62a29d63ed82ae93761bdd76e38c21e2822d6ebee5eb1c5b8a03eaf9df749e2490eda9d8ac27d1f71150de93668074d18d1c3a",
        "697c1aed3cd8858515d4be8ac158b229fe184d79cb2b06e49210a6f3a7cd537bcd9bd390d96c4ab6a4406da5d93640726285370cfa95df80",

        // Non-square x^2.
        "58ad48715c9a102569b68b88362a4b0645781f5a19eb7e59c6a4686fd0f0750ff42e3d7af1ab38c29d69b670f31258919c9fdbf6093d06c0",
        "8ca37ee2b15693f06e910cf43c4e32f1d5551dda8b1e48cb6ddd55e440dbc7b296b601919a4e4069f59239ca247ff693f7daa42f086122b1",
        "982c0ec7f43d9f97c0a74b36db0abd9ca6bfb98123a90782787242c8a523cdc76df14a910d54471127e7662a1059201f902940cd39d57af5",
        "baa9ab82d07ca282b968a911a6c3728d74bf2fe258901925787f03ee4be7e3cb6684fd1bcfe5071a9a974ad249a4aaa8ca81264216c68574",
        "2ed9ffe2ded67a372b181ac524996402c42970629db03f5e8636cbaf6074b523d154a7a8c4472c4c353ab88cd6fec7da7780834cc5bd5242",
        "f063769e4241e76d815800e4933a3a144327a30ec40758ad3723a788388399f7b3f5d45b6351eb8eddefda7d5bff4ee920d338a8b89d8b63",
        "5a0104f1f55d152ceb68bc138182499891d90ee8f09b40038ccc1e07cb621fd462f781d045732a4f0bda73f0b2acf94355424ff0388d4b9c",
    ];

    #[test]
    fn invalid() {
        for s in VEC_INVALID.iter() {
            let buf = hex::decode(s).unwrap();
            assert!(Point::decode(&buf[..]).is_none());
        }
    }

    // Test vectors from draft-irtf-cfrg-ristretto255-decaf448-07,
    // section A.B.
    struct Decaf448MapTestVector<'a> {
        I: &'a str,
        O: &'a str,
    }
    const VEC_MAP: [Decaf448MapTestVector; 7] = [
        Decaf448MapTestVector {
            I: "cbb8c991fd2f0b7e1913462d6463e4fd2ce4ccdd28274dc2ca1f4165d5ee6cdccea57be3416e166fd06718a31af45a2f8e987e301be59ae6673e963001dbbda80df47014a21a26d6c7eb4ebe0312aa6fffb8d1b26bc62ca40ed51f8057a635a02c2b8c83f48fa6a2d70f58a1185902c0",
            O: "0c709c9607dbb01c94513358745b7c23953d03b33e39c7234e268d1d6e24f34014ccbc2216b965dd231d5327e591dc3c0e8844ccfd568848",
        },

        Decaf448MapTestVector {
            I: "b6d8da654b13c3101d6634a231569e6b85961c3f4b460a08ac4a5857069576b64428676584baa45b97701be6d0b0ba18ac28d443403b45699ea0fbd1164f5893d39ad8f29e48e399aec5902508ea95e33bc1e9e4620489d684eb5c26bc1ad1e09aba61fabc2cdfee0b6b6862ffc8e55a",
            O: "76ab794e28ff1224c727fa1016bf7f1d329260b7218a39aea2fdb17d8bd9119017b093d641cedf74328c327184dc6f2a64bd90eddccfcdab",
        },

        Decaf448MapTestVector {
            I: "36a69976c3e5d74e4904776993cbac27d10f25f5626dd45c51d15dcf7b3e6a5446a6649ec912a56895d6baa9dc395ce9e34b868d9fb2c1fc72eb6495702ea4f446c9b7a188a4e0826b1506b0747a6709f37988ff1aeb5e3788d5076ccbb01a4bc6623c92ff147a1e21b29cc3fdd0e0f4",
            O: "c8d7ac384143500e50890a1c25d643343accce584caf2544f9249b2bf4a6921082be0e7f3669bb5ec24535e6c45621e1f6dec676edd8b664",
        },

        Decaf448MapTestVector {
            I: "d5938acbba432ecd5617c555a6a777734494f176259bff9dab844c81aadcf8f7abd1a9001d89c7008c1957272c1786a4293bb0ee7cb37cf3988e2513b14e1b75249a5343643d3c5e5545a0c1a2a4d3c685927c38bc5e5879d68745464e2589e000b31301f1dfb7471a4f1300d6fd0f99",
            O: "62beffc6b8ee11ccd79dbaac8f0252c750eb052b192f41eeecb12f2979713b563caf7d22588eca5e80995241ef963e7ad7cb7962f343a973",
        },

        Decaf448MapTestVector {
            I: "4dec58199a35f531a5f0a9f71a53376d7b4bdd6bbd2904234a8ea65bbacbce2a542291378157a8f4be7b6a092672a34d85e473b26ccfbd4cdc6739783dc3f4f6ee3537b7aed81df898c7ea0ae89a15b5559596c2a5eeacf8b2b362f3db2940e3798b63203cae77c4683ebaed71533e51",
            O: "f4ccb31d263731ab88bed634304956d2603174c66da38742053fa37dd902346c3862155d68db63be87439e3d68758ad7268e239d39c4fd3b",
        },

        Decaf448MapTestVector {
            I: "df2aa1536abb4acab26efa538ce07fd7bca921b13e17bc5ebcba7d1b6b733deda1d04c220f6b5ab35c61b6bcb15808251cab909a01465b8ae3fc770850c66246d5a9eae9e2877e0826e2b8dc1bc08009590bc6778a84e919fbd28e02a0f9c49b48dc689eb5d5d922dc01469968ee81b5",
            O: "7e79b00e8e0a76a67c0040f62713b8b8c6d6f05e9c6d02592e8a22ea896f5deacc7c7df5ed42beae6fedb9000285b482aa504e279fd49c32",
        },

        Decaf448MapTestVector {
            I: "e9fb440282e07145f1f7f5ecf3c273212cd3d26b836b41b02f108431488e5e84bd15f2418b3d92a3380dd66a374645c2a995976a015632d36a6c2189f202fc766e1c82f50ad9189be190a1f0e8f9b9e69c9c18cc98fdd885608f68bf0fdedd7b894081a63f70016a8abf04953affbefa",
            O: "20b171cb16be977f15e013b9752cf86c54c631c4fc8cbf7c03c4d3ac9b8e8640e7b0e9300b987fe0ab5044669314f6ed1650ae037db853f1",
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
        let mut sh = Sha512::new();
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
