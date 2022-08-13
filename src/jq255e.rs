//! Jq255e implementation.
//!
//! This module implements generic group operations on the jq255e
//! group, which is itself isomorphic to a subgroup of the
//! double-odd elliptic curve of equation `y^2 = x*(x^2 - 2)` over
//! the finite field GF(2^255 - 18651). This group is described
//! on the [double-odd site]. The group has a prime order order `r`
//! (an integer slightly below 2^254). A conventional base point is
//! defined; like all non-neutral elements in a prime order group, it
//! generates the whole group.
//!
//! A group element is represented by the `Point` structure. Group
//! elements are called "points" because they are internally represented
//! by points on an elliptic curve; however, the `Point` structure, by
//! construction, contains only proper representatives of the group
//! element, not just any point. `Point` instances can be used in
//! additions and subtractions with the usual `+` and `-` operators; all
//! combinations of raw values and references are accepted, as well as
//! compound assignment operators `+=` and `-=`. Specialized functions
//! are available, in particular for point doubling (`Point::double()`)
//! and for sequences of successive doublings (`Point::xdouble()`), the
//! latter using some extra optimizations. Multiplication by an integer
//! (`u64` type) or a scalar (`Scalar` structure) is also accepted, using
//! the `*` and `*=` operators. Scalars are integers modulo `r`. The
//! `Scalar` structure represents such a value; it implements all usual
//! arithmetic operators (`+`, `-`, `*` and `/`, as well as `+=`, `-=`,
//! `*=` and `/=`).
//!
//! Scalars can be encoded over 32 bytes (using unsigned little-endian
//! convention) and decoded back. Encoding is always canonical, and
//! decoding always verifies that the value is indeed in the canonical
//! range.
//!
//! Points can be encoded over 32 bytes, and decoded back. As with
//! scalars, encoding is always canonical, and verified upon decoding.
//! Point encoding uses only 255 bits; the top bit (most significant bit
//! of the last byte) is always zero. The decoding process verifies that
//! the top bit is indeed zero.
//!
//! [double-odd site]: https://doubleodd.group/

// Projective/fractional coordinates traditionally use uppercase letters,
// using lowercase only for affine coordinates.
#![allow(non_snake_case)]

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::convert::TryFrom;
use super::field::{GF255e, ModInt256};
use blake2::{Blake2s256, Digest};
use rand_core::{CryptoRng, RngCore};

/// An element in the jq255e group.
#[derive(Clone, Copy, Debug)]
pub struct Point {
    // We use extended coordinates on the Jacobi quartic curve with
    // equation: e^2 = (a^2 - 4*b)*u^4 + u^2 + 1
    // The map from the base curve is defined as:
    //   u = x/y
    //   e = u^2*(x - b/x)
    // For the point (0,0) (the neutral in the jq255e group, which is the
    // unique point of order 2 on the curve), we set u = 0 and e = -1.
    // From the curve equation, e = (x^2 - b)/(x^2 + a*x + b), so that
    // it is always defined and non-zero, and e = -1 for x = 0; as for u,
    // it is the inverse of the slope of the line from (0,0) to the point,
    // so the extreme case for (0,0) itself is a vertical tangent, which
    // is why we use u = 0. Since addition of (0,0) on the curve becomes
    // on the quartic the transform (e,u) -> (-e,-u), we can also map
    // the point-at-infinity of the initial curve into (1,0) on the quartic.
    //
    // In extended coordinates, we have:
    //   Z != 0 and E != 0 for all points
    //   e = E/Z
    //   u = U/Z
    //   u^2 = T/Z   (hence U^2 = T*Z)
    E: GF255e,
    U: GF255e,
    Z: GF255e,
    T: GF255e,
}

/// Integers modulo r = 2^254 - 131528281291764213006042413802501683931.
///
/// `r` is the prime order of the jq255e group.
pub type Scalar = ModInt256<0x1F52C8AE74D84525, 0x9D0C930F54078C53,
                            0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF>;

impl Point {

    /// The group neutral element.
    pub const NEUTRAL: Self = Self {
        E: GF255e::MINUS_ONE,
        Z: GF255e::ONE,
        U: GF255e::ZERO,
        T: GF255e::ZERO,
    };

    /// The conventional base point (group generator).
    ///
    /// This point generates the whole group, which as prime order r
    /// (integers modulo r are represented by the `Scalar` type).
    pub const BASE: Self = Self {
        E: GF255e::w64be(0, 0, 0, 3),
        Z: GF255e::ONE,
        U: GF255e::ONE,
        T: GF255e::ONE,
    };

    /* unused
    /// The curve `a` constant (0).
    const A: GF255e = GF255e::ZERO;
    /// The curve `b` constant (-2).
    const B: GF255e = GF255e::w64be(
        0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFB723);
    */

    /// A square root of -1 in GF255e (we use the non-negative root).
    const ETA: GF255e = GF255e::w64be(
        0x10ED2DB33C69B85F, 0xE414983FE53688E3,
        0xA60D864FB30E6336, 0xD99E0F1BAA938AEE);

    /// Tries to decode a point from bytes.
    ///
    /// If the source slice has not length exactly 32 bytes, then
    /// decoding fails. If the source bytes are not a valid, canonical
    /// encoding of a group element, then decoding fails. On success,
    /// 0xFFFFFFFF is returned; on failure, 0x00000000 is returned. On
    /// failure, this point is set to the neutral.
    ///
    /// If the source length is exactly 32 bytes, then the decoding
    /// outcome (success or failure) should remain hidden from
    /// timing-based side channels.
    pub fn set_decode(&mut self, buf: &[u8]) -> u32 {
        // Check that the input length is correct.
        if buf.len() != 32 {
            *self = Self::NEUTRAL;
            return 0;
        }

        // Decode the u coordinate.
        let (u, mut r) = GF255e::decode32(buf);

        // e^2 = (a^2-4*b)*u^4 - 2*a*u^2 + 1
        let uu = u.square();
        let ee = uu.square().mul8() + GF255e::ONE;
        let (e, r2) = ee.sqrt();
        r &= r2;
        // GF255e::sqrt() already returns the non-negative root, we do
        // not have to test the sign of e and adjust.

        // We have the point in affine coordinates, except on failure,
        // in which case we have to adjust the values.
        self.E = GF255e::select(&GF255e::MINUS_ONE, &e, r);
        self.Z = GF255e::ONE;
        self.U = GF255e::select(&GF255e::ZERO, &u, r);
        self.T = GF255e::select(&GF255e::ZERO, &uu, r);
        r
    }

    /// Tries to decode a point from some bytes.
    ///
    /// Decoding succeeds only if the source slice has length exactly 32
    /// bytes, and contains the canonical encoding of a valid curve
    /// point. Sicne this method returns an `Option<Point>`, it
    /// inherently leaks (through timing-based side channels) whether
    /// decoding succeeded or not; to avoid that, consider using
    /// `set_decode()`. The decoded point itself, however, does not leak.
    pub fn decode(buf: &[u8]) -> Option<Point> {
        let mut P = Point::NEUTRAL;
        if P.set_decode(buf) != 0 {
            Some(P)
        } else {
            None
        }
    }

    /// Encodes this point into exactly 32 bytes.
    ///
    /// Encoding is always canonical.
    pub fn encode(self) -> [u8; 32] {
        // - Choose the element representant whose e coordinate is
        //   non-negative.
        // - Encode the u coordinate of that point.
        let iZ = GF255e::ONE / self.Z;
        let mut u = self.U * iZ;
        let sgn = (((self.E * iZ).encode32()[0] & 1) as u32).wrapping_neg();
        u.set_cond(&-u, sgn);
        u.encode32()
    }

    /// Creates a point by converting a point in extended affine
    /// coordinates (e, u, u^2).
    fn from_affine_extended(P: &PointAffineExtended) -> Self {
        Self {
            E: P.e,
            Z: GF255e::ONE,
            U: P.u,
            T: P.t,
        }
    }

    /// Adds another point (`rhs`) to this point.
    fn set_add(&mut self, rhs: &Self) {
        let (E1, Z1, U1, T1) = (&self.E, &self.Z, &self.U, &self.T);
        let (E2, Z2, U2, T2) = (&rhs.E, &rhs.Z, &rhs.U, &rhs.T);

        // Generic case (8M+3S):
        //   constants on the dual curve:
        //      a' = -2*a          (jq255e: a' = 0)
        //      b' = a^2 - 4*b     (jq255e: b' = 8)
        //   e1e2 = E1*E2
        //   z1z2 = Z1*Z2
        //   u1u2 = U1*U2
        //   t1t2 = T1*T2
        //     zt = (Z1 + T1)*(Z2 + T2) - z1z2 - t1t2
        //     eu = (E1 + U1)*(E2 + U2) - e1e2 - u1u2
        //     hd = z1z2 - b'*t1t2
        //     E3 = (z1z2 + b'*t1t2)*(e1e2 + a'*u1u2) + 2*b'*u1u2*zt
        //     Z3 = hd^2
        //     T3 = eu^2
        //     U3 = ((hd + eu)^2 - Z3 - T3)/2  # Or: U3 = hd*eu
        let e1e2 = E1 * E2;
        let u1u2 = U1 * U2;
        let z1z2 = Z1 * Z2;
        let t1t2 = T1 * T2;
        let eu = (E1 + U1) * (E2 + U2) - e1e2 - u1u2;
        let zt = (Z1 + T1) * (Z2 + T2) - z1z2 - t1t2;
        let bpt1t2 = t1t2.mul8();  // (a^2 - 4*b)*T1*T2
        let hd = z1z2 - bpt1t2;
        let T3 = eu.square();
        let Z3 = hd.square();
        let E3 = (z1z2 + bpt1t2) * e1e2 + u1u2.mul16() * zt;
        let U3 = hd * eu;  // faster than: ((hd + eu)^2 - Z3 - T3)/2
        self.E = E3;
        self.Z = Z3;
        self.U = U3;
        self.T = T3;
    }

    /// Specialized point addition routine when the other operand is in
    /// affine extended coordinates (used in the pregenerated tables for
    /// multiples of the base point).
    fn set_add_affine_extended(&mut self, rhs: &PointAffineExtended) {
        let (E1, Z1, U1, T1) = (&self.E, &self.Z, &self.U, &self.T);
        let (e2, u2, t2) = (&rhs.e, &rhs.u, &rhs.t);

        // Generic case (7M+3S):
        //   constants on the dual curve:
        //      a' = -2*a          (jq255e: a' = 0)
        //      b' = a^2 - 4*b     (jq255e: b' = 8)
        //   e1e2 = E1*E2
        //   u1u2 = U1*U2
        //   t1t2 = T1*T2
        //     zt = Z1*t2 + T1
        //     eu = (E1 + U1)*(E2 + U2) - e1e2 - u1u2
        //     hd = Z1 - b'*t1t2
        //     E3 = (Z1 + b'*t1t2)*(e1e2 + a'*u1u2) + 2*b'*u1u2*zt
        //     Z3 = hd^2
        //     T3 = eu^2
        //     U3 = ((hd + eu)^2 - Z3 - T3)/2  # Or: U3 = hd*eu
        let e1e2 = E1 * e2;
        let u1u2 = U1 * u2;
        let t1t2 = T1 * t2;
        let eu = (E1 + U1) * (e2 + u2) - e1e2 - u1u2;
        let zt = Z1 * t2 + T1;
        let bpt1t2 = t1t2.mul8();  // (a^2 - 4*b)*T1*T2
        let hd = Z1 - bpt1t2;
        let T3 = eu.square();
        let Z3 = hd.square();
        let E3 = (Z1 + bpt1t2) * e1e2 + u1u2.mul16() * zt;
        let U3 = hd * eu;  // faster than: ((hd + eu)^2 - Z3 - T3)/2
        self.E = E3;
        self.Z = Z3;
        self.U = U3;
        self.T = T3;
    }

    /// Specialized point subtraction routine when the other operand is in
    /// affine extended coordinates (used in the pregenerated tables for
    /// multiples of the base point).
    fn set_sub_affine_extended(&mut self, rhs: &PointAffineExtended) {
        let mrhs = PointAffineExtended {
            e: rhs.e,
            u: -rhs.u,
            t: rhs.t,
        };
        self.set_add_affine_extended(&mrhs);
    }

    /// Doubles this point (in place).
    pub fn set_double(&mut self) {
        let (E, Z, U) = (&self.E, &self.Z, &self.U);

        // P ezut -> 2*P xwj  (1M+3S)
        //    ee = E^2
        //    X  = ee^2
        //    W  = 2*Z^2 - ee
        //    J  = 2*E*U
        let ee = E.square();
        let J = E * U.mul2();
        let X = ee.square();
        let W = Z.square().mul2() - ee;

        // P xwj -> P ezut  (3S)
        //    ww = W^2
        //    jj = J^2
        //    E  = 2*X - ww
        //    Z  = ww
        //    U  = ((W + J)^2 - ww - jj)/2  # Or: U = W*J
        //    T  = jj
        let ww = W.square();
        let jj = J.square();
        self.E = X.mul2() - ww;
        self.Z = ww;
        self.U = W * J;  // faster than: ((W + J).square() - ww - jj).half()
        self.T = jj;
    }

    /// Doubles this point.
    #[inline(always)]
    pub fn double(self) -> Self {
        let mut r = self;
        r.set_double();
        r
    }

    /// Doubles this point n times (in place).
    pub fn set_xdouble(&mut self, n: u32) {
        if n == 0 {
            return;
        }

        // First doubling switches to xwj, with cost 1M+3S; subsequent
        // doublings work on the xwj representation (1M+5S each). At the
        // end, we convert back to ezut in cost 3S.
        let (E, Z, U) = (&self.E, &self.Z, &self.U);

        // P ezut -> 2*P xwj  (1M+3S)
        let ee = E.square();
        let mut J = E * U.mul2();
        let mut X = ee.square();
        let mut W = Z.square().mul2() - ee;

        // Subsequent doublings in xwj  (n-1)*(1M+5S)
        for _ in 1..n {
            // ww = W^2
            // t1 = ww - 2*X
            // t2 = t1^2
            // J' = ((W + t1)^2 - ww - t2)*J  # Or: J' = 2*W*t1*J
            // W' = t2 - 2*ww^2
            // X' = t2^2
            let ww = W.square();
            let t1 = ww - X.mul2();
            let t2 = t1.square();
            J *= t1 * W.mul2();  // faster than (W + t1)^2 - ww - t2
            W = t2 - ww.square().mul2();
            X = t2.square();
        }

        // Conversion xwj -> ezut  (3S)
        let ww = W.square();
        let jj = J.square();
        self.E = X.mul2() - ww;
        self.Z = ww;
        self.U = W * J;  // faster than: ((W + J).square() - ww - jj).half()
        self.T = jj;
    }

    /// Doubles this point n times.
    #[inline(always)]
    pub fn xdouble(self, n: u32) -> Self {
        let mut r = self;
        r.set_xdouble(n);
        r
    }

    /// Negates this point (in place).
    #[inline(always)]
    pub fn set_neg(&mut self) {
        self.U.set_neg();
    }

    /// Subtract another point (`rhs`) from this point.
    fn set_sub(&mut self, rhs: &Self) {
        self.set_add(&-rhs);
    }

    /// Multiplies this point by a small integer.
    ///
    /// This operation is constant-time with regard to the source point,
    /// but NOT with regard to the multiplier; the multiplier `n` MUST
    /// NOT be secret.
    pub fn set_mul_small(&mut self, n: u64) {
        if n == 0 {
            *self = Self::NEUTRAL;
            return;
        }
        if n == 1 {
            return;
        }

        let nlen = 64 - n.leading_zeros();
        let T = *self;
        let mut ndbl = 0u32;
        for i in (0..(nlen - 1)).rev() {
            ndbl += 1;
            if ((n >> i) & 1) == 0 {
                continue;
            }
            self.set_xdouble(ndbl);
            ndbl = 0;
            self.set_add(&T);
        }
        self.set_xdouble(ndbl);
    }

    /// Compares two points for equality.
    ///
    /// Returned value is 0xFFFFFFFF if the two points are equal,
    /// 0x00000000 otherwise.
    #[inline]
    pub fn equals(self, rhs: Self) -> u32 {
        // Points are equal if and only if they have the same image through
        // isogeny theta1:
        //    theta1(e, u) = (f, v)
        //    with f = (a^2 - 4*b)*u^2, and v = u/e
        // In the theta1 output, coordinate v of a point uniquely identifies
        // the point. Thus, we only need to compare u1/e1 and u2/e2, which
        // is equivalent to comparing u1*e2 and u2*e1 (since e1 and e2 are
        // never zero).
        (self.U * rhs.E).equals(rhs.U * self.E)
    }

    /// Tests whether this point is the neutral (identity point on the
    /// curve).
    ///
    /// Returned value is 0xFFFFFFFF for the neutral, 0x00000000
    /// otherwise.
    #[inline(always)]
    pub fn isneutral(self) -> u32 {
        self.U.iszero()
    }

    /// Conditionally copies the provided point (`P`) into `self`.
    ///
    ///  - If `ctl` = 0xFFFFFFFF, then the value of `P` is copied into `self`.
    ///
    ///  - If `ctl` = 0x00000000, then the value of `self` is unchanged.
    ///
    /// `ctl` MUST be equal to 0x00000000 or 0xFFFFFFFF.
    #[inline]
    pub fn set_cond(&mut self, P: &Self, ctl: u32) {
        self.E.set_cond(&P.E, ctl);
        self.Z.set_cond(&P.Z, ctl);
        self.U.set_cond(&P.U, ctl);
        self.T.set_cond(&P.T, ctl);
    }

    /// Returns a point equal to `P0` (if `ctl` = 0x00000000) or `P1` (if
    /// `ctl` = 0xFFFFFFFF).
    ///
    /// Value `ctl` MUST be either 0x00000000 or 0xFFFFFFFF.
    #[inline(always)]
    pub fn select(P0: &Self, P1: &Self, ctl: u32) -> Self {
        let mut P = *P0;
        P.set_cond(P1, ctl);
        P
    }

    /// Conditionally negates this point.
    ///
    /// This point is negated if `ctl` = 0xFFFFFFFF, but kept unchanged
    /// if `ctl` = 0x00000000. `ctl` MUST be equal to 0x00000000 or
    /// 0xFFFFFFFF.
    #[inline]
    pub fn set_condneg(&mut self, ctl: u32) {
        self.U.set_cond(&-self.U, ctl);
    }

    /// Maps a field element into a point.
    ///
    /// This map output is not uniformly distributed; in general, it should
    /// be used only through `hash_to_curve()`, which invokes this map
    /// twice.
    fn map_to_curve(f: &GF255e) -> Self {
        // We map to the dual curve E(a',b') with:
        //   a' = -2*a = 0
        //   b' = a^2 - 4*b = 8

        // x1num = 4*f^2 + (1 - b')
        // x2num = eta*(4*f^2 - (1 - b'))
        // x12den = 4*f
        let f2_4 = f.mul2().square();
        let x1num = f2_4 - GF255e::w64be(0, 0, 0, 7);
        let x2num = Self::ETA * (f2_4 + GF255e::w64be(0, 0, 0, 7));
        let x12den = f.mul4();

        // yy1num = 64*f^7 + 16*(3+b')*f^5 + 4*(3-2*b'-b'^2)*f^3 + (1-b')^3*f
        // yy2num = -eta*(64*f^7 - 16*(3+b')*f^5 + 4*(3-2*b'-b'^2)*f^3 - (1-b')^3*f)
        // y12den = 8*f^2
        let f3_4 = f2_4 * f;
        let f5_16 = f3_4 * f2_4;
        let f7_64 = f5_16 * f2_4;
        let yt1 = f7_64 - f3_4.mul_small(77);
        let yt2 = f5_16.mul_small(11) - f.mul_small(343);
        let yy1num = yt1 + yt2;
        let yy2num = Self::ETA * (yt2 - yt1);
        let y12den = f2_4.mul2();

        // Use x1 and y1 if yy1num is square.
        // Otherwise, use x2 and y2 if yy2num is square.
        // Otherwise, use x3 and y3.
        let ctl1 = !((yy1num.legendre() >> 1) as u32);
        let ctl2 = !ctl1 & !((yy2num.legendre() >> 1) as u32);
        let ctl3 = !ctl1 & !ctl2;
        let mut xnum = x1num;
        let mut xden = x12den;
        let mut yynum = yy1num;
        let mut yden = y12den;
        xnum.set_cond(&x2num, ctl2);
        yynum.set_cond(&yy2num, ctl2);
        xnum.set_cond(&(x1num * x2num), ctl3);
        xden.set_cond(&x12den.square(), ctl3);
        yynum.set_cond(&(yy1num * yy2num), ctl3);
        yden.set_cond(&y12den.square(), ctl3);
        let (ynum, _) = yynum.sqrt();  // sqrt() returns the non-negative root

        // u = x/y
        let unum = xnum * yden;
        let uden = xden * ynum;

        // Apply the theta_{1/2} isogeny to get back to curve E[a,b].
        //   x' = 4*b*u^2
        //   u' = 2*x/(u*(x^2 - b'))
        let Xnum = -unum.square().mul8();
        let mut Xden = uden.square();
        let Unum = (xnum * xden * uden).mul2();
        let mut Uden = unum * (xnum.square() - xden.square().mul8());

        // If the source scalar was zero, then computations above were not
        // good and we got zero in all values; we must fix the denominators.
        let fz = f.iszero();
        Xden.set_cond(&GF255e::ONE, fz);
        Uden.set_cond(&GF255e::ONE, fz);

        // Compute the 'e' coordinate with e = (x^2 - b)/(x^2 + a*x + b).
        let xx = Xnum.square();
        let mbzz = Xden.square().mul2();
        let Enum = xx + mbzz;
        let Eden = xx - mbzz;

        // Convert to extended coordinates.
        let ud2 = Uden.square();
        let uned = Unum * Eden;
        let E = Enum * ud2;
        let Z = Eden * ud2;
        let U = Uden * uned;
        let T = Unum * uned;

        Self { E, Z, U, T }
    }

    /// Hashes some data into a point.
    ///
    /// Given some input bytes, a group element is deterministically
    /// generated; the output distribution should be indistinguishable
    /// from uniform random generation, and the discrete logarithm of the
    /// output relatively to any given point is unknown.
    ///
    /// The input bytes are provided as `data`. If these bytes are a
    /// hash value, then the hash function name should be provided as
    /// `hash_name`, corresponding to one of the defined constants
    /// (`HASHNAME_SHA256`, `HASHNAME_BLAKE2S`, etc). In general, the
    /// name to use is the "formal" name of the hash function, converted
    /// to lowercase and without punctuation signs (e.g. SHA-256 uses
    /// the name `sha256`). If the input bytes are not an already
    /// computed hash value, but some other raw data, then `hash_name`
    /// shall be set to an empty string.
    pub fn hash_to_curve(hash_name: &str, data: &[u8]) -> Self {
        let mut sh = Blake2s256::new();
        let (blob1, blob2);
        if hash_name.len() == 0 {
            sh.update(&[0x01u8, 0x52u8]);
            sh.update(data);
            blob1 = sh.finalize_reset();
            sh.update(&[0x02u8, 0x52u8]);
            sh.update(data);
            blob2 = sh.finalize_reset();
        } else {
            sh.update(&[0x01u8, 0x48u8]);
            sh.update(hash_name.as_bytes());
            sh.update(&[0x00u8]);
            sh.update(data);
            blob1 = sh.finalize_reset();
            sh.update(&[0x02u8, 0x48u8]);
            sh.update(hash_name.as_bytes());
            sh.update(&[0x00u8]);
            sh.update(data);
            blob2 = sh.finalize_reset();
        }
        let f1 = GF255e::decode_reduce(&blob1);
        let f2 = GF255e::decode_reduce(&blob2);
        Self::map_to_curve(&f1) + Self::map_to_curve(&f2)
    }

    pub const HASHNAME_SHA224:      &'static str = "sha224";
    pub const HASHNAME_SHA256:      &'static str = "sha256";
    pub const HASHNAME_SHA384:      &'static str = "sha384";
    pub const HASHNAME_SHA512:      &'static str = "sha512";
    pub const HASHNAME_SHA512_224:  &'static str = "sha512224";
    pub const HASHNAME_SHA512_256:  &'static str = "sha512256";
    pub const HASHNAME_SHA3_224:    &'static str = "sha3224";
    pub const HASHNAME_SHA3_256:    &'static str = "sha3256";
    pub const HASHNAME_SHA3_384:    &'static str = "sha3384";
    pub const HASHNAME_SHA3_512:    &'static str = "sha3512";
    pub const HASHNAME_BLAKE2B:     &'static str = "blake2b";
    pub const HASHNAME_BLAKE2S:     &'static str = "blake2s";
    pub const HASHNAME_BLAKE3:      &'static str = "blake3";

    /// Recodes a scalar into 51 signed digits.
    ///
    /// Each digit is in -15..+16, top digit is in 0..+16.
    fn recode_scalar(n: &Scalar) -> [i8; 51] {
        let mut sd = [0i8; 51];
        let bb = n.encode32();
        let mut cc: u32 = 0;       // carry from lower digits
        let mut i: usize = 0;      // index of next source byte
        let mut acc: u32 = 0;      // buffered bits
        let mut acc_len: i32 = 0;  // number of buffered bits
        for j in 0..51 {
            if acc_len < 5 {
                acc |= (bb[i] as u32) << acc_len;
                acc_len += 8;
                i += 1;
            }
            let d = (acc & 0x1F) + cc;
            acc >>= 5;
            acc_len -= 5;
            let m = 16u32.wrapping_sub(d) >> 8;
            sd[j] = (d.wrapping_sub(m & 32)) as i8;
            cc = m & 1;
        }
        sd
    }

    /// Recodes a half-width scalar into 26 signed digits.
    ///
    /// Each digit is in -15..+16, top digit is in 0..+8.
    fn recode_u128(n: u128) -> [i8; 26] {
        let mut sd = [0i8; 26];
        let mut x = n;
        let mut cc: u32 = 0;       // carry from lower digits
        for j in 0..26 {
            let d = ((x as u32) & 0x1F) + cc;
            x >>= 5;
            let m = 16u32.wrapping_sub(d) >> 8;
            sd[j] = (d.wrapping_sub(m & 32)) as i8;
            cc = m & 1;
        }
        sd
    }

    /// Lookups a point from a window, with sign handling (constant-time).
    fn lookup(win: &[Self; 16], k: i8) -> Self {
        // Split k into its sign s (0xFFFFFFFF for negative) and
        // absolute value (f).
        let s = ((k as i32) >> 8) as u32;
        let f = ((k as u32) ^ s).wrapping_sub(s);
        let mut P = Self::NEUTRAL;
        for i in 0..16 {
            // win[i] contains (i+1)*P; we want to keep it if (and only if)
            // i+1 == f.
            // Values a-b and b-a both have their high bit equal to 0 only
            // if a == b.
            let j = (i as u32) + 1;
            let w = !(f.wrapping_sub(j) | j.wrapping_sub(f));
            let w = ((w as i32) >> 31) as u32;

            P.E.set_cond(&win[i].E, w);
            P.Z.set_cond(&win[i].Z, w);
            P.U.set_cond(&win[i].U, w);
            P.T.set_cond(&win[i].T, w);
        }

        // Negate the returned value if needed.
        P.U.set_cond(&-P.U, s);

        P
    }

    /// Supports scalar splitting.
    ///
    /// Given a 256-bit integer k (unsigned, provided as 8 32-bit limbs in
    /// little-endian order, less than the group order r) and a multiplier
    /// integer e (lower than 2^127 - 2), compute y = round(k*e / r).
    fn mul_divr_rounded(k: &[u32; 8], e: &[u32; 4]) -> [u32; 4] {
        // TODO: see if this code should be moved to the backends, so
        // that an optimized 64-bit version could be made. This is
        // probably not worth the effort.

        // Computations are done over 32-bit limbs because we do not
        // trust the standard support for integers larger than the
        // native register size to be constant-time.

        // z <- k*e
        let mut z = [0u32; 12];
        for i in 0..8 {
            let w = (k[i] as u64) * (e[0] as u64) + (z[i] as u64);
            z[i] = w as u32;
            let cc = w >> 32;
            let w = (k[i] as u64) * (e[1] as u64) + (z[i + 1] as u64) + cc;
            z[i + 1] = w as u32;
            let cc = w >> 32;
            let w = (k[i] as u64) * (e[2] as u64) + (z[i + 2] as u64) + cc;
            z[i + 2] = w as u32;
            let cc = w >> 32;
            let w = (k[i] as u64) * (e[3] as u64) + (z[i + 3] as u64) + cc;
            z[i + 3] = w as u32;
            z[i + 4] = (w >> 32) as u32;
        }

        // (r-1)/2
        const HR: [u32; 12] = [
            0x3A6C2292, 0x8FA96457, 0xAA03C629, 0xCE864987,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x1FFFFFFF,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ];

        // z <- z + (r-1)/2
        let mut cc = 0u32;
        for i in 0..12 {
            let w = (z[i] as u64) + (HR[i] as u64) + (cc as u64);
            z[i] = w as u32;
            cc = (w >> 32) as u32;
        }

        // y <- floor(z / 2^254) + 1
        let mut y = [
            (z[ 7] >> 30) | (z[ 8] << 2),
            (z[ 8] >> 30) | (z[ 9] << 2),
            (z[ 9] >> 30) | (z[10] << 2),
            (z[10] >> 30) | (z[11] << 2),
        ];
        let mut cc = 1u32;
        for i in 0..4 {
            let w = (y[i] as u64) + (cc as u64);
            y[i] = w as u32;
            cc = (w >> 32) as u32;
        }

        // r0 = 2^255 - r
        const R0: [u32; 4] = [
            0x8B27BADB, 0xE0AD3751, 0xABF873AC, 0x62F36CF0,
        ];

        // t <- y*r0
        let mut t = [0u32; 8];
        for i in 0..4 {
            let w = (y[i] as u64) * (R0[0] as u64) + (t[i] as u64);
            t[i] = w as u32;
            let cc = w >> 32;
            let w = (y[i] as u64) * (R0[1] as u64) + (t[i + 1] as u64) + cc;
            t[i + 1] = w as u32;
            let cc = w >> 32;
            let w = (y[i] as u64) * (R0[2] as u64) + (t[i + 2] as u64) + cc;
            t[i + 2] = w as u32;
            let cc = w >> 32;
            let w = (y[i] as u64) * (R0[3] as u64) + (t[i + 3] as u64) + cc;
            t[i + 3] = w as u32;
            t[i + 4] = (w >> 32) as u32;
        }

        // t <- t + z0
        // We are only interested in the high limb.
        z[7] &= 0x3FFFFFFF;
        let mut w = (z[0] as u64) + (t[0] as u64);
        for i in 1..8 {
            w = (z[i] as u64) + (t[i] as u64) + (w >> 32);
        }

        // The high limb is in w and is lower than 2^31. If it is
        // lower than 2^30, then y is too large and we must decrement
        // it; otherwise, we keep it unchanged.
        let w = (y[0] as u64).wrapping_sub(1 - (w >> 30));
        y[0] = w as u32;
        let w = (y[1] as u64).wrapping_sub(w >> 63);
        y[1] = w as u32;
        let w = (y[2] as u64).wrapping_sub(w >> 63);
        y[2] = w as u32;
        y[3] -= (w >> 32) as u32;

        y
    }

    /// Splits a scalar k into k0 and k1 (signed) such that
    /// k = k0 + k1*mu (with mu being a given square root of -1 modulo r).
    ///
    /// This function returns |k0|, sgn(k0), |k1| and sgn(k1), with
    /// sgn(x) = 0xFFFFFFFF for x < 0, 0x00000000 for x >= 0.
    fn split_mu(k: &Scalar) -> (u128, u32, u128, u32) {
        // Obtain k as an integer t in the 0..r-1 range.
        let mut t = [0u32; 8];
        let bb = k.encode32();
        for i in 0..8 {
            t[i] = u32::from_le_bytes(*<&[u8; 4]>::try_from(
                &bb[(4 * i)..(4 * i + 4)]).unwrap());
        }

        const EU: [u32; 4] = [
            0xC93F6111, 0x2ACCF9DE, 0x53C2C6E6, 0x1A509F7A
        ];

        const EV: [u32; 4] = [
            0x5466F77E, 0x0B7A3130, 0xFFBB3A93, 0x7D440C6A
        ];

        // c <- round(t*v / r)
        // d <- round(t*u / r)
        let c = Self::mul_divr_rounded(&t, &EV);
        let d = Self::mul_divr_rounded(&t, &EU);

        // k0 = k - d*u - c*v
        // k1 = d*v - c*u

        fn mul128(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
            let w = (a[0] as u64) * (b[0] as u64);
            let d0 = w as u32;
            let w = (a[0] as u64) * (b[1] as u64) + (w >> 32);
            let mut d1 = w as u32;
            let w = (a[0] as u64) * (b[2] as u64) + (w >> 32);
            let mut d2 = w as u32;
            let mut d3 = a[0].wrapping_mul(b[3]).wrapping_add((w >> 32) as u32);

            let w = (a[1] as u64) * (b[0] as u64) + (d1 as u64);
            d1 = w as u32;
            let w = (a[1] as u64) * (b[1] as u64) + (d2 as u64) + (w >> 32);
            d2 = w as u32;
            d3 = a[1].wrapping_mul(b[2]).wrapping_add(d3)
                .wrapping_add((w >> 32) as u32);

            let w = (a[2] as u64) * (b[0] as u64) + (d2 as u64);
            d2 = w as u32;
            d3 = a[2].wrapping_mul(b[1]).wrapping_add(d3)
                .wrapping_add((w >> 32) as u32);

            d3 = a[3].wrapping_mul(b[0]).wrapping_add(d3);

            [ d0, d1, d2, d3 ]
        }

        fn sub128(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
            let w = (a[0] as u64).wrapping_sub(b[0] as u64);
            let d0 = w as u32;
            let w = (a[1] as u64).wrapping_sub(b[1] as u64)
                .wrapping_sub(w >> 63);
            let d1 = w as u32;
            let w = (a[2] as u64).wrapping_sub(b[2] as u64)
                .wrapping_sub(w >> 63);
            let d2 = w as u32;
            let d3 = a[3].wrapping_sub(b[3]).wrapping_sub((w >> 63) as u32);
            [ d0, d1, d2, d3 ]
        }

        fn cneg128(a: [u32; 4]) -> (u128, u32) {
            let s = ((a[3] as i32) >> 31) as u32;
            let w = ((a[0] ^ s) as u64).wrapping_sub(s as u64);
            let d0 = w as u32;
            let w = ((a[1] ^ s) as u64).wrapping_sub(s as u64)
                .wrapping_sub(w >> 63);
            let d1 = w as u32;
            let w = ((a[2] ^ s) as u64).wrapping_sub(s as u64)
                .wrapping_sub(w >> 63);
            let d2 = w as u32;
            let d3 = (a[3] ^ s).wrapping_sub(s).wrapping_sub((w >> 63) as u32);

            let d = (d0 as u128)
                  | ((d1 as u128) << 32)
                  | ((d2 as u128) << 64)
                  | ((d3 as u128) << 96);
            (d, s)
        }

        let tt = [ t[0], t[1], t[2], t[3] ];
        let k0 = sub128(sub128(tt, mul128(d, EU)), mul128(c, EV));
        let k1 = sub128(mul128(d, EV), mul128(c, EU));

        let (n0, s0) = cneg128(k0);
        let (n1, s1) = cneg128(k1);
        (n0, s0, n1, s1)
    }

    /// Endomorphism on the group.
    fn zeta(self) -> Self {
        Self {
            E: self.E,
            Z: self.Z,
            U: self.U * Self::ETA,
            T: -self.T
        }
    }

    /// Multiplies this point by a scalar (in place).
    ///
    /// This operation is constant-time with regard to both the points
    /// and the scalar value.
    pub fn set_mul(&mut self, n: &Scalar) {
        // Split the scalar with the endomorphism.
        let (n0, s0, n1, s1) = Self::split_mu(n);

        // Compute the 5-bit windows:
        //   win0[i] = (i+1)*sgn(k0)*P
        //   win1[i] = (i+1)*sgn(k1)*zeta(P)
        // with zeta(e, u) = (e, u*eta) for eta = sqrt(-1) (this is an
        // endomorphism on the group).
        let mut win0 = [Self::NEUTRAL; 16];
        win0[0] = *self;
        win0[0].set_condneg(s0);
        for i in 1..8 {
            let j = 2 * i;
            win0[j - 1] = win0[i - 1].double();
            win0[j] = win0[j - 1] + win0[0];
        }
        win0[15] = win0[7].double();
        let mut win1 = [Self::NEUTRAL; 16];
        for i in 0..16 {
            win1[i] = win0[i].zeta();
            win1[i].set_condneg(s0 ^ s1);
        }

        // Recode the two half-width scalars into 26 digits each.
        let sd0 = Self::recode_u128(n0);
        let sd1 = Self::recode_u128(n1);

        // Process the two digit sequences in high-to-low order.
        *self = Self::lookup(&win0, sd0[25]);
        self.set_add(&Self::lookup(&win1, sd1[25]));
        for i in (0..25).rev() {
            self.set_xdouble(5);
            self.set_add(&Self::lookup(&win0, sd0[i]));
            self.set_add(&Self::lookup(&win1, sd1[i]));
        }
    }

    /// Lookups a point from a window of points in affine extended
    /// coordinates, with sign handling (constant-time).
    fn lookup_affine_extended(win: &[PointAffineExtended; 16], k: i8)
        -> PointAffineExtended
    {
        // Split k into its sign s (0xFFFFFFFF for negative) and
        // absolute value (f).
        let s = ((k as i32) >> 8) as u32;
        let f = ((k as u32) ^ s).wrapping_sub(s);
        let mut P = PointAffineExtended::NEUTRAL;
        for i in 0..16 {
            // win[i] contains (i+1)*P; we want to keep it if (and only if)
            // i+1 == f.
            // Values a-b and b-a both have their high bit equal to 0 only
            // if a == b.
            let j = (i as u32) + 1;
            let w = !(f.wrapping_sub(j) | j.wrapping_sub(f));
            let w = ((w as i32) >> 31) as u32;

            P.e.set_cond(&win[i].e, w);
            P.u.set_cond(&win[i].u, w);
            P.t.set_cond(&win[i].t, w);
        }

        // Negate the returned value if needed.
        P.u.set_cond(&-P.u, s);

        P
    }

    /// Sets this point by multiplying the conventional generator by the
    /// provided scalar.
    ///
    /// This operation is constant-time. It is faster than using the
    /// generic multiplication on `Self::BASE`.
    pub fn set_mulgen(&mut self, n: &Scalar) {
        // Recode the scalar into 51 signed digits.
        let sd = Self::recode_scalar(n);

        // We process four chunks in parallel. Each chunk is 13 digits,
        // except the top one which is 12 digits only.
        *self = Self::from_affine_extended(
            &Self::lookup_affine_extended(&PRECOMP_B, sd[12]));
        self.set_add_affine_extended(
            &Self::lookup_affine_extended(&PRECOMP_B65, sd[25]));
        self.set_add_affine_extended(
            &Self::lookup_affine_extended(&PRECOMP_B130, sd[38]));

        // Process the digits in high-to-low order.
        for i in (0..12).rev() {
            self.set_xdouble(5);
            self.set_add_affine_extended(
                &Self::lookup_affine_extended(&PRECOMP_B, sd[i]));
            self.set_add_affine_extended(
                &Self::lookup_affine_extended(&PRECOMP_B65, sd[i + 13]));
            self.set_add_affine_extended(
                &Self::lookup_affine_extended(&PRECOMP_B130, sd[i + 26]));
            self.set_add_affine_extended(
                &Self::lookup_affine_extended(&PRECOMP_B195, sd[i + 39]));
        }
    }

    /// Creates a point by multiplying the conventional generator by the
    /// provided scalar.
    ///
    /// This operation is constant-time.
    #[inline]
    pub fn mulgen(n: &Scalar) -> Self {
        let mut P = Self::NEUTRAL;
        P.set_mulgen(n);
        P
    }

    /// 5-bit wNAF recoding of a scalar; output is a sequence of 255
    /// digits.
    ///
    /// Non-zero digits have an odd value, between -15 and +15
    /// (inclusive). (The recoding is constant-time, but use of wNAF is
    /// inherently non-constant-time.)
    fn recode_scalar_NAF(n: &Scalar) -> [i8; 255] {
        // We use a branchless algorithm to avoid misprediction
        // penalties.
        //
        // Let x be the current (complete) integer:
        //  - If x is even, then the next digit is 0.
        //  - Otherwise, we produce a digit from the low five bits of
        //    x. If these low bits have value v (odd, 1..31 range):
        //     - If v <= 15, then the next digit is v.
        //     - Otherwise, the next digit is v - 32, and we add 32 to x.
        //    When then subtract v from x (i.e. we clear the low five bits).
        // Once the digit has been produced, we divide x by 2 and loop.
        //
        // Since r < 2^254, only 255 digits are necessary at most.

        let mut sd = [0i8; 255];
        let bb = n.encode32();
        let mut x = bb[0] as u32;
        for i in 0..255 {
            if (i & 7) == 4 && i < 252 {
                x += (bb[(i + 4) >> 3] as u32) << 4;
            }
            let m = (x & 1).wrapping_neg();  // -1 if x is odd, 0 otherwise
            let v = x & m & 31;              // low 5 bits if x odd, or 0
            let c = (v & 16) << 1;           // carry (0 or 32)
            let d = v.wrapping_sub(c);       // next digit
            sd[i] = d as i8;
            x = x.wrapping_sub(d) >> 1;
        }
        sd
    }

    /// 5-bit wNAF recoding of a nonnegative integer.
    ///
    /// 129 digits are produced (array has size 130, extra value is 0).
    /// Non-zero digits have an odd value, between -15 and +15
    /// (inclusive). (The recoding is constant-time, but use of wNAF is
    /// inherently non-constant-time.)
    fn recode_u128_NAF(n: u128) -> [i8; 130] {
        // See recode_scalar_NAF() for details.
        let mut sd = [0i8; 130];
        let mut y = n;
        for i in 0..129 {
            let x = y as u32;
            let m = (x & 1).wrapping_neg();  // -1 if x is odd, 0 otherwise
            let v = x & m & 31;              // low 5 bits if x odd, or 0
            let c = (v & 16) << 1;           // carry (0 or 32)
            sd[i] = v.wrapping_sub(c) as i8;
            y = y.wrapping_sub(v as u128).wrapping_add(c as u128) >> 1;
        }
        sd
    }

    /// Given scalars `u` and `v`, sets this point to `u*self + v*B`
    /// (with `B` being the conventional generator of the prime order
    /// subgroup).
    ///
    /// This can be used to support Schnorr signature verification.
    ///
    /// THIS FUNCTION IS NOT CONSTANT-TIME; it shall be used only with
    /// public data.
    pub fn set_mul_add_mulgen_vartime(&mut self, u: &Scalar, v: &Scalar) {
        // Split the first scalar with the endomorphism.
        let (u0, s0, u1, s1) = Self::split_mu(u);

        // Compute the window for the current point:
        //   win[i] = (2*i+1)*self    (i = 0 to 7)
        let mut win = [Self::NEUTRAL; 8];
        let Q = self.double();
        win[0] = *self;
        for i in 1..8 {
            win[i] = win[i - 1] + Q;
        }

        // Compute the 5-bit windows for the first scalar:
        //   win0[i] = (2*i+1)*sgn(k0)*self         (i = 0 to 7)
        //   win1[i] = (2*i+1)*sgn(k1)*zeta(self)   (i = 0 to 7)
        // with zeta(e, u) = (e, u*eta) for eta = sqrt(-1) (this is an
        // endomorphism on the group).
        let mut win0 = [Self::NEUTRAL; 8];
        win0[0] = *self;
        win0[0].set_condneg(s0);
        let Q = win0[0].double();
        for i in 1..8 {
            win0[i] = win0[i - 1] + Q;
        }
        let mut win1 = [Self::NEUTRAL; 8];
        for i in 0..8 {
            win1[i] = win0[i].zeta();
            win1[i].set_condneg(s0 ^ s1);
        }

        // Recode the two half-width scalars u0 and u1, and the
        // full-width scalar v, into 5-bit wNAF.
        let sd0 = Self::recode_u128_NAF(u0);
        let sd1 = Self::recode_u128_NAF(u1);
        let sd2 = Self::recode_scalar_NAF(v);

        let mut zz = true;
        let mut ndbl = 0u32;
        for i in (0..130).rev() {
            // We have one more doubling to perform.
            ndbl += 1;

            // Get next digits. If they are all zeros, then we can loop
            // immediately.
            let e0 = sd0[i];
            let e1 = sd1[i];
            let e2 = sd2[i];
            let e3 = if i < 125 { sd2[i + 130] } else { 0 };
            if ((e0 as u32) | (e1 as u32) | (e2 as u32) | (e3 as u32)) == 0 {
                continue;
            }

            // Apply accumulated doubles.
            if zz {
                *self = Self::NEUTRAL;
                zz = false;
            } else {
                self.set_xdouble(ndbl);
            }
            ndbl = 0u32;

            // Process digits.
            if e0 != 0 {
                if e0 > 0 {
                    self.set_add(&win0[e0 as usize >> 1]);
                } else {
                    self.set_sub(&win0[(-e0) as usize >> 1]);
                }
            }
            if e1 != 0 {
                if e1 > 0 {
                    self.set_add(&win1[e1 as usize >> 1]);
                } else {
                    self.set_sub(&win1[(-e1) as usize >> 1]);
                }
            }
            if e2 != 0 {
                if e2 > 0 {
                    self.set_add_affine_extended(&PRECOMP_B[e2 as usize - 1]);
                } else {
                    self.set_sub_affine_extended(&PRECOMP_B[(-e2) as usize - 1]);
                }
            }
            if e3 != 0 {
                if e3 > 0 {
                    self.set_add_affine_extended(&PRECOMP_B130[e3 as usize - 1]);
                } else {
                    self.set_sub_affine_extended(&PRECOMP_B130[(-e3) as usize - 1]);
                }
            }
        }

        if zz {
            *self = Self::NEUTRAL;
        } else {
            if ndbl > 0 {
                self.set_xdouble(ndbl);
            }
        }
    }

    /// Given scalars `u` and `v`, returns `u*self + v*B` (with `B` being
    /// the conventional generator of the prime order subgroup).
    ///
    /// This can be used to support Schnorr signature verification.
    ///
    /// THIS FUNCTION IS NOT CONSTANT-TIME; it shall be used only with
    /// public data.
    #[inline(always)]
    pub fn mul_add_mulgen_vartime(self, u: &Scalar, v: &Scalar) -> Self {
        let mut R = self;
        R.set_mul_add_mulgen_vartime(u, v);
        R
    }

    /// Given integer `u` and scalar `v`, sets this point to `u*self + v*B`
    /// (with `B` being the conventional generator of the prime order
    /// subgroup).
    ///
    /// This can be used to support Schnorr signature verification.
    ///
    /// THIS FUNCTION IS NOT CONSTANT-TIME; it shall be used only with
    /// public data.
    pub fn set_mul128_add_mulgen_vartime(&mut self, u: u128, v: &Scalar) {
        // Recode the integer and scalar in 5-bit wNAF.
        let sdu = Self::recode_u128_NAF(u);
        let sdv = Self::recode_scalar_NAF(&v);

        // Compute the window for the current point:
        //   win[i] = (2*i+1)*self    (i = 0 to 7)
        let mut win = [Self::NEUTRAL; 8];
        let Q = self.double();
        win[0] = *self;
        for i in 1..8 {
            win[i] = win[i - 1] + Q;
        }

        let mut zz = true;
        let mut ndbl = 0u32;
        for i in (0..130).rev() {
            // We have one more doubling to perform.
            ndbl += 1;

            // Get next digits. If they are all zeros, then we can loop
            // immediately.
            let e1 = sdu[i];
            let e2 = sdv[i];
            let e3 = if i < 125 { sdv[i + 130] } else { 0 };
            if ((e1 as u32) | (e2 as u32) | (e3 as u32)) == 0 {
                continue;
            }

            // Apply accumulated doubles.
            if zz {
                *self = Self::NEUTRAL;
                zz = false;
            } else {
                self.set_xdouble(ndbl);
            }
            ndbl = 0u32;

            // Process digits.
            if e1 != 0 {
                if e1 > 0 {
                    self.set_add(&win[e1 as usize >> 1]);
                } else {
                    self.set_sub(&win[(-e1) as usize >> 1]);
                }
            }
            if e2 != 0 {
                if e2 > 0 {
                    self.set_add_affine_extended(&PRECOMP_B[e2 as usize - 1]);
                } else {
                    self.set_sub_affine_extended(&PRECOMP_B[(-e2) as usize - 1]);
                }
            }
            if e3 != 0 {
                if e3 > 0 {
                    self.set_add_affine_extended(&PRECOMP_B130[e3 as usize - 1]);
                } else {
                    self.set_sub_affine_extended(&PRECOMP_B130[(-e3) as usize - 1]);
                }
            }
        }

        if zz {
            *self = Self::NEUTRAL;
        } else {
            if ndbl > 0 {
                self.set_xdouble(ndbl);
            }
        }
    }

    /// Given integer `u` and scalar `v`, returns `u*self + v*B` (with
    /// `B` being the conventional generator of the prime order subgroup).
    ///
    /// This can be used to support Schnorr signature verification.
    ///
    /// THIS FUNCTION IS NOT CONSTANT-TIME; it shall be used only with
    /// public data.
    #[inline(always)]
    pub fn mul128_add_mulgen_vartime(self, u: u128, v: &Scalar) -> Self {
        let mut R = self;
        R.set_mul128_add_mulgen_vartime(u, v);
        R
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

/// A jq255e private key.
///
/// Such a key wraps around a secret non-zero scalar. It also contains
/// a copy of the public key.
#[derive(Clone, Copy, Debug)]
pub struct PrivateKey {
    sec: Scalar,                // secret scalar
    pub public_key: PublicKey,  // public key
}

/// A jq255e public key.
///
/// It wraps around a jq255e element, but also includes a copy of the
/// encoded point. The point and its encoded version can be accessed
/// directly; if modified, then the two values MUST match.
#[derive(Clone, Copy, Debug)]
pub struct PublicKey {
    pub point: Point,
    pub encoded: [u8; 32],
}

impl PrivateKey {

    /// Generates a new private key from a cryptographically secure RNG.
    pub fn generate<T: CryptoRng + RngCore>(rng: &mut T) -> Self {
        loop {
            let mut tmp = [0u8; 32];
            rng.fill_bytes(&mut tmp);
            let sec = Scalar::decode_reduce(&tmp);
            if sec.iszero() == 0 {
                return Self::from_scalar(&sec);
            }
        }
    }

    /// Instantiates a private key from a secret scalar.
    ///
    /// If the provided scalar is zero, then a panic is triggered.
    pub fn from_scalar(sec: &Scalar) -> Self {
        assert!(sec.iszero() == 0);
        let point = Point::mulgen(&sec);
        let encoded = point.encode();
        Self { sec: *sec, public_key: PublicKey { point, encoded } }
    }

    /// Instantiates a private key by decoding it from bytes.
    ///
    /// If the source bytes do not encode a correct private key,
    /// then None is returned.
    pub fn decode(buf: &[u8]) -> Option<Self> {
        let (sec, mut ok) = Scalar::decode32(buf);
        ok &= !sec.iszero();
        if ok != 0 {
            Some(Self::from_scalar(&sec))
        } else {
            None
        }
    }

    /// Encode a private key into bytes.
    ///
    /// This encodes the private scalar into exactly 32 bytes.
    pub fn encode(self) -> [u8; 32] {
        self.sec.encode32()
    }

    /// Signs a message with this private key.
    ///
    /// The data to sign is provided as `data`. When using raw data,
    /// the `hash_name` string should be an empty string; otherwise,
    /// `data` is supposed to be a hash value computed over the message
    /// data, and `hash_name` identifies the hash function. Rules for
    /// the hash name are identical to `Point::hash_to_curve()`.
    ///
    /// This function uses a deterministic process to compute the
    /// per-signature secret scalar. Signing the same message twice
    /// with the same key yields the same signature.
    pub fn sign(self, hash_name: &str, data: &[u8]) -> [u8; 48] {
        self.sign_seeded(&[0u8; 0], hash_name, data)
    }

    /// Signs a message with this private key.
    ///
    /// The data to sign is provided as `data`. When using raw data,
    /// the `hash_name` string should be an empty string; otherwise,
    /// `data` is supposed to be a hash value computed over the message
    /// data, and `hash_name` identifies the hash function. Rules for
    /// the hash name are identical to `Point::hash_to_curve()`.
    ///
    /// This function uses a randomized process to compute the
    /// per-signature secret scalar. The provided `rng` is supposed to
    /// be cryptographically secure (it implements the `CryptoRng`
    /// trait) but signatures are still safe even if the `rng` turns out
    /// to be flawed and entirely predictable.
    pub fn sign_randomized<T: CryptoRng + RngCore>(self, rng: &mut T,
        hash_name: &str, data: &[u8]) -> [u8; 48]
    {
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        self.sign_seeded(&seed, hash_name, data)
    }

    /// Signs a message with this private key.
    ///
    /// The data to sign is provided as `data`. When using raw data,
    /// the `hash_name` string should be an empty string; otherwise,
    /// `data` is supposed to be a hash value computed over the message
    /// data, and `hash_name` identifies the hash function. Rules for
    /// the hash name are identical to `Point::hash_to_curve()`.
    ///
    /// This function uses a deterministic process to compute the
    /// per-signature secret scalar. The provided `seed` is included
    /// in that process. Having a varying seed (not necessarily secret
    /// or random) improves resistance to fault attack (where an
    /// attacker forces glitches in the hardware through physically
    /// intrusive actions, and tries to infer information on the private
    /// key from the result).
    pub fn sign_seeded(self, seed: &[u8], hash_name: &str, data: &[u8])
        -> [u8; 48]
    {
        // Make the per-signature k value. We use a derandomized process
        // which is deterministic: a BLAKE2s hash is computed over the
        // concatenation of:
        //    the private key (encoded)
        //    the public key (encoded)
        //    the length of the seed, in bytes (over 8 bytes, little-endian)
        //    the seed
        //    if data is raw:
        //        one byte of value 0x52
        //        the data
        //    else:
        //        one byte of value 0x48
        //        the hash function name
        //        one byte of value 0x00
        //        the data (supposedly, a hash value)
        // The BLAKE2s output (32 bytes) is then interpreted as an
        // integer (unsigned little-endian convention) and reduced modulo
        // the group order (i.e. turned into a scalar). This induces
        // negligible bias because the jq255e order is close enough to
        // a power of 2.
        let mut sh = Blake2s256::new();
        sh.update(&self.sec.encode32());
        sh.update(&self.public_key.encoded);
        sh.update(&(seed.len() as u64).to_le_bytes());
        sh.update(seed);
        if hash_name.len() == 0 {
            sh.update(&[0x52u8]);
        } else {
            sh.update(&[0x48u8]);
            sh.update(hash_name.as_bytes());
            sh.update(&[0x00u8]);
        }
        sh.update(data);
        let k = Scalar::decode_reduce(&sh.finalize());

        // Use k to generate the signature.
        let R = Point::mulgen(&k);
        let cb = make_challenge(&R, &self.public_key.encoded, hash_name, data);
        let s = k + self.sec * Scalar::from_u128(u128::from_le_bytes(cb));
        let mut sig = [0u8; 48];
        sig[ 0..16].copy_from_slice(&cb);
        sig[16..48].copy_from_slice(&s.encode32());
        sig
    }

    /// ECDH key exchange.
    ///
    /// Given this private key, and the provided peer public key (encoded),
    /// return the 32-byte shared key. The process fails if the `peer_pk`
    /// slice does not have length exactly 32 bytes, or does not encode
    /// a valid jq255e element, or encodes the neutral element. On success,
    /// the 32-byte key is returned along with 0xFFFFFFFFu32. On failure,
    /// a different key (unguessable by outsiders) is returned, along with
    /// 0x00000000u32.
    ///
    /// Processing is constant-time. If the `peer_pk` slice has length
    /// exactly 32 bytes, then outsiders cannot know through timing-based
    /// side-channels whether the process succeeded or failed.
    pub fn ECDH(self, peer_pk: &[u8]) -> ([u8; 32], u32) {
        // Decode peer public key.
        let mut Q = Point::NEUTRAL;
        let mut ok = Q.set_decode(peer_pk);
        ok &= !Q.isneutral();

        // Compute shared output. If the process failed, our private key
        // is used instead, so that the derived key is unknown by outsiders
        // but still appears to be deterministic relatively to the
        // received peer bytes.
        let mut shared = (self.sec * Q).encode();
        let alt = self.sec.encode32();
        let z = (!ok) as u8;
        for i in 0..32 {
            shared[i] ^= z & (shared[i] ^ alt[i]);
        }

        // We use BLAKE2s for the key derivation.
        let mut sh = Blake2s256::new();

        // If the source slice did not have length 32 bytes, then the
        // exchange necessarily fails and the memory access pattern is
        // distinguished from a success, so that we can use a separate
        // path in that case. We also do not both with ordering public
        // keys.
        if peer_pk.len() == 32 {
            // Compare the two public keys lexicographically, so that
            // we inject the "lowest" first.
            let mut cc = 0u32;
            for i in (0..32).rev() {
                let v1 = self.public_key.encoded[i] as u32;
                let v2 = peer_pk[i] as u32;
                cc = v1.wrapping_sub(v2 + cc) >> 31;
            }
            let z1 = cc.wrapping_neg() as u8;
            let z2 = !z1;
            let mut pk1 = [0u8; 32];
            let mut pk2 = [0u8; 32];
            for i in 0..32 {
                let b1 = self.public_key.encoded[i];
                let b2 = peer_pk[i];
                pk1[i] = (b1 & z1) | (b2 & z2);
                pk2[i] = (b1 & z2) | (b2 & z1);
            }
            sh.update(&pk1);
            sh.update(&pk2);
        } else {
            sh.update(&self.public_key.encoded);
            sh.update(peer_pk);
        }

        // Leading byte denotes the success (0x53) or failure (0x46).
        sh.update(&[(0x46 + (ok & 0x0D)) as u8]);
        sh.update(&shared);

        // Output key is the hash output.
        let mut key = [0u8; 32];
        key[..].copy_from_slice(&sh.finalize());
        (key, ok)
    }
}

impl PublicKey {

    /// Creates and instance from a curve point.
    ///
    /// A panic is triggered if the point is the neutral.
    pub fn from_point(point: &Point) -> Self {
        assert!(point.isneutral() == 0);
        Self { point: *point, encoded: point.encode() }
    }

    /// Decodes the provided bytes as a public key.
    ///
    /// If the source slice does not have length exactly 32 bytes,
    /// or the bytes do not encode a valid jq255e element, or the bytes
    /// encode the neutral element, then the process fails and this
    /// function returns `None`. Otherwise, the decoded public key
    /// is returned.
    pub fn decode(buf: &[u8]) -> Option<PublicKey> {
        let point = Point::decode(buf)?;
        let mut encoded = [0u8; 32];
        encoded[..].copy_from_slice(&buf[0..32]);
        Some(Self { point, encoded })
    }

    /// Encode this public key into exactly 32 bytes.
    ///
    /// This simply returns the contents of the `encoded` field.
    pub fn encode(self) -> [u8; 32] {
        self.encoded
    }

    /// Verifies a signature on a message against this public key.
    ///
    /// The message is provided as `data`, which is a hash value that
    /// was computed over the actual message data with the hash function
    /// identified by `hash_name` (see `Point::hash_to_curve()` for
    /// naming rules). If `data` contains the raw message data, to be
    /// used directly without an intermediate hashing, then `hash_name`
    /// shall be an empty string.
    ///
    /// Note: this function is not constant-time; it assumes that the
    /// public key and signature value are public data.
    pub fn verify(self, sig: &[u8], hash_name: &str, data: &[u8]) -> bool {
        if sig.len() != 48 {
            return false;
        }
        let c = u128::from_le_bytes(*<&[u8; 16]>::try_from(&sig[0..16]).unwrap());
        let (s, ok) = Scalar::decode32(&sig[16..48]);
        if ok == 0 {
            return false;
        }
        let R = (-self.point).mul128_add_mulgen_vartime(c, &s);
        let cb = make_challenge(&R, &self.encoded, hash_name, data);
        return cb[..] == sig[0..16];
    }
}

/// Computes the 16-byte "challenge" of a signature.
///
/// The per-signature point R, encoded public key, and (hashed) data
/// are provided. Use an empty string for `hash_name` if the `data`
/// is raw (unhashed). This function is used for both signature generation
/// and signature verification.
fn make_challenge(R: &Point, enc_pk: &[u8; 32], hash_name: &str, data: &[u8])
    -> [u8; 16]
{
    let mut sh = Blake2s256::new();
    sh.update(&R.encode());
    sh.update(enc_pk);
    if hash_name.len() == 0 {
        sh.update(&[0x52u8]);
    } else {
        sh.update(&[0x48u8]);
        sh.update(hash_name.as_bytes());
        sh.update(&[0x00u8]);
    }
    sh.update(data);
    let mut c = [0u8; 16];
    c[..].copy_from_slice(&sh.finalize()[0..16]);
    c
}

// ========================================================================

// We hardcode known multiples of the points B, (2^65)*B, (2^130)*B
// and (2^195)*B, with B being the conventional base point. These are
// used to speed mulgen() operations up. The points are moreover stored
// in a three-coordinate format (e, u, u^2).

/// A point in affine extended coordinates (e, u, u^2)
#[derive(Clone, Copy, Debug)]
struct PointAffineExtended {
    e: GF255e,
    u: GF255e,
    t: GF255e,
}

impl PointAffineExtended {

    const NEUTRAL: Self = Self {
        e: GF255e::MINUS_ONE,
        u: GF255e::ZERO,
        t: GF255e::ZERO,
    };
}

// Points i*B for i = 1 to 16, affine extended format
static PRECOMP_B: [PointAffineExtended; 16] = [
    // B * 1
    PointAffineExtended {
        e: GF255e::w64be(0x0000000000000000, 0x0000000000000000,
                         0x0000000000000000, 0x0000000000000003),
        u: GF255e::w64be(0x0000000000000000, 0x0000000000000000,
                         0x0000000000000000, 0x0000000000000001),
        t: GF255e::w64be(0x0000000000000000, 0x0000000000000000,
                         0x0000000000000000, 0x0000000000000001),
    },
    // B * 2
    PointAffineExtended {
        e: GF255e::w64be(0x1A1F58D0FAC687D6, 0x343EB1A1F58D0FAC,
                         0x687D6343EB1A1F58, 0xD0FAC687D6342FD1),
        u: GF255e::w64be(0x36DB6DB6DB6DB6DB, 0x6DB6DB6DB6DB6DB6,
                         0xDB6DB6DB6DB6DB6D, 0xB6DB6DB6DB6D97A3),
        t: GF255e::w64be(0x414E5E0A72F05397, 0x829CBC14E5E0A72F,
                         0x05397829CBC14E5E, 0x0A72F05397827791),
    },
    // B * 3
    PointAffineExtended {
        e: GF255e::w64be(0x6E6BA44DDB3919FD, 0x4D5065C0BA270925,
                         0xAD8273027F0E7D48, 0x26AFA803D61A9E2F),
        u: GF255e::w64be(0x12358E75D30336A0, 0xAB617909A3E20224,
                         0x6B1CEBA6066D4156, 0xC2F21347C4043E79),
        t: GF255e::w64be(0x7287BBB2DC59141C, 0x0509961D71E284EF,
                         0xC58EF652F0485A50, 0xC4FAF5442BDDB3C7),
    },
    // B * 4
    PointAffineExtended {
        e: GF255e::w64be(0x0F1371769561D944, 0x96E4BB9C95BDA924,
                         0xCBFF478EE825BA04, 0xBCE7BEB9F8390C16),
        u: GF255e::w64be(0x7A47C8BB65A29F71, 0x130DFA47C8BB65A2,
                         0x9F71130DFA47C8BB, 0x65A29F71130DB4AD),
        t: GF255e::w64be(0x2375E8119918E929, 0x03ADCBE22101F311,
                         0x853223D7F44E59F2, 0x4D0A213B4402088D),
    },
    // B * 5
    PointAffineExtended {
        e: GF255e::w64be(0x3A3E1251980D6493, 0x74304444BCDB09C0,
                         0x1BD8155773C41197, 0xD23D2C8BE875C86A),
        u: GF255e::w64be(0x185034D250F768D7, 0x5866F1F8B35FB70C,
                         0xE40F8B8BC44A0C63, 0x1F2B6B08DA5B43EE),
        t: GF255e::w64be(0x3E3560E7BB5DF4DA, 0x8982206A724B43CC,
                         0xE00C1E20C1C66FF4, 0xC91927493D361051),
    },
    // B * 6
    PointAffineExtended {
        e: GF255e::w64be(0x124FA9DF5844C804, 0x1673C5B96A6D2E2A,
                         0xC712074807471E77, 0x643AD390229AD5F2),
        u: GF255e::w64be(0x4FA6D8B6AFDDC92B, 0xA1AB0B9D98F35F00,
                         0xBB4A410D263610A7, 0x0BD0C5F1F91D6B18),
        t: GF255e::w64be(0x34F3562FDA88753E, 0xD7991971E94A460E,
                         0x76ED99CCAEE9D26F, 0x355D1614AEB11ACD),
    },
    // B * 7
    PointAffineExtended {
        e: GF255e::w64be(0x4C5C3FA382AAF7AC, 0xC36F69694BFC77E7,
                         0xD86308C5232C88D0, 0xE8D6CBC7678229FB),
        u: GF255e::w64be(0x6DD8CDFA520AED5A, 0x350914B3EE64BF7F,
                         0xB885C9D1EB4CC9E1, 0x7EB52414159EF4EA),
        t: GF255e::w64be(0x56861F4D0A217C1C, 0x0957AB7A60AC1520,
                         0x818C73930C2A0899, 0x59DAD0E634C75544),
    },
    // B * 8
    PointAffineExtended {
        e: GF255e::w64be(0x2C922B2CB2A6146C, 0x86AD97E42FD5BA49,
                         0xAD881FFB515E9113, 0x4E11C11353187B28),
        u: GF255e::w64be(0x4766315BFA2E63E5, 0xC54D8F471F778CE9,
                         0x0706B8B95DEDFC87, 0x99DA8C93EB513A8B),
        t: GF255e::w64be(0x532698BCB811270A, 0xF3C520B8FCA311FD,
                         0x89B68A9C8B0077A8, 0x4D9B8F5639729F9A),
    },
    // B * 9
    PointAffineExtended {
        e: GF255e::w64be(0x2289F3ABFA293050, 0x55B6D3E8852C1B0B,
                         0x675E5BCC38AA1784, 0x6FB66DF6B52FBCC4),
        u: GF255e::w64be(0x6957FDFF940E4159, 0x498C7D8B01F68C40,
                         0x27E9084D132CCAC1, 0xA84A27A9D0A08E61),
        t: GF255e::w64be(0x431DA1A672CB2D3C, 0x56244B8A32B42796,
                         0x76CA668F88C812F9, 0x8D2F2DE6815F2EFF),
    },
    // B * 10
    PointAffineExtended {
        e: GF255e::w64be(0x30290CF961B06E3A, 0xAF5539730762C505,
                         0x803FC1C6AAD5CD46, 0x8EB44683ACC048BE),
        u: GF255e::w64be(0x7B37113C916F803C, 0x4AA37581A9B6AD5E,
                         0x55838146CC140A37, 0x3AA366BBB889903E),
        t: GF255e::w64be(0x252DC97C189ECFD9, 0x9EDDA370E828C438,
                         0xC70EAC518AE5C163, 0xD912EBC4E1C6283E),
    },
    // B * 11
    PointAffineExtended {
        e: GF255e::w64be(0x467C84CA2424A548, 0x6F385F29F643AF23,
                         0x09DF0EB0A3919A65, 0x38A7E99599D93A3B),
        u: GF255e::w64be(0x158521078CD1F209, 0x8C133221E772B327,
                         0x65CF6B9CAB741725, 0xA9A8911D864E7F82),
        t: GF255e::w64be(0x5D4A07E9BC1F036B, 0xC3FC1F026C5406EA,
                         0xFAE4DC5553E938EB, 0x41583C9A8F92D685),
    },
    // B * 12
    PointAffineExtended {
        e: GF255e::w64be(0x66FF84AF5A9F7719, 0x4B3B87C8238DA4B1,
                         0x06E6D0CF8DC7807E, 0xA49A13F883BF7B93),
        u: GF255e::w64be(0x67E65CF7E1787343, 0x53EF35932E1297EB,
                         0xFB902D6A3EF5D5E0, 0xE4C1C725087640AA),
        t: GF255e::w64be(0x2213A3D93959DA85, 0x4742F3CB3DEB52CB,
                         0xAE5BB5318E506208, 0x4203ACE2FF9309B4),
    },
    // B * 13
    PointAffineExtended {
        e: GF255e::w64be(0x6E7036BE9D4C885B, 0x3E7BB13DFFB6AD2D,
                         0x22587604B5D2A716, 0xA99622782ED04723),
        u: GF255e::w64be(0x17F96D76306A3580, 0x913200FCCD1396D8,
                         0x67D0394FF2BFCB98, 0x90F8839881061965),
        t: GF255e::w64be(0x0E2364672DB22F61, 0x028EBB02AB0AE384,
                         0xE02396E9E43361F5, 0xF1F0EC984099DC93),
    },
    // B * 14
    PointAffineExtended {
        e: GF255e::w64be(0x74C367ABF8A08989, 0x069708CC706DD30D,
                         0xA59973DD5CB6D116, 0xC6C60BB30F2521F0),
        u: GF255e::w64be(0x42C8034547A6A04A, 0x7C6DAA970635E1C0,
                         0xA016A6890D77E4E6, 0x05B49673D2AC4172),
        t: GF255e::w64be(0x0C4AC28DE6AF9B7D, 0xE13C4A7639E5E234,
                         0x41B211E505C5A659, 0x0266FAA875DEF4DC),
    },
    // B * 15
    PointAffineExtended {
        e: GF255e::w64be(0x1110D92782C497CD, 0x257B5FE5EE7E01D9,
                         0x626226C09EA21055, 0x543AA085E86224A7),
        u: GF255e::w64be(0x2F4CA2B6265D7456, 0x4AD1FA649FAE1F07,
                         0x705D12571BF74984, 0x7FFA4AF719120727),
        t: GF255e::w64be(0x0CCF7FF00D563132, 0xCC7CCE327009957D,
                         0x54BD0CFFC1B29AC7, 0xB111F1B5F7C6525E),
    },
    // B * 16
    PointAffineExtended {
        e: GF255e::w64be(0x676824C9A296F053, 0x03D9F770D7F5F415,
                         0xF8FAE043DB5120DD, 0x97DA44A024F31B2E),
        u: GF255e::w64be(0x251A19311A6DB76E, 0x706D8A1F41E90ED8,
                         0x15064C9132683177, 0x2D808316E1227049),
        t: GF255e::w64be(0x2C8D5EC4B15D04AE, 0x7C16D8FF7BF95128,
                         0x0DB49CC10C6EE0A8, 0x95191DCA9E05F91A),
    },
];

// Points i*(2^65)*B for i = 1 to 16, affine extended format
static PRECOMP_B65: [PointAffineExtended; 16] = [
    // (2^65)*B * 1
    PointAffineExtended {
        e: GF255e::w64be(0x6CE0B77B634A53B5, 0xAB3BAC1AB41DD08C,
                         0x4629856EF94734AB, 0x886FF6278E08211F),
        u: GF255e::w64be(0x1B313D36A8A7626E, 0x395D2181E50A4384,
                         0x415893549DE223FF, 0xAD5F8FBFC596FA71),
        t: GF255e::w64be(0x6E78594A21A5AAD2, 0x48E8393CFF13EC0E,
                         0x222B7C0CD599D9CB, 0x17E71DD1EB9D832B),
    },
    // (2^65)*B * 2
    PointAffineExtended {
        e: GF255e::w64be(0x7521310F8DD93292, 0x8AA629F4CE6C2582,
                         0xE4584C1E3417B3E6, 0x111922C72716A209),
        u: GF255e::w64be(0x0BE56F359985D602, 0x5CD27D909976BFE7,
                         0x4CA4C049A5E65CF4, 0x2B7C2F71889E3F33),
        t: GF255e::w64be(0x40534B306985620E, 0x6C805A5DABAECD17,
                         0xC9233D8219ECC70E, 0xC699D8CD1BE2027B),
    },
    // (2^65)*B * 3
    PointAffineExtended {
        e: GF255e::w64be(0x390940AD8D0885BF, 0x7B11CD4B6C9CC38D,
                         0x5A40972FCDA92791, 0x33F6746ED6A45A0E),
        u: GF255e::w64be(0x098D3AB90B09C2B4, 0x0519A68AAB4295EB,
                         0xAF41508342D7801D, 0x4A504A5DED61CB7F),
        t: GF255e::w64be(0x41A590927B5F6FD7, 0x0696A39545C181DE,
                         0x92D534263FFE78C9, 0xA06F9DAD59A456C6),
    },
    // (2^65)*B * 4
    PointAffineExtended {
        e: GF255e::w64be(0x10D60536FA934AC1, 0xBD9A0206C0DF4A3B,
                         0xEADD9BBED6E7A570, 0x6C4E581C32A76153),
        u: GF255e::w64be(0x3540EB32AA6E2C23, 0x3B845EF3AFAD3191,
                         0x5A0FCD3AE1067F08, 0xD98BB32E81B4D89F),
        t: GF255e::w64be(0x6470D11011381CFF, 0x893FFC0CA3CB3C98,
                         0x4563C7AD47CE6A39, 0x1D5909B1A76336E1),
    },
    // (2^65)*B * 5
    PointAffineExtended {
        e: GF255e::w64be(0x742236A2B9985165, 0x9AE9B0736E87C51E,
                         0x74427A8818227B9E, 0x648443E48591C275),
        u: GF255e::w64be(0x4B36B3BE374E74CB, 0x9DD36D925FB45810,
                         0x695C5863633AC0BB, 0xDB2EB15DD80EB7AB),
        t: GF255e::w64be(0x7563AAD0685E3293, 0x52217A6F3280BAD3,
                         0xBCF6E190CBC99BC3, 0x3B02030497F19A1B),
    },
    // (2^65)*B * 6
    PointAffineExtended {
        e: GF255e::w64be(0x637BA56A42C4587D, 0x9318569D6BC1A5D5,
                         0x748A8C77D6669E2A, 0xD8AD585D0C14DC83),
        u: GF255e::w64be(0x143CF8A090ACB085, 0x0C0776C9F6ED32DE,
                         0x763067186E58108F, 0x2473D6DB1425C001),
        t: GF255e::w64be(0x2C301B6BCC1F09C1, 0x3A3345F98418DD26,
                         0xDE3B04BA9040AF87, 0x73CDF9E505579C38),
    },
    // (2^65)*B * 7
    PointAffineExtended {
        e: GF255e::w64be(0x5C780EAA96A1BB91, 0x1094A46B6F56FA43,
                         0x0E16666D8273B21D, 0xE14526AA9D051B56),
        u: GF255e::w64be(0x7EE7B8C1D61C83DD, 0xBAC06821F2BF788C,
                         0x33088969E8FB612B, 0xE95E218127D26209),
        t: GF255e::w64be(0x72AB912B16C33383, 0x617E1C89A1A25303,
                         0x624FFDB35FA59240, 0x0474E6971F3502A9),
    },
    // (2^65)*B * 8
    PointAffineExtended {
        e: GF255e::w64be(0x6B3865BE473732FF, 0xE95C514D17DD5343,
                         0xD6C684EDF26076D9, 0xE800A094D67CC3ED),
        u: GF255e::w64be(0x235872299B5B142E, 0x34DFEFB2A6C39FFC,
                         0x3260DA08A5D337CB, 0x4C033B2301944CB8),
        t: GF255e::w64be(0x2961CB37E34C35A1, 0x7B1EF0DEBBFAA0EE,
                         0x08AA04266FC23C15, 0x439324FCD04090FB),
    },
    // (2^65)*B * 9
    PointAffineExtended {
        e: GF255e::w64be(0x32138C48916B15C7, 0xF8815AEC9C0105B4,
                         0xDD56EA4894B8F6C1, 0xE7847250A19EF549),
        u: GF255e::w64be(0x07DF0CC5D7C9E88D, 0xADD356D1CA14E352,
                         0x53193A718D75331F, 0x6742B67C6086DF92),
        t: GF255e::w64be(0x713B783B3766078D, 0x0D31937E03003B68,
                         0x643500CDE6FADF0D, 0x5C9C41C3B8594D9B),
    },
    // (2^65)*B * 10
    PointAffineExtended {
        e: GF255e::w64be(0x5C9329786A90B198, 0x83A3CE92BEFEC84E,
                         0x4FD5981D24E25FBE, 0x25F90BCEB299A5D0),
        u: GF255e::w64be(0x79E82E420C17C457, 0x3F6064CC7E287D25,
                         0x476A1D01EBBFE2BB, 0x7982C54051C4DF08),
        t: GF255e::w64be(0x1813C6ED70DAFA38, 0x27984BF87B6F0AD1,
                         0x73165189E2EF99F8, 0x8CE01EA2E8B54114),
    },
    // (2^65)*B * 11
    PointAffineExtended {
        e: GF255e::w64be(0x4515BBA1AEAFD937, 0x8D0D667D16879C64,
                         0x383BFB84CDDA791E, 0x7CAD33020884167D),
        u: GF255e::w64be(0x3CC8BC3168B5A376, 0xB97069DD65593890,
                         0x854CBABF2D02F041, 0x00FF128613403E58),
        t: GF255e::w64be(0x058DA5FED462C39D, 0x206DA25D9FE2783E,
                         0xF8F2F98E5BEF4381, 0xC303EBB8F65D98BA),
    },
    // (2^65)*B * 12
    PointAffineExtended {
        e: GF255e::w64be(0x3E658213D95C2805, 0x68592F914D7E5167,
                         0x4F20F40E6D616E5E, 0x7118E3577FDDE52D),
        u: GF255e::w64be(0x51CBB2B65B751D79, 0x23447D42F7311FA5,
                         0xA50B5985163B51DB, 0x3688782C6588D284),
        t: GF255e::w64be(0x60570F440CB85A8A, 0x15EB451E43025B6A,
                         0xDC1EF290D77A0305, 0x877FB541EB5BEC17),
    },
    // (2^65)*B * 13
    PointAffineExtended {
        e: GF255e::w64be(0x40255DC81F644B88, 0x3CA545A989E728B3,
                         0x0DAC5E2EBE8D1017, 0xBF9758CD45295FBE),
        u: GF255e::w64be(0x052051EDAFC87131, 0xAA36C89F8D5B502A,
                         0xFDEE555C9009703F, 0x6AF277E67A9FAA68),
        t: GF255e::w64be(0x50C64067319D2237, 0x0F6D233DC4F2C4BA,
                         0x9B85D45C9AA3A8E1, 0xDED77C4486AAB04C),
    },
    // (2^65)*B * 14
    PointAffineExtended {
        e: GF255e::w64be(0x631EFE8EF7F8E0F4, 0x8E7CF5FB0130A4FF,
                         0x52126D4376848E4E, 0x13D6BB104910DE30),
        u: GF255e::w64be(0x7182E92FA95620EB, 0x214282B022098B86,
                         0xF694DF6ED62219A0, 0x86A494FC08EB56DD),
        t: GF255e::w64be(0x136A3753275A04FE, 0x2F8383D036FFDEC3,
                         0x6E06A2EFFBF79614, 0x8A96C7C2F4D51B6A),
    },
    // (2^65)*B * 15
    PointAffineExtended {
        e: GF255e::w64be(0x09C0518DA671868A, 0xBF13A51C8C3DB802,
                         0x533027C62A4254F0, 0x714774712F78D8DC),
        u: GF255e::w64be(0x406E88577206C6C4, 0xEDF6131ABB08D620,
                         0xAE53B9A6D617E037, 0xD25E4076E71417FD),
        t: GF255e::w64be(0x7A35DA9A106BD4A6, 0x69CB583FD3DDA8C2,
                         0x5FD6950193780470, 0xA7B19AB7E5261DBE),
    },
    // (2^65)*B * 16
    PointAffineExtended {
        e: GF255e::w64be(0x0DA857B86203938C, 0xA8D3D5EBA8F300F7,
                         0x16B77AD03CBC9CE8, 0xB7E60EBD3E152D96),
        u: GF255e::w64be(0x7B916A823C090D35, 0x6CB90D1DDFBA601C,
                         0x1ADF4CFE82DAF6AB, 0x7D37BBB6D0781C73),
        t: GF255e::w64be(0x78BE6830E382EC23, 0xAA62937A623721E5,
                         0x75D85076E298321B, 0xFE1CAB3D1875E47C),
    },
];

// Points i*(2^130)*B for i = 1 to 16, affine extended format
static PRECOMP_B130: [PointAffineExtended; 16] = [
    // (2^130)*B * 1
    PointAffineExtended {
        e: GF255e::w64be(0x06CB9DE70E47A525, 0x9FBB8D6BBC2E0657,
                         0x04DA93AE9BE27159, 0x869B548FDE63606E),
        u: GF255e::w64be(0x1E3A7692F3E02AB1, 0x2B998E19EE40437C,
                         0xEF9A2E5ACD520CD9, 0x6A6F8767CAF58BCB),
        t: GF255e::w64be(0x6383719F46FAC0AE, 0x2FCDD9E72158E8D4,
                         0xC1FEC4C782CB44C6, 0x12DCF83EC50BD952),
    },
    // (2^130)*B * 2
    PointAffineExtended {
        e: GF255e::w64be(0x69E07759E1492E65, 0xE12975DE1B6AA715,
                         0x137941C5ADF1CAD3, 0xB08E5A163D54F276),
        u: GF255e::w64be(0x1DE0359F8936CEEA, 0xAC5EBA85AFFFBCEB,
                         0x64C33BBEA0416456, 0x97643EA7C28259E8),
        t: GF255e::w64be(0x3A18C76587D69006, 0xF83CBCACE081587F,
                         0xC8A5F632AB9427B0, 0xF32ACE061879EA59),
    },
    // (2^130)*B * 3
    PointAffineExtended {
        e: GF255e::w64be(0x7F9D9B87A431B60A, 0x7A63055985C3FC53,
                         0x9225D9C973152188, 0xA69544938C9DC498),
        u: GF255e::w64be(0x62BAF3EF3C842009, 0x4FA122A357313D72,
                         0xC29D0227105E6338, 0x4A8F6858185A63C2),
        t: GF255e::w64be(0x54C34BDF628B654A, 0xC5FE42DEAF33DE83,
                         0x799295C376EF453F, 0x183908457B30E0BC),
    },
    // (2^130)*B * 4
    PointAffineExtended {
        e: GF255e::w64be(0x180B366A5F951D11, 0x6E7804AC92C3BE8D,
                         0x8777F23BA9BA461C, 0x607C64DAEC32F903),
        u: GF255e::w64be(0x37EDC360CFBBEDE2, 0xDCCCD3E1EF458EDC,
                         0xEF697901DA783099, 0x17625F88FFC35397),
        t: GF255e::w64be(0x515891835D836913, 0x62C64632720B072F,
                         0x3E9CC11638D69625, 0x5F7FC0E7A0B94FB0),
    },
    // (2^130)*B * 5
    PointAffineExtended {
        e: GF255e::w64be(0x552022F75A7FF671, 0xA458CD165446291A,
                         0x75DE8E82AE7AD735, 0xB5DA3AAFCEA3BDDC),
        u: GF255e::w64be(0x282E209409ADD692, 0x6DDAC4AB9F7AE17C,
                         0xEB86F24ADFCADBF5, 0x055D6CCDE9EC95DF),
        t: GF255e::w64be(0x775F596632FB79E4, 0x541CE0C53EC38691,
                         0x5AF736BE387EA15D, 0x60BA3A7F6B02F394),
    },
    // (2^130)*B * 6
    PointAffineExtended {
        e: GF255e::w64be(0x5EF32CC1B5A5952D, 0x66ECF43864D273CD,
                         0xA45999A2258AEB93, 0x7C2346E4D222B565),
        u: GF255e::w64be(0x7E4F46D5131E195C, 0xF406AD1FFA1ACD6B,
                         0xB6FA24A8892C93FF, 0xBA9DA77EDCAD7528),
        t: GF255e::w64be(0x3306A93BCA2FEDC4, 0xA68B3F0189B1BA23,
                         0xEB5AD1F6B9C3F150, 0x85A61A2DEA698452),
    },
    // (2^130)*B * 7
    PointAffineExtended {
        e: GF255e::w64be(0x4830813F189FDCFE, 0x5E05B4C4DB704F5D,
                         0xE9BDF6AD5FC76584, 0x70FD1203988BD912),
        u: GF255e::w64be(0x372239482AEC172D, 0x6FB5672D05CA0B5B,
                         0xC1293137B7F95D67, 0x9E343B6FEBEF1413),
        t: GF255e::w64be(0x35266C9ACED8D90B, 0x21A50BBDCF12747E,
                         0xF796A60E89BAA4F4, 0xEDC0FD9628A8DD36),
    },
    // (2^130)*B * 8
    PointAffineExtended {
        e: GF255e::w64be(0x303B08CF4D486C20, 0x5D86EBC524ED9795,
                         0x27214239C8731082, 0x68EE4317D1F7E9CE),
        u: GF255e::w64be(0x4E1CF6CCC48289F4, 0x7BFEE5167F928B59,
                         0x0F110F3BEDDC580B, 0x21232B19EA446499),
        t: GF255e::w64be(0x1DFD019EB5360B34, 0x94AAE19ECEC7737D,
                         0x0834BAF860B519A9, 0x1768F10E04969A8C),
    },
    // (2^130)*B * 9
    PointAffineExtended {
        e: GF255e::w64be(0x4262554CF57EA2FC, 0x2830FF4E2FBB9A1B,
                         0x9259F250CE8DDF8F, 0xFA3E5E1F454A9789),
        u: GF255e::w64be(0x6F671B022C83BC57, 0x24120A75934E696A,
                         0x06134380E791A9D0, 0xFFFA0532A28FF940),
        t: GF255e::w64be(0x03BBDFC635166005, 0x4CA31E3B83E29BB6,
                         0xBB03E3FDF6E447C2, 0x0BE5ED639180368F),
    },
    // (2^130)*B * 10
    PointAffineExtended {
        e: GF255e::w64be(0x776F6C1C18E51DF5, 0x2970E4C0A4D753B1,
                         0xBC5D29A73202BCE6, 0xE1B24ECCE93AA158),
        u: GF255e::w64be(0x64ECAAC0EBA1A511, 0x108D5A1C5EC7534A,
                         0x3EC320448D80BCA7, 0x0EE1B5003B071E37),
        t: GF255e::w64be(0x4BC50BF7A3AC991A, 0x8C125FB0A13E6050,
                         0xD00F9D216D74A32A, 0x10E500131535C514),
    },
    // (2^130)*B * 11
    PointAffineExtended {
        e: GF255e::w64be(0x3622E7BF2A3899E8, 0x2C33E16E5D45B50A,
                         0x31DC4E881F4F95D8, 0x10DC6E263EB40DBD),
        u: GF255e::w64be(0x029CAFA5599B372A, 0xD51C733D4564D8F4,
                         0x0BC92225712BA4D4, 0x6CC3477AD4AB70A1),
        t: GF255e::w64be(0x30F0D235E547B2AD, 0x396102BA67CD8FA6,
                         0x84FD6E87816D4B31, 0x3D63011A6518ADF3),
    },
    // (2^130)*B * 12
    PointAffineExtended {
        e: GF255e::w64be(0x70F7A39B8561373F, 0x3F2A874D3EE0C99A,
                         0xB3E38AE4779DEFF8, 0x5424663282E719F6),
        u: GF255e::w64be(0x6E81F9FA7E661D8D, 0x41F420B5A91ED717,
                         0x960067A5C064493E, 0xCB9E1D5CB7854025),
        t: GF255e::w64be(0x797141D1E37DDC45, 0x4E40C5DCD5EB0507,
                         0xFA5B99758B260B44, 0x96737B3BD1B3FF89),
    },
    // (2^130)*B * 13
    PointAffineExtended {
        e: GF255e::w64be(0x4489AC6538239FA8, 0x87620DD6665947AC,
                         0xC52014D6E24C4D0F, 0x3A2AA626DE84F742),
        u: GF255e::w64be(0x653EE371D2801E6A, 0x59F3D4C5BB3DFDCC,
                         0xF94983893D4AABD6, 0x1A722773E489DDB8),
        t: GF255e::w64be(0x6E8DE13723DFA5BC, 0x3AF72BF16ED6198C,
                         0x336E96DD99975786, 0xA14B344A108032A6),
    },
    // (2^130)*B * 14
    PointAffineExtended {
        e: GF255e::w64be(0x0698AEC8D7B177EE, 0xE23DA12DBE2A729B,
                         0x30C254D3EE996605, 0x5BDDDC0121EA7956),
        u: GF255e::w64be(0x50E1BEE9AC402680, 0x216A1928595E868E,
                         0x96FABF7744D22EC2, 0x460E164D09693F50),
        t: GF255e::w64be(0x52BC8B51287FBDD0, 0x73999AF999779B03,
                         0x3076D3BE8227BB81, 0x2EA3B4425FE17CBC),
    },
    // (2^130)*B * 15
    PointAffineExtended {
        e: GF255e::w64be(0x67C8ED8C09C7A4DA, 0xC7562833544AC635,
                         0x7269B6542711C087, 0x4DE169D233632097),
        u: GF255e::w64be(0x5CE94782B63D2983, 0x2323BA05744FB271,
                         0x5486463AAA3D41CD, 0x442C914C64C6EE61),
        t: GF255e::w64be(0x79B75334C85A090C, 0x4B8F046B40C3632A,
                         0xFE575987C8449D15, 0x5E85E0F841CFEA05),
    },
    // (2^130)*B * 16
    PointAffineExtended {
        e: GF255e::w64be(0x5F00BEB6EDB8A088, 0xAFF58E0E6F53165A,
                         0xB956B5A6669E52E3, 0xA1638BEC45B50B50),
        u: GF255e::w64be(0x0F2ED8418B17674E, 0x9CAE33A6B5F9B94C,
                         0x68337F979386B815, 0x20DDE2D9560BD063),
        t: GF255e::w64be(0x2EED8F30CF10FA1C, 0xBB88653D342DE052,
                         0x3721E53E5901899E, 0x42082E618690FF50),
    },
];

// Points i*(2^195)*B for i = 1 to 16, affine extended format
static PRECOMP_B195: [PointAffineExtended; 16] = [
    // (2^195)*B * 1
    PointAffineExtended {
        e: GF255e::w64be(0x26FE2B2D2FF7C515, 0x4F9CCA6483A60AC4,
                         0x05A13379CA5B1805, 0x259381D52FD3D48B),
        u: GF255e::w64be(0x7A2058682117A352, 0x5A78F8FDAFE0F2B2,
                         0x3FACF03027F2FE05, 0xC6C4EA52864B8022),
        t: GF255e::w64be(0x6CF49CB73C1C85C0, 0x759EDBD5AA299C4A,
                         0xD2369B352A0FAC70, 0x827479FBF869915D),
    },
    // (2^195)*B * 2
    PointAffineExtended {
        e: GF255e::w64be(0x26E11FA33F986A71, 0xFEFB94253EE54433,
                         0x2F4C099F8F9811D4, 0xBEA01D24F6BA6D06),
        u: GF255e::w64be(0x597FE7EC4E366C38, 0xCBC1DD8A3179B032,
                         0xB043C24E23C52866, 0xE7A9C455FDCCE69F),
        t: GF255e::w64be(0x5D82A3622AECDB76, 0x31EC395A62DF9B38,
                         0xB52F9C48D79CEDB6, 0xF41D8A2928BB5A33),
    },
    // (2^195)*B * 3
    PointAffineExtended {
        e: GF255e::w64be(0x7A97A73E152AC3D0, 0x5D9FD304B83A88A9,
                         0xE3C20E9FA8811C74, 0x9902C1E090F2D551),
        u: GF255e::w64be(0x5D529C2B2E3222F2, 0xB0C0AEB839C2A9BB,
                         0xDAC867CE0228990B, 0xEBF1C192782AD7E7),
        t: GF255e::w64be(0x25F396E870747870, 0x4CC66CF8E7017825,
                         0x8692B8050BB45ACF, 0xDF8E63928F1AE0C2),
    },
    // (2^195)*B * 4
    PointAffineExtended {
        e: GF255e::w64be(0x68BEB6AD6278C4C5, 0x23A615DE55EB285B,
                         0x532E7F73C34493C0, 0x825E9E80D9017500),
        u: GF255e::w64be(0x73E9DA1D7C583AB6, 0x880F0F9CA2B23DE9,
                         0xD5179DCB398FE9E2, 0xE5AD1FDA050CE7CF),
        t: GF255e::w64be(0x2F2A10845751514B, 0x55CA5A73E2FDED3F,
                         0x56A385AC631A3736, 0x1BED8C4A161AD03A),
    },
    // (2^195)*B * 5
    PointAffineExtended {
        e: GF255e::w64be(0x1A980F7E359D5D64, 0xDD94B1823583A50F,
                         0x39A98FB9520B23D9, 0xEF721415A10674B8),
        u: GF255e::w64be(0x3FB05C6B656FCDE7, 0x3428954D0B0B6412,
                         0x2B41B323457375B0, 0x81533FD0CED00FE0),
        t: GF255e::w64be(0x1FA8E20753FBE8B0, 0x0156D1B1D6CF146F,
                         0x3EB933A63E59FA2B, 0xD6A2CFBECD1FF35F),
    },
    // (2^195)*B * 6
    PointAffineExtended {
        e: GF255e::w64be(0x53505DC8D146C39F, 0xD1328954C1969616,
                         0xCE6BBA92A647E0CA, 0xC32013800A7FA85F),
        u: GF255e::w64be(0x1EEE7F4380019FB8, 0xC70C1F018B95875D,
                         0xA6B25F5370FF8BED, 0xABECE8DEAA4DEFF3),
        t: GF255e::w64be(0x35C68ADE3CAC8CA8, 0x5D617DE816798EB4,
                         0x93B664F242B13EF8, 0xCC3FC741CC1E0562),
    },
    // (2^195)*B * 7
    PointAffineExtended {
        e: GF255e::w64be(0x203E41FE3594BCCC, 0xC95A3B2FA7918F82,
                         0x023E52D8EE207D6F, 0x293F3FD57745E14A),
        u: GF255e::w64be(0x1C54D437C147EB47, 0xDF748C1856292D78,
                         0x8AF5B5BD4C9ADB58, 0xC2513D53CE5A6CD2),
        t: GF255e::w64be(0x791ECB5E7F925E67, 0xB8E346F880AA5525,
                         0xB56CC897BA7BA956, 0x02C3EA61A5ABCB56),
    },
    // (2^195)*B * 8
    PointAffineExtended {
        e: GF255e::w64be(0x5A50303BB7FBF10D, 0xA2654498E4A81FA1,
                         0x9EAB45BEF1F6D3FE, 0x9E436CB767A86CB5),
        u: GF255e::w64be(0x47EC89341C754A72, 0x23187D9C49399AB4,
                         0x76396E28425429D3, 0x961441BFA4853698),
        t: GF255e::w64be(0x00D090A5B2E0E163, 0x47DBD9D756A70427,
                         0x4E7E59858CDEC91B, 0xB5B36F776AF04917),
    },
    // (2^195)*B * 9
    PointAffineExtended {
        e: GF255e::w64be(0x4E1EFB691EF8D7C0, 0x3ADC11E0F4D52DD6,
                         0xD1492FB1E8EDE015, 0x1390F65E3B6C3E84),
        u: GF255e::w64be(0x0C4E4B67E97A3F02, 0xB055E943E160FF52,
                         0x899CA30F8A6C1157, 0x790DDCB656DA5EBB),
        t: GF255e::w64be(0x5C4551BBBCA04267, 0x0E7DEA665667BD29,
                         0x90F6162E6D8BF1FD, 0xE665E42E421C783F),
    },
    // (2^195)*B * 10
    PointAffineExtended {
        e: GF255e::w64be(0x0BBAE0E5BEFBB5CF, 0x433AC8B15317BE46,
                         0x82E93F7DB218D2B4, 0x28CA2A1B76E757EA),
        u: GF255e::w64be(0x44B8942C5DF07889, 0x7C80807D104B0670,
                         0xEB725D50999A4399, 0x7CBAD6A204D1F6B9),
        t: GF255e::w64be(0x3B052B97144BDDCC, 0x4584662C9819C28B,
                         0x4555C3E786404CA2, 0x617C6A396D315D12),
    },
    // (2^195)*B * 11
    PointAffineExtended {
        e: GF255e::w64be(0x2CCB2CF7E40BF4D0, 0xD28F560EA8A29B73,
                         0x200EE9E2631EC7C4, 0x0D101C917FFDE613),
        u: GF255e::w64be(0x7DD81F0AD475BB65, 0xEB5423DB7543C3F6,
                         0xB6A214969A98317F, 0x30FE96716E7DF796),
        t: GF255e::w64be(0x69561B43851B3032, 0x76EEE28489A2673B,
                         0x4ED760314652F82F, 0x6E9D90B85FC133E0),
    },
    // (2^195)*B * 12
    PointAffineExtended {
        e: GF255e::w64be(0x2DFA0BA127D42687, 0xB3A61F723EE958E7,
                         0xA66737D8513DFDB2, 0x50C89E4AB1499911),
        u: GF255e::w64be(0x60406418E0A8E484, 0xF6647FFFA6239BFD,
                         0x0620628690F070A8, 0x041881DC4BB15593),
        t: GF255e::w64be(0x3C01E53D24508710, 0x5F7FF4B10823ABA8,
                         0xCC5BCAEE53A27045, 0xB186E710CF384958),
    },
    // (2^195)*B * 13
    PointAffineExtended {
        e: GF255e::w64be(0x0B0051552C1C58B5, 0x16210166F7E6E8DA,
                         0x039A3283169BCBD8, 0xA54981754921B43D),
        u: GF255e::w64be(0x03C8B0610E5DE932, 0xEDC81102A9A286D9,
                         0x80A9A62960313C1E, 0x02A5C58CE250DF40),
        t: GF255e::w64be(0x5C2FDF1BE72A0992, 0x220189A7901D370E,
                         0x7F61E2E190853286, 0x9B656127329A1F5C),
    },
    // (2^195)*B * 14
    PointAffineExtended {
        e: GF255e::w64be(0x6021348E9D2BB8D8, 0xB7E063EA06740CDF,
                         0x117438E7D83A2644, 0x1725A4D445249491),
        u: GF255e::w64be(0x63F775224E36165F, 0x97D9FFD6286A6BD1,
                         0x3498D65BDB7611FC, 0x2359C265188EE74D),
        t: GF255e::w64be(0x534AFBC1B782F441, 0x6492A9AFA98D71BD,
                         0xA31076307FE06CED, 0xD5B3AA1D5B583047),
    },
    // (2^195)*B * 15
    PointAffineExtended {
        e: GF255e::w64be(0x6C435D4A27D08486, 0xD35D4667B3E6947A,
                         0x07778715A12C63DB, 0x66C6224B1937FD6E),
        u: GF255e::w64be(0x5C69FCC38DA29629, 0xDD4D47FA083CC9BF,
                         0x58F89E267B42BEA3, 0x14BD04EFD34FC573),
        t: GF255e::w64be(0x0B9C09556A326820, 0x5A90BF07D1C9E81B,
                         0x4F944F4CDD9551F4, 0x98A6F2B6623C4605),
    },
    // (2^195)*B * 16
    PointAffineExtended {
        e: GF255e::w64be(0x4DA95AF12CBA888E, 0x6217606AC5C899F0,
                         0xB4BD88C5AC6C7FE8, 0xF4448E39173AD989),
        u: GF255e::w64be(0x162581D9B675A4E1, 0xDA636B907FED771B,
                         0x3E6A069F39FCEFCB, 0x805E028481B3D10D),
        t: GF255e::w64be(0x02FA9D8FB083E563, 0xEEAA462F252FD554,
                         0x44FDFE7C39061B8D, 0x52668A8902F702CE),
    },
];

// ========================================================================

#[cfg(test)]
mod tests {

    use super::{Point, Scalar, PrivateKey, PublicKey};
    use sha2::{Sha256, Digest};
    use blake2::{Blake2s256};
    use crate::field::GF255e;

    /* unused

    fn print_gf(name: &str, x: GF255e) {
        print!("{} = 0x", name);
        let bb = x.encode32();
        for i in (0..32).rev() {
            print!("{:02X}", bb[i]);
        }
        println!();
    }

    fn print(name: &str, P: Point) {
        println!("{}:", name);
        print_gf("  E", P.E);
        print_gf("  Z", P.Z);
        print_gf("  U", P.U);
        print_gf("  T", P.T);
    }
    */

    static KAT_DECODE_OK: [&str; 21] = [
"0000000000000000000000000000000000000000000000000000000000000000",
"8d94d29406f613c642019d7dbda7121510be572d323181af1a0914e8d1080903",
"23a121249d7a31475a8603bfab1ec71d94fd0dd033c4a6111937ce8649474b51",
"a040d74b0374447b31b3c70c815c380c5ab9f9efaca7ee75a51a2a7fbfabc615",
"892180c7ba29f4980fbd56219704b6eeb161053867294aed2cdb520301721a61",
"8eabe2a42cca04c02a009faab48eb9255c9594b6c9467b03479c00c22bcc423f",
"ac3bd3c64e2e441b19b7f35e450c6c68346e76bcb2a8b137a7de9e0185bd9603",
"e53a91660b5e8e8cbdf8603af5b6e4dca4b70d400739c292cc85ff3f4ba2e875",
"8d33a31606aec0eae122d81896982c8866a1a9f1ebd750d944e83cac3f2f6041",
"39415ca52c40d1233087280b99a80bf946156cefc5a0fcfc4cd6f640dfbcfc45",
"4a7bb7961c063e7e1aad732efc6f2a4ac990318b0a9b406ae3c20c8b64bef239",
"36530914a8e71d6092ef0414e7adfc2485870dd3b2c967b245a0de3f11f84941",
"36170b60f0fbfda49917de84b99c3f3d901c7fad3e4210fff5f8be7aaf3b1a6c",
"2a9d3fa655934f2909de0a158ae66e35b268b2682c795e77c5e9d3cdecf76142",
"e6eb5e9ed983e1f270a72bd0c4a868798ea042efc8bda3adb78d5bc2438b5411",
"f153b63a831d9a5309dc6936453883c31a929af9943bf7bfb35c10fa9f4ff968",
"89aeb7a251b63d70ece76dd1b16e14bb97a6c181c880d369e373c4d2854f9315",
"bd66a62f55d033b7c9d8dc68714787b1baed252c66220e0a5d6960ddc8f0d362",
"ef782de017160be60b3bfff454b0fe589c34a3e6f1f9275b520a91c4ebc85945",
"faaf35e1911512270789f220d2d31e21cf4b2ba6b98da2dffb5fb0dbb06ae402",
"8f4b5186669f788374e3759badbc10384e9b1661090c2183ee19b4990056b15f",
    ];

    static KAT_DECODE_BAD: [&str; 40] = [
        // These values cannot be decoded (w is out of range).
"26b7ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
"e77f719c0bb70774d38b65f54e686da67c5cb40327dc54a35e4eede259c40ef5",
"1da164d9f10443a44adaa689b304fc620b0084b83a3a980745f0a7bc909736a5",
"6685163e6eb1ef12173b1292a5ce168f0902df2d0cc5f8d84221e218455e66cc",
"9be913019bee05796667cdfa16f24cd352d966dd6f7d0c34760e6ab6a72aa4bc",
"c7d0dc80aaa13be5c83eab4a2690c8689a86b192848e0b23037e6432afccc0bd",
"ddb2ada3211019b1f9360c1c773a935b3e0dbb554db3e3296b3edbe31468cdbd",
"f190aba112cc538499ccb40d0a03981a357fcad60bc6a0b76e2b01a3189c09f8",
"d689e881ac051dd2d853d91a2abebdea4c587696a207e05e77c1b4228bdfabea",
"4f62d302458e99fa0d7d3f09dd853fd341414db3c0c84bac9c61b18edfc5b8ac",
"fcd17606aacd5f2d45360a26c8709680766b6a3167815ac591168b16bd299ce2",
"ec4ecc7c8683f7ff435012a8458650e4edc3faef26dd02c455b05d659db194d2",
"b85eb4cf83a14cff1c6360d619a61062fa2a12e52be9d03ac39a4baae25b87f2",
"692ab83e0120189cab03921949edc4181310f984f2d611f53a1f944c62dfbba4",
"d22892e900c7fb06a8b481348e56cb6c6ac7dee630ea8d7750cd5d7c809108f4",
"8f275e14c648647cffe8968b8f463a9e6625510c13b90121e8679a4f18af3dea",
"38cb8c5808478faabc4aa9bdc27469867d968b3db335ad5711971bc7e23107b2",
"af39fab77d9de6dfe58ab9f5973588211fb12b6213ffa66038b07fb571c85bb5",
"f16d6faed0a0acf357900acb15d8b27d10dadf5db0b5fd545eeb8b4784e2aeed",
"f3af66c2a8f69db459cffb0f18136c08751596c03c2380bd8bec413e758ea79c",

        // These values cannot be decoded (w matches no point).
"464b242d1c4c8860f05c36c21568a428433c32efe8634382bdb38ef392527840",
"3a7f1237413bafd046210bc6974b04f8e51c944eaf6b0346c9ebde1dd3dfca4d",
"840b63fa443bf4f8a491bd26abbfe415fe0898c176b3cc74f54494b15d01fd14",
"38c4ef0dd9b16de9f38f2915fba12ea7579687b3d045680aab2781abbb54e476",
"b0215102edc6a067471dfaecd71f7339a08ba005461ee652f2eee219806a330a",
"7ede36ac262d13c1445e84c7cb0643e3dfc81d0d2b45a8efc0c5eb0fef9adc65",
"225ee2a9f8cbf55ba44529e5e06db708b0bb9f0724654120a7f642d9f90f722a",
"1648b3502c9406fb650ffd5fd185f7f04c7a37183e024345e4c060c11da4334a",
"877c919914c70af38631f5a28ee98b2c85c0188facc3158a44a529f840552849",
"7112294bcc34bfed1f4050df27e669721d4943387893c806c9b436f5e7666711",
"e219124c9afa4f7c69b2a010c6a9cf82a8aaae67d28d01c274432f9018d2dc78",
"98a7fa5f994c06c15b8e919c4281459e1d3f9f6477ba2589346b11fa97088e73",
"73a881c422e9594ac9d8fd3eff56b8a920f50819916c96aab5b0da0912d75f3b",
"1fcbb61c5c447348aca7d005f605787950a6e58d91ab4481ce88bc93e7771921",
"886e412db5af6173f59696d486d6fd80c302febeb740bf64ced4dfdc0b7cff7a",
"8d7d3d424e413e7e0d984b518dca099f0a6d95d7335ecf72b369c0f4fa12524a",
"d9321c34e92d4faff2e22e6b45883bd1cc02f30c21f9c36924e4ca9e481d994b",
"7c33da291f0df8180c031975041b05d684ae46ace81023ff1ef36d137e174328",
"0577366515537e4d4129c358306105508d405cb310888a02b0361815aa91b769",
"b17a73c541514d26a361f23ffb635e3ec4d671deb26364a8b608c9b4e200b506",
    ];

    static KAT_ADD: [[&str; 6]; 20] = [
        // Each group of 6 values is encodings of points:
        // P1, P2, P1+P2, 2*P1, 2*P1+P2, 2*P1+P2
        [
"dfb1ca41e8528d308ccd8bcba774df736d45c046be7ddbd16cf4d09e4bd6713d",
"5ca1f4dcdc62691e34a39ef351d89b23279bc0c9df39ce041ea1c1797c2dc954",
"1965427d058cb8177e90a5347a71c4d8e8c984efb059b391b834fa020276336f",
"7f95cd111ebcc875c2901d8cbe1a998e515da8dfda172afe7ef831ab7cd9fa34",
"9dd5baa72fb2a502a92ebce19770953b61edbbe918468c8de917dd0fe5d45333",
"979158d8691c7905469f7767a093250277fe35814d0d7660a91c701c8ea67c4c",
        ], [
"c314c44cfdedb72330565f3484f0b9ca5af5f8bd07c435e295ef236f25d41a5e",
"21107c12f9ffd2fd3add888ec2db9b1c90dd2af8e3c85d4de5188e711c4a277b",
"469506f2b9102b9749cfc3b5ee3bd40fe5431671095d816fe4c532a8644c9b52",
"d9465de29203ad90622c4effd6ff37c99298e7a973742a7ce41c926df2ee5c61",
"be51fb71259eaa489a24af77e2eeb319dc57b9e5c5fd02fc573eae58e3c93001",
"9fcd219e21e802f9ca690fed8c88136bad85243b3e58524f7ee0ff0d5a87fb66",
        ], [
"1608e067352acfe93e4d9925860e7fe0036cee802a41187f138c7407148b6249",
"8f7a3119ed80d0f37fee244ce0ad6ee64d9ff33ef96c870a87dbc0e5405a9d46",
"b0c6ebe2dfb85ac3d7c707d3999728cc8dd835ef8cb96fd139fe6e1828841357",
"85fa17f27db6337446b9285f037cb4ef13da278189864d5c080b375915ead92e",
"c86e83cb4da7e786ea6b60d783c8ee438cbeb6d3b48712fe7a05f4ab5799724e",
"277703b788d7c22c3ef83cb1dadb8f12c2cde24273aa347e93cb53039e0dfc36",
        ], [
"9ed7b348a6c89cf4ae7686ca60d7e8e4d84aa8914902de9b0c3c0e773a5d2338",
"3ae19805b28e8fd11aa08e2b3504a5f929c7d195108e9f1feb6151b50c16b002",
"b2b6a406654c7605a1ffade8ede158e306e44ebc6c9df121be93d386803d713d",
"fc9f740da4eb3f789ff9f2de63566c1cfe82b7e9469312f92e5f06add5b42c01",
"9a19d29014ee1166eb8075802e473fce54ed18ed64dd5d177b3609579314735d",
"5a16f010530c9e34358a20f5da9a341553fef73b8d0469f025586316c46bf519",
        ], [
"e13b4bbc01698f5ad066af91836e421fd222c6ff9155ecbc8343eb596c96625e",
"304dd928342859ecc0b8f8f8a04653dab139e3b5c19dc6d94285c5426e0ff81b",
"43b646939e1818e4233641697b95a0a498d14b94c75f2fa40f87ec5f1068822a",
"b8864ffb961f3902316463cc1c10242eb21783807dab44155e5ac0cf1c79671e",
"799f6b2a9765582995e8af3c409d662edcee092ef8b90b6fc616753d610be611",
"e5aeb24c1e7c70a09405918bff754d00514e21e78b30bfb16f4afe0afba04026",
        ], [
"456e102ab8cd3de4da2f87aef97de56322cfd090cd1bdfce2f7d1a12f3044c03",
"dab7d1e828f3a0fa442813138b9500d5a6e8593cc2b258c38f4bc2f6677c2b4f",
"4ee98f9b18ab3e943dc44ef8a1d34cad80010d56da720ad481d106ead5f8237a",
"69c3d94898c16ef6a8e8b704bd75f795d5f61a2d0e5ad88ede10f0463511fb4e",
"adef55541436f51828188f2262d14d0338e29e225619e8b4367373b56e717505",
"779053e1aa82aab9b4db23a6c62b20c1302ed57e0050f9ccc6c962abba9a2919",
        ], [
"47c3a0316cbe7eefd75e9ac28d048ce4a66c4d1f4b6e1ac2f01b92202b0de45e",
"868e38053880afaedea957f8a15f5fc1a7d259a9e9c81d17c7c6e459a242e343",
"1ea1e821d1c0455a3bce1b02cba1954222eaec267609f0e020df07a30f4d651e",
"fd15ff8f1ffb8ab93202477a50bffcb51919f9601f00dc4e8d5f7f67bd2e5961",
"5dd3d5b70105d7b9489f371a118982ace633004d0d6bd0826b6ce531f1ee973a",
"6d03d8bf6b7151c4bbcde92060f50cd2079893a4008285ee398fb52ae9ff3f79",
        ], [
"ee848f7817d3e971b2b54c8eb7e10b1fbab5ad289923734eb44823a02527f072",
"a01f4dc1ad54ea5fc196970626b5cf4ef85f70c653b094611d38a5066f2bfb45",
"11e843d9e9ef7f61e7832326db3ba20c0d32e8536a2d626eb8972288a8415053",
"cf17d00741fa91e8e3a8e5a9567fc470a37124ad7bb1e3261d42e06c61d94f40",
"2787fc966f26beb21166505bd8086538d7bab1c478d40afd4aaa6566d1f20c43",
"6a6a2ed9fbcc30fca854f44c5bbb461f08d8496e51e4448860fbba36131c214f",
        ], [
"7a0a00759d1cd4d5e8b105de55df8e24c4c0f77126d93012d821f7329ee8bf4a",
"896782828d9778495af5e81c1d2728eb65701b5702f5c76339c6327f747f2d79",
"5fdcd0e79e3710af16a8429971d604a37219df13a81db02f17a075959636aa78",
"38377262c89d13499fec4289370b84d2f8909b5ff7f78f852fa58de75b501212",
"33101a47819a4cb010a2fafa2200b9eea178eb295d827aa70ac1eab90f209847",
"07e8fdc7e3c6f94c7ea00db775c57a78e276a9bdadd9bf48edc8fa2e7d84be7f",
        ], [
"20a3145554aa24612918a65010f5721dc35f1b521643bb7a9bd128283eccf801",
"7936bc60842ba0473107baab33f72323434774cdfd1fdee13d4ec630173bbf07",
"d093d883ce426874c3eb10d07d742945e54dbadfd3d0987ff2257b2d6e36de41",
"5d7790edde5459bac7944afee96d11ecf7a09fa908d017a1dc0bfa21f48c6c5b",
"6aca1b02bfa6d8a3ad6cd8885bae5b67f25988e0482b2d89066ae73240fc340c",
"3d6f0b6f496aa935d91165d70460f2f98077fb07c8aa440220fcd185a9bbe804",
        ], [
"161c59317e19a3563366812fece786e996484b8fd5d1c148ebd7dcdc8e88be5c",
"7f3715d5ce4db0494b4a3385a04611b18a4d7bd22765a212beba300a33dadb6a",
"aa27b14f18dcd37ea3cc8d240b643c87c26f6aa76ed7e4fe96557c84e10f820b",
"a016d56b4104213befa959dc13b34adf4721e848c7e06908c8dc549d873d9d64",
"a15e3b933bd4ec8f3a81a17249cdc110478cf1f936755f854ae1512df2f3ed45",
"8e0c587d9ebedf7ae2be0a40670068b7a063ed82d0bd2cf8488c83a9a2578e58",
        ], [
"716a5981c982111525789284a3b28513647e27d94a8bb8d070f0a4b7d994844e",
"ef84ea063af30e199e5d1b68c7d0e2ba3474cafe2b1e68b65a9882b660ced801",
"a30e0c2d8d09e41927fc91751d9c7b93cd857805fe3ba094cbc0f2d81dd47f20",
"36954215cc4c68afda107cf709bc02b293951cefeb3cdf255c7057bcd0594f2a",
"ff26b0308c876b707eb05fdde588efcee433302123afe241a7779622ebb00475",
"f241b4863d570b3400e052c190a58262d7350d4ea23b8a3a51d46c7fb616d373",
        ], [
"eaa1f6043a6fc4aacf323f037b7f2d3fb4e76b4294b9ef70bb917c9cd9254d76",
"592f8c5cee2995928ecb7a4a5bc4c900c90fc3ce160829c99e7d9bd94944de39",
"a875ac408e17b91a87f02699cada2e861157f431e3a534ec2596ad7f9b5ec606",
"1eefe83ba8687ec47cdd7f3f452429dcf7f409471a90591a0448cc1ecf0e5e3e",
"29a810343e6df60509302e68b2a5d0afd8e0052ce66c96c12f1731aaac9c8a26",
"c9191c9a53b59af68a329c0ecd9bce33b859f37fa5bf810dd13cdad11b2fd941",
        ], [
"030605efbc3bb274c4a57facc65d60def032e2c5fc8fcbffa0132ccfd05fc342",
"b35a69133de37947d0d4a53c978ec6411be97da282e36596ebd3d7c6ae53af7d",
"c3438989ccd4966ca03afbf5ae4d9b4fffb63233ff5cf62ff2f89518ab859845",
"03267f801c066774e3ce7ca37c70e2b011389b46ed48cc2bc506deaf349a4167",
"06169d3dc96c474d31dedfbe5c7b5ebece757f55da11cd39dffa1d1f5876f71a",
"f30fed8c79c1c5accff0bf1027b3a22dc4766cb123f86a6e2429fff470d1da7a",
        ], [
"ee24e224fdf4cb9e03719ab4e621eef4bce1257ae337c9b55306d78d8801df15",
"bd80f4653d286e6150247bcc0fc3028857fe1af36d034e410928c0a3e6ec0522",
"16fa8cdf21208d7cfe1a32b1b8a76fad08e3bd4388559d60e551505b0ff1bc2a",
"124859494ea846482a104ffedac43faaa90255fa16c4e2ad6d56f2e9d928b40c",
"6fb267fc018e758ae323c287639c3bd070128f060417cd9120d0f050f8015d57",
"7ddf2a10f4a37299caeec5ccc9a0d601e3276ee0ec3a280e63cc8e777dead80f",
        ], [
"89ce0216734a92f4c20f35baede5961645da4429a90ed58380e662d1efc41213",
"419a372d235448fbae16df3a8e2cb663372e8d90cab86a4cd6644f864957a22a",
"fe7b16216610acad06fb1b94f0d9ffc61f8bcb54f7dcb3cf51c3b6ac194e6156",
"4d49806b7bccc8be2aac9cdd76ebf704bb441ff57e714844ea7eaf0a7e19e77a",
"fb199c8b717288f1d3d4050ffabd0c074a5a3958064ec96b8909f5d55bb1eb70",
"d01b9c987ba6049327c96f9b2b5dc6b83cd26044522b676c1949197288827a6d",
        ], [
"67dbfa93752bb778433352037a2a9074442cd4d245ae48cf1144606cbb0a6752",
"79611d1fa2eaee38bcb69e015a1ad7b76486171f1730124c8b1bb2df8bcc5218",
"4f66724b154d19e0ba806e7877f668c9fd36a5e82e5ea42f2f7d887dc8ea5d7f",
"eac67bac9ab914eda3279f7e6c1b7d65497899b6254fade7eb06d5e7f9fc9615",
"411b470ae6914ce475c102cc3a4137b164cc261f7651ee66c1eca8d3aa1eaf56",
"5300a7cd35274040890994d59131dd845c935db8b9dddb073f6fde47f4633430",
        ], [
"fce47a5b4d1663b9eeb8d93012096f12a6509f45a7d2de5269b6b12686492320",
"b607242257bd8805c6798b11b8ff30b3bfa5a186d0bca466a938d2f01bcbbf42",
"9339b879a6afbfabc0d3cb8477b639d0d2d5483782b4ab92199677cd15e76647",
"995456185abe9a56a94027a6708f7931795fa8509454a2d0b5f4c87d2182bb5c",
"2bfc84e851330e272ec66f518cfe84d7d444072b835c4f4b8b9c55dda8b9dd62",
"8950242ea5180f4a6c6aaa57539ce4d861be902d81a3e2668a0b62a6352e6b58",
        ], [
"9af6d2b370ef3d90d1fedc3dd4b77ecb895416a51ac291f42e281ede5c80aa2c",
"ea82dc22ee1cab5b512aed06024124c270088f54fa257d3a4ef74e311c7bbb0d",
"44729e9aafc33a1f1d981b86909395f3d5b2aa4da2d2289e0df06232419dc35c",
"98837dc918083bb59db02f472aaf329ff678a92d96a4b26106ed353fa4a80248",
"4a0d207b569516ef57f8d11ab777b1cc75ff0e37aec2439bb43b151215bb8852",
"a4732968e39260f27f1e2f9e91bb29438ab11e8ae03916c914f64bb83864b128",
        ], [
"5f65a4910e32732bce8611479b2c64276b426e935981cac8071222fbbec93f52",
"d382bd0d89a9d52dbc4aac37ca3cd8dc7b656235561baf43c4d9894ffd498f0d",
"d88df2e1d8d02ba7957fee18949be935e9b28b4e6ff842a020255a5a8e1d517a",
"4df39c13b5a53ee6cc628f0f6aa833c4263cb19e94790fc9f9d9cb7e701d781d",
"6c48a7564f72c814bb855604f64387a88a5cc0489f26399b1e646dc4d93ab642",
"75d6d4dd5bbcccc2c881ee126812a34712e7be5f7bcb4111751aeaabdfd7b23d",
        ]
    ];

    #[test]
    fn encode_decode() {
        for i in 0..KAT_DECODE_OK.len() {
            let buf = hex::decode(KAT_DECODE_OK[i]).unwrap();
            let Q = Point::decode(&buf).unwrap();
            assert!(Q.encode()[..] == buf);
        }
        for i in 0..KAT_DECODE_BAD.len() {
            let buf = hex::decode(KAT_DECODE_BAD[i]).unwrap();
            assert!(Point::decode(&buf).is_none());
        }
    }

    #[test]
    fn base_arith() {
        for i in 0..KAT_ADD.len() {
            let buf1 = hex::decode(KAT_ADD[i][0]).unwrap();
            let buf2 = hex::decode(KAT_ADD[i][1]).unwrap();
            let buf3 = hex::decode(KAT_ADD[i][2]).unwrap();
            let buf4 = hex::decode(KAT_ADD[i][3]).unwrap();
            let buf5 = hex::decode(KAT_ADD[i][4]).unwrap();
            let buf6 = hex::decode(KAT_ADD[i][5]).unwrap();
            let P1 = Point::decode(&buf1).unwrap();
            let P2 = Point::decode(&buf2).unwrap();
            let P3 = Point::decode(&buf3).unwrap();
            let P4 = Point::decode(&buf4).unwrap();
            let P5 = Point::decode(&buf5).unwrap();
            let P6 = Point::decode(&buf6).unwrap();
            assert!(P1.equals(P1) == 0xFFFFFFFF);
            assert!(P2.equals(P2) == 0xFFFFFFFF);
            assert!(P3.equals(P3) == 0xFFFFFFFF);
            assert!(P4.equals(P4) == 0xFFFFFFFF);
            assert!(P5.equals(P5) == 0xFFFFFFFF);
            assert!(P6.equals(P6) == 0xFFFFFFFF);
            assert!(P1.equals(P2) == 0x00000000);
            assert!(P1.equals(P3) == 0x00000000);
            assert!(P1.equals(P4) == 0x00000000);
            assert!(P1.equals(P5) == 0x00000000);
            assert!(P1.equals(P6) == 0x00000000);
            let Q3 = P1 + P2;
            assert!(Q3.equals(P3) != 0);
            assert!(Q3.encode()[..] == buf3);
            let Q4 = P1.double();
            assert!(Q4.equals(P4) != 0);
            assert!(Q4.encode()[..] == buf4);
            let R4 = P1 + P1;
            assert!(R4.equals(P4) != 0);
            assert!(R4.equals(Q4) != 0);
            assert!(R4.encode()[..] == buf4);
            let Q5 = P4 + P2;
            assert!(Q5.equals(P5) != 0);
            assert!(Q5.encode()[..] == buf5);
            let R5 = Q4 + P2;
            assert!(R5.equals(P5) != 0);
            assert!(R5.equals(Q5) != 0);
            assert!(R5.encode()[..] == buf5);
            let S5 = P1 + Q3;
            assert!(S5.equals(P5) != 0);
            assert!(S5.equals(Q5) != 0);
            assert!(S5.equals(R5) != 0);
            assert!(S5.encode()[..] == buf5);
            let Q6 = Q3.double();
            assert!(Q6.equals(P6) != 0);
            assert!(Q6.encode()[..] == buf6);
            let R6 = Q4 + P2.double();
            assert!(R6.equals(P6) != 0);
            assert!(R6.equals(Q6) != 0);
            assert!(R6.encode()[..] == buf6);
            let S6 = R5 + P2;
            assert!(S6.equals(P6) != 0);
            assert!(S6.equals(Q6) != 0);
            assert!(S6.equals(R6) != 0);
            assert!(S6.encode()[..] == buf6);

            let mut T = Q6;
            for j in 0..10 {
                let S = R6.xdouble(j as u32);
                assert!(T.equals(S) != 0);
                assert!(T.encode() == S.encode());
                T = T.double();
            }

            assert!((R6 + Point::NEUTRAL).encode()[..] == buf6);
        }
    }

    #[test]
    fn mulgen() {
        let sbuf = hex::decode("938b4583a72eb5382f3a2fa2ce57c3a4e5de0bbf30042ef0a86e36f4b8600d14").unwrap();
        let (s, ok) = Scalar::decode32(&sbuf);
        assert!(ok == 0xFFFFFFFF);
        let rbuf = hex::decode("02907f5f9930501d9fb42fd62653a149e2155d7ef8ff3dc82deb8783b5ead353").unwrap();
        let R = Point::decode(&rbuf).unwrap();
        let P = Point::BASE * s;
        assert!(P.equals(R) == 0xFFFFFFFF);
        assert!(P.encode()[..] == rbuf);
        let Q = Point::mulgen(&s);
        assert!(Q.equals(R) == 0xFFFFFFFF);
        assert!(Q.encode()[..] == rbuf);
    }

    #[test]
    fn split_mu() {

        const MU: Scalar = Scalar::w64be(
            0x3304A73398CAEADB, 0x37382C8933C3F6D9,
            0xB153382D88E2CF39, 0x9C46EF0C23DF370D);

        let mut sh = Sha256::new();
        for i in 0..100 {
            sh.update((i as u64).to_le_bytes());
            let v = sh.finalize_reset();
            let k = Scalar::decode_reduce(&v);
            let (n0, s0, n1, s1) = Point::split_mu(&k);
            let mut k0 = Scalar::from_u128(n0);
            k0.set_cond(&-k0, s0);
            let mut k1 = Scalar::from_u128(n1);
            k1.set_cond(&-k1, s1);
            assert!(k.equals(k0 + MU * k1) == 0xFFFFFFFF);
        }
    }

    #[test]
    fn mul() {
        let mut sh = Sha256::new();
        for i in 0..20 {
            // Build pseudorandom s1 and s2
            sh.update(((2 * i + 0) as u64).to_le_bytes());
            let v1 = sh.finalize_reset();
            sh.update(((2 * i + 1) as u64).to_le_bytes());
            let v2 = sh.finalize_reset();

            let s1 = Scalar::decode_reduce(&v1);
            let s2 = Scalar::decode_reduce(&v2);
            let s3 = s1 * s2;
            let P1 = Point::mulgen(&s1);
            let Q1 = s1 * Point::BASE;
            assert!(P1.equals(Q1) == 0xFFFFFFFF);
            let P2 = Point::mulgen(&s3);
            let Q2 = s2 * Q1;
            assert!(P2.equals(Q2) == 0xFFFFFFFF);
        }

        let mut T = Point::BASE.xdouble(120);
        assert!(T.encode()[..] == hex::decode("40bb85fb77b5bc0729686725ff9a89c749d64471d4e994931e834d6972fb652e").unwrap());
        for _ in 0..1000 {
            let n = Scalar::decode_reduce(&T.encode());
            T *= n;
        }
        assert!(T.encode()[..] == hex::decode("d3c47ce3a042da4f3da80852a8c5bbbda0dcdf4b1ad51c5da9f746e0dc5e5760").unwrap());
    }

    #[test]
    fn mul_add_mulgen() {
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

    static KAT_MAP_TO_CURVE: [[&str; 2]; 40] = [
        // Each group of two values is: input bytes, mapped point
        [
"0000000000000000000000000000000000000000000000000000000000000000",
"0000000000000000000000000000000000000000000000000000000000000000",
        ], [
"723a35545d0a7a28943ca04c4b617ed4dcbbb7849eea0a44559756bf8d45d680",
"f3cba9554c4c06272bfe3f51261c7cb3678c527495137618530c74da2a43414d",
        ], [
"0e285c3b0dd3bc06de5f84508a82548f95abea361cd851473572343d5f1b2c41",
"9c87035855c71ba2691011b35e0dca62a2bc707a532ee80dc64c40e80e828e1b",
        ], [
"b09e6ff542a126b128d0c959a7dddb74defddba36804d37a8ff618df0090d443",
"54d75a829df2b19a2a4d91e700ad918bc02bb6f22107b4983bc1e05ff3190d41",
        ], [
"6752e27a0ad5286a189940663c8703891d794d225c70a43733fc7121a9888026",
"1530d881b5d802744c6994b4244b9cf21da2701d6a0a7a8ef0e29b9f4ff63f74",
        ], [
"1f32b5d73288138bb2bf2366bd22e454ddfd8c48feecfbc33ef840c04e45af27",
"a6dfea8c3442c5ba1fc653f196215590b025f081a538ca51d72869e7ad1ef343",
        ], [
"7bb73f7b85dbf165fb2c185bea9bab794730ada7ac12787785555e3b628325f2",
"ac41b778c468b8c939c251a03dec03c2a49cb82d672a800dec8eee5a237b9958",
        ], [
"73689d8361ffa3d866db43f4f07cdd3cc40dcfeb4c87a7bcd0707caeb8b87d7c",
"4e7960431a67ea2498e8b5bcb515c56de6304a1f486144961ac30b242a16cd33",
        ], [
"e69f4b22d05dc7d00763063fd29c4dec1d3250f10d245087549260f257752f9a",
"73f89a33902e496c5d61ca609fcbe0b385b5ef0357420f23809fd60d703af603",
        ], [
"fc911bbdadd2ae6010b5fdff3a236eea7270f63977c6f2d7cb7e404131941e17",
"b40efdd01d6b1678345352d17a93596873cba4f861f7febbc75dc729b0b3a967",
        ], [
"5cf1c3c547c6b6c35a6f1e0e18836ba9495a1340eb650795a3c0353441988e49",
"8319ff7a59791640b8d811ad0903907e4ad092505dc4f8ffa6431cd6928df844",
        ], [
"aa147bc7eb3659dfa327f0af067ef8ebf233e8b95633a26f97da175372a81501",
"9d3e1587230c097098229a28487ffbbb2ac7007ebefe3d2be631b9925c9f2901",
        ], [
"512729297f0cd62a02a7ca546ab8666785aa63a51d0e2edb1d1cc5e760793907",
"e787a112ac61bc5ce399565ae6c37225a424834c87d2a91dfe345a01f688034e",
        ], [
"25195837a7a1fe5585a9fd125809fd6caad0217d01cc0707bc2591f0dd406932",
"756e0e09b3b485ab4c6b876b4be36e3ae2d57ecb9afdf317ade3f2bd21ee3507",
        ], [
"5e78ba1b1d0f5c006b5c0e740d8f37b3f3332c51aaa876debd169d629d313d8f",
"52595e618a146ee56e11d3a2aa12a7746c57dd8bab13f56f122fb5b425cbfe68",
        ], [
"abd9fbee5625b2b9527e6fc007fa8e5166fc4c3b107486fedf27ee9a54b6fd60",
"26a742ac2bc65720790960915af5652952356d5ee84cf61da72f67a064ce967e",
        ], [
"c25163787f737f1e016a67d4dbad8cbe4331787cc65f23d7383f93888b2b9d52",
"3a214343560cee77d9065f6a1f3aecf523e3caec815c39ce83c70441dc03cb3e",
        ], [
"39f5603a06d2958dd485c9862aedf0bca0e71dee6fb1509c1fc51bd457fd19ff",
"7ea9f6b375a5e4877aec88c0a151d6e7f0e17bf890d63a3ee798d346f7df305e",
        ], [
"274822b4713f0d03cde0410fc70545ee608aabd2adbd671495e1737b5f4a0ba1",
"6d2c6af450083f10abb853449e3a02ed946b4eaeab83b24d031c4a66d66c975b",
        ], [
"320ada54aff47138c3a85852524536b4f8817226dec94d9b48ad3b0bdf9677ce",
"352a4595e6144cc3ebdc5e7ba9d2d5a70f52f4f07d0580cca42304436b815642",
        ], [
"2b626f86360f5e1b9f8236e3990d5594b477bc52e0465777be80cb8b021bf8b1",
"e1b3b53403a843269201e043a73916bd0be4577bfd6b78fb941437f42ea4dd7a",
        ], [
"e0549613fd58b01c1e32404c83d390c6acbc1cd69fdde89bb0f4074723f734ad",
"b4a49c5529ae2f3ad3a7796f95a00ead0969243558f3b6ff6a359fc21191490f",
        ], [
"6b1d7b15aaf41ac151923d58a4c8c027566cec8220d98cd4501309daa259d605",
"1451d6ea3f3c93dedc3b3159c33c3c84c2c3766ee1d3ef96955914bbdf15f069",
        ], [
"c49b1777775059bbfa05fd83ec6d14d46cbef91fa3e9fe9a2e07885d926947d5",
"f9d09fae607c55af5a4dbe073f1cc19f034f8ac1106ed9279c0bb766896a5f5f",
        ], [
"134c2597128a8867f90dcfd1c33f3ffc57638534d0e3491cc43749afe7be5fed",
"ab2473703c6f3b602539b9a6616e8e5dfabaaccf39a4a2c021423e1f2b292104",
        ], [
"1bf00cc0ca59dedccac2da93b5a3fec9ff65acd0ce92dbe13fb53d6fb688833b",
"cd452b2bd9cb2128e29b37627865a0becd5cc85f7fd7e5dc61287d33698a7158",
        ], [
"0ba02b6a7ce2e2556739299745ea20ffd4df8556b983bb71ddf4d2cbd8aac77e",
"b934730eaeb0851d5d4db5d2428d485e029103b88c68e6fc9c24af636f86ba23",
        ], [
"ef50a39fdb8f88b94f98877cdf97800dbdd3e95f93415b924d985c846e4a5754",
"e1906955c28b5034e2dbffc4b9429b9540a7441a094fa453bc97cc4fae500a4a",
        ], [
"ea5eea9023cd657d1bab53b4a709e961f03672d975154cb7f0b04191c666fd3a",
"ed99fd5f5344c641326ab44a1670e1ef7af9cbf10ae4d948556fee4ab0e9260a",
        ], [
"f1d6ca77d1ad6b53db5f55f97628879e6dbe819f355d97fa3077d899e98cb6ae",
"1d348e1c2c68c9ecf1494db397547018c4417268e3218a9cdf8bb30e6444fb58",
        ], [
"5793d8a97e650da4e9ca4540eb52fa565dab309eb0e9bafb8659f7d47098b007",
"3b9c9c2623e23be1b99c3169d7b08fb73ca892962233a01cfe1b86e2d0fd9903",
        ], [
"0712f766c5f6db6ccd926d92bec2d6a35391ccbced21661acc6ec57d6757d5a0",
"c49595a2bce3a434d8701ce7bfd8d5241912bb032ac250afe152122e24fc2865",
        ], [
"ad80519e5b13da5e05e0652435d944656a28af7734bbd05428cdd8b99609e9d1",
"fa104d70219c53df9df3dc2370d6a0ace0806068e16ee803e3e2695110396614",
        ], [
"c909546acd3ff1d750ae50b886dc83d2e9034a3972d5de3891825ae41b60afef",
"ccb7ce67b471335891b82a1301d7e98fe8c30d4972f5d7b1b22b418f2eb1253e",
        ], [
"abafb6538877ab38cb909307eb11ead4e11ac2054110f174abec506276c30559",
"b5e61c01ce7f0294e7289cedb19c973ef0ee600285bc3626ec02e40c1293f565",
        ], [
"c266be61b0724dad07a2417c78710767baebf168e71984ab084f808528a3fd36",
"d1a0eb95a8681eb0bf72cc7426e708cbe0acc4d1fac64a0f0b0e70841894df08",
        ], [
"ec7ac0c0e996296d3e5c19dc7b14fb3b140cecdfeb7909c8afbc7441b1a25851",
"36cdb3ed03e9f17378f36ce5b851c86f31b594204e42873329def6991548484c",
        ], [
"4b6e7b392d706eeb04c76e552d0ebca92b9155770c169dc40b93fa9b6568c0bc",
"451bafd5654f097900ad0bc3f860937284187fbfdaeb0ec5a2da083c3d585030",
        ], [
"9c31cf29b6fb775ff36abfba52a800ca75ca5ae9ec70e6e334e72af0cd4f6d44",
"13b10053406de3298cbb83ff7aec300dc66f3b381edf51220a4fe59879daca0a",
        ], [
"f1969458ac27c6dd5bbafb01fe0b572b7a2f099737f67289f6124134f2e35c40",
"08c2bd77a2793ae156c89043a0d41a5dff7c3cc33fee347e7f4a987b0ad0c44e",
        ]
    ];

    #[test]
    fn map_to_curve() {
        for i in 0..KAT_MAP_TO_CURVE.len() {
            let buf1 = hex::decode(KAT_MAP_TO_CURVE[i][0]).unwrap();
            let buf2 = hex::decode(KAT_MAP_TO_CURVE[i][1]).unwrap();
            let f = GF255e::decode_reduce(&buf1);
            let Q = Point::map_to_curve(&f);
            assert!(Q.encode()[..] == buf2);
        }
    }

    static KAT_HASH1: [&str; 100] = [
        // For i = 0..99, hash-to-curve using as data the first i bytes
        // of the sequence 00 01 02 03 .. 62  (raw data, no hash function)
"ea5af1b80af04ff3efee57f0a97cdee34686ab6038c28c09fec9c95b57f7b454",
"061cad337330c4f365092f07dab0c05e715e50897d288843ecacd349d4bf9b14",
"1f77e336d1810d3747830fe1c6533f69f65ff4045049308941f6acb05111560e",
"04b9fd9d18bdc3a601194037c435e48a23304f2da8885aadfd9343f812ac8633",
"372562db7b536a2fd960ba3521b93e8cf8303ff73a37b261cfdd5f5121689e16",
"1c1fbed5c4f710076cc550d3486c838f510670876b41097b96e64f3a5c29d33a",
"de3731edfea76b0c10f0bd5a2901beb95b8b509e0fd59d5311fa6e5681d76817",
"078dfea0b1e8cad647af96e30a031458b18e24130553ea7424c6ffec3dd2cb78",
"1f4b6bc8a64c88de144f4bc009aaad628d908df81995356ee2a4d85a91328e10",
"66e58ad880b5ce8b17d6b5fd4470845d48bbe3973e0ba3b244a0054cda188c23",
"69931df9a6e6d5872742aee633518373d5f38529cdcf182494921b5ad0d0d05c",
"241ffca335f346f30cdf6ccabe87dde0545c30dbfb270894837ad330315ee503",
"ec57e395ee1287aac0bed4fee327081a8c0b63c69bdb6fd300c721bc0302a90f",
"86156fb75fc6c53dbbcd40c6b310fbd1f436d660c3badb2ca54123ecd3f46c31",
"a2a88863d1067100d46aa77f6bac68af5c9977a303ad4a4851be274b4e704277",
"64c1d9f08912dedb5c10ed972b078f9bd0eb335d232b98c7adfaee8ac02cba27",
"75f37a183101ed3d09178fe8d4dfa88ac446a8bd38e251db25657cfa49161659",
"8a5111b149a22a26e8771cbc08bbfaccd37136750400eeebb1cb2f5f4236f93a",
"e2d67210a9d4ade9209b8bab4aae5d23d6003f1f4d2c86902ea2ff78f338e35b",
"1d1494b9eff0669514d9a5b95d1d93e2974c936fe6a5f0891d8c0a8fb3703426",
"03407d01f2261f7ea11f060c11fa334caacf070091f316c60dee1ee9c9e32054",
"3325b4cd99b2b15a16dec5147c36c949c68bf0a11aed4af888381978ef7b231e",
"3d5c44bb387298de396b5ff2c21e7ea7df4388137e7d94b82614a7aff1fa110a",
"bbc9df0d359f305629182a462f59760cab2716194b0ede35cef1b55581985a6f",
"cccbc6c821b2666dc674306c777d10e0ea45c0f8bcd1c80ca3b97e7e0635f744",
"50893c06248ce299fcb07a8e9a69bec1e37a3365858ba5820c8812879ec6f647",
"2371e553cc971af5b27c51dbf3d97bd2c40f7440f9de8f0834c86e09f643973b",
"380cff51d54f153e50bc6bb3c102b7aa293166ed8ed90cf4cf28446311bb7f39",
"6e04d42b1dc364c0fed689ebcae48a36d6cd3a227b50d30c489d63bbdaa28826",
"199ef6706a42d10eeda4add04cb6ea2a7ccd818fd00ac34985e98522c7d0fb66",
"83fcb078a23b92d0a9918f36d81ab87b7efc4a0e3c4007d195da59c3af1be228",
"31c4a7af764846813587fb7d02482d797c85f8d8e549ab414b60b30a18126475",
"fe99a252275a49b60e79583f9b442eff20fa5dee990e1942cf130db58c918b55",
"87d2ef19a8ab402aa2f8afb5f5614062522568aff3fdc0554d2ae17ecb1b7f28",
"7b80858594e29abc534d1deebecd27fb5f190b8099de8f85fa571cf538c5a533",
"50ac79e72d05ab2eb101740ecb3d514b09a23e30dce2737371d791ad5d331470",
"8bcec431440081c08ccc81889a79607af663a3f052987b319ae75e44ffbdbd4d",
"a4fbb6a452282e68a7c2abbae31950bc5a5a6b5831cf15290b8381689b023c01",
"39d26112e82b84a4263bf6c6d35f8c1ad3bbbc522149edb6de29caa6b6cc5c3b",
"8ac6c95be88a132dba4b235b0cdd78b2838e47365dbca0fd6c597db80607cb00",
"a3be23f26ac3e37a86242a3df61b655e1a5e5f67888336aaf105ef4ea153a669",
"9c3d9821bbd84521082387a9593605109434b8548497096147f17750eabdc778",
"fe375d92eafdea8f3a8885dc014022ddc5806a52081b4a6cffd477a6d45c8f2c",
"1ffada1e8589c3a22c4667f00cc494d3638a1021bd37476ba18e73fcfad2297a",
"b627268344164762b879c97c2af7b8f1e39ae1b090c96c4c2c76ee4da6ad7111",
"1f983d4c111ffea3dbde3c272456d37cdd5e26f579d10c49ce51d570ccdfc30d",
"187ae3f089466e208b5a92cd238f684c393ca7c0401de03a8436f20e36451048",
"2c56ad19e29ca45bb1493eb292aa50e3dcc1afac9e9b4a227a411d86123db73b",
"29709e5b4d73f76f98ce756f3546be0b405a28741027cdf01af67b3c2f58622e",
"134654b17d79fee88c695b868e806f6e6c8f8a61d3b3e3de7d3fc2e031a1482a",
"4c607b8036775f91d839d1886149986f9b4cd0fb2662731bef3fca0ae93fda2f",
"f2712f11d3728449fa13604a998d2a42e126f997dfe0241ef76ddb0f613aa722",
"9c1295bd24147d4f65a0904c37f0df6011513b98eb2242d9f96d7dcb2909c861",
"499e4b77146070f43d756d2972ac4fe0bb9ecb4ea31e2661851c4fa753261603",
"f8a7d6f2e46ceb766b082cd71ee042a2b7c9e0dc34b72a28eddb19449e9ea748",
"46c8b88b6b297826dbc60173861174e1335945a06346132a387e72fbfcbf0429",
"12a91c2cd8c98850f2cf152e6c7675072d40d82079b740afc73470d5083b714f",
"9a3ff5663fc4984d4b2d574e005463b4ea6946293dfaf0b9e7fa20b333dee13c",
"99ecacdc83164feefe463d9a0f2c664a1d3df5708bef679b3ab18c1766adf43a",
"4d79d4f75b6420b4d8c655f9edf9cd6e3e35f31caba239bcc77ba54260632114",
"27e076ee9a2764557144d3fd582392b9ac3d62555465ffd62c0f5b50e2ca4e2b",
"a95cbdd42ed59b40056a7d8a0dca6bfb182c89b7470d8ef37e0ee05818812756",
"fb73526b9a92d0ac9e99fc5dfe1c0d8b7d9cc76dc5f83ddee9b1ffa77d4a4675",
"64816f3e0d8a2927fa436610a56cbc98efa03a94cb76035f4500c4ee72d2b511",
"a9814706e9e9b7466b940707fbd01efbf17cb0a3ce6883f477eccdfc6cc5bf0e",
"1f249011e0547fd832d01d544ecce48bf2d6aae19a554828a55463043d254b5f",
"12d756781f3678803f8172ec76eee3ea31bbf9d4a0007689af696dfc93080250",
"a76136861eec114a12ec54e2c456ba7b7c15ca7f2789d0999e84d23eda03ea15",
"7c340a48954bd74952c4e7ce02f41136d49c62a173311f15d800523acbfcf914",
"4ccdfc53c12c6fd1f411ab0d4dcb199e3b233233f41e363ae95b70ef6977d049",
"a0c81c623907fb5d0b622aa130eb67135dea275dcf0dbbeb6cbff08215eb4908",
"d5d1e3b30c3a371da222ec51d65bd5336c5065c9c16e9d9133b00737c6305c48",
"525091d72c384b01c91c26a5da768fd68b6b36cc229ee8891385d4403915eb6c",
"01a7d44c8f874f71e7a70e1b55f2fc8d71a99f459ef8ae70d34d66a8d71ea30e",
"d6dda270e11589af3b1c705b9251900dd3b81f9e554956b8eb3a7a4ae916e36d",
"1aeddb680087fb91167eb61957750d57451ed71f84e07c016f4ca3cb8e583152",
"6d56a726a1cc5c7ca4ad3ea3bc5d41706226ffacce94ab0a85f31954dc41e753",
"af3ad2a1cedb9dffa4f69409010282f0300ba04ca88ea5ef87317d83d3010828",
"bfb4780461d488cc2cfc3b8f1ebec04d7fdeee95cf59e07fadd2bb9184f5f556",
"a96ab0695d4c2cff8f41a8ab7868c15b0b5e70577c8c56627021e9982dbea84f",
"6711a9f24b4ff403bd1ce746537e58a3c0c45977a9eedd555407ce2751d0ce27",
"3cc85d0baf0b26a256a720ce3cbb91b311e426bf716473f3859ece8a36f8fb6c",
"1e5cf78356dcff1f8fe58ebcfc1e90292f6142e054421a774a887cbc336bdd35",
"a00f56894911407c4512024fd759f0718f3a3d0655d9c9852a843012a344087d",
"59797550857639dd58fa3f66c31bd72a952afde111003978dd2569635cf6540f",
"feef6d815d0cbb61c91a12d73e24d4351882792468af06862daf10b79ce42907",
"bc98b35e139c8f719988cbf6e5136354f456574b254ebc493a71230431508248",
"72e5540c1a098b440965b225a236ab602294de18ba2e4c352a2b10e04f852c12",
"927538bc57870541c7de1e79c84c83b7ee05dc7ebd644492412596be3c5c1726",
"f17a2db81a86b89d5bfa43e621e22beac92b3d926aaf384be9dd1ee708534c30",
"187f0bf6ab3371698442799b5e735c9e641b0315a4ed030d7c40726991929b00",
"547462c858e44eb8017175d3e0be3750393d055385ea78b0466dcd8b258bf11f",
"f39ad973a6c75065ca09c035b6348c1cd5aab7d33a44d5981e1dc18a162b8d16",
"6bfa4dd66e1f9095f2c6556f4b5063e84a3aba74abf31f85c95ac38e38169045",
"ae180573571bd83f2ca4a9b950083da0e0600caedf3026cd84ecc5c5d4a83c1f",
"0b57aa0c4911f3126927481df5c8d9bba655873ae155db0278b56dc59904b44a",
"250201733273378d852141e5daef96491d6e4a37566515a629052ae09256cf3b",
"60a388ccc29d8b053ba1ab8f69145f9fdfc482f10d15ce74426ccf6cf1ebb93d",
"455b14a130fbb2541c05aedd8769fde56775e6cad121bbc04ca303835e8c703d",
"b6f338df7475aeff91a409db0b55ef806752e2cc17c8c4147514f3d3d70f2827",
    ];

    static KAT_HASH2: [&str; 10] = [
        // For i = 0..10, hash-to-curve using as data the BLAKE2s hash of
        // a single byte of value i.
"bf7e5b0c9466d872b546438e856fc2ef3d2cf2eb11403cbae8dd7422674d0517",
"57008d73669f292b351ee7ccf0981e63fc678535822b2d9c90df67d855b31262",
"f59238b5b070edb198cb709605a162b9056391304d9fcea64605543a58357d66",
"2f763b1ace8ce6f0a032dc593ff95810314691433eadd6ff663f0378a4589d68",
"78700053b6e785a077dfe6f386d525ff51579e321062c1e08164ba0fa3ea4e65",
"7bdfb71b647d61b47eb46b996f26930f80f36a4959729b42cf2b62c7e51e2e57",
"2a63f2b4b05943bd18cac172f2915c4ae801320d257b354c72fff1de469f2a2f",
"73f5c0b5229d9c6e0dbc4578d20db00efe718b3e6a975a1627fec75206f18736",
"7d671c6dd8d448d11c31216477adc91ad5ac189946cb61534e7f615b57c16716",
"ea2e3c8084d87bff62973a4e02896f42787b247416477d1a84323627e30e3447",
    ];

    #[test]
    fn hash_to_curve() {
        let mut data = [0u8; 100];
        for i in 0..data.len() {
            data[i] = i as u8;
        }
        for i in 0..KAT_HASH1.len() {
            let Q = Point::hash_to_curve("", &data[0..i]);
            let out = hex::decode(KAT_HASH1[i]).unwrap();
            assert!(Q.encode()[..] == out);
        }

        for i in 0..KAT_HASH2.len() {
            let mut sh = Blake2s256::new();
            sh.update(&[i as u8]);
            let data = sh.finalize();
            let Q = Point::hash_to_curve(Point::HASHNAME_BLAKE2S, &data);
            let out = hex::decode(KAT_HASH2[i]).unwrap();
            assert!(Q.encode()[..] == out);
        }
    }

    static KAT_SIGN: [[&str; 5]; 20] = [
        // Each group of five values is:
        //   private key
        //   public key
        //   seed ("-" for an empty seed)
        //   data (BLAKE2s of "sample X" with X = 0, 1,...)
        //   signature
        [
"b0c4721e9e9b534aacf9700b127be576bcf8506ad19819f809626296bf218038",
"1b9327a8d6c8e9445b41ba3fc4125521611bfabfb404668a78a13972c2ce7232",
"",
"ec14004660d4b02da3b86b1bc5afa7b2e4827f0ee1c9a25472a2bcac521bc231",
"fcb4f45ad552974fe6273a5c2e44caf83ea4712046dde4b60fa64dbaa26477a19f91acd5a648d0348c2b7f2d19034e07",
        ], [
"46e0d7d2cf8801383489a33d7d0bfc7f0ee8169f219040dd44b01e733aa6a625",
"54200642cb45da3d6deaea3243459c510f204f14f7952ff9f29ffcf3ac0ea020",
"97",
"bd1a4655b90f873c53fe908f4109bb8dfcd9096312b447a6434af3c35304b7d1",
"03ea3b335fdf8436a3ddd0417b89a83ab939620efb5f0208698a0062556e1c25247f699388d8b66f58ae99efec6d812a",
        ], [
"6b7edc4accfa2e5ead52447fa1bba0878d5a799f7744644eac43a38094fef919",
"eba8a65eebbca7960de2ee4c60ef7745eb1284f324c5bc2c762d74ed2f95683a",
"aef7",
"6230441be7f030f180e81dc44502b24ed94260490d140ae738bb80746051651e",
"80fcf39eeb55cc737eb711644725cbe8f5e69f538e59d9acc6ea7ef1f5d497989d27815efb49774f5c0b34f3f60b8507",
        ], [
"899f693d23f9bb913fd46df6b5959f0536efc1d4516e89decb4f1227d07e0b36",
"38ba0f050fa355290f7fa46620b95e648271eedbde3f17f393c7a914de313c4b",
"ea57fb",
"e877b70f8c12aff466a4dbd6284bd0c6ad7cf66376bdad599f22145f8277bc52",
"5b62b6e459bed4133e78002aa9f01cd425cd3bf8f6c3c8618cef38b61ee9954108f7e4df635b6c8f7b7b51eb4be5d805",
        ], [
"88b8143ff1bd263488da0de8756235e64a993a02d3821cdc920bae07d765683d",
"d92d7d971129858b84f25c785e857e5210f8d4d67c5eeb015d89ac20fd72191e",
"842d1ac7",
"b4c94e55cc622b96b49fcfe6b913ce3a06050b7e9b26fe840389145088d59502",
"7188f8faacea2d6e306a466868e439cf8b2cb0d629aefe429bb403d8ef6b01b4e5c3edd651f7e75ef3ecff9025163e22",
        ], [
"3055cc4730d43086f1de2de3218fe523e5023b90aa350647ac66cc6fc8c0b909",
"f45d1721efa8f19ad4951b307215fd0d1f2d2c88b367f1e8de4594884c913c2f",
"159c3e27b1",
"a7ad895209663ae35bfb3fb0e44cc83616bb876d14608e5b09c20d19f57839d4",
"cdbe66382372b0bde4cf0aeb161222adcd3e9f73831a9c2992b7ab056b265f3618bc04963d40f8639a95f7bf2b3d3a11",
        ], [
"86d83fd4eacfc4183b434d4b53774e7b96d3b90e60be4825d4f11356d6406802",
"79f5b593358afe4eeab86b8ce03ddb0948f557a19c4f1518c9f41786304a5716",
"b07511b1ea1f",
"0bd1a3ff8506a918b8bd733c31cec084927241dda2ede63f719a6758872c94ab",
"7a4b305ec8aa070908383db3bab19fca96634a10a144bb1b0a411073e2ec67b09ca34561f0809662bb49823391f43836",
        ], [
"1a2bb01bd52783d80234995836e324576f379a1eafc641082b789551e5e61e3b",
"c605d076744aa10cd8b210d8eb72a10d30d4e7bf90e1dc66a9a95bd4f7e35a24",
"ffc13f192d2bcd",
"f328909fd158f3541c2da54b758ccf750bfe4afa717b00094fd30e7fd69661e3",
"07b120787a23c7949d5af517e022fd9edd98c2c66cb49ec980c58a36c88b09581455b6314d59eefc030281c67d40dc1d",
        ], [
"6da52d7184e46888cf28f426a55d4959b4e0b8742989955511221f5fd11ba11f",
"357b8ce82ab53b84c00f57bad4c7cac2a9031cda35b739b93b9e9816e0196e46",
"9bdc3225b951dcb7",
"7de5f8c2c35149558c0a6bef84596669100f6350f07aefed58120d6dc3531231",
"8c2f8691e8e3d571ddf4c9ab4031ae35b6ed58efcdc0c7094392d33ed98a6b22fdb5a63ef2975c5ee47a8459b7f51706",
        ], [
"5d8c7af067d55e821a02d897defc1e1ec7b958a35ef88ff5d6e6ccbb46ea1425",
"b81b33ce13b84b7e16ddbdfea36e60d41f2b8d5296dcdf37e6f94855fc44d614",
"d7cdf50fa8f045a646",
"9fbcee44419bc19b97bb673d0055faa0aae1861f44c682345fb3494e610e26da",
"404225638891273ea82119a46d6ee821b62e63a736e1110678e908f16906a199730ff39f1c390277a2605ff3cb0b3303",
        ], [
"2d6d24dc143d8ee3edee22588e6dce7b182a99d16c71626cfaee3545fbf69119",
"beec5d4ac050aee9d8cf762e9344492117713c4ec66a5f4bbe89fc119d3cea47",
"3106a8fd984b8b531081",
"4ca14993a888660c624f816db0c893bfac69d5ddf04cced60333d94ac1b0e2f5",
"6454711333f86f606a97eb1260a2d14eed0d6d378c47fb64dedb0712a618387ea785668988f0e806e779417342aab704",
        ], [
"52138cbf38a8489588bf9df27424d1ae6aa558f4d14cb48d6db2e2acb8e6fc34",
"4be1272e9ce768cd66da0ae13db31e0d43df1141ac548d75550db5bb34c8b308",
"7018379cfee44be20b27c0",
"ed427029b6afbfe2a73c7a73605bfb47b4db8eadc940bddc103098a06d7b7daf",
"daa96d208afe11b333bf0c640ebe5356ecbe43ccc62d98a5d9862b35d6542054bda991ea83f562754bd2ed3e3dca640c",
        ], [
"41d050597273cef83ff9d472b755c97503ce591fd0bea4a95e342572be560317",
"b249e18d4cc019b65ebac2725c8f9d2215916ea4abad71c6f8ed7cf2c734ad4a",
"2cc3a1dacfe47cbfbe6bf9a1",
"2b083962ac0f0d9421bffdf9377f06e7152c3677e911029b08f9d40688c8aaa8",
"5e3caec500d71a35db64e477dd3c802de1bfbabb57b0e4b06604f86b71e43458e8ba26e517c79ac04d7fe9cac21d1010",
        ], [
"65ca33e7427779d99685cc79290352e7a50e2e00c8e56cbfa8113c84d5ad5013",
"280e5ac034faf3e5e77c4096a81cd2f5158a153fb6990c52170ddd9cada3e954",
"d25216525cc3e3e5b1d2a8f0ed",
"cf44d2ca3441b9089e99a00eb90fe161bc994990469a46b488e08711a7ba8d9e",
"dd3550665d38232217e82fdf9f6616e499e132cfba0b411ac899347023bbfaa846f22583b7cfb6ee1f68ae83a7c57023",
        ], [
"42997b627130ad187138ba99ea491919876c2ac11a7072f63134dc66198b4430",
"446ff1e2e8b1c3d88777619edaf906cd1a0e9d19183c352b025b897cbdb2fe0e",
"cc01f34688fe08b6390452072519",
"79d41d37434fa78c4cd3fb421c7caa26704df53c215adcc4f7807adde10c7438",
"2506dcf20f46b27e488855e8bcbcfb8ff9c21cbdd42dac379bf594d128b4a3aa2b987aa3f97594b66c58a3b95898b50c",
        ], [
"0b8a8838ab2eba7c1e6844bf99fd2c8643292fc6e4af5929607d06a4ef86271f",
"d84e000c6b08e19ac2c5be79ae814c9ab371d6947a2a943bc65861eb067f255e",
"6168bb30469723d4ad04f80e42103e",
"0756a67df9f84be0d319c4e8d324f3b77077f9322f9603f015df27f2804b17a2",
"d56d36d4f15f2b565c505e14001b97408d75e67c5ed5bedff00ddda06efaab6323fb22ff96c81bf737a07e114486c812",
        ], [
"f115a80e22a3c99fadf4c270f1c3ea44d8525469b1f6e79bd94e46857a74263a",
"910aeb2fb18804c5e6275302a2f873167f1ea749df6e3fd88feec6633cc9394e",
"c26c33d95cc9ecac0d6f2081e887e23b",
"86b36dde6d628b67332456b5d41d09737a057215f72f89094d071422e705b82e",
"5a060f9dbbde6d2c3ce85fffe8c9d28011517d7970efcbba6f89133508b92e9b94672ae9f7e252609aca1f6a0ca1bb05",
        ], [
"bbaa3f6a2949e50460ef2a74b448641639e08374396371479e234a9d21152504",
"f2295930923865a14602e81f75a5caec9d59952c8102a0003d143b9c53bd0e12",
"3b5838b50fe7674016ca8f07e62c785be8",
"86497726e18b409075f7036b1c65deacab22cf85d2ae64ef1857e17a9713e4fb",
"2be07b51ed87a4f0e2008aa113b3002f567fd6a200fb8fcdadcfb6a702472cefd40c7a0d72f6bc2ca6d82e9db54d4304",
        ], [
"115dbea4c2fa4052f9493a5ec191fec5166ab7698e596b3f6e63d4270622b934",
"f4e080af64068465bec753da1b152d1fabea5a283420d8737019bb3fa01cff55",
"8701db242b917c34be6f66b979756f8eab58",
"a2edb2c979a443ff733c32453d350f09af33068a5640af90940315e7d3c87957",
"f6de011b6f6900f63083b727c1115e29cec046b25a9e2483f00789602653addcb3000ad316d0bede5ca0fbd6c0e44e29",
        ], [
"d6aacd7885ee0183ca96cdf71c49e3596073d0894e9100880d7b38f9f3521400",
"66fde0f01d063ce623b604811beae9ddc5661f0d85e097d320df77aeecbb3e2e",
"5d8b25fffbc925783eaf34dab0c582e6432815",
"ba789f5876b8db6ae44d0e4507de9993c83e504804c1f3f8619adbd717847b77",
"48ca28bd49f281f2620a4cc87adf301ae980c5154a35a9483f4aaab9ae791b44afa6be2053ef2100718f44006645651f",
        ]
    ];

    #[test]
    fn signature() {
        for i in 0..KAT_SIGN.len() {
            let sk = PrivateKey::decode(&hex::decode(KAT_SIGN[i][0]).unwrap()).unwrap();
            let pk = PublicKey::decode(&hex::decode(KAT_SIGN[i][1]).unwrap()).unwrap();
            assert!(sk.public_key.encoded == pk.encoded);
            let seed = hex::decode(KAT_SIGN[i][2]).unwrap();
            let mut hv = hex::decode(KAT_SIGN[i][3]).unwrap();
            let expected = hex::decode(KAT_SIGN[i][4]).unwrap();
            let sig = sk.sign_seeded(&seed, Point::HASHNAME_BLAKE2S, &hv);
            assert!(sig[..] == expected);
            assert!(pk.verify(&sig, Point::HASHNAME_BLAKE2S, &hv) == true);
            hv[31] ^= 0x80;
            assert!(pk.verify(&sig, Point::HASHNAME_BLAKE2S, &hv) == false);
        }
    }

    static KAT_ECDH: [[&str; 5]; 20] = [
        // Each group of five values is:
        //   private key
        //   public peer point (valid)
        //   secret from ECDH with valid peer point
        //   public peer point (invalid)
        //   secret from ECDH with invalid peer point
        [
"9dcb9151ce0a7cb41dd14737338dc87d75f3ef7552bd17d53a442d0ed1b21931",
"ec85965eba95780181cb469f6db8ad117c5cfa941286ded7b839d34525768e4d",
"82e849c462d9efe63e7575552a4ff9ffaa065064fe33145efa013d8f50365db3",
"d5dabc3ef4d9e0b4162c2ba16a4938f0e4129172d54fd61880a7ee7e25fcc524",
"e805b0e59b201f0eab9844f0f3281ecf97046b2cbc017f27b6ac607af6337f95",
        ], [
"034c40f610153bf9997909131996d5f80e889594398d90af4a355c9fce97f83c",
"eb224cadb595d0b02c5b96b86faad18f149bfd1cb8c0760239c568b62def3d59",
"7f5830c8eb95d3aa46687a5699987dd905453a23d59a9e83d94dbfcd1489b6be",
"9f9c9aa9a69c750305ed05cb1cc180b566464486cee544c6c9ce901b2e155002",
"607ef72e0956680a4fb1681a9738e0b3a4f805741dcfc239d6645f775f357f22",
        ], [
"3f25bd6eda431ff6c03751121be1e25efda115bdaf67cdb84a0b6b2ee75c461d",
"e13dc3c334027453db737f6c9454451d2c567da110876a8c1f2d7cb94079e022",
"d7095e7aa9c8f96ca0c94f322e53980ea7f1d11daedb05e90c907c30838e28e2",
"8f77a4f12394390b5cb1df34ca9e3baf9751cb789d3fdd215a7216c30c6f687a",
"140970b23bf8b7949ab64d2fd1683d1c402f88ff3874af1bcdddfa4b273d24fb",
        ], [
"d072edc3292fc9ab93e7757122c814509043f3cff33b4d1be8478c50a2afbf0d",
"195a4427064cf84d045a978f0ae6b0d2a9665b69c46fcd997d1cf902db8af932",
"4e8ac122f0ca6f480ba93b6763b359b0fa07975b36d712932b938e3c4e421879",
"b408e4965f8e483231668d53ca896ea6f44f8bb3c073a051084454654c99c337",
"ab171ef85183b7773a41015b7e036920421257697f442f196e9e15d4c04747c9",
        ], [
"debf0abb614a6664e92d73a93073edb5229d282045ea556a4177177bb424ec08",
"695b1b056a4e1af72d0d402202dff3ce673a9503a8cfba9f1d604d8ef8ff6463",
"a95bc6e922cf178f93ad2ac9254e8cd9786eccea9e0d698954ef8950ece7e372",
"d3c8e5260b4e9460c07c33fc8bf2b025718fe7d94f0d57f3a00696e90cb6f942",
"4a7cd7502678863fe49a57d931c4a68c717681e574de5373146ad381ef3e8f5c",
        ], [
"f241d84958ecfa4da98dbc655e4ecdb39f82aec2d93bbddf56e7edbbc1ada03e",
"2d0a234ccc310f215317660ca94f65a1f62e45b117fb72bc4b66cd737f2b146b",
"72308c39d148a74c0913bc88c8f3673661983427f21e41a4bb576daa7c6fac31",
"095155875b4f029d25962fe5021091b7d1a8c3e345b00c318099d15c2ac9ed71",
"538fd628a643e63924cce9d26380b5cbffcd388e80cf4ef8d1257af0e5475275",
        ], [
"7adea62155559a0b3ad074a4ef04b0ddf6ccadcb7bc4717eb5d803371aae7c2a",
"29493a7661a957cbbd100b76fa76da1725d1ea688f21c0656e53fe9175510b0b",
"fe5cfc84ef368e43b530fe352475784d24bcda02fe133b85b0e63bcb865cb337",
"e365ab6bda9f5876281b87b64a51885dcc38bc55d2a39cb4a1870a137e71363a",
"5cb0777c26dde80eccd62d2ad682f68dc750fd7bac0ec32b8fd5bcf5d0f95d45",
        ], [
"455c966ba7bd0674a899a6506ae0950ace532047a3b9ccb1aaff473e6c15b21e",
"f2a562c8c170af956619a9832d7ff5a8cb437fff9b4da54e32ccb553c22eaa78",
"b389326411e8fdcdc28c8ff2c8d1bb96bd9ee47d1e842d6855da5e46aa82ff42",
"66b4ef586c05db3748c9703bd0b2ea07b627e847d653b7aea9e7c6067ee94942",
"3cccf46cecf4f29d8729ba4619fd246b98af32efc9eccd82955f4b3a9117084b",
        ], [
"265affbed17b2e7bf8f1f1674ee2b5e4df8f94b63da296e6b1399f8612d67224",
"0ce6d5fef2aa6dcbee3836da27927be205e5ef293d8110e3a6bb683c5ab03307",
"f905e2cfa22342282ba96c517d4a312b47222fec94c2438492f0653d361b4798",
"2df3ee157bea2a10db450426efdf8c8e09b825fea293a93988e48b4e7c392f00",
"ef4ddaa976ceeaf48dac9e769b2e11e9adf281b4c8107b6c11114c617776b406",
        ], [
"068c07d784663308e53c4cbb8d030c09155640f2b29c80e55d6f9c9250f96938",
"7cd059a1236216d01c0b6eb1bfc498fdba92ad9b35f1952290528aceb01f2e7b",
"8241202843de3638dc4eabfa9ba4bb1fcd583ed41613716f22b284f71f23866e",
"1d9ffe1bf02362267a7347540daf49c17cfe759fce0d6dad80805ad8ff359678",
"a1a6ed4088a2ce83a78caf90ce2dc79eb6402d8bfee8eefdef1a44e832f68365",
        ], [
"97a281e2545a9de414f544cb9830d8cf7975a0c02c756d1631d6d5ca364de637",
"f12c27746e758baf98829e787a3351654f2680128be5a393325e2e4f71a4c94a",
"4fcb2e817eda1c8874bcd34cf38537f26fe5dbd4856d1d75b7f2fabb72503bc6",
"ee1137915736672af7368617b0c018597e70dc688c6514ee17c1bc9c78469108",
"47db4fcb16e258d02ce0e8bd5a0f737f14e1caf9d78c05008703f8980190f0c0",
        ], [
"857aede049b75cc9e877a24722626c91a85cc62851cfe7494660095cebd6b516",
"ef76d2d5ed8dd0a7ccecc48b3623f589df0b31e461276b79183b25d43fc8d52c",
"4ffb1be9de0c203e472d8061445b5bbdbbedfd4746a9de810404c079f0e7954a",
"fb84621dc915729c2b5f439f4e2d82a4acc5d4399d1122b21b2da5292036e017",
"8a9bf15b8f33a3f1a83b94e36e9c0c24b2bf6bb1178485b7d9e679e40c56640d",
        ], [
"acd4f4526043d14c626e752bd9e46e3acc67468c077b9c09bb85c2ca2849be16",
"0fd69b2fe624b85d4d313c3679ec1dacafd005c7002e925f56940d872f339c7b",
"c87f53b379da8ee65258c6b398d6d72637280ed3760c5ebf61c92419ef963ec0",
"8cfa6f50f3fd35c653e45dd8d6babf483c0d8e7f219953b8cb148dc5e5e91016",
"0205fe0fa6796de6e212e8c9994f8cb71e5b48cca64340d77742ea57b753f338",
        ], [
"7df1a00a4271e729f336af36ff8e20f14af5259f0aa1ce1f77ec4ee9ae004130",
"20798df98a93c43e5f95b1bda47bac292be89384de6039c614f026dc520dff68",
"404b819a79f76b5be9178fe5a14f300b135cf83313d1e3bec440d706ccd04080",
"fe78461564f462328e61640690f1481ac29ca83239891e5967f10d38aac0e658",
"d4259f680aaecf8957f04a9e55bf5c37ff11f3c8a3abed884eb25cec775f7e6b",
        ], [
"6a72fec38380c931b3d8cab9543ba041af9e20c73b8b086c8d08a711455dfe3e",
"9a242966308adc876b1a00db83f3f8144e8b942a3afa498cebc676cd2f5ea16e",
"a176318ca194e5ffdf3bfc639ff7e8c194ddb8250ae43823780e110d510bbb17",
"8a1cda96b144a0d39e070c5a6d741847f1e877348b95b3dbcf8a29ebb613d44c",
"23d10eb01d1d574f04a429c11b1a0478ac7302b83b4d161ff544270251affda8",
        ], [
"02c1b0268c67747a76ca378e2d7e93a6de1083de5e24f57074c960627292b431",
"d08cddf88afef688c60f2c8a9718a458cac8ccd39269099753366518fee08575",
"8c1b9d64b8ea71970453fd38f771772e1628168181be107a2891f782d408938d",
"c1973b4cf687a2b59399e7c3b6f7edbfc159a23a2ff2595f8a44c2ba6e072d1a",
"607f822fe40334f1ce90c467e33db59268969764efdc9cc861873d0f9be5e1e3",
        ], [
"85b8bc942b9a68e4417452af0d59ca0f611744e6a5248c238166d37a2c0bfa0c",
"3903e118bfdc70203e7dce44a05438855ade072b7280fb47784404e5bb52115b",
"8a1c7a17baf71b7219f4021cad96e39dcc3842a04c42a4be4e53cc04db0ceef5",
"7e6e7c8911a6a8d3ff80324a7f051ca7aa83fdbe7d6580ccee14fe2be1171439",
"8d865c6dba70aeb9aa0de9a4d21a9a7171b10831a1e849c9039fb2b65993e76e",
        ], [
"072a97460c465e2b6896fedb93fac4f6c2955a57c0488b3e086eac2a9e714e3c",
"bfd29c67c5d2e30903c1734ef38364f7530af59b08ecc52e9d204e05c7ad270e",
"2bc2b8f386cebb93e3924018cb63c0e803548e499539110033ec00cf4f16e0e7",
"4b8a9308e575fb549c300c0dabd1f835432e581d0c58ece1e039104ee5e37c10",
"8167c8df36cd17c9abfbbcafdc9ec0579cd36ee2e6a4b53e0d1f7df1a8492d34",
        ], [
"6548f8d6120770714a7c3e266c3f7d06f64eb78cee2b878f1f1b268dfa558e29",
"5c7c7d5ecef71cafaa97ca1257986522c21e116daf8dc74cf716a67c50b07d50",
"89cbb2d7a029af9c60481a0593e46cd6d2ed8639fb99fdd0cdd54500b136583a",
"9491d56f59b71b06dab44959074447cf673ca7f6eafba946e0a212b8aa8d2732",
"f1e8cfbdf3e7aff1740cfdb5aae772a3afefb1706237bd9109127157a6a4b3c0",
        ], [
"61cebf8acf562c79813e85102fa9f310fdd1525490a0a1f324e427649d4b3524",
"a84fad2c58abe53609e745b2ddda507d4d1c58a4a77973ef55d7e81884d18211",
"725a9db4fb2cd41ae8e15b7f5662e57197f2c9f3ea3fd799ee99466d2a149eaa",
"1207ae7f8c51a690c60f4a31dd3141169f57a9c3eb37a872de27c9c5eadae00e",
"90b406707f7f32ea205e524462f83cd9846420c48cb29bb659db97426241c8fa",
        ]
    ];

    #[test]
    fn ECDH() {
        for i in 0..KAT_ECDH.len() {
            let sk = PrivateKey::decode(&hex::decode(KAT_ECDH[i][0]).unwrap()).unwrap();
            let peer1 = hex::decode(KAT_ECDH[i][1]).unwrap();
            let refkey1 = hex::decode(KAT_ECDH[i][2]).unwrap();
            let peer2 = hex::decode(KAT_ECDH[i][3]).unwrap();
            let refkey2 = hex::decode(KAT_ECDH[i][4]).unwrap();
            let (key1, ok1) = sk.ECDH(&peer1);
            assert!(ok1 == 0xFFFFFFFF);
            assert!(key1[..] == refkey1);
            let (key2, ok2) = sk.ECDH(&peer2);
            assert!(ok2 == 0x00000000);
            assert!(key2[..] == refkey2);
        }
    }
}
