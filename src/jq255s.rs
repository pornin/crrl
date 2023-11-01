//! Jq255s implementation.
//!
//! This module implements generic group operations on the jq255s
//! group, which is itself isomorphic to a subgroup of the
//! double-odd elliptic curve of equation `y^2 = x*(x^2 - x + 1/2)` over
//! the finite field GF(2^255 - 3957). This group is described
//! on the [double-odd site]. The group has a prime order order `r`
//! (an integer slightly above 2^254). A conventional base point is
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
use super::field::{GF255s, ModInt256};
use super::blake2s::Blake2s256;
use super::{CryptoRng, RngCore};

/// An element in the jq255s group.
#[derive(Clone, Copy, Debug)]
pub struct Point {
    // We use extended coordinates on the Jacobi quartic curve with
    // equation: e^2 = (a^2 - 4*b)*u^4 + u^2 + 1
    // The map from the base curve is defined as:
    //   u = x/y
    //   e = u^2*(x - b/x)
    // For the point (0,0) (the neutral in the jq255s group, which is the
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
    E: GF255s,
    U: GF255s,
    Z: GF255s,
    T: GF255s,
}

/// Integers modulo r = 2^254 + 56904135270672826811114353017034461895.
///
/// `r` is the prime order of the jq255s group.
pub type Scalar = ModInt256<0xDCF2AC65396152C7, 0x2ACF567A912B7F03,
                            0x0000000000000000, 0x4000000000000000>;

impl Scalar {
    /// Encodes a scalar element into bytes (little-endian).
    pub fn encode(self) -> [u8; 32] {
        self.encode32()
    }
}

impl Point {

    /// The group neutral element.
    pub const NEUTRAL: Self = Self {
        E: GF255s::MINUS_ONE,
        Z: GF255s::ONE,
        U: GF255s::ZERO,
        T: GF255s::ZERO,
    };

    /// The conventional base point (group generator).
    ///
    /// This point generates the whole group, which as prime order r
    /// (integers modulo r are represented by the `Scalar` type).
    pub const BASE: Self = Self {
        E: GF255s::w64be(
            0x0F520B1BA747ADAC, 0x55E452A64612D10E,
            0x6D7386B2348CC437, 0x104220CDA2789410),
        Z: GF255s::ONE,
        U: GF255s::w64be(0, 0, 0, 3),
        T: GF255s::w64be(0, 0, 0, 9),
    };

    /* unused
    /// The curve `a` constant (-1).
    const A: GF255s = GF255s::MINUS_ONE;
    /// The curve `b` constant (1/2).
    const B: GF255s = GF255s::w64be(
        0x3FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF846);
    */

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
        let (u, mut r) = GF255s::decode32(buf);

        // e^2 = (a^2-4*b)*u^4 - 2*a*u^2 + 1
        let uu = u.square();
        let ee = -uu.square() + uu.mul2() + GF255s::ONE;
        let (e, r2) = ee.sqrt();
        r &= r2;
        // GF255s::sqrt() already returns the non-negative root, we do
        // not have to test the sign of e and adjust.

        // We have the point in affine coordinates, except on failure,
        // in which case we have to adjust the values.
        self.E = GF255s::select(&GF255s::MINUS_ONE, &e, r);
        self.Z = GF255s::ONE;
        self.U = GF255s::select(&GF255s::ZERO, &u, r);
        self.T = GF255s::select(&GF255s::ZERO, &uu, r);
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
        let iZ = GF255s::ONE / self.Z;
        let mut u = self.U * iZ;
        let sgn = (((self.E * iZ).encode()[0] & 1) as u32).wrapping_neg();
        u.set_cond(&-u, sgn);
        u.encode()
    }

    /// Creates a point by converting a point in extended affine
    /// coordinates (e, u, u^2).
    fn from_affine_extended(P: &PointAffineExtended) -> Self {
        Self {
            E: P.e,
            Z: GF255s::ONE,
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
        //      a' = -2*a          (jq255s: a' = 2)
        //      b' = a^2 - 4*b     (jq255s: b' = -1)
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
        let hd = z1z2 + t1t2;   // Z1*Z2 - (a^2 - 4*b)*T1*T2
        let T3 = eu.square();
        let Z3 = hd.square();
        let u1u2d = u1u2.mul2();
        let E3 = (z1z2 - t1t2) * (e1e2 + u1u2d) - u1u2d * zt;
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
        //      a' = -2*a          (jq255s: a' = 2)
        //      b' = a^2 - 4*b     (jq255s: b' = -1)
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
        let hd = Z1 + t1t2;   // Z1*Z2 - (a^2 - 4*b)*T1*T2
        let T3 = eu.square();
        let Z3 = hd.square();
        let u1u2d = u1u2.mul2();
        let E3 = (Z1 - t1t2) * (e1e2 + u1u2d) - u1u2d * zt;
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
        let (E, Z, U, T) = (&self.E, &self.Z, &self.U, &self.T);

        // P ezut -> 2*P+N xwj  (1M+3S)
        //    uu = U^2
        //    X  = 8*(uu^2)
        //    W  = 2*uu - (T + Z)^2  # -(T^2 + Z^2)
        //    J  = 2*E*U
        let uu = U.square();
        let J = E * U.mul2();
        let X = uu.square().mul8();
        let W = uu.mul2() - (T + Z).square();

        // P xwj -> P ezut  (3S)
        //    ww = W^2
        //    jj = J^2
        //    E  = 2*X - ww - jj
        //    Z  = ww
        //    U  = ((W + J)^2 - ww - jj)/2  # Or: U = W*J
        //    T  = jj
        let ww = W.square();
        let jj = J.square();
        self.E = X.mul2() - ww - jj;
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
        let (E, Z, U, T) = (&self.E, &self.Z, &self.U, &self.T);

        // P ezut -> 2*P+N xwj  (1M+3S)
        //    uu = U^2
        //    X  = 8*(uu^2)
        //    W  = 2*uu - (T + Z)^2  # -(T^2 + Z^2)
        //    J  = 2*E*U
        let uu = U.square();
        let mut J = E * U.mul2();
        let mut X = uu.square().mul8();
        let mut W = uu.mul2() - (T + Z).square();

        // Subsequent doublings in xwj  (n-1)*(2M+4S)
        for _ in 1..n {
            // t1 = W*J
            // t2 = t1^2
            // X' = 8*t2^2
            // t3 = (W + J)^2 - 2*t1  # W^2 + J^2
            // W' = 2*t2 - t3^2
            // J' = 2*t1*(2*X - t3)
            // We scale down J' by 1/2, W' by 1/2, X' by 1/4
            let t1 = W * J;
            let t3 = (W + J).square();
            let t2 = t1.square();
            let t4 = t3 - t1.mul2();
            let t5 = X.mul2() - t4;
            X = t2.square().mul2();
            let t6 = t4.square();
            J = t1 * t5;
            W = t2 - t6.half();
        }

        // Conversion xwj -> ezut  (3S)
        let ww = W.square();
        let jj = J.square();
        self.E = X.mul2() - ww - jj;
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
    fn map_to_curve(f: &GF255s) -> Self {
        // We map to the dual curve E(a',b') with:
        //   a' = -2*a = 2
        //   b' = a^2 - 4*b = -1

        // yy1num = -2*f^6 + 14*f^4 - 14*f^2 + 2
        // yy2num = -yy1num*f^2
        // xden = 1 - f^2
        // yden = (1 - f^2)^2
        let ff = f.square();
        let yy1num = ((GF255s::w64be(0, 0, 0, 14) - ff.mul2()) * ff
                     - GF255s::w64be(0, 0, 0, 14)) * ff
                     + GF255s::w64be(0, 0, 0, 2);
        let yy2num = -yy1num * ff;
        let xden = GF255s::ONE - ff;
        // unused: let yden = xden.square();

        // If yy1num is square, set xnum = -2 and ynum = sqrt(yy1num)
        // Otherwise, set xnum = 2*f^2 and ynum = -sqrt(yy2num)
        let nqr = (yy1num.legendre() >> 1) as u32;
        let xnum = GF255s::select(
            &GF255s::w64be(
                0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
                0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF089),
            &ff.mul2(), nqr);
        let (mut ynum, _) = GF255s::select(&yy1num, &yy2num, nqr).sqrt();
        ynum.set_cond(&-ynum, nqr);

        // We have x = xnum / xden and y = ynum / yden, with yden = xden^2.
        // Therefore: u = x/y = xnum*xden / ynum
        let unum = xnum * xden;
        let uden = ynum;

        // Note: special cases:
        //   If f = 1 or -1, then:
        //      xnum = -2, xden = 0, unum = 0, uden = 0
        //   If f != 1 and -1 but ynum = 0, then:
        //      xnum = -2, xden != 0, unum != 0, uden = 0
        // For these cases, we want to map to the neutral point.
        // In all other cases, uden != 0.
        let to_fix = uden.iszero();

        // Apply the theta_{1/2} isogeny to get back to curve E[a,b].
        //   x' = 4*b*u^2
        //   u' = 2*x/(u*(x^2 - b'))
        let mut Xnum = unum.square().mul2();
        let mut Xden = uden.square();
        let Unum = uden.mul2();
        let Uden = xnum.square() + xden.square();

        // If we are in a special case, then Unum and Uden are correct
        // (Unum = 0, Uden != 0), but Xnum and Xden must be fixed.
        Xnum.set_cond(&GF255s::ZERO, to_fix);
        Xden.set_cond(&GF255s::ONE, to_fix);

        // Compute the 'e' coordinate with e = (x^2 - b)/(x^2 + a*x + b).
        let t1 = Xnum * (Xnum.mul2() - Xden);
        let t2 = Xden * (Xnum - Xden);
        let Enum = t1 + t2;
        let Eden = t1 - t2;

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
        let f1 = GF255s::decode_reduce(&blob1);
        let f2 = GF255s::decode_reduce(&blob2);
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

    /// Recodes a scalar into 52 signed digits.
    ///
    /// Each digit is in -15..+16, top digit is 0 or 1.
    fn recode_scalar(n: &Scalar) -> [i8; 52] {
        let mut sd = [0i8; 52];
        let bb = n.encode();
        let mut cc: u32 = 0;       // carry from lower digits
        let mut i: usize = 0;      // index of next source byte
        let mut acc: u32 = 0;      // buffered bits
        let mut acc_len: i32 = 0;  // number of buffered bits
        for j in 0..52 {
            if acc_len < 5 && i < 32 {
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

    /// Multiplies this point by a scalar (in place).
    ///
    /// This operation is constant-time with regard to both the points
    /// and the scalar value.
    pub fn set_mul(&mut self, n: &Scalar) {
        // Make a 5-bit window: win[i] contains (i+1)*P
        let mut win = [Self::NEUTRAL; 16];
        win[0] = *self;
        for i in 1..8 {
            let j = 2 * i;
            win[j - 1] = win[i - 1].double();
            win[j] = win[j - 1] + win[0];
        }
        win[15] = win[7].double();

        // Recode the scalar into 52 signed digits.
        let sd = Self::recode_scalar(n);

        // Process the digits in high-to-low order.
        *self = Self::lookup(&win, sd[51]);
        for i in (0..51).rev() {
            self.set_xdouble(5);
            self.set_add(&Self::lookup(&win, sd[i]));
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

        // We process four chunks in parallel. Each chunk is 13 digits.
        *self = Self::from_affine_extended(
            &Self::lookup_affine_extended(&PRECOMP_B, sd[12]));
        self.set_add_affine_extended(
            &Self::lookup_affine_extended(&PRECOMP_B65, sd[25]));
        self.set_add_affine_extended(
            &Self::lookup_affine_extended(&PRECOMP_B130, sd[38]));
        self.set_add_affine_extended(
            &Self::lookup_affine_extended(&PRECOMP_B195, sd[51]));

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

    /// 5-bit wNAF recoding of a scalar; output is a sequence of 256
    /// digits.
    ///
    /// Non-zero digits have an odd value, between -15 and +15
    /// (inclusive). (The recoding is constant-time, but use of wNAF is
    /// inherently non-constant-time.)
    fn recode_scalar_NAF(n: &Scalar) -> [i8; 256] {
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
        // Since r < 2^255, only 256 digits are necessary at most.

        let mut sd = [0i8; 256];
        let bb = n.encode();
        let mut x = bb[0] as u32;
        for i in 0..256 {
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
        // Recode the scalars in 5-bit wNAF.
        let sdu = Self::recode_scalar_NAF(&u);
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
        for i in (0..256).rev() {
            // We have one more doubling to perform.
            ndbl += 1;

            // Get next digits. If they are all zeros, then we can loop
            // immediately.
            let e1 = sdu[i];
            let e2 = sdv[i];
            if ((e1 as u32) | (e2 as u32)) == 0 {
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
            let e3 = if i < 126 { sdv[i + 130] } else { 0 };
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

/// A jq255s private key.
///
/// Such a key wraps around a secret non-zero scalar. It also contains
/// a copy of the public key.
#[derive(Clone, Copy, Debug)]
pub struct PrivateKey {
    sec: Scalar,                // secret scalar
    pub public_key: PublicKey,  // public key
}

/// A jq255s public key.
///
/// It wraps around a jq255s element, but also includes a copy of the
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
        self.sec.encode()
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
        // negligible bias because the jq255s order is close enough to
        // a power of 2.
        let mut sh = Blake2s256::new();
        sh.update(&self.sec.encode());
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
        sig[16..48].copy_from_slice(&s.encode());
        sig
    }

    /// ECDH key exchange.
    ///
    /// Given this private key, and the provided peer public key (encoded),
    /// return the 32-byte shared key. The process fails if the `peer_pk`
    /// slice does not have length exactly 32 bytes, or does not encode
    /// a valid jq255s element, or encodes the neutral element. On success,
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
        let alt = self.sec.encode();
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
    /// or the bytes do not encode a valid jq255s element, or the bytes
    /// encode the neutral element, then the process fails and this
    /// function returns `None`. Otherwise, the decoded public key
    /// is returned.
    pub fn decode(buf: &[u8]) -> Option<PublicKey> {
        let point = Point::decode(buf)?;
        if point.isneutral() != 0 {
            None
        } else {
            let mut encoded = [0u8; 32];
            encoded[..].copy_from_slice(&buf[0..32]);
            Some(Self { point, encoded })
        }
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
    e: GF255s,
    u: GF255s,
    t: GF255s,
}

impl PointAffineExtended {

    const NEUTRAL: Self = Self {
        e: GF255s::MINUS_ONE,
        u: GF255s::ZERO,
        t: GF255s::ZERO,
    };
}

// Points i*B for i = 1 to 16, affine extended format
static PRECOMP_B: [PointAffineExtended; 16] = [
    // B * 1
    PointAffineExtended {
        e: GF255s::w64be(0x0F520B1BA747ADAC, 0x55E452A64612D10E,
                         0x6D7386B2348CC437, 0x104220CDA2789410),
        u: GF255s::w64be(0x0000000000000000, 0x0000000000000000,
                         0x0000000000000000, 0x0000000000000003),
        t: GF255s::w64be(0x0000000000000000, 0x0000000000000000,
                         0x0000000000000000, 0x0000000000000009),
    },
    // B * 2
    PointAffineExtended {
        e: GF255s::w64be(0x7DCAB2C9EFDB7348, 0xA479391BDE7F0296,
                         0xC474598BF46D0A34, 0x155215A00E9EA08D),
        u: GF255s::w64be(0x6F44EC749C5869ED, 0x256C2BE75887FD30,
                         0xE5427944219E490E, 0xB3E22F8D0D1657FC),
        t: GF255s::w64be(0x42B401D3D5F7C6BD, 0x2501ACD978762D61,
                         0xEC33C75916FEEF18, 0x84CC11AA69B07916),
    },
    // B * 3
    PointAffineExtended {
        e: GF255s::w64be(0x407A13E6346F39AA, 0xFAC101D18780329D,
                         0xE7931687A657BB05, 0x1C2AE84D369F681F),
        u: GF255s::w64be(0x7204233F36F07204, 0x233F36F07204233F,
                         0x36F07204233F36F0, 0x7204233F36F06441),
        t: GF255s::w64be(0x3E21C102D9DADBFA, 0x10163C9F79D25F29,
                         0x9F6417AAAE3D0228, 0xB582FD5064ED4BDD),
    },
    // B * 4
    PointAffineExtended {
        e: GF255s::w64be(0x1153E6DF83525815, 0xADBF0638585A38A4,
                         0x879B715FCBC752B5, 0xFCB9E72D0A0C8EFE),
        u: GF255s::w64be(0x54E64904662F1657, 0xF2A1145C9F5D3475,
                         0x4E645F874B12D8E7, 0x9204A59E69223E39),
        t: GF255s::w64be(0x11F4E2818B86CA68, 0xAB4BE93A7FC59C00,
                         0x013E9EBA0D1CB6FA, 0x330B2867F87650FD),
    },
    // B * 5
    PointAffineExtended {
        e: GF255s::w64be(0x67D128B9D764C85E, 0x850ECBC936A0B118,
                         0x89E1BB43C57ADC57, 0xDE7D4909B7768768),
        u: GF255s::w64be(0x52932B9A9F0CC65D, 0xC17C3E9333A6D7CE,
                         0x58856B292FBA673A, 0xDF0337C00667B64D),
        t: GF255s::w64be(0x591738A93D12426F, 0x3B8EBB2DDCBD289A,
                         0x635DC68819778FD9, 0x1C824AAF423C38C3),
    },
    // B * 6
    PointAffineExtended {
        e: GF255s::w64be(0x1AE40CE8E4758D2C, 0x3F0A5A3647C0974E,
                         0xBAF1C9AE563AB60C, 0xDB908132D0DC5ED9),
        u: GF255s::w64be(0x7B55CE1D18E39726, 0x05D59CECEE1D7E76,
                         0xF71199A7E92DA598, 0x2378FCE7659F8304),
        t: GF255s::w64be(0x42B373007F5A5422, 0x002552F2D3C7EDD4,
                         0xFB8008F322A48DBC, 0x5C964B33DBE905C9),
    },
    // B * 7
    PointAffineExtended {
        e: GF255s::w64be(0x778691038A25A05C, 0x1F836A920FFADF5D,
                         0x8FF2EAAA7A24E5FE, 0xBCF616435A305CD1),
        u: GF255s::w64be(0x6EB76B2647D95DA4, 0x150E35D2C3D060D0,
                         0x62AE8CABB5C7CED6, 0xBB70A3099712F248),
        t: GF255s::w64be(0x59A0770044734AB4, 0x6A2EACE6BC1B7A38,
                         0x64F44E4B72EDC8DB, 0x5429F9F2E42EC752),
    },
    // B * 8
    PointAffineExtended {
        e: GF255s::w64be(0x4CB7E380E64F8650, 0x188E0332E4A5FB16,
                         0x342B4BA3A435E2AC, 0x1EA227FB56AF9D20),
        u: GF255s::w64be(0x26DCD74FEC331ED1, 0x4BA8485FACF9CC03,
                         0x13A2A2DE4BD9AB93, 0x2DE04D6F93F6EEA0),
        t: GF255s::w64be(0x08F2F46F791D0D36, 0x27641FC9B91D9EE2,
                         0xD79E9196AC529912, 0x008F881CFF3A091D),
    },
    // B * 9
    PointAffineExtended {
        e: GF255s::w64be(0x186E71D5F58996EC, 0x28DAB54C47465363,
                         0x9E0262DD1BBD9331, 0x7064AA193867E249),
        u: GF255s::w64be(0x641BC3EAE8B16348, 0xC25DB7DB3DC0038B,
                         0x6544BF679F64AADF, 0xF293B01428B2AEF7),
        t: GF255s::w64be(0x5344F6CBF78889DE, 0xC30E7E91EBC5E399,
                         0x5AFEBCBFF210B104, 0xC9769297553D878E),
    },
    // B * 10
    PointAffineExtended {
        e: GF255s::w64be(0x24D9FE2BF2E5C1DF, 0xF1BEAC3346435A2C,
                         0x55781708B257A28D, 0xEC41FB9869A38BAF),
        u: GF255s::w64be(0x11EB002E4DD0CC8A, 0x15C5CFFBF6CC3634,
                         0xA5FBAFCA4BE098FF, 0xAC2450CD35E0E33E),
        t: GF255s::w64be(0x3361CE26431C24BB, 0xFE443CA5E072548A,
                         0xA1D9E1AC087CD2FE, 0xB57551F40B3E9878),
    },
    // B * 11
    PointAffineExtended {
        e: GF255s::w64be(0x01511890814FCFF1, 0xB18CAB0F6A54EB86,
                         0xA02A4D6FD4645856, 0x640788E8FDB68B79),
        u: GF255s::w64be(0x44204FCFD6AF1B44, 0xD7761D7533C3A9ED,
                         0x343EAC1284D73CA8, 0x4845EE2F9FDD13DE),
        t: GF255s::w64be(0x10E31B1DBFBE67E2, 0xF25BD01AAB8AE0C4,
                         0xDEC8E3E4B025867C, 0x88C209C7EA04605B),
    },
    // B * 12
    PointAffineExtended {
        e: GF255s::w64be(0x42B172782E25AB56, 0x8AB438671165365F,
                         0x3E87D66869F1BBA3, 0xC31D48EEAC5599AB),
        u: GF255s::w64be(0x1EA722FF296F8D92, 0xC2A3F9623BA7F9DB,
                         0x9BBB9D365457C4A7, 0xEDFD8B4BAD8160D1),
        t: GF255s::w64be(0x466ED47592E58063, 0xA86F76D59422C162,
                         0x060E2D9DB05F1736, 0xDBD377FD4727F1F8),
    },
    // B * 13
    PointAffineExtended {
        e: GF255s::w64be(0x72491B477910438F, 0xDB27E2AA724C430C,
                         0xCC5F0E316B1677A5, 0xFFAD311FC42DF019),
        u: GF255s::w64be(0x56187141071FC1D5, 0xA8C80C9E1E78A79B,
                         0x12E5410DA8EA7661, 0x4DEC233876049D71),
        t: GF255s::w64be(0x3E2BE3EF9C950F96, 0x37A78C21BA3C4BCD,
                         0xFADED89F6751C8BA, 0x5F87D7B161AB363D),
    },
    // B * 14
    PointAffineExtended {
        e: GF255s::w64be(0x5E430A47DD41A2D1, 0xA0040674E6CCAC7F,
                         0xB119CC9192E72DD0, 0xB1620E564FE98F30),
        u: GF255s::w64be(0x60EF0D216D3123EB, 0x2EDB82538E0CEEAE,
                         0xC3E0CD34567E3C9A, 0x0693069387390957),
        t: GF255s::w64be(0x681E3041257F38AE, 0x917BC5415D3B524A,
                         0x18B961A309C8339D, 0xCEB292E0EC603124),
    },
    // B * 15
    PointAffineExtended {
        e: GF255s::w64be(0x3BE6B06EB08C48A2, 0x51EBFEAAC016B5CD,
                         0x98B526861DE23B6B, 0x871FFDDBCBB41565),
        u: GF255s::w64be(0x100615BCCFB9A5A5, 0x5594A4E0449B9EFD,
                         0x99E3DE25AE95E20B, 0x3010093957EA5D5A),
        t: GF255s::w64be(0x172AAB628A8B99AA, 0x79BCF0468E59F7E5,
                         0x07A2C5FB49523FF8, 0x2CB15E4AD534F12E),
    },
    // B * 16
    PointAffineExtended {
        e: GF255s::w64be(0x2DE21A5F5B5F3676, 0xE6355FA5083B2FE3,
                         0xEE888DF3C9AF901F, 0x18BF1FBA94943C22),
        u: GF255s::w64be(0x1880F6E65E61EDCA, 0xF17032C1930CC3DF,
                         0x70631241B6132ED0, 0xFE9991C8BEE08226),
        t: GF255s::w64be(0x5C75B961A2BDDB5E, 0xB6548C27E54A4C3C,
                         0xB0627DFEE2C9B21E, 0x7D5A50285053E953),
    },
];

// Points i*(2^65)*B for i = 1 to 16, affine extended format
static PRECOMP_B65: [PointAffineExtended; 16] = [
    // (2^65)*B * 1
    PointAffineExtended {
        e: GF255s::w64be(0x145C3F79A555FEFD, 0xE59F700CF1498B64,
                         0xB7D10E9710F4FB5A, 0xAD671792FC850F1F),
        u: GF255s::w64be(0x3AA6C1241FDF29EA, 0x59EF9CAA0E041627,
                         0xC3F5757559BEB30B, 0xC504DF70301870EC),
        t: GF255s::w64be(0x6F9938F04DD20707, 0x5728E3A866E7AD73,
                         0x7E905634B7B83754, 0x6D1B6960A73A5D82),
    },
    // (2^65)*B * 2
    PointAffineExtended {
        e: GF255s::w64be(0x2A078A32D8A60452, 0x450F6EC619A10A0D,
                         0xF3B3A271D9C85032, 0xAEA44289F769D548),
        u: GF255s::w64be(0x0C4B1FDF2A2835EA, 0xE555C4B2316756DE,
                         0xBEEAEA12D994BBBA, 0xE395C342A49AB090),
        t: GF255s::w64be(0x5138EFFC5F9EE71D, 0x2B40BE24F305035E,
                         0x3B828FBF6A20F966, 0xACC1B6C0E71A2679),
    },
    // (2^65)*B * 3
    PointAffineExtended {
        e: GF255s::w64be(0x6F9AECFE00B2FBF9, 0xC58DE35B727B63A1,
                         0x2EB5074C5FB663E6, 0xFCB8737061C3B389),
        u: GF255s::w64be(0x05DADA9DFEACCC60, 0x84993635937FCD8B,
                         0x8235DE8C4CCCB7AD, 0xDD2736E064D47F35),
        t: GF255s::w64be(0x367779B57A3314D7, 0x146AA2BAF6EC5EF7,
                         0x128FA18D8F430D4E, 0x041CF837CCDEEA2C),
    },
    // (2^65)*B * 4
    PointAffineExtended {
        e: GF255s::w64be(0x168FE09EBFF84166, 0xB844A3A38291AF02,
                         0x07B12ACEF6DB6D43, 0x02E353146AB767DD),
        u: GF255s::w64be(0x637DD68BAA0DCC5D, 0x39FB3501747B5584,
                         0xB05A144640E6146F, 0x20CCC8942E0DD4AF),
        t: GF255s::w64be(0x081DAA0CD8F74B16, 0x5B82CA8E72C504DC,
                         0x1CD2F537C049D501, 0xF858BE49C8D19D78),
    },
    // (2^65)*B * 5
    PointAffineExtended {
        e: GF255s::w64be(0x12B18E684B0A57E8, 0x0930AFE6F7D251E0,
                         0x9B679600DF4B5715, 0xEAAC278EE492D81E),
        u: GF255s::w64be(0x0430DD4DB72FDAE3, 0x355389746FE3860D,
                         0x034912BE87333A80, 0x8E3257DDE0B61BA4),
        t: GF255s::w64be(0x6CE5827F833EE541, 0x80E1FCACE9A31DF3,
                         0x9DB06D02B9CA5DEA, 0x8F69AB4C10703FE2),
    },
    // (2^65)*B * 6
    PointAffineExtended {
        e: GF255s::w64be(0x18BD1D68A45DEDDC, 0x9A382C9BFE2755E0,
                         0xF9ABC6FE17A925CB, 0x296B5AC7D0BE8BD1),
        u: GF255s::w64be(0x6BC51220E2A6FEA1, 0x3C89F25AEBEFEA23,
                         0xEB9F7DAEFD0E5177, 0x41C6037317995DE4),
        t: GF255s::w64be(0x5198473D3A06C5F9, 0xF3632B4CFD05C7A5,
                         0x46B4997F9117FF61, 0xA8BE5154DC0850D8),
    },
    // (2^65)*B * 7
    PointAffineExtended {
        e: GF255s::w64be(0x2C841A6C3AE0F758, 0x4E0E09389BAC268D,
                         0x16DFAAD15E54A341, 0x0C025505E54EC194),
        u: GF255s::w64be(0x6C76A36690BF3721, 0x68532DB386C5311B,
                         0xDE22614DA09D7B7D, 0x6E0C6EC00B45CB36),
        t: GF255s::w64be(0x2C44B584B042AE8F, 0xCC0911EBF7E93EA0,
                         0xBF20C5FEEE045148, 0xEF4DC9C561021E16),
    },
    // (2^65)*B * 8
    PointAffineExtended {
        e: GF255s::w64be(0x32F8215138DF8DF9, 0xADEE45CF23AA330B,
                         0xA9BB3C1344A2B7AC, 0x0D409883C7DFAAFF),
        u: GF255s::w64be(0x320289A4D4AD38D8, 0x77F9F481B54C83A0,
                         0x2E0E5F286C358DFB, 0xCF5B38B898027EEE),
        t: GF255s::w64be(0x0A7DECD4F28D875D, 0xB56B41AAC63C7E31,
                         0x69338F651A87EED4, 0x850595A42D050368),
    },
    // (2^65)*B * 9
    PointAffineExtended {
        e: GF255s::w64be(0x6BFAD4535C93F764, 0x32D2ACB4DF4CBE2B,
                         0xF4121ED9B11E2F78, 0x3EA9CA530CD6C445),
        u: GF255s::w64be(0x6A58C4C447F58EDA, 0xE362CB1B14264D84,
                         0x2D6EB28373FFD2AB, 0xE9F73ACE30D70239),
        t: GF255s::w64be(0x05EA57EAA00332B5, 0x43D7F20CB3184DFA,
                         0x8C5750AC973B8F0F, 0xD421A3CDEB81C4F0),
    },
    // (2^65)*B * 10
    PointAffineExtended {
        e: GF255s::w64be(0x594955BFF388397E, 0xFC3A41A8997B623A,
                         0xCFA05077AC82CA7F, 0x6E80B93F8C55EFC5),
        u: GF255s::w64be(0x6F6BF0FD0A760CE6, 0xFDF4EBAD6DA9D7C5,
                         0xFDF2BE7E13D824B7, 0xBD4ABD3A1D23F58F),
        t: GF255s::w64be(0x5B172E38809C36C6, 0x3160D36F471E8803,
                         0xBDD6F1B6D18BC1A6, 0xB81C256655EADF3C),
    },
    // (2^65)*B * 11
    PointAffineExtended {
        e: GF255s::w64be(0x0D9EFEC28CA0A54C, 0xE1EC96F492FEAA7B,
                         0x9FE3E489FB0E0E43, 0x80DDD82DF53D8310),
        u: GF255s::w64be(0x3F27081B1A793D69, 0x7CB1DF78AE0F6520,
                         0x1ED56E4F26768EEA, 0xAC82BC835BEF5D82),
        t: GF255s::w64be(0x5FD614C1FCF6A3EE, 0xC51142A5B0691BD2,
                         0xDA034A47FB299F2D, 0x057471C1F93EB90E),
    },
    // (2^65)*B * 12
    PointAffineExtended {
        e: GF255s::w64be(0x1A664CA56EBCC5BD, 0x8173D517B7E0B2B0,
                         0xCA182B05C7FCE11E, 0xDBEFD63B52CB2A5C),
        u: GF255s::w64be(0x770D858C1138F9B1, 0xB0A6FCE83F628FFF,
                         0x744209C1D5AA1E98, 0x3C2CD45E00488C1B),
        t: GF255s::w64be(0x10420882FFE4F21E, 0xA04AF9D178D61F3C,
                         0x733C7CA714153EC3, 0x6A96C56588D37A03),
    },
    // (2^65)*B * 13
    PointAffineExtended {
        e: GF255s::w64be(0x035DE04A61BABE1E, 0xC3729090E26F9B9C,
                         0x7B6D5AAC61E9960D, 0x855811311CCCC730),
        u: GF255s::w64be(0x3EA45CDA4A4335B5, 0xF578275E6C3BD8ED,
                         0x5176B2243AD75522, 0x2B0FECC1159B7C7E),
        t: GF255s::w64be(0x70977639FA69E967, 0xEA10679B009BFFEE,
                         0x7BB775A10203F23D, 0x399CEF5FFB8296AD),
    },
    // (2^65)*B * 14
    PointAffineExtended {
        e: GF255s::w64be(0x513E12D64D4EAAF8, 0x03B072171B964F50,
                         0xC10608CEDBFB267E, 0x969B92FA2E20E4A3),
        u: GF255s::w64be(0x52A2BF2E7D7202E9, 0xB2C96A085D26A03D,
                         0xC0B470A297317B12, 0xDCABCD1C2B792081),
        t: GF255s::w64be(0x43FC22627E870D74, 0x8A3F159339E5051E,
                         0x8164FFF4195F29D9, 0x92EA95A74111A128),
    },
    // (2^65)*B * 15
    PointAffineExtended {
        e: GF255s::w64be(0x07D749F9FAB7DFA1, 0x899B72B33AED803B,
                         0x26E5E74F1817958B, 0x5B4B80D1C3DF479B),
        u: GF255s::w64be(0x5B29A9D46B6557D1, 0x436574C311E4EAE5,
                         0x95E7B3DAE6F438C6, 0xCB4E8E8682ED1F56),
        t: GF255s::w64be(0x04B77CA16130D747, 0xB279CEBEF2B0454A,
                         0xD3C936A9527B4E88, 0xA0FF9F128E2A8D1F),
    },
    // (2^65)*B * 16
    PointAffineExtended {
        e: GF255s::w64be(0x191D614E6AD8C891, 0x4E2ED8BC0B19751E,
                         0xE55F1507DB1386D1, 0xD82E7D79DB9A5B83),
        u: GF255s::w64be(0x00D98818FA1C5F43, 0x9698A42E4CDA26D4,
                         0xCFD4B9B4407ADF90, 0xA0A662EF07947CAE),
        t: GF255s::w64be(0x214C5B3CB2D89DF0, 0x31ED70E89727A4CA,
                         0xA09CE0B0CDA1B6AB, 0xF048C3E420D474E3),
    },
];

// Points i*(2^130)*B for i = 1 to 16, affine extended format
static PRECOMP_B130: [PointAffineExtended; 16] = [
    // (2^130)*B * 1
    PointAffineExtended {
        e: GF255s::w64be(0x5FEB95EE492D6D08, 0x7A7BA22355F5D398,
                         0x907B64252E9A0A54, 0xC547E3D2287B7A8C),
        u: GF255s::w64be(0x661229C7BADE4F32, 0xB14CDFC79E957BD2,
                         0x95542A3E6973F13C, 0xC17AB82D247C18A0),
        t: GF255s::w64be(0x02802A6A03E1A1E7, 0x2EC475815BC7D073,
                         0x5B912D60442F368E, 0x5541C59928E441B7),
    },
    // (2^130)*B * 2
    PointAffineExtended {
        e: GF255s::w64be(0x2B3CECA02688A0B0, 0x319418ED83354F29,
                         0x2AEBD6074550787C, 0x15A644D90F28E7B4),
        u: GF255s::w64be(0x17A619CA7283DADF, 0xEA24368F35C3A22D,
                         0xF443CA1C94A3CAEB, 0x72C27DF187129578),
        t: GF255s::w64be(0x649637A5340F9542, 0x4B3F9D68A82A6DEE,
                         0x1A6558F98FD7DD2D, 0x0EC06DB43459B171),
    },
    // (2^130)*B * 3
    PointAffineExtended {
        e: GF255s::w64be(0x168AAAD797CE7EDA, 0xD922BF8C18633561,
                         0xBF1723E466338796, 0xA065A1038F86DA38),
        u: GF255s::w64be(0x22A23BF61CDCF306, 0x86F51D53192BE4B6,
                         0x7A195520CEEAA6A9, 0x31D729A12CB400ED),
        t: GF255s::w64be(0x66C538B71AA1CCE1, 0xC73BF67BBC73A980,
                         0x8E87630B3381880A, 0xCACA700AE212FCC2),
    },
    // (2^130)*B * 4
    PointAffineExtended {
        e: GF255s::w64be(0x0DD0B15BCCECE2DE, 0x199AF9DFF3BB8550,
                         0xF5B27B3B050B1CC0, 0xA7C985807B396AC1),
        u: GF255s::w64be(0x188F708521AD21CA, 0xEBD2AC8233E1DE6F,
                         0xA2DC84A037DCA7EE, 0x767FEA8208A28FA3),
        t: GF255s::w64be(0x2BEF1130670533CF, 0x2FBD912DEC2F8A1A,
                         0xEF5048FF3B333AA4, 0x3AC52264BA423FFD),
    },
    // (2^130)*B * 5
    PointAffineExtended {
        e: GF255s::w64be(0x2392F5F42DA132F4, 0x0D8952811E9106B9,
                         0xF737A550BC29AA93, 0xCD8A44DA9FC7024F),
        u: GF255s::w64be(0x7F08F18F636EA208, 0xC66B48D80BE03B56,
                         0x5B4DD1BB342A7459, 0xB820181764875239),
        t: GF255s::w64be(0x67B29276BA073D7D, 0xB9CE10F55DE4C795,
                         0x14E86A7605B43C11, 0x2CC6020672019CF2),
    },
    // (2^130)*B * 6
    PointAffineExtended {
        e: GF255s::w64be(0x5C5A93242CA0CD5B, 0xEBB4F5A33EDD77B8,
                         0x69548FFF3BE07C7C, 0x43D22D5EF864F06F),
        u: GF255s::w64be(0x7D453CC948C9FA1D, 0xE1C94F9AB14804D7,
                         0xFB480B5678DA5C00, 0x9FFA28B92E2ECC30),
        t: GF255s::w64be(0x5A40858A46BA5349, 0x8FE833DB15F7685C,
                         0xAA8361C9702948B9, 0x99B10D2684BD090A),
    },
    // (2^130)*B * 7
    PointAffineExtended {
        e: GF255s::w64be(0x7D201AFC52734277, 0xFEB5667662B03E4E,
                         0xEC532CBC96052BB3, 0xFE714DB9C806064F),
        u: GF255s::w64be(0x1A086A6FE01B2E7D, 0xAF3F2A3FA6C81237,
                         0x48D2662133271AC2, 0x8784811955F5DEF5),
        t: GF255s::w64be(0x7444100416332425, 0xAAE6A9C9C053632B,
                         0x06226B3A465F8A77, 0xE60717C5AA0289C2),
    },
    // (2^130)*B * 8
    PointAffineExtended {
        e: GF255s::w64be(0x507DF5A8263956AC, 0xA0478BE33AECC482,
                         0x3FD38D2D5E9E8C93, 0x9C901178C3D8E685),
        u: GF255s::w64be(0x3DE43A8017A5525E, 0x2A8D65D0B2CAB143,
                         0xBF15C84C6245BBB3, 0x1DC9A2A859DBD92B),
        t: GF255s::w64be(0x1D2C24FA4A76A7C6, 0x2468A4B876BC1D61,
                         0x98B672949E66D5ED, 0x2040089013E8DD16),
    },
    // (2^130)*B * 9
    PointAffineExtended {
        e: GF255s::w64be(0x2875F4526CD0570A, 0x895F51290AB84B6A,
                         0x25107B05B78EFCC3, 0x4E0B3042A15BFEC5),
        u: GF255s::w64be(0x17052800E0B4DA8B, 0x5CF867DE0ADED951,
                         0xA1E811D5F01D372F, 0xDB52105D891C8CF1),
        t: GF255s::w64be(0x12DB650B53E17835, 0x565E0443A3FBDE66,
                         0x61C1C858ED9253B0, 0x465AF151D990BDFE),
    },
    // (2^130)*B * 10
    PointAffineExtended {
        e: GF255s::w64be(0x52B3D9F485A3DB38, 0xB4E997E9293857A0,
                         0x01916903ED0CA017, 0x0149667229D2E833),
        u: GF255s::w64be(0x234345752143582E, 0x46505E02982B39EB,
                         0x23D37316626487D7, 0x126C847746496B31),
        t: GF255s::w64be(0x625467C6A6568CBE, 0x2D86F80589C6CAF4,
                         0x9684CAEB425C1B2E, 0x6BBD04A3A0420BAA),
    },
    // (2^130)*B * 11
    PointAffineExtended {
        e: GF255s::w64be(0x7B0AF5F2B95FF650, 0x78F69BE716B2CFD7,
                         0xC67ADEEDBBF1E2D0, 0x8444EE09228C2235),
        u: GF255s::w64be(0x6BB94CDFEF6F6D88, 0x8349ED268E6F750E,
                         0x792F3E2ED6FA4957, 0xD51EDD52BECE2651),
        t: GF255s::w64be(0x66EC80D6A1180C6F, 0x2CBAD6B90644B58D,
                         0x43CAA7ECFFA8E04D, 0xE4774F545088C5A8),
    },
    // (2^130)*B * 12
    PointAffineExtended {
        e: GF255s::w64be(0x7DD5EFEF2F7E3A00, 0x67D0F7B0D86050C8,
                         0x3494662617CEAF23, 0xAF4B65BA8591C1AD),
        u: GF255s::w64be(0x1F3592DE203C2DA4, 0xA17D1BEB2C69ABE2,
                         0x10CF7923E46000DB, 0x89340045DE2129A6),
        t: GF255s::w64be(0x643EC0482633F97D, 0x932CF781C2D4D2BA,
                         0xB19B06C590529CD9, 0x946A6220C9A50DDD),
    },
    // (2^130)*B * 13
    PointAffineExtended {
        e: GF255s::w64be(0x19430AF80693A8A2, 0xD874B17723DB7E8C,
                         0x1321374B6FBEA675, 0x681F8DB202E72852),
        u: GF255s::w64be(0x7AB6CB353A7E1058, 0xB4B045E5E702318A,
                         0xCAD900D819124EA8, 0x0F4767AD57355C90),
        t: GF255s::w64be(0x33E0CFF16CAB14B8, 0xC425B464B3606AA3,
                         0x0C528F0B20133EC8, 0xF5572A210505358F),
    },
    // (2^130)*B * 14
    PointAffineExtended {
        e: GF255s::w64be(0x26B43F065FC95D4F, 0x394668C34B06CE7A,
                         0xEAEA8FAEB3D5ED7B, 0x1739D8DE862E3AF0),
        u: GF255s::w64be(0x32D3B1A49A0A50A1, 0x08F76F6E267875A4,
                         0x84BC9EEC48398A95, 0xE62B26D301A95CF8),
        t: GF255s::w64be(0x720DA33C8F5136E0, 0x5FD8D77B29295DB8,
                         0x5FAE0C5B31875B73, 0x1E1D0050EAB69B8C),
    },
    // (2^130)*B * 15
    PointAffineExtended {
        e: GF255s::w64be(0x40D8B2EDD4386D2C, 0xC69D5CC6EC35FF3A,
                         0x1FDF44E6D229E43B, 0x53614A3BC609C455),
        u: GF255s::w64be(0x082617E146EBF733, 0xAD341ED8B1E1776A,
                         0xF789BF221086D125, 0x9A0B63FF58819185),
        t: GF255s::w64be(0x52BAFECDE0C48A92, 0x60A041C742E3E324,
                         0xCE7FD95867032F80, 0x94BAFDC6B0AF72AF),
    },
    // (2^130)*B * 16
    PointAffineExtended {
        e: GF255s::w64be(0x2DBB938F43FCED53, 0x0A7A9E0DFA237DD5,
                         0xFCAABBE442C88D93, 0xB59722476686F776),
        u: GF255s::w64be(0x588C179EE277A32E, 0x64BAB1680DE07B83,
                         0x7E7D988FECBBDB1D, 0xF50D8973C1409964),
        t: GF255s::w64be(0x60C42361D4635289, 0x52196FBDBC75755C,
                         0xAA7B23F4A4AEC308, 0x62A6B63BF1FFD3B3),
    },
];

// Points i*(2^195)*B for i = 1 to 16, affine extended format
static PRECOMP_B195: [PointAffineExtended; 16] = [
    // (2^195)*B * 1
    PointAffineExtended {
        e: GF255s::w64be(0x1982687F23A0A46D, 0x20A728914AA70C44,
                         0x7BFFA172511D0BF8, 0xEB4B028B52EF088D),
        u: GF255s::w64be(0x02B1FE40C35B1108, 0xD13612F77F90F84F,
                         0xB46F6A19F2C711D7, 0xD5386C024F6950E1),
        t: GF255s::w64be(0x4ECAC7AF734C65E6, 0xF54207D14BAA772F,
                         0xA15165FBD4AE4DC9, 0x24BC6E863C8937C9),
    },
    // (2^195)*B * 2
    PointAffineExtended {
        e: GF255s::w64be(0x32453BFF3E754687, 0x08BF6AEC6C9AF425,
                         0xAAC155C49E622FFF, 0x6736CB36F5DF3C6B),
        u: GF255s::w64be(0x207807A3C7DA4C9E, 0x0394879561348B5B,
                         0x491F6D9530005684, 0x4D20522801D9BE69),
        t: GF255s::w64be(0x7BE8AD2F073E261D, 0x8BAB5AC42AECE484,
                         0xE3BB41D1EEDD6F53, 0x39C460145081D09A),
    },
    // (2^195)*B * 3
    PointAffineExtended {
        e: GF255s::w64be(0x616BB17981AD5800, 0xB5DA9A3A84DA7A94,
                         0x0D36DDAFD2EDE213, 0xE2163ED40240D5AB),
        u: GF255s::w64be(0x2C2FEC4277A075E5, 0x052C83B745F76715,
                         0xD15830A754BDD8E5, 0x62329BF642399CCB),
        t: GF255s::w64be(0x614659258F72400F, 0xEC7B337F46B0A4C3,
                         0x8FA110DC3CABC581, 0xCA181F67050E4AFB),
    },
    // (2^195)*B * 4
    PointAffineExtended {
        e: GF255s::w64be(0x21345004520D6147, 0xBD182106BE833EB9,
                         0x00905EC73DCFAB11, 0x11A5ECC6C01F036B),
        u: GF255s::w64be(0x20B293F739F3AA65, 0xB28C3C8D49110514,
                         0x45EB4CD78646E9E2, 0x33F1C936823F9793),
        t: GF255s::w64be(0x395BB5CD3D65F08E, 0x11CAE9FC0D23AB0D,
                         0x6E3D99DD3E295315, 0xECE0E7C9F465E491),
    },
    // (2^195)*B * 5
    PointAffineExtended {
        e: GF255s::w64be(0x0D654BC97CC560D9, 0x3FC45AA3BB68E927,
                         0xDDD7400D7DF1D9F8, 0xC3465879B04EFBCB),
        u: GF255s::w64be(0x356B0675A08113C1, 0x9FFA3C3AAC578533,
                         0xE47EE2E2B553CA96, 0x3F09E8FE4D4F0DAA),
        t: GF255s::w64be(0x431D2978BDA14A65, 0x6DA80DAB9FB489BE,
                         0x3739979BDD5F3D89, 0x2F2E5ED16CF3E78E),
    },
    // (2^195)*B * 6
    PointAffineExtended {
        e: GF255s::w64be(0x7A900684605D8B43, 0x5F345C36300E35B7,
                         0x7A150F1801C67235, 0x93683AAAE71501A7),
        u: GF255s::w64be(0x076CCDEE12F67001, 0xE32F52E0BB72837C,
                         0x62D889136CC108D4, 0xA143C257C8E7B387),
        t: GF255s::w64be(0x580DEEF025934593, 0x75B858E15A115558,
                         0xC600EF045B679BDB, 0x02F43BF7262D9E46),
    },
    // (2^195)*B * 7
    PointAffineExtended {
        e: GF255s::w64be(0x4AB3F0BEB592478D, 0x6BE0EB313BA1A346,
                         0x56CCAFC57F61B1E3, 0xCAF3D56D09001CAE),
        u: GF255s::w64be(0x2D2D355F27445B8E, 0x100B543BDE5AD3F6,
                         0xD10EB0D73B802862, 0x971F761E8AEDB101),
        t: GF255s::w64be(0x44B393C42E5F9548, 0xD33B393FC4662114,
                         0x6AF732299368BB18, 0xBB6B514090669B3D),
    },
    // (2^195)*B * 8
    PointAffineExtended {
        e: GF255s::w64be(0x30E51AB070B031C5, 0x47F0972818DC4F59,
                         0xF9AAA7896642E5AB, 0x073F794DC96EBE06),
        u: GF255s::w64be(0x72F0811D6A945237, 0x0F5C5D706C51FC54,
                         0xB754075F43904E3A, 0x6AC9953979CB6EA2),
        t: GF255s::w64be(0x2CEB4E800F2A9875, 0x326CCE6372A57EE6,
                         0x9F321433405FA3BF, 0xBB17DDD37656E362),
    },
    // (2^195)*B * 9
    PointAffineExtended {
        e: GF255s::w64be(0x074A5FED55883EDC, 0xBF6F1A757B6325F6,
                         0xFF7E1AF27A122549, 0x39611A8FE1675B2C),
        u: GF255s::w64be(0x13349ABFC999DD67, 0x466067CE1442F46D,
                         0x2D0EFBE6A1A2C97B, 0xF2B5C06FCAFEB583),
        t: GF255s::w64be(0x2C1C0AAC64969117, 0xA0761711E6393676,
                         0xC003C77D16DF1800, 0xA4D40FF42D09A54C),
    },
    // (2^195)*B * 10
    PointAffineExtended {
        e: GF255s::w64be(0x4643A40C89936E48, 0x60DC0E92531F05A1,
                         0xE5D5C2E0726FAB1E, 0xA3062A05A4EEFBF4),
        u: GF255s::w64be(0x5774724E6436D94B, 0xC77308C7C7664BF3,
                         0xDDCF1D005DA27586, 0x1F87C746453A8A8A),
        t: GF255s::w64be(0x1C9D11E40471A372, 0x58B8811ACCBF5685,
                         0xAFD7FD333DC2B252, 0xAA14EFB76CB73C25),
    },
    // (2^195)*B * 11
    PointAffineExtended {
        e: GF255s::w64be(0x5884A598A4404810, 0x012B93CE6A5FE90E,
                         0x59FB0704B8D4C07B, 0xD13062D63B886BC0),
        u: GF255s::w64be(0x4CD603F72497FC6D, 0x5FDD560F0207E6AA,
                         0x05428BBDC4D7168A, 0x08A8CB0C9987EBED),
        t: GF255s::w64be(0x6942504CFB557160, 0x7241E80F6BE0B6EA,
                         0x1502C7440024CECC, 0x3F08842CBAB59BD5),
    },
    // (2^195)*B * 12
    PointAffineExtended {
        e: GF255s::w64be(0x5396F0454D545564, 0xF97BFF023D44A6C4,
                         0x98E6462E9368F9D1, 0x94CCC8EE7B7A2079),
        u: GF255s::w64be(0x14C0F255E1CDCAD1, 0x6492E0F04B64FC09,
                         0xC0F01E802AC5B1C3, 0xB4754DE27F72D537),
        t: GF255s::w64be(0x6FC86668C96482D6, 0x7F75ADBF7BDCED64,
                         0x4EC3DC5605B71FB1, 0x392EB3FD516E0A97),
    },
    // (2^195)*B * 13
    PointAffineExtended {
        e: GF255s::w64be(0x4F9DA6DF6C6C9E5B, 0xC58E3723BA25DFF8,
                         0xBCB0813E9F6E73B8, 0xEAD383ABBD6BFE31),
        u: GF255s::w64be(0x420C0614ED165A4B, 0x60D780AFF882B1C7,
                         0x0E14A880D3E90C45, 0xCF158917F580DF1C),
        t: GF255s::w64be(0x5F142474E4E1F3CE, 0xB43A2ACF31FEC743,
                         0x31611DAEC9A6880F, 0xCBDDA904B121A745),
    },
    // (2^195)*B * 14
    PointAffineExtended {
        e: GF255s::w64be(0x6713DDBB85F6BE8F, 0xF618BF0ED36E5167,
                         0x9BD20465051FC1FE, 0x37DADA6DD884B154),
        u: GF255s::w64be(0x40C860215AE057DB, 0x645809F39DD03BA1,
                         0x0AEB0A55D410C765, 0xE3DF626AA182F658),
        t: GF255s::w64be(0x15D34F3283722184, 0xB0115208919623A2,
                         0xB136DC9FFC64B6A1, 0xB2C0705FCF581D58),
    },
    // (2^195)*B * 15
    PointAffineExtended {
        e: GF255s::w64be(0x2AE8064429DB3D90, 0x4A240F132CDBD9D0,
                         0x8482C85F3145A960, 0xA304489B2ECAC269),
        u: GF255s::w64be(0x0C078789FF8301C5, 0x43EFF53756D1C683,
                         0xFD932027B0906610, 0xBA4D3B438436B829),
        t: GF255s::w64be(0x71698D07904FD7A5, 0xE3FA97515BE2309A,
                         0xBFEFB28BC630CCDC, 0x206A563B0F98A082),
    },
    // (2^195)*B * 16
    PointAffineExtended {
        e: GF255s::w64be(0x255DDF7E0DF3849B, 0x4BAD3FD43C3A18CF,
                         0x5AA587118E79B929, 0x320F03CA494F6306),
        u: GF255s::w64be(0x50A69209BB0B3E0B, 0x2E1D217E9768A122,
                         0xAAC1F71E18F00C14, 0x2C94910518A991C5),
        t: GF255s::w64be(0x7AB91A184C260615, 0x1F0F91EC44969A4C,
                         0x5711285559145CCC, 0xF93D8C083C888F1E),
    },
];

// ========================================================================

#[cfg(test)]
mod tests {

    use super::{Point, Scalar, PrivateKey, PublicKey};
    use sha2::{Sha256, Digest};
    use crate::blake2s::Blake2s256;
    use crate::field::GF255s;

    /* unused

    fn print_gf(name: &str, x: GF255s) {
        print!("{} = 0x", name);
        let bb = x.encode();
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
"b1c26ad124f0b9b235e811fb02576bdeed7f6f2bdb9625ce867bbc2bbd751e59",
"aef7d911554e0e1527f49d49d303e7da00532359589f1d56528005cbac70b153",
"318b64b9ae83047670e6256054e6b5c0ffc3f9cb24659d809d946a7cf72a7521",
"66a968dfe975050781f837e3ee6bf0352a969e53e4f66d1276b2c433568ea26b",
"285611df1d4459f29fe7cbcbb98c752306d75664851836e50c347401a664f270",
"851a83fabfc063883c88904475b53dce04d664f31e0fa678e297e3d2dee58732",
"50ce42f8d53d0e1b79185c5be821578dd2a86b20084a6c0eff47b6431457ec64",
"77859488c089928c355581d2a8e86154b94373ebcaaba991ad4ef19e4fb0352a",
"bd9f4067b33ea224f281e8fb8b89c9317151796b415d7e491fb469c7289ed058",
"3a89543388ce55e915cc870a4db2dd4fd9ad2e618f5a5013a45b67505859957e",
"1bb1041d4123f462baf68ad3e6fc2aee7b2d30291ffd478379ead57470b1281a",
"8d902970535a38b5e8d283a55d8cfe4ad62291916b000e561ec78e81d30f3160",
"5758687972e24623a80af50fb524ecced34ade656a701644c764e30517a37713",
"7f013fcabb9d8d29c2d44764703a4d69b1841db608bb1043e51d39dbd29f8739",
"9afcc2b95049ee0feb5787056d301115e0b4d22b8ec82cdc4c37e80a7990b901",
"d696e2470eb8a2a9c86a87791ee99852e4e51cb2be78e826e73fd73037df3a00",
"510467bcd2af8b15e4833126e9ab0057c86e743420bf6681a664447f115d0118",
"14ed9c64000ce1100827fa3ec256b7b09d0f3e009dbaccc17ad4e08d62172b37",
"51fe4489d8c6f777feb613dd9cfc5160a12cc30c2e48bb1594e32bdbfa8a5f49",
"dcc3f03e68c699d5be18958e90d9efad3ffec41de365807633592ff89d306771",
    ];

    static KAT_DECODE_BAD: [&str; 40] = [
        // These values cannot be decoded (w is out of range).
"8ef0ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
"f40f8584aa9e36f1be98b045dfe8573213557185500de9519c317926c9fc48ed",
"28e123fc26680524a94a51d3e9a120a39e329c383c41c3e7ac87f173c722f0ee",
"3319b08127d291d4480b441b7489886483ec88eeb77c165bd3698c0b305c0dc3",
"1895a6aad79f9738f2f6e925843737a9ffb01fc803f948a217255e35cf91e3a3",
"300ed3309d2df52ae6773cf378be94c961cd1cdf34480a3604efd3840f56f9a9",
"08572a152d45763d1def57969530cd09b9a52ac7da82b241c0364d081b526af9",
"bd3bd83424f7ea3f6cba33e25ba93ba0c4c10b6645508048c9e6e467ce1578bc",
"31d9540603feb90ffc81e6f94975b077ab88448245f78392d52c56334e998aae",
"7064abf64fc84a0667abd7396c30b63406b677ae078df897b7193208d845a5f2",
"f35b45b66084fe0b497696f517e3ca9eabcc8d152dcb961d6795fa6ba8c0ada7",
"0718ea4ce976ab9f64f6c62a5a73c1406aeb4c3c7be2d6b277cf07a8f3b1e58e",
"25db3d20f89dd4b9cf111912fc3427092464292c8634519e73afafd3821e73d0",
"e7b7af235613713fbdcfc825c2d877fd3f87df7e83191134e249906060e36fb7",
"458737cec3f41fe9506af823fadcf0271dd81fd656038f0c3dbbf7e93b98bed1",
"98061b024220305afb556de3a2becc081d40cdb2e32e0f932f61dde8d668d8b2",
"1e2f702bd67d6ad8710fea61dd8f5b4e96128ad528ec66f697d73e54d2cb46d8",
"2da6629388de5b7c5a6ce5a739f8e2f0e6bcec00465f083412161dbb871cf3e9",
"584eaf85c8017a9c8b40974974f423d882f946ca91f9dc317a62ca65b6380392",
"6f7b1a58344d99a20c77923f7eb0b09bee724ece288c9624d2343c962c710fe5",

        // These values cannot be decoded (w matches no point).
"029db5a37622f1b2e4bedb93f641f98b4b26a06936e80a8beb5a0a81c350bb6c",
"eb23b93c99b8d786f9772371d35b16a444bd22151bf34f97cee4f11f2b54b53d",
"2c57cc483998414a721a4736f89312ebb2623cd7c7b04318ad2755950002797d",
"56196701642782ddccf9e812ea8bfe7cbf430519e1f5aea99fa23ba784d0d653",
"2d144fafa94c2ec516e71aed7967a3aff354bfec9002751a0a07042f4af14943",
"cc8a98aeb704ac2e884ebba9f9832d686a8e0237efcef408d4deeff7bd248d64",
"f3c76f1a6c84603b2327e6a09e7c311d3dff636edfd63a6803b6a18543717f4b",
"6afe78e4600c57add45f3f763536f2828e1ea1a37de45382991fe3ea61957e17",
"15ff8793e536af65cfa1f3ec63925c2a504d155cbf857023a3905f88af5f5e09",
"e99847c33b43dceb1605b4ee22cc8e7f1936cac38840b2f851b538f286ec8609",
"1b57817ef3887174b39bb3fad25cf2a72f9c9bd948c60efdc5e7953973ffd860",
"bbc110ba28ee2fc04021e12362889b8762b8764c424df62db2d2a57da351c27f",
"4e2b59d3ea5ea3140b9e0316f793b94bf6af4470f03dc6bed6a2c0abced9476e",
"4bdeedcc7050b0ff28c7092af6ef78b5bd0e369dea29453850cbdda019155d7f",
"522db2987ffdead41a2d731fdb7a09122b6ad9909b1fcfdbfdbb5c3b641edf11",
"f79ca01c0d421023824d0f081aa5b5004d7222e8b1583ff9697632aaffc4ce61",
"f1c9835aa95a7783884c13bc3963fa29afea36aaa1114b26184319e05190df3d",
"84f489ca4d49d8b4db13b0ee9cce45e10a4c537fda03941ba8bce5f6a8ef5a0a",
"9f99165adf8d45e538b4d180747793473eee05e5a42d5db44af76e29b489c058",
"74f084660923f4de26538adef1b8424de0c2ad4e942b5e1b7c8af056b939e57c",
    ];

    static KAT_ADD: [[&str; 6]; 20] = [
        // Each group of 6 values is encodings of points:
        // P1, P2, P1+P2, 2*P1, 2*P1+P2, 2*(P1+P2)
        [
"381c4e1998ea738fcee1d18779bebb36271ef55a5bcc3a5595717576e2ce704d",
"db479deeb0d8e6096b33dd9dd4e18d385308cb553faa89ce8681df208860a110",
"e0e467a709f40a935a8b1c5e78a745924a0eecc58576c99866b20e435fd7a821",
"e02e26cc7657170f7e3ca5e9fd7fb11712928b55fb007df9d62f805c49be5673",
"d60aacf148e642d5f31a4307300ffd46f6235dee653e7ea611484586669aee01",
"03955a234b2760184666f43a9f3aa1b94330a37d4a2722cfc2dcf436fb67ad52",
        ], [
"60f36f052234d98450e0131f5cea458af31b5e36df99c7ac77e8febfd6bd4562",
"eec699499ef822479c929686a6fcfb68bd7c03004af40a6e7120e61bf946ff0f",
"011e0b7c6a3332b8d81a44cb34a4a6e5428d1a1e2f0b68973f65331a2aa37421",
"075b9051e80d2d25c8f289331e5abda12a9046c8269994fbbcd0408f8196805d",
"a5d16235f08f5db6c03138c280e7b79fdf3abb1f5c9cbbaebbac00d7b5e58d15",
"90a2dd8393d8f80f07b323a6a3f71aca392ee9355a285d3a924c0a08fdcdb12f",
        ], [
"5f999a844a1142c521b30d7962c5d1cc51f44b73310f2859fba957bbde789250",
"0e463bcd3d7a94dbe0c1db26bb8a1ceaa02f3bb5a158690959aae534a99d5753",
"8532065bd0f3442f01b315eb0088f42656959eb082e6ff5064e7739383a1e50a",
"44d0ee20649420c8514f2b152e8551334df9755f18eed00225c0ea4aaff52262",
"5e026732df7af8d6265a9021cbccd6b4653353171fe228ab3c3a26911a72b421",
"b0e8da551c65613e0f8e90bbb987941b55eb1818f9c4747cfeba1f6bd715e139",
        ], [
"3ea95fbb3baca80e0340315c688ec8b898b17a491f452e2a2e7d7b7d948d2171",
"2cc136fb22b80d783f6a8162b1292fc9192cd5be4d719c52cff60e6cdd845119",
"dda4bf35cd7d5c05e49b37640df81a385e4cd9e540e6a84782a57f53543e3b2d",
"c11fe3be926b10336cbb8f191e31459c6fa492629c773f5d7214536cd299a367",
"3ba12da38f381c060ac30c20adb436cb816b01fcf72ab777cbae7525767ec044",
"61987cfeb7e4c45f742d1d6efd5d04b7d7a10246e591d947791d64da569c2d6e",
        ], [
"6c44d7a7569c28f80b12273d5ad2308f1241eb1e42d0b92a67213b7e908ad14d",
"e9b5567f03125c2357ad2190a852c3dec98dfd919721f385066b28f797c64e6c",
"acedfb558a43ef11ece45956d15a4ce8e206f101311ef7ae51d89fdf1d593304",
"5974f71abe178125b92b559ee5025e004c37b29e570e1b05702bac49160b221e",
"9814abe67e6f98b67c7abb8f192d9177b5b707f3e0eec52a08442f3bb6e80464",
"9e6725094371bfe29b6e114b29bf12eff0167b93b98b611734cdd3e196a0f52f",
        ], [
"bee7f22a49b9e5404162552437d11fff2b773b4111ada2ec8dc0faab67243261",
"c7c2c162ba7a3c0b8ed64176b959207e396b43b9eb5429c418528bfc1370331a",
"f0615e44ab681f5f6823b3b441a39fb5f518ba448eacce1f104008771e33fb29",
"95e9fd29ea0c0bacabc1d4158503890784c3c37f3b35c70c9ae45d94caf58329",
"02c5f804b8607d5525e52810a3d000cd489fe7b79109239b26af3d9d4085d658",
"95faa802ad86ba6929fc3b01730daa232076aef3b063ebb6b309cfea61f9e741",
        ], [
"19ea6d24f1511a233c198ed0895d3fc76161a3e48f310b15cc8e6730cc2c1026",
"5755e6da5c23c5f0c67b9b1c7f7e8b02aea0ac07498d206b5efa045489e1d737",
"096ab36803970aa88bf6000f889267543299016c63ea2c629df4e9ed985d862d",
"8ad8630a4117f64a7d78deded3400ea6adf8f92f98ddfc85c1b1b5819b5d4046",
"8cbc2d5148c3c78acca57399073370089f535da059a93415788a1528469eea4a",
"6bbb60d1fc041b826313dc3c0a5116e71767e252cee93c8dc512516979b4ea12",
        ], [
"88b303961b1ada474b2294750bd00f1c935f0e6f1de20b07c4e971fb67527f51",
"823978efdd5b7b305262287900a4eb96cf5e61c351a13a9e3bd61efa687a5866",
"dd5c8e2ede2ad115a1edcc72c2e3c264e300fce3a4e3d0f09f504e023ac9e31e",
"20049f4a29799b829533ffe67cbf70279b88bb19a6ffe62b670914b5bafae025",
"1b7ade87df630960d5295480937545974949714c66184ad3770542f7377e2301",
"ea9fd3440c0a193d56e93589541d55f25bafcda89d66986689ee171823951666",
        ], [
"add5c3e78cb10a1f9ba69997804838250506cc7678649b483e6299e045497371",
"7483ee768a54e1f3b19422c9bdab65ed8c0688483cf8fc3f80a08d07bcac0622",
"c9b187ab32b93e9acbd2600d79985eb6e1b0daaea7dee8a372955bf36beba27b",
"8ae1eecc6db1a3866c2ace308989a0f8ddbfb9b182da844aadc05148aca0e25b",
"4c8e8c39fe3370d9fa1797dacc26aa93463bbb2cf8309140c43bcb0e16842e39",
"92082e52bdd0091846e0be899144cc553379b723f609947ff95c6308b3e55a25",
        ], [
"00219fcb3c4a5cfb84c0cfb2e0be0c2fc0dc50be01bd15c0849cd2b4e86a8753",
"61cba11b9c106484a8d3b677fd42ea9aed3c4e3211cf87845db59d085406385b",
"5c32d46832a9e72ba1b334176d5afb4b499915479559c99ac900abde03d3a503",
"599f2e7e822c06eeeb98477975b18664556c9de5070e296bb547a5d6b064d153",
"4b9d81c62c6b1ed1159db5a6b4af8161b7ec864740bc2bb3e49627997043bb0a",
"4cd2ca03930823ea167e6cdc738cf4bba68960999d26ef6b748ba8ac1fb9f61c",
        ], [
"f881d61f1d74ec291e82eb12b274789ceb22541a1d62f58f26c946aa8262954f",
"a26b9edd4f33a8d27ec17b0834ab1631198bc7a86d34f22eb88b5b59dee40a5b",
"9bc3688ede79d6d6fcb5f29576e8f4fe00a47ab49913b3ee3a66ec5c901e4137",
"716f532220d997612022d61df0f2aa890dc087039ddbe6fa315d89e58ffadd72",
"1c209ccdc7fc984d4d03b763cdfdf0e5ce81d22eeb710057e80b49f2fe7bf74d",
"8067c4ac59dc0f95470c06efa63820c3ed714146167216eff0ba61fd0ab12c04",
        ], [
"6d06d460392a61e9011d18622610ad4ae2b40352a787c440e458bd67e81bac6c",
"acbe4ea45fb6c601d0b0da1346ddc2315e851def7555c90ef460f9088cd7114b",
"ed3af00b64e16034cdcd3b5344ca4bcb686f61af5894f685f353b06140b8ef75",
"d4cff64435af65e2d73db04984c9b1724614827ad272ebb9fc76646885bf5328",
"2309d522f92b184f7c290a71823f79b9f0c72020c5b6554b4647f813da77ff3d",
"2f910bd99152d1b56542fd3965ede38ba9647516dddcc1a5bc9db0aa645b8103",
        ], [
"94fed56d5124d271fd3b63cd17f4f3bd6f3fb9c9b3a2f2dee7abacfada3ab969",
"25debeac62ccaf5c8d5019580c7ef97e78c769ca6514e6873e9622cb74b18171",
"7413d872bb58181d56d0e288d2993b2a552845292632b45bceb2fd8f3a4ef614",
"fd3526dc4a70604ac9986f916129cfb5995b8aac2093dd06c338cfebcd23062d",
"8ded66b6e0bfa56a27a5fa4e763545064849e35ee855d65f73546832f9dc0d54",
"835d25ab5b57ed33402ab7c6701755117619948a558878193fecee12cd6bb150",
        ], [
"780fc3cd9224249d1df0ff142f552629846bb1f1beba2621bf79f85a9d314d56",
"3175dac57c3a693980287572a2c14d0a59a4f078af8d00a79b38ec982b25730d",
"da270fc356d1a119132b75504682dd56ccbac891d100e8c794d0833285c1c34f",
"abe881a298959bac0e8ab0b111348a31c3c3c9dada2ccae15a853685f9ac2b46",
"bb1344014186f4219716a740867b0e7dfc3297de7b35240ec0cfcccb383be57f",
"9e0bbb128e78780dfa725cd8934e025097be87672f24b5733b41f5f23a5a3549",
        ], [
"eed74e9cbd63ddb8f3ca345ebab1b0ba9cc6bd53ec6096fc059354319957fa74",
"f921125ccf44d17ce2d8d0f29a9f59e0f73d8a759ac8f7ef9250f4d5fb0d7426",
"1f36c8e821941eb90d24c6341e75d94aeeb2527863a6a634fca5d38f42663430",
"7e82ad365a5f308390a0ebfd73952cf924050f03620a1e0fec69797a2179235d",
"aca89dc458bd0e2ca0f73c09e495b2241c085011ab786f707d8c3c1c2f7ebc4a",
"5c52e4759076e2558be0b5ade6e1bd53c99514c656d61ae4dadbcd63b0934a30",
        ], [
"97dc9ac2154cc71d4bb42fb46afb064924e538a88db86d52397cebe124b4b075",
"3ccc7026466bae54512a3f4ff7f85e703872b74f517a4b607dd5be5a8412a16d",
"2ddfb977bb7fdc7880efed6953ee1604de69df616964f8a127437e0e98dba361",
"7185bd0d698a74852ac7639fc9b955bc722f30a019170f01644a353c24e8cd2f",
"47229988a81941e122554b34bf81e2a75bbacaf44bf263c716cb2725665aa37b",
"414b57d106ac19110db4815c8e59be5c47af99b139e8ccf13025c39be320195e",
        ], [
"fca688816addd850c8930696a49a1e6fb9e14255b03aa86dddd049f6cce8ee67",
"53f9d223569e492d78fd4b6665ae8994c6a1f852a6c38a60072a87240062d808",
"706ba0b5f5ce767b83fa2c539e4daaa750e5bd43c95acd5a1eac3d4afd6d6508",
"9a26cbf9ea28d031068bcf81b642aa9044f0fda980766d22a6b2c8e479b30840",
"18a65ad2f85cd3c3f2fdb6fd4a4d88e523ec93b19894ffce6f3fb931740d1607",
"36522522608857cd94e8dd9814bed64f12bf8a6845098b284457376544afd820",
        ], [
"293907dbcfab4d4dab6b9edc338997da0d107f3cb47f3531e75c04d23171ee6f",
"e35d3b0cea71bdfa31be71d19eb58c7a86a4566169f17980d2b32a95b9f3ee54",
"2f78f90f07e8d242bed8fd5a160fdeda7924d045b77b228ae7d2b9a46e1b1e07",
"68564e3f97a87ee54e66bdd31f98efd95f90c0081bb348dd179ed547d4732c55",
"5005f9700eed61c2e68e37bb0607d4236ada34c602ed86e0160b0146a9e1b339",
"595fe35df7e10174845e5ce6f334f7be092a82b56bdaccbbf19bb5b11d33d171",
        ], [
"78cb0a784a503a4386928f22754e772223587faf2cc8950dd6e386680a86ba65",
"a4b5c97f1972b54c13d7b050666d6d198ad63e21a9c06e965025ca7e44e9f000",
"5764e56b2d8896fedf97c30fc2a9d5ad5c2900a0389dd6234c55f166a036ad18",
"500269aaa0bfe99b6f16a1cd2dd1b21668c5e085b1782de1307449732009cd1b",
"b936238a475ae0851b89b4c63e094791ac6a03846afad8c7300c77a456bd6e58",
"6a7690687763248b89a41ac65607a3a88c539a7cfdd3079102b33b2ea607506b",
        ], [
"c671c46c3f35165106b26b49873a17a202004883c5a56962d5a13fc4dbcaec44",
"a04f9846c83204d54f26bab44530c83b4aaa5dfa7772084fcc30a1ab1a9bff1c",
"552db8f5fad91c4b2d81a021e0e5ef3683c279d8ffa40097417e2fb483c3e23c",
"aeb8c01dd8a194abb8a6a34c482627e8acbb3b64c8b5470edb9ebddfadc4c61a",
"8e57d5813af955b94447161bb9965da513f4595e6dc961767cb123657aa1bd50",
"5b5a5a734813bd45fc5c08015eaca263fa715584b15409730a64b4c20123af7a",
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
        let sbuf = hex::decode("f17dbcbef04a157b7e470b6563940017e5de0bbf30042ef0a86e36f4b8600d14").unwrap();
        let (s, ok) = Scalar::decode32(&sbuf);
        assert!(ok == 0xFFFFFFFF);
        let rbuf = hex::decode("10121d78458c788d9729a165e4b5f0b308f23d75eaf5c370fa98eb8a7dafec49").unwrap();
        let R = Point::decode(&rbuf).unwrap();
        let P = Point::BASE * s;
        assert!(P.equals(R) == 0xFFFFFFFF);
        assert!(P.encode()[..] == rbuf);
        let Q = Point::mulgen(&s);
        assert!(Q.equals(R) == 0xFFFFFFFF);
        assert!(Q.encode()[..] == rbuf);
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
        assert!(T.encode()[..] == hex::decode("a0df5043c5cf6695dd10e3492495821b68457cb2645979ca2fb3c936544d2a18").unwrap());
        for _ in 0..1000 {
            let n = Scalar::decode_reduce(&T.encode());
            T *= n;
        }
        assert!(T.encode()[..] == hex::decode("ba6f94d4e9120982a4a821a3335223d33d89ba373a3a91d4c5cbab3cc28fb252").unwrap());
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
"0100000000000000000000000000000000000000000000000000000000000000",
"0000000000000000000000000000000000000000000000000000000000000000",
        ], [
"4f347267b8e1655a0a25cc5d302116baba0ceecc838986132a0dfa5bc74e2993",
"e18ffe262e994600c1471c741ad8e7ae11abb640c398c8b0306cf6d76add594d",
        ], [
"97f335d4eada682f09239a3b71a66c11a531064b09328ad8c4ca436a05e2d816",
"a8eb52e69e2c473318a61c66bc6303666ff2c34ebdace7bead72ba693387c54f",
        ], [
"a07e536399349201a5765a9edc51189366797999dff2181b14a9af0da320fad4",
"eb1e9cd186459e27cb83e6b847ec3153fe3c6c8edcc21f5f4186a060af381149",
        ], [
"8350bf235e37b4acfae66083b52061f53b2894a7db9a5f13881670fc48b4309e",
"31e9a5352c29b0a8c57582cafe6dd7aa2b094f98c1783849c58471551b51e103",
        ], [
"3894717e7ed532f279d486c0dd72dc2f319d86cf0407f7975e5ef0c8f6f2d68b",
"2dd1f1b53821b86fe6784a27fde7cdba5201a09718659e621126cbb757073409",
        ], [
"98fea1a8a6c415f0072281f8e11de091e9fc8dee90e8a19c57af2c3fbed4df8b",
"f93f5d3c0b7a3513982fd68c95ff0ac27e3c6f41e30d49a06de94e33e0c30372",
        ], [
"24bc0e9df5b0f627a27eef1ed2e45115f4e04e8865cb0d3d15e9abd6f8d1be2f",
"7b9526bdb6f92ed5bae05b00000d02b8f348b7be60ebd44cb251ba6e8a8ba13b",
        ], [
"b3e07d3bbdc330ffd662d42c37e4c8274bc8b7704eee118e66f6eadba50d4fca",
"9ec145cf7c074984124bb5cc62998bdc9fe5ec9c34881ecd6041620b564af53f",
        ], [
"5ec437395a7df0f0e849985fabc22ca914c1afe4b974de4fe651169a05174d1f",
"eb0200bb95e62ccac6df2d756051e909a1831c84707c119c1063501c15ba8c74",
        ], [
"b79f138c5eac38b5f07bc0245e9aed2179c4d6f90766956b922747b224ef9cd5",
"ee48b384cf94a9a920692ec527a9116e1de7caacd75c788866d494ffc5f53f5b",
        ], [
"cbb53297a9c53c82d57ce8ba26812c788932d19280b45752cad47bf5b81f7dfc",
"b04ea34f214377afe8a9b75dd252c8a0538e8139aeca59a80490702ba865045d",
        ], [
"16161f740ea4c4a639151ef923d5351fe96ab11fb3e84fc708dffedb4efb33b4",
"7e7988e6dfe4994546f31912fc7752138a144857052f91c1c5ddd7c1cfd76153",
        ], [
"e132790de50058fd94aa558c9e9bbdcb87c0b62feea19a83696a90e8b52308c3",
"37031786f7a6fe73394f8400027901f7534c2a83cf06230198c6f6e0ccc5f374",
        ], [
"6a26aeeb3cdfb7bc6bef9f0a76300ef2609f126a1d08128b45da19cfacd7f497",
"9e008fa86aa90db2a473edd99fd3bf361b232d698a6accf583c922c2fec1270d",
        ], [
"c754e692b9c948d513694919e6f9d417cf42a5565d729cfbfe1cd080f4623de9",
"a7748228ecf7a3afc6a915656719254483d26a753bafbe73b0a02ee84da1e924",
        ], [
"3e688cc3e0875232c152b653dbbcc1a22739287a315909f58e04e55e7d2121ac",
"b5465cdfb8a776430845e856c5de93ad25f98ea78860a0d9a52087942e607762",
        ], [
"e054d6b59905617d42b9d31cfb740b9219701f7a902ec91c0c115bf9b5a28583",
"2aecfc00575cf8e3aa6b981fd2b53fe1846ac17455831fe42c096218be57c504",
        ], [
"43df64360cbbd06bf7247495563fc880887451e0c5701a25bc9284fa2ebcd856",
"5bef1bf644e7c43b6eb46079bfee57fa9efa1ab9f29eb53be6d627625799fc32",
        ], [
"fd2e1e57e40c1ef60a21931bdf2329b04edba8d72bfed9fdc186ef76b5075859",
"4755c542dcd87ac4fb6c4bef8d3b7ff711d58c18d61c04dac44aa391fe60510f",
        ], [
"3cee890ab9ad19dd018f37f9162f78f8bed98c0594e50275192d25df071b8ca9",
"9e0e84fd5bc9e752b7f205962e177068a7458e66764ac19e7c561001bc3fca2c",
        ], [
"fe7fe03fe62bd0fe11aa1b3863f3ad86c04d1ddc391bd818528676b84d740c6a",
"090624bcfe3aae535d039daa55616a1c03f493e7372d0e1870ef4c3142d0132a",
        ], [
"82afef12e6bc1ce76f139b581c3e2bde4ca09fe15280ffe32ca2cce87b5475f0",
"2bc0e5e125ea6754c4df0f49e070094a79f675f3f42b9f880cb894d1aeb69437",
        ], [
"97d2c5245f76665faf903a023ac69d02b0399f3c7c5a325300569a5a605fdfdb",
"51a503689ed77ffd4ee5fbf749ea59a8305bdacc284d5b99853a40544dba5f68",
        ], [
"1955f6b258d7283ac4b10ff2913586a834dbfecce17e5cd491054152ab0c88e6",
"ab718403165ef2bb3d3b2d2450917312b541689585c0a030b5e3b24e58884424",
        ], [
"851edef2f6eabe94467279024d96d08314566c64e2e945d95db73607c3a18fef",
"0055a3b2e5df1e950cd0a85ce97b454546bf35bb422ddb452ed8beacd664d761",
        ], [
"ce58e5fb30564c468d34d3b68f61249a27c8fd1e0ebf51c47f68a2465d24bfcf",
"5f56b38fd1681ffe0d64cb2ac017c0efe8fe18bb2cbb15ff58c03e05d3066d45",
        ], [
"04c0845caf76658619d20cfc73a1b52d559325053ee10d867a2739697292d60a",
"a0ca4ffef9e12b33dd14bc8f58d809ad47326e1f1348c9561a34e3e7f91e4761",
        ], [
"19bd05ed8a0deaca33d5782d223aafa6da90edba5b2443b38c9268f130aeddfa",
"94da1ccd869ff797701897535c00f7d10b488de6ae082902439faa3af4444e52",
        ], [
"3b1202c82b34a81a4c996e337492ff99c52284d4c1b00d2fceaffe4c8e95e9c7",
"a8911426b9a05cae2bc5c6f0c9d9604408efd82972e33417b76d5ae65d61c21f",
        ], [
"a98120c5a3b4f86a77b7e6bcd8f35ae136a47c7a9feb9be0907d043bba80ac34",
"e18cf3de24ea143ae56c6bfc539c49ec3553cde185f47a1a530322527a9d3a32",
        ], [
"bf1351db9b57bbb6ee712a2bbdcc6e2df2d54ad62d51a50bdbb13b2f7ac06c3c",
"2cf4c9072a8df20da57b50576997fd5521d5001bd390e42bac4bccd1a895da5d",
        ], [
"dd4e3e5b80c45573e31ae8e8fe82c9cbf7b0dc1f1139718d34807530aa8447a5",
"fd6ce57786494f651237004effda2ab2adccd861d8823381de97ed427070da7a",
        ], [
"433f4deb8e26cdffc6224b3cb4aa3bcb8aa2a6a08f88a7a40f1b2b386560a3ae",
"66e801a030461a44e160b204a4c45a1bd61f172b493e42eed35f333617a39575",
        ], [
"464ed28003f7b8413cdeb77d8f32e45975b349fcbd7ee4e5e5653065f1a460f3",
"37599895c6a0b2771cf741010790c7022bf181d2367d38d483ccaf088f80146b",
        ], [
"e627cdf1507bd03e5fe45c0f29d12ba71076af1f42013d5e24abc0e285b2042f",
"6bd6607e348174b5c10003e57fbead121468fc847fe358fa2602c9df6d51f253",
        ], [
"7a838bd079ef3e37426ff3bcdac05c3d1e259fff639876753012edad3d64977c",
"f9f4e18b5d503306164434a655546f02f7e4362e833362bf75b64cf4516ae62d",
        ], [
"0a89fd51a9abba6f86f2db7c1cbf4346b6b46b6966667d4542a177bac8177370",
"15bc61c129ffde2c73b1c347274b828955ca01a1df27770404f44f0b020f7010",
        ], [
"26ed1f06e41118d43c874bb29d73485734effaf019e23fa8f48d80982318df02",
"7e23de36a843af132ad5751bfbf21825faca306207f57269db95f53e489aaa56",
        ], [
"6d9175799c8ffadeca0ead71dad0557836905b9bc0d94d332f22e4df25ea15da",
"304c8ab25fe8c3b1e3c49fdc75e769b48f3def51d4505d280e69df7eeb4eb809",
        ]
    ];

    #[test]
    fn map_to_curve() {
        for i in 0..KAT_MAP_TO_CURVE.len() {
            let buf1 = hex::decode(KAT_MAP_TO_CURVE[i][0]).unwrap();
            let buf2 = hex::decode(KAT_MAP_TO_CURVE[i][1]).unwrap();
            let f = GF255s::decode_reduce(&buf1);
            let Q = Point::map_to_curve(&f);
            assert!(Q.encode()[..] == buf2);
        }
    }

    static KAT_HASH1: [&str; 100] = [
        // For i = 0..99, hash-to-curve using as data the first i bytes
        // of the sequence 00 01 02 03 .. 62  (raw data, no hash function)
"c6fe2de08312096a3c5193b401b5e76737f8a5a93b839b0348ae30a9f89ad827",
"7a1b35e36be0e66c04a75e10a3b4f292008136e2a1caadfab6788d002b92c35a",
"ba207ae99a510681dabc21744b42c14b4491c7dd005e35a134bb7693060def2f",
"5817da14941f320026d5e867d8a77276deab2d6e5fb0274f25586967e416a65e",
"7480e3ec537d6ad5d63d4a5661ad0c209b90919ed7707469fb30540162c4b00b",
"cf8b08a1f72cdd6f73fba537c3053da1c4fca7204c18fad84565621bd70f0f79",
"91f5d409538f9b4a6069f70164558b6cdf1a53f203cf98a8ee0d84ce06c20748",
"1dd1699d3eb2d8876af8446952353521ed5ec4082a6c8ecec280fa8227914144",
"130a65f7b0764e4e0a89d89f2395ac850316e3cc4cdd19b2883dc04f68e87128",
"45a7a0b9e1c6e4754a3737ca57b55e424eac3d4cd02bc0441b38876174418e13",
"74dc61ba5401b62adbea6bda308bc2a0e494ab23e7b7848d6c3ded542a896c4c",
"125841b618bc99c07935aa9b85ec9d930723fd6a9c4603ef74d2dab09dc65a27",
"c9d0c77e3daf5c768623cf75d44fc1e9e5f5acf34aa1de54dc6671eaf9c06722",
"7812bb0109ac1597dbe26ec24f33beb277ebdcf8ece0957fdf81004f06936c7c",
"6ad83aab969fb9235a1c9ebc342f5fb9e7066c1ac09c816f58be2873d34a266a",
"902e5112f5563ad47f5056ab626d3dfcc6f365071cc81cbb1dd54498950e3364",
"de9664a687fa74bbc0c9ca809266225b2eb5d3b1d70e65b94bf83d6d738eb534",
"64a5b18cb77a133d4d1d22566336192becd67f2f6dac6a9a034e58de65694b4f",
"d7dc1326767c206fe0c133782db458b0fa47c88a5fbe3924e70d8936b0b15379",
"025fd9ca74ffac04677260ef58c25bbff30c226bc9cd38c5edf359f33901cc63",
"079e75fdad559d3ded439d47c115c013aad2446c05d0c735356620c51fc46766",
"693e12d46c86214b1daa1616b775557599ec8aaa2fdff8647e43f13ba8c24a7c",
"22c001224fe39465c0e6c65cbd43e1f75d180995490e5bd5414b03e39ecb2a13",
"2cb948c300b9bbe2798481725d73cd7adb66897d2cd2f817eec5580ed0ebe263",
"4dff5fccd7c808c174401c6b519ffef72738e12991d8fef02b0ef1789da70f0f",
"1f807a008529de48c2195d12ca298210d741ef138f88fde4e034791aef92e82d",
"d6ec9b52b199aac98d731fb0c8e4ce7f96284aff06e1b1184c701e2e8ca4937c",
"59367dccc051bd64e7a294454cf3ad484e6d3aad9c6edc183e70c8be51005c06",
"08fd7d5844ada5fec834bd1e777da652ac4e6e9e67352c02a0da3d7444aada41",
"664855d427ecdcfa5cdeb704f83cde5ee07d144da912a507b58157bf21a38d2c",
"86b240ac411f285635e564ca73823118b1b7948bf28ab12993be307e06c03626",
"99da63c4ebe34e6dea1e8ae62538850df98292b2f0f3a9ccd55dc3231c4fa35b",
"0f62b9420b101da1504539081c9acd19cb222ded19b6475eebc83ab7523a9b4e",
"6422b92ecc8af1d2fee44e94f674436cc1010439487eb99e7be69131bae6fd0f",
"c58921108b946fb6600ee5dfd2bb5930c6a65db18856bbc420a9ee6a321b1133",
"025bcaf1156882b422b90df9fbe7721c43f770cca447f19e506a4b7d282b6f60",
"5656820880ae0e85b4445538cde4f5f34cf3b67bb00c4a5ad053cd88f13d3939",
"d1084cd4a85306532128b076d1a5f41d0146a22a2d096ba3f0fd89d81213ef68",
"0f993a8b73946c5208c5e75b4270e6dec950ac97d746ca30faa79a66c525c619",
"ef9104d7664cf568d75b84b9078d0165adfb5834b34e9126cf97f3307f1bfe52",
"a529f59fa0eea94127a89069b0a443351092e2c26a5b74bf9e3eb0d4ca3c721a",
"08843f62e44eddd78b30e39d6664958f0eb6bf5e682737aa8967cedb2907a57f",
"6b708ac287a4b7ae37427d1f7c3740c005947d7e45e9678037a6d7898e212843",
"d1df7c6a7a74586c076baaf4c8e824a0871f4411d47d0187f8330e6f6781ff18",
"463dc46ca9b88bc680e2f1cd1c9f20262ded9655adf1b944bf0c341fcfd64063",
"77c3500132ee573c837c867b76cf5a39ae8925c55fb738db8a3fc07ab2c3003b",
"cdd6879bfa5236a8a0e9d249f70b6d2846a709d5c479e0efd19ebbbf8ba09d3c",
"9f42be4410bfa790432b952c93d2ba7d43bd6d86b6baba58b296dde441c97275",
"216fcc320858b408aa0075b6a67a299607f814808f9f01f19605baf9cb00924e",
"0951ca3b75edd46dfa440b2aa2c71b1b8ac073db7e955cc14185e45ae0475970",
"441ab52a5afa98696712ba25a1d13b9563115222868b4beda751f6b2a8dda009",
"ff92cb314d6bb1ee6120f5770f6935bedea28eba4f98b2470ec6585a179d9c6e",
"cc082bd1e7cad632e557f778c6590acddaf07d1346b182b8d0d81fab035e010c",
"4b396e83d9ba265de3d0997e272e68460a5c647420074cc4f1194ab4c1317935",
"6012cc8631bfbd53124833837a31dffb691e49145e7ca2dd0162169e71bbea6e",
"9e8e01df2e91b030c678e9144973e1623d85e1a79bb80304367d8c46e050506d",
"8e6b91336a52a588c5519a12875aacb0c348ee43a65ee70815849a94c73f3a71",
"da334677e48877a6c340cd596bd0a86d4094e323af7baf002d598e1b93c1dc4e",
"72f1164c03337477b084d6df75d7f9bf2b6bec6a180bed623c1d78467e5ce819",
"fed8e00ddb6ed9a8c805922a6ffef8b84c2aa1b6b33aa00bced017d8381cc35c",
"a9ef1a4b3abb1aa879c370deb5b1b951038903860d511db92541a3b845f6e10a",
"9825b55cecaf4f9eb60359bf480d91016b45bb9c2cb0b61f19a20a0029c65601",
"3979dd887aa5df449730bc7f45a7078ef8198dc18bd4630490010cbe29d7d813",
"4fda717fe4643ea79d254aacf6a6901076e65c5d65cabc79f328d22ecf5bae5e",
"eb49e2d61b37e5bdc4e66dc71607b996bda6f028b22d8d19d1186bb6720d7140",
"2da11c7c1f24dee64d8a68f7f105320f0c23e73836ddb2b2fad259d2cf57d54f",
"3e0e2e01f445d6be5f4cdbe13c12d44721707d1edacc48cd1461a37af6847405",
"ab3c1931429c3213edcf0982b665538ffcb4518c317fa225c91e64f475efc61b",
"dc9333c6969409a900d1bf9bfcbf6e9bfa0af61931bd33fd60980355455d0c6b",
"bcfaae0161751d496a126a5a8747fc2162a8fe9d9ec069d7b02c36b87929cc7d",
"32c47d0b8e11e735c2037da7f8414da1391678111d49978fe2664a7b94aacb7d",
"c692fd04c779b8d022e1ca7d79d40396e00d28141568b560bd77c2407485996d",
"154e1092de9c032b6c7b2badf0973cd8d39983e6f8f42a71a073d3adefb31a7a",
"9c867c17181d1d9309dd607b573640c91d1f314cdd44c169ef3c0b235ac0c714",
"e9acc70bd21f69be5169dddd034391a0b7f1a7d30c526e3bedee4c7768870731",
"f1a46c3a2ee9b66d45558ce1003511c1d756b7e9f5232fed3bdb2e6b4e652069",
"a5686bfc87ace34eb69238e3cb2df2b601d3984de3a65a49e91e5235b757c37a",
"f845f2b59c457428c3c63aa84452d29c15b7a422e9081042010f17e926443739",
"eda2f6544dad4d19c6500d9af584a0d1adfca1d99941138f4f612cd0e6126263",
"47fd035dc410c9561c728715bc68ff01a7da12ae032182b1e018a7556ece0102",
"14f5c5c373c59d5010e239f8566af474859b56ab22b010cb5db5391d328ca412",
"31b4ce9bfab00a3845977c6c7c946b37cb9a46668b88249739cc95596956ab53",
"718f4d992ec6d1201a3520d6089b85483ad4f6b8bf0b46396cfbccef39eb4438",
"14caf7680daa3459ebf28f3a39f1e2033e32d7ac2499d1d61def048aebc57805",
"ba66268dd83766986216348bbf5fbb03149fafd0f20858783eea184d1b3bda70",
"d166300d138ab99a2e357cb02bda78b17de1f676acb2fc39359bd457bf81ac46",
"fb556b0a449253e14850e21d1a1028dfc9fb1425c33c1de3e9fd0334eb7fd478",
"7b6c356e7c8e6f1a53ce74495e60d80cf7c14e6617b81e56015b012b6716de43",
"0e806a5c5ca01a8bcb5fec9b7e5f4fddda50d3f4e3eee3ac93705ed81f6db04c",
"9995a1a7d21dfa46c9f6088335d558da608ebf0042f117459abfe4036600e26a",
"6b0a737cdcdab38189c2bde6cedbe0cd8a5230201624deb7bb4d8f24f942107f",
"1cb89ce0f691c9712a7f631a6a56b5143e3fa3a4682c44e85507ab2238a11071",
"2555b88a7865b86b6570b77bdef3e74134c985655b18f70b41e4e5c8abc28e53",
"95aa344162c3947ec8e03c25966c0764ef9bf3ca4a73eabbbf1da6a35abfc007",
"fc618a6d1d2bbd9d19a786bb39ae36caef4314c33097fd118c312d4165c05a45",
"92f16142695ceeb8b60a1340c945490d0379665eb626761f032c248a1aedfb54",
"8cecbabbf54f5fa3cf2c23ae2de3b54117d4469b1522d24666dba02314473147",
"d62cb538aec856d1ff607df853549ee6ea30331ca1fb8df0cb14b3b0b91d4b3d",
"73f9634c4bc4ad9dc83c395ab68336e1d1dfda3c323d7507228405ac0be9ad0f",
"4e714f3beb3b7e656b7262f25c08a0afa366139273e61f4aa6a59255c9da9005",
    ];

    static KAT_HASH2: [&str; 10] = [
        // For i = 0..10, hash-to-curve using as data the BLAKE2s hash of
        // a single byte of value i.
"fc8b574bff48f44c918627d9c5b7c29eaaa42bff178dc5dc5ce9fe2ecf800217",
"2809656525a5ae6c81396c69fbdbb0046ea9341d029fd8c8e6fc977c1357a51f",
"ada2ab68510fe755fd8f73f450a030111b2aac1e44432ea0d6ddc27d3b1de966",
"ae23156a5fc4d11a3427edd23ad40ff94815ce6a108a11c7f76c8d4829aff93e",
"ff354536db9b148840e938a9b4d2075628b6fd5a55e7450b60f4670b72fd6a6e",
"23616414f783da2c12f83dbea9eefc4687847bba2aa3f96e449fb994be72055c",
"ea4a3e8791fb366d0fd3eb2428d4544936ca169144505dd849352a93238c1674",
"a1d14c87913520649bd22a3e6a431d289c71137e7aa56ea19d91093781f17f0e",
"71ccb7e6b84a3de606d4b8b6869e814f70100cb03c837c7f7d750b56e28c7d31",
"8550b7e7cca2b515c094d84b8ed30dd553cb3f5633284313df373d0ef905c540",
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
"44d148505c1e49d76904300350c4a22029269fa17c7b950835601aba55f5cd3f",
"7ec3566f181fee5b8e3395032e9dfd604746d04741a4bd194f3914ee8c2f2f09",
"",
"ec14004660d4b02da3b86b1bc5afa7b2e4827f0ee1c9a25472a2bcac521bc231",
"6beb8fa58895214166fc1456e7d55ed9cf988a149a2aa246fcd3204436c9ddf8c98ceb542efd28e1422a3e93364b9739",
        ], [
"6dc9600492a3027dab067dd19acda6c1103e42c3a76f1414caee702672b4832b",
"5eb8f1406fb20166f5d477e429030cc1efd9186a829e7d117858608144516b0a",
"28",
"bd1a4655b90f873c53fe908f4109bb8dfcd9096312b447a6434af3c35304b7d1",
"86a11485e8747a668fffeda4e65bb4f9bfef0a20976d004e3b6057253c4c2f67c2689f948d5cb75ce718ec00d6828038",
        ], [
"899bece1c0a52828f763c2dfcecd0230da61ca06d2f36498dbbaa97fb710ce3a",
"71ae147bd7aeb936744196bffab557882cad4f1d40cb5f66a0d3675242bef62d",
"886f",
"6230441be7f030f180e81dc44502b24ed94260490d140ae738bb80746051651e",
"4cede9ea3e01d72546681ea22056b385a068127d895a46f907eaecec35ba91d34f552ce5d1668b5fe9b88b9004c1ad0b",
        ], [
"3b041a54fc960a41bba568a06e1f02ef2b9a296396adf6f9343e79b5f494f838",
"02bfaaa73335f342f2bcf03ff53ac5317ddeddd4cb55885785fbcbd362f0a44a",
"77c941",
"e877b70f8c12aff466a4dbd6284bd0c6ad7cf66376bdad599f22145f8277bc52",
"21011d4b3ad0f6c4a54f84bc7f8103ae044b3a8343b4d40c422a085fa3cf6479760cb1182702e00bf42a0cd021fa1220",
        ], [
"c62a8f3d12bef46c1b0a05ba830b2c43541ef2bf156e01cc6acb392b1754cc0e",
"c47650e29993d826c5c958003c0a20b88d5132bbbee8c5c3be64de92828fdb09",
"8ce55953",
"b4c94e55cc622b96b49fcfe6b913ce3a06050b7e9b26fe840389145088d59502",
"dd5799ff85ec893bd769db09f31792de779631a0c6cc74416b1382dbed783cf781ee99c14e7fd93427237f2939fa030c",
        ], [
"40e8ac753a7ca7dc54907e10d1f586730516831c1eefd0a6a7f43fcf1b18ce3d",
"68c04bc8fbc6e907257fdbe05817ff8192db31495b28240808cc024e448e9c4e",
"2de4b45f74",
"a7ad895209663ae35bfb3fb0e44cc83616bb876d14608e5b09c20d19f57839d4",
"814b044812dbf7a2960b66a0fb6091f08026b2cfdd4644e97ebecca8a425bd9bf2958e064534a1d2e369ad607d7e070e",
        ], [
"c9df9a54d6210ec0427aad7701202116a53c4b6f0ae2aec1ccc88a3815a8450d",
"265a74e78ebb564aa2549afb8b32e4213165cf72b0371cf9a98dd5f478d22b79",
"390da74e4d36",
"0bd1a3ff8506a918b8bd733c31cec084927241dda2ede63f719a6758872c94ab",
"93dcb37516ea8b6e9b664650e828bfebb86a4bb5fac63c1ebe77d5193b9cfa48d4806818bb4e0a5df202b8bd35a6f122",
        ], [
"1b7388f24651e893b190339f7311a7131b1a20db1c6d367f6cc2daae38f62220",
"6d57eaf852fe4e02face6fdd2472f9a1118e543e140b26c41a32f464fdb2954d",
"d4793450f2ab8d",
"f328909fd158f3541c2da54b758ccf750bfe4afa717b00094fd30e7fd69661e3",
"b60efaf4b8c78bf964dd08b15e26487b39e34de730f81f4f204937ba86b48f46ac9ac97e81b18474ff9a54104c47ef31",
        ], [
"43b0604981453629d445e721a002549d3a67539837d4b42803199bfd941d2722",
"d53cf56919db086bfb9295124cbd42e90625f1e0737016bc5252e028cd433074",
"4b60d66f4259f746",
"7de5f8c2c35149558c0a6bef84596669100f6350f07aefed58120d6dc3531231",
"507aff485887e755c969509c13451e4654d0176183ebddd58449e8455e4fd209eaf0611e990bbea92c4b34057fdcc20d",
        ], [
"40f531164f591328487c958910c32c8ddef7438265b58f88c52d4cad7d4ee11c",
"b7835dc5220efb6da27393aeec5805f9b86721892da1cc2ef4340f7712871e4f",
"6492d2cf155c1c0b9a",
"9fbcee44419bc19b97bb673d0055faa0aae1861f44c682345fb3494e610e26da",
"1bbeaa61241dbe92be3c72ab4eecb7d299daef91044350b1e28dc5ab0ad2fc3bdde2b87e5cb9dcf0f5cfc183b2f18b12",
        ], [
"78be8fb66e5b281cc839bbaf4edc14052269010c914e120ff6298f4a9e164c13",
"6c86a99601d1ea0dba0639523c631c4bd169153f1408adf3779eaae64d9ced1a",
"29f1f5d318ed07778a62",
"4ca14993a888660c624f816db0c893bfac69d5ddf04cced60333d94ac1b0e2f5",
"981c02d1246bc5d8e035cee31a1401a207fd81b3be5205fd8def10ee9dd7b4707286d40882a191939ba8042ce4b07e04",
        ], [
"ce78c297c44c76db62e0ecd4c157fe1495881a26fdd0f97442e08987e59dd301",
"6f56cf57bc21e8a4c9523683bd670c43aff19e4297ae62e8e6dffa40b6d5710a",
"60e16e31ff2209949495c6",
"ed427029b6afbfe2a73c7a73605bfb47b4db8eadc940bddc103098a06d7b7daf",
"4fbcd966d5f718cb419fd86c25b9b92f6185afd888330f2bdc5754b08410f09e146c4a502877a020e187ba1d2056853c",
        ], [
"f194079ab80a7b4c4b4c6a7f22f7b34d7e9fcd03c53593d067689f19bcdb9e1a",
"0f59f99ae181d9de6fc4ee383555970646a39ccd66920e87e3073ebfccd9835c",
"a9c87c4d948254fedeb3dc88",
"2b083962ac0f0d9421bffdf9377f06e7152c3677e911029b08f9d40688c8aaa8",
"1f4452a956e35365cf9088daa76bdf4e386bf3b47ccf097871d841e03874067cca94ad4e8176bf88d9522096bde4fd3e",
        ], [
"a903e2211ca913be5884f85ace2f0d8985753666d4b28ab4776fad85be6cce0e",
"1123349523fb96f89c484b1c6b88c5af4c7dd97ca020f550abbb30722d8fc83a",
"057f57f0468a62a12c7bd68cac",
"cf44d2ca3441b9089e99a00eb90fe161bc994990469a46b488e08711a7ba8d9e",
"7ecf72b0e37bc81a83df0c716822ccdca54ea34b1378335604acb4002eff0b606695c4c5ef18eefdfaf31fe05e3b6907",
        ], [
"96e928159796d2546d71253b3d15561cca131d2c5822e0c2dc4bc5b303b45932",
"a6ade506570c580ee1c065d6210280f075060fd81316b69838ef1c14f15e6951",
"0e97ec23af84fc1ed53ca9af0b62",
"79d41d37434fa78c4cd3fb421c7caa26704df53c215adcc4f7807adde10c7438",
"1fe913b6be257cbf7eea7cda8e6818314eaee29523d435f5539b5f75c2ec4beed495dafd1cc11c50b2fa7bc6f0433333",
        ], [
"7f2fca4dbbd1c853b0d2b459fdfa7ae524640e7fa54700bbcef2fb4f1c85d00b",
"0c1e5ac0eaa6d864b9e6fc497d0d8f624f222551d998541a53b91ae326e95a05",
"0f30a3a1da33359704f44cac29200b",
"0756a67df9f84be0d319c4e8d324f3b77077f9322f9603f015df27f2804b17a2",
"27d2f08e2a5d7bf9f5d1b66c2d22e9a9cfe709f5dcf9b299101a518d33db8d70e995d7243c562a8e9500ec280190e93a",
        ], [
"c7399bc25f9369303b630bfadc51990dd369e19e9b790f14f42e91b8856a5424",
"1ff5e73a0ef6bd6f1e9bf4240dff4035c64ca01c2cd55e2b0c0c77ed2e9d594b",
"bf0e62025628565a960505740dbb330f",
"86b36dde6d628b67332456b5d41d09737a057215f72f89094d071422e705b82e",
"ac01832b18585aa10a35c1956752d8f7763766ef11a3f9c0605c175cfca4f3bb7037f7c07b3319798b9c71531e6eab0f",
        ], [
"317b87e497da33ba7c213be9ea2644baf90e222a50fe81ce85e78bb9b85c8c11",
"266a647561333dc99bff485fb04374484c766754bba1dd5c647c37348e4d7c4c",
"fefbe9b06ef6a865c92fe28ec9f442f594",
"86497726e18b409075f7036b1c65deacab22cf85d2ae64ef1857e17a9713e4fb",
"8ef452bb82b45406f539b7bae4bb4c41fc07eeeb9161ac8a5b461811858e46133479e64b21bbde6cf35326f9dc18023d",
        ], [
"7631d80ae612bfda8da1b0c0b9f20da732a149e3b055d871faaf83d1bea82f1a",
"60c3d7dc1d1a6e2287161d09b05d217d545d1b7d6952fc4bf96a4087501c0e28",
"80e283c7d4f2137313fc740fd4602f9d9900",
"a2edb2c979a443ff733c32453d350f09af33068a5640af90940315e7d3c87957",
"e24ee81e1181cb497bc8c30a92439cc8362cdca8d04039343a4fea16fa6418d112523ca427abc3057daba39ce1ceb817",
        ], [
"63991a7f4047abc57e62052220973ee9dbb3e304dc0dfa23fb5b3444a6ad930b",
"9d1bc352dbba595a0f28b88bde70731e4c5965d8922239f64c92ee9e36e2062c",
"dea1840613f0b9fd8a183ab1671f9ab2590354",
"ba789f5876b8db6ae44d0e4507de9993c83e504804c1f3f8619adbd717847b77",
"dad236db7694c7fe3136facd2d9c21f822e47400d93ab4d4b5500d9a608e56d43c26d9a1cda93ecebc4be98529104136",
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
"2ce6a3022224881fa4cac94bb1f1cdd5f512c60cc8cfe8d3d1bf7d3907815b22",
"0c954b27b026d9d2e674f45c1ffc4733123cdfcd296fe5dc10eda85c9ce38175",
"bff8f1e8e180736688435f45f9985cbcf8062fb19a510d6d52e63f284abe3874",
"83129bac9ece882792cf11f962eb6ff175511914d5c3fc1762be549ff873470d",
"94cf6ad65028d5b65866adb96332abc749fc4d4e5b0ec425b9e8bd2cbdf69c6b",
        ], [
"4d7583ae41b468dc745187cfcaf9a92783790a02e1a2599468b41f0c5fbde039",
"f897c3a95b1607a0f758c67788570b6fac5afd1bf8f0f8cd9920063c480b615d",
"28201e9c2090b7bbee214c2cb34ece082f458f3c55eba8a6882a768984036bbb",
"43b023fe48c0e26764538806296217b58c7a26458a9491e74b303b0b1ba42732",
"ec9058eae2bcb1820890b7d942406596b4b5a750a8d7c656c289237c9e373168",
        ], [
"3a82c8500d9deb7348e2dd182d76b28184571507463a30e3dd6125dd3216fd37",
"89a286eb077ac8abd3550a08c578496a9159e58a2c4e4c86a2b77edc1c7a1b38",
"4744b14ff920567f60ec079a5a9ee66ca4109853f39d5485288f6da2e489e4b5",
"6d071a85f7cdda1867c7e2a0494749970181be21ae9231e7bec95399c70b6728",
"ed85673f100f7067641ee4d7469b09be303ec03e11364a24c1df84ceea24e7c4",
        ], [
"1dfecb52856f645fe27b73848f9f6cf959a17cc22ae9e93f417de1d78f5b022b",
"1e43b0416db5e8d114cf8c37c72cc18754f3930072fd6f1169bb96cf51f21719",
"d4da9fbbd7837aea06f1ea415a9490ca465d5c64b74dbae37d74c39f05ed0888",
"9cd5c268535a49a798f051ea58c4cbea87308f882a318139df7fdbb20959ef51",
"db056bdbf9c390ccbf84d1f821409a12efba0976b09be4c224b7c2e0564e60b4",
        ], [
"f93389968f863587a02c4d3d023a88f6348de90069a3f2697d9d37cb49f3ff21",
"26f22b8594808944de6db8a46724428ea8e44669ccda34ceed064e13d9c4b221",
"564dc4b140b7556c9f5cbae3c315ae824f75cd9271cb1e32cea55de7223b1825",
"33416510e3aeba7dc089466f87bec6be8fae73bcf513c10e6e3cf343cd87c124",
"918825d1c54c13dff8bd444fb24ff2351df6cabeab6ca7c9502fc49f8db4787c",
        ], [
"33bfde4aac21966176f5e5148d5b8c25470695fdaf068f4f718c0613a6afff2f",
"00aed36111831d40a4379d10c4fc0640f0a27a361ec6bf90d9d1421dd5e89224",
"d7a1938568bcc9acb137a336484c7d669b5b7cbc1c5eb9464715ea6312306641",
"a8a5b4c5bd41acdda08dccba573961cb7eae8a91d8ff387071328e6df75b7c4a",
"f0a7742feb7fdd5850134c62d435f4fb338246271f6829488f83b38309ea9c61",
        ], [
"9bd86fe3e3b288882036dae8a712f6eeac0a00737f159ee815ec2a01bc31ef1b",
"dbb12363bcbacd3514815706df030972dcc2394bb06651f6cee5795329586c73",
"4762a117f63999a7b661d73320fc190541125fa149abe37ddc01cf1460b740a9",
"7d28599d1308306a38740b032bbb0c80e0b31b84ece0bf5f23e46c53ed0f1f7a",
"d8d232bd469de86cf648fdd8bcd6f2dcf73ab88c554b8e044cbda2c2c49a3120",
        ], [
"4b80c2ed5a5fc1e2ca4a4cfecf785ed9c4dd41becb7e27b5af43f90addf1a922",
"9a32baae43098b855d018f45fae3af949b7de036935edb5bc21584e60640aa46",
"3955008ed07d6e1d82d13a19983ebfbed29632df447b92b24c7f4232f098361e",
"9727891b391ace0996a5fd66ea56b69351096662d6dab9bda1e4e9d3a1eb982f",
"63bf4bc35f7cf073084a2adecb520939e9f7c4042c0b94a7149deb0123d0d859",
        ], [
"7c78a65c58796318ee0a42058713fe881e20f7d71e06b8feae9c387794c84e18",
"24fa74f08b3a80316dba48f830065ee4833344b11c836215c79a7d014d2f7608",
"6b87c285f4b52cda98e8bc247f4d21ffb1c224fd25c140ea573f927d9e29b11c",
"059a26f7b26c39707b2b0f540dcbf991217500df23ff058aed9d0f031f90ed69",
"ea3b5071e55e231f06a117d4c42e931ffb77487f06a1dfac983ff75e904a6bc6",
        ], [
"5aae94acd70415b7b73ec6d40daa9ef1a72bcbb50e0b44744bf8db031ecc5f39",
"03c451ee809f9eb338a29fc636633caef5065690573017b014f7e0f2496c6523",
"3780c986a4f052d6c9dd1a83ebee54dcd673c8fb1f6ccfca9d9d2570cb5cc355",
"95d014459d19585cd4ecfb4f27beb4ea7a015abb4773dcfcd15e757d98534075",
"9f78587cdcac1e67848c17895f70b17eb17d30b9d5da88409dc312a122f23275",
        ], [
"e4a8a2c8049f504acdd06a27bb5a8f99c19d9980a157732912cb627825194300",
"df25864e27c5d275465e458c8bf6fe9f09c4697d5749e49b7127f95a6f9d1020",
"d45a834737cb0f2ea0a239eec624bb4872e8394b6a62389448f3d333b0077dfa",
"1bb067a317144e439894e0dd3feedb6f8a03ca6b0e6893a7d363ed07fdb92034",
"29d5f4a0568d34631aa82ec022375cdc1b21c85039489ce54f01dd1f6741e411",
        ], [
"e2ddc61e6c6ca46274f31568a67d623dbfc155f1c71b9ba756feff8751d0831c",
"889123806a3a59de088546dc412154cb88bfdfb8bc4817a9c29259ea34540c0c",
"05370debb414ad6160f3e7cb20d3543d7713d3cdf28bdd9d6b1fab7e3c098246",
"bbb9278ff3aebd78b8aa992dffbb60b8914c422c86ef753dec3e748cec733953",
"f15cdb4bd5f7d392ff8002773999e4eac098387746ce4ef68c5d9815e87d87b5",
        ], [
"fdbe81d0228b0f4ebcf15873e9d8f5b93e3c810d3a506de7cb27a4c1c967ec14",
"9de7cbf1ee57aff85d2376032c823a413fa2473707aaa62a426ff9ca77bab940",
"d1936d883b179c696c0646df406292daf97cc2a55a479c6df83f4f076cab30d1",
"304f39a669c65db03e98c81cb0273f5aa73c82958654256933fe6b7186c0862c",
"bd0fb8f39f17408221ddc3374c0a9ed97805d2eebf6491b6537661e45f813b57",
        ], [
"68c30778bfedd83e55b223e9fb4054ac4be0aa4667dbf1d9e9a34fc5bfd47e21",
"cc6821938a0f5f5ed46c48435dfceb173f0129dca91a0fe001ca9ea29581805a",
"9146537af3b5edd0d446e8401f8faaeff23764a04bdc4e9ebf9493986b5ca288",
"acc9924832e02591226dc457a9dbf1538811c8cb4b958910d094fce2d7e9d122",
"433feaa0ab3d43ce0bd84b9341c52ac43e8f34b1c92dd42195ab47a12efdb63d",
        ], [
"ef0c699fcc1a97ecd50f47724120cd5eea382b4062f6b39ae469fbf68e61563c",
"f6d0433e2130604a86ae83e9637354343508bd44d7d3b518fd6b6e6f91d9af0e",
"5ae49bbb8746288ba35c53ea423d6d3256e46b8643d02bc109a849acc7fca42d",
"e0ae6f872715d131a786d1d65d95e92a2ad476797d57c216397243e34ce11b5b",
"06be64a5ffd9200ec3825ec36e2751b92ac2d440f9afe096b7ba531c2e68d0a5",
        ], [
"4e06c65ae6a8008e49d96cedb15b9997236c9b045af475a4f04f64fba1e07e2d",
"f672237f68c7e7648fefcb682de66d48b9d47fd1e9ab2b1ca9a5ebb9f439e510",
"695d406e143b9db9b7916619fd198a4d86ede73143cb6e40d5f7dff107d36f46",
"045dcdcf5d14fed0122328a04a152a9874a99375974c8b56b5524129013cef05",
"e2e55cd8bc8905127d4fe90ebd6715c01f8569c59e46cb18bbaf1b02f29f1e10",
        ], [
"f8f6701ad99636a4e3ef7769fea68a1e19abaa81924ef3344fcbbb0690c2d310",
"156d1484dec6689923cbd301d44ac093fff06f1e10c3d88d5f0e1be95b17a333",
"a8e4c1c3fb07ea2d02a5f48d67eceaa18ce7b1bb0f846c66ac5919122ffa3712",
"7622480c01d4d674ded6e257247a0b1281356855e2178dc3ce3c832a34234576",
"c0fc4d32a0aff5ef25740fa8a512ee56ae60f0d9e5202a425749e655162c5fef",
        ], [
"3014cc8164ce0d26b9413f9193d1d649977a3b5a99dda74436aa58d7d8ac4f13",
"de2190610edf00d7ee6903b5193c532441feccdfd4f34e2b155f4f3f0076f35b",
"8493d63c43b1d19722d1ce39f4283f9e0879072c8f11a6f0cf7d279b6bc4022d",
"91a1a25e538f047fb5bf71bc6a38f76f2579f7f60ecdf33d532c861cd4651926",
"f86d3c74224276c1603fa02d7790f93d824f19a057a2103bbbd015f6ccc99a72",
        ], [
"4396098733154911565c8cba3b3a4d179b01a60992fc27ec98d21ac2cad72723",
"a33e1064b9ac643c9d12e3e04588666976f0d656d06d2d6ed381ded82a52ad29",
"aecb4ff8cb1dbf314edec1144c541c82e430014b3d297de502a9c000c2663b78",
"270a1b39a4e1bbefa55475b90272baa4916e0355a618ae0aaa5cce9b3412c647",
"9bb224260b1e228ef7279a1557eb613561174c91eafd6defd4cf6bd7c09d3a9f",
        ], [
"241502ea684057cbdc5a8f523503dee94933ff0622fb89adad6d022bee966338",
"e809a28e1f672a870c00652bb41e6b455013cb11073e8be75500e76d1611f409",
"7054ce87174d29d970b59dd27cd31d55adca70a904a81b70a2e03c426d9cf322",
"3b3c149867c44030878a4374275e938bc99725f2b182e33bb925870e68251166",
"391049ce061ae09766252cecc6a7b23b90e007497412c668ef178dee1897b63b",
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
