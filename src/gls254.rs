//! GLS254 implementation.
//!
//! This module implements generic group operations on a prime-order group
//! backed by the GLS254 elliptic curve. That curve is formally defined
//! over GF(2^254), a finite field itself defined as a degree-2 extensionm
//! over GF(2^127). We follow the initial parameters used in some previous
//! papers, in particular: <https://eprint.iacr.org/2022/748>
//!
//! Field GF(2^127) is the quotient `GF(2)[z]/(1 + z^63 + z^127)`.
//! We then define GF(2^254) as `GF(2^127)[u]/(1 + u + u^2)`. This is
//! the base field for curve point coordinates. Elements of GF(2^254) are
//! represented as combinations `x0 + x1*u` for `x0` and `x1` both in
//! GF(2^127). The types `GFb127` and `GFb254` implement these fields.
//!
//! The GLS254 curve equation is `Y^2 + X*Y = X^3 + A*X^2 + B` with
//! `A = u` and `B = 1 + z^27` (`B` is part of GF(2^127)). Its order is
//! `2*r` for prime `r = 2^253 + 83877821160623817322862211711964450037`.
//!
//! We map this curve using a bijective isogeny: `x = X^4` and
//! `y = Y^4 + B^2`. This transforms the curve equation into
//! `y^2 + x*y = x^3 + a*x^2 + b*x`, with `a = A^4 = u` and
//! `b = B^2 = 1 + z^54`. Since the isogeny preserves the group structure
//! and is efficiently computable in both direction, this new curve is
//! fully equivalent, from a security point of view, to the original
//! GLS254. We can then apply the representations and formulas from:
//! <https://eprint.iacr.org/2022/1325>
//!
//! Group elements are curve points `P+N`, where `N = (0,0)` is the
//! unique point of order 2 on the curve, and `P` is any `r`-torsion point.
//! `N` is the group neutral. The sum (in the group) of `P+N` and `Q+N`
//! is `P+Q+N`. `(x,s)` coordinates are used, with `s = y + x^2 + a*x + b`.
//! The formulas for such operations are efficient and complete (no special
//! case). A group element can be encoded canonically into a field element
//! `w = sqrt(s/x)` (the neutral `N` is encoded as zero); the decoding
//! process ensures that the input is valid and canonical. Field elements
//! are serialized to sequences of 32 bytes:
//!
//!  - For an element `x = \sum_{i=0}^{126} x_i z^i` of GF(2^127), we
//!    define the integer `vx = \sum_{i=0}^{126} x_i 2^i`, which we
//!    serialize into exactly 16 bytes with the unsigned little-endian
//!    conventions (the most significant bit of the 16th byte is then
//!    always 0).
//!
//!  - For an element `x = x0 + x1*u` of GF(2^254), with `x0` and `x1`
//!    being elements of GF(2^127), the serialization f `x` is the
//!    concatenation of the serializations of `x0` and `x1`, in that
//!    order, for a total of of 32 bytes.
//!
//! The most significant bit of byte 15 and the most significant bit of
//! byte 31 are both always zero. The deserialization process checks that
//! these bits are indeed zero, and rejects the input otherwise. Provided
//! that these bits are both zero, then the input can always be decoded into
//! an element of GF(2^254). About half of the elements of GF(2^254) can
//! be then decoded into a valid group element; the decoding process rejects
//! invalid inputs. A field element can be decoded into at most a single
//! group element; decoding is unambiguous.

#![allow(non_snake_case)]

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::convert::TryFrom;
use super::field::{GFb127, GFb254, ModInt256ct};
use super::blake2s::Blake2s256;
use super::{CryptoRng, RngCore};
use super::{Zu128, Zu256, Zu384};

/// An element of the GLS254 group.
#[derive(Clone, Copy, Debug)]
pub struct Point {
    // Z != 0
    // x = sqrt(b)*X/Z
    // s = sqrt(b)*S/Z^2
    // T = X*Z
    X: GFb254,
    S: GFb254,
    Z: GFb254,
    T: GFb254,
}

/// Integers modulo r = 2^253 + 83877821160623817322862211711964450037.
///
/// `r` is the prime order of the GLS254 group.
pub type Scalar = ModInt256ct<0x3CBDE37CF43A8CF5, 0x3F1A47DEDC1A1DAD,
                              0x0000000000000000, 0x2000000000000000>;

impl Scalar {
    /// Encodes a scalar element into bytes (little-endian).
    pub fn encode(self) -> [u8; 32] {
        self.encode32()
    }

    /// The square root of -1 that corresponds to the zeta() endomorphism
    /// on the curve.
    pub const MU: Self = Self::w64be(
        0x17E6D0D00F54BC93, 0x9F58BDDA363FE499,
        0x1EEFADF1FAE163FC, 0x1B8487FC89A1F614);

    // mu + 1
    const MU_PLUS_ONE: Self = Self::w64be(
        0x17E6D0D00F54BC93, 0x9F58BDDA363FE499,
        0x1EEFADF1FAE163FC, 0x1B8487FC89A1F615);
}

impl Point {

    /// The group neutral element.
    pub const NEUTRAL: Self = Self {
        // Neutral is N = (0,b) (in (x,s) coordinates).
        // Since s = sqrt(b)*S/Z^2, we need S = sqrt(b) when Z = 1.
        X: GFb254::ZERO,
        S: Self::SB,
        Z: GFb254::ONE,
        T: GFb254::ZERO,
    };

    /// The conventional base point (group generator).
    ///
    /// This point generates the whole group, which has prime order r
    /// (integers modulo r are represented by the `Scalar` type).
    pub const BASE: Self = Self {
        // Generator point comes from: https://github.com/dfaranha/gls254/blob/6f8b07ce848fc8990627fe6dfdf5aae7a61c1a65/sage/ec.sage#L20-L28
        // The original (X,Y) coordinates are mapped with the isogeny
        // to (x,y) = (X^4, Y^4 + B^2), then the point N = (0,b) is
        // added to obtain the point whose coordinates are provided here.
        X: GFb254::w64le(0xB6412F20326B8675, 0x657CB9F79AE29894,
                         0x3932450FF66DD010, 0x14C6F62CB2E3915E),
        S: GFb254::w64le(0x5FADCA04023DC896, 0x763522ADA04300F1,
                         0x206E4C1E9E07345A, 0x4F69A66A2381CA6D),
        Z: GFb254::ONE,
        T: GFb254::w64le(0xB6412F20326B8675, 0x657CB9F79AE29894,
                         0x3932450FF66DD010, 0x14C6F62CB2E3915E),
    };

    // Curve constant a = u.
    const A: GFb254 = GFb254::b127(GFb127::ZERO, GFb127::ONE);

    // Curve constant b = 1 + z**54
    const B: GFb254 = GFb254::w64le(0x0040000000000001, 0, 0, 0);

    // sqrt(b) = 1 + z**27
    const SB: GFb254 = GFb254::w64le(0x0000000008000001, 0, 0, 0);

    // sqrt(b) as an element of GF(2^127)
    #[allow(dead_code)]
    const SB_GF127: GFb127 = GFb127::w64le(0x0000000008000001, 0);

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

        // Decode the value w. Remember if w is zero (but was successfully
        // decoded as a zero).
        let (w, mut r) = GFb254::decode_ct(buf);
        let wz = r & w.iszero();

        // d = w^2 + w + a
        let d = w.square() + w + Self::A;

        // e = b/d^2
        // d cannot be 0 because Tr(a) = 1
        // Input is valid only if Tr(e) = 0
        let e = Self::B / d.square();
        r &= e.trace().wrapping_sub(1);

        // f = Qsolve(e)
        // x = d*f
        // If Tr(x) = 1 then we want to use the other solution f+1,
        // i.e. add d to x.
        let f = e.qsolve();
        let mut x = d * f;
        x.set_cond(&(x + d), x.trace().wrapping_neg());

        // s = x*w^2
        let s = x * w.square();

        self.X = x;
        self.S = s.mul_sb();
        self.Z = Self::SB;
        self.T = x.mul_sb();

        // If one of the checks above failed, then we want the neutral point.
        self.set_cond(&Self::NEUTRAL, !r);

        // If w = 0 then we get a failure here, because this yields
        // d = a, then e = b/a^2 = u*(1 + z^54), whose trace is 1.
        // However, we want to successfully decode 0 as the neutral;
        // self was already set to the neutral, so we just have to adjust
        // the status flag.
        r |= wz;
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
        // The neutral encodes to field element 0.
        // Other points encode to w = sqrt(s/x).
        //    x = sqrt(b)*X/Z
        //    s = sqrt(b)*S/Z^2
        //    T = X*Z
        // Thus:
        //    w = sqrt(S/X*Z) = sqrt(S/T)
        // For the neutral, T = 0, and the division by 0 on GFb254 yields 0,
        // which is exactly what we want here.
        let w = (self.S / self.T).sqrt();
        w.encode()
    }

    /// Creates a point by converting a point in affine coordinates.
    #[allow(dead_code)]
    #[inline(always)]
    fn from_affine(P: &PointAffine) -> Self {
        Self {
            X: P.scaled_x,
            S: P.scaled_s,
            Z: GFb254::ONE,
            T: P.scaled_x,
        }
    }

    /// Adds another point (`rhs`) to this point.
    fn set_add(&mut self, rhs: &Self) {
        let (X1, S1, Z1, T1) = (&self.X, &self.S, &self.Z, &self.T);
        let (X2, S2, Z2, T2) = (&rhs.X, &rhs.S, &rhs.Z, &rhs.T);

        // Formulas from eprint.iacr.org/2022/1325, figure 1.
        //  X1X2 = X1*X2
        //  S1S2 = S1*S2
        //  Z1Z2 = Z1*Z2
        //  T1T2 = T1*T2
        //     D = (S1 + T1)*(S2 + T2)
        //     E = aa*T1T2      # aa == a^2
        //     F = X1X2**2
        //     G = Z1Z2**2
        //    X3 = D + S1S2
        //    S3 = sqrt_b*(G*(S1S2 + E) + F*(D + E))
        //    Z3 = sqrt_b*(F + G)
        //    T3 = X3*Z3
        // cost: 8M+2S
        let X1X2 = X1 * X2;
        let S1S2 = S1 * S2;
        let Z1Z2 = Z1 * Z2;
        let T1T2 = T1 * T2;
        let D = (S1 + T1) * (S2 + T2);
        let E = T1T2.mul_u1();         // a^2 = u^2 = u + 1
        let F = X1X2.square();
        let G = Z1Z2.square();
        let X3 = D + S1S2;
        let S3 = (G * (S1S2 + E) + F * (D + E)).mul_sb();
        let Z3 = (F + G).mul_sb();
        let T3 = X3 * Z3;
        self.X = X3;
        self.S = S3;
        self.Z = Z3;
        self.T = T3;
    }

    /// Negates this point (in place).
    #[inline(always)]
    pub fn set_neg(&mut self) {
        self.S += self.T;
    }

    /// Subtract another point (`rhs`) from this point.
    fn set_sub(&mut self, rhs: &Self) {
        self.set_add(&-rhs);
    }

    /// Specialized point addition routine when the other operand is in
    /// affine coordinates (used in the pregenerated tables for multiples
    /// of the base point).
    fn set_add_affine(&mut self, rhs: &PointAffine) {
        // This is similar to set_add() except that:
        //   Z2 = 1
        //   T2 = X2
        // cost: 7M+2S
        let (X1, S1, Z1, T1) = (&self.X, &self.S, &self.Z, &self.T);
        let (X2, S2) = (&rhs.scaled_x, &rhs.scaled_s);

        let X1X2 = X1 * X2;
        let S1S2 = S1 * S2;
        let T1T2 = T1 * X2;
        let D = (S1 + T1) * (S2 + X2);
        let E = T1T2.mul_u1();         // a^2 = u^2 = u + 1
        let F = X1X2.square();
        let G = Z1.square();
        let X3 = D + S1S2;
        let S3 = (G * (S1S2 + E) + F * (D + E)).mul_sb();
        let Z3 = (F + G).mul_sb();
        let T3 = X3 * Z3;
        self.X = X3;
        self.S = S3;
        self.Z = Z3;
        self.T = T3;
    }

    #[allow(dead_code)]
    #[inline(always)]
    fn add_affine(self, rhs: &PointAffine) -> Self {
        let mut P = self;
        P.set_add_affine(rhs);
        P
    }

    /// Specialized point addition and subtraction routine. This returns
    /// `self + rhs` and `self - rhs` more efficiently than doing both
    /// operations separately. Moreover, the two returned points have the
    /// same Z coordinate.
    fn add_sub(self, rhs: &Self) -> (Self, Self) {
        // Starting from set_add(), we also compute self - rhs.
        // If P1 = self, P2 = rhs, P3 = P1 + P2 and P3' = P1 - P2,
        // then S2 = S1 + T1.
        // Cost: 12M+2S

        let (X1, S1, Z1, T1) = (&self.X, &self.S, &self.Z, &self.T);
        let (X2, S2, Z2, T2) = (&rhs.X, &rhs.S, &rhs.Z, &rhs.T);

        let X1X2 = X1 * X2;
        let S1S2 = S1 * S2;
        // sub: S1S2' = S1S2 + S1*T2
        let Z1Z2 = Z1 * Z2;
        let T1T2 = T1 * T2;
        let D = (S1 + T1) * (S2 + T2);
        // sub: D' = (S1 + T1)*S2 = D + S1*T2 + T1T2
        let E = T1T2.mul_u1();         // a^2 = u^2 = u + 1
        let F = X1X2.square();
        let G = Z1Z2.square();
        let X3 = D + S1S2;
        // sub: X3' = D' + S1*(S2 + T2) = X3 + T1T2
        let S3 = (G * (S1S2 + E) + F * (D + E)).mul_sb();
        // sub: S3' = (G*(S1S2 + S1*T2 + E) + F*(D + S1*T2 + T1T2 + E))*sqrt(b)
        //          = S3 + sqrt(b)*((G + F)*S1*T2 + F*T1T2)
        let Z3 = (F + G).mul_sb();
        let T3 = X3 * Z3;
        let X4 = X3 + T1T2;
        let S4 = S3 + ((F + G) * S1 * T2 + F * T1T2).mul_sb();
        let Z4 = Z3;
        let T4 = X4 * Z4;
        (Self { X: X3, S: S3, Z: Z3, T: T3 },
         Self { X: X4, S: S4, Z: Z4, T: T4 })
    }

    /// Specialized point addition and subtraction routine when the other
    /// operand is in affine coordinates. This returns self + rhs and
    /// self - rhs more efficiently than doing both operations separately.
    /// Moreover, the two returned points have the same Z coordinate.
    #[allow(dead_code)]
    fn add_sub_affine(self, rhs: &PointAffine) -> (Self, Self) {
        // Starting from set_add_affine(), we also compute self - rhs, with:
        //   (-rhs).scaled_s = rhs.scaled_s + rhs.scaled_x.
        // cost: 11M+2S
        let (X1, S1, Z1, T1) = (&self.X, &self.S, &self.Z, &self.T);
        let (X2, S2) = (&rhs.scaled_x, &rhs.scaled_s);

        let X1X2 = X1 * X2;
        let S1S2 = S1 * S2;
        // sub: S1S2' = S1*S2 + S1*X2 = S1S2 + S1*X2
        let T1T2 = T1 * X2;
        let D = (S1 + T1) * (S2 + X2);
        // sub: D' = (S1 + T1)*S2 = D + S1*X2 + T1*X2 = D + S1*X2 + T1T2
        let E = T1T2.mul_u1();
        let F = X1X2.square();
        let G = Z1.square();
        let X3 = D + S1S2;
        // sub: X4 = D' + S1S2' = X3 + T1T2
        let S3 = (G * (S1S2 + E) + F * (D + E)).mul_sb();
        // sub: S4 = (G*(S1S2' + E) + F*(D' + E))*sqrt(b)
        //         = S3 + (G*S1*X2 + F*S1*X2 + F*T1T2)*sqrt(b)
        //         = S3 + X2*((F + G)*S1 + F*T1)*sqrt(b)
        let Z3 = (F + G).mul_sb();
        // sub: Z4 = Z3
        let T3 = X3 * Z3;
        // sub: T4 = X4*Z4 = X3 + T1T2*Z3
        let X4 = X3 + T1T2;
        let S4 = S3 + X2 * ((F + G) * S1 + F * T1).mul_sb();
        let Z4 = Z3;
        let T4 = X4 * Z4;
        (Self { X: X3, S: S3, Z: Z3, T: T3 },
         Self { X: X4, S: S4, Z: Z4, T: T4 })
    }

    /// Specialized point addition routine when both operands are in
    /// affine coordinates.
    fn add_affine_affine(P1: &PointAffine, P2: &PointAffine) -> Self {
        // This is similar to set_add() except that:
        //   Z1 = Z2 = 1
        //   T1 = X1
        //   T2 = X2
        // cost: 5M+1S
        let (X1, S1) = (&P1.scaled_x, &P1.scaled_s);
        let (X2, S2) = (&P2.scaled_x, &P2.scaled_s);

        let X1X2 = X1 * X2;
        let S1S2 = S1 * S2;
        let D = (S1 + X1) * (S2 + X2);
        let E = X1X2.mul_u1();         // a^2 = u^2 = u + 1
        let F = X1X2.square();
        let X3 = D + S1S2;
        let S3 = (S1S2 + E + F * (D + E)).mul_sb();
        let Z3 = F.mul_sb() + &Self::SB;
        let T3 = X3 * Z3;
        Self { X: X3, S: S3, Z: Z3, T: T3 }
    }

    /// Specialized point doubling routine when the operand is in
    /// affine coordinates.
    #[allow(dead_code)]
    fn double_affine(P: &PointAffine) -> Self {
        // X' = X^2
        // Z' = sqrt(b)*(X^2 + 1)^2
        // S' = Z'*(S + (a + 1)*X)^2 + sqrt(b)*X^2
        // T' = X'*Z'
        // cost: 2M+3S
        let (X, S) = (&P.scaled_x, &P.scaled_s);

        let Xp = X.square();
        let Zp = (Xp + GFb254::ONE).square().mul_sb();
        let Sp = Zp * (S + X.mul_u1()).square() + Xp.mul_sb();
        let Tp = Xp * Zp;
        Self { X: Xp, S: Sp, Z: Zp, T: Tp }
    }

    /// Specialized point tripling routine when the operand is in
    /// affine coordaintes.
    #[allow(dead_code)]
    fn triple_affine(P: &PointAffine) -> Self {
        // cost: 6M+4S
        let (X, S) = (&P.scaled_x, &P.scaled_s);
        let D = X.square();
        let E = D.square();
        let F = (E + GFb254::ONE).mul_sb().square();
        let G = D * X;
        let H = G.square();
        let J = F * X;
        let X3 = G + J;
        let Z3 = F + H;
        let S3 = Z3 * (S * (D + F) + J) + H * X3;
        let T3 = X3 * Z3;
        Self { X: X3, S: S3, Z: Z3, T: T3 }
    }

    /// Specialized point doubling and tripling routine when the operand is
    /// in affine coordaintes. This returns `2*P` and `3*P`; this is a bit
    /// cheaper than using `double_affine()` and `triple_affine()` separately.
    #[allow(dead_code)]
    fn double_and_triple_affine(P: &PointAffine) -> (Self, Self) {
        // cost: 7M+5S
        let (X, S) = (&P.scaled_x, &P.scaled_s);
        let D = X.square();
        let E = D.square();
        let Z2 = (E + GFb254::ONE).mul_sb();
        let F = Z2.square();
        let G = D * X;
        let H = G.square();
        let J = F * X;
        let X2 = D;
        let T2 = (D + H).mul_sb();
        let S2 = Z2 * (S + X.mul_u1()).square() + D.mul_sb();
        let X3 = G + J;
        let Z3 = F + H;
        let S3 = Z3 * (S * (D + F) + J) + H * X3;
        let T3 = X3 * Z3;
        (Self { X: X2, S: S2, Z: Z2, T: T2 },
         Self { X: X3, S: S3, Z: Z3, T: T3 })
    }

    /// Specialized point addition routine when adding an affine point P
    /// to zeta(P).
    /// Note: the Z coordinate of the output point is in GF(2^127).
    #[allow(dead_code)]
    fn add_affine_selfzeta(P: &PointAffine, neg: u32) -> Self {
        // This is similar to set_add_affine_affine() except that:
        //   X2 = phi(X1)
        //   S2 = phi(S1) + (u + 1)*phi(X1)
        // If neg != 0, we also negate the point afterwards.
        // For a value v = v0 + u*v1:
        //    phi(v) = v0 + v1 + u*v1
        //    v*phi(v) = v0^2 + v1^2 + v0*v1   \in GF(2^127)
        //    v + phi(v) = v0                  \in GF(2^127)
        //    v + u*phi(v) = v1 + u*v0
        // We decompose:
        //    X1 = x0 + u*x1
        //    S1 = s0 + u*s1
        // Then:
        //    X2 = (x0 + x1) + u*x1
        //    S2 = (s0 + s1 + x0) + u*(s1 + x0 + x1)
        // cost: 5M+1S - savings
        // (savings:
        //    X1*X2 is X1*phi(X1), which is slightly less expensive
        //    F = X1X2^2 is now computed over GF(2^127)
        //    (D + E)*F is now a multiplication by an element of GF(2^127)
        //    X3*Z3 is now a multiplication by an element of GF(2^127))
        let (x0, x1) = P.scaled_x.to_components();
        let (s0, s1) = P.scaled_s.to_components();
        let X2 = GFb254::from_b127(x0 + x1, x1);
        let mut S2 = GFb254::from_b127(s0 + s1 + x0, s1 + x0 + x1);
        S2.set_cond(&(S2 + X2), neg);

        let X1X2f = (x0 + x1).square() + x0 * x1;
        let S1S2 = P.scaled_s * S2;
        let D = GFb254::from_b127(s0 + x0, s1 + x1) * (S2 + X2);
        let E = GFb254::from_b127(X1X2f, X1X2f);
        let Ff = X1X2f.square();
        let X3 = D + S1S2;
        let S3 = (S1S2 + E + (D + E).mul_b127(&Ff)).mul_sb();
        let Z3f = Ff.mul_sb() + &Self::SB_GF127;
        let Z3 = GFb254::from_b127(Z3f, GFb127::ZERO);
        let T3 = X3.mul_b127(&Z3f);
        Self { X: X3, S: S3, Z: Z3, T: T3 }
    }

    /// Adds zeta(self) to self. If neg != 0, this uses -zeta(self) instead.
    /// Note: the Z coordinate of the output point is in GF(2^127).
    #[allow(dead_code)]
    fn set_add_selfzeta(&mut self, neg: u32) {
        let (X1, S1, Z1, T1) = (&self.X, &self.S, &self.Z, &self.T);
        let rhs = self.zeta(neg);
        let (S2, T2) = (&rhs.S, &rhs.T);

        // Based on set_add(), with the additional rules:
        //    X2 = phi(X1)                   = x0 + x1 + u*x1
        //    S2 = phi(S1) + (u + 1)*phi(T1) = s0 + s1 + t0 + u*(s1 + t0 + t1)
        //    Z2 = phi(Z1)                   = z0 + z1 + u*z1
        //    T2 = phi(T1)                   = t0 + t1 + u*t1
        // This simplifies some of the operations below.
        // cost: 8M+2S - savings:
        //    X1*X2, Z1*Z2 and T1*T2 become mul_selfphi() calls
        //    F = X1X2^2 and G = Z1Z2^2 are over GF(2^127)
        //    multiplications by F, G and Z3 are mixed
        let X1X2f = X1.mul_selfphi();
        let S1S2 = S1 * S2;
        let Z1Z2f = Z1.mul_selfphi();
        let T1T2f = T1.mul_selfphi();
        let D = (S1 + T1) * (S2 + T2);
        let E = GFb254::from_b127(T1T2f, T1T2f);
        let Ff = X1X2f.square();
        let Gf = Z1Z2f.square();
        let X3 = D + S1S2;
        let S3 = ((S1S2 + E).mul_b127(&Gf) + (D + E).mul_b127(&Ff)).mul_sb();
        let Z3f = (Ff + Gf).mul_sb();
        let Z3 = GFb254::from_b127(Z3f, GFb127::ZERO);
        let T3 = X3.mul_b127(&Z3f);
        self.X = X3;
        self.S = S3;
        self.Z = Z3;
        self.T = T3;
    }

    /// Specialized point addition and subtraction routine when both
    /// operands are in affine coordinates. This returns P1 + P2 and
    /// P1 - P2 more efficiently than doing both operations separately.
    /// Moreover, the two returned point have the same Z coordinate.
    #[allow(dead_code)]
    fn add_sub_affine_affine(P1: &PointAffine, P2: &PointAffine)
        -> (Self, Self)
    {
        // Starting from add_affine_affine(), we also compute P1 - P2 with:
        //   (-P2).scaled_s = P2.scaled_s + P2.scaled_x.
        // cost: 8M+1S
        let (X1, S1) = (&P1.scaled_x, &P1.scaled_s);
        let (X2, S2) = (&P2.scaled_x, &P2.scaled_s);

        let X1X2 = X1 * X2;
        let S1S2 = S1 * S2;
        let S1X2 = S1 * X2;
        let S2X1 = S2 * X1;
        // D = S1S2 + S1X2 + S2X1 + X1X2
        let E = X1X2.mul_u1();
        let F = X1X2.square();
        let X4 = S1X2 + S2X1;
        let X3 = X4 + X1X2;
        let S3 = (S1S2 + E + F * (S1S2 + X3 + E)).mul_sb();
        let S4 = (S1S2 + S1X2 + E + F * (S1S2 + S2X1 + E)).mul_sb();
        let Z3 = F.mul_sb() + &Self::SB;
        let Z4 = Z3;
        let T3 = X3 * Z3;
        let T4 = X4 * Z4;
        (Self { X: X3, S: S3, Z: Z3, T: T3 },
         Self { X: X4, S: S4, Z: Z4, T: T4 })
    }

    /// Specialized point addition routine when the other operand is in
    /// affine extended coordinates (used in the pregenerated tables for
    /// multiples of the base point).
    #[allow(dead_code)]
    #[inline(always)]
    fn set_sub_affine(&mut self, rhs: &PointAffine) {
        let mrhs = PointAffine {
            scaled_x: rhs.scaled_x,
            scaled_s: rhs.scaled_s + rhs.scaled_x,
        };
        self.set_add_affine(&mrhs);
    }

    /// Doubles this point (in place).
    pub fn set_double(&mut self) {
        // Note: this implementation uses the 3M+4S formulas. The
        // formulas used in set_xdouble() would yield a cost of 2M+6S,
        // which might be faster on architectures without a carryless
        // multiplication opcode, and where GF(2^254) multiplications
        // are typically 5x the cost of squarings.

        let (S, Z, T) = (&self.S, &self.Z, &self.T);

        // Formulas from eprint.iacr.org/2022/1325, figure 2.
        //   D = (S + a*T)*(S + a*T + T)
        //   E = (S + a*T + T + sqrt_b*Z**2)**2
        //   F = T**2
        //  Xp = sqrt_b*F
        //  Zp = D + a*F
        //  Sp = sqrt_b*(E*(E + Zp + F) + (D + sqrt_b*Xp)**2)
        //  Tp = Xp*Zp
        // cost: 3M+4S
        let H = S + T.mul_u();
        let J = H + T;
        let D = H * J;
        let E = (J + Z.square().mul_sb()).square();
        let F = T.square();
        let Xp = F.mul_sb();
        let Zp = D + F.mul_u();
        let Sp = (E * (E + Zp + F) + (D + Xp.mul_sb()).square()).mul_sb();
        let Tp = Xp * Zp;
        self.X = Xp;
        self.S = Sp;
        self.Z = Zp;
        self.T = Tp;
    }

    #[inline(always)]
    pub fn double(self) -> Self {
        let mut r = self;
        r.set_double();
        r
    }

    /// Doubles this point n times (in place).
    pub fn set_xdouble(&mut self, n: u32) {
        // Handle special cases.
        if n <= 1 {
            if n == 1 {
                self.set_double();
            }
            return;
        }

        // Method: conversion to (x,y) on the short Weierstraß curve,
        // doublings on that curve, then conversion back to our curve
        // and to the group by adding N, with switch back to (x,s)
        // coordinates. Each doubling on the short Weierstraß curve has
        // cost 2M+4S. Initial and final conversions cost 1S each.
        // Total cost: n*(2M+4S) + 2S

        // Convert the point to the short Weierstraß curve:
        //    y^2 + x*y = x^3 + a*x^2 + b^2
        // with the representation:
        //    Z != 0
        //    x = X/Z
        //    y = Y/Z^2
        //    T = X*Z
        // x is unchanged by the conversion, but we must account for
        // the scaling factor. We start with (X1:S1:Z1:T1); then:
        //    X = sqrt(b)*X1
        //    T = sqrt(b)*T1
        //    Z = Z1
        // We have y = s + x^2 + a*x, hence:
        //    Y = sqrt(b)*S1 + X^2 + a*T
        let mut X = self.X.mul_sb();
        let mut T = self.T.mul_sb();
        let mut Z = self.Z;
        let mut Y = self.S.mul_sb() + X.square() + T.mul_u();

        // Doubles over the short Weierstraß curve:
        //   Z' = T^2
        //   D' = (X + sqrt(b)*Z)^2
        //   X' = D'^2
        //   T' = X'*Z'
        //   Y' = (Y*(Y + D' + T) + (a + b)*Z')^2 + (a + 1)*T'
        //
        // Special cases:
        //
        //  - Input is N = (0, b):
        //        X = 0, T = 0, Z != 0, Y = b*Z^2
        //    Then:
        //        Z' = 0
        //        D' = b*Z^2    <-- equal to Y
        //        X' = b^2*Z^4  <-- not zero
        //        T' = 0
        //        Y' = 0
        //    Output:  X != 0, T = 0, Z = 0, Y = 0
        //
        //  - Input is the point-at-infinity:
        //        X != 0, T = 0, Z = 0, Y = 0
        //    Then:
        //        Z' = 0
        //        D' = X^2
        //        X' = X^4      <-- not zero
        //        T' = 0
        //        Y' = 0
        //    Output:  X != 0, T = 0, Z = 0, Y = 0
        //
        // Thus, the formulas are complete, provided that the
        // point-at-infinity is represented as (X:Y:Z:T) = (X:0:0:0)
        // for some X != 0.
        for _ in 0..n {
            let D = (X + Z.mul_sb()).square();
            Z = T.square();
            X = D.square();
            let E = Y + T + D;
            T = X * Z;
            Y = (Y * E + Z.mul_u() + Z.mul_b()).square() + T.mul_u1();
        }

        // Convert back:
        //   curve change: y <- y + b
        //   add N to move to the group: (x, y) <- (b/x, b*(y + x)/x^2)
        //   get the s coordinate: s = y + x^2 + a*x + b
        //   apply scaling to our normal (X:S:Z:T) representation
        //
        //   X3 = Z*sqrt(b)
        //   S3 = (Y + (a + 1)*T + X^2)*sqrt(b)
        //   Z3 = X
        //   T3 = T*sqrt(b)
        //
        // Note: since the point is the output of at least one doublings on
        // E[r] (on the short Weierstraß curve), it cannot be N, hence X != 0.
        // If the point is the point-at-infinity, then Y = Z = T = 0 but
        // X != 0, and we get (X3:S3:Z3:T3) = (0:sqrt(b)*X^2:X:0), which is
        // a correct representation of the group neutral.
        self.X = Z.mul_sb();
        self.T = T.mul_sb();
        self.Z = X;
        self.S = (Y + X.square() + T.mul_u1()).mul_sb();
    }

    #[inline(always)]
    pub fn xdouble(self, n: u32) -> Self {
        let mut r = self;
        r.set_xdouble(n);
        r
    }

    /// Doubles this point n times (in place) then add an affine point.
    #[cfg(feature = "gls254bench")]
    #[inline(always)]
    fn set_xdouble_add_affine(&mut self, n: u32, rhs: &PointAffine) {
        // Handle special cases.
        if n == 0 {
            self.set_add_affine(rhs);
            return;
        }

        // Convert the point to the short Weierstraß curve:
        //    y^2 + x*y = x^3 + a*x^2 + b^2
        // with the representation:
        //    Z != 0
        //    x = X/Z
        //    y = Y/Z^2
        //    T = X*Z
        // x is unchanged by the conversion, but we must account for
        // the scaling factor. We start with (X1:S1:Z1:T1); then:
        //    X = sqrt(b)*X1
        //    T = sqrt(b)*T1
        //    Z = Z1
        // We have y = s + x^2 + a*x, hence:
        //    Y = sqrt(b)*S1 + X^2 + a*T
        let mut X = self.X.mul_sb();
        let mut T = self.T.mul_sb();
        let mut Z = self.Z;
        let mut Y = self.S.mul_sb() + X.square() + T.mul_u();

        // Doubles over the short Weierstraß curve:
        //   Z' = T^2
        //   D' = (X + sqrt(b)*Z)^2
        //   X' = D'^2
        //   T' = X'*Z'
        //   Y' = (Y*(Y + D' + T) + (a + b)*Z')^2 + (a + 1)*T'
        for _ in 0..n {
            let D = (X + Z.mul_sb()).square();
            Z = T.square();
            X = D.square();
            let E = Y + T + D;
            T = X * Z;
            Y = (Y * E + Z.mul_u() + Z.mul_b()).square() + T.mul_u1();
        }

        // Convert back:
        //   curve change: y <- y + b
        //   add N to move to the group: (x, y) <- (b/x, b*(y + x)/x^2)
        //   get the s coordinate: s = y + x^2 + a*x + b
        //   apply scaling to our normal (X:S:Z:T) representation
        let X1 = Z.mul_sb();
        let T1 = T.mul_sb();
        let Z1 = X;
        let S1 = (Y + T + X.square() + T.mul_u()).mul_sb();

        let (X2, S2) = (&rhs.scaled_x, &rhs.scaled_s);

        let X1X2 = X1 * X2;
        let S1S2 = S1 * S2;
        let T1T2 = T1 * X2;
        let D = (S1 + T1) * (S2 + X2);
        let E = T1T2.mul_u1();         // a^2 = u^2 = u + 1
        let F = X1X2.square();
        let G = Z1.square();
        let X3 = D + S1S2;
        let S3 = (G * (S1S2 + E) + F * (D + E)).mul_sb();
        let Z3 = (F + G).mul_sb();
        let T3 = X3 * Z3;
        self.X = X3;
        self.S = S3;
        self.Z = Z3;
        self.T = T3;
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
        // eprint.iacr.org/2022/1325, section 5.5
        (self.S * rhs.T).equals(rhs.S * self.T)
    }

    /// Tests whether this point is the neutral (identity point on the
    /// curve).
    ///
    /// Returned value is 0xFFFFFFFF for the neutral, 0x00000000
    /// otherwise.
    #[inline(always)]
    pub fn isneutral(self) -> u32 {
        self.X.iszero()
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
        self.X.set_cond(&P.X, ctl);
        self.S.set_cond(&P.S, ctl);
        self.Z.set_cond(&P.Z, ctl);
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
        self.S.set_cond(&(self.S + self.T), ctl);
    }

    /// Maps a field element into a point.
    ///
    /// This map output is not uniformly distributed; in general, it should
    /// be used only through `hash_to_curve()`, which invokes this map
    /// twice.
    fn map_to_curve(c: &GFb254) -> Self {
        // Ensure that Tr(c) = 1 and Tr(c/z) = 0.
        //
        // If we write c = c0 + c1*u, Tr(c) is the least significant bit
        // of c1; moreover, c/z implies a simple right shift by 1 bit
        // for c1 (the former bit 0 "wraps around" by becoming z^62 +
        // z^126) so Tr(c/z) is simply the value of the second bit of
        // c1. Thus, we just need to make sure that the two least
        // significant bits of c1 are 1 (bit 0) and 0 (bit 1).
        // We also remember the original trace of c.
        let (c0, mut c1) = c.to_components();
        let orig_trace = c1.get_bit(0);
        c1.set_bit(0, 1);
        c1.set_bit(1, 0);
        let c = GFb254::from_b127(c0, c1);

        // Note that:
        //   Tr(c) = 1
        //   Tr(c + z^2) = Tr(c) + Tr(z^2) = 1
        //   Tr(c + c^2/z^2) = Tr(c) + Tr(c^2/z^2) = Tr(c) + Tr(c/z) = 1
        // Set:
        //   m_1 = c
        //   m_2 = c + z^2
        //   m_3 = c + c^2/z^2
        // All three values m_i have trace 1.
        // Define:
        //   e_1 = b/m_1
        //   e_2 = b/m_2
        //   e_3 = b/m_3
        // (since the m_i all have trace 1, none of them is zero)
        // We have:
        //   e_1 + e_2 + e_3 = b/c + b/(c + z^2) + b*z^2/(c*z^2 + c^2)
        //                   = 0
        // Therefore, e_1, e_2 and e_3 cannot be all of trace 1. We set
        // (m, e) = (m_i, e_i) for the lowest i such that Tr(e_i) = 0. We
        // can define d = sqrt(m), then compute w such that:
        //   w^2 + w = d + a
        // Since Tr(d + a) = Tr(m) + Tr(a) = 0, we know that the equation
        // has two solutions (w and w+1); we select the solution w = w0 + w1*u
        // which is such that the least significant bit of w0 matches the
        // original trace of c (orig_trace).
        //
        // At that point, we have a value w such that:
        //   d = w^2 + w + a
        //   e = b/d^2
        //   Tr(e) = 0
        // These are the exact conditions for a successful and unambiguous
        // decoding of w into a point; it has a unique solution in the group:
        //   f = Qsolve(e)
        //   x = d*f
        //   if Tr(x) = 1, then: x <- x+d
        //   s = x*w^2
        //
        // Main cost are one inversion (mutualized inversion for m_1,
        // m_2 and m_3) and two Qsolve (for w and for f), which is
        // slightly above the cost of point decoding (one inversion and
        // one Qsolve; inversion is more expensive than Qsolve). The
        // extra operations are a few squarings and multiplications,
        // which are much cheaper. Division by z^2 is inexpensive.

        // m_1 = c
        // m_2 = c + z^2
        // m_3 = c + c^2/z^2
        let m_1 = c;
        let m_2 = c + GFb254::w64le(4, 0, 0, 0);
        let m_3 = c + c.square().div_z2();

        // e_i = b/m_i for i \in {1, 2, 3}
        // We mutualize the division. All m_i are non-zero since they all
        // have trace 1, so there is no special case in this division.
        let mm = m_1 * m_2;
        let ii = Self::B / (mm * m_3);
        let e_3 = ii * mm;
        let jj = ii * m_3;
        let e_2 = jj * m_1;
        let e_1 = jj * m_2;

        // (m, e) = (m_i, e_i) for the smallest i such that Tr(e_i) = 0.
        // Since e_1 + e_2 + e_3 = 0, at least one of them matches the
        // condition.
        let t_1 = e_1.trace().wrapping_sub(1);
        let t_2 = e_2.trace().wrapping_sub(1);
        let e = GFb254::select(&GFb254::select(&e_3, &e_2, t_2), &e_1, t_1);
        let m = GFb254::select(&GFb254::select(&m_3, &m_2, t_2), &m_1, t_1);

        // d = sqrt(m)
        // w such that w^2 + w + a = d, and the lsb of w0 matches orig_trace.
        let d = m.sqrt();
        let w = d.qsolve();
        let (mut w0, w1) = w.to_components();
        w0.set_bit(0, orig_trace);
        let w = GFb254::from_b127(w0, w1);

        // We now have:
        //   w^2 + w + a = d
        //   e = b/d^2
        //   Tr(e) = 1
        // We can finish the computation in the same way as normal point
        // decoding.
        let f = e.qsolve();
        let mut x = d * f;
        x.set_cond(&(x + d), x.trace().wrapping_neg());
        let s = x * w.square();

        Self {
            X: x,
            S: s.mul_sb(),
            Z: Self::SB,
            T: x.mul_sb(),
        }
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

        // Decode 32 bytes into a field element, ignoring the extra bits.
        fn decode_trunc(buf: &[u8]) -> GFb254 {
            let mut tmp = [0u8; 32];
            tmp[..].copy_from_slice(buf);
            tmp[15] &= 0x7F;
            tmp[31] &= 0x7F;
            let (x, _) = GFb254::decode_ct(&tmp);
            x
        }

        let c1 = decode_trunc(&blob1);
        let c2 = decode_trunc(&blob2);
        Self::map_to_curve(&c1) + Self::map_to_curve(&c2)
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

    /// Recodes a 64-bit integer into 13 signed digits.
    ///
    /// Each digit is in -15..+16, top digit is in 0..+16.
    fn recode5_u64(n: u64) -> [i8; 13] {
        let mut sd = [0i8; 13];
        let mut x = n;
        let mut cc: u32 = 0;       // carry from lower digits
        for j in 0..13 {
            let d = ((x as u32) & 0x1F) + cc;
            x >>= 5;
            let m = 16u32.wrapping_sub(d) >> 8;
            sd[j] = (d.wrapping_sub(m & 32)) as i8;
            cc = m & 1;
        }
        sd
    }

    /// Recodes a half-width scalar into 26 signed digits.
    ///
    /// Each digit is in -15..+16, top digit is in 0..+8.
    fn recode5_u128(n: u128) -> [i8; 26] {
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

    /// Recodes a half-width scalar into 32 signed digits.
    /// WARNING: this assumes that the scalar fits on 127 bits.
    ///
    /// Each digit is in -7..+8, top digit is in 0..+8.
    fn recode4_u128(n: u128) -> [i8; 32] {
        let mut sd = [0i8; 32];
        let mut x = n;
        let mut cc: u32 = 0;       // carry from lower digits
        for j in 0..32 {
            let d = ((x as u32) & 0x0F) + cc;
            x >>= 4;
            let m = 8u32.wrapping_sub(d) >> 8;
            sd[j] = (d.wrapping_sub(m & 16)) as i8;
            cc = m & 1;
        }
        sd
    }

    /// Recodes a half-width scalar into 43 signed digits.
    ///
    /// Each digit is in -3..+4, top digit is in 0..+4.
    #[allow(dead_code)]
    fn recode3_u128(n: u128) -> [i8; 43] {
        let mut sd = [0i8; 43];
        let mut x = n;
        let mut cc: u32 = 0;       // carry from lower digits
        for j in 0..43 {
            let d = ((x as u32) & 0x07) + cc;
            x >>= 3;
            let m = 4u32.wrapping_sub(d) >> 8;
            sd[j] = (d.wrapping_sub(m & 8)) as i8;
            cc = m & 1;
        }
        sd
    }

    // Let:
    //    s = 85070591730234615854573802599387326102
    //    t = 85070591730234615877113501116496779625
    // We have s**2 + t**2 = r
    // Let mu = s/t mod r
    // This is a square root of -1 modulo r. The endomorphism zeta()
    // efficiently computes the multiplication by mu on the group.
    //
    // split_mu(k), for a scalar k, finds k0 and k1 such that
    // k = k0 + k1*mu, with |k0| and |k1| of minimal size. Computation
    // is the following:
    //    we consider k as an integer in the 0 to r-1 range
    //    c = round(k*t/r)
    //    d = round(k*s/r)
    //    k0 = k - d*s - c*t
    //    k1 = d*t - c*s
    // It can be shown that k0^2 + k1^2 <= s^2 + t^2 = r. Thus, both k0
    // and k1 are lower than sqrt(r) =~ 2^126.5 in absolute value. We
    // can compute k0 and k1 as signed 128-bit integers, without minding
    // about the upper bits.
    //
    // To compute the rounded divisions, we can use the following; this
    // leverages the fact that r is close to 2^253. Indeed, we can write:
    //    r = 2^253 + r0
    // and the value of r0 is a bit lower than 2^126. We can thus use
    // 2^(-253) as an approximation of 1/r. More precisely, for an input
    // integer 0 <= x <= (r - 1)*m for an integer m <= 2^127:
    //
    //   Let z = x + (r - 1)/2
    //   We have: round(x/r) = floor(z/r)
    //   Split z:
    //      z = z0 + z1*2^253
    //        = z0 - z1*r0 + z1*r
    //   We have:
    //      0 <= z1*r0 <= r0*floor((r - 1)*(m + 1/2)/2^253)
    //   That upper bound is approximately 2^252.98; it is lower than r.
    //   Thus:
    //      0 <= z0 < 2^253 < r
    //      0 <= z1*r0 < 2^253 < r
    //   Therefore:
    //      floor(z/r) = z1      if z1*r0 <= z0
    //      floor(z/r) = z1 - 1  if z1*r0 > z0
    //
    // The function mul_divr_rounded() computes round(k*e/r) on inputs k
    // and e with the above method. It outputs a nonnegative integer which
    // is necessarily not greater than e. The function is used by split_mu()
    // with parameters (k, t) and (k, s).
    //
    // All computations on such integers are done over representations with
    // 32-bit limbs, because we do not trust 23-bit architectures to implement
    // 64-bit code (and 64x64->128 multiplications) in a constant-time way.
    //
    // This is essentially the same code as in jq255e.rs, except that
    // mul_divr_rounded() is slightly different because here r is slightly
    // above a power of 2, instead of slightly below.

    /// Given integers k and e, with k < r and e < 2^127 - 2, returns
    /// round(k*e/r).
    fn mul_divr_rounded(k: &Zu256, e: &Zu128) -> Zu128 {
        // z <- k*e
        let mut z = k.mul256x128(e);

        // (r-1)/2 (padded to 384 bits)
        const HR: Zu384 = Zu384::w64le(
            0x9E5EF1BE7A1D467A, 0x1F8D23EF6E0D0ED6,
            0x0000000000000000, 0x1000000000000000,
            0x0000000000000000, 0x0000000000000000);

        // r0 = r - 2^253
        const R0: Zu128 = Zu128::w64le(0x3CBDE37CF43A8CF5, 0x3F1A47DEDC1A1DAD);

        // z <- z + (r-1)/2
        z.set_add(&HR);

        // Split z = z0 + z1*2^253
        let (z0, mut z1) = z.trunc_and_rsh_cc(0, 253);

        // t <- z1*r0
        let t = z1.mul128x128(&R0);

        // Subtract z1*r0 (in t) from z0; we are only interested in the
        // resulting borrow. The borrow is 0 or 1, and must be subtracted
        // from z1.
        z1.set_sub_u32(z0.borrow(&t));

        z1
    }

    /// Splits a scalar k into k0 and k1 (signed) such that k = k0 + k1*mu
    /// (for mu a specific square root of -1 modulo r that matches the
    /// GLS curve endomorphism).
    ///
    /// This function returns |k0|, sgn(k0), |k1| and sgn(k1), with
    /// sgn(x) = 0xFFFFFFFF if x < 0, 0x00000000 for x >= 0. It is
    /// guaranteed that |k0| and |k1| are lower than about 2^126.5.
    pub fn split_mu(k: &Scalar) -> (u128, u32, u128, u32) {
        let (k0, k1) = Self::split_mu_inner(k);
        let (n0, s0) = k0.abs();
        let (n1, s1) = k1.abs();
        (n0, s0, n1, s1)
    }

    /// Splits a scalar k into _odd_ k0 and k1 (signed) such that
    /// k = k0 + k1*mu (for mu a specific square root of -1 modulo r
    /// that matches the GLS curve endomorphism). This is a variant
    /// of `split_mu()`, except that it guarantees that the two
    /// obtained integers are odd. On the other hand, the integers
    /// may be up to one bit larger than the output of `split_mu()`.
    ///
    /// This function returns |k0|, sgn(k0), |k1| and sgn(k1), with
    /// sgn(x) = 0xFFFFFFFF if x < 0, 0x00000000 for x >= 0. It is
    /// guaranteed that |k0| and |k1| are lower than about 2^127.5.
    pub fn split_mu_odd(k: &Scalar) -> (u128, u32, u128, u32) {
        // We split m = (k - (mu + 1))/2. Then:
        //   c0 + c1*mu = m
        //   (2*c0 + 1) + (2*c1 + 1)*mu = 2*m + 1 + u = k
        // We thus return 2*c0 + 1 and 2*c1 + 1.
        let m = (k - Scalar::MU_PLUS_ONE).half();
        let (c0, c1) = Self::split_mu_inner(&m);
        let (n0, s0) = c0.double_inc_abs();
        let (n1, s1) = c1.double_inc_abs();
        (n0, s0, n1, s1)
    }

    // Inner function for split_mu(); it returns the two integers in
    // signed little-endian representation, with 32-bit limbs.
    #[inline(always)]
    fn split_mu_inner(k: &Scalar) -> (Zu128, Zu128) {
        // Obtain k as an integer ki in the 0..r-1 range.
        let ki = Zu256::decode(&k.encode()).unwrap();

        // Constants s and t such that mu = s/t mod r.
        const ES: Zu128 = Zu128::w64le(0x639973CF3FA56696, 0x3FFFFFFFFFFFFFFF);
        const ET: Zu128 = Zu128::w64le(0x9C668C30C05A9969, 0x4000000000000000);

        // c <- round(ki*t/r)
        // d <- round(ki*s/r)
        let c = Self::mul_divr_rounded(&ki, &ET);
        let d = Self::mul_divr_rounded(&ki, &ES);

        // k0 = ki - d*s - c*t
        // k1 = d*t - c*s
        let mut k0 = ki.trunc128();
        k0.set_sub(&d.mul128x128trunc(&ES));
        k0.set_sub(&c.mul128x128trunc(&ET));
        let mut k1 = d.mul128x128trunc(&ET);
        k1.set_sub(&c.mul128x128trunc(&ES));

        (k0, k1)
    }

    /// Applies the GLS endomorphism on this point. This (efficiently)
    /// computes the product of this point by the scalar `mu` which
    /// is a square root of -1 modulo `r`.
    /// Parameter `neg` must be either 0x00000000 or 0xFFFFFFFF; if
    /// `neg` is not zero, then the returned point is negated.
    #[inline(always)]
    pub fn set_zeta(&mut self, neg: u32) {
        // On the original curve, the endomorphism psi() is:
        //   psi(x, y) = (phi(x), phi(y) + phi(x)*(phi(alpha) + alpha))
        // with phi(t) = t^(2^127) (the Frobenius automorphism for
        // GF(2^127)) and alpha an element of GF(2^508) such that
        // alpha^2 + alpha = a + 1.
        //
        // In GLS254, we have a = u, and phi(alpha) + alpha = u. Hence:
        //    psi(x, y) = (phi(x), phi(y) + u*phi(x))
        //
        // Since psi(N) = N, we can define our endomorphism zeta() as
        // being the same as psi(), except that it works in (x,s)
        // coordinates with our modified curve equation. The change of
        // variable was:
        //    (x, y) -> (x^4, y^4 + B^2)
        // We have:
        //    phi(x^4) = phi(x)^4
        //    phi(y^4 + B^2) + u*phi(x^4) = (phi(y) + u*phi(x))^4 + B^2
        // since B^2 = b \in GF(2^127). Thus, the endomorphism formulas
        // are not affected by the change of variable.
        //
        // The s coordinate is:
        //    s = y + x^2 + a*x + b
        // Thus, if zeta(x,s) = (x',s'), we have:
        //    x' = phi(x)
        //    s' = phi(s + x^2 + a*x + b) + u*phi(x) + phi(x)^2 + a*phi(x) + b
        // a = u, phi(a) = u + 1, and phi(b) = b. We thus get:
        //    x' = phi(x)
        //    s' = phi(s) + u*phi(x) + phi(x)
        //
        // In affine (x,s) coordinates, with x = x0 + u*x1 and s = s0 + u*s1,
        // we get:
        //    x' = (x0 + x1) + x1*u
        //    s' = (s0 + s1 + x0) + (s1 + x0 + x1)*u
        // Note that these formulas are preserved if x and s are scaled by
        // a non-zero value in GF(2^127). This is convenient because we
        // keep x and s with such a scaling (by 1/sqrt(b)) in our affine
        // representation.
        //
        // Here, we have extended coordinates (X,S,Z,T). phi() is a field
        // automorphism, hence:
        //    x' = phi(x) = phi(sqrt(b)*X/Z)
        //       = sqrt(b)*phi(X)/phi(Z)
        //    s' = phi(s) + (u + 1)*phi(x)
        //       = sqrt(b)*phi(S)/phi(Z)^2 + (u + 1)*sqrt(b)*phi(T)/phi(Z)^2
        // We can thus define:
        //    X' = phi(X)                  = x0 + x1 + u*x1
        //    S' = phi(S) + (u + 1)*phi(T) = s0 + s1 + t0 + u*(s1 + t0 + t1)
        //    Z' = phi(Z)                  = z0 + z1 + u*z1
        //    T' = phi(T)                  = t0 + t1 + u*t1
        let (x0, x1) = self.X.to_components();
        let (s0, s1) = self.S.to_components();
        let (z0, z1) = self.Z.to_components();
        let (t0, t1) = self.T.to_components();
        self.X = GFb254::from_b127(x0 + x1, x1);
        self.S = GFb254::from_b127(s0 + s1 + t0, s1 + t0 + t1);
        self.Z = GFb254::from_b127(z0 + z1, z1);
        self.T = GFb254::from_b127(t0 + t1, t1);

        // Also apply the conditional negation.
        self.set_condneg(neg);
    }

    /// Applies the GLS endomorphism on this point. This (efficiently)
    /// computes the product of this point by the scalar `mu` which
    /// is a square root of -1 modulo `r`.
    /// Parameter `neg` must be either 0x00000000 or 0xFFFFFFFF; if
    /// `neg` is not zero, then the returned point is negated.
    #[inline(always)]
    pub fn zeta(self, neg: u32) -> Self {
        let mut P = self;
        P.set_zeta(neg);
        P
    }

    #[inline(always)]
    fn lookup16_affine(win: &[GFb254; 32], k: i8) -> PointAffine {
        PointAffine::lookup16(win, k)
    }

    #[inline(always)]
    fn lookup16_affine_zeta(win: &[GFb254; 32], k: i8, neg: u32) -> PointAffine {
        let mut P = PointAffine::lookup16(win, k);
        P.set_zeta(neg);
        P
    }

    #[inline(always)]
    fn lookup8_affine(win: &[GFb254; 16], k: i8) -> PointAffine {
        PointAffine::lookup8(win, k)
    }

    #[inline(always)]
    fn lookup8_affine_zeta(win: &[GFb254; 16], k: i8, neg: u32) -> PointAffine {
        let mut P = PointAffine::lookup8(win, k);
        P.set_zeta(neg);
        P
    }

    #[allow(dead_code)]
    #[inline(always)]
    fn lookup4_affine(win: &[GFb254; 8], k: i8) -> PointAffine {
        PointAffine::lookup4(win, k)
    }

    #[allow(dead_code)]
    #[inline(always)]
    fn lookup4_affine_zeta(win: &[GFb254; 8], k: i8, neg: u32) -> PointAffine {
        let mut P = PointAffine::lookup4(win, k);
        P.set_zeta(neg);
        P
    }

    #[inline(always)]
    fn lookup16_affine_vartime(win: &[GFb254; 32], k: i8, neg: u32)
        -> PointAffine
    {
        let mut P = PointAffine::lookup16_vartime(win, k);
        P.set_condneg(neg);
        P
    }

    #[inline(always)]
    fn lookup16_affine_zeta_vartime(win: &[GFb254; 32], k: i8, neg: u32)
        -> PointAffine
    {
        let mut P = PointAffine::lookup16_vartime(win, k);
        P.set_zeta(neg);
        P
    }

    #[inline(always)]
    fn lookup16_vartime(win_ex: &[Self; 16], k: i8) -> Self {
        // Get sign and absolute value of index, and actual array index.
        let sk = ((k as i32) >> 31) as u32;
        let uk = ((k as u32) ^ sk).wrapping_sub(sk);
        let uj = uk.wrapping_sub(1);

        // Get point.
        let mut P = win_ex[(uj & 0x0F) as usize];

        // Adjust sign; set to neutral if 0.
        P.set_cond(&Self::NEUTRAL, ((uj as i32) >> 31) as u32);
        P.set_condneg(sk);
        P
    }

    #[inline(always)]
    fn lookup16_zeta_vartime(win_ex: &[Self; 16], k: i8, neg: u32) -> Self {
        let mut P = Self::lookup16_vartime(win_ex, k);
        P.set_zeta(neg);
        P
    }

    // Recodes an odd integer n into signed digits; the digits are all
    // non-zero and odd, with values in {-7, -5, -3, -1, +1, +3, +5, +7}.
    // This is the Joye-Tunstall recoding (from M. Joye and M. Tunstall,
    // "Exponent recoding and regular exponentiation algorithms",
    // AFRICACRYPT 2009).
    #[cfg(feature = "gls254bench")]
    fn recode3_u128_odd(n: u128) -> [i8; 43] {
        let mut sd = [0i8; 43];
        let mut cc = 0;
        for i in (0..43).rev() {
            let x = (((n >> (3 * i)) as u32) & 0x07).wrapping_sub(cc);
            cc = (x & 1).wrapping_sub(1) & 0x08;
            sd[i] = (x | 1) as i8;
        }
        sd
    }

    // Recodes an odd integer n into signed digits; the digits are all
    // non-zero and odd, with values in {-3, -1, +1, +3}. This is the
    // Joye-Tunstall recoding (from M. Joye and M. Tunstall, "Exponent
    // recoding and regular exponentiation algorithms", AFRICACRYPT
    // 2009).
    #[cfg(feature = "gls254bench")]
    fn recode2_u128_odd(n: u128) -> [i8; 64] {
        let mut sd = [0i8; 64];
        let mut cc = 0;
        for i in (0..64).rev() {
            let x = (((n >> (2 * i)) as u32) & 0x03).wrapping_sub(cc);
            cc = (x & 1).wrapping_sub(1) & 0x04;
            sd[i] = (x | 1) as i8;
        }
        sd
    }

    /// Multiplies this point by a scalar (in place).
    ///
    /// This operation is constant-time with regard to both the points
    /// and the scalar value.
    pub fn set_mul(&mut self, n: &Scalar) {
        // This uses the GLS endomorphism along with a "normal" lookup
        // table, with two lookups and two additions after every sequence
        // of doubling. 4-bit Booth recoding is used.

        // Split the scalar with the endomorphism.
        let (n0, s0, n1, s1) = Self::split_mu(n);

        // Compute the 4-bit window, first in extended coordinates,
        // then normalized to affine.
        let mut win_ex = [Point::NEUTRAL; 8];
        win_ex[0] = *self;
        win_ex[0].set_condneg(s0);
        win_ex[1] = win_ex[0].double();
        win_ex[2] = win_ex[1] + win_ex[0];
        win_ex[3] = win_ex[1].double();
        win_ex[5] = win_ex[2].double();
        (win_ex[6], win_ex[4]) = win_ex[5].add_sub(&win_ex[0]);
        win_ex[7] = win_ex[3].double();

        // Batch inversion for normalization. Since Z != 0 for all points
        // (including the neutral), this always works.
        // Note: win_ex[4] and win_ex[6] have the same Z coordinate (since
        // they came from add_sub()), so we can skip a few operations.
        let mut win = [GFb254::ZERO; 16];
        for i in 0..8 {
            if i != 6 {
                win_ex[i].Z.set_square();
            }
        }
        let mut zz = win_ex[0].Z;
        for i in 1..8 {
            if i != 6 {
                win[2 * i] = zz;
                zz *= win_ex[i].Z;
            }
        }
        zz.set_invert();
        for i in (1..8).rev() {
            match i {
                6 => (),
                4 => {
                    let iZ = zz * win[2 * i];
                    zz *= win_ex[i].Z;
                    win[(2 * i) + 0] = win_ex[i].T * iZ;
                    win[(2 * i) + 1] = win_ex[i].S * iZ;
                    win[(2 * i) + 4] = win_ex[i + 2].T * iZ;
                    win[(2 * i) + 5] = win_ex[i + 2].S * iZ;
                },
                _ => {
                    let iZ = zz * win[2 * i];
                    zz *= win_ex[i].Z;
                    win[(2 * i) + 0] = win_ex[i].T * iZ;
                    win[(2 * i) + 1] = win_ex[i].S * iZ;
                },
            }
        }
        win[0] = win_ex[0].T * zz;
        win[1] = win_ex[0].S * zz;

        // If n0 and n1 have distinct signs, then we need to apply
        // the -zeta endomorphism instead of zeta.
        let zn = s0 ^ s1;

        // Recode the two half-width scalars into 32 digits each.
        let sd0 = Self::recode4_u128(n0);
        let sd1 = Self::recode4_u128(n1);

        // Process the two digit sequences in high-to-low order.
        let P = Self::lookup8_affine(&win, sd0[31]);
        let Q = Self::lookup8_affine_zeta(&win, sd1[31], zn);
        *self = Self::add_affine_affine(&P, &Q);
        for i in (0..31).rev() {
            self.set_xdouble(4);
            let P = Self::lookup8_affine(&win, sd0[i]);
            let Q = Self::lookup8_affine_zeta(&win, sd1[i], zn);
            self.set_add(&Self::add_affine_affine(&P, &Q));
        }
    }

    /* unused -- kept only for reference.

    /// Multiplication of a point by a scalar using the ladder formulas.
    /// This is slower than `set_mul()`, mostly because it does not
    /// leverage the endomorphism.
    pub fn set_mul_ladder(&mut self, n: &Scalar) {
        // The ladder is a bit-by-bit process, with cost 5M+4S per scalar
        // bit. To leverage the endomorphism, one would have to switch
        // to a double ladder that mutualizes the doublings; this is
        // conceptually feasible, see:
        //    https://cr.yp.to/ecdh/diffchain-20060219.pdf
        // This would lower the cost to 2*(4M+1S)+(1M+3S) = 9M+5S per 2
        // bits, hence 45M+25S per 10 bits (compared to 50M+40S for the
        // code below). A contrario, the set_mul() function uses a
        // 5-doubling call (5*(2M+4S)+2S), an affine-affine addition
        // (5M+1S), and a normal addition (8M+2S), for 10 scalar bits,
        // i.e. 23M+25S, which should be substantially faster, even
        // taking into account the extra costs for building the table
        // and normalizing it, and the lookups.
        //
        // These counts of multiplications and squarings are only a crude
        // approximation. Actual measurements on an x86 "Coffee Lake" core:
        //   set_mul()         30900 cycles
        //   set_mul_ladder()  52800 cycles
        // It seems indeed improbable that the shift to a double-ladder
        // would gain enough to be competitive here. On systems without a
        // carryless multiplications, where multiplications are a lot more
        // expensive than squarings, the gain would be indeed minimal at
        // best.

        // Encode the scalar to access its bits.
        let nn = n.encode();

        // Normalize the point to affine.
        // Note: x0 and s0 are unscaled here (contrary to PointAffine).
        let iZ = self.Z.invert();
        let x0 = self.X.mul_sb() * iZ;
        let s0 = self.S.mul_sb() * iZ.square();

        // Ladder.
        //  (X1, Z1) = (0, 1)
        //  (X2, Z2) = (x0, 1)
        //  for i in reversed(range(0, n.bit_length()):
        //      # swap P1 and P2 if the scalar bit is 1
        //      bit = (n >> i) & 1
        //      if bit != 0:
        //          (X1, Z1, X2, Z2) = (X2, Z2, X1, Z1)
        //      X1X2 = X1*X2
        //      Z1Z2 = Z1*Z2
        //      D = (X1X2 + Z1Z2)**2
        //      X3 = X1X2*Z1Z2 + x0*D
        //      Z3 = sqrt_b*D
        //      X4 = (X1*Z1)**2
        //      Z4 = sqrt_b*(X1 + Z1)**4
        //      if bit != 0:
        //          (X3, Z3, X4, Z4) = (X4, Z4, X3, Z3)
        //      (X1, Z1, X2, Z2) = (X3, Z3, X4, Z4)
        let (mut X1, mut Z1) = (GFb254::ZERO, GFb254::ONE);
        let (mut X2, mut Z2) = (x0, GFb254::ONE);
        for i in (0..256).rev() {
            // Next scalar bit (as a 32-bit mask)
            let bit = (((nn[i >> 3] >> (i & 7)) as u32) & 1).wrapping_neg();

            // Swap (X1, Z1) <-> (X2, Z2) if bit != 0
            GFb254::cswap(&mut X1, &mut X2, bit);
            GFb254::cswap(&mut Z1, &mut Z2, bit);

            // Sum and double.
            let X1X2 = X1 * X2;
            let Z1Z2 = Z1 * Z2;
            let D = (X1X2 + Z1Z2).square();
            let X3 = X1X2 * Z1Z2 + x0 * D;
            let Z3 = D.mul_sb();
            let X4 = (X1 * Z1).square();
            let Z4 = (X1 + Z1).xsquare(2).mul_sb();
            X1 = GFb254::select(&X4, &X3, bit);
            Z1 = GFb254::select(&Z4, &Z3, bit);
            X2 = GFb254::select(&X3, &X4, bit);
            Z2 = GFb254::select(&Z3, &Z4, bit);
        }

        // Rebuild the full result coordinates.
        //  Z1Z2 = Z1*Z2
        //  D = x0*X1
        //  Xp = D*Z2
        //  Sp = x0*Z2*(X1*Z1Z2*(x0 + s0) + X2*(D + sqrt_b*Z1)**2)
        //  Zp = x0*Z1Z2
        //  Tp = Xp*Zp
        let Z1Z2 = Z1 * Z2;
        let D = x0 * X1;
        self.X = D * Z2;
        self.S = x0 * Z2
            * (X1 * Z1Z2 * (x0 + s0) + X2 * (D + Z1.mul_sb()).square());
        self.Z = x0 * Z1Z2;
        self.T = self.X * self.Z;

        // Corrective action in case the source was the neutral.
        let wz = x0.iszero();
        self.S.set_cond(&Self::NEUTRAL.S, wz);
        self.Z.set_cond(&Self::NEUTRAL.Z, wz);
    }
    */

    #[allow(dead_code)]
    fn to_affine(self) -> PointAffine {
        let iZ = self.Z.square().invert();
        let qx = self.T * iZ;
        let qs = self.S * iZ;
        PointAffine { scaled_x: qx, scaled_s: qs }
    }

    /// This function is defined only for benchmarking purposes. It
    /// implements a kind of ECDH key exchange, with the following
    /// characteristics:
    ///
    ///  - Input point `pp` is a raw, uncompressed point (64 bytes).
    ///  - Output is the resulting point, also uncompressed; no hashing
    ///    is performed.
    ///  - Scalar `sk` is provided encoded over 32 bytes.
    ///  - Input point is not considered secret (only the scalar).
    ///
    /// For a proper key exchange, with compressed points and extra
    /// hashing to get an unbiased output cryptographically bounded to
    /// the source points, and constant-time processing so that even the
    /// input point can be secret, use `PrivateKey::ECDH()`.
    ///
    /// This function uses a one-dimensional lookup table, and processes
    /// scalar bits in groups of 3.
    #[cfg(feature = "gls254bench")]
    pub fn for_benchmarks_only_1dt_3(pp: &[u8], sk: &[u8]) -> Option<[u8; 64]> {
        // Decode the input point in scaled affine coordinates; fail if
        // the decoding fails (invalid representation of field elements)
        // or if the resulting point is not on the curve, or is not in
        // the proper group (the points P+N, for P \in E[r]).

        // x and s must decode as field elements.
        if pp.len() != 64 {
            return None;
        }
        let px = GFb254::decode(&pp[..32])?;
        let ps = GFb254::decode(&pp[32..])?;

        // Curve equation: s^2 + x*s = (x^2 + a*x + b)^2
        // Our (x,s) values are scaled by 1/sqrt(b), so we must first
        // "unscale" them.
        let ux = px.mul_sb();
        let us = ps.mul_sb();
        let uv = (us + ux.square() + ux.mul_u() + Self::B).square() + us * ux;
        if uv.iszero() == 0 {
            return None;
        }

        // Points P+N are the points with Tr(x) != Tr(a). We must use the
        // unscaled x coordinate.
        if ux.trace() != 0 {
            return None;
        }

        let mut Pa = PointAffine { scaled_x: px, scaled_s: ps };
        let n = Scalar::decode(sk)?;

        // Split the scalar with the endomorphism.
        let (n0, s0, n1, s1) = Self::split_mu(&n);

        // Compute the 3-bit window, first in extended coordinates,
        // then normalized to affine.
        let mut win_ex = [Point::NEUTRAL; 4];
        Pa.set_condneg(s0);
        (win_ex[1], win_ex[2]) = Self::double_and_triple_affine(&Pa);
        win_ex[3] = win_ex[1].double();

        // Batch inversion for normalization. Since Z != 0 for all points
        // (including the neutral), this always works. The first point
        // is Pa and is already affine.
        let mut win = [GFb254::ZERO; 8];
        for i in 1..4 {
            win_ex[i].Z.set_square();
        }
        let zz = win_ex[1].Z * win_ex[2].Z;
        let mut izz = (zz * win_ex[3].Z).invert();
        let iZ = izz * zz;
        win[6] = win_ex[3].T * iZ;
        win[7] = win_ex[3].S * iZ;
        izz *= win_ex[3].Z;
        let iZ = izz * win_ex[1].Z;
        win[4] = win_ex[2].T * iZ;
        win[5] = win_ex[2].S * iZ;
        let iZ = izz * win_ex[2].Z;
        win[2] = win_ex[1].T * iZ;
        win[3] = win_ex[1].S * iZ;
        win[0] = Pa.scaled_x;
        win[1] = Pa.scaled_s;

        // If n0 and n1 have distinct signs, then we need to apply
        // the -zeta endomorphism instead of zeta.
        let zn = s0 ^ s1;

        // Recode the two half-width scalars into 43 digits each.
        let sd0 = Self::recode3_u128(n0);
        let sd1 = Self::recode3_u128(n1);

        // Process the two digit sequences in high-to-low order.
        let Q0 = Self::lookup4_affine(&win, sd0[42]);
        let Q1 = Self::lookup4_affine_zeta(&win, sd1[42], zn);
        let mut P = Self::add_affine_affine(&Q0, &Q1);
        for i in (0..42).rev() {
            P.set_xdouble(3);
            let Q0 = Self::lookup4_affine(&win, sd0[i]);
            let Q1 = Self::lookup4_affine_zeta(&win, sd1[i], zn);
            P.set_add(&Self::add_affine_affine(&Q0, &Q1));
        }

        let Qa = P.to_affine();
        let mut qq = [0u8; 64];
        qq[..32].copy_from_slice(&Qa.scaled_x.encode());
        qq[32..].copy_from_slice(&Qa.scaled_s.encode());
        Some(qq)
    }

    /// This function is defined only for benchmarking purposes. It
    /// implements a kind of ECDH key exchange, with the following
    /// characteristics:
    ///
    ///  - Input point `pp` is a raw, uncompressed point (64 bytes).
    ///  - Output is the resulting point, also uncompressed; no hashing
    ///    is performed.
    ///  - Scalar `sk` is provided encoded over 32 bytes.
    ///  - Input point is not considered secret (only the scalar).
    ///
    /// For a proper key exchange, with compressed points and extra
    /// hashing to get an unbiased output cryptographically bounded to
    /// the source points, and constant-time processing so that even the
    /// input point can be secret, use `PrivateKey::ECDH()`.
    ///
    /// This function uses a one-dimensional lookup table, and processes
    /// scalar bits in groups of 4.
    #[cfg(feature = "gls254bench")]
    pub fn for_benchmarks_only_1dt_4(pp: &[u8], sk: &[u8]) -> Option<[u8; 64]> {
        // Decode the input point in scaled affine coordinates; fail if
        // the decoding fails (invalid representation of field elements)
        // or if the resulting point is not on the curve, or is not in
        // the proper group (the points P+N, for P \in E[r]).

        // x and s must decode as field elements.
        if pp.len() != 64 {
            return None;
        }
        let px = GFb254::decode(&pp[..32])?;
        let ps = GFb254::decode(&pp[32..])?;

        // Curve equation: s^2 + x*s = (x^2 + a*x + b)^2
        // Our (x,s) values are scaled by 1/sqrt(b), so we must first
        // "unscale" them.
        let ux = px.mul_sb();
        let us = ps.mul_sb();
        let uv = (us + ux.square() + ux.mul_u() + Self::B).square() + us * ux;
        if uv.iszero() == 0 {
            return None;
        }

        // Points P+N are the points with Tr(x) != Tr(a). We must use the
        // unscaled x coordinate.
        if ux.trace() != 0 {
            return None;
        }

        let mut Pa = PointAffine { scaled_x: px, scaled_s: ps };
        let n = Scalar::decode(sk)?;

        // Split the scalar with the endomorphism.
        let (n0, s0, n1, s1) = Self::split_mu(&n);

        // Compute the 4-bit window, first in extended coordinates,
        // then normalized to affine.
        let mut win_ex = [Point::NEUTRAL; 8];
        Pa.set_condneg(s0);
        (win_ex[1], win_ex[2]) = Self::double_and_triple_affine(&Pa);
        win_ex[3] = win_ex[1].double();
        win_ex[5] = win_ex[2].double();
        (win_ex[6], win_ex[4]) = win_ex[5].add_sub_affine(&Pa);
        win_ex[7] = win_ex[3].double();

        // Batch inversion for normalization. Since Z != 0 for all points
        // (including the neutral), this always works. The first point
        // is Pa and is already affine.
        // Note: win_ex[4] and win_ex[6] have the same Z coordinate (since
        // they came from add_sub_affine()), so we can skip a few operations.
        let mut win = [GFb254::ZERO; 16];
        for i in 1..8 {
            if i != 6 {
                win_ex[i].Z.set_square();
            }
        }
        let mut zz = win_ex[1].Z;
        for i in 2..8 {
            if i != 6 {
                win[2 * i] = zz;
                zz *= win_ex[i].Z;
            }
        }
        zz.set_invert();
        for i in (2..8).rev() {
            match i {
                6 => (),
                4 => {
                    let iZ = zz * win[2 * i];
                    zz *= win_ex[i].Z;
                    win[(2 * i) + 0] = win_ex[i].T * iZ;
                    win[(2 * i) + 1] = win_ex[i].S * iZ;
                    win[(2 * i) + 4] = win_ex[i + 2].T * iZ;
                    win[(2 * i) + 5] = win_ex[i + 2].S * iZ;
                },
                _ => {
                    let iZ = zz * win[2 * i];
                    zz *= win_ex[i].Z;
                    win[(2 * i) + 0] = win_ex[i].T * iZ;
                    win[(2 * i) + 1] = win_ex[i].S * iZ;
                },
            }
        }
        win[2] = win_ex[1].T * zz;
        win[3] = win_ex[1].S * zz;
        win[0] = Pa.scaled_x;
        win[1] = Pa.scaled_s;

        // If n0 and n1 have distinct signs, then we need to apply
        // the -zeta endomorphism instead of zeta.
        let zn = s0 ^ s1;

        // Recode the two half-width scalars into 32 digits each.
        let sd0 = Self::recode4_u128(n0);
        let sd1 = Self::recode4_u128(n1);

        // Process the two digit sequences in high-to-low order.
        let Q0 = Self::lookup8_affine(&win, sd0[31]);
        let Q1 = Self::lookup8_affine_zeta(&win, sd1[31], zn);
        let mut P = Self::add_affine_affine(&Q0, &Q1);
        for i in (0..31).rev() {
            P.set_xdouble(4);
            let Q0 = Self::lookup8_affine(&win, sd0[i]);
            let Q1 = Self::lookup8_affine_zeta(&win, sd1[i], zn);
            P.set_add(&Self::add_affine_affine(&Q0, &Q1));
        }

        let Qa = P.to_affine();
        let mut qq = [0u8; 64];
        qq[..32].copy_from_slice(&Qa.scaled_x.encode());
        qq[32..].copy_from_slice(&Qa.scaled_s.encode());
        Some(qq)
    }

    /// This function is defined only for benchmarking purposes. It
    /// implements a kind of ECDH key exchange, with the following
    /// characteristics:
    ///
    ///  - Input point `pp` is a raw, uncompressed point (64 bytes).
    ///  - Output is the resulting point, also uncompressed; no hashing
    ///    is performed.
    ///  - Scalar `sk` is provided encoded over 32 bytes.
    ///  - Input point is not considered secret (only the scalar).
    ///
    /// For a proper key exchange, with compressed points and extra
    /// hashing to get an unbiased output cryptographically bounded to
    /// the source points, and constant-time processing so that even the
    /// input point can be secret, use `PrivateKey::ECDH()`.
    ///
    /// This function uses a one-dimensional lookup table, and processes
    /// scalar bits in groups of 5.
    #[cfg(feature = "gls254bench")]
    pub fn for_benchmarks_only_1dt_5(pp: &[u8], sk: &[u8]) -> Option<[u8; 64]> {
        // Decode the input point in scaled affine coordinates; fail if
        // the decoding fails (invalid representation of field elements)
        // or if the resulting point is not on the curve, or is not in
        // the proper group (the points P+N, for P \in E[r]).

        // x and s must decode as field elements.
        if pp.len() != 64 {
            return None;
        }
        let px = GFb254::decode(&pp[..32])?;
        let ps = GFb254::decode(&pp[32..])?;

        // Curve equation: s^2 + x*s = (x^2 + a*x + b)^2
        // Our (x,s) values are scaled by 1/sqrt(b), so we must first
        // "unscale" them.
        let ux = px.mul_sb();
        let us = ps.mul_sb();
        let uv = (us + ux.square() + ux.mul_u() + Self::B).square() + us * ux;
        if uv.iszero() == 0 {
            return None;
        }

        // Points P+N are the points with Tr(x) != Tr(a). We must use the
        // unscaled x coordinate.
        if ux.trace() != 0 {
            return None;
        }

        let mut Pa = PointAffine { scaled_x: px, scaled_s: ps };
        let n = Scalar::decode(sk)?;

        // Split the scalar with the endomorphism.
        let (n0, s0, n1, s1) = Self::split_mu(&n);

        // Compute the 4-bit window, first in extended coordinates,
        // then normalized to affine.
        let mut win_ex = [Point::NEUTRAL; 16];
        Pa.set_condneg(s0);
        (win_ex[1], win_ex[2]) = Self::double_and_triple_affine(&Pa);
        win_ex[3] = win_ex[1].double();
        win_ex[5] = win_ex[2].double();
        (win_ex[6], win_ex[4]) = win_ex[5].add_sub_affine(&Pa);
        win_ex[7] = win_ex[3].double();
        win_ex[9] = win_ex[4].double();
        (win_ex[10], win_ex[8]) = win_ex[9].add_sub_affine(&Pa);
        win_ex[11] = win_ex[5].double();
        win_ex[13] = win_ex[6].double();
        (win_ex[14], win_ex[12]) = win_ex[13].add_sub_affine(&Pa);
        win_ex[15] = win_ex[7].double();

        // Batch inversion for normalization. Since Z != 0 for all points
        // (including the neutral), this always works. The first point
        // is Pa and is already affine.
        // Note: elements 6, 10 and 14 have the same Z coordinate as
        // elements 4, 8 and 12, respectively (they were obtained from
        // add_sub_affine()), so we can skip a few multiplications and
        // squarings.
        let mut win = [GFb254::ZERO; 32];
        for i in 1..16 {
            match i {
                6 | 10 | 14 => (),
                _ => { win_ex[i].Z.set_square(); },
            }
        }
        let mut zz = win_ex[1].Z;
        for i in 2..16 {
            match i {
                6 | 10 | 14 => (),
                _ => {
                    win[2 * i] = zz;
                    zz *= win_ex[i].Z;
                },
            }
        }
        zz.set_invert();
        for i in (2..16).rev() {
            match i {
                6 | 10 | 14 => (),
                4 | 8 | 12 => {
                    let iZ = zz * win[2 * i];
                    zz *= win_ex[i].Z;
                    win[(2 * i) + 0] = win_ex[i].T * iZ;
                    win[(2 * i) + 1] = win_ex[i].S * iZ;
                    win[(2 * i) + 4] = win_ex[i + 2].T * iZ;
                    win[(2 * i) + 5] = win_ex[i + 2].S * iZ;
                },
                _ => {
                    let iZ = zz * win[2 * i];
                    zz *= win_ex[i].Z;
                    win[(2 * i) + 0] = win_ex[i].T * iZ;
                    win[(2 * i) + 1] = win_ex[i].S * iZ;
                },
            }
        }
        win[2] = win_ex[1].T * zz;
        win[3] = win_ex[1].S * zz;
        win[0] = Pa.scaled_x;
        win[1] = Pa.scaled_s;

        // If n0 and n1 have distinct signs, then we need to apply
        // the -zeta endomorphism instead of zeta.
        let zn = s0 ^ s1;

        // Recode the two half-width scalars into 26 digits each.
        let sd0 = Self::recode5_u128(n0);
        let sd1 = Self::recode5_u128(n1);

        // Process the two digit sequences in high-to-low order.
        let Q0 = Self::lookup16_affine(&win, sd0[25]);
        let Q1 = Self::lookup16_affine_zeta(&win, sd1[25], zn);
        let mut P = Self::add_affine_affine(&Q0, &Q1);
        for i in (0..25).rev() {
            P.set_xdouble(5);
            let Q0 = Self::lookup16_affine(&win, sd0[i]);
            let Q1 = Self::lookup16_affine_zeta(&win, sd1[i], zn);
            P.set_add(&Self::add_affine_affine(&Q0, &Q1));
        }

        let Qa = P.to_affine();
        let mut qq = [0u8; 64];
        qq[..32].copy_from_slice(&Qa.scaled_x.encode());
        qq[32..].copy_from_slice(&Qa.scaled_s.encode());
        Some(qq)
    }

    /// This function is defined only for benchmarking purposes. It
    /// implements a kind of ECDH key exchange, with the following
    /// characteristics:
    ///
    ///  - Input point `pp` is a raw, uncompressed point (64 bytes).
    ///  - Output is the resulting point, also uncompressed; no hashing
    ///    is performed.
    ///  - Scalar `sk` is provided encoded over 32 bytes.
    ///  - Input point is not considered secret (only the scalar).
    ///
    /// For a proper key exchange, with compressed points and extra
    /// hashing to get an unbiased output cryptographically bounded to
    /// the source points, and constant-time processing so that even the
    /// input point can be secret, use `PrivateKey::ECDH()`.
    ///
    /// This function uses a two-dimensional lookup table, and processes
    /// scalar bits in groups of 2.
    #[cfg(feature = "gls254bench")]
    pub fn for_benchmarks_only_2dt_2(pp: &[u8], sk: &[u8]) -> Option<[u8; 64]> {
        // Decode the input point in scaled affine coordinates; fail if
        // the decoding fails (invalid representation of field elements)
        // or if the resulting point is not on the curve, or is not in
        // the proper group (the points P+N, for P \in E[r]).

        // x and s must decode as field elements.
        if pp.len() != 64 {
            return None;
        }
        let px = GFb254::decode(&pp[..32])?;
        let ps = GFb254::decode(&pp[32..])?;

        // Curve equation: s^2 + x*s = (x^2 + a*x + b)^2
        // Our (x,s) values are scaled by 1/sqrt(b), so we must first
        // "unscale" them.
        let ux = px.mul_sb();
        let us = ps.mul_sb();
        let uv = (us + ux.square() + ux.mul_u() + Self::B).square() + us*ux;
        if uv.iszero() == 0 {
            return None;
        }

        // Points P+N are the points with Tr(x) != Tr(a). We must use the
        // unscaled x coordinate.
        if ux.trace() != 0 {
            return None;
        }

        let mut P = PointAffine { scaled_x: px, scaled_s: ps };
        let n = Scalar::decode(sk)?;

        // We follow here the "2DT" method described in:
        //   https://eprint.iacr.org/2022/748
        // It is slightly modified here, in the following ways:
        //  - We use (x,s) coordinates and complete formulas, so we do
        //    not have to make special provisions for a neutral input
        //    or the last iteration; we simply do not have any exceptional
        //    case to care about.
        //  - The scalar splitting is a bit different; the split_mu_inner()
        //    function returns the optimal solution (no off-by-1) and we
        //    ensure odd integers in the output by simply modifying the
        //    input scalar, and adjusting the result (see split_mu_odd()).

        // Split the scalar into odd integers (128 bits + sign).
        let (n0, s0, n1, s1) = Self::split_mu_odd(&n);

        // The scalar n was split into n0 and n1, with n = n0 + n1*mu.
        // We have abs(n0) and abs(n1) into the `n0` and `n1` variables;
        // we will conditionally negatd the source operand point, and also
        // the output of zeta(), so as to take into account the signs
        // of n0 and n1, and avoid any further sign handling. Thus, we do
        // the following:
        //   If s0 < 0 then we negate the base point (self).
        //   All applications of zeta really are zeta(nz), with nz = s0 XOR s1.
        P.set_condneg(s0);
        let nz = s0 ^ s1;

        // Compute the 2D table:
        //   P + zeta(P)
        //   3*P + zeta(P)
        //   P + zeta(3*P)
        //   3*P + zeta(3*P)
        let zP = P.zeta(nz);
        let P3 = Self::triple_affine(&P);
        let Q0 = Self::add_affine_selfzeta(&P, nz);
        let (mut Q1, mut Q2) = P3.add_sub_affine(&zP);
        Q2.set_zeta(nz);
        let mut Q3 = P3;
        Q3.set_add_selfzeta(nz);

        // Normalize the table to affine.
        // Note:
        //   Q0.Z and Q3.Z are in GF(2^127)
        //   Q2.Z = phi(Q1.Z)  (add_sub_affine() + zeta() on Q2)
        let mut win = [GFb254::ZERO; 8];
        let (mut q0z, _) = Q0.Z.to_components();
        q0z.set_square();
        let (mut q3z, _) = Q3.Z.to_components();
        q3z.set_square();
        Q1.Z.set_square();
        let sav1 = q0z * q3z;
        let mut zz = Q1.Z.mul_b127(&sav1);
        zz.set_invert();
        let iZ = zz.mul_b127(&sav1);
        zz *= Q1.Z;
        win[2] = Q1.T * iZ;
        win[3] = Q1.S * iZ;
        let (iz0, iz1) = iZ.to_components();
        let iZ = GFb254::from_b127(iz0 + iz1, iz1);
        win[4] = Q2.T * iZ;
        win[5] = Q2.S * iZ;
        let (zzf, _) = zz.to_components();
        let izz = zzf * q0z;
        win[6] = Q3.T.mul_b127(&izz);
        win[7] = Q3.S.mul_b127(&izz);
        let izz = zzf * q3z;
        win[0] = Q0.T.mul_b127(&izz);
        win[1] = Q0.S.mul_b127(&izz);

        // The window contains 4 points:
        //   win[4*j + i] = (2*i+1)*P + (2*j+1)*zeta(P)
        // (Each point is two coordinates: scaled affine representation.)
        // In other words, win[] contains all points e*P + f*zeta(P) for
        // e and f in {1, 3}.
        //
        // This function returns i*P + j*zeta(P) for i and j both in
        // {-3, -1, +1, +3}. The sign adjustments leverage
        // the fact that zeta^2 = -1, i.e.:
        //   zeta(e*P + f*zeta(P)) = -f*P + e*zeta(P)
        // If neg is 0xFFFFFFFF, then '-zeta' is used instead of 'zeta' in
        // all of the above.
        #[inline(always)]
        fn lookup_2dt(win: &[GFb254; 8], neg: u32, i: i8, j: i8)
            -> PointAffine
        {
            // We have two conditional operations:
            //   swap: swap i and j (before the lookup), apply zeta on output
            //   neg:  negate the output
            // The lookup itself uses |i| and |j| (possibly swapped).
            //    sign(i)  sign(j)  operations
            //     >= 0     >= 0     none
            //     >= 0     < 0      swap + negate
            //     < 0      >= 0     swap
            //     < 0      < 0      negate

            // Get absolute value and sign for each index.
            let si = ((i as i32) >> 8) as u32;
            let mut ui = ((i as u32) ^ (si as u32)).wrapping_sub(si);
            let sj = ((j as i32) >> 8) as u32;
            let mut uj = ((j as u32) ^ (sj as u32)).wrapping_sub(sj);

            // Swap absolute values if the signs differ.
            let do_swap = si ^ sj;
            let t = do_swap & (ui ^ uj);
            ui ^= t;
            uj ^= t;

            // Lookup the point.
            // The two absolute values are at most 3 each, so the combined
            // index cannot exceed 3.
            let k = (ui >> 1) + (2 * (uj >> 1));
            let v = GFb254::lookup4_x2_nocheck(&win, k);
            let mut P = PointAffine { scaled_x: v[0], scaled_s: v[1] };

            // Post-lookup adjustments.
            P.set_condzeta(do_swap, neg);
            P.set_condneg(sj);

            P
        }

        // Recode the two half-width odd scalars into 64 digits each.
        let sd0 = Self::recode2_u128_odd(n0);
        let sd1 = Self::recode2_u128_odd(n1);

        // Process the two digit sequences in high-to-low order.
        let mut Q = Self::from_affine(&lookup_2dt(&win, nz, sd0[63], sd1[63]));
        for i in (0..63).rev() {
            Q.set_xdouble_add_affine(2,
                &lookup_2dt(&win, nz, sd0[i], sd1[i]));
        }

        let Qa = Q.to_affine();
        let mut qq = [0u8; 64];
        qq[..32].copy_from_slice(&Qa.scaled_x.encode());
        qq[32..].copy_from_slice(&Qa.scaled_s.encode());
        Some(qq)
    }

    /// This function is defined only for benchmarking purposes. It
    /// implements a kind of ECDH key exchange, with the following
    /// characteristics:
    ///
    ///  - Input point `pp` is a raw, uncompressed point (64 bytes).
    ///  - Output is the resulting point, also uncompressed; no hashing
    ///    is performed.
    ///  - Scalar `sk` is provided encoded over 32 bytes.
    ///  - Input point is not considered secret (only the scalar).
    ///
    /// For a proper key exchange, with compressed points and extra
    /// hashing to get an unbiased output cryptographically bounded to
    /// the source points, and constant-time processing so that even the
    /// input point can be secret, use `PrivateKey::ECDH()`.
    ///
    /// This function uses a two-dimensional lookup table, and processes
    /// scalar bits in groups of 3.
    #[cfg(feature = "gls254bench")]
    pub fn for_benchmarks_only_2dt_3(pp: &[u8], sk: &[u8]) -> Option<[u8; 64]> {
        // Decode the input point in scaled affine coordinates; fail if
        // the decoding fails (invalid representation of field elements)
        // or if the resulting point is not on the curve, or is not in
        // the proper group (the points P+N, for P \in E[r]).

        // x and s must decode as field elements.
        if pp.len() != 64 {
            return None;
        }
        let px = GFb254::decode(&pp[..32])?;
        let ps = GFb254::decode(&pp[32..])?;

        // Curve equation: s^2 + x*s = (x^2 + a*x + b)^2
        // Our (x,s) values are scaled by 1/sqrt(b), so we must first
        // "unscale" them.
        let ux = px.mul_sb();
        let us = ps.mul_sb();
        let uv = (us + ux.square() + ux.mul_u() + Self::B).square() + us*ux;
        if uv.iszero() == 0 {
            return None;
        }

        // Points P+N are the points with Tr(x) != Tr(a). We must use the
        // unscaled x coordinate.
        if ux.trace() != 0 {
            return None;
        }

        let mut P = PointAffine { scaled_x: px, scaled_s: ps };
        let n = Scalar::decode(sk)?;

        // We follow here the "2DT" method described in:
        //   https://eprint.iacr.org/2022/748
        // It is slightly modified here, in the following ways:
        //  - We use (x,s) coordinates and complete formulas, so we do
        //    not have to make special provisions for a neutral input
        //    or the last iteration; we simply do not have any exceptional
        //    case to care about.
        //  - The scalar splitting is a bit different; the split_mu_inner()
        //    function returns the optimal solution (no off-by-1) and we
        //    ensure odd integers in the output by simply modifying the
        //    input scalar, and adjusting the result (see split_mu_odd()).
        //  - We process bits by groups of 3 bits at a time because our
        //    multi-doubling formulas (xdouble()) are better from longer
        //    runs.

        // Split the scalar into odd integers (128 bits + sign).
        let (n0, s0, n1, s1) = Self::split_mu_odd(&n);

        // The scalar n was split into n0 and n1, with n = n0 + n1*mu.
        // We have abs(n0) and abs(n1) into the `n0` and `n1` variables;
        // we will conditionally negatd the source operand point, and also
        // the output of zeta(), so as to take into account the signs
        // of n0 and n1, and avoid any further sign handling. Thus, we do
        // the following:
        //   If s0 < 0 then we negate the base point (self).
        //   All applications of zeta really are zeta(nz), with nz = s0 XOR s1.
        P.set_condneg(s0);
        let nz = s0 ^ s1;

        // Compute the 2D table: i*P + j*zeta(P), for all combinations of
        // (i,j) in {1, 3, 5, 7}.

        // We first compute P, 3*P, 5*P and 7*P, and normalize them to
        // affine coordinates.
        let R1 = P;
        let mut P3 = Self::triple_affine(&P);
        let (P7, mut P5) = P3.double().add_sub_affine(&P);
        // Note: P5 and P7 come from add_sub_affine(); they have the same Z.
        P3.Z.set_square();
        P5.Z.set_square();
        let zz = (P3.Z * P5.Z).invert();
        let iZ = zz * P3.Z;
        let R7 = PointAffine {
            scaled_x: P7.T * iZ,
            scaled_s: P7.S * iZ,
        };
        let R5 = PointAffine {
            scaled_x: P5.T * iZ,
            scaled_s: P5.S * iZ,
        };
        let iZ = zz * P5.Z;
        let R3 = PointAffine {
            scaled_x: P3.T * iZ,
            scaled_s: P3.S * iZ,
        };

        //   win[i + 4*j] = (2*i+1)*P + (2*j+1)*zeta(P)
        let mut win_ex = [Self::NEUTRAL; 16];

        // Diagonal elements: i*P + j*zeta(P)
        win_ex[0] = Self::add_affine_selfzeta(&R1, nz);
        win_ex[5] = Self::add_affine_selfzeta(&R3, nz);
        win_ex[10] = Self::add_affine_selfzeta(&R5, nz);
        win_ex[15] = Self::add_affine_selfzeta(&R7, nz);

        // Non-diagonal elements. We leverage the fact that:
        //   zeta(i*P - j*zeta(P)) = j*P + i*zeta(P)
        let Q = R3.zeta(nz);
        (win_ex[4], win_ex[1]) = Self::add_sub_affine_affine(&R1, &Q);
        win_ex[1].set_zeta(nz);
        let Q = R5.zeta(nz);
        (win_ex[8], win_ex[2]) = Self::add_sub_affine_affine(&R1, &Q);
        win_ex[2].set_zeta(nz);
        (win_ex[9], win_ex[6]) = Self::add_sub_affine_affine(&R3, &Q);
        win_ex[6].set_zeta(nz);
        let Q = R7.zeta(nz);
        (win_ex[12], win_ex[3]) = Self::add_sub_affine_affine(&R1, &Q);
        win_ex[3].set_zeta(nz);
        (win_ex[13], win_ex[7]) = Self::add_sub_affine_affine(&R3, &Q);
        win_ex[7].set_zeta(nz);
        (win_ex[14], win_ex[11]) = Self::add_sub_affine_affine(&R5, &Q);
        win_ex[11].set_zeta(nz);

        // Normalize the table to affine.
        // Note: since the two outputs of add_sub*() have the same Z, and
        // zeta() applies the Frobenius automorphism phi() on Z, we have:
        //   win_ex[1].Z = phi(win_ex[4]).Z
        //   win_ex[2].Z = phi(win_ex[8]).Z
        //   win_ex[3].Z = phi(win_ex[12]).Z
        //   win_ex[6].Z = phi(win_ex[9]).Z
        //   win_ex[7].Z = phi(win_ex[13]).Z
        //   win_ex[11].Z = phi(win_ex[14]).Z
        let mut win = [GFb254::ZERO; 32];
        for i in 0..16 {
            match i {
                1 | 2 | 3 | 6 | 7 | 11 => (),
                _ => {
                    win_ex[i].Z.set_square();
                },
            }
        }
        let mut zz = win_ex[0].Z;
        for i in 1..16 {
            match i {
                1 | 2 | 3 | 6 | 7 | 11 => (),
                _ => {
                    win[2 * i] = zz;
                    zz *= win_ex[i].Z;
                },
            }
        }
        zz.set_invert();
        for i in (1..16).rev() {
            match i {
                1 | 2 | 3 | 6 | 7 | 11 => (),
                _ => {
                    let iZ = zz * win[2 * i];
                    zz *= win_ex[i].Z;
                    win[(2 * i) + 0] = win_ex[i].T * iZ;
                    win[(2 * i) + 1] = win_ex[i].S * iZ;
                    match i {
                        4 => {
                            let (iz0, iz1) = iZ.to_components();
                            let iZ = GFb254::from_b127(iz0 + iz1, iz1);
                            win[(2 * 1) + 0] = win_ex[1].T * iZ;
                            win[(2 * 1) + 1] = win_ex[1].S * iZ;
                        },
                        8 => {
                            let (iz0, iz1) = iZ.to_components();
                            let iZ = GFb254::from_b127(iz0 + iz1, iz1);
                            win[(2 * 2) + 0] = win_ex[2].T * iZ;
                            win[(2 * 2) + 1] = win_ex[2].S * iZ;
                        },
                        12 => {
                            let (iz0, iz1) = iZ.to_components();
                            let iZ = GFb254::from_b127(iz0 + iz1, iz1);
                            win[(2 * 3) + 0] = win_ex[3].T * iZ;
                            win[(2 * 3) + 1] = win_ex[3].S * iZ;
                        },
                        9 => {
                            let (iz0, iz1) = iZ.to_components();
                            let iZ = GFb254::from_b127(iz0 + iz1, iz1);
                            win[(2 * 6) + 0] = win_ex[6].T * iZ;
                            win[(2 * 6) + 1] = win_ex[6].S * iZ;
                        },
                        13 => {
                            let (iz0, iz1) = iZ.to_components();
                            let iZ = GFb254::from_b127(iz0 + iz1, iz1);
                            win[(2 * 7) + 0] = win_ex[7].T * iZ;
                            win[(2 * 7) + 1] = win_ex[7].S * iZ;
                        },
                        14 => {
                            let (iz0, iz1) = iZ.to_components();
                            let iZ = GFb254::from_b127(iz0 + iz1, iz1);
                            win[(2 * 11) + 0] = win_ex[11].T * iZ;
                            win[(2 * 11) + 1] = win_ex[11].S * iZ;
                        },
                        _ => (),
                    }
                },
            }
        }
        win[0] = win_ex[0].T * zz;
        win[1] = win_ex[0].S * zz;

        // The window contains 16 points:
        //   win[i + 4*j] = (2*i+1)*P + (2*j+1)*zeta(P)
        // (Each point is two coordinates: scaled affine representation.)
        // In other words, win[] contains all points e*P + f*zeta(P) for
        // e and f in {1, 3, 5, 7}.
        //
        // This function returns i*P + j*zeta(P) for i and j both in
        // {-7, -5, -3, -1, +1, +3, +5, +7}. The sign adjustments leverage
        // the fact that zeta^2 = -1, i.e.:
        //   zeta(e*P + f*zeta(P)) = -f*P + e*zeta(P)
        // If neg is 0xFFFFFFFF, then '-zeta' is used instead of 'zeta' in
        // all of the above.
        #[inline(always)]
        fn lookup_2dt(win: &[GFb254; 32], neg: u32, i: i8, j: i8)
            -> PointAffine
        {
            // We have two conditional operations:
            //   swap: swap i and j (before the lookup), apply zeta on output
            //   neg:  negate the output
            // The lookup itself uses |i| and |j| (possibly swapped).
            //    sign(i)  sign(j)  operations
            //     >= 0     >= 0     none
            //     >= 0     < 0      swap + negate
            //     < 0      >= 0     swap
            //     < 0      < 0      negate

            // Get absolute value and sign for each index.
            let si = ((i as i32) >> 8) as u32;
            let mut ui = ((i as u32) ^ (si as u32)).wrapping_sub(si);
            let sj = ((j as i32) >> 8) as u32;
            let mut uj = ((j as u32) ^ (sj as u32)).wrapping_sub(sj);

            // Swap absolute values if the signs differ.
            let do_swap = si ^ sj;
            let t = do_swap & (ui ^ uj);
            ui ^= t;
            uj ^= t;

            // Lookup the point.
            // The two absolute values are at most 7 each, so the combined
            // index cannot exceed 15.
            let k = (ui >> 1) + (4 * (uj >> 1));
            let v = GFb254::lookup16_x2(&win, k);
            let mut P = PointAffine { scaled_x: v[0], scaled_s: v[1] };

            // Post-lookup adjustments.
            P.set_condzeta(do_swap, neg);
            P.set_condneg(sj);

            P
        }

        // Recode the two half-width odd scalars into 43 digits each.
        let sd0 = Self::recode3_u128_odd(n0);
        let sd1 = Self::recode3_u128_odd(n1);

        // Process the two digit sequences in high-to-low order.
        let mut Q = Self::from_affine(&lookup_2dt(&win, nz, sd0[42], sd1[42]));
        for i in (0..42).rev() {
            Q.set_xdouble_add_affine(3,
                &lookup_2dt(&win, nz, sd0[i], sd1[i]));
        }

        let Qa = Q.to_affine();
        let mut qq = [0u8; 64];
        qq[..32].copy_from_slice(&Qa.scaled_x.encode());
        qq[32..].copy_from_slice(&Qa.scaled_s.encode());
        Some(qq)
    }

    /// Sets this point by multiplying the conventional generator by the
    /// provided scalar.
    ///
    /// This operation is constant-time. It is faster than using the
    /// generic multiplication on `Self::BASE`.
    pub fn set_mulgen(&mut self, n: &Scalar) {
        // Split the scalar with the endomorphism.
        let (n0, s0, n1, s1) = Self::split_mu(n);

        // If n0 and n1 have distinct signs, then we need to apply
        // the -zeta endomorphism instead of zeta.
        let zn = s0 ^ s1;

        // Recode the two half-width scalars into 26 digits each.
        let sd0 = Self::recode5_u128(n0);
        let sd1 = Self::recode5_u128(n1);

        // We process eight chunks in parallel, with alternating sizes
        // (in digits): 6, 7, 6, 7, 6, 7, 6, 7. First four chunks are
        // for n0, and work over the precomputed tables for B, B30, B65
        // and B95; the four other chunks work over the same tables but
        // with the zeta endomorphism applied.
        let P = Self::lookup16_affine(&PRECOMP_B30, sd0[12]);
        let Q = Self::lookup16_affine(&PRECOMP_B95, sd0[25]);
        *self = Self::add_affine_affine(&P, &Q);
        let P = Self::lookup16_affine_zeta(&PRECOMP_B30, sd1[12], zn);
        let Q = Self::lookup16_affine_zeta(&PRECOMP_B95, sd1[25], zn);
        self.set_add(&Self::add_affine_affine(&P, &Q));

        for i in (0..6).rev() {
            self.set_xdouble(5);

            let P = Self::lookup16_affine(&PRECOMP_B, sd0[i]);
            let Q = Self::lookup16_affine(&PRECOMP_B30, sd0[i + 6]);
            self.set_add(&Self::add_affine_affine(&P, &Q));
            let P = Self::lookup16_affine(&PRECOMP_B65, sd0[i + 13]);
            let Q = Self::lookup16_affine(&PRECOMP_B95, sd0[i + 19]);
            self.set_add(&Self::add_affine_affine(&P, &Q));

            let P = Self::lookup16_affine_zeta(&PRECOMP_B, sd1[i], zn);
            let Q = Self::lookup16_affine_zeta(&PRECOMP_B30, sd1[i + 6], zn);
            self.set_add(&Self::add_affine_affine(&P, &Q));
            let P = Self::lookup16_affine_zeta(&PRECOMP_B65, sd1[i + 13], zn);
            let Q = Self::lookup16_affine_zeta(&PRECOMP_B95, sd1[i + 19], zn);
            self.set_add(&Self::add_affine_affine(&P, &Q));
        }

        // We need to negate the point if n0 was negative.
        self.set_condneg(s0);
    }

    /// Creates a point by multiplying the conventional generator by the
    /// provided scalar.
    ///
    /// This operation is constant-time.
    #[inline(always)]
    pub fn mulgen(n: &Scalar) -> Self {
        let mut P = Self::NEUTRAL;
        P.set_mulgen(n);
        P
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
        // Since there is an overhead to each sequence of successive
        // point doublings, it is advantageous to stick to long sequences.
        // This is why we use Booth recoding at a regular 5-bit window
        // rather than wNAF. The lookups are still "vartime" (direct
        // copy from the relevant index instead of whole table scanning).

        // Split both scalars with the endomorphism.
        let (u0, s0, u1, s1) = Self::split_mu(u);
        let (v0, t0, v1, t1) = Self::split_mu(v);

        // Apply the sign of the first half of the first scalar to the
        // current point. This way, only the second half (combined with
        // the zeta() endomorphism) needs to care about signs.
        self.set_condneg(s0);
        let sz = s0 ^ s1;

        // Compute the window for the current point:
        //   win[i] = i*self   (i = 1 to 16)
        let mut win_ex = [Self::NEUTRAL; 16];
        win_ex[0] = *self;
        win_ex[1] = win_ex[0].double();
        win_ex[2] = win_ex[1] + win_ex[0];
        win_ex[3] = win_ex[1].double();
        for i in 1..4 {
            win_ex[4 * i + 1] = win_ex[2 * i].double();
            (win_ex[4 * i + 2], win_ex[4 * i]) =
                win_ex[4 * i + 1].add_sub(&win_ex[0]);
            win_ex[4 * i + 3] = win_ex[2 * i + 1].double();
        }

        // It is not worth the effort to normalize the window to affine.
        // Costs below (for addition of points from the window):
        //   normalized:       26*(5M+1S + 8M+2S) = 338M + 78S
        //   not normalized:   26*(7M+2S + 7M+2S) = 364M + 104S
        // Making the normalized window has an overhead larger than the costs:
        //   normalization:    16S + inversion + 15*5M + 2M = 77M + 16S + I

        // 5-bit Booth recoding.
        let sd0 = Self::recode5_u128(u0);
        let sd1 = Self::recode5_u128(u1);
        let sd2 = Self::recode5_u128(v0);
        let sd3 = Self::recode5_u128(v1);

        *self = Self::add_affine_affine(
            &Self::lookup16_affine_vartime(&PRECOMP_B, sd2[25], t0),
            &Self::lookup16_affine_zeta_vartime(&PRECOMP_B, sd3[25], t1));
        *self += Self::lookup16_vartime(&win_ex, sd0[25]);
        *self += Self::lookup16_zeta_vartime(&win_ex, sd1[25], sz);

        for i in (0..25).rev() {
            self.set_xdouble(5);
            *self += Self::lookup16_vartime(&win_ex, sd0[i]);
            *self += Self::lookup16_zeta_vartime(&win_ex, sd1[i], sz);
            *self += Self::add_affine_affine(
                &Self::lookup16_affine_vartime(&PRECOMP_B, sd2[i], t0),
                &Self::lookup16_affine_zeta_vartime(&PRECOMP_B, sd3[i], t1));
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

    /// Given integer `u0` and `u1`, and scalar `v`, sets this point to
    /// `u0*self + u1*mu*self + v*B`, where `mu` is a specific square
    /// root of -1 modulo `r` (the prime subgroup order) and `B` is the
    /// conventional generator of the prime order subgroup. The value
    /// `mu` is the unique root of -1 which, when represented as an
    /// integer in the 0 to `r-1` range, is even.
    ///
    /// THIS FUNCTION IS NOT CONSTANT-TIME; it shall be used only with
    /// public data.
    pub fn set_mul64mu_add_mulgen_vartime(
        &mut self, u0: u64, u1: u64, v: &Scalar)
    {
        // Since there is an overhead to each sequence of successive
        // point doublings, it is advantageous to stick to long sequences.
        // This is why we use Booth recoding at a regular 5-bit window
        // rather than wNAF. The lookups are still "vartime" (direct
        // copy from the relevant index instead of whole table scanning).

        // Split the second scalar with the endomorphism.
        let (v0, t0, v1, t1) = Self::split_mu(v);

        // Compute the window for the current point:
        //   win[i] = i*self   (i = 1 to 16)
        let mut win_ex = [Self::NEUTRAL; 16];
        win_ex[0] = *self;
        win_ex[1] = win_ex[0].double();
        win_ex[2] = win_ex[1] + win_ex[0];
        win_ex[3] = win_ex[1].double();
        for i in 1..4 {
            win_ex[4 * i + 1] = win_ex[2 * i].double();
            (win_ex[4 * i + 2], win_ex[4 * i]) =
                win_ex[4 * i + 1].add_sub(&win_ex[0]);
            win_ex[4 * i + 3] = win_ex[2 * i + 1].double();
        }

        // It is not worth the effort to normalize the window to affine.
        // Costs below (for addition of points from the window):
        //   normalized:       13*(5M+1S + 8M+2S) = 169M + 39S
        //   not normalized:   13*(7M+2S + 7M+2S) = 182M + 52S
        // Making the normalized window has an overhead larger than the costs:
        //   normalization:    16S + inversion + 15*5M + 2M = 77M + 16S + I

        // 5-bit Booth recoding.
        let sd0 = Self::recode5_u64(u0);
        let sd1 = Self::recode5_u64(u1);
        let sd2 = Self::recode5_u128(v0);
        let sd3 = Self::recode5_u128(v1);

        *self = Self::add_affine_affine(
            &Self::lookup16_affine_vartime(&PRECOMP_B, sd2[12], t0),
            &Self::lookup16_affine_vartime(&PRECOMP_B65, sd2[25], t0));
        *self += Self::add_affine_affine(
            &Self::lookup16_affine_zeta_vartime(&PRECOMP_B, sd3[12], t1),
            &Self::lookup16_affine_zeta_vartime(&PRECOMP_B65, sd3[25], t1));
        *self += Self::lookup16_vartime(&win_ex, sd0[12]);
        *self += Self::lookup16_zeta_vartime(&win_ex, sd1[12], 0);

        for i in (0..12).rev() {
            self.set_xdouble(5);

            if sd0[i] != 0 {
                *self += Self::lookup16_vartime(&win_ex, sd0[i]);
            }
            if sd1[i] != 0 {
                *self += Self::lookup16_zeta_vartime(&win_ex, sd1[i], 0);
            }
            if sd2[i] != 0 && sd2[i + 13] != 0 {
                *self += Self::add_affine_affine(
                    &Self::lookup16_affine_vartime(
                        &PRECOMP_B, sd2[i], t0),
                    &Self::lookup16_affine_vartime(
                        &PRECOMP_B65, sd2[i + 13], t0));
            } else if sd2[i] != 0 {
                self.set_add_affine(
                    &Self::lookup16_affine_vartime(
                        &PRECOMP_B, sd2[i], t0));
            } else if sd2[i + 13] != 0 {
                self.set_add_affine(
                    &Self::lookup16_affine_vartime(
                        &PRECOMP_B65, sd2[i + 13], t0));
            }
            if sd3[i] != 0 && sd3[i + 13] != 0 {
                *self += Self::add_affine_affine(
                    &Self::lookup16_affine_zeta_vartime(
                        &PRECOMP_B, sd3[i], t1),
                    &Self::lookup16_affine_zeta_vartime(
                        &PRECOMP_B65, sd3[i + 13], t1));
            } else if sd3[i] != 0 {
                self.set_add_affine(
                    &Self::lookup16_affine_zeta_vartime(
                        &PRECOMP_B, sd3[i], t1));
            } else if sd3[i + 13] != 0 {
                self.set_add_affine(
                    &Self::lookup16_affine_zeta_vartime(
                        &PRECOMP_B65, sd3[i + 13], t1));
            }
        }
    }

    /// Given integer `u0` and `u1`, and scalar `v`, returns
    /// `u0*self + u1*mu*self + v*B`, where `mu` is a specific square
    /// root of -1 modulo `r` (the prime subgroup order) and `B` is the
    /// conventional generator of the prime order subgroup. The value
    /// `mu` is the unique root of -1 which, when represented as an
    /// integer in the 0 to `r-1` range, is even.
    ///
    /// THIS FUNCTION IS NOT CONSTANT-TIME; it shall be used only with
    /// public data.
    #[inline(always)]
    pub fn mul64mu_add_mulgen_vartime(self, u0: u64, u1: u64, v: &Scalar)
        -> Self
    {
        let mut R = self;
        R.set_mul64mu_add_mulgen_vartime(u0, u1, v);
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

/// A GLS254 private key.
///
/// Such a key wraps around a secret non-zero scalar. It also contains
/// a copy of the public key.
#[derive(Clone, Copy, Debug)]
pub struct PrivateKey {
    sec: Scalar,                // secret scalar
    pub public_key: PublicKey,  // public key
}

/// A GLS254 public key.
///
/// It wraps around a GLS254 element, but also includes a copy of the
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
    /// or random) improves resistance to fault attacks (where an
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
        // negligible bias because the GLS254 order is close enough to
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

        let c0 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&cb[..8]).unwrap());
        let c1 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&cb[8..]).unwrap());
        let c = Scalar::from_u64(c0) + Scalar::MU * Scalar::from_u64(c1);
        let s = k + self.sec * c;
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
    /// a valid GLS254 element, or encodes the neutral element. On success,
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
    /// or the bytes do not encode a valid GLS254 element, or the bytes
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
        let (s, ok) = Scalar::decode32(&sig[16..48]);
        if ok == 0 {
            return false;
        }

        let cb = &sig[..16];
        let c0 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&cb[..8]).unwrap());
        let c1 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&cb[8..]).unwrap());
        let R = (-self.point).mul64mu_add_mulgen_vartime(c0, c1, &s);
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
// used to speed mulgen() operations up.

/// A point in affine extended coordinates (x, s)
/// WARNING: the stored values are actually scaled by 1/sqrt(b) (i.e.
/// they must be multiplied by sqrt(b) to yield the actual (x, s), if
/// such values are needed).
#[derive(Clone, Copy, Debug)]
struct PointAffine {
    scaled_x: GFb254,
    scaled_s: GFb254,
}

impl PointAffine {

    const NEUTRAL: Self = Self {
        scaled_x: GFb254::ZERO,
        scaled_s: Point::SB,
    };

    /// Lookups a point from a window of points in affine
    /// coordinates, with sign handling (constant-time).
    #[inline(always)]
    fn lookup16(win: &[GFb254; 32], k: i8) -> Self {
        // Split k into its sign sf (0xFFFFFFFF for negative) and
        // absolute value (f).
        let sf = ((k as i32) >> 8) as u32;
        let f = ((k as u32) ^ sf).wrapping_sub(sf);

        // Get the correct coordinates for the absolute value.
        let fj = f.wrapping_sub(1);
        let mut vv = GFb254::lookup16_x2(win, fj);

        // If the absolute value was zero then we get only zeros and we
        // must adjust the s coordinate.
        let fz = ((fj as i32) >> 31) as u32;
        vv[1].set_cond(&PointAffine::NEUTRAL.scaled_s, fz);

        // Negate the point if the original index was negative.
        vv[1].set_cond(&(vv[0] + vv[1]), sf);

        Self {
            scaled_x: vv[0],
            scaled_s: vv[1],
        }
    }

    /// Lookups a point from a window of points in affine
    /// coordinates, with sign handling (variable time: side channels
    /// may leak the value of the index `k`).
    #[inline(always)]
    fn lookup16_vartime(win: &[GFb254; 32], k: i8) -> Self {
        if k > 0 {
            Self {
                scaled_x: win[(k as usize) * 2 - 2],
                scaled_s: win[(k as usize) * 2 - 1],
            }
        } else if k < 0 {
            let scaled_x = win[(-k as usize) * 2 - 2];
            Self {
                scaled_x: scaled_x,
                scaled_s: scaled_x + win[(-k as usize) * 2 - 1],
            }
        } else {
            Self::NEUTRAL
        }
    }

    /// Apply the zeta() endomorphism on this point. Value `neg` must be
    /// 0x00000000 or 0xFFFFFFFF; if it is `0xFFFFFFFF`, then the point
    /// is also negated.
    #[allow(dead_code)]
    #[inline(always)]
    fn set_zeta(&mut self, neg: u32) {
        // (x, s) -> (x', s'), with:
        //    x' = (x0 + x1) + x1*u
        //    s' = (s0 + s1 + x0) + (s1 + x0 + x1)*u
        // These formulas still apply on scaled_x and scaled_s, since
        // the scaling factor is 1/sqrt(b), which is in GF(2^127).
        let (x0, x1) = self.scaled_x.to_components();
        let (s0, s1) = self.scaled_s.to_components();
        self.scaled_x = GFb254::from_b127(x0 + x1, x1);
        self.scaled_s = GFb254::from_b127(s0 + s1 + x0, s1 + x0 + x1);
        self.scaled_s.set_cond(&(self.scaled_x + self.scaled_s), neg);
    }

    #[allow(dead_code)]
    #[inline(always)]
    fn zeta(self, neg: u32) -> Self {
        let mut P = self;
        P.set_zeta(neg);
        P
    }

    /// Negate this point (conditionally).
    #[inline(always)]
    fn set_condneg(&mut self, neg: u32) {
        self.scaled_s.set_cond(&(self.scaled_x + self.scaled_s), neg);
    }

    /// Apply the zeta() endomorphism on this point conditionally.
    /// Two flags are applied; each flag is either 0xFFFFFFFF or 0x00000000:
    ///   neg: if non-zero, negate the output of zeta
    ///   ctl: apply the transform only if non-zero
    #[allow(dead_code)]
    #[inline(always)]
    fn set_condzeta(&mut self, ctl: u32, neg: u32) {
        // (x, s) -> (x', s'), with:
        //    x' = (x0 + x1) + x1*u
        //    s' = (s0 + s1 + x0) + (s1 + x0 + x1)*u
        // These formulas still apply on scaled_x and scaled_s, since
        // the scaling factor is 1/sqrt(b), which is in GF(2^127).
        let (x0, x1) = self.scaled_x.to_components();
        let (s0, s1) = self.scaled_s.to_components();
        let zeta_x = GFb254::from_b127(x0 + x1, x1);
        let mut zeta_s = GFb254::from_b127(s0 + s1 + x0, s1 + x0 + x1);
        zeta_s.set_cond(&(zeta_x + zeta_s), neg);
        self.scaled_x.set_cond(&zeta_x, ctl);
        self.scaled_s.set_cond(&zeta_s, ctl);
    }

    /// Lookups a point from a window of points in affine
    /// coordinates, with sign handling (constant-time).
    #[inline(always)]
    fn lookup8(win: &[GFb254; 16], k: i8) -> Self {
        // Split k into its sign sf (0xFFFFFFFF for negative) and
        // absolute value (f).
        let sf = ((k as i32) >> 8) as u32;
        let f = ((k as u32) ^ sf).wrapping_sub(sf);

        // Get the correct coordinates for the absolute value.
        let fj = f.wrapping_sub(1);
        let mut vv = GFb254::lookup8_x2(win, fj);

        // If the absolute value was zero then we get only zeros and we
        // must adjust the s coordinate.
        let fz = ((fj as i32) >> 31) as u32;
        vv[1].set_cond(&PointAffine::NEUTRAL.scaled_s, fz);

        // Negate the point if the original index was negative.
        vv[1].set_cond(&(vv[0] + vv[1]), sf);

        Self {
            scaled_x: vv[0],
            scaled_s: vv[1],
        }
    }

    /// Lookups a point from a window of points in affine
    /// coordinates, with sign handling (constant-time).
    #[inline(always)]
    fn lookup4(win: &[GFb254; 8], k: i8) -> Self {
        // Split k into its sign sf (0xFFFFFFFF for negative) and
        // absolute value (f).
        let sf = ((k as i32) >> 8) as u32;
        let f = ((k as u32) ^ sf).wrapping_sub(sf);

        // Get the correct coordinates for the absolute value.
        let fj = f.wrapping_sub(1);
        let mut vv = GFb254::lookup4_x2(win, fj);

        // If the absolute value was zero then we get only zeros and we
        // must adjust the s coordinate.
        let fz = ((fj as i32) >> 31) as u32;
        vv[1].set_cond(&PointAffine::NEUTRAL.scaled_s, fz);

        // Negate the point if the original index was negative.
        vv[1].set_cond(&(vv[0] + vv[1]), sf);

        Self {
            scaled_x: vv[0],
            scaled_s: vv[1],
        }
    }
}

// Point i*B for i = 1 to 16, affine format (scaled_x, scaled_s)
static PRECOMP_B: [GFb254; 32] = [
    // B * 1
    GFb254::w64le(0xB6412F20326B8675, 0x657CB9F79AE29894,
                  0x3932450FF66DD010, 0x14C6F62CB2E3915E),
    GFb254::w64le(0x5FADCA04023DC896, 0x763522ADA04300F1,
                  0x206E4C1E9E07345A, 0x4F69A66A2381CA6D),
    // B * 2
    GFb254::w64le(0x415A7930D693FA8F, 0x1D78874EDF2F1CA6,
                  0xF61DEA7CDAE036F7, 0x4B30C0F5E5F279EA),
    GFb254::w64le(0xC19ED043FBD6BE01, 0x693D8F2F6ABE9465,
                  0x0F2F0D9CD452AB50, 0x19720E490A6EE21C),
    // B * 3
    GFb254::w64le(0x0BC573551889FE19, 0x665C451B1393238B,
                  0xE053B1D027CA6F4D, 0x5C27A07D34043EA7),
    GFb254::w64le(0xFE1E7723A1F56BB6, 0x7B7805107D15931D,
                  0xAE7D87EFE184E5DF, 0x0F6F5F4EF11925D5),
    // B * 4
    GFb254::w64le(0xA11DB5F206C9A0C8, 0x061309D0C72A3AB3,
                  0x91999BBEEED4F57B, 0x77F10DBDC3C0D1DA),
    GFb254::w64le(0x38EE9EC6812A13C2, 0x77FBC24A9DCA6BB5,
                  0x181DB8C3C034074B, 0x6D296D30A8E44BBD),
    // B * 5
    GFb254::w64le(0xC715B038CF1FAB5F, 0x0DA235C1610AD947,
                  0xD3AC0FF57E52B936, 0x7094DAC342EA1434),
    GFb254::w64le(0x06A589BB32462848, 0x0F8767251566BBAF,
                  0x9F808AC917C2DAAB, 0x32B14A6855FE4D2C),
    // B * 6
    GFb254::w64le(0xB210B5452FEA71F8, 0x14D11ED1921194F5,
                  0x476FF44B4E3E4518, 0x6F68AAC2007A5A24),
    GFb254::w64le(0x57BE3BF043C891FA, 0x4F28EEAF548C5D6C,
                  0x72895485E898732D, 0x5683B98CB3EB369B),
    // B * 7
    GFb254::w64le(0x1F6121CEA16EAC69, 0x19EB28FDBC02778C,
                  0x0E86728BB2803207, 0x03E9B9FCD9893789),
    GFb254::w64le(0x13DE2DAE7604ABE1, 0x5121D6B7A6611933,
                  0xAFC835F39644C754, 0x0A1F6E2DE19E6CB3),
    // B * 8
    GFb254::w64le(0xCDCB2821F80BD001, 0x4D1FCC11C02477B7,
                  0x2A6A17AF237C442C, 0x1301DB82D4D6114C),
    GFb254::w64le(0x83CF1AA244C7077A, 0x327AC316BC942DCB,
                  0xAA4C2E848D0BBFA4, 0x235DF1F92A0788B2),
    // B * 9
    GFb254::w64le(0x444147D32B7B07D2, 0x455A58853AE73AC5,
                  0xA35643E9C3143DC0, 0x2B58E48503E13B83),
    GFb254::w64le(0x88ED4A7D6F9404C9, 0x3B0D7C2C4DB7771D,
                  0xE61555B4857B56BE, 0x49E00A9CF2B0ACC8),
    // B * 10
    GFb254::w64le(0x739D6E316C22A135, 0x0B95BCFBD37F497D,
                  0x58A06533B085A0A9, 0x7EF979FB05F280EF),
    GFb254::w64le(0xF45DABD58B91BE7A, 0x43D9530172714341,
                  0x33252D1C10F42D0A, 0x6055FA3A8BAAE885),
    // B * 11
    GFb254::w64le(0x9B9E97048BA89C69, 0x1F684809930A36DC,
                  0x4C41318CFFCD063C, 0x64EB28667FD6BD8B),
    GFb254::w64le(0x90D3FD0748513CCE, 0x58C79A98F6DE8087,
                  0x654CADE0754630EC, 0x798833049A86E32D),
    // B * 12
    GFb254::w64le(0xC304DE6E55BB5B8C, 0x69A471500725B96F,
                  0x2A1B94BAA5169F8B, 0x2FA2EE3E46C1EAB2),
    GFb254::w64le(0x3678C2C98E4C81F1, 0x738C07EBAEEA7A60,
                  0x94D4021C576E6711, 0x5602BB5EEBC8003D),
    // B * 13
    GFb254::w64le(0x2E5F37C9420A1C76, 0x3A7D7C7BB2357F2F,
                  0xAFB34113A907F216, 0x3C8E95CB823230DD),
    GFb254::w64le(0xCC4F7898746279A6, 0x1AB1756EC0119FEC,
                  0xB7793E62B203CE10, 0x05E599E37D57B92B),
    // B * 14
    GFb254::w64le(0x84127C771EA031C2, 0x04BFE708B478EDB3,
                  0x37B151C13EDAF4FF, 0x0BDA56F5244B609C),
    GFb254::w64le(0x926779C226ECBFF7, 0x4CC8D8D0CC5BEFD5,
                  0xF5C39769DF3DF9BE, 0x5C6BD1BFED7A9384),
    // B * 15
    GFb254::w64le(0xC964F07A5A95E9FB, 0x220BD9620F169909,
                  0xCC1E67CABB2D3A20, 0x35E2D10E9787A5EC),
    GFb254::w64le(0x0C91CE6A452AB7DE, 0x515E70DA29F38FB7,
                  0x57357CA25C31A581, 0x790F54EFFFC32009),
    // B * 16
    GFb254::w64le(0xB7DB2F25542502B7, 0x7FA2C6414A5A33BA,
                  0x94A863D4A653DD5F, 0x7B4E3179221F8FD2),
    GFb254::w64le(0xAF32E1F83787F6B7, 0x0BFC7AE55AE7A619,
                  0x733C08179EE9B5CC, 0x48249E0F9B0A6F2C),
];

// Point i*(2^30)*B for i = 1 to 16, affine format (scaled_x, scaled_s)
static PRECOMP_B30: [GFb254; 32] = [
    // (2^30)*B * 1
    GFb254::w64le(0x7A56D2210A13763D, 0x1E542B0E5D47C05A,
                  0x3168E88573A50D88, 0x62F6BCC43FEE0180),
    GFb254::w64le(0xC444141E5801B50C, 0x780B51CC88014C52,
                  0xA4AAC5E3593032C9, 0x6060B45F6796915C),
    // (2^30)*B * 2
    GFb254::w64le(0xED7F66A6361C5516, 0x69DEF675D69B0740,
                  0x682F4F3069C20D5E, 0x79A3F98A3185576E),
    GFb254::w64le(0x283E68992835C186, 0x162A3E630AB5CE8D,
                  0x8037A08B2FB10C29, 0x700F06438EC75716),
    // (2^30)*B * 3
    GFb254::w64le(0x8835600BC01C4F28, 0x4DC82A0512F4BD3F,
                  0x96003F2F77E561E6, 0x5168EE07C04C2372),
    GFb254::w64le(0x284EC470CF54FBB0, 0x7CB1DD324C66F261,
                  0xAAAC25DE495A62B4, 0x4AF6172829D9E5D8),
    // (2^30)*B * 4
    GFb254::w64le(0x635575814DDB30B8, 0x5B61982B5030FA03,
                  0x11DFBA3C22FC0A21, 0x59B8AAF20F317C69),
    GFb254::w64le(0x24CCD3E54BA656F7, 0x75E449438F12A690,
                  0x35A7574A83593FAD, 0x605B7617D281984B),
    // (2^30)*B * 5
    GFb254::w64le(0xC5EB165C6C9594E5, 0x516184959269C502,
                  0x3B8462AD3F8492DD, 0x18D8DD1294EE066C),
    GFb254::w64le(0x9E6D5F7EAD1F3145, 0x318C94659BFC47A9,
                  0x50CB9574857351C1, 0x15C0C33C8B44A500),
    // (2^30)*B * 6
    GFb254::w64le(0x2D61BDE43AF95541, 0x32617878B8B2801B,
                  0x34FD35CAFC13BA3C, 0x5A804BA39DE9C543),
    GFb254::w64le(0x5CE4B65AA180D4FE, 0x34EBCD4AC4BA5E91,
                  0x1E0117B3264D968E, 0x6459490F49FC500B),
    // (2^30)*B * 7
    GFb254::w64le(0x6065DB6B6105DABD, 0x767311CBDA1BFCC1,
                  0xDCEBA36D41941AF5, 0x45C51A9C2052FE56),
    GFb254::w64le(0xC0D85D21AB36ABD3, 0x67F348E6F3D14493,
                  0xBC50B22467C4B8DE, 0x4BC9197D7BA1DCAA),
    // (2^30)*B * 8
    GFb254::w64le(0x90CF4E3563E928F5, 0x50074E815223D2E7,
                  0x5C404A45354B113C, 0x0FA6E6AEC8167241),
    GFb254::w64le(0xA1301F5B6DA726AA, 0x417E796A36FADE6F,
                  0x132B507CA030F951, 0x1B05958227837BD6),
    // (2^30)*B * 9
    GFb254::w64le(0x5C9BC34398AB0764, 0x45744DDAD1A49DC2,
                  0x635379A050BF4093, 0x28DF869F3FCA8B56),
    GFb254::w64le(0xE29FE6783388AC3F, 0x5136C5B1AF79BDA7,
                  0x527B529946A90F0B, 0x3581CE6C1C62EFB9),
    // (2^30)*B * 10
    GFb254::w64le(0x975A5045320A84F2, 0x60EBBA63A6956114,
                  0x6C2E83C307B17892, 0x01F721CED6A34C05),
    GFb254::w64le(0xC570EDF2D16449B1, 0x0F881B33F8779780,
                  0x668722EFEC6C7C91, 0x3045185F169B4351),
    // (2^30)*B * 11
    GFb254::w64le(0xCAE20C539BF4D07C, 0x1A57F886ED98E5E0,
                  0x7412722428A14D50, 0x4A1DF544A4FAA190),
    GFb254::w64le(0xA3E486047AB17051, 0x21D04154207897F6,
                  0x2A36E62B05BB6BC5, 0x5D0C81A78299081C),
    // (2^30)*B * 12
    GFb254::w64le(0x3EB8194BBD1848ED, 0x49233033A973E23F,
                  0x162E3AC59659B3C6, 0x55D7E164CF1B0A47),
    GFb254::w64le(0x8408AE6F50D0746F, 0x54B1EF88DA5B5D8C,
                  0xBEEF1BC0E0266218, 0x47AEBA1631BD68F4),
    // (2^30)*B * 13
    GFb254::w64le(0xE0C8D5DAC3D5948F, 0x46963075502C5AAD,
                  0x0A6A50E69F60887C, 0x31235FE2E5217346),
    GFb254::w64le(0xDC2971A02CF3DA10, 0x57B7535F9EC75E1D,
                  0xCC636FF68B8C057A, 0x5639D9E05E04C8F8),
    // (2^30)*B * 14
    GFb254::w64le(0x7899DEAD26B296BD, 0x727699A88B809EEC,
                  0xB5CBAFDA74FEF16C, 0x320B842E11CFD114),
    GFb254::w64le(0x02CB8B9D77D21590, 0x21F9599B631A0928,
                  0x0B86AD1DF6A170EA, 0x4F3A42E20172D2BE),
    // (2^30)*B * 15
    GFb254::w64le(0xA59C0DF2AC51117D, 0x32637AB5FE79FD68,
                  0xB077D3EAEB119174, 0x63998A2B9D159E74),
    GFb254::w64le(0x29218E8EC5C203F6, 0x405C960055FF5571,
                  0xFF4C0486AD5CD6F8, 0x74AD45FA3ECC2E39),
    // (2^30)*B * 16
    GFb254::w64le(0xACDCDE13FEBCA318, 0x2054A0686F23CA1C,
                  0x4FC664CE9A944830, 0x0EE627625CC70929),
    GFb254::w64le(0x10FFCF13F712C3D2, 0x7AEF8651378DADCF,
                  0x83BF078A3A88BB41, 0x6540AA59ED94CCB7),
];

// Point i*(2^65)*B for i = 1 to 16, affine format (scaled_x, scaled_s)
static PRECOMP_B65: [GFb254; 32] = [
    // (2^65)*B * 1
    GFb254::w64le(0x05704BF4F207FAC6, 0x0F16C7B1161BD3A2,
                  0x1AD76AF2870DEC6E, 0x4FB614A7D0BF2740),
    GFb254::w64le(0x45D7C01C28566D8A, 0x005002FF4077ABED,
                  0x6542A7765672D4B3, 0x04137083A98AB48D),
    // (2^65)*B * 2
    GFb254::w64le(0x23CBD429DDA5DC0B, 0x27DF09B66A5208C3,
                  0x10BCC45E8B8FF984, 0x4D7FE346205DF31F),
    GFb254::w64le(0x0CB81A89C97F02A7, 0x3C1C9D277D64DBF2,
                  0xF84A977B704354B3, 0x2C8704A6368738E4),
    // (2^65)*B * 3
    GFb254::w64le(0xCBD5E2459A07D071, 0x578067F7CE94BD91,
                  0x393D9B5722EBB7B9, 0x07F1E938F4C2C566),
    GFb254::w64le(0xAF27AF4B7ACE6FEC, 0x6DE1B7A62CE0A5CF,
                  0xD0C6FCA2633B4D64, 0x2813A2EA989F7B92),
    // (2^65)*B * 4
    GFb254::w64le(0x43086DD4CD1523B9, 0x25B6941E4CF14DC9,
                  0x0C30580B40028B29, 0x6B6816FFA4F8EDDF),
    GFb254::w64le(0xB9FFB6EF84749178, 0x16BFA2F78D83172B,
                  0xCD9F9599577E2135, 0x0B9E5031C1FB34BF),
    // (2^65)*B * 5
    GFb254::w64le(0x358328BD421DF834, 0x5611F401AB1C9E65,
                  0x460F60B7D14F18A5, 0x6F38C6BB5317F7CF),
    GFb254::w64le(0x3E8CD8A1BC4490A9, 0x2352A5D6576FEF2F,
                  0x3F9866F7CBDB2CCB, 0x6BB4712925B7B963),
    // (2^65)*B * 6
    GFb254::w64le(0x352773F55A13E056, 0x775B6CD7F9AD958F,
                  0x56A78DD6D33E733B, 0x74C3747984E07536),
    GFb254::w64le(0x2A9696958FD485F5, 0x396E58B5ECD07EAB,
                  0xAAC48F6BF3DB335B, 0x79F7353906495B14),
    // (2^65)*B * 7
    GFb254::w64le(0x614892FB61D46ABA, 0x2C21B4117E3F9489,
                  0xCA31F25261DE6AD3, 0x0F994BD01FA51D53),
    GFb254::w64le(0x70FE6DEA9E701971, 0x24D8A9E3A68BEC9A,
                  0x7AFF38546103857E, 0x6B88616CF8F59990),
    // (2^65)*B * 8
    GFb254::w64le(0xCE33CBF4468B34B4, 0x336B43EF8088F6B4,
                  0xBF3D7155590BF9AC, 0x467EE5EE7B1FB471),
    GFb254::w64le(0x155767E7DB653538, 0x6F3C38130E198094,
                  0x36DF50400EAEC1D7, 0x1249AC09FF06C86A),
    // (2^65)*B * 9
    GFb254::w64le(0xFD9EEDF23468F8B8, 0x219139455B11A8C4,
                  0x995EDCD86308D4FA, 0x30FFB38F6317E62F),
    GFb254::w64le(0xB1637F71A83274F9, 0x232C576130EE2FEF,
                  0x54E044A561EB0EBF, 0x3B3B04EC001E207A),
    // (2^65)*B * 10
    GFb254::w64le(0x3C861AD5F3A19EE1, 0x557E83E89F0BDD71,
                  0xB70BC49CE79C21DA, 0x266D488ADAB25D6E),
    GFb254::w64le(0x0458D61F5A9B2DC4, 0x476D10383890E062,
                  0x50720E8442894031, 0x71D0A50286770D8E),
    // (2^65)*B * 11
    GFb254::w64le(0xC430CC5D6E7C4889, 0x6BC6F5C76CDCF8DC,
                  0x067D2FA0F4B89533, 0x2722D3327E5E7DBD),
    GFb254::w64le(0x6BBF56E470253942, 0x483866BF0DF62089,
                  0x86BBB475F0035F12, 0x3821788E9D849934),
    // (2^65)*B * 12
    GFb254::w64le(0xEA573E74F5B99345, 0x63A807400558EA20,
                  0x0A31970DE8E74D1A, 0x631E7520058F489D),
    GFb254::w64le(0x9CA0CA9C83474217, 0x11FED7537D232344,
                  0x4AF67D2AEC370F88, 0x7887F41E98B4D64E),
    // (2^65)*B * 13
    GFb254::w64le(0xECB9652FE337EF98, 0x6F1027049D45A808,
                  0x7FC6EF7C014D455E, 0x6FA926666A7B58B4),
    GFb254::w64le(0x36F80A94CA59E87E, 0x3FBBCCF4746B33B2,
                  0xBD6C89CA31938A05, 0x053EBEC4BA8562BC),
    // (2^65)*B * 14
    GFb254::w64le(0x5F63C4A2BF4B9B86, 0x2CC4B89DFDC1F9B8,
                  0x603B31E6E027A251, 0x243C6F343E212AC7),
    GFb254::w64le(0xA57665FFEFEE3B75, 0x0E052DF063F77B28,
                  0xE02F22763906D0A7, 0x77613CE28EBB36F4),
    // (2^65)*B * 15
    GFb254::w64le(0x62456F4D7BA9B4E8, 0x3982A81AF78B6B26,
                  0x7AFBE01B4C3798E7, 0x620D3F3615843BCE),
    GFb254::w64le(0x1557321F829A7230, 0x179D109D430C908A,
                  0x2D9EF485B2BA39E6, 0x4A1A56AA589DD58D),
    // (2^65)*B * 16
    GFb254::w64le(0x7804988218AF38F3, 0x0663D7424707BDAE,
                  0x25B10DD322E37BB0, 0x42F080645F332894),
    GFb254::w64le(0x8854245FADAEAF9A, 0x5209FDAC1F0B3D0E,
                  0x1AB17A89F6DAB37D, 0x04417E929A2B83C1),
];

// Point i*(2^95)*B for i = 1 to 16, affine format (scaled_x, scaled_s)
static PRECOMP_B95: [GFb254; 32] = [
    // (2^95)*B * 1
    GFb254::w64le(0x8F59C9C28AE0ED7D, 0x2D95BCDA12F8114D,
                  0x8FF5D4DCD1FB0EC9, 0x432888FD44772C7B),
    GFb254::w64le(0x54BC518A5F2ABF58, 0x0953A61792521BAB,
                  0x0AC8F1E9E9717890, 0x34D3D70D80AB185D),
    // (2^95)*B * 2
    GFb254::w64le(0x653346E6DA88E093, 0x300022659CD13872,
                  0x65532D395F29D20B, 0x30FE4C5C7CB5DE42),
    GFb254::w64le(0x0D181FE3421D4A31, 0x35F3E72694F4D3F7,
                  0x0AB661ADDD3ED40C, 0x542B83C04F2CADE5),
    // (2^95)*B * 3
    GFb254::w64le(0x5AF2BE4FF78C86DE, 0x694F35C59E513B50,
                  0x172F417587AF524C, 0x215665C3575AF084),
    GFb254::w64le(0x9B2D01B343FCCFAB, 0x7566BC373468D0D2,
                  0x8CCE877D7DA40360, 0x5B07DD299B48D24E),
    // (2^95)*B * 4
    GFb254::w64le(0x5450A803CF11A8C7, 0x1A3EFC521DB4620C,
                  0x3FA30220B4D6810F, 0x56C042181BC8AF08),
    GFb254::w64le(0x97E3B24DFCE09354, 0x7B0F3BAFE7E9C001,
                  0x2DD1D729BD91FC40, 0x05C74680C21B1AD2),
    // (2^95)*B * 5
    GFb254::w64le(0x426D9C88E74337CD, 0x6E150B1AA9E4D273,
                  0x96B78126643F32F4, 0x5AD2EBAFC67A5ED9),
    GFb254::w64le(0xBEC8485BEADAC38A, 0x6AC7EC38FD1E9A5E,
                  0x4FFB22F4D090AD16, 0x27BD191DB1B2D42C),
    // (2^95)*B * 6
    GFb254::w64le(0x8F7A7F37431C5C00, 0x4487CC9622605514,
                  0x754A0DB2955E5D1C, 0x6AA1BE4AB8D0072A),
    GFb254::w64le(0xA6D4611F6B1BFC14, 0x003903646B2E8951,
                  0x723A689D0D536882, 0x3B33B3BD973B29AB),
    // (2^95)*B * 7
    GFb254::w64le(0xFBC0A501E981BDE1, 0x2F6A937FB8CA47E0,
                  0x6BBD3AA37F545D65, 0x6100717368B352B9),
    GFb254::w64le(0xC2B933A73DD09647, 0x4D3172829BBFF4EA,
                  0x53FDD229CDA72F35, 0x30B5BE7498B7C799),
    // (2^95)*B * 8
    GFb254::w64le(0xE2D4EE8AF4444850, 0x7C4CCD23D2D38B53,
                  0x66C8957AECC474E6, 0x702916069CF325E5),
    GFb254::w64le(0x6FEC1E66E0752CC9, 0x3E40F3D73FC42538,
                  0x5E66D9FE8A03A6D1, 0x73FDAD6877C4AEDF),
    // (2^95)*B * 9
    GFb254::w64le(0x18FFCCF71F8B1373, 0x400C9DABF9EA3B8C,
                  0xB506CD2368C99B43, 0x7F030B567FFB1422),
    GFb254::w64le(0x4378E63CE6B86A4F, 0x0A96B88FBA2034C2,
                  0xF51D6B6006E1752F, 0x22CF0CB871F4FF14),
    // (2^95)*B * 10
    GFb254::w64le(0x20505FA34F97E0A6, 0x79ACB74516909F86,
                  0xA163A5DC82094271, 0x1B6E54562F63A6BC),
    GFb254::w64le(0x9EFD3DD17E812C96, 0x6901EB6C136FD51D,
                  0x13157F6FC0488EEA, 0x67729C400270A4C0),
    // (2^95)*B * 11
    GFb254::w64le(0xA7C68204C001A764, 0x0AC00B0B708C8E94,
                  0x23A50CDB0F711893, 0x0C83F8DED755265C),
    GFb254::w64le(0xC750A647E7F1CE4A, 0x6943E20A57503CB6,
                  0x6729448938C7F4AD, 0x5FE8D54F1EBC782C),
    // (2^95)*B * 12
    GFb254::w64le(0xDBEAF734E30AA449, 0x2E1D908EB81EC506,
                  0xF261172761127B0E, 0x2DC2FA82BA512D9F),
    GFb254::w64le(0x4417289968E311D9, 0x57F6D770D5748EBC,
                  0x97723CD499E2D413, 0x283638AECC746EF0),
    // (2^95)*B * 13
    GFb254::w64le(0x5899F2FF726E75CF, 0x1CDB5D072ACECC4A,
                  0x33665414B3DB0259, 0x5409951A65581FA4),
    GFb254::w64le(0xD4242294B6214BD6, 0x473CE92F5248083D,
                  0x7911B42CEAEBD1FA, 0x73AB7FB037F5D735),
    // (2^95)*B * 14
    GFb254::w64le(0xE16BBA3D8B0BCCC7, 0x29BE1EE444C9E28F,
                  0x6E4A728A751536A3, 0x08FD01F000888F7C),
    GFb254::w64le(0x3346C2076105457B, 0x290BC8D967B0008B,
                  0xCC0E64B78C9C3D6E, 0x14197A7C2E01B797),
    // (2^95)*B * 15
    GFb254::w64le(0x2A4D9DE387F6E30D, 0x0E3C2ABF41C7EC49,
                  0x317F191F70096BA0, 0x6D5A14E904171167),
    GFb254::w64le(0x729763344C77D82D, 0x00CA6CB7F8B2D293,
                  0x9CCCBFC54493507F, 0x5EFF0C63B68C2FC1),
    // (2^95)*B * 16
    GFb254::w64le(0x891B5765F4B109E4, 0x4C341F7803AA5B0A,
                  0x7DF0A0F3B329C9A0, 0x6E637EAE55940920),
    GFb254::w64le(0x81C1B2EF7624B8A0, 0x528F805E54F22B55,
                  0x43A540E67A0FFB48, 0x7A79D0B607BE133F),
];

// ========================================================================

#[cfg(test)]
mod tests {

    use super::{Point, Scalar, PrivateKey, PublicKey};
    use sha2::{Sha256, Digest};
    use crate::blake2s::Blake2s256;
    use crate::field::GFb254;

    use core::convert::TryFrom;

    /* unused

    fn print_gf(name: &str, x: GFb254) {
        print!("{} = ", name);
        let bb = x.encode();
        for i in (0..32).rev() {
            print!("{:02x}", bb[i]);
        }
        println!();
    }

    fn print(name: &str, P: Point) {
        println!("{}:", name);
        print_gf("  X", P.X);
        print_gf("  S", P.S);
        print_gf("  Z", P.Z);
        print_gf("  T", P.T);
    }
    */

    static KAT_DECODE_OK: [&str; 21] = [
"0000000000000000000000000000000000000000000000000000000000000000",
"cbd10bd0365bcd76de1b2418d01a906c61bb948da5f84f1866f62ab301d9870f",
"6aeb610d4a16d7632c0209704e27c27adafb3825c4a446f7181b219c5a280d36",
"84bd6a2d2af05abd13433e3a5133245f2e36b5cb9d9861bf4e8cc224a7287b6a",
"4bfa00756e0b43b2e8424c971c7f930a1f8d62d792d245a82aaffe9b09004273",
"40acd94753c08aa352824049b87a211a2ffb23ebf05fd2231f5f5153da06591d",
"11dd6f132cf1c3601628b6998e7c0e2f039ef726b298662e2ec76465fba3cb4e",
"6f2110c1e88b12e750ca9cb2d7d6b044b2ee5b5c47ec56e3f867f2e486fa8c4a",
"1c94e2cce8426abc891f4066dab0245349fa07c65665d1deaae287c350644c0a",
"b479ff50dcdc3c45cae258bcf7685d6d0ec0dd6f267f9cf3211763d8b273dd68",
"09cccfe1ff69d31c38b328f26bb5b976093d9dd0d65f7921714b26989c97b559",
"40a7a912e6eec20305569fa56b01e475c2e46f8e3370877c67551424d923fe79",
"2be7959c9e0b491bf75001f261aa453166abac274f3c43b7e88614f4463bf50e",
"78fe600050c526e269e0ae75fdea027dbd32d5644ec39dd7cf42887d8c288f29",
"dd40017bf526e3c8c19b2c34c5528645458c7e7d1db6c6e8e746be53f09c2659",
"2db49a993475789b2661b944f244412f7b3fc1306664c8e96290f9c457fab724",
"b3ffaeca7b8ec292983c3de6734a636c1f0742b5a4977d2b77add7a8f61d3810",
"b581ea3d7de746cd96b29878c2e92d7909c2882c36e698916dd5be27566d8760",
"8d224a26157643bcd22d8fbc4199af4994f3dd08c41a4708050e605443adf168",
"4ddbc4e7ce2ff43f19edea4472eb754076b01062e83de82efdeb58224e39c77c",
"239f1ebb2acf00d3334d4c04df45d558a89837da3ddb48ed3b7bc488266b0d35",
    ];

    static KAT_DECODE_BAD: [&str; 20] = [
"105bf9e33fb81d01d91fa654cc6c3336737769caa64eed272c84ad26a88ece46",
"7c0abc802bb637d213220093bea30674b600e33a72fe1fc3d32153ec9416ab5a",
"53bb562569dd42fc9c19ae0f9961e95d50722cc9a2c4842a906a1f360d01db24",
"3ab2b22de7e879efa3ee5aaebc9ceb11edade9e541938f2f2a84c285e685a131",
"f34201da0c73d1575562b00bdbcb8221d93e6aef119f7a50986517788a192d7b",
"51e885251baaa8389ef82ac57fd9b029dda6a0db3a6371d8e76dc6cf36034454",
"4846d6cbb55e41a11cc70683a8221c4727a9042764add54c977690800c41340e",
"ca67d13126559b7dc7e34b1e4a1f720ad2749ffab2afccbda708658942e61637",
"bdb68995508e52fc778a6e14f3939d2506de28e4d07cc13bef265d7f0eca6d73",
"3e40adf4128a8be4e7bf78c4d6882b4f6ea22035498c4b5c431bbb96396bfc27",
"eb7502768f86f7e06dc7be9fa6aafa57fae3ad9f3f3bc0694b6b6068e7e7e562",
"336c9617613efd1316f914e4248e6045a64b900c7dddd571f2c55c3f3aa1ea4b",
"ac600877fc0c09e4212e234d17a8eb560cc26066b96c73868bbb2d93a2e80917",
"8a580f479987ff9330f4a4a1b72b217aa08c79c2d4b1020e8ce16075a6a60e46",
"99b9f8148f894acd53b4c4f4881a130a5670e47a87f49a2db1a0c10d7795af43",
"29dbe07e0ba63b0542a0b45e07d47d5e0407df4420c14db6f3c8f9a01ab1ce31",
"56a00a70550e365fc4d23c36d96d7823da084c6ef64f996178795e4d5c25e777",
"e36a1a362e8c4e6b6629c086defe720250994a4a0a859f8415a01eb206c11648",
"15c28f1ae6fb7dc3b57ff5f731f2b23e3cf6205ac442f008b49dd352e7ba4346",
"b203cdd232198a1a3cec2d9270ebc1493e88d5206e2f3b8834ea69d8d7797b29",
    ];

    static KAT_ADD: [[&str; 6]; 20] = [
        // Each group of 6 values is encodings of points:
        // P1, P2, P1+P2, 2*P1, 2*P1+P2, 2*(P1+P2)
        // (points randomly generated with Sage)
        [
"94d5f4bae9121a19c57110b50ab85a45d86768c170fa0f898b0e4514fbbadc07",
"4bc0a4701f3bd5647e737eb229e55a4ad6617fe853f6df0760682ca26ed5fb5f",
"0bc6c0f1ea55ce2d58ef439018bb3f3b87e7e469841eeb73ce62e2d4d707d253",
"73ae6e40fb9b6157048e54c94bbdb764c07c84a0fdf6dd93c25c940161340c67",
"a933ad92f379c3cf47fdfdc6623a875ce1225223e40de2448e9cfe3c520a8d1c",
"1e19f4d5341e736f1701f150e750ad2a4a4abc6459bdbefe2b16ed6beb5a9035",
        ], [
"70dafd65c8d22e882cdd2c8605836219d9ed37b86d0fd4003b05a92a89407018",
"da4587d68b0bd5be2a0f41fd05bcf61eb51f0124b158ca01a8ff2005e543b56f",
"30a544ca2ce27c50438871e0556c604194160bb66ab75af7fbe6dd584bcd530b",
"9015220008bf9c1c4729ea667e49d5240189db0ad548d868a68db0a87988800e",
"8791605ac3d93aad52bd13674769e97ea8217b1b3275f6437c74cbd3edd5e87b",
"603af6a2a915d9bad1e7bb36c659e82c0e80f437b1a4e959221749abd80c0e79",
        ], [
"a57bb7cb871acf1bcb8d9f60382e0c612e79c2e5297cf7b4e4f03b1e37ab2c40",
"32a5c7c7b38120d2a1604694b733e9510c1e50db41e6237debde0ccaf9cc6368",
"56bb66baff5fb145d84ce65eef1eab25607453048e2f03f610391796c8b48328",
"b402c4fa79de650b10f0c1aca9e96048b91b19fe3d12a556f587900332f7b767",
"0fb56df72c8f0746bcd0b4bd9b2a4e79b9a18373fb91d27b23b832094590ba03",
"9c86cd3e7dc906278d25ec8944f75d116e1df56fa25f41b74c8f0837f3c5b428",
        ], [
"d585c4caeb5ad5ee65c95516631b7e0a3d3dd3aff2b5d2e7f408573f5ad00a28",
"d4363cd4c92f46e27c3ecda7e0d3ab600a1f1dbf7a76a9c590ae551a72a73d71",
"d16d27659ad81fbe5493633d4def6d224d91d02648a8de724c6c284338acf50a",
"7e9d24c4dcad65f94952db0775eb3c31cf0277b60b08547e43fbadb92c587106",
"4aea15b4ac9a397bf656ea78c69ede3b5c35244bc1b7dde064707b96102b1574",
"0cc02b81a15368d854b3c7cc9345bc39dafe628645d1f4daddbebac1fa79c51a",
        ], [
"32bdb70caf703cbcdf35eb8e890ed61a2ec5f43cd1ed0827c773b64be0a92c7c",
"f91c194f2053d8ad90c93eb445ed6617ee9ee6bc1106a678c9038d1947811f1e",
"35d928b773cb2f19d91d23b975b1041513ce88147a89496605d75578a6434e59",
"ac971551707a68c0685213277c4386567c78f9540f816f445637ba6adce66764",
"54b00f6e8086fdf6378240201e7b244e22d4aea6f718f80836fb41003613883a",
"8c9708fe406ee39d411804ee15a43c3c02ac5b9b5351c9a7ad3b680026f4ac52",
        ], [
"bfb6790f2a4eb3aa2227ab334d97bc578a05da8b8c87887bb86bac028719222b",
"3fa757ae29e4d7f4e8fd370584af305107c088117aa17cd71bee36fe32617f33",
"39579fe6d10d6ec82922d9c16ffedf08f5e77c9224a4cadc44500b88f167687b",
"0035eeb77eb648ba8443ebe3e334df42226303b9f0eb4120ebbba2ce0e862f09",
"cff2405b060fd293c6c0291caf7bff0c6e966853ba8896be831ad73b7117907c",
"bb3545afeb4566a3e9f7dd469e6f58119fbc1eee85dbe1f2fa50cba3df9ebf20",
        ], [
"2bf8dd630dc1954a33453439fff6f75bec34ccfdb3d5e54b2b4a0aef6072b22b",
"7f3cf960972df858e1eed009a810e04528f60e304d1b7509a2f59ff0e832bf25",
"261c17ecd98cb6c39d8a945ff93bde0d2cd2912c35f8454f469cfa6ace65e944",
"826186473bca039c92aebdfa2e30a32f0acfc7318e24aa4e8cd4b325ebf7b772",
"3553dae3dd2c8431f7cc448ff78fda3902225926c66fc4525e35f8f136510c78",
"2d57fd866f7711a3b29ddfc2cc419a571fc65f25578ccb8550ae793e7ded9727",
        ], [
"f5526cf94b6360e394a73da959c5da5f8e927ac0476d3fd5a9d263e71fa4691c",
"7993d8e7edd72b4af3e70ab3e429341d8a001ae4cafd897263bca446b0eb8470",
"e63d4ef78e84128c099c886e6b151d3657f1f81d1be9398448d07be84e214c3f",
"4d47da4f69ac711afe1b6d3d8dfcbd129d8d059f3fe5fb398069bc436b146127",
"2c5e5d9be466c2c6495e5cb7b479315eda990b59ffa1b65edf3b76ff43a7a937",
"b0616ef58cfa4e0eb5c87e56e0a95123c02ca9686863cae0c65368099c0a1e11",
        ], [
"b963b2b3898944a87fbf9bfee3dab20f93157f4977ef51eec2268da43827013d",
"9486bf8c4731e85e1b05ed802d505d08f172ed797e752327bc1c9a74ba179d40",
"5cde0c1e3c74287b582d228c77274e26ee42c538cbd3bf6065ff35f7f879c222",
"91dfd1b855ea5f0f5dc5cfeba713d16d4b9275d721d76bfaf72c3b9017297076",
"d5ce2884f64694e8743e2109b3a9f76f08935b036ad1bad95fa09de5dcfce528",
"ebdec8fb0615167d3442b267f77d0a3a570faad06c3d0e33bd625f9a1e08bf73",
        ], [
"a934a7f2a91218dd50ed68c2a23f512bec9ba9c91b52e0a5eb2dafda65f02708",
"38c8ddc6f673690c7b648b8de6e0be371ff874cacc598cb3f54b3b69cecaf938",
"5f6a261d448d01e5becfe180bc4b953d0bbfd0952e266e67833ab6de6c13a568",
"5a173b80cd44291b91c92d2429f4a5549b3d2bd3852164f4ddceaa9288667559",
"e47f811f3efd31972d820f51fefff47c9c755c8a60f4e2910b712d19a1ee0d62",
"436ceea7a2673b4709c567d60535f4185ad10efaafaa48580c6e45ba5eb9cb6f",
        ], [
"e02ae0dd198ad9ba5a05d1af8888991fcbe1c6ef8ed9236edb50e494d894dd54",
"f9d68bfc6e2bea9d80a656efeb897d68724033a1f27de4a50d85de69acfc2a59",
"d2cf5623d0330576aa104a0e9ae0215fef169d6f84efddcfc824ad62749d9c1d",
"22322eb119221f4b55cef3a375185b365710ae01884415d50cf80e2438361332",
"5a6fca50f85ccd987b264b68f45ac8008438caf4e5edb531561a1f9674f44308",
"f79473d5352c991f53bcf6e3ffe13d60d01ed506b40eb81aac32316d2e632331",
        ], [
"1a953b6a305b08f979c95f2767a99e50ef6ba4a729fa46c2a9c35951210c2216",
"40586276a0a27de3231abf85685ddc256940bb31e47ec894d3d1a32033e00337",
"b1e5cfa28033947c5a8ee389ed020f1f2a5f0532650a7c73e5a4ee878f8f6335",
"e5c221a103894510687c492ce015510ab43c93dc8592fbccd748e6af724aee7f",
"96db506fbb440f9ffd3a1ee8e7cba008759f815f1746f25dc379de79e486e802",
"acb7e61f28f6a214a634b5eb7c0e313e69c57f70cc1eb6fe5a44c910fef6f80a",
        ], [
"e4a0c0347d53036b9b24d1c1ad508233362e365ea8643eb73fbba29d2965d658",
"41a140d1fccc86cf0013c01d58cba02efab5bc618eaaade9003c69fc06605447",
"075da096356d298348d54c466e2427561f8fc01621678e239fcf66027a55053e",
"03f72d0d67f65555a34d6d3fffe6d51c34744778ab642017828959ae4e70221b",
"4dc518d9990aba63584604a85aa6005f31fbb50a01f763da5f4c3016b5f3f467",
"a378cdb44a171b07b838195548c3e05e8cdc376579037b0923bb1986bd06c065",
        ], [
"4161b36bbced2460bdd8298450d28e48b293e5abf4993458057b82fb3530c20a",
"52e194460cf20f62e0979e69ded92d0bab87b2060d7cd0b23ce23afac608283e",
"0417448e0011bdabed512a517d8c731d13b0c1b623e437c68db785fffc0a7865",
"64d600a2dc1e577e628d8eb0db522f32abacddea209d7efc13f44f9825ac8674",
"0a7abe492f6de7d98cb1cfd5b7e1fe63509f3fddeee4e0ab11f4fa2036787602",
"e4af7e308f64032e1710e0c29f9c7d38a8f99b44176f6a4a5b708fb219763836",
        ], [
"1a4b218e2abbbe73a7d1d16970b4e62244a7a8f33ce0dc8a136a2ef070129953",
"230ed9c76a65e0c321c7a34ba867ea754a4315823257f3e740e69f976aa02b3f",
"c8872f2ee158bf466b121ed7df3c4f12a1c37ab43f56104af5745ba57cd8e278",
"5e2da8ac31308839d156643b1cd14c15c9c8334eff2a99a0cc01c78dd0f84841",
"6f0cfea5c7f4d8fe97551454fd1f3018b611d393a49f800024cb64332378864e",
"b8caa845e0f0cdde18426a8015980e30ce9a745b458db34804929b8d1a565111",
        ], [
"f65b443c06216f94f292db2e52ecf05f61814d9c674df7d6022ebded87d0d47b",
"07d8dfc8e5d6c0ecdcc4ad391fcf4f7b8cf95f84c7a087953f42843397a47a0a",
"e31456234385822032872a78a18c4a552d277a118048449e3500b3502a9db125",
"2dc3026ef9108702b8fdbda142e9531ea2ce60a79e718192787d77a3364b3520",
"60f4c1f37620b0ab209415804f636616d7b9161a0ec1c08b7b9b32615724f511",
"88e678d3e77794da66012ab8279858170504cfe56ca102b461cab1801cadc45c",
        ], [
"ff12c92d545b154bb01b97f73e3c9842a3459f6060d397325dcb74495b0a7050",
"6ab785216b20309fa95521bc68632273c4c2ae1ff2d23f385441b468f830cb4d",
"89339bfcbcc9bbc895cbedad3427e75755fe975e65f810a5c6af975a8062c622",
"b7c5b06ed3cef8c54c0edf36b0bd48022e9ab2e2aa7a1712602bf0be5c487b6c",
"afa79291112d4c3284487166773ca747fa019841570f852e48258c109b905911",
"d860c7a2834f3bff2981fbdc3589e81484a6382754d8952a53d1aa074c10682c",
        ], [
"6ffd71b6080088a1ac7ab66b2d60dd74fe559956e8a00310052e3bb0aaeaad0c",
"e81ffaac4cf15066633b34daef1c5e433636fb8c0524025cdb310d2ccec80453",
"e5ba686596aa3e1d9a5774b5df193c19c20dd7f70525b122ab464c9a67e88f5b",
"cfeba433d736ea12a2ad777ec899fc262fa7fcefb9b2393e7aeebc334582fd78",
"de30dc12a96279ed8a9685de2d289424147995c0b0f5214c41f8a6818504152e",
"2e705cc87d66348e31db3ca299851d03f5e7a0058e5597b1b41b8112c26eed6b",
        ], [
"950a5fce5f917cf3f32300275d814235cda77ad91c1b6b940bbab6beaa901603",
"19063ca0f6e050c2c82f0597f5e2cf5bc19e50e433f316dca8af4c23dba07707",
"68c03e3c7bde8fd0f7a6b5e0919ec66139a1b375c0b078c057205a53102e5227",
"24ae96b1446c349926e72e1514fca568870a3cdefa562b642e81123293199972",
"da50c6993bf4b0d38ec90d6567905b081897a205938e6aeb054a115c7bc2af59",
"4bac02328418fff1a9600c925de437305c58906b62a43c6c3d917ce93fcb6519",
        ], [
"5664397fc93ce68a751dd06a0276345cd7786293b96efa76512fa990e188655f",
"48377bca5a35e3be8e97bce252da010396f5986e1eaea3fcba68fec527d2a025",
"716d35f7e38791c9123310a739ba857ba44d22562dd70ede859b6991f7034747",
"7bb0bf14fd50c7e31d5f6e3b11fb41610404ac7450c68675a6a3db59f517e57f",
"36e644fe707eabbfbf6f2bd8b6367d7b87361b7eaec183746ce4dbd86d17dd73",
"1463f9cb32eda42898a02f0141ec5c5d1a000011166192f9498415ead3a43e43",
        ]
    ];

    #[test]
    fn encode_decode() {
        for i in 0..KAT_DECODE_OK.len() {
            let buf = hex::decode(KAT_DECODE_OK[i]).unwrap();
            let Q = Point::decode(&buf).unwrap();
            assert!(Q.encode()[..] == buf);
            let mut buf2 = [0u8; 32];
            buf2[..].copy_from_slice(&buf);
            buf2[15] |= 0x80;
            assert!(Point::decode(&buf2).is_none());
            buf2[31] |= 0x80;
            assert!(Point::decode(&buf2).is_none());
            buf2[15] &= 0x7F;
            assert!(Point::decode(&buf2).is_none());
            buf2[31] &= 0x7F;
            assert!(Point::decode(&buf2).unwrap().encode()[..] == buf);
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
            let T6 = S6 - R5;
            assert!(T6.equals(P2) != 0);
            assert!(T6.encode()[..] == buf2);

            let mut T = Q6;
            for j in 0..10 {
                let S = R6.xdouble(j as u32);
                assert!(T.equals(S) != 0);
                assert!(T.encode() == S.encode());
                T = T.double();
            }

            assert!((R6 + Point::NEUTRAL).encode()[..] == buf6);

            let (Q1, Q2) = P6.add_sub(&P1);
            assert!(Q1.equals(P6 + P1) == 0xFFFFFFFF);
            assert!(Q2.equals(P6 - P1) == 0xFFFFFFFF);

            let P1a = super::PointAffine {
                scaled_x: P1.X / P1.Z,
                scaled_s: P1.S / P1.Z.square(),
            };
            let P1na = Point {
                X: P1a.scaled_x,
                S: P1a.scaled_s,
                Z: GFb254::ONE,
                T: P1a.scaled_x,
            };
            assert!(P1.equals(P1na) == 0xFFFFFFFF);
            let (Q1, Q2) = P6.add_sub_affine(&P1a);
            assert!(Q1.equals(P6 + P1) == 0xFFFFFFFF);
            assert!(Q2.equals(P6 - P1) == 0xFFFFFFFF);

            assert!(Point::double_affine(&P1a).equals(P1.double()) == 0xFFFFFFFF);
        }
    }

    #[test]
    fn mulgen() {
        let sbuf = hex::decode("d2d85b649ca1cb28cf6a710ea180864b48be872c7a9585fafc01ff8259ee4e09").unwrap();
        let (s, ok) = Scalar::decode32(&sbuf);
        assert!(ok == 0xFFFFFFFF);
        let rbuf = hex::decode("6832ca87b11a5efd7718bc3cff30dc7e2fe8dd0309aa4744208c43157cc1eb46").unwrap();
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
            0x17E6D0D00F54BC93, 0x9F58BDDA363FE499,
            0x1EEFADF1FAE163FC, 0x1B8487FC89A1F614);

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

            let (n0, s0, n1, s1) = Point::split_mu_odd(&k);
            let mut k0 = Scalar::from_u128(n0);
            k0.set_cond(&-k0, s0);
            let mut k1 = Scalar::from_u128(n1);
            k1.set_cond(&-k1, s1);
            assert!(k.equals(k0 + MU * k1) == 0xFFFFFFFF);
            assert!((n0 & 1) == 1);
            assert!((n1 & 1) == 1);
        }
    }

    #[test]
    fn mul() {
        assert!(Point::BASE.encode()[..] == hex::decode("797d4a56f3e74d615aad09b2f7dd600af7f64865a867c511262181889b6cc133").unwrap());
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
            /*
            let mut Q3 = Q1;
            Q3.set_mul_alt(&s2);
            assert!(P2.equals(Q3) == 0xFFFFFFFF);
            let mut Q4 = Q1;
            Q4.set_mul_ladder(&s2);
            assert!(P2.equals(Q4) == 0xFFFFFFFF);
            */
        }

        let mut T = Point::BASE.xdouble(120);
        assert!(T.encode()[..] == hex::decode("18e08856b0ee260dd4bb2c94e52044378415677408e515f7fb22fbd6215c2a4b").unwrap());
        for _ in 0..1000 {
            let n = Scalar::decode_reduce(&T.encode());
            T *= n;
        }
        assert!(T.encode()[..] == hex::decode("4af66e2bd76b1cbdc04913cbd8b66d4e04f7935cae2ca489dd60a43b98db0f59").unwrap());
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

    #[test]
    fn mul64mu_add_mulgen() {
        let mut sh = Sha256::new();
        let mu = Scalar::decode(&hex::decode("14f6a189fc87841bfc63e1faf1adef1e99e43f36dabd589f93bc540fd0d0e617").unwrap()).unwrap();
        for i in 0..20 {
            // Build pseudorandom A, u0, u1 and v
            sh.update(((3 * i + 0) as u64).to_le_bytes());
            let v1 = sh.finalize_reset();
            sh.update(((3 * i + 1) as u64).to_le_bytes());
            let v2 = sh.finalize_reset();
            sh.update(((3 * i + 2) as u64).to_le_bytes());
            let v3 = sh.finalize_reset();
            let A = Point::mulgen(&Scalar::decode_reduce(&v1));
            let u0 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&v2[0..8]).unwrap());
            let u1 = u64::from_le_bytes(*<&[u8; 8]>::try_from(&v2[8..16]).unwrap());
            let v = Scalar::decode_reduce(&v3);

            // u = u0 + u1*mu
            let u = Scalar::from_u64(u0) + Scalar::from_u64(u1) * mu;

            // Compute u*A + v*B in two different ways; check that they
            // match.
            let R1 = u * A + Point::mulgen(&v);
            let R2 = A.mul64mu_add_mulgen_vartime(u0, u1, &v);
            assert!(R1.equals(R2) == 0xFFFFFFFF);
        }
    }

    static KAT_MAP_TO_CURVE: [[&str; 2]; 40] = [
        // Each group is: input field element, mapped point
        [
"1fd6a4420e7bbdf24133096e1528fe727a9f57c9074b61a8af111f241d6f166a",
"7e98d209635a08b896104923cc224f66442d488585b61f88b4db12be62705644",
        ], [
"033f18d29f561043f6dd852af914815d14112d278d0b97150ce81c922f0eec47",
"a8230361227b63e341ad3e02ffc10f6d3bbedba3234efd159a3670c79725c520",
        ], [
"b388dc2f0f84fd925a0e276610d7bc3d0e0c0429e365970f7c0fec67d4cfa230",
"b0ea5be5ea714172538aaa3e4e04da1d1693947e2e932dc670d0cc9bf4f6e960",
        ], [
"0bb58d35373a061187664f39b3713816454a0af5cf3e1e81b0426771e8c2cd17",
"951d4e43e8ca3e0e768f6baf44a7703afe25d81856c9b9f988d77d640ceaf067",
        ], [
"1432af36d4423cfeadc73bf9e6e7936356fe12af86531fa60478f8ccf33a3b0f",
"d43b93ab0835773c462adb334c0287062bef060ce754a3fdf86da607370f6729",
        ], [
"46b1149877793e714f2da1c7c6ee433c8787e517881786615824ea69c8b5686d",
"7bba003b2d9a085c57353e9f90b820084f4939287f89a8f5b1df3732e72d1c39",
        ], [
"4c9b6166e0a376ef53df4bcee675152a7394fdcc3dbaf71b272a8932b14c5868",
"537550590f57fe81d54d1bc63b7938023b62372da43ff9f8e78e6c83f00d4a03",
        ], [
"1e15e17915bcec56257a18a8278d6218031ef41c546ff9b6b052f3d6f7543706",
"1fd3f7b89cdb854ce3b692081beb2b64bfa13721cb6cb6770e2e49dbb504ae6d",
        ], [
"d1c4c82236756e448df547a0fd0bff200b99d02bcdc57de59c260a7c24b5145b",
"611cb59a05a96556565c5bb0195bd42866203931c53dfb9dd2d5fe9eb76c3e3f",
        ], [
"3e35e31cc1367739dcbf17bec6e4a374e92b7f202aa5d281347541084b0bdb4c",
"bd40634f54f679361ccd04373f800c489164a820e8a874881e0ed0da64640435",
        ], [
"4d5ecf0ab094a2238dfef8a0edeaa037bd0d01ad9d679ff3da19a381e7815409",
"476d54469249b6cc406ff2b8620fb409b27e616a7b4fa77b80f76bbdb81e9940",
        ], [
"2a308772b05e5f486d6a80c3c0f3e56727fe18e070343f22e42f426c34cfdf57",
"4119a979fbaf24fa16e1134109dbae4aa9c884fdd4a80b77b5746cdd968aff63",
        ], [
"8b6cdd3efd76711c9257900ed85da0061dc02f720f798f72c4cb0d5ab6c62a26",
"e12ee72b132f406105e76ff67e40f16781690fecc8a0dad4b455158486ceb416",
        ], [
"0d2481388c05ddfdb233e856e8a3e5662dac01599f23e11da629dc0eb96c5d78",
"0966447eb06aafda6cb5ffa7c5ee6d55e8afe8b7be78e124106282bc6bddd12f",
        ], [
"0c95a69895da0dfa4924ac633fe3651e91d5cc67384a460417941aaf8c634f15",
"29bd953b3a5095e7d33b54947af1303e9b738b1da00677f39ea37db939f01e10",
        ], [
"ef45b300c5a3d1a71db2b4dd989587656f7329dabbb6a9d2a5c83e35e4c2de6f",
"6b454fbaa0bad3e01b9470c5c7beea3175ed80f28ad9a75ee0bdcfeb62015e76",
        ], [
"6f782d135fb90d15f9d304280ad64063ea69b5bbe876afdd7d170dab3227de40",
"aa6e28c982fd988fd97cd49d584b3a7726f909a4d04552eda5286abf5a349219",
        ], [
"174da2cf649a6e36dab18c80304ab66eef81f1410a51a35d6f2371d2b343dc4e",
"4fb3d4d5d78eddf2dcf24bc6e6a85c2d261913d8906276fa4f395b35c9508b59",
        ], [
"280cd6ad59a47eaf0e20ce2026d28e18f61f8a0e0e058451870d5ab0811e9209",
"14070e7b73e38aa61132118eed7c746d6b9b30b48a78b811476efdf4882fdc36",
        ], [
"3a6dab01abb780f9b4072ea7179fb105d03f4c8ca24baca11de4552c64ddb21c",
"103e4b895750d4ac5d1bd300d71591052daec65c4b5d1f315d30cbf7dd01434d",
        ], [
"f0d4894b6c295b2e5a0f686e905d870587edcf4af93f2626132cf5919f9b906c",
"1dba1ba0721332105dd82bb6f46e262899becec26f0b365223fe125fbdf6511d",
        ], [
"3e5d9fe52d9eca109fe9cd93356d8a7f5ce094643827bc466bd1e6f023cc6c0a",
"04e78ea15d6be00e6f79c32a5dc7917ceb5a5037b199c893541e2aabb09a153c",
        ], [
"d19f3e961e943031f7ef78de827cd81c9647763aca9c591a3ef33bd3ba3ff213",
"14725de901c513bab21359874ec0573edabf350affafb34c0e47a748c855931e",
        ], [
"3fc852735b02b49cc818768a938c335c3f9c6c7687b7c00c5072b91b46bde47f",
"af253cc999dbdd25280a672de33d6f05f41d0aa1a6812a34ab572ffa72f1b520",
        ], [
"836b7ddbdf083910e787ae21412d931d4ad6c4371963945bee3960373b5b567d",
"a6031d1a6ae02089752972beda74e7018810b57d5bc7dc71082248fe7cd6912d",
        ], [
"91cb535f9d1801c11bb6cba4f8ca4704b1f4b319564fa492f6715d24b7b3d510",
"add8f2d0d696e48eeb1a88e8084f554562c9d732ad79e0fc11e5b319b8a3cc60",
        ], [
"d13634dfc5071664c57ad88b99587e55bd5d10dd9617c1237fa0d0fdbd326d45",
"8d81f19cf45575a02080d2b2e7a41d4e9a97eb72cc0858f42f23c1697959cf78",
        ], [
"709e769413668c944e922308bbb3ee328a1b3dd3714fb45436fa4d49245c845a",
"128965f29455b2ac8334e41f0545987b516683fadbef9211829558ee42b28d5e",
        ], [
"ed08481212de7c21f46c6e6bb403165ed5f6b6040330441fe809e30e3c92b241",
"c52b8b2f456e0a053d0505db8230c51474b9083f8006520133b5edb788b7bd77",
        ], [
"1de84b2b45e1ff0f6b2baacc833f6d5f37fc717f50569166a48225cfa427cf5e",
"4f3150b8df513df81be786cd2fc27b48bc32e88a8639eef961672771d217d34b",
        ], [
"76a9329fd3b6bad8f95fd998c1741c0a7dbf3b6d6147994407d65d69cfa0ba59",
"2b9cf6f01ac512fd8ff284b4c2698b4bd3cc41f7bc6888ab495cc31027424d59",
        ], [
"803b0bd34b259edfd95938a43f6cda58ebb1ab6c5c333b218c6ce3117cae795a",
"3b11f3f4cb26a54e63e1ac54bb230d4aaf67c3a3694cf96ebe3f929823e5e250",
        ], [
"06dc547b7dccf1aee12673a8b6339421df26c88e0e640fcdb5751949b159d73d",
"0bcc440754312b0334b89b7380bb4027e0e57f807faa11ba63d05bb8df9e0338",
        ], [
"94ff0b92dddb730f1d472de6a285595eaca1c9fa155498eacb45f8fed9f0d074",
"dc2ed058d58ec68f46e080ffef08c868f9b6dbc91acf26c4561823ec293e8573",
        ], [
"f2fbe53749eff134e6edcdeda3f55b5d8e7a94302934074c49234dd529559b4c",
"7a1c06704d6e66034959894e40559e5d61724f1c348b6fa549df90170428f15d",
        ], [
"638280f5fc76063e36089b6443e83969279201d65cb781b9ce6ccc4bbaf37d6c",
"8b8276287d923d9b11134abcecb9b6474710d4938aa0e3927dca9cb31c25df17",
        ], [
"fd460d3a94a1caac733f35f6b6e2c31e36ab3bf3a92e7fe26caad70166cef260",
"c09a92269daa9de74786543d4fbbb909ec89624558d6e76b9741de24b8f0e361",
        ], [
"a273ab53ea0ca2979bbb1854ff8c1362166b7118e152d126953c4976316ae706",
"240a36220b86b9f2fe27bf678160db0f4df5707237503dc64aaf3ed3a1919a1f",
        ], [
"9ea17dff46e618beba058f397124ec1f65ed76562a9f06c875b6e0236a9bce63",
"790e38bc001787a9e9f04e79bfa8021c07c646fa1f97094c93f60e2dfd93b670",
        ], [
"b261e5a4ecee50b861e864c68f94db34295490e22594214d914ba904abc29966",
"05a59ca022c433fc91368f98038a11004496bd817adc968dcafa54c7a673ab36",
        ]
    ];

    #[test]
    fn map_to_curve() {
        for i in 0..KAT_MAP_TO_CURVE.len() {
            let buf1 = hex::decode(KAT_MAP_TO_CURVE[i][0]).unwrap();
            let buf2 = hex::decode(KAT_MAP_TO_CURVE[i][1]).unwrap();
            let f = GFb254::decode(&buf1).unwrap();
            let Q = Point::map_to_curve(&f);
            assert!(Q.encode()[..] == buf2);
        }
    }

    static KAT_HASH1: [&str; 100] = [
        // For i = 0 to 99, hash-to-curve using as data the first i bytes
        // of the sequence 00 01 02 03 .. 62 (raw data, no hash function)
"6af795c7563d68eaad7eaee938e70e4664b4f4cb90359ca814fa8a46bda5fe4d",
"ae842d9baaf34c55a5430409de01a4135c5c019670d1a283d7c2d8e3bc2b0351",
"a1ba722ccc3ffb533446b7c1347b380e3e2ded837fdd200d1e0b9b4769751326",
"6e4c78646686f73394ea5f4e82e2a86496ded5909c5b53a8ef8507d036f01826",
"4b57080067b53784b33fd442ca7c2164a6f54e0d31e53314b112ebe641c77c5f",
"780ceee9bf7df21db01371ef5e20a9658dfa6137769bc30cb4598fd30718d218",
"dbb6969d16e809f06754d0167de9402f9ce34e3634af0b4ca24036ff582a3d3a",
"7d60c9504d87920a527301ece992111d095f39b63a21a13521449596a3eeed5e",
"d829733616b83850101d74388d385f34ba928dbe904c72d42c3d293fa36aa83d",
"ad95cbab565a826391abc7317a67146d3b09e6edb46b486fa6e77a5fbce6e077",
"bce87c2ca561a19deb9ca83af3ccae6b832b13a31b6a10049cf263fbd12f150a",
"50e2b93801bef67616bd9971bfe6110eb5ba9299d02e19633a6c1da45aad090a",
"4ca997b43491dd2954987f6d075cda1fd384ae24fc0980e98afd3bf90c107e5b",
"12da8fdb3fc09696619d422f1e607723dccbc2d5d3104a768343003e1530df36",
"e5a82a42a03e65e66f534440d0a894539ef9b80e8c0141ba920889c176056f6f",
"8e35cb5b17a579a3959a4a6a3297034c30d1805e8554f30507b4b98802703311",
"b10c604f6076c70cac209ef309978e5296ff701ab43ebd4db74ead091788b821",
"5aca4aec264cc298ababa55671b1a51c00ab232f7c86f2a1c8eb1be9d3a1a933",
"163172d96e57e3b1bc72f7387d8af659f3f031135a3afdbfd53a69d3913a054c",
"5a4c9c4ce7af71d87647696ff1076933782ba95ded63407503ff0f6747337c19",
"a441e512c81d31b4de44549cc282571cc4c648065fbd0def621a338363b2591b",
"386bf9b14f5be9d11b62a0deaa5b0c0ac6adbb3426bd7de7c8795255c62bce68",
"59c1c323353a7045b8bd5063251c4c7ad5074c8e72b7e02f520ed612a5d73e61",
"eb7bbd64e37c7868345783e99c188d65344b491c57d65afee6f05f31da0a7a63",
"bac5d33dc34f732a07bc1e64d7dda95096ae8171f5e1122630f327cef8cfc22e",
"026fd98c04a3791211c12fc4959b1b2e0be878334ef5bb719363ab3a9144f36b",
"b269ca4a71f2dbec6a34733d30de260db0debc0406c3ee8a29ca815fefd79f63",
"fd65af02ac1358f2892dea8c29b6427d4a316774a31eed9733f86ff35813f409",
"486fd0ad00fac25fe247851221a2865bfa124991c25551b9c48807df9a7cb53a",
"58e45d524f40abf97785c8cc91dac66fad4799aeb76fa1c33bb288a53003c602",
"47fb4ef3fa4632da4f849993ba4853579784ac9871ee0052b0f30cae26a2f003",
"2dcc4d1e7d1bcd78eecda3b7ee39d74769182fdb6f06781de52086086ed97a1b",
"778a77bb965f4777ada7f159c7555b2b541d1a54c062c8f3eca43e85d0a80203",
"205203b5ac6df5590a33e6c232e60d50950d0fc85cd05c242d0e43c97b1aa020",
"f8bd428335eaa272451f4ebc447fdc6d458c19a42716125302b32fbae23f2b49",
"82f8458bcb0df058351b282ed284504708722572f3b85d7b9db313a87514f321",
"cb403daa8d7e6b0e96687c0363ab2513add942b3b2bac07a0d76752d02cedb54",
"de180f7d5ef5ce4b366b64be216d7c7fa4fb25e6f77b87bbc5856421b9e05859",
"1fbd7df96ed85911f3198caeeaf546117a36b08e3961b3dd0406012326384b10",
"b75025058a8b123542b7145fd7230e328e6b70becdd2a4ff61d97f1bb0ddd07a",
"1d114823a1e9cc5866b13e594052e34914882bbc0bd3ec78f1943d3bfe62ad38",
"6748a0728604c97b5f2b9bdcf4173b2e4b3a997d7c5763292da6ae10d230772d",
"042c01ab08e74200d9bd273c1d221169bfbcdac9afa5b453df212602a24a215e",
"100950ae92efa862dae749971fc5b644c2173f6210cc3c08aea20c5ba151fd06",
"9a57ba8a59fa9de45c414796887c54078d5d0f56a1a2e7d75cbd533db691de4c",
"09183158574310c489df842ba6a3c71c07386c3e5bc98d664b2a9cbcd69b4b32",
"b74eb45ded2684c8bf42210f05fde01d11a61b72f7ac33f7a58a6b1cad47291f",
"70394ef4ee5c257b3b1696de2c1e151c63584ed79e2dd95549d07151771b874b",
"e61588f80fed7338d56bff60b0145826d01acef5597483bb18c9f3a1cdec295b",
"2ef74bb77ac7152c2006a2f58f73be553049320d69d294ed42aa4f8e21704f7e",
"95088b65d3652adb2556baa4c94dbf6fcb5ec08920c97e43b9277abc3f376067",
"02a5c4f3f4c13092920a53ce38605f706f66cf4e062a475798c71f28f880ac55",
"0869ffe7893e62eb4882d6b37e15c61e9f259d127b5cec40a962c33172c1ec0d",
"ae619274e8378942854540f92d361e114d7ad619211cf9a5015ef4b005746465",
"8986c350152febe95656f8096d5a917c9193010920f568154d8ae8b9ae516775",
"ab670c4fafcd2e61b30d914ec83702663b2b5d07281dd4fa37540ba0ce8d7f59",
"093eb91eaf5c8aba2a1710f53fe998125498d015cf3c1324a68c1fdf635f0863",
"1aec392791a31d6c06ce6f4b5333436c53a8faf10658580de8e179faf3762133",
"31a5c67b86c55f7038c57f848407624c4364d816eb991bdfca1dea4eeacf5200",
"ac87a37a2f0d6b151c987ca673b9dd792a8c3b3acaf78790dd35ed62a6823d00",
"03ecc758606cac9526dc52b9d444a467ae2dd47f5f65bd6d825738bc1db7c34a",
"bbfd4e41526acb5804f66131936772428a72d2151fc11b361f1e71cc0a4be75d",
"5853684a2056cb289e4621dd201c5c0cac05ef23228791989da581827a9db266",
"9f1e98fcb3cd67e4d0166fdc42585b68836305ce612ab00110293fc0ee3aa34a",
"8b36e7926d73751bcad84e96c05d1445f3b1fce7ebde5bdcad63e1b11ddf4705",
"8fd0fc5fc708d6e6ee23c98bc3bb7e30dc9ee047fdf9bb3ef7d18541f617f771",
"1a25afc2eb78b710c8b8dcf996e9140ce4321d2a74c1eb077b4994a32c4ee471",
"8febe0d70ef71c62186630bb010a257830261a1993e6cab96b83aaa942b1cd48",
"613f795dc36217b4f9a4ed435000514742f06bc2cb29cc8c99632e4e02127b7e",
"4b83a94bca39bb07be53d40e4b0c530ca1ae85f9b4283f8f57000a59e7eccf56",
"8dc2c41527d4cc11be582d7c29b6d55392365adeeebdb19f4e3105ce9875740c",
"dcf964e96cb3d5ef362da9d2467cee6e50396e290ab89d7c0846a7d989ab6848",
"97297f266aee72fd8b4a030f08d1762f285bd5a25a631f820a8757c844fdc744",
"b9775ba5ab5bab7e4b46719db640462e716e0c1310724900753d7247b69f7a13",
"224cbaa7bcd59547c354ce80b28b797e12d6c08ab8ad2635bf069081bad9dc43",
"d0c50cfe6125d048957ccb41e61fd8042857f03c439dcad1ccbeada8d018100b",
"ef17e5fb67c9bcababb137f553f3d5392f4861d7686bd5a8fedd57ef4d42515b",
"780ced0930f23d9eb4edc3aed931bb6a459c02c616cd31af0d58fab7109ac028",
"52f578d6efb1db0ea8530fd18803da00aad8080328c81c8868187c505e36ba72",
"03a53b75c6bc97a37333ac65aa1be747a247afc50ca4dd95e719113174da1a08",
"d4e0b7979e60631e905c315ecc3ed14cdf1b2d3d5020014cda256b71e35f9070",
"223d4bb8969b6f2fed55591a03d0211d2b0f2e5fd1bc748ac31524536c61e911",
"cb0be4b6c1c0f5a1eac6487d22f74b49db5f2d16b65fafbda86a26af6357ba0b",
"1dc6eeebe5974108f66c2b69741ab84ac3e82c311c69630453789c6d1527c420",
"0919f9499d4d9160e53bcae2266e69544533cd2fd8b551500f6a8f98246a8028",
"d20dce7ff19d2d7cea5a32138a6c212c38e0b76fa63457dbb7d3e07ffc117b50",
"79175bfdd77bd630efaa7958f8e4ed37dbe03c1aabf3bd5675d0b63d17be405c",
"c981fa60ae83e38dda0412fc16d207067d4bd6aaa0b4c19153ab5723ce80b06f",
"d65b7ddb137319bef9ab0c4540ef9962e35bb27b9186f6e7ae9aa581c6c3ec48",
"8a9986da2e304daa564f75fc3afa0d62fb7be8e68f055c7d91d9855f4824ab6c",
"32ed68aba78e6aaa37df5d3113940165d26a5264bb8f89a6bc415160e8a6af33",
"524864775d0206a2a49d6095ce01c97f65497d78d4826ec658c8c3d4e0648e2d",
"03cd9d01e3564c79f5e9051b47bcd8438f20123efa497f067d4a7a1b07001f0b",
"20c87e0b39d6a92505150dc5fea2aa5373df09faca8259a7137ee074230b145b",
"9a03cceed0e10aece557e98a41bcb9562d7ee8488bbd9352c4195b5fb2ac1431",
"01c285042be6b83819e7b6768b714f36d5629d33c65e0e13055c5fc6e978242a",
"0152b50944652f92c30f01dc1270ac158c3eb0f6f2a58d67bd0f2f570fcc654c",
"dafe815d212af1e7729ad8e43af10e129dc5bb80b57f2724d68d5d1adafacf61",
"eea07f90a65cd2991c00b70f8f132d4691df52f874937d6e632ddaea6b462134",
"7e069d51723e31f6d7e7a5fb6423121a7297b8984ad1a3cb721211f574b6e76d",
    ];

    static KAT_HASH2: [&str; 10] = [
        // For i = 0 to 9, hash-to-curve using as data the BLAKE2s hash of
        // a single byte of value i.
"89c0d5e5d8b85dc64aa01128392fde67cfdf3d134d1c1cb2454f49a7fa332d10",
"146108e34523fb073833c020c39c47695cec4e8ebc9df120b380f75ccf480266",
"2e1eb08730fb1029454cc363c0341472e38f0ad5ffd205e767cf935f1af76737",
"24416b842dcbf578b38a00b0565c8e2387d9d7391d846ed746685bb4afc3ee0b",
"e9f8bbd2dad91f2c4aca106f953a53026f3a4b0e2d04e49b279a6f45d8fb5347",
"7d57851cb89cf067255171d8c29be752a9d1b0337b3ac446ba35d9daab6e0c0f",
"dcc2b346f340cbb9e2c2e2a75c14997bcb828135a8e6996345effff4f7b89127",
"585cc121fbcf8fded35d98f232f2100bb6758d76c30ba3c9c9d0dc2aa5818a29",
"b93b56b759832e62911dc203a8f9225e150f33b6daacd22559715611b808bc29",
"f04a925c43f525efce533891141f0352b91537b1bb56f2e0740bd6695dc9e62b",
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
"1dbf78267a3c78d87e567b309ec8d053c83325b32353220aa82add4d8a77b51d",
"91cac0f40686c2e29c007f6db14e8e110708ce9c1dcbc8b57e8167c3b2c9c77e",
"",
"ec14004660d4b02da3b86b1bc5afa7b2e4827f0ee1c9a25472a2bcac521bc231",
"85787ce65a0679bed708dd2655dd7e0fe1c677c1f09690fcbcb721737b942b77fe2a1fb9e90cf7f1d0807c11435adf0d",
        ], [
"c69e2616d78bddabe8a3ca558c9e399ed2945665c96494891c46dcd8c116d103",
"891b2b8b544486b3a458159dc324ee0eb4f091eb3d23526bd917c636b28b7d69",
"11",
"bd1a4655b90f873c53fe908f4109bb8dfcd9096312b447a6434af3c35304b7d1",
"4eaca5162addfbaf45cb1009d963c904a5a1b8eb67677d41f9686ed68434493df08bf07ef7286f402242028e30d4020c",
        ], [
"0f4f28a0a11c48fb9c61fc8f5842347a613e7bcb24f0a0cbc4f9dc0e51aae917",
"3b2f0706d2c51ab5e627a1cec8246664f739020c8025c6898a00e905cca4ce37",
"5b6e",
"6230441be7f030f180e81dc44502b24ed94260490d140ae738bb80746051651e",
"885dfbae8e88f5fac429bc628c9942c432a7e2cd06ddacdec9cf3abc644d1b2f1d29deca504e9fe67d040141f50d0701",
        ], [
"78d4ab8b96ae6bd93823cf8e2f79ab9f5b4df74a54974bac023c7fdc1d162a1e",
"176963b138dd2b5e8f9c8a5f2fe50e198a65a16820fde23d9cbc505fc6c3201e",
"fe7b59",
"e877b70f8c12aff466a4dbd6284bd0c6ad7cf66376bdad599f22145f8277bc52",
"2096ce24c16aae16117fe0f03c20b27c9e51bbb48ea8527773f0b435a27d8d3d90a5c05d625ce3d4cae066684bc02a1a",
        ], [
"68cd1cfad4651bb160abef55f88710bde5ee6636b637ce4aaf027da99baa871b",
"f74c41414d731f85a1cc0d277ec0b177abcb8e22b970b9bb626952e7c400f90e",
"91d66d55",
"b4c94e55cc622b96b49fcfe6b913ce3a06050b7e9b26fe840389145088d59502",
"d4390b07858770cca8abc9915085e914048e712de6f7ec412440024defe90100921724d18a882c340b407fd99bbba21b",
        ], [
"d881adfd928d96b413e15e35a55a1a144d4fea4dfca24df5357fc276c84a3103",
"0a72c07740eeb9795310ccc7527bae56f884c24d679139ac4da1209346e8ce64",
"b768840af3",
"a7ad895209663ae35bfb3fb0e44cc83616bb876d14608e5b09c20d19f57839d4",
"2b52d00f71dad9707aeb329eef0beb0ac2d65850928c6981e048e46d40b36d5b7c9064a8e6dd06d88bd478b763bb1516",
        ], [
"3b7ff4b04edc7e95d5a5f4d75756c178a21b76c01b32375baab60d46bd608f0d",
"ec3a048a22b148d88b8c82c9d00949276e15b8246b21438b3854d5b53385c818",
"92c1c211204d",
"0bd1a3ff8506a918b8bd733c31cec084927241dda2ede63f719a6758872c94ab",
"17501cc4379c33edef42bf87a41be514903c0719c65eae67abe622d8fa3e196a31db2f430ee5c5e3a93d824eb02f7d0e",
        ], [
"d2c3599251bc3e8412f06c336136fe6a206563b128ab817f21ac2c07b6f1a412",
"976468d48c037f40997e37d74115c8647d4933f7f03d174319e64a794513e238",
"d5bfee51716f4c",
"f328909fd158f3541c2da54b758ccf750bfe4afa717b00094fd30e7fd69661e3",
"beda93626600f76a265dcb15d7611ef8f3714ee4e334b4c0f040db12e42cc229fcefb159f19927decc439a4942ea7007",
        ], [
"2e6ad8738eab9d29811feda93fbc71f87621a444514b61940892e5ef24c99518",
"15e419df2569d74c8f0491747b10f74cfa5952e2dd84611f345e26edaf31a848",
"25d55a3117fcfb1d",
"7de5f8c2c35149558c0a6bef84596669100f6350f07aefed58120d6dc3531231",
"be6d4facaff0115f886e5cbf701a6baf55af24d53fce8c6c3dbdee14b652762d1c6b97c83aa3d00682ebe31c2e4f5c18",
        ], [
"c13ff9d1170875eb7eab4f0013cd1945092917b705fc3dc8f18bc33f62443d14",
"74ada21664eb808ba4fbe1b150c61d207547e3b2b0588f732dc18127f4be7a58",
"540398f5f8ac3bd048",
"9fbcee44419bc19b97bb673d0055faa0aae1861f44c682345fb3494e610e26da",
"0606b1a812dd9163ef721de26de05e41f1a371d29dce4f123a5ac51963ab43999ae9e283194ec31736ce12e5938f4713",
        ], [
"ceb71f69091862d376bd54d86483bd8122464f6b2f6fcbb3a92f3d1604336a17",
"c0dd1b298fafb007016288f0fe19a179ba61c265502a04162bb3377e845fec48",
"f7dc92978d97e11aacbc",
"4ca14993a888660c624f816db0c893bfac69d5ddf04cced60333d94ac1b0e2f5",
"d108ca24db6dc30cd36b5eb0266ae495255851a72a078ceb2394702c959adaa0ae9bef52f534bffb93d4a53cf1febb15",
        ], [
"8dac486f7f8c7e0f8530f113e4b9a754fb6c4bc807a0eaad4f2965dcc574cf02",
"0b8b6baa03341662361e3dbecc933e28c40030cca49b8cc272609e2cf9c64656",
"282ce007c2f416cf4eff41",
"ed427029b6afbfe2a73c7a73605bfb47b4db8eadc940bddc103098a06d7b7daf",
"f15d35f52952c6c7644e66331fa5b78468b46e6e08fe0f83c4f56e0d8cb14abce98414fec58df3e2773ecba970c55101",
        ], [
"b8a52aca615df8fc7cd48a7391ff8c101494f43ef4a95c49808d754112409816",
"471ebd8a26d3857957721e5139f3117d48dc691eb4dc3f4030405721eeb1c250",
"6bce806f389db2ae12e9fd9f",
"2b083962ac0f0d9421bffdf9377f06e7152c3677e911029b08f9d40688c8aaa8",
"702ba7a045265d5e0d516e8df6ed1f6ea5399f6b07eb2adf75a23e914c8670c3ab1e44f717de3c528f7936bb888d1216",
        ], [
"0dda8526b240c1544d3c39f43bdd16984fd33dcd2fb464b8066c889daeabce1b",
"3a1ef01cb3345d888bc078a05569ce2b11ef37eb388dc6157e8e786fba08146b",
"c360afc16a5fe7e775739e7fa1",
"cf44d2ca3441b9089e99a00eb90fe161bc994990469a46b488e08711a7ba8d9e",
"71bd7242279e9f555b8bd8578b29803409fb215f81b31192a6b13552c10dfec7029297c5dd3a64aa596df0f2de01b71a",
        ], [
"45164615ada309a0564da2c6f81c798e255e491659b3199af834252990890d1a",
"e51fcd414416561f91fffe4435c3232544961897fd27bada532d0b9e702ba206",
"ece299589e82b605a1e20723de3a",
"79d41d37434fa78c4cd3fb421c7caa26704df53c215adcc4f7807adde10c7438",
"3fc0755f21a565b7a49fc6447884ffd73c10752b8b0219fb650a015c4feeabcccd70f15ba0c6a689ccf3ba1960995a06",
        ], [
"592382f4b473bd254c710ec78461d68caff55b5bd9cd71303fe5fcf8b34ef108",
"7b9ba7240bc13d361315efb641ed776d61a12c10f1bd2df16999dc2b92a75f6e",
"a541117e1b92ae3d2e5fceddcb1a58",
"0756a67df9f84be0d319c4e8d324f3b77077f9322f9603f015df27f2804b17a2",
"31a66f0e2054db3fa6a33eae6ceb8f7725fb79dcab20fd727857fb69ebfc8956c10e6b7163950410574e736a034bcb0f",
        ], [
"9b44f46bb75e7e710882365d02be0d40c211d926921629d2105a068eaa541008",
"f0cb399edcbd2d72928787e09b3cef576940c2f2be1a780831a4e299b001aa41",
"f3a9781c21ea1c8fbf65677793cc9449",
"86b36dde6d628b67332456b5d41d09737a057215f72f89094d071422e705b82e",
"fcb261760103af02ef4c2e768c6ec67049eb26878cb7bb3b17154e332e63d2335a125bbc295fcefba030634b051b9a09",
        ], [
"0ab6fa175df3ee6ee0ba766652d5583032d2f60ce8e4bc4afdde81f94b935a07",
"b1bd4d4ac2cada44053708afe4ad200b9d2cf74b2e87212365271db21948db1a",
"62e978968fca2374a00cd45788172b866a",
"86497726e18b409075f7036b1c65deacab22cf85d2ae64ef1857e17a9713e4fb",
"4cb2068f5a0fd13f7668dd0bc95af371ad95f024ffb6060c035fb251c2d2e4161cddc0e8ccf0d991446a33efec403319",
        ], [
"17b237e86dbc84537be61750d9237dadf8b779a080b7c159d25cf6a8727bea06",
"dc852b6cdaddcdf3963c96cc7511dc1a84a6a7c5e9d243f4d86fee0a36a7d726",
"5bf1bc88327a261bfeb63e7a9da4d6930cc5",
"a2edb2c979a443ff733c32453d350f09af33068a5640af90940315e7d3c87957",
"8f2784580557ffd3a7907c7eac293c8fe162231c2145987207200a52ea8d7ed6bc2ac31ff029f8776652302ba0c2c615",
        ], [
"29567c5fee0619ee539d0a835e90683f796caca4b20ea671cc8e75a47574d006",
"ebdd954c92cc91b565f11dda690bfb158f2579582a624dd6553508b4a0484a3e",
"c266e16ad55956e200d682349704f04454a79b",
"ba789f5876b8db6ae44d0e4507de9993c83e504804c1f3f8619adbd717847b77",
"fc52da48297237021dacec1c9cae7b6f99885dd5a679528f7ee5b835e8725be444a838715e75202870ace597cd5a8104",
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
"efa5335c242f3fa3460af5edbb6d4ec580805d92fbdd6a74d823ca7eced1e913",
"2d0df2615ae681cd2734bd905dad7017061b41a5559a28b60c9ec6e3ef23eb63",
"527ab0a449272ba46cfcde5080277ae686de3d9380b3b48d8018cb7478f13022",
"56dd447da86d00199a7aa9700ac6a04ce8bf17dcd63c4852a944cce18956d8df",
"3264d7a70609782179f2f503e7d8a984b96d05e316a63f4b307ac6b71f071cd4",
        ], [
"eb7d1f13254c97ed868c48b437198443d52e89776dc36d55556f391283402e13",
"4a98f953b3235351e9920e9a400ade7faa285d836e133e4f5d7583bf6b65c443",
"dd4b60dc669d857307fdc0648cc7006ba916dbcdfef98d5e6ebfbc609e31b753",
"784fdbfbfb8390d648db26ff5257ef297ad42e583cffa4097e066a554dcd68e1",
"dc29ca8aa56d3c655b3bf54fa89488eb4b6ca8530f1eadfabf0e26f5120a46f9",
        ], [
"ab0ac3d0cda9e1c8ae4b3cd8c8a1035d1b481d49bf0db3f26231222e348ad110",
"8b9259f63f63a281096e271510e87030bb0aa7ca39c1d60aaed2508316df8d4d",
"822b628932f27a2cf111a7b1c969b8546122df0f9d66f33d658c64d1ab2d3ed0",
"c1ab9fd79840127eae427a76fcecee00ae1f6c433b89da8afbe48f88f99bec18",
"bb863d6b699612d0979d738e76805217b7464fe90105263b7680365a4de0220b",
        ], [
"72c32ea104b590c07cc19ccf0cb8c37ff0b050f995b68758753b548f4c2fd016",
"cfcc2f4f284bfda1698f84f6abc0a051e83f885982d3f40da3b9f66f4fe6ce77",
"0c70336df7bfbc428ef358636795d9770e94dfc1a7f386b4ef27dfe6edc76e8c",
"015bc69f16e32d20718ec2f764fda2cfe4fa18f26ea953fe979d5a351b640f85",
"7813d018d0de012b018024e5201227f4b0c493e1f2b374738ee2949894efc941",
        ], [
"f36d7dbd24414accc953191f6a223ea786a840f7c9dae46748848aa8c861760e",
"4d73d669fa5499630a697cbc46d69c41e93f228fb7835fe32c0bc0d44536c24f",
"0224aaf6009c89e9b3b96e70df9f0194071d5776470cbbc472953d309161f267",
"917300222b564fc52e9c9de57df5ca87e2cbbd0ef8997946da0538eba3181b4f",
"22c5125e020c4f4df71504b8b10e990af8e35aef02038a86f4a768111f2398d7",
        ], [
"db200046d4c6a9582518905450aeec6986dd29af62fa9ff72b1008e154545c00",
"9aea9fa32969d4271f85474cc01ae24831ab62c4e55a2c5298732cda9d1b504a",
"627e355cf8749fc778122d03f8a96dd0830404750fec3fa860f8bf8e98bade41",
"53b112c4663a3e3518a795470dc89a315cff943406dabf5ee7029daa47016832",
"f5555105d30629bbbe326ff746f59df940c8f90e5f7d23789e70e383354fdea0",
        ], [
"3849adc00abb42c53ad88d9321b8e7a87911ca33be530d0836073c8324ff4118",
"4f09dddadd97fd62c5b042ee13b902696b8c5722c4635cf7f5a6602637ad5d63",
"f3e4b28341d0be8c46f0ab0ca078560e36467195ef9e8a965436f0b12f146def",
"623ac592961d2a91872138b1541dec4df67c4f90f47d2002e3b3ccaf511306e6",
"eef21bae28cc67a99789678c3a5d2f7530c1944d496ed874e14f19826bb53829",
        ], [
"866140149b9ea7be1b15389b8872a5fff3e21547a43dd814fb223c164d77660a",
"54f9982d97a33034e4e242413945bd7dee991b946ab5d64234a2f05ed674fe30",
"4260598f7c3f908f74feb1860625a5fe1492410ea7c6e0cfbec226fd261773a1",
"637cc59fac87980e66fe84ccb3c7ecb31694eeff78697155556a5dae6e88c0e7",
"a991edf71253568dbec60100611bc8e53e7f16186472cafe14f5ee4dfe6ca37b",
        ], [
"b65cbbf772e0a85a7fac2ce1815df97185618d1d95a1d3eeddc60ef74f25c118",
"7c515892f0d7f150a55c422274f21d7a2668f10ba17ee9cc03c39e2c894ea511",
"ee1216e0f0d71caaa3a4168bfa3630a4499dffaf42cc4cde94b16b7b216b4a4e",
"2412bb0dc99084fa88af4dc6f175bdfb3e48f1fd9737ae20d35ad7c7cf9ad7e6",
"14e58ed179e5ff113e6c0fe6421f2d8ae872cb8ab3e02357fafc34d24799167a",
        ], [
"96725c787c7c6598c065d671c7ff9ea8d7e6c5d2bf43feaf6a9a1de3b4db2717",
"c48d0a03c6f89cda616857eec962392a8473722c56d9fec6f08bbfd792b36c1c",
"4b10091e1036ddb1f69a1299a7ff938a65ba095e087118e7900f11baaa729aad",
"a5c172c8f415e4d680cd5684f2a629f260ac21694d4a812b5e9c63e93a117150",
"516ea9297a13940f75cae38bca1c8e36b6c04ff7b53d3eda885d6408f555cede",
        ], [
"93a5bc7b61b87e2c73a96711524aebb61a08c0f86f07b76f8fcdead6081ce50c",
"747f23e58a49cacac7ee2235e533f1676ae6b0fba3ecec66ec2fb1135008af68",
"4b33a2006c0738463a08c2711a5777ffc1f4a0e0ceaf1c021459ed1334834852",
"f09821ee19466e5d3fcd3e71b94084dd2a5e2ab2edf7ff5355983f2ea1c5fd07",
"47a621d21f7e29885ba221aabde5f27a103f424782231d2f28a1c7653fc876f3",
        ], [
"7aba07d7b0eb9b91942c34d4b7d3109c169e16942bd8a10486bd91ae06385e05",
"a5dce975971eb9a64003a540a6442f168919369ecdf04ae55b493a60c0b0b106",
"8fa93f3e6a34d25fb9f0747c2684b54daebaa28b93d08da52f910c52f349609a",
"aaa53549035e84af4b0dd9b9bf7ba124f619b7ee717755ce0e65fd08c8c32a96",
"233836eb88365a170b5590b5b3fcde4d95afb9916a0dc34e8db685a80f933f02",
        ], [
"47aea7b0b75fd88ba8d25c2b5f2b5a5bfcb563fdde482ef17f9dc8908b649906",
"0648d22d5fe5b96da305b049c2f6d076109fff5779555660f9bad7b1d9f4eb13",
"2c6bcff0f28cb3238a6de3d21d256a714f327d65c834686dd013b200e4c74a3f",
"0b24b3a777bd9281c29e37ebe395f33484b80355ee528539d752ba859d4ca94c",
"4e4d9126698a5876a86f020dd297243a4464bb82dd350e47ce4f5f2ce3efbcfe",
        ], [
"e223af2985452c4873e22a8054c3db51d48841cd6888dcbcd953cdba4345c005",
"270e8959897e231cccb7efa6bcd9341e5094e0f589d706b30d56ec6a7d6d494c",
"008dd7dff93cd282b562fc7ee1f2e0f2c540b9a72c9d294bb261abd6bd1c92da",
"7622faa5152516defc8df28b8c9c891bb3a20e06b6b255ed2e569e76521ee69f",
"17777e61af7fd8c97c981b2f8731f7a153b4d98ae8185cbb68cfc15e1cb2117e",
        ], [
"333bfdb81bcf8fa7913263ed1d80533dfebc774c81c70dd285a69fbbf0508510",
"c74aed16a68de4a045dcf9d111c87a041540c12cca1eab8748d4d4bd552ab701",
"a495d9953f8c751191d9bccf8a25340d004b076a83f4d78c1e9a804518014a7e",
"fcb135f686469182f8be6b771edcab95b9948abb16de7c972fce8edf5d709282",
"4585b322626b6fca3767eccb347f1b47b06431a027d898bd668caed022fd43a3",
        ], [
"95cb2548d0205ba7141711002f1d331494c5a556800b31619e794139bd482605",
"89fb4784ca2fd64308b8bfc5de0b9b3b28e3fd3bd809e92e3974508b762acb21",
"54e6e8cd15c76989965526ed1f5c1f152395cf3291887b788f9e9e32bf02bc8d",
"b86e4019bd672905f50e0f21050aaed117e65e475b9f465f449f91020dca90cd",
"e222e92d4c16c0dc8cd662891e35a8c2b286cc9f581e46e29f55215192ef2cf0",
        ], [
"ade45d8a677bf6e66c4326360f09db3ac0cba03c5a7b80d8688d4a03b799760b",
"98363d6f2c3deae404863fc2b43698244e626587ee11b34dd73907379bf7f153",
"73af72e90d985a53ddbd71c0bdd7cda3994590f08ff2032375021b201b5d7d62",
"6d4ddb5037a1df6b5d8dc214a205b3104c4487283739130700226352ee4d4197",
"f82855d6a49d11aef755cc26c843ecbcb0267ea6f0643ea387c62cfb7d1a7e24",
        ], [
"2d9c4afe9d71deb4694fd00d5cb74c61d59bfdcefabfaa78c6b8bb5fc1fd831a",
"4405a2f934eaf30dc1dc6f665844a72c179ffab7098042942e32914786e04820",
"c735033bc49ecaf07e8936930cc853460a624c1026005b01f0907f08c1e447f9",
"aa3a8b3048358a6ee8bc63462a15df75ff8c7f8bb77566d4a14cdb39d44ad3f5",
"1ba9ccf41c40c334dfb3229670957e89b857770814d82c81053a9cfb916e645a",
        ], [
"b799b16aa1c859a5baecee1a32bdaaa6447e8a78c781fe257b57ed08b6d29703",
"85d24ee9425ae5bbbfb5f972e7212741a4baff12814d1c15257a4d33df552963",
"e65696b330ae269a21dd38355b3b0e0e8ba063b1b67af2f6140f5ea60e42e926",
"66110cad99421b884a8a981f14545b6c3f51d1cb71f0dd0bb032b4a7a043bd8a",
"6430c21a8fb58d23313e5f2adfbaa8610d4e9bf6ba6cd37a59ae6e74e796a0e9",
        ], [
"bff62ebd0cf6603ce6dbf09d1fed616db866d024a77260e78e4eac3445595e08",
"20f70a98b6467a7d27a04ed34dc0db6bf7771b722ded7f96b7995768f9c6f925",
"0b7aa162b8a7ca9d78da1dfeea1dd3b2c0cdcd088746a054c51f7f1ac5e5abb2",
"4a88e940513e625a8519517ecbdd70914cfc3dcd38914086a34c0c9e2b1f7a57",
"b4c1a5e6a04eb51cb1fe10a1d109cc1aae1a2ee0143d437e9038374ee9299ccf",
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

    #[cfg(feature = "gls254bench")]
    #[test]
    fn for_benchmarks_only() {
        let mut sh = Sha256::new();
        for i in 0..20 {
            // Build pseudorandom private keys
            sh.update(((2 * i + 0) as u64).to_le_bytes());
            let v1 = sh.finalize_reset();
            sh.update(((2 * i + 1) as u64).to_le_bytes());
            let v2 = sh.finalize_reset();
            let k1 = Scalar::decode_reduce(&v1);
            let k2 = Scalar::decode_reduce(&v2);
            let sk1 = k1.encode();
            let sk2 = k2.encode();

            let P1 = Point::mulgen(&k1);
            let P2 = Point::mulgen(&k2);
            let P1a = P1.to_affine();
            let mut pp1 = [0u8; 64];
            pp1[..32].copy_from_slice(&P1a.scaled_x.encode());
            pp1[32..].copy_from_slice(&P1a.scaled_s.encode());
            let P2a = P2.to_affine();
            let mut pp2 = [0u8; 64];
            pp2[..32].copy_from_slice(&P2a.scaled_x.encode());
            pp2[32..].copy_from_slice(&P2a.scaled_s.encode());

            let Qa = (P1 * k2).to_affine();
            let mut q = [0u8; 64];
            q[..32].copy_from_slice(&Qa.scaled_x.encode());
            q[32..].copy_from_slice(&Qa.scaled_s.encode());
            assert!(Point::for_benchmarks_only_1dt_3(&pp1, &sk2).unwrap() == q);
            assert!(Point::for_benchmarks_only_1dt_3(&pp2, &sk1).unwrap() == q);
            assert!(Point::for_benchmarks_only_1dt_4(&pp1, &sk2).unwrap() == q);
            assert!(Point::for_benchmarks_only_1dt_4(&pp2, &sk1).unwrap() == q);
            assert!(Point::for_benchmarks_only_1dt_5(&pp1, &sk2).unwrap() == q);
            assert!(Point::for_benchmarks_only_1dt_5(&pp2, &sk1).unwrap() == q);
            assert!(Point::for_benchmarks_only_2dt_2(&pp1, &sk2).unwrap() == q);
            assert!(Point::for_benchmarks_only_2dt_2(&pp2, &sk1).unwrap() == q);
            assert!(Point::for_benchmarks_only_2dt_3(&pp1, &sk2).unwrap() == q);
            assert!(Point::for_benchmarks_only_2dt_3(&pp2, &sk1).unwrap() == q);
        }
    }
}
