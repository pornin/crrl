//! Edwards448 curve implementation.
//!
//! This module implements generic group operations on the Edwards curve
//! of equation `x^2 + y^2 = 1 + d*x^2*y^2`, over the finite field
//! GF(2^448 - 2^224 - 1), for the constant `d` = -39081. This curve is
//! described in [RFC 7748]. The signature algorithm Ed448, which operates
//! on that curve, is described in [RFC 8032].
//!
//! The curve has order `4*L` for a given prime integer `L` (which is
//! slightly lower than 2^446). A conventional base point is defined,
//! that generates the subgroup of order `L`. Points used in public keys
//! and signatures are normally part of that subgroup, though this is not
//! verified (the function `Point::is_in_subgroup()` can be used to
//! validate that a given point is in that subgroup).
//!
//! A curve point is represented by the `Point` structure. `Point`
//! instances can be used in additions and subtractions with the usual
//! `+` and `-` operators; all combinations of raw values and references
//! are accepted, as well as compound assignment operators `+=` and `-=`.
//! Specialized functions are available, in particular for point doubling
//! (`Point::double()`) and for sequences of successive doublings
//! (`Point::xdouble()`), the latter using some extra optimizations.
//! Multiplication by an integer (`u64` type) or a scalar (`Scalar`
//! structure) is also accepted, using the `*` and `*=` operators.
//! Scalars are integers modulo `L`. The `Scalar` structure represents
//! such a value; it implements all usual arithmetic operators (`+`, `-`,
//! `*` and `/`, as well as `+=`, `-=`, `*=` and `/=`).
//!
//! Scalars can be encoded over 56 bytes (using unsigned little-endian
//! convention) and decoded back. Encoding is always canonical, and
//! decoding always verifies that the value is indeed in the canonical
//! range (note: within the context of Ed448, the second half of a
//! signature is an encoded scalar but is represented over 57 bytes, not
//! 56, as per the specification in RFC 8032; the last byte of the
//! signature is then always zero).
//!
//! Points can be encoded over 57 bytes, and decoded back. As with
//! scalars, encoding is always canonical, and verified upon decoding.
//!
//! The `PrivateKey` structure represents a private key for the Ed448
//! signature algorithm. It is instantiated from a 57-byte seed; the seed
//! MUST have been generated with a cryptographically secure generator
//! (this library does not include provisions for this generation step).
//! Following [RFC 8032], the seed is derived into a secret scalar, and
//! an extra private value used for deterministic signature generation.
//! The private key allows signature generation with the Ed448
//! and Ed448ph variants (in the latter, the pre-hashed message is provided
//! by the caller). The `PublicKey` structure represents a public key, i.e.
//! a curve point (and its 57-byte encoding as an additional field).
//! Signature verification functions are available on `PublicKey`, again
//! for Ed448 and Ed448ph.
//!
//! # Ed448 Edge Cases
//!
//! Like Ed25519, there exist multiple variants of Ed448 implementations,
//! with regard to the handling of some edge cases, e.g. non-canonical
//! inputs. This implementation follows the strict RFC 8032 rules as
//! follows:
//!
//!   - Canonical encoding of both points and scalars is enforced.
//!   Non-canonical encodings are rejected.
//!
//!   - The cofactored verification equation is used (i.e. including the
//!   multiplication by 4).
//!
//!   - Points outside of the subgroup of order `L`, including low-order
//!   points and the identity point, are accepted both for the `R`
//!   component of the signatures, and for public keys.
//!
//!   - The `S` component of a signature is accepted as long as it is
//!   canonically encoded (i.e. in the 0 to `L-1` range). Zero is
//!   accepted. The full 57 bytes are used: the last byte, and the top
//!   two bits of the second to last byte, though always of value 0, are
//!   checked.

// Projective/fractional coordinates traditionally use uppercase letters,
// using lowercase only for affine coordinates.
#![allow(non_snake_case)]

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::convert::TryFrom;
use super::field::GF448;
use sha3::{Shake256, digest::{Update, ExtendableOutputReset, XofReader}};
use super::{CryptoRng, RngCore};
use crate::backend::define_gfgen;
use crate::backend::define_gfgen_tests;

/// A point on the Edwards curve edwards448.
#[derive(Clone, Copy, Debug)]
pub struct Point {
    // We use projective coordinates, as suggested by RFC 8032.
    //
    // TODO: investigate extended coordinate + formulas from:
    //   https://eprint.iacr.org/2008/522
    // (section 3.1). For a non-twisted Edwards curve, these yield a cost
    // in 9M, which is better than the 10M+1S of projective coordinates.
    // However, it also raises the doubling cost from 3M+4S to 4M+4S,
    // though we can save some multiplications when doing several doublings
    // in a row. So we get this:
    //
    //                 projective   extended
    //   add             10M+1S        9M
    //   double           3M+4S        4M+4S
    //   n*double      n*(3M+4S)    n*(3M+4S)+1M
    //   size (bytes)      168          224
    //   size affine       112          168 (Duif)
    //
    // In general, computations can be described as alternating normal
    // additions and sequences of doublings; there may be several successive
    // general additions, then one sequence of doublings. We can thus
    // account for the "+1M" in a sequence of doublings as being part of
    // the add that preceeded it, so (somehow) this would make doublings
    // at the same cost for both representations, and general adds at
    // 10M for extended coordinates, vs 10M+1S for projective.
    //
    // An extra variant of the above is to switch to an isogenous twisted
    // Edwards curve, that might lower the point addition cost to 8M,
    // but the isogeny is not a bijection, so there is some fiddling to
    // account for the "low-order contribution", and the trick might not
    // be applicable generically, though it can help with some controlled
    // operations (e.g. multiplication by a scalar, or as backend for the
    // decaf448 group).
    //
    // On the other hand, extended coordinates use more RAM/registers, so
    // the precomputed tables of constants are 50% larger, constant-time
    // table lookups are more expensive, and register pressure is larger
    // as well. Each field element is alrady 7 registers (on 64-bit systems),
    // this is already a lot. Thus, it is unclear whether, _in practice_,
    // extended coordinates are better here.
    pub(crate) X: GF448,
    pub(crate) Y: GF448,
    pub(crate) Z: GF448,
}

// Scalars are integer modulo the prime L = 2^446 -
// 13818066809895115352007386748515426880336692474882178609894547503885.
//
// The complete curve contains 4*L points. The conventional base point
// generates a subgroup of order exactly L.
struct ScalarParams;
impl ScalarParams {

    const MODULUS: [u64; 7] = [
        0x2378C292AB5844F3,
        0x216CC2728DC58F55,
        0xC44EDB49AED63690,
        0xFFFFFFFF7CCA23E9,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x3FFFFFFFFFFFFFFF,
    ];
}
define_gfgen!(Scalar, ScalarParams, scalarmod, true);
define_gfgen_tests!(Scalar, 2, tests_scalarmod);

impl Point {

    /// The group neutral (identity point) in the curve.
    ///
    /// Affine coordinates of the neutral are (0,1).
    pub const NEUTRAL: Self = Self {
        X: GF448::ZERO,
        Y: GF448::ONE,
        Z: GF448::ONE,
    };

    /* unused
    /// The point of order 2 on the curve.
    ///
    /// Affine coordinate of this point are (0,-1).
    const ORDER2: Self = Self {
        X: GF448::ZERO,
        Y: GF448::MINUS_ONE,
        Z: GF448::ONE,
    };
    */

    /// The conventional base point in the curve.
    ///
    /// This point generates the subgroup of prime order L (integers
    /// modulo L are represented by the `Scalar` type).
    pub const BASE: Self = Self {
        X: GF448::w64be([
            0x4F1970C66BED0DED,
            0x221D15A622BF36DA,
            0x9E146570470F1767,
            0xEA6DE324A3D3A464,
            0x12AE1AF72AB66511,
            0x433B80E18B00938E,
            0x2626A82BC70CC05E,
        ]),
        Y: GF448::w64be([
            0x693F46716EB6BC24,
            0x8876203756C9C762,
            0x4BEA73736CA39840,
            0x87789C1E05A0C2D7,
            0x3AD3FF1CE67C39C4,
            0xFDBD132C4ED7C8AD,
            0x9808795BF230FA14,
        ]),
        Z: GF448::ONE,
    };

    /// Curve equation parameter is d = -39081; this is -d = +39081.
    pub(crate) const MINUS_D: u32 = 39081;

    /// Tries to decode a point from bytes.
    ///
    /// If the source slice has not length exactly 57 bytes, then
    /// decoding fails. If the source bytes are not a valid, canonical
    /// encoding of a curve point, then decoding fails. On success,
    /// 0xFFFFFFFF is returned; on failure, 0x00000000 is returned. On
    /// failure, this point is set to the neutral.
    ///
    /// If the source length is exactly 56 bytes, then the decoding
    /// outcome (success or failure) should remain hidden from
    /// timing-based side channels.
    pub fn set_decode(&mut self, buf: &[u8]) -> u32 {
        // Reference: RFC 8032, section 5.2.3.

        // Check input length.
        if buf.len() != 57 {
            *self = Self::NEUTRAL;
            return 0;
        }

        // The sign-of-x is the top bit of byte 56.
        let sign_x = buf[56] >> 7;

        // Decode y. It uses bytes 0 to 55.
        let (mut y, mut r) = GF448::decode_ct(&buf[..56]);

        // If any of bits 0 to 6 of byte 56 is non-zero, then this is
        // an incorrect encoding.
        r &= ((((buf[56] as i32) & 0x7F) - 1) >> 8) as u32;

        // u = y^2 - 1
        // v = d*y^2 - 1
        let y2 = y.square();
        let u = y2 - GF448::ONE;
        let v = -y2.mul_small(Self::MINUS_D) - GF448::ONE;

        // s = u^3*v
        // t = u^5*v^3
        let uv = u * v;
        let s = uv * u.square();
        let t = s * uv.square();

        // x = s*t^((p-3)/4)  (candidate)
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
        let mut x = t223.xsquare(223) * t222 * s;

        // If x*v^2 == u, then x is correct; otherwise, there is no solution.
        r &= (v * x.square()).equals(u);

        // Normalize x so that its "sign" (least significant bit) matches
        // the sign bit provided in the encoding. Note that if x is zero,
        // then both x and -x has a least significant bit equal to zero,
        // and if the provided sign bit is 1 then this is an error.
        let nx = (((x.encode()[0] & 0x01) ^ sign_x) as u32).wrapping_neg();
        r &= !(x.iszero() & nx);
        x.set_cond(&-x, nx);

        // If the process failed at any point, then we set (x,y) to (0,1)
        // (which is the neutral point).
        x.set_cond(&GF448::ZERO, !r);
        y.set_cond(&GF448::ONE, !r);

        // We got the point in affine coordinates, hence Z = 1.
        self.X = x;
        self.Y = y;
        self.Z = GF448::ONE;
        r
    }

    /// Tries to decode a point from some bytes.
    ///
    /// Decoding succeeds only if the source slice has length exactly 57
    /// bytes, and contains the canonical encoding of a valid curve
    /// point. Since this method returns an `Option<Point>`, it
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

    /// Encodes this point into exactly 57 bytes.
    ///
    /// Encoding is always canonical.
    pub fn encode(self) -> [u8; 57] {
        let iZ = GF448::ONE / self.Z;
        let (x, y) = (self.X * iZ, self.Y * iZ);
        let mut bb = [0u8; 57];
        bb[..56].copy_from_slice(&y.encode());
        bb[56] |= x.encode()[0] << 7;
        bb
    }

    /// Creates a point by converting a point in affine coordinates.
    fn from_affine(P: &PointAffine) -> Self {
        Self { X: P.x, Y: P.y, Z: GF448::ONE }
    }

    /// Adds another point (`rhs`) to this point.
    fn set_add(&mut self, rhs: &Self) {
        let (X1, Y1, Z1) = (&self.X, &self.Y, &self.Z);
        let (X2, Y2, Z2) = (&rhs.X, &rhs.Y, &rhs.Z);

        // Formulas from RFC 8032, section 5.2.4.
        // Since d < 0, we multiply by -d, which negates the sign of E;
        // that reversed sign is taken into account in the formulas
        // for F and G.
        let A = Z1 * Z2;
        let B = A.square();
        let C = X1 * X2;
        let D = Y1 * Y2;
        let E = (C * D).mul_small(Self::MINUS_D);
        let F = B + E;
        let G = B - E;
        let H = (X1 + Y1) * (X2 + Y2);
        self.X = A * F * (H - C - D);
        self.Y = A * G * (D - C);
        self.Z = F * G;
    }

    /// Specialized point addition when the other operand is in
    /// affine coordinates.
    fn set_add_affine(&mut self, rhs: &PointAffine) {
        let (X1, Y1, Z1) = (&self.X, &self.Y, &self.Z);
        let (X2, Y2) = (&rhs.x, &rhs.y);

        // Formulas from RFC 8032, section 5.2.4 (with Z2 = 1).
        // Same remark as in set_add() for the sign of E.
        let A = Z1;
        let B = A.square();
        let C = X1 * X2;
        let D = Y1 * Y2;
        let E = (C * D).mul_small(Self::MINUS_D);
        let F = B + E;
        let G = B - E;
        let H = (X1 + Y1) * (X2 + Y2);
        self.X = A * F * (H - C - D);
        self.Y = A * G * (D - C);
        self.Z = F * G;
    }

    /// Specialized point subtraction when the other operand is in
    /// affine coordinates.
    fn set_sub_affine(&mut self, rhs: &PointAffine) {
        self.set_add_affine(&PointAffine { x: -rhs.x, y: rhs.y });
    }

    /// Doubles this point (in place).
    pub fn set_double(&mut self) {
        let (X1, Y1, Z1) = (&self.X, &self.Y, &self.Z);

        // Formulas from RFC 8032, section 5.2.4.
        let B = (X1 + Y1).square();
        let C = X1.square();
        let D = Y1.square();
        let E = C + D;
        let H = Z1.square();
        let J = E - H.mul2();
        self.X = (B - E) * J;
        self.Y = E * (C - D);
        self.Z = E * J;
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
        for _ in 0..n {
            self.set_double();
        }
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
        self.X.set_neg();
    }

    /// Subtract another point (`rhs`) from this point.
    #[inline(always)]
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
        (self.X * rhs.Z).equals(rhs.X * self.Z)
        & (self.Y * rhs.Z).equals(rhs.Y * self.Z)
    }

    /// Tests whether this point is the neutral (identity point on the
    /// curve).
    ///
    /// Returned value is 0xFFFFFFFF for the neutral, 0x00000000
    /// otherwise.
    #[inline(always)]
    pub fn isneutral(self) -> u32 {
        // The neutral is the only point with y == 1.
        self.Y.equals(self.Z)
    }

    /// Tests whether this point is a low-order point, i.e. a point of
    /// order 1, 2 or 4.
    ///
    /// Returned value is 0xFFFFFFFF for a low order point, 0x00000000
    /// otherwise. The curve neutral point (group identity) is a
    /// low-order point.
    pub fn has_low_order(self) -> u32 {
        // There are exactly four points of low order on the curve:
        //   (0, 1) has order 1 (neutral point)
        //   (0, -1) has order 2
        //   (1, 0) and (-1, 0) have order 4
        // These are exactly the curve points (x, y) with x = 0 or y = 0;
        // for all other points, x and y are both non-zero.
        self.X.iszero() | self.Y.iszero()
    }

    /// Tests whether this point is in the proper subgroup of order `L`.
    ///
    /// Returned value is 0xFFFFFFFF for a point in the prime-order
    /// subgroup, 0x00000000 otherwise. Note that the curve neutral point
    /// (group identity) is also part of that subgroup.
    ///
    /// This function is relatively expensive (about 40% of the cost of
    /// multiplying a random point with a scalar).
    pub fn is_in_subgroup(self) -> u32 {
        // Reference: https://eprint.iacr.org/2022/1164
        // Curve equation is: a*x^2 + y^2 = 1 + d*x^2*y^2
        //
        // We suppose that the point is not of low order (this case is
        // handled at the end of the function). We map to curve:
        //   u*w^2 = u^2 + A*u + B
        // with:
        //   A = 2*(a + d)
        //   B = (a - d)^2
        //
        // To compute point halvings, we consider the following curves.
        // We write "Curve(A,B)" to designate the curve of equation
        // u*w^2 = u^2 + A*u + B. For some given A and B, we also
        // define:
        //   Ap = -2*A
        //   Bp = A^2 - 4*B
        //   As = -2*Ap = 4*A
        //   Bs = Ap^2 - 4*Bp = 16*B
        // We then define these functions:
        //   psi1: Curve(A,B)   --> Curve(Ap,Bp)
        //         (u, w)       |-> (w^2, -(u - B/u)/w)
        //   psi2: Curve(Ap,Bp) --> Curve(As,Bs)
        //         (u, w)       |-> (w^2, -(u - Bp/u)/w)
        //   iso:  Curve(As,Bs) --> Curve(A,B)
        //         (u, w)       |-> (u/4, w/2)
        // iso() is an isomorphism (i.e. Curve(As,Bs) is isomorphic to
        // Curve(A,B)). psi1() and psi2() are isogenies; they are obtained
        // by applying Vélu's formulas over the subgroup {inf,(0,0)} (i.e.
        // the subgroup generated by the 2-torsion point (0,0)). Thus, for
        // any point P in Curve(A,B), we have:
        //   2*P = iso(psi2(psi1(P)))
        // To halve a point, we thus only need to invert iso, psi2 and psi1,
        // in that order.

        // a - d
        const A_MINUS_D: u32 = 39082;

        // MINUS_A0 contains -A = -2*(a + d)
        const MINUS_A0: u32 = 78160;

        // AP0 contains Ap = -2*A
        const AP0: u32 = 156320;

        // SQRT_MINUS_BP0 constains sqrt(-Bp)
        const SQRT_MINUS_BP0: GF448 = GF448::w64be([
            0x749A7410536C225F, 0x1025CA374176557D,
            0x7839611D691CAAD2, 0x6D74A1FCA5CFAD15,
            0xF196642C0A4484B6, 0x7F321025577CC6B5,
            0xA6F443C2EAA36327,
        ]);

        // 1. Map to the Weierstraß curve.
        //   u = (a - d)*(1 + y)/(1 - y)
        //   w = 2/x
        // We switch to the isomorphic curve with e = X*(Z - Y), so that
        // we get division-less expressions:
        //   u = (a - d)*(Z + Y)*(Z - Y)*X^2
        //   w = 2*Z*(Z - Y)
        let mut e = self.X * (self.Z - self.Y);
        let mut u = (self.Z + self.Y).mul_small(A_MINUS_D) * self.X * e;
        let mut w = (self.Z * (self.Z - self.Y)).mul2();

        // We are now on curve Curve(A*e^2, B*e^4).
        // We try to halve the point.

        let mut ok = 0xFFFFFFFFu32;

        // Inverse iso().
        // We get (us, ws) in curve(As*e^2, Bs*e^4).
        let us = u.mul4();
        let ws = w.mul2();

        // Inverse psi2(). This works if and only if us is a square.
        // If us is not a square, then the point cannot be halved, and
        // therefore cannot be in the prime-order subgroup.
        let (mut wp, cc) = us.sqrt();
        ok &= cc;
        let up = (us - e.square().mul_small(AP0) - wp * ws).half();

        // We now have (if ok != 0 and the point is not low order):
        //   up*wp^2 = up^2 + up*Ap*e^2 + Bp*e^4

        // Inverse psi1().
        // If up is a square, then formulas are:
        //   w = sqrt(up)
        //   u = (w^2 - A*e^2 - w*wp)/2
        // If up happens not to be a square, then sqrt(up) fails. We
        // use sqrt_ext(), which returns sqrt(-up) in that case (since we
        // are in a finite field GF(q) with q = 3 mod 4, if up is not
        // a square then -up must be a square). In such a case, we should
        // replace (up,wp) with ((Bp*e^4)/up, -wp). To avoid the division,
        // we instead switch to an isomorphic curve; namely:
        //   up2 = Bp*(e^4)*up
        //   wp2 = -wp*up
        //   e2 = e*up
        // Then:
        //   w = sqrt(up2) = sqrt(-up)*sqrt(-Bp)*(e^2)
        //   u = (w^2 - A*e^2 - w*wp)/2
        //
        // The square root of -Bp is a precomputed constant, and sqrt(-up)
        // is what sqrt_ext() returned when up is not a square.
        let (tt, cc) = up.sqrt_ext();
        w = tt;
        wp.set_cond(&-(wp * up), !cc);
        w.set_cond(&(w * SQRT_MINUS_BP0 * e.square()), !cc);
        e.set_cond(&(e * up), !cc);
        u = (w.square() + e.square().mul_small(MINUS_A0) - w * wp).half();

        // If ok != 0, then we could halve the point. If the point is
        // in the prime-order subgroup, then we can halve it again, in
        // which case its u coordinate is square. We only need a Legendre
        // symbol here, because we do not have to compute that extra half,
        // only to check whether it would be computable.
        ok &= !((u.legendre() >> 1) as u32);

        // If the source point was a low-order point, then the computations
        // above are incorrect. We handle this case here; among the
        // low-order points, only the neutral point is in the prime-order
        // subgroup.
        let lop = self.has_low_order();
        let neu = self.isneutral();
        ok ^= lop & (ok ^ neu);

        ok
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
        self.Y.set_cond(&P.Y, ctl);
        self.Z.set_cond(&P.Z, ctl);
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
        self.X.set_cond(&-self.X, ctl);
    }

    /// Maps this point to the corresponding Montgomery curve and returns
    /// the affine u coordinate of the resulting point.
    ///
    /// If this point is the neutral, then 0 is returned.
    pub fn to_montgomery_u(&self) -> GF448 {
        // u = y^2/x^2
        (self.Y / self.X).square()
    }

    /// Recodes a scalar into 90 signed digits.
    ///
    /// Each digit is in -15..+16, top digit is in 0..+2.
    fn recode_scalar(n: &Scalar) -> [i8; 90] {
        let mut sd = [0i8; 90];
        let bb = n.encode();
        let mut cc: u32 = 0;       // carry from lower digits
        let mut i: usize = 0;      // index of next source byte
        let mut acc: u32 = 0;      // buffered bits
        let mut acc_len: i32 = 0;  // number of buffered bits
        for j in 0..89 {
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
        sd[89] = (acc + cc) as i8;
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

            P.X.set_cond(&win[i].X, w);
            P.Y.set_cond(&win[i].Y, w);
            P.Z.set_cond(&win[i].Z, w);
        }

        // Negate the returned value if needed.
        P.X.set_cond(&-P.X, s);

        P
    }

    /// Multiplies this point by a scalar (in place).
    ///
    /// This operation is constant-time with regard to both the points
    /// and the scalar value.
    ///
    /// Note: a scalar is nominally an integer modulo the prime L, which
    /// is the order of a specific subgroup of the curve. If the source
    /// point is NOT in that subgroup, then what is computed is the
    /// product of the point by an integer in the 0 to L-1 range.
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

        // Recode the scalar into 90 signed digits.
        let sd = Self::recode_scalar(n);

        // Process the digits in high-to-low order.
        *self = Self::lookup(&win, sd[89]);
        for i in (0..89).rev() {
            self.set_xdouble(5);
            self.set_add(&Self::lookup(&win, sd[i]));
        }
    }

    /// Lookups a point from a window of points in affine coordinates, with
    /// sign handling (constant-time).
    fn lookup_affine(win: &[PointAffine; 16], k: i8) -> PointAffine {
        // Split k into its sign s (0xFFFFFFFF for negative) and
        // absolute value (f).
        let s = ((k as i32) >> 8) as u32;
        let f = ((k as u32) ^ s).wrapping_sub(s);
        let mut x = GF448::ZERO;
        let mut y = GF448::ONE;
        for i in 0..16 {
            // win[i] contains (i+1)*P; we want to keep it if (and only if)
            // i+1 == f.
            // Values a-b and b-a both have their high bit equal to 0 only
            // if a == b.
            let j = (i as u32) + 1;
            let w = !(f.wrapping_sub(j) | j.wrapping_sub(f));
            let w = ((w as i32) >> 31) as u32;

            x.set_cond(&win[i].x, w);
            y.set_cond(&win[i].y, w);
        }

        // Negate the returned value if needed.
        x.set_cond(&-x, s);

        PointAffine { x, y }
    }

    /// Sets this point by multiplying the conventional generator by the
    /// provided scalar.
    ///
    /// This operation is constant-time. It is faster than using the
    /// generic multiplication on `Self::BASE`.
    pub fn set_mulgen(&mut self, n: &Scalar) {
        // Recode the scalar into 90 signed digits.
        let sd = Self::recode_scalar(n);

        // We process six chunks in parallel; chunks use 15 digits each.
        *self = Self::from_affine(&Self::lookup_affine(&PRECOMP_B, sd[14]));
        self.set_add_affine(&Self::lookup_affine(&PRECOMP_B75, sd[29]));
        self.set_add_affine(&Self::lookup_affine(&PRECOMP_B150, sd[44]));
        self.set_add_affine(&Self::lookup_affine(&PRECOMP_B225, sd[59]));
        self.set_add_affine(&Self::lookup_affine(&PRECOMP_B300, sd[74]));
        self.set_add_affine(&Self::lookup_affine(&PRECOMP_B375, sd[89]));

        // Process the digits in high-to-low order.
        for i in (0..14).rev() {
            self.set_xdouble(5);
            self.set_add_affine(
                &Self::lookup_affine(&PRECOMP_B, sd[i]));
            self.set_add_affine(
                &Self::lookup_affine(&PRECOMP_B75, sd[i + 15]));
            self.set_add_affine(
                &Self::lookup_affine(&PRECOMP_B150, sd[i + 30]));
            self.set_add_affine(
                &Self::lookup_affine(&PRECOMP_B225, sd[i + 45]));
            self.set_add_affine(
                &Self::lookup_affine(&PRECOMP_B300, sd[i + 60]));
            self.set_add_affine(
                &Self::lookup_affine(&PRECOMP_B375, sd[i + 75]));
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

    /// 5-bit wNAF recoding of a scalar; output is a sequence of 447
    /// digits.
    ///
    /// Non-zero digits have an odd value, between -15 and +15
    /// (inclusive). (The recoding is constant-time, but use of wNAF is
    /// inherently non-constant-time.)
    fn recode_scalar_NAF(n: &Scalar) -> [i8; 447] {
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
        // Since L < 2^446, only 447 digits are necessary at most.

        let mut sd = [0i8; 447];
        let bb = n.encode();
        let mut x = bb[0] as u32;
        for i in 0..447 {
            if (i & 7) == 4 && i < 444 {
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

    /// Given scalars `u` and `v`, sets this point to `u*self + v*B`
    /// (with `B` being the conventional generator of the prime order
    /// subgroup).
    ///
    /// This can be used to support EdDSA signature verification, though
    /// for that task `verify_helper_vartime()` is faster.
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
        for i in (0..447).rev() {
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
                    self.set_add_affine(&PRECOMP_B[e2 as usize - 1]);
                } else {
                    self.set_sub_affine(&PRECOMP_B[(-e2) as usize - 1]);
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

    // Given scalars `u` and `v`, returns `u*self + v*B` (with `B` being
    // the conventional generator of the prime order subgroup).
    //
    // This can be used to support EdDSA signature verification, though
    // for that task `verify_helper_vartime()` is faster.
    //
    // THIS FUNCTION IS NOT CONSTANT-TIME; it shall be used only with
    // public data.
    #[inline(always)]
    pub fn mul_add_mulgen_vartime(self, u: &Scalar, v: &Scalar) -> Self {
        let mut R = self;
        R.set_mul_add_mulgen_vartime(u, v);
        R
    }

    /// 5-bit wNAF recoding of a half-width integer. Input integer is
    /// in unsigned little-endian convention. Output is a sequence of
    /// 225 digits.
    ///
    /// Non-zero digits have an odd value, between -15 and +15
    /// (inclusive). (The recoding is constant-time, but use of wNAF is
    /// inherently non-constant-time.)
    fn recode_halfwidth_NAF(n: &[u8; 28]) -> [i8; 225] {
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

        let mut sd = [0i8; 225];
        let mut x = n[0] as u32;
        for i in 0..225 {
            if (i & 7) == 4 && i < 220 {
                x += (n[(i + 4) >> 3] as u32) << 4;
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

    /// Check whether `4*s*B = 4*R + 4*k*A`, for the provided scalars `s`
    /// and `k`, provided points `A` (`self`) and `R`, and conventional
    /// generator `B`.
    ///
    /// Returned value is true on match, false otherwise. This function
    /// is meant to support Ed448 signature verification.
    ///
    /// THIS FUNCTION IS NOT CONSTANT-TIME; it shall be used only with
    /// public data.
    pub fn verify_helper_vartime(self,
        R: &Point, s: &Scalar, k: &Scalar) -> bool
    {
        // We split k into (c0, c1) such that k = c0/c1.
        // c0 and c1 are signed, with length 225 bits each (including the
        // sign bit). In absolute value, they are at most
        // sqrt(L*(2/sqrt(3))), which is approximately 1.075*2^223.
        //
        // Let:
        //   T = s*B - k*A - R
        // The equation to check is, formally, 4*T = neutral. This is
        // equivalent to checking that T is a point of low order.
        // Since k = c0/c1, we can check instead define the point U as:
        //   U = (s*c1)*B + c0*(-A) + c1*(-R)
        // and verify that U has low order. Indeed:
        //   U = c1*T
        //   c1 != 0 and |c1| < L, hence c1 is invertible modulo L
        // If we write T = T0 + T1, with T0 a point of low-order and T1
        // a point of order L (the decomposition always exist and is unique),
        // then U = c1*T0 + c1*T1, and c1*T1 = 0 if and only if T1 = 0.
        // Hence, U has low order if and only if T has low order.
        //
        // We can split s*c1 = s0 + s1*2^225, which allows to write U as
        // a linear combination of four points, with four half-width
        // coefficients (at most 225 bits).

        let (mut c0, mut c1) = k.split_vartime();
        let mut ss = *s;
        let mut P0 = self;
        let mut P1 = *R;
        if c0[c0.len() - 1] >= 0x80 {
            let mut zz = 1;
            for i in 0..c0.len() {
                let z = (!c0[i] as u32) + zz;
                c0[i] = z as u8;
                zz = z >> 8;
            }
        } else {
            P0.set_neg();
        }
        if c1[c1.len() - 1] >= 0x80 {
            let mut zz = 1;
            for i in 0..c1.len() {
                let z = (!c1[i] as u32) + zz;
                c1[i] = z as u8;
                zz = z >> 8;
            }
            ss = -ss;
        } else {
            P1.set_neg();
        }
        ss *= Scalar::decode_reduce(&c1);

        // Now, c0 and c1 are non-negative, and thus fit on 224 bits.

        let mut h0 = [0u8; 28];
        let mut h1 = [0u8; 28];
        h0[..].copy_from_slice(&c0[..28]);
        h1[..].copy_from_slice(&c1[..28]);
        let sd0 = Self::recode_halfwidth_NAF(&h0);
        let sd1 = Self::recode_halfwidth_NAF(&h1);
        let sds = Self::recode_scalar_NAF(&ss);

        // Compute windows for points P0 and P1:
        //    win0[i] = (2*i+1)*P0   (i = 0 to 7)
        //    win1[i] = (2*i+1)*P1   (i = 0 to 7)
        let Q0 = P0.double();
        let Q1 = P1.double();
        let mut win0 = [Self::NEUTRAL; 8];
        let mut win1 = [Self::NEUTRAL; 8];
        win0[0] = P0;
        win1[0] = P1;
        for i in 1..8 {
            win0[i] = win0[i - 1] + Q0;
            win1[i] = win1[i - 1] + Q1;
        }

        // Initialize the accumulator.
        let mut T = Self::NEUTRAL;
        let mut isneu = true;

        // Process all other digits. We coalesce long sequences of
        // doublings to leverage the optimizations of xdouble().
        let mut ndbl = 0u32;
        for i in (0..225).rev() {
            // We have one more doubling to perform.
            ndbl += 1;

            // Get next digits. If they are all zeros, then we can loop
            // immediately.
            let e0 = sd0[i];
            let e1 = sd1[i];
            let e2 = sds[i];
            let e3 = if i < 222 { sds[i + 225] } else { 0 };
            if ((e0 as u32) | (e1 as u32) | (e2 as u32) | (e3 as u32)) == 0 {
                continue;
            }

            // Apply accumulated doubles.
            if isneu {
                isneu = false;
            } else {
                T.set_xdouble(ndbl);
            }
            ndbl = 0u32;

            // Process digits.
            if e0 != 0 {
                if e0 > 0 {
                    T.set_add(&win0[e0 as usize >> 1]);
                } else {
                    T.set_sub(&win0[(-e0) as usize >> 1]);
                }
            }
            if e1 != 0 {
                if e1 > 0 {
                    T.set_add(&win1[e1 as usize >> 1]);
                } else {
                    T.set_sub(&win1[(-e1) as usize >> 1]);
                }
            }
            if e2 != 0 {
                if e2 > 0 {
                    T.set_add_affine(&PRECOMP_B[e2 as usize - 1]);
                } else {
                    T.set_sub_affine(&PRECOMP_B[(-e2) as usize - 1]);
                }
            }
            if e3 != 0 {
                if e3 > 0 {
                    T.set_add_affine(&PRECOMP_B225[e3 as usize - 1]);
                } else {
                    T.set_sub_affine(&PRECOMP_B225[(-e3) as usize - 1]);
                }
            }
        }

        // We can skip the accumulated doubles (if any) because they
        // won't change the status of T as a low-order point.

        T.has_low_order() == 0xFFFFFFFF
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

/// An Ed448 private key.
///
/// It is built from a 57-byte seed (which should be generated from a
/// cryptographically secure random source with at least 224 bits of
/// entropy). From the seed are derived the secret scalar and the public
/// key. The public key is a curve point, that can be encoded as such.
#[derive(Clone, Copy, Debug)]
pub struct PrivateKey {
    s: Scalar,                  // secret scalar
    seed: [u8; 57],             // source seed
    h: [u8; 57],                // derived seed (second half of SHAKE256(seed))
    pub public_key: PublicKey,  // public key
}

/// An Ed448 public key.
///
/// It wraps around the curve point, but also includes a copy of the
/// encoded point. The point and its encoded version can be accessed
/// directly; if modified, then the two values MUST match.
#[derive(Clone, Copy, Debug)]
pub struct PublicKey {
    pub point: Point,
    pub encoded: [u8; 57],
}

/// Constant string "SigEd448".
const HASH_HEAD: [u8; 8] = [
    0x53, 0x69, 0x67, 0x45, 0x64, 0x34, 0x34, 0x38,
];

impl PrivateKey {

    /// Generates a new private key from a cryptographically secure RNG.
    pub fn generate<T: CryptoRng + RngCore>(rng: &mut T) -> Self {
        let mut seed = [0u8; 57];
        rng.fill_bytes(&mut seed);
        Self::from_seed(&seed)
    }

    /// Instantiates a private key from the provided seed.
    ///
    /// The seed length MUST be exactly 57 bytes (a panic is triggered
    /// otherwise).
    pub fn from_seed(seed: &[u8]) -> Self {
        // We follow RFC 8032, section 5.2.5.

        // The seed MUST have length 57 bytes.
        assert!(seed.len() == 57);
        let mut bseed = [0u8; 57];
        bseed[..].copy_from_slice(seed);

        // Hash the seed with SHAKE256, with a 114-byte output.
        let mut sh = Shake256::default();
        sh.update(seed);
        let mut hh = [0u8; 114];
        sh.finalize_xof_reset().read(&mut hh);

        // Prune the first half and decode it as a scalar (with
        // reduction).
        hh[0] &= 0xFC;
        hh[56] = 0;
        hh[55] |= 0x80;
        let s = Scalar::decode_reduce(&hh[..57]);

        // Save second half of the hashed seed for signing operations.
        let mut h = [0u8; 57];
        h[..].copy_from_slice(&hh[57..]);

        // Public key is obtained from the secret scalar.
        let public_key = PublicKey::from_point(&Point::mulgen(&s));

        Self { s, seed: bseed, h, public_key }
    }

    /// Decodes a private key from bytes.
    ///
    /// If the source slice has length exactly 57 bytes, then these bytes
    /// are interpreted as a seed, and the private key is built on that
    /// seed (see `from_seed()`). Otherwise, `None` is returned.
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() == 57 {
            Some(Self::from_seed(<&[u8; 57]>::try_from(buf).unwrap()))
        } else {
            None
        }
    }

    /// Encodes a private key into 57 bytes.
    ///
    /// This actually returns a copy of the seed.
    pub fn encode(self) -> [u8; 57] {
        self.seed
    }

    /// Signs a message.
    ///
    /// This is the "Ed448" mode of RFC 8032 (no pre-hashing),
    /// also known as "PureEdDSA on Curve448". No context is provided;
    /// this is equivalent to `sign_ctx()` with an empty (zero-length)
    /// context.
    pub fn sign_raw(self, m: &[u8]) -> [u8; 114] {
        self.sign_inner(0, &[0u8; 0], m)
    }

    /// Signs a message (with context).
    ///
    /// This is the "Ed448" mode of RFC 8032 (no pre-hashing),
    /// also known as "PureEdDSA on Curve448". A context string is also
    /// provided; it MUST have length at most 255 bytes.
    pub fn sign_ctx(self, ctx: &[u8], m: &[u8]) -> [u8; 114] {
        self.sign_inner(0, ctx, m)
    }

    /// Signs a pre-hashed message.
    ///
    /// This is the "Ed448ph" mode of RFC 8032 (message is pre-hashed),
    /// also known as "HashEdDSA on Curve448". The hashed message `hm`
    /// is provided (presumably, that hash value was obtained with
    /// SHAKE256 and an output of 64 bytes; the caller does the hashing
    /// itself). A context string is also provided; it MUST have length
    /// at most 255 bytes.
    pub fn sign_ph(self, ctx: &[u8], hm: &[u8]) -> [u8; 114] {
        self.sign_inner(1, ctx, hm)
    }

    /// Inner signature generation function.
    fn sign_inner(self, phflag: u8, ctx: &[u8], m: &[u8]) -> [u8; 114] {
        // SHAKE256(dom4(F, C) || prefix || PH(M), 114) -> scalar r
        let mut sh = Shake256::default();
        assert!(ctx.len() <= 255);
        let clen = ctx.len() as u8;
        sh.update(&HASH_HEAD);
        sh.update(&[phflag]);
        sh.update(&[clen]);
        sh.update(ctx);
        sh.update(&self.h);
        sh.update(m);
        let mut hv1 = [0u8; 114];
        sh.finalize_xof_reset().read(&mut hv1);
        let r = Scalar::decode_reduce(&hv1);

        // R = r*B
        let R = Point::mulgen(&r);
        let R_enc = R.encode();

        // SHAKE256(dom4(F, C) || R || A || PH(M), 114) -> scalar k
        sh.update(&HASH_HEAD);
        sh.update(&[phflag]);
        sh.update(&[clen]);
        sh.update(ctx);
        sh.update(&R_enc);
        sh.update(&self.public_key.encoded);
        sh.update(m);
        let mut hv2 = [0u8; 114];
        sh.finalize_xof_reset().read(&mut hv2);
        let k = Scalar::decode_reduce(&hv2);

        // Signature is (R, S) with S = r + k*s mod L
        // S is encoded over 57 bytes, even though scalars use only 56;
        // the last byte is left at zero.
        let mut sig = [0u8; 114];
        sig[0..57].copy_from_slice(&R_enc);
        sig[57..113].copy_from_slice(&(r + k * self.s).encode());

        sig
    }
}

impl PublicKey {

    /// Creates an instance from a curve point.
    pub fn from_point(point: &Point) -> Self {
        Self { point: *point, encoded: point.encode() }
    }

    // Decodes the provided bytes as a public key.
    //
    // This process may fail if the source slice does not have length
    // exactly 57 bytes, or if it has length 57 bytes but these bytes
    // are not the valid encoding of a curve point.
    //
    // Note: decoding success does not guarantee that the point is in
    // the proper subgroup of prime order L. The point may be outside of
    // the subgroup. The point may also be the curve neutral point, or a
    // low order point.
    pub fn decode(buf: &[u8]) -> Option<PublicKey> {
        let point = Point::decode(buf)?;
        let mut encoded = [0u8; 57];
        encoded[..].copy_from_slice(&buf[0..57]);
        Some(Self { point, encoded })
    }

    /// Encodes the key into exactly 57 bytes.
    ///
    /// This simply returns the contents of the `encoded` field.
    pub fn encode(self) -> [u8; 57] {
        self.encoded
    }

    /// Verifies a signature on a message.
    ///
    /// This is the "Ed448" mode of RFC 8032 (no pre-hashing, a
    /// context is provided). This is equivalent to `verify_ctx()`
    /// with an empty (zero-length) context.
    ///
    /// Note: this function is not constant-time; it assumes that the
    /// public key and signature value are public data.
    pub fn verify_raw(self, sig: &[u8], m: &[u8]) -> bool {
        self.verify_inner(sig, 0, &[0u8; 0], m)
    }

    /// Verifies a signature on a message (with context).
    ///
    /// This is the "Ed448" mode of RFC 8032 (no pre-hashing, a
    /// context is provided). The context string MUST have length at most
    /// 255 bytes. Return value is `true` on a valid signature, `false`
    /// otherwise.
    ///
    /// Note: this function is not constant-time; it assumes that the
    /// public key and signature value are public data.
    pub fn verify_ctx(self, sig: &[u8], ctx: &[u8], m: &[u8]) -> bool {
        self.verify_inner(sig, 0, ctx, m)
    }

    /// Verifies a signature on a hashed message.
    /// 
    /// This is the "Ed448ph" mode of RFC 8032 (message is pre-hashed),
    /// also known as "HashEdDSA on Curve448". The hashed message `hm`
    /// is provided (presumably, that hash value was obtained with
    /// SHAKE256 and a 64-byte output; the caller does the hashing itself).
    /// A context string is
    /// also provided; it MUST have length at most 255 bytes. Return
    /// value is `true` on a valid signature, `false` otherwise.
    /// 
    /// Note: this function is not constant-time; it assumes that the
    /// public key and signature value are public data.
    pub fn verify_ph(self, sig: &[u8], ctx: &[u8], hm: &[u8]) -> bool {
        self.verify_inner(sig, 1, ctx, hm)
    }

    /// Inner signature verification function.
    fn verify_inner(self, sig: &[u8], phflag: u8, ctx: &[u8], m: &[u8])
        -> bool
    {
        // Signature must have length 114 bytes exactly.
        if sig.len() != 114 {
            return false;
        }

        // First half of the signature is the encoded point R;
        // second half is the scalar S. Both must decode successfully.
        // Note that the scalar itself uses only 56 bytes; the extra
        // 57th byte must be 0x00. The decoding functions enforce
        // canonicality (but point R may be outside of the order-L subgroup).
        if sig[113] != 0x00 {
            return false;
        }
        let R_enc = &sig[0..57];
        let R = match Point::decode(R_enc) {
            Some(R) => R,
            None    => { return false; }
        };
        let (S, ok) = Scalar::decode_ct(&sig[57..113]);
        if ok == 0 {
            return false;
        }

        // SHA-512(dom4(F, C) || R || A || PH(M)) -> scalar k
        // R is encoded over the first 57 bytes of the signature.
        let mut sh = Shake256::default();
        assert!(ctx.len() <= 255);
        let clen = ctx.len() as u8;
        sh.update(&HASH_HEAD);
        sh.update(&[phflag]);
        sh.update(&[clen]);
        sh.update(ctx);
        sh.update(R_enc);
        sh.update(&self.encoded);
        sh.update(m);
        let mut hv2 = [0u8; 114];
        sh.finalize_xof_reset().read(&mut hv2);
        let k = Scalar::decode_reduce(&hv2);

        // Check the verification equation 4*S*B = 4*R + 4*k*A.
        self.point.verify_helper_vartime(&R, &S, &k)
    }
}

// ========================================================================

// We hardcode known multiples of the points B, (2^115)*B, (2^225)*B
// and (2^340)*B, with B being the conventional base point. These are
// used to speed mulgen() operations up. The points are moreover stored
// in a affine format (only two coordinates x and y).
//
// The use of four sub-tables is a trade-off. With more sub-tables (e.g.
// a 6-way split with B, (2^75)*B, (2^150)*B...), some point doublings
// would be avoided in mulgen(), but the cumulative table size would be
// larger. In general, we prefer to keep relatively small tables so that
// use of this code does not kick out of L1 cache too much data and slows
// down whatever _other_ computations the calling app performs. Each
// table contains 16 points = 1792 bytes.

/// A point in affine coordinates (x,y).
#[derive(Clone, Copy, Debug)]
struct PointAffine {
    x: GF448,
    y: GF448,
}

impl PointAffine {

    /* unused
    const NEUTRAL: Self = Self {
        x: GF448::ZERO,
        y: GF448::ONE,
    };
    */
}

// Points i*B for i = 1 to 16, in affine format.
static PRECOMP_B: [PointAffine; 16] = [
    // B * 1
    PointAffine { x: GF448::w64be([ 0x4F1970C66BED0DED, 0x221D15A622BF36DA,
                                    0x9E146570470F1767, 0xEA6DE324A3D3A464,
                                    0x12AE1AF72AB66511, 0x433B80E18B00938E,
                                    0x2626A82BC70CC05E ]),
                  y: GF448::w64be([ 0x693F46716EB6BC24, 0x8876203756C9C762,
                                    0x4BEA73736CA39840, 0x87789C1E05A0C2D7,
                                    0x3AD3FF1CE67C39C4, 0xFDBD132C4ED7C8AD,
                                    0x9808795BF230FA14 ]) },
    // B * 2
    PointAffine { x: GF448::w64be([ 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
                                    0xAAAAAAAAAAAAAAAA, 0xAAAAAAA955555555,
                                    0x5555555555555555, 0x5555555555555555,
                                    0x5555555555555555 ]),
                  y: GF448::w64be([ 0xAE05E9634AD7048D, 0xB359D6205086C2B0,
                                    0x036ED7A035884DD7, 0xB7E36D728AD8C4B8,
                                    0x0D6565833A2A3098, 0xBBBCB2BED1CDA06B,
                                    0xDAEAFBCDEA9386ED ]) },
    // B * 3
    PointAffine { x: GF448::w64be([ 0x0865886B9108AF64, 0x55BD64316CB69433,
                                    0x32241B8B8CDA82C7, 0xE2BA077A4A3FCFE8,
                                    0xDAA9CBF7F6271FD6, 0xE862B769465DA857,
                                    0x5728173286FF2F8F ]),
                  y: GF448::w64be([ 0xE005A8DBD5125CF7, 0x06CBDA7AD43AA644,
                                    0x9A4A8D952356C3B9, 0xFCE43C82EC4E1D58,
                                    0xBB3A331BDB6767F0, 0xBFFA9A68FED02DAF,
                                    0xB822AC13588ED6FC ]) },
    // B * 4
    PointAffine { x: GF448::w64be([ 0x49DCBC5C6C0CCE2C, 0x1419A17226F929EA,
                                    0x255A09CF4E0891C6, 0x93FDA4BE70C74CC3,
                                    0x01B7BDF1515DD8BA, 0x21AEE1798949E120,
                                    0xE2CE42AC48BA7F30 ]),
                  y: GF448::w64be([ 0xD49077E4ACCDE527, 0x164B33A5DE021B97,
                                    0x9CB7C02F0457D845, 0xC90DC3227B8A5BC1,
                                    0xC0D8F97EA1CA9472, 0xB5D444285D0D4F5B,
                                    0x32E236F86DE51839 ]) },
    // B * 5
    PointAffine { x: GF448::w64be([ 0x7A9F9335A48DCB0E, 0x2BA7601EEDB50DEF,
                                    0x80CBCF728562ADA7, 0x56D761E895881280,
                                    0x8BC0D57A920C3C96, 0xF07B2D8CEFC6F950,
                                    0xD0A99D1092030034 ]),
                  y: GF448::w64be([ 0xADFD751A2517EDD3, 0xB9109CE4FD580ADE,
                                    0x260CA1823AB18FCE, 0xD86551F7B6980171,
                                    0x27D7A4EE59D2B33C, 0x58405512881F2254,
                                    0x43B4731472F435EB ]) },
    // B * 6
    PointAffine { x: GF448::w64be([ 0x54523E04498B722E, 0x00349AEBD97125D2,
                                    0xE673FB02603BED0B, 0x4AD95F654A06E4A8,
                                    0x76451C5C1840D12A, 0x71096C20A07443A8,
                                    0x787FD7652ABEF79C ]),
                  y: GF448::w64be([ 0x3F99B9CD8AB61C2D, 0x743EC056C667BFD0,
                                    0x53A2765A1A7B1218, 0x69C5852248DE76B6,
                                    0xE1FF1B80284E861D, 0x2B3ECCE776938EA9,
                                    0x8CEA5D1DA07C7BCC ]) },
    // B * 7
    PointAffine { x: GF448::w64be([ 0x079748E5C89BEF77, 0x467B9A5291B6D78A,
                                    0xF2D8C3719BBA4223, 0xD568E6840DF9C7B0,
                                    0xB5A2092D33501470, 0x6E0B110A6B478AC7,
                                    0xD7DF9567CEB5EAF7 ]),
                  y: GF448::w64be([ 0x7DDEDA3C90FD699C, 0x7373BCB23583B875,
                                    0x94417E7179D00E40, 0xCA78F50E7B3946FD,
                                    0x2F84C9D8687A3C40, 0xBBB734E866972B5C,
                                    0x09E20D3FADAC377F ]) },
    // B * 8
    PointAffine { x: GF448::w64be([ 0xD1F1BC5520E8A35B, 0x63FB45D0E66BD97B,
                                    0x037C08E2D7BC2856, 0x85DC8B976C4CB47D,
                                    0x816A64F9080DC1AD, 0x713E223ACA9406B6,
                                    0x962538A67153BDE0 ]),
                  y: GF448::w64be([ 0xC7564F4E9BEA9BDA, 0xE87D1F3574FF46BA,
                                    0x408C6B38B3A4B93B, 0x7BD94C7B4B98EAD3,
                                    0x86AD8208F7003BA8, 0xD89F1663164BC8EE,
                                    0x454EB873CE69E09B ]) },
    // B * 9
    PointAffine { x: GF448::w64be([ 0xEF1650CED584FCAA, 0x8CD7D824AD4DAEFA,
                                    0x6CDAE1D134A3AE73, 0x18AD44EE101E3444,
                                    0x30A0D93E53F4FC89, 0x89ED1A467EC49F7C,
                                    0x793D0DEF76AB686B ]),
                  y: GF448::w64be([ 0x1560B6298F12AE2D, 0xAF7ACBD5A84DF251,
                                    0x8C687993B9C165FA, 0xBA605671391C15DA,
                                    0x256FBB47C32D4297, 0x1140F52CEA8EF3FB,
                                    0x8BA74DF674F4754F ]) },
    // B * 10
    PointAffine { x: GF448::w64be([ 0x77486F9D19F6411C, 0xDD35D30D1C3235F7,
                                    0x1936452C787E5C03, 0x4134D3E8172278AC,
                                    0xA61622BC805761CE, 0x3DAB65118A0122D7,
                                    0x3B403165D0ED303D ]),
                  y: GF448::w64be([ 0x4D2FEA0B026BE110, 0x24F1F0FE7E94E618,
                                    0xE8AC17381ADA1D1B, 0xF7EE293A68FF5D0B,
                                    0xF93C1997DC1AABDC, 0x0C7E6381428D85B6,
                                    0xB1954A89E4CDDF67 ]) },
    // B * 11
    PointAffine { x: GF448::w64be([ 0xBF12ABBC2408EEB1, 0xE56D71A4C6405D44,
                                    0x0DB5122D34422EE7, 0x104F502967B93775,
                                    0x581DBAE20B14982F, 0x294863118655940D,
                                    0xB8EF4DA0254DC10A ]),
                  y: GF448::w64be([ 0x42D35B554BF2FDAD, 0x308123E0173328C9,
                                    0x00FF0C5D20F18F58, 0x06D187F7390DF605,
                                    0xD0067485DDA13A2E, 0xFE1288366ABF3D8C,
                                    0xAB0CC9F86016AF01 ]) },
    // B * 12
    PointAffine { x: GF448::w64be([ 0x20DA15621953A323, 0xA58A17D226879A69,
                                    0x6E54388D690753CE, 0xA1A4BFFA38EC916A,
                                    0x57B108DB485DE913, 0x6D4D8C1E6FF99976,
                                    0x8658F07A037DACC4 ]),
                  y: GF448::w64be([ 0x4DD13BBA92CA3C03, 0xFF7344560BBE52E7,
                                    0x646321065845849C, 0x23A3FD49D7E2DC4C,
                                    0xAD1DA346C255A6C7, 0xB42FB94C4FEBEBB2,
                                    0xD17E441CC3D41EA7 ]) },
    // B * 13
    PointAffine { x: GF448::w64be([ 0x0BAEDD2EF0F50F06, 0xBAAEC1E928E770A5,
                                    0x44FC9708C5B52325, 0x01B339E1C186D123,
                                    0x44552BCC67DA626B, 0xFA5A099FC7E8F9C6,
                                    0x05009135F2FFB1F1 ]),
                  y: GF448::w64be([ 0xC59B84F0E52A4C34, 0xD56E0804A2685460,
                                    0x4214E6900213D8A7, 0x82E33DDD988AF715,
                                    0x9FD7363165741DDB, 0x9BE2156E536E4EC3,
                                    0xE9535D6D8BF479E5 ]) },
    // B * 14
    PointAffine { x: GF448::w64be([ 0xC8D60E7EBF7F4116, 0x43F405852C65556D,
                                    0x8915CFF643ABFB9D, 0x9E3D34553CB6E837,
                                    0x3E4D688CADED78B3, 0x7367E3D39362C9CD,
                                    0x639B527C6E67A9EA ]),
                  y: GF448::w64be([ 0x66A27BAE2CE4DB64, 0xC867F6BD4CEB7171,
                                    0xDD04222BFC80EEDC, 0x1E7B71AFC911C28E,
                                    0x7DF2197CAD92A0EA, 0xBF766541D96C0347,
                                    0x0F17EEC500CF807C ]) },
    // B * 15
    PointAffine { x: GF448::w64be([ 0x30D6D8C216D8D3B6, 0xF721A37BA96945E2,
                                    0xFBFD508AF9A785D2, 0x1605611215A1EE89,
                                    0x27EF21A25E43B430, 0x35DB5768ABA87123,
                                    0x27525D45F24729D9 ]),
                  y: GF448::w64be([ 0xF48FB8F28ADE0AC9, 0x86C2B28DB8027B05,
                                    0x2223312990799C86, 0x9F665756D4EF19C9,
                                    0xF25900851325A763, 0xF955FA40B4A2A068,
                                    0x4E3065E0852074C3 ]) },
    // B * 16
    PointAffine { x: GF448::w64be([ 0xDD8402F36C2E9F96, 0x6E9290104C4F302D,
                                    0xE8A918066E09D5DD, 0x6F5B419EC49F9EEA,
                                    0xFD74C278D5B37ED6, 0x8C9C6300B41A0776,
                                    0x8FEDB5FBC9120262 ]),
                  y: GF448::w64be([ 0x1AB2B2D015957671, 0x740AB813879AFEFA,
                                    0xF0A16ACD03FB9C2B, 0x9A0193724AB99426,
                                    0x46BC0936D4EDC831, 0x58EF53C0ED47CA82,
                                    0x0CCDE6EC861D7985 ]) },
];

// Points i*(2^75)*B for i = 1 to 16, in affine format.
static PRECOMP_B75: [PointAffine; 16] = [
    // (2^75)*B * 1
    PointAffine { x: GF448::w64be([ 0x4C733E0AFAE0EAB6, 0xEA78CFC253A53108,
                                    0x6002796E33425AF6, 0xAC29A15E9BAF9626,
                                    0x71F86AE325334C6A, 0x159377751DC0047B,
                                    0x8AD24AE1D77E9709 ]),
                  y: GF448::w64be([ 0x8552089F6E36B402, 0x2C937DDCE3D9B3E4,
                                    0xDF9EED7DA0A66E38, 0xC169E1A0A56A3EFD,
                                    0x55533AD1BB515526, 0x80BAC14E9E450203,
                                    0xA64B7443A97C99B9 ]) },
    // (2^75)*B * 2
    PointAffine { x: GF448::w64be([ 0x9F98BF3E5D932FB5, 0xCBA335FEDDA7EE3B,
                                    0x44650A232C79414D, 0x59352A919E5B8F6E,
                                    0x4C626EFDEBD84EB2, 0x5732BF47B00E29F2,
                                    0x8BDDA5651483F233 ]),
                  y: GF448::w64be([ 0x67A9741019C9E4B8, 0xA885212E8A2F5E1E,
                                    0x0865C38141E696FB, 0x4BD267EA55BD7062,
                                    0xAFBE74D0296542BC, 0xCAB913F1EBE99523,
                                    0x464B7677E64B1919 ]) },
    // (2^75)*B * 3
    PointAffine { x: GF448::w64be([ 0xCA57FB9690A80C01, 0xC8A0FF490E15036F,
                                    0xA812C289EE84CB2A, 0x3CA043A3B967CB47,
                                    0x5BD84965CBFDFCCA, 0x480F25B79AE9DCAA,
                                    0xDF768D84C693F339 ]),
                  y: GF448::w64be([ 0x1932F12AC69704BD, 0xA13134B04A5884DD,
                                    0xEB87992B53565551, 0xC178C3F550E6C38D,
                                    0xC40A40B287486026, 0x1FA9613FDB4FCAEB,
                                    0x2B1E5C5CDBB1A44D ]) },
    // (2^75)*B * 4
    PointAffine { x: GF448::w64be([ 0x2B0B7EF7FA010D8F, 0x9B4732AB2CBC8A5B,
                                    0x7D4ECCA6ACC3F1C5, 0x541CD70077FBC1E4,
                                    0x90D0C0BCF1A04162, 0x0D2933EB62DDF19C,
                                    0xD3EAB8BB0EEAF0A4 ]),
                  y: GF448::w64be([ 0x929863C5671C2F96, 0x5D23363DE6E069B5,
                                    0x7DA6EB184A449E0A, 0x5AA8D7FB70E8613D,
                                    0x58A63D27D868DE44, 0x019C04C9C19CD5EE,
                                    0x81B75ADB7640778F ]) },
    // (2^75)*B * 5
    PointAffine { x: GF448::w64be([ 0xAE13F0F0B55F9519, 0x7118315A0CD33F22,
                                    0xB44793560E23A6C5, 0x209C3D68B97AF72D,
                                    0x1B30C294C43CD73B, 0x201A75A5DB92D29D,
                                    0xD5981C0ABC23991D ]),
                  y: GF448::w64be([ 0xB9F1F32654A4EA94, 0xAB0EAB72696DFD7D,
                                    0x1C18911F21D1C1A9, 0xBEC5556B3FF2F709,
                                    0x7A6EA503C216ABD7, 0x3D76ECD6CE3C82D8,
                                    0x69FD246E739160D5 ]) },
    // (2^75)*B * 6
    PointAffine { x: GF448::w64be([ 0x6263F5541D80FE05, 0xDCEE0960CBC7DA55,
                                    0x775C9B1D492A11DC, 0x408B44481686A0A0,
                                    0x2DD82BF9D6E792AB, 0xA817ADE8E77B3DD8,
                                    0xE41B04534FFE0B49 ]),
                  y: GF448::w64be([ 0xD97A45E725F8C70E, 0xB577C8F74EE9F6A8,
                                    0x6608F958780F34EA, 0x70372E08BF82B9C2,
                                    0xC496D503504B00A1, 0xC211E18CEAFB5FBA,
                                    0x4CF75A3E463DC1B0 ]) },
    // (2^75)*B * 7
    PointAffine { x: GF448::w64be([ 0x0C81C5E86D6CAD67, 0x05C653A165C06757,
                                    0x45E99CA9A15245AE, 0x0B6E2247985D4364,
                                    0x83FFA8D9055C9D4B, 0xD5B34B3F26D3345B,
                                    0x96FCE9CFA24364FD ]),
                  y: GF448::w64be([ 0x9B2DCB1C72324E32, 0xF5CE8608ACD82388,
                                    0xE0EA7D1351ABAF9A, 0x95B77C35F8F22553,
                                    0x73BF8CBAE1B5C810, 0x087CAC0625FC7C1D,
                                    0xB2935D71D6E4501B ]) },
    // (2^75)*B * 8
    PointAffine { x: GF448::w64be([ 0x303F8869F1C0DBBF, 0x349DEC3537D17FE8,
                                    0xC4530A690E883070, 0x4E28BA1C86C92C1D,
                                    0x5B2AC822DF238DFB, 0x6CA5A2E81DB119D7,
                                    0x99105130FCFB7BA2 ]),
                  y: GF448::w64be([ 0x74821B0AA368EC1D, 0x11E96CF11C59BC3D,
                                    0x6D63635613227583, 0xD41D6C5262F9B9B3,
                                    0xF835B27BBB1150C7, 0x8D632B3B0E1E6423,
                                    0x656B7E4AD759BFD1 ]) },
    // (2^75)*B * 9
    PointAffine { x: GF448::w64be([ 0x4B6053760521966C, 0x2C465B882111474C,
                                    0xAF1C7BBBD18D62B7, 0xA4F7218EFAB8ABFC,
                                    0xA0C4953441FF7D29, 0x978A76901537C0E0,
                                    0x68A29D99762068C1 ]),
                  y: GF448::w64be([ 0xD69C61202E32D78E, 0x108A255E96452F3C,
                                    0x6008A80D0E055B3D, 0xB0AA5EFAD2444B07,
                                    0xF392C9AAEA4C3284, 0x43EB4B08C7BCE0BC,
                                    0x406021CF619E2E0D ]) },
    // (2^75)*B * 10
    PointAffine { x: GF448::w64be([ 0x3F7F889A30C81D95, 0xE275177379EF2491,
                                    0x60D24BEB494BC9F8, 0x4D3B2C2B9AE178C1,
                                    0x4DF841CC0A5759BA, 0x9961B0EBA8D22CBE,
                                    0x7E8B732E33A4390A ]),
                  y: GF448::w64be([ 0xA11D189C6DB5F4FF, 0xBE74E3E1059EEF80,
                                    0x391BEC6FEA8E90F4, 0xF2EE450A5196B583,
                                    0x57AD38F2180C42DF, 0x2B66D30E4F9BD0FB,
                                    0xF8DD43FE27F65987 ]) },
    // (2^75)*B * 11
    PointAffine { x: GF448::w64be([ 0x4792939144A75768, 0x2CB2BA123EEAA74D,
                                    0xEF3A66C359B70442, 0xC6E94350C5C92CF1,
                                    0x6A485B04D905CD98, 0x14808564DC014378,
                                    0xAE142B243FCD21EF ]),
                  y: GF448::w64be([ 0x5460FBD34E3C9780, 0x05B2B7D1BAAD3757,
                                    0x99FFFA3CFD9D3A85, 0xA5D9654C527BAA86,
                                    0xB0C7921B662F3AA3, 0x0C5D8DD77940DC45,
                                    0xC4DD50A74C5D60E4 ]) },
    // (2^75)*B * 12
    PointAffine { x: GF448::w64be([ 0xFFBECB4792E9B019, 0xDDEC5C54358ADCE1,
                                    0x209C3A9E75BDEF96, 0xC0874A626463E305,
                                    0xFEC7FC64A9702AF6, 0x15E668B9E6B5B85D,
                                    0x7C516C16D6F21BE0 ]),
                  y: GF448::w64be([ 0xA364CBDE03ACABB8, 0x42D4C024FFFD3710,
                                    0x97C8290A680A58A6, 0xAD20D4C0A0421448,
                                    0x3E5DA4A1DFE60BFF, 0x843DFB0C8AA0FB89,
                                    0x3B015735A9B5C2A9 ]) },
    // (2^75)*B * 13
    PointAffine { x: GF448::w64be([ 0x7960C4B6B7CDF1B1, 0x9FDCC83176BB6012,
                                    0xD070FA68C116BFBC, 0x189DD395AB558DC6,
                                    0x9BDB3EE913C1899E, 0xFDE6634770EFA676,
                                    0xC0A1A78F2C56E258 ]),
                  y: GF448::w64be([ 0x7BD5869CBF0AD32C, 0x264052A293734A9B,
                                    0x3007CCDAADEDA776, 0x5A6F8040D4366B94,
                                    0x3A0C5F6B097DCCA0, 0x78348B27BFC91801,
                                    0x596C338ED48789D4 ]) },
    // (2^75)*B * 14
    PointAffine { x: GF448::w64be([ 0xD4A195509EF6C8AB, 0x5175C5ACC2F304A1,
                                    0xD85E7F4E047F5D1E, 0x7ADFB3882C7ACEA7,
                                    0xBE5B3A6612D1D4EA, 0x116A7BC1747E6A1F,
                                    0x1EA7E00733702A74 ]),
                  y: GF448::w64be([ 0x5DFBB8E445E10BC3, 0x8EECD4583CCD16D4,
                                    0x860A83CCAB59D02E, 0x1F56731D116BB595,
                                    0x87FF037F0FB4DB2C, 0x0BFB66BB8B88C808,
                                    0xAD3E2B82DBCFCDA5 ]) },
    // (2^75)*B * 15
    PointAffine { x: GF448::w64be([ 0x1EB42FFBBC516356, 0x5E11C8853732C91A,
                                    0x580ED963C9BE70CD, 0xF8DFBB8463CD89F8,
                                    0x55AF41A8F4F5EF2F, 0x33FD0EA1E80F180B,
                                    0xB2E279B7CDDCEB56 ]),
                  y: GF448::w64be([ 0x53D1E5D29E268020, 0x5281CFAA874AB656,
                                    0x72A1F6802A0E8DBC, 0x860F53B9D20E822F,
                                    0x6953DF6F65735C6E, 0x1007D61DA3CF5C38,
                                    0x984A169D5BDF65B6 ]) },
    // (2^75)*B * 16
    PointAffine { x: GF448::w64be([ 0x740E230765BE619C, 0x3542133D29A830E3,
                                    0x7ABEA84F77F4C25D, 0x00F624A81060A44C,
                                    0x790823452D67D931, 0x6EAA4F5505F4B2EF,
                                    0xFD99C2DCC17E9E01 ]),
                  y: GF448::w64be([ 0xACF60852C91786EF, 0x7FA80226258CC1F1,
                                    0x49F07C6F4901D491, 0xFE01D58B7095F6D8,
                                    0x98A8ADFA5C538A86, 0xFA7BA66F33453B62,
                                    0xA89C2B98EE4A9908 ]) },
];

// Points i*(2^150)*B for i = 1 to 16, in affine format.
static PRECOMP_B150: [PointAffine; 16] = [
    // (2^150)*B * 1
    PointAffine { x: GF448::w64be([ 0x530EEE10F9C1C4A2, 0xE39558BC78FD9649,
                                    0x3D77C7D57A5C8671, 0x08C3913975147CC3,
                                    0x1660658128AE0D91, 0x6D47EF49EA94F050,
                                    0xEC972A6C29752104 ]),
                  y: GF448::w64be([ 0x006B04238D404B59, 0x3FCC958BB30AD397,
                                    0xC961225897A54DFF, 0x01A9298B7D015D9B,
                                    0x89897C9447FCF092, 0x18F772DDF9ED36BC,
                                    0x9646831F78D76976 ]) },
    // (2^150)*B * 2
    PointAffine { x: GF448::w64be([ 0xDE58BD78AFF3B7CA, 0xBD4AF0C971596EDB,
                                    0x5CEB1BD1C1D261E6, 0x4942B4ACB05892D2,
                                    0x8038F65E6F482DC5, 0xF3D77AFBF6696854,
                                    0xDA804A3AD4AF930A ]),
                  y: GF448::w64be([ 0x8F73DACA908507F1, 0x9884528A52B995D3,
                                    0xB74C905E8259543E, 0x2C2DE25656F41E9C,
                                    0xB421C2C378930675, 0x10960ADCA620379F,
                                    0x511278FA98CDBFFD ]) },
    // (2^150)*B * 3
    PointAffine { x: GF448::w64be([ 0xA0D21178FCAE90BA, 0x4F9238F2AC7436A3,
                                    0x7A61B3F90D2161F4, 0x51EEA60CE9386E01,
                                    0xB3AF818D351F6287, 0x4A32AD1A07213D85,
                                    0x0706FA71586CFC85 ]),
                  y: GF448::w64be([ 0x0831322F84262A26, 0x200E597199EE27AB,
                                    0x9B3A350FBD9FAB25, 0x852D6D763FBBDA04,
                                    0xC1EF37BE4D8AC7D0, 0x05DE1EC112CB97E2,
                                    0x4804321A7683A80A ]) },
    // (2^150)*B * 4
    PointAffine { x: GF448::w64be([ 0x01942F6983E46CD4, 0xDA719F57928A043D,
                                    0x59710118BF781668, 0xF35A7376ABD868F9,
                                    0x0BD99737D960B0D9, 0xD5E594B352D38ACE,
                                    0xCBDC53EA2E5B2DE2 ]),
                  y: GF448::w64be([ 0x4D9AB23505193CC7, 0x34E4124044CED280,
                                    0x18043E066AA41C70, 0x1C2181AFDB63E9CC,
                                    0x8E3132202809FC82, 0xA104825956B4B5C1,
                                    0xC5AB97FC8728BD76 ]) },
    // (2^150)*B * 5
    PointAffine { x: GF448::w64be([ 0x0667D814E85912E7, 0x7287F2CBD8D1FD84,
                                    0x7601DAA9DA83F37D, 0xDAE8B3FD10EAA27E,
                                    0x6553635922F16BAA, 0x67C87E2772D99BDD,
                                    0xF73361AA8DF14548 ]),
                  y: GF448::w64be([ 0xA2C7BF0543FD3522, 0xC863862F5D1697F2,
                                    0x656FA511752E3A1B, 0x9D78A75A335CFA6E,
                                    0x2C59FA34AF2E2DD9, 0xBDBCDF4B946199B1,
                                    0x032FCF621CB3BB0B ]) },
    // (2^150)*B * 6
    PointAffine { x: GF448::w64be([ 0xFBF732824F438C23, 0xFD6A52AD36E278A3,
                                    0x4303E9E05AE96F4E, 0x8FC3A156D38FFEDC,
                                    0x3B86B2F1EB38177A, 0x7E3A611F7D5EB275,
                                    0xCD56BFEF21985BA7 ]),
                  y: GF448::w64be([ 0x02969FDA2A4F4044, 0xE74B81C9368F6B25,
                                    0x7F10B487ADE28B8E, 0xD2F4269359E3FB92,
                                    0xF2BFB26D07EEC147, 0x5C54061A76F7D8B2,
                                    0xC9485311A0D0F8BA ]) },
    // (2^150)*B * 7
    PointAffine { x: GF448::w64be([ 0x347E3EAB0AB4DBFA, 0x739A20A5CC0A0917,
                                    0x68DB5B548908967E, 0xB1AE1339554376A3,
                                    0xE686A7CF4C0E3672, 0x8B0296BE01269337,
                                    0xABC0F15A8F619F45 ]),
                  y: GF448::w64be([ 0xFF12BCACFF409CEE, 0x408332AA43B3AD51,
                                    0x6E3310E9465DC886, 0x7F26BAE2F1CDC756,
                                    0xFA6F3BFBDF9BDE88, 0xDCD386D1F11FBE71,
                                    0x70055EB9348509D7 ]) },
    // (2^150)*B * 8
    PointAffine { x: GF448::w64be([ 0xD3D972494F792A3D, 0x310BA8AD2E5C5D15,
                                    0x232EC1DCE58C2938, 0xB79AF9CB318AAE64,
                                    0x7A5551E55243C0FF, 0xC788DA21A46B94A2,
                                    0x180BCD42AF9F0C3F ]),
                  y: GF448::w64be([ 0x48DF9D80DF75151D, 0x1D04CD2E242F35F9,
                                    0xE1CBA7D234EE3EDD, 0xCF5F201FEB73C3A5,
                                    0x4DCD3AAA9C9CF69F, 0xB7A12F1ED04EB54D,
                                    0x9DDEB4CA112A9553 ]) },
    // (2^150)*B * 9
    PointAffine { x: GF448::w64be([ 0xA9EA3AC1769EB93E, 0x7981664CDB27E447,
                                    0x446AE04F04CCDDFF, 0x85BBD41DE4B62D2B,
                                    0xB8F4FCF1FE61E6E1, 0x14FE57EFA786F8C1,
                                    0x712965F0E997C5E7 ]),
                  y: GF448::w64be([ 0x838D8091B69DB0D4, 0x23471C5772092C3C,
                                    0xCF750C94FB776A4C, 0xA7FA83938B9DC0D5,
                                    0x5217347556B5F5B1, 0x8171F23F2F87FA72,
                                    0x1D86DA6A01440B1D ]) },
    // (2^150)*B * 10
    PointAffine { x: GF448::w64be([ 0x2A580B5A7A0FA08A, 0xE3BF2811DECA8611,
                                    0x77406032D377DE6F, 0x581E99A5317EEE6D,
                                    0xFDD1D116E1DD381B, 0xEE5EFFF760D1247E,
                                    0xFBD75953A92769EC ]),
                  y: GF448::w64be([ 0xCD75FB05A2C4E85D, 0xBB778E94693DB167,
                                    0x7A8FAD6575A08306, 0x1EB16026F1F20470,
                                    0xDF08C66730B155B7, 0xC6501D035486906E,
                                    0x89E906B9F9F2A3BD ]) },
    // (2^150)*B * 11
    PointAffine { x: GF448::w64be([ 0xD479D9BEAA3A597E, 0x05EBF0B6B7884ED1,
                                    0x714FF884E6A9C4F4, 0x35EAF1AA42104449,
                                    0x8AA4CE7F210E329A, 0x01869A9D27AC848B,
                                    0x5964A3568A832919 ]),
                  y: GF448::w64be([ 0xB74C70BC823FCB6E, 0xC886242DD6F2E7C7,
                                    0x20E5000F922A74B8, 0xF68BE425A8AEBCB4,
                                    0x53906ADA70B7ABA4, 0x80B34036DC9667E3,
                                    0x9DC6CE1C31E60C13 ]) },
    // (2^150)*B * 12
    PointAffine { x: GF448::w64be([ 0xA8F2DF07999AF81E, 0x74DCCAAACFBE7C9D,
                                    0xA37C2CC175F8FF03, 0x35917E0B9AE34CD9,
                                    0xD6CF527645842BBC, 0x4FEDA6AC4BB03AA0,
                                    0x154ECC77DA81DD9A ]),
                  y: GF448::w64be([ 0x2B57D05DAEFBCC5B, 0xD38A20478B65BAA3,
                                    0xE50A1CD104110FAE, 0x6904F81E49EDFFD2,
                                    0x44A6FEC95DB8AEC4, 0xA8632156E87AE190,
                                    0xDDD06C4EA102A466 ]) },
    // (2^150)*B * 13
    PointAffine { x: GF448::w64be([ 0x8892D60C15D35173, 0xAA12A349C66C78D0,
                                    0xB4E09F8603180B38, 0x2D31720FC5CF6F0B,
                                    0xABFC3465D80A027F, 0xA0D9DC349AF56C33,
                                    0x33A922CBEC5AA7DC ]),
                  y: GF448::w64be([ 0x232B509A69025E7C, 0xAF55B349F80BFE74,
                                    0x3A05347765BED2E6, 0xADCD0FF25D54D135,
                                    0x361657FFCA3E495D, 0x6BC3E95C6C802EFD,
                                    0x12AADD280C4D8D62 ]) },
    // (2^150)*B * 14
    PointAffine { x: GF448::w64be([ 0xF5DDE0249C6C621A, 0x7D44474B32CBAF62,
                                    0x315AAC06C210860D, 0xCA3EAECE34962029,
                                    0xD67682A898FA5D17, 0xABEED0FA2406234F,
                                    0x3F4870A3F7FAA510 ]),
                  y: GF448::w64be([ 0x2A685792D6D8C821, 0xA1DAFEE9EA36E3FD,
                                    0x0A866120EAE047B1, 0x633A108248936D34,
                                    0xA4481FB3556E8064, 0xBF713CA7CD2D9E55,
                                    0xB999F765A01BE912 ]) },
    // (2^150)*B * 15
    PointAffine { x: GF448::w64be([ 0x38AADFC7D82FEDF5, 0x602EC5EED4598B57,
                                    0x7F10C70962DC5B7F, 0x1DDFBA88B7713439,
                                    0xDB2B7B276EBCA207, 0xDE5F10978C1B1EA3,
                                    0x57296A1DE85505EF ]),
                  y: GF448::w64be([ 0x4C0C2DA0D90AE9B9, 0x1A672AAE2C3F71A9,
                                    0xF6FDC3B949E16725, 0x76CE5150E9025781,
                                    0xE05EDC768451B74C, 0x6B9B34AFF8ABCD3C,
                                    0xB79FDC0E18EAD832 ]) },
    // (2^150)*B * 16
    PointAffine { x: GF448::w64be([ 0x54048F53F917B6AE, 0xF6541A2377D97447,
                                    0x5CC401A29FE0CE4D, 0x3E2D4F647599AFF3,
                                    0xF0ADD7DD7A4C6997, 0x9436B2A8592414F5,
                                    0xE31CE92BA86F4534 ]),
                  y: GF448::w64be([ 0xFC14E790B6CA4361, 0x033FE2586342E68E,
                                    0xDA0A1ACA4617B1F1, 0x0F7187FAEAB01A86,
                                    0x17FCE4C718369AAC, 0x9320779BA2231493,
                                    0xCB1B86B2205312EC ]) },
];

// Points i*(2^225)*B for i = 1 to 16, in affine format.
static PRECOMP_B225: [PointAffine; 16] = [
    // (2^225)*B * 1
    PointAffine { x: GF448::w64be([ 0x61A15FDBD4BD7107, 0x164A73A44575D449,
                                    0xD1E5627B54A240C1, 0x3AEE9C5507F0EF61,
                                    0x225F552D82E9DCF2, 0xB361FC4D970995EC,
                                    0x8A1D7605C6909EE2 ]),
                  y: GF448::w64be([ 0xA1C3CB8BD9FD574A, 0x8B05E2D256BC78A8,
                                    0xDCEB1C6539ECA503, 0xB5E5DB0B83EE3193,
                                    0x04DB8E3AC222BCE0, 0xB8C868308C77E7E3,
                                    0x2630696B9D3A9FE4 ]) },
    // (2^225)*B * 2
    PointAffine { x: GF448::w64be([ 0x5603FD84F4FDA12F, 0x06CB31F4E15D786C,
                                    0x784EBB013E9EEEE4, 0x431A4592B15CECF9,
                                    0x83EA9A517F88CDF6, 0xEC7BEA712EF24D51,
                                    0x707B8820F4BDE3EE ]),
                  y: GF448::w64be([ 0x50E2340280B91766, 0x61C72B027EACFAF9,
                                    0x2761A477E0E54A60, 0xD51BE8DC7700BDDA,
                                    0xDA5157A4B70B49A4, 0x1A4E274C66A74A4C,
                                    0x09F6790E99E1321F ]) },
    // (2^225)*B * 3
    PointAffine { x: GF448::w64be([ 0x71B31E85634AB8F3, 0x0CEAAB0905856463,
                                    0xDE90D5AABE389F80, 0x77F3BA8E9B004A22,
                                    0x787082F9074A2676, 0x7ACDECA122E9FB96,
                                    0xEFA1EA3259C46FD8 ]),
                  y: GF448::w64be([ 0xED946122B5C03404, 0x19265F0FDE703D5D,
                                    0xE1966C27FD46E443, 0xADC6B089B9B784DF,
                                    0x2BB3280665F7886B, 0x8A59506886E20AC2,
                                    0x520DEE65CAF02AED ]) },
    // (2^225)*B * 4
    PointAffine { x: GF448::w64be([ 0x2F850464731F6137, 0x8B2D14ADF9E59673,
                                    0x4D51530D56358F81, 0xB3EAF48FEF94FDD2,
                                    0x9E7666DBFFA1936B, 0xA8899DD05C48524C,
                                    0xF9CA884CFBF5844B ]),
                  y: GF448::w64be([ 0x8D57FB354DF89521, 0x8925B00AC3012C7B,
                                    0x0D04C3708064681A, 0x2F551C0F2866D99C,
                                    0xC2770764052498F0, 0xF958A4F89E061992,
                                    0x39D6AE90599DCB83 ]) },
    // (2^225)*B * 5
    PointAffine { x: GF448::w64be([ 0xB195D13D6A0B215E, 0xD275AFB93E40E852,
                                    0x3733283591F2DF60, 0xA6DD1ABB2F92955E,
                                    0x269AC33FC19B95B2, 0x4CBA71BF4F5E19A0,
                                    0x3268BAF10F0CBD72 ]),
                  y: GF448::w64be([ 0x2BE92BCC9C7D8EB6, 0xE4491F3BC5746C72,
                                    0x07A860DB2C9EFD60, 0x4CB4BE5361ED01CC,
                                    0xEBE5138687EC7736, 0xFB48C6808AEC9DBA,
                                    0xE26A5819F37F3E5A ]) },
    // (2^225)*B * 6
    PointAffine { x: GF448::w64be([ 0xC5BFEBABE34EBB11, 0xFD61B9EB6758B779,
                                    0x187CFB0329CBAF3D, 0x407B6229EEBAA6C8,
                                    0x915758457354F4A8, 0x6648C327EF9EDB74,
                                    0x5D230C788522EC8F ]),
                  y: GF448::w64be([ 0x31CD6CA1E59BB375, 0x42B99EC06646DB6D,
                                    0xA2B791A253C589C4, 0xA827832DE02CF203,
                                    0xF786DDB43FD0F3FD, 0x85E13161E3F61FDA,
                                    0xA3B4E2BB542D239C ]) },
    // (2^225)*B * 7
    PointAffine { x: GF448::w64be([ 0x05715E730A1933C9, 0xAAFBC86184D5186B,
                                    0x9E59729DEA7BD2B5, 0xF566CA9958F5C8E1,
                                    0x81FFCA610E3F8FBB, 0x9855183CF53E973F,
                                    0x7862E130F8614743 ]),
                  y: GF448::w64be([ 0xB949CC274C4B88D0, 0xDC7E0EA1631EBE22,
                                    0xD283B71DCC7327FD, 0x2963B7F03A584616,
                                    0xCC9EC956CAD904D9, 0x7B6131CF12E164EB,
                                    0xB2F4B632DC4F342E ]) },
    // (2^225)*B * 8
    PointAffine { x: GF448::w64be([ 0x0439679376816A06, 0x604B0305B8060285,
                                    0x0BB55FFFDFF8DC68, 0xCD0639C0A1F71031,
                                    0x934305654D840EA2, 0x101B77DAFA2B35EA,
                                    0x4BF76968758EB660 ]),
                  y: GF448::w64be([ 0xE80946965F1A2E7D, 0x20307CEC6C108ABB,
                                    0x3576044DC57D02A7, 0xE946C6EFFCE7E47F,
                                    0xA229153B571E5EE0, 0x70205FA7073785DA,
                                    0x9FE37B7913627295 ]) },
    // (2^225)*B * 9
    PointAffine { x: GF448::w64be([ 0xAD422D53E68AC6ED, 0x88AD7ADAB6F1C829,
                                    0xA24852B989EBD2AE, 0x7F83643B45A88937,
                                    0x2A2ACC7544015873, 0x9334A43127BC2B39,
                                    0x23F2444095A0FCD1 ]),
                  y: GF448::w64be([ 0xF2A6DF63F102D82D, 0xA96AACB4DA5CE6DB,
                                    0x51414FE04C524181, 0x011740CFD990AAA0,
                                    0xC084D2ECEC105387, 0x66F31DE7B9B3BDD4,
                                    0x3785FA5AA0FA99DC ]) },
    // (2^225)*B * 10
    PointAffine { x: GF448::w64be([ 0x012DDEB5AE496A58, 0x5C2A6661A0EEDDF0,
                                    0xC9290720D63194D4, 0x67F668F86F3CECB6,
                                    0x44EBC41E44BC2AD7, 0x5713BCFEB2815C5B,
                                    0x53F202C5196A9EC2 ]),
                  y: GF448::w64be([ 0x0D2A31353F7162DA, 0x21E60E08C210B93A,
                                    0xAA2CDA76609734A8, 0x4BB17CB10E3E379A,
                                    0xE16CD0D651A6FBFB, 0x7E6B6DF9F0922AA4,
                                    0x5C64C8E4693F834B ]) },
    // (2^225)*B * 11
    PointAffine { x: GF448::w64be([ 0x3B978145EEF978D3, 0xB6E07E88B952572E,
                                    0x9724FCFA5C2C3F79, 0x93DF6194B374742E,
                                    0x11B71F93202AB6F4, 0x44C8B8D07A2E5342,
                                    0xD5FC0C5C77F8716F ]),
                  y: GF448::w64be([ 0xF63B05F29A4D2D2F, 0xDFBBF92FD7AD5B46,
                                    0xC46D518FA822B741, 0xBD3A5753F1B9BF29,
                                    0xDFAB490422112724, 0xA374680CBB77B31E,
                                    0xF215082DB27CA2A8 ]) },
    // (2^225)*B * 12
    PointAffine { x: GF448::w64be([ 0xB2F34AA6675B0574, 0x20E2E18C9CA61C19,
                                    0x3B10B4B251877B62, 0x48B6A74C24D75EC0,
                                    0x3A29E37E9DB41D40, 0x1B1752720E39C79A,
                                    0x825665646E4C4F47 ]),
                  y: GF448::w64be([ 0x1B596F6BE6465D72, 0x734BE8C9CAE8E96A,
                                    0x43A47708915C406B, 0x947E5487833B18BD,
                                    0xFEA681A7992E9EF8, 0xA7DDA035F6C282F3,
                                    0xE43E04357A10A5C0 ]) },
    // (2^225)*B * 13
    PointAffine { x: GF448::w64be([ 0xE1499D7B9B3D4200, 0xB2848127D13551C9,
                                    0xD7EC615628E317CE, 0x20E555BD39EF016E,
                                    0x99D8D9889635CC65, 0x9FA3ABEF602967CD,
                                    0xB64B05D5519C777A ]),
                  y: GF448::w64be([ 0x0E028C95A614A085, 0x82C9350A3B9A071E,
                                    0x22BB831AFC43C9D4, 0xFCC83111C355CE57,
                                    0x5258DB079B668EA8, 0xCDC6666782BE31D3,
                                    0x441D58D9C608074F ]) },
    // (2^225)*B * 14
    PointAffine { x: GF448::w64be([ 0x3AADA1989BF7F450, 0xE87A36EA51BBFA9E,
                                    0x2530686798EAC886, 0xD81A622692D04EB7,
                                    0x8C352FC644268014, 0x28680DB83F74A7DF,
                                    0xA6DA4DF73FF2801B ]),
                  y: GF448::w64be([ 0x6663B9DAE28F4D51, 0x4731A12CC0C99DA2,
                                    0x0473B3395DF5A32A, 0xBA8BEF1329C08078,
                                    0x89AAB483E549E182, 0x22E4D50B00633B2E,
                                    0x83EE50F3A5FDEE62 ]) },
    // (2^225)*B * 15
    PointAffine { x: GF448::w64be([ 0x04AE082AFEC32DAC, 0xF30DB502BD9AA26E,
                                    0x0A504B5ABEBE4FD7, 0x320D9B7433A28676,
                                    0xC1473770CA88525F, 0x902537EB8CEC6CC7,
                                    0x3C83D4B7DF72AA94 ]),
                  y: GF448::w64be([ 0xF7748E4F1D86A719, 0x0B42A792B35B1BDD,
                                    0xCFA217C3424A6AD3, 0xF00C8F6A5D93BF6F,
                                    0xC743BA426FE90D16, 0x63AF3DF6A46E6DBD,
                                    0xDC22F154633C0B48 ]) },
    // (2^225)*B * 16
    PointAffine { x: GF448::w64be([ 0x7F6F6B6410DECCAA, 0x9B22A5DE9857A0C9,
                                    0xAAF4D11639FDE2DA, 0x73E3A887B4DEFF23,
                                    0xE3AC88A516982E8A, 0x44FF36351C8FEE57,
                                    0xDB800065701D5599 ]),
                  y: GF448::w64be([ 0x17FE0CA69BD7D653, 0x88E8325993161DC7,
                                    0xBEDBDF529168B592, 0x8C9318AED2F06A56,
                                    0x36027F799CCA2D67, 0xDF63BFFF90E2E9E9,
                                    0xBC28846A3A626697 ]) },
];

// Points i*(2^300)*B for i = 1 to 16, in affine format.
static PRECOMP_B300: [PointAffine; 16] = [
    // (2^300)*B * 1
    PointAffine { x: GF448::w64be([ 0xCB30B2DFE34FF479, 0x6B84FD21C04F0BEF,
                                    0x8E5346C6DB064D57, 0x115BCF46AA247DB3,
                                    0xE334FF4842AAC396, 0x8483295148F07612,
                                    0xF834DD6C87FC13A6 ]),
                  y: GF448::w64be([ 0x6BCB8351A68AD287, 0xFB17361E9926D7D8,
                                    0xD0A393AEDBB45187, 0xA91F5BC4A1024A71,
                                    0xE4E73DEC0EC43CAA, 0x8A1F57F4AAC8C7EC,
                                    0x6AE7B762A9DE4308 ]) },
    // (2^300)*B * 2
    PointAffine { x: GF448::w64be([ 0x7AD7D58FC74676A1, 0xA30D24FC5B87701A,
                                    0x03C018AF46437E0D, 0xA79C545762817825,
                                    0xC09866C9CECC5FFE, 0x844801908F409192,
                                    0x3272401B6541DFCD ]),
                  y: GF448::w64be([ 0xF37FF5219708EB4E, 0x93CE05D2CB7DF0B9,
                                    0x74527C693E5FE447, 0x8C0742C3DA01DF4F,
                                    0xF838A69802A15E44, 0xB93D299B1445D128,
                                    0xE929607A590F2730 ]) },
    // (2^300)*B * 3
    PointAffine { x: GF448::w64be([ 0x5149368A6D327B12, 0x9A6023BB142863E4,
                                    0x857D91E74C4137DD, 0xDE6AD4B9BFE5065E,
                                    0xBE5F92994D304169, 0xECCB8A414D840E6E,
                                    0x4886F95D5A8B0681 ]),
                  y: GF448::w64be([ 0x0B0467B7C987156D, 0xF4B340062C5F32A7,
                                    0x1755603C7680A0CC, 0xAE3B583414C75D58,
                                    0xFE02353B4118311A, 0x3A0D35D9E605AADC,
                                    0x0240660700CC61D7 ]) },
    // (2^300)*B * 4
    PointAffine { x: GF448::w64be([ 0x5DF918D123D08C66, 0xF7770FE6BB3279CB,
                                    0xD26495C99862A3E0, 0x983EFD067459C177,
                                    0x90428C4CDF9FE320, 0xAF14124E67446613,
                                    0xD99354E26D1CEC2F ]),
                  y: GF448::w64be([ 0x590B617745EB3750, 0xC8791094CB43D3A5,
                                    0x225D02860CEBF02A, 0xF0784CA4B4D6D351,
                                    0x36366920876E32F7, 0x996D7E149E0CB0CF,
                                    0x4ACA4B7D8D788F2C ]) },
    // (2^300)*B * 5
    PointAffine { x: GF448::w64be([ 0x33497B08C810E27E, 0xC8EB702811300C50,
                                    0x83C1FFA724155DF6, 0x01B3111CA3DCAFE2,
                                    0x03085510BB6BBBA4, 0x8D5833013937F32E,
                                    0xC4BBA18727116578 ]),
                  y: GF448::w64be([ 0xE9C011D04BF5421D, 0x6D2906AFE06A3603,
                                    0x63AC5EDA93A1B100, 0x03D0DE50D3D0F9A6,
                                    0xC6B909FCEFDD9CA4, 0xF4122BA081FFA8E3,
                                    0x7BDD5E69ECE59C75 ]) },
    // (2^300)*B * 6
    PointAffine { x: GF448::w64be([ 0x80332B4272B57C77, 0xD2984804DE05274F,
                                    0x898AE7DF31A2D2CD, 0xC343AB61C3DFDD72,
                                    0xCE4B0C07C51493C8, 0xAD5731DA6476D489,
                                    0x0EEA43CCC8387D2E ]),
                  y: GF448::w64be([ 0xCF734CC4B0F589C3, 0xC35AC5867121E203,
                                    0xECB858009DA9A5C8, 0x7C418918D19DFCCD,
                                    0x6372D04279350E42, 0xDBA3B9E96631F611,
                                    0x7BA36934D624AEB5 ]) },
    // (2^300)*B * 7
    PointAffine { x: GF448::w64be([ 0x427682DA71FCA480, 0x80736171B2694986,
                                    0x3EBFB10B60E13671, 0x301DEB00B165E3E9,
                                    0x883FABA6EFBBBB5A, 0x4EBFC1C4F7090BA3,
                                    0x45F31A334FD081BD ]),
                  y: GF448::w64be([ 0xEE7881E27EA1B62F, 0xC497A377C2377650,
                                    0x937708904479397A, 0xCBDC74A2F88D53B1,
                                    0xFA0CF27C21C62EF4, 0xEC8F4CDC5B0AE693,
                                    0x8C305AAEEDCBA2C8 ]) },
    // (2^300)*B * 8
    PointAffine { x: GF448::w64be([ 0x58D7A19CA8D5A47F, 0x0C4B9B6CB3F8193C,
                                    0xDAAD1398B333EF72, 0xFF9D531AD766C07E,
                                    0x83B7DB74BA971136, 0x0CC911651952BA58,
                                    0x99C0DCD94716BACB ]),
                  y: GF448::w64be([ 0x20348FD73BF81B92, 0x04F381C74A227D0F,
                                    0xB9CAD2A68F534A89, 0xFC58C406CCE1D41A,
                                    0x7057AFB32BB3A3C6, 0x56D5DD9402B460E5,
                                    0xB96EF80496443A81 ]) },
    // (2^300)*B * 9
    PointAffine { x: GF448::w64be([ 0xB2B621A968C583C8, 0xD0088997407E4A4C,
                                    0x16193D6476B28187, 0x357F87EC6C12C3DC,
                                    0xA274BAD4CB89EAA3, 0xFAF9584C1073D44C,
                                    0x7F994025242728C5 ]),
                  y: GF448::w64be([ 0xAAD2696E9B469276, 0xFFF92C7E7D59938A,
                                    0xBC94D5F353305459, 0xDC8EE9B0B049D9B3,
                                    0x1903A6A0DB99F942, 0x56E384A1CF43BFB1,
                                    0x751F6CB0095F05DA ]) },
    // (2^300)*B * 10
    PointAffine { x: GF448::w64be([ 0xDB44C833FF2E03B6, 0x80F44271A7CE0E08,
                                    0x63134C74521C8579, 0x95EDD79C674F2E1C,
                                    0x70B415942E889F0C, 0x640EC55311FD2322,
                                    0x7B8684945849F25A ]),
                  y: GF448::w64be([ 0x0B4A2A43D48BEFD4, 0x17DA0D20AF4567FA,
                                    0x946D9F6B85C45A31, 0x40419FC73524E69D,
                                    0x022976C4D17E061C, 0x00FF3D4664542E3B,
                                    0xEC1414460BC23E1A ]) },
    // (2^300)*B * 11
    PointAffine { x: GF448::w64be([ 0x0060DB2C8AAC8878, 0xD89AAE937EA5910F,
                                    0x97129C89BBC86861, 0x045413B59F81433C,
                                    0x2957E6AC8714531A, 0xA36D01A581BFF5DC,
                                    0x02782981FB8EDDAC ]),
                  y: GF448::w64be([ 0xF5E4C8CDF7FC4DAC, 0xFDD1F766A669F094,
                                    0x9220F5BB8E7EC1F8, 0xF100D6775B7AAD66,
                                    0x96A7D9269246E6DC, 0x4618CA3E5EA58F06,
                                    0x2306BEF32B09E547 ]) },
    // (2^300)*B * 12
    PointAffine { x: GF448::w64be([ 0xAB409309FA522B57, 0x8A730479C395F449,
                                    0x49B5CA33FD354810, 0x9C84E59FBF83DFDA,
                                    0x4112AB46EAD3789B, 0xC4B1BCC5E5700A0D,
                                    0x951F1F49EEBB2A6D ]),
                  y: GF448::w64be([ 0x9F01D4AF536FDB41, 0x071AC90C8FAB4111,
                                    0x48B8A9DAAECBE4E9, 0xD33C3A9819D7689E,
                                    0x03A928E243CF521E, 0x50CD20BFE655DBDA,
                                    0xFC1E0587BC1EE7B3 ]) },
    // (2^300)*B * 13
    PointAffine { x: GF448::w64be([ 0x21DA74DAE0DA2A7D, 0xED05EBEE56D035A2,
                                    0x68B170621971C88D, 0x27062D0894D3B1BB,
                                    0xCE3D20B3DF980D67, 0x29E0EE25708E0C0B,
                                    0xCCADF3B692A98FA8 ]),
                  y: GF448::w64be([ 0x35E9BF830938997F, 0xCDCC482E65EF3595,
                                    0x9CC8A444A5D7CBD3, 0xD809A637BAF4936C,
                                    0x2921D499554DAEB1, 0x40A316D222E669F2,
                                    0x25D070358A3A5483 ]) },
    // (2^300)*B * 14
    PointAffine { x: GF448::w64be([ 0x0C542D28639930E3, 0xA0EC640666E7A3C4,
                                    0xD0D163A3E1C9E0ED, 0xA5E0CB802A9ADA89,
                                    0xBE705B28B08FE26D, 0xE7B9080AEBDCBE6B,
                                    0x7B09B1C17D2265F5 ]),
                  y: GF448::w64be([ 0x63D7E448C81B4E76, 0xF2AA968C7C3422F8,
                                    0xDA4D64A90B78A845, 0x4A390FE1650F18FD,
                                    0x58FEC917BF8CFCF1, 0xAFE10D142C71BB1B,
                                    0xE87CC9CD388DEBE5 ]) },
    // (2^300)*B * 15
    PointAffine { x: GF448::w64be([ 0xA535F6F84A4D9FA8, 0x26AD99BC2180388A,
                                    0x1429DECE4043C0BC, 0x403B0DCDEC1DFCC4,
                                    0xFB5022689BA50638, 0x1E9E448B97F56E0C,
                                    0x6816677E3F9F7945 ]),
                  y: GF448::w64be([ 0x0F08B42FBE8250D0, 0x95B73B1C42683429,
                                    0x6DF625C6999ECA9E, 0x7089AF502373591F,
                                    0xD2AF723D8D6D8ED5, 0xE8E5193837F91EA8,
                                    0x157274BE64AC33A8 ]) },
    // (2^300)*B * 16
    PointAffine { x: GF448::w64be([ 0xB5AF09F651B99515, 0x55CCBFF2898505EB,
                                    0x0ECFD787D4C108CF, 0x573FE2E70193FD88,
                                    0x48E04D26556642A5, 0xA761E0069A5C36F7,
                                    0xF40508B340F0B450 ]),
                  y: GF448::w64be([ 0x8BEF69332110FE80, 0x5188365A80E032B6,
                                    0xFADD9A0C1E300FF0, 0xE09B75B0EABBF202,
                                    0xA41B3140FB7166DD, 0x76FAD6D98BF57C66,
                                    0x9A167D72CE1134BE ]) },
];

// Points i*(2^375)*B for i = 1 to 16, in affine format.
static PRECOMP_B375: [PointAffine; 16] = [
    // (2^375)*B * 1
    PointAffine { x: GF448::w64be([ 0xF903D37923D9831D, 0x69CB97C650B25D86,
                                    0x2F3160947FF9BF59, 0x5E97B53B56F7EA03,
                                    0x223E4D4DF414EAF5, 0x95D94CEE65250B9E,
                                    0x03161D66EC4BD38D ]),
                  y: GF448::w64be([ 0x977B45653D87FD5D, 0x0D7E7195400B6E0D,
                                    0x0E4D86DD0DB8946D, 0x58314CE931863CF3,
                                    0x62CF43354A3E4119, 0xD2C6CF9C667496E8,
                                    0x4F49B5476B786836 ]) },
    // (2^375)*B * 2
    PointAffine { x: GF448::w64be([ 0x974A8399A2437A0C, 0x182A13A68351D462,
                                    0x8710F069A6799B3F, 0xBE53184D97109B66,
                                    0x3632C9933D432F7E, 0xBE4C946559C7B264,
                                    0x2F4C88B2BA29629C ]),
                  y: GF448::w64be([ 0xD2F66391A457E881, 0x0440C55365BEEE0F,
                                    0xE4DDE84AE2F92211, 0x5DDFFB0FCC3523F2,
                                    0xB1DA15D85A60B08F, 0x4C3701B98B6D9C42,
                                    0x4B29F19972A70278 ]) },
    // (2^375)*B * 3
    PointAffine { x: GF448::w64be([ 0x07B42010AEB9ADA2, 0xD8181A4022AB24AE,
                                    0x3D53FD753FDE91CD, 0x22F7DED65E934344,
                                    0x31A3F05BC27146E1, 0x3B5D4B99B28E0626,
                                    0x94742F6631ED390A ]),
                  y: GF448::w64be([ 0xDC5B71E88F8AE200, 0x0235A91F04BEB171,
                                    0xEFC787786DBD70FC, 0xDE2A2E992BA6EEB4,
                                    0xADD01268E5E92DA3, 0x678D47FBA626875D,
                                    0xC5CCF57AEB62B3C9 ]) },
    // (2^375)*B * 4
    PointAffine { x: GF448::w64be([ 0x7EB85BBDF7AA5045, 0x37F9EA9DEE05372F,
                                    0xDE610F3FB2A5D1F6, 0x591F6CBB8A03B3F7,
                                    0x0DF4BD7DEDD63A5C, 0xEDE892A7545ABDFC,
                                    0x615E6879FE2DDD05 ]),
                  y: GF448::w64be([ 0x626720DEB405EF8D, 0x314509B84763ADB6,
                                    0xCF005860DD2AA748, 0xD0270A5475533400,
                                    0xC55BDEA246E4C6FE, 0xDF2D53C8DCAE7BDB,
                                    0x6B963EDF8E8C504D ]) },
    // (2^375)*B * 5
    PointAffine { x: GF448::w64be([ 0x8CA5B4D31806F729, 0xB6E6B8988CE97D4D,
                                    0xBBD5C3C8467CADC8, 0x13D2AD330B6AA4E0,
                                    0x101A2160C0D34624, 0x644E1BC8867400FB,
                                    0xDF14FE28DEEA8203 ]),
                  y: GF448::w64be([ 0x0701817FE55A196E, 0x7CF109F3C24A8763,
                                    0xE86B7080A7E9F6EF, 0x87CF47AC6373B6A7,
                                    0xC4BBEF9477267341, 0x2D0B6DACA86E216B,
                                    0x4BBE9E99EE2EDBF6 ]) },
    // (2^375)*B * 6
    PointAffine { x: GF448::w64be([ 0xAA38895244A39D79, 0xF76581D3762F8C58,
                                    0x12261792A849E7B0, 0xD15DF607FA467AF3,
                                    0x92C19838793439D5, 0xD24768E94FD2AC24,
                                    0x78A3709AE6601328 ]),
                  y: GF448::w64be([ 0x58EB0A0192818C83, 0x3F022FF418931DDA,
                                    0x8B91C6F7B9E52B05, 0x703651AAC608AFCE,
                                    0x64489EE3E0B1F251, 0x015D2B0DB53A33B3,
                                    0xBBEF60AF9C5CD0BC ]) },
    // (2^375)*B * 7
    PointAffine { x: GF448::w64be([ 0x043F850B74FD5DDC, 0xED916A2CEAB0E5FE,
                                    0xC1AF746C84131B00, 0x32D3CCAED74B767D,
                                    0xF498DDF4F2101BFD, 0xE3FA6D0D5800C033,
                                    0xCAE486437D2DDC29 ]),
                  y: GF448::w64be([ 0x4D521748532091E6, 0x34BA57E6D888AAD1,
                                    0x54BD0D5DFDA80CDA, 0x8A854AD57BC6982D,
                                    0x6975890A78C29712, 0x086850FF6795EEC4,
                                    0x946391B36918384F ]) },
    // (2^375)*B * 8
    PointAffine { x: GF448::w64be([ 0x35C0E081E02E1AC1, 0x0A2EE1B8153B11B2,
                                    0x4AEDBE47B12E9C16, 0xC19E2F736C64559B,
                                    0xA60D557B3CD019C0, 0x2EEF6D2FFE1A83AB,
                                    0x336C1300CFC7B5A7 ]),
                  y: GF448::w64be([ 0x057AADC3B8C0577E, 0x4E5A59231142BF25,
                                    0x4DA417BAC1378A64, 0xA1BE2F8F6CD88ADB,
                                    0x85AABDAC240F825E, 0xE03F4EA93B6F41A2,
                                    0x90A9F470C1DD6F16 ]) },
    // (2^375)*B * 9
    PointAffine { x: GF448::w64be([ 0x18118C6E1F420F32, 0xE84EC364CD9B961D,
                                    0x16249A983BF85326, 0xB5F6977B5BC8C674,
                                    0xDE78A8EBE0BB7192, 0x6F3863BD7A8466B8,
                                    0xEDC2CF6AAD812AF8 ]),
                  y: GF448::w64be([ 0x94D0128D79FFCC74, 0x637416234F941F86,
                                    0x055740D7EBA9675F, 0xCCDC4675B9065DDD,
                                    0x35D38B7519DD011B, 0x768ACD8A7A5FC601,
                                    0x3BC56FFEE1DF6C0A ]) },
    // (2^375)*B * 10
    PointAffine { x: GF448::w64be([ 0x16E05F57B25C2FC1, 0x70D6939E601C35DA,
                                    0xF2E4640B5292A788, 0xDB72D782D2A7BE12,
                                    0xFC50A69680AAF101, 0xB4C7D3E01FF5CCDF,
                                    0x74607C77A80AF479 ]),
                  y: GF448::w64be([ 0x868ABE3592853A2F, 0x261FB09931546847,
                                    0x758FCEA0B1D657D6, 0x35BB4783FB8DA82A,
                                    0xA53873401E11E179, 0xDA16EB74271BECA2,
                                    0x9847A42A66FE37F8 ]) },
    // (2^375)*B * 11
    PointAffine { x: GF448::w64be([ 0x4306315FA7A87116, 0x247F65947B73BD46,
                                    0xE30AC755ECE9949F, 0xA582B707585054E2,
                                    0xF0DEAED7EE5B563F, 0x56485B519F8E7B66,
                                    0xC1C64ECF59CCE468 ]),
                  y: GF448::w64be([ 0x1AAA13DF127EDECA, 0xA0BA88985E459504,
                                    0x7E6627CE8638E320, 0xD7130EE74222E9FE,
                                    0x221CD98D1D6B9031, 0xC02A1D5CDB36847D,
                                    0xE3C9EE20584BE518 ]) },
    // (2^375)*B * 12
    PointAffine { x: GF448::w64be([ 0x34781416AC17110B, 0x278494F2A37DCAEC,
                                    0x82DC262B4D207877, 0xFFD967D34A7B1E29,
                                    0x8B15A25D737A660B, 0xA199A1C84D66F4E4,
                                    0xEA1A4C54335766AB ]),
                  y: GF448::w64be([ 0x789747EC129690E9, 0xF90D99D902145B9C,
                                    0x8C24A53332ABAFA7, 0xD5C0E7CE1DCB53B2,
                                    0xD776D32BD705BFF3, 0x9EBDBEC03F816D00,
                                    0x3F28E3DF18FC6856 ]) },
    // (2^375)*B * 13
    PointAffine { x: GF448::w64be([ 0x1A925B4A16F6B7E0, 0x61B01308D66ACD0B,
                                    0x7DC1473FAE467324, 0x4D88B4D8081FACCD,
                                    0x1BBCA4BC548C6885, 0x2FE5A1E4FBAAD28E,
                                    0x6C76372BD4DE32A4 ]),
                  y: GF448::w64be([ 0xE788BC63724294CD, 0x57717F3ADD26B00D,
                                    0x13F841950F925B3D, 0x62E68000F661B1FA,
                                    0xE70DEB40EA248F3E, 0xAAA2BD4F931F7702,
                                    0xC1FCA5277A52C028 ]) },
    // (2^375)*B * 14
    PointAffine { x: GF448::w64be([ 0x104E5016657B9525, 0x28150E34E7819586,
                                    0x7C00F90242A23E5E, 0x8288F337CDB4D14A,
                                    0x165C37159EF0534E, 0x3927919F406AFB2A,
                                    0x8F94D3C3954E2DFA ]),
                  y: GF448::w64be([ 0xE18FA9C62188BED6, 0x5F4D4CD4CD4A129A,
                                    0x1928C716A13075AD, 0xA20C811FEC63F7DF,
                                    0x084A29CBCE4F516A, 0x60F8609B99523EB5,
                                    0xFA695A6C9BCDDA24 ]) },
    // (2^375)*B * 15
    PointAffine { x: GF448::w64be([ 0x2B46E89AACABD017, 0x6C8441C529FE8342,
                                    0xE256C05AE49FA0F1, 0x8072176BC07924A7,
                                    0xFCC3C2CB5202CAD5, 0x9D25785FFCE0CB49,
                                    0x339B66464D1347C9 ]),
                  y: GF448::w64be([ 0x81F3B3FE34940E17, 0x31077C165F4814BE,
                                    0xCEC819632B4FB600, 0xB10AD4F315C18527,
                                    0xF51BE4C3458D423C, 0xC559BA12CACABAF8,
                                    0x577E8C27A094020F ]) },
    // (2^375)*B * 16
    PointAffine { x: GF448::w64be([ 0x8A73E3450E20E64F, 0x3ED0A22287DC783C,
                                    0xBDC5962AC0884F17, 0x3CBA2FCF7A771263,
                                    0x1E440838FC997887, 0x055B525B3B957714,
                                    0x4B1672757BAC60E3 ]),
                  y: GF448::w64be([ 0xD6499CCCAEC313DE, 0x47BBD87528A98B5A,
                                    0x9E8783F13C068523, 0xD5B37BF4ED478D0C,
                                    0xF363CB0EA08E9AA1, 0x3ED68997D8D0D58E,
                                    0xE6E7A1CD0D764583 ]) },
];

// ========================================================================

#[cfg(test)]
mod tests {

    use super::{Point, Scalar, PrivateKey, PublicKey};
    use sha3::{Shake256, digest::{Update, ExtendableOutputReset, XofReader}};

    /* unused
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
        print_gf("  X", P.X);
        print_gf("  Y", P.Y);
        print_gf("  Z", P.Z);
    }
    */

    #[test]
    fn base_arith() {
        // For a point P (randomly generated on the curve with Sage),
        // points i*P for i = 0 to 6, encoded.
        const EPP: [[u8; 57]; 7] = [
            [
                0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ], [
                0x8D, 0xCF, 0x19, 0x75, 0x8C, 0x0B, 0x73, 0xED, 0x64, 0x1C,
                0xCB, 0xA8, 0x09, 0x19, 0x96, 0x74, 0x51, 0x20, 0x59, 0x1C,
                0xAE, 0xAD, 0xA9, 0x4E, 0x05, 0x5B, 0x00, 0x8E, 0xE5, 0x41,
                0x24, 0xD6, 0x66, 0x2F, 0xE7, 0xC3, 0x38, 0xAA, 0xF3, 0xDF,
                0x2B, 0xE7, 0x43, 0xED, 0x0F, 0x64, 0x19, 0x6B, 0xF4, 0x51,
                0x91, 0x79, 0xE6, 0x61, 0x79, 0xA0, 0x80,
            ], [
                0x01, 0xB4, 0x4C, 0xBD, 0xF8, 0x95, 0x8B, 0x34, 0x98, 0xA7,
                0x4E, 0x9A, 0x4A, 0xCE, 0x09, 0x23, 0x61, 0x57, 0xC9, 0xB5,
                0x3F, 0xC9, 0x8A, 0x4D, 0x3D, 0xA4, 0xA9, 0xE7, 0x7C, 0xE9,
                0x8E, 0xB1, 0x0D, 0x13, 0xE9, 0x4C, 0x48, 0x12, 0x4D, 0xC9,
                0xDF, 0xCB, 0x91, 0xDA, 0x0F, 0x21, 0xC8, 0x17, 0x55, 0xE1,
                0xC4, 0xF9, 0xED, 0xE1, 0xD5, 0xCB, 0x80,
            ], [
                0xA8, 0x00, 0xE1, 0x95, 0x4A, 0xF3, 0x26, 0x17, 0x7C, 0x3D,
                0xF5, 0xC7, 0xBD, 0x40, 0x05, 0xDA, 0xBC, 0x90, 0xD8, 0xD9,
                0x2D, 0x1B, 0x12, 0x39, 0x54, 0x2E, 0xE5, 0x92, 0x63, 0xC8,
                0xF1, 0x29, 0xA1, 0x6C, 0xCF, 0x9F, 0x08, 0x18, 0xCF, 0xA9,
                0xA8, 0x60, 0x14, 0xF7, 0x24, 0xA5, 0x7D, 0xD7, 0x78, 0xA1,
                0xFE, 0x97, 0x11, 0xE2, 0x6A, 0x5E, 0x80,
            ], [
                0x9D, 0x3E, 0x45, 0xDE, 0x29, 0x44, 0xF6, 0x51, 0xA7, 0xD7,
                0x5C, 0x8B, 0x6C, 0x6A, 0x4B, 0x36, 0x3A, 0xA4, 0x1B, 0x52,
                0x14, 0x70, 0xDF, 0xE8, 0x4D, 0x83, 0xF7, 0xFB, 0x5B, 0x86,
                0x88, 0x55, 0x60, 0x05, 0xA5, 0x93, 0x45, 0x0D, 0xD4, 0x96,
                0xF5, 0xC2, 0xBB, 0xC5, 0x90, 0xC3, 0xB3, 0x90, 0xB9, 0x48,
                0x34, 0xD0, 0xB9, 0xF8, 0xE8, 0xB4, 0x80,
            ], [
                0xDE, 0x38, 0x73, 0xCB, 0x14, 0x84, 0x99, 0x05, 0x43, 0x78,
                0xAA, 0xDA, 0x27, 0x6E, 0xFB, 0x1A, 0x1E, 0x08, 0x2A, 0x20,
                0xB7, 0x67, 0x3B, 0x3F, 0x1A, 0x9A, 0x1F, 0x8F, 0x4E, 0xA7,
                0x2C, 0x44, 0x3A, 0xE8, 0x7D, 0xCB, 0x2A, 0xE6, 0xAF, 0x97,
                0x53, 0xE5, 0x3E, 0xC9, 0x16, 0x55, 0x36, 0xC4, 0xB2, 0x17,
                0x4A, 0x89, 0x91, 0x81, 0x1A, 0x1B, 0x00,
            ], [
                0x71, 0x21, 0xC4, 0xD1, 0x1D, 0xE4, 0x7D, 0x02, 0x3F, 0x67,
                0xEF, 0x82, 0x5B, 0x7B, 0x3F, 0x65, 0xF0, 0xB9, 0xB9, 0x98,
                0x29, 0xBE, 0x14, 0x57, 0x10, 0xFC, 0x24, 0x3B, 0xBA, 0x89,
                0xE1, 0x7B, 0xF1, 0xAF, 0xFA, 0xCB, 0x99, 0xBB, 0x10, 0xD2,
                0x5F, 0x5E, 0x30, 0xAB, 0xD0, 0x84, 0x93, 0x39, 0x87, 0xBE,
                0x6D, 0xF7, 0xD6, 0x4C, 0x60, 0x57, 0x80,
            ],
        ];

        let mut PP = [Point::NEUTRAL; 7];
        for i in 0..7 {
            let P = Point::decode(&EPP[i][..]).unwrap();
            assert!(EPP[i] == P.encode());
            PP[i] = P;
            if i == 0 {
                assert!(P.isneutral() == 0xFFFFFFFF);
            } else {
                assert!(P.isneutral() == 0x00000000);
            }
        }

        let P0 = PP[0];
        let P1 = PP[1];
        let P2 = PP[2];
        let P3 = PP[3];
        let P4 = PP[4];
        let P5 = PP[5];
        let P6 = PP[6];

        for i in 1..7 {
            assert!(PP[i].equals(PP[i - 1]) == 0);
            let Q = PP[i - 1] + PP[1];
            assert!(PP[i].equals(Q) == 0xFFFFFFFF);
            assert!((Q + Point::NEUTRAL).equals(Q) == 0xFFFFFFFF);
            let R = Q + P0;
            assert!(PP[i].equals(R) == 0xFFFFFFFF);
        }

        let Q2 = P1 + P1;
        assert!(Q2.encode() == EPP[2]);
        assert!(Q2.equals(P2) == 0xFFFFFFFF);
        let R2 = P1.double();
        assert!(R2.encode() == EPP[2]);
        assert!(R2.equals(P2) == 0xFFFFFFFF);
        assert!(R2.equals(Q2) == 0xFFFFFFFF);

        let Q3 = P2 + P1;
        assert!(Q3.encode() == EPP[3]);
        assert!(Q3.equals(P3) == 0xFFFFFFFF);
        let R3 = Q2 + P1;
        assert!(R3.encode() == EPP[3]);
        assert!(R3.equals(P3) == 0xFFFFFFFF);
        assert!(R3.equals(Q3) == 0xFFFFFFFF);

        let Q4 = Q2.double();
        assert!(Q4.encode() == EPP[4]);
        assert!(Q4.equals(P4) == 0xFFFFFFFF);
        let R4 = P1.xdouble(2);
        assert!(R4.encode() == EPP[4]);
        assert!(R4.equals(P4) == 0xFFFFFFFF);
        assert!(R4.equals(Q4) == 0xFFFFFFFF);
        let R4 = P1 + Q3;
        assert!(R4.encode() == EPP[4]);
        assert!(R4.equals(P4) == 0xFFFFFFFF);
        assert!(R4.equals(Q4) == 0xFFFFFFFF);

        let Q5 = Q3 + R2;
        assert!(Q5.encode() == EPP[5]);
        assert!(Q5.equals(P5) == 0xFFFFFFFF);
        let R5 = R3 + Q2;
        assert!(R5.encode() == EPP[5]);
        assert!(R5.equals(P5) == 0xFFFFFFFF);
        assert!(R5.equals(Q5) == 0xFFFFFFFF);

        assert!((R5 - Q3).equals(Q2) == 0xFFFFFFFF);

        let Q6 = Q3.double();
        assert!(Q6.encode() == EPP[6]);
        assert!(Q6.equals(P6) == 0xFFFFFFFF);
        let R6 = Q2 + Q4;
        assert!(R6.encode() == EPP[6]);
        assert!(R6.equals(P6) == 0xFFFFFFFF);
        assert!(R6.equals(Q6) == 0xFFFFFFFF);
    }

    #[test]
    fn mulgen() {
        // Test vector generated randomly with Sage.
        let s = Scalar::from_w64be([ 0x39D0079B0AD77868, 0x3B665ADEB2FB084F,
                                     0x2726C2C4802203B0, 0x1D2FF77F290A90D5,
                                     0x16B96E1A808A6B3D, 0x5CB74DB41301996A,
                                     0xE3DA32CF511A55BC ]);
        let enc: [u8; 57] = [
            0x38, 0x7B, 0xF4, 0x5E, 0xF3, 0x62, 0xDC, 0x53, 0x1A, 0x49,
            0xF7, 0x93, 0x86, 0x6A, 0xC5, 0x73, 0x40, 0x72, 0xB5, 0x96,
            0x5A, 0x84, 0xB0, 0x57, 0xA8, 0xDC, 0x7E, 0x77, 0x98, 0x04,
            0x51, 0x23, 0xD4, 0xD7, 0x1B, 0x5F, 0x42, 0x24, 0x87, 0x2C,
            0x4F, 0xB9, 0x1D, 0x68, 0xEB, 0x54, 0x16, 0xC1, 0xEF, 0x29,
            0x9A, 0x2A, 0x4F, 0xEF, 0x51, 0x6C, 0x80,
        ];

        let R = Point::decode(&enc).unwrap();
        let P = Point::BASE * s;
        assert!(P.equals(R) == 0xFFFFFFFF);
        assert!(P.encode() == enc);
        let Q = Point::mulgen(&s);
        assert!(Q.equals(R) == 0xFFFFFFFF);
        assert!(Q.encode() == enc);
    }

    #[test]
    fn mul() {
        let mut sh = Shake256::default();
        for i in 0..20 {
            // Build pseudorandom s1 and s2
            let mut v1 = [0u8; 64];
            sh.update(&((2 * i + 0) as u64).to_le_bytes());
            sh.finalize_xof_reset().read(&mut v1);
            let mut v2 = [0u8; 64];
            sh.update(&((2 * i + 1) as u64).to_le_bytes());
            sh.finalize_xof_reset().read(&mut v2);

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
    }

    #[test]
    fn mul_add_mulgen() {
        let mut sh = Shake256::default();
        for i in 0..20 {
            // Build pseudorandom A, u and v
            let mut v1 = [0u8; 64];
            sh.update(&((3 * i + 0) as u64).to_le_bytes());
            sh.finalize_xof_reset().read(&mut v1);
            let mut v2 = [0u8; 64];
            sh.update(&((3 * i + 1) as u64).to_le_bytes());
            sh.finalize_xof_reset().read(&mut v2);
            let mut v3 = [0u8; 64];
            sh.update(&((3 * i + 2) as u64).to_le_bytes());
            sh.finalize_xof_reset().read(&mut v3);
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
    fn verify_helper() {
        // Low-order points (encoded).
        const LOW_ENC: [[u8; 57]; 4] = [
            [
                0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
            [
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
            [
                0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF,
                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00,
            ],
            [
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80,
            ],
        ];

        let mut low = [Point::NEUTRAL; 4];
        for i in 0..4 {
            low[i] = Point::decode(&LOW_ENC[i]).unwrap();
            assert!(low[i].has_low_order() == 0xFFFFFFFF);
        }

        let mut sh = Shake256::default();
        for i in 0..20 {
            // Build pseudorandom A, s and k
            // Compute R = s*B - k*A
            let mut v1 = [0u8; 64];
            sh.update(&((3 * i + 0) as u64).to_le_bytes());
            sh.finalize_xof_reset().read(&mut v1);
            let mut v2 = [0u8; 64];
            sh.update(&((3 * i + 1) as u64).to_le_bytes());
            sh.finalize_xof_reset().read(&mut v2);
            let mut v3 = [0u8; 64];
            sh.update(&((3 * i + 2) as u64).to_le_bytes());
            sh.finalize_xof_reset().read(&mut v3);
            let A = Point::mulgen(&Scalar::decode_reduce(&v1));
            let s = Scalar::decode_reduce(&v2);
            let k = Scalar::decode_reduce(&v3);
            let R = Point::mulgen(&s) - k * A;

            for j in 0..4 {
                // The equation must be verified even if we add
                // low-order points to either A or R.
                assert!(A.verify_helper_vartime(&(R + low[j]), &s, &k));
                assert!((A + low[j]).verify_helper_vartime(&R, &s, &k));
                let j2 = (j + i) & 3;
                assert!((A + low[j]).verify_helper_vartime(&(R + low[j2]), &s, &k));
            }

            // The equation must NOT match if we change k or s.
            assert!(!A.verify_helper_vartime(&R, &(s + Scalar::ONE), &k));
            assert!(!A.verify_helper_vartime(&R, &s, &(k + Scalar::ONE)));
        }
    }

    struct Ed448TestVector<'a> {
        s: &'a str,
        Q: &'a str,
        m: &'a str,
        ph: bool,
        ctx: &'a str,
        sig: &'a str,
    }

    // Test vectors from RFC 8032.
    const TEST_VECTORS: [Ed448TestVector; 6] = [
        // Empty message, empty context.
        Ed448TestVector {
            s:   "6c82a562cb808d10d632be89c8513ebf6c929f34ddfa8c9f63c9960ef6e348a3528c8a3fcc2f044e39a3fc5b94492f8f032e7549a20098f95b",
            Q:   "5fd7449b59b461fd2ce787ec616ad46a1da1342485a70e1f8a0ea75d80e96778edf124769b46c7061bd6783df1e50f6cd1fa1abeafe8256180",
            m:   "",
            ph:  false,
            ctx: "",
            sig: "533a37f6bbe457251f023c0d88f976ae2dfb504a843e34d2074fd823d41a591f2b233f034f628281f2fd7a22ddd47d7828c59bd0a21bfd3980ff0d2028d4b18a9df63e006c5d1c2d345b925d8dc00b4104852db99ac5c7cdda8530a113a0f4dbb61149f05a7363268c71d95808ff2e652600",
        },
        // 1-byte message, empty context.
        Ed448TestVector {
            s:   "c4eab05d357007c632f3dbb48489924d552b08fe0c353a0d4a1f00acda2c463afbea67c5e8d2877c5e3bc397a659949ef8021e954e0a12274e",
            Q:   "43ba28f430cdff456ae531545f7ecd0ac834a55d9358c0372bfa0c6c6798c0866aea01eb00742802b8438ea4cb82169c235160627b4c3a9480",
            m:   "03",
            ph:  false,
            ctx: "",
            sig: "26b8f91727bd62897af15e41eb43c377efb9c610d48f2335cb0bd0087810f4352541b143c4b981b7e18f62de8ccdf633fc1bf037ab7cd779805e0dbcc0aae1cbcee1afb2e027df36bc04dcecbf154336c19f0af7e0a6472905e799f1953d2a0ff3348ab21aa4adafd1d234441cf807c03a00",
        },
        // 1-byte message, 3-byte context.
        Ed448TestVector {
            s:   "c4eab05d357007c632f3dbb48489924d552b08fe0c353a0d4a1f00acda2c463afbea67c5e8d2877c5e3bc397a659949ef8021e954e0a12274e",
            Q:   "43ba28f430cdff456ae531545f7ecd0ac834a55d9358c0372bfa0c6c6798c0866aea01eb00742802b8438ea4cb82169c235160627b4c3a9480",
            m:   "03",
            ph:  false,
            ctx: "666f6f",
            sig: "d4f8f6131770dd46f40867d6fd5d5055de43541f8c5e35abbcd001b32a89f7d2151f7647f11d8ca2ae279fb842d607217fce6e042f6815ea000c85741de5c8da1144a6a1aba7f96de42505d7a7298524fda538fccbbb754f578c1cad10d54d0d5428407e85dcbc98a49155c13764e66c3c00",
        },
        // 256-byte message, empty context.
        Ed448TestVector {
            s:   "2ec5fe3c17045abdb136a5e6a913e32ab75ae68b53d2fc149b77e504132d37569b7e766ba74a19bd6162343a21c8590aa9cebca9014c636df5",
            Q:   "79756f014dcfe2079f5dd9e718be4171e2ef2486a08f25186f6bff43a9936b9bfe12402b08ae65798a3d81e22e9ec80e7690862ef3d4ed3a00",
            m:   "15777532b0bdd0d1389f636c5f6b9ba734c90af572877e2d272dd078aa1e567cfa80e12928bb542330e8409f3174504107ecd5efac61ae7504dabe2a602ede89e5cca6257a7c77e27a702b3ae39fc769fc54f2395ae6a1178cab4738e543072fc1c177fe71e92e25bf03e4ecb72f47b64d0465aaea4c7fad372536c8ba516a6039c3c2a39f0e4d832be432dfa9a706a6e5c7e19f397964ca4258002f7c0541b590316dbc5622b6b2a6fe7a4abffd96105eca76ea7b98816af0748c10df048ce012d901015a51f189f3888145c03650aa23ce894c3bd889e030d565071c59f409a9981b51878fd6fc110624dcbcde0bf7a69ccce38fabdf86f3bef6044819de11",
            ph:  false,
            ctx: "",
            sig: "c650ddbb0601c19ca11439e1640dd931f43c518ea5bea70d3dcde5f4191fe53f00cf966546b72bcc7d58be2b9badef28743954e3a44a23f880e8d4f1cfce2d7a61452d26da05896f0a50da66a239a8a188b6d825b3305ad77b73fbac0836ecc60987fd08527c1a8e80d5823e65cafe2a3d00",
        },
        // 3-byte message, pre-hashed, empty context.
        Ed448TestVector {
            s:   "833fe62409237b9d62ec77587520911e9a759cec1d19755b7da901b96dca3d42ef7822e0d5104127dc05d6dbefde69e3ab2cec7c867c6e2c49",
            Q:   "259b71c19f83ef77a7abd26524cbdb3161b590a48f7d17de3ee0ba9c52beb743c09428a131d6b1b57303d90d8132c276d5ed3d5d01c0f53880",
            m:   "616263",
            ph:  true,
            ctx: "",
            sig: "822f6901f7480f3d5f562c592994d9693602875614483256505600bbc281ae381f54d6bce2ea911574932f52a4e6cadd78769375ec3ffd1b801a0d9b3f4030cd433964b6457ea39476511214f97469b57dd32dbc560a9a94d00bff07620464a3ad203df7dc7ce360c3cd3696d9d9fab90f00",
        },
        Ed448TestVector {
            s:   "833fe62409237b9d62ec77587520911e9a759cec1d19755b7da901b96dca3d42ef7822e0d5104127dc05d6dbefde69e3ab2cec7c867c6e2c49",
            Q:   "259b71c19f83ef77a7abd26524cbdb3161b590a48f7d17de3ee0ba9c52beb743c09428a131d6b1b57303d90d8132c276d5ed3d5d01c0f53880",
            m:   "616263",
            ph:  true,
            ctx: "666f6f",
            sig: "c32299d46ec8ff02b54540982814dce9a05812f81962b649d528095916a2aa481065b1580423ef927ecf0af5888f90da0f6a9a85ad5dc3f280d91224ba9911a3653d00e484e2ce232521481c8658df304bb7745a73514cdb9bf3e15784ab71284f8d0704a608c54a6b62d97beb511d132100",
        },
    ];

    #[test]
    fn signatures() {
        for tv in TEST_VECTORS.iter() {
            let seed = hex::decode(tv.s).unwrap();
            let Q_enc = hex::decode(tv.Q).unwrap();
            let msg = hex::decode(tv.m).unwrap();
            let ctx = hex::decode(tv.ctx).unwrap();
            let mut sig = [0u8; 114];
            hex::decode_to_slice(tv.sig, &mut sig[..]).unwrap();

            let skey = PrivateKey::from_seed(&seed[..]);
            assert!(&Q_enc[..] == skey.public_key.encode());
            if tv.ph {
                let mut sh = Shake256::default();
                sh.update(&msg[..]);
                let mut hm = [0u8; 64];
                sh.finalize_xof_reset().read(&mut hm);
                assert!(skey.sign_ph(&ctx[..], &hm) == sig);
            } else {
                assert!(skey.sign_ctx(&ctx[..], &msg[..]) == sig);
                if ctx.len() == 0 {
                    assert!(skey.sign_raw(&msg[..]) == sig);
                }
            }

            let pkey = PublicKey::decode(&Q_enc[..]).unwrap();
            if tv.ph {
                let mut sh = Shake256::default();
                sh.update(&msg[..]);
                let mut hm = [0u8; 64];
                sh.finalize_xof_reset().read(&mut hm);
                assert!(pkey.verify_ph(&sig, &ctx[..], &hm));
                assert!(!pkey.verify_ph(&sig, &[1u8], &hm));
                hm[42] ^= 0x08;
                assert!(!pkey.verify_ph(&sig, &ctx[..], &hm));
            } else {
                assert!(pkey.verify_ctx(&sig, &ctx[..], &msg[..]));
                assert!(!pkey.verify_ctx(&sig, &[1u8], &msg[..]));
                assert!(!pkey.verify_ctx(&sig, &ctx[..], &[0u8]));
                if ctx.len() == 0 {
                    assert!(pkey.verify_raw(&sig, &msg[..]));
                }
            }
        }
    }

    #[test]
    fn in_subgroup() {
        // A generator for the low-order points (i.e. a point of order
        // exactly 4).
        let T4_enc = [0u8; 57];
        let T4 = Point::decode(&T4_enc).unwrap();
        assert!(T4.isneutral() == 0x00000000);
        assert!(T4.double().isneutral() == 0x00000000);
        assert!(T4.xdouble(2).isneutral() == 0xFFFFFFFF);
        let mut sh = Shake256::default();
        for i in 0..30 {
            let mut P = Point::NEUTRAL;
            if i > 0 {
                sh.update(&(i as u64).to_le_bytes());
                let mut v = [0u8; 64];
                sh.finalize_xof_reset().read(&mut v);
                P.set_mulgen(&Scalar::decode_reduce(&v[..]));
            }
            assert!(P.is_in_subgroup() == 0xFFFFFFFF);
            for _ in 0..3 {
                P += T4;
                assert!(P.is_in_subgroup() == 0x00000000);
            }
        }
    }
}
