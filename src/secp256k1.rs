//! secp256k1 curve implementation.
//!
//! This module implements generic group operations on the secp256k1
//! elliptic curve, a short Weierstraß curve with equation `y^2 = x^3 + 7`.
//! This curve is standardized in SEC 2.
//!
//! The curve has prime order. "Scalars" are integers modulo that prime
//! order, and are implemented by the `Scalar` structure. This structure
//! supports the usual arithmetic operators (`+`, `-`, `*`, `/`, and the
//! compound assignments `+=`, `-=`, `*=` and `/=`).
//!
//! A point on the curve is represented by the `Point` structure. The
//! additive arithmetic operators can be applied on `Point` instances
//! (`+`, `-`, `+=`, `-=`); multiplications by an integer (`u64` type) or
//! by a scalar (`Scalar` type) are also supported with the `*` and `*=`
//! operators. Point doublings can be performed with the `double()`
//! function (which is somewhat faster than general addition), and
//! additional optimizations are obtained in the context of multiple
//! successive doublings by calling the `xdouble()` function. All these
//! operations are implemented with fully constant-time code and are
//! complete, i.e. they work with all points, even when adding a point
//! with itself or when operations involve the curve point-at-infinity
//! (the neutral element for the curve as a group).
//!
//! Scalars can be encoded over 32 bytes, using unsigned
//! **little-endian** convention) and decoded back. Encoding is always
//! canonical, and decoding always verifies that the value is indeed in
//! the canonical range. Take care that many standards related to
//! secp256k1 tend to use big-endian for encoding scalars (and often use
//! a variable-length encoding, e.g. an ASN.1 `INTEGER`).
//!
//! Points can be encoded in compressed (33 bytes) or uncompressed (65
//! bytes) formats. These formats internally use big-endian. The nominal
//! encoding of the point-at-infinity is a single byte of value 0x00; the
//! `encode_compressed()` and `encode_uncompressed()` functions cannot
//! produce that specific encoding (since they produce fixed-length
//! outputs), and instead yield a sequence of 33 or 65 zeros in that
//! case. Point decoding accepts compressed and uncompressed formats, and
//! also the one-byte encoding of the point-at-infinity, but they do not
//! accept a sequence of 33 or 65 zeros as a valid input. Thus, point
//! decoding is stricly standards-conforming. All decoding operations
//! enforce canonicality of encoding, and verify that the point is indeed
//! on the curve.
//!
//! The `PrivateKey` structure represents a private key for the ECDSA
//! signature algorithm; it is basically a wrapper around a private
//! scalar value. The `PrivateKey::encode()` and `PrivateKey::decode()`
//! functions encode a private key to exactly 32 bytes, and decode it
//! back, this time using unsigned big-endian, as per SEC 1 encoding
//! rules (which represents private keys with the ASN.1 `OCTET STRING`
//! type). The `PrivateKey::from_seed()` allows generating a private key
//! from a source seed, which is presumed to have been obtained
//! from a cryptographically secure random source.
//!
//! The `PublicKey` structure represents a public key for the ECDSA
//! signature algorithm; it is a wrapper around a `Point`. It has its own
//! `decode()`, `encode_compressed()` and `encode_uncompressed()` which
//! only wrap around the corresponding `Point` functions, except that
//! `decode()` explicitly rejects the point-at-infinity: an ECDSA public
//! key is never the identity point.
//!
//! ECDSA signatures are generated with `PrivateKey::sign_hash()`, and
//! verified with `PublicKey::verify_hash()`. The signature process is
//! deterministic, using the SHA-256 function, following the description
//! in [RFC 6979]. The caller is provides the pre-hashed message
//! (normally, this hashing uses SHA-256, but the functions accept hash
//! values of any length). In this implementation, the ECDSA signatures
//! follow the non-ASN.1 format: the two `r` and `s` halves of the
//! signature are encoded in unsigned big-endian format and concatenated,
//! in that order. When generating a signature, exactly 32 bytes are used
//! for each of `r` and `s`, so the signature has length 64 bytes
//! exactly. When verifying a signature, any input size is accepted
//! provided that it is even (so that it is unambiguous where `r` stops
//! and `s` starts), and that the two `r` and `s` values are still in the
//! proper range (i.e. lower than the curve order).
//!
//! [FIPS 186-4]: https://csrc.nist.gov/publications/detail/fips/186/4/final
//! [RFC 6979]: https://datatracker.ietf.org/doc/html/rfc6979

// Projective/fractional coordinates traditionally use uppercase letters,
// using lowercase only for affine coordinates.
#![allow(non_snake_case)]

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use super::field::{GFsecp256k1, ModInt256};
use sha2::{Sha512, Digest};
use super::{CryptoRng, RngCore};
use core::convert::TryFrom;

/// A point on the short Weierstraß curve secp256k1.
#[derive(Clone, Copy, Debug)]
pub struct Point {
    X: GFsecp256k1,
    Y: GFsecp256k1,
    Z: GFsecp256k1,
}

/// Integers modulo the curve order n (a 256-bit prime).
pub type Scalar = ModInt256<0xBFD25E8CD0364141, 0xBAAEDCE6AF48A03B,
                            0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFF>;

impl Scalar {
    /// Encodes a scalar element into bytes (little-endian).
    pub fn encode(self) -> [u8; 32] {
        self.encode32()
    }
}

/// Reverses a 32-byte sequence (i.e. switches between big-endian and
/// little-endian conventions).
///
/// Source slice MUST have length at least 32 (only the first 32 bytes
/// are accessed).
fn bswap32(x: &[u8]) -> [u8; 32] {
    let mut y = [0u8; 32];
    for i in 0..32 {
        y[i] = x[31 - i];
    }
    y
}

impl Point {

    // Curve equation is: y^2 = x^3 + b  (for a given constant b)
    // We use projective coordinates:
    //   (x, y) -> (X:Y:Z) such that x = X/Z and y = Y/Z
    //   Y is never 0 (not even for the neutral)
    //   X = 0 and Z = 0 for the neutral
    //   Z != 0 for all non-neutral points
    // X = 0 is conceptually feasible for some non-neutral points, but
    // it does not happen with secp256k1.
    // 
    // Note that the curve does not have a point of order 2.
    //
    // For point additions, we use the formulas from:
    //    https://eprint.iacr.org/2015/1060
    // The formulas are complete (on this curve), with cost 14M (including
    // two multiplications by the constant 3*b).
    //
    // For point doublings, the formulas have cost 7M+2S (including one
    // multiplication by the constant 3*b).

    /// The neutral element (point-at-infinity) in the curve.
    pub const NEUTRAL: Self = Self {
        X: GFsecp256k1::ZERO,
        Y: GFsecp256k1::ONE,
        Z: GFsecp256k1::ZERO,
    };

    /// The conventional base point in the curve.
    ///
    /// Like all non-neutral points in secp256k1, it generates the whole curve.
    pub const BASE: Self = Self {
        X: GFsecp256k1::w64be(
            0x79BE667EF9DCBBAC, 0x55A06295CE870B07,
            0x029BFCDB2DCE28D9, 0x59F2815B16F81798),
        Y: GFsecp256k1::w64be(
            0x483ADA7726A3C465, 0x5DA4FBFC0E1108A8,
            0xFD17B448A6855419, 0x9C47D08FFB10D4B8),
        Z: GFsecp256k1::ONE,
    };

    /// Curve equation parameter b.
    const B: GFsecp256k1 = GFsecp256k1::w64be(0, 0, 0, 7);

    /// Tries to decode a point.
    ///
    /// This function accepts the following encodings and lengths:
    ///
    ///  - A single byte of value 0x00: the point-at-infinity.
    ///
    ///  - A byte of value 0x02 or 0x03, followed by exactly 32 bytes
    ///    (unsigned big-endian encoding of the x coordinate): compressed
    ///    encoding of a non-neutral point.
    ///
    ///  - A byte of value 0x04, followed by exactly 64 bytes (unsigned
    ///    big-endian encodings of x and y): uncompressed encoding of a
    ///    non-neutral point.
    ///
    /// The (very rarely encountered) "hybrid" encoding (like
    /// uncompressed, but the least significant bit of y is also copied
    /// into the first byte, which has value 0x06 or 0x07) is not
    /// supported.
    ///
    /// On success, this structure is set to the decoded point, and
    /// 0xFFFFFFFF is returned. On failure, this structure is set to the
    /// neutral point, and 0x00000000 is returned. A failure is reported
    /// if the coordinates can be decoded but do not correspond to a
    /// point on the curve.
    ///
    /// Constant-time behaviour: timing-based side channels may leak
    /// which encoding type was used (neutral, compressed, uncompressed)
    /// but not the value of the obtained point, nor whether the encoding
    /// was for a valid point.
    pub fn set_decode(&mut self, buf: &[u8]) -> u32 {
        *self = Self::NEUTRAL;

        if buf.len() == 1 {

            // Single-byte encoding is for the point-at-infinity.
            // Return 0xFFFFFFFF if and only if the byte has value 0x00.
            return (((buf[0] as i32) - 1) >> 8) as u32;

        } else if buf.len() == 33 {

            // Compressed encoding.
            // Check that the first byte is 0x02 or 0x03.
            let mut r = (((((buf[0] & 0xFE) ^ 0x02) as i32) - 1) >> 8) as u32;

            // Decode x.
            let (x, rx) = GFsecp256k1::decode32(&bswap32(&buf[1..33]));
            r &= rx;

            // Compute: y = sqrt(x^3 + b)
            let (mut y, ry) = (x * x.square() + Self::B).sqrt();
            r &= ry;

            // Negate y if the sign does not match the bit provided in the
            // first encoding byte. Note that there is no valid point with
            // y = 0, thus we do not have to check that the sign is correct
            // after the conditional negation.
            let yb = y.encode()[0];
            let ws = (((yb ^ buf[0]) & 0x01) as u32).wrapping_neg();
            y.set_cond(&-y, ws);

            // Set the coordinates, adjusting them if the process failed.
            self.X = GFsecp256k1::select(&GFsecp256k1::ZERO, &x, r);
            self.Y = GFsecp256k1::select(&GFsecp256k1::ONE, &y, r);
            self.Z = GFsecp256k1::select(
                &GFsecp256k1::ZERO, &GFsecp256k1::ONE, r);
            return r;

        } else if buf.len() == 65 {

            // Uncompressed encoding.
            // First byte must have value 0x04.
            let mut r = ((((buf[0] ^ 0x04) as i32) - 1) >> 8) as u32;

            // Decode x and y.
            let (x, rx) = GFsecp256k1::decode32(&bswap32(&buf[1..33]));
            let (y, ry) = GFsecp256k1::decode32(&bswap32(&buf[33..65]));
            r &= rx & ry;

            // Verify that the coordinates match the curve equation.
            r &= y.square().equals(x * x.square() + Self::B);

            // Set the coordinates, adjusting them if the process failed.
            self.X = GFsecp256k1::select(&GFsecp256k1::ZERO, &x, r);
            self.Y = GFsecp256k1::select(&GFsecp256k1::ONE, &y, r);
            self.Z = GFsecp256k1::select(
                &GFsecp256k1::ZERO, &GFsecp256k1::ONE, r);
            return r;

        } else {

            // Invalid encoding length, return 0.
            return 0;

        }
    }

    /// Tries to decode a point.
    ///
    /// This function accepts the following encodings and lengths:
    ///
    ///  - A single byte of value 0x00: the point-at-infinity.
    ///
    ///  - A byte of value 0x02 or 0x03, followed by exactly 32 bytes
    ///    (unsigned big-endian encoding of the x coordinate): compressed
    ///    encoding of a non-neutral point.
    ///
    ///  - A byte of value 0x04, followed by exactly 64 bytes (unsigned
    ///    big-endian encodings of x and y): uncompressed encoding of a
    ///    non-neutral point.
    ///
    /// The (very rarely encountered) "hybrid" encoding (like
    /// uncompressed, but the least significant bit of y is also copied
    /// into the first byte, which has value 0x06 or 0x07) is not
    /// supported.
    ///
    /// On success, the decoded point is returned; on failure, `None` is
    /// returned. A failure is reported if the coordinates can be decoded
    /// but do not correspond to a point on the curve.
    ///
    /// Constant-time behaviour: timing-based side channels may leak
    /// which encoding type was used (neutral, compressed, uncompressed)
    /// but not the value of the obtained point, nor whether the encoding
    /// was for a valid point.
    pub fn decode(buf: &[u8]) -> Option<Point> {
        let mut P = Point::NEUTRAL;
        if P.set_decode(buf) != 0 {
            Some(P)
        } else {
            None
        }
    }

    /// Encodes this point in compressed format (33 bytes).
    ///
    /// If the point is the neutral then `[0u8; 33]` is returned, which
    /// is NOT the standard encoding of the neutral (standard is a single
    /// byte of of value 0x00); for a non-neutral point, the first byte
    /// is always equal to 0x02 or 0x03, never to 0x00.
    pub fn encode_compressed(self) -> [u8; 33] {
        let r = !self.isneutral();
        let iZ = GFsecp256k1::ONE / self.Z;  // this is 0 if Z = 0
        let x = self.X * iZ;  // 0 for the neutral
        let y = self.Y * iZ;  // 0 for the neutral
        let mut b = [0u8; 33];
        b[0] = ((y.encode()[0] & 0x01) | 0x02) & (r as u8);
        b[1..33].copy_from_slice(&bswap32(&x.encode()));
        b
    }

    /// Encodes this point in uncompressed format (65 bytes).
    ///
    /// If the point is the neutral then `[0u8; 65]` is returned, which
    /// is NOT the standard encoding of the neutral (standard is a single
    /// byte of of value 0x00); for a non-neutral point, the first byte
    /// is always equal to 0x04, never to 0x00.
    pub fn encode_uncompressed(self) -> [u8; 65] {
        let r = !self.isneutral();
        let iZ = GFsecp256k1::ONE / self.Z;  // this is 0 if Z = 0
        let x = self.X * iZ;  // 0 for the neutral
        let y = self.Y * iZ;  // 0 for the neutral
        let mut b = [0u8; 65];
        b[0] = 0x04 & (r as u8);
        b[ 1..33].copy_from_slice(&bswap32(&x.encode()));
        b[33..65].copy_from_slice(&bswap32(&y.encode()));
        b
    }

    /// Gets the affine (x, y) coordinates for this point.
    ///
    /// Values (x, y, r) are returned, with x and y being field elements,
    /// and r a `u32` value that qualifies the outcome:
    ///
    ///  - if the point is the neutral, then x = 0, y = 0 and r = 0x00000000;
    ///
    ///  - otherwise, x and y are the affine coordinates, and r = 0xFFFFFFFF.
    ///
    /// Note that there is no point with x = 0 or with y = 0 on the curve.
    pub fn to_affine(self) -> (GFsecp256k1, GFsecp256k1, u32) {
        // Uncompressed format contains both coordinates.
        let bb = self.encode_uncompressed();

        // First byte is 0x00 for the neutral, 0x04 for other points.
        let r = (((bb[0] as i32) - 1) >> 8) as u32;

        // The values necessarily decode successfully.
        let (x, _) = GFsecp256k1::decode32(&bswap32(&bb[1..33]));
        let (y, _) = GFsecp256k1::decode32(&bswap32(&bb[33..65]));
        (x, y, r)
    }

    /// Gets the projective coordinates (X:Y:Z) for this point.
    ///
    /// Values (X, Y, Z) are returned, such that:
    ///
    ///  - if the point is the neutral (point-at-infinity), then X and Z
    ///    are 0;
    ///
    ///  - otherwise, Z != 0, and the affine point coordinates are
    ///    x = X/Z and y = Y/Z.
    ///
    /// By definition, projective coordinates for a given point are not
    /// unique; two equal points may have distinct projective coordinates.
    ///
    /// The Y coordinate is never 0. The X coordinate may be 0 only for
    /// the neutral point.
    pub fn to_projective(self) -> (GFsecp256k1, GFsecp256k1, GFsecp256k1) {
        (self.X, self.Y, self.Z)
    }

    /// Sets this instance from the provided affine coordinates.
    ///
    /// If the coordinates designate a valid curve point, then the
    /// function returns 0xFFFFFFFF; otherwise, this instance is set to
    /// the neutral, and the function returns 0x00000000.
    pub fn set_affine(&mut self, x: GFsecp256k1, y: GFsecp256k1) -> u32 {
        *self = Self::NEUTRAL;
        let y2 = x * x.square() + Self::B;
        let r = y.square().equals(y2);
        self.X.set_cond(&x, r);
        self.Y.set_cond(&y, r);
        self.Z.set_cond(&GFsecp256k1::ONE, r);
        r
    }

    /// Creates an instance from the provided affine coordinates.
    ///
    /// The coordinates are verified to comply with the curve equation;
    /// if they do not, then `None` is returned.
    ///
    /// Note: whether the point is on the curve or not may leak through
    /// side channels; however, the actual value of the point should not
    /// leak.
    pub fn from_affine(x: GFsecp256k1, y: GFsecp256k1) -> Option<Self> {
        let mut P = Self::NEUTRAL;
        if P.set_affine(x, y) != 0 {
            Some(P)
        } else {
            None
        }
    }

    /// Sets this instance from the provided projective coordinates.
    ///
    /// If the coordinates designate a valid curve point, then the
    /// function returns 0xFFFFFFFF; otherwise, this instance is set to
    /// the neutral, and the function returns 0x00000000.
    ///
    /// This function accepts any (X:Y:0) triplet as a representation of
    /// the point-at-infinity.
    pub fn set_projective(&mut self, X: GFsecp256k1, Y: GFsecp256k1,
                          Z: GFsecp256k1) -> u32
    {
        *self = Self::NEUTRAL;

        // Detect the point-at-infinity.
        let zn = Z.iszero();

        // Verify the equation, assuming a non-infinity point.
        let Y2 = X * X.square() + Self::B * Z * Z.square();
        let r = (Y.square() * Z).equals(Y2) & !zn;

        // r is 0xFFFFFFFF is the point is non-infinity and the coordinates
        // are valid.

        // Set the coordinates in the point if the equation is fulfilled
        // and Z != 0 (which also implies Y != 0, since there is no point
        // of order 2 on the curve).
        self.X.set_cond(&X, r);
        self.Y.set_cond(&Y, r);
        self.Z.set_cond(&Z, r);

        r | zn
    }

    /// Creates an instance from the provided projective coordinates.
    ///
    /// The coordinates are verified to comply with the curve equation;
    /// if they do not, then `None` is returned.
    ///
    /// This function accepts any (X:Y:0) triplet as a representation of
    /// the point-at-infinity.
    ///
    /// Note: whether the point is on the curve or not may leak through
    /// side channels; however, the actual value of the point should not
    /// leak.
    pub fn from_projective(X: GFsecp256k1, Y: GFsecp256k1, Z: GFsecp256k1) 
        -> Option<Self>
    {
        let mut P = Self::NEUTRAL;
        if P.set_projective(X, Y, Z) != 0 {
            Some(P)
        } else {
            None
        }
    }

    /// Adds point `rhs` to `self`.
    fn set_add(&mut self, rhs: &Self) {
        let (X1, Y1, Z1) = (&self.X, &self.Y, &self.Z);
        let (X2, Y2, Z2) = (&rhs.X, &rhs.Y, &rhs.Z);

        // Formulas from Renes-Costello-Batina 2016:
        // https://eprint.iacr.org/2015/1060
        // (algorithm 7, with some renaming and expression compaction)
        let x1x2 = X1 * X2;
        let y1y2 = Y1 * Y2;
        let z1z2 = Z1 * Z2;
        let C = (X1 + Y1) * (X2 + Y2) - x1x2 - y1y2;  // X1*Y2 + X2*Y1
        let D = (Y1 + Z1) * (Y2 + Z2) - y1y2 - z1z2;  // Y1*Z2 + Y2*Z1
        let E = (X1 + Z1) * (X2 + Z2) - x1x2 - z1z2;  // X1*Z2 + X2*Z1
        let F = x1x2.mul3();
        let G = z1z2.mul21();
        let H = y1y2 + G;
        let I = y1y2 - G;
        let J = E.mul21();
        let X3 = C * I - D * J;
        let Y3 = J * F + I * H;
        let Z3 = H * D + F * C;

        self.X = X3;
        self.Y = Y3;
        self.Z = Z3;
    }

    /// Adds the affine point `rhs` to `self`.
    ///
    /// If the point to add is the neutral, then `rhs.x` and `rhs.y` can
    /// be arbitrary, and `rz` is 0xFFFFFFFF; otherwise, `rhs.x` and `rhs.y`
    /// are the affine coordinates of the point to add, and `rz` is
    /// 0x00000000.
    fn set_add_affine(&mut self, rhs: &PointAffine, rz: u32) {
        let (X1, Y1, Z1) = (&self.X, &self.Y, &self.Z);
        let (X2, Y2) = (&rhs.x, &rhs.y);

        // Same formulas as in set_add(), but modified to account for
        // Z2 = 1 (implicitly).
        let x1x2 = X1 * X2;
        let y1y2 = Y1 * Y2;
        let C = (X1 + Y1) * (X2 + Y2) - x1x2 - y1y2;  // X1*Y2 + X2*Y1
        let D = Y2 * Z1 + Y1;                         // Y1*Z2 + Y2*Z1
        let E = X2 * Z1 + X1;                         // X1*Z2 + X2*Z1
        let F = x1x2.mul3();
        let G = Z1.mul21();
        let H = y1y2 + G;
        let I = y1y2 - G;
        let J = E.mul21();
        let X3 = C * I - D * J;
        let Y3 = J * F + I * H;
        let Z3 = H * D + F * C;

        // If rhs is the neutral, then we computed the wrong output and
        // we must fix it, namely by discarding the computed values in
        // that case.
        self.X.set_cond(&X3, !rz);
        self.Y.set_cond(&Y3, !rz);
        self.Z.set_cond(&Z3, !rz);
    }

    /// Subtract the affine point `rhs` from `self`.
    ///
    /// If the point to add is the neutral, then `rhs.x` and `rhs.y` can
    /// be arbitrary, and `rz` is 0xFFFFFFFF; otherwise, `rhs.x` and `rhs.y`
    /// are the affine coordinates of the point to add, and `rz` is
    /// 0x00000000.
    fn set_sub_affine(&mut self, rhs: &PointAffine, rz: u32) {
        self.set_add_affine(&PointAffine { x: rhs.x, y: -rhs.y }, rz);
    }

    /// Doubles this point (in place).
    ///
    /// This function is somewhat faster than using plain point addition.
    pub fn set_double(&mut self) {
        let (X, Y, Z) = (&self.X, &self.Y, &self.Z);

        // Formulas from Renes-Costello-Batina 2016:
        // https://eprint.iacr.org/2015/1060
        // (algorithm 9, with some renaming and expression compaction)
        let yy = Y.square();
        let yy8 = yy.mul8();
        let C = Z.square().mul21();
        let Z3 = Y * Z * yy8;
        let D = yy - C.mul3();
        let Y3 = D * (yy + C) + C * yy8;
        let X3 = (D * X * Y).mul2();

        self.X = X3;
        self.Y = Y3;
        self.Z = Z3;
    }

    /// Doubles this point.
    ///
    /// This function is somewhat faster than using plain point addition.
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
    ///
    /// When n > 1, this function is faster than calling `double()`
    /// n times.
    #[inline(always)]
    pub fn xdouble(self, n: u32) -> Self {
        let mut r = self;
        r.set_xdouble(n);
        r
    }

    /// Negates this point (in place).
    #[inline(always)]
    pub fn set_neg(&mut self) {
        self.Y.set_neg();
    }

    /// Subtracts point `rhs` from `self`.
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
        // If both points are non-neutral, then their Zs are non-zero
        // and we check that their affine coordinates match.
        // Since Y != 0 for all points, the test on Y cannot match between
        // a neutral and a non-neutral point.
        (self.X * rhs.Z).equals(rhs.X * self.Z)
        & (self.Y * rhs.Z).equals(rhs.Y * self.Z)
    }

    /// Tests whether this point is the neutral (point-at-infinity).
    ///
    /// Returned value is 0xFFFFFFFF for the neutral, 0x00000000 otherwise.
    #[inline(always)]
    pub fn isneutral(self) -> u32 {
        self.Z.iszero()
    }

    // Conditionally copies the provided point (`P`) into `self`.
    //
    //  - If `ctl` is 0xFFFFFFFF, then the value of `P` is copied into `self`.
    //
    //  - if `ctl` is 0x00000000, then the value of `self` is unchanged.
    //
    // Value `ctl` MUST be either 0x00000000 or 0xFFFFFFFF.
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
    /// if `ctl` = 0x00000000.
    ///
    /// Value `ctl` MUST be either 0x00000000 or 0xFFFFFFFF.
    #[inline]
    pub fn set_condneg(&mut self, ctl: u32) {
        self.Y.set_cond(&-self.Y, ctl);
    }

    // GLV endomorphism
    // ================
    //
    // Let epsilon be a cube root of 1 modulo p. The function
    // zeta(x, y) = (epsilon*x, y) is an endomorphism over the curve;
    // moreover, f(P) = theta*P for a value theta which is a cube root of
    // 1 modulo n (the curve order). We choose (arbitrarily) to use the
    // lower (as an integer) of the two non-trivial cube roots of 1 for
    // epsilon, which then imposes a specific theta.
    //
    // Using Lagrange's algorithm on the lattice ((theta, 1), (n, 0)), we
    // can find a size-reduced basis (v1, v2). These two vectors can be
    // described with two small integers s and t:
    //   s =  64502973549206556628585045361533709077
    //   t = 303414439467246543595250775667605759171
    // Then, v1 = (s, -t) and v2 = (s+t, s). This particular format of v1
    // and v2 comes from the fact that theta is a cube root of 1 modulo n
    // (in particular, 1 + theta + theta^2 = 0 mod n). Note that we also
    // have theta = s/t = -(s+t)/s, and s^2 + t^2 + s*t = n.
    //
    // Given a scalar k, interpreted as an integer in the 0..n-1 range, we
    // can find two small integers k0 and k1 such that k = k0 + k1*theta,
    // by computing:
    //   c = round(s*k / n)
    //   d = round(t*k / n)
    //   k0 = k - c*s - d*(s + t)
    //   k1 = c*t - d*s
    // The GLV paper (https://www.iacr.org/archive/crypto2001/21390189.pdf)
    // shows that the obtained solution is correct, and that:
    //   sqrt(k0^2 + k1^2) <= 0.5*sqrt(s^2 + t^2) + 0.5*sqrt((s + t)^2 + s^2)
    // The right-hand side of this inequality yields an upper bound on
    // |k0| and |k1| which is slightly _above_ 2^128, which is inconvenient
    // to us; however, the bound is not tight and we can do better.
    //
    // Indeed, we have:
    //   (k0,k1) = (k,0) - c*(s,-t) - d*(s+t,s)
    // and:
    //   (k,0) = (s*k/n)*(s,-t) + (t*k/n)*(s+t,s)
    // since s^2 + t^2 + s*t = n.
    // We thus obtain that:
    //   (k0,k1) = e*(s,-t) + f*(s+t,s)
    // with e = c - s*k/n and f = d - t*k/n. Given the definitions of c and
    // d, we necessarily have |e| <= 1/2 and |f| <= 1/2. Writing N(x) the
    // L2 norm of vector x, the GLV paper then uses the triangular inequality
    // to state that:
    //   N(k0,k1) <= |e|*N(v1) + |f|*N(v2) <= 0.5*N(v1) + 0.5*N(v2)
    // for vectors v1 = (s,-t) and v2 = (s+t,s) in our case. This is true,
    // but leads to N(k0,k1) <= 2^128.0067. The triangular inequality is a
    // worst case, in case the vectors v1 and v2 are colinear, but our
    // specific v1 and v2 are certainly not colinear (in fact, as a
    // size-reduced lattice basis, they are as orthogonal as can be in that
    // lattice). We can write:
    //   N(k0,k1)^2 = <e*v1 + f*v2, e*v1 + f*v2>
    //              = e^2*N(v1)^2 + f^2*N(v2)^2 + 2*e*f*<v1,v2>
    //              <= (1/4)*N(v1)^2 + (1/4)*N(v2)^2 + (1/2)*|<v1,v2>|
    // We know the vectors v1 = (s,-t) and v2 = (s+t,s), so we can compute
    // the above expression; in particular, <v1,v2> = s*(s+t)-t*s = s^2.
    // We thus find that N(k0,k1)^2 < 2^255.08, which implies that |k0|
    // and |k1| are both lower than 2^127.54.

    const EPSILON: GFsecp256k1 = GFsecp256k1::w64be(
        0x7AE96A2B657C0710, 0x6E64479EAC3434E9,
        0x9CF0497512F58995, 0xC1396C28719501EE);
    /* unused
    const THETA: Scalar = Scalar::w64be(
        0x5363AD4CC05C30E0, 0xA5261C028812645A,
        0x122E22EA20816678, 0xDF02967C1B23BD72);
    */

    /// Endomorphism on the group.
    fn zeta(self) -> Self {
        Self {
            X: self.X * Self::EPSILON,
            Y: self.Y,
            Z: self.Z
        }
    }

    /// Computes round(e*k/n).
    ///
    /// Values are exchanged as arrays of 32-bit limbs, in little-endian
    /// order (least significant first). n is the curve order. Input k must
    /// be lower than n; input e is less than 2^128. Output is lower than
    /// or equal to e.
    fn mul_divr_rounded(k: &[u32; 8], e: &[u32; 4]) -> [u32; 4] {
        // We compute round(e*k/n) = floor((e*k + (n-1)/2)/n). Since
        // k < n < 2^256, we know that e*k + (n-1)/2 < 2^384.
        // For the division, we apply the Granlund-Montgomery method from:
        // "Division by Invariant Integers using Multiplication"
        //    https://dl.acm.org/doi/pdf/10.1145/178243.178249
        //
        // Specifically, for the divisor d = curve order, and prec = 384,
        // the CHOOSE_MULTIPLIER() process (figure 6.2) returns a 382-bit
        // odd multiplier m, and shift count sh_post = 253. Applying the
        // optimized algorithm from figure 4.2, we get sh_pre = 0, and the
        // quotient of a 384-bit integer z by d (rounded low) is obtained as:
        //   q = floor((m*n)/(2^637))

        // m
        const M: [u32; 12] = [
            0x8B79A0F9, 0xBCD2FEBC, 0xB038D378, 0x13ACE39A,
            0x65F937D8, 0x8805B42E, 0x2A16EBF8, 0x28AA2463,
            0x00000000, 0x00000000, 0x00000000, 0x20000000,
        ];

        // (n-1)/2
        const HN: [u32; 12] = [
            0x681B20A0, 0xDFE92F46, 0x57A4501D, 0x5D576E73,
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ];

        // z <- k*e + (n-1)/2
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
        let mut cc = 0u32;
        for i in 0..12 {
            let w = (z[i] as u64) + (HN[i] as u64) + (cc as u64);
            z[i] = w as u32;
            cc = (w >> 32) as u32;
        }

        // t <- m*z
        let mut t = [0u32; 24];
        for i in 0..12 {
            let mut cc = 0u32;
            for j in 0..12 {
                let w = (M[i] as u64) * (z[j] as u64)
                    + (t[i + j] as u64) + (cc as u64);
                t[i + j] = w as u32;
                cc = (w >> 32) as u32;
            }
            t[i + 12] = cc;
        }

        // q = floor(t / 2^637)
        let q0 = (t[19] >> 29) | (t[20] << 3);
        let q1 = (t[20] >> 29) | (t[21] << 3);
        let q2 = (t[21] >> 29) | (t[22] << 3);
        let q3 = (t[22] >> 29) | (t[23] << 3);

        [ q0, q1, q2, q3 ]
    }

    /// Splits a scalar k into k0 and k1 (signed) such that
    /// k = k0 + k1*mu (with mu being a given square root of -1 modulo r).
    ///
    /// This function returns |k0|, sgn(k0), |k1| and sgn(k1), with
    /// sgn(x) = 0xFFFFFFFF for x < 0, 0x00000000 for x >= 0.
    fn split_theta(k: &Scalar) -> (u128, u32, u128, u32) {
        // s =  64502973549206556628585045361533709077
        const S: [u32; 4] = [
            0x9284EB15, 0xE86C90E4, 0xA7D46BCD, 0x3086D221,
        ];

        // t = 303414439467246543595250775667605759171
        const T: [u32; 4] = [
            0x0ABFE4C3, 0x6F547FA9, 0x010E8828, 0xE4437ED6,
        ];

        // s+t (mod 2^128)
        const ST: [u32; 4] = [
            0x9D44CFD8, 0x57C1108D, 0xA8E2F3F6, 0x14CA50F7,
        ];

        // Convert k into 32-bit limbs.
        let kb = k.encode();
        let mut kw = [0u32; 8];
        for i in 0..8 {
            let j = 4 * i;
            kw[i] = u32::from_le_bytes(*<&[u8; 4]>::try_from(&kb[j..j + 4]).unwrap());
        }

        // c = round(s*k / n)
        // d = round(t*k / n)
        let c = Self::mul_divr_rounded(&kw, &S);
        let d = Self::mul_divr_rounded(&kw, &T);

        // Since we know that |k0| and |k1| are both less than 2^128, we
        // can compute the values modulo 2^160.

        // k0 = k - c*s - d*(s + t)
        let mut kw0 = sub160(
            &sub160(
                &[ kw[0], kw[1], kw[2], kw[3], kw[4] ],
                &mul128_t160(&c, &S)),
            &mul128_t160(&d, &ST));
        // Correction: ST contains s + t - 2^128, so we must furthermore
        // subtract d*2^128 from kw0.
        kw0[4] = kw0[4].wrapping_sub(d[0]);

        // k1 = c*t - d*s
        let kw1 = sub160(
            &mul128_t160(&c, &T),
            &mul128_t160(&d, &S));

        // Compute abs(k0) and abs(k1); top limb of kw0 (resp. kw1) is
        // either 0x00000000 (non-negative) or 0xFFFFFFFF (negative).
        let (k0, sk0) = abs128(&kw0);
        let (k1, sk1) = abs128(&kw1);

        return (k0, sk0, k1, sk1);

        // =========== helper functions ===========

        // d <- a - b mod 2^160
        fn sub160(a: &[u32; 5], b: &[u32; 5]) -> [u32; 5] {
            let w = (a[0] as u64).wrapping_sub(b[0] as u64);
            let d0 = w as u32;
            let w = (a[1] as u64).wrapping_sub(b[1] as u64)
                .wrapping_sub(w >> 63);
            let d1 = w as u32;
            let w = (a[2] as u64).wrapping_sub(b[2] as u64)
                .wrapping_sub(w >> 63);
            let d2 = w as u32;
            let w = (a[3] as u64).wrapping_sub(b[3] as u64)
                .wrapping_sub(w >> 63);
            let d3 = w as u32;
            let d4 = a[4].wrapping_sub(b[4]).wrapping_sub((w >> 63) as u32);

            [ d0, d1, d2, d3, d4 ]
        }

        // d <- (a*b) mod 2^160
        fn mul128_t160(a: &[u32; 4], b: &[u32; 4]) -> [u32; 5] {
            let w = (a[0] as u64) * (b[0] as u64);
            let d0 = w as u32;
            let w = (a[1] as u64) * (b[0] as u64) + (w >> 32);
            let d1 = w as u32;
            let w = (a[2] as u64) * (b[0] as u64) + (w >> 32);
            let d2 = w as u32;
            let w = (a[3] as u64) * (b[0] as u64) + (w >> 32);
            let d3 = w as u32;
            let d4 = (w >> 32) as u32;

            let w = (a[0] as u64) * (b[1] as u64) + (d1 as u64);
            let d1 = w as u32;
            let w = (a[1] as u64) * (b[1] as u64) + (d2 as u64) + (w >> 32);
            let d2 = w as u32;
            let w = (a[2] as u64) * (b[1] as u64) + (d3 as u64) + (w >> 32);
            let d3 = w as u32;
            let d4 = d4.wrapping_add(a[3].wrapping_mul(b[1]))
                .wrapping_add((w >> 32) as u32);

            let w = (a[0] as u64) * (b[2] as u64) + (d2 as u64);
            let d2 = w as u32;
            let w = (a[1] as u64) * (b[2] as u64) + (d3 as u64) + (w >> 32);
            let d3 = w as u32;
            let d4 = d4.wrapping_add(a[2].wrapping_mul(b[2]))
                .wrapping_add((w >> 32) as u32);

            let w = (a[0] as u64) * (b[3] as u64) + (d3 as u64);
            let d3 = w as u32;
            let d4 = d4.wrapping_add(a[1].wrapping_mul(b[3]))
                .wrapping_add((w >> 32) as u32);

            [ d0, d1, d2, d3, d4 ]
        }

        // Given g such that |g| < 2^128, return |g| and sgn(g).
        fn abs128(g: &[u32; 5]) -> (u128, u32) {
            let gs = g[4];
            let w = ((g[0] ^ gs) as u64).wrapping_sub(gs as u64);
            let d0 = w as u32;
            let w = ((g[1] ^ gs) as u64).wrapping_sub(gs as u64)
                .wrapping_sub(w >> 63);
            let d1 = w as u32;
            let w = ((g[2] ^ gs) as u64).wrapping_sub(gs as u64)
                .wrapping_sub(w >> 63);
            let d2 = w as u32;
            let d3 = (g[3] ^ gs).wrapping_sub(gs)
                .wrapping_sub((w >> 63) as u32);

            let d = (d0 as u128)
                | ((d1 as u128) << 32)
                | ((d2 as u128) << 64)
                | ((d3 as u128) << 96);
            (d, gs)
        }
    }

    /// Recodes a scalar into 52 signed digits.
    ///
    /// Each digit is in -15..+16, top digit is in 0..+2.
    fn recode_scalar(n: &Scalar) -> [i8; 52] {
        let mut sd = [0i8; 52];
        let bb = n.encode();
        let mut cc: u32 = 0;       // carry from lower digits
        let mut i: usize = 0;      // index of next source byte
        let mut acc: u32 = 0;      // buffered bits
        let mut acc_len: i32 = 0;  // number of buffered bits
        for j in 0..52 {
            if acc_len < 5 && j < 51 {
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

            P.X.set_cond(&win[i].X, w);
            P.Y.set_cond(&win[i].Y, w);
            P.Z.set_cond(&win[i].Z, w);
        }

        // Negate the returned value if needed.
        P.Y.set_cond(&-P.Y, s);

        P
    }

    /// Multiplies this point by a scalar (in place).
    ///
    /// This operation is constant-time with regard to both the points
    /// and the scalar value.
    pub fn set_mul(&mut self, n: &Scalar) {
        // Split the scalar with the endomorphism.
        let (n0, s0, n1, s1) = Self::split_theta(n);

        // Compute the 5-bit windows:
        //   win0[i] = (i+1)*sgn(n0)*P
        //   win1[i] = (i+1)*sgn(n1)*zeta(P)
        // with zeta(x, y) = (x*epsilon, y) for epsilon^3 = 1 (this is an
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

    /// Lookups a point from a window in affine coordinates, with sign
    /// handling (constant-time).
    ///
    /// The returned point is in affine coordinates, and an extra "output
    /// is neutral" flag is also returned (since the neutral point does
    /// not have defined affine coordinates).
    fn lookup_affine(win: &[PointAffine; 16], k: i8) -> (PointAffine, u32) {
        // Split k into its sign s (0xFFFFFFFF for negative) and
        // absolute value (f).
        let s = ((k as i32) >> 8) as u32;
        let f = ((k as u32) ^ s).wrapping_sub(s);
        let mut P = PointAffine { x: GFsecp256k1::ZERO, y: GFsecp256k1::ONE };
        for i in 0..16 {
            // win[i] contains (i+1)*P; we want to keep it if (and only if)
            // i+1 == f.
            // Values a-b and b-a both have their high bit equal to 0 only
            // if a == b.
            let j = (i as u32) + 1;
            let w = !(f.wrapping_sub(j) | j.wrapping_sub(f));
            let w = ((w as i32) >> 31) as u32;

            P.x.set_cond(&win[i].x, w);
            P.y.set_cond(&win[i].y, w);
        }

        // Negate the returned value if needed.
        P.y.set_cond(&-P.y, s);
        let fz = (((f as i32) - 1) >> 8) as u32;

        (P, fz)
    }

    /// Lookups a point from a window in affine coordinates, with sign
    /// handling (constant-time).
    ///
    /// The returned point is projective coordinates (which can represent
    /// the neutral).
    #[inline]
    fn lookup_affine_proj(win: &[PointAffine; 16], k: i8) -> Self {
        let (P, rz) = Self::lookup_affine(win, k);
        Self {
            X: P.x,
            Y: P.y,
            Z: GFsecp256k1::select(&GFsecp256k1::ONE, &GFsecp256k1::ZERO, rz),
        }
    }

    /// Lookups a point from a window in affine coordinates, with sign
    /// handling (constant-time), and adds it to the current point.
    #[inline]
    fn set_lookup_affine_add(&mut self, win: &[PointAffine; 16], k: i8) {
        let (P, rz) = Self::lookup_affine(win, k);
        self.set_add_affine(&P, rz);
    }

    /// Sets this point by multiplying the conventional generator by the
    /// provided scalar.
    ///
    /// This operation is constant-time. It is faster than using the
    /// generic multiplication on `Self::BASE`.
    pub fn set_mulgen(&mut self, n: &Scalar) {
        // Recode the scalar into 52 signed digits.
        let sd = Self::recode_scalar(n);

        // We process four chunks in parallel. Each chunk is 13 digits.
        *self = Self::lookup_affine_proj(&PRECOMP_G, sd[12]);
        self.set_lookup_affine_add(&PRECOMP_G65, sd[25]);
        self.set_lookup_affine_add(&PRECOMP_G130, sd[38]);
        self.set_lookup_affine_add(&PRECOMP_G195, sd[51]);

        // Process the digits in high-to-low order.
        for i in (0..12).rev() {
            self.set_xdouble(5);
            self.set_lookup_affine_add(&PRECOMP_G, sd[i]);
            self.set_lookup_affine_add(&PRECOMP_G65, sd[i + 13]);
            self.set_lookup_affine_add(&PRECOMP_G130, sd[i + 26]);
            self.set_lookup_affine_add(&PRECOMP_G195, sd[i + 39]);
        }
    }

    /// Creates a point by multiplying the conventional generator by the
    /// provided scalar.
    ///
    /// This operation is constant-time. It is faster than using the
    /// generic multiplication on `Self::BASE`.
    #[inline]
    pub fn mulgen(n: &Scalar) -> Self {
        let mut P = Self::NEUTRAL;
        P.set_mulgen(n);
        P
    }

    /// 5-bit wNAF recoding of a scalar; output is a sequence of 257
    /// digits.
    ///
    /// Non-zero digits have an odd value, between -15 and +15
    /// (inclusive). (The recoding is constant-time, but use of wNAF is
    /// inherently non-constant-time.)
    fn recode_scalar_NAF(n: &Scalar) -> [i8; 257] {
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
        // Since a scalar fits on 256 bits, at most 257 digits are needed.

        let mut sd = [0i8; 257];
        let bb = n.encode();
        let mut x = bb[0] as u32;
        for i in 0..257 {
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

    /// Given scalars `u` and `v`, sets this point to `u*self + v*G`
    /// (with `G` being the conventional generator point, aka
    /// `Self::BASE`).
    ///
    /// This function can be used to support ECDSA signature
    /// verification.
    ///
    /// THIS FUNCTION IS NOT CONSTANT-TIME; it shall be used only with
    /// public data.
    pub fn set_mul_add_mulgen_vartime(&mut self, u: &Scalar, v: &Scalar) {
        // Split the first scalar with the endomorphism.
        let (u0, s0, u1, s1) = Self::split_theta(u);

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
        // with zeta(x, y) = (x*epsilon, y) for epsilon^3 = 1 (this is an
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
            let e3 = if i < 127 { sd2[i + 130] } else { 0 };
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
                    self.set_add_affine(&PRECOMP_G[e2 as usize - 1], 0);
                } else {
                    self.set_sub_affine(&PRECOMP_G[(-e2) as usize - 1], 0);
                }
            }
            if e3 != 0 {
                if e3 > 0 {
                    self.set_add_affine(&PRECOMP_G130[e3 as usize - 1], 0);
                } else {
                    self.set_sub_affine(&PRECOMP_G130[(-e3) as usize - 1], 0);
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

    /// Given scalars `u` and `v`, returns point `u*self + v*G`
    /// (with `G` being the conventional generator point, aka
    /// `Self::BASE`).
    ///
    /// This function can be used to support ECDSA signature
    /// verification.
    ///
    /// THIS FUNCTION IS NOT CONSTANT-TIME; it shall be used only with
    /// public data.
    #[inline(always)]
    pub fn mul_add_mulgen_vartime(self, u: &Scalar, v: &Scalar) -> Self {
        let mut R = self;
        R.set_mul_add_mulgen_vartime(u, v);
        R
    }

    /// Check whether `s*G = R + k*Q`, for the provided scalars `s`
    /// and `k`, provided points `Q` (`self`) and `R`, and conventional
    /// generator `G`.
    ///
    /// Returned value is true on match, false otherwise. This function
    /// is meant to support Schnorr signature verification (e.g. as defined
    /// in FROST).
    ///
    /// THIS FUNCTION IS NOT CONSTANT-TIME; it shall be used only with
    /// public data.
    pub fn verify_helper_vartime(self,
        R: &Point, s: &Scalar, k: &Scalar) -> bool
    {
        // We use mul_add_mulgen_vartime(), which leverages the fast
        // endomorphism on the curve.
        let T = self.mul_add_mulgen_vartime(&(-k), s);
        T.equals(*R) != 0
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

/// A secp256k1 private key simply wraps around a scalar.
#[derive(Clone, Copy, Debug)]
pub struct PrivateKey {
    x: Scalar,   // secret scalar
}

/// A secp256k1 public key simply wraps around a curve point.
#[derive(Clone, Copy, Debug)]
pub struct PublicKey {
    pub point: Point,
}

impl PrivateKey {

    /// Generates a new private key from a cryptographically secure RNG.
    pub fn generate<T: CryptoRng + RngCore>(rng: &mut T) -> Self {
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        Self::from_seed(&seed)
    }

    /// Instantiates a private key by decoding the provided 32-byte
    /// array.
    ///
    /// The 32 bytes contain the unsigned **big-endian** encoding of the
    /// secret scalar (as per SEC1 and RFC 5915). The decoding may fail
    /// in the following cases:
    ///
    ///  - The source slice does not have length exactly 32 bytes.
    ///
    ///  - The scalar value is zero.
    ///
    ///  - The scalar value is not lower than the curve order.
    ///
    /// Decoding is constant-time; side-channels may leak whether the
    /// value was valid or not, but not the value itself (nor why it was
    /// deemed invalid, if decoding failed).
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() != 32 {
            return None;
        }
        let (x, r) = Scalar::decode32(&bswap32(buf));
        if (r & !x.iszero()) != 0  {
            Some(Self { x })
        } else {
            None
        }
    }

    /// Encodes this private key into exactly 32 bytes.
    ///
    /// Encoding uses the unsigned big-endian convention, as per SEC1 and
    /// RFC 5915.
    pub fn encode(self) -> [u8; 32] {
        let buf = self.x.encode();
        bswap32(&buf)
    }

    /// Instantiates a private key from a random seed.
    ///
    /// The seed MUST have been generated from a cryptographically secure
    /// random source that ensured an entropy of at least 128 bits (which
    /// implies that the seed cannot logically have length less than 16
    /// bytes). The transform from the seed to the private key is not
    /// described by any standard; therefore, for key storage, the
    /// private key itself should be stored, not the seed.
    ///
    /// This process guarantees that the output key is valid (i.e. it is
    /// in the proper range, and it is non-zero).
    pub fn from_seed(seed: &[u8]) -> Self {
        // We use SHA-512 over the input seed to get a pseudo-random
        // 512-bit value, which is then reduced modulo the curve order.
        // A custom prefix ("crrl scep256k1" in ASCII) is used to avoid
        // collisions.
        let mut sh = Sha512::new();
        sh.update(&[ 0x63, 0x72, 0x72, 0x6c, 0x20, 0x73, 0x65,
                     0x63, 0x70, 0x32, 0x35, 0x36, 0x6b, 0x31 ]);
        sh.update(seed);
        let mut x = Scalar::decode_reduce(&sh.finalize()[..]);

        // We make sure we do not get zero by replacing the value with 1
        // in that case. The probability that such a thing happens is
        // negligible.
        x.set_cond(&Scalar::ONE, x.iszero());
        Self { x }
    }

    /// Gets the public key corresponding to that private key.
    pub fn to_public_key(self) -> PublicKey {
        PublicKey { point: Point::mulgen(&self.x) }
    }

    /// Signs a hash value with ECDSA.
    ///
    /// The hash value may have an arbitrary length, but in general
    /// should be a SHA-256 output. The provided hash value (`hv`) MUST
    /// be a real hash value, not a raw unhashed message (in particular,
    /// if `hv` is longer than 256 bits, it is internally truncated).
    ///
    /// An ECDSA signature is a pair of integers (r, s), both being taken
    /// modulo the curve order n. This function encodes r and s over 32
    /// bytes each (unsigned big-endian notation), and returns their
    /// concatenation.
    ///
    /// Additional randomness can be provided as the `extra_rand` slice.
    /// It is not necessary for security that the extra randomness is
    /// cryptographically secure. If `extra_rand` has length 0, then the
    /// signature generation process is deterministic (but still safe!).
    /// Note: this does not follow the exact process of RFC 6979, but the
    /// same principle is applied.
    pub fn sign_hash(self, hv: &[u8], extra_rand: &[u8]) -> [u8; 64] {

        // Convert the input hash value into an integer modulo n:
        //  - If hv.len() > 32, keep only the leftmost 32 bytes.
        //  - Interpret the value as big-endian.
        //  - Reduce the integer modulo n.
        // The result is h.
        let mut tmp = [0u8; 32];
        if hv.len() >= 32 {
            tmp[..].copy_from_slice(&hv[..32]);
        } else {
            tmp[(32 - hv.len())..32].copy_from_slice(hv);
        }
        let h = Scalar::decode_reduce(&bswap32(&tmp));

        // Compute k by reducing the SHA-512 hash value of the concatenation
        // of the private key (over 32 bytes, little-endian), the h scalar
        // (32 bytes, little-endian), and the extra randomness (if provided).
        // If 0 is obtained (this has negligible probability), then 1 is
        // used instead.
        let mut sh = Sha512::new();
        sh.update(&self.x.encode());
        sh.update(&h.encode());
        if extra_rand.len() > 0 {
            sh.update(&extra_rand);
        }
        let mut k = Scalar::decode_reduce(&sh.finalize());
        k.set_cond(&Scalar::ONE, k.iszero());

        loop {
            // R = k*G; then encode x(R), and decode-reduce as a scalar
            let R = Point::mulgen(&k);
            let xR_le = bswap32(&R.encode_compressed()[1..33]);
            let r = Scalar::decode_reduce(&xR_le);

            // Compute s.
            let s = (h + self.x * r) / k;

            // If s and r are both non-zero, then we have our signature.
            if (r.iszero() | s.iszero()) == 0 {
                let mut sig = [0u8; 64];
                sig[..32].copy_from_slice(&bswap32(&r.encode()));
                sig[32..].copy_from_slice(&bswap32(&s.encode()));
                return sig;
            }

            // It is extremely improbable that either r or s is zero, and
            // nobody knows an input that would yield such a result.
            // Just in case, though, we increment k in that case, and
            // try again.
            k += Scalar::ONE;
            k.set_cond(&Scalar::ONE, k.iszero());
        }
    }
}

impl PublicKey {

    /// Decodes a public key from bytes.
    ///
    /// This function accepts both compressed (33 bytes) and uncompressed
    /// (65 bytes) formats. The point is always verified to be a valid
    /// curve point. Note that the neutral point (the
    /// "point-at-infinity") is explicitly rejected.
    pub fn decode(buf: &[u8]) -> Option<Self> {
        let point = Point::decode(buf)?;
        if point.isneutral() != 0 {
            return None;
        }
        Some(Self { point })
    }

    /// Encodes this public key into the compressed format (33 bytes).
    ///
    /// The first byte of the encoding always has value 0x02 or 0x03.
    pub fn encode_compressed(self) -> [u8; 33] {
        self.point.encode_compressed()
    }

    /// Encodes this public key into the compressed format (65 bytes).
    ///
    /// The first byte of the encoding always has value 0x04.
    pub fn encode_uncompressed(self) -> [u8; 65] {
        self.point.encode_uncompressed()
    }

    /// Verifies a signature on a given hashed message.
    ///
    /// The signature (`sig`) MUST have an even length; the first half of
    /// the signature is interpreted as the "r" integer, while the second
    /// half is "s" (both use unsigned big-endian convention).
    /// Out-of-range values are rejected. The hashed message is provided
    /// as `hv`; it is nominally the output of a suitable hash function
    /// (often SHA-256) computed over the actual message. This function
    /// can tolerate arbitrary hash output lengths; however, for proper
    /// security, the hash output must not be too short, and it must be
    /// an actual hash function output, not raw structured data.
    ///
    /// Note: this function is not constant-time; it assumes that the
    /// public key and signature value are public data.
    pub fn verify_hash(self, sig: &[u8], hv: &[u8]) -> bool {
        // Recover r and s as scalars. We truncate/pad them to 32 bytes
        // (verifying that the removed bytes are all zeros), then decode
        // them as scalars. Zeros and out-of-range values are rejected.
        let sig_len = sig.len();
        if (sig_len & 1) != 0 {
            return false;
        }
        let rlen = sig_len >> 1;
        let mut rb = [0u8; 32];
        let mut sb = [0u8; 32];
        if rlen > 32 {
            for i in 0..(rlen - 32) {
                if sig[i] != 0 || sig[rlen + i] != 0 {
                    return false;
                }
            }
            rb[..].copy_from_slice(&sig[(rlen - 32)..rlen]);
            sb[..].copy_from_slice(&sig[(sig_len - 32)..sig_len]);
        } else {
            rb[(32 - rlen)..].copy_from_slice(&sig[..rlen]);
            sb[(32 - rlen)..].copy_from_slice(&sig[rlen..]);
        }
        let (r, cr) = Scalar::decode32(&bswap32(&rb));
        if cr == 0 || r.iszero() != 0 {
            return false;
        }
        let (s, cs) = Scalar::decode32(&bswap32(&sb));
        if cs == 0 || s.iszero() != 0 {
            return false;
        }

        // Convert the input hash value into an integer modulo n.
        let mut tmp = [0u8; 32];
        if hv.len() >= 32 {
            tmp[..].copy_from_slice(&hv[..32]);
        } else {
            tmp[32 - hv.len() .. 32].copy_from_slice(hv);
        }
        let h = Scalar::decode_reduce(&bswap32(&tmp));

        // Verification algorithm.
        let w = Scalar::ONE / s;
        let R = self.point.mul_add_mulgen_vartime(&(r * w), &(h * w));
        let xR_le = bswap32(&R.encode_compressed()[1..33]);
        let rr = Scalar::decode_reduce(&xR_le);

        // Signature is valid if the rebuilt r value (in rr) matches
        // the one that was received.
        return r.equals(rr) != 0;
    }
}

// ========================================================================

// We hardcode known multiples of the points G, (2^65)*G, (2^130)*G
// and (2^195)*G, with G being the conventional base point. These are
// used to speed mulgen() operations up. The points are stored in affine
// coordinates, i.e. their Z coordinate is implicitly equal to 1.

/// A curve point (non-infinity) in affine coordinates.
#[derive(Clone, Copy, Debug)]
struct PointAffine {
    x: GFsecp256k1,
    y: GFsecp256k1,
}

// Points i*G for i = 1 to 16, in affine coordinates.
static PRECOMP_G: [PointAffine; 16] = [
    // G * 1
    PointAffine { x: GFsecp256k1::w64be(0x79BE667EF9DCBBAC, 0x55A06295CE870B07,
                                        0x029BFCDB2DCE28D9, 0x59F2815B16F81798),
                  y: GFsecp256k1::w64be(0x483ADA7726A3C465, 0x5DA4FBFC0E1108A8,
                                        0xFD17B448A6855419, 0x9C47D08FFB10D4B8) },
    // G * 2
    PointAffine { x: GFsecp256k1::w64be(0xC6047F9441ED7D6D, 0x3045406E95C07CD8,
                                        0x5C778E4B8CEF3CA7, 0xABAC09B95C709EE5),
                  y: GFsecp256k1::w64be(0x1AE168FEA63DC339, 0xA3C58419466CEAEE,
                                        0xF7F632653266D0E1, 0x236431A950CFE52A) },
    // G * 3
    PointAffine { x: GFsecp256k1::w64be(0xF9308A019258C310, 0x49344F85F89D5229,
                                        0xB531C845836F99B0, 0x8601F113BCE036F9),
                  y: GFsecp256k1::w64be(0x388F7B0F632DE814, 0x0FE337E62A37F356,
                                        0x6500A99934C2231B, 0x6CB9FD7584B8E672) },
    // G * 4
    PointAffine { x: GFsecp256k1::w64be(0xE493DBF1C10D80F3, 0x581E4904930B1404,
                                        0xCC6C13900EE07584, 0x74FA94ABE8C4CD13),
                  y: GFsecp256k1::w64be(0x51ED993EA0D455B7, 0x5642E2098EA51448,
                                        0xD967AE33BFBDFE40, 0xCFE97BDC47739922) },
    // G * 5
    PointAffine { x: GFsecp256k1::w64be(0x2F8BDE4D1A072093, 0x55B4A7250A5C5128,
                                        0xE88B84BDDC619AB7, 0xCBA8D569B240EFE4),
                  y: GFsecp256k1::w64be(0xD8AC222636E5E3D6, 0xD4DBA9DDA6C9C426,
                                        0xF788271BAB0D6840, 0xDCA87D3AA6AC62D6) },
    // G * 6
    PointAffine { x: GFsecp256k1::w64be(0xFFF97BD5755EEEA4, 0x20453A14355235D3,
                                        0x82F6472F8568A18B, 0x2F057A1460297556),
                  y: GFsecp256k1::w64be(0xAE12777AACFBB620, 0xF3BE96017F45C560,
                                        0xDE80F0F6518FE4A0, 0x3C870C36B075F297) },
    // G * 7
    PointAffine { x: GFsecp256k1::w64be(0x5CBDF0646E5DB4EA, 0xA398F365F2EA7A0E,
                                        0x3D419B7E0330E39C, 0xE92BDDEDCAC4F9BC),
                  y: GFsecp256k1::w64be(0x6AEBCA40BA255960, 0xA3178D6D861A54DB,
                                        0xA813D0B813FDE7B5, 0xA5082628087264DA) },
    // G * 8
    PointAffine { x: GFsecp256k1::w64be(0x2F01E5E15CCA351D, 0xAFF3843FB70F3C2F,
                                        0x0A1BDD05E5AF888A, 0x67784EF3E10A2A01),
                  y: GFsecp256k1::w64be(0x5C4DA8A741539949, 0x293D082A132D13B4,
                                        0xC2E213D6BA5B7617, 0xB5DA2CB76CBDE904) },
    // G * 9
    PointAffine { x: GFsecp256k1::w64be(0xACD484E2F0C7F653, 0x09AD178A9F559ABD,
                                        0xE09796974C57E714, 0xC35F110DFC27CCBE),
                  y: GFsecp256k1::w64be(0xCC338921B0A7D9FD, 0x64380971763B61E9,
                                        0xADD888A4375F8E0F, 0x05CC262AC64F9C37) },
    // G * 10
    PointAffine { x: GFsecp256k1::w64be(0xA0434D9E47F3C862, 0x35477C7B1AE6AE5D,
                                        0x3442D49B1943C2B7, 0x52A68E2A47E247C7),
                  y: GFsecp256k1::w64be(0x893ABA425419BC27, 0xA3B6C7E693A24C69,
                                        0x6F794C2ED877A159, 0x3CBEE53B037368D7) },
    // G * 11
    PointAffine { x: GFsecp256k1::w64be(0x774AE7F858A9411E, 0x5EF4246B70C65AAC,
                                        0x5649980BE5C17891, 0xBBEC17895DA008CB),
                  y: GFsecp256k1::w64be(0xD984A032EB6B5E19, 0x0243DD56D7B7B365,
                                        0x372DB1E2DFF9D6A8, 0x301D74C9C953C61B) },
    // G * 12
    PointAffine { x: GFsecp256k1::w64be(0xD01115D548E7561B, 0x15C38F004D734633,
                                        0x687CF4419620095B, 0xC5B0F47070AFE85A),
                  y: GFsecp256k1::w64be(0xA9F34FFDC815E0D7, 0xA8B64537E17BD815,
                                        0x79238C5DD9A86D52, 0x6B051B13F4062327) },
    // G * 13
    PointAffine { x: GFsecp256k1::w64be(0xF28773C2D975288B, 0xC7D1D205C3748651,
                                        0xB075FBC6610E58CD, 0xDEEDDF8F19405AA8),
                  y: GFsecp256k1::w64be(0x0AB0902E8D880A89, 0x758212EB65CDAF47,
                                        0x3A1A06DA521FA91F, 0x29B5CB52DB03ED81) },
    // G * 14
    PointAffine { x: GFsecp256k1::w64be(0x499FDF9E895E719C, 0xFD64E67F07D38E32,
                                        0x26AA7B63678949E6, 0xE49B241A60E823E4),
                  y: GFsecp256k1::w64be(0xCAC2F6C4B54E8551, 0x90F044E4A7B3D464,
                                        0x464279C27A3F95BC, 0xC65F40D403A13F5B) },
    // G * 15
    PointAffine { x: GFsecp256k1::w64be(0xD7924D4F7D43EA96, 0x5A465AE3095FF411,
                                        0x31E5946F3C85F79E, 0x44ADBCF8E27E080E),
                  y: GFsecp256k1::w64be(0x581E2872A86C72A6, 0x83842EC228CC6DEF,
                                        0xEA40AF2BD896D3A5, 0xC504DC9FF6A26B58) },
    // G * 16
    PointAffine { x: GFsecp256k1::w64be(0xE60FCE93B59E9EC5, 0x3011AABC21C23E97,
                                        0xB2A31369B87A5AE9, 0xC44EE89E2A6DEC0A),
                  y: GFsecp256k1::w64be(0xF7E3507399E59592, 0x9DB99F34F5793710,
                                        0x1296891E44D23F0B, 0xE1F32CCE69616821) },
];

// Points i*(2^65)*G for i = 1 to 16, in affine coordinates.
static PRECOMP_G65: [PointAffine; 16] = [
    // (2^65)*G * 1
    PointAffine { x: GFsecp256k1::w64be(0x8D26200250CEBDAE, 0x120EF31B04C80CD5,
                                        0x0D4CDDC8EADBCF29, 0xFC696D32C0ADE462),
                  y: GFsecp256k1::w64be(0xEBED3BB4715BF437, 0xD31F6F2DC3EE36BA,
                                        0x1D4AFB4E72678B3A, 0xD8E0A8B90F26470C) },
    // (2^65)*G * 2
    PointAffine { x: GFsecp256k1::w64be(0x1238C0766EAEBEA9, 0xCE4068A1F594D03B,
                                        0x8ED4930D072D9C8B, 0x9164643E1516E633),
                  y: GFsecp256k1::w64be(0x8A9DB02DBB271359, 0xD6C979E2D1C3DC17,
                                        0x0946252DCC740228, 0x05CDB728C77B7805) },
    // (2^65)*G * 3
    PointAffine { x: GFsecp256k1::w64be(0x17C072D56BDD1382, 0xA782481B8AA4D223,
                                        0x2DB794385870BCAD, 0xC3063330A5CD5379),
                  y: GFsecp256k1::w64be(0xD901BDF4283DA064, 0xE77C1247AF1D034F,
                                        0x8959AC76265BAD0D, 0xF7CAE051B108CD25) },
    // (2^65)*G * 4
    PointAffine { x: GFsecp256k1::w64be(0x271D5B0770CB9C15, 0xE7B2EA758A6A11B9,
                                        0xCDDCD7282B0EC216, 0x19B01552788E7A66),
                  y: GFsecp256k1::w64be(0x5D3AA45834E7F491, 0xE457D09949AC877F,
                                        0xE2A065E3508A824E, 0x7A8D7258E03C9727) },
    // (2^65)*G * 5
    PointAffine { x: GFsecp256k1::w64be(0xAC2ACB9B21999A70, 0x540708AB68338266,
                                        0xAEF650EED81C5B30, 0xDA1E87D8A8A923B7),
                  y: GFsecp256k1::w64be(0x7684428511C1724D, 0x1C9AFA0DF13D9EB3,
                                        0x60B0D0BF12D27A4F, 0xA2DC124AD7CD20A6) },
    // (2^65)*G * 6
    PointAffine { x: GFsecp256k1::w64be(0x88271C02621192F9, 0xBA6B25EF9CB2256E,
                                        0xAC32A5F91FD25EA9, 0x5793C018CA2D8DAE),
                  y: GFsecp256k1::w64be(0xD719DD53507176AA, 0x401C8B3AE5ABF5AC,
                                        0xC300876DC717D099, 0xFB426C0F3E1E77D9) },
    // (2^65)*G * 7
    PointAffine { x: GFsecp256k1::w64be(0x15B8390D652D7338, 0xE18EE09197E0E176,
                                        0x74F8C4BAFA2E7B85, 0x8F5BADC99C89240F),
                  y: GFsecp256k1::w64be(0x786CF20C8EFE8D08, 0x3ABDD7CCC7A59F99,
                                        0xB30367AB5C1A3335, 0x2E2F9EF8E326F04A) },
    // (2^65)*G * 8
    PointAffine { x: GFsecp256k1::w64be(0x85672C7D2DE0B7DA, 0x2BD1770D89665868,
                                        0x741B3F9AF7643397, 0x721D74D28134AB83),
                  y: GFsecp256k1::w64be(0x7C481B9B5B43B2EB, 0x6374049BFA62C2E5,
                                        0xE77F17FCC5298F44, 0xC8E3094F790313A6) },
    // (2^65)*G * 9
    PointAffine { x: GFsecp256k1::w64be(0xED621F7798ADD722, 0xB0DC5E529C6FEC6B,
                                        0xDFF60827B0B12C85, 0x18D798DC761F1075),
                  y: GFsecp256k1::w64be(0x5768C18656350E03, 0x1CE9AEBA20F74824,
                                        0x948E785AD74ED8ED, 0x939D44A1B0F3B558) },
    // (2^65)*G * 10
    PointAffine { x: GFsecp256k1::w64be(0xEAD4FA2F0A1516E0, 0xD92A75CB7AF3930E,
                                        0x6A25734CC87BCC49, 0x5F29B66EB89447A0),
                  y: GFsecp256k1::w64be(0xB45174E03831FF21, 0xCE27BB0B2B6F2CB9,
                                        0xF5D2A845D92EDA06, 0x5A6036BC79163281) },
    // (2^65)*G * 11
    PointAffine { x: GFsecp256k1::w64be(0x5F950F20B610C06B, 0x76949DAB52FC6149,
                                        0x97D254BE0A1330A0, 0x493F1EA21D608864),
                  y: GFsecp256k1::w64be(0x26F67B7E7DC4C006, 0x2E3F482F4316F7A9,
                                        0xE794BA1390DF25D9, 0x64EA7D7B75B36550) },
    // (2^65)*G * 12
    PointAffine { x: GFsecp256k1::w64be(0xCEB67E812E3E4A29, 0xAA6A8311986EC5AB,
                                        0x431E8524F124E1FB, 0xA950FAFD1EE503A6),
                  y: GFsecp256k1::w64be(0x5E6A8545AC390613, 0xB823DF78109CD86B,
                                        0x4B896D95EE69E2F3, 0x1FFD40D94E98C4D1) },
    // (2^65)*G * 13
    PointAffine { x: GFsecp256k1::w64be(0xA9AAF56B5016DB58, 0x5B8116DDCBAD1169,
                                        0x4B16DE8D9DB5EA5A, 0x279CCF4D091B1D7A),
                  y: GFsecp256k1::w64be(0xDE7012BEC765B543, 0xBB04D57C8FE914AF,
                                        0x663BF17944BE6D9A, 0x80A88EB0E6B5A32F) },
    // (2^65)*G * 14
    PointAffine { x: GFsecp256k1::w64be(0x07758C6DE814678E, 0xEAFE2E753F2C0693,
                                        0x84AD1C823F952889, 0xCADB1BE5796C687A),
                  y: GFsecp256k1::w64be(0x6B6039EA9CDC8488, 0xAA540ADD077202B0,
                                        0x949F9331AA048403, 0xD9D1005ED089DDA2) },
    // (2^65)*G * 15
    PointAffine { x: GFsecp256k1::w64be(0x9F46479A69411D57, 0xC3C7EA6ADFA833F9,
                                        0x1FB2109AFD30C790, 0x2CE323AE4B14BE0C),
                  y: GFsecp256k1::w64be(0x9329281F7B6B346A, 0x61983DA7E41BD909,
                                        0xB111BAEB7C16565E, 0xD874F8C18A7B746C) },
    // (2^65)*G * 16
    PointAffine { x: GFsecp256k1::w64be(0x534CCF6B740F9EC0, 0x36C1861215C8A61F,
                                        0x3B89EA46DF2E6D96, 0x998B90BC1F17FC25),
                  y: GFsecp256k1::w64be(0xD5715CB09C8B2DDB, 0x462AE3DD32D54355,
                                        0x0AE3D277BFDD28DD, 0xD71C7F6ECFE86E76) },
];

// Points i*(2^130)*G for i = 1 to 16, in affine coordinates.
static PRECOMP_G130: [PointAffine; 16] = [
    // (2^130)*G * 1
    PointAffine { x: GFsecp256k1::w64be(0x7564539E85D56F85, 0x37D6619E1F5C5AA7,
                                        0x8D2A3DE0889D1D4E, 0xE8DBCB5729B62026),
                  y: GFsecp256k1::w64be(0xC1D685413749B3C6, 0x5231DF524A722925,
                                        0x684AACD954B79F33, 0x4172C8FADACE0CF3) },
    // (2^130)*G * 2
    PointAffine { x: GFsecp256k1::w64be(0x210A917AD9DF2779, 0x6746FF301AD9CCC8,
                                        0x78F61A5F1FF4082B, 0x5364DACD57B4A278),
                  y: GFsecp256k1::w64be(0x670E1B5450B5E57B, 0x7A39BE81F8D6737D,
                                        0x3789E61AAFF20BFC, 0x7F2713FD0C7B2231) },
    // (2^130)*G * 3
    PointAffine { x: GFsecp256k1::w64be(0x5568DAC679F74A32, 0xEBB5FAD219547AD1,
                                        0x66F440ABC1C017B4, 0x70F702D505ED815E),
                  y: GFsecp256k1::w64be(0x7A85F8742788BA64, 0x580D6FE01D073F2B,
                                        0xEB05F7EEE2582151, 0xD9BBF64C00602DF0) },
    // (2^130)*G * 4
    PointAffine { x: GFsecp256k1::w64be(0xE4F3FB0176AF85D6, 0x5FF99FF9198C3609,
                                        0x1F48E86503681E3E, 0x6686FD5053231E11),
                  y: GFsecp256k1::w64be(0x1E63633AD0EF4F1C, 0x1661A6D0EA02B728,
                                        0x6CC7E74EC951D1C9, 0x822C38576FEB73BC) },
    // (2^130)*G * 5
    PointAffine { x: GFsecp256k1::w64be(0x9AA9A7FF54DEBAA0, 0xD30DC06917144F0B,
                                        0x1DF5E7985B188A46, 0x56D823710F6AEB45),
                  y: GFsecp256k1::w64be(0x5336F7FC662565B2, 0x6A39B258D8C74CF7,
                                        0x578DD3874035A888, 0x6AB18C2A27479FAB) },
    // (2^130)*G * 6
    PointAffine { x: GFsecp256k1::w64be(0x87195A80DC83BE4E, 0xCFC9D4B829725CBE,
                                        0x11101C26013C98F2, 0x641753AF1EE840F8),
                  y: GFsecp256k1::w64be(0x06031DCC996CE3AE, 0xB15F6DDB4A9A2138,
                                        0xDD89C27090A8DFA8, 0x0228269067EED395) },
    // (2^130)*G * 7
    PointAffine { x: GFsecp256k1::w64be(0x2D492168934B4CE5, 0xBE6F8E222161DE2C,
                                        0x80ECCA1E6812AB39, 0xD33B1534E53DADAC),
                  y: GFsecp256k1::w64be(0x6B3A38C9BB39F399, 0x8884199D07AC87F8,
                                        0xEDCDD04FDBA090C8, 0xE3D18704585E8EB4) },
    // (2^130)*G * 8
    PointAffine { x: GFsecp256k1::w64be(0x4B30CBB7686773E0, 0x1EC64110ABDB362F,
                                        0x88531A825BA17295, 0x3BFEE2233BCDAF2F),
                  y: GFsecp256k1::w64be(0x74C6350265BB629B, 0x6F9E2C5777C3C4A9,
                                        0x1FDF3C81E4348575, 0x68033D463D26B5B7) },
    // (2^130)*G * 9
    PointAffine { x: GFsecp256k1::w64be(0x84A517B7E05290EA, 0xD10A1B5E4DCE4564,
                                        0xF7B6EAACD75F9C4B, 0x3E6AE00FD4077638),
                  y: GFsecp256k1::w64be(0x7B9F0BF5B60EC494, 0x8886EA84D4BC2D84,
                                        0x1972106C6C41DCC0, 0x1F86D469AC415EB7) },
    // (2^130)*G * 10
    PointAffine { x: GFsecp256k1::w64be(0xA4D2802411F577C1, 0xC5D08FBC457A46BD,
                                        0x428F4D2AB29475EA, 0xEF622876593E49F0),
                  y: GFsecp256k1::w64be(0x1B7AAB6E53FCBD4E, 0x237FB43D851DC788,
                                        0x7D1150DDAD78B5FF, 0xB2B1F2984F84B8E0) },
    // (2^130)*G * 11
    PointAffine { x: GFsecp256k1::w64be(0x5DA4E742B7CB76F4, 0xB6F4FEAABDF4DD5A,
                                        0xC8C08D998634A645, 0x2BAAC486B31F9A77),
                  y: GFsecp256k1::w64be(0xEE8ECA8A1BDC1F8C, 0x09DDF91432C74CD6,
                                        0x3C40261FDE2016A4, 0x3722B1E48FD36174) },
    // (2^130)*G * 12
    PointAffine { x: GFsecp256k1::w64be(0x900C3241BEE44FE9, 0x0832F51FEB470DEC,
                                        0xA2F56E03212A9946, 0x5399F04E6BF05BD6),
                  y: GFsecp256k1::w64be(0x6C31F9E8E8B1F0F5, 0xF95C7204570B2439,
                                        0xD69853583C4EFB15, 0xDE52AD3BF00D358B) },
    // (2^130)*G * 13
    PointAffine { x: GFsecp256k1::w64be(0x94B995F51E4B0976, 0x694BEB6BC0698E28,
                                        0x0B71CBF2AB17753A, 0xA6D22DACAB359D6C),
                  y: GFsecp256k1::w64be(0xCC2C70F0E8B49742, 0xB57CF18D760E7059,
                                        0xCE7B03B2E136412B, 0x5BFF9A4C52C9F14C) },
    // (2^130)*G * 14
    PointAffine { x: GFsecp256k1::w64be(0xF79781E7A4137AC4, 0x7A9A9D009D239B37,
                                        0x6CD0FA3CB9F5DE46, 0x8CBA5A110FFCDD69),
                  y: GFsecp256k1::w64be(0xF2EF45877691F792, 0xFA1DABBBE9A18626,
                                        0xF84C2B7AE5BB71FC, 0xD9276F93D0D887D4) },
    // (2^130)*G * 15
    PointAffine { x: GFsecp256k1::w64be(0xFDD2FCE57C54C676, 0x553205C63EE71C28,
                                        0xC3A2597AC35C0E7D, 0xC14197C5E08ADAEA),
                  y: GFsecp256k1::w64be(0x5D5C412719B293AB, 0x8F1AF2F983763114,
                                        0x148359BB0D4BFF4D, 0x251A5A6FBED748D1) },
    // (2^130)*G * 16
    PointAffine { x: GFsecp256k1::w64be(0xCBB434AA7AE1700D, 0xCD15B20B17464817,
                                        0xEC11715050E0FA19, 0x2FFE9C29A673059F),
                  y: GFsecp256k1::w64be(0x4A1A200AB4DABD17, 0x562D492338B5DFAD,
                                        0x41D45E4F0AD5F845, 0xB7DA9642227C070C) },
];

// Points i*(2^195)*G for i = 1 to 16, in affine coordinates.
static PRECOMP_G195: [PointAffine; 16] = [
    // (2^195)*G * 1
    PointAffine { x: GFsecp256k1::w64be(0x60144494C8F69448, 0x5B85ECB6AEE10956,
                                        0xC756267D12894711, 0x922243D5E855B8DA),
                  y: GFsecp256k1::w64be(0x8BB5D669F681E646, 0x9E8BE1FD9132E65B,
                                        0x543955C27E3F2A4B, 0xAD500590F34E4BBD) },
    // (2^195)*G * 2
    PointAffine { x: GFsecp256k1::w64be(0xE4A42D43C5CF169D, 0x9391DF6DECF42EE5,
                                        0x41B6D8F0C9A13740, 0x1E23632DDA34D24F),
                  y: GFsecp256k1::w64be(0x4D9F92E716D1C735, 0x26FC99CCFB8AD34C,
                                        0xE886EEDFA8D8E4F1, 0x3A7F7131DEBA9414) },
    // (2^195)*G * 3
    PointAffine { x: GFsecp256k1::w64be(0x1EB7CB4D971E5316, 0xA209BD338FF36ED9,
                                        0xCF4F0DA811F362CD, 0x4A95838EB84DA233),
                  y: GFsecp256k1::w64be(0xD984328AE47C84FF, 0x826F3BCD0BDED0AB,
                                        0xA336C99981CF0AE9, 0xCB8EA55317C43F18) },
    // (2^195)*G * 4
    PointAffine { x: GFsecp256k1::w64be(0xFD6451FB84CFB18D, 0x3EF0ACF856C4EF4D,
                                        0x0553C562F7AE4D2A, 0x303F2EA33E8F62BB),
                  y: GFsecp256k1::w64be(0xE745CEB2B1871578, 0xB6FE7A5C1BC344CC,
                                        0xFA2AB492D200E83F, 0xD0AD9086132C0911) },
    // (2^195)*G * 5
    PointAffine { x: GFsecp256k1::w64be(0xAAA48545E0E226E6, 0x7FBE4AC6C9040AFF,
                                        0xB3D427C61FF6C3B8, 0xC3208D14B5BF37FF),
                  y: GFsecp256k1::w64be(0xA6BC6AA6CD2927B1, 0x2FFD61B7491637D7,
                                        0xBE7E72A29E8F5CD8, 0x72E2CD7501F263F0) },
    // (2^195)*G * 6
    PointAffine { x: GFsecp256k1::w64be(0x3E419634E156A3A2, 0x4949BC8E8D396FAF,
                                        0x09430123677B392B, 0x5C8410AF3BEA0C68),
                  y: GFsecp256k1::w64be(0x0123C59D924B21F7, 0xF373CBFE37069306,
                                        0x2FA11946303CDA1A, 0xBCBB6FF71A45EDB6) },
    // (2^195)*G * 7
    PointAffine { x: GFsecp256k1::w64be(0xC7E511DC9DABD507, 0x72576532EFD7DBAE,
                                        0xE18BC312477E1DD4, 0x8BEACBB385152AA8),
                  y: GFsecp256k1::w64be(0xE9BF1F86DFFE772E, 0xCD7B66963E4F7FA0,
                                        0xB3B581714DFD63B1, 0xDA805AA7A782AA01) },
    // (2^195)*G * 8
    PointAffine { x: GFsecp256k1::w64be(0x1EEE207CB24086BC, 0x716E81A06F9EDBBB,
                                        0x0042E2D5DCF3C7A1, 0xFA1D1FB9D5FE696B),
                  y: GFsecp256k1::w64be(0x652CBD19AEF6269C, 0xD2B196D12461C95F,
                                        0x7A02062E0AFD694E, 0xBB45670E7429337B) },
    // (2^195)*G * 9
    PointAffine { x: GFsecp256k1::w64be(0x0CFFD9693EB29213, 0x750CC57B7FABCE74,
                                        0xD43E6BAB95215B83, 0x6FE50CE90FEF8C18),
                  y: GFsecp256k1::w64be(0x831163EB4A1FEB00, 0xD59A834A392A66A2,
                                        0xDAFD902840D1AF47, 0x8B41CCDDB1E0280E) },
    // (2^195)*G * 10
    PointAffine { x: GFsecp256k1::w64be(0x8D9438F5455D7508, 0xEED4A3E62F7F0B57,
                                        0x6EB7B64C351C9897, 0xAF75D23C939824D7),
                  y: GFsecp256k1::w64be(0x3261E0734FEE6C2A, 0x2CA60BD31AB6EF6F,
                                        0x8FB9E2B8326B063D, 0x8A004F489366489F) },
    // (2^195)*G * 11
    PointAffine { x: GFsecp256k1::w64be(0xCEDC08639C64CD25, 0x38608DB2FD6574FF,
                                        0x200255A33F3B48CE, 0x2907F6D12C317482),
                  y: GFsecp256k1::w64be(0x413ED3F381BF024F, 0xB8C73D2D1570DE86,
                                        0x7FACF5881D6CDFA8, 0x99F2332FE064E123) },
    // (2^195)*G * 12
    PointAffine { x: GFsecp256k1::w64be(0xF13A99E58DC72FCB, 0x0C62A492D2850704,
                                        0x621DDF48F1F433E6, 0x9A9814C417D4B84A),
                  y: GFsecp256k1::w64be(0x33C2C8CD0F0BE995, 0xAA6B91CD1E3FE06E,
                                        0xB6E37D4710F2D962, 0x85990FC553FD1C81) },
    // (2^195)*G * 13
    PointAffine { x: GFsecp256k1::w64be(0x5FFAA262A47FAD9E, 0xF51FBF6C76DCFCC2,
                                        0xDDE8172EED32DEC4, 0x031D668832363481),
                  y: GFsecp256k1::w64be(0x545A43ADE0D50DAE, 0xE362A4FADB98225C,
                                        0xD276A0F973BEF10D, 0x45C2A243C3C014F7) },
    // (2^195)*G * 14
    PointAffine { x: GFsecp256k1::w64be(0xB72524C558EE5442, 0x0D4A912A2FE54543,
                                        0x9360C2FB7428E620, 0x8E48071A98D713DE),
                  y: GFsecp256k1::w64be(0x4C51B39A8A283E45, 0x1042D182E9D69415,
                                        0x0482D26FE44A5FCB, 0x76FFE5259B8350E9) },
    // (2^195)*G * 15
    PointAffine { x: GFsecp256k1::w64be(0x1E5635B05FA1850B, 0xC7ADF807F79D2294,
                                        0xC74DCC1F17092700, 0xD2A125AA698BC489),
                  y: GFsecp256k1::w64be(0x09A0088FE337E6DE, 0x8A61A9873FBF3BBA,
                                        0x961A9FBDB9B5A056, 0x3BAC9BB85E183204) },
    // (2^195)*G * 16
    PointAffine { x: GFsecp256k1::w64be(0xCC0EA33EA8A9EB14, 0xD465AB2C346E2111,
                                        0xE1C0FC017C572579, 0x08D40F19EF94C0D5),
                  y: GFsecp256k1::w64be(0xF9907A3B711C8A2F, 0xB23DD203B5FBE663,
                                        0xF6074F266113F543, 0xDEABE597AF452FE6) },
];

// ========================================================================

#[cfg(test)]
mod tests {

    use super::{Point, Scalar, PrivateKey, PublicKey};
    use sha2::{Sha256, Digest};

    /* unused
    fn print_gf(name: &str, x: GFsecp256k1) {
        print!("{} = 0x", name);
        let bb = x.encode();
        for i in (0..32).rev() {
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

    fn print_sc(name: &str, x: Scalar) {
        print!("{} = 0x", name);
        let bb = x.encode();
        for i in (0..32).rev() {
            print!("{:02X}", bb[i]);
        }
        println!();
    }
    */

    #[test]
    fn base_arith() {
        // Encoding of neutral.
        const EP0: [u8; 1] = [ 0 ];

        // For a point P (randomly generated on the curve with Sage),
        // points i*P for i = 0 to 6, encoded (compressed).
        // (Point 0*P is here represented as 33 bytes of value 0x00.)
        const EPC: [[u8; 33]; 7] = [
            [
                0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
            ],
            [
                0x02,
                0x85, 0xFC, 0x56, 0xC5, 0xD6, 0xCC, 0xD9, 0x8A,
                0x3D, 0x61, 0x14, 0xAB, 0x0C, 0x8B, 0x09, 0xCD,
                0x5E, 0x8F, 0xD9, 0x0D, 0x6C, 0x96, 0x6E, 0xD9,
                0xF9, 0xE1, 0x92, 0xB2, 0xF7, 0x39, 0x42, 0x88
            ],
            [
                0x02,
                0x1E, 0x15, 0x0E, 0x10, 0x08, 0x66, 0x3C, 0xAA,
                0xB3, 0x54, 0xD9, 0x24, 0x55, 0x31, 0x0A, 0xCF,
                0x5A, 0x51, 0xD1, 0x4C, 0xCA, 0xEB, 0x1B, 0xEC,
                0xB1, 0x48, 0xD7, 0xDD, 0x79, 0x7E, 0xA5, 0x5A
            ],
            [
                0x02,
                0x60, 0x0C, 0x54, 0xB9, 0x68, 0x05, 0xC8, 0xAD,
                0xF7, 0x11, 0xEC, 0xF0, 0x35, 0xEF, 0xFB, 0x42,
                0x60, 0x9F, 0x4C, 0xE5, 0x80, 0x12, 0xBE, 0xF1,
                0xA6, 0x8C, 0xE6, 0x43, 0x22, 0x5B, 0x6D, 0xBF
            ],
            [
                0x02,
                0xCA, 0xA2, 0x44, 0xDD, 0xBF, 0x5E, 0xD5, 0xCB,
                0x13, 0x84, 0xA4, 0x68, 0x9E, 0xEC, 0xCA, 0xAA,
                0x08, 0x40, 0x80, 0xAA, 0x53, 0xCC, 0xA3, 0x4B,
                0xC5, 0x2F, 0xBC, 0x90, 0xA5, 0x3E, 0xB1, 0xE1
            ],
            [
                0x03,
                0x6B, 0xD1, 0x67, 0x5D, 0x24, 0x45, 0xC1, 0x84,
                0xE0, 0xCD, 0x49, 0xED, 0x12, 0x5E, 0x98, 0x89,
                0x6B, 0xB6, 0xF0, 0xBB, 0xD0, 0x1F, 0x3F, 0x49,
                0xDF, 0x67, 0xC8, 0xBA, 0x58, 0xD5, 0xE6, 0x16
            ],
            [
                0x03,
                0x56, 0xFF, 0xC1, 0x9E, 0xAE, 0xD6, 0xD4, 0x6B,
                0xD7, 0x3A, 0x0E, 0x3F, 0xB4, 0x77, 0x59, 0xC9,
                0xFA, 0x58, 0xFF, 0x10, 0xA6, 0x37, 0xF4, 0xBF,
                0x5E, 0x1E, 0x96, 0xE2, 0x08, 0xAD, 0x42, 0x66
            ],
        ];

        // Same points, but with uncompressed encoding.
        // (Point 0*P is here represented as 65 bytes of value 0x00.)
        const EPU: [[u8; 65]; 7] = [
            [
                0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
            ],
            [
                0x04,
                0x85, 0xFC, 0x56, 0xC5, 0xD6, 0xCC, 0xD9, 0x8A,
                0x3D, 0x61, 0x14, 0xAB, 0x0C, 0x8B, 0x09, 0xCD,
                0x5E, 0x8F, 0xD9, 0x0D, 0x6C, 0x96, 0x6E, 0xD9,
                0xF9, 0xE1, 0x92, 0xB2, 0xF7, 0x39, 0x42, 0x88,
                0x9B, 0x59, 0x87, 0xFF, 0x8B, 0x5B, 0x16, 0x12,
                0x86, 0x43, 0xB8, 0x3D, 0xF2, 0x6F, 0xF7, 0x66,
                0x24, 0x45, 0x62, 0x70, 0xE8, 0x6B, 0x4F, 0xE4,
                0x92, 0x13, 0x0F, 0x61, 0x3B, 0x95, 0x04, 0x72
            ],
            [
                0x04,
                0x1E, 0x15, 0x0E, 0x10, 0x08, 0x66, 0x3C, 0xAA,
                0xB3, 0x54, 0xD9, 0x24, 0x55, 0x31, 0x0A, 0xCF,
                0x5A, 0x51, 0xD1, 0x4C, 0xCA, 0xEB, 0x1B, 0xEC,
                0xB1, 0x48, 0xD7, 0xDD, 0x79, 0x7E, 0xA5, 0x5A,
                0x23, 0x3A, 0xF4, 0x50, 0xE5, 0x46, 0x3A, 0x91,
                0x3A, 0x53, 0xE3, 0xCC, 0xFC, 0x92, 0x77, 0x94,
                0xB8, 0x6C, 0x43, 0x9D, 0x43, 0xAD, 0x31, 0x52,
                0xD1, 0xB1, 0x05, 0x3C, 0x16, 0x26, 0x9B, 0x32
            ],
            [
                0x04,
                0x60, 0x0C, 0x54, 0xB9, 0x68, 0x05, 0xC8, 0xAD,
                0xF7, 0x11, 0xEC, 0xF0, 0x35, 0xEF, 0xFB, 0x42,
                0x60, 0x9F, 0x4C, 0xE5, 0x80, 0x12, 0xBE, 0xF1,
                0xA6, 0x8C, 0xE6, 0x43, 0x22, 0x5B, 0x6D, 0xBF,
                0xC8, 0x45, 0x8C, 0xCB, 0xA6, 0x41, 0xB7, 0x18,
                0x0D, 0x47, 0xE9, 0xC0, 0x64, 0xCB, 0x6C, 0xF4,
                0x9E, 0xD6, 0x26, 0x7D, 0xBC, 0x4C, 0xA4, 0xA0,
                0xB6, 0xB5, 0x9C, 0xDD, 0xF3, 0x07, 0xC1, 0xF6
            ],
            [
                0x04,
                0xCA, 0xA2, 0x44, 0xDD, 0xBF, 0x5E, 0xD5, 0xCB,
                0x13, 0x84, 0xA4, 0x68, 0x9E, 0xEC, 0xCA, 0xAA,
                0x08, 0x40, 0x80, 0xAA, 0x53, 0xCC, 0xA3, 0x4B,
                0xC5, 0x2F, 0xBC, 0x90, 0xA5, 0x3E, 0xB1, 0xE1,
                0x19, 0xD0, 0x27, 0x56, 0x2B, 0x06, 0x31, 0xE9,
                0x77, 0x35, 0xB7, 0x71, 0x88, 0x90, 0xAF, 0x11,
                0x18, 0x19, 0x97, 0x12, 0xD4, 0x73, 0x63, 0x2C,
                0x59, 0x4A, 0x56, 0x64, 0x8E, 0x89, 0xD0, 0x44
            ],
            [
                0x04,
                0x6B, 0xD1, 0x67, 0x5D, 0x24, 0x45, 0xC1, 0x84,
                0xE0, 0xCD, 0x49, 0xED, 0x12, 0x5E, 0x98, 0x89,
                0x6B, 0xB6, 0xF0, 0xBB, 0xD0, 0x1F, 0x3F, 0x49,
                0xDF, 0x67, 0xC8, 0xBA, 0x58, 0xD5, 0xE6, 0x16,
                0xA0, 0x10, 0x2A, 0xDB, 0xEE, 0x27, 0x3B, 0x6B,
                0xA3, 0x02, 0x66, 0xC3, 0x36, 0xEC, 0x5C, 0xC2,
                0xBA, 0x3D, 0x3B, 0x25, 0xCB, 0xD6, 0x93, 0xAA,
                0xD4, 0x72, 0x0F, 0x72, 0x9E, 0x6B, 0x5F, 0x81
            ],
            [
                0x04,
                0x56, 0xFF, 0xC1, 0x9E, 0xAE, 0xD6, 0xD4, 0x6B,
                0xD7, 0x3A, 0x0E, 0x3F, 0xB4, 0x77, 0x59, 0xC9,
                0xFA, 0x58, 0xFF, 0x10, 0xA6, 0x37, 0xF4, 0xBF,
                0x5E, 0x1E, 0x96, 0xE2, 0x08, 0xAD, 0x42, 0x66,
                0x42, 0xDA, 0xDD, 0x63, 0xF7, 0xCB, 0x8B, 0x3B,
                0x0F, 0x77, 0x34, 0x5D, 0x98, 0xEA, 0xDF, 0x4B,
                0xBC, 0x71, 0xE0, 0x6B, 0x6C, 0x51, 0x86, 0xEE,
                0xAA, 0x55, 0x29, 0x1F, 0x13, 0x28, 0xDB, 0x0F
            ],
        ];

        let P0 = Point::decode(&EP0).unwrap();
        assert!(P0.isneutral() == 0xFFFFFFFF);

        let mut PP = [P0; 7];
        for i in 1..7 {
            let P = Point::decode(&EPC[i]).unwrap();
            let Q = Point::decode(&EPU[i]).unwrap();
            assert!(P.isneutral() == 0);
            assert!(Q.isneutral() == 0);
            assert!(P.equals(Q) == 0xFFFFFFFF);
            assert!(P.encode_compressed() == EPC[i]);
            assert!(P.encode_uncompressed() == EPU[i]);
            PP[i] = P;
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
        assert!(Q2.encode_compressed() == EPC[2]);
        assert!(Q2.equals(P2) == 0xFFFFFFFF);
        let R2 = P1.double();
        assert!(R2.encode_compressed() == EPC[2]);
        assert!(R2.equals(P2) == 0xFFFFFFFF);
        assert!(R2.equals(Q2) == 0xFFFFFFFF);

        let Q3 = P2 + P1;
        assert!(Q3.encode_compressed() == EPC[3]);
        assert!(Q3.equals(P3) == 0xFFFFFFFF);
        let R3 = Q2 + P1;
        assert!(R3.encode_compressed() == EPC[3]);
        assert!(R3.equals(P3) == 0xFFFFFFFF);
        assert!(R3.equals(Q3) == 0xFFFFFFFF);

        let Q4 = Q2.double();
        assert!(Q4.encode_compressed() == EPC[4]);
        assert!(Q4.equals(P4) == 0xFFFFFFFF);
        let R4 = P1.xdouble(2);
        assert!(R4.encode_compressed() == EPC[4]);
        assert!(R4.equals(P4) == 0xFFFFFFFF);
        assert!(R4.equals(Q4) == 0xFFFFFFFF);
        let R4 = P1 + Q3;
        assert!(R4.encode_compressed() == EPC[4]);
        assert!(R4.equals(P4) == 0xFFFFFFFF);
        assert!(R4.equals(Q4) == 0xFFFFFFFF);

        let Q5 = Q3 + R2;
        assert!(Q5.encode_compressed() == EPC[5]);
        assert!(Q5.equals(P5) == 0xFFFFFFFF);
        let R5 = R3 + Q2;
        assert!(R5.encode_compressed() == EPC[5]);
        assert!(R5.equals(P5) == 0xFFFFFFFF);
        assert!(R5.equals(Q5) == 0xFFFFFFFF);

        assert!((R5 - Q3).equals(Q2) == 0xFFFFFFFF);

        let Q6 = Q3.double();
        assert!(Q6.encode_compressed() == EPC[6]);
        assert!(Q6.equals(P6) == 0xFFFFFFFF);
        let R6 = Q2 + Q4;
        assert!(R6.encode_compressed() == EPC[6]);
        assert!(R6.equals(P6) == 0xFFFFFFFF);
        assert!(R6.equals(Q6) == 0xFFFFFFFF);

        let mut P = Q6;
        let mut Q = R6;
        for _ in 0..8 {
            P += P;
        }
        Q.set_xdouble(8);
        assert!(P.equals(Q) == 0xFFFFFFFF);

        let P = P1 + P0.double();
        assert!(P.equals(P1) == 0xFFFFFFFF);
        assert!(P.equals(P2) == 0x00000000);
    }

    #[test]
    fn split_theta() {
        const THETA: Scalar = Scalar::w64be(
            0x5363AD4CC05C30E0, 0xA5261C028812645A,
            0x122E22EA20816678, 0xDF02967C1B23BD72);

        let mut sh = Sha256::new();
        for i in 0..100 {
            sh.update(&(i as u64).to_le_bytes());
            let k: Scalar = Scalar::decode_reduce(&sh.finalize_reset());
            let (k0, sk0, k1, sk1) = Point::split_theta(&k);
            let mut t0 = Scalar::from_u128(k0);
            if sk0 != 0 {
                t0 = -t0;
            }
            let mut t1 = Scalar::from_u128(k1);
            if sk1 != 0 {
                t1 = -t1;
            }
            let t = t0 + t1 * THETA;
            assert!(t.equals(k) == 0xFFFFFFFF);
        }
    }

    #[test]
    fn mulgen() {
        // Test vector generated randomly with Sage.
        let s = Scalar::w64be(0xF0FCA55C06488D1C, 0x6CA454ED29573B6C,
                              0x89D4F76592F96F10, 0x98BD4A5F08DF863E);
        let enc: [u8; 33] = [
            0x02,
            0x08, 0x28, 0x9C, 0x90, 0x62, 0x82, 0x49, 0x71,
            0x94, 0x38, 0x9E, 0xA3, 0x2B, 0xD6, 0x35, 0x18,
            0xAD, 0xEA, 0xE8, 0x4C, 0x17, 0x9F, 0xEA, 0x6F,
            0xD2, 0x53, 0x1A, 0x71, 0x14, 0x4C, 0x94, 0xFA
        ];

        let R = Point::decode(&enc).unwrap();
        let P = Point::BASE * s;
        assert!(P.equals(R) == 0xFFFFFFFF);
        assert!(P.encode_compressed() == enc);
        let Q = Point::mulgen(&s);
        assert!(Q.equals(R) == 0xFFFFFFFF);
        assert!(Q.encode_compressed() == enc);
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
    fn verify_helper() {
        let mut sh = Sha256::new();
        for i in 0..20 {
            // Build pseudorandom Q, s and k.
            // Compute R = s*G - k*Q
            sh.update(((3 * i + 0) as u64).to_le_bytes());
            let v1 = sh.finalize_reset();
            sh.update(((3 * i + 1) as u64).to_le_bytes());
            let v2 = sh.finalize_reset();
            sh.update(((3 * i + 2) as u64).to_le_bytes());
            let v3 = sh.finalize_reset();
            let Q = Point::mulgen(&Scalar::decode_reduce(&v1));
            let s = Scalar::decode_reduce(&v2);
            let k = Scalar::decode_reduce(&v3);
            let R = Point::mulgen(&s) - k * Q;

            // verify_helper_vartime() must return true, but this
            // must change to false if we change a scalar or a point.
            assert!(Q.verify_helper_vartime(&R, &s, &k));
            assert!(!Q.verify_helper_vartime(&R, &(s + Scalar::ONE), &k));
            assert!(!Q.verify_helper_vartime(&R, &s, &(k + Scalar::ONE)));
            assert!(!Q.verify_helper_vartime(&(R + Point::BASE), &s, &k));
            assert!(!(Q + Point::BASE).verify_helper_vartime(&R, &s, &k));
        }
    }

    #[test]
    fn signatures() {
        // Test vector from project Wycheproof
        // (ecdsa_secp256k1_sha256_p1363_test.json)
        let pub_enc: [u8; 65] = [
            0x04,
            0xB8, 0x38, 0xFF, 0x44, 0xE5, 0xBC, 0x17, 0x7B,
            0xF2, 0x11, 0x89, 0xD0, 0x76, 0x60, 0x82, 0xFC,
            0x9D, 0x84, 0x32, 0x26, 0x88, 0x7F, 0xC9, 0x76,
            0x03, 0x71, 0x10, 0x0B, 0x7E, 0xE2, 0x0A, 0x6F,
            0xF0, 0xC9, 0xD7, 0x5B, 0xFB, 0xA7, 0xB3, 0x1A,
            0x6B, 0xCA, 0x19, 0x74, 0x49, 0x6E, 0xEB, 0x56,
            0xDE, 0x35, 0x70, 0x71, 0x95, 0x5D, 0x83, 0xC4,
            0xB1, 0xBA, 0xDA, 0xA0, 0xB2, 0x18, 0x32, 0xE9,
        ];
        let msg = b"123400";
        let sig: [u8; 64] = [
            0x81, 0x3E, 0xF7, 0x9C, 0xCE, 0xFA, 0x9A, 0x56,
            0xF7, 0xBA, 0x80, 0x5F, 0x0E, 0x47, 0x85, 0x84,
            0xFE, 0x5F, 0x0D, 0xD5, 0xF5, 0x67, 0xBC, 0x09,
            0xB5, 0x12, 0x3C, 0xCB, 0xC9, 0x83, 0x23, 0x65,
            0x90, 0x0E, 0x75, 0xAD, 0x23, 0x3F, 0xCC, 0x90,
            0x85, 0x09, 0xDB, 0xFF, 0x59, 0x22, 0x64, 0x7D,
            0xB3, 0x7C, 0x21, 0xF4, 0xAF, 0xD3, 0x20, 0x3A,
            0xE8, 0xDC, 0x4A, 0xE7, 0x79, 0x4B, 0x0F, 0x87,
        ];

        let pkey = PublicKey::decode(&pub_enc).unwrap();
        let mut sh = Sha256::new();
        sh.update(&msg);
        let hv1: [u8; 32] = sh.finalize_reset().into();
        sh.update(&msg);
        sh.update(&[0u8]);
        let hv2: [u8; 32] = sh.finalize_reset().into();
        assert!(pkey.verify_hash(&sig, &hv1));
        assert!(!pkey.verify_hash(&sig, &hv2));

        for i in 0..20 {
            sh.update((i as u64).to_le_bytes());
            let seed: [u8; 32] = sh.finalize_reset().into();
            let sk = PrivateKey::from_seed(&seed);
            let pk = sk.to_public_key();
            let sig1 = sk.sign_hash(&hv1, &[]);
            let sig2 = sk.sign_hash(&hv2, &[]);
            assert!(pk.verify_hash(&sig1, &hv1));
            assert!(pk.verify_hash(&sig2, &hv2));
            assert!(!pk.verify_hash(&sig1, &hv2));
            assert!(!pk.verify_hash(&sig2, &hv1));
            assert!(!pkey.verify_hash(&sig1, &hv1));
            assert!(!pkey.verify_hash(&sig2, &hv2));
        }
    }
}
