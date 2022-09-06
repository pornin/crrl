//! NIST P-256 curve implementation.
//!
//! This module implements generic group operations on the NIST P-256
//! elliptic curve, a short Weierstraß curve with equation `y^2 = x^3 -
//! 3*x + b` for a given constant `b`. This curve is standardized in
//! [FIPS 186-4] as well as in other standards such as SEC 2 or ANSI
//! X9:62. It is also known under the names "secp256r1" and "prime256v1".
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
//! the canonical range. Take care that many standards related to P-256
//! tend to use big-endian for encoding scalars (and often use a
//! variable-length encoding, e.g. an ASN.1 `INTEGER`).
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
//! # Truncated Signatures
//!
//! The `PublicKey::verify_trunc_hash()` function supports _truncated
//! signatures_: a 64-byte signature is provided, but the last few bits
//! are considered to have been reused for encoding other data, and thus
//! are ignored. The truncation requires that the original signature is
//! first processed through `PrivateKey::prepare_truncate()`; this
//! utility function does not use the private key itself, but it modifies
//! the encoding format of the `s` part of the signature so that
//! truncation removes the high-order bits of the value, instead of the
//! low-order bits.
//!
//! The verification function then tries to recompute the complete,
//! original, untruncated signature. This process is safe since neither
//! truncation nor reconstruction involve usage of the private key, and
//! the original signature is obtained as an outcome of the process. Up
//! to 32 bits (i.e. four whole bytes) can be rebuilt by this
//! implementation, which corresponds to shrinking the signature encoding
//! size from 64 down to 60 bytes.
//!
//! Signature reconstruction cost increases with the number of ignored
//! bits (asymptotically, cost doubles for every 2 removed bits, so
//! 32-bit truncation sboud be about 16 times more expensive than 24-bit
//! truncation); when 32 bits are ignored, the verification cost is about
//! 300 to 450 times the cost of verifying an untruncated signature.
//!
//! Truncated signature verification on curve P-256 requires dynamic
//! memory allocation; it is not available if this library is compiled
//! without the `std` or `alloc` feature (default compilation uses `std`
//! and thus `std::vec::Vec`; without `std` but with `alloc`,
//! `alloc::vec::Vec` is used).
//!
//! [FIPS 186-4]: https://csrc.nist.gov/publications/detail/fips/186/4/final
//! [RFC 6979]: https://datatracker.ietf.org/doc/html/rfc6979

// Projective/fractional coordinates traditionally use uppercase letters,
// using lowercase only for affine coordinates.
#![allow(non_snake_case)]

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use super::field::{GFp256, ModInt256};
use sha2::{Sha256, Sha512, Digest};
use rand_core::{CryptoRng, RngCore};

#[cfg(feature = "alloc")]
use crate::Vec;

/// A point on the short Weierstraß curve P-256.
#[derive(Clone, Copy, Debug)]
pub struct Point {
    X: GFp256,
    Y: GFp256,
    Z: GFp256,
}

/// Integers modulo the curve order n (a 256-bit prime).
pub type Scalar = ModInt256<0xF3B9CAC2FC632551, 0xBCE6FAADA7179E84,
                            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFF00000000>;

impl Scalar {
    /// Scalar encoding length (in bytes).
    pub const ENC_LEN: usize = 32;

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

    // Curve equation is: y^2 = x^3 - 3*x + b  (for a given constant b)
    // We use projective coordinates:
    //   (x, y) -> (X:Y:Z) such that x = X/Z and y = Y/Z
    //   Y is never 0 (not even for the neutral)
    //   X = 0 and Z = 0 for the neutral
    //   X = 0 is possible for some non-neutral points as well
    //   Z != 0 for all non-neutral points
    // 
    // Note that the curve does not have a point of order 2.
    //
    // For point additions, we use the formulas from:
    //    https://eprint.iacr.org/2015/1060
    // The formulas are complete (on this curve), with cost 14M (including
    // two multiplications by the curve constant b, which is a non-simple
    // value on curve P-256).
    //
    // For point doublings, we use some formulas from Bernstein and
    // Lange with cost 7M+3S; they require a corrective step when
    // doubling the neutral (because they then yield (0:0:0), which is
    // not valid). The corrective step is inexpensive (conditional
    // setting of Y to 1 in case Z is 0).
    //
    // For sequences of several doublings, we use the Bernstein-Lange
    // formulas for the first doubling, modified to yield an output in
    // Jacobian coordinates (i.e. x = X/Z^2 and y = Y/Z^3). The cost of
    // the first doubling is then lowered to 5M+2S. Subsequent doublings
    // then use the classic Hankerson-Vanstone formulas on Jacobian
    // coordinates, with cost 4M+4S. At the end, we convert back to
    // projective coordinates; the conversion has cost 2M+1S. In total,
    // we make n doublings with cost (7M+3S)+(n-1)*(4M+4S).

    /// The neutral element (point-at-infinity) in the curve.
    pub const NEUTRAL: Self = Self {
        X: GFp256::ZERO,
        Y: GFp256::ONE,
        Z: GFp256::ZERO,
    };

    /// The conventional base point in the curve.
    ///
    /// Like all non-neutral points in P-256, it generates the whole curve.
    pub const BASE: Self = Self {
        X: GFp256::w64be(
            0x6B17D1F2E12C4247, 0xF8BCE6E563A440F2,
            0x77037D812DEB33A0, 0xF4A13945D898C296),
        Y: GFp256::w64be(
            0x4FE342E2FE1A7F9B, 0x8EE7EB4A7C0F9E16,
            0x2BCE33576B315ECE, 0xCBB6406837BF51F5),
        Z: GFp256::ONE,
    };

    /// Curve equation parameter b.
    const B: GFp256 = GFp256::w64be(
        0x5AC635D8AA3A93E7,
        0xB3EBBD55769886BC,
        0x651D06B0CC53B0F6,
        0x3BCE3C3E27D2604B,
    );

    /// 2*b
    const B2: GFp256 = GFp256::w64be(
        0xB58C6BB1547527CF,
        0x67D77AAAED310D78,
        0xCA3A0D6198A761EC,
        0x779C787C4FA4C096,
    );

    /// 4*b
    const B4: GFp256 = GFp256::w64be(
        0x6B18D763A8EA4F9D,
        0xCFAEF555DA621AF1,
        0x94741AC2314EC3D8,
        0xEF38F0F89F49812D,
    );

    /// 8*b
    const B8: GFp256 = GFp256::w64be(
        0xD631AEC751D49F3B,
        0x9F5DEAABB4C435E3,
        0x28E83584629D87B1,
        0xDE71E1F13E93025A,
    );

    /// Constant 3 (in the field).
    const THREE: GFp256 = GFp256::w64be(
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000003,
    );

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
            let (x, rx) = GFp256::decode32(&bswap32(&buf[1..33]));
            r &= rx;

            // Compute: y = sqrt(x^3 - 3*x + b)
            let (mut y, ry) = (x * (x.square() - Self::THREE) + Self::B).sqrt();
            r &= ry;

            // Negate y if the sign does not match the bit provided in the
            // first encoding byte. Note that there is no valid point with
            // y = 0, thus we do not have to check that the sign is correct
            // after the conditional negation.
            let yb = y.encode()[0];
            let ws = (((yb ^ buf[0]) & 0x01) as u32).wrapping_neg();
            y.set_cond(&-y, ws);

            // Set the coordinates, adjusting them if the process failed.
            self.X = GFp256::select(&GFp256::ZERO, &x, r);
            self.Y = GFp256::select(&GFp256::ONE, &y, r);
            self.Z = GFp256::select(&GFp256::ZERO, &GFp256::ONE, r);
            return r;

        } else if buf.len() == 65 {

            // Uncompressed encoding.
            // First byte must have value 0x04.
            let mut r = ((((buf[0] ^ 0x04) as i32) - 1) >> 8) as u32;

            // Decode x and y.
            let (x, rx) = GFp256::decode32(&bswap32(&buf[1..33]));
            let (y, ry) = GFp256::decode32(&bswap32(&buf[33..65]));
            r &= rx & ry;

            // Verify that the coordinates match the curve equation.
            r &= y.square().equals(x * (x.square() - Self::THREE) + Self::B);

            // Set the coordinates, adjusting them if the process failed.
            self.X = GFp256::select(&GFp256::ZERO, &x, r);
            self.Y = GFp256::select(&GFp256::ONE, &y, r);
            self.Z = GFp256::select(&GFp256::ZERO, &GFp256::ONE, r);
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
        let iZ = GFp256::ONE / self.Z;  // this is 0 if Z = 0
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
        let iZ = GFp256::ONE / self.Z;  // this is 0 if Z = 0
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
    ///  - if the point is the neutral, then x = 1, y = 0 and r = 0x00000000;
    ///
    ///  - otherwise, x and y are the affine coordinates, and r = 0xFFFFFFFF.
    ///
    /// Note that there is no point with x = 1 or with y = 0 on the curve.
    pub fn to_affine(self) -> (GFp256, GFp256, u32) {
        // Uncompressed format contains both coordinates.
        let mut bb = self.encode_uncompressed();

        // First byte is 0x00 for the neutral, 0x04 for other points.
        let r = (((bb[0] as i32) - 1) >> 8) as u32;

        // For the neutral, we got zeros for x and y, but we want x = 1
        // in that case.
        bb[32] |= (r & 1) as u8;

        // The values necessarily decode successfully.
        let (x, _) = GFp256::decode32(&bswap32(&bb[1..33]));
        let (y, _) = GFp256::decode32(&bswap32(&bb[33..65]));
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
    /// The Y coordinate is never 0. The X coordinate may be 0 for a
    /// non-neutral point (there are two such points on curve P-256);
    /// it is always 0 for the neutral point.
    pub fn to_projective(self) -> (GFp256, GFp256, GFp256) {
        (self.X, self.Y, self.Z)
    }

    /// Sets this instance from the provided affine coordinates.
    ///
    /// If the coordinates designate a valid curve point, then the
    /// function returns 0xFFFFFFFF; otherwise, this instance is set to
    /// the neutral, and the function returns 0x00000000.
    pub fn set_affine(&mut self, x: GFp256, y: GFp256) -> u32 {
        *self = Self::NEUTRAL;
        let y2 = x * (x.square() - Self::THREE) + Self::B;
        let r = y.square().equals(y2);
        self.X.set_cond(&x, r);
        self.Y.set_cond(&y, r);
        self.Z.set_cond(&GFp256::ONE, r);
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
    pub fn from_affine(x: GFp256, y: GFp256) -> Option<Self> {
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
    pub fn set_projective(&mut self, X: GFp256, Y: GFp256, Z: GFp256) -> u32 {
        *self = Self::NEUTRAL;

        // Detect the point-at-infinity.
        let zn = Z.iszero();

        // Verify the equation, assuming a non-infinity point.
        let Z2 = Z.square();
        let Y2 = X * X.square() + Z2 * (Self::B * Z - X.mul3());
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
    pub fn from_projective(X: GFp256, Y: GFp256, Z: GFp256) -> Option<Self> {
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
        // (algorithm 4, with some renaming and expression compaction)
        let x1x2 = X1 * X2;
        let y1y2 = Y1 * Y2;
        let z1z2 = Z1 * Z2;
        let C = (X1 + Y1) * (X2 + Y2) - x1x2 - y1y2;  // X1*Y2 + X2*Y1
        let D = (Y1 + Z1) * (Y2 + Z2) - y1y2 - z1z2;  // Y1*Z2 + Y2*Z1
        let E = (X1 + Z1) * (X2 + Z2) - x1x2 - z1z2;  // X1*Z2 + X2*Z1
        let F = (E - Self::B * z1z2).mul3();
        let G = y1y2 - F;
        let H = y1y2 + F;
        let I = z1z2.mul3();
        let J = (Self::B * E - x1x2 - I).mul3();
        let K = x1x2.mul3() - I;
        let L = D * J;
        let M = K * J;
        let N = K * C;
        let Y3 = H * G + M;
        let X3 = H * C - L;
        let Z3 = G * D + N;

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
        let F = (E - Self::B * Z1).mul3();
        let G = y1y2 - F;
        let H = y1y2 + F;
        let I = Z1.mul3();
        let J = (Self::B * E - x1x2 - I).mul3();
        let K = x1x2.mul3() - I;
        let L = D * J;
        let M = K * J;
        let N = K * C;
        let Y3 = H * G + M;
        let X3 = H * C - L;
        let Z3 = G * D + N;

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

        // We need to remember whether the source was the neutral.
        let zn = Z.iszero();

        // Formulas from Bernstein-Lange 2007:
        // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective-3.html#doubling-dbl-2007-bl-2
        let s = (Y * Z).mul2();
        let w = ((X - Z) * (X + Z)).mul3();
        let R = Y * s;
        let ss = s.square();
        let RR = R.square();
        let B = (X * R).mul2();
        let h = w.square() - B - B;
        let Z3 = s * ss;
        let X3 = s * h;
        let Y3 = w * (B - h) - RR.mul2();

        // When Z = 0 (i.e. input is the neutral), this yields
        // (0:-27*X^6:0), which is a valid representation of the neutral
        // only if X != 0 (since we must keep Y != 0 at all times,
        // otherwise the addition formulas fail). However, we normally
        // have X = 0 in a neutral representation, so we get (0:0:0) in
        // that case, and it is not valid (it will make our point
        // addition formulas fail). We thus need to add a corrective
        // step to avoid getting the invalid (0:0:0) triplet.
        self.X = X3;
        self.Y = GFp256::select(&Y3, &GFp256::ONE, zn);
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
    ///
    /// When n > 1, this function is faster than calling `set_double()`
    /// n times.
    pub fn set_xdouble(&mut self, n: u32) {
        if n == 0 {
            return;
        }
        if n == 1 {
            self.set_double();
            return;
        }

        // If doing two or more doublings, we switch to Jacobian
        // coordinates temporarily.

        // The first doubling uses formulas that are derived from
        // the dbl-2007-bl-2 formulas we use in set_double(): we
        // can do the doubling AND convert to Jacobian coordinates
        // in cost 5M+2S.
        let (X, Y, Z) = (&self.X, &self.Y, &self.Z);
        let s = (Y * Z).mul2();
        let w = ((X - Z) * (X + Z)).mul3();
        let R = Y * s;
        let RR = R.square();
        let B = (X * R).mul2();
        let mut X = w.square() - B - B;
        let mut Y = w * (B - X) - RR.mul2();
        let mut Z = s;

        // We now are in Jacobian coordinates. We perform the remaining
        // doublings.
        for _ in 1..n {
            // Using Hankerson-Menezes-Vanstone 2004 formulas (4M+4S)
            let Z2 = Z.square();
            let A = ((X - Z2) * (X + Z2)).mul3();
            let B = Y.mul2();
            Z *= B;
            let C = B.square();
            let D = C.square().half();
            let E = C * X;
            X = A.square() - E.mul2();
            Y = (E - X) * A - D;
        }

        // Conversion back to projective.
        // Only special case is when the source was the neutral; conversion
        // to Jacobian yielded (0:0:0), which we still have here. We need
        // to set Y back to a non-zero value in that case.
        self.X = X * Z;
        self.Y = GFp256::select(&Y, &GFp256::ONE, Z.iszero());
        self.Z = Z * Z.square();
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
        let mut P = PointAffine { x: GFp256::ZERO, y: GFp256::ONE };
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
            Z: GFp256::select(&GFp256::ONE, &GFp256::ZERO, rz),
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
        for i in (0..257).rev() {
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
                    self.set_add_affine(&PRECOMP_G[e2 as usize - 1], 0);
                } else {
                    self.set_sub_affine(&PRECOMP_G[(-e2) as usize - 1], 0);
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

    /// From points P0 and P1, returns the affine x coordinates of P0, P1
    /// and P1 - P0, in that order.
    ///
    /// For the point-at-infinity (which does not have a defined x
    /// coordinate), value 1 is used (there is no point with x = 1 on the
    /// curve).
    ///
    /// These values are what `x_sequence_vartime()` expects.
    pub fn to_x_affine_diff(P0: Self, P1: Self) -> (GFp256, GFp256, GFp256) {
        let Q = P1 - P0;
        let mut x0 = P0.X;
        let mut z0 = P0.Z;
        x0.set_cond(&GFp256::ONE, P0.isneutral());
        z0.set_cond(&GFp256::ONE, P0.isneutral());
        let mut x1 = P1.X;
        let mut z1 = P1.Z;
        x1.set_cond(&GFp256::ONE, P1.isneutral());
        z1.set_cond(&GFp256::ONE, P1.isneutral());
        let mut xq = Q.X;
        let mut zq = Q.Z;
        xq.set_cond(&GFp256::ONE, Q.isneutral());
        zq.set_cond(&GFp256::ONE, Q.isneutral());

        let z0z1 = z0 * z1;
        let mut k = GFp256::ONE / (z0z1 * zq);
        xq *= k * z0z1;
        k *= zq;
        x1 *= k * z0;
        k *= z1;
        x0 *= k;
        (x0, x1, xq)
    }

    /// Given the x coordinates of points P0, P1 and Q (with Q = P1 - P0),
    /// computes the x coordinates of points P\_i = P0 + i*Q for i = 0
    /// to n-1, where n = `xx.len()`.
    ///
    /// The values are stored in the provided slice `xx[]`. Moreover, the
    /// x coordinates of P\_n and P\_(n+1) are returned.
    ///
    /// For the purposes of this function, the x coordinate of the
    /// point-at-infinity (the curve neutral element) is set to 1, both
    /// in inputs and outputs. There is no non-infinity point with x = 1
    /// on curve P-256. The x coordinates of P0, P1 and Q can be obtained
    /// from the `to_x_affine_diff()` function.
    /// 
    /// THIS FUNCTION IS NOT CONSTANT-TIME; it shall be used only with
    /// public data.
    pub fn x_sequence_vartime(x0: GFp256, x1: GFp256, xq: GFp256,
        xx: &mut [GFp256]) -> (GFp256, GFp256)
    {
        // We use general x-line arithmetics to get the x coordinate of
        // P_(i+2) from the x coordinates of P_i, P_(i+1) and P_(i+1)-P_i;
        // initial formulas are due to Brier and Joye; see also section
        // 3.4 of: https://eprint.iacr.org/2017/212
        // We change the notations because we are not computing a
        // Montgomery ladder.
        //
        // We represent x coordinates as fractions X / Z. For the neutral
        // (point-at-infinity), we set Z = 0 (the convention x = 1 is only
        // for input/output, not internally).
        //
        //    Input:
        //       x(P_i)     = X0 / Z0
        //       x(P_(i+1)) = X1 / Z1
        //       x(P_(i+2)) = X2 / Z2
        //       x(Q)       = xq
        //       Q = P_(i+1) - P_i
        //       P_(i+2) = P_(i+1) + P_i
        //    
        //    Assumptions:
        //       Q != inf
        //       P_i != inf      (this implies that P_(i+1) != Q)
        //       P_(i+1) != inf
        //       x(P_i) != 0
        //    
        //    Formulas:
        //       X2 = Z0*((X1*xq - a*Z1)^2 - 4*b*(X1 + xq*Z1)*Z1)
        //       Z2 = X0*(X1 - xq*Z1)^2
        //
        // Note that if P_(i+1) = -Q, then this yields Z2 = 0, i.e.
        // P_(i+1) = inf, which is correct.
        //
        // Case Q = inf can be handled easily: if Q = inf then
        // P_0 = P_1 = P_i for all i. We cover this case early in the
        // function; afterwards, we can assume that Q != inf.
        //
        // If P_i = inf, then P_(i+1) = Q, and P_(i+2) = 2*Q; we can use
        // the pseudo-doubling formulas to get 2*Q:
        //    x(2*Q) = X' / Z'
        //    X' = (xq^2 - a)^2 - 8*b*xq
        //    Z' = 4*(xq^3 + a*xq + b)
        // We'll get this case only once per call (at most) so we can
        // compute X' and Z' on the spot if the situation arises.
        //
        // If P_(i+1) = inf, then P_i = -Q, and P_(i+2) = Q.
        //
        // If x(P_i) = 0, then we have alternate formulas:
        //
        //    Input:
        //       x(P_i)     = 0         (i.e. P_i = (0, sqrt(b)))
        //       x(P_(i+1)) = X1 / Z1
        //       x(P_(i+2)) = X2 / Z2
        //       x(Q)       = xq
        //       Q = P_(i+1) - P_i
        //       P_(i+2) = P_(i+1) + P_i
        //    
        //    Assumptions:
        //       Q != inf
        //       P_(i+1) != inf
        //       P_(i+1) != Q   (implicit in x(P_i) = 0, i.e. P_i != inf)
        //    
        //    Formulas:
        //       X2 = 2*((X1*xq + a*Z1)*(X1 + xq*Z1) + 2*b*Z1^2)
        //       Z2 = (X1 - xq*Z1)^2
        //
        // Note that if P_(i+1) = -Q, then Z2 = 0, which is again the
        // correct result.

        // Pseudo-addition, general case.
        //    Input:
        //       x(P0) = X0 / Z0
        //       x(P1) = X1 / Z1
        //       x(Q)  = xq
        //       Q = P1 - P0
        //    
        //    Assumptions:
        //       Q != inf  (xq is properly defined)
        //       Z0 != 0   (P0 != inf)
        //       Z1 != 0   (P1 != inf)
        //       X0 != 0   (P0 != (0, +/-sqrt(b)))
        //
        //    Output:
        //       x(P1 + Q) = X2 / Z2
        fn xadd(X0: GFp256, Z0: GFp256, X1: GFp256, Z1: GFp256, xq: GFp256)
            -> (GFp256, GFp256)
        {
            // X2 = Z0*((X1*xq - a*Z1)^2 - 4*b*(X1 + xq*Z1)*Z1)
            // Z2 = X0*(X1 - xq*Z1)^2
            let C = xq * Z1;                  // C = xq*Z1
            let D = X1 * xq;                  // D = X1*xq
            let E = (X1 + C) * Z1;            // E = (X1 + xq*Z1)*Z1
            let F = (D + Z1.mul3()).square(); // F = (X1*xq - a*Z1)^2
            let G = E * Point::B4;            // G = 4*b*(X1 + xq*Z1)*Z1
            let H = (X1 - C).square();        // H = (X1 - xq*Z1)^2
            let X2 = Z0 * (F - G);
            let Z2 = X0 * H;
            (X2, Z2)
        }

        // Pseudo-addition, special case: x(P0) = 0
        //    Input:
        //       x(P0) = 0
        //       x(P1) = X1 / Z1
        //       x(Q)  = xq
        //       Q = P1 - P0
        //    
        //    Assumptions:
        //       Q != inf  (xq is properly defined)
        //       Z1 != 0   (P1 != inf)
        //
        //    Output:
        //       x(P1 + Q) = X2 / Z2
        fn xadd_spec(X1: GFp256, Z1: GFp256, xq: GFp256)
            -> (GFp256, GFp256)
        {
            // X2 = 2*((X1*xq + a*Z1)*(X1 + xq*Z1) + 2*b*Z1^2)
            // Z2 = (X1 - xq*Z1)^2
            let C = X1 * xq;
            let D = xq * Z1;
            let E = Z1.square();
            let F = (C - Z1.mul3()) * (X1 + D);
            let G = E * Point::B2;
            let Z2 = (X1 - D).square();
            let X2 = (F + G).mul2();
            (X2, Z2)
        }

        let n = xx.len();
        if n == 0 {
            return (x0, x1);
        }

        // Special case: Q is the point-at-infinity.
        if xq.equals(GFp256::ONE) != 0 {
            for i in 0..n {
                xx[i] = x0;
            }
            return (x0, x0);
        }

        // We keep x(P_i) = X0 / Z0 and x(P_(i+1)) = X1 / Z1; in
        // fractional representation, we use Z = 0 for the point-at-infinity.
        let mut X0 = x0;
        let mut Z0 = if x0.equals(GFp256::ONE) != 0 {
            GFp256::ZERO
        } else {
            GFp256::ONE
        };
        let mut X1 = x1;
        let mut Z1 = if x1.equals(GFp256::ONE) != 0 {
            GFp256::ZERO
        } else {
            GFp256::ONE
        };

        let mut i = 0;
        loop {
            // We use batches of up to 198 values so that we can perform
            // batch normalization to affine coordinates. The fixed
            // maximum batch size allows std-less stack allocation.
            // We make arrays with 200 slots so that we may also normalize
            // the two output points with the final batch.
            //
            // (Length 200 was chosen so as to match the batch size used
            // by batch_invert().)
            let blen = if (n - i) < 198 { n - i } else { 198 };
            let mut XX = [GFp256::ZERO; 200];
            let mut ZZ = [GFp256::ZERO; 200];
            for j in 0..blen {
                // Write the current P_i in the array.
                XX[j] = X0;
                ZZ[j] = Z0;

                // Compute P_(i+2).
                let (X2, Z2) = if Z0.iszero() != 0 {
                    // P_i = inf
                    // P_(i+1) = Q
                    // P_(i+2) = 2*Q
                    let xqxq = xq.square();
                    let Xt = (xqxq + Self::THREE).square() - xq * Self::B8;
                    let Zt = ((xqxq - Self::THREE) * xq + Self::B).mul4();
                    (Xt, Zt)
                } else if Z1.iszero() != 0 {
                    // P_(i+1) = inf
                    // P_(i+2) = Q
                    (xq, GFp256::ONE)
                } else if X0.iszero() != 0 {
                    // P_i = (0, +/-sqrt(b))
                    xadd_spec(X1, Z1, xq)
                } else {
                    // General case
                    xadd(X0, Z0, X1, Z1, xq)
                };
                (X0, Z0) = (X1, Z1);
                (X1, Z1) = (X2, Z2);
            }

            // We also add the two output points when processing the
            // final batch.
            let ilen = if (i + blen) == n {
                XX[blen] = X0;
                ZZ[blen] = Z0;
                XX[blen + 1] = X1;
                ZZ[blen + 1] = Z1;
                blen + 2
            } else {
                blen
            };

            // Normalize the batch to affine coordinates (applying the
            // convention that the point-at-infinity normalizes to 1).
            GFp256::batch_invert(&mut ZZ[0..ilen]);
            for j in 0..ilen {
                let iZ = ZZ[j];
                let mut x = XX[j] * iZ;
                if iZ.iszero() != 0 {
                    x = GFp256::ONE;
                }
                if j < blen {
                    xx[i + j] = x;
                } else {
                    XX[j] = x;
                }
            }

            // Batch process, get to the next one.
            i += blen;
            if i == n {
                // We are finished; the two output points were normalized
                // with the last batch.
                return (XX[blen], XX[blen + 1]);
            }
        }
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

/// A P-256 private key simply wraps around a scalar.
#[derive(Clone, Copy, Debug)]
pub struct PrivateKey {
    x: Scalar,   // secret scalar
}

/// A P-256 public key simply wraps around a curve point.
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
        // A custom prefix ("crrl P-256" in ASCII) is used to avoid
        // collisions.
        let mut sh = Sha512::new();
        sh.update(&[ 0x63, 0x72, 0x72, 0x6c, 0x20,
                     0x50, 0x2d, 0x32, 0x35, 0x36 ]);
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
    /// If `extra_rand` has length 0, and `hv` is indeed the SHA-256 hash
    /// of the actual message, then the signature generation process
    /// follows RFC 6979.
    pub fn sign_hash(self, hv: &[u8], extra_rand: &[u8]) -> [u8; 64] {

        // Feed a SHA-256 context with the starter block for HMAC/SHA-256,
        // using a 32-byte key.
        fn hmac_start(sh: &mut Sha256, key: &[u8; 32]) {
            let mut tmp = [0x36u8; 64];
            for i in 0..32 {
                tmp[i] ^= key[i];
            }
            sh.update(&tmp);
        }

        // Finalize a HMAC/SHA-256 computation; the 32-byte key is provided
        // again. The SHA-256 context is automatically reinitialized.
        fn hmac_end(sh: &mut Sha256, key: &[u8; 32]) -> [u8; 32] {
            let v = sh.finalize_reset();
            let mut tmp = [0x5Cu8; 64];
            for i in 0..32 {
                tmp[i] ^= key[i];
            }
            sh.update(&tmp);
            sh.update(&v);
            sh.finalize_reset().into()
        }

        // Convert the input hash value into an integer modulo n:
        //  - If hv.len() > 32, keep only the leftmost 32 bytes.
        //  - Interpret the value as big-endian.
        //  - Reduce the integer modulo n.
        // The result is h. We also re-encode h over 32 bytes (exactly),
        // in unsigned big-endian notation, to get hb (in RFC 6979
        // notations, h = bits2int(hv), and hb = bits2octets(hv)).
        let mut tmp = [0u8; 32];
        if hv.len() >= 32 {
            tmp[..].copy_from_slice(&hv[..32]);
        } else {
            tmp[(32 - hv.len())..32].copy_from_slice(hv);
        }
        let h = Scalar::decode_reduce(&bswap32(&tmp));
        let hb = bswap32(&h.encode());

        // Get the byte representation of the private key itself.
        let xb = bswap32(&self.x.encode());

        // Generate a pseudorandom k as per RFC 6979, section 3.2.
        let mut sh = Sha256::new();
        let V = [0x01u8; 32];
        let K = [0x00u8; 32];

        // 3.2.d
        hmac_start(&mut sh, &K);
        sh.update(&V);
        sh.update(&[0x00u8]);
        sh.update(&xb);
        sh.update(&hb);
        if extra_rand.len() > 0 {
            sh.update(&extra_rand);
        }
        let K = hmac_end(&mut sh, &K);

        // 3.2.e
        hmac_start(&mut sh, &K);
        sh.update(&V);
        let V = hmac_end(&mut sh, &K);

        // 3.2.f
        hmac_start(&mut sh, &K);
        sh.update(&V);
        sh.update(&[0x01u8]);
        sh.update(&xb);
        sh.update(&hb);
        if extra_rand.len() > 0 {
            sh.update(&extra_rand);
        }
        let mut K = hmac_end(&mut sh, &K);

        // 3.2.g
        hmac_start(&mut sh, &K);
        sh.update(&V);
        let mut V = hmac_end(&mut sh, &K);

        // 3.2.h
        // We loop in case we get a zero for k or for s (either case is
        // so improbable that it won't happen in practice).
        loop {
            // Get k. Since SHA-256 outputs 256 bits, and the curve order
            // has size 256 bits as well, we only need one HMAC call, with
            // no truncation.
            hmac_start(&mut sh, &K);
            sh.update(&V);
            V[..].copy_from_slice(&hmac_end(&mut sh, &K));
            let (k, cc) = Scalar::decode32(&bswap32(&V));
            if cc != 0 && k.iszero() == 0 {
                // We got k, compute the signature.

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
            }

            // Bad k, try again (very improbable).
            hmac_start(&mut sh, &K);
            sh.update(&V);
            sh.update(&[0x00u8]);
            let nK = hmac_end(&mut sh, &K);
            K[..].copy_from_slice(&nK);
            hmac_start(&mut sh, &K);
            sh.update(&V);
            V[..].copy_from_slice(&hmac_end(&mut sh, &K));
        }
    }

    /// Prepares a signature value for truncation.
    ///
    ///  - Signature is parsed into (r,s) values (unsigned big-endian).
    ///
    ///  - If s >= 2^255 then it is replaced with -s (mod n).
    ///
    ///  - s is reencoded in little-endian format.
    ///
    /// A failure is reported (`None` is returned) if r or s is
    /// out-of-range (invalid signature) or if r < p-n (with p = modulus,
    /// n = curve order). The latter may theoretically happen with
    /// probability about 2^(-128.9), i.e. never in practice (or, more
    /// accurately, when it seems to happen, it is much more likely to be
    /// due to a hardware failure than to the value falling in that
    /// range).
    ///
    /// This function does not use the private key; it was defined in the
    /// `PrivateKey` structure only because in a typical context where
    /// truncated signatures are relevant, this operation should happen
    /// on the signer's side (i.e. after signature generation but before
    /// transmission to the verifier).
    pub fn prepare_truncate(sig: &[u8]) -> Option<[u8; 64]> {
        // Ensure that the signature has length exactly 64 bytes
        // (Shorter lengths are possible if the source integers happen
        // to be both lower than 2^248).
        let siglen = sig.len();
        if (siglen & 1) != 0 || siglen == 0 || siglen > 64 {
            return None;
        }
        let numlen = siglen >> 1;
        let mut tmp = [0u8; 64];
        tmp[(32 - numlen)..32].copy_from_slice(&sig[..numlen]);
        tmp[(64 - numlen)..64].copy_from_slice(&sig[numlen..]);

        // Decode each of r and s with unsigned big-endian convention;
        // we obtain the high and low halves of each as 128-bit integers.
        use core::convert::TryFrom;
        let rh = u128::from_be_bytes(*<&[u8; 16]>::try_from(
            &tmp[ 0..16]).unwrap());
        let rl = u128::from_be_bytes(*<&[u8; 16]>::try_from(
            &tmp[16..32]).unwrap());
        let mut sh = u128::from_be_bytes(*<&[u8; 16]>::try_from(
            &tmp[32..48]).unwrap());
        let mut sl = u128::from_be_bytes(*<&[u8; 16]>::try_from(
            &tmp[48..64]).unwrap());

        // Check ranges:
        //   p-n <= r < n
        //   0 < s < n
        const NH: u128 = 340282366841710300967557013911933812735u128;
        const NL: u128 = 251094175845612772866266697226726352209u128;
        const PMN: u128 = 89188191154553853111372247798585809582u128;

        if (rh == 0 && rl < PMN) || rh > NH || (rh == NH && rl >= NL)
            || (sh == 0 && sl == 0) || sh > NH || (sh == NH && sl >= NL)
        {
            return None;
        }

        // If s does not fit in 255 bits, then replace it with n - s.
        if tmp[32] >= 0x80 {
            sl = NL.wrapping_sub(sl);
            sh = NH.wrapping_sub(sh);
            if sl > NL {
                sh = sh.wrapping_sub(1);
            }
        }

        // Reencode r and s. r was not changed from the source signature;
        // s was possibly changed, and we want s in little-endian format.
        let mut nsig = [0u8; 64];
        nsig[..32].copy_from_slice(&sig[..32]);
        nsig[32..48].copy_from_slice(&sl.to_le_bytes());
        nsig[48..64].copy_from_slice(&sh.to_le_bytes());

        Some(nsig)
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

    /// Verifies a truncated signature on a given hashed message.
    ///
    /// The signature (`sig`) MUST have length 64 bytes and MUST have
    /// been prepated with `PrivateKey::prepare_truncate()`. The last
    /// `rm` bits are ignored (i.e. the last `floor(rm/8)` bytes are
    /// ignored, as well as the top `rm%8` bits of the last non-ignored
    /// byte.
    ///
    /// The hashed message is provided as `hv`; it is nominally the
    /// output of a suitable hash function (often SHA-256) computed over
    /// the actual message. This function can tolerate arbitrary hash
    /// output lengths; however, for proper security, the hash output
    /// must not be too short, and it must be an actual hash function
    /// output, not raw structured data.
    ///
    /// Returned value on success is the complete, untruncated signature
    /// (reencoded in the standard all-big-endian format); otherwise,
    /// `None` is returned (if a rebuilt signature value is returned,
    /// then it has been verified to be valid and there is no need to
    /// validate it again).
    ///
    /// Note: this function is not constant-time; it assumes that the
    /// public key and signature value are public data.
    ///
    /// Note: this function is available only if heap allocation support
    /// was configured. Only public data is allocated on the heap.
    #[cfg(feature = "alloc")]
    pub fn verify_trunc_hash(self,
        sig: &[u8], rm: usize, hv: &[u8]) -> Option<[u8; 64]>
    {
        // Check that we removed at least 8 bits. We also prevent trying
        // to remove more than 32 bits because the cost would be excessive;
        // it also allows keeping U_i[] indices over 16 bits.
        assert!(rm >= 8 && rm <= 32);

        // Signature array must have length 64 bytes exactly; but
        // we ignore the last rm bits. We copy the non-ignored bits
        // to sig2, and clear the rest.
        if sig.len() != 64 {
            return None;
        }
        let n = (519 - rm) >> 3;
        let mut sig2 = [0u8; 64];
        sig2[0..n].copy_from_slice(&sig[0..n]);
        if (rm & 7) != 0 {
            sig2[n - 1] &= 0xFFu8 >> (rm & 7);
        }

        // First half of the signature is r, which is supposed to be the
        // x coordinate of the point R (in big-endian format). We assume
        // that the x value was not altered when reduced modulo n (since
        // n is very close to p, alteration is very improbable; moreover,
        // this assumption cannot make us accept an invalid signature).
        // If no matching point R can be found, then we report a failure.
        let mut R_enc = [0u8; 33];
        R_enc[0] = 0x02;
        R_enc[1..33].copy_from_slice(&sig2[..32]);
        let R = Point::decode(&R_enc)?;

        // We also want r as a scalar; and we are expected to reject
        // signatures with r = 0.
        let (r, cr) = Scalar::decode32(&bswap32(&sig2[..32]));
        if cr == 0 || r.iszero() != 0 {
            return None;
        }

        // Second half of the signature is s0 (in little-endian). Since we
        // ensured that at least one top bit was cleared, the value cannot be
        // out-of-range.
        let (s0, _) = Scalar::decode32(&sig2[32..64]);

        // Convert the input hash value into an integer modulo n.
        let mut tmp = [0u8; 32];
        if hv.len() >= 32 {
            tmp[..].copy_from_slice(&hv[..32]);
        } else {
            tmp[32 - hv.len() .. 32].copy_from_slice(hv);
        }
        let h = Scalar::decode_reduce(&bswap32(&tmp));

        // Signature verification equation can be written as:
        //   s*R = h*G + r*Q
        // Note that our R might be -R instead, since we only had the x
        // coordinate of the point. The signature is also valid if the
        // alternate equation matches:
        //   s*R = -(h*G + r*Q)
        //
        // We know that s fits on 255 bits (this was ensured by the
        // prepare_truncate() function), so we can write:
        //   s = s0 + s1*2^n
        // with n = 256 - rm (the number of bits of s that we received),
        // s0 the value that we could decode from the truncated signature
        // (0 <= s0 < 2^n), and s1 such that:
        //   0 <= s1 < 2^m
        // for m = 255 - n.
        // Let a and b such that:
        //   s1 = a + 2^k*b
        // with:
        //   k = ceil(m/2)
        //   0 <= a < 2^k
        //   0 <= b < +2^(m-k)
        //
        // Let:
        //   U = (2^n)*R
        //   V = h*G + r*Q
        //   U_i = s0*R + i*(2^k)*U  for 0 <= i <= +2^(m-k)
        //   V_j = V - j*U           for 0 <= j <= +2^k
        // If s*R = h*G + r*Q, then:
        //   s0*R + (a + b*2^k)*(2^n)*R = V
        // hence:
        //   U_b = V_a
        // If instead we used the wrong sign for R, and the equation
        // really is s*R = -(h*G + r*Q), then:
        //   -(s0*R + (a + b*2^k)*(2^n)*R) = V
        // hence:
        //   -(s0*R + (b + 1)*(2^k)*U) + (2^k - a)*U = V
        //   -U_(b+1) = V_(2^k-a)
        // (Note that we included 2^(m-k) in the range of i, hence U_(b+1)
        // is part of the list of computed U_i values; similarly, we
        // includes 2^k in the range of j, hence V_(2^k-a) is also part of
        // the list of computed V_j values.)
        //
        // Since points P and -P have the same x coordinate, it follows
        // that in both cases, one of the U_i and one of the V_j will have
        // the same x coordinate. If U_i and V_j have the same x coordinate,
        // the the potential solutions (a,b) are (j,i) and (2^k-a,i-1).
        // We can reconstruct s1 as:
        //   s1 = a + b*2^k = j + i*2^k (first case)
        //   s1 = 2^k - j + (i - 1)*2^k = -j + i*2^k (second case)
        //
        // We compute all U_i, then extract 64 bits out of each x coordinate
        // and sort them so that we may do efficient binary searches. We
        // then compute all V_j and perform lookups. We proceed by batches
        // since normalization to affine coordinates is much more efficient
        // that way.

        let n = 256 - rm;      // s0 has size n bits
        let m = 255 - n;
        let k = (m + 1) >> 1;
        let U = R.xdouble(n as u32);
        let V = self.point.mul_add_mulgen_vartime(&r, &h);
        let Rb = s0 * R;

        // 0 <= i <= I
        // 0 <= j <= J
        let I = 1usize << (m - k);
        let J = 1usize << k;

        // Compute all U_i = s0*R + i*(2^k)*U for 0 <= i <= +2^(m-k)
        // Since 8 <= rm <= 32, we have 7 <= m <= 31, hence m-k <= 15.
        // We only need the x coordinates.
        let Uk = U.xdouble(k as u32);
        let mut Ui = Vec::with_capacity(I + 1);
        Ui.resize(I + 1, GFp256::ZERO);
        let (x0, x1, xq) = Point::to_x_affine_diff(Rb, Rb + Uk);
        Point::x_sequence_vartime(x0, x1, xq, &mut Ui[..]);

        let mut ux = Vec::with_capacity(Ui.len());
        for i in 0..Ui.len() {
            let mut tmp = [0u8; 8];
            tmp[0..2].copy_from_slice(&(i as u16).to_le_bytes());
            tmp[2..8].copy_from_slice(&Ui[i].encode()[0..6]);
            ux.push(u64::from_le_bytes(tmp));
        }
        ux.sort();

        // For all V_j = V - j*U, look for the x coordinates among those
        // of the U_i; we use the ux[] array for that, and confirm any
        // match with the complete X coordinates in the Ui[] array.
        // We compute the V_j by batches of 100 so that normalization
        // to affine coordinates is properly optimized, but we still
        // stop early when possible.
        let (mut x0, mut x1, xq) = Point::to_x_affine_diff(V, V - U);
        let mut jj = 0;
        loop {
            let mut Vj = [GFp256::ZERO; 100];
            let mut blen = J + 1 - jj;
            if blen > 100 {
                blen = 100;
            }
            (x0, x1) = Point::x_sequence_vartime(x0, x1, xq, &mut Vj);
            for j in 0..blen {
                // Extract the search key from the x coordinate of V_j.
                // We set the low 16 bits to 1, so that potential matches
                // are keys which are lower than this value.
                let mut tmp = [0u8; 8];
                tmp[0] = 0xFF;
                tmp[1] = 0xFF;
                tmp[2..8].copy_from_slice(&Vj[j].encode()[0..6]);
                let x = u64::from_le_bytes(tmp);

                // Perform a search in ux[].
                if x < ux[0] {
                    continue;
                }
                let mut i1 = 0usize;
                let mut i2 = ux.len();
                while (i2 - i1) > 1 {
                    let im = (i1 + i2) >> 1;
                    if x < ux[im] {
                        i2 = im;
                    } else {
                        i1 = im;
                    }
                }

                // Potential matches are lower than x but have the same high
                // 48 bits.
                loop {
                    let xm = ux[i1];
                    if ((xm ^ x) >> 16) != 0 {
                        break;
                    }
                    let i = (xm & 0xFFFF) as usize;
                    if Vj[j].equals(Ui[i]) != 0 {
                        // U_i and V_j have the same x coordinate. Therefore,
                        // one of the following holds:
                        //   U_i = V_j
                        //  -U_i = V_j
                        // First case yields:
                        //   s0*R + i*(2^k)*U = h*G + r*Q - j*U
                        // i.e.:
                        //   s = s0 + (j + i*2^k)*2^n
                        // Second case yields:
                        //  -(s0*R + i*(2^k)*U) = h*G + r*Q - j*U
                        // i.e.:
                        //   s = s0 + (-j + i*2^k)*2^n
                        //
                        // We do not have the complete U_i and V_j (we
                        // computed only the x coordinates) but we can
                        // recompute them cheaply using the values we
                        // already have:
                        //   Rb = s0*R
                        //   U = (2^n)*R
                        //   Uk = (2^k)*(2^n)*R
                        //   V = h*G + r*Q
                        let j = jj + j;  // the "real" j
                        let zUi = Rb + (i as u64) * Uk;
                        let zVj = V - (j as u64) * U;
                        let ni = Scalar::from_w64le(0, 0, 0,
                            (i as u64) << (n + k - 192));
                        let nj = Scalar::from_w64le(0, 0, 0,
                            (j as u64) << (n - 192));
                        let s = if zUi.equals(zVj) != 0 {
                            s0 + ni + nj
                        } else {
                            debug_assert!(zUi.equals(-zVj) != 0);
                            s0 + ni - nj
                        };
                        // sig2[] already contains r, we just have to encode
                        // the complete s in it.
                        sig2[32..64].copy_from_slice(&bswap32(&s.encode()));
                        return Some(sig2);
                    }

                    if i1 == 0 {
                        break;
                    }
                    i1 -= 1;
                }
            }

            jj += blen;
            if jj >= J + 1 {
                break;
            }
        }

        // No match; signature is not valid.
        None
    }
}

// ========================================================================

// We hardcode known multiples of the points B, (2^65)*B, (2^130)*B
// and (2^195)*B, with B being the conventional base point. These are
// used to speed mulgen() operations up. The points are stored in affine
// coordinates, i.e. their Z coordinate is implicitly equal to 1.

/// A curve point (non-infinity) in affine coordinates.
#[derive(Clone, Copy, Debug)]
struct PointAffine {
    x: GFp256,
    y: GFp256,
}

// Points i*G for i = 1 to 16, in affine coordinates.
static PRECOMP_G: [PointAffine; 16] = [
    // G * 1
    PointAffine { x: GFp256::w64be(0x6B17D1F2E12C4247, 0xF8BCE6E563A440F2,
                                   0x77037D812DEB33A0, 0xF4A13945D898C296),
                  y: GFp256::w64be(0x4FE342E2FE1A7F9B, 0x8EE7EB4A7C0F9E16,
                                   0x2BCE33576B315ECE, 0xCBB6406837BF51F5) },
    // G * 2
    PointAffine { x: GFp256::w64be(0x7CF27B188D034F7E, 0x8A52380304B51AC3,
                                   0xC08969E277F21B35, 0xA60B48FC47669978),
                  y: GFp256::w64be(0x07775510DB8ED040, 0x293D9AC69F7430DB,
                                   0xBA7DADE63CE98229, 0x9E04B79D227873D1) },
    // G * 3
    PointAffine { x: GFp256::w64be(0x5ECBE4D1A6330A44, 0xC8F7EF951D4BF165,
                                   0xE6C6B721EFADA985, 0xFB41661BC6E7FD6C),
                  y: GFp256::w64be(0x8734640C4998FF7E, 0x374B06CE1A64A2EC,
                                   0xD82AB036384FB83D, 0x9A79B127A27D5032) },
    // G * 4
    PointAffine { x: GFp256::w64be(0xE2534A3532D08FBB, 0xA02DDE659EE62BD0,
                                   0x031FE2DB785596EF, 0x509302446B030852),
                  y: GFp256::w64be(0xE0F1575A4C633CC7, 0x19DFEE5FDA862D76,
                                   0x4EFC96C3F30EE005, 0x5C42C23F184ED8C6) },
    // G * 5
    PointAffine { x: GFp256::w64be(0x51590B7A515140D2, 0xD784C85608668FDF,
                                   0xEF8C82FD1F5BE524, 0x21554A0DC3D033ED),
                  y: GFp256::w64be(0xE0C17DA8904A727D, 0x8AE1BF36BF8A7926,
                                   0x0D012F00D4D80888, 0xD1D0BB44FDA16DA4) },
    // G * 6
    PointAffine { x: GFp256::w64be(0xB01A172A76A4602C, 0x92D3242CB897DDE3,
                                   0x024C740DEBB215B4, 0xC6B0AAE93C2291A9),
                  y: GFp256::w64be(0xE85C10743237DAD5, 0x6FEC0E2DFBA70379,
                                   0x1C00F7701C7E16BD, 0xFD7C48538FC77FE2) },
    // G * 7
    PointAffine { x: GFp256::w64be(0x8E533B6FA0BF7B46, 0x25BB30667C01FB60,
                                   0x7EF9F8B8A80FEF5B, 0x300628703187B2A3),
                  y: GFp256::w64be(0x73EB1DBDE0331836, 0x6D069F83A6F59000,
                                   0x53C73633CB041B21, 0xC55E1A86C1F400B4) },
    // G * 8
    PointAffine { x: GFp256::w64be(0x62D9779DBEE9B053, 0x4042742D3AB54CAD,
                                   0xC1D238980FCE97DB, 0xB4DD9DC1DB6FB393),
                  y: GFp256::w64be(0xAD5ACCBD91E9D824, 0x4FF15D771167CEE0,
                                   0xA2ED51F6BBE76A78, 0xDA540A6A0F09957E) },
    // G * 9
    PointAffine { x: GFp256::w64be(0xEA68D7B6FEDF0B71, 0x878938D51D71F872,
                                   0x9E0ACB8C2C6DF8B3, 0xD79E8A4B90949EE0),
                  y: GFp256::w64be(0x2A2744C972C9FCE7, 0x87014A964A8EA0C8,
                                   0x4D714FEAA4DE823F, 0xE85A224A4DD048FA) },
    // G * 10
    PointAffine { x: GFp256::w64be(0xCEF66D6B2A3A993E, 0x591214D1EA223FB5,
                                   0x45CA6C471C48306E, 0x4C36069404C5723F),
                  y: GFp256::w64be(0x878662A229AAAE90, 0x6E123CDD9D3B4C10,
                                   0x590DED29FE751EEE, 0xCA34BBAA44AF0773) },
    // G * 11
    PointAffine { x: GFp256::w64be(0x3ED113B7883B4C59, 0x0638379DB0C21CDA,
                                   0x16742ED0255048BF, 0x433391D374BC21D1),
                  y: GFp256::w64be(0x9099209ACCC4C8A2, 0x24C843AFA4F4C68A,
                                   0x090D04DA5E9889DA, 0xE2F8EEFCE82A3740) },
    // G * 12
    PointAffine { x: GFp256::w64be(0x741DD5BDA817D95E, 0x4626537320E5D551,
                                   0x79983028B2F82C99, 0xD500C5EE8624E3C4),
                  y: GFp256::w64be(0x0770B46A9C385FDC, 0x567383554887B154,
                                   0x8EEB912C35BA5CA7, 0x1995FF22CD4481D3) },
    // G * 13
    PointAffine { x: GFp256::w64be(0x177C837AE0AC495A, 0x61805DF2D85EE2FC,
                                   0x792E284B65EAD58A, 0x98E15D9D46072C01),
                  y: GFp256::w64be(0x63BB58CD4EBEA558, 0xA24091ADB40F4E72,
                                   0x26EE14C3A1FB4DF3, 0x9C43BBE2EFC7BFD8) },
    // G * 14
    PointAffine { x: GFp256::w64be(0x54E77A001C3862B9, 0x7A76647F4336DF3C,
                                   0xF126ACBE7A069C5E, 0x5709277324D2920B),
                  y: GFp256::w64be(0xF599F1BB29F43175, 0x42121F8C05A2E7C3,
                                   0x7171EA7773509008, 0x1BA7C82F60D0B375) },
    // G * 15
    PointAffine { x: GFp256::w64be(0xF0454DC6971ABAE7, 0xADFB378999888265,
                                   0xAE03AF92DE3A0EF1, 0x63668C63E59B9D5F),
                  y: GFp256::w64be(0xB5B93EE3592E2D1F, 0x4E6594E51F9643E6,
                                   0x2A3B21CE75B5FA3F, 0x47E59CDE0D034F36) },
    // G * 16
    PointAffine { x: GFp256::w64be(0x76A94D138A6B4185, 0x8B821C629836315F,
                                   0xCD28392EFF6CA038, 0xA5EB4787E1277C6E),
                  y: GFp256::w64be(0xA985FE61341F260E, 0x6CB0A1B5E11E8720,
                                   0x8599A0040FC78BAA, 0x0E9DDD724B8C5110) },
];

// Points i*(2^65)*G for i = 1 to 16, in affine coordinates.
static PRECOMP_G65: [PointAffine; 16] = [
    // (2^65)*G * 1
    PointAffine { x: GFp256::w64be(0x031A8747DF8DC746, 0xE4C13D0306960801,
                                   0x53FE448A57324591, 0x794A16BAA05F57B5),
                  y: GFp256::w64be(0x883A2C64FDA8D586, 0x60E8AA6C1E387A32,
                                   0x1431C18C42B8DEF2, 0x1827EE579C0343FD) },
    // (2^65)*G * 2
    PointAffine { x: GFp256::w64be(0xA7163C2B9B973C17, 0xF9571975C0D5934A,
                                   0x4ECCAC6096513CCA, 0x015E2E65580B2322),
                  y: GFp256::w64be(0x308A9A797AF31FA5, 0x63389991545A6B7A,
                                   0xB841A4F4E09952D7, 0x3933B2232197FFE9) },
    // (2^65)*G * 3
    PointAffine { x: GFp256::w64be(0x3C714524875D4EED, 0xE22A0772517690AC,
                                   0xA18159998EF9AFB9, 0x9AFBD3916A334020),
                  y: GFp256::w64be(0x7F090565771AAF5C, 0x4889E729FF7F3D8C,
                                   0x152FCAAFE7938C2C, 0x6102A85CBA701655) },
    // (2^65)*G * 4
    PointAffine { x: GFp256::w64be(0x1BFFAB8C03AB8279, 0x811B3923EF4B991F,
                                   0x02A50C382DB2670C, 0x6CB04A6A5C42A280),
                  y: GFp256::w64be(0x2982B620CEB1D098, 0x332AC758529F1AAA,
                                   0x982A369AA9D24FB6, 0xC9C742364DDB9261) },
    // (2^65)*G * 5
    PointAffine { x: GFp256::w64be(0xD9A822BA07B6BC2A, 0xFB88219241F2D6C5,
                                   0x9E5CB1D7B595F7FF, 0xF0EBCB0ED3444C74),
                  y: GFp256::w64be(0x48E4749ECB370ECB, 0x0E7B8740FF7B504F,
                                   0x2D6ABBD72AEF8A78, 0x5F9ED1AFC2338891) },
    // (2^65)*G * 6
    PointAffine { x: GFp256::w64be(0x66AA4FD12ADD747A, 0x9D76B8258B28B28C,
                                   0x18B0C59B8AC074B7, 0xE788E04AB021A9BE),
                  y: GFp256::w64be(0x10C65E609047474C, 0xCA94F6C567E1DC3F,
                                   0x9D64422D106AA108, 0x88B5A3AC03A15A99) },
    // (2^65)*G * 7
    PointAffine { x: GFp256::w64be(0xE31D414BC13EA842, 0x7C2B1A4EBB312CC8,
                                   0xD10A694EA5FF8400, 0xEF0F43B30A7338BF),
                  y: GFp256::w64be(0xF8AB200A672A9E53, 0xA87559754BF051FC,
                                   0x35CEE4F5D7185BC0, 0xD23C40B0F878F170) },
    // (2^65)*G * 8
    PointAffine { x: GFp256::w64be(0x54BC18D7A9989954, 0x7DDC6988D7EE1B3F,
                                   0x2B481AB443DA43FF, 0x68F41305B76A6987),
                  y: GFp256::w64be(0x4B2C8C1211E6EAF3, 0x7391B851BA73E2FD,
                                   0x52EB8ED4BB73B119, 0xFE457CD05B9AAE49) },
    // (2^65)*G * 9
    PointAffine { x: GFp256::w64be(0x2C8AA1BBBF526DA4, 0xE347384662E54903,
                                   0xCE0DC533C33CBF11, 0xBDA3E5F081CEC610),
                  y: GFp256::w64be(0x1A0FC88AF38D3CB8, 0x6BE3230B9C93D0DF,
                                   0x2FE6EA380C43AB69, 0x469EA1B6B2D38B5B) },
    // (2^65)*G * 10
    PointAffine { x: GFp256::w64be(0xD3DB9B013B334BB5, 0x64A8C9CFFE7D63E6,
                                   0x395E493A27511BEF, 0x5E5E48320038A4D3),
                  y: GFp256::w64be(0xAF36ED9E441C40EE, 0xDDEC0D6B3E427380,
                                   0xF702E94394A0415C, 0xF5885A882DE9744C) },
    // (2^65)*G * 11
    PointAffine { x: GFp256::w64be(0xA6C3411CBD535430, 0xA9E5CF878C1CFAEA,
                                   0xB1676B74715F9C2A, 0x904CCE874666F4DF),
                  y: GFp256::w64be(0x2A2C152EB75FA9B1, 0x4AEC9626316F4989,
                                   0x135DC727867A1E9D, 0x3ACEF8A2C01456D5) },
    // (2^65)*G * 12
    PointAffine { x: GFp256::w64be(0xAC3857170B8B6252, 0x8C14F0F0B8ACBEAE,
                                   0x074D3FB83DB5ADF3, 0xAAFF87BF8A68FD2D),
                  y: GFp256::w64be(0x07726BD947412104, 0x78658166B27E4C56,
                                   0xDC5689FD1DD39426, 0x65D518497B48707B) },
    // (2^65)*G * 13
    PointAffine { x: GFp256::w64be(0xE2ADA2707FC3F4F2, 0x28219C10361C0666,
                                   0xED04CB3E11151FAE, 0xC71275A29DDF5AF3),
                  y: GFp256::w64be(0x163BA05BF87116C8, 0x447F9D6EC3ECB071,
                                   0x93E3591219126A1B, 0x0D4C0667ACFB630D) },
    // (2^65)*G * 14
    PointAffine { x: GFp256::w64be(0x2CEFE861BF2C3184, 0xAB2426302BFC3BC7,
                                   0xB410EC4B7440CCA5, 0x68E1C196CDBADC1D),
                  y: GFp256::w64be(0x1FCBC458DF871736, 0x0F1B16C5D31AECDF,
                                   0x0BB3958FF42B1056, 0xE52E15FFD37537AC) },
    // (2^65)*G * 15
    PointAffine { x: GFp256::w64be(0xE56081647CE4BC50, 0x579CE8BD6E1C6731,
                                   0xE06F5F6D466A7576, 0x199ACEC99B94CC00),
                  y: GFp256::w64be(0xAA059E3B7C2C9C86, 0xC60C19BEF5048D65,
                                   0x8C62E0BA91735EF8, 0xB5DDC1B356AC47C7) },
    // (2^65)*G * 16
    PointAffine { x: GFp256::w64be(0x1F380071781DFF16, 0xF33D8173A6FB4D96,
                                   0xBA770F18355CCA4C, 0xB2A64C6196260250),
                  y: GFp256::w64be(0xB1521EC4BA6C681D, 0xA33705176CA2CF62,
                                   0x1B8567660590D693, 0xB1519B2E6B011955) },
];

// Points i*(2^130)*G for i = 1 to 16, in affine coordinates.
static PRECOMP_G130: [PointAffine; 16] = [
    // (2^130)*G * 1
    PointAffine { x: GFp256::w64be(0x2890D721E57E1961, 0x18E63ADD579547F0,
                                   0xACAD16E63BE0AEA8, 0xF6F1D3AC4D771F0C),
                  y: GFp256::w64be(0x69B5B8159DDC032A, 0xAF77E1D1416752EC,
                                   0x7DC0E7F77EF54069, 0x0A5728ECB5890D78) },
    // (2^130)*G * 2
    PointAffine { x: GFp256::w64be(0x7B8B8867DD4D9C6F, 0x81690628033DE305,
                                   0xD4E3FF5E71DA1781, 0x4F5822EFAC7E9E44),
                  y: GFp256::w64be(0x8BC6273E511680B4, 0x2246073C7DD5ADB5,
                                   0x64057E6DBBD72365, 0x0767022AA8CD12C3) },
    // (2^130)*G * 3
    PointAffine { x: GFp256::w64be(0x05F0C3CABE40EDE9, 0x02FAEA8E759ACD21,
                                   0xCF8A00249E4D1558, 0xEEE143C1F0F870D2),
                  y: GFp256::w64be(0x34B55D934F6D2C56, 0xBCE7A5998AB1AC23,
                                   0xC2F725737AEB5F8A, 0x5F537C956E60E8CC) },
    // (2^130)*G * 4
    PointAffine { x: GFp256::w64be(0x9022E314949CCF3E, 0x8937542B8CDEC18E,
                                   0xA2F8D5618688CE24, 0x1EBD8BAC137DE736),
                  y: GFp256::w64be(0x2FAE5E4F2904A394, 0x66D0BB045226CE08,
                                   0x7F49366C44EA7657, 0xF4EF5C0844C42ECC) },
    // (2^130)*G * 5
    PointAffine { x: GFp256::w64be(0xA7F78E1EC93CBBD6, 0xF8FC6D4127ED7269,
                                   0xB9D7F4F0A675F38F, 0x0292ECB3AED00393),
                  y: GFp256::w64be(0x74385589986B6CFB, 0x55B693F3E38D980F,
                                   0xE99F2570EB82ABA9, 0x19D07787B7FEF35A) },
    // (2^130)*G * 6
    PointAffine { x: GFp256::w64be(0x0029C229E3736FCA, 0x260FA106D35E9AEA,
                                   0xE5CB8D2F032C89E8, 0xC9FBED5AC0E99BE1),
                  y: GFp256::w64be(0xDCE5DE5CBAE2D493, 0x986D607FEF4AACD3,
                                   0x5C1B783D383671FE, 0x5C3583389E77EDB0) },
    // (2^130)*G * 7
    PointAffine { x: GFp256::w64be(0x847021C276877E98, 0xE4B6CE7093E79C53,
                                   0x320440AE7941CFEA, 0x0FA90FA2B7F21984),
                  y: GFp256::w64be(0x72B8BC670713B0C8, 0x7902B8C7723B15EC,
                                   0x45A79737F0DD70AB, 0xC8C8A0CA549DEAE8) },
    // (2^130)*G * 8
    PointAffine { x: GFp256::w64be(0xA77663F5FCDF189C, 0xC2731F7801B1133E,
                                   0xE130A96964396297, 0x872E52CF757A603C),
                  y: GFp256::w64be(0x4E139CBDADDAFC58, 0xDAE982FA5476A35A,
                                   0x8AB86C568CBACE48, 0xD91745804CA8E468) },
    // (2^130)*G * 9
    PointAffine { x: GFp256::w64be(0xE5D817CA33C4E2FE, 0x60C06F3595DED4B8,
                                   0x00FE36FF7435D008, 0x603810E5CC6A453E),
                  y: GFp256::w64be(0xA4C9D402134C2096, 0x7B562F4E55850BA0,
                                   0xDF8B7D3BCFC9A287, 0xC62349BAEC9EE85D) },
    // (2^130)*G * 10
    PointAffine { x: GFp256::w64be(0x85D2322A5DAEBAF1, 0xCFA07F5F23768F84,
                                   0x0C2ACF8A3A05AF3D, 0x13EF0C6AB6531CEE),
                  y: GFp256::w64be(0xFBB6BF19376E06EF, 0x76A17B861DDA119F,
                                   0xA974DF60FBA1B683, 0x5448F9E51FFF61F6) },
    // (2^130)*G * 11
    PointAffine { x: GFp256::w64be(0x6D420C785F05B0D1, 0x792AAF68692B6A7E,
                                   0xD4180CE48549F248, 0xD2E327E28014ED0D),
                  y: GFp256::w64be(0x706934B5ECC0921B, 0x03C0F2593F66094C,
                                   0x64A84889C456A6F0, 0x27C246F735A6D65A) },
    // (2^130)*G * 12
    PointAffine { x: GFp256::w64be(0x6E5872E3076FE945, 0xFB0F2D3F1AF7857A,
                                   0xF73F35183273C051, 0x91706CB8D09B01F5),
                  y: GFp256::w64be(0x4213C02EC8D77171, 0x80CC1D4AF3BEF0EB,
                                   0xC13179D4E09A85A7, 0xC30758D9ABCD5FDE) },
    // (2^130)*G * 13
    PointAffine { x: GFp256::w64be(0x56F5C4B85C0574DC, 0x078E6CFA9B9AF7F9,
                                   0xB7A0E24BB0CB03AC, 0x2EC6F1FFBFED0AF3),
                  y: GFp256::w64be(0x54E716D27CA28344, 0x99C70ECBD1B77EFC,
                                   0x36CCC70A13343147, 0xB16512E5EB7720CB) },
    // (2^130)*G * 14
    PointAffine { x: GFp256::w64be(0x604AC042B04668BB, 0xE0D905ACA59AB261,
                                   0x2625F0B09E37C379, 0x6C0882668EBB9BB6),
                  y: GFp256::w64be(0x370FA451C3ECD3E4, 0xD603305526FFAEBF,
                                   0xFDBAAE00E30B932F, 0xDB5172B771028158) },
    // (2^130)*G * 15
    PointAffine { x: GFp256::w64be(0xB5DC5FC512D1C157, 0x66F5CBF729D95ECB,
                                   0x68F2A620A52E5FE3, 0x2CC1CF471362C215),
                  y: GFp256::w64be(0x6EFA9672CA18B8CE, 0xAC9ADA314F5285DA,
                                   0x3FC7CF1166E0E7DB, 0xE925490007E98290) },
    // (2^130)*G * 16
    PointAffine { x: GFp256::w64be(0x1A7098D2DB889A11, 0x12911D2965F2870E,
                                   0xA93D696BF79B5582, 0xE83E2130461DFFE4),
                  y: GFp256::w64be(0x39D474F5EE8B69FA, 0xE5A42C936F76F327,
                                   0x68682783DE26B553, 0xF3BA5EED161315DA) },
];

// Points i*(2^195)*G for i = 1 to 16, in affine coordinates.
static PRECOMP_G195: [PointAffine; 16] = [
    // (2^195)*G * 1
    PointAffine { x: GFp256::w64be(0x9A79BFBFE71E347F, 0x4D6C6698316797E2,
                                   0xF5AC2A3900F5ABF0, 0xC409332DE46E2050),
                  y: GFp256::w64be(0xE98B4DE6D316E200, 0xB6F671F3B224EFA9,
                                   0xCA94FACCB6DFDE31, 0x7A3F4781926250D2) },
    // (2^195)*G * 2
    PointAffine { x: GFp256::w64be(0xAC25DA80089CF4E0, 0x33D4DB5710FF5936,
                                   0xFD683B4D0DAB013E, 0x6EEF62FF4514C6FD),
                  y: GFp256::w64be(0xEBC69D985CB44C7B, 0x883DA9312A1B338C,
                                   0x810983E8243BF37A, 0x60B5397705830541) },
    // (2^195)*G * 3
    PointAffine { x: GFp256::w64be(0x3A52D92C9B4DF939, 0xD0B45C92FC82055A,
                                   0x6087250028AC0ECD, 0x3A611C1E24B91CD0),
                  y: GFp256::w64be(0x9EBBCBDB18D87820, 0xE0362AC11F589476,
                                   0xF9D2197601A1C427, 0xDA58C4ED72313BA9) },
    // (2^195)*G * 4
    PointAffine { x: GFp256::w64be(0xB467CD65660C13B8, 0x4AEC42B5296BC037,
                                   0xC1E6B5EA71A0D289, 0xB456511069962F3C),
                  y: GFp256::w64be(0xDC91B9BC4D36FFEA, 0xBB3F5D2A011664AC,
                                   0x3CB212DF6CBDA472, 0xF75584CB22877596) },
    // (2^195)*G * 5
    PointAffine { x: GFp256::w64be(0xCB32B06A74454973, 0x61DC3CD62B330851,
                                   0xB5A5817C274859C8, 0x4BD01EC7B5C8A53E),
                  y: GFp256::w64be(0xEECBCB591D8E55F7, 0x2DBEDABC226784F1,
                                   0xAED376EDC810D12F, 0xD7E9D40D0709AB85) },
    // (2^195)*G * 6
    PointAffine { x: GFp256::w64be(0xEEA7E66C49BA3C8B, 0xB4D748671E165547,
                                   0xE631E4DC5427AD60, 0x81941C7E32FD0E3D),
                  y: GFp256::w64be(0x83FED45B9F889261, 0x28843F27BE9B7D98,
                                   0x5ACBADA23019E3CC, 0x10411F68D91D5868) },
    // (2^195)*G * 7
    PointAffine { x: GFp256::w64be(0xF3736791C368A886, 0x44D8B2AB44362269,
                                   0x452772FC2EBBB531, 0xD961A14460FFAF46),
                  y: GFp256::w64be(0xFF18EA13C19FA5F8, 0x0FE7B2615346DF42,
                                   0xCED765F8C33E1679, 0xBEE3806EA38FE7AC) },
    // (2^195)*G * 8
    PointAffine { x: GFp256::w64be(0x92F9A6F93FA37761, 0x979280DE26DA6CDE,
                                   0x08533ABEEF2160F4, 0x919B1597B43C7E8D),
                  y: GFp256::w64be(0xBF618D715147AFE8, 0xED41D3A6F5E542DF,
                                   0x7309284ECEA31289, 0x7746055552BF1285) },
    // (2^195)*G * 9
    PointAffine { x: GFp256::w64be(0xA022A1FA97E7AAFE, 0x3D78A9AFCD130CD8,
                                   0x6D2CEA57032119E2, 0x25475B10E116E28A),
                  y: GFp256::w64be(0x39A9060D8AE899FF, 0x294E50155C1A2436,
                                   0x687F6DBBF07C9BF2, 0x83CCB7FD708A2008) },
    // (2^195)*G * 10
    PointAffine { x: GFp256::w64be(0x9A3DE2D3E93E21BD, 0xBAD194B35403C67B,
                                   0x7DDFFE6C44408E5D, 0x119F14F26CCB63B3),
                  y: GFp256::w64be(0x2F809B90611112C8, 0x809F2736EE813C4C,
                                   0x49C01D9FEF582260, 0x01C532977DF93FCF) },
    // (2^195)*G * 11
    PointAffine { x: GFp256::w64be(0x3AABC6CEC406EFDE, 0x93A0AAEC8366DE41,
                                   0x33D6D33959266E23, 0xA38069C0A5CEDE86),
                  y: GFp256::w64be(0xF9B11C9A08A65B42, 0xBE21897E20D27EB3,
                                   0xEE6C0F0C0FDD87AD, 0xF5E941F6788BEA75) },
    // (2^195)*G * 12
    PointAffine { x: GFp256::w64be(0x3CC4767A06772486, 0x65FBB469F6E48D44,
                                   0x7DB4BA1F065935FB, 0xB5CD34F827D3C9FF),
                  y: GFp256::w64be(0x557C538D8CD381AD, 0x94F22765E37B7EDF,
                                   0x772241168378473A, 0x3016C53CC5AA7383) },
    // (2^195)*G * 13
    PointAffine { x: GFp256::w64be(0x81E200EB9A14795D, 0xC21A735B26096917,
                                   0x814C37604036C2E1, 0x1847121C6DD6F7D4),
                  y: GFp256::w64be(0x4F3DBBD140393479, 0x0AF1ABB04967D8AB,
                                   0x2215774AA9EF4A5F, 0xC653CEAE0007051D) },
    // (2^195)*G * 14
    PointAffine { x: GFp256::w64be(0xA7C382F4B53B92DB, 0x05100D30123FADAF,
                                   0x34896F70BA7F6D8B, 0xE4ECA34DD63F1117),
                  y: GFp256::w64be(0x1B91E5116761BF88, 0x0459F777705DCFB5,
                                   0xAE4EB76ED3C4A68C, 0xC1220C5E1934CB9A) },
    // (2^195)*G * 15
    PointAffine { x: GFp256::w64be(0x46D7E1C662885744, 0x35E92201E54A33E8,
                                   0x7399020B7D586074, 0x6AC4CE68ECB57693),
                  y: GFp256::w64be(0x98E98E39CB7F8AD0, 0x412F1731CBBBAED9,
                                   0x2CA811498DC3BE4E, 0x14747481CA2A8770) },
    // (2^195)*G * 16
    PointAffine { x: GFp256::w64be(0x5CE4FD836044BAF3, 0xA1623065BCE83B11,
                                   0xFD7516A5B209D887, 0xD79A7081AFAEFC6A),
                  y: GFp256::w64be(0x61042108AA50F368, 0x1E31BC1EA832C714,
                                   0x4ABFBD4AB9E4ACCE, 0x38606DFBBF9DB9A6) },
];

// ========================================================================

#[cfg(test)]
mod tests {

    use super::{Point, Scalar, PrivateKey};
    use sha2::{Sha256, Digest};

    #[cfg(feature = "alloc")]
    use crate::Vec;

    #[cfg(feature = "alloc")]
    use crate::field::GFp256;

    /* unused
    fn print_gf(name: &str, x: GFp256) {
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
                0xAA, 0x0E, 0xB9, 0x89, 0xA0, 0x7C, 0x30, 0xF9,
                0xEC, 0x83, 0xC1, 0xF1, 0x02, 0x76, 0x2F, 0x75,
                0x2D, 0x77, 0xD8, 0xD7, 0x22, 0x71, 0xE5, 0x5B,
                0xDB, 0xA6, 0x21, 0x6A, 0x97, 0x6B, 0x1E, 0xAF
            ],
            [
                0x02,
                0xBB, 0x49, 0xE8, 0xA7, 0x67, 0x7E, 0x4C, 0xBA,
                0xB7, 0x58, 0x55, 0xB3, 0x09, 0xF3, 0x33, 0x6D,
                0xAD, 0xB8, 0xAA, 0xFF, 0xF9, 0x54, 0x7A, 0x39,
                0xC4, 0xB5, 0x86, 0x8D, 0x2F, 0xE9, 0xD4, 0xD6
            ],
            [
                0x02,
                0xC4, 0xC3, 0x08, 0x93, 0x37, 0x35, 0x33, 0x1D,
                0xBD, 0x22, 0xD8, 0x4A, 0x02, 0x6F, 0xEA, 0x53,
                0xA1, 0x86, 0x42, 0xF6, 0x27, 0xEF, 0x9E, 0xB0,
                0xD6, 0xE2, 0xA6, 0x8A, 0x2E, 0xB8, 0xB4, 0x7C
            ],
            [
                0x02,
                0x7F, 0xAC, 0x28, 0xE6, 0xB5, 0x2B, 0xA8, 0x2E,
                0x83, 0x1E, 0xDC, 0x29, 0x3D, 0x59, 0x73, 0xB9,
                0xC6, 0x5F, 0x43, 0xF6, 0x4A, 0xB4, 0xF3, 0x7C,
                0x38, 0x58, 0x80, 0x2A, 0x99, 0x4F, 0x34, 0xE8
            ],
            [
                0x03,
                0xAA, 0x1A, 0x33, 0x26, 0xBF, 0xBB, 0x57, 0x8D,
                0x4B, 0x16, 0xBD, 0x94, 0xA1, 0x8E, 0x88, 0x5C,
                0x6F, 0x53, 0x6E, 0xE1, 0xF4, 0x6A, 0x99, 0xAF,
                0x43, 0xF0, 0x91, 0x2E, 0xFD, 0x44, 0x6B, 0x85
            ],
            [
                0x02,
                0x14, 0x58, 0xDE, 0x7A, 0x34, 0x09, 0x4E, 0x68,
                0x31, 0x59, 0x2D, 0x48, 0x13, 0x5F, 0xDC, 0xC5,
                0x8A, 0xA5, 0x25, 0xBF, 0x1B, 0xF7, 0x65, 0xCE,
                0x40, 0x5B, 0x53, 0x36, 0x2F, 0x36, 0xDE, 0xA4
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
                0xAA, 0x0E, 0xB9, 0x89, 0xA0, 0x7C, 0x30, 0xF9,
                0xEC, 0x83, 0xC1, 0xF1, 0x02, 0x76, 0x2F, 0x75,
                0x2D, 0x77, 0xD8, 0xD7, 0x22, 0x71, 0xE5, 0x5B,
                0xDB, 0xA6, 0x21, 0x6A, 0x97, 0x6B, 0x1E, 0xAF,
                0x7D, 0x04, 0xEB, 0xEF, 0x40, 0xBF, 0x57, 0xF4,
                0xAF, 0x34, 0xD2, 0xEB, 0x59, 0x14, 0x84, 0xFA,
                0xD2, 0x67, 0xBB, 0x92, 0x28, 0x8A, 0x6C, 0x8C,
                0x88, 0x3D, 0xD1, 0x24, 0xA7, 0xF9, 0xB8, 0xD6
            ],
            [
                0x04,
                0xBB, 0x49, 0xE8, 0xA7, 0x67, 0x7E, 0x4C, 0xBA,
                0xB7, 0x58, 0x55, 0xB3, 0x09, 0xF3, 0x33, 0x6D,
                0xAD, 0xB8, 0xAA, 0xFF, 0xF9, 0x54, 0x7A, 0x39,
                0xC4, 0xB5, 0x86, 0x8D, 0x2F, 0xE9, 0xD4, 0xD6,
                0x53, 0x7B, 0xB0, 0x46, 0x10, 0xF8, 0x0E, 0x00,
                0x43, 0xA7, 0x9F, 0x52, 0xE4, 0xF8, 0xB8, 0x5C,
                0x88, 0x74, 0x5E, 0x72, 0xE0, 0xCD, 0xE9, 0x70,
                0x4B, 0x19, 0x82, 0xFA, 0x92, 0x97, 0x6B, 0xF6
            ],
            [
                0x04,
                0xC4, 0xC3, 0x08, 0x93, 0x37, 0x35, 0x33, 0x1D,
                0xBD, 0x22, 0xD8, 0x4A, 0x02, 0x6F, 0xEA, 0x53,
                0xA1, 0x86, 0x42, 0xF6, 0x27, 0xEF, 0x9E, 0xB0,
                0xD6, 0xE2, 0xA6, 0x8A, 0x2E, 0xB8, 0xB4, 0x7C,
                0x86, 0xB7, 0x70, 0xA3, 0xDE, 0x94, 0x0A, 0x78,
                0x6F, 0xC9, 0x97, 0x0E, 0x9B, 0x41, 0x8A, 0x7E,
                0x26, 0xEA, 0xCD, 0x70, 0x52, 0x3F, 0x17, 0xA1,
                0x2C, 0x6A, 0xF4, 0xFD, 0x00, 0x47, 0xB5, 0x2C
            ],
            [
                0x04,
                0x7F, 0xAC, 0x28, 0xE6, 0xB5, 0x2B, 0xA8, 0x2E,
                0x83, 0x1E, 0xDC, 0x29, 0x3D, 0x59, 0x73, 0xB9,
                0xC6, 0x5F, 0x43, 0xF6, 0x4A, 0xB4, 0xF3, 0x7C,
                0x38, 0x58, 0x80, 0x2A, 0x99, 0x4F, 0x34, 0xE8,
                0x80, 0xE9, 0x49, 0x0B, 0xFB, 0x97, 0x75, 0x84,
                0x37, 0xC6, 0xE2, 0x82, 0x68, 0x6C, 0x08, 0x7D,
                0xDB, 0x21, 0x23, 0xDC, 0x44, 0x56, 0x15, 0xB0,
                0x01, 0x71, 0x61, 0x42, 0x79, 0xC3, 0x64, 0x0C
            ],
            [
                0x04,
                0xAA, 0x1A, 0x33, 0x26, 0xBF, 0xBB, 0x57, 0x8D,
                0x4B, 0x16, 0xBD, 0x94, 0xA1, 0x8E, 0x88, 0x5C,
                0x6F, 0x53, 0x6E, 0xE1, 0xF4, 0x6A, 0x99, 0xAF,
                0x43, 0xF0, 0x91, 0x2E, 0xFD, 0x44, 0x6B, 0x85,
                0x78, 0x46, 0x19, 0xA3, 0xEF, 0xE1, 0xD0, 0xCC,
                0xD8, 0x61, 0x6A, 0xF1, 0x14, 0x47, 0xBF, 0xD7,
                0x7E, 0x36, 0xB5, 0xF7, 0x8D, 0x53, 0x1C, 0xC8,
                0x6B, 0x8D, 0x7B, 0x2B, 0x58, 0xE6, 0x26, 0x8F
            ],
            [
                0x04,
                0x14, 0x58, 0xDE, 0x7A, 0x34, 0x09, 0x4E, 0x68,
                0x31, 0x59, 0x2D, 0x48, 0x13, 0x5F, 0xDC, 0xC5,
                0x8A, 0xA5, 0x25, 0xBF, 0x1B, 0xF7, 0x65, 0xCE,
                0x40, 0x5B, 0x53, 0x36, 0x2F, 0x36, 0xDE, 0xA4,
                0x20, 0x27, 0xDF, 0xC5, 0x9C, 0x29, 0xD1, 0xDB,
                0x2D, 0x5B, 0x67, 0x6F, 0x36, 0xC8, 0xC7, 0xDA,
                0xC1, 0x63, 0x76, 0x69, 0xD1, 0xAA, 0xD8, 0x46,
                0x63, 0x26, 0xFE, 0xD2, 0x0F, 0x62, 0x6B, 0x9C
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
    fn mulgen() {
        // Test vector generated randomly with Sage.
        let s = Scalar::w64be(0x7DC39B763DF3A5EA, 0x46AC87887B246E48,
                              0xD9DC3839C0D466E4, 0x6DFE006C126C829B);
        let enc: [u8; 33] = [
            0x02,
            0x53, 0x13, 0x52, 0x93, 0xE1, 0xF3, 0xD3, 0xBE,
            0x74, 0xBF, 0x7D, 0x50, 0xD9, 0x9C, 0xA0, 0x85,
            0x41, 0xB0, 0x36, 0xE0, 0x9D, 0xB7, 0x83, 0xFC,
            0x79, 0x08, 0xA0, 0xDA, 0xF3, 0x94, 0xDA, 0x6F
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

    #[cfg(feature = "alloc")]
    #[test]
    fn x_sequence() {
        fn tt(P0: Point, P1: Point, n: usize) {
            let mut xx = Vec::with_capacity(n);
            xx.resize(n, GFp256::ZERO);
            let Q = P1 - P0;
            let (x0, _, _) = P0.to_affine();
            let (x1, _, _) = P1.to_affine();
            let (xq, _, _) = Q.to_affine();
            let (xf0, xf1) = Point::x_sequence_vartime(x0, x1, xq, &mut xx[..]);
            let mut T = P0;
            for i in 0..n {
                if T.isneutral() != 0 {
                    assert!(xx[i].equals(GFp256::ONE) != 0);
                } else {
                    assert!(T.X.equals(xx[i] * T.Z) != 0);
                }
                T += Q;
            }
            let T0 = T;
            let T1 = T + Q;
            if T0.isneutral() != 0 {
                assert!(xf0.equals(GFp256::ONE) != 0);
            } else {
                assert!(T0.X.equals(xf0 * T0.Z) != 0);
            }
            if T1.isneutral() != 0 {
                assert!(xf1.equals(GFp256::ONE) != 0);
            } else {
                assert!(T1.X.equals(xf1 * T1.Z) != 0);
            }
        }

        let mut sh = Sha256::new();
        sh.update(&[0u8]);
        let U = Point::mulgen(&Scalar::decode_reduce(&sh.finalize_reset()[..]));
        sh.update(&[1u8]);
        let V = Point::mulgen(&Scalar::decode_reduce(&sh.finalize_reset()[..]));

        // Normal tests with pseudorandom points.
        tt(U, V, 0);
        tt(U, V, 1);
        tt(U, V, 2);
        tt(U, V, 3);
        tt(U, V, 100);
        tt(U, V, 101);
        tt(U, V, 102);

        // Tests with the neutral.
        tt(5 * U, 4 * U, 3);
        tt(5 * U, 4 * U, 4);
        tt(5 * U, 4 * U, 5);
        tt(5 * U, 4 * U, 6);
        tt(5 * U, 4 * U, 7);
        tt(U, U, 10);

        // Tests with Q = (0, +/-sqrt(b)).
        let mut Q_enc = [0u8; 33];
        Q_enc[0] = 0x02;
        let Q = Point::decode(&Q_enc).unwrap();
        tt(U, U + Q, 10);
        tt(Q - 5 * U, Q - 4 * U, 10);
    }

    #[test]
    fn signatures() {
        // Test vector from RFC 6979, section A.2.5
        let priv_enc: [u8; 32] = [
            0xC9, 0xAF, 0xA9, 0xD8, 0x45, 0xBA, 0x75, 0x16,
            0x6B, 0x5C, 0x21, 0x57, 0x67, 0xB1, 0xD6, 0x93,
            0x4E, 0x50, 0xC3, 0xDB, 0x36, 0xE8, 0x9B, 0x12,
            0x7B, 0x8A, 0x62, 0x2B, 0x12, 0x0F, 0x67, 0x21,
        ];
        let pub_enc: [u8; 65] = [
            0x04,
            0x60, 0xFE, 0xD4, 0xBA, 0x25, 0x5A, 0x9D, 0x31,
            0xC9, 0x61, 0xEB, 0x74, 0xC6, 0x35, 0x6D, 0x68,
            0xC0, 0x49, 0xB8, 0x92, 0x3B, 0x61, 0xFA, 0x6C,
            0xE6, 0x69, 0x62, 0x2E, 0x60, 0xF2, 0x9F, 0xB6,
            0x79, 0x03, 0xFE, 0x10, 0x08, 0xB8, 0xBC, 0x99,
            0xA4, 0x1A, 0xE9, 0xE9, 0x56, 0x28, 0xBC, 0x64,
            0xF2, 0xF1, 0xB2, 0x0C, 0x2D, 0x7E, 0x9F, 0x51,
            0x77, 0xA3, 0xC2, 0x94, 0xD4, 0x46, 0x22, 0x99,
        ];
        let msg1: &[u8] = b"sample";
        let expected_sig1: [u8; 64] = [
            0xEF, 0xD4, 0x8B, 0x2A, 0xAC, 0xB6, 0xA8, 0xFD,
            0x11, 0x40, 0xDD, 0x9C, 0xD4, 0x5E, 0x81, 0xD6,
            0x9D, 0x2C, 0x87, 0x7B, 0x56, 0xAA, 0xF9, 0x91,
            0xC3, 0x4D, 0x0E, 0xA8, 0x4E, 0xAF, 0x37, 0x16,
            0xF7, 0xCB, 0x1C, 0x94, 0x2D, 0x65, 0x7C, 0x41,
            0xD4, 0x36, 0xC7, 0xA1, 0xB6, 0xE2, 0x9F, 0x65,
            0xF3, 0xE9, 0x00, 0xDB, 0xB9, 0xAF, 0xF4, 0x06,
            0x4D, 0xC4, 0xAB, 0x2F, 0x84, 0x3A, 0xCD, 0xA8,
        ];
        let msg2: &[u8] = b"test";
        let expected_sig2: [u8; 64] = [
            0xF1, 0xAB, 0xB0, 0x23, 0x51, 0x83, 0x51, 0xCD,
            0x71, 0xD8, 0x81, 0x56, 0x7B, 0x1E, 0xA6, 0x63,
            0xED, 0x3E, 0xFC, 0xF6, 0xC5, 0x13, 0x2B, 0x35,
            0x4F, 0x28, 0xD3, 0xB0, 0xB7, 0xD3, 0x83, 0x67,
            0x01, 0x9F, 0x41, 0x13, 0x74, 0x2A, 0x2B, 0x14,
            0xBD, 0x25, 0x92, 0x6B, 0x49, 0xC6, 0x49, 0x15,
            0x5F, 0x26, 0x7E, 0x60, 0xD3, 0x81, 0x4B, 0x4C,
            0x0C, 0xC8, 0x42, 0x50, 0xE4, 0x6F, 0x00, 0x83,
        ];

        let skey = PrivateKey::decode(&priv_enc).unwrap();
        let pkey = skey.to_public_key();
        assert!(pkey.encode_uncompressed() == pub_enc);
        let mut sh = Sha256::new();
        sh.update(&msg1);
        let hv1: [u8; 32] = sh.finalize_reset().into();
        let sig1 = skey.sign_hash(&hv1, &[]);
        assert!(sig1 == expected_sig1);
        sh.update(&msg2);
        let hv2: [u8; 32] = sh.finalize_reset().into();
        let sig2 = skey.sign_hash(&hv2, &[]);
        assert!(sig2 == expected_sig2);

        assert!(pkey.verify_hash(&sig1, &hv1));
        assert!(pkey.verify_hash(&sig2, &hv2));
        assert!(!pkey.verify_hash(&sig1, &hv2));
        assert!(!pkey.verify_hash(&sig2, &hv1));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn signatures_trunc() {
        let mut seed = [0u8; 48];
        let mut sh = Sha256::new();
        sh.update(&[0u8]);
        seed[0..32].copy_from_slice(&sh.finalize_reset()[..]);
        sh.update(&[1u8]);
        seed[32..48].copy_from_slice(&sh.finalize_reset()[0..16]);
        let skey = PrivateKey::from_seed(&seed);
        let pkey = skey.to_public_key();
        for i in 0..2 {
            let mut msg = [0u8; 8];
            msg[..].copy_from_slice(&(i as u64).to_le_bytes());
            sh.update(&msg);
            let hv = sh.finalize_reset();
            let sig1 = skey.sign_hash(&hv[..], &[]);
            let mut sig2 = PrivateKey::prepare_truncate(&sig1).unwrap();
            sig2[63] = 0;
            for rm in 8..(if i == 0 { 33 } else { 25 }) {
                let n = 512 - rm;
                sig2[n >> 3] &= !(0x01u8 << (n & 7));
                let vv = pkey.verify_trunc_hash(&sig2, rm, &hv[..]);
                assert!(vv.is_some());
                let sig3 = vv.unwrap();
                assert!(pkey.verify_hash(&sig3, &hv[..]));
                msg[0] ^= 1;
                assert!(pkey.verify_trunc_hash(&sig2, rm, &msg).is_none());
                msg[0] ^= 1;
            }
        }
    }
}
