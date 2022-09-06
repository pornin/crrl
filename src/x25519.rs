//! X25519 key-exchange algorithm.
//!
//! This module implements the X25519 primitive, as defined by [RFC
//! 7748]. The primitive takes as input two 32-byte values, the first
//! being the representation of a point on Curve25519 (a Montgomery
//! curve) or on the quadratic twist of Curve25519, and the second being
//! a scalar (a big integer). The scalar is internally "clamped" (some
//! bits are set to specific values), then the point is multiplied by the
//! scalar, and the output point is reencoded into 32 bytes.
//!
//! The `x25519()` function implements exactly the process described in
//! RFC 7748 (section 5). The `x25519_base()` function is an optimization
//! of the specific case of the input point being the conventional
//! generator point on Curve25519; `x25519_base()` is fully compatible
//! with `x25519()`, but also substantially faster.
//!
//! The `x25519()` function does NOT filter out any value from its input;
//! any input sequence of 32 bytes is accepted, even if it encodes a
//! low-order curve point. As per RFC 7748 requirements, the top point
//! bit (most significant bit of the last byte) is ignored. As for
//! scalars, the clamping process ensures that the integer used for the
//! multiplication is a multiple of 8, at least 2^254, and lower than
//! 2^255; the three least significant bits of the first byte, and two
//! most significant bits of the last byte, are ignored.
//!
//! [RFC 7748]: https://datatracker.ietf.org/doc/html/rfc7748

// Projective/fractional coordinates traditionally use uppercase letters,
// using lowercase only for affine coordinates.
#![allow(non_snake_case)]

use super::field::GF25519;
use super::ed25519::{Point, Scalar};

/// X25519 function (from RFC 7748), general case.
///
/// The source point is provided as an array of 32 bytes (`point`), as
/// well as the scalar (`scalar`). In RFC 7748 terminology, the `point`
/// parameter is the little-endian encoding of the u coordinate of a
/// point on the Montgomery curve or on its quadratic twist, and the
/// `scalar` parameter is the little-endian encoding of the scalar. The
/// function "clamps" the scalar (bits 0, 1, 2 and 255 are cleared, bit
/// 254 is set) then interprets the clamped scalar as an integer
/// (little-endian convention), with which the provided curve point is
/// multiplied; the u coordinate of the resulting point is then encoded
/// and returned.
pub fn x25519(point: &[u8; 32], scalar: &[u8; 32]) -> [u8; 32] {
    // Make clamped scalar.
    let mut s = *scalar;
    s[0] &= 248;
    s[31] &= 127;
    s[31] |= 64;

    // Decode the source point. As per RFC 7748 rules, the top bit is
    // ignored, and non-canonical values are acceptable.
    let mut u = *point;
    u[31] &= 127;
    let x1 = GF25519::decode_reduce(&u[..]);

    // Apply the RFC 7748 section 5 algorithm.
    let mut x2 = GF25519::ONE;
    let mut z2 = GF25519::ZERO;
    let mut x3 = x1;
    let mut z3 = GF25519::ONE;
    let mut swap = 0u32;

    for t in (0..255).rev() {
        let kt = (((s[t >> 3] >> (t & 7)) & 1) as u32).wrapping_neg();
        swap ^= kt;
        GF25519::cswap(&mut x2, &mut x3, swap);
        GF25519::cswap(&mut z2, &mut z3, swap);
        swap = kt;

        let A = x2 + z2;
        let B = x2 - z2;
        let AA = A.square();
        let BB = B.square();
        let C = x3 + z3;
        let D = x3 - z3;
        let E = AA - BB;
        let DA = D * A;
        let CB = C * B;
        x3 = (DA + CB).square();
        z3 = x1 * (DA - CB).square();
        x2 = AA * BB;
        z2 = E * (AA + E.mul_small(121665));
    }
    GF25519::cswap(&mut x2, &mut x3, swap);
    GF25519::cswap(&mut z2, &mut z3, swap);

    (x2 / z2).encode()
}

/// Specialized version of X25519, when applied to the conventional
/// generator point (u = 9).
///
/// See `x25519()` for details. This function is significantly faster than
/// the general `x25519()` function.
pub fn x25519_base(scalar: &[u8; 32]) -> [u8; 32] {
    // Make clamped scalar, and decode it as an integer modulo L.
    let mut sb = *scalar;
    sb[0] &= 248;
    sb[31] &= 127;
    sb[31] |= 64;
    let s = Scalar::decode_reduce(&sb[..]);

    // Perform the multiplication on the Edwards curve.
    let P = Point::mulgen(&s);

    // Apply the birational map to get the Montgomery point (u coordinate
    // only). When the point is the neutral, we want to return 0.
    let u = P.to_montgomery_u();
    u.encode()
}

// ========================================================================

#[cfg(test)]
mod tests {

    use super::{x25519, x25519_base};
    use sha2::{Sha256, Digest};

    #[test]
    fn x25519_mc() {
        let mut k = [0u8; 32];
        k[0] = 9;
        let mut u = k;
        let mut ref1 = [0u8; 32];
        hex::decode_to_slice("422c8e7a6227d7bca1350b3e2bb7279f7897b87bb6854b783c60e80311ae3079", &mut ref1[..]).unwrap();
        let mut ref1000 = [0u8; 32];
        hex::decode_to_slice("684cf59ba83309552800ef566f2f4d3c1c3887c49360e3875f2eb94d99532c51", &mut ref1000[..]).unwrap();
        for i in 0..1000 {
            let old_k = k;
            k = x25519(&u, &k);
            u = old_k;
            if i == 0 {
                assert!(k == ref1);
            }
        }
        assert!(k == ref1000);
    }

    #[test]
    fn x25519_basepoint() {
        let mut sh = Sha256::new();
        let mut b = [0u8; 32];
        b[0] = 9;
        for i in 0..20 {
            sh.update(&(i as u64).to_le_bytes());
            let v = sh.finalize_reset();
            let mut k = [0u8; 32];
            k[..].copy_from_slice(&v);
            assert!(x25519(&b, &k) == x25519_base(&k));
        }
    }
}
