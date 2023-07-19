//! X448 key-exchange algorithm.
//!
//! This module implements the X448 primitive, as defined by [RFC 7748].
//! The primitive takes as input two 56-byte values, the first
//! being the representation of a point on Curve448 (a Montgomery
//! curve) or on the quadratic twist of Curve448, and the second being
//! a scalar (a big integer). The scalar is internally "clamped" (some
//! bits are set to specific values), then the point is multiplied by the
//! scalar, and the output point is reencoded into 56 bytes.
//!
//! The `x448()` function implements exactly the process described in
//! RFC 7748 (section 5). The `x448_base()` function is an optimization
//! of the specific case of the input point being the conventional
//! generator point on Curve448; `x448_base()` is fully compatible
//! with `x448()`, but also substantially faster.
//!
//! The `x448()` function does NOT filter out any value from its input;
//! any input sequence of 56 bytes is accepted, even if it encodes a
//! low-order curve point. As per RFC 7748 requirements, the top point
//! bit (most significant bit of the last byte) is ignored. As for
//! scalars, the clamping process ensures that the integer used for the
//! multiplication is a multiple of 4, at least 2^447, and lower than
//! 2^448; the two least significant bits of the first byte, and the
//! most significant bit of the last byte, are ignored.
//!
//! [RFC 7748]: https://datatracker.ietf.org/doc/html/rfc7748

// Projective/fractional coordinates traditionally use uppercase letters,
// using lowercase only for affine coordinates.
#![allow(non_snake_case)]

use super::field::GF448;
use super::ed448::{Point, Scalar};

/// X448 function (from RFC 7748), general case.
///
/// The source point is provided as an array of 56 bytes (`point`), as
/// well as the scalar (`scalar`). In RFC 7748 terminology, the `point`
/// parameter is the little-endian encoding of the u coordinate of a
/// point on the Montgomery curve or on its quadratic twist, and the
/// `scalar` parameter is the little-endian encoding of the scalar. The
/// function "clamps" the scalar (bits 0 and 1 are cleared, bit 447 is
/// set) then interprets the clamped scalar as an integer (little-endian
/// convention), with which the provided curve point is multiplied; the u
/// coordinate of the resulting point is then encoded and returned.
pub fn x448(point: &[u8; 56], scalar: &[u8; 56]) -> [u8; 56] {
    // Make clamped scalar.
    let mut s = *scalar;
    s[0] &= 252;
    s[55] |= 128;

    // Decode the source point. As per RFC 7748 rules, non-canonical
    // values are acceptable.
    let x1 = GF448::decode_reduce(point);

    // Apply the RFC 7748 section 5 algorithm.
    let mut x2 = GF448::ONE;
    let mut z2 = GF448::ZERO;
    let mut x3 = x1;
    let mut z3 = GF448::ONE;
    let mut swap = 0u32;

    for t in (0..448).rev() {
        let kt = (((s[t >> 3] >> (t & 7)) & 1) as u32).wrapping_neg();
        swap ^= kt;
        GF448::cswap(&mut x2, &mut x3, swap);
        GF448::cswap(&mut z2, &mut z3, swap);
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
        z2 = E * (AA + E.mul_small(39081));
    }
    GF448::cswap(&mut x2, &mut x3, swap);
    GF448::cswap(&mut z2, &mut z3, swap);

    (x2 / z2).encode()
}

/// Specialized version of X448, when applied to the conventional
/// generator point (u = 9).
///
/// See `x448()` for details. This function is significantly faster than
/// the general `x448()` function.
pub fn x448_base(scalar: &[u8; 56]) -> [u8; 56] {
    // Make clamped scalar, and decode it as an integer modulo L.
    let mut sb = *scalar;
    sb[0] &= 252;
    sb[55] |= 128;
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

    use super::{x448, x448_base};
    use sha2::{Sha512, Digest};

    #[test]
    fn x448_mc() {
        let mut k = [0u8; 56];
        k[0] = 5;
        let mut u = k;
        let mut ref1 = [0u8; 56];
        hex::decode_to_slice("3f482c8a9f19b01e6c46ee9711d9dc14fd4bf67af30765c2ae2b846a4d23a8cd0db897086239492caf350b51f833868b9bc2b3bca9cf4113", &mut ref1[..]).unwrap();
        let mut ref1000 = [0u8; 56];
        hex::decode_to_slice("aa3b4749d55b9daf1e5b00288826c467274ce3ebbdd5c17b975e09d4af6c67cf10d087202db88286e2b79fceea3ec353ef54faa26e219f38", &mut ref1000[..]).unwrap();
        for i in 0..1000 {
            let old_k = k;
            k = x448(&u, &k);
            u = old_k;
            if i == 0 {
                assert!(k == ref1);
            }
        }
        assert!(k == ref1000);
    }

    #[test]
    fn x448_basepoint() {
        let mut sh = Sha512::new();
        let mut b = [0u8; 56];
        b[0] = 5;
        for i in 0..20 {
            sh.update(&(i as u64).to_le_bytes());
            let v = sh.finalize_reset();
            let mut k = [0u8; 56];
            k[..].copy_from_slice(&v[..56]);
            assert!(x448(&b, &k) == x448_base(&k));
        }
    }
}
