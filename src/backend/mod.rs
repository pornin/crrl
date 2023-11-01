//! Architecture-specific implementations of finite fields.
//!
//! This module provides type aliases for the structures that implement
//! some finite fields. Each structure is specialized for a single field
//! (possibly through some type parameters, known at compile-time). There
//! can be several actual implementations for each type; a relevant
//! implementation is selected based on configured compilation features,
//! or through auto-detection of the current target.
//!
//! In general, the following properties apply to finite field implementations:
//!
//!  - An instance encapsulates a field element.
//!
//!  - The constant values `Self::ZERO` and `Self::ONE` contain the
//!    elements of value 0 and 1, respectively.
//!
//!  - Usual arithmetic operators can be used on field elements (`+`, `-`,
//!    `*`, `/`, and the compound assignments `+=`, `-=`, `*=` and `/=`).
//!    Division by zero is tolerated, and yields zero (regardless of the
//!    dividend). Operators can use both the raw types, and references
//!    thereof.
//!
//!  - Function `set_square(&mut self)` squares a field element (in place).
//!    Corresponding function `square(self) -> Self` returns the result
//!    as a new instance. These functions are somewhat faster than general
//!    multiplications. Sequences of multiple squarings can be performed
//!    with `set_xsquare(&mut self, n: u32)` (and a corresponding `xsquare()`
//!    to get the result as a new instance).
//!
//!  - Function `set_neg(&mut self)` negates the instance on which it is
//!    applied.
//!
//!  - Function `set_cond(&mut self, a: &Self, ctl: u32)` sets
//!    the instance to the value of the other instance `a` if `ctl` is
//!    equal to 0xFFFFFFFF, or leaves the instance value unmodified if
//!    `ctl` is equal to 0x00000000.
//!
//!  - Function `select(a0: &Self, a1: &Self, ctl: u32) -> Self` returns
//!    a copy of `a0` if `ctl` is 0x00000000, or a copy of `a1` if
//!    `ctl` is 0xFFFFFFFF.
//!
//!  - Function `cswap(a: &mut Self, b: &mut Self, ctl: u32)`
//!    exchanges the contents of `a` and `b` if `ctl` is 0xFFFFFFFF,
//!    or leaves them unmodified if `ctl` is 0x00000000.
//!
//!  - Functions `set_half()`, `set_mul2()`, `set_mul4()`, `set_mul8()`,
//!    `set_mul16()` and `set_mul32()`, all applied on `&mut self`,
//!    multiply their operand (in place) by 1/2, 2, 4, 8, 16 or 32,
//!    respectively. Corresponding functions `half()`, `mul2()`, `mul4()`,
//!    `mul8()`, `mul16()` and `mul32()` operate on `self` and return
//!    a new instance. These functions are normally faster than a
//!    generic multiplication in the field.
//!
//!  - Some fields also provide `set_mul3()` (and `mul3()`) for
//!    multiplication by 3.
//!
//!  - Some fields also provide `set_mul_small(&mut self, x: u32)`
//!    (and `mul_small(self, x: 32) -> Self`) for multiplication by a
//!    small 32-bit integer provided at runtime.
//!
//!  - Constant values can be defined with the const-qualified `w64le()`
//!    and `w64be()` functions, which take the value as four 64-bit limbs
//!    in little-endian and big-endian order, respectively. The 256-bit
//!    value is implicitly reduced modulo the field order.
//!
//!  - Non-const functions `from_w64le()` and `from_w64be()` are also
//!    provided; they yield the same output as, but are potentially faster
//!    than, the const functions `w64le()` and `w64be()`. Note that
//!    both the const and the non-const functions are safe (they should
//!    have no side-channels); the non-const functions are nonetheless
//!    preferred at runtime.
//!
//!  - Conversions from `i32`, `u32`, `i64`, `u64`, `i128` ans `u128`
//!    can use the functions `from_i32()`, `from_u32()`, and so on.
//!
//!  - Function `equals(self, rhs: Self) -> u32` returns 0xFFFFFFFF
//!    if `self` and `rhs` represent the same value, or 0x00000000
//!    otherwise. Function `iszero(self) -> u32` is a specialized
//!    subcase that compares `self` with zero.
//!
//!  - The `legendre(self) -> i32` function returns the Legendre symbol
//!    for an element (0 for zero, +1 for non-zero squares, -1 for
//!    non-squares).
//!
//!  - The `batch_invert(xx: &mut[Self])` function performs inversion
//!    of all field elements in the provided slice. It works by
//!    combining internal elements in batches (normally of 200 elements)
//!    and mutualizing the internal inversion; this is vastly faster
//!    than inverting each element independently. Elements of value zero
//!    are tolerated (the "inverse" of zero is zero).
//!
//!  - The `set_sqrt(&mut self) -> u32` function computes the square root
//!    of an element. On success, 0xFFFFFFFF is returned. On failure (input
//!    is not a square), the element is set to zero, and 0x00000000 is
//!    returned. The chosen square root is the one whose least significant
//!    bit (when represented as an integer lower than the field order) is
//!    a zero. The non-in-place variant of this function is
//!    `sqrt(self) -> (Self, u32)`. Note that field implementations may
//!    not provide square root computations for all supported moduli.
//!
//!  - The `set_sqrt_ext(&mut self) -> u32` is similar to `set_sqrt()`,
//!    except that it set the element to a predictable value on failure;
//!    that value depends on the field and on the input value. For integers
//!    modulo a prime q = 3 mod 4, if the input value is x, and x is not
//!    a square, then the element is set to a square root of -x. For integers
//!    modulo a prime q = 5 mod 8, if the input value is x, and x is not
//!    a square, then the element is set to a square root of either 2*x or
//!    -2*x. A non-in-place variant is provided as
//!    `sqrt_ext(self) -> (Self, u32)`. Note that field implementations may
//!    not provide square root computations for all supported moduli.
//!
//!  - Function `split_vartime(self) -> (i128, i128)` returns two signed
//!    integers c0 and c1 such that `self` is equal to c0/c1. Note that
//!    if the field modulus is greater than about 1.73\*2^253, then
//!    the two values c0 and c1 may be truncated. In all generality,
//!    `self` is equal to `(c0 + a*2^128) / (c1 + b*2^128)` for two
//!    integers `a` and `b` such that:
//!
//!      - If the field modulus is at most 1.73\*2^253, then `a` and `b`
//!        are both 0 (the returned values are exact).
//!
//!      - If the field modulus is between 1.73\*2^253 and 1.73\*2^255,
//!        than `a` and `b` may range between -1 and +1 (inclusive).
//!
//!      - For larger moduli (up to 2^256), `a` and `b` may range between
//!        -2 and +2 (inclusive).
//!
//!    This function uses Lagrange's algorithm for lattice basis reduction
//!    in dimension 2. It is sufficiently fast to be considered for
//!    optimizing verification of Schnorr signatures. WARNING: this
//!    function is not constant-time; it MUST NOT be applied on secret
//!    data.
//!
//!    This function is defined only for fields wioth a modulus less than
//!    2^256. It is not defined for larger fields.
//!
//!  - Function `encode(self) -> [u8; Self::ENC_LEN]` encodes an element as
//!    bytes. Unsigned little-endian convention is used. The encoding length
//!    (`ENC_LEN`) is the length of the modulus, in bytes. Encoding is
//!    always canonical (i.e. the encoding always uses the integer which is
//!    lower than the field modulus).
//!
//!  - Function `set_decode_ct(&mut self, buf: &[u8]) -> u32` decodes some
//!    bytes with the unsigned little-endian convention. If the source slice
//!    does not have the length of the modulus, then the decoding fails. If
//!    the source slice has the proper length, but the byte contents yield a
//!    non-canonical value, then decoding fails. On success, the element
//!    is set to the decoded value, and 0xFFFFFFFF is returned; on failure,
//!    the element is set to zero, and 0x00000000 is returned. If the source
//!    has the proper length, then not only the decoded value, but also the
//!    operation outcome (success or failure), are shielded from side-channel
//!    attacks. A non-in-place variant of this function is provided as
//!    `decode_ct(buf: &[u8]) -> (Self, u32)`.
//!
//!  - Function `set_decode_reduce(buf: &[u8])` decodes some bytes
//!    with the unsigned little-endian convention. The obtained integer is
//!    reduced modulo the field order, so the process never fails.
//!    It is fully constant-time (only the length of the source slice
//!    may leak through timing-based side channels). A non-in-place
//!    variant is provided as `decode_reduce(buf: &[u8]) -> Self`.
//!
//!  - Function `decode(buf: &[u8]) -> Option<Self>` decodes some bytes
//!    with the unsigned little-endian convention. This is a wrapper
//!    around `decode_ct()`, except that it returns `None` on decoding
//!    failure. Due to the use of the option type, side-channel analysis
//!    may reveal to outsiders whether the decoding succeeded or not.

#[cfg(not(any(
    feature = "w32_backend",
    feature = "w64_backend",
    target_pointer_width = "32",
    target_pointer_width = "64",
)))]
compile_error!("no backend specified; cannot infer from pointer size");

#[cfg(all(
    feature = "w32_backend",
    feature = "w64_backend",
))]
compile_error!("cannot use w32 and w64 backends simultaneously");

#[cfg(any(
    feature = "w32_backend",
    all(not(feature = "w64_backend"), target_pointer_width = "32"),
))]
pub mod w32;

/// Fixed-size integers with constant-time operations: 128-bit, 256-bit,
/// and 384-bit. Only a few operations are implemented.
#[cfg(any(
    feature = "w32_backend",
    all(not(feature = "w64_backend"), target_pointer_width = "32"),
))]
pub use w32::{Zu128, Zu256, Zu384};

/// Finite field: integers modulo 2^255 - `MQ`.
///
/// The modulus MUST be prime. The type parameter `MQ` MUST be an odd
/// integer between 1 and 32767. This type implements `mul_small()`
/// and `set_mul_small()`. Square root computations are possible
/// if the modulus is equal to 3, 5 or 7 modulo 8, but not if the
/// modulus is equal to 1 modulo 8 (this would trigger a panic).
#[cfg(all(
    any(
        feature = "w32_backend",
        all(not(feature = "w64_backend"), target_pointer_width = "32")),
    feature = "gf255",
))]
pub use w32::GF255;

/// Finite field: integers modulo 2^255 - 19.
///
/// This type implements `mul_small()` and `set_mul_small()`.
#[cfg(all(
    any(
        feature = "w32_backend",
        all(not(feature = "w64_backend"), target_pointer_width = "32")),
    feature = "gf25519",
))]
pub use w32::GF25519;

/// Finite field: integers modulo 2^255 - 18651.
///
/// This type implements `mul_small()` and `set_mul_small()`.
#[cfg(all(
    any(
        feature = "w32_backend",
        all(not(feature = "w64_backend"), target_pointer_width = "32")),
    feature = "gf255e",
))]
pub use w32::GF255e;

/// Finite field: integers modulo 2^255 - 3957.
///
/// This type implements `mul_small()` and `set_mul_small()`.
#[cfg(all(
    any(
        feature = "w32_backend",
        all(not(feature = "w64_backend"), target_pointer_width = "32")),
    feature = "gf255s",
))]
pub use w32::GF255s;

/// Finite field: generic 256-bit modulus.
///
/// The modulus is provided as four 64-bit type parameters, that encode
/// the modulus in base 2^64 (`M0` is the least significant limb,
/// `M3` is the most significant limb). The modulus MUST have length
/// at least 193 bits (i.e. `M3` must not be zero). The modulus MUST be
/// odd (i.e. `M0` must be odd). The modulus SHOULD be prime; if the
/// modulus is not prime, then division by a non-invertible divisor
/// yields 0 (regardless of dividend), and square root computations return
/// unspecified results (the `legendre()` function should still work,
/// though).
///
/// Square root computations are possible if the modulus is equal to 3, 5
/// or 7 modulo 8, but not if the modulus is equal to 1 modulo 8 (this
/// would trigger a panic).
///
/// This type implements `set_mul3()` and `mul3()`.
///
/// This type does NOT implement the `encode()` function; it provides
/// an `encode32()` function that returns an encoding over 32 bytes;
/// the proper `ENC_LEN` constant is still defined.
///
/// The internal implementation strategy uses Montgomery multiplication.
/// Some moduli yield better performance, especially moduli that contain
/// limbs of value 0, and moduli such that `M0` is 0xFFFFFFFFFFFFFFFF.
#[cfg(all(
    any(
        feature = "w32_backend",
        all(not(feature = "w64_backend"), target_pointer_width = "32")),
    feature = "modint256",
))]
pub use w32::ModInt256;

/// A type similar to `ModInt256` but with "enforced constant-time". This
/// is meant to support some uncooperative platforms where multiplication
/// opcodes are not constant-time in their full application range; in
/// that case, `ModInt256ct` ensures constant-time processing, but at
/// a cost (it will be substantially slower than `ModInt256` in those cases).
#[cfg(all(
    any(
        feature = "w32_backend",
        all(not(feature = "w64_backend"), target_pointer_width = "32")),
    feature = "modint256",
))]
pub use w32::ModInt256ct;

/// Finite field: integers modulo 2^256 - 2^32 - 977.
///
/// This is a dedicated type for the base field used by curve secp256k1.
#[cfg(all(
    any(
        feature = "w32_backend",
        all(not(feature = "w64_backend"), target_pointer_width = "32")),
    feature = "gfsecp256k1",
))]
pub use w32::GFsecp256k1;

/// Finite field: integers modulo 2^256 - 2^224 + 2^192 + 2^96 - 1.
///
/// This is a dedicated type for the base field used by curve P-256.
#[cfg(all(
    any(
        feature = "w32_backend",
        all(not(feature = "w64_backend"), target_pointer_width = "32")),
    feature = "gfp256",
))]
pub use w32::GFp256;

/// Finite field: integers modulo 2^448 - 2^224 - 1.
///
/// This is a dedicated type for the base field used by curve Curve448.
#[cfg(all(
    any(
        feature = "w32_backend",
        all(not(feature = "w64_backend"), target_pointer_width = "32")),
    feature = "gf448",
))]
pub use w32::GF448;

/// Finite fields: GF(2^127) and GF(2^254)
///
/// These are dedicated types used for the base field of curve GLS-254.
#[cfg(all(
    any(
        feature = "w32_backend",
        all(not(feature = "w64_backend"), target_pointer_width = "32")),
    feature = "gfb254",
))]
pub use w32::{GFb127, GFb254};

/// Finite field generic implementation: support macro.
#[cfg(all(
    any(
        feature = "w32_backend",
        all(not(feature = "w64_backend"), target_pointer_width = "32")),
    feature = "gfgen",
))]
pub use w32::gfgen::define_gfgen;

/// Finite field generic implementation: support macro (tests).
#[cfg(all(
    any(
        feature = "w32_backend",
        all(not(feature = "w64_backend"), target_pointer_width = "32")),
    feature = "gfgen",
))]
pub use w32::gfgen::define_gfgen_tests;

#[cfg(any(
    feature = "w64_backend",
    all(not(feature = "w32_backend"), target_pointer_width = "64"),
))]
pub mod w64;

/// Fixed-size integers with constant-time operations: 128-bit, 256-bit,
/// and 384-bit. Only a few operations are implemented.
#[cfg(any(
    feature = "w64_backend",
    all(not(feature = "w32_backend"), target_pointer_width = "64"),
))]
pub use w64::{Zu128, Zu256, Zu384};

/// Finite field: integers modulo 2^255 - `MQ`.
///
/// The modulus MUST be prime. The type parameter `MQ` MUST be an odd
/// integer between 1 and 32767. This type implements `mul_small()`
/// and `set_mul_small()`. Square root computations are possible
/// if the modulus is equal to 3, 5 or 7 modulo 8, but not if the
/// modulus is equal to 1 modulo 8 (this would trigger a panic).
#[cfg(all(
    any(
        feature = "w64_backend",
        all(not(feature = "w32_backend"), target_pointer_width = "64")),
    feature = "gf255",
))]
pub use w64::GF255;

/// Finite field: integers modulo 2^255 - 19.
///
/// This type implements `mul_small()` and `set_mul_small()`.
#[cfg(all(
    any(
        feature = "w64_backend",
        all(not(feature = "w32_backend"), target_pointer_width = "64")),
    feature = "gf25519",
))]
pub use w64::GF25519;

/// Finite field: integers modulo 2^255 - 18651.
///
/// This type implements `mul_small()` and `set_mul_small()`.
#[cfg(all(
    any(
        feature = "w64_backend",
        all(not(feature = "w32_backend"), target_pointer_width = "64")),
    feature = "gf255e",
))]
pub use w64::GF255e;

/// Finite field: integers modulo 2^255 - 3957.
///
/// This type implements `mul_small()` and `set_mul_small()`.
#[cfg(all(
    any(
        feature = "w64_backend",
        all(not(feature = "w32_backend"), target_pointer_width = "64")),
    feature = "gf255s",
))]
pub use w64::GF255s;

/// Finite field: generic 256-bit modulus.
///
/// The modulus is provided as four 64-bit type parameters, that encode
/// the modulus in base 2^64 (`M0` is the least significant limb,
/// `M3` is the most significant limb). The modulus MUST have length
/// at least 193 bits (i.e. `M3` must not be zero). The modulus MUST be
/// odd (i.e. `M0` must be odd). The modulus SHOULD be prime; if the
/// modulus is not prime, then division by a non-invertible divisor
/// yields 0 (regardless of dividend), and square root computations return
/// unspecified results (the `legendre()` function should still work,
/// though).
///
/// Square root computations are possible if the modulus is equal to 3, 5
/// or 7 modulo 8, but not if the modulus is equal to 1 modulo 8 (this
/// would trigger a panic).
///
/// This type implements `set_mul3()` and `mul3()`.
///
/// The internal implementation strategy uses Montgomery multiplication.
/// Some moduli yield better performance, especially moduli that contains
/// limbs of value 0, and moduli such that `M0` is 0xFFFFFFFFFFFFFFFF.
#[cfg(all(
    any(
        feature = "w64_backend",
        all(not(feature = "w32_backend"), target_pointer_width = "64")),
    feature = "modint256",
))]
pub use w64::ModInt256;

/// A type similar to `ModInt256` but with "enforced constant-time". This
/// is meant to support some uncooperative platforms where multiplication
/// opcodes are not constant-time in their full application range; in
/// that case, `ModInt256ct` ensures constant-time processing, but at
/// a cost (it will be substantially slower than `ModInt256` in those cases).
#[cfg(all(
    any(
        feature = "w64_backend",
        all(not(feature = "w32_backend"), target_pointer_width = "64")),
    feature = "modint256",
))]
pub use w64::ModInt256ct;

/// Finite field: integers modulo 2^256 - 2^32 - 977.
///
/// This is a dedicated type for the base field used by curve secp256k1.
#[cfg(all(
    any(
        feature = "w64_backend",
        all(not(feature = "w32_backend"), target_pointer_width = "64")),
    feature = "gfsecp256k1",
))]
pub use w64::GFsecp256k1;

/// Finite field: integers modulo 2^256 - 2^224 + 2^192 + 2^96 - 1.
///
/// This is a dedicated type for the base field used by curve P-256.
#[cfg(all(
    any(
        feature = "w64_backend",
        all(not(feature = "w32_backend"), target_pointer_width = "64")),
    feature = "gfp256",
))]
pub use w64::GFp256;

/// Finite field: integers modulo 2^448 - 2^224 - 1.
///
/// This is a dedicated type for the base field used by curve Curve448.
#[cfg(all(
    any(
        feature = "w64_backend",
        all(not(feature = "w32_backend"), target_pointer_width = "64")),
    feature = "gf448",
))]
pub use w64::GF448;

/// Finite fields: GF(2^127) and GF(2^254)
///
/// These are dedicated types used for the base field of curve GLS-254.
#[cfg(all(
    any(
        feature = "w64_backend",
        all(not(feature = "w32_backend"), target_pointer_width = "64")),
    feature = "gfb254",
))]
pub use w64::{GFb127, GFb254};

/// Finite field generic implementation: support macro.
#[cfg(all(
    any(
        feature = "w64_backend",
        all(not(feature = "w32_backend"), target_pointer_width = "64")),
    feature = "gfgen",
))]
pub use w64::gfgen::define_gfgen;

/// Finite field generic implementation: support macro (tests).
#[cfg(all(
    any(
        feature = "w64_backend",
        all(not(feature = "w32_backend"), target_pointer_width = "64")),
    feature = "gfgen",
))]
pub use w64::gfgen::define_gfgen_tests;
