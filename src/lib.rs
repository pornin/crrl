//! Crrl is a Rust library for cryptographic research.
//!
//! This library implements computations in some finite fields and
//! elliptic curves. It aims at providing efficient and secure
//! (constant-time) implementations, but with portable code, and with a
//! convenient API so that scalars, curve points, and other field
//! elements may be used in straightforward expressions with normal
//! arihtmetic operators.
//!
//! Finite fields are implemented through some customizable types defined
//! in `backend` (a 32-bit and a 64-bit backends are provided, the "right
//! one" is automatically selected, unless overridden by a compile-time
//! feature). The types may support several distinct moduli, chosen
//! through compile-time type parameter.
//!
//! Curve Edwards25519 is implemented in the `ed25519` module. The
//! specialized X25519 function is in `x25519`. The prime-order group
//! Ristretto255 (internally based on Edwards25519) is defined in the
//! `ristretto255` module. NIST curve P-256 (aka "secp256r1" and
//! "prime256v1") is implemented in the `p256` module (with the ECDSA
//! signature algorithm). Double-odd curves jq255e and jq255s are
//! implemented by `jq255e` and `jq255s`, respectively (including
//! signature and key exchange schemes). Other curves will be implemented
//! in the future (e.g. secp256k1).
//!
//! # Usage
//!
//! The library is "mostly `no_std`". By default, it compiles against the
//! standard library. It can be compiled in `no_std` mode, in which case
//! all functionality is still available, except verification of truncated
//! ECDSA signatures with curve P-256.
//!
//! # Conventions
//!
//! All implemented functions should be strictly constant-time, unless
//! explicitly documented otherwise (non-constant-time functions normally
//! have "vartime" in their name). In order to avoid unwanted side-channel
//! leaks, Booleans are avoided (compilers tend to "optimize" things a bit
//! too eagerly when handling `bool` values). All functions that return or
//! use a potentially secret Boolean value use the `u32` type; the convention
//! is that 0xFFFFFFFF means "true", and 0x00000000 means "false". No other
//! value shall be used, for they would lead to unpredictable results.
//! Similarly, the `Eq` or `PartialEq` traits are not implemented.
//!
//! Algebraic operations on field elements and curve points are performed
//! with the usual operators (e.g. `+`); appropriate traits are defined
//! so that structure types and pointers to structure types can be used
//! more or less interchangeably. Throughout the code, functions that
//! modify the object on which they are called tend to have a name in
//! `set_*()` (e.g. for a curve point `P`, if we want to compute the
//! double of that point, then `P.set_double()` modifies the point
//! structure in place, while `P.double()` leaves `P` unmodified and
//! returns the double as a new structure instance).
//!
//! # Truncated Signatures
//!
//! Apart from standard support for curve operations and signature
//! algorithms, _truncated signatures_ are implemented for both Ed25519
//! (Schnorr signatures over Edwards25519) and ECDSA (over P-256). A
//! truncated signature is a shrunk version, by up to 32 bits, of a
//! normal signature; the verification process is then more expensive,
//! though not necessarily intolerably expensive, depending on usage
//! context (the most expensive verification function is for ECDSA on
//! P-256, with maximal 32-bit truncation; in that case, verification
//! cost can be up to about 0.65 seconds on a 500 MHz ARM Cortex A53; but
//! Ed25519 signatures with 32-bit truncation can be verified in less
//! than 0.05 seconds on the same hardware). Signature truncation can be
//! useful in situations with strong I/O constraints, where every data
//! bit counts, but where use of fully standard Ed25519 or ECDSA
//! signature generators is made mandatory because of some regulatory or
//! physical constraints of the signing hardware.
//! 
//! # Performance
//!
//! On an Intel i5-8259U CPU (Coffee Lake core), Ed25519 signatures have
//! been benchmarked at about 51600 cycles for signing, 114000 cycles for
//! verification; these are not bad values, and are competitive or at
//! least within 30% of performance obtained from assembly-optimized
//! implementations on the same hardware. For P-256, signing time is
//! about 125000 cycles, verification is 256000 cycles. For the jq255e
//! curve, signatures are generated in about 54700 cycles, and verified
//! in only 82800 cycles (56200 and 86800, respectively, for jq255s).
//! These figures have been obtained by compiling with Rust 1.59 in
//! release mode, with the flags `-C target-cpu=native`.
//!
//! On an ARM Cortex A53 (RaspberryPi Model 3B), Ed25519 signing was
//! measured at 213000 cycles, verification at 479000 cycles; for P-256,
//! the figures were 389000 and 991000, respectively. With jq255e,
//! signature generation and verification use 241000 and 358000 cycles,
//! respectively (248000 and 369000 for jq255s).
//!
//! No inline assembly is used. On x86-64 architectures, the
//! `_addcarry_u64()` and `_subborrow_u64()` intrinsics are used
//! (from `core::arch::x86_64`); however, plain implementations with
//! no intrinsics are available (and used on aarch64).

#![no_std]

#[cfg(all(feature = "alloc", not(feature = "std")))]
#[macro_use]
extern crate alloc;

#[cfg(feature = "std")]
#[macro_use]
extern crate std;

#[cfg(all(feature = "alloc", not(feature = "std")))]
pub(crate) use alloc::vec::Vec;

#[cfg(feature = "std")]
pub(crate) use std::vec::Vec;

pub use rand_core::{CryptoRng, RngCore, Error as RngError};

macro_rules! static_assert {
    ($condition:expr) => {
        let _ = &[()][1 - ($condition) as usize];
    }
}

pub mod backend;
pub mod field;
pub mod ed25519;
pub mod x25519;
pub mod p256;
pub mod ristretto255;
pub mod jq255e;
pub mod jq255s;
pub mod secp256k1;

#[cfg(feature = "alloc")]
pub mod frost;
