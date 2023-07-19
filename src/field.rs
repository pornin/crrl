//! Finite fields.
//!
//! This module defines a few specific finite fields, used as base fields
//! by various curves. These are merely specializations of the
//! backend-provided `GF255` and `ModInt256` types.

pub use crate::backend::{GF255, ModInt256, GFsecp256k1, GF448};

/// Field: integers modulo 2^255 - 19
/// (base field for Curve25519 and derivatives: X25519, ed25519, ristretto255).
pub type GF25519 = GF255<19>;

/// Field: integers modulo 2^255 - 18651
/// (base field for double-odd curve do255e).
pub type GF255e = GF255<18651>;

/// Field: integers modulo 2^255 - 3957
/// (base field for double-odd curve do255s).
pub type GF255s = GF255<3957>;

/// Field: integers modulo 2^256 - 2^224 + 2^192 + 2^96 - 1
/// (base field for NIST curve P-256).
pub type GFp256 = ModInt256<0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFF,
                            0x0000000000000000, 0xFFFFFFFF00000001>;

impl GFp256 {
    /// Encodes a field element into bytes (little-endian).
    pub fn encode(self) -> [u8; 32] {
        self.encode32()
    }
}
