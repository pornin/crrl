//! Finite fields.
//!
//! This module defines a few specific finite fields, used as base fields
//! by various curves. These are merely specializations of the
//! backend-provided `GF255` and `ModInt256` types.

#[cfg(feature = "gf255e")]
pub use crate::backend::GF255e;

#[cfg(feature = "gf255s")]
pub use crate::backend::GF255s;

#[cfg(feature = "gf25519")]
pub use crate::backend::GF25519;

#[cfg(feature = "modint256")]
pub use crate::backend::ModInt256;

#[cfg(feature = "gfsecp256k1")]
pub use crate::backend::GFsecp256k1;

#[cfg(feature = "gfp256")]
pub use crate::backend::GFp256;

#[cfg(feature = "gf448")]
pub use crate::backend::GF448;
