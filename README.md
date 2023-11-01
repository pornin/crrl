# crrl

This library implements some primitives for purposes of cryptographic
research. Its point is to provide efficient, optimized and constant-time
implementations that are supposed to be representative of
production-ready code, so that realistic performance benchmarks may be
performed. Thus, while meant primarily for research, the code here
*should* be fine for production use (though of course I offer no such
guarantee; use at your own risks).

So far, only some primitives related to elliptic curve cryptography
are implemented:

  - A generic type `GF255<MQ>` for finite fields of integers modulo a
    prime 2^255-`MQ` (for a value of `MQ` between 1 and 32767). The `MQ`
    value is provided as a type parameter, i.e. the exact field is known
    at compile time. This type covers the usual modulus 2^255-19 (used
    in Curve25519) as well as 2^255-18651 and 2^255-3957 (used in
    [double-odd curves do255e and do255s](https://doubleodd.group/)).

  - A generic type `ModInt256<M0, M1, M2, M3>` for arbitrary finite
    fields of integers modulo a prime between 2^192 and 2^256.
    Montgomery representation is internally used. The modulus is
    provided as type parameters, allowing the compiler to apply
    optimizations when some parts of the modulus allow them (in
    particular with the modulus used for NIST curve P-256).

  - Type `GFsecp256k1` implements the specific base field for curve
    secp256k1 (integers modulo 2^256-4294968273). The 64-bit backend
    has a dedicated implementation, while the 32-bit version of this
    type uses `ModInt256`.

  - The macro `define_gfgen` allows defining arbitrary finite fields
    of integers modulo a prime, with a large range of modulus size.
    It uses Montgomery representation internally.

  - Type `GF448` implements the specific base field for Curve448.
    The 64-bit backend has a dedicated implementation, while the 32-bit
    backend uses `define_gfgen`.

  - Type `ed25519::Point` provides generic group operations in the
    twisted Edwards curve Curve25519. Ed25519 signatures (as per [RFC
    8032](https://datatracker.ietf.org/doc/html/rfc8032)) are
    implemented. Type `ed25519::Scalar` implements operations on
    integers modulo the curve subgroup order.

  - Type `ristretto255::Point` provides generic group operations in the
    [ristretto255 group](https://ristretto.group/), whose prime order is
    exactly the size of the interesting subgroup of Curve25519.

  - Type `ed448::Point` provides generic group operations in the
    Edwards curve edwards448. Ed448 signatures (as per [RFC
    8032](https://datatracker.ietf.org/doc/html/rfc8032)) are
    implemented. Type `ed448::Scalar` implements operations on
    integers modulo the curve subgroup order.

  - Type `decaf448::Point` provides generic group operations in the
    [decaf448 group](https://ristretto.group/), whose prime order is
    exactly the size of the interesting subgroup of Curve448.

  - Type `p256::Point` provides generic group operations in the NIST
    P-256 curve (aka "secp256r1" aka "prime256v1"). ECDSA signatures are
    supported. The `p256::Scalar` type implements the corresponding
    scalars (integers modulo the curve order).

  - Type `secp256k1::Point` provides generic group operations in the
    secp256k1 curve (aka "the Bitcoin curve"). ECDSA signatures are
    supported. The `secp256k1::Scalar` type implements the corresponding
    scalars (integers modulo the curve order). The GLV endomorphism is
    leveraged to speed-up point multiplication (key exchange) and
    signature verification.

  - Types `jq255e::Point` and `jq255s::Point` implement the
    [double-odd curves](https://doubleodd.group/) jq255e and jq255s
    (along with the corresponding scalar types `jq255e::Scalar` and
    `jq255s::Scalar`). Key exchange and Schnorr signatures are
    implemented. These curves provide a prime-order group abstraction,
    similar to ristretto255, but with somewhat better performance at the
    same security level. Moreover, the relevant signatures are both
    shorter (48 bytes instead of 64) and faster than the usual Ed25519
    signatures.

  - Function `x25519::x25519()` implements the
    [X25519 function](https://datatracker.ietf.org/doc/html/rfc7748#section-5).
    An optimized `x25519::x2559_base()` function is provided when X25519
    is applied to the conventional base point. Similarly, `x448::x448()`
    and `x448::x448_base()` provide the same functionality for the
    X448 function.

  - Type `gls254::Point` implements the GLS254 curve (or, more precisely,
    a prime-order group homomorphic to a subgroup of that curve), which is
    defined over a binary field. `gls254::Scalar` is the type for integers
    modulo the curve order. `gls254::PrivateKey` and `gls254:PublicKey`
    implement high-level operations such as key exchange and signatures,
    using that group.

  - Module `blake2s` contains some BLAKE2s implementations, with
    optional SSE2 and AVX2 optimizations.

Types `GF255` and `ModInt256` have a 32-bit and a 64-bit implementations
each (actually two 64-bit implementations, see later the discussion
about the `gf255_m51` feature). The code is portable (it was tested on
32-bit and 64-bit x86, 64-bit aarch64, and 64-bit riscv64). Performance
is quite decent; e.g. Ed25519 signatures are computed in about 51500
cycles, and verified in about 114000 cycles, on an Intel "Coffee Lake"
CPU; this is not too far from the best assembly-optimized
implementations. At the same time, use of operator overloading allows to
express formulas on points and scalar with about the same syntax as
their mathematical description. For instance, the core of the X25519
implementation looks like this:

```
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
```

which is quite close to the corresponding description in RFC 7748:

```
       A = x_2 + z_2
       AA = A^2
       B = x_2 - z_2
       BB = B^2
       E = AA - BB
       C = x_3 + z_3
       D = x_3 - z_3
       DA = D * A
       CB = C * B
       x_3 = (DA + CB)^2
       z_3 = x_1 * (DA - CB)^2
       x_2 = AA * BB
       z_2 = E * (AA + a24 * E)
```

# Optional Features

By default, everything in crrl is compiled, which unfortunately makes for
a relatively long compilation time, especially on not-so-fast systems.
To only compile support for some primitives, use `--no-default-features`
then add selectively the features you are interested in with `-F`; e.g.
use `cargo build --no-default-features -F ed25519` to only compile the
Ed25519 support (and the primitives that it needs, such as its base
field). The defined primitive-controlling features are the following:

  - `omnes`: enables all of the following.

  - `decaf448`: decaf448 prime-order group (based on edwards448)

  - `ed25519`: edwards25519 curve and signatures (RFC 8032: Ed25519)

  - `ed448`: edwards448 curve and signatures (RFC 8032: Ed448)
  
  - `frost`: FROST threshold signatures (support macros + standard
    ciphersuites, but only for the curves which are also enabled in
    this build)

  - `jq255e`: jq255e prime-order group and signatures

  - `jq255s`: jq255s prime-order group and signatures

  - `lms`: LMS support (hash-based signatures)

  - `p256`: NIST P-256 curve and signatures (ECDSA)

  - `ristretto255`: ristretto255 prime-order group (based on edwards25519)

  - `secp256k1`: secp256k1 curve and signatures (ECDSA)

  - `x25519`: X25519 key exchange primitive (RFC 7748)

  - `x448`: X448 key exchange primitive (RFC 7748)

  - `modint256`: generic finite field implementation (prime order of up to
    256 bits)

  - `gf255`: generic finite field implementation (for prime order
    `q = 2^255 - MQ` with `MQ < 2^15`)

  - `gfgen`: generic finite field implementation (generating macro; prime
    modulus of arbitrary length)

  - `gls254`: GLS254 prime-order group and signatures

  - `gls254bench`: additional benchmarking code for GLS254

  - `blake2s`: BLAKE2s hash function

Some operations have multiple backends. An appropriate backend is selected
at compile-time, but this can be overridden by enabling some features:

  - `w32_backend`: enforce use of the 32-bit code, even on 64-bit systems.

  - `w64_backend`: enforce use of the 64-bit code, even on 32-bit systems.

  - `gf255_m64`: enforce use of 64-bit limbs for `GF255<MQ>`; this is the
    default on 64-bit machines, except RISC-V (riscv64) where 51-bit
    limbs are used by default. This feature has no effect if the 32-bit code
    is used.

  - `gf255_m51`: encode use of 51-bit limbs for `GF255<MQ>`; this is the
    default on 64-bit RISC-V targets (riscv64), but not on other 64-bit
    architectures where 64-bit limbs are normally preferred. This feature
    has no effect if the 32-bit code is used.

  - `gfb254_m64`: enforce use of the generic implementation of the
    binary field GF(2^254). This feature has no effect if the 32-bit code
    is used.

  - `gfb254_x86clmul`: enforce use of the AVX2+pclmulqdq implementation of
    the binary field GF(2^254). This code is used automatically if the
    compilation target is an x86 with the relevant hardware support; this
    feature bypasses the automatic detection. This feature has no effect
    if the 32-bit code is used.

  - `gfb254_arm64pmull`: enforce use of the NEON+pmull implementation of
    the binary field GF(2^254). This code is used automatically if the
    compilation target is an aarch64 system; this feature bypasses the
    automatic detection. This feature has no effect if the 32-bit code
    is used.

# Security and Compliance

All the code is strict, both in terms of timing-based side-channels
(everything is constant-time, except if explicitly stated otherwise,
e.g. in a function whose name includes `vartime`) and in compliance to
relevant standards. For instance, the Ed25519 signature support applies
and enforces canonical encodings of both points and scalars.

There is no attempt at "zeroizing memory" anywhere in the code. In
general, such memory cleansing is a fool's quest. Note that since most
of the library use `no_std` rules, dynamic allocation happens only on
the stack, thereby limiting the risk of leaving secret information
lingering all over the RAM. The only functions that use heap allocation
only store public data there.

**WARNING:** I reiterate what was written above: although all of the
code aims at being representative of optimized production-ready code, it
is still fairly recent and some bugs might still lurk, however careful I
am when writing code. Any assertion of suitability to any purpose is
explcitly denied. The primary purpose is to help with "trying out stuff"
in cryptographic research, by offering an easy-to-use API backed by
performance close enough to what can be done in actual applications.

# Truncated Signatures

Support for truncated signatures is implemented for Ed25519 and
ECDSA/P-256. Standard signatures can be shortened by 8 to 32 bits (i.e.
the size may shrink from 64 down to 60 bytes), and the verifier rebuilds
the original signature during verification (at some computational cost).
This is not a ground-breaking feature, but it can be very convenient in
some situations with tight constraints on bandwidth and a requirement to
work with standard signature formats. See
`ed25519::PublicKey::verify_trunc_raw()` and
`p256::PublicKey::verify_trunc_hash()` for details.

# FROST Threshold Schnorr Signatures

The FROST protocol for a distributed Schnorr signature generation scheme
has been implemented, as per the v14 draft specification:
[draft-irtf-cfrg-frost-14](https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-frost-14).
Four ciphersuites are provided, with similar APIs, in the
`frost::ed25519`, `frost::ristretto255`, `frost::ed448`, `frost::p256` and
`frost::secp256k1` modules. A sample code showing how to use the API is
provided in the [frost-sample.rs](extra/frost-sample.rs) file.

While FROST is inherently a distributed scheme, the implementation can
also be used in a single signer mode by using the "group" private key
directly.

# Benchmarks

`cargo bench` runs some benchmarks, but there are a few caveats:

  - The cycle counter is used on x86. If frequency scaling ("TurboBoost")
    is not disabled, then you'll get wrong and meaningless results.

  - On aarch64, the cycle counter is also accessed directly, which will
    in general fail with some CPU exception. Access to the counter must
    first be enabled, which requires (on Linux) a specific kernel
    module. [This
    one](https://github.com/jerinjacobk/armv8_pmu_cycle_counter_el0)
    works for me.

  - On riscv64gc, the cycle counter is accessed directly. In general,
    that counter is not enabled and all benches return zero; to enable
    the cycle counter, run the benchmark binary inside the `perf`
    tool (which comes with the `linux-tools`).

  - On architectures other than i386, x86-64, aarch64 and riscv64gc,
    benchmark code will simply not compile.

# TODO

In general, about anything related to cryptography may show up here,
if there is a use case for it.
