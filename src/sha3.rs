#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

//! # SHAKE Implementation
//!
//! This module includes a (somewhat perfunctory) implementation of SHA3
//! and SHAKE.

// Keccak state (25*8 = 200 bytes).
#[derive(Copy, Clone, Debug)]
struct KeccakState([u64; 25]);

impl KeccakState {

    const RC: [u64; 24] = [
        0x0000000000000001, 0x0000000000008082,
        0x800000000000808A, 0x8000000080008000,
        0x000000000000808B, 0x0000000080000001,
        0x8000000080008081, 0x8000000000008009,
        0x000000000000008A, 0x0000000000000088,
        0x0000000080008009, 0x000000008000000A,
        0x000000008000808B, 0x800000000000008B,
        0x8000000000008089, 0x8000000000008003,
        0x8000000000008002, 0x8000000000000080,
        0x000000000000800A, 0x800000008000000A,
        0x8000000080008081, 0x8000000000008080,
        0x0000000080000001, 0x8000000080008008,
    ];

    // Create a new KeccakState initialized at zero.
    fn new() -> Self {
        Self([0u64; 25])
    }

    fn process(&mut self) {
        let mut A: [u64; 25] = self.0;

        // Invert some words (alternate internal representation, which
        // saves some operations).
        A[ 1] = !A[ 1];
        A[ 2] = !A[ 2];
        A[ 8] = !A[ 8];
        A[12] = !A[12];
        A[17] = !A[17];
        A[20] = !A[20];

        // Compute 24 rounds. The loop is partially unrolled (two rounds
        // per iteration).
        for i in 0..12 {
            let (mut t0, mut t1, mut t2, mut t3, mut t4);
            let (mut tt0, mut tt1, mut tt2, mut tt3);
            let (mut t, mut kt);
            let (mut c0, mut c1, mut c2, mut c3, mut c4, mut bnn);

            tt0 = A[ 1] ^ A[ 6];
            tt1 = A[11] ^ A[16];
            tt0 ^= A[21] ^ tt1;
            tt0 = (tt0 << 1) | (tt0 >> 63);
            tt2 = A[ 4] ^ A[ 9];
            tt3 = A[14] ^ A[19];
            tt0 ^= A[24];
            tt2 ^= tt3;
            t0 = tt0 ^ tt2;

            tt0 = A[ 2] ^ A[ 7];
            tt1 = A[12] ^ A[17];
            tt0 ^= A[22] ^ tt1;
            tt0 = (tt0 << 1) | (tt0 >> 63);
            tt2 = A[ 0] ^ A[ 5];
            tt3 = A[10] ^ A[15];
            tt0 ^= A[20];
            tt2 ^= tt3;
            t1 = tt0 ^ tt2;

            tt0 = A[ 3] ^ A[ 8];
            tt1 = A[13] ^ A[18];
            tt0 ^= A[23] ^ tt1;
            tt0 = (tt0 << 1) | (tt0 >> 63);
            tt2 = A[ 1] ^ A[ 6];
            tt3 = A[11] ^ A[16];
            tt0 ^= A[21];
            tt2 ^= tt3;
            t2 = tt0 ^ tt2;

            tt0 = A[ 4] ^ A[ 9];
            tt1 = A[14] ^ A[19];
            tt0 ^= A[24] ^ tt1;
            tt0 = (tt0 << 1) | (tt0 >> 63);
            tt2 = A[ 2] ^ A[ 7];
            tt3 = A[12] ^ A[17];
            tt0 ^= A[22];
            tt2 ^= tt3;
            t3 = tt0 ^ tt2;

            tt0 = A[ 0] ^ A[ 5];
            tt1 = A[10] ^ A[15];
            tt0 ^= A[20] ^ tt1;
            tt0 = (tt0 << 1) | (tt0 >> 63);
            tt2 = A[ 3] ^ A[ 8];
            tt3 = A[13] ^ A[18];
            tt0 ^= A[23];
            tt2 ^= tt3;
            t4 = tt0 ^ tt2;

            A[ 0] = A[ 0] ^ t0;
            A[ 5] = A[ 5] ^ t0;
            A[10] = A[10] ^ t0;
            A[15] = A[15] ^ t0;
            A[20] = A[20] ^ t0;
            A[ 1] = A[ 1] ^ t1;
            A[ 6] = A[ 6] ^ t1;
            A[11] = A[11] ^ t1;
            A[16] = A[16] ^ t1;
            A[21] = A[21] ^ t1;
            A[ 2] = A[ 2] ^ t2;
            A[ 7] = A[ 7] ^ t2;
            A[12] = A[12] ^ t2;
            A[17] = A[17] ^ t2;
            A[22] = A[22] ^ t2;
            A[ 3] = A[ 3] ^ t3;
            A[ 8] = A[ 8] ^ t3;
            A[13] = A[13] ^ t3;
            A[18] = A[18] ^ t3;
            A[23] = A[23] ^ t3;
            A[ 4] = A[ 4] ^ t4;
            A[ 9] = A[ 9] ^ t4;
            A[14] = A[14] ^ t4;
            A[19] = A[19] ^ t4;
            A[24] = A[24] ^ t4;
            A[ 5] = (A[ 5] << 36) | (A[ 5] >> (64 - 36));
            A[10] = (A[10] <<  3) | (A[10] >> (64 -  3));
            A[15] = (A[15] << 41) | (A[15] >> (64 - 41));
            A[20] = (A[20] << 18) | (A[20] >> (64 - 18));
            A[ 1] = (A[ 1] <<  1) | (A[ 1] >> (64 -  1));
            A[ 6] = (A[ 6] << 44) | (A[ 6] >> (64 - 44));
            A[11] = (A[11] << 10) | (A[11] >> (64 - 10));
            A[16] = (A[16] << 45) | (A[16] >> (64 - 45));
            A[21] = (A[21] <<  2) | (A[21] >> (64 - 2));
            A[ 2] = (A[ 2] << 62) | (A[ 2] >> (64 - 62));
            A[ 7] = (A[ 7] <<  6) | (A[ 7] >> (64 -  6));
            A[12] = (A[12] << 43) | (A[12] >> (64 - 43));
            A[17] = (A[17] << 15) | (A[17] >> (64 - 15));
            A[22] = (A[22] << 61) | (A[22] >> (64 - 61));
            A[ 3] = (A[ 3] << 28) | (A[ 3] >> (64 - 28));
            A[ 8] = (A[ 8] << 55) | (A[ 8] >> (64 - 55));
            A[13] = (A[13] << 25) | (A[13] >> (64 - 25));
            A[18] = (A[18] << 21) | (A[18] >> (64 - 21));
            A[23] = (A[23] << 56) | (A[23] >> (64 - 56));
            A[ 4] = (A[ 4] << 27) | (A[ 4] >> (64 - 27));
            A[ 9] = (A[ 9] << 20) | (A[ 9] >> (64 - 20));
            A[14] = (A[14] << 39) | (A[14] >> (64 - 39));
            A[19] = (A[19] <<  8) | (A[19] >> (64 -  8));
            A[24] = (A[24] << 14) | (A[24] >> (64 - 14));

            bnn = !A[12];
            kt = A[ 6] | A[12];
            c0 = A[ 0] ^ kt;
            kt = bnn | A[18];
            c1 = A[ 6] ^ kt;
            kt = A[18] & A[24];
            c2 = A[12] ^ kt;
            kt = A[24] | A[ 0];
            c3 = A[18] ^ kt;
            kt = A[ 0] & A[ 6];
            c4 = A[24] ^ kt;
            A[ 0] = c0;
            A[ 6] = c1;
            A[12] = c2;
            A[18] = c3;
            A[24] = c4;
            bnn = !A[22];
            kt = A[ 9] | A[10];
            c0 = A[ 3] ^ kt;
            kt = A[10] & A[16];
            c1 = A[ 9] ^ kt;
            kt = A[16] | bnn;
            c2 = A[10] ^ kt;
            kt = A[22] | A[ 3];
            c3 = A[16] ^ kt;
            kt = A[ 3] & A[ 9];
            c4 = A[22] ^ kt;
            A[ 3] = c0;
            A[ 9] = c1;
            A[10] = c2;
            A[16] = c3;
            A[22] = c4;
            bnn = !A[19];
            kt = A[ 7] | A[13];
            c0 = A[ 1] ^ kt;
            kt = A[13] & A[19];
            c1 = A[ 7] ^ kt;
            kt = bnn & A[20];
            c2 = A[13] ^ kt;
            kt = A[20] | A[ 1];
            c3 = bnn ^ kt;
            kt = A[ 1] & A[ 7];
            c4 = A[20] ^ kt;
            A[ 1] = c0;
            A[ 7] = c1;
            A[13] = c2;
            A[19] = c3;
            A[20] = c4;
            bnn = !A[17];
            kt = A[ 5] & A[11];
            c0 = A[ 4] ^ kt;
            kt = A[11] | A[17];
            c1 = A[ 5] ^ kt;
            kt = bnn | A[23];
            c2 = A[11] ^ kt;
            kt = A[23] & A[ 4];
            c3 = bnn ^ kt;
            kt = A[ 4] | A[ 5];
            c4 = A[23] ^ kt;
            A[ 4] = c0;
            A[ 5] = c1;
            A[11] = c2;
            A[17] = c3;
            A[23] = c4;
            bnn = !A[ 8];
            kt = bnn & A[14];
            c0 = A[ 2] ^ kt;
            kt = A[14] | A[15];
            c1 = bnn ^ kt;
            kt = A[15] & A[21];
            c2 = A[14] ^ kt;
            kt = A[21] | A[ 2];
            c3 = A[15] ^ kt;
            kt = A[ 2] & A[ 8];
            c4 = A[21] ^ kt;
            A[ 2] = c0;
            A[ 8] = c1;
            A[14] = c2;
            A[15] = c3;
            A[21] = c4;
            A[ 0] = A[ 0] ^ Self::RC[2 * i + 0];

            tt0 = A[ 6] ^ A[ 9];
            tt1 = A[ 7] ^ A[ 5];
            tt0 ^= A[ 8] ^ tt1;
            tt0 = (tt0 << 1) | (tt0 >> 63);
            tt2 = A[24] ^ A[22];
            tt3 = A[20] ^ A[23];
            tt0 ^= A[21];
            tt2 ^= tt3;
            t0 = tt0 ^ tt2;

            tt0 = A[12] ^ A[10];
            tt1 = A[13] ^ A[11];
            tt0 ^= A[14] ^ tt1;
            tt0 = (tt0 << 1) | (tt0 >> 63);
            tt2 = A[ 0] ^ A[ 3];
            tt3 = A[ 1] ^ A[ 4];
            tt0 ^= A[ 2];
            tt2 ^= tt3;
            t1 = tt0 ^ tt2;

            tt0 = A[18] ^ A[16];
            tt1 = A[19] ^ A[17];
            tt0 ^= A[15] ^ tt1;
            tt0 = (tt0 << 1) | (tt0 >> 63);
            tt2 = A[ 6] ^ A[ 9];
            tt3 = A[ 7] ^ A[ 5];
            tt0 ^= A[ 8];
            tt2 ^= tt3;
            t2 = tt0 ^ tt2;

            tt0 = A[24] ^ A[22];
            tt1 = A[20] ^ A[23];
            tt0 ^= A[21] ^ tt1;
            tt0 = (tt0 << 1) | (tt0 >> 63);
            tt2 = A[12] ^ A[10];
            tt3 = A[13] ^ A[11];
            tt0 ^= A[14];
            tt2 ^= tt3;
            t3 = tt0 ^ tt2;

            tt0 = A[ 0] ^ A[ 3];
            tt1 = A[ 1] ^ A[ 4];
            tt0 ^= A[ 2] ^ tt1;
            tt0 = (tt0 << 1) | (tt0 >> 63);
            tt2 = A[18] ^ A[16];
            tt3 = A[19] ^ A[17];
            tt0 ^= A[15];
            tt2 ^= tt3;
            t4 = tt0 ^ tt2;

            A[ 0] = A[ 0] ^ t0;
            A[ 3] = A[ 3] ^ t0;
            A[ 1] = A[ 1] ^ t0;
            A[ 4] = A[ 4] ^ t0;
            A[ 2] = A[ 2] ^ t0;
            A[ 6] = A[ 6] ^ t1;
            A[ 9] = A[ 9] ^ t1;
            A[ 7] = A[ 7] ^ t1;
            A[ 5] = A[ 5] ^ t1;
            A[ 8] = A[ 8] ^ t1;
            A[12] = A[12] ^ t2;
            A[10] = A[10] ^ t2;
            A[13] = A[13] ^ t2;
            A[11] = A[11] ^ t2;
            A[14] = A[14] ^ t2;
            A[18] = A[18] ^ t3;
            A[16] = A[16] ^ t3;
            A[19] = A[19] ^ t3;
            A[17] = A[17] ^ t3;
            A[15] = A[15] ^ t3;
            A[24] = A[24] ^ t4;
            A[22] = A[22] ^ t4;
            A[20] = A[20] ^ t4;
            A[23] = A[23] ^ t4;
            A[21] = A[21] ^ t4;
            A[ 3] = (A[ 3] << 36) | (A[ 3] >> (64 - 36));
            A[ 1] = (A[ 1] <<  3) | (A[ 1] >> (64 -  3));
            A[ 4] = (A[ 4] << 41) | (A[ 4] >> (64 - 41));
            A[ 2] = (A[ 2] << 18) | (A[ 2] >> (64 - 18));
            A[ 6] = (A[ 6] <<  1) | (A[ 6] >> (64 -  1));
            A[ 9] = (A[ 9] << 44) | (A[ 9] >> (64 - 44));
            A[ 7] = (A[ 7] << 10) | (A[ 7] >> (64 - 10));
            A[ 5] = (A[ 5] << 45) | (A[ 5] >> (64 - 45));
            A[ 8] = (A[ 8] <<  2) | (A[ 8] >> (64 - 2));
            A[12] = (A[12] << 62) | (A[12] >> (64 - 62));
            A[10] = (A[10] <<  6) | (A[10] >> (64 -  6));
            A[13] = (A[13] << 43) | (A[13] >> (64 - 43));
            A[11] = (A[11] << 15) | (A[11] >> (64 - 15));
            A[14] = (A[14] << 61) | (A[14] >> (64 - 61));
            A[18] = (A[18] << 28) | (A[18] >> (64 - 28));
            A[16] = (A[16] << 55) | (A[16] >> (64 - 55));
            A[19] = (A[19] << 25) | (A[19] >> (64 - 25));
            A[17] = (A[17] << 21) | (A[17] >> (64 - 21));
            A[15] = (A[15] << 56) | (A[15] >> (64 - 56));
            A[24] = (A[24] << 27) | (A[24] >> (64 - 27));
            A[22] = (A[22] << 20) | (A[22] >> (64 - 20));
            A[20] = (A[20] << 39) | (A[20] >> (64 - 39));
            A[23] = (A[23] <<  8) | (A[23] >> (64 -  8));
            A[21] = (A[21] << 14) | (A[21] >> (64 - 14));

            bnn = !A[13];
            kt = A[ 9] | A[13];
            c0 = A[ 0] ^ kt;
            kt = bnn | A[17];
            c1 = A[ 9] ^ kt;
            kt = A[17] & A[21];
            c2 = A[13] ^ kt;
            kt = A[21] | A[ 0];
            c3 = A[17] ^ kt;
            kt = A[ 0] & A[ 9];
            c4 = A[21] ^ kt;
            A[ 0] = c0;
            A[ 9] = c1;
            A[13] = c2;
            A[17] = c3;
            A[21] = c4;
            bnn = !A[14];
            kt = A[22] | A[ 1];
            c0 = A[18] ^ kt;
            kt = A[ 1] & A[ 5];
            c1 = A[22] ^ kt;
            kt = A[ 5] | bnn;
            c2 = A[ 1] ^ kt;
            kt = A[14] | A[18];
            c3 = A[ 5] ^ kt;
            kt = A[18] & A[22];
            c4 = A[14] ^ kt;
            A[18] = c0;
            A[22] = c1;
            A[ 1] = c2;
            A[ 5] = c3;
            A[14] = c4;
            bnn = !A[23];
            kt = A[10] | A[19];
            c0 = A[ 6] ^ kt;
            kt = A[19] & A[23];
            c1 = A[10] ^ kt;
            kt = bnn & A[ 2];
            c2 = A[19] ^ kt;
            kt = A[ 2] | A[ 6];
            c3 = bnn ^ kt;
            kt = A[ 6] & A[10];
            c4 = A[ 2] ^ kt;
            A[ 6] = c0;
            A[10] = c1;
            A[19] = c2;
            A[23] = c3;
            A[ 2] = c4;
            bnn = !A[11];
            kt = A[ 3] & A[ 7];
            c0 = A[24] ^ kt;
            kt = A[ 7] | A[11];
            c1 = A[ 3] ^ kt;
            kt = bnn | A[15];
            c2 = A[ 7] ^ kt;
            kt = A[15] & A[24];
            c3 = bnn ^ kt;
            kt = A[24] | A[ 3];
            c4 = A[15] ^ kt;
            A[24] = c0;
            A[ 3] = c1;
            A[ 7] = c2;
            A[11] = c3;
            A[15] = c4;
            bnn = !A[16];
            kt = bnn & A[20];
            c0 = A[12] ^ kt;
            kt = A[20] | A[ 4];
            c1 = bnn ^ kt;
            kt = A[ 4] & A[ 8];
            c2 = A[20] ^ kt;
            kt = A[ 8] | A[12];
            c3 = A[ 4] ^ kt;
            kt = A[12] & A[16];
            c4 = A[ 8] ^ kt;
            A[12] = c0;
            A[16] = c1;
            A[20] = c2;
            A[ 4] = c3;
            A[ 8] = c4;
            A[ 0] = A[ 0] ^ Self::RC[2 * i + 1];

            t = A[ 5];
            A[ 5] = A[18];
            A[18] = A[11];
            A[11] = A[10];
            A[10] = A[ 6];
            A[ 6] = A[22];
            A[22] = A[20];
            A[20] = A[12];
            A[12] = A[19];
            A[19] = A[15];
            A[15] = A[24];
            A[24] = A[ 8];
            A[ 8] = t;
            t = A[ 1];
            A[ 1] = A[ 9];
            A[ 9] = A[14];
            A[14] = A[ 2];
            A[ 2] = A[13];
            A[13] = A[23];
            A[23] = A[ 4];
            A[ 4] = A[21];
            A[21] = A[16];
            A[16] = A[ 3];
            A[ 3] = A[17];
            A[17] = A[ 7];
            A[ 7] = t;
        }

        // Invert some words back to normal representation.
        A[ 1] = !A[ 1];
        A[ 2] = !A[ 2];
        A[ 8] = !A[ 8];
        A[12] = !A[12];
        A[17] = !A[17];
        A[20] = !A[20];

        self.0 = A;
    }
}

macro_rules! sha3_impl { ($typename:ident, $size:expr) => {

    #[doc = concat!("SHA3-", stringify!($size), " implementation.\n\n",
        "Instances are cloneable, which captures the current object state.")]
    #[derive(Copy, Clone, Debug)]
    pub struct $typename (SHA3Core<$size>);

    impl $typename {
        /// Create a new SHA3 instance.
        pub fn new() -> Self {
            Self(SHA3Core::<$size>::new())
        }

        /// Process some input bytes; this function can be called repeatedly.
        pub fn update(&mut self, src: impl AsRef <[u8]>) {
            self.0.update(src.as_ref());
        }

        /// Compute the hash of all bytes injected since the last state
        /// reset of this instance. The instance is automatically reset.
        pub fn digest(&mut self) -> [u8; $size >> 3] {
            let mut r = [0u8; $size >> 3];
            self.0.digest_to(&mut r);
            self.0.reset();
            r
        }

        /// Reset this instance to its initial state.
        pub fn reset(&mut self) {
            self.0.reset();
        }

        /// An alias on `self.digest()` (for syntactic compatibility).
        pub fn finalize(&mut self) -> [u8; $size >> 3] {
            self.digest()
        }

        /// An alias on `self.digest()` (for syntactic compatibility).
        pub fn finalize_reset(&mut self) -> [u8; $size >> 3] {
            self.digest()
        }

        /// Finalize this context and write the output into the
        /// provided slice. The number of written bytes is returned.
        /// This instance is automatically reset.
        pub fn finalize_write(&mut self, out: &mut [u8]) -> usize {
            out[..($size >> 3)].copy_from_slice(&self.digest());
            $size >> 3
        }

        /// An alias on `self.finalize_write()` (for syntactic
        /// compatibility).
        pub fn finalize_reset_write(&mut self, out: &mut [u8]) -> usize {
            self.finalize_write(out)
        }

        /// One-call hash of a given input.
        pub fn hash(src: impl AsRef<[u8]>) -> [u8; $size >> 3] {
            let mut s = Self::new();
            s.update(src);
            s.digest()
        }
    }

} }

sha3_impl!(SHA3_224, 224);
sha3_impl!(SHA3_256, 256);
sha3_impl!(SHA3_384, 384);
sha3_impl!(SHA3_512, 512);

// SHA3 core implementation. The type parameter (SZ) MUST be 224, 256, 384
// or 512, corresponding to the implemented SHA3 variant.
//
// An instance is created with the new() method. Input bytes are processed
// with update(). The output is obtained with digest_to(); this automatically
// resets the instance to its initial state, so that it may be reused for
// a new hashing operation. The instance can also be reset at any time with
// reset(). The actual public types are defined with the sha3_impl macro
// as wrappers, so that they may have a digest() function that returns an
// array type with an appropriate size.

#[derive(Clone, Copy, Debug)]
struct SHA3Core<const SZ: usize> {
    state: KeccakState,
    ptr: usize,
}

impl<const SZ: usize> SHA3Core<SZ> {

    // A custom compile-time check; it should prevent compilation from
    // succeeding if SZ is not in {224, 256, 384, 512}.
    #[allow(dead_code)]
    const COMPILE_TIME_CHECKS: () = Self::compile_time_checks();
    const fn compile_time_checks() {
        let _ = &[()][1 -
            ((SZ == 224 || SZ == 256 || SZ == 384 || SZ == 512) as usize)];
    }
    const RATE: usize = 200 - (SZ >> 2);

    fn new() -> Self {
        Self {
            state: KeccakState::new(),
            ptr: 0,
        }
    }

    fn update(&mut self, src: &[u8]) {
        let mut ptr = self.ptr;
        let mut i = 0;
        while i < src.len() {
            let clen = core::cmp::min(src.len() - i, Self::RATE - ptr);
            for _ in 0..clen {
                self.state.0[ptr >> 3] ^= (src[i] as u64) << ((ptr & 7) << 3);
                i += 1;
                ptr += 1;
            }
            if ptr == Self::RATE {
                self.state.process();
                ptr = 0;
            }
        }
        self.ptr = ptr;
    }

    fn digest_to(&mut self, dst: &mut [u8]) {
        assert!(dst.len() == (SZ >> 3));
        let i = self.ptr;
        self.state.0[i >> 3] ^= 0x06u64 << ((i & 7) << 3);
        let i = Self::RATE - 1;
        self.state.0[i >> 3] ^= 0x80u64 << ((i & 7) << 3);
        self.state.process();
        for i in 0..dst.len() {
            dst[i] = (self.state.0[i >> 3] >> ((i & 7) << 3)) as u8;
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

/// SHAKE implementation.
///
/// The type parameter `SZ` must be either 128 or 256, for SHAKE128 and
/// SHAKE256, respectively. An instance is, at any time, in "initial"
/// or "flipped" state. The instance is created in input mode, and goes
/// to output mode when `flip()` is called. In input mode, data can
/// be injected with the `inject()` method. In output mode, data can
/// be extracted with the `extract()` method. The `reset()` method sets
/// back the instance to its starting state (input mode, and empty).
///
/// Instances are cloneable, which captures the current engine state.
#[derive(Clone, Copy, Debug)]
pub struct SHAKE<const SZ: usize> {
    state: KeccakState,
    ptr: usize,
    flipped: bool,
}

/// Type specialization for SHAKE128.
pub type SHAKE128 = SHAKE<128>;

/// Alias on `SHAKE128` (source code compatibility).
pub type Shake128 = SHAKE128;

/// Type specialization for SHAKE256.
pub type SHAKE256 = SHAKE<256>;

/// Alias on `SHAKE256` (source code compatibility).
pub type Shake256 = SHAKE256;

impl<const SZ: usize> SHAKE<SZ> {

    // A custom compile-time check; it should prevent compilation from
    // succeeding if SZ is not 128 or 256.
    #[allow(dead_code)]
    const COMPILE_TIME_CHECKS: () = Self::compile_time_checks();
    const fn compile_time_checks() {
        let _ = &[()][1 - ((SZ == 128 || SZ == 256) as usize)];
    }
    const RATE: usize = 200 - (SZ >> 2);

    /// Create a new instance.
    pub fn new() -> Self {
        Self {
            state: KeccakState::new(),
            ptr: 0,
            flipped: false,
        }
    }

    /// Inject some bytes into the engine.
    ///
    /// This function can be called repeatedly. If the engine is in output
    /// mode, then a panic is triggered.
    pub fn inject(&mut self, src: impl AsRef <[u8]>) {
        assert!(!self.flipped);
        let mut ptr = self.ptr;
        let src = src.as_ref();
        let mut i = 0;
        while i < src.len() {
            let clen = core::cmp::min(src.len() - i, Self::RATE - ptr);
            for _ in 0..clen {
                self.state.0[ptr >> 3] ^= (src[i] as u64) << ((ptr & 7) << 3);
                i += 1;
                ptr += 1;
            }
            if ptr == Self::RATE {
                self.state.process();
                ptr = 0;
            }
        }
        self.ptr = ptr;
    }

    /// An alias on `inject()` (for source compatibility purposes).
    #[inline(always)]
    pub fn update(&mut self, src: impl AsRef <[u8]>) {
        self.inject(src);
    }

    /// Flip the engine from input to output mode.
    ///
    /// If the engine is already in output mode, then a panic is triggered.
    pub fn flip(&mut self) {
        assert!(!self.flipped);
        let i = self.ptr;
        self.state.0[i >> 3] ^= 0x1Fu64 << ((i & 7) << 3);
        let i = Self::RATE - 1;
        self.state.0[i >> 3] ^= 0x80u64 << ((i & 7) << 3);
        self.ptr = Self::RATE;
        self.flipped = true;
    }

    /// Extract some bytes from the engine.
    ///
    /// This function can be called repeatedly. If the engine is in input
    /// mode, then a panic is triggered.
    pub fn extract(&mut self, dst: &mut [u8]) {
        assert!(self.flipped);
        let mut ptr = self.ptr;
        let mut i = 0;
        while i < dst.len() {
            if ptr == Self::RATE {
                self.state.process();
                ptr = 0;
            }
            let clen = core::cmp::min(dst.len() - i, Self::RATE - ptr);
            for _ in 0..clen {
                dst[i] = (self.state.0[ptr >> 3] >> ((ptr & 7) << 3)) as u8;
                i += 1;
                ptr += 1;
            }
        }
        self.ptr = ptr;
    }

    /// Reset this engine to the initial state (empty, input mode).
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Combined `flip()` and `extract()` calls. The engine is _not_ reset
    /// and further `extract()` calls can be performed.
    pub fn flip_extract(&mut self, dst: &mut [u8]) {
        self.flip();
        self.extract(dst);
    }

    /// Combined `flip()`, `extract()` and `reset()` calls.
    pub fn flip_extract_reset(&mut self, dst: &mut [u8]) {
        self.flip();
        self.extract(dst);
        self.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inner_shake<const SZ: usize>(kat: &[&str]) {
        for i in 0..(kat.len() >> 1) {
            let src = hex::decode(&kat[2 * i]).unwrap();
            let dst = hex::decode(&kat[2 * i + 1]).unwrap();
            let r = &mut [0u8; 512][..dst.len()];

            // Test 1: inject and extract in one go.
            let mut sh = SHAKE::<SZ>::new();
            sh.inject(&src);
            sh.flip();
            sh.extract(r);
            assert!(r == dst);

            // Test 2: inject and extract byte by byte.
            for j in 0..r.len() {
                r[j] = 0;
            }
            sh.reset();
            for j in 0..src.len() {
                sh.inject(&[src[j]]);
            }
            sh.flip();
            for j in 0..r.len() {
                let mut tt = [0u8];
                sh.extract(&mut tt[..]);
                r[j] = tt[0];
            }
            assert!(r == dst);
        }
    }

    // This function tests SHAKE128 against test vectors from the NIST Web
    // site.
    #[test]
    fn shake128() {
        inner_shake::<128>(&KAT_SHAKE128);
    }

    static KAT_SHAKE128: [&str; 6] = [
        // Each pair of strings is: input (hex), output (hex)
        "",
        "7f9c2ba4e88f827d616045507605853e",

        "a6fe00064257aa318b621c5eb311d32bb8004c2fa1a969d205d71762cc5d2e633907992629d1b69d9557ff6d5e8deb454ab00f6e497c89a4fea09e257a6fa2074bd818ceb5981b3e3faefd6e720f2d1edd9c5e4a5c51e5009abf636ed5bca53fe159c8287014a1bd904f5c8a7501625f79ac81eb618f478ce21cae6664acffb30572f059e1ad0fc2912264e8f1ca52af26c8bf78e09d75f3dd9fc734afa8770abe0bd78c90cc2ff448105fb16dd2c5b7edd8611a62e537db9331f5023e16d6ec150cc6e706d7c7fcbfff930c7281831fd5c4aff86ece57ed0db882f59a5fe403105d0592ca38a081fed84922873f538ee774f13b8cc09bd0521db4374aec69f4bae6dcb66455822c0b84c91a3474ffac2ad06f0a4423cd2c6a49d4f0d6242d6a1890937b5d9835a5f0ea5b1d01884d22a6c1718e1f60b3ab5e232947c76ef70b344171083c688093b5f1475377e3069863",
        "3109d9472ca436e805c6b3db2251a9bc",

        "0a13ad2c7a239b4ba73ea6592ae84ea9",
        "5feaf99c15f48851943ff9baa6e5055d8377f0dd347aa4dbece51ad3a6d9ce0c01aee9fe2260b80a4673a909b532adcdd1e421c32d6460535b5fe392a58d2634979a5a104d6c470aa3306c400b061db91c463b2848297bca2bc26d1864ba49d7ff949ebca50fbf79a5e63716dc82b600bd52ca7437ed774d169f6bf02e46487956fba2230f34cd2a0485484d",
    ];

    // This function tests SHAKE128 against test vectors from the NIST Web
    // site.
    #[test]
    fn shake256() {
        inner_shake::<256>(&KAT_SHAKE256);
    }

    static KAT_SHAKE256: [&str; 6] = [
        // Each pair of strings is: input (hex), output (hex)
        "",
        "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762fd75dc4ddd8c0f200cb05019d67b592f6fc821c49479ab48640292eacb3b7c4be",

        "dc5a100fa16df1583c79722a0d72833d3bf22c109b8889dbd35213c6bfce205813edae3242695cfd9f59b9a1c203c1b72ef1a5423147cb990b5316a85266675894e2644c3f9578cebe451a09e58c53788fe77a9e850943f8a275f830354b0593a762bac55e984db3e0661eca3cb83f67a6fb348e6177f7dee2df40c4322602f094953905681be3954fe44c4c902c8f6bba565a788b38f13411ba76ce0f9f6756a2a2687424c5435a51e62df7a8934b6e141f74c6ccf539e3782d22b5955d3baf1ab2cf7b5c3f74ec2f9447344e937957fd7f0bdfec56d5d25f61cde18c0986e244ecf780d6307e313117256948d4230ebb9ea62bb302cfe80d7dfebabc4a51d7687967ed5b416a139e974c005fff507a96",
        "2bac5716803a9cda8f9e84365ab0a681327b5ba34fdedfb1c12e6e807f45284b",

        "8d8001e2c096f1b88e7c9224a086efd4797fbf74a8033a2d422a2b6b8f6747e4",
        "2e975f6a8a14f0704d51b13667d8195c219f71e6345696c49fa4b9d08e9225d3d39393425152c97e71dd24601c11abcfa0f12f53c680bd3ae757b8134a9c10d429615869217fdd5885c4db174985703a6d6de94a667eac3023443a8337ae1bc601b76d7d38ec3c34463105f0d3949d78e562a039e4469548b609395de5a4fd43c46ca9fd6ee29ada5efc07d84d553249450dab4a49c483ded250c9338f85cd937ae66bb436f3b4026e859fda1ca571432f3bfc09e7c03ca4d183b741111ca0483d0edabc03feb23b17ee48e844ba2408d9dcfd0139d2e8c7310125aee801c61ab7900d1efc47c078281766f361c5e6111346235e1dc38325666c",
    ];

    // SHA3 test vectors from:
    //    https://csrc.nist.gov/Projects/cryptographic-algorithm-validation-program/Secure-Hashing
    // (file sha-3bytetestvectors.zip).
    //
    // Each entry is two strings: message, and output. Both are hexadecimal.
    const KAT_SHA3: [&str; 930] = [
        // SHA3-224.
        "",
        "6b4e03423667dbb73b6e15454f0eb1abd4597f9a1b078e3f5b5a6bc7",

        "01",
        "488286d9d32716e5881ea1ee51f36d3660d70f0db03b3f612ce9eda4",

        "69cb",
        "94bd25c4cf6ca889126df37ddd9c36e6a9b28a4fe15cc3da6debcdd7",

        "bf5831",
        "1bb36bebde5f3cb6d8e4672acf6eec8728f31a54dacc2560da2a00cc",

        "d148ce6d",
        "0b521dac1efe292e20dfb585c8bff481899df72d59983315958391ba",

        "91c71068f8",
        "989f017709f50bd0230623c417f3daf194507f7b90a11127ba1638fa",

        "e7183e4d89c9",
        "650618f3b945c07de85b8478d69609647d5e2a432c6b15fbb3db91e4",

        "d85e470a7c6988",
        "8a134c33c7abd673cd3d0c33956700760de980c5aee74c96e6ba08b2",

        "e4ea2c16366b80d6",
        "7dd1a8e3ffe8c99cc547a69af14bd63b15ac26bd3d36b8a99513e89e",

        "b29373f6f8839bd498",
        "e02a13fa4770f824bcd69799284878f19bfdc833ac6d865f28b757d0",

        "49ec72c29b63036dbecd",
        "47cab44618f62dd431ccb13b3b9cd985d816c5d6026afc38a281aa00",

        "502f4e28a6feb4c6a1cc47",
        "bbe61d85b4cae716329e2bcc4038e282b4d7836eb846228835f65308",

        "e723c64b2258b5124f88405f",
        "d09da094cfefaad46b7b335830a9305570f4f4afe79f8629ff9d0c3d",

        "0d512eceb74d8a047531c1f716",
        "29ae0744051e55167176317eb17850a22939d8d94ebb0a90b6d98fde",

        "3b9ab76a23ae56340b5f4b80e1f3",
        "c0903be96f38051cfc2a5ad256aa0b8332217f450eab904ee84b6541",

        "e9fef751a20297ad1938662d131e7a",
        "48eba36dfe0575597d13ca26133267199dae76d63d1b9e9612720d08",

        "2bbb42b920b7feb4e3962a1552cc390f",
        "0dfa61f6b439bf8e3a6f378fe30a4134e8b2dfb652997a2a76c2789f",

        "2254e100bde9295093565a94877c21d05a",
        "6965256463276dbb26ad34a378c4bacaeae79d700283b188d44d73eb",

        "784ef7adecbb9a4cb5ac1df8513d87ae9772",
        "e918a5d52a0d42ab8ba2ea386eb6ad83cb8dd9a6bd461506be356ead",

        "f4e68964f784fe5c4d0e00bb4622042fa7048e",
        "765f050c95ae3347cf3f4f5032b428faeab13694e8c7798eafb82475",

        "a9ca7ec7aaf89db352fecba646ff73efe8e4a7e8",
        "65d6a49739c0e287584ff9d1f3463ce2e555ae9678147e21b5889e98",

        "b2f7018581a4e459cf9b9d9816fc17903ba8033f13",
        "c6837f12227bfbd86ccfe794053ce3a54052c8ca8430f526fd64b5f2",

        "f50086b4dc7bca0baec0076a878dd89571d52e47855b",
        "e39aa96fad581961bda032ed33dce36defde958baf9bae5dc558cf89",

        "6e6ef963f5000d0b91b0ad537ddc9697f8db8f10a3d5ee",
        "66dcb292b4d6bb4cdd4099b8e7bfea9658680c92c51562c091577056",

        "12a7b1a73b0b26a66362ec2a91ea5ff11af49a7a148a8cc5",
        "6fc91ec8ad448173f591b865ed3eb89115a278003376523c00e22f2a",

        "8a4768add4a9bd7b3f27461220ceae0218cf3322f4d2a980d1",
        "9a88bc64e743f2acaa1670cca7e201a299e1cce6df7015b0d2535213",

        "5c5b8c1902c8608c204e72a813e2b625021b3182c48b00f7fe4f",
        "31802a0fa9ae7ae88626604ad9ae41381d6f7c3c90effcfcf70efcf7",

        "e89e5cf07afb4a58ebeee17ff596d90b3274ba348f14f284fff025",
        "3bc9b7973f55735b612ddee8cc7907a3f1429b06df7cb1293b989802",

        "eb9e1143782a0f9fa815261c2adc2758fb1d88ffe40a0ae144189a48",
        "9d70d22520094a113297a192ead33e316924fdc7a2a9f8ea7098b84b",

        "c4ba3bff885fb78357221a9a903bc7ebd11c771faf5789e5aabc993a5f",
        "7b0212b4ee0b14dba62c2db7a765ac56db46e0b06eb744ee35726ddd",

        "07810e6b785177e52d0feac0394f3ecc41f35aa08ff1ed8162575f85888c",
        "b413d6f0cce14b7a1044a14bb2803d53bef907093769a5aa63a8e316",

        "01c742dc9ab0b05df925d4a351e38bea7ca7ad783594e22487d5b8198583f3",
        "c42c707ddc7b630939544adbdbe567a333ac88c3b5e738dee8f862be",

        "dd0f85b55fdf56ba254e06f8c2b650cc6b86bf28a14d714011141a86b8f14bd9",
        "0fe92469297c2c34911eae424710db6d312047898b9756edc5c2deb2",

        "ddf48f4cdc856c448326092dcf6bfc4ebcf4b36fc2e516eba0956807588b6e827b",
        "6cd83ba70e1bd387d603ab14c9fdcbf9862d2ebf0987215f011abee8",

        "c34d1f8729663569569f87b1fd6e0b954ae2e3b723d6c9fcae6ab09b13b4a87483b2",
        "e57e1d24dbd9a30ab311291f5d6a95530caa029c421dde0b487a577e",

        "808de7cbf8d831ad4f17eb58031daed38bdab82f467f87c6b2e3a7c5de25c8e8229413",
        "b3c13f11227f4386afdcf7663a120990f27da205ffb9bf83676f86dc",

        "5204a0a63707bd1cab67a8797994a052ee73884b325fdf37d86ef280b3f550c9eb4e7fd3",
        "6aa1060f84127bf2c988230a907242e7d6972a01c6772ba0f7b8bc86",

        "da9439bd090dfc2eccc1203a7a82c5d6467fec4e5b0a2b2c2b9ea65b03203a8ce365fbd98e",
        "e8f0929f1f6209d41185292d35ebbf5a3bfe5492713b06d56579458d",

        "668bbd38c0ad0881a7f095157d00f29b576b01ba54a8f1392e586c640ecb12b2a5c627a67884",
        "75dd056962c5bb5d6f616a9f57892992946d048df57c0a36a40a365a",

        "d63ac3bcfee3a5bc503cf20fe8ff496bf7a8064769870c8fc514c29b55825b6288975beb94ba56",
        "c694da941a7a506cef471fdffb5230bb6c3cd2715341033ab7268e9b",

        "985f06121aed603171020badc2075fd33256d67d40430839575ddaa7a3f1f22325d06ea40252d5e4",
        "29f8846aaf234281b515ea1d45674535a6126c38bd959c1995cad7c9",

        "8783849552be4540cb24d67996a10d16444b2d936d2fa5fcff51fb0dd5ee03998c0454289215fce47f",
        "84502256e3f4291ef4d15e8705e579951fc0e39a2d58fda74852551f",

        "dab31c7b3f40825aac13f6772771b7e7fbc09fedf6eff778d51190ecfd4b0f256cf189baeeec507e945f",
        "97168a9c3b07ec4987a4cf1f2478731fc674f56a2caeef074590ed6b",

        "1119b962bed5815734af7827ec536701a494ac5d4ab83eea1b16ecc80ce4e5f8694a7d11bcba2e34f084dd",
        "205d89e032f03c8519cf43b720478389b1788f3522c3d347febd2c70",

        "d2c45e2c1fa0c44efc84e6c0654cc0d867a3e33733c725aa718d974ed6a4b7f8f91de7d3622b1e4be428de2a",
        "d483e39b7add050eb4a793e54c85b250746e382399c74736f33da890",

        "a873b148fe1807b89cbed930a7802abad6ca0442340e62ed21b84ead9a634713bb4de5648208c0eed6738d9cc8",
        "c86bcc12a6ab792c149aa83a6783ca8bb52b0ca4b2c12661c0a25d22",

        "b3008f6f567d1eed9ab5b3bbce824d290e66f66bcfcff7f9b8994835b4d54a4e45c9b8651b37dbefe5e3fe5b674f",
        "23929753ad07e8476e7bdac8a0ca39e9aac158132653be10ebeeb50c",

        "78d073b4e13f6850dc1ca36683abac72336465d790eb3575c942667d1e3ecc849f37a8d73604cb0fe726ffe55744a2",
        "6229233fc655ea48bb5b48b73a081897d855f6cf10478228fc305842",

        "45325b80e043c0cdce3ec421ecda529481910c09730128b4bb927dda1659ddd8fd3ca667d857941e6f9fd939a1c57098",
        "776aa1f54e038f390491a5d69bde7a2dbcba97c35574ebe60c9a772f",

        "3bdd6821d938fac52101fbee5d6ba191fb3b6cb634dbf42cebaae57bd897481ae5ee04e2d871a4c333ab5ab6588144f2f1",
        "62f8f3baea6dcf5af25d53ddfdac0bdcde88e3895df567c6c416a541",

        "86fc66f2618c98fe9efa1e3ac04e340385dc2b746cbc0f7c757b88342810fe70d81200952928e7aad0c0b6b19a044537b009",
        "20a21eb1d3130a4519ce6abd5ab6817081ae1bef3603056476a00e41",

        "f2a6168e7f92d313fc30f9e6f825a480916216f02e0308db70773ec165e25e81ffbf0220c5ca0cc6c91d3a09da99fa6efa877f",
        "5d6e5c82574f5e5c0339d3af1f9c28e17bcddc306a15187aff5d3dd7",

        "5e3b6b75b54f21b8016effb39276f5e7f493117ac4c0f2dec38a80ae2917dad83c68900120db1325f1f4697e0f5c25a8b92a9702",
        "5dc2147f1cf655dabb5ca4b2970b4564eb19ec456e6f966bbae19762",

        "e7f17c131950c06311f47799a0f5a6b4996f4cc890334450e1bd6cc6f5670771c0dc607f8eceb15300ec4220510ed5b7deb3429de6",
        "4ce80dab9f933112a3fd78c1f76434b197806eddfe35cb0bdd845c15",

        "c9aa3d0f6d878db11235e7b028f8d67e2ce26eee718f308e21132e377e3170e26ece95bd37a4bd7f873ba7f8b71517ec50297b21cf94",
        "5963b41b13925a90c9e8fbcded9a82ade8aae36dee920199f6d6ac7f",

        "0f170afafcefdfa8b0de328dab30b4e44d98d6aea2bc39557ff4658fce4fbf8526d8b5359f173c14e4da7cf88935c9369fc7d607863f25",
        "fe7e59028c7855c37ae3dc5ee324864cfee6b8bccc2c3b5a410b65d9",

        "6b2b92584146a433bee8b947cc1f35b617b73f5b1e0376ac8bdadfe5bfdf2263b205f74dfa53db7a29e5078f5c34a268119736ba390961f6",
        "132cfa7e71fe0991abbd88ef588ac95ac9289b1d775b42033567dd33",

        "39f7a94312bea1b4fa989f5a6775df538f01704120838c4a3104256478b5c0cfbe8b86e2912c980b390ea412edddb69d461e50f9f313bc17af",
        "fcc59655b8fec1a3d878345df9108bd99f4dd0e5218a55fc335e57f7",

        "ac582b5a4bb0c5e9c40d8f277bda9de3d07fff01e820a1cdaf88708f1d60be60b9a5e83b5c593657387802b4182d1df4e9466e6d7ae6dc7c8079",
        "5c2e10fae8f4304cd9361690e5d2c4cd15f10a7b14ea60208739579b",

        "072753981998453438a520d9de2d5704292910148b8f794ec3765b240c7af1b79462fa9a2f000dd94d592d3a2a069dc244daf57b12c57675f3f89b",
        "b0d290a6ebdd950811a2715f354b0d8935cb610a471cfc5dff5e0660",

        "66a9a6d0a322ed2852378af82c0a2c027b1082098ab750925a4dc2e8961d0062c9db02e8cf42a6b48afb0056d6c1f1fbbec3fbeef049535f6e9b3864",
        "d683488c8420eb2d61e528ab0a7b73aa780a085b9c7982293b2ac6ad",

        "18419a8498d4e9bfaa911748186c5753d5da5aa033371ffc56650d0ae9b73f430f0d1f3c9d40362786c0429d977b899b64016eca82e64203f6685c12ee",
        "51d0cd33fd6579b05c366c6fcc653638b7b13b62798b99b36792cdc4",

        "4fc52009d58a0fc2573e83fa335b5c1df8c14b2e6daaf05bd6e13fd5722f28de4816772424c2f94ddc3de0d3d7e26812d014bb9fd83012dc9abf1ec9e3f1",
        "630ee2beaf1c1592eaa6263fc562a260b6054e9eab1aa19536fda170",

        "acdaa28692f334732088f5efab2c7951fe0f845b9e2c6f1253c3cdcde30a4e8d2120e38c26422219df41eda2c8334e13f669a65f5ba2075b467eded32936d5",
        "7d4991d54c78af5809cd17024cadae783c6f5a1f0feb365b532580c2",

        "d1593cd338b7a25bb5413f112a639fe31c981e505c81a820e638c25209e2ce56c8838a7c8117dbadccdec959a6f7cab0cf304315701d4ccf0167b4026a6744de",
        "84e18330723e4f90520d0b051a9bf9bd7b5c7ec0177803f15cf740e5",

        "8cf8ea25310126ae1fdce3c9195395a9d45051a2a3f08ce154d8265b54cca7031a7ec840c3a3359efa4c91c41b74baa698d54ffb9b0170f2edadc5201650c2bdc6",
        "75de14169d16a9902f6e8a3359d94594a889c4aed9246caa6cf5612c",

        "e0320fee19af5bfd511a23cabba75acb0815525a3734305aafa49c1d8bdfbd853579646a36a7873c4cfff2eabd7e3902eccff1192aca1f6dce3cf1c988e6aca9f2c8",
        "d7f2018c303ee045de4b8cdefcfb5395674e3a8770d65f0757b4cd5e",

        "1a424ecce1a82c47742171a701ad6e0ff1a762ce26f8e332818a7fa1a800a4e506a4bdc813a09ee1d57222ada79a12e2399549ffd80f1628ef55e231ce0913f9ab1930",
        "277f96fca5d9ab055fae5d4dd10cc49c2237bd38d95bd8dbd168ec21",

        "af172809570cc306333c25523f863c6d0e0154c55e404722f0d4ed419713dabf8e18493a0e0b53b220a36535b1e8f0bbe43e624fac9f566f992807b6f2d70bb805933e2e",
        "9581170093600cb67063a314d8decf109ff9368ffbc90ea2d3250577",

        "a62f4b43250cdf3f43c1da439bc5e4224b15185b60d615e38e3c512425aab145401b57ac3fc0bcc178eafef52a2b7b04b2b89e760212f96c4ee694990831858f0fa7c13c24",
        "a0f5775a2d001a66f0882ce1415261994021988690840c6b4a3470c8",

        "fcf81c93f917bb06f278f48826ef9ca8ba99ac8f00129fd9f8e81ca31750d5e54818af0331dd239eb77ee4b0c4d0c2d84794cef27da6bfeb707794d3bdbc7b349968f2a316d8",
        "a97a74fb01fec5caf3477220eef6e7c36d0ba4199ddc755f7ccf94ee",

        "e61d24b500581734c29902ade4c5035c090868df9f24bb330609fcdff4a72d6f18001424fd813cea32923d8aa86c3d215b2ab7d134237bb62e78f61cb9e9b4ef5ced23729d019a",
        "40758314f1abbd43e0bc9c73a1c7e24719d56eebcd967b39d355e978",

        "37b14f04233dfb4da5e5bd1852f77c41e25c4926936fe414c8108200f6f3cd78c03e2dd9615446c14bebc2c70d65506a7a5dec4808806291769e0dbab200e576f9fdb9e240c8b8ff",
        "2d36af0dd95619a96c5664d8987bbb82d183466ff44151034fed687b",

        "45efb0a3d8fb7bb683913459727e8756d67959cfdd4f5b80e13ddf45e09debdc2cc68ceb632d6d45a2d0a869f6d4dc4c136c805849fe77b4b381e4c6b22a3ff69947a9b5aa6b7cbe42",
        "125e983229f65bf01b59a9b619810a88f1c53b4c3b1960b52a205d99",

        "9b6c3c77746219dd88976966c68ead59eb62aa3cf6647798dc06d4fc7ef8bd44d8903f1b7b6f8bbf3d6249052f862e9ccfb0d1957f0bba233603bca0766286d17eb9746bc002abd69583",
        "762629518833ba68333fc3e3b4d482c60b4e0e828872826b68313315",

        "9f452f900219017199edfc5d7d86a162d9750bba4cec77428ed1032e5711b6fb7c37c1a65b3d041c7aa1d4f16bbcfc54f35001436b60abfb6544c0b393fc1389e5c5bdbdf2eaab1d99dd59",
        "19b432f5c38f665441d36c472d386008a5bbd82aa4eabeaabe3d28cc",

        "cbfd186592fa68dc3a21d62db1ba55121f58fecb11695859d70bd7ed2a21a2a013a699640842973b571bf4a7c8ee4f617d5e8a4d1e8c15ae33e77097d146eba27934b1e33d8a041f2444ca3a",
        "b32ad13ba4a0b9fc1aa9a1a57bdbfbebdfab71cf5a16e06040f75787",

        "173225324c6c350ddba227b89a651e576d1ab6a96895453c33ea61ddb37fa253e666a84d0fea609814688495246161eb9cccdd792cb1b88f36f3125d766e2eabe84175cbe66dbecc91a0ccf173",
        "fc8feecaefffdaa966e9536b91dfc85ea5113a01d6b320677d727a7d",

        "6999f398407480cd43bafdaedb8624d9ba0972aa5a2f3504a67fe54ef744b7bb41ea70cf8faa771fac6a2f5823de83826af4c3865b6faeeee3d1d0edfe7f0e9fe3207f917b467d841850fc6e648f",
        "e7abcb4c0f218814ecf45fbf28a3f286d90c5e740aafd1647437c1e1",

        "2727eeb1d51098c69fd8141d78f21275b2bb949e7115fd3860526bbda25547c20cf31b79919fa37bfd4726c4e77906ffe0ca9705f1782da0454e799422c815e01e785d418fa881f84341d8cd71ec77",
        "2be332c873ed4fb70bc1916c76bef2cd3385e674b83aa1ee8ad28a01",

        "1f48a5b401d88e6cbe37f3f634d55462865f7cde7990052a1e4a1e4cb2e58c84c2c7ef82923447d7c068b6aa25e388acfc05704e46da14316d37ccdd2706a7b79ddeb02dcdd76f342c9cb2f490c18dc1",
        "448b70f575a8a1eb74030a985e9c504d4eaf6b1814e1146f782c9af5",

        "6dce9a9ecb48b9da8aef51a89e7f7fc1a6a78966b7bac0ac5ba7ab18d92b616bb74537bf7eeb9bd3bdfb40a450747c3de2e6eecfb12763049148fa9134c7870ba80636fb21fc7134f92b0364f5d27deaca",
        "df855d544e17f01125022bc18e9ffced12f3cd39674e68184657ec7e",

        "d498b6901345afddc5aa50cac77f7f794d7929eed571d95b59c289a0c9f3b812b896bc7b566f5a639ed9948ed066c2c622c6e4dbb2ea37e7c06806d61a22c326d72356ec48c9b5182c29b5f923af20046605",
        "5b225c29e4547777a2c6a1a2bbe9da2dc6a8c6d0d01d8d8022988be2",

        "e958b80489aa6a38526244da165dc4464e7961e457f763abdb23f7e48d368331197b37cd5ab1e515ceb1124848504d8be587bf3041d10437ebd53915164556b59106bebdf99115122d99529e02ee155138a13a",
        "364a988400424557a9c60e4e1f32f0855a3383c90b007d30ee3ec333",

        "f33ba982bc2c3308f948a1b64c7fb68fb891bc05fa18781b1dc95dc749f7009adc58cca2bb0cf790ebdbb4165bbfab9304a2a6f234688dcf273094dcd8d7b38416be57cedace5783d8b92993548256b5373f2b4e",
        "ca37e52f2843a0f65692c5aeed0169601da3275dfb3ee6d81b467f60",

        "8b5d77a906c7ec7563af7551a796e5d5dcf02c42121d7b13a49aa9d4bc79d637190e4e6510ecaf92d1104fd4ec5bd8351446350722d1b2775dbc5e65f8fab473dc637b5ca8a9eb88f68d11dde15275d7c472f9db43",
        "9337537de482f0cf88cad6b86e195a1e422e59cc60d41d0eca8b0091",

        "3a564a84c2b48ee26da138ce2d1ae3c7933bcd65e40288406e56f30d1c48690a4998389dd27b55376f9b4e7f43607fadb16e8933726f00a3e41264cda553532761fefc73e86ed79b849b94e0895451332dc80fe39a4b",
        "88eab3e16ca8da5716542bae3c7c736b541c896199b2cb941213767b",

        "618a53989ffbbf54a76f01f9b87772491d87c8f25c58eb11b18a04f5ba8ed62574c351a466df64731c911458d765cbde83e7f29de90bc1bb26cc56b35c140555a7dcf00f5394d76a4cc531d7d5f57bac7dcbd06a4f73ba",
        "4a727cc6b4bd93d5ff2ecb81ab5057dfdcbe3e0c49436a58b9ff3ef2",

        "31857bb4e82497b526e426de6920a6063d02264d5249feffd14abdbbf03563d4c59ad1f7572c7d0efbc46a65dea9580bde0e387c9edce27cd9b20a46f62a70e6dd5f58e40aac3a22dfb6ba073facdadd58cd6f78c02bd219",
        "9e614fc139645e158cd1b216e2623e586242af64f8483e6fca20ed4b",

        "14859008c83f2831be4d6e54b781b9fb61dadc40c459a93ede11b4c78a7e5a55a71701427526a03b42d883f247904813cd812e7a947c8fa37406aa6145aea6d3fd9ed494186f35333d423ce31e0cd473a031a5803c5593e9a4",
        "545fafa43afcaf38063d8a312c3a27e0d74bff957f8ef4d51cb29698",

        "267a14bad702ef0a8468b31c72715f0533f6b97e6e943839dea420719d6defc5a399f84689e64ecf931ee395ee49f1fe362199b73cc6cb0105b3654b16f19f06ee8aa6b5d5418743d4804f9a059270710d126765e6a49c4ce2e3",
        "9b9360a5c747e6e1288f6f9d971051ffd84641f6d64e0a4b5142e4ec",

        "6c98a8eb3ea4451401e0424c10cb722683b23f75ae254d62eba75abb9aa9698e65ba1ff7c9f86d36d1ca6f0425d19428441b00450e9a2ef685d5da1cd4de1e779184db743fc95a461797333808ae6e42fce1e9da5d82f90cd71b54",
        "0c6f33f9534fc52f3700f37b9ee678b4c5c8a90b1a2eb1574002e377",

        "4bae62a008d9fdba351a1903c66d58e587361990f7c9eea05a2f51f90a2892f60e6c14c4ed36b908c4039bc89797fd88e54281b37f619b3d9a274587229ef48351e8cb1881cb0fc83e6ddc90a05b160fd7d0a1eb0835d57158e42c7b",
        "989c156ba1fd1f70deb378e46ffcbf6f2cf9cf977a92ac51643c97b4",

        "83ca6d4ebdf1c04062ca1abb977670ef9bcc889906935fd64ff4c739912e541b8f8c7932f595ef66e18256dfa1f51f63bfe7a9df3ae2aa431771d19318d6aa3bccfac1a4c8aa0a0433ff807a881e0d5a9722aac6cd57c77eb6a9edf8c0",
        "fb831f2456595fabee9d458625283a80bb4f8f031e9abdbf48b7b51e",

        "f4c7ad8d24ed5a682c473463e85391050c026fef0b0e6dca388e1a7e2bc872a46746a63f3a2c1ca6e4c8b7c5fb6b58850d77a58988ba091bd7fafb66ced184e548bcfb1b0e6e1485fb6a19cd5ed07640a0777b82273d5e80799b7fa7a57d",
        "13bee617474b3fc3447025f2a488dba8825d46a4e128b9a8bdeb1b85",

        "5f81c5aec92385bfdc55ebd600f23cb04ac9d5c7a1396f801ffea1a6b94aa617231761bdeebc9ec0f4bf9bfaf5ebc7ac82a2c96f1a74c46d94f0dad0bcb9ef7b41ddaff8cf63d2b278239e6558dbaed2797ef3b7f4cff8fe592f6a3551b3d7",
        "143a6f0a20d5b4dbc5df64a7e50f9985631453eb09ded71667709083",

        "0735cecaedef99bf4c53242f0552f49f56bbe589a2f611af75f4f3aec366cdd6702d46391512580202b869097fceb8a45889fbbf9852472f94bc2f432bb8309c4d0c4d3fba01f6e90c5c2ea3f890ed95d132c31f4dadbf268c378fac5604e8a4",
        "9f5e9f7429e5488a843c52ffb46ae2e84228919d32330a9193af3b21",

        "9b4e4df92e5152fe1ec56a9fc865f30bac7e949fc4f62f0b158d10b083636b4de9bb05db69fe31b50103fefc5f8daf3af7156b4552ca3667a9d720bbb2e4bcdabadfd4b7f4fc5bc811faa36710a9d17758a98d4a0474fec27e9ef5b74f5c689935",
        "487a6f2f875cb253de4cef18ecb4f2a54388ebaffbfc4259bdd97f09",

        "a61bef838867710ff4341b26b13b8d7af7e461ccd317b160cc4fdaaec7f1805a28ddd3663a4210a7d1b64a752e866aa7224a75bf77bd0d618bcc3b0a3eed6bfe0eb2b882819e6a4cc437bd38915ce53c55d94e9e9339286483dc230d0049777ea1c4",
        "e257bc45b62d0853ba4b0f8578698f4262c31a778cb6a6317b6e6d60",

        "c0bd79e0c5f72fcb1de6c234bdb67bd0d3f481b962a3a01f2d8c483bd7d5d98548d51d27532716b195fdfb0ea0b77db759b54e269e69e48e2cb07bc9c06259927d2755f48e8d9a020c58a9c9221a9d836f03b30eabf9099c8eeba6abed63bb38275b28",
        "92df7f848ada8a9698ddc2e7452ac8fc43cf83d2ca2cadd712c595f2",

        "77823af9b8796c63baebe7ba9dcde12c626b840ea04f42d878646970ca5bf7aba94eaf110da36ce0c834b654bcac93264a349f520e505f1ec903d3589e3a4adf82687a65ee6dd072d6bc05acdfbdf257cd70a5183a54b4fe8e87d1c22b2e9f4ee817c57d",
        "819a4340938497cd8b1def8444bb03f8429b9e87bad8000002d60b83",

        "ada5651b4e240335600940f207b98371f7e743988957bffe0de8ef0862d1ba52c52b6950e7b05c3542c2fb13acaff0442d33940a0e3ea67232f8437eaa02128283ffc0cfe254ac8f542be3f05fbe4e855dd22ae98a81b9a55b3d3753111210048f2b50e068",
        "b6177d179cf17eddcd8988c9108b42af9c41adcc5942c4d33b0f1be2",

        "ff4704bbbd719b011244ebedf2f2355338fcc7d64844c3a0f36a21569b55f74a9710f8f3d8d83b9bcd733f5885c32b3d149a5ad137d016c03b93a4d11aff8218e8eeec6d6d12a41d1441f3df040feb098ca2f003c4c277fc71300cdd2a399a7bb98ae711c446",
        "a1072b28f3453422e611421309aa49aaebba0273c72b835fdeea1132",

        "eae4b62f697cf0bf40a1c2c109143c1dde18e24f1c289aba67e5c83eef52b70cf1433bb98013949285969630054e074ca2e249d465cb383dba51561cbcb626f0b3b1d542db1e1ff168f371c7c6764b4f25ade9eb351622212e99903614bbf1fe3914cdf203035a",
        "f5273e4d0bf9779a0975fee23c447b3abb1cd17c34c723d62f3a2fd1",

        "0e39e0e6933c6104984fffe115dd8cde77edfee495480aa5e5def424f066a5770345fecb28b16caa5416bc79e2b83145409bd4bfe9a00c8493f06ea2a99dd658fb87b71eb57dafe58da55fa0411e790341e31a8ba8f35bbe71af23b4e8833fd65ec8b4e621e95340",
        "62fb7d6b3810d0fd7d96b4ff5efe7bd283ddbbeda4a21a62f985a3dc",

        "e32bea9ab02de7d893ecb7857ba66df2c35ed258123065ca80e2a067fabb7dd4e79839ea0b3c58abace8e97bf42b0b8d97fcb09bb606a1da0243c32d24cc98985df008f8698362f2aa789e2a82b3e5b5011853d0c0e8fbd20c4d2b5f4733f2df8c5ae02e92a90d95d3",
        "278e06fd12a3e314f60d59a323673ba0a22003e42ac48e1cd04a70d0",

        "4157752d3d175a4bc1334fd42c204111728e7059659dcedf334ea7ce30378798d67c598a0afacca5a1c5fba923d54c72cffc9887df1b8df10d96514955056815fd2dd855d32e8b58b6fdf4d45715f636416a0137179f7eb01d786daffa924ccabd523bb31d1b5f0d05c4",
        "1cab43635d501e43ac42beee263755b9a29827e2a18b21d7be42e447",

        "2df12d8c256cd1a127e525ac3763e30c895982eee67ab7c150ce3deae906d2b9110d829ccfdf2793729e31e478e3a310ae525e059971a29515bad2273cee77ad89ad88d63d44e98402c63180cf5eb06d0be3b1faf5adfc5c43a79ffc09a6ee6cddf9c9a039421d5b2184ad",
        "ee60f0d01008cface49af2ee5780ccdee37404c37642008a55fafaf2",

        "03be6940e859f9b072660dff28a187551c2425481dd0555d2dee4acc36164f84f8505b6f467ae6f772eafcc9065490d9b4ed12a690d044bf7da14986e571fe34aee28e1d698c4136cc9f95d462c990b6815a54467da6f41c1baa86c448f37ac10bbc2ad1b957b17368ce01a7",
        "a8aa80d4c925889b58eff41b89682b92bea60c1c3995043dac312d2d",

        "0baf1ac243c1f34ca5e00aed4d867f967bc2b963e93956c35b6b68da7737de23d7a1405a5dd4a099c663cdc182d4c91bc35f7d3fd5f3ac35ad7a26dbc45e3e86264c7decc538984214a1a0a1d11679ae22f98d7ae483c1a74008a9cd7f7cf71b1f373a4226f5c58eb621ec56e2",
        "f12f7a1c5c1c383a2a5fff8932e2ae9dc342b37652d47356ffc1cb37",

        "3c29a8c83e48194a7b87b69e376a06063de2449bd171fa91e58ed2bc904ba853bb35e3f51e7c06e96b5482aac89acfa383bbba3701d20104f8101d69de615f45a24c3e02991bf0d3bb3d37390fe87ecc64032438424218862093a69dd7b99008573661f9996ffe8ed50b7e54f49c",
        "5c6b29c3cbfd1d2eadf7c791513b27f21c934de6378ef748b779b71d",

        "68a3c06e0740b569c72ea6a90d8b45e83c7c350d2bcf1cf6d6dffa7553b8b998087c052e1c065d862bcc6a7a3e0a90acfa1dc410172c9dab140ead9a296811557e1647359acd40341efeb6f5b3fdc0044162a45e62b0ec341634bcecb830626930392f8c6bde85fa088a322054acfc",
        "58a691524398a5746df28ac083f15861750e0cdd1fd5e5f57c982c18",

        "d4f757d1c33b9c0b38b4e93e8e2483ec51b4861299f1d650961457496d86614d42a36e3696bf168fd4663efc26e88cd58d151e1531467b73f69dc9ce4f8d41ce579ce1c91e6760e340e7677abdf4fec1040745aa5144640a39b8c4f884df80753a691653003d634fa5bfce81f94ec3f6",
        "be11259377f09821d9dc358592b6565d8ef2b414dfaa7db5609fb751",

        "ecd9e95f7c5efc8336f80fe67e113657b31482bafc22dc5b45073482846cdc48414d2ea855ae75d9f28a0bdbe30dbe511503788e578f20f25e20bb770ca1d787f2f02911139275dbeaa5ae1aaf155f40d7134915dac34d0938358dc8be97cf1005a922bf3d71c331282f41c86993e0ccff",
        "6950ad0f91398b39965b1859ea918c531212face1e51d4d390f094e1",

        "834ddd8fc7ea0c3385ef8280d3a7b22d59ad17d710a51a544a293544f30659e816a98d38a2d4d92f6f96626a7c79d6f17bfd0a558f45e2fb541172b720ec629c88a7971326050f2b9ab80d30cf8c777f80e37c98fa61797523e81e1bbbc7cd6ee22e4249dae679ce0f3eccfb54495d7e7046",
        "ef21ee8d568c009eaa8d1ea770968cb718c4d56e7b2d966bfcbbf398",

        "6ff611208395d81500505dae050ff0c29c0afde2a8e89c96192863ea62c17e292d0502e94dcb7f47f4cdd574264f48716d02d616cf27c759fdf787cdcd43b169ea586c8bca25fa3ce1a08eb615655e2471a0faa81d2edca28eff4030fabf36f10fb5f50fe4eb727c308f317bba995b6310ae12",
        "8a29f2c0d564935b8d31b7d007f58138489d140917a28ee85d43b6f2",

        "f977ea38076328bb0ee2297cbe3b2a9755fe8bb95ae726298e04df05201a7ccf2046b82836e092da94a4eb1c291450121718159468e8a330fc2b1272c661fb62397e874ffcd7cccbe5425af725791001c0c035ea41c8c48dabd206ddb217666e2b688237c2127e96eb049d941b34126b373e1345",
        "15180df5554387337f04de2f37a16b28125adbd02b6fa6cfdb24195d",

        "22a8fb43d54fff82749cdce98abe8adafcd443ffe16bf0e99341e1f7064fc07a5907c816abdb326c30fef0f5846e9e313f32b602c9e00352706358fcb7fb81eaf1857a7b0ffddf27b741a465961806ccf672c17993f284b2aaa9a2c854250a4212aa7937a9bfeefc30ec5f0067c3aaf34a1dce2ee6",
        "d11fcbbb2fa03109f952a56e16867c70904552eb580a6659314bd5fe",

        "68727636ff38c0ba8999dde3cbd9503900d5ccb01d3c9b7959fb411eedf95cce1805cef6670d1e1133901cc06b55c41d945e654c0d18035498d4f92d167ae21b927cba3a810a41594885a00bff354ffc753e368274d01374469f1b3f7793e436ddc0822ad698f13bd15fb3ed10e0b97fac5f8778d9ce",
        "21c71bd09ebf5d09155347c4f476b8f9c5aed4579573211887ab6084",

        "167cb772f096b2e3b1599cce3440d1af57c5b7df5d2f460b91acc7e52c9fdb19793bc0833751d09f3f664a4167095586a564420a7810125b832e38ae7bb3a0d14403ef6157c20d3d67e6e13a44115b19ff1fb8b64ffa018133b6d532d9da69b9bffbcd74189071a57101e7239401ea50ad1ea04aab961c",
        "c46cb2dfeb8b961e6e84d72e05111e04d62e3f93a055164b135b9072",

        "b88ff728c8f829841a14e56194bbf278d69f88317a81b4749aa5fdbc9383486e09bff96a2c5b5bdf392c4263438aef43334c33170ef4d89a76263cb9745f3fea74e35fbf91f722bb1351b56436cdd2992e61e6266753749611a9b449dce281c600e37251813446c1b16c858cf6ea6424cdc6e9860f07510f",
        "8891cdfe486a582e8340bd8b893996d7a4e547e3bf50551902e722f2",

        "520f27a4d096d4193d2bc0983cf83bbb5084845b41844800c1f5669b4f67f5785c9c886eac51b059005cc3caf2f7dcfc205c230a8c924f604386696f3d5dd2a68509879d991aa49314d7271a8a8ef711b42825d3cd0071ae3bf6109772bfac1b167fad995f99b7afc2c573f2ce6493e25411101dca79b6d2f1",
        "216ea50997596f71edc94ed96e2b686628640f94a3c64adef05c2b63",

        "75c23e556178f00440533bcd25257934d0c6f5e68a64f1aa511bee9435c5277b02145fae1fdedce3b6b7b47015c547be55d00dfa3999920d586dbecf7ff95a775160d057308b32c661c17e5d6a772166bf69b9919ee91fe93877a50711939c85a9cf1ab65c28fa94879623faece20e1458b8821383fda2253762",
        "d1631028a8e0ec4adc689cabba8bf681d11e2e2a5059f293f7ef5be3",

        "d23373b9405024d0c4b17aa503f7e2ff7d308083124ed2cbc4d990b9bee0d70b9635872fcfdaea58a2b696d1fd8c9492cd2ec11179ee755aae5663626219c0981348a8be50c9bdf77b061121cde246649af1f30bd7e84a93d952f8025f854d7bd3d59d0ecd07e6d4d909b23c7ae03fa06fe1de1c3424999fcc3618",
        "726f6584ff9ea998ff326c9f73291ace8726d8697e7aa94f1ed42f7e",

        "6f057f91480fecee8a7e3879dbf8c52040f96f5929c6b8b6aea223b91843ddeba387a2288264df3d241d14b5b6bc7defe9bcf174f5060a88de1f86fff59fed52a3e574f2620922dc0c12316e5869b779a18e8697ea0a50bf20a50f169ed8a308f785bd98efe6fdf4cac4574dcae9bbe5f3d7f56a11bad282fc9c84a7",
        "6b40e5c86db3d9c384c22a46cbef5f8e8c427bb6bf43268edd918aeb",

        "6f77874dcad9479f5bcac3763662cc30cb99823c5ff469dcbd64c028286b0e579580fd3a17b56b099b97bf62d555798f7a250e08b0e4f238c3fcf684198bd48a68c208a6268be2bb416eda3011b523388bce8357b7f26122640420461abcabcb5004519adfa2d43db718bce7d0c8f1b4645c89315c65df1f0842e57412",
        "0228626c63c20465d5139d1af0b9ce17e334ebe10a5eee2cafe96cb1",

        "ea841bd41b22e4c98b223332918eb791f51d1978540785f9c617675dbd02721831f7e7fdfa7714af7d671b588a64f49d8556b5d1c448116839771faf51a85dbb1bbff59fad8e3fe3c4eb8631aa050f505df85757ed9e9d1a26a8a0e96feeaa7af204cd23fd0e6d4ca8d5ff25b91a0f94c42a887297b230f6d5d57271e07c",
        "ff33c64231dedfc247e11e35aaf82d283a9ad62034102ee2bb5d4609",

        "7216a825029da1c9a9328d499b3ff98f6e18b8af368e2b19efc1c0121b35b965ab282f55232356d7fad002fe3f0b6ab7833b2cb6f2e392b0c37414cbd3661e538c8613ae0c9291928303f775dd2a2445a27e825a1a3544a9b411eb3aa87d0fdcdcd85c170511db620e747296bdc3afa39489c181f5abc76a8a404e47e4a214",
        "9440d3710b43e79899e116987366b2dd36b44b2f39e377fa2d4fe143",

        "44a8508a3c3976d563e933705be4dbeebc726304b511203df7c7d1efceb6e06e91f1e57f3d8e6c105dfdf8262d984816fe7ad8f8dc95ab596fff48301f8d03137ba37dabdc4a6e664583a26b8edc42d3c2405516c51386c33a7f2875a3087702ca6721f56195053fe5263a29c8d8538dce6ce146b8b43ae520ee79a5a450c6a2",
        "a2743d341023ff5f775d90185d3139a7756b0a65c19ee876ebeb92ae",

        "a8ef4107f41ebbc5799a716b6b50e87c19e976042afca7702682e0a2398b42453430d15ed5c9d62448608212ed65d33a5ca2bcdca7728037df2e5f9fd9e974d0315dde8290241e3e2b2cc06f8c653ebc95bc2195c24d690caed42fe7d96589f3a85eae9bad995ab829e674abcfb8efaacb1eee5703f52b979d5d99a1c1694855a0",
        "b411a28ff46513d0c3d63cf78a9b6353466cba3b926a8d895ee14fdd",

        "f649d801b4040b7b5152f58a01e7852f565efc77b5dafe4607eee953b0ba6774c5573f1c79767121d94381c3ba9013ebef2fb8b0bf9f081f96ecf13cfad04e44c11ebb358160a89049bfad5e8e241d71689ddeecff0278063fd86b0ad475c6a25265f556b30ddb50078e216267edcd4a2b7016345d4b76806d7b02c625f3f717e0f6",
        "b94debadc833d5706cd4736bb1dc75039827832ae408859e2e6a6941",

        "eb71b45a494e76462edf41a9fdcbb3f46fb863b9e259d0c8f4a79898516eebe8c90c3ea5a675440f3c7b1a18c14dc20c5f3dd27788c66d448acd73226327f52cd65cecc8beaa2acfa34d90ef8bfe824e12ba9870bdc4965b8ced9ff9ce13a5bd39e824893af410d08ade0cf802e7dc02b0b71d6c2a5c3356229084e53b3ae4e51b384f",
        "fbbec05ee1fb5f5cd1106ed7384850059cdcda474ba7cec0407a272b",

        "4eca0c51d30829b9a1d2712da1fac31f52942d77c9f20c2bf6d3751028d7d4f0d336d3dc92b27ec368caa4444b3180c1e37e98b58f25e647a9a6361f0b04cf78d17955766168eebaa993a435a88e0b39307423d6ead87f639afea75ba44bbc6bd0fb5ac84a12c2c6ed9539a7c0f9abb0c1dc9483e2f321a85244926dfd95e2f05624aa7a",
        "fe313eb74f955c0cbb1c446dd4ff853f32b3232d93faba7db6d1fab8",

        "97784d14db62a7f98f5ac3df742e013489ec0b8777b05ef82bba06edc5c3a807b191c65513ca3fc7690615e56c2773c036edef29aac50c2211e20392018fc33d83c436f274f7c6062c3420025e7037993f1b8cddebf4aeb20421fc829c7fb23255372455c69244a0210e6a9e13b155a5ec9d6d0900e54a8f4d9f7a255e3a7fd06f1218e5d1",
        "5504f39131773550b6f459f33a5b57a2ce60ce8bb78c574fef83dcf7",

        "1ee9047351e2a13e4a2d5a826e304fef82241fbab5100835e1f850a20e51e34938b93dc852e58aab8adb0c3ccf61be9c90b53713c77ed0a5370309e6f19b290f1d642550f738c36818ddff74f77cae04af55617403b08c7a9f17e8fba0c21523575384b44ac4949e7c9dfbd1ef6a684f666c67856f8f84dba19cb38a23b0efad6eed229c536f",
        "b8f253512dabf9d89d2080830f23da5893b0f87edc0bd624ea767f14",

        "1f363d2f7aa89e2b6c5e172f530d1a35531d0083a5acfcd232d64db06134b8232da2368f7a46ead9a9ce55cd6af8cdbdd1582b6bad56c52a15769c3f43dcd68da60f6e7232fd2aecfb3fcd00029f8e5c4ed7ca3b3f9cf68920dbd747fb43f532b1034d9f49d546aa893be68fc3084658f22343b9068877387b8f68903071fe5877083be068d626",
        "e59a19686df36bf5fe798a9565722b8e0bdd9f8eedbbb4a34a9ca7ab",

        "ecf5d9e29c1c04c11a9503cc223d0cee4866fa26df2b4f7c1a017939718f545746c0f137c9169692194105b2acf001e2f0e70f2332517a20c05899644af454cb8e00e5363593dc83f78d66bd0670ce8faa7244ff28d0de59e964dc68d87a30ec0ce03e49a73ce07dfea2ad54fa667bdfbe2f2222894d830dde4dc9aee3caefa4088683d7e8b9a966",
        "a886eb94f15df208be122912d4edf02561482278a9f847ddc91c9bd2",

        "9f44357664b5e3a958780641cca52049f3b49f07484b5f762a5571f7c9541b4346f81fa416f04065a80003864754b3b54114a77a4938c8b21a9e4d3e5d59c9fccd4d68f699f975da099320ab655a7fb51328d2c6ff460b9b40858e99f88a35be7b6a97d6b4778af2c559e616ee608c32b018a753321e321be333bb6f618f666f9a7734ab3112859323",
        "8839f755eee84e15c586b52e29a41ddc640ac432cf31370680987a44",

        "c1aa1266f223c148bfa3d0ab29f278334d8fcbfbf0f4ebef5c1b7a766b415155e1ea75d0fe2546115411faced7a04a27339b6bcd62e740697d06ce3cd2e0f00238c44c1d9faa85efebbbb3880313108124c5f3277c1f03ddf430a4bb4d88b67b6e3f7f96fc39e5aa2ca7e11fd5d1300aca144c5166269a1168a2e53c01c00b872c63f6833e5ace09bedf",
        "439e3c7a0d655a30a9749afdefb7e048814335849df76d526c287727",

        "0a367d3789827ccd4bef5fe8eb78c20503241f07fb8c41d81e97fb53f3891962ca3c976395ac11d1f9ba7b20a52912e8e3ed92466ca5aa808166ade737ba8a0213e8fee8d67608ee9aed9e821edc9e575f1f07c3686169656ae09a0a0f70abd10cc31a8ef6e7496d56102fd8ff984e9a9f44e54495c966cf028f2a8423b46419de54541d9a08bd9654ac98",
        "40318036a595630e4135f10703be1d759a6c7e5146e0fc82abeba184",

        "8a05b00ae2d5f652f02f98a1b035003f8fa7ba1b17fc3778cdb1cae35ae1f768ea16ed05d25f515f75a23db468348911d4a749c51ce39615c07892318233a667c7f00e973fae98e7c8e9a8b7902480d87ac5bef8c4252661e6e8a2e4bd8a870fe83b1aa773ed5352b2abe193702c6dfb4aa8239e55ea6fc507a704e2540e23c917a01a1cb4420b07fb90ee2e",
        "9a26f054e57aea14242d7801f3d61ddca1523b738fc26fecfa5d9a6a",

        "ba6442c6d2139201dfef32c1ffb0ce92dd64091bd507c250595395e993d9a5124b5199640c2fe51482774b6a27d1a1751fe0d4fe5fd02dba152ed3c344fd9249af06da85f96f0bef0a8fefb1b501885b97f70dd842d12fa19befa03080c3d6b8ae2a0d13e2fc8bfc3fe1277ef0670cac0e52bb93c4344f6db13d05188d53fbc6106538f50ffdeda2e915fab921",
        "58470da58476bcb89450c521fc396c6dc51b9fb6465c979aba5f8eb4",

        "96fdb76f83bf12b3f4f322bf613fc38b2c8e0678856230418b6b062fb358488d6eed7c5c0656ec48c9bbf2da6a1473eea43faa68204f27239928172a3e49c52b58e861282c4401702337e5ce280aff00528eb26ac368db0cd0ad0eb262af226a9b16ef3bbd325614488f820363ca6ea77da4a7e8345554e57623732ee6326534819eadfe81c7f51d81ec51e1e3fc",
        "be92d4a6946de0e93d5bbe420651a8befb97cbdb5d63b22aaecf453d",

        "0eef947f1e4f01cdb5481ca6eaa25f2caca4c401612888fecef52e283748c8dfc7b47259322c1f4f985f98f6ad44c13117f51e0517c0974d6c7b78af7419bcce957b8bc1db8801c5e280312ef78d6aa47a9cb98b866aaec3d5e26392dda6bbde3fece8a0628b30955b55f03711a8e1eb9e409a7cf84f56c8d0d0f8b9ba184c778fae90dc0f5c3329cb86dcf743bbae",
        "98ec52c21cb988b1434b1653dd4ac806d118de6af1bb471c16577c34",

        "e65de91fdcb7606f14dbcfc94c9c94a57240a6b2c31ed410346c4dc011526559e44296fc988cc589de2dc713d0e82492d4991bd8c4c5e6c74c753fc09345225e1db8d565f0ce26f5f5d9f404a28cf00bd655a5fe04edb682942d675b86235f235965ad422ba5081a21865b8209ae81763e1c4c0cccbccdaad539cf773413a50f5ff1267b9238f5602adc06764f775d3c",
        "26ec9df54d9afe11710772bfbeccc83d9d0439d3530777c81b8ae6a3",

        "31c82d71785b7ca6b651cb6c8c9ad5e2aceb0b0633c088d33aa247ada7a594ff4936c023251319820a9b19fc6c48de8a6f7ada214176ccdaadaeef51ed43714ac0c8269bbd497e46e78bb5e58196494b2471b1680e2d4c6dbd249831bd83a4d3be06c8a2e903933974aa05ee748bfe6ef359f7a143edf0d4918da916bd6f15e26a790cff514b40a5da7f72e1ed2fe63a05b8149587bea05653718cc8980eadbfeca85b7c9c286dd040936585938be7f98219700c83a9443c2856a80ff46852b26d1b1edf72a30203cf6c44a10fa6eaf1920173cedfb5c4cf3ac665b37a86ed02155bbbf17dc2e786af9478fe0889d86c5bfa85a242eb0854b1482b7bd16f67f80bef9c7a628f05a107936a64273a97b0088b0e515451f916b5656230a12ba6dc78",
        "aab23c9e7fb9d7dacefdfd0b1ae85ab1374abff7c4e3f7556ecae412",

        // SHA3-256
        "",
        "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a",

        "e9",
        "f0d04dd1e6cfc29a4460d521796852f25d9ef8d28b44ee91ff5b759d72c1e6d6",

        "d477",
        "94279e8f5ccdf6e17f292b59698ab4e614dfe696a46c46da78305fc6a3146ab7",

        "b053fa",
        "9d0ff086cd0ec06a682c51c094dc73abdc492004292344bd41b82a60498ccfdb",

        "e7372105",
        "3a42b68ab079f28c4ca3c752296f279006c4fe78b1eb79d989777f051e4046ae",

        "0296f2c40a",
        "53a018937221081d09ed0497377e32a1fa724025dfdc1871fa503d545df4b40d",

        "e6fd42037f80",
        "2294f8d3834f24aa9037c431f8c233a66a57b23fa3de10530bbb6911f6e1850f",

        "37b442385e0538",
        "cfa55031e716bbd7a83f2157513099e229a88891bb899d9ccd317191819998f8",

        "8bca931c8a132d2f",
        "dbb8be5dec1d715bd117b24566dc3f24f2cc0c799795d0638d9537481ef1e03e",

        "fb8dfa3a132f9813ac",
        "fd09b3501888445ffc8c3bb95d106440ceee469415fce1474743273094306e2e",

        "71fbacdbf8541779c24a",
        "cc4e5a216b01f987f24ab9cad5eb196e89d32ed4aac85acb727e18e40ceef00e",

        "7e8f1fd1882e4a7c49e674",
        "79bef78c78aa71e11a3375394c2562037cd0f82a033b48a6cc932cc43358fd9e",

        "5c56a6b18c39e66e1b7a993a",
        "b697556cb30d6df448ee38b973cb6942559de4c2567b1556240188c55ec0841c",

        "9c76ca5b6f8d1212d8e6896ad8",
        "69dfc3a25865f3535f18b4a7bd9c0c69d78455f1fc1f4bf4e29fc82bf32818ec",

        "687ff7485b7eb51fe208f6ff9a1b",
        "fe7e68ae3e1a91944e4d1d2146d9360e5333c099a256f3711edc372bc6eeb226",

        "4149f41be1d265e668c536b85dde41",
        "229a7702448c640f55dafed08a52aa0b1139657ba9fc4c5eb8587e174ecd9b92",

        "d83c721ee51b060c5a41438a8221e040",
        "b87d9e4722edd3918729ded9a6d03af8256998ee088a1ae662ef4bcaff142a96",

        "266e8cbd3e73d80df2a49cfdaf0dc39cd1",
        "6c2de3c95900a1bcec6bd4ca780056af4acf3aa36ee640474b6e870187f59361",

        "a1d7ce5104eb25d6131bb8f66e1fb13f3523",
        "ee9062f39720b821b88be5e64621d7e0ca026a9fe7248d78150b14bdbaa40bed",

        "d751ccd2cd65f27db539176920a70057a08a6b",
        "7aaca80dbeb8dc3677d18b84795985463650d72f2543e0ec709c9e70b8cd7b79",

        "b32dec58865ab74614ea982efb93c08d9acb1bb0",
        "6a12e535dbfddab6d374058d92338e760b1a211451a6c09be9b61ee22f3bb467",

        "4e0cc4f5c6dcf0e2efca1f9f129372e2dcbca57ea6",
        "d2b7717864e9438dd02a4f8bb0203b77e2d3cd8f8ffcf9dc684e63de5ef39f0d",

        "d16d978dfbaecf2c8a04090f6eebdb421a5a711137a6",
        "7f497913318defdc60c924b3704b65ada7ca3ba203f23fb918c6fb03d4b0c0da",

        "47249c7cb85d8f0242ab240efd164b9c8b0bd3104bba3b",
        "435e276f06ae73aa5d5d6018f58e0f009be351eada47b677c2f7c06455f384e7",

        "cf549a383c0ac31eae870c40867eeb94fa1b6f3cac4473f2",
        "cdfd1afa793e48fd0ee5b34dfc53fbcee43e9d2ac21515e4746475453ab3831f",

        "9b3fdf8d448680840d6284f2997d3af55ffd85f6f4b33d7f8d",
        "25005d10e84ff97c74a589013be42fb37f68db64bdfc7626efc0dd628077493a",

        "6b22fe94be2d0b2528d9847e127eb6c7d6967e7ec8b9660e77cc",
        "157a52b0477639b3bc179667b35c1cdfbb3eef845e4486f0f84a526e940b518c",

        "d8decafdad377904a2789551135e782e302aed8450a42cfb89600c",
        "3ddecf5bba51643cd77ebde2141c8545f862067b209990d4cb65bfa65f4fa0c0",

        "938fe6afdbf14d1229e03576e532f078898769e20620ae2164f5abfa",
        "9511abd13c756772b852114578ef9b96f9dc7d0f2b8dcde6ea7d1bd14c518890",

        "66eb5e7396f5b451a02f39699da4dbc50538fb10678ec39a5e28baa3c0",
        "540acf81810a199996a612e885781308802fe460e9c638cc022e17076be8597a",

        "de98968c8bd9408bd562ac6efbca2b10f5769aacaa01365763e1b2ce8048",
        "6b2f2547781449d4fa158180a178ef68d7056121bf8a2f2f49891afc24978521",

        "94464e8fafd82f630e6aab9aa339d981db0a372dc5c1efb177305995ae2dc0",
        "ea7952ad759653cd47a18004ac2dbb9cf4a1e7bba8a530cf070570c711a634ea",

        "c178ce0f720a6d73c6cf1caa905ee724d5ba941c2e2628136e3aad7d853733ba",
        "64537b87892835ff0963ef9ad5145ab4cfce5d303a0cb0415b3b03f9d16e7d6b",

        "6ef70a3a21f9f7dc41c553c9b7ef70db82ca6994ac89b3627da4f521f07e1ae263",
        "0afe03b175a1c9489663d8a6f66d1b24aba5139b996400b8bd3d0e1a79580e4d",

        "0c4a931ff7eace5ea7cd8d2a6761940838f30e43c5d1253299abd1bd903fed1e8b36",
        "dc5bebe05c499496a7ebfe04309cae515e3ea57c5d2a5fe2e6801243dd52c93b",

        "210f7b00bf8b4337b42450c721c3f781256359d208733846b97c0a4b7b044c38dbb219",
        "3305c9d28e05288a2d13994d64c88d3506399cd62b2b544213cf3539a8e92e2e",

        "3cb8992759e2dc60ebb022bd8ee27f0f98039e6a9fe360373b48c7850ce113a0ff7b2ae5",
        "3c00bf3e12ade9d2de2756506f809f147c8d6adc22e7bb666e0b1d26469e65a5",

        "22634f6ba7b4fccaa3ba4040b664dbe5a72bf394fb534e49c76ec4cdc223f4969e2d37e899",
        "a87e5c78837d7be0060d8f5eda975489ec961b28d7088f42a70f92414ae17793",

        "6e1dcd796b2015ee6760f98fdb40e668b2cf38b05c91f6a91e83bcc8ac59f816f90a59d64e8e",
        "746bf845c08aa186b5fe1ca35528232c4a491a3a2a32cd23e990bc603f3268ae",

        "ee0be20320f9d44073281265a6e9fa6b9d252495624b8d016b8ef57e1b4e859d8ad3b50b89416d",
        "a3257baf14ca16e1137dc5158703f3b02ebc74fc7677165fe86d4be1f38e2f7c",

        "8ae2da242635b6568289bf6bec8a438dbac1f5b4d50a90bb7449bdb92a59378e23452dbcabbbe879",
        "e25c44802c5cf2e9f633e683d37aa8c8db8a0e21c367808121d14d96c8a400b5",

        "bdd0252dec5b798ef20e51791a18e8ca234d9bfde632a9e5395337a112dd97cdf068c9f57615424f59",
        "e02c1b197979c44a5a50d05ea4882c16d8205c2e3344265f8fe0e80aed06c065",

        "c4c7b6315cb60b0e6cd01ef0b65f6486fdae4b94c6be21465c3a31c416ad2f06dcf3d6eae8eecf84ca7a",
        "2da21867cd6b5402d3caff92a05fddfca90199fd51a94a066af164ce3d36c949",

        "b17977aced3a1184b14b0e41a04dd8b513c925ca19211e1abdc6c1b987ac845545fb3b820a083b4f7883c0",
        "f91b016d013ede8d6a2e1efd4c0dd99417da8b0222d787867ca02b0ea2e80e45",

        "f65c3aa1d9981a84e49fc86d938f3f756f60e3858d5e1f6957dd4d268e28d68e90ba9a11d7b192d6c37fb30b",
        "3acbebf8eda9d3c99a6b6b666366c391e8200d55fd33ad8680734def1dc7ae85",

        "49abba1fa98f3c4470d5dd4ed36924af4a7ad62f4c2dd13e599238883ed7d0cb95bbaae58b460332e6b7681446",
        "02bcd9ea4f1aa5276f38e30351a14a072bc5d53a52d04d559a65ca46f1bcb56e",

        "275645b5a2514fe65a82efac57e406f224e0259677674f1d133f00a5ee9a6d1a8fed0eadbbff5a825041d2a9715d",
        "c70a874d786cd0f3f09fa4dc1bb8f551d45f26d77ad63de1a9fdfb3b7c09c041",

        "cd02b32107b9a640fc1bf439ac81a5c27d037c6076e1cfe6ad229638037ac1550e71cf9557c29c2fc6017afd5a8184",
        "36c73d11d450784eb99af068cd4e1cbc5768c8a2118010aceec6d852dda80d95",

        "5a72e0e1aec82a6541f04883bb463b0c39c22b59431cfb8bfd332117a1afb5832ce5c76a58fcf6c6cb4e3e6f8e1112de",
        "90fc3193552ec71d3315ebbb807913afd4cd2f0833a65e40d011d64de5e66513",

        "43402165911890719f9179f883bbbc2a3be77682e60dd24b356a22621c6d2e3dcdd4cb2ce613b0dfe9f58629ee853e0394",
        "5c4b6ceac9441defa99b10b805a725d4018b74b3e1f24ad8934fc89b41b8fd9e",

        "fc56ca9a93982a4669ccaba6e3d184a19de4ce800bb643a360c14572aedb22974f0c966b859d91ad5d713b7ad99935794d22",
        "e21806ce766bbce8b8d1b99bcf162fd154f54692351aec8e6914e1a694bda9ee",

        "ace6297e50d50a11388118efc88ef97209b11e9dfcb7ad482fc9bf7d8deecc237ad163d920c51f250306d6cedc411386a457c7",
        "f5581403a082bbf5ad7e09bdfccc43bf9683ebc88291d71d9ce885a37e952bd6",

        "3bad18046e9424de24e12944cd992cfba4556f0b2ae88b7bd342be5cff9586092bb66fac69c529040d10dd66aa35c1023d87eb68",
        "faed76ff5a1cd99183b311e502c54e516d70a87050cf8961c8cd46f65c1358cd",

        "e564c9a1f1aaf8545a259f52c3fd1821ed03c22fd7424a0b2ad629d5d3026ef4f27cbe06f30b991dfa54de2885f192af4dc4ddc46d",
        "811529c600c9d780f796a29a6b3e89f8a12b3f29c36f72b06cca7edc36f48dc0",

        "6043fa6465d69cab45520af5f0fd46c81dbf677531799802629863681cea30ffa3b00836fbf49f87051d92aaeac0ed09bcb9f0755b7b",
        "b0fceecdaef6c76d5fc3835b523ce2416f4a9b9bd1f90234445df0f2b689f2f5",

        "2040c538c79237e6f2b8188c6375ec2f610ac2301607b9c23660c3a1e1c3a902cb2950c59aac3af28f984f6369c4debe8623dfa74c967b",
        "e33dbdc0acc23fcfad3c759c4333410bd3a40efb1366ade157d2c81d65a0a6c7",

        "00ff6c96b7aa3cf27d036cf20af7031434113252574bda9cf9244d85aef2593d3a7a83bff6be904b75164a1766828042bc3f4f090d98a03d",
        "d000eafca34815783bed9b050c6901c97f2e77d4771a0ed724dd8f6ff1448791",

        "e8df14936cce118139e690f1662f88cfbc9c333b6dea658c02cb1d959644592842542fd9d8d61a04d4a892128f0ddff7b6502efffbabe5cb0a",
        "3479a9617a3adca35854c08fe987c2fe7ff2b01b04f2d952c107b3f066420551",

        "4ed981a31f70dd6b70c161be1f01fc1bba54d06d9494e7eb194e213d5e0e71e0fddd49cb1f075353da22624cbe4ba871aab32906e45b6fbb691b",
        "9c824a00e068d2fda73f9c2e7798e8d9394f57f94df0edeb132e78e8a379a0cf",

        "7802b70c6158bc26d5f157671c3f3d81ab399db552b9f851b72333770348eb1fdb8a085f924095eb9d5ccfd8474b7ba5a61c7d7bcde5a7b44362cf",
        "fa9726ccb068c0adb5d20079c35a318b3d951eb43b196c509ab790b7e9202207",

        "ff83dcd7c1a488e5a128d5b746284552f1f2c091615d9519f459bc9010ca5e0ac19796c4a3fd7a15032a55a1410737d07855b07f61fbd8f5759e9218",
        "8bd8d494a41acda4b7cd2994badaecff0f46ba2743458f6c3fdc0226f9492ede",

        "afd4764cc7d5de16a3cf80c51d0c0d919f18700c7dc9bc4e887d634fe0a3aa94097d590e4123b73f11ccb59e23496a3d53d2bfa908056c11c52c23abfb",
        "e9e3b3da648cf230f1973f3814eb81316d2a496826ea39adf4674576f97e1167",

        "6fa6de509719ffbf17759f051453c0ac3cbe13346546bbc17050541074b034af197af06e41142211ee906a476039b3e07d6cb83a76aac6fca8eac307c034",
        "766630993fbb651fd8d3603e3eebc81931fb1302a46791df259a6e13ca2cba9f",

        "93cbb7e47c8859bef939155bea488090283ecf5023d99767c960d86baa333af05aa696fc170fb8bbac1e6473956d96b964580ee6640f0cc57be9598e55fc86",
        "d3212abca1100eb7658c0f916daf2692c57a47b772ee031c4ec6ad28a4a46de9",

        "67e384d209f1bc449fa67da6ce5fbbe84f4610129f2f0b40f7c0caea7ed5cb69be22ffb7541b2077ec1045356d9db4ee7141f7d3f84d324a5d00b33689f0cb78",
        "9c9160268608ef09fe0bd3927d3dffa0c73499c528943e837be467b50e5c1f1e",

        "4bef1a43faacc3e38412c875360606a8115d9197d59f61a85e0b48b433db27695dc962ed75d191c4013979f401cf3a67c472c99000d3a152227db61de313ab5a1c",
        "8703a1f7424c3535f1d4f88c9b03d194893499478969fbb0a5dc2808a069ab8f",

        "f0be5e961bb55b3a9452a536504f612a3e66aec8160a882e5156eb7278433b7ea21de31e39383d57fcdfb2fb4a8d227a9d6085fb55cad3abb78a225535da0e34efea",
        "2fa180209bf6b4ad13c357d917fabb3e52c101a0cdb3f2299fa0f7f81dfb848e",

        "206f1c36ba25aea73398fffc9b65c4637cc1f05a6bbee014dccbd61e3b7aa9423887bbac62152a4bf73a4b7afabe54e08720589464da7985d8e6591ac081d115df2fe6",
        "558ea7c800b687380cce7e06006e1ebe0b89973f788c4caac5780f22dbf382e8",

        "8cd71434c00663f3bda0205508a4a266548dc69e00ca91fde06d165b40279af92674f75bd8133e5a9eb9a075c9068f68f4b820008a1fb42d89d1d759859e68f8efc6fb60",
        "085b343b08516f320a9b90fe50440a8bc51ae0850fa38d88724a4d6bd3df1ad4",

        "4cf5bbd91cac61c21102052634e99faedd6cdddcd4426b42b6a372f29a5a5f35f51ce580bb1845a3c7cfcd447d269e8caeb9b320bb731f53fe5c969a65b12f40603a685afe",
        "f9dbb88c5bb4415e17dee9222174538eeab371b12d8d572cfdf55b806e3158e4",

        "e00e46c96dec5cb36cf4732048376657bcd1eff08ccc05df734168ae5cc07a0ad5f25081c07d098a4b285ec623407b85e53a0d8cd6999d16d3131c188befbfc9ebb10d62daf9",
        "3571326a1577c400b967ac1c26df2a0dcf5db7070eac262a8071da16afa7c419",

        "981f41a83d8f17f71fc03f915a30cd8ac91d99aa1b49ef5c29fb88c68646b93a588debcd67474b457400c339cca028731df0b599875ab80df6f18b11b0b1c62f2a07b3d8209402",
        "62aea8760759a996f4d855e99bcd79e9a57ea362522d9b42fd82c12c9294a217",

        "5c589fc54fefc4d6e2249a36583e1992fc6b8a9c070e8e00c45a639af22063e66ae5cdb80238c82db043a5e1f39f65626e6d7be5d6a2d3380fa212f89211200412e5e4315fc04e40",
        "18deba74e9d93ae7df93c6c316ef201bf5e3a661e68868e14d4f56264f5d858c",

        "7c8691e7b2560fe87fcc5e2877f7e3c84d9101eca4818f6322a58986c6cf05627c0d6919ef2edc859f81fa1f33e0cc1f10edf7e52a9c33981af2ff0d720c94ea4d62170b2a4d1224fa",
        "5a5a438b57c1b3ce8756094252362afeaa9fc91cd45b385d16994ec8af49aa6b",

        "97359b564b2bc20800ed1e5151b4d2581a0427ce9539d324c3637cfb0e5378dc2cf6d72946e2a3535a2f664ede88ed42a6814c84072b22c43de71e880a77c2d9a05b673bc15a82e3255f",
        "be54f2e435f760d5b77c0ae61ef0aa7f5f3366f47819f350dc8a39aff8c73a8f",

        "a0dfaecd3e307c5ddf9a93603f7e19725a779218734904525b14586ff0ce0425e4efe7e1c06e745c28ed136f6031c4280fd4061d433ef700b6d1bc745064231fecf387015f94f504b6ad8c",
        "60d80f1c703dad5da93db222fb45fb7fa768c8aa2787f4b81f1e00365b8f49e2",

        "568d66d061306c3419a1928ce7edc8e3400c30998f09bdac6f63ff351eb23d362e8dc5927eac805d694ac9563dcd7fb2efa9591c0d827af9f39146f0424873aa8e3963d65734b1713baf0a44",
        "7a4fe37f296991121792dd7c2c30390725a1eebbf20b766a5a1c3c6c3646d996",

        "d65b9f881d1fc7f17d6dd429faca8404e6ce60fba7d89b7fba003c8ef84d8083182979327611fc341291ba80dc70ad3b2f28b6d29b988445e7fdb7c6561f45822ac81dbf677a0b27d961dc6358",
        "51cc71b6934afcf28fa49942b76323f36cd6a0aecc5a0e49c10994ddcabdbb80",

        "711c88adf13e7a0e694652f2b9a397543f4937fafb4ccca7f1ad1d93cf74e818d0fedfaee099f019014ec9e1edfe9c03fdb11fe6492ad89011bf971a5c674461de15daff1f44b47adad308baa314",
        "1780e52e306858478290c46b04d8068f078a7f6ad8e3790a68fc40dccfbdadc9",

        "f714a27cd2d1bc754f5e4972ab940d366a754e029b6536655d977956a2c53880332424ddf597e6866a22bfca7aa26b7d74bc4c925014c4ed37bfe37245fa42628d1c2ee75dc909edc469ee3452d894",
        "f4afa72f3e489ad473dc247aae353da99fb005b490e2c4e1f5bd16a99732b100",

        "fe0c3280422c4ef6c82116e947da89f344d6ff997bf1aec6807e7379a695d0ba20ae31d2666f73bbdbc3a6d6ac2c12dcfb5a79173dfc9cd2e0d6000e3114f2767edec995772c6b47dadc136d500251e5",
        "89198e2363efd4e0ba7a8a45f690f02712e6f856668517bae118d11e9a9dc7cc",

        "02e238461d0a99d49c4cd16f442edf682c39b93114fc3d79f8546a99e5ead02f0cfc45081561da44b5c70eb48340418707fd6b2614580d5c581868ba32f1ee3ac34bf6224845b32ba7f867e34700d45025",
        "abef81b33591eedcac0cf32fb5a91c931f2d719c37801409133552170ce50dbf",

        "fb7c8cd4031007f8159d5c4c6120dee6777a3ace0a245b56f31e8aae7828dab3cf35c308de1d0d684592ef3a9e55796603a92f68d109f7a3ac1635f7c4d334955614c812753431bb0a0743291a0fc41547f3",
        "5a67284d39e4f37caa64ca1a54593c35f6d8f3a3ec20d460393a39f6f57c4486",

        "6b2e868c7d0ee1c240d3a67e2fdf36e8e23817c02644a54453d10454da5859d41e833a5285ec63e8ce28aa64a50435a7740eea4b7d5827892678b35993d3f5da7a1c64f533173f3d0fa37e1aebf70827052c26",
        "aecf5dab6fea9ffd1bce2cdfeec0bee9d214a669e8306d5b6688afa8957fc91f",

        "e5f3ba000c43bb6aca4e0a711a75912a48241cffa5b4b0b17f901f9e5097d94036c205f7a307d008567d05e58ac0dfaf6d971bf9d3d450cf2c7c83f6b328f676e9ab425642f5a5a71e389dc4fa49b6d7e848a09f",
        "182d6e4316f4bc18d7163b1b21462d99f99c6f34d2c00ee771ce54fd6c5018b9",

        "939c61e68af5e2fdb75a2eebb159a85b0c87a126ce22701622f5c5ef517c3ab0ed492b1650a6c862457c685c04732198645b95f84ccb0e726a07ce132827a044dc76b34d3f19a81721f1ea365bc23e2604949bd5e8",
        "121057b0b9a627be07dc54e7d1b719f0a3df9d20d29a03a38b5df0a51503df93",

        "9eadaf4811a604c65eaa7b1c6e89f2c0ab96bebec25a950ba78aac16d9371ca1e7458acf331e077ef6a735d68474ab22d2389bdf357fb2136c9f40e1e1eb99592c2bbb95d94931016b4d37faa08b1e9bf71bf2d3708a",
        "c237194b902e48dca5bd096cb51562079d0cdccb2af8088197676c17b0896be2",

        "71dcca239dced2ac5cc49a9bf9ea69a99be22ba62216716b524db80f337dee5eb7e032869e4adc1497babd1fa82fa8c3cfbd30d2eadfb4c5d40f99f9d194d7182c9cb7d41e8adbdcf2917e086782fdd756e2961c944070",
        "377d1cffb626735810b613fd31ef9bbb4577cd752521abe3a41afa921e623da0",

        "ea130d3236bca7dffb4b9e50e805309a503e7347227aeb9f1bd15c263a98dd65753d2eedaa734b9ad88f41158f32419ca529f3062b910c019f3f239f635fc1116e5ab7b242feb4471ed9168474e501d39d6bae52cc21061a",
        "85c7a52d53f7b41162ea9f1ef0d07c3fb8f0ec621617f88cb3828ebe5388ab3d",

        "28f1be1156792af95c6f72e971bf1b64e0127b7653ff1e8c527f698907a27d1544815e38c7745529bc859260832416f2b41cd01e60c506239a7bf7553650bf70d1fe7a2c1220ac122ea1e18db27490447d8545a70bf0ffc8fa",
        "b2eb3762a743d252567796692863b55636cb088e75527efd7306a2f6e3a48a85",

        "c8400ef09c13e8acc8a72258f5d1d20302c6e43b53250c2f6c38ff15be77e3cac04d04b8421fc8fdff8be5ca71edd108e9287b42dea338bf859100eea376da08a0e695f0dc90b95e467cbd3c2a917a504a5ae01c310ae802c4bd",
        "69966e89b7bc7f39cd85791b92180ff3fed658d8240e393e1e6d7c24b8d0ac95",

        "a48950c961438e09f4d054ac66a498e5f1a4f6eabfde9b4bf5776182f0e43bcbce5dd436318f73fa3f92220cee1a0ff07ef132d047a530cbb47e808f90b2cc2a80dc9a1dd1ab2bb274d7a390475a6b8d97dcd4c3e26ffde6e17cf6",
        "44c00cf622beca0fad08539ea466dcbe4476aef6b277c450ce8282fbc9a49111",

        "e543edcff8c094c0b329c8190b31c03fa86f06ace957918728692d783fa824ba4a4e1772afbe2d3f5cba701250d673405d2c38d52c52522c818947bcc0373835b198c4cc80b029d20884ac8c50893c3f565d528a0cb51bf8a197d9d6",
        "6d5260384f3cefd3758fb900dcba3730d2b23cee03d197abeff01369dc73c180",

        "4e10ab631718aa5f6e69ee2c7e17908ec82cb81667e508f6981f3814790cfd5d112a305c91762c0bd9dd78e93ef3a64c8be77af945b74ff234a0b78f1ed962d0d68041f276d5ea40e8a63f2cab0a4a9ed3526c8c523db7cb776b9825b4",
        "d88e5f3b2d0a698fd943233760a3000a3360d9040e7374b22e39ea58d868102d",

        "604d8842855354811cd736d95c7f46d043a194048b64bf6cda22c3e0391113dcc723e881ae2ad8dc5740aa6bda6669ddb96bb71acd10648380693f7b3d862c262553777004bd6852831618519fbb824759f4dd65af1b2a79cc01096d7c8d",
        "8a8ab6cf5c02b9ae8f4c170740eff1592f3eda11d3420ac8b421d93cfbb35db8",

        "628180e14f41ebdfde3b4439de55ee9cd743d41040f3457ef2280370dd659619fa0ce69580c709725b275a6eda8bcb82a8447c20fdf68cba15412f83e2a10079fe9399a3e3fa61975ec0a64041c0ecde59e4844e9f8a608cb22d2576854182",
        "8d154bf6f9cb72efc0d8b3927a8f690060d1d48bbe5cc72094d2c8b149a75132",

        "fc150b1619d5c344d615e86fca1a723f4eeb24fbe21b12facde3615a04744ef54d8a7191a4454357de35df878cb305692278648759681919d1af73c1fb0ff9783678aec838da933db0376e1629fcca3f32913f84bc2ff3ffc3f261d2312f591c",
        "3f626c8bb20a132495bd3022b3fcd0ce0604b91a9d70132dab4099f73dde23d5",

        "6dadbecdd15e5646e3f37a6fe5b328e06113cce3c8cf07285939afba44d117321017902b3a9d2ff51f60d18e1b585dcdf34e49e170ee60fa4d1dc246548d2c1fc38e7983f42769c43d65a28016f3f4d479ebe1cd8fec5d1f886dd21aca5067d94f",
        "9098ea34c40b541b153e80a8bd92da19432b18b7d329760b302f8a54c395dd06",

        "9cc5fd3035b72dc63b8c3c326fd013081e6b8716f526d3fe176b45256d4c37cc3dc8417dff49ada96c702b8fd715c65fc08a17a0a720b9cf1eedfd4922ccde6baba437f782ee33b95371056b0350dad743470c3b663299f16fcfd34f6fc459cd0ee4",
        "b0c04f24bb6d3d4fcbfdf9222d0e886f1eb60a0566a478085f7623a025a5b981",

        "f3f063fbcf2d74aa5a02d240c962ed7bb119b3a212bdb41594e28428108e613152ed16e01e451fcf702b0e5a08f82eb12677652b93e05fdee00ae86cf2dc9a1fbf05b93952ec5b8515eacc324fb830e1ec236afd7d073d4b7f7ab1c2e048b99cbfa012",
        "f930d79360b581b1bbfdeac57133a339444f5c44538c921631eabaf058277d32",

        "840739a3d6992c13ec63e6dbf46f9d6875b2bd87d8878a7b265c074e13ab17643c2de356ad4a7bfda6d3c0cc9ff381638963e46257de087bbdd5e8cc3763836b4e833a421781791dfcae9901be5805c0bbf99cca6daf574634ec2c61556f32e642730510",
        "19795657e08cfbb247a17cf209a4905f46e4ddf58eea47feee0be9bb9f5c460f",

        "4a51b49393ab4d1b44fb6dc6628855a34e7c94d13b8b2142e5d5a7bf810e202cefdca50e3780844a33b9942f89e5c5b7dd6afb0a44541d44fb40687859780af5025fecc85e10cf8249429a3b0c6ff2d68c350c87c2fcbf936bd9de5701b2c48ce9a330c9ee",
        "128fb4114e43eefd19277c708be9e6873e66d7fd59c58a1485b7b015facfa795",

        "afc309e6b7b74dfb0d368e3894266fc4a706c3325e21f5550d07a6560e3d9703c134ca6ad078e4a7b82ad6fa85b0bc1ddcab05d43f29d5c58d1da78ac80c37051b089ff31ce2c0c44e9ce3abea1da0f1df28008e178fdefafca493413bf1d256c729d0a9225e",
        "03e782b01a4ba10f640470bb3cae487eb9cbbaab8c9941978b194f6a312cf79e",

        "c5ae750f2230642092397b84ad5526c46ae9480ada16892816e0f2db7690b751035653ea2f33da3cc4168b591b46a5548eff7d012f60ccfdbb854deec9f0880c472de8e127b5144c56147cccee4732fbac68fc59a48da74b33ed9e643644bbe279795c7c737eba",
        "f64b7ab243ce6e6c04b483888ba8a655465c21d95eb60c7b8d6e566a3811bae2",

        "603e13f61499e12ec6b33b68847a281d314f54dc705c0f3fc428981ff5689c04b519fadf83cbc9fcd0409c326035045df480570e265bb080940037ce4076a36437aafdb371c1a62af9ad9b614dfef89708fbbb5ebef2cb9528cc399781e4c5b22f1aa4dba623809f",
        "5f76962fd3d373e5db2953c0823a51fe81f874450bedf7e46876394b04d3ef66",

        "e03115cfa19efcd796da389063c4be6acce684d983f8edfb3da6887b0b94fbb5e89e3a1a8e64fdd68f0670b1a02c2c33384a660c5a2266b3ae8a3b4cd76faecf011a7467b9b2a818020278a5a57d1eb1c87f1224c2d67dd02e81f1553eb75841532c2b7cca8fe5e418",
        "d107ee6ee4a58871a33c49657faa2573e475f11918c4a4e3801d0e17fb93c6e3",

        "0e6c1d58b1b9d3a2d399aafd60529e07d483a2755bb7e44c373b5355632d5fca76d6ff56c93af93ddcec5ed6f62753420c1b1758e48542df7b824b00a3a54dfaf0470b18d51e31e10b12dd8e324b5dc1bb8f3b7305cb762ec6ef137dadffd4a2466748861d9004f626b0",
        "02ab2dbb02944354799051247b1a25c19f3696e1afcb502b859e83798b33fd77",

        "6db2a43a229b10c3629249fc5136468b4d84df7b89ec90ebf7aa7a036c53aa2dffae9e81b2c60580543dc706a5e3457abc87e248a60ec29150c2d221a6ec08a1fda4ec0daee8576904ec7ab059b1230e7bd93c4e55ba9496cbb1e352e5b8086e303b94c861288ce53c466b",
        "8cc4d39b2f5ba0bc9d2ee2a8777cf08533e60cc69b65a7b31c5c2121193aa31e",

        "31d995f7ff8b6de70829a8336c610f10df2c866107a4922b25151849f8566861df5a79163d02767f21357ad82733997899261f03dafb1ce1056f20efd16d4374b89768565823c38e19e899d910b847b023f1867b6e4fed02e604b8243c0bc7cb05b9ea1f17955bfa36698c9c",
        "c99c7191b34c9ad3f941d4ad442cc865205cbb4c2a6927c592e831cbc4d36fcf",

        "cb0b8cb7de621c8e0a0fc6be2fc18d0e8818a2c2dd0b3219fa87831a61583f903c4d105495976ccac973b3ae3a09771145931a9e74c19f22f45cba4c492b29b1401347122581dfe2370d3e0359578cd10a355c619711810a8f8c232578671312c0a45c7cf7e81bdd3b249044f3",
        "6d2f57a7e42b35369cf2cd60caf9e65aca7d9aa019e6824bb806348f1acf3c7c",

        "48dff78aed5f6e823054924a78dc1b8e51a117f1610181529f6d164ebf0f6406f0b02422cad8c916823759a361437ca17423d3fd84cc8afe486a31ccda01c732685418a32c064a7b9effb288e811ecc99adb2a759feecc3f702f31d9877dcdb717937c15fa2f163bea744400f58c",
        "14b631f0f00a3024ad1810dabf02711e28449668abe27f69380942268968d4f6",

        "06cc9fa542ceb35c88fb6ab82c29d5dcd530f807d3f1c3bcb3974421101d1aa6ac112de6bf979cd28eb0f70c40bcaf91ed3eca9bf9e0dbc6a0b73271d1c7506740ca9ebfb72d5e00ac5ce189193ffa308804b42a6d20402bb99031cdac65ec36eb7f59f5d299df2e0b8690f760b9a0",
        "574fd82a9fceb8f7bbbf244d16e0412cbda8153b720846c32b8f10fe5779a881",

        "8d93627c0b7cbf61a7fe70e78c2c8ed23b1344b4cfed31bd85980dd37b4690e5b8758f7d6d2269957a39a1ac3451cc196696ae9e9606a04089e13456095a1ce1e593481b3ac84f53f1cb10f789b099f316c948398ad52fa13474bdf486de9b431bd5d57ef9d83a42139a05f112b2bd08",
        "344ec86642eabb206b2fd930e4c5dde78aa878577d6c271cb0069d4999495652",

        "d0af484b8be6b41c1971ae9d90650a1e894356c9191d6be303fa424f2b7c09544ec076a0f1865c8c97927ca137529d5bedc0df2ef08a4cc7c470b094b1eeaa86731c041633d24086b60f7369d59c57652dec9b3817477df9db289ba020e306c9a78a99b539128992deb23cfc508c5fc3af",
        "b7ba998726477c32792e9c3eddc1cb6feb7c3933e49f2e7590d8ce7a2113e6f8",

        "b212f7ef04ffcdcf72c39a6309486c0eeb390ff8f218d6bd978b976612f7f898c350e90bd130723e1126af69295019b4f52c06a629ab74e03887020b75d73f0f78e12785c42feb70a7e5f12761511c9688c44da6aaa02afa35b31edc94c3a0779b6ab9462525c0ccfba76986f873fe1e6ba9",
        "2f26b96c1fa3f3dee728f17584e733b4189821c659b8885a5fb1d12d60d2aaa9",

        "86591ada83fba8175a0fe91d264e7f9b2df97ee4c32570e76b579d6140508951932abdadd6a4ca53b8bb8c42927aac0a02126881d52d97b82b80e72dd59f6a42021651ee1bb5f7b3eb2b21d003d784b75dda87c13f714b216282e8175474fa661b445d071bd5341f3a88302f410d0f8a857962",
        "e3edbc8c42ce5d2384dfb24fb1de5d4798b1bc3cc78c97033894040dfa6feb6c",

        "92b5a8e84b6a2ac4d5b1e61d63804abd641dd630058ec6d5f752f135724ef1947a0a84c6611d32448de6307f7b7d857404e96b81df94f87768fcfdf09faa2fe37468847542afe012995ff1bd40b257a47a7309f8896bf4fb711de55bfeb3a8be0837729ef6067c578182f17ebb080a754f22773c",
        "80ed0a702812297c2aa1b6b4b530c2b5ed17ecfba6d51791cf152d4303ced2e6",

        "d284a0a9a4de5d4c68cc23884c95ad7619aa39b20a2cf401deaeb3362c3ce356f79cc3fa82d3d1f565ec8137e1f435f171496afaa1152f722315dca5209f0031cce39b6c3d718e007dfb4fd8de5ce1408dda04476aa8a96817afa86a4f8fb5857ae091c67ebd7db5d783f434ead699aa96e56f610d",
        "654eccefd0a4fdb2ac0ab56288c64399b37bc4d57ff4a9f1cce94362fc491bda",

        "f57f0f8795385b805246a0a2573afc274346a9eccf50c626b0455a50bfb09668578b5a5afe54fbbd486444bdf97dba586aa224ce2e2b4b52f418ff06afa65a26f5204983a5f84734cd166c88cb70a73fb2db48f9ef20c1ee2c53ade07460114e98e7e2ebd24ac84ea90422eb143c4a42e2991a565959",
        "135ec8b144a667dceae8fadd287df81c10ef3ebef87ff2fb56e60ae708a88f3b",

        "2a41a52e6578873588a57f11f1be7c7eb398d01f3bfdec2c33fe6b65a68a534a6540978daa82e0c8fccb8c6c5242f7f97b8ffa75bdedb217bd8083439eea5cbb6d193c13bd62f5658ed4304774c6b1faf5b3dce432487840cabab415fb5d67640a739ca6e5414e760869708a9d7331e7e7ad7d55e035c7",
        "a6a1b8a26f6f440f19f16dce1d3001477d73ee7f6c374bce2922167b81970d6a",

        "4d11aa5d3c6b6900f49ff90dd815744572be5648b64bde638b9db7a9877dd745fa8ea80e2f7f655cee85c71a4509e21d899e49b4973579815f947587a404ad83fd4a248020d9d2a65f46485373fc926d793161f63a196ae0af590923c5be2a0e5d2f69da97e0788550c9c1dee9574ddc4a61e533275d7729",
        "fc5159f0ddd6d765c85fcc3fc3ac1dc0d317d8ea0b110e96ac9f7a398dc386c5",

        "05cd99bfe031d123ca7061d3de0956f4bbf164bad792db881713d6599ddab55ee24fcee804e360896152c8766424f8309f7a24641a07be0feb5da5e5076a9af45842f385101f93433ca5199f9c6b5872b2b808e4198aba8e18dd12db772930b4912d6f5cabeb529884f4bb142de55e021b3276047b22b64cc5",
        "8aa07742e6f1f47ad020ed6684edc8dba4af36b782955f0f972be3ae980aea0e",

        "529684398d68bdc19e7a00ce32cc1a8c1315b97f07137474f61f0cb84a04f2879b1109c78c6dacf7f0abf362329e3298f36fc31ef4ec06653723a5f961301dfb63537ad15946611cb2cd54ea928e322e7423fd6d146ee0b98c2c71e3bdcd33edf0845fbebd9ae4192d07acd01b432135e05af0d22f3f0c5a3d62",
        "a07049b6ebd7b355479a3d802fda436b83ae6747d741cf9626f7c62f47cbd563",

        "982fb5f4af498a4a75e33a033235ea3ddb70d9d236519f883ff5b388cbef30126b98d96e93a65a26fb00d17246d18cf4e2db14a52f0f6b10e35a93beadc14ff118b02e95b38fc4736f973ba848e40b5527cb0599076d96bc578c4aada09e8faf6820bc4f562d5199974f808b7f95edca74e6b3940894a7f66534e0",
        "09c60fec5a089a23f5da3ed2492aa21fcf7aa36183850fafc15ae8c63f596db0",

        "ca88614828f8acdb5fcffab6bb2fb62d932b7808e4d9cc3139a835b0cef471d9f4d8ffc4b744dffebf4f997e74ce80db662538bceb5d768f0a77077e9700149ea0e6a46a088a62717216a14b60119dd19c31038ed870b4709161c6c339c5cc60945a582263f3be9a40cd1a04c921947900f6e266f2390f3c970f7b69",
        "fe2d4183ccdaa816b4446a9b6c07d0ba4b42ac743599db5dc482b1941f443c71",

        "ab6b92daf83275cb9c1b76cfb59fbcc8ac53188e0b6980918e7ac0c07c836ca9372d19e11251cca664bbb3c3db2e13b412a9820b65e95612042f5db24643cf9340b9808597735a1f92670ba573a2fb2f088d81087d70565574344af7576d35b2ed98318e2ca0067d4fa8e63f28045b83b6887d4ffa0668a10712ed5759",
        "744538e1ae1cd7357710b56c3bc6f1bd7a8564118a1e0f9acc30fcf0b5396eef",

        "bfd4c7c8e90858ccf9c8834abefd9c1846ca4a11966fdd139d6de24a6bebf4b19f58d5d51e52bddd0bc6f1c7f35998f44707cae7100aeb4adefe373101429da3fca1d15737329dbbf47c783a84de59bfbb2fcd75a1a148d26aebb8d3a9a76089c0f8e4d49b71a06f9e323e2cdb54888189887a44b1fa9cb32b7c8fb7c9e0",
        "58b17843bc851a721c5a258eef57b3854d02190e732d9b8e7a9f926ac409c173",

        "c5019433c285da2bb93f119e58b4f36cd1e4d99dda35dbf4f8ae39c7fe65fa0ed03bd2b96dc649472d8f1a94477ed9f29592d97c9cd54da7c790ad1af3bb5cc030b7871bc64050db779d2caf0419895bf3b7b50b8e22fbe62fe30fe7bbd6ace86ddf7b00d5d9370f20cf0f97996f4bce70bb33f1ba022cdaba0f25d55fa031",
        "f7c92a3fb7f180370d628be78de874d693f74ccc7a54c741634258d8c512fd7f",

        "84b60cb3720bf29748483cf7abd0d1f1d9380459dfa968460c86e5d1a54f0b19dac6a78bf9509460e29dd466bb8bdf04e5483b782eb74d6448166f897add43d295e946942ad9a814fab95b4aaede6ae4c8108c8edaeff971f58f7cf96566c9dc9b6812586b70d5bc78e2f829ec8e179a6cd81d224b161175fd3a33aacfb1483f",
        "8814630a39dcb99792cc4e08cae5dd078973d15cd19f17bacf04deda9e62c45f",

        "14365d3301150d7c5ba6bb8c1fc26e9dab218fc5d01c9ed528b72482aadee9c27bef667907797d55514468f68791f053daa2df598d7db7d54beea493bdcbb0c75c7b36ad84b9996dca96354190bd96d9d7fbe8ff54ffaf77c55eb92985da50825ee3b4179f5ec88b6fa60bb361d0caf9493494fe4d28ef843f0f498a2a9331b82a",
        "9b690531dee948a9c559a2e0efab2ec824151a9175f2730a030b748d07cbaa7f",

        "4a757db93f6d4c6529211d70d5f8491799c0f73ae7f24bbd2138db2eaf2c63a85063b9f7adaa03fc348f275323248334e3ffdf9798859f9cf6693d29566ff7d50976c505ecb58e543c459b39acdf4ce4b5e80a682eaa7c1f1ce5fe4acb864ff91eb6892b23165735ea49626898b40ceeb78161f5d0ea4a103cb404d937f9d1dc362b",
        "1ac7cc7e2e8ea14fb1b90096f41265100712c5dd41519d78b2786cfb6355af72",

        "da11c39c77250f6264dda4b096341ff9c4cc2c900633b20ea1664bf32193f790a923112488f882450cf334819bbaca46ffb88eff0265aa803bc79ca42739e4347c6bff0bb9aa99780261ffe42be0d3b5135d03723338fb2776841a0b4bc26360f9ef769b34c2bec5ed2feb216e2fa30fa5c37430c0360ecbfba3af6fb6b8dedacbb95c",
        "c163cd43de224ac5c262ae39db746cfcad66074ebaec4a6da23d86b310520f21",

        "3341ca020d4835838b0d6c8f93aaaebb7af60730d208c85283f6369f1ee27fd96d38f2674f316ef9c29c1b6b42dd59ec5236f65f5845a401adceaa4cf5bbd91cac61c21102052634e99faedd6cdddcd4426b42b6a372f29a5a5f35f51ce580bb1845a3c7cfcd447d269e8caeb9b320bb731f53fe5c969a65b12f40603a685afed86bfe53",
        "6c3e93f2b49f493344cc3eb1e9454f79363032beee2f7ea65b3d994b5cae438f",

        "989fc49594afc73405bacee4dbbe7135804f800368de39e2ea3bbec04e59c6c52752927ee3aa233ba0d8aab5410240f4c109d770c8c570777c928fce9a0bec9bc5156c821e204f0f14a9ab547e0319d3e758ae9e28eb2dbc3d9f7acf51bd52f41bf23aeb6d97b5780a35ba08b94965989744edd3b1d6d67ad26c68099af85f98d0f0e4fff9",
        "b10adeb6a9395a48788931d45a7b4e4f69300a76d8b716c40c614c3113a0f051",

        "e5022f4c7dfe2dbd207105e2f27aaedd5a765c27c0bc60de958b49609440501848ccf398cf66dfe8dd7d131e04f1432f32827a057b8904d218e68ba3b0398038d755bd13d5f168cfa8a11ab34c0540873940c2a62eace3552dcd6953c683fdb29983d4e417078f1988c560c9521e6f8c78997c32618fc510db282a985f868f2d973f82351d11",
        "3293a4b9aeb8a65e1014d3847500ffc8241594e9c4564cbd7ce978bfa50767fe",

        "b1f6076509938432145bb15dbe1a7b2e007934be5f753908b50fd24333455970a7429f2ffbd28bd6fe1804c4688311f318fe3fcd9f6744410243e115bcb00d7e039a4fee4c326c2d119c42abd2e8f4155a44472643704cc0bc72403b8a8ab0fd4d68e04a059d6e5ed45033b906326abb4eb4147052779bad6a03b55ca5bd8b140e131bed2dfada",
        "f82d9602b231d332d902cb6436b15aef89acc591cb8626233ced20c0a6e80d7a",

        "56ea14d7fcb0db748ff649aaa5d0afdc2357528a9aad6076d73b2805b53d89e73681abfad26bee6c0f3d20215295f354f538ae80990d2281be6de0f6919aa9eb048c26b524f4d91ca87b54c0c54aa9b54ad02171e8bf31e8d158a9f586e92ffce994ecce9a5185cc80364d50a6f7b94849a914242fcb73f33a86ecc83c3403630d20650ddb8cd9c4",
        "4beae3515ba35ec8cbd1d94567e22b0d7809c466abfbafe9610349597ba15b45",

        "b1caa396771a09a1db9bc20543e988e359d47c2a616417bbca1b62cb02796a888fc6eeff5c0b5c3d5062fcb4256f6ae1782f492c1cf03610b4a1fb7b814c057878e1190b9835425c7a4a0e182ad1f91535ed2a35033a5d8c670e21c575ff43c194a58a82d4a1a44881dd61f9f8161fc6b998860cbe4975780be93b6f87980bad0a99aa2cb7556b478ca35d1f3746c33e2bb7c47af426641cc7bbb3425e2144820345e1d0ea5b7da2c3236a52906acdc3b4d34e474dd714c0c40bf006a3a1d889a632983814bbc4a14fe5f159aa89249e7c738b3b73666bac2a615a83fd21ae0a1ce7352ade7b278b587158fd2fabb217aa1fe31d0bda53272045598015a8ae4d8cec226fefa58daa05500906c4d85e7567",
        "cb5648a1d61c6c5bdacd96f81c9591debc3950dcf658145b8d996570ba881a05",

        // SHA3-384
        "",
        "0c63a75b845e4f7d01107d852e4c2485c51a50aaaa94fc61995e71bbee983a2ac3713831264adb47fb6bd1e058d5f004",

        "80",
        "7541384852e10ff10d5fb6a7213a4a6c15ccc86d8bc1068ac04f69277142944f4ee50d91fdc56553db06b2f5039c8ab7",

        "fb52",
        "d73a9d0e7f1802352ea54f3e062d3910577bf87edda48101de92a3de957e698b836085f5f10cab1de19fd0c906e48385",

        "6ab7d6",
        "ea12d6d32d69ad2154a57e0e1be481a45add739ee7dd6e2a27e544b6c8b5ad122654bbf95134d567987156295d5e57db",

        "11587dcb",
        "cb6e6ce4a266d438ddd52867f2e183021be50223c7d57f8fdcaa18093a9d0126607df026c025bff40bc314af43fd8a08",

        "4d7fc6cae6",
        "e570d463a010c71b78acd7f9790c78ce946e00cc54dae82bfc3833a10f0d8d35b03cbb4aa2f9ba4b27498807a397cd47",

        "5a6659e9f0e7",
        "21b1f3f63b907f968821185a7fe30b16d47e1d6ee5b9c80be68947854de7a8ef4a03a6b2e4ec96abdd4fa29ab9796f28",

        "17510eca2fe11b",
        "35fba6958b6c68eae8f2b5f5bdf5ebcc565252bc70f983548c2dfd5406f111a0a95b1bb9a639988c8d65da912d2c3ea2",

        "c44a2c58c84c393a",
        "60ad40f964d0edcf19281e415f7389968275ff613199a069c916a0ff7ef65503b740683162a622b913d43a46559e913c",

        "a36e5a59043b6333d7",
        "bd045661663436d07720ff3c8b6f922066dfe244456a56ca46dfb3f7e271116d932107c7b04cc7c60173e08d0c2e107c",

        "c0920f2bd1e2d302259b",
        "3d1584220409f88d38409a29ecaebb490ef884b5acba2c7eaf23914bab7f5f0fc97ee1e6336f88dfd4d0a06e902ccd25",

        "70ae731af5e0d92d264ec9",
        "563359fd93fe09f3fe49fcf5f17e7f92aab589cdec3e55e4c3715e7775814bbbfb8c4c732e28d3b6e6404860812dc6e9",

        "69c74a9b0db538eeff64d93d",
        "88c66389ca2c320a39022aa441fa884fbc6ed2d3cc9ac475372d947d4960579a64e061a297d1831d3524f98d8094404b",

        "a4a9327be21b9277e08c40abc7",
        "751f5da5ff9e2460c99348070d5068d8a3d7ffcec7fd0e6f68f6cd4a2ef4226df8d9b4613c3b0d10a168eaf54eabe01a",

        "cc4764d3e295097298f2af8882f6",
        "10f287f256643ad0dfb5955dd34587882e445cd5ae8da337e7c170fc0c1e48a03fb7a54ec71335113dbdccccc944da41",

        "5a23ad0ce89e0fb1df4a95bb2488f0",
        "23840671e7570a248cf3579c7c8810b5fcc35b975a3a43b506cc67faefa6dbe1c945abc09a903e199f759dcbc7f2c4d0",

        "65b27f6c5578a4d5d9f6519c554c3097",
        "dd734f4987fe1a71455cf9fb1ee8986882c82448827a7880fc90d2043c33b5cbc0ed58b8529e4c6bc3a7288829e0a40d",

        "a74847930a03abeea473e1f3dc30b88815",
        "dba6f929fe55f9d66c5f67c0af3b82f17bcf58b36752f3165c16083fea8fd478ee6903f27f820ad2dd9950afb48c6700",

        "6efaf78ed4d293927eef2c3a71930e6e887a",
        "8218498ab01b63041c2ba0709e3309496124ddf0904543a9e0d9d096a750dda97f7a02208af3d8c618d4be7c2bb2a288",

        "fd039eb6e4657388b947ec01e737efbbad47da",
        "c5b3130ef8dbc580e1103fecae69c9a882d9ebf5a3def5938b07f843452a09c9f72f0dbca91d33b021cf6aa6fe60d2ed",

        "9c694943389bdc4e05ad7c2f63ceac2820e1d2d7",
        "f692c025c5c5f3d1275213c1df9bf9eb6d2188eda90ab5bffe631f1dbf70ebd628caee88b7d149e1ac4e262873979afe",

        "0fb18357b018b9bbb2cbb4cac50bc85609c92b8e7f",
        "d164306c99e3798790f0923fe92dbf2f96c3907127dacaa467c766ac75788062589272cb7690b8af2030dd8bd61a3df2",

        "26cb40a460e2e727aeb867e0140d0f34790110deb5d7",
        "af2a42a4c67c3226c55b89605b0dee27e796c2792115f6097203db5aed89e35f563a8246d399fde00c2a5b97ed5a5e17",

        "6690a3a0373c829facc56f824382f4feed6eb184642b4f",
        "84e1b68bc9e2daefc19b567dec911ef46f5f37a74fdbbb6155e7e646f2735df2ac44e239689eb5b536465dc571e55cb2",

        "7d80b160c4b536a3beb79980599344047c5f82a1dfc3eed4",
        "041cc5861ba334563c61d4ef9710d4896c311c92edbe0d7cd53e803bf2f4eb6057235570770ce87c5520d7ec14198722",

        "02128283ffc0cfe254ac8f542be3f05fbe4e855dd22ae98a81",
        "3840981a766d725f83d334e8982965033a5fbb5107d94ffef33b1f700cd46348091a49f6620c37ae3ef5b20513494826",

        "27911dd0a6843ccae965d876aa1916f1dcd71e518f7f2197152e",
        "f59f8428555984d1526cded8129c649fb1b683d35cec7c5e1209441a6a9e7c17f0784151b5ab8a8c492b402a3acb98c4",

        "d9378bb66e8c8dee556d691cbc9fdddd6333ca5d50668862c3c57d",
        "994532d1a557e990b1cc9e0395a2ad8b05619ca322db9da3c4ed2ee194c051d04582fde72dd2b8f674cf6ec958db75da",

        "ae1828047c5f82a7b9712f3399832124b892f2f7aea51c8fe3536cd6",
        "d51111f8bffb44d81ad19683198f29d2033144d3cd856c749cac5b9cae0e712f500f8d0ef813f38e305ce175a7d6162c",

        "7dd2d76fa054cf461e132e9ef914acdc53080a508cdc5368ab8c6224ff",
        "6c0b3395e4c86518ab0a06267320ee9ec95e50385b7a2527ddaa1bd0ead262c56122d4f4eb08b0ae22b3ee7e6f44dd18",

        "6fd72888a021f36e550967cb5605b55b78657c9272d93c3ded340d67da6f",
        "0551583a5b4007401c77ef4382fd8e245c9cf12e976c9766af6b7ae3c7e07a82b3079f903b083d5ec85cb94e46a85ac0",

        "d500eb9546553619cdc31e0848c502db92d547efef3ae5eeaa22258afcf0a9",
        "5edde2f94f8695f277ec05efcc00761fafd272200aed0e63d221c2b6c65b4972a6526f9a1f2e6ace0e81938f043fe877",

        "6189597e0198a18c65fa0bdd0797f13037c75c4058b7d3454c0f71bd2dd13b6c",
        "110630ca7631b7620e6bee6ed6e929098965571936c34829484983eba9532b8175528c228c57439453f027a4f7c83ca3",

        "243b941d748541af303f8e9d2c371cd03e437d62a9df485ddc176dc65da8c7da00",
        "5884201f7a555ea3c5deeb019fd9e8c161e1b89756045e475b141ec5135ce5a41c93e5e1f79534d36fd8345ba434da43",

        "2dc3d789582c1a806c3b491d5972ef8f1733f1f5e02866dc9de2a8029ec0ab608d13",
        "05a3903b519cdf679120c7ccb4ef178b58e4502fcd461360988fa06669294851e629d9dd3e77ffb73d24599d5d3edd36",

        "e5b3f6962fe57230780b3d55b29effe0dfebde2c81ba97d4512ecdbd33eca1576a7f82",
        "7ac2776afb74f55bbc4f6eccf825ee13ac7445fb54974e6c24ebc0f03fdcd8530199a61106a31b4279e02201ee0f54fd",

        "da03486aa3cebbd6502e9f5a6f0f835e973a581befcc1aadefe7b3696ba71c70cd58c584",
        "02c44ceec0bb7dc0f664ebe44230192b5b0bb646bb944d23fa1ff3586dc0523fa9d7f0dd6df5449ab9edd9a1096b07dc",

        "3c686d321ba66185cdca83ba9f41984fa61b826ef56b136e13f1239dadf6e03d877866ccb8",
        "ad624edd9f2c3a32b56c53d9e813c01d66bcfe424c4a96907d52ac1ddd68370ec86dac67504a90e8a8e75502e01081d2",

        "4dcff99fac33840f6532547fb69b456902d6718fd5d4538e23462db6d00da61975f2b8e26298",
        "cf37dd27997c1bb7e6dc405170066e74c6ce517c029ed8dce126d025da74e0b8e86da567e8d7d8d5b5d3e2a546df7489",

        "2799f672328834d7eaef9439795d35ce93c9094f58ded9f17c968a97a50a9e461489fed988e7f6",
        "85cfc23c97cb13910b808e7033809a45aa0b7f7138de618c2ca622c8b813c988e264af3b96c7925dcbd1d2761757d800",

        "c7e947507822f28a562745a8fe6fed6cb47d73145804c894954e21245cde04fa9155a35904926aca",
        "8bddf3baebbc5b04fe0b0a9c3c2b730abe918ce4892d2843c613ee96da0228512f0d1307c7d1a8922e79a92e957dd18e",

        "6c497bf6ff69cb39e3faa349212b8b6691ca237905ac0099c450b6d33abf362bedb65bdeb307bfea23",
        "3639fab6191b35246278522cfacee0cd5b15580a26c505ae3c46b4b1c2572016b48f1b012bbbedec47916950fbb33a1d",

        "d15936f3b0c9018271812b4c81453c4457c7edd110bcea7f5735d6f5882d8f27155eb4cc285a65138ad6",
        "0293eeef0aa3392c93d9c6ca89c08b317622572d4de2286a4b9ae6c2f9c9e0e64ee6c483d4f10859077e3c6868430214",

        "df18139f34b8904ef0681c1b7a3c86653e44b2535d6cecd1a2a17cd5b9357be79b85e5e04dd9eff2ca8b9a",
        "db9e171d6e3336631c9ceec6b4d732ce62b015939269fb69fae7d22725500e8a2fc9f1459cf0a31fb9d16d7c44583f52",

        "0459dcbc149333ea2f937b779a5f3728148449a9aea3662cdd2cc653ce6a2050f9c0d54bf9326c039b263eb9",
        "464ba409fbb45e985f84ee24662eb7c042c3c2ad9649f1ac4a8b2be9c07d37ed2e4284362057493f6a7e52c356b05bc5",

        "eb3f7002c8352270340b8da8643622e5f7e32cdb208a0dec06c6cb9e6b64cc4d8cb9de1d49397b3386464a25d1",
        "a26bd76ce42d818dbec462d8fe7cdd957e6b84ae8750fb5e1c9c76bc6000e23737e073a59b4600e5056524edc667909d",

        "47e3e3d8c68ac9d9f4b3759d8c7d9dd901e35b096ee4c8b6cbe0cdf467463630926c08289abe153bfa1bcde3cd7c",
        "b504ef475a568f9caba8352a0b2d243acdf3d2b41d8890a6fb3abb8aa28a29e0c7527d20e2d79b25b400ec27c314db72",

        "838d9c181c5ab59592723bd69360e0d7fd15232beada7591ea899ac78ffd53a32fc73a5fe522ed35d92a6e2bc148ca",
        "53e99e1158d59032ffe4b5ea304c7d2f7a61b6b2a96ac97832ca26013549fe3f7dcdf926bd74ceabe4f1ff172daed6e6",

        "a90d2aa5b241e1ca9dab5b6dc05c3e2c93fc5a2210a6315d60f9b791b36b560d70e135ef8e7dba9441b74e53dab0606b",
        "4a16881ce156f45fdfdb45088e3f23be1b4c5a7a6a35315d36c51c75f275733319aca185d4ab33130ffe45f751f1bbc5",

        "8c29345d3a091a5d5d71ab8f5a068a5711f7ba00b1830d5ed0bcdfb1bb8b03cd0af5fe78789c7314f289df7eee288735fe",
        "e27b39a96255ff69c45285fca6edaaa3954ce32c1e3d9b1f60c1b6676594bb45caf0889fc11daf93a1b60746229689dd",

        "32876feefe9915a32399083472e3c3805ef261800b25582aa7c36395fd3ec05d47b49c4944bbcc2b8b5ebd081f63ae7943d0",
        "f96433cdb69a607433ea2eb77d87d3328867dc4076b67ccf17f50f9e08e89a86624b60f2ecdb8affcd431fc13173fe75",

        "e2e77eb54f321f86f52ea3d3c8cdc3bc74d8b4f2f334591e5e63b781034da9d7b941d5827037dee40c58dc0d74c00996e582bc",
        "a352ab33ca730482c376bdc573c9d1dc6d3597f9be9f798b74a57beaa8e9c57b78ee6761056eb67363e882fefcad4fb9",

        "da14b6d0b2ec4cf1e7c790e7f8f4212b8f4d05f50e75e2a56a5d70623c0d2e0115a15428129109b3b136d756e38a5c8463304290",
        "aae7ad977e17ac0e560c0e0186433420f9fddcd191b9e91567cee05df88f1e1aee50424a313998a873f7a9c289a02217",

        "2db06f09abaa6a9e942d62741eacd0aa3b60d868bddf8717bef059d23f9efe170f8b5dc3ef87da3df361d4f12bfd720083a7a035e8",
        "85d4e3e5abcb1b59ca6f551eb43b43ff64890511f73a9083a2ce6e9c2861c6e9664c765629024f4b01b0cd1594a5981b",

        "26bad23e51c4560c172076538b28716782ee6304962f68e27182048948d5c367a51a1c206a3e9b25135b40883b2e220f61cb5787ed8f",
        "a44c7f84ab962f68283404f8c5c4029dbc35d2138e075c9327580baf89f292937bf99422e45756b3f942bf0a5ae4acb6",

        "77a9f652a003a83d22fb849b73fed7d37830c0dc53f89cea7dbec24e14f37197765206fe0e6672016e4dec4d9ebbe3e1b4423771a5d0a8",
        "29c8bb39bb2aad419a00a80216ec71ec5ec9ab54c41927e3e3f2f48f079a5886d7fe89db98c807ab686d2339001d6252",

        "268c7b3a84849fec5c769bc4ad377dea10c9d20c91dd17fdbd9670a2fc909d0e212129ec40dee41dbf6194a3b04ae8be5e84ad5426ca4496",
        "0dfc6ffcf4a387ec09ff862c6139a6f7ac77abb2b5e1f6dc814eb71525f8657ac74a7697c2975c70a543af0e227d03ca",

        "b8324341a6891a6b5e001a7d2ebba6e02e8335c124185309a4c9e9907c43bd8d4fa73c527fdf783650316dd24b148870e1436ac05111e9cdcc",
        "6278d1cc17fb6d54129d04987d4774fa846dcac4ba8b6b72f41e63dc387ce0081ba29fb2c17c6744edae24e669cc9e75",

        "5ef8b3d79d299bee2c414560c7de626cc0d9fb429884aa69cc30095ef1f36b7e03a8ca25fb3601189f163b209e0facf8dc447f690b710fb47b72",
        "7ec9505f33f4a5493574422de078e0490b61be8e8d6f158192bb7d2bdc2dc335598dc88d9b443cd1c14b883a77119df1",

        "ad7321c9a8b8f0bfe100811114270daad57f6e88772326b62d88a37a6f55c2cf9f759115ed6a590878e4dcefb592db151538db7de20229d26a181c",
        "3782d2caa537294e809e9df837b1b07e2f1df07d0f4c12e12459f56eeaa478d5b3a41e519d9414eafa5ddd5661c831ba",

        "0719d9664541f0a824f71c83b809bb6afc973c9f7428e1ed11f7c29a558e1698b796aefb49eec2b098faf06bd43e82e1312bf0388c38a5bb523506d3",
        "362c05f678df92883d56e19221391fb00d0f0afcec51d3e0feb15ba2fb60693b09d69118af649648933259d7b1e240ab",

        "5415c2596aa7d21e855be98491bd702357c19f21f46294f98a8aa37b3532ee1541ca35509adbef9d83eb99528ba14ef0bd2998a718da861c3f16fe6971",
        "8f9fd7d879d6b51ee843e1fbcd40bb67449ae744db9f673e3452f028cb0189d9cb0fef7bdb5c760d63fea0e3ba3dd8d1",

        "b979a25a424b1e4c7ea71b6645545248498a2b8c4b568e4c8f3ff6e58d2ac8fbe97be4bea57d796b96041d1514511da5f6351120be7ab428107ef3c66921",
        "e248a64b6ef112bf3d29948b1c995808e506c049f3906d74c3ee1e4d9f351658681901fe42c8e28024fe31014e2d342b",

        "e64c7bb9cd99ce547d43de3cc3b6f7d87a2df9d8a4760c18baf590c740ec53c89bfa075827e1f3f2858ce86f325077725e726103fbe94f7a1466c39f60924f",
        "d1e5a72d2595f38714c6198ac14f8a5cdd894dcf9b4b8e975174b100df7bbf4f7ce291b4864f27c0b64e6330f6c1c82c",

        "91b7a1fd0e20072d9c5be7196e5eaf8df36fdf145895b30d4e4c02010d7c663499ac9d7a44732f4c7430511ba6fb0ae4b3dc9405523a054fdf962f5c5b79c423",
        "07c2e0aeae30da83b5a6b320aa1cf727b10c2034583d7acda55648fa3daa017aa15588b6e2149101c56e3d7df7c76df1",

        "5bbc2d4efe63cbfc9fc221dd8d8384075a79c80a27d6a8c5219e677f4c5bb8338013dc2ab1770acf735d13c0bc704621ec2691350cf3ea2f53bded45ef8fc70702",
        "dd0bbfe4b799642191abe316df9d59a3743566778b4459c51c3be3f658bdce45516ad188fbe1a8cad8a1fa78f8ebb645",

        "129549278e8976c38b5505815725400c3d2081edf141ad002e62ff299d9a0743f9c9f25971710b194dc88285d50b6cec6e140c19072f51cab32a9f6497abd3e407c6",
        "ca26aec527fadcd5ebeb4eafa7c102f79a3c2edb452afd04f6162dd7a17bdd1aad7d616508a89a3ec6a40791d915acc8",

        "b9a9f378adeff4337bc7ec10d526c6dda07028375549f7fda7a81d05662c8a0da3b478f4152af42abb9f9a65c39da095abb8161ba6676b35411234bd466c2914e00370",
        "99914f684e0b317f9338af0c71e9655a3af7153eb9fabaae61454bf8de9e0bfd274c1eff6c4b550e47afcb3b20fa7d9e",

        "101da5b09700dcadf80e5b7900f4e94c54d5f175569a854e488aa36fb41ab7220b0662178ca07a596768528123de3b2a3d944aa412875cedfeaf58dcc6d5b4a033a53b69",
        "d3e32c9b271e11e4968397d85d76938b974ac1ba55bcbe8d7b7da02dbd7e3b9c9af0d98bbd7e50c436fcf9e3551e3432",

        "14761bbc5685b5de692973e2df7c9c4750889c19a952f912c817890546d5e37d940d13a14ac7925abbd875b8cd60e4920896ce6decc8db9f889da2b5489e1d110ff459d885",
        "272222ed50631aff465c0e6fe49ecdfdca983bcb7231e50903e200b335b845108202c28315912c9c4fd50e2c6f13a9ea",

        "ed538009aeaed3284c29a6253702904967e0ea979f0a34a5f3d7b5ab886662da9b8e01efc4188e077c2cdeb5de0a8252aafbee948f86db62aae6e9e74abc89e6f6021a4db140",
        "8361b680243b1661d6f1df53db363cae41c2ebb7438c00606d76b9c2a253faa1f09d6f520d69d692ec1dca0c7885119c",

        "c434d88468f1eda23848d0804b476933f24baeadec69743dd90d8455f1e1f290f6f1aaf3670c4c74f76d3ab83e9bef21ad8d9208c712ca478e70d5fb3c4bd48834c969dd38f484",
        "9c26e96fcc09a76cc13d24ad25c9cef4300e96e97e4fb59b441baffed07f6a70b1464f2548c7fd7839810dbb9e9c1e18",

        "3064e5ba1e7751bf7198e0811ff4d4ca17d1311c25d9c3a316b562691cde75c974b0b52645c134ddcc709d77b6c1bd24cd684265d723c308bb4d0159e6b16d97ed9ceaa57436d302",
        "1ea779739b204abe911b4923e6f60fece271eedfc7f074fe1919f0cbc6ce2a99234b003389520884b660165f5a1e80f8",

        "89d9521ad84b1c9afc2fbd0edc227193acd3330764b0d2cb71bf47c7aac946af85be13858b55976009f3b36b09ced4308052c817c9c4d0295225f61a9659a0874b88667cdcc5213919",
        "4209bb8f869f6f17c8d5c368c489ac51a75e24a85a12de1b16fefc292ce636ff8fa360e82f05684f6b0b074ba370a933",

        "3216662da0227993d88288187177a0287de4eccf245d7c718b8045bbfb8869d93f1fb9e94d7478b0298e628c07e0edaab01dcf79264dc05f8b2181aa3f831dc949726fbcf80de4c9c9ed",
        "64c45e018cfbc88f8f4ffe3cef0df3a94aab3049fafae28e28efbb2a4b94809eb302caf901010abfa194f72965663d35",

        "e776e6749c5b6c7def59cb98340984539280a9874f80412d4df0ee73d58acd1094d49ed4e35125834cf8cfe349e599144e4f2e200aba4fd3eb6d78cde027c1d5620e0270b5e83ab26b8d32",
        "94bd67b7f2587b0bda5487cc45d00e4365f1ee40073cdf0d23a5ea3fba01eef42a46bfbac5306d67be02d8d918ae5c9a",

        "5d8f84b2f208b58a68e88ce8efb543a8404f0ec0c9805c760ad359d13faab84d3f8bb1d2a4bb45e72c0ec9245ffda2e572f94e466cffa44b876d5c5ed914d1ff338e06b74ad1e74d1405d23d",
        "947350307748c29467f00103d0a07c3c228c5f494fc88fe2352ca5d10449d0dda7076780c05439a09694eb528d1f477a",

        "357d5765595065efe281afb8d021d4764fba091adde05e02af0a437051a04a3b8e552ec48fb7152c470412c40e40eec58b842842d8993a5ae1c61eb20de5112321bc97af618bbfbaf8e2a87699",
        "32286970204c3451958f5155f090448f061dd81b136a14592a3204c6b08e922ee5bb6d6534dbf8efb4bb7387092c8400",

        "a8cb78e1485cbb7a9474c1c1f8e0f307cda5139a7e947df5ea20ac330a6dffcad4a9bd755f9f58724789eeee532615be550dd84f5241fde0e3058aeedbf287f02a460445027f5e6b3829bf71ecf4",
        "51168bfeef8a981c0def0c4cb067baf15ce5feb8d5f7e9d6076b2836267391aee1fd3a0b5d3434ceb5cf2d6fa06fa063",

        "81acca82545e767ab59dcc750a09849cebad08ff31c9297f4fd510ebe6c27769938319180ccc66f36b1a7cf9c9f3538b0f6f371509f77cf0bc4d6d87facc85b933f2e27f8e1bf6cf388f80c0fcbfba",
        "4ae44d6509986893a8414753b57d11f9c554d89c15ad6d70687c56c6c2ac73537acbb0d51f48e6bea6cf762d58890d7a",

        "94987498b1ca87a6f3fa4b999db726115c455d0ec24029b2f5810e49a94668864b8c470f7fc07c3dcd97f41c973b45ba4fa7879ee7546596881573b6863fc39d940eb3fa3444084f721341f5d23d2561",
        "a733b118be72a187ddcbe5ba67e04b589f9cd9f8482c4bd9d64c580aba7d19d2d1f9c1ddf95fe6efdeffd44f67fcabb5",

        "de6b32c2d40d0659166db235259b530ea43f44e75d8b3e9e856ec4c1410bbea3696964af8b6c5dfd3304282369a4bc4e7cf66b91fecd0c7c105b59f1e0a496336f327440980a34614ee00fff2587d6b813",
        "17ba30c0b5fc185b3245313b83dd0481145953101128914765784af751745b8a2b6a90a434548f3adaf1f07f18649890",

        "854211bedacc19f77b46cfa447a4ad672ea9b643f09f5cf5274ba28888207e2466b38127776fb976db8ad7165a378df6ee1e3a0f8109c9aff7e0d6126fd71333c6e6ebe15d7a65151d6a4a83b82c8a6f3149",
        "ca85632a9f7c32ac4705c6458770025dda4fd07a8d5d6921b897b0da490d64400587649f2d20bf608b9a18d071b63b48",

        "822373d9d3d5b06a8da48a43095740fb98c9caf717350fd2c3b058024ff705b9346b7f0a495a6d4d93802bc45ece777f8c6a6e7c2ef6b8135115ff911a2ba5241665b6f7cbfa1b9d93b011b3aaa1dac1853fb2",
        "6e84587c8c6e54353a6032e7505902ef7f0f0538dd1bb32922e13a7d4d98c47a541015381eab27e9186398120da7fb32",

        "c04b701f688092bbd1cf4217bc4b5877f2e60c087bdac46611482a61d51f820140403bc85be0c336332da0938734bde8c502014f3509266c73c6c93c22a1bd0ddf15a5ce7410c2894e9d092e32c079922ba1abb7",
        "75c585503f15a526113608bc183180b1cb80f4d1b466c576bf021b1ce7a1528391f70e10446681849fa8a643cb2b6828",

        "009dd821cbed1235880fe647e191fe6f6555fdc98b8aad0ff3da5a6df0e5799044ef8e012ad54cb19a46fdd5c82f24f3ee77613d4bed961f6b7f4814aaac48bdf43c9234ce2e759e9af2f4ff16d86d5327c978dad5",
        "02a09d37d31e4365c26bec0eaacecf29eea4e8d21ab915dd605248764d964f10ebb8fafdb591982d33869a1d08a7e313",

        "0b7dd6709d55e0d526d64c0c5af40acf595be353d705be7b7a0b1c4c83bbe6a1b1ec681f628e9d6cfc85ad9c8bb8b4ecac64c5b3a9b72f95e59afefa7bcec5be223a9b2b54836424afde52a29b22ab652d22cce34b39",
        "5c84ae39d959b79555231746ad5b33689a31720ed0070f6772147977edd0aead07fb8b7b71b0bd587ebc5c1a80d564c7",

        "3e9b65d7bf4239420afa8639c8195b63902b24495b95c4143978e49843d88a92d1feed2eed1a88cd072d6d04ea26dce8ee4b14896fdb69bc7ff2971ed8ac5655148d2e9921218d74efdf17c56b533d0bb17d11e07d7458",
        "ab7890d1b51af10285752bf9da5eee5c3e87a285dc33262d0261aa9a575f303e94845d7ab21b48f4e6884568cd78b550",

        "9436da433d1ebd10b946b129cb34bccec9b8f705aaba3f8561352ed36a8449aba2dd7ba15b1bc308b0c02913163af63a346524dff5521432db477f529606afb5d552efc95cb040db566b4d39eddaa19319e518a7b5c6931e",
        "968ae9104f9c907c5a72936250dfedd62cd04f6e5ddd2c113490808a11884449aaef5d013ea3993a6cb6fc5c08754408",

        "37254bf9bc7cd4ed72e72b6bb623a0cc8eeb963d827aef65ad4bc54913235b6d3551533ce33421aa52ffbf186eb9a2787188eeb1b52ee645c6d4a631bc071415c80014940c28fbfeb0db472c326c8dacfd6ab21f3e225edef3",
        "975e10fac9aa77b780e5f6c2151ec4a3c72ff26e41233cc774c074df1b78cce5af1191ba955a0bce15926ae691b0ffe7",

        "79e77cd08a6ef770bbe4bedf61557ea632b42d78637149670d4d6157d56ed7b2ccaee45d9439dcebc557b4118e86c15aa0ccc21c474b21abda1676cc56434d6d46422993e66dc99387dfa985358accf69884b9dd18a2c4d04448",
        "94729f5f99a54f5a3ea69233ff9d522392d4596eb6ac2bbb07492ece3c67317412bb47ae317ddd20536c3adc003862f1",

        "64b76cb554f6becc238a3fcfc3eb97993667ec82fdc3fb28d42567709c3250c7997328aeddfdc2750451ac462281bf66fa94f4b8712c7a8342660574f20268e707c466627519c56259fea55be91e10faab3ad2ade6ce8b6557f202",
        "26d48ef5067d704ee9e2a64e399de23068908b3c911ffc4056c168362c37385c92d37d51354b6505a82c4d22fec37eaa",

        "3df27829bfb1ab7d381f146b30370ef56b392b73b35b1be5d8bbcf88f499dda7f3c327b45350b8972991ee466545de96560cf451711fda884e3d9b2af3e909d655d25cee1c931beda79c40fa507097bdf1126771a7b9543ad5cb84b9",
        "5fa4ebfa24150236c03409f0857b31cb95b0150f381c8858b01559957b1268f73c698709233e6b15468675a102d0c5e5",

        "b00f4e67ca08ccfa32b2698f70411d8f570f69c896e18ec8896cfe89551810543303f7df0c49f5b94783cce7df8d76d0b88d155633302d46003711f233339b1c9a8c20164ec8a328890a4932b7d90d92d023b548e4820558f8bd327010",
        "eaa756b5892fdfc793d74e3f9f4d6c7a5a6a2241dd11e0c38ced59c8ec7be377a41d1d06774a5970ce9722d8e119d0ad",

        "a4f95f6a46a9cbf384a7e98e102d1fdc96839d1bf26b35a5a0bd6cb9734fd17e8a178d4581943c0fe469fb4fe94cc2f15e1ef59ae05b35324eb57ca07dfc69d42d41d80b3c3bb64e1aea143c7d79790a56697dc803ec93e6c68f27f6761c",
        "1aff8d9c64f0c162ed0195d1f3a342a010d14be0636903c48020ba42de1cfa8b98ae2142d89af3e69e9eb4c735857dd1",

        "02713084bf93fdc35135515243c3bc0f4b2b447f2d3461c0dc104cbfe23479ab036762a91d1987c953f7b3386abc80b8734a1d4eabf94f3a9f2fb62c943152b5253846fc2ec8dbb2e93dc74857a7b05fe2d7ec8040ba8b0d9ae69777ee739a",
        "84da02114e341a3636f00822b32bd21a8a1f7b39f2956bd97f39346fedf9aae63b304c65c93a541e8bcda549576d5f27",

        "00ce225eaea24843406fa42cc8450e66f76ac9f549b8591f7d40942f4833fc734a034c8741c551d57ddafb5d94ceb4b25680f045038306e6bcc53e88386e2b45b80b3ba23dec8c13f8ca01c202ae968c4d0df04cdb38395d2df42a5aff646928",
        "81d6e0d96575a9b8ca083ee9ec2ead57ddf72b97d7709086a2f4a749d3f61d16423463487562c7f09aba1b26e8cae47b",

        "7af3feed9b0f6e9408e8c0397c9bb671d0f3f80926d2f48f68d2e814f12b3d3189d8174897f52a0c926ccf44b9d057cc04899fdc5a32e48c043fd99862e3f761dc3115351c8138d07a15ac23b8fc5454f0373e05ca1b7ad9f2f62d34caf5e1435c",
        "00e95f4e8a32a03e0a3afba0fd62c7c3c7120b41e297a7ff14958c0bdf015a478f7bab9a22082bfb0d206e88f4685117",

        "2eae76f4e7f48d36cd83607813ce6bd9ab0ecf846ad999df67f64706a4708977f0e9440f0b31dc350c17b355007fed90d4b577b175014763357ce5a271212a70702747c98f8f0ad89bf95d6b7fbb10a51f34d8f2835e974038a3dd6df3f2affb7811",
        "eb396cfaf26ee2775af3c9a3a3047664ca34cbc228ccbb966df187d518717df6a328ecc316ed0ed09b170080eccc486f",

        "093e56d33bd9337ad2ad268d14bac69a64a8a7361350cf9f787e69a043f5beb50eb460703578a81be882639f7e9ac9a50c54affa3792fd38464a61a37c8a4551a4b9ff8eed1f487ef8a8f00430e4d0e35a53ff236ce049b7a3abdc5cd00b45c4f3d49b",
        "4a339128486e5b274fc4ed538c0ec9e57f780e9c500c5f92b04ae81a22fbeebf3785259a0bb3b6d9b47f31873cd8dffa",

        "0593babe7a6202077c026e253cb4c60ee7bad7b1c31a20da7aa0ce56b622eb57ed07d21a7f0ae6c6fe3c8398cc48353decfb287f1204e024fcf82a13059953b9f85797ab2217dc8dab34a13226c33104661c1ca79396e7d97e91039d32bafc98cc8af3bb",
        "5981815c1618cc49cd5cf71a4b7b32b8cd7b7ef553bfaef2149ac723ff2582a2d345c5bd05943e155ced1e5f091c5601",

        "ae1828047c5f82a7b9712f3399832124b892f2f7aea51c8fe3536cd6a584b4a7777cc1ecac158c03354bb467b8fe2c8ce2f4310afd1e80fec51cc5ad7702566b2c5d21bc6571e4b8e7c59cb4c9e23f1ecb57ada9e900e4aa308874c2d12d34be74c332bbce",
        "7257f5bfa7d33d1cf5f4550d0cb78750e84c5b7d25027da6acec64bdf30879a0e5c97fe7c468e743aa5ec2bddb29d193",

        "3bceedf5df8fe699871decb7dd48203e2518fb0fce0f865f46adce5c133a921320bf40915456204869a3ceb5fca3ed40e0a41a64b8951f0fc580694cfc55bd1f5ce926b07e3e32ac6e055de9b961ce49c7ee41e06b024559b933a79518192e969855889c85d1",
        "60d7f8bd85fb7a13701db5aded2b7771ab5e476ec34f1fd4298978defbd2b31bb2979391559a164b3ed28f6a39031a11",

        "6c36147652e71b560becbca1e7656c81b4f70bece26321d5e55e67a3db9d89e26f2f2a38fd0f289bf7fa22c2877e38d9755412794cef24d7b855303c332e0cb5e01aa50bb74844f5e345108d6811d5010978038b699ffaa370de8473f0cda38b89a28ed6cabaf6",
        "b1319192df11faa00d3c4b068becc8f1ba3b00e0d1ff1f93c11a3663522fdb92ab3cca389634687c632e0a4b5a26ce92",

        "92c41d34bd249c182ad4e18e3b856770766f1757209675020d4c1cf7b6f7686c8c1472678c7c412514e63eb9f5aee9f5c9d5cb8d8748ab7a5465059d9cbbb8a56211ff32d4aaa23a23c86ead916fe254cc6b2bff7a9553df1551b531f95bb41cbbc4acddbd372921",
        "71307eec1355f73e5b726ed9efa1129086af81364e30a291f684dfade693cc4bc3d6ffcb7f3b4012a21976ff9edcab61",

        "5fe35923b4e0af7dd24971812a58425519850a506dfa9b0d254795be785786c319a2567cbaa5e35bcf8fe83d943e23fa5169b73adc1fcf8b607084b15e6a013df147e46256e4e803ab75c110f77848136be7d806e8b2f868c16c3a90c14463407038cb7d9285079ef162c6a45cedf9c9f066375c969b5fcbcda37f02aacff4f31cded3767570885426bebd9eca877e44674e9ae2f0c24cdd0e7e1aaf1ff2fe7f80a1c4f5078eb34cd4f06fa94a2d1eab5806ca43fd0f06c60b63d5402b95c70c21ea65a151c5cfaf8262a46be3c722264b",
        "3054d249f916a6039b2a9c3ebec1418791a0608a170e6d36486035e5f92635eaba98072a85373cb54e2ae3f982ce132b",

        // SHA3-512
        "",
        "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a615b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26",

        "e5",
        "150240baf95fb36f8ccb87a19a41767e7aed95125075a2b2dbba6e565e1ce8575f2b042b62e29a04e9440314a821c6224182964d8b557b16a492b3806f4c39c1",

        "ef26",
        "809b4124d2b174731db14585c253194c8619a68294c8c48947879316fef249b1575da81ab72aad8fae08d24ece75ca1be46d0634143705d79d2f5177856a0437",

        "37d518",
        "4aa96b1547e6402c0eee781acaa660797efe26ec00b4f2e0aec4a6d10688dd64cbd7f12b3b6c7f802e2096c041208b9289aec380d1a748fdfcd4128553d781e3",

        "fc7b8cda",
        "58a5422d6b15eb1f223ebe4f4a5281bc6824d1599d979f4c6fe45695ca89014260b859a2d46ebf75f51ff204927932c79270dd7aef975657bb48fe09d8ea008e",

        "4775c86b1c",
        "ce96da8bcd6bc9d81419f0dd3308e3ef541bc7b030eee1339cf8b3c4e8420cd303180f8da77037c8c1ae375cab81ee475710923b9519adbddedb36db0c199f70",

        "71a986d2f662",
        "def6aac2b08c98d56a0501a8cb93f5b47d6322daf99e03255457c303326395f765576930f8571d89c01e727cc79c2d4497f85c45691b554e20da810c2bc865ef",

        "ec83d707a1414a",
        "84fd3775bac5b87e550d03ec6fe4905cc60e851a4c33a61858d4e7d8a34d471f05008b9a1d63044445df5a9fce958cb012a6ac778ecf45104b0fcb979aa4692d",

        "af53fa3ff8a3cfb2",
        "03c2ac02de1765497a0a6af466fb64758e3283ed83d02c0edb3904fd3cf296442e790018d4bf4ce55bc869cebb4aa1a799afc9d987e776fef5dfe6628e24de97",

        "3d6093966950abd846",
        "53e30da8b74ae76abf1f65761653ebfbe87882e9ea0ea564addd7cfd5a6524578ad6be014d7799799ef5e15c679582b791159add823b95c91e26de62dcb74cfa",

        "1ca984dcc913344370cf",
        "6915ea0eeffb99b9b246a0e34daf3947852684c3d618260119a22835659e4f23d4eb66a15d0affb8e93771578f5e8f25b7a5f2a55f511fb8b96325ba2cd14816",

        "fc7b8cdadebe48588f6851",
        "c8439bb1285120b3c43631a00a3b5ac0badb4113586a3dd4f7c66c5d81012f7412617b169fa6d70f8e0a19e5e258e99a0ed2dcfa774c864c62a010e9b90ca00d",

        "ecb907adfb85f9154a3c23e8",
        "94ae34fed2ef51a383fb853296e4b797e48e00cad27f094d2f411c400c4960ca4c610bf3dc40e94ecfd0c7a18e418877e182ca3ae5ca5136e2856a5531710f48",

        "d91a9c324ece84b072d0753618",
        "fb1f06c4d1c0d066bdd850ab1a78b83296eba0ca423bb174d74283f46628e6095539214adfd82b462e8e9204a397a83c6842b721a32e8bb030927a568f3c29e6",

        "c61a9188812ae73994bc0d6d4021",
        "069e6ab1675fed8d44105f3b62bbf5b8ff7ae804098986879b11e0d7d9b1b4cb7bc47aeb74201f509ddc92e5633abd2cbe0ddca2480e9908afa632c8c8d5af2a",

        "a6e7b218449840d134b566290dc896",
        "3605a21ce00b289022193b70b535e6626f324739542978f5b307194fcf0a5988f542c0838a0443bb9bb8ff922a6a177fdbd12cf805f3ed809c48e9769c8bbd91",

        "054095ba531eec22113cc345e83795c7",
        "f3adf5ccf2830cd621958021ef998252f2b6bc4c135096839586d5064a2978154ea076c600a97364bce0e9aab43b7f1f2da93537089de950557674ae6251ca4d",

        "5b1ec1c4e920f5b995b6a788b6e989ac29",
        "135eea17ca4785482c19cd668b8dd2913216903311fa21f6b670b9b573264f8875b5d3c071d92d63556549e523b2af1f1a508bd1f105d29a436f455cd2ca1604",

        "133b497b00932773a53ba9bf8e61d59f05f4",
        "783964a1cf41d6d210a8d7c81ce6970aa62c9053cb89e15f88053957ecf607f42af08804e76f2fbdbb31809c9eefc60e233d6624367a3b9c30f8ee5f65be56ac",

        "88c050ea6b66b01256bda299f399398e1e3162",
        "6bf7fc8e9014f35c4bde6a2c7ce1965d9c1793f25c141021cc1c697d111363b3854953c2b4009df41878b5558e78a9a9092c22b8baa0ed6baca005455c6cca70",

        "d7d5363350709e96939e6b68b3bbdef6999ac8d9",
        "7a46beca553fffa8021b0989f40a6563a8afb641e8133090bc034ab6763e96d7b7a0da4de3abd5a67d8085f7c28b21a24aefb359c37fac61d3a5374b4b1fb6bb",

        "54746a7ba28b5f263d2496bd0080d83520cd2dc503",
        "d77048df60e20d03d336bfa634bc9931c2d3c1e1065d3a07f14ae01a085fe7e7fe6a89dc4c7880f1038938aa8fcd99d2a782d1bbe5eec790858173c7830c87a2",

        "73df7885830633fc66c9eb16940b017e9c6f9f871978",
        "0edee1ea019a5c004fd8ae9dc8c2dd38d4331abe2968e1e9e0c128d2506db981a307c0f19bc2e62487a92992af77588d3ab7854fe1b68302f796b9dcd9f336df",

        "14cb35fa933e49b0d0a400183cbbea099c44995fae1163",
        "af2ef4b0c01e381b4c382208b66ad95d759ec91e386e953984aa5f07774632d53b581eba32ed1d369c46b0a57fee64a02a0e5107c22f14f2227b1d11424becb5",

        "75a06869ca2a6ea857e26e78bb78a139a671ccb098d8205a",
        "88be1934385522ae1d739666f395f1d7f99978d62883a261adf5d618d012dfab5224575634446876b86b3e5f7609d397d338a784b4311027b1024ddfd4995a0a",

        "b413ab364dd410573b53f4c2f28982ca07061726e5d999f3c2",
        "289e889b25f9f38facfccf3bdbceea06ef3baad6e9612b7232cd553f4884a7a642f6583a1a589d4dcb2dc771f1ff6d711b85f731145a89b100680f9a55dcbb3f",

        "d7f9053984213ebabc842fd8ce483609a9af5dc140ecdbe63336",
        "f167cb30e4bacbdc5ed53bc615f8c9ea19ad4f6bd85ca0ff5fb1f1cbe5b576bda49276aa5814291a7e320f1d687b16ba8d7daab2b3d7e9af3cd9f84a1e9979a1",

        "9b7f9d11be48e786a11a472ab2344c57adf62f7c1d4e6d282074b6",
        "82fa525d5efaa3cce39bffef8eee01afb52067097f8965cde71703345322645eae59dbaebed0805693104dfb0c5811c5828da9a75d812e5562615248c03ff880",

        "115784b1fccfabca457c4e27a24a7832280b7e7d6a123ffce5fdab72",
        "ec12c4ed5ae84808883c5351003f7e26e1eaf509c866b357f97472e5e19c84f99f16dbbb8bfff060d6c0fe0ca9c34a210c909b05f6a81f441627ce8e666f6dc7",

        "c3b1ad16b2877def8d080477d8b59152fe5e84f3f3380d55182f36eb5f",
        "4b9144edeeec28fd52ba4176a78e080e57782d2329b67d8ac8780bb6e8c2057583172af1d068922feaaff759be5a6ea548f5db51f4c34dfe7236ca09a67921c7",

        "4c66ca7a01129eaca1d99a08dd7226a5824b840d06d0059c60e97d291dc4",
        "567c46f2f636223bd5ed3dc98c3f7a739b42898e70886f132eac43c2a6fadabe0dd9f1b6bc4a9365e5232295ac1ac34701b0fb181d2f7f07a79d033dd426d5a2",

        "481041c2f56662316ee85a10b98e103c8d48804f6f9502cf1b51cfa525cec1",
        "46f0058abe678195b576df5c7eb8d739468cad1908f7953ea39c93fa1d96845c38a2934d23804864a8368dae38191d983053ccd045a9ab87ef2619e9dd50c8c1",

        "7c1688217b313278b9eae8edcf8aa4271614296d0c1e8916f9e0e940d28b88c5",
        "627ba4de74d05bb6df8991112e4d373bfced37acde1304e0f664f29fa126cb497c8a1b717b9929120883ec8898968e4649013b760a2180a9dc0fc9b27f5b7f3b",

        "785f6513fcd92b674c450e85da22257b8e85bfa65e5d9b1b1ffc5c469ad337d1e3",
        "5c11d6e4c5c5f76d26876c5976b6f555c255c785b2f28b6700ca2d8b3b3fa585636239277773330f4cf8c5d5203bcc091b8d47e7743bbc0b5a2c54444ee2acce",

        "34f4468e2d567b1e326c0942970efa32c5ca2e95d42c98eb5d3cab2889490ea16ee5",
        "49adfa335e183c94b3160154d6698e318c8b5dd100b0227e3e34cabea1fe0f745326220f64263961349996bbe1aae9054de6406e8b350408ab0b9f656bb8daf7",

        "53a0121c8993b6f6eec921d2445035dd90654add1298c6727a2aed9b59bafb7dd62070",
        "918b4d92e1fcb65a4c1fa0bd75c562ac9d83186bb2fbfae5c4784de31a14654546e107df0e79076b8687bb3841c83ba9181f9956cd43428ba72f603881b33a71",

        "d30fa4b40c9f84ac9bcbb535e86989ec6d1bec9b1b22e9b0f97370ed0f0d566082899d96",
        "39f104c1da4af314d6bceb34eca1dfe4e67484519eb76ba38e4701e113e6cbc0200df86e4439d674b0f42c72233360478ba5244384d28e388c87aaa817007c69",

        "f34d100269aee3ead156895e8644d4749464d5921d6157dffcbbadf7a719aee35ae0fd4872",
        "565a1dd9d49f8ddefb79a3c7a209f53f0bc9f5396269b1ce2a2b283a3cb45ee3ae652e4ca10b26ced7e5236227006c94a37553db1b6fe5c0c2eded756c896bb1",

        "12529769fe5191d3fce860f434ab1130ce389d340fca232cc50b7536e62ad617742e022ea38a",
        "daee10e815fff0f0985d208886e22f9bf20a3643eb9a29fda469b6a7dcd54b5213c851d6f19338d63688fe1f02936c5dae1b7c6d5906a13a9eeb934400b6fe8c",

        "b2e3a0eb36bf16afb618bfd42a56789179147effecc684d8e39f037ec7b2d23f3f57f6d7a7d0bb",
        "04029d6d9e8e394afa387f1d03ab6b8a0a6cbab4b6b3c86ef62f7142ab3c108388d42cb87258b9e6d36e5814d8a662657cf717b35a5708365e8ec0396ec5546b",

        "25c4a5f4a07f2b81e0533313664bf615c73257e6b2930e752fe5050e25ff02731fd2872f4f56f727",
        "ec2d38e5bb5d7b18438d5f2029c86d05a03510db0e66aa299c28635abd0988c58be203f04b7e0cc25451d18f2341cd46f8705d46c2066dafab30d90d63bf3d2c",

        "134bb8e7ea5ff9edb69e8f6bbd498eb4537580b7fba7ad31d0a09921237acd7d66f4da23480b9c1222",
        "8f966aef96831a1499d63560b2578021ad970bf7557b8bf8078b3e12cefab122fe71b1212dc704f7094a40b36b71d3ad7ce2d30f72c1baa4d4bbccb3251198ac",

        "f793256f039fad11af24cee4d223cd2a771598289995ab802b5930ba5c666a24188453dcd2f0842b8152",
        "22c3d9712535153a3e206b1033929c0fd9d937c39ba13cf1a6544dfbd68ebc94867b15fda3f1d30b00bf47f2c4bf41dabdeaa5c397dae901c57db9cd77ddbcc0",

        "23cc7f9052d5e22e6712fab88e8dfaa928b6e015ca589c3b89cb745b756ca7c7634a503bf0228e71c28ee2",
        "6ecf3ad6064218ee101a555d20fab6cbeb6b145b4eeb9c8c971fc7ce05581a34b3c52179590e8a134be2e88c7e549875f4ff89b96374c6995960de3a5098cced",

        "a60b7b3df15b3f1b19db15d480388b0f3b00837369aa2cc7c3d7315775d7309a2d6f6d1371d9c875350dec0a",
        "8d651605c6b32bf022ea06ce6306b2ca6b5ba2781af87ca2375860315c83ad88743030d148ed8d73194c461ec1e84c045fc914705747614c04c8865b51da94f7",

        "2745dd2f1b215ea509a912e5761cccc4f19fa93ba38445c528cb2f099de99ab9fac955baa211fd8539a671cdb6",
        "4af918eb676ce278c730212ef79d818773a76a43c74d643f238e9b61acaf4030c617c4d6b3b7514c59b3e5e95d82e1e1e35443e851718b13b63e70b123d1b72c",

        "88adee4b46d2a109c36fcfb660f17f48062f7a74679fb07e86cad84f79fd57c86d426356ec8e68c65b3caa5bc7ba",
        "6257acb9f589c919c93c0adc4e907fe011bef6018fbb18e618ba6fcc8cbc5e40641be589e86dbb0cf7d7d6bf33b98d8458cce0af7857f5a7c7647cf350e25af0",

        "7d40f2dc4af3cfa12b00d64940dc32a22d66d81cb628be2b8dda47ed6728020d55b695e75260f4ec18c6d74839086a",
        "5c46c84a0a02d898ed5885ce99c47c77afd29ae015d027f2485d630f9b41d00b7c1f1faf6ce57a08b604b35021f7f79600381994b731bd8e6a5b010aeb90e1eb",

        "3689d8836af0dc132f85b212eb670b41ecf9d4aba141092a0a8eca2e6d5eb0ba4b7e61af9273624d14192df7388a8436",
        "17355e61d66e40f750d0a9a8e8a88cd6f9bf6070b7efa76442698740b4487ea6c644d1654ef16a265204e03084a14cafdccf8ff298cd54c0b4009967b6dd47cc",

        "58ff23dee2298c2ca7146227789c1d4093551047192d862fc34c1112d13f1f744456cecc4d4a02410523b4b15e598df75a",
        "aca89aa547c46173b4b2a380ba980da6f9ac084f46ac9ddea5e4164aeef31a9955b814a45aec1d8ce340bd37680952c5d68226dda1cac2677f73c9fd9174fd13",

        "67f3f23df3bd8ebeb0096452fe4775fd9cc71fbb6e72fdcc7eb8094f42c903121d0817a927bcbabd3109d5a70420253deab2",
        "f4207cc565f266a245f29bf20b95b5d9a83e1bb68ad988edc91faa25f25286c8398bac7dd6628259bff98f28360f263dfc54c4228bc437c5691de1219b758d9f",

        "a225070c2cb122c3354c74a254fc7b84061cba33005cab88c409fbd3738ff67ce23c41ebef46c7a61610f5b93fa92a5bda9569",
        "e815a9a4e4887be014635e97958341e0519314b3a3289e1835121b153b462272b0aca418be96d60e5ab355d3eb463697c0191eb522b60b8463d89f4c3f1bf142",

        "6aa0886777e99c9acd5f1db6e12bda59a807f92411ae99c9d490b5656acb4b115c57beb3c1807a1b029ad64be1f03e15bafd91ec",
        "241f2ebaf7ad09e173b184244e69acd7ebc94774d0fa3902cbf267d4806063b044131bcf4af4cf180eb7bd4e7960ce5fe3dc6aebfc6b90eec461f414f79a67d9",

        "6a06092a3cd221ae86b286b31f326248270472c5ea510cb9064d6024d10efee7f59e98785d4f09da554e97cdec7b75429d788c112f",
        "d14a1a47f2bef9e0d4b3e90a6be9ab5893e1110b12db38d33ffb9a61e1661aecc4ea100839cfee58a1c5aff72915c14170dd99e13f71b0a5fc1985bf43415cb0",

        "dfc3fa61f7fffc7c88ed90e51dfc39a4f288b50d58ac83385b58a3b2a3a39d729862c40fcaf9bc308f713a43eecb0b72bb9458d204ba",
        "947bc873dc41df195f8045deb6ea1b840f633917e79c70a88d38b8862197dc2ab0cc6314e974fb5ba7e1703b22b1309e37bd430879056bdc166573075a9c5e04",

        "52958b1ff0049efa5d050ab381ec99732e554dcd03725da991a37a80bd4756cf65d367c54721e93f1e0a22f70d36e9f841336956d3c523",
        "9cc5aad0f529f4bac491d733537b69c8ec700fe38ab423d815e0927c8657f9cb8f4207762d816ab697580122066bc2b68f4177335d0a6e9081540779e572c41f",

        "302fa84fdaa82081b1192b847b81ddea10a9f05a0f04138fd1da84a39ba5e18e18bc3cea062e6df92ff1ace89b3c5f55043130108abf631e",
        "8c8eaae9a445643a37df34cfa6a7f09deccab2a222c421d2fc574bbc5641e504354391e81eb5130280b1226812556d474e951bb78dbdd9b77d19f647e2e7d7be",

        "b82f500d6bc2dddcdc162d46cbfaa5ae64025d5c1cd72472dcd2c42161c9871ce329f94df445f0c8aceecafd0344f6317ecbb62f0ec2223a35",
        "55c69d7accd179d5d9fcc522f794e7af5f0eec7198ffa39f80fb55b866c0857ff3e7aeef33e130d9c74ef90606ca821d20b7608b12e6e561f9e6c7122ace3db0",

        "86da9107ca3e16a2b58950e656a15c085b88033e79313e2c0f92f99f06fa187efba5b8fea08eb7145f8476304180dd280f36a072b7eac197f085",
        "0d3b1a0459b4eca801e0737ff9ea4a12b9a483a73a8a92742a93c297b7149326bd92c1643c8177c8924482ab3bbd916c417580cc75d3d3ae096de531bc5dc355",

        "141a6eafe157053e780ac7a57b97990616ce1759ed132cb453bcdfcabdbb70b3767da4eb94125d9c2a8d6d20bfaeacc1ffbe49c4b1bb5da7e9b5c6",
        "bdbdd5b94cdc89466e7670c63ba6a55b58294e93b351261a5457bf5a40f1b5b2e0acc7fceb1bfb4c8872777eeeaff7927fd3635ca18c996d870bf86b12b89ba5",

        "6e0c65ee0943e34d9bbd27a8547690f2291f5a86d713c2be258e6ac16919fe9c4d491895d3a961bb97f5fac255891a0eaa18f80e1fa1ebcb639fcfc1",
        "39ebb992b8d39daae973e3813a50e9e79a67d8458a6f17f97a6dd30dd7d11d95701a11129ffeaf7d45781b21cac0c4c034e389d7590df5beeb9805072d0183b9",

        "57780b1c79e67fc3beaabead4a67a8cc98b83fa7647eae50c8798b96a516597b448851e93d1a62a098c4767333fcf7b463ce91edde2f3ad0d98f70716d",
        "3ef36c3effad6eb5ad2d0a67780f80d1b90efcb74db20410c2261a3ab0f784429df874814748dc1b6efaab3d06dd0a41ba54fce59b67d45838eaa4aa1fadfa0f",

        "bcc9849da4091d0edfe908e7c3386b0cadadb2859829c9dfee3d8ecf9dec86196eb2ceb093c5551f7e9a4927faabcfaa7478f7c899cbef4727417738fc06",
        "1fcd8a2c7b4fd98fcdc5fa665bab49bde3f9f556aa66b3646638f5a2d3806192f8a33145d8d0c535c85adff3cc0ea3c2715b33cec9f8886e9f4377b3632e9055",

        "05a32829642ed4808d6554d16b9b8023353ce65a935d126602970dba791623004dede90b52ac7f0d4335130a63cba68c656c139989614de20913e83db320db",
        "49d8747bb53ddde6d1485965208670d1130bf35619d7506a2f2040d1129fcf0320207e5b36fea083e84ffc98755e691ad8bd5dc66f8972cb9857389344e11aad",

        "56ac4f6845a451dac3e8886f97f7024b64b1b1e9c5181c059b5755b9a6042be653a2a0d5d56a9e1e774be5c9312f48b4798019345beae2ffcc63554a3c69862e",
        "5fde5c57a31febb98061f27e4506fa5c245506336ee90d595c91d791a5975c712b3ab9b3b5868f941db0aeb4c6d2837c4447442f8402e0e150a9dc0ef178dca8",

        "8a229f8d0294fe90d4cc8c875460d5d623f93287f905a999a2ab0f9a47046f78ef88b09445c671189c59388b3017cca2af8bdf59f8a6f04322b1701ec08624ab63",
        "16b0fd239cc632842c443e1b92d286dd519cfc616a41f2456dd5cddebd10703c3e9cb669004b7f169bb4f99f350ec96904b0e8dd4de8e6be9953dc892c65099f",

        "87d6aa9979025b2437ea8159ea1d3e5d6f17f0a5b913b56970212f56de7884840c0da9a72865e1892aa780b8b8f5f57b46fc070b81ca5f00eee0470ace89b1e1466a",
        "d816acf1797decfe34f4cc49e52aa505cc59bd17fe69dc9543fad82e9cf96298183021f704054d3d06adde2bf54e82a090a57b239e88daa04cb76c4fc9127843",

        "0823616ab87e4904308628c2226e721bb4169b7d34e8744a0700b721e38fe05e3f813fe4075d4c1a936d3a33da20cfb3e3ac722e7df7865330b8f62a73d9119a1f2199",
        "e1da6be4403a4fd784c59be4e71c658a78bb8c5d7d571c5e816fbb3e218a4162f62de1c285f3779781cb5506e29c94e1b7c7d65af2aa71ea5c96d9585b5e45d5",

        "7d2d913c2460c09898b20366ae34775b1564f10edea49c073cebe41989bb93f38a533af1f425d3382f8aa40159b567358ee5a73b67df6d0dc09c1c92bf3f9a28124ab07f",
        "3aa1e19a52b86cf414d977768bb535b7e5817117d436b4425ec8d775e8cb0e0b538072213884c7ff1bb9ca9984c82d65cb0115cc07332b0ea903e3b38650e88e",

        "fca5f68fd2d3a52187b349a8d2726b608fccea7db42e906b8718e85a0ec654fac70f5a839a8d3ff90cfed7aeb5ea9b08f487fc84e1d9f7fb831dea254468a65ba18cc5a126",
        "2c74f846ecc722ea4a1eb1162e231b6903291fffa95dd5e1d17dbc2c2be7dfe549a80dd34487d714130ddc9924aed904ad55f49c91c80ceb05c0c034dae0a0a4",

        "881ff70ca34a3e1a0e864fd2615ca2a0e63def254e688c37a20ef6297cb3ae4c76d746b5e3d6bb41bd0d05d7df3eeded74351f4eb0ac801abe6dc10ef9b635055ee1dfbf4144",
        "9a10a7ce23c0497fe8783927f833232ae664f1e1b91302266b6ace25a9c253d1ecab1aaaa62f865469480b2145ed0e489ae3f3f9f7e6da27492c81b07e606fb6",

        "b0de0430c200d74bf41ea0c92f8f28e11b68006a884e0d4b0d884533ee58b38a438cc1a75750b6434f467e2d0cd9aa4052ceb793291b93ef83fd5d8620456ce1aff2941b3605a4",
        "9e9e469ca9226cd012f5c9cc39c96adc22f420030fcee305a0ed27974e3c802701603dac873ae4476e9c3d57e55524483fc01adaef87daa9e304078c59802757",

        "0ce9f8c3a990c268f34efd9befdb0f7c4ef8466cfdb01171f8de70dc5fefa92acbe93d29e2ac1a5c2979129f1ab08c0e77de7924ddf68a209cdfa0adc62f85c18637d9c6b33f4ff8",
        "b018a20fcf831dde290e4fb18c56342efe138472cbe142da6b77eea4fce52588c04c808eb32912faa345245a850346faec46c3a16d39bd2e1ddb1816bc57d2da",

        "664ef2e3a7059daf1c58caf52008c5227e85cdcb83b4c59457f02c508d4f4f69f826bd82c0cffc5cb6a97af6e561c6f96970005285e58f21ef6511d26e709889a7e513c434c90a3cf7448f0caeec7114c747b2a0758a3b4503a7cf0c69873ed31d94dbef2b7b2f168830ef7da3322c3d3e10cafb7c2c33c83bbf4c46a31da90cff3bfd4ccc6ed4b310758491eeba603a76",
        "e5825ff1a3c070d5a52fbbe711854a440554295ffb7a7969a17908d10163bfbe8f1d52a676e8a0137b56a11cdf0ffbb456bc899fc727d14bd8882232549d914e",

        "991c4e7402c7da689dd5525af76fcc58fe9cc1451308c0c4600363586ccc83c9ec10a8c9ddaec3d7cfbd206484d09634b9780108440bf27a5fa4a428446b3214fa17084b6eb197c5c59a4e8df1cfc521826c3b1cbf6f4212f6bfb9bc106dfb5568395643de58bffa2774c31e67f5c1e7017f57caadbb1a56cc5b8a5cf9584552e17e7af9542ba13e9c54695e0dc8f24eddb93d5a3678e10c8a80ff4f27b677d40bef5cb5f9b3a659cc4127970cd2c11ebf22d514812dfefdd73600dfc10efba38e93e5bff47736126043e50f8b9b941e4ec3083fb762dbf15c86",
        "cd0f2a48e9aa8cc700d3f64efb013f3600ebdbb524930c682d21025eab990eb6d7c52e611f884031fafd9360e5225ab7e4ec24cbe97f3af6dbe4a86a4f068ba7",
    ];

    fn sha3_224(m: &[u8], d: &[u8]) {
        let mut sh = SHA3_224::new();
        sh.update(m);
        assert!(sh.digest() == d);
        for i in 0..m.len() {
            sh.update(&m[i..i + 1]);
        }
        assert!(sh.digest() == d);
    }

    fn sha3_256(m: &[u8], d: &[u8]) {
        let mut sh = SHA3_256::new();
        sh.update(m);
        assert!(sh.digest() == d);
        for i in 0..m.len() {
            sh.update(&m[i..i + 1]);
        }
        assert!(sh.digest() == d);
    }

    fn sha3_384(m: &[u8], d: &[u8]) {
        let mut sh = SHA3_384::new();
        sh.update(m);
        assert!(sh.digest() == d);
        for i in 0..m.len() {
            sh.update(&m[i..i + 1]);
        }
        assert!(sh.digest() == d);
    }

    fn sha3_512(m: &[u8], d: &[u8]) {
        let mut sh = SHA3_512::new();
        sh.update(m);
        assert!(sh.digest() == d);
        for i in 0..m.len() {
            sh.update(&m[i..i + 1]);
        }
        assert!(sh.digest() == d);
    }

    #[test]
    fn sha3() {
        for i in 0..(KAT_SHA3.len() >> 1) {
            let m = hex::decode(KAT_SHA3[(i << 1) + 0]).unwrap();
            let d = hex::decode(KAT_SHA3[(i << 1) + 1]).unwrap();
            match d.len() {
                28 => { sha3_224(&m, &d); },
                32 => { sha3_256(&m, &d); },
                48 => { sha3_384(&m, &d); },
                64 => { sha3_512(&m, &d); },
                _ => { assert!(false); },
            }
        }
    }
}
