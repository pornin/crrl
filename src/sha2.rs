#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_imports)]

use core::convert::{AsRef, TryFrom};

macro_rules! sha2_impl {
    ($typename:ident, $docname:expr, $size:expr, $corename:ident, $iv:expr) => {

        #[doc = concat!($docname, " implementation.\n\n",
            "Cloning captures the current object state.")]
        #[derive(Copy, Clone, Debug)]
        pub struct $typename ($corename<$size>);

        impl $typename {

            /// Create a new instance.
            pub fn new() -> Self {
                Self($corename::<$size>::new(&$iv))
            }

            /// Process some input bytes; this function can be called
            /// repeatedly.
            pub fn update(&mut self, src: impl AsRef <[u8]>) {
                self.0.update(src.as_ref());
            }

            /// Compute the hash of all bytes injected since the last
            /// reset of this instance. The instance is automatically reset.
            pub fn digest(&mut self) -> [u8; $size >> 3] {
                let mut r = [0u8; $size >> 3];
                self.0.digest_to(&mut r);
                self.reset();
                r
            }

            /// Reset this instance to its initial state.
            pub fn reset(&mut self) {
                self.0.reset(&$iv);
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
    }
}

sha2_impl!(Sha224, "SHA-224", 224, SHA2Small,
    [ 0xC1059ED8u32, 0x367CD507u32, 0x3070DD17u32, 0xF70E5939u32,
      0xFFC00B31u32, 0x68581511u32, 0x64F98FA7u32, 0xBEFA4FA4u32, ]);
sha2_impl!(Sha256, "SHA-256", 256, SHA2Small,
    [ 0x6A09E667u32, 0xBB67AE85u32, 0x3C6EF372u32, 0xA54FF53Au32,
      0x510E527Fu32, 0x9B05688Cu32, 0x1F83D9ABu32, 0x5BE0CD19u32, ]);

sha2_impl!(Sha384, "SHA-384", 384, SHA2Big,
    [ 0xCBBB9D5DC1059ED8, 0x629A292A367CD507, 0x9159015A3070DD17,
      0x152FECD8F70E5939, 0x67332667FFC00B31, 0x8EB44A8768581511,
      0xDB0C2E0D64F98FA7, 0x47B5481DBEFA4FA4, ]);
sha2_impl!(Sha512, "SHA-512", 512, SHA2Big,
    [ 0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, 0x3C6EF372FE94F82B,
      0xA54FF53A5F1D36F1, 0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
      0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179, ]);
sha2_impl!(Sha512_224, "SHA-512/224", 224, SHA2Big,
    [ 0x8C3D37C819544DA2, 0x73E1996689DCD4D6, 0x1DFAB7AE32FF9C82,
      0x679DD514582F9FCF, 0x0F6D2B697BD44DA8, 0x77E36F7304C48942,
      0x3F9D85A86A1D36C8, 0x1112E6AD91D692A1, ]);
sha2_impl!(Sha512_256, "SHA-512/256", 256, SHA2Big,
    [ 0x22312194FC2BF72C, 0x9F555FA3C84C64C2, 0x2393B86B6F53B151,
      0x963877195940EABD, 0x96283EE2A88EFFE3, 0xBE5E1E2553863992,
      0x2B0199FC2C85B8AA, 0x0EB72DDC81C52CA2, ]);

#[derive(Clone, Copy, Debug)]
struct SHA2Small<const SZ: usize> {
    h: [u32; 8],
    buf: [u8; 64],
    ctr: u64,
}

impl<const SZ: usize> SHA2Small<SZ> {

    // A custom compile-time check; it should prevent compilation from
    // succeeding if SZ is not in {224, 256}.
    #[allow(dead_code)]
    const COMPILE_TIME_CHECKS: () = Self::compile_time_checks();
    const fn compile_time_checks() {
        let _ = &[()][1 -
            ((SZ == 224 || SZ == 256) as usize)];
    }

    fn new(iv: &[u32; 8]) -> Self {
        Self {
            h: *iv,
            buf: [0u8; 64],
            ctr: 0,
        }
    }

    fn update(&mut self, src: &[u8]) {
        let mut j = 0;
        let mut ptr = (self.ctr as usize) & 63;
        while j < src.len() {
            let clen = core::cmp::min(src.len() - j, 64 - ptr);
            self.buf[ptr..(ptr + clen)].copy_from_slice(&src[j..(j + clen)]);
            ptr += clen;
            if ptr == 64 {
                self.process();
                ptr = 0;
            }
            j += clen;
        }
        self.ctr += src.len() as u64;
    }

    fn digest_to(&mut self, dst: &mut [u8]) {
        assert!(dst.len() == (SZ >> 3));
        let mut ptr = (self.ctr as usize) & 63;
        self.buf[ptr] = 0x80;
        ptr += 1;
        if ptr > 56 {
            while ptr < 64 {
                self.buf[ptr] = 0;
                ptr += 1;
            }
            self.process();
            ptr = 0;
        }
        while ptr < 56 {
            self.buf[ptr] = 0;
            ptr += 1;
        }
        self.buf[56..].copy_from_slice(&(self.ctr << 3).to_be_bytes());
        self.process();
        for i in 0..(SZ >> 5) {
            let j = i << 2;
            dst[j..(j + 4)].copy_from_slice(&self.h[i].to_be_bytes());
        }
    }

    fn reset(&mut self, iv: &[u32; 8]) {
        self.h.copy_from_slice(&iv[..]);
        self.ctr = 0;
    }

    const K: [u32; 64] = [
        0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
        0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
        0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
        0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
        0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
        0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
        0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
        0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
        0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
        0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
        0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
        0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
        0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
        0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
        0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
        0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
    ];

    fn process(&mut self) {

        #[inline(always)]
        fn ch(x: u32, y: u32, z: u32) -> u32 {
            z ^ (x & (y ^ z))
        }

        #[inline(always)]
        fn maj(x: u32, y: u32, z: u32) -> u32 {
            (x & y) | (z & (x | y))
        }

        #[inline(always)]
        fn bsig0(x: u32) -> u32 {
            x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
        }

        #[inline(always)]
        fn bsig1(x: u32) -> u32 {
            x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
        }

        #[inline(always)]
        fn ssig0(x: u32) -> u32 {
            x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
        }

        #[inline(always)]
        fn ssig1(x: u32) -> u32 {
            x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
        }

        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(*<&[u8; 4]>::try_from(
                &self.buf[(i << 2)..((i << 2) + 4)]).unwrap());
        }
        for i in 16..64 {
            w[i] = ssig1(w[i - 2]).wrapping_add(w[i - 7])
                .wrapping_add(ssig0(w[i - 15])).wrapping_add(w[i - 16]);
        }

        let mut a = self.h[0];
        let mut b = self.h[1];
        let mut c = self.h[2];
        let mut d = self.h[3];
        let mut e = self.h[4];
        let mut f = self.h[5];
        let mut g = self.h[6];
        let mut h = self.h[7];
        for i in 0..8 {
            macro_rules! rf {
                ($a:ident, $b:ident, $c:ident, $d:ident,
                 $e:ident, $f:ident, $g:ident, $h:ident, $idx:expr) => {

                    let t1 = $h.wrapping_add(bsig1($e))
                        .wrapping_add(ch($e, $f, $g))
                        .wrapping_add(Self::K[$idx])
                        .wrapping_add(w[$idx]);
                    let t2 = bsig0($a).wrapping_add(maj($a, $b, $c));
                    $d = $d.wrapping_add(t1);
                    $h = t1.wrapping_add(t2);
                }
            }

            let j = i << 3;
            rf!(a, b, c, d, e, f, g, h, j + 0);
            rf!(h, a, b, c, d, e, f, g, j + 1);
            rf!(g, h, a, b, c, d, e, f, j + 2);
            rf!(f, g, h, a, b, c, d, e, j + 3);
            rf!(e, f, g, h, a, b, c, d, j + 4);
            rf!(d, e, f, g, h, a, b, c, j + 5);
            rf!(c, d, e, f, g, h, a, b, j + 6);
            rf!(b, c, d, e, f, g, h, a, j + 7);
        }
        self.h[0] = self.h[0].wrapping_add(a);
        self.h[1] = self.h[1].wrapping_add(b);
        self.h[2] = self.h[2].wrapping_add(c);
        self.h[3] = self.h[3].wrapping_add(d);
        self.h[4] = self.h[4].wrapping_add(e);
        self.h[5] = self.h[5].wrapping_add(f);
        self.h[6] = self.h[6].wrapping_add(g);
        self.h[7] = self.h[7].wrapping_add(h);
    }
}

#[derive(Clone, Copy, Debug)]
struct SHA2Big<const SZ: usize> {
    h: [u64; 8],
    buf: [u8; 128],
    ctr: u128,
}

impl<const SZ: usize> SHA2Big<SZ> {

    // A custom compile-time check; it should prevent compilation from
    // succeeding if SZ is not in {224, 256, 384, 512}.
    #[allow(dead_code)]
    const COMPILE_TIME_CHECKS: () = Self::compile_time_checks();
    const fn compile_time_checks() {
        let _ = &[()][1 -
            ((SZ == 224 || SZ == 256 || SZ == 384 || SZ == 512) as usize)];
    }

    fn new(iv: &[u64; 8]) -> Self {
        Self {
            h: *iv,
            buf: [0u8; 128],
            ctr: 0,
        }
    }

    fn update(&mut self, src: &[u8]) {
        let mut j = 0;
        let mut ptr = (self.ctr as usize) & 127;
        while j < src.len() {
            let clen = core::cmp::min(src.len() - j, 128 - ptr);
            self.buf[ptr..(ptr + clen)].copy_from_slice(&src[j..(j + clen)]);
            ptr += clen;
            if ptr == 128 {
                self.process();
                ptr = 0;
            }
            j += clen;
        }
        self.ctr += src.len() as u128;
    }

    fn digest_to(&mut self, dst: &mut [u8]) {
        assert!(dst.len() == (SZ >> 3));
        let mut ptr = (self.ctr as usize) & 127;
        self.buf[ptr] = 0x80;
        ptr += 1;
        if ptr > 112 {
            while ptr < 128 {
                self.buf[ptr] = 0;
                ptr += 1;
            }
            self.process();
            ptr = 0;
        }
        while ptr < 112 {
            self.buf[ptr] = 0;
            ptr += 1;
        }
        self.buf[112..].copy_from_slice(&(self.ctr << 3).to_be_bytes());
        self.process();
        for i in 0..(SZ >> 6) {
            let j = i << 3;
            dst[j..(j + 8)].copy_from_slice(&self.h[i].to_be_bytes());
        }
        if (SZ & 63) != 0 {
            let j = (SZ >> 3) & !7usize;
            let r = (self.h[SZ >> 6] >> 32) as u32;
            dst[j..(j + 4)].copy_from_slice(&r.to_be_bytes());
        }
    }

    fn reset(&mut self, iv: &[u64; 8]) {
        self.h.copy_from_slice(&iv[..]);
        self.ctr = 0;
    }

    const K: [u64; 80] = [
        0x428A2F98D728AE22, 0x7137449123EF65CD,
        0xB5C0FBCFEC4D3B2F, 0xE9B5DBA58189DBBC,
        0x3956C25BF348B538, 0x59F111F1B605D019,
        0x923F82A4AF194F9B, 0xAB1C5ED5DA6D8118,
        0xD807AA98A3030242, 0x12835B0145706FBE,
        0x243185BE4EE4B28C, 0x550C7DC3D5FFB4E2,
        0x72BE5D74F27B896F, 0x80DEB1FE3B1696B1,
        0x9BDC06A725C71235, 0xC19BF174CF692694,
        0xE49B69C19EF14AD2, 0xEFBE4786384F25E3,
        0x0FC19DC68B8CD5B5, 0x240CA1CC77AC9C65,
        0x2DE92C6F592B0275, 0x4A7484AA6EA6E483,
        0x5CB0A9DCBD41FBD4, 0x76F988DA831153B5,
        0x983E5152EE66DFAB, 0xA831C66D2DB43210,
        0xB00327C898FB213F, 0xBF597FC7BEEF0EE4,
        0xC6E00BF33DA88FC2, 0xD5A79147930AA725,
        0x06CA6351E003826F, 0x142929670A0E6E70,
        0x27B70A8546D22FFC, 0x2E1B21385C26C926,
        0x4D2C6DFC5AC42AED, 0x53380D139D95B3DF,
        0x650A73548BAF63DE, 0x766A0ABB3C77B2A8,
        0x81C2C92E47EDAEE6, 0x92722C851482353B,
        0xA2BFE8A14CF10364, 0xA81A664BBC423001,
        0xC24B8B70D0F89791, 0xC76C51A30654BE30,
        0xD192E819D6EF5218, 0xD69906245565A910,
        0xF40E35855771202A, 0x106AA07032BBD1B8,
        0x19A4C116B8D2D0C8, 0x1E376C085141AB53,
        0x2748774CDF8EEB99, 0x34B0BCB5E19B48A8,
        0x391C0CB3C5C95A63, 0x4ED8AA4AE3418ACB,
        0x5B9CCA4F7763E373, 0x682E6FF3D6B2B8A3,
        0x748F82EE5DEFB2FC, 0x78A5636F43172F60,
        0x84C87814A1F0AB72, 0x8CC702081A6439EC,
        0x90BEFFFA23631E28, 0xA4506CEBDE82BDE9,
        0xBEF9A3F7B2C67915, 0xC67178F2E372532B,
        0xCA273ECEEA26619C, 0xD186B8C721C0C207,
        0xEADA7DD6CDE0EB1E, 0xF57D4F7FEE6ED178,
        0x06F067AA72176FBA, 0x0A637DC5A2C898A6,
        0x113F9804BEF90DAE, 0x1B710B35131C471B,
        0x28DB77F523047D84, 0x32CAAB7B40C72493,
        0x3C9EBE0A15C9BEBC, 0x431D67C49C100D4C,
        0x4CC5D4BECB3E42B6, 0x597F299CFC657E2A,
        0x5FCB6FAB3AD6FAEC, 0x6C44198C4A475817,
    ];

    fn process(&mut self) {

        #[inline(always)]
        fn ch(x: u64, y: u64, z: u64) -> u64 {
            z ^ (x & (y ^ z))
        }

        #[inline(always)]
        fn maj(x: u64, y: u64, z: u64) -> u64 {
            (x & y) | (z & (x | y))
        }

        #[inline(always)]
        fn bsig0(x: u64) -> u64 {
            x.rotate_right(28) ^ x.rotate_right(34) ^ x.rotate_right(39)
        }

        #[inline(always)]
        fn bsig1(x: u64) -> u64 {
            x.rotate_right(14) ^ x.rotate_right(18) ^ x.rotate_right(41)
        }

        #[inline(always)]
        fn ssig0(x: u64) -> u64 {
            x.rotate_right(1) ^ x.rotate_right(8) ^ (x >> 7)
        }

        #[inline(always)]
        fn ssig1(x: u64) -> u64 {
            x.rotate_right(19) ^ x.rotate_right(61) ^ (x >> 6)
        }

        let mut w = [0u64; 80];
        for i in 0..16 {
            w[i] = u64::from_be_bytes(*<&[u8; 8]>::try_from(
                &self.buf[(i << 3)..((i << 3) + 8)]).unwrap());
        }
        for i in 16..80 {
            w[i] = ssig1(w[i - 2]).wrapping_add(w[i - 7])
                .wrapping_add(ssig0(w[i - 15])).wrapping_add(w[i - 16]);
        }

        let mut a = self.h[0];
        let mut b = self.h[1];
        let mut c = self.h[2];
        let mut d = self.h[3];
        let mut e = self.h[4];
        let mut f = self.h[5];
        let mut g = self.h[6];
        let mut h = self.h[7];
        for i in 0..10 {
            macro_rules! rf {
                ($a:ident, $b:ident, $c:ident, $d:ident,
                 $e:ident, $f:ident, $g:ident, $h:ident, $idx:expr) => {

                    let t1 = $h.wrapping_add(bsig1($e))
                        .wrapping_add(ch($e, $f, $g))
                        .wrapping_add(Self::K[$idx])
                        .wrapping_add(w[$idx]);
                    let t2 = bsig0($a).wrapping_add(maj($a, $b, $c));
                    $d = $d.wrapping_add(t1);
                    $h = t1.wrapping_add(t2);
                }
            }

            let j = i << 3;
            rf!(a, b, c, d, e, f, g, h, j + 0);
            rf!(h, a, b, c, d, e, f, g, j + 1);
            rf!(g, h, a, b, c, d, e, f, j + 2);
            rf!(f, g, h, a, b, c, d, e, j + 3);
            rf!(e, f, g, h, a, b, c, d, j + 4);
            rf!(d, e, f, g, h, a, b, c, j + 5);
            rf!(c, d, e, f, g, h, a, b, j + 6);
            rf!(b, c, d, e, f, g, h, a, j + 7);
        }
        self.h[0] = self.h[0].wrapping_add(a);
        self.h[1] = self.h[1].wrapping_add(b);
        self.h[2] = self.h[2].wrapping_add(c);
        self.h[3] = self.h[3].wrapping_add(d);
        self.h[4] = self.h[4].wrapping_add(e);
        self.h[5] = self.h[5].wrapping_add(f);
        self.h[6] = self.h[6].wrapping_add(g);
        self.h[7] = self.h[7].wrapping_add(h);
    }
}

#[cfg(test)]
mod tests {

    use super::{Sha224, Sha256, Sha384, Sha512, Sha512_224, Sha512_256};

    fn inner_sha224(src: &[u8], hexdst: &str) {
        let dst = hex::decode(hexdst).unwrap();

        assert!(Sha224::hash(src) == *dst);
        let mut sh = Sha224::new();
        for i in 0..src.len() {
            sh.update(&src[i..(i + 1)]);
        }
        assert!(sh.digest() == *dst);
    }

    #[test]
    fn sha224() {
        inner_sha224(&[0x61u8, 0x62u8, 0x63u8],
            "23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7");

        inner_sha224(&[0xFFu8],
            "e33f9d75e6ae1369dbabf81b96b4591ae46bba30b591a6b6c62542b5");
        inner_sha224(&[0xE5u8, 0xE0u8, 0x99u8, 0x24u8],
            "fd19e74690d291467ce59f077df311638f1c3a46e510d0e49a67062d");
        inner_sha224(&[0u8; 56],
            "5c3e25b69d0ea26f260cfae87e23759e1eca9d1ecc9fbf3c62266804");
        inner_sha224(&[0x51u8; 1000],
            "3706197f66890a41779dc8791670522e136fafa24874685715bd0a8a");
        inner_sha224(&[0x41u8; 1000],
            "a8d0c66b5c6fdfd836eb3c6d04d32dfe66c3b1f168b488bf4c9c66ce");
        inner_sha224(&[0x99u8; 1005],
            "cb00ecd03788bf6c0908401e0eb053ac61f35e7e20a2cfd7bd96d640");
    }

    fn inner_sha256(src: &[u8], hexdst: &str) {
        let dst = hex::decode(hexdst).unwrap();

        assert!(Sha256::hash(src) == *dst);
        let mut sh = Sha256::new();
        for i in 0..src.len() {
            sh.update(&src[i..(i + 1)]);
        }
        assert!(sh.digest() == *dst);
    }

    #[test]
    fn sha256() {
        inner_sha256(&[0x61u8, 0x62u8, 0x63u8],
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

        inner_sha256(&[0xBDu8],
            "68325720aabd7c82f30f554b313d0570c95accbb7dc4b5aae11204c08ffe732b");
        inner_sha256(&[0xC9u8, 0x8Cu8, 0x8Eu8, 0x55u8],
            "7abc22c0ae5af26ce93dbb94433a0e0b2e119d014f8e7f65bd56c61ccccd9504");
        inner_sha256(&[0u8; 55],
            "02779466cdec163811d078815c633f21901413081449002f24aa3e80f0b88ef7");
        inner_sha256(&[0u8; 56],
            "d4817aa5497628e7c77e6b606107042bbba3130888c5f47a375e6179be789fbb");
        inner_sha256(&[0u8; 57],
            "65a16cb7861335d5ace3c60718b5052e44660726da4cd13bb745381b235a1785");
        inner_sha256(&[0u8; 64],
            "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b");
        inner_sha256(&[0u8; 1000],
            "541b3e9daa09b20bf85fa273e5cbd3e80185aa4ec298e765db87742b70138a53");
        inner_sha256(&[0x41u8; 1000],
            "c2e686823489ced2017f6059b8b239318b6364f6dcd835d0a519105a1eadd6e4");
        inner_sha256(&[0x55u8; 1005],
            "f4d62ddec0f3dd90ea1380fa16a5ff8dc4c54b21740650f24afc4120903552b0");
    }

    fn inner_sha384(src: &[u8], hexdst: &str) {
        let dst = hex::decode(hexdst).unwrap();

        assert!(Sha384::hash(src) == *dst);
        let mut sh = Sha384::new();
        for i in 0..src.len() {
            sh.update(&src[i..(i + 1)]);
        }
        assert!(sh.digest() == *dst);
    }

    #[test]
    fn sha384() {
        inner_sha384(&[0x61u8, 0x62u8, 0x63u8],
            "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7");

        inner_sha384(&[0u8; 0],
            "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b");
        inner_sha384(&[0u8; 111],
            "435770712c611be7293a66dd0dc8d1450dc7ff7337bfe115bf058ef2eb9bed09cee85c26963a5bcc0905dc2df7cc6a76");
        inner_sha384(&[0u8; 112],
            "3e0cbf3aee0e3aa70415beae1bd12dd7db821efa446440f12132edffce76f635e53526a111491e75ee8e27b9700eec20");
        inner_sha384(&[0u8; 113],
            "6be9af2cf3cd5dd12c8d9399ec2b34e66034fbd699d4e0221d39074172a380656089caafe8f39963f94cc7c0a07e3d21");
        inner_sha384(&[0u8; 122],
            "12a72ae4972776b0db7d73d160a15ef0d19645ec96c7f816411ab780c794aa496a22909d941fe671ed3f3caee900bdd5");
        inner_sha384(&[0u8; 1000],
            "aae017d4ae5b6346dd60a19d52130fb55194b6327dd40b89c11efc8222292de81e1a23c9b59f9f58b7f6ad463fa108ca");
        inner_sha384(&[0x41u8; 1000],
            "7df01148677b7f18617eee3a23104f0eed6bb8c90a6046f715c9445ff43c30d69e9e7082de39c3452fd1d3afd9ba0689");
        inner_sha384(&[0x55u8; 1005],
            "1bb8e256da4a0d1e87453528254f223b4cb7e49c4420dbfa766bba4adba44eeca392ff6a9f565bc347158cc970ce44ec");
    }

    fn inner_sha512(src: &[u8], hexdst: &str) {
        let dst = hex::decode(hexdst).unwrap();

        assert!(Sha512::hash(src) == *dst);
        let mut sh = Sha512::new();
        for i in 0..src.len() {
            sh.update(&src[i..(i + 1)]);
        }
        assert!(sh.digest() == *dst);
    }

    #[test]
    fn sha512() {
        inner_sha512(&[0x61u8, 0x62u8, 0x63u8],
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");

        inner_sha512(&[0u8; 0],
            "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
        inner_sha512(&[0u8; 111],
            "77ddd3a542e530fd047b8977c657ba6ce72f1492e360b2b2212cd264e75ec03882e4ff0525517ab4207d14c70c2259ba88d4d335ee0e7e20543d22102ab1788c");
        inner_sha512(&[0u8; 112],
            "2be2e788c8a8adeaa9c89a7f78904cacea6e39297d75e0573a73c756234534d6627ab4156b48a6657b29ab8beb73334040ad39ead81446bb09c70704ec707952");
        inner_sha512(&[0u8; 113],
            "0e67910bcf0f9ccde5464c63b9c850a12a759227d16b040d98986d54253f9f34322318e56b8feb86c5fb2270ed87f31252f7f68493ee759743909bd75e4bb544");
        inner_sha512(&[0u8; 122],
            "4f3f095d015be4a7a7cc0b8c04da4aa09e74351e3a97651f744c23716ebd9b3e822e5077a01baa5cc0ed45b9249e88ab343d4333539df21ed229da6f4a514e0f");
        inner_sha512(&[0u8; 1000],
            "ca3dff61bb23477aa6087b27508264a6f9126ee3a004f53cb8db942ed345f2f2d229b4b59c859220a1cf1913f34248e3803bab650e849a3d9a709edc09ae4a76");
        inner_sha512(&[0x41u8; 1000],
            "329c52ac62d1fe731151f2b895a00475445ef74f50b979c6f7bb7cae349328c1d4cb4f7261a0ab43f936a24b000651d4a824fcdd577f211aef8f806b16afe8af");
        inner_sha512(&[0x55u8; 1005],
            "59f5e54fe299c6a8764c6b199e44924a37f59e2b56c3ebad939b7289210dc8e4c21b9720165b0f4d4374c90f1bf4fb4a5ace17a1161798015052893a48c3d161");
    }

    fn inner_sha512_224(src: &[u8], hexdst: &str) {
        let dst = hex::decode(hexdst).unwrap();

        assert!(Sha512_224::hash(src) == *dst);
        let mut sh = Sha512_224::new();
        for i in 0..src.len() {
            sh.update(&src[i..(i + 1)]);
        }
        assert!(sh.digest() == *dst);
    }

    #[test]
    fn sha512_224() {
        inner_sha512_224(&[0x61u8, 0x62u8, 0x63u8],
            "4634270f707b6a54daae7530460842e20e37ed265ceee9a43e8924aa");
        inner_sha512_224(&b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"[..],
            "23fec5bb94d60b23308192640b0c453335d664734fe40e7268674af9");
    }

    fn inner_sha512_256(src: &[u8], hexdst: &str) {
        let dst = hex::decode(hexdst).unwrap();

        assert!(Sha512_256::hash(src) == *dst);
        let mut sh = Sha512_256::new();
        for i in 0..src.len() {
            sh.update(&src[i..(i + 1)]);
        }
        assert!(sh.digest() == *dst);
    }

    #[test]
    fn sha512_256() {
        inner_sha512_256(&[0x61u8, 0x62u8, 0x63u8],
            "53048e2681941ef99b2e29b76b4c7dabe4c2d0c634fc6d46e0e2f13107e7af23");
        inner_sha512_256(&b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"[..],
            "3928e184fb8690f840da3988121d31be65cb9d3ef83ee6146feac861e19b563a");
    }
}
