//! LMS implementation.
//!
//! This follows RFC 8554, and additional parameter sets from
//! draft-fluhrer-lms-more-parm-sets-09.txt (which itself copies
//! the parameters from NIST SP 800-208).
//!
//! WARNING: LMS is a stateful signature scheme; each signature modifies
//! the private key. If the same private key state is used to generate
//! two signatures (even if both are on the same data), then attackers
//! observing the two signature values learn enough to make forgeries.
//! This implementation always mutates the `PrivateKey` structure when
//! generating a signature, but it is up to the caller to ensure that the
//! new private key state is properly committed to stable storage before
//! showing the signature value to any third party.
//!
//! This code was written mostly for verifying the test vectors in the
//! parameter sets in the new draft; it does not include facilities for
//! serialization of private and public keys. It also does NOT implement
//! HSS, the hierarchical scheme that builds on top of LMS (in RFC 8554,
//! section 6).

// We use the constant names from RFC 8554, which do not following the
// default casing style rules of Rust.
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

macro_rules! define_lms_core { () => {

    use crate::{CryptoRng, RngCore};
    use core::convert::TryFrom;

    #[derive(Clone, Copy, Debug)]
    pub struct PrivateKey {
        I: [u8; 16],
        SEED: [u8; m],
        current_leaf: u32,
        T: [[u8; m]; 1usize << (h + 1)],
    }

    #[derive(Clone, Copy, Debug)]
    pub struct PublicKey {
        I: [u8; 16],
        T1: [u8; m],
    }

    const p: usize = make_p();
    const ls: usize = make_ls();

    const fn make_p() -> usize {
        let (zp, _) = make_p_ls();
        zp
    }

    const fn make_ls() -> usize {
        let (_, zls) = make_p_ls();
        zls
    }

    const fn make_p_ls() -> (usize, usize) {
        let u = (8 * n + w - 1) / w;
        let x = ((1usize << w) - 1) * u;
        let (x0, t0) = if x > 0xFFFF { (x >> 16, 16) } else { (x, 0) };
        let (x1, t1) = if x0 > 0xFF { (x0 >> 8, 8) } else { (x0, 0) };
        let (x2, t2) = if x1 > 0x0F { (x1 >> 4, 4) } else { (x1, 0) };
        let (x3, t3) = if x2 > 0x03 { (x2 >> 2, 2) } else { (x2, 0) };
        let t4 = if x3 > 0x01 { 1 } else { 0 };
        let lg = t0 + t1 + t2 + t3 + t4;
        let v = (lg + w) / w;
        let zp = u + v;
        let zls = 16 - (v * w);
        (zp, zls)
    }

    const Z: [u8; 0] = [];
    const D_PBLC: [u8; 2] = [ 0x80, 0x80, ];
    const D_MESG: [u8; 2] = [ 0x81, 0x81, ];
    const D_LEAF: [u8; 2] = [ 0x82, 0x82, ];
    const D_INTR: [u8; 2] = [ 0x83, 0x83, ];

    const ots_siglen: usize = 4 + n + n * p;
    const lms_siglen: usize = 4 + ots_siglen + 4 + h * m;

    fn checksum(Q: &[u8]) -> u16 {
        let mut sum = 0u16;
        for i in 0..((n * 8) / w) {
            sum = sum.wrapping_add((1u16 << w) - 1);
            sum = sum.wrapping_sub(coef(Q, i) as u16);
        }
        sum << ls
    }

    fn coef(Q: &[u8], i: usize) -> u8 {
        let m8 = ((1u32 << w) - 1) as u8;
        (Q[(i * w) / 8] >> (8 - (w * (i % (8 / w)) + w))) & m8
    }

    impl PrivateKey {

        pub fn generate<T: CryptoRng + RngCore>(rng: &mut T) -> Self {
            let mut I = [0u8; 16];
            let mut SEED = [0u8; m];
            rng.fill_bytes(&mut I);
            rng.fill_bytes(&mut SEED);
            let current_leaf = 0u32;
            let mut sk = Self {
                I, SEED, current_leaf,
                T: [[0u8; m]; 1usize << (h + 1)],
            };
            sk.compute_tree();
            sk
        }

        fn compute_tree(&mut self) {
            for r in (1u32 << h)..(1u32 << (h + 1)) {
                let q = r - (1u32 << h);
                let x = self.make_ots_x(q);
                let y = self.make_ots_pub_y(q, &x);
                self.T[r as usize] = Hm(&self.I, &r.to_be_bytes(), &D_LEAF,
                    &self.make_ots_pub_hash(q, &y), &Z);
            }
            for r in (1..(1u32 << h)).rev() {
                self.T[r as usize] = Hm(&self.I, &r.to_be_bytes(), &D_INTR,
                    &self.T[(2 * r) as usize], &self.T[(2 * r + 1) as usize]);
            }
        }

        pub fn compute_public(self) -> PublicKey {
            PublicKey { I: self.I, T1: self.T[1] }
        }

        fn make_ots_x(self, q: u32) -> [[u8; n]; p] {
            let mut x = [[0u8; n]; p];
            let eq = q.to_be_bytes();
            for i in 0..p {
                x[i] = Hn(&self.I, &eq, &(i as u16).to_be_bytes(),
                    &[0xFFu8], &self.SEED);
            }
            x
        }

        fn make_ots_pub_y(self, q: u32, x: &[[u8; n]; p]) -> [[u8; n]; p] {
            let mut y = [[0u8; n]; p];
            let eq = q.to_be_bytes();
            for i in 0..p {
                let mut tmp = x[i];
                for j in 0..((1 << w) - 1) {
                    tmp = Hn(&self.I, &eq, &(i as u16).to_be_bytes(),
                        &[j as u8], &tmp);
                }
                y[i] = tmp;
            }
            y
        }

        fn make_ots_pub_hash(self, q: u32, y: &[[u8; n]; p]) -> [u8; n] {
            Hnx(&self.I, &q.to_be_bytes(), &D_PBLC, &y)
        }

        fn ots_sign<T: CryptoRng + RngCore>(self, rng: &mut T,
            q: u32, msg: &[u8]) -> [u8; ots_siglen]
        {
            let mut sig = [0u8; ots_siglen];
            sig[0..4].copy_from_slice(&ots_type.to_be_bytes());

            let mut C = [0u8; n];
            rng.fill_bytes(&mut C);
            sig[4..(n + 4)].copy_from_slice(&C);

            let Q = Hn(&self.I, &q.to_be_bytes(), &D_MESG, &C, msg);
            let mut Qck = [0u8; n + 2];
            Qck[..n].copy_from_slice(&Q);
            Qck[n..].copy_from_slice(&(checksum(&Q).to_be_bytes()));
            let x = self.make_ots_x(q);
            let eq = q.to_be_bytes();
            for i in 0..p {
                let a = coef(&Qck, i);
                let mut tmp = x[i];
                for j in 0..(a as usize) {
                    tmp = Hn(&self.I, &eq, &(i as u16).to_be_bytes(),
                        &[j as u8], &tmp);
                }
                sig[(4 + n * (i + 1))..(4 + n * (i + 2))].copy_from_slice(&tmp);
            }
            sig
        }

        pub fn sign<T: CryptoRng + RngCore>(&mut self, rng: &mut T, msg: &[u8])
            -> Option<[u8; 4 + ots_siglen + 4 + h * m]>
        {
            let q = self.current_leaf;
            if q >= (1u32 << h) {
                return None;
            }
            self.current_leaf = q + 1;
            let ots_sig = self.ots_sign(rng, q, msg);
            let mut sig = [0u8; 4 + ots_siglen + 4 + h * m];
            sig[0..4].copy_from_slice(&q.to_be_bytes());
            sig[4..(ots_siglen + 4)].copy_from_slice(&ots_sig);
            sig[(ots_siglen + 4)..(ots_siglen + 8)].copy_from_slice(&key_type.to_be_bytes());
            let mut r = q + (1u32 << h);
            for i in 0..h {
                let k = if (r & 1) == 0 { r + 1 } else { r - 1 };
                let j = 4 + ots_siglen + 4 + i * m;
                sig[j..(j + m)].copy_from_slice(&self.T[k as usize]);
                r = r >> 1;
            }
            Some(sig)
        }
    }

    impl PublicKey {

        fn ots_verify(self, q: u32, sig: &[u8], msg: &[u8]) -> Option<[u8; n]> {
            if sig.len() != ots_siglen {
                return None;
            }
            let st = u32::from_be_bytes(*<&[u8; 4]>::try_from(&sig[0..4]).unwrap());
            if st != ots_type {
                return None;
            }
            let C = &sig[4..(4 + n)];
            let yy = &sig[(4 + n)..];
            let eq = q.to_be_bytes();
            let Q = Hn(&self.I, &eq, &D_MESG, C, msg);
            let mut Qck = [0u8; n + 2];
            Qck[..n].copy_from_slice(&Q);
            Qck[n..].copy_from_slice(&(checksum(&Q).to_be_bytes()));
            let mut z = [[0u8; n]; p];
            for i in 0..p {
                let a = coef(&Qck, i);
                let mut tmp = [0u8; n];
                tmp.copy_from_slice(&yy[(i * n)..((i + 1) * n)]);
                for j in (a as usize)..((1usize << w) - 1) {
                    tmp = Hn(&self.I, &eq, &(i as u16).to_be_bytes(),
                        &[j as u8], &tmp);
                }
                z[i] = tmp;
            }
            Some(Hnx(&self.I, &eq, &D_PBLC, &z))
        }

        pub fn verify(self, sig: &[u8], msg: &[u8]) -> bool {
            if sig.len() != lms_siglen {
                return false;
            }
            let q = u32::from_be_bytes(*<&[u8; 4]>::try_from(&sig[0..4]).unwrap());
            if q >= (1u32 << h) {
                return false;
            }
            let st = u32::from_be_bytes(*<&[u8; 4]>::try_from(&sig[(ots_siglen + 4)..(ots_siglen + 8)]).unwrap());
            if st != key_type {
                return false;
            }
            let ots_sig = &sig[4..(4 + ots_siglen)];
            let Kc = match self.ots_verify(q, ots_sig, msg) {
                None => return false,
                Some(kk) => kk,
            };
            let mut r = (1u32 << h) + q;
            let mut tmp = Hm(&self.I, &r.to_be_bytes(), &D_LEAF, &Kc, &Z);
            let path = &sig[(4 + ots_siglen + 4)..];
            for i in 0..h {
                let nno = (r & 1) != 0;
                r = r >> 1;
                if nno {
                    tmp = Hm(&self.I, &r.to_be_bytes(), &D_INTR,
                        &path[(i * m)..((i + 1) * m)], &tmp);
                } else {
                    tmp = Hm(&self.I, &r.to_be_bytes(), &D_INTR,
                        &tmp, &path[(i * m)..((i + 1) * m)]);
                }
            }
            tmp == self.T1
        }
    }

} } // end of macro define_lms_core

// ========================================================================

#[cfg(test)]
macro_rules! define_lms_tests { () => {

    use super::{PrivateKey};
    use crate::{CryptoRng, RngCore, RngError};
    use core::num::NonZeroU32;

    // A pretend RNG for test purposes (returns fixed values).
    struct FRNG<'a> {
        tape: &'a [u8],
        ptr: usize,
    }

    impl<'a> FRNG<'a> {

        fn from_tape(tape: &'a [u8]) -> Self {
            Self { tape, ptr: 0 }
        }
    }

    impl<'a> RngCore for FRNG<'a> {

        fn next_u32(&mut self) -> u32 {
            let mut buf = [0u8; 4];
            self.fill_bytes(&mut buf);
            u32::from_le_bytes(buf)
        }

        fn next_u64(&mut self) -> u64 {
            let mut buf = [0u8; 8];
            self.fill_bytes(&mut buf);
            u64::from_le_bytes(buf)
        }

        fn fill_bytes(&mut self, dst: &mut [u8]) {
            let ptr = self.ptr;
            let dlen = dst.len();
            dst.copy_from_slice(&self.tape[ptr..(ptr + dlen)]);
            self.ptr = ptr + dlen;
        }

        fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), RngError> {
            if (self.tape.len() - self.ptr) < dst.len() {
                return Err(RngError::from(NonZeroU32::new(RngError::CUSTOM_START + 1).unwrap()));
            }
            self.fill_bytes(dst);
            Ok(())
        }
    }

    impl<'a> CryptoRng for FRNG<'a> { }

    #[test]
    fn kat_lms() {
        let rng_tape = hex::decode(KAT_RNG_TAPE).unwrap();
        let mut rng = FRNG::from_tape(&rng_tape);
        let mut sk = PrivateKey::generate(&mut rng);
        sk.current_leaf = KAT_LEAFNUM;
        let pk = sk.compute_public();

        let Iref = hex::decode(KAT_PK_I).unwrap();
        let T1ref = hex::decode(KAT_PK_T1).unwrap();
        assert!(pk.I[..] == Iref);
        assert!(&pk.T1[..] == T1ref);

        let msg = hex::decode(KAT_MSG).unwrap();
        let sig = sk.sign(&mut rng, &msg).unwrap();
        let sigref = hex::decode(KAT_SIG).unwrap();
        assert!(sig[..] == sigref);

        assert!(pk.verify(&sig, &msg) == true);
        assert!(pk.verify(&sig, &msg[1..]) == false);
    }

} } // end of macro define_lms_tests

// ========================================================================

/// LMS_SHA256_M32_H5 with LMOTS_SHA256_N32_W8
pub mod LMS_SHA256_M32_H5_SHA256_N32_W8 {

    use sha2::{Sha256, Digest};

    define_lms_core!{}

    const n: usize = 32;
    const m: usize = 32;
    const w: usize = 8;
    const h: usize = 5;
    const key_type: u32 = 0x00000005;
    const ots_type: u32 = 0x00000004;

    fn Hn(m1: &[u8], m2: &[u8], m3: &[u8], m4: &[u8], m5: &[u8]) -> [u8; n] {
        let mut sh = Sha256::new();
        sh.update(m1);
        sh.update(m2);
        sh.update(m3);
        sh.update(m4);
        sh.update(m5);
        let mut r = [0u8; n];
        r[..].copy_from_slice(&sh.finalize());
        r
    }

    fn Hnx(m1: &[u8], m2: &[u8], m3: &[u8], mm: &[[u8; n]; p]) -> [u8; n] {
        let mut sh = Sha256::new();
        sh.update(m1);
        sh.update(m2);
        sh.update(m3);
        for i in 0..p {
            sh.update(&mm[i]);
        }
        let mut r = [0u8; n];
        r[..].copy_from_slice(&sh.finalize());
        r
    }

    fn Hm(m1: &[u8], m2: &[u8], m3: &[u8], m4: &[u8], m5: &[u8]) -> [u8; m] {
        let mut sh = Sha256::new();
        sh.update(m1);
        sh.update(m2);
        sh.update(m3);
        sh.update(m4);
        sh.update(m5);
        let mut r = [0u8; m];
        r[..].copy_from_slice(&sh.finalize());
        r
    }

    #[cfg(test)]
    mod tests {

        // Test vector from RFC 8554 (appendix F, test case 2,
        // "final signature")

        static KAT_RNG_TAPE: &str = "215f83b7ccb9acbcd08db97b0d04dc2ba1c4696e2608035a886100d05cd99945eb3370731884a8235e2fb3d4d71f25470eb1ed54a2460d512388cad533138d240534e97b1e82d33bd927d201dfc24ebb";
        static KAT_PK_I: &str = "215f83b7ccb9acbcd08db97b0d04dc2b";
        static KAT_PK_T1: &str = "a1cd035833e0e90059603f26e07ad2aad152338e7a5e5984bcd5f7bb4eba40b7";
        static KAT_MSG: &str = "54686520656e756d65726174696f6e20696e2074686520436f6e737469747574696f6e2c206f66206365727461696e207269676874732c207368616c6c206e6f7420626520636f6e73747275656420746f2064656e79206f7220646973706172616765206f74686572732072657461696e6564206279207468652070656f706c652e0a";
        static KAT_LEAFNUM: u32 = 4u32;
        static KAT_SIG: &str =
            "00000004\
            00000004\
            0eb1ed54a2460d512388cad533138d24\
            0534e97b1e82d33bd927d201dfc24ebb\
            11b3649023696f85150b189e50c00e98\
            850ac343a77b3638319c347d7310269d\
            3b7714fa406b8c35b021d54d4fdada7b\
            9ce5d4ba5b06719e72aaf58c5aae7aca\
            057aa0e2e74e7dcfd17a0823429db629\
            65b7d563c57b4cec942cc865e29c1dad\
            83cac8b4d61aacc457f336e6a10b6632\
            3f5887bf3523dfcadee158503bfaa89d\
            c6bf59daa82afd2b5ebb2a9ca6572a60\
            67cee7c327e9039b3b6ea6a1edc7fdc3\
            df927aade10c1c9f2d5ff446450d2a39\
            98d0f9f6202b5e07c3f97d2458c69d3c\
            8190643978d7a7f4d64e97e3f1c4a08a\
            7c5bc03fd55682c017e2907eab07e5bb\
            2f190143475a6043d5e6d5263471f4ee\
            cf6e2575fbc6ff37edfa249d6cda1a09\
            f797fd5a3cd53a066700f45863f04b6c\
            8a58cfd341241e002d0d2c0217472bf1\
            8b636ae547c1771368d9f317835c9b0e\
            f430b3df4034f6af00d0da44f4af7800\
            bc7a5cf8a5abdb12dc718b559b74cab9\
            090e33cc58a955300981c420c4da8ffd\
            67df540890a062fe40dba8b2c1c548ce\
            d22473219c534911d48ccaabfb71bc71\
            862f4a24ebd376d288fd4e6fb06ed870\
            5787c5fedc813cd2697e5b1aac1ced45\
            767b14ce88409eaebb601a93559aae89\
            3e143d1c395bc326da821d79a9ed41dc\
            fbe549147f71c092f4f3ac522b5cc572\
            90706650487bae9bb5671ecc9ccc2ce5\
            1ead87ac01985268521222fb9057df7e\
            d41810b5ef0d4f7cc67368c90f573b1a\
            c2ce956c365ed38e893ce7b2fae15d36\
            85a3df2fa3d4cc098fa57dd60d2c9754\
            a8ade980ad0f93f6787075c3f680a2ba\
            1936a8c61d1af52ab7e21f416be09d2a\
            8d64c3d3d8582968c2839902229f85ae\
            e297e717c094c8df4a23bb5db658dd37\
            7bf0f4ff3ffd8fba5e383a48574802ed\
            545bbe7a6b4753533353d73706067640\
            135a7ce517279cd683039747d218647c\
            86e097b0daa2872d54b8f3e508598762\
            9547b830d8118161b65079fe7bc59a99\
            e9c3c7380e3e70b7138fe5d9be255150\
            2b698d09ae193972f27d40f38dea264a\
            0126e637d74ae4c92a6249fa103436d3\
            eb0d4029ac712bfc7a5eacbdd7518d6d\
            4fe903a5ae65527cd65bb0d4e9925ca2\
            4fd7214dc617c150544e423f450c99ce\
            51ac8005d33acd74f1bed3b17b7266a4\
            a3bb86da7eba80b101e15cb79de9a207\
            852cf91249ef480619ff2af8cabca831\
            25d1faa94cbb0a03a906f683b3f47a97\
            c871fd513e510a7a25f283b196075778\
            496152a91c2bf9da76ebe089f4654877\
            f2d586ae7149c406e663eadeb2b5c7e8\
            2429b9e8cb4834c83464f079995332e4\
            b3c8f5a72bb4b8c6f74b0d45dc6c1f79\
            952c0b7420df525e37c15377b5f09843\
            19c3993921e5ccd97e097592064530d3\
            3de3afad5733cbe7703c5296263f7734\
            2efbf5a04755b0b3c997c4328463e84c\
            aa2de3ffdcd297baaaacd7ae646e44b5\
            c0f16044df38fabd296a47b3a838a913\
            982fb2e370c078edb042c84db34ce36b\
            46ccb76460a690cc86c302457dd1cde1\
            97ec8075e82b393d542075134e2a17ee\
            70a5e187075d03ae3c853cff60729ba4\
            00000005\
            4de1f6965bdabc676c5a4dc7c35f97f8\
            2cb0e31c68d04f1dad96314ff09e6b3d\
            e96aeee300d1f68bf1bca9fc58e40323\
            36cd819aaf578744e50d1357a0e42867\
            04d341aa0a337b19fe4bc43c2e79964d\
            4f351089f2e0e41c7c43ae0d49e7f404\
            b0f75be80ea3af098c9752420a8ac0ea\
            2bbb1f4eeba05238aef0d8ce63f0c6e5\
            e4041d95398a6f7f3e0ee97cc1591849\
            d4ed236338b147abde9f51ef9fd4e1c1";

        define_lms_tests!{}
    }
}

/// LMS_SHA256_M24_H5 with LMOTS_SHA256_N24_W8
pub mod LMS_SHA256_M24_H5_SHA256_N24_W8 {

    use sha2::{Sha256, Digest};

    define_lms_core!{}

    const n: usize = 24;
    const m: usize = 24;
    const w: usize = 8;
    const h: usize = 5;
    const key_type: u32 = 0x0000000a;
    const ots_type: u32 = 0x00000008;

    fn Hn(m1: &[u8], m2: &[u8], m3: &[u8], m4: &[u8], m5: &[u8]) -> [u8; n] {
        let mut sh = Sha256::new();
        sh.update(m1);
        sh.update(m2);
        sh.update(m3);
        sh.update(m4);
        sh.update(m5);
        let mut r = [0u8; n];
        r[..].copy_from_slice(&sh.finalize()[..24]);
        r
    }

    fn Hnx(m1: &[u8], m2: &[u8], m3: &[u8], mm: &[[u8; n]; p]) -> [u8; n] {
        let mut sh = Sha256::new();
        sh.update(m1);
        sh.update(m2);
        sh.update(m3);
        for i in 0..p {
            sh.update(&mm[i]);
        }
        let mut r = [0u8; n];
        r[..].copy_from_slice(&sh.finalize()[..24]);
        r
    }

    fn Hm(m1: &[u8], m2: &[u8], m3: &[u8], m4: &[u8], m5: &[u8]) -> [u8; m] {
        let mut sh = Sha256::new();
        sh.update(m1);
        sh.update(m2);
        sh.update(m3);
        sh.update(m4);
        sh.update(m5);
        let mut r = [0u8; m];
        r[..].copy_from_slice(&sh.finalize()[..24]);
        r
    }

    #[cfg(test)]
    mod tests {

        // Test vector from draft-fluhrer-lms-more-parm-sets-09.txt
        // ("Test Case 1")

        static KAT_RNG_TAPE: &str = "202122232425262728292a2b2c2d2e2f000102030405060708090a0b0c0d0e0f10111213141516170b5040a18c1b5cabcbc85b047402ec6294a30dd8da8fc3da";
        static KAT_PK_I: &str = "202122232425262728292a2b2c2d2e2f";
        static KAT_PK_T1: &str = "2c571450aed99cfb4f4ac285da14882796618314508b12d2";
        static KAT_MSG: &str = "54657374206d65737361676520666f72205348413235362d3139320a";
        static KAT_LEAFNUM: u32 = 5u32;
        static KAT_SIG: &str =
            "00000005\
            00000008\
            0b5040a18c1b5cabcbc85b047402ec62\
            94a30dd8da8fc3da\
            e13b9f0875f09361dc77fcc4481ea463\
            c073716249719193\
            614b835b4694c059f12d3aedd34f3db9\
            3f3580fb88743b8b\
            3d0648c0537b7a50e433d7ea9d6672ff\
            fc5f42770feab4f9\
            8eb3f3b23fd2061e4d0b38f832860ae7\
            6673ad1a1a52a900\
            5dcf1bfb56fe16ff723627612f9a48f7\
            90f3c47a67f870b8\
            1e919d99919c8db48168838cece0abfb\
            683da48b9209868b\
            e8ec10c63d8bf80d36498dfc205dc45d\
            0dd870572d6d8f1d\
            90177cf5137b8bbf7bcb67a46f86f26c\
            fa5a44cbcaa4e18d\
            a099a98b0b3f96d5ac8ac375d8da2a7c\
            248004ba11d7ac77\
            5b9218359cddab4cf8ccc6d54cb7e1b3\
            5a36ddc9265c0870\
            63d2fc6742a7177876476a324b03295b\
            fed99f2eaf1f3897\
            0583c1b2b616aad0f31cd7a4b1bb0a51\
            e477e94a01bbb4d6\
            f8866e2528a159df3d6ce244d2b6518d\
            1f0212285a3c2d4a\
            927054a1e1620b5b02aab0c8c10ed48a\
            e518ea73cba81fcf\
            ff88bff461dac51e7ab4ca75f47a6259\
            d24820b9995792d1\
            39f61ae2a8186ae4e3c9bfe0af2cc717\
            f424f41aa67f03fa\
            edb0665115f2067a46843a4cbbd297d5\
            e83bc1aafc18d1d0\
            3b3d894e8595a6526073f02ab0f08b99\
            fd9eb208b59ff631\
            7e5545e6f9ad5f9c183abd043d5acd6e\
            b2dd4da3f02dbc31\
            67b468720a4b8b92ddfe7960998bb7a0\
            ecf2a26a37598299\
            413f7b2aecd39a30cec527b4d9710c44\
            73639022451f50d0\
            1c0457125da0fa4429c07dad859c846c\
            bbd93ab5b91b01bc\
            770b089cfede6f651e86dd7c15989c8b\
            5321dea9ca608c71\
            fd862323072b827cee7a7e28e4e2b999\
            647233c3456944bb\
            7aef9187c96b3f5b79fb98bc76c3574d\
            d06f0e95685e5b3a\
            ef3a54c4155fe3ad817749629c30adbe\
            897c4f4454c86c49\
            0000000a\
            e9ca10eaa811b22ae07fb195e3590a33\
            4ea64209942fbae3\
            38d19f152182c807d3c40b189d3fcbea\
            942f44682439b191\
            332d33ae0b761a2a8f984b56b2ac2fd4\
            ab08223a69ed1f77\
            19c7aa7e9eee96504b0e60c6bb5c942d\
            695f0493eb25f80a\
            5871cffd131d0e04ffe5065bc7875e82\
            d34b40b69dd9f3c1";

        define_lms_tests!{}
    }
}

/// LMS_SHAKE_M24_H5 with LMOTS_SHAKE_N24_W8
pub mod LMS_SHAKE_M24_H5_SHAKE_N24_W8 {

    use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};

    define_lms_core!{}

    const n: usize = 24;
    const m: usize = 24;
    const w: usize = 8;
    const h: usize = 5;
    const key_type: u32 = 0x00000014;
    const ots_type: u32 = 0x00000010;

    fn Hn(m1: &[u8], m2: &[u8], m3: &[u8], m4: &[u8], m5: &[u8]) -> [u8; n] {
        let mut sh = Shake256::default();
        sh.update(m1);
        sh.update(m2);
        sh.update(m3);
        sh.update(m4);
        sh.update(m5);
        let mut r = [0u8; n];
        sh.finalize_xof().read(&mut r);
        r
    }

    fn Hnx(m1: &[u8], m2: &[u8], m3: &[u8], mm: &[[u8; n]; p]) -> [u8; n] {
        let mut sh = Shake256::default();
        sh.update(m1);
        sh.update(m2);
        sh.update(m3);
        for i in 0..p {
            sh.update(&mm[i]);
        }
        let mut r = [0u8; n];
        sh.finalize_xof().read(&mut r);
        r
    }

    fn Hm(m1: &[u8], m2: &[u8], m3: &[u8], m4: &[u8], m5: &[u8]) -> [u8; m] {
        let mut sh = Shake256::default();
        sh.update(m1);
        sh.update(m2);
        sh.update(m3);
        sh.update(m4);
        sh.update(m5);
        let mut r = [0u8; m];
        sh.finalize_xof().read(&mut r);
        r
    }

    #[cfg(test)]
    mod tests {

        // Test vector from draft-fluhrer-lms-more-parm-sets-09.txt
        // ("Test Case 2")

        static KAT_RNG_TAPE: &str = "505152535455565758595a5b5c5d5e5f303132333435363738393a3b3c3d3e3f404142434445464784219da9ce9fffb16edb94527c6d10565587db28062deac4";
        static KAT_PK_I: &str = "505152535455565758595a5b5c5d5e5f";
        static KAT_PK_T1: &str = "db54a4509901051c01e26d9990e550347986da87924ff0b1";
        static KAT_MSG: &str = "54657374206d65737361676520666f72205348414b453235362d3139320a";
        static KAT_LEAFNUM: u32 = 6u32;
        static KAT_SIG: &str =
            "00000006\
            00000010\
            84219da9ce9fffb16edb94527c6d1056\
            5587db28062deac4\
            208e62fc4fbe9d85deb3c6bd2c01640a\
            ccb387d8a6093d68\
            511234a6a1a50108091c034cb1777e02\
            b5df466149a66969\
            a498e4200c0a0c1bf5d100cdb97d2dd4\
            0efd3cada278acc5\
            a570071a043956112c6deebd1eb3a7b5\
            6f5f6791515a7b5f\
            fddb0ec2d9094bfbc889ea15c3c7b9be\
            a953efb75ed648f5\
            35b9acab66a2e9631e426e4e99b733ca\
            a6c55963929b77fe\
            c54a7e703d8162e736875cb6a455d4a9\
            015c7a6d8fd5fe75\
            e402b47036dc3770f4a1dd0a559cb478\
            c7fb1726005321be\
            9d1ac2de94d731ee4ca79cff454c811f\
            46d11980909f047b\
            2005e84b6e15378446b1ca691efe491e\
            a98acc9d3c0f785c\
            aba5e2eb3c306811c240ba2280292382\
            7d582639304a1e97\
            83ba5bc9d69d999a7db8f749770c3c04\
            a152856dc726d806\
            7921465b61b3f847b13b2635a45379e5\
            adc6ff58a99b00e6\
            0ac767f7f30175f9f7a140257e218be3\
            07954b1250c9b419\
            02c4fa7c90d8a592945c66e86a76defc\
            b84500b55598a199\
            0faaa10077c74c94895731585c8f900d\
            e1a1c675bd8b0c18\
            0ebe2b5eb3ef8019ece3e1ea7223eb79\
            06a2042b6262b4aa\
            25c4b8a05f205c8befeef11ceff12825\
            08d71bc2a8cfa0a9\
            9f73f3e3a74bb4b3c0d8ca2abd0e1c2c\
            17dafe18b4ee2298\
            e87bcfb1305b3c069e6d385569a4067e\
            d547486dd1a50d6f\
            4a58aab96e2fa883a9a39e1bd45541ee\
            e94efc32faa9a94b\
            e66dc8538b2dab05aee5efa6b3b2efb3\
            fd020fe789477a93\
            afff9a3e636dbba864a5bffa3e28d13d\
            49bb597d94865bde\
            88c4627f206ab2b465084d6b780666e9\
            52f8710efd748bd0\
            f1ae8f1035087f5028f14affcc5fffe3\
            32121ae4f87ac5f1\
            eac9062608c7d87708f1723f38b23237\
            a4edf4b49a5cd3d7\
            00000014\
            dd4bdc8f928fb526f6fb7cdb944a7eba\
            a7fb05d995b5721a\
            27096a5007d82f79d063acd434a04e97\
            f61552f7f81a9317\
            b4ec7c87a5ed10c881928fc6ebce6dfc\
            e9daae9cc9dba690\
            7ca9a9dd5f9f573704d5e6cf22a43b04\
            e64c1ffc7e1c442e\
            cb495ba265f465c56291a902e62a461f\
            6dfda232457fad14";

        define_lms_tests!{}
    }
}

/// LMS_SHAKE_M32_H5 with LMOTS_SHAKE_N32_W8
pub mod LMS_SHAKE_M32_H5_SHAKE_N32_W8 {

    use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};

    define_lms_core!{}

    const n: usize = 32;
    const m: usize = 32;
    const w: usize = 8;
    const h: usize = 5;
    const key_type: u32 = 0x0000000f;
    const ots_type: u32 = 0x0000000c;

    fn Hn(m1: &[u8], m2: &[u8], m3: &[u8], m4: &[u8], m5: &[u8]) -> [u8; n] {
        let mut sh = Shake256::default();
        sh.update(m1);
        sh.update(m2);
        sh.update(m3);
        sh.update(m4);
        sh.update(m5);
        let mut r = [0u8; n];
        sh.finalize_xof().read(&mut r);
        r
    }

    fn Hnx(m1: &[u8], m2: &[u8], m3: &[u8], mm: &[[u8; n]; p]) -> [u8; n] {
        let mut sh = Shake256::default();
        sh.update(m1);
        sh.update(m2);
        sh.update(m3);
        for i in 0..p {
            sh.update(&mm[i]);
        }
        let mut r = [0u8; n];
        sh.finalize_xof().read(&mut r);
        r
    }

    fn Hm(m1: &[u8], m2: &[u8], m3: &[u8], m4: &[u8], m5: &[u8]) -> [u8; m] {
        let mut sh = Shake256::default();
        sh.update(m1);
        sh.update(m2);
        sh.update(m3);
        sh.update(m4);
        sh.update(m5);
        let mut r = [0u8; m];
        sh.finalize_xof().read(&mut r);
        r
    }

    #[cfg(test)]
    mod tests {

        // Test vector from draft-fluhrer-lms-more-parm-sets-09.txt
        // ("Test Case 3")

        static KAT_RNG_TAPE: &str = "808182838485868788898a8b8c8d8e8f606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7fb82709f0f00e83759190996233d1ee4f4ec50534473c02ffa145e8ca2874e32b";
        static KAT_PK_I: &str = "808182838485868788898a8b8c8d8e8f";
        static KAT_PK_T1: &str = "9bb7faee411cae806c16a466c3191a8b65d0ac31932bbf0c2d07c7a4a36379fe";
        static KAT_MSG: &str = "54657374206d657361676520666f72205348414b453235362d3235360a";
        static KAT_LEAFNUM: u32 = 7u32;
        static KAT_SIG: &str =
            "00000007\
            0000000c\
            b82709f0f00e83759190996233d1ee4f\
            4ec50534473c02ffa145e8ca2874e32b\
            16b228118c62b96c9c77678b33183730\
            debaade8fe607f05c6697bc971519a34\
            1d69c00129680b67e75b3bd7d8aa5c8b\
            71f02669d177a2a0eea896dcd1660f16\
            864b302ff321f9c4b8354408d0676050\
            4f768ebd4e545a9b0ac058c575078e6c\
            1403160fb45450d61a9c8c81f6bd69bd\
            fa26a16e12a265baf79e9e233eb71af6\
            34ecc66dc88e10c6e0142942d4843f70\
            a0242727bc5a2aabf7b0ec12a99090d8\
            caeef21303f8ac58b9f200371dc9e41a\
            b956e1a3efed9d4bbb38975b46c28d5f\
            5b3ed19d847bd0a737177263cbc1a226\
            2d40e80815ee149b6cce2714384c9b7f\
            ceb3bbcbd25228dda8306536376f8793\
            ecadd6020265dab9075f64c773ef97d0\
            7352919995b74404cc69a6f3b469445c\
            9286a6b2c9f6dc839be76618f053de76\
            3da3571ef70f805c9cc54b8e501a98b9\
            8c70785eeb61737eced78b0e380ded4f\
            769a9d422786def59700eef3278017ba\
            bbe5f9063b468ae0dd61d94f9f99d5cc\
            36fbec4178d2bda3ad31e1644a2bcce2\
            08d72d50a7637851aa908b94dc437612\
            0d5beab0fb805e1945c41834dd6085e6\
            db1a3aa78fcb59f62bde68236a10618c\
            ff123abe64dae8dabb2e84ca705309c2\
            ab986d4f8326ba0642272cb3904eb96f\
            6f5e3bb8813997881b6a33cac0714e4b\
            5e7a882ad87e141931f97d612b84e903\
            e773139ae377f5ba19ac86198d485fca\
            97742568f6ff758120a89bf19059b8a6\
            bfe2d86b12778164436ab2659ba86676\
            7fcc435584125fb7924201ee67b535da\
            f72c5cb31f5a0b1d926324c26e67d4c3\
            836e301aa09bae8fb3f91f1622b1818c\
            cf440f52ca9b5b9b99aba8a6754aae2b\
            967c4954fa85298ad9b1e74f27a46127\
            c36131c8991f0cc2ba57a15d35c91cf8\
            bc48e8e20d625af4e85d8f9402ec44af\
            bd4792b924b839332a64788a7701a300\
            94b9ec4b9f4b648f168bf457fbb3c959\
            4fa87920b645e42aa2fecc9e21e000ca\
            7d3ff914e15c40a8bc533129a7fd3952\
            9376430f355aaf96a0a13d13f2419141\
            b3cc25843e8c90d0e551a355dd90ad77\
            0ea7255214ce11238605de2f000d2001\
            04d0c3a3e35ae64ea10a3eff37ac7e95\
            49217cdf52f307172e2f6c7a2a4543e1\
            4314036525b1ad53eeaddf0e24b1f369\
            14ed22483f2889f61e62b6fb78f5645b\
            dbb02c9e5bf97db7a0004e87c2a55399\
            b61958786c97bd52fa199c27f6bb4d68\
            c4907933562755bfec5d4fb52f06c289\
            d6e852cf6bc773ffd4c07ee2d6cc55f5\
            7edcfbc8e8692a49ad47a121fe3c1b16\
            cab1cc285faf6793ffad7a8c341a49c5\
            d2dce7069e464cb90a00b2903648b23c\
            81a68e21d748a7e7b1df8a593f3894b2\
            477e8316947ca725d141135202a9442e\
            1db33bbd390d2c04401c39b253b78ce2\
            97b0e14755e46ec08a146d279c67af70\
            de256890804d83d6ec5ca3286f1fca9c\
            72abf6ef868e7f6eb0fddda1b040ecec\
            9bbc69e2fd8618e9db3bdb0af13dda06\
            c6617e95afa522d6a2552de15324d991\
            19f55e9af11ae3d5614b564c642dbfec\
            6c644198ce80d2433ac8ee738f9d825e\
            0000000f\
            71d585a35c3a908379f4072d070311db\
            5d65b242b714bc5a756ba5e228abfa0d\
            1329978a05d5e815cf4d74c1e547ec4a\
            a3ca956ae927df8b29fb9fab3917a7a4\
            ae61ba57e5342e9db12caf6f6dbc5253\
            de5268d4b0c4ce4ebe6852f012b162fc\
            1c12b9ffc3bcb1d3ac8589777655e22c\
            d9b99ff1e4346fd0efeaa1da044692e7\
            ad6bfc337db69849e54411df8920c228\
            a2b7762c11e4b1c49efb74486d3931ea";

        define_lms_tests!{}
    }
}
