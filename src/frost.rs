//! FROST implementation.
//!
//! This follows the v8 draft specification: [draft-irtf-cfrg-frost-08]
//!
//! FROST is a threshold Schnorr signature scheme: the group private key
//! is split into individual signer shares. If enough signers (with a
//! configurable threshold) collaborate, then they can conjointly generate
//! a signature on a given message. The individual signers do not have to
//! trust each other (the protocol is resilient to actively malicious
//! signers, who may at worst prevent the generation of a valid signature).
//! Output signatures are "plain" Schnorr signatures, verifiable against the
//! group public key. When the ciphersuite is FROST(Ed25519, SHA-512), the
//! generated signatures can also be successfully verified with a plain
//! Ed25519 verifier (as per RFC 8032).
//!
//! Single-signer usage is also supported: message signatures can be
//! generated from the group private key itself. In distributed signature
//! usage, _nobody_ knows the group private key itself once it has been
//! split into individual signer key shares.
//!
//! Sub-modules are defined for several ciphersuites:
//!
//!  - `ed25519`: FROST(Ed25519, SHA-512)
//!  - `ristretto255`: FROST(ristretto255, SHA-512)
//!  - `p256`: FROST(P-256, SHA-256)
//!  - `secp256k1`: FROST(secp256k1, SHA-256)
//!
//! All sub-modules implement the same API, with the following types:
//!
//!  - `GroupPrivateKey`: a group private key
//!  - `GroupPublicKey`: a group public key
//!  - `SignerPrivateKeyShare`: an individual signer's private key share
//!  - `SignerPublicKey`: an individual signer's public key
//!  - `KeySplitter`: tagging structure for the trusted dealer, who
//!    splits the group private into individual key shares
//!  - `VSSElement`: an element of the VSS commitment produced by the trusted
//!    dealer (the VSS commitment allows individual signers to validate that
//!    their private key share was properly generated)
//!  - `Coordinator`: the permanent state of a coordinator, who organizes
//!    the signature generation and assembles the signature shares (that
//!    state consists of the signature threshold and the group public key)
//!  - `Nonce`: a per-signature nonce produced by an individual signer
//!  - `Commitment`: a per-signature commitment produced by an individual signer
//!  - `SignatureShare`: a signature share, produced by an individual signer
//!  - `Signature`: a generated FROST signature
//!
//! All the types that are meant to be either transmitted or stored on a
//! non-volatile medium have encoding and decoding functions; the encoding
//! functions return a fixed-size array of bytes (the size is published as
//! the `ENC_LEN` constant in the structure) while the decoding function
//! takes as input a slice of bytes and returns an `Option` type.
//!
//! Sample code using the FROST API is available in [frost-sample.rs].
//!
//! The implementation of all operations involving secret values is
//! constant-time.
//!
//! [draft-irtf-cfrg-frost-08]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-frost-08
//! [frost-sample.rs]: https://github.com/pornin/crrl/extra/frost-sample.rs

#![allow(non_snake_case)]

/// Most functions are generic, provided that the relevant Point and
/// Scalar types are in scope, and a few constants defined. This macro
/// generates the generic functions. The caller is supposed to invoke it
/// in an appropriate module with Point and Scalar already defined.
macro_rules! define_frost_core { () => {

    use crate::{CryptoRng, RngCore};
    use core::convert::TryFrom;
    use crate::Vec;

    /// A group private key.
    ///
    /// In normal FROST usage, the group private key is not supposed to be
    /// kept anywhere once the private key shares have been computed. In
    /// single-signer usage, the group private key is handled like a
    /// normal cryptographic private key.
    #[derive(Clone, Copy, Debug)]
    pub struct GroupPrivateKey {
        sk: Scalar,
        pk: Point,
        pk_enc: [u8; NE],   // keep cached copy of the encoded public key
    }

    /// A group public key.
    #[derive(Clone, Copy, Debug)]
    pub struct GroupPublicKey {
        pk: Point,
        pk_enc: [u8; NE],   // keep cached copy of the encoded public key
    }

    /// A tagging structure for functions related to key splitting; it
    /// does not contain any state.
    #[derive(Clone, Copy, Debug)]
    pub struct KeySplitter { }

    /// A private key share.
    ///
    /// This structure contains the private key share of a given signer.
    /// It includes the signer's identifier, private key (a scalar),
    /// corresponding public key (a point), and group public key. The
    /// signer's public key is recomputed from the signer's private key
    /// when decoding, so it always matches the signers private key.
    #[derive(Clone, Copy, Debug)]
    pub struct SignerPrivateKeyShare {
        /// Signer identifier
        pub ident: u16,
        sk: Scalar,
        pk: Point,
        group_pk: GroupPublicKey,
    }

    /// A signer's public key.
    ///
    /// This structure contains the public key of a given signer. It
    /// includes the signer's identifier. It does NOT include the group
    /// public key.
    #[derive(Clone, Copy, Debug)]
    pub struct SignerPublicKey {
        /// Signer identifier
        pub ident: u16,
        pk: Point,
    }

    /// A VSS element.
    ///
    /// The key split process yields, along with the private key shares,
    /// a sequence of public VSS elements which can be used by individual
    /// signers to verify that their share was properly computed, and
    /// also to derive the group information (all signers' public keys,
    /// and the group public key).
    #[derive(Clone, Copy, Debug)]
    pub struct VSSElement(Point);

    /// A signer's nonce.
    ///
    /// A nonce and a commitment are generated by a signer when starting
    /// the computation of a new signature. The signer must remember them
    /// for the second round of the signature generation protocol. The
    /// nonce is secret; the commitment is public and must be sent to the
    /// coordinator.
    #[derive(Clone, Copy, Debug)]
    pub struct Nonce {
        ident: u16,
        hiding: Scalar,
        binding: Scalar,
    }

    /// A signer's commitment.
    ///
    /// A nonce and a commitment are generated by a signer when starting
    /// the computation of a new signature. The signer must remember them
    /// for the second round of the signature generation protocol. The
    /// nonce is secret; the commitment is public and must be sent to the
    /// coordinator.
    #[derive(Clone, Copy, Debug)]
    pub struct Commitment {
        /// Signer identifier
        pub ident: u16,
        hiding: Point,
        binding: Point,
    }

    /// A signature share.
    ///
    /// A signature share is computed by an individual signer, and sent
    /// to the coordinator for assembly of the complete group signature.
    #[derive(Clone, Copy, Debug)]
    pub struct SignatureShare {
        /// Signer identifier
        pub ident: u16,
        zi: Scalar,
    }

    /// A FROST signature.
    #[derive(Clone, Copy, Debug)]
    pub struct Signature {
        R: Point,
        z: Scalar,
    }

    /// A coordinator's permanent state.
    ///
    /// The coordinator knows the signature threshold and the group
    /// public key.
    #[derive(Clone, Copy, Debug)]
    pub struct Coordinator {
        min_signers: usize,
        group_pk: GroupPublicKey,
    }

    impl GroupPrivateKey {

        /// Encoded private key length (in bytes).
        pub const ENC_LEN: usize = NS;

        /// Generates a new (group) private key.
        ///
        /// A private key is a randomly selected non-zero scalar.
        pub fn generate<T: CryptoRng + RngCore>(rng: &mut T) -> Self {
            let mut sk = random_scalar(rng);
            sk.set_cond(&Scalar::ONE, sk.iszero());
            let pk = Point::mulgen(&sk);
            let pk_enc = point_encode(pk);
            Self { sk, pk, pk_enc }
        }

        /// Gets the public key corresponding to this private key.
        pub fn get_public_key(self) -> GroupPublicKey {
            GroupPublicKey { pk: self.pk, pk_enc: self.pk_enc }
        }

        /// Encodes this private key into bytes.
        ///
        /// In normal FROST usage, group private keys are only transient
        /// in-memory object discarded at the end of the key split process.
        /// Private key encoding is meant to support single-signer FROST
        /// usage.
        pub fn encode(self) -> [u8; Self::ENC_LEN] {
            scalar_encode(self.sk)
        }

        /// Decodes this private key from bytes.
        ///
        /// This function may fail (i.e. return `None`) if the source does
        /// not have the length of an encoded private key, or if the
        /// provided bytes are not a proper canonical encoding for a
        /// non-zero scalar.
        pub fn decode(buf: &[u8]) -> Option<Self> {
            let sk = scalar_decode(buf)?;
            if sk.iszero() != 0 {
                return None;
            }
            let pk = Point::mulgen(&sk);
            let pk_enc = point_encode(pk);
            Some(Self { sk, pk, pk_enc })
        }

        /// Generates a signature (single-signer version).
        ///
        /// This function uses the (group) private key to sign the
        /// provided message. The signature is randomized, though it also
        /// uses a derandomization process internally so that safety is
        /// maintained even if the provided random generator has poor
        /// quality.
        pub fn sign<T: CryptoRng + RngCore>(self, rng: &mut T, msg: &[u8])
            -> Signature
        {
            let mut seed = [0u8; 32];
            rng.fill_bytes(&mut seed);
            self.sign_seeded(&seed, msg)
        }

        /// Generates a signature (single-signer version, seeded).
        ///
        /// This function uses the (group) private key to sign the
        /// provided message. The signature uses an internal derandomization
        /// process to compute the per-signature nonce; an additional seed
        /// can be provided, which is integrated in that process. If that
        /// extra seed is fixed (e.g. it is empty), then the signature
        /// is deterministic (but still safe).
        pub fn sign_seeded(self, seed: &[u8], msg: &[u8]) -> Signature {
            // Per-signature nonce is obtained with hash function H6(),
            // over the public key, private key, seed, and message. The
            // seed length is included before the seed to make the hashing
            // unambiguous; we append that length to the encoded private
            // key since we statically know its length.
            let mut esksl = [0u8; NS + 8];
            esksl[0..NS].copy_from_slice(&scalar_encode(self.sk));
            esksl[NS..NS + 8].copy_from_slice(
                &(seed.len() as u64).to_le_bytes());
            let k = H6(&self.pk_enc, &esksl, seed, msg);
            let R = Point::mulgen(&k);
            let challenge = compute_challenge(R, &self.pk_enc, msg);
            let z = k + challenge * self.sk;
            Signature { R, z }
        }
    }

    impl GroupPublicKey {

        /// Encoded public key length (in bytes).
        pub const ENC_LEN: usize = NE;

        /// Encodes this public key into bytes.
        pub fn encode(self) -> [u8; Self::ENC_LEN] {
            self.pk_enc
        }

        /// Decodes this public key from bytes.
        ///
        /// This function may fail (i.e. return `None`) if the source does
        /// not have the length of an encoded public key, or if the
        /// provided bytes are not a proper canonical encoding for a
        /// non-neutral group element.
        pub fn decode(buf: &[u8]) -> Option<Self> {
            // If the source bytes decode properly then we can use them
            // as the cached encoded public key.
            let pk = point_decode(buf)?;
            let mut pk_enc = [0u8; NE];
            pk_enc[..].copy_from_slice(buf);
            Some(Self { pk, pk_enc })
        }

        /// Verifies a FROST signature.
        ///
        /// The provided signature (`sig`) is verified against this
        /// public key, for the given message (`msg`). FROST is nominally
        /// a distributed signature scheme, but this function also works
        /// with singler-signer signatures.
        pub fn verify(self, sig: Signature, msg: &[u8]) -> bool {
            // Compute the challenge.
            let challenge = compute_challenge(sig.R, &self.pk_enc, msg);

            // Verify the equation.
            self.pk.verify_helper_vartime(&sig.R, &sig.z, &challenge)
        }

        /// Verifies a FROST signature.
        ///
        /// This function decodes the signature from its encoded format
        /// (`esig`), then calls `self.verify()`. If the signature cannot
        /// be decoded, or if the signature is syntactically correct but
        /// the verification algorithm fails, then `false` is returned.
        pub fn verify_esig(self, esig: &[u8], msg: &[u8]) -> bool {
            match Signature::decode(esig) {
                Some(sig) => self.verify(sig, msg),
                None      => false,
            }
        }
    }

    impl KeySplitter {

        /// Split a group private key into shares.
        ///
        /// This function corresponds to the `trusted_dealer_keygen`
        /// function in the FROST specification.
        ///
        /// `group_sk` is the group private key.
        /// `min_signers` is the signing threshold; it must be at least 2.
        /// `max_signers` is the number of shares; it must not be lower than
        /// `min_signers`, and must not exceed 65535.
        /// Returned values are:
        ///   - a vector of `max_signers` signing shares;
        ///   - a vector os `min_signers` VSS elements, that allow
        ///     individual signers to verify that their respective shares
        ///     were correctly generated.
        pub fn trusted_split<T: CryptoRng + RngCore>(rng: &mut T,
            group_sk: GroupPrivateKey, min_signers: usize, max_signers: usize)
            -> (Vec<SignerPrivateKeyShare>, Vec<VSSElement>)
        {
            assert!(min_signers >= 2);
            assert!(min_signers <= max_signers);
            assert!(max_signers <= 65535);

            let group_pk = GroupPublicKey {
                pk: group_sk.pk,
                pk_enc: group_sk.pk_enc,
            };
            let mut coefficients: Vec<Scalar> = Vec::new();
            let mut vsscomm: Vec<VSSElement> = Vec::new();
            coefficients.push(group_sk.sk);
            vsscomm.push(VSSElement(group_pk.pk));
            for _ in 1..min_signers {
                let coef = random_scalar(rng);
                coefficients.push(coef);
                vsscomm.push(VSSElement(Point::mulgen(&coef)));
            }

            let mut shares: Vec<SignerPrivateKeyShare> = Vec::new();
            for i in 0..max_signers {
                let x = Scalar::from_u32((i + 1) as u32);
                let mut y = coefficients[min_signers - 1];
                for j in (0..(min_signers - 1)).rev() {
                    y = (y * x) + coefficients[j];
                }
                let pk = Point::mulgen(&y);
                shares.push(SignerPrivateKeyShare {
                    ident: (i + 1) as u16,
                    sk: y,
                    pk: pk,
                    group_pk: group_pk,
                });
            }

            (shares, vsscomm)
        }

        /// Derives the group information (individual signer public keys, and
        /// group public key) from the key sharing output.
        ///
        /// `max_signers` is the total number of signers (computed key shares).
        /// `vsscomm` is the VSS commitment from the sharing step; it contains
        /// exactly `min_signers` points (where `min_signers` is the signing
        /// threshold).
        ///
        /// This function assumes that the provided parameters are correct,
        /// i.e. that the VSS commitment has been duly verified.
        pub fn derive_group_info(max_signers: usize, vsscomm: Vec<VSSElement>)
            -> (Vec<SignerPublicKey>, GroupPublicKey)
        {
            assert!(vsscomm.len() >= 2);
            assert!(max_signers >= vsscomm.len());
            assert!(max_signers <= 65535);
            let group_pk = GroupPublicKey {
                pk: vsscomm[0].0,
                pk_enc: point_encode(vsscomm[0].0),
            };
            let min_signers = vsscomm.len();
            let mut signer_pk_list: Vec<SignerPublicKey> = Vec::new();
            for i in 1..=max_signers {
                let mut Q = group_pk.pk;
                let k = Scalar::from_u32(i as u32);
                let mut z = k;
                for j in 1..min_signers {
                    Q += vsscomm[j].0 * z;
                    z *= k;
                }
                signer_pk_list.push(SignerPublicKey {
                    ident: i as u16,
                    pk: Q,
                });
            }
            (signer_pk_list, group_pk)
        }
    }

    impl SignerPrivateKeyShare {

        /// Private key share encoded length (in bytes).
        pub const ENC_LEN: usize = 2 + NS + NE;

        /// Encodes this private key share into bytes.
        pub fn encode(self) -> [u8; Self::ENC_LEN] {
            let mut buf = [0u8; Self::ENC_LEN];
            buf[0..2].copy_from_slice(&self.ident.to_be_bytes());
            buf[2..2 + NS].copy_from_slice(&scalar_encode(self.sk));
            buf[2 + NS..2 + NS + NE].copy_from_slice(&self.group_pk.pk_enc);
            buf
        }

        /// Decodes this share from bytes.
        ///
        /// The process fails (i.e. returns `None`) if the source slice
        /// does not have a proper length or does not contain properly
        /// canonical encodings of the share identifier, share of the
        /// private key, or group public key.
        pub fn decode(buf: &[u8]) -> Option<Self> {
            if buf.len() != Self::ENC_LEN {
                return None;
            }
            let ident = u16::from_be_bytes(*<&[u8; 2]>::try_from(
                &buf[0..2]).unwrap());
            if ident == 0 {
                return None;
            }
            let sk = scalar_decode(&buf[2..2 + NS])?;
            if sk.iszero() != 0 {
                // We explicitly reject a zero scalar here because:
                //  - We don't want to get a neutral point as public
                //    key, since that is not supported in most ciphersuites,
                //    and that would make some later encoding/decoding fail.
                //  - Zero private keys are just plain abnormal. They may
                //    theoretically happen with a negligible probability;
                //    in practice, if a zero is obtained, then it is almost
                //    surely because of a bug, an attack, or a hardware
                //    failure.
                return None;
            }
            let group_pk = GroupPublicKey::decode(&buf[2 + NS..2 + NS + NE])?;
            Some(Self {
                ident: ident,
                sk: sk,
                pk: Point::mulgen(&sk),
                group_pk: group_pk,
            })
        }

        /// Get the public key for this signer.
        pub fn get_public_key(self) -> SignerPublicKey {
            SignerPublicKey {
                ident: self.ident,
                pk: self.pk,
            }
        }

        /// Verifies that this share was properly computed, given the VSS
        /// commitments.
        ///
        /// This function is called `verify` in the FROST specification.
        pub fn verify_split(self, vsscomm: &[VSSElement]) -> bool {
            // We don't need to check that the private key is not zero, or
            // that the public key matches it, because this was already
            // verified when decoding.

            let mut Q = vsscomm[0].0;
            let k = Scalar::from_u32(self.ident as u32);
            let mut z = k;
            for j in 1..vsscomm.len() {
                Q += vsscomm[j].0 * z;
                z *= k;
            }
            if self.pk.equals(Q) == 0 {
                return false;
            }

            true
        }

        /// Internal generation of a new nonce.
        ///
        /// As per the specification, the nonce is obtained by hashing the
        /// concatenation of 32 random bytes and the private key.
        fn nonce_generate<T: CryptoRng + RngCore>(self, rng: &mut T) -> Scalar {
            let mut buf = [0u8; 32 + NS];
            rng.fill_bytes(&mut buf[0..32]);
            buf[32..32 + NS].copy_from_slice(&scalar_encode(self.sk));
            H3(&buf)
        }

        /// Generates nonces and commitments for a new signature generation.
        ///
        /// The returned `Nonce` and `Commitment` should be remembered by the
        /// signer for round 2. The `Commitment` should be sent to the
        /// coordinator (`Nonce` is secret and MUST NOT be revealed to
        /// anybody).
        pub fn commit<T: CryptoRng + RngCore>(self, rng: &mut T)
            -> (Nonce, Commitment)
        {
            let ident = self.ident;
            let hiding = self.nonce_generate(rng);
            let binding = self.nonce_generate(rng);
            let nonce = Nonce { ident, hiding, binding };
            (nonce, nonce.get_commitment())
        }

        /// Computes a signature share.
        ///
        /// The nonce and commitment (previously generated with a `commit()`
        /// call, the commitment was sent to the coordinator) is combined with
        /// the message and list of signer commitments selected by the
        /// coordinator.
        ///
        /// This function may fail if the list of commitments is too short
        /// (less than two commitments), or not in the expected order
        /// (by ascending signer identifier), or contains duplicates (two
        /// commitments with the same identifier), or does not contain this
        /// signer's identifier, or contains this signer's identifier but
        /// with a different commitment. In all failure cases, `None` is
        /// returned.
        ///
        /// The signer's own nonce (`nonce`) and commitment (`comm`) MUST
        /// match each other.
        pub fn sign(self, nonce: Nonce, comm: Commitment,
            msg: &[u8], commitment_list: &[Commitment])
            -> Option<SignatureShare>
        {
            // Verify that the commitment list is ordered with no duplicate,
            // that we are part of the list of signers, and that our commitment
            // indeed appears there.
            if commitment_list.len() < 2 {
                return None;
            }
            for i in 0..(commitment_list.len() - 1) {
                if commitment_list[i].ident >= commitment_list[i + 1].ident {
                    return None;
                }
            }
            let mut ff = false;
            for i in 0..commitment_list.len() {
                if commitment_list[i].ident == self.ident {
                    ff = true;
                    if commitment_list[i].hiding.equals(comm.hiding) == 0
                        || commitment_list[i].binding.equals(comm.binding) == 0
                    {
                        return None;
                    }
                }
            }
            if !ff {
                return None;
            }

            // The caller should remember both the nonce and the commitment;
            // thus, a mismatch here is a programming bug, not a case of
            // incoming malicious data.
            assert!(nonce.ident == comm.ident);

            // Compute the binding factors. Since we verified that our
            // commitment is in the provided list,
            // binding_factor_for_participant() cannot fail.
            let binding_factor_list = compute_binding_factors(
                commitment_list, msg);
            let binding_factor = binding_factor_for_participant(
                &binding_factor_list, self.ident).unwrap();

            // Compute the group commitment.
            let group_commitment = compute_group_commitment(
                commitment_list, &binding_factor_list);

            // Compute the Lagrange coefficient.
            let participant_list = participants_from_commitment_list(
                commitment_list);
            let lambda = derive_lagrange_coefficient(
                self.ident, &participant_list);

            // Compute the per-message challenge.
            let challenge = compute_challenge(
                group_commitment, &self.group_pk.pk_enc, msg);

            // Compute the signature share.
            let sig_share = nonce.hiding + nonce.binding * binding_factor
                + lambda * self.sk * challenge;
            Some(SignatureShare {
                ident: self.ident,
                zi: sig_share,
            })
        }
    }

    impl SignerPublicKey {

        /// Signer's public key encoded length (in bytes).
        pub const ENC_LEN: usize = 2 + NE;

        /// Encodes this public key into bytes.
        pub fn encode(self) -> [u8; Self::ENC_LEN] {
            let mut buf = [0u8; Self::ENC_LEN];
            buf[0..2].copy_from_slice(&self.ident.to_be_bytes());
            buf[2..2 + NE].copy_from_slice(&point_encode(self.pk));
            buf
        }

        /// Decodes this public key from bytes.
        ///
        /// The process fails (i.e. returns `None`) if the source slice
        /// does not have a proper length or does not contain properly
        /// canonical encodings of the share identifier and signer's
        /// public key.
        pub fn decode(buf: &[u8]) -> Option<Self> {
            if buf.len() != Self::ENC_LEN {
                return None;
            }
            let ident = u16::from_be_bytes(*<&[u8; 2]>::try_from(
                &buf[0..2]).unwrap());
            if ident == 0 {
                return None;
            }
            let pk = point_decode(&buf[2..2 + NE])?;
            Some(Self { ident, pk })
        }

        /// Verifies a signature share relatively to this signer's public key,
        /// for a given signature generation process.
        ///
        /// This function can be used by the coordinator to check that the
        /// signer computed its signature share properly. It is implictly
        /// called by `Coordinator::assemble_signature()`.
        pub fn verify_signature_share(self, sig_share: SignatureShare,
            commitment_list: &[Commitment], group_pk: GroupPublicKey,
            msg: &[u8]) -> bool
        {
            let binding_factor_list = compute_binding_factors(
                commitment_list, msg);
            let group_commitment = compute_group_commitment(
                commitment_list, &binding_factor_list);
            let challenge = compute_challenge(
                group_commitment, &group_pk.pk_enc, msg);
            self.inner_verify_signature_share(sig_share, commitment_list,
                &binding_factor_list, challenge)
        }

        /// Verifies a signature share relatively to this signer's public key,
        /// for a given signature generation process (inner function).
        fn inner_verify_signature_share(self, sig_share: SignatureShare,
            commitment_list: &[Commitment],
            binding_factor_list: &[BindingFactor], challenge: Scalar) -> bool
        {
            // Verify that the share is really ours.
            if sig_share.ident != self.ident {
                return false;
            }

            // Find our commitment in the list.
            let mut comm = Commitment::INVALID;
            for c in commitment_list.iter() {
                if c.ident == self.ident {
                    comm = *c;
                    break;
                }
            }
            if comm.is_invalid() {
                return false;
            }

            // Get the correct binding factor.
            let binding_factor = binding_factor_for_participant(
                binding_factor_list, self.ident).unwrap();

            // Compute the commitment share.
            let comm_share = comm.hiding + binding_factor * comm.binding;

            // Compute the Lagrange coefficient.
            let participant_list = participants_from_commitment_list(
                commitment_list);
            let lambda = derive_lagrange_coefficient(
                self.ident, &participant_list);

            // Compute relation values.
            // We want to verify that P1 = P2, with:
            //  P1 = sig_share*G
            //  P2 = comm_share + (challenge * lambda)*Q
            // (with Q = public key)
            // Everything here is public so we can use verify_helper_vartime().
            self.pk.verify_helper_vartime(
                &comm_share, &sig_share.zi, &(challenge * lambda))
        }
    }

    impl VSSElement {

        /// Encodes a VSS commitment (list of VSS elements) into bytes.
        pub fn encode_list(vss: &[VSSElement]) -> Vec<u8> {
            let mut r: Vec<u8> = Vec::with_capacity(32 * vss.len());
            for v in vss.iter() {
                r.extend_from_slice(&point_encode(v.0));
            }
            r
        }

        /// Decodes a VSS commitment (list of VSS elements) from bytes.
        ///
        /// This function returns `None` if the source slice does not split
        /// evenly into at least two encoded points (with no trailing garbage),
        /// or if any of the encodings is not a valid point encoding, or if any
        /// of the points is the neutral.
        pub fn decode_list(buf: &[u8]) -> Option<Vec<VSSElement>> {
            if buf.len() % NE != 0 {
                return None;
            }
            let n = buf.len() / NE;
            if n < 2 {
                return None;
            }
            let mut r: Vec<VSSElement> = Vec::with_capacity(n);
            for i in 0..n {
                r.push(VSSElement(point_decode(&buf[NE * i .. NE * (i + 1)])?));
            }
            Some(r)
        }
    }

    impl Nonce {

        /// Encoded nonce length (in bytes).
        pub const ENC_LEN: usize = 2 + 2 * NS;

        /// Encodes this nonce into bytes.
        ///
        /// In normal FROST usage, nonces are transient, remembered only
        /// by the individual signer who generated them, and not transmitted.
        /// Encoding nonces into bytes is possible to allow long-latency
        /// scenarios in which the signer cannot reliably maintain the nonce
        /// in RAM only between the two rounds.
        pub fn encode(self) -> [u8; Self::ENC_LEN] {
            let mut buf = [0u8; Self::ENC_LEN];
            buf[0..2].copy_from_slice(&self.ident.to_be_bytes());
            buf[2..2 + NS].copy_from_slice(&scalar_encode(self.hiding));
            buf[2 + NS..2 + 2 * NS].copy_from_slice(
                &scalar_encode(self.binding));
            buf
        }

        /// Decodes this nonce from bytes.
        pub fn decode(buf: &[u8]) -> Option<Self> {
            if buf.len() != Self::ENC_LEN {
                return None;
            }
            let ident = u16::from_be_bytes(*<&[u8; 2]>::try_from(
                &buf[0..2]).unwrap());
            if ident == 0 {
                return None;
            }
            let hiding = scalar_decode(&buf[2..2 + NS])?;
            let binding = scalar_decode(&buf[2 + NS..2 + 2 * NS])?;
            Some(Self { ident, hiding, binding })
        }

        /// (Re)computes the commitment corresponding to this nonce.
        pub fn get_commitment(self) -> Commitment {
            Commitment {
                ident: self.ident,
                hiding: Point::mulgen(&self.hiding),
                binding: Point::mulgen(&self.binding),
            }
        }
    }

    impl Commitment {

        /// Invalid commitment value, used as a placeholder.
        const INVALID: Commitment = Self {
            ident: 0,
            hiding: Point::NEUTRAL,
            binding: Point::NEUTRAL,
        };

        /// Encoded length (in bytes).
        pub const ENC_LEN: usize = 2 + 2 * NE;

        fn is_invalid(self) -> bool {
            self.ident == 0
        }

        /// Encodes this commitment into bytes.
        pub fn encode(self) -> [u8; Self::ENC_LEN] {
            let mut buf = [0u8; Self::ENC_LEN];
            buf[0..2].copy_from_slice(&self.ident.to_be_bytes());
            buf[2..2 + NE].copy_from_slice(&point_encode(self.hiding));
            buf[2 + NE..2 + 2 * NE].copy_from_slice(
                &point_encode(self.binding));
            buf
        }

        /// Decodes this commitment from bytes.
        ///
        /// The process fails (i.e. returns `None`) if the source slice
        /// does not have a proper length or does not contain properly
        /// canonical encodings of the signer identifier and commitment.
        pub fn decode(buf: &[u8]) -> Option<Self> {
            if buf.len() != Self::ENC_LEN {
                return None;
            }
            let ident = u16::from_be_bytes(*<&[u8; 2]>::try_from(
                &buf[0..2]).unwrap());
            if ident == 0 {
                return None;
            }
            let hiding = point_decode(&buf[2..2 + NE])?;
            let binding = point_decode(&buf[2 + NE..2 + 2 * NE])?;
            Some(Self { ident, hiding, binding })
        }

        /// Encodes a commitment list into bytes.
        pub fn encode_list(commitment_list: &[Commitment]) -> Vec<u8> {
            let mut r: Vec<u8> = Vec::with_capacity(
                Commitment::ENC_LEN * commitment_list.len());
            for c in commitment_list.iter() {
                r.extend_from_slice(&c.encode());
            }
            r
        }

        /// Decodes a commitment list from bytes.
        ///
        /// This function verifies that there are at least two commitments, that
        /// the source slice does not have any trailing unused bytes, that all
        /// commitments are syntactically correct (in particular that the
        /// identifiers are in the 1 to 65535 range and that the points are not
        /// the neutral), and that the list is properly ordered in ascending
        /// order of identifiers with no duplicate. If any of these verification
        /// fails, then this function returns `None`.
        pub fn decode_list(buf: &[u8]) -> Option<Vec<Commitment>> {
            if buf.len() % Commitment::ENC_LEN != 0 {
                return None;
            }
            let n = buf.len() / Commitment::ENC_LEN;
            if n < 2 {
                return None;
            }
            let mut cc: Vec<Commitment> = Vec::with_capacity(n);
            for i in 0..n {
                let c = Commitment::decode(&buf[i * Commitment::ENC_LEN
                    .. (i + 1) * Commitment::ENC_LEN])?;
                if i > 0 && c.ident <= cc[i - 1].ident {
                    return None;
                }
                cc.push(c);
            }
            Some(cc)
        }
    }

    impl SignatureShare {

        /// Encoded length (in bytes) of a signature share.
        pub const ENC_LEN: usize = 2 + NS;

        /// Encode a signature share into bytes.
        pub fn encode(self) -> [u8; Self::ENC_LEN] {
            let mut buf = [0u8; Self::ENC_LEN];
            buf[0..2].copy_from_slice(&self.ident.to_be_bytes());
            buf[2..2 + NS].copy_from_slice(&scalar_encode(self.zi));
            buf
        }

        /// Decode a signature share from bytes.
        pub fn decode(buf: &[u8]) -> Option<Self> {
            if buf.len() != Self::ENC_LEN {
                return None;
            }
            let ident = u16::from_be_bytes(*<&[u8; 2]>::try_from(
                &buf[0..2]).unwrap());
            if ident == 0 {
                return None;
            }
            let zi = scalar_decode(&buf[2..2 + NS])?;
            Some(Self { ident, zi })
        }
    }

    impl Signature {

        /// Encoded length (in bytes) of a signature.
        pub const ENC_LEN: usize = NE + NS;

        /// Encode a signature into bytes.
        pub fn encode(self) -> [u8; Self::ENC_LEN] {
            let mut buf = [0u8; Self::ENC_LEN];
            buf[0..NE].copy_from_slice(&point_encode(self.R));
            buf[NE..NE + NS].copy_from_slice(&scalar_encode(self.z));
            buf
        }

        /// Decode a signature from bytes.
        ///
        /// `None` is returned if the source bytes do not have the proper
        /// length for a signature, or if the signature is syntactically
        /// incorrect.
        pub fn decode(buf: &[u8]) -> Option<Self> {
            if buf.len() != Self::ENC_LEN {
                return None;
            }
            let R = point_decode(&buf[0..NE])?;
            let z = scalar_decode(&buf[NE..NE + NS])?;
            Some(Signature { R, z })
        }
    }

    impl Coordinator {

        /// Create an instance over the provided group public key and
        /// signature threshold.
        ///
        /// If the threshold is invalid (less than 2 or greater than 65535),
        /// then this function returns `None`.
        pub fn new(min_signers: usize, group_pk: GroupPublicKey)
            -> Option<Self>
        {
            if min_signers < 2 || min_signers > 65535 {
                return None;
            }
            Some(Self { min_signers, group_pk })
        }

        /// Choose the signers from a list of commitments.
        ///
        /// If there are enough commitments (given the group threshold),
        /// then this function chooses `min_signers` of them and returns
        /// the corresponding ordered list of commitments. The list must
        /// be sent to all
        pub fn choose(self, comms: &[Commitment]) -> Option<Vec<Commitment>> {
            let mut r: Vec<Commitment> = Vec::with_capacity(self.min_signers);
            for i in 0..comms.len() {
                let c = comms[i];
                let mut ff = false;
                for j in 0..r.len() {
                    if r[j].ident >= c.ident {
                        if r[j].ident > c.ident {
                            r.insert(j, c);
                        }
                        ff = true;
                        break;
                    }
                }
                if !ff {
                    r.push(c);
                }
                if r.len() >= self.min_signers {
                    // We got enough distinct commitments, and they are
                    // already sorted.
                    return Some(r);
                }
            }
            None
        }

        /// Verifies signature shares received from the signers, and assembles
        /// the signature value.
        ///
        /// The obtained signature shares are provided as `sig_shares`;
        /// they can be provided in any order, and duplicates are tolerated,
        /// as long as one share can be found for each commitment (extra
        /// share values and duplicates are ignored). This function also
        /// needs the signers' public keys, which are provided in the
        /// `signer_public_keys` slice (the signers' public keys need not be
        /// in any particular order, and the `signer_public_keys` slice may
        /// contain extra public keys for signers who were not involved in
        /// the list of commitments). The assembled signature is
        /// automatically verified against the group public key, and
        /// returned. If the process fails for any reason, then this function
        /// returns `None`.
        pub fn assemble_signature(self,
            sig_shares: &[SignatureShare], commitment_list: &[Commitment],
            signer_public_keys: &[SignerPublicKey], msg: &[u8])
            -> Option<Signature>
        {
            // Verify all shares.
            let binding_factor_list = compute_binding_factors(
                commitment_list, msg);
            let group_commitment = compute_group_commitment(
                commitment_list, &binding_factor_list);
            let challenge = compute_challenge(
                group_commitment, &self.group_pk.pk_enc, msg);
            let mut verified_shares: Vec::<SignatureShare> =
                Vec::with_capacity(commitment_list.len());
            for c in commitment_list.iter() {
                // Find the signature share and the signer public key for
                // this commitment (by identifier).
                let id = c.ident;
                let ss = sig_shares.into_iter().find(
                    |&x| x.ident == id)?;
                let spk = signer_public_keys.into_iter().find(
                    |&x| x.ident == id)?;

                // Verify the share.
                if !spk.inner_verify_signature_share(
                    *ss, commitment_list, &binding_factor_list, challenge)
                {
                    return None;
                }

                verified_shares.push(*ss);
            }

            // Assemble the signature value.
            let (R, z) = aggregate(group_commitment, &verified_shares);

            // Verify the signature. We already computed the challenge,
            // so we only have to check the verification equation.
            if !self.group_pk.pk.verify_helper_vartime(&R, &z, &challenge) {
                return None;
            }

            // All good, return the signature.
            Some(Signature { R, z })
        }
    }

    // ---------------- internal helper functions ------------------

    /// A binding factor.
    #[derive(Clone, Copy, Debug)]
    struct BindingFactor {
        ident: u16,
        factor: Scalar,
    }

    /// Generates a random scalar.
    fn random_scalar<T: CryptoRng + RngCore>(rng: &mut T) -> Scalar {
        let mut buf = [0u8; 2 * NS];
        rng.fill_bytes(&mut buf);
        Scalar::decode_reduce(&buf)
    }

    /// Computes the bindings factors for a list of commitments and a
    /// nessage.
    fn compute_binding_factors(commitment_list: &[Commitment], msg: &[u8])
        -> Vec<BindingFactor>
    {
        let msg_hash = H4(msg);
        let encoded_commitment_list = Commitment::encode_list(commitment_list);
        let encoded_commitment_hash = H5(&encoded_commitment_list);
        let mut rho_input_prefix: Vec<u8> = Vec::with_capacity(
            msg_hash.len() + encoded_commitment_hash.len());
        rho_input_prefix.extend_from_slice(&msg_hash);
        rho_input_prefix.extend_from_slice(&encoded_commitment_hash);

        let mut binding_factor_list: Vec<BindingFactor> = Vec::new();
        for c in commitment_list.iter() {
            let mut rho_input: Vec<u8> = Vec::with_capacity(
                rho_input_prefix.len() + 2);
            rho_input.extend_from_slice(&rho_input_prefix[..]);
            rho_input.extend_from_slice(&c.ident.to_be_bytes());
            binding_factor_list.push(BindingFactor {
                ident: c.ident,
                factor: H1(&rho_input[..]),
            });
        }
        binding_factor_list
    }

    /// Finds the binding factor specific to a given participant in a list
    /// of binding factors.
    fn binding_factor_for_participant(bfl: &[BindingFactor], ident: u16)
        -> Option<Scalar>
    {
        for bf in bfl.iter() {
            if bf.ident == ident {
                return Some(bf.factor);
            }
        }
        None
    }

    /// Computes the group commitment.
    ///
    /// This function assumes that the binding factors match the commitments,
    /// i.e. that they designate the same signers in the same order.
    fn compute_group_commitment(commitment_list: &[Commitment],
        binding_factor_list: &[BindingFactor]) -> Point
    {
        let mut Q = Point::NEUTRAL;
        for (c, bf) in commitment_list.iter().zip(binding_factor_list) {
            assert!(c.ident == bf.ident);
            Q += c.hiding + bf.factor * c.binding;
        }
        Q
    }

    /// Derive the list of participants (identifers) from
    /// a list of commitments.
    fn participants_from_commitment_list(commitment_list: &[Commitment])
        -> Vec<u16>
    {
        let mut ids: Vec<u16> = Vec::with_capacity(commitment_list.len());
        for c in commitment_list.iter() {
            ids.push(c.ident)
        }
        ids
    }

    /// Derive the Lagrange interpolation coefficient for a given scalar x,
    /// and a set of x-coordinates.
    ///
    /// The provided `x` MUST be part of the list `L`. All elements of `L`
    /// (including `x`) MUST be integers in the 1 to 65535 range. Elements of
    /// `L` MUST be sorted in ascending order.
    fn derive_lagrange_coefficient(x: u16, L: &[u16]) -> Scalar {
        // Check that the parameters are correct.
        let mut ff = false;
        for i in 0..L.len() {
            if x == L[i] {
                ff = true;
            }
            assert!(L[i] >= 1);
            if i > 0 {
                assert!(L[i] > L[i - 1]);
            }
        }
        assert!(ff);

        // Compute the coefficient.
        let mut numerator = Scalar::ONE;
        let mut denominator = Scalar::ONE;
        let xi = Scalar::from_u32(x as u32);
        for y in L.iter() {
            if x != *y {
                let xj = Scalar::from_u32(*y as u32);
                numerator *= xj;
                denominator *= xj - xi;
            }
        }
        numerator / denominator
    }

    /// Computes the challenge.
    fn compute_challenge(group_commitment: Point,
        encoded_group_public_key: &[u8], msg: &[u8]) -> Scalar
    {
        H2(&point_encode(group_commitment), encoded_group_public_key, msg)
    }

    /// Aggregates the signature shares into a signature.
    fn aggregate(group_commitment: Point, sig_shares: &[SignatureShare])
        -> (Point, Scalar)
    {
        let mut z = Scalar::ZERO;
        for ss in sig_shares.iter() {
            z += ss.zi;
        }
        (group_commitment, z)
    }

} } // End of macro: define_frost_core

// ========================================================================

#[cfg(test)]
macro_rules! define_frost_tests { () => {

    use super::{GroupPrivateKey, GroupPublicKey, KeySplitter, VSSElement};
    use super::{SignerPrivateKeyShare, SignerPublicKey};
    use super::{Nonce, Commitment, SignatureShare, Signature, Coordinator};
    use super::{Point, compute_binding_factors, point_decode, scalar_decode};
    use crate::{CryptoRng, RngCore, RngError};
    use sha2::{Sha512, Digest};
    use crate::Vec;

    // A pretend RNG for test purposes (deterministic from a given seed).
    struct DRNG {
        buf: [u8; 64],
        ptr: usize,
    }

    impl DRNG {

        fn from_seed(seed: &[u8]) -> Self {
            let mut d = Self {
                buf: [0u8; 64],
                ptr: 0,
            };
            let mut sh = Sha512::new();
            sh.update(seed);
            d.buf[..].copy_from_slice(&sh.finalize());
            d
        }
    }

    impl RngCore for DRNG {

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

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            let len = dest.len();
            let mut off = 0;
            while off < len {
                let mut clen = 32 - self.ptr;
                if clen > (len - off) {
                    clen = len - off;
                }
                dest[off .. off + clen].copy_from_slice(
                    &self.buf[self.ptr .. self.ptr + clen]);
                self.ptr += clen;
                off += clen;
                if self.ptr == 32 {
                    let mut sh = Sha512::new();
                    sh.update(&self.buf);
                    self.buf[..].copy_from_slice(&sh.finalize());
                    self.ptr = 0;
                }
            }
        }

        fn try_fill_bytes(&mut self, dest: &mut [u8])
            -> Result<(), RngError>
        {
            self.fill_bytes(dest);
            Ok(())
        }
    }

    impl CryptoRng for DRNG { }

    fn test_self_ops(min_signers: usize, max_signers: usize) {
        // Initialize a reproducible RNG.
        let mut rng = DRNG::from_seed(
            &((min_signers + (max_signers << 16)) as u32).to_le_bytes());

        // Keygen.
        let group_sk = GroupPrivateKey::generate(&mut rng);
        let group_pk = group_sk.get_public_key();
        let gsk2 = GroupPrivateKey::decode(&group_sk.encode()).unwrap();
        assert!(gsk2.sk.equals(group_sk.sk) != 0);
        assert!(gsk2.pk.equals(group_sk.pk) != 0);
        assert!(gsk2.pk_enc == group_sk.pk_enc);
        let gpk2 = GroupPublicKey::decode(&group_pk.encode()).unwrap();
        assert!(gpk2.pk.equals(group_sk.pk) != 0);
        assert!(gpk2.pk_enc == group_sk.pk_enc);

        // Verify that single-signer usage works.
        for i in 0..10 {
            let msg1 = [i as u8];
            let msg2 = [(i + 1) as u8];
            let esig = group_sk.sign(&mut rng, &msg1).encode();
            assert!(group_pk.verify_esig(&esig, &msg1));
            assert!(!group_pk.verify_esig(&esig, &msg2));
        }

        // Key split.
        let (sk_shares, vss) = KeySplitter::trusted_split(
            &mut rng, group_sk, min_signers, max_signers);
        assert!(sk_shares.len() == max_signers);
        assert!(vss.len() == min_signers);
        for i in 0..max_signers {
            let ssk = sk_shares[i];
            assert!(ssk.ident == ((i + 1) as u16));
            assert!(ssk.group_pk.pk.equals(group_pk.pk) != 0);
            assert!(ssk.group_pk.pk_enc == group_pk.pk_enc);
            assert!(ssk.pk.equals(Point::mulgen(&ssk.sk)) != 0);
        }

        // Verify shares, including encoding/decoding. We also extract
        // the signer public keys.
        let vss2 = VSSElement::decode_list(
            &VSSElement::encode_list(&vss)).unwrap();
        assert!(vss2.len() == vss.len());
        for i in 0..vss.len() {
            assert!(vss[i].0.equals(vss2[i].0) != 0);
        }
        let mut signer_public_keys: Vec<SignerPublicKey> = Vec::new();
        for ssk in sk_shares.iter() {
            let ssk2 = SignerPrivateKeyShare::decode(&ssk.encode()).unwrap();
            assert!(ssk2.ident == ssk.ident);
            assert!(ssk2.sk.equals(ssk.sk) != 0);
            assert!(ssk2.pk.equals(ssk.pk) != 0);
            assert!(ssk2.group_pk.pk.equals(group_pk.pk) != 0);
            assert!(ssk2.group_pk.pk_enc == group_pk.pk_enc);
            assert!(ssk2.verify_split(&vss2));
            signer_public_keys.push(ssk2.get_public_key());
        }

        assert!(signer_public_keys.len() == sk_shares.len());
        for i in 0..sk_shares.len() {
            assert!(signer_public_keys[i].ident == sk_shares[i].ident);
            assert!(signer_public_keys[i].pk.equals(sk_shares[i].pk) != 0);
        }

        // Make each signer generate a new nonce, with the corresponding
        // commitment. Both a remembered on the signer's side for the next
        // round; commitments are encoded and sent to the coordinator. We
        // also randomize the order of encoded commitments (possibly with
        // duplicates).
        struct SignerState {
            nonce: Nonce,
            comm: Commitment,
        }
        let mut signer_states: Vec<SignerState> = Vec::new();
        let mut ecomms: Vec<[u8; Commitment::ENC_LEN]> = Vec::new();
        for _ in 0..max_signers {
            ecomms.push([0u8; Commitment::ENC_LEN]);
        }
        for ssk in sk_shares.iter() {
            let (nonce, comm) = ssk.commit(&mut rng);
            signer_states.push(SignerState { nonce, comm });
            ecomms.push(comm.encode());
        }
        for i in 0..max_signers {
            let k = (rng.next_u64() as usize) % max_signers;
            ecomms[i] = ecomms[max_signers + k];
        }

        // Submit the encoded commitments to the coordinator.
        let mut round1_comms: Vec<Commitment> = Vec::new();
        for e in ecomms {
            round1_comms.push(Commitment::decode(&e).unwrap());
        }
        let coor = Coordinator::new(min_signers, group_pk).unwrap();
        let comms1 = coor.choose(&round1_comms).unwrap();
        assert!(comms1.len() == min_signers);
        for i in 1..comms1.len() {
            assert!(comms1[i].ident > comms1[i - 1].ident);
        }
        let comms2 = Commitment::decode_list(
            &Commitment::encode_list(&comms1)).unwrap();
        assert!(comms1.len() == comms2.len());
        for i in 0..comms2.len() {
            assert!(comms1[i].ident == comms2[i].ident);
            assert!(comms1[i].hiding.equals(comms2[i].hiding) != 0);
            assert!(comms1[i].binding.equals(comms2[i].binding) != 0);
            let j = (comms2[i].ident as usize) - 1;
            assert!(comms2[i].hiding.equals(signer_states[j].comm.hiding) != 0);
            assert!(comms2[i].binding.equals(signer_states[j].comm.binding) != 0);
        }

        // Have the chosen signers perform the second round.
        let msg: &[u8] = b"sample";
        let mut sig_shares: Vec<SignatureShare> = Vec::new();
        for c in comms2.iter() {
            let i = (c.ident as usize) - 1;
            let ss = sk_shares[i].sign(signer_states[i].nonce,
                signer_states[i].comm, msg, &comms2).unwrap();
            sig_shares.push(ss);
        }

        // Coordinator receives all signature shares, verifies them,
        // and assembles the signature.
        let sig = coor.assemble_signature(
            &sig_shares, &comms1, &signer_public_keys, msg).unwrap();

        // Verify that the signature is correct (this was already
        // verified by the coordinator, we just want to confirm here).
        // We also check that the signature is not valid for a different
        // message.
        assert!(group_pk.verify(sig, msg));
        assert!(!group_pk.verify(sig, b"not the same message"));
    }

    #[test]
    fn self_ops() {
        for max_signers in 2..6 {
            for min_signers in 2..=max_signers {
                test_self_ops(min_signers, max_signers);
            }
        }
    }

    #[test]
    fn KAT() {
        let group_sk = GroupPrivateKey::decode(&hex::decode(KAT_GROUP_SK).unwrap()).unwrap();
        let group_pk = GroupPublicKey::decode(&hex::decode(KAT_GROUP_PK).unwrap()).unwrap();
        assert!(group_pk.pk.equals(Point::mulgen(&group_sk.sk)) != 0);
        let msg = hex::decode(KAT_MSG).unwrap();

        let sk1 = scalar_decode(&hex::decode(KAT_SK1).unwrap()).unwrap();
        let sk2 = scalar_decode(&hex::decode(KAT_SK2).unwrap()).unwrap();
        let sk3 = scalar_decode(&hex::decode(KAT_SK3).unwrap()).unwrap();
        let S1 = SignerPrivateKeyShare {
            ident: 1,
            sk: sk1,
            pk: Point::mulgen(&sk1),
            group_pk: group_pk,
        };
        let S2 = SignerPrivateKeyShare {
            ident: 2,
            sk: sk2,
            pk: Point::mulgen(&sk2),
            group_pk: group_pk,
        };
        let S3 = SignerPrivateKeyShare {
            ident: 3,
            sk: sk3,
            pk: Point::mulgen(&sk3),
            group_pk: group_pk,
        };

        let S1_hn = scalar_decode(&hex::decode(KAT_S1_HN).unwrap()).unwrap();
        let S1_bn = scalar_decode(&hex::decode(KAT_S1_BN).unwrap()).unwrap();
        let S1_hc = point_decode(&hex::decode(KAT_S1_HC).unwrap()).unwrap();
        let S1_bc = point_decode(&hex::decode(KAT_S1_BC).unwrap()).unwrap();
        let S1_bf = scalar_decode(&hex::decode(KAT_S1_BF).unwrap()).unwrap();
        assert!(S1_hc.equals(Point::mulgen(&S1_hn)) != 0);
        assert!(S1_bc.equals(Point::mulgen(&S1_bn)) != 0);
        let S1_nonce = Nonce { ident: 1, hiding: S1_hn, binding: S1_bn };
        let S1_comm = S1_nonce.get_commitment();
        assert!(S1_comm.ident == 1);
        assert!(S1_comm.hiding.equals(S1_hc) != 0);
        assert!(S1_comm.binding.equals(S1_bc) != 0);

        let S3_hn = scalar_decode(&hex::decode(KAT_S3_HN).unwrap()).unwrap();
        let S3_bn = scalar_decode(&hex::decode(KAT_S3_BN).unwrap()).unwrap();
        let S3_hc = point_decode(&hex::decode(KAT_S3_HC).unwrap()).unwrap();
        let S3_bc = point_decode(&hex::decode(KAT_S3_BC).unwrap()).unwrap();
        let S3_bf = scalar_decode(&hex::decode(KAT_S3_BF).unwrap()).unwrap();
        assert!(S3_hc.equals(Point::mulgen(&S3_hn)) != 0);
        assert!(S3_bc.equals(Point::mulgen(&S3_bn)) != 0);
        let S3_nonce = Nonce { ident: 3, hiding: S3_hn, binding: S3_bn };
        let S3_comm = S3_nonce.get_commitment();
        assert!(S3_comm.ident == 3);
        assert!(S3_comm.hiding.equals(S3_hc) != 0);
        assert!(S3_comm.binding.equals(S3_bc) != 0);

        let coor = Coordinator::new(2, group_pk).unwrap();
        let comms = coor.choose(&[S3_comm, S3_comm, S1_comm]).unwrap();
        assert!(comms.len() == 2);
        assert!(comms[0].ident == 1);
        assert!(comms[1].ident == 3);

        let bfs = compute_binding_factors(&comms, &msg);
        assert!(bfs.len() == 2);
        assert!(bfs[0].ident == 1);
        assert!(bfs[0].factor.equals(S1_bf) != 0);
        assert!(bfs[1].ident == 3);
        assert!(bfs[1].factor.equals(S3_bf) != 0);

        let S1_sig_share = S1.sign(S1_nonce, S1_comm, &msg, &comms).unwrap();
        let S3_sig_share = S3.sign(S3_nonce, S3_comm, &msg, &comms).unwrap();
        let S1_ss_ref = scalar_decode(&hex::decode(KAT_S1_SIG_SHARE).unwrap()).unwrap();
        let S3_ss_ref = scalar_decode(&hex::decode(KAT_S3_SIG_SHARE).unwrap()).unwrap();
        assert!(S1_sig_share.ident == 1);
        assert!(S1_sig_share.zi.equals(S1_ss_ref) != 0);
        assert!(S3_sig_share.ident == 3);
        assert!(S3_sig_share.zi.equals(S3_ss_ref) != 0);

        let sig = coor.assemble_signature(
            &[S3_sig_share, S1_sig_share, S3_sig_share], &comms,
            &[S1.get_public_key(), S2.get_public_key(), S3.get_public_key()],
            &msg).unwrap();
        let esig = sig.encode();
        let mut sig_ref = [0u8; Signature::ENC_LEN];
        hex::decode_to_slice(KAT_SIG, &mut sig_ref[..]).unwrap();
        assert!(esig == sig_ref);
        assert!(group_pk.verify_esig(&esig, &msg));
    }

} } // End of macro: define_frost_tests

// ========================================================================

/// FROST(Ed25519, SHA-512)
pub mod ed25519 {

    pub use crate::ed25519::{Point, Scalar};
    use sha2::{Sha512, Digest};

    define_frost_core!{}

    /// Decodes a point from bytes.
    fn point_decode(buf: &[u8]) -> Option<Point> {
        // As per draft-irtf-cfrg-frost-08, points must be verified to be
        // in the proper prime-order subgroup, and not the neutral element.
        let P = Point::decode(buf)?;
        if P.isneutral() != 0 || P.is_in_subgroup() == 0 {
            None
        } else {
            Some(P)
        }
    }

    /// Encodes a point into bytes.
    fn point_encode(P: Point) -> [u8; 32] {
        P.encode()
    }

    /// Decodes a scalar from bytes.
    fn scalar_decode(buf: &[u8]) -> Option<Scalar> {
        Scalar::decode(buf)
    }

    /// Encodes a scalar into bytes.
    fn scalar_encode(x: Scalar) -> [u8; 32] {
        x.encode()
    }

    const NE: usize = 32;
    const NS: usize = 32;

    const CONTEXT_STRING: &[u8] = b"FROST-ED25519-SHA512-v8";

    fn H1(msg: &[u8]) -> Scalar {
        let mut sh = Sha512::new();
        sh.update(CONTEXT_STRING);
        sh.update(b"rho");
        sh.update(msg);
        Scalar::decode_reduce(&sh.finalize())
    }

    fn H2(gc_enc: &[u8], pk_enc: &[u8], msg: &[u8]) -> Scalar {
        let mut sh = Sha512::new();
        // No context string or label, for compatibility with RFC 8032.
        sh.update(gc_enc);
        sh.update(pk_enc);
        sh.update(msg);
        Scalar::decode_reduce(&sh.finalize())
    }

    fn H3(msg: &[u8]) -> Scalar {
        let mut sh = Sha512::new();
        sh.update(CONTEXT_STRING);
        sh.update(b"nonce");
        sh.update(msg);
        Scalar::decode_reduce(&sh.finalize())
    }

    fn H4(msg: &[u8]) -> [u8; 64] {
        let mut sh = Sha512::new();
        sh.update(CONTEXT_STRING);
        sh.update(b"msg");
        sh.update(msg);
        let mut r = [0u8; 64];
        r[..].copy_from_slice(&sh.finalize());
        r
    }

    fn H5(msg: &[u8]) -> [u8; 64] {
        let mut sh = Sha512::new();
        sh.update(CONTEXT_STRING);
        sh.update(b"com");
        sh.update(msg);
        let mut r = [0u8; 64];
        r[..].copy_from_slice(&sh.finalize());
        r
    }

    fn H6(pk_enc: &[u8], sk_enc: &[u8], seed: &[u8], msg: &[u8]) -> Scalar {
        let mut sh = Sha512::new();
        sh.update(CONTEXT_STRING);
        sh.update(b"single-signer");
        sh.update(pk_enc);
        sh.update(sk_enc);
        sh.update(seed);
        sh.update(msg);
        Scalar::decode_reduce(&sh.finalize())
    }

    #[cfg(test)]
    mod tests {

        static KAT_GROUP_SK: &str = "7b1c33d3f5291d85de664833beb1ad469f7fb6025a0ec78b3a790c6e13a98304";
        static KAT_GROUP_PK: &str = "15d21ccd7ee42959562fc8aa63224c8851fb3ec85a3faf66040d380fb9738673";
        static KAT_MSG: &str = "74657374";

        static KAT_SK1: &str = "929dcc590407aae7d388761cddb0c0db6f5627aea8e217f4a033f2ec83d93509";
        static KAT_SK2: &str = "a91e66e012e4364ac9aaa405fcafd370402d9859f7b6685c07eed76bf409e80d";
        static KAT_SK3: &str = "d3cb090a075eb154e82fdb4b3cb507f110040905468bb9c46da8bdea643a9a02";

        static KAT_S1_HN: &str = "1c406170127e33142b8611bc02bf14d5909e49d5cb87150eff3ec9804212920c";
        static KAT_S1_BN: &str = "5be4edde8b7acd79528721191626810c94fbc2bcc814b7a67d301fbd7fc16e07";
        static KAT_S1_HC: &str = "eab073cf90278e1796c2db197566c8d1f62f9992d399a5329239481f9cbb5811";
        static KAT_S1_BC: &str = "13172c94dec7b22eb0a910e93fa1af8a79e27f61b69981e1ebb227438ca3be84";
        static KAT_S1_BF: &str = "c538ab7707e484ba5d29bb80d9ac795e0542e8089debbaca4df090e92a6d5504";

        static KAT_S3_HN: &str = "795f87122f05b7efc4b1a52f435c3d28597411b1a6fec198ce9c818c5451c401";
        static KAT_S3_BN: &str = "c9193aaef37bc074ea3286d0361c815f7201bf764cd9e7d8bb4eb5ecca840a09";
        static KAT_S3_HC: &str = "049e0a8d62db8fd2f8401fb027e0a51374f5c4c796a1765ecf14467df8c4829a";
        static KAT_S3_BC: &str = "eeb691d3dc19e0dbc33471c7a7681a51801c481da34a8f55efe3070a75e9991d";
        static KAT_S3_BF: &str = "9e4474b925576c54c8e50ec27e09a04537c837f38b0f71312a58a8c12861b408";

        static KAT_S1_SIG_SHARE: &str = "1f16a3989b4aa2cc3782a503331b9a21d7ba56c9c5455d06981b5425306c9d01";
        static KAT_S3_SIG_SHARE: &str = "4c8f33c301c05871b434a686847d5818417a01e50a59e9e7fddaefde7d244207";

        static KAT_SIG: &str = "1aff2259ecb59cfcbb36ae77e02a9b134422abeae47cf7ff56c85fdf90932b186ba5d65b9d0afb3decb64b8ab798f239183558aed09e46ee95f64304ae90df08";

        define_frost_tests!{}

        #[test]
        fn interop_ed25519() {
            // FROST signatures are supposed to be verifiable with a
            // plain RFC 8032 Ed25519 verifier.
            use crate::ed25519::PublicKey;

            let mut rng = DRNG::from_seed(b"interop_ed25519");
            let msg = b"sample";
            let group_sk = GroupPrivateKey::generate(&mut rng);
            let esig = group_sk.sign(&mut rng, msg).encode();

            let group_pk = group_sk.get_public_key();
            let ed_pk = PublicKey::decode(&group_pk.encode()).unwrap();
            assert!(ed_pk.verify_raw(&esig, msg));
        }
    }
}

/// FROST(ristretto255, SHA-512)
pub mod ristretto255 {
    pub use crate::ristretto255::{Point, Scalar};
    use sha2::{Sha512, Digest};

    define_frost_core!{}

    /// Decodes a point from bytes.
    fn point_decode(buf: &[u8]) -> Option<Point> {
        // As per draft-irtf-cfrg-frost-08, point decoding is allowed to
        // return the neutral element.
        Point::decode(buf)
    }

    /// Encodes a point into bytes.
    fn point_encode(P: Point) -> [u8; 32] {
        P.encode()
    }

    /// Decodes a scalar from bytes.
    fn scalar_decode(buf: &[u8]) -> Option<Scalar> {
        Scalar::decode(buf)
    }

    /// Encodes a scalar into bytes.
    fn scalar_encode(x: Scalar) -> [u8; 32] {
        x.encode()
    }

    const NE: usize = 32;
    const NS: usize = 32;

    const CONTEXT_STRING: &[u8] = b"FROST-RISTRETTO255-SHA512-v8";

    fn H1(msg: &[u8]) -> Scalar {
        let mut sh = Sha512::new();
        sh.update(CONTEXT_STRING);
        sh.update(b"rho");
        sh.update(msg);
        Scalar::decode_reduce(&sh.finalize())
    }

    fn H2(gc_enc: &[u8], pk_enc: &[u8], msg: &[u8]) -> Scalar {
        let mut sh = Sha512::new();
        sh.update(CONTEXT_STRING);
        sh.update(b"chal");
        sh.update(gc_enc);
        sh.update(pk_enc);
        sh.update(msg);
        Scalar::decode_reduce(&sh.finalize())
    }

    fn H3(msg: &[u8]) -> Scalar {
        let mut sh = Sha512::new();
        sh.update(CONTEXT_STRING);
        sh.update(b"nonce");
        sh.update(msg);
        Scalar::decode_reduce(&sh.finalize())
    }

    fn H4(msg: &[u8]) -> [u8; 64] {
        let mut sh = Sha512::new();
        sh.update(CONTEXT_STRING);
        sh.update(b"msg");
        sh.update(msg);
        let mut r = [0u8; 64];
        r[..].copy_from_slice(&sh.finalize());
        r
    }

    fn H5(msg: &[u8]) -> [u8; 64] {
        let mut sh = Sha512::new();
        sh.update(CONTEXT_STRING);
        sh.update(b"com");
        sh.update(msg);
        let mut r = [0u8; 64];
        r[..].copy_from_slice(&sh.finalize());
        r
    }

    fn H6(pk_enc: &[u8], sk_enc: &[u8], seed: &[u8], msg: &[u8]) -> Scalar {
        let mut sh = Sha512::new();
        sh.update(CONTEXT_STRING);
        sh.update(b"single-signer");
        sh.update(pk_enc);
        sh.update(sk_enc);
        sh.update(seed);
        sh.update(msg);
        Scalar::decode_reduce(&sh.finalize())
    }

    #[cfg(test)]
    mod tests {

        static KAT_GROUP_SK: &str = "1b25a55e463cfd15cf14a5d3acc3d15053f08da49c8afcf3ab265f2ebc4f970b";
        static KAT_GROUP_PK: &str = "e2a62f39eede11269e3bd5a7d97554f5ca384f9f6d3dd9c3c0d05083c7254f57";
        static KAT_MSG: &str = "74657374";

        static KAT_SK1: &str = "5c3430d391552f6e60ecdc093ff9f6f4488756aa6cebdbad75a768010b8f830e";
        static KAT_SK2: &str = "b06fc5eac20b4f6e1b271d9df2343d843e1e1fb03c4cbb673f2872d459ce6f01";
        static KAT_SK3: &str = "f17e505f0e2581c6acfe54d3846a622834b5e7b50cad9a2109a97ba7a80d5c04";

        static KAT_S1_HN: &str = "1eaee906e0554a5e533415e971eefa909f3c614c7c75e27f381b0270a9afe308";
        static KAT_S1_BN: &str = "16175fc2e7545baf7180e8f5b6e1e73c4f2769323cc76754bdd79fe93ab0bd0b";
        static KAT_S1_HC: &str = "80d35700fda011d9e2b2fad4f237bf88f2978d954382dfd36a517ab0497a474f";
        static KAT_S1_BC: &str = "40f0fecaf94e656b3f802ba9827fca9fa994c13c98a5ff257973f8bdbc733324";
        static KAT_S1_BF: &str = "c0f5ee2613c448137bae256a4e95d56deb8c59f934332c0c0041720b8819680f";

        static KAT_S3_HN: &str = "48d78b8c2de1a515513f9d3fc464a19a72304fac522f17cc647706cb22c21403";
        static KAT_S3_BN: &str = "5c0f10966b3f1386660a87de0fafd69decbe9ffae1a152a88b7d83bb4fb1c908";
        static KAT_S3_HC: &str = "20dec6ad0795f82009a1a94b6ad79f01a1e95ae8e308d8d8fae8285982308113";
        static KAT_S3_BC: &str = "98437dafb20fdb18255464072bee514889aeeec324f149d49747143c3613056d";
        static KAT_S3_BF: &str = "8ea449e545706bb3b42c66423005451457e4bb4dea2c2d0b1d157e6bb652ec09";

        static KAT_S1_SIG_SHARE: &str = "5ae13621ebeef844e39454eb3478a50c4531d25939e1065f44f5b04a8535090e";
        static KAT_S3_SIG_SHARE: &str = "aa432dcf274a9441c205e76fe43497be99efe374f9853477bd5add2075f6970c";

        static KAT_SIG: &str = "9c407badb8cacf10f306d94e31fb2a71d6a8398039802b4d80a127847239720617516e93f8d57a2ecffd43b83ab35db6de20b6ce32673bd601508e6bfa2ba10a";

        define_frost_tests!{}
    }
}

/// FROST(P-256, SHA-256)
pub mod p256 {
    pub use crate::p256::{Point, Scalar};
    use sha2::{Sha256, Digest};

    define_frost_core!{}

    /// Decodes a point from bytes.
    fn point_decode(buf: &[u8]) -> Option<Point> {
        // As per draft-irtf-cfrg-frost-08, points use the compressed
        // encoding, and we do not accept the point-at-infinity. It suffices
        // to verify that the encoded length is 33 bytes, since only
        // non-infinity compressed encodings have that length.
        if buf.len() != 33 {
            return None;
        }
        Point::decode(buf)
    }

    /// Encodes a point into bytes.
    fn point_encode(P: Point) -> [u8; 33] {
        P.encode_compressed()
    }

    /// Decodes a scalar from bytes.
    fn scalar_decode(buf: &[u8]) -> Option<Scalar> {
        // SEC1 rules mandate big-endian.
        if buf.len() != 32 {
            return None;
        }
        let mut ex = [0u8; 32];
        for i in 0..32 {
            ex[i] = buf[31 - i];
        }
        Scalar::decode(&ex)
    }

    /// Encodes a scalar into bytes.
    fn scalar_encode(x: Scalar) -> [u8; 32] {
        // SEC1 rules mandate big-endian.
        let mut buf = [0u8; 32];
        let ex = x.encode();
        for i in 0..32 {
            buf[i] = ex[31 - i];
        }
        buf
    }

    const NE: usize = 33;
    const NS: usize = 32;

    const CONTEXT_STRING: &[u8] = b"FROST-P256-SHA256-v8";

    fn expand_message_xmd(label: &[u8],
        msg1: &[u8], msg2: &[u8], msg3: &[u8], msg4: &[u8]) -> Scalar
    {
        let mut sh = Sha256::new();

        // b_0 = H(msg_prime)
        sh.update(&[0u8; 64]);            // Z_pad
        sh.update(msg1);                  // msg (four chunks)
        sh.update(msg2);
        sh.update(msg3);
        sh.update(msg4);
        sh.update(&48u16.to_be_bytes());  // l_i_b_str
        sh.update(&[0u8]);                // I2OSP(0, 1)
        sh.update(CONTEXT_STRING);        // DST_prime
        sh.update(label);
        sh.update(&[(CONTEXT_STRING.len() + label.len()) as u8]);
        let b0 = sh.finalize_reset();

        // b_1 = H(b_0 || I2OSP(1, 1) || DST_prime)
        sh.update(&b0);
        sh.update(&[1u8]);
        sh.update(CONTEXT_STRING);
        sh.update(label);
        sh.update(&[(CONTEXT_STRING.len() + label.len()) as u8]);
        let b1 = sh.finalize_reset();

        // b_2 = H(b_0 XOR b_1 || I2OSP(2, 1) || DST_prime)
        let mut xb01 = [0u8; 32];
        for i in 0..32 {
            xb01[i] = b0[i] ^ b1[i];
        }
        sh.update(&xb01);
        sh.update(&[2u8]);
        sh.update(CONTEXT_STRING);
        sh.update(label);
        sh.update(&[(CONTEXT_STRING.len() + label.len()) as u8]);
        let b2 = sh.finalize_reset();

        // Truncate b_1 || b_2 to the first 48 bytes, then interpret as
        // an integer with the big-endian convention, and reduce.
        let mut x = [0u8; 48];
        for i in 0..32 {
            x[47 - i] = b1[i];
        }
        for i in 0..16 {
            x[15 - i] = b2[i];
        }
        Scalar::decode_reduce(&x)
    }

    const U8_EMPTY: [u8; 0] = [];

    fn H1(msg: &[u8]) -> Scalar {
        expand_message_xmd(b"rho", msg, &U8_EMPTY, &U8_EMPTY, &U8_EMPTY)
    }

    fn H2(gc_enc: &[u8], pk_enc: &[u8], msg: &[u8]) -> Scalar {
        expand_message_xmd(b"chal", gc_enc, pk_enc, msg, &U8_EMPTY)
    }

    fn H3(msg: &[u8]) -> Scalar {
        expand_message_xmd(b"nonce", msg, &U8_EMPTY, &U8_EMPTY, &U8_EMPTY)
    }

    fn H4(msg: &[u8]) -> [u8; 32] {
        let mut sh = Sha256::new();
        sh.update(CONTEXT_STRING);
        sh.update(b"msg");
        sh.update(msg);
        let mut r = [0u8; 32];
        r[..].copy_from_slice(&sh.finalize());
        r
    }

    fn H5(msg: &[u8]) -> [u8; 32] {
        let mut sh = Sha256::new();
        sh.update(CONTEXT_STRING);
        sh.update(b"com");
        sh.update(msg);
        let mut r = [0u8; 32];
        r[..].copy_from_slice(&sh.finalize());
        r
    }

    fn H6(pk_enc: &[u8], sk_enc: &[u8], seed: &[u8], msg: &[u8]) -> Scalar {
        expand_message_xmd(b"single-signer", pk_enc, sk_enc, seed, msg)
    }

    #[cfg(test)]
    mod tests {

        static KAT_GROUP_SK: &str = "8ba9bba2e0fd8c4767154d35a0b7562244a4aaf6f36c8fb8735fa48b301bd8de";
        static KAT_GROUP_PK: &str = "023a309ad94e9fe8a7ba45dfc58f38bf091959d3c99cfbd02b4dc00585ec45ab70";
        static KAT_MSG: &str = "74657374";

        static KAT_SK1: &str = "0c9c1a0fe806c184add50bbdcac913dda73e482daf95dcb9f35dbb0d8a9f7731";
        static KAT_SK2: &str = "8d8e787bef0ff6c2f494ca45f4dad198c6bee01212d6c84067159c52e1863ad5";
        static KAT_SK3: &str = "0e80d6e8f6192c003b5488ce1eec8f5429587d48cf001541e713b2d53c09d928";

        static KAT_S1_HN: &str = "e9165dad654fc20a9e31ca6f32ac032ec327b551a50e8ac5cf25f5c4c9e20757";
        static KAT_S1_BN: &str = "e9059a232598a0fba0e495a687580e624ab425337c3221246fb2c716905bc9e7";
        static KAT_S1_HC: &str = "0228df2e7f6c254b40a9f8853cf6c4f21eacbb6f0663027384966816b57e513304";
        static KAT_S1_BC: &str = "02f5b7f48786f8b83ebefed6249825650c4fa657da66ae0da1b2613dedbe122ec8";
        static KAT_S1_BF: &str = "95f987c0ab590507a8c4deaf506ffc182d3626e30386306f7ab3aaf0b0013cd3";

        static KAT_S3_HN: &str = "b9d136e29eb758bd77cb83c317ac4e336cf8cda830c089deddf6d5ec81da9884";
        static KAT_S3_BN: &str = "5261e2d00ce227e67bb9b38990294e2c82970f335b2e6d9f1d07a72ba43d01f0";
        static KAT_S3_HC: &str = "02f87bd95ab5e08ea292a96e21caf9bdc5002ebf6e3ce14f922817d26a4d08144d";
        static KAT_S3_BC: &str = "0263cb513e347fcf8492c7f97843ed4c3797f2f3fe925b1e68f65fb90826fe9597";
        static KAT_S3_BF: &str = "2f21db4f811b13f938a13b8f2633467d250703fe5bd63cd24f08bef6fd2f3c29";

        static KAT_S1_SIG_SHARE: &str = "bdaa275f10ca57e3a3a9a7a0d95aeabb517897d8482873a8f9713d458f94756f";
        static KAT_S3_SIG_SHARE: &str = "0e8fd85386939e8974a8748e66641df0fe043323c52487a2b10b8a397897de21";

        static KAT_SIG: &str = "03c41521412528dce484c35b6b9b7cc8150102ab3e4bdf858d702270c05098e6c6cc39ffb2975df66d18521c2f3fbf08ac4f7ccafc0d4cfb4baa7cc77f082c5390";

        define_frost_tests!{}
    }
}

/// FROST(secp256k1, SHA-256)
pub mod secp256k1 {
    pub use crate::secp256k1::{Point, Scalar};
    use sha2::{Sha256, Digest};

    define_frost_core!{}

    /// Decodes a point from bytes.
    fn point_decode(buf: &[u8]) -> Option<Point> {
        // As per draft-irtf-cfrg-frost-08, points use the compressed
        // encoding, and we do not accept the point-at-infinity. It suffices
        // to verify that the encoded length is 33 bytes, since only
        // non-infinity compressed encodings have that length.
        if buf.len() != 33 {
            return None;
        }
        Point::decode(buf)
    }

    /// Encodes a point into bytes.
    fn point_encode(P: Point) -> [u8; 33] {
        P.encode_compressed()
    }

    /// Decodes a scalar from bytes.
    fn scalar_decode(buf: &[u8]) -> Option<Scalar> {
        // SEC1 rules mandate big-endian.
        if buf.len() != 32 {
            return None;
        }
        let mut ex = [0u8; 32];
        for i in 0..32 {
            ex[i] = buf[31 - i];
        }
        Scalar::decode(&ex)
    }

    /// Encodes a scalar into bytes.
    fn scalar_encode(x: Scalar) -> [u8; 32] {
        // SEC1 rules mandate big-endian.
        let mut buf = [0u8; 32];
        let ex = x.encode();
        for i in 0..32 {
            buf[i] = ex[31 - i];
        }
        buf
    }

    const NE: usize = 33;
    const NS: usize = 32;

    const CONTEXT_STRING: &[u8] = b"FROST-secp256k1-SHA256-v8";

    fn expand_message_xmd(label: &[u8],
        msg1: &[u8], msg2: &[u8], msg3: &[u8], msg4: &[u8]) -> Scalar
    {
        let mut sh = Sha256::new();

        // b_0 = H(msg_prime)
        sh.update(&[0u8; 64]);            // Z_pad
        sh.update(msg1);                  // msg (four chunks)
        sh.update(msg2);
        sh.update(msg3);
        sh.update(msg4);
        sh.update(&48u16.to_be_bytes());  // l_i_b_str
        sh.update(&[0u8]);                // I2OSP(0, 1)
        sh.update(CONTEXT_STRING);        // DST_prime
        sh.update(label);
        sh.update(&[(CONTEXT_STRING.len() + label.len()) as u8]);
        let b0 = sh.finalize_reset();

        // b_1 = H(b_0 || I2OSP(1, 1) || DST_prime)
        sh.update(&b0);
        sh.update(&[1u8]);
        sh.update(CONTEXT_STRING);
        sh.update(label);
        sh.update(&[(CONTEXT_STRING.len() + label.len()) as u8]);
        let b1 = sh.finalize_reset();

        // b_2 = H(b_0 XOR b_1 || I2OSP(2, 1) || DST_prime)
        let mut xb01 = [0u8; 32];
        for i in 0..32 {
            xb01[i] = b0[i] ^ b1[i];
        }
        sh.update(&xb01);
        sh.update(&[2u8]);
        sh.update(CONTEXT_STRING);
        sh.update(label);
        sh.update(&[(CONTEXT_STRING.len() + label.len()) as u8]);
        let b2 = sh.finalize_reset();

        // Truncate b_1 || b_2 to the first 48 bytes, then interpret as
        // an integer with the big-endian convention, and reduce.
        let mut x = [0u8; 48];
        for i in 0..32 {
            x[47 - i] = b1[i];
        }
        for i in 0..16 {
            x[15 - i] = b2[i];
        }
        Scalar::decode_reduce(&x)
    }

    const U8_EMPTY: [u8; 0] = [];

    fn H1(msg: &[u8]) -> Scalar {
        expand_message_xmd(b"rho", msg, &U8_EMPTY, &U8_EMPTY, &U8_EMPTY)
    }

    fn H2(gc_enc: &[u8], pk_enc: &[u8], msg: &[u8]) -> Scalar {
        expand_message_xmd(b"chal", gc_enc, pk_enc, msg, &U8_EMPTY)
    }

    fn H3(msg: &[u8]) -> Scalar {
        expand_message_xmd(b"nonce", msg, &U8_EMPTY, &U8_EMPTY, &U8_EMPTY)
    }

    fn H4(msg: &[u8]) -> [u8; 32] {
        let mut sh = Sha256::new();
        sh.update(CONTEXT_STRING);
        sh.update(b"msg");
        sh.update(msg);
        let mut r = [0u8; 32];
        r[..].copy_from_slice(&sh.finalize());
        r
    }

    fn H5(msg: &[u8]) -> [u8; 32] {
        let mut sh = Sha256::new();
        sh.update(CONTEXT_STRING);
        sh.update(b"com");
        sh.update(msg);
        let mut r = [0u8; 32];
        r[..].copy_from_slice(&sh.finalize());
        r
    }

    fn H6(pk_enc: &[u8], sk_enc: &[u8], seed: &[u8], msg: &[u8]) -> Scalar {
        expand_message_xmd(b"single-signer", pk_enc, sk_enc, seed, msg)
    }

    #[cfg(test)]
    mod tests {

        static KAT_GROUP_SK: &str = "0d004150d27c3bf2a42f312683d35fac7394b1e9e318249c1bfe7f0795a83114";
        static KAT_GROUP_PK: &str = "02f37c34b66ced1fb51c34a90bdae006901f10625cc06c4f64663b0eae87d87b4f";
        static KAT_MSG: &str = "74657374";

        static KAT_SK1: &str = "08f89ffe80ac94dcb920c26f3f46140bfc7f95b493f8310f5fc1ea2b01f4254c";
        static KAT_SK2: &str = "04f0feac2edcedc6ce1253b7fab8c86b856a797f44d83d82a385554e6e401984";
        static KAT_SK3: &str = "00e95d59dd0d46b0e303e500b62b7ccb0e555d49f5b849f5e748c071da8c0dbc";

        static KAT_S1_HN: &str = "95f352cf568508bce96ef3cb816bf9229eb521ca9c2aff6a4fe8b86bf49ae16f";
        static KAT_S1_BN: &str = "c675aea50ff2510ae6b0fcb55432b97ad0b55a28b959bacb0e8b466dbf43dd26";
        static KAT_S1_HC: &str = "028acf8c9e345673e2544248006f4ba7ead5e94e170062b86eb532a74c26f79f98";
        static KAT_S1_BC: &str = "0314c33f75948224dd39cdffc68fa0faeeb42f7ef94f1552c920196d53fbda04ce";
        static KAT_S1_BF: &str = "6c7933abb7bc86bcc5c549ba984b9526dca099f9d9b787cedde20c70d36f5fc1";

        static KAT_S3_HN: &str = "b5089ebf363630d3477711005173c1419f4f40514f7287b4ca6ff110967a2d70";
        static KAT_S3_BN: &str = "5e50ce9975cfc6164e85752f52094b11091fdbca846a9c245fdbfa4bab1ae28c";
        static KAT_S3_HC: &str = "039121f05be205b6a52ffdfdcd5f9cdc3b074a7f0f031dac294e747b7ca83567d5";
        static KAT_S3_BC: &str = "0265c40f57bdcdcd0dfa43a8d353301e1474517b70da29ddb1cb4461cd09eee1ce";
        static KAT_S3_BF: &str = "1b18e710a470fe513e4387c613321aa41151990f65a8577343b45d6883ab877d";

        static KAT_S1_SIG_SHARE: &str = "280c44c6c37cd64c7f5a552ae8416a57d21c115cab524dbff5fbcebbf5c0019d";
        static KAT_S3_SIG_SHARE: &str = "e372bca35133a80ca140dcac2125c966b763a934678f40e09fb8b0ae9d4aee1b";

        static KAT_SIG: &str = "0364b02292a4b0e61f849f4d6fac0e67c2f698a21e1cba9e4a5b8fa535f2f9310d0b7f016a14b07e59209b31d7096733bfced0ddaa6398ee64d5e220ddc2d4ae77";

        define_frost_tests!{}
    }
}
