//! FROST implementation.
//!
//! This follows the v14 draft specification: [draft-irtf-cfrg-frost-14]
//!
//! FROST is a threshold Schnorr signature scheme: the group private key
//! is split into individual signer shares. If enough signers (with a
//! configurable threshold) collaborate, then they can conjointly
//! generate a signature on a given message. The individual signers do
//! not have to trust each other (the protocol is resilient to actively
//! malicious signers, who may at worst prevent the generation of a valid
//! signature). Output signatures are "plain" Schnorr signatures,
//! verifiable against the group public key. When the ciphersuite is
//! FROST(Ed25519, SHA-512), the generated signatures can also be
//! successfully verified with a plain Ed25519 verifier (as per RFC
//! 8032); the same applies to FROST(Ed448, SHAKE256) relatively to Ed448
//! verifiers (also as defined in RFC 8032).
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
//!  - `ed448`: FROST(Ed448, SHAKE256)
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
//!    splits the group private key into individual key shares
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
//! [draft-irtf-cfrg-frost-14]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-frost-14
//! [frost-sample.rs]: https://github.com/pornin/crrl/extra/frost-sample.rs

#![allow(non_snake_case)]
#![allow(unused_macros)]

/// Most functions are generic, provided that the relevant Point and
/// Scalar types are in scope, and a few constants defined. This macro
/// generates the generic functions. The caller is supposed to invoke it
/// in an appropriate module with Point and Scalar already defined.
macro_rules! define_frost_core { () => {

    use crate::{CryptoRng, RngCore};
    use crate::Vec;
    use core::cmp::Ordering;

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
        pub ident: Scalar,
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
        pub ident: Scalar,
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
        ident: Scalar,
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
        pub ident: Scalar,
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
        pub ident: Scalar,
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
        /// FROST specification, in appendix D, mandates that the
        /// key generation process with a trusted dealers does not generate
        /// more than 65535 shares. This is an arbitrary limit that may
        /// change in the future (it was historically related to the
        /// encoding of share identifiers over two bytes in older draft
        /// specifications, but the encoding has changed and no longer has
        /// this limitation).
        pub const MAX_MAX_SIGNERS: usize = 65535;

        /// Split a group private key into shares.
        ///
        /// This function corresponds to the `trusted_dealer_keygen`
        /// function in the FROST specification.
        ///
        /// `group_sk` is the group private key.
        /// `min_signers` is the signing threshold; it must be at least 2.
        /// `max_signers` is the number of shares; it must not be lower than
        /// `min_signers`, and must not exceed `MAX_MAX_SIGNERS`.
        ///
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
            assert!(max_signers <= Self::MAX_MAX_SIGNERS);

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
                let x = Scalar::from_u64((i as u64) + 1);
                let mut y = coefficients[min_signers - 1];
                for j in (0..(min_signers - 1)).rev() {
                    y = (y * x) + coefficients[j];
                }
                let pk = Point::mulgen(&y);
                shares.push(SignerPrivateKeyShare {
                    ident: x,
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
            assert!(max_signers <= Self::MAX_MAX_SIGNERS);
            let group_pk = GroupPublicKey {
                pk: vsscomm[0].0,
                pk_enc: point_encode(vsscomm[0].0),
            };
            let min_signers = vsscomm.len();
            let mut signer_pk_list: Vec<SignerPublicKey> = Vec::new();
            for i in 1..=max_signers {
                let mut Q = group_pk.pk;
                let k = Scalar::from_u64(i as u64);
                let mut z = k;
                for j in 1..min_signers {
                    Q += vsscomm[j].0 * z;
                    z *= k;
                }
                signer_pk_list.push(SignerPublicKey {
                    ident: Scalar::from_u64(i as u64),
                    pk: Q,
                });
            }
            (signer_pk_list, group_pk)
        }
    }

    impl SignerPrivateKeyShare {

        /// Private key share encoded length (in bytes).
        pub const ENC_LEN: usize = NS + NS + NE;

        /// Encodes this private key share into bytes.
        pub fn encode(self) -> [u8; Self::ENC_LEN] {
            let mut buf = [0u8; Self::ENC_LEN];
            buf[0..NS].copy_from_slice(&scalar_encode(self.ident));
            buf[NS..NS + NS].copy_from_slice(&scalar_encode(self.sk));
            buf[NS + NS..NS + NS + NE].copy_from_slice(&self.group_pk.pk_enc);
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
            let ident = scalar_decode(&buf[0..NS])?;
            if ident.iszero() != 0 {
                // Share identifiers are not allowed to be zero.
                return None;
            }
            let sk = scalar_decode(&buf[NS..NS + NS])?;
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
            let group_pk = GroupPublicKey::decode(&buf[NS + NS..NS + NS + NE])?;
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
        /// This function is called `vss_verify` in the FROST specification.
        pub fn verify_split(self, vsscomm: &[VSSElement]) -> bool {
            // We don't need to check that the private key is not zero, or
            // that the public key matches it, because this was already
            // verified when decoding.

            let mut Q = vsscomm[0].0;
            let k = self.ident;
            let mut z = k;
            for j in 1..vsscomm.len() {
                Q += vsscomm[j].0 * z;
                z *= k;
            }
            self.pk.equals(Q) != 0
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
                if scalar_cmp_vartime(commitment_list[i].ident,
                    commitment_list[i + 1].ident) != Ordering::Less
                {
                    return None;
                }
            }
            let mut ff = false;
            for i in 0..commitment_list.len() {
                if commitment_list[i].ident.equals(self.ident) != 0 {
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
            assert!(nonce.ident.equals(comm.ident) != 0);

            // Compute the binding factors. Since we verified that our
            // commitment is in the provided list,
            // binding_factor_for_participant() cannot fail.
            let binding_factor_list = compute_binding_factors(
                self.group_pk, commitment_list, msg);
            let binding_factor = binding_factor_for_participant(
                &binding_factor_list, self.ident).unwrap();

            // Compute the group commitment.
            let group_commitment = compute_group_commitment(
                commitment_list, &binding_factor_list);

            // Compute the Lagrange coefficient.
            let participant_list = participants_from_commitment_list(
                commitment_list);
            let lambda = derive_interpolating_value(
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
        pub const ENC_LEN: usize = NS + NE;

        /// Encodes this public key into bytes.
        pub fn encode(self) -> [u8; Self::ENC_LEN] {
            let mut buf = [0u8; Self::ENC_LEN];
            buf[0..NS].copy_from_slice(&scalar_encode(self.ident));
            buf[NS..NS + NE].copy_from_slice(&point_encode(self.pk));
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
            let ident = scalar_decode(&buf[0..NS])?;
            if ident.iszero() != 0 {
                return None;
            }
            let pk = point_decode(&buf[NS..NS + NE])?;
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
                group_pk, commitment_list, msg);
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
            if sig_share.ident.equals(self.ident) == 0 {
                return false;
            }

            // Find our commitment in the list.
            let mut comm = Commitment::INVALID;
            for c in commitment_list.iter() {
                if c.ident.equals(self.ident) != 0 {
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
            let lambda = derive_interpolating_value(
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
            let mut r: Vec<u8> = Vec::with_capacity(NE * vss.len());
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
        pub const ENC_LEN: usize = 3 * NS;

        /// Encodes this nonce into bytes.
        ///
        /// In normal FROST usage, nonces are transient, remembered only
        /// by the individual signer who generated them, and not transmitted.
        /// Encoding nonces into bytes is possible to allow long-latency
        /// scenarios in which the signer cannot reliably maintain the nonce
        /// in RAM only between the two rounds.
        pub fn encode(self) -> [u8; Self::ENC_LEN] {
            let mut buf = [0u8; Self::ENC_LEN];
            buf[0..NS].copy_from_slice(&scalar_encode(self.ident));
            buf[NS..2 * NS].copy_from_slice(&scalar_encode(self.hiding));
            buf[2 * NS..3 * NS].copy_from_slice(&scalar_encode(self.binding));
            buf
        }

        /// Decodes this nonce from bytes.
        pub fn decode(buf: &[u8]) -> Option<Self> {
            if buf.len() != Self::ENC_LEN {
                return None;
            }
            let ident = scalar_decode(&buf[0..NS])?;
            if ident.iszero() != 0 {
                return None;
            }
            let hiding = scalar_decode(&buf[NS..2 * NS])?;
            let binding = scalar_decode(&buf[2 * NS..3 * NS])?;
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
            ident: Scalar::ZERO,
            hiding: Point::NEUTRAL,
            binding: Point::NEUTRAL,
        };

        /// Encoded length (in bytes).
        pub const ENC_LEN: usize = NS + 2 * NE;

        fn is_invalid(self) -> bool {
            self.ident.iszero() != 0
        }

        /// Encodes this commitment into bytes.
        pub fn encode(self) -> [u8; Self::ENC_LEN] {
            let mut buf = [0u8; Self::ENC_LEN];
            buf[0..NS].copy_from_slice(&scalar_encode(self.ident));
            buf[NS..NS + NE].copy_from_slice(&point_encode(self.hiding));
            buf[NS + NE..NS + 2 * NE].copy_from_slice(
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
            let ident = scalar_decode(&buf[0..NS])?;
            if ident.iszero() != 0 {
                return None;
            }
            let hiding = point_decode(&buf[NS..NS + NE])?;
            let binding = point_decode(&buf[NS + NE..NS + 2 * NE])?;
            Some(Self { ident, hiding, binding })
        }

        /// Encodes a commitment list into bytes.
        pub fn encode_list(commitment_list: &[Commitment]) -> Vec<u8> {
            // This is encode_group_commitment_list() from the FROST spec.
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
        /// identifiers are canonically-encoded non-zero scalars and the
        /// commitment points are canonically encoded), and that the list
        /// is properly ordered in ascending order of identifiers with no
        /// duplicate. If any of these verification fails, then this function
        /// returns `None`.
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
                if i > 0 {
                    if scalar_cmp_vartime(cc[i - 1].ident, c.ident)
                        != Ordering::Less
                    {
                        return None;
                    }
                }
                cc.push(c);
            }
            Some(cc)
        }
    }

    impl SignatureShare {

        /// Encoded length (in bytes) of a signature share.
        pub const ENC_LEN: usize = NS + NS;

        /// Encode a signature share into bytes.
        pub fn encode(self) -> [u8; Self::ENC_LEN] {
            let mut buf = [0u8; Self::ENC_LEN];
            buf[0..NS].copy_from_slice(&scalar_encode(self.ident));
            buf[NS..NS + NS].copy_from_slice(&scalar_encode(self.zi));
            buf
        }

        /// Decode a signature share from bytes.
        pub fn decode(buf: &[u8]) -> Option<Self> {
            if buf.len() != Self::ENC_LEN {
                return None;
            }
            let ident = scalar_decode(&buf[0..NS])?;
            if ident.iszero() != 0 {
                return None;
            }
            let zi = scalar_decode(&buf[NS..NS + NS])?;
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
        /// If the threshold is invalid (less than 2), then this function
        /// returns `None`.
        pub fn new(min_signers: usize, group_pk: GroupPublicKey)
            -> Option<Self>
        {
            if min_signers < 2 {
                return None;
            }
            Some(Self { min_signers, group_pk })
        }

        /// Choose the signers from a list of commitments.
        ///
        /// If there are enough commitments (given the group threshold),
        /// then this function chooses `min_signers` of them and returns
        /// the corresponding ordered list of commitments. The list must
        /// be sent to all chosen participants.
        pub fn choose(self, comms: &[Commitment]) -> Option<Vec<Commitment>> {
            // TODO: maybe do a better sort? This is an insertion sort, with
            // a cost quadratic min_signers. Normally, min_signers is rather
            // small, so this does not matter much.
            let mut r: Vec<Commitment> = Vec::with_capacity(self.min_signers);
            for i in 0..comms.len() {
                let c = comms[i];
                let mut ff = false;
                for j in 0..r.len() {
                    let cv = scalar_cmp_vartime(r[j].ident, c.ident);
                    if cv != Ordering::Less {
                        if cv == Ordering::Greater {
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
                self.group_pk, commitment_list, msg);
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
                    |&x| x.ident.equals(id) != 0)?;
                let spk = signer_public_keys.into_iter().find(
                    |&x| x.ident.equals(id) != 0)?;

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
        ident: Scalar,
        factor: Scalar,
    }

    /// Generates a random scalar.
    fn random_scalar<T: CryptoRng + RngCore>(rng: &mut T) -> Scalar {
        let mut buf = [0u8; NS + ((NS + 1) >> 1)];
        rng.fill_bytes(&mut buf);
        Scalar::decode_reduce(&buf)
    }

    /// Computes the bindings factors for a list of commitments and a
    /// nessage.
    fn compute_binding_factors(group_pk: GroupPublicKey,
        commitment_list: &[Commitment], msg: &[u8]) -> Vec<BindingFactor>
    {
        let gpk_enc = group_pk.pk_enc;
        let msg_hash = H4(msg);
        let encoded_commitment_list = Commitment::encode_list(commitment_list);
        let encoded_commitment_hash = H5(&encoded_commitment_list);
        let mut rho_input_prefix: Vec<u8> = Vec::with_capacity(
            gpk_enc.len() + msg_hash.len() + encoded_commitment_hash.len());
        rho_input_prefix.extend_from_slice(&gpk_enc);
        rho_input_prefix.extend_from_slice(&msg_hash);
        rho_input_prefix.extend_from_slice(&encoded_commitment_hash);

        let mut binding_factor_list: Vec<BindingFactor> = Vec::new();
        for c in commitment_list.iter() {
            let mut rho_input: Vec<u8> = Vec::with_capacity(
                rho_input_prefix.len() + NS);
            rho_input.extend_from_slice(&rho_input_prefix[..]);
            rho_input.extend_from_slice(&scalar_encode(c.ident));
            binding_factor_list.push(BindingFactor {
                ident: c.ident,
                factor: H1(&rho_input[..]),
            });
        }
        binding_factor_list
    }

    /// Finds the binding factor specific to a given participant in a list
    /// of binding factors.
    fn binding_factor_for_participant(bfl: &[BindingFactor], ident: Scalar)
        -> Option<Scalar>
    {
        for bf in bfl.iter() {
            if bf.ident.equals(ident) != 0 {
                return Some(bf.factor);
            }
        }
        None
    }

    /// Computes the group commitment.
    ///
    /// This function assumes that the binding factors match the commitments,
    /// i.e. that they designate the same signers in the same order
    /// (the FROST spec does not make that ordering assumption, but the
    /// caller can easily enforce it).
    fn compute_group_commitment(commitment_list: &[Commitment],
        binding_factor_list: &[BindingFactor]) -> Point
    {
        let mut Q = Point::NEUTRAL;
        for (c, bf) in commitment_list.iter().zip(binding_factor_list) {
            assert!(c.ident.equals(bf.ident) != 0);
            Q += c.hiding + bf.factor * c.binding;
        }
        Q
    }

    /// Derive the list of participants (identifers) from
    /// a list of commitments.
    fn participants_from_commitment_list(commitment_list: &[Commitment])
        -> Vec<Scalar>
    {
        let mut ids: Vec<Scalar> = Vec::with_capacity(commitment_list.len());
        for c in commitment_list.iter() {
            ids.push(c.ident)
        }
        ids
    }

    /// Derive the Lagrange interpolation coefficient for a given scalar x,
    /// and a set of x-coordinates.
    ///
    /// The provided `x` MUST be part of the list `L`. All elements of `L`
    /// must be non-zero. Elements of `L` MUST be sorted in ascending order.
    fn derive_interpolating_value(x: Scalar, L: &[Scalar]) -> Scalar {
        // The FROST specification does not include the sorting requirement
        // on elements of `L`, but it is easy to apply by the caller, and
        // it makes the non-duplicate check much easier.

        // Check that the parameters are correct.
        let mut ff = false;
        for i in 0..L.len() {
            if x.equals(L[i]) != 0 {
                ff = true;
            }
            assert!(x.iszero() == 0);
            if i > 0 {
                assert!(scalar_cmp_vartime(L[i - 1], L[i]) == Ordering::Less);
            }
        }
        assert!(ff);

        // Compute the coefficient.
        let mut numerator = Scalar::ONE;
        let mut denominator = Scalar::ONE;
        let xi = x;
        for xj in L.iter() {
            if xi.equals(*xj) == 0 {
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

    /// Compare scalars numerically. For comparison purposes, scalars are
    /// converted to their unique integer representative in the 0 to p-1
    /// range (for a scalar modulus p).
    /// Note: this is not constant-time.
    fn scalar_cmp_vartime(x: Scalar, y: Scalar) -> Ordering {
        let xb = scalar_encode_le(x);
        let yb = scalar_encode_le(y);
        for i in (0..xb.len()).rev() {
            if xb[i] < yb[i] {
                return Ordering::Less;
            } else if xb[i] > yb[i] {
                return Ordering::Greater;
            }
        }
        Ordering::Equal
    }

} } // End of macro: define_frost_core

// ========================================================================

#[cfg(test)]
macro_rules! define_frost_tests { () => {

    use super::{GroupPrivateKey, GroupPublicKey, KeySplitter, VSSElement};
    use super::{SignerPrivateKeyShare, SignerPublicKey};
    use super::{Nonce, Commitment, SignatureShare, Signature, Coordinator};
    use super::{Point, Scalar, scalar_cmp_vartime};
    use super::{compute_binding_factors, point_decode, scalar_decode};
    use crate::{CryptoRng, RngCore, RngError};
    use sha2::{Sha512, Digest};
    use crate::Vec;
    use core::cmp::Ordering;

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
            assert!(ssk.ident.equals(Scalar::from_u64((i as u64) + 1)) != 0);
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
            assert!(ssk2.ident.equals(ssk.ident) != 0);
            assert!(ssk2.sk.equals(ssk.sk) != 0);
            assert!(ssk2.pk.equals(ssk.pk) != 0);
            assert!(ssk2.group_pk.pk.equals(group_pk.pk) != 0);
            assert!(ssk2.group_pk.pk_enc == group_pk.pk_enc);
            assert!(ssk2.verify_split(&vss2));
            signer_public_keys.push(ssk2.get_public_key());
        }

        assert!(signer_public_keys.len() == sk_shares.len());
        for i in 0..sk_shares.len() {
            assert!(signer_public_keys[i].ident.equals(sk_shares[i].ident) != 0);
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
            assert!(scalar_cmp_vartime(comms1[i].ident, comms1[i - 1].ident)
                == Ordering::Greater);
        }
        let comms2 = Commitment::decode_list(
            &Commitment::encode_list(&comms1)).unwrap();
        assert!(comms1.len() == comms2.len());
        for i in 0..comms2.len() {
            assert!(comms1[i].ident.equals(comms2[i].ident) != 0);
            assert!(comms1[i].hiding.equals(comms2[i].hiding) != 0);
            assert!(comms1[i].binding.equals(comms2[i].binding) != 0);
            let mut ff = false;
            for ss in signer_states.iter() {
                if ss.comm.ident.equals(comms2[i].ident) != 0 {
                    ff = true;
                    assert!(comms2[i].hiding.equals(ss.comm.hiding) != 0);
                    assert!(comms2[i].binding.equals(ss.comm.binding) != 0);
                    break;
                }
            }
            assert!(ff);
        }

        // Have the chosen signers perform the second round.
        let msg: &[u8] = b"sample";
        let mut sig_shares: Vec<SignatureShare> = Vec::new();
        for c in comms2.iter() {
            let mut ff = false;
            for i in 0..sk_shares.len() {
                if sk_shares[i].ident.equals(c.ident) != 0 {
                    ff = true;
                    let s = sk_shares[i].sign(signer_states[i].nonce,
                        signer_states[i].comm, msg, &comms2).unwrap();
                    sig_shares.push(s);
                    break;
                }
            }
            assert!(ff);
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

    // A pretend RNG for test purposes (deterministic engine that returns
    // a preset stream of 64 bytes).
    struct R64RNG {
        buf: [u8; 64],
        ptr: usize,
    }

    impl R64RNG {

        fn from_seed(seed: &[u8]) -> Self {
            let mut d = Self {
                buf: [0u8; 64],
                ptr: 0,
            };
            d.buf.copy_from_slice(seed);
            d
        }
    }

    impl RngCore for R64RNG {

        fn next_u32(&mut self) -> u32 {
            unimplemented!();
        }

        fn next_u64(&mut self) -> u64 {
            unimplemented!();
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            let len = dest.len();
            assert!(len <= self.buf.len() - self.ptr);
            dest.copy_from_slice(&self.buf[self.ptr..self.ptr + len]);
            self.ptr += len;
        }

        fn try_fill_bytes(&mut self, dest: &mut [u8])
            -> Result<(), RngError>
        {
            self.fill_bytes(dest);
            Ok(())
        }
    }

    impl CryptoRng for R64RNG { }

    #[test]
    fn KAT() {
        let group_sk = GroupPrivateKey::decode(&hex::decode(KAT_GROUP_SK).unwrap()).unwrap();
        let group_pk = GroupPublicKey::decode(&hex::decode(KAT_GROUP_PK).unwrap()).unwrap();
        assert!(group_pk.pk.equals(Point::mulgen(&group_sk.sk)) != 0);
        let msg = hex::decode(KAT_MSG).unwrap();
        let pcoeff = scalar_decode(&hex::decode(KAT_PCOEFF).unwrap()).unwrap();

        let sk1 = scalar_decode(&hex::decode(KAT_SK1).unwrap()).unwrap();
        let sk2 = scalar_decode(&hex::decode(KAT_SK2).unwrap()).unwrap();
        let sk3 = scalar_decode(&hex::decode(KAT_SK3).unwrap()).unwrap();
        assert!(sk1.equals(group_sk.sk + pcoeff) != 0);
        assert!(sk2.equals(group_sk.sk + pcoeff.mul2()) != 0);
        assert!(sk3.equals(group_sk.sk + pcoeff.mul3()) != 0);
        let S1 = SignerPrivateKeyShare {
            ident: Scalar::from_u32(1),
            sk: sk1,
            pk: Point::mulgen(&sk1),
            group_pk: group_pk,
        };
        let S2 = SignerPrivateKeyShare {
            ident: Scalar::from_u32(2),
            sk: sk2,
            pk: Point::mulgen(&sk2),
            group_pk: group_pk,
        };
        let S3 = SignerPrivateKeyShare {
            ident: Scalar::from_u32(3),
            sk: sk3,
            pk: Point::mulgen(&sk3),
            group_pk: group_pk,
        };

        let S1_nr = hex::decode(KAT_S1_NR).unwrap();
        let S1_hn = scalar_decode(&hex::decode(KAT_S1_HN).unwrap()).unwrap();
        let S1_bn = scalar_decode(&hex::decode(KAT_S1_BN).unwrap()).unwrap();
        let S1_hc = point_decode(&hex::decode(KAT_S1_HC).unwrap()).unwrap();
        let S1_bc = point_decode(&hex::decode(KAT_S1_BC).unwrap()).unwrap();
        let S1_bf = scalar_decode(&hex::decode(KAT_S1_BF).unwrap()).unwrap();
        assert!(S1_hc.equals(Point::mulgen(&S1_hn)) != 0);
        assert!(S1_bc.equals(Point::mulgen(&S1_bn)) != 0);
        let (S1_nonce, S1_comm) = S1.commit(&mut R64RNG::from_seed(&S1_nr));
        assert!(S1_nonce.ident.equals(S1.ident) != 0);
        assert!(S1_nonce.hiding.equals(S1_hn) != 0);
        assert!(S1_nonce.binding.equals(S1_bn) != 0);
        assert!(S1_comm.ident.equals(S1.ident) != 0);
        assert!(S1_comm.hiding.equals(S1_hc) != 0);
        assert!(S1_comm.binding.equals(S1_bc) != 0);
        /*
        let S1_nonce = Nonce { ident: S1.ident, hiding: S1_hn, binding: S1_bn };
        let S1_comm = S1_nonce.get_commitment();
        */

        let S3_nr = hex::decode(KAT_S3_NR).unwrap();
        let S3_hn = scalar_decode(&hex::decode(KAT_S3_HN).unwrap()).unwrap();
        let S3_bn = scalar_decode(&hex::decode(KAT_S3_BN).unwrap()).unwrap();
        let S3_hc = point_decode(&hex::decode(KAT_S3_HC).unwrap()).unwrap();
        let S3_bc = point_decode(&hex::decode(KAT_S3_BC).unwrap()).unwrap();
        let S3_bf = scalar_decode(&hex::decode(KAT_S3_BF).unwrap()).unwrap();
        assert!(S3_hc.equals(Point::mulgen(&S3_hn)) != 0);
        assert!(S3_bc.equals(Point::mulgen(&S3_bn)) != 0);
        let (S3_nonce, S3_comm) = S3.commit(&mut R64RNG::from_seed(&S3_nr));
        assert!(S3_nonce.ident.equals(S3.ident) != 0);
        assert!(S3_nonce.hiding.equals(S3_hn) != 0);
        assert!(S3_nonce.binding.equals(S3_bn) != 0);
        assert!(S3_comm.ident.equals(S3.ident) != 0);
        assert!(S3_comm.hiding.equals(S3_hc) != 0);
        assert!(S3_comm.binding.equals(S3_bc) != 0);
        /*
        let S3_nonce = Nonce { ident: S3.ident, hiding: S3_hn, binding: S3_bn };
        let S3_comm = S3_nonce.get_commitment();
        */

        let coor = Coordinator::new(2, group_pk).unwrap();
        let comms = coor.choose(&[S3_comm, S3_comm, S1_comm]).unwrap();
        assert!(comms.len() == 2);
        assert!(comms[0].ident.equals(S1.ident) != 0);
        assert!(comms[1].ident.equals(S3.ident) != 0);

        let bfs = compute_binding_factors(group_pk, &comms, &msg);
        assert!(bfs.len() == 2);
        assert!(bfs[0].ident.equals(S1.ident) != 0);
        assert!(bfs[0].factor.equals(S1_bf) != 0);
        assert!(bfs[1].ident.equals(S3.ident) != 0);
        assert!(bfs[1].factor.equals(S3_bf) != 0);

        let S1_sig_share = S1.sign(S1_nonce, S1_comm, &msg, &comms).unwrap();
        let S3_sig_share = S3.sign(S3_nonce, S3_comm, &msg, &comms).unwrap();
        let S1_ss_ref = scalar_decode(&hex::decode(KAT_S1_SIG_SHARE).unwrap()).unwrap();
        let S3_ss_ref = scalar_decode(&hex::decode(KAT_S3_SIG_SHARE).unwrap()).unwrap();
        assert!(S1_sig_share.ident.equals(S1.ident) != 0);
        assert!(S1_sig_share.zi.equals(S1_ss_ref) != 0);
        assert!(S3_sig_share.ident.equals(S3.ident) != 0);
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
#[cfg(feature = "ed25519")]
pub mod ed25519 {

    pub use crate::ed25519::{Point, Scalar};
    use sha2::{Sha512, Digest};

    define_frost_core!{}

    /// Decodes a point from bytes.
    fn point_decode(buf: &[u8]) -> Option<Point> {
        // As per draft-irtf-cfrg-frost-14, points must be verified to be
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

    /// Convert a scalar to its minimal integer representative (in the 0
    /// to p-1 range, for a modulus p), in unsigned little-endian convention.
    fn scalar_encode_le(x: Scalar) -> [u8; 32] {
        x.encode()
    }

    const NE: usize = 32;
    const NS: usize = 32;

    const CONTEXT_STRING: &[u8] = b"FROST-ED25519-SHA512-v1";

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
        static KAT_PCOEFF: &str = "178199860edd8c62f5212ee91eff1295d0d670ab4ed4506866bae57e7030b204";

        static KAT_SK1: &str = "929dcc590407aae7d388761cddb0c0db6f5627aea8e217f4a033f2ec83d93509";
        static KAT_SK2: &str = "a91e66e012e4364ac9aaa405fcafd370402d9859f7b6685c07eed76bf409e80d";
        static KAT_SK3: &str = "d3cb090a075eb154e82fdb4b3cb507f110040905468bb9c46da8bdea643a9a02";

        static KAT_S1_NR: &str = "0fd2e39e111cdc266f6c0f4d0fd45c947761f1f5d3cb583dfcb9bbaf8d4c9fec69cd85f631d5f7f2721ed5e40519b1366f340a87c2f6856363dbdcda348a7501";
        static KAT_S1_HN: &str = "812d6104142944d5a55924de6d49940956206909f2acaeedecda2b726e630407";
        static KAT_S1_BN: &str = "b1110165fc2334149750b28dd813a39244f315cff14d4e89e6142f262ed83301";
        static KAT_S1_HC: &str = "b5aa8ab305882a6fc69cbee9327e5a45e54c08af61ae77cb8207be3d2ce13de3";
        static KAT_S1_BC: &str = "67e98ab55aa310c3120418e5050c9cf76cf387cb20ac9e4b6fdb6f82a469f932";
        static KAT_S1_BF: &str = "f2cb9d7dd9beff688da6fcc83fa89046b3479417f47f55600b106760eb3b5603";

        static KAT_S3_NR: &str = "86d64a260059e495d0fb4fcc17ea3da7452391baa494d4b00321098ed2a0062f13e6b25afb2eba51716a9a7d44130c0dbae0004a9ef8d7b5550c8a0e07c61775";
        static KAT_S3_HN: &str = "c256de65476204095ebdc01bd11dc10e57b36bc96284595b8215222374f99c0e";
        static KAT_S3_BN: &str = "243d71944d929063bc51205714ae3c2218bd3451d0214dfb5aeec2a90c35180d";
        static KAT_S3_HC: &str = "cfbdb165bd8aad6eb79deb8d287bcc0ab6658ae57fdcc98ed12c0669e90aec91";
        static KAT_S3_BC: &str = "7487bc41a6e712eea2f2af24681b58b1cf1da278ea11fe4e8b78398965f13552";
        static KAT_S3_BF: &str = "b087686bf35a13f3dc78e780a34b0fe8a77fef1b9938c563f5573d71d8d7890f";

        static KAT_S1_SIG_SHARE: &str = "001719ab5a53ee1a12095cd088fd149702c0720ce5fd2f29dbecf24b7281b603";
        static KAT_S3_SIG_SHARE: &str = "bd86125de990acc5e1f13781d8e32c03a9bbd4c53539bbc106058bfd14326007";

        static KAT_SIG: &str = "36282629c383bb820a88b71cae937d41f2f2adfcc3d02e55507e2fb9e2dd3cbebd9d2b0844e49ae0f3fa935161e1419aab7b47d21a37ebeae1f17d4987b3160b";

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
#[cfg(feature = "ristretto255")]
pub mod ristretto255 {
    pub use crate::ristretto255::{Point, Scalar};
    use sha2::{Sha512, Digest};

    define_frost_core!{}

    /// Decodes a point from bytes.
    fn point_decode(buf: &[u8]) -> Option<Point> {
        // As per draft-irtf-cfrg-frost-14, point decoding is NOT allowed to
        // return the neutral element.
        let P = Point::decode(buf)?;
        if P.isneutral() != 0 {
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

    /// Convert a scalar to its minimal integer representative (in the 0
    /// to p-1 range, for a modulus p), in unsigned little-endian convention.
    fn scalar_encode_le(x: Scalar) -> [u8; 32] {
        x.encode()
    }

    const NE: usize = 32;
    const NS: usize = 32;

    const CONTEXT_STRING: &[u8] = b"FROST-RISTRETTO255-SHA512-v1";

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
        static KAT_PCOEFF: &str = "410f8b744b19325891d73736923525a4f596c805d060dfb9c98009d34e3fec02";

        static KAT_SK1: &str = "5c3430d391552f6e60ecdc093ff9f6f4488756aa6cebdbad75a768010b8f830e";
        static KAT_SK2: &str = "b06fc5eac20b4f6e1b271d9df2343d843e1e1fb03c4cbb673f2872d459ce6f01";
        static KAT_SK3: &str = "f17e505f0e2581c6acfe54d3846a622834b5e7b50cad9a2109a97ba7a80d5c04";

        static KAT_S1_NR: &str = "f595a133b4d95c6e1f79887220c8b275ce6277e7f68a6640e1e7140f9be2fb5c34dd1001360e3513cb37bebfabe7be4a32c5bb91ba19fbd4360d039111f0fbdc";
        static KAT_S1_HN: &str = "214f2cabb86ed71427ea7ad4283b0fae26b6746c801ce824b83ceb2b99278c03";
        static KAT_S1_BN: &str = "c9b8f5e16770d15603f744f8694c44e335e8faef00dad182b8d7a34a62552f0c";
        static KAT_S1_HC: &str = "965def4d0958398391fc06d8c2d72932608b1e6255226de4fb8d972dac15fd57";
        static KAT_S1_BC: &str = "ec5170920660820007ae9e1d363936659ef622f99879898db86e5bf1d5bf2a14";
        static KAT_S1_BF: &str = "8967fd70fa06a58e5912603317fa94c77626395a695a0e4e4efc4476662eba0c";

        static KAT_S3_NR: &str = "daa0cf42a32617786d390e0c7edfbf2efbd428037069357b5173ae61d6dd5d5eb4387e72b2e4108ce4168931cc2c7fcce5f345a5297368952c18b5fc8473f050";
        static KAT_S3_HN: &str = "3f7927872b0f9051dd98dd73eb2b91494173bbe0feb65a3e7e58d3e2318fa40f";
        static KAT_S3_BN: &str = "ffd79445fb8030f0a3ddd3861aa4b42b618759282bfe24f1f9304c7009728305";
        static KAT_S3_HC: &str = "480e06e3de182bf83489c45d7441879932fd7b434a26af41455756264fbd5d6e";
        static KAT_S3_BC: &str = "3064746dfd3c1862ef58fc68c706da287dd925066865ceacc816b3a28c7b363b";
        static KAT_S3_BF: &str = "f2c1bb7c33a10511158c2f1766a4a5fadf9f86f2a92692ed333128277cc31006";

        static KAT_S1_SIG_SHARE: &str = "9285f875923ce7e0c491a592e9ea1865ec1b823ead4854b48c8a46287749ee09";
        static KAT_S3_SIG_SHARE: &str = "7cb211fe0e3d59d25db6e36b3fb32344794139602a7b24f1ae0dc4e26ad7b908";

        static KAT_SIG: &str = "fc45655fbc66bbffad654ea4ce5fdae253a49a64ace25d9adb62010dd9fb25552164141787162e5b4cab915b4aa45d94655dbb9ed7c378a53b980a0be220a802";

        define_frost_tests!{}
    }
}

/// FROST(Ed448, SHAKE256)
#[cfg(feature = "ed448")]
pub mod ed448 {

    pub use crate::ed448::{Point, Scalar};
    use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};

    define_frost_core!{}

    /// Decodes a point from bytes.
    fn point_decode(buf: &[u8]) -> Option<Point> {
        // As per draft-irtf-cfrg-frost-14, points must be verified to be
        // in the proper prime-order subgroup, and not the neutral element.
        let P = Point::decode(buf)?;
        if P.isneutral() != 0 || P.is_in_subgroup() == 0 {
            None
        } else {
            Some(P)
        }
    }

    /// Encodes a point into bytes.
    fn point_encode(P: Point) -> [u8; 57] {
        P.encode()
    }

    /// Decodes a scalar from bytes.
    fn scalar_decode(buf: &[u8]) -> Option<Scalar> {
        // In draft-irtf-cfrg-frost-14, scalars are encoded over 57 bytes,
        // even though only 56 bytes are needed; the extra byte must
        // always be zero.
        if buf.len() != 57 || buf[56] != 0 {
            None
        } else {
            Scalar::decode(&buf[..56])
        }
    }

    /// Encodes a scalar into bytes.
    fn scalar_encode(x: Scalar) -> [u8; 57] {
        // In draft-irtf-cfrg-frost-14, scalars are encoded over 57 bytes,
        // even though only 56 bytes are needed; the extra byte must
        // always be zero.
        let mut buf = [0u8; 57];
        buf[..56].copy_from_slice(&x.encode());
        buf
    }

    /// Convert a scalar to its minimal integer representative (in the 0
    /// to p-1 range, for a modulus p), in unsigned little-endian convention.
    fn scalar_encode_le(x: Scalar) -> [u8; 56] {
        x.encode()
    }

    const NE: usize = 57;
    const NS: usize = 57;

    const CONTEXT_STRING: &[u8] = b"FROST-ED448-SHAKE256-v1";

    fn H1(msg: &[u8]) -> Scalar {
        let mut sh = Shake256::default();
        sh.update(CONTEXT_STRING);
        sh.update(b"rho");
        sh.update(msg);
        let mut buf = [0u8; 114];
        sh.finalize_xof().read(&mut buf);
        Scalar::decode_reduce(&buf)
    }

    fn H2(gc_enc: &[u8], pk_enc: &[u8], msg: &[u8]) -> Scalar {
        let mut sh = Shake256::default();
        // Prefix compatible with RFC 8032:
        //    dom4(F, C)     constant string "SigEd448"
        //    phflag         0 for raw (non pre-hashed) mode
        //    ctxlen         0 for an empty context string
        sh.update(b"SigEd448");
        sh.update(&[0u8; 2]);
        sh.update(gc_enc);
        sh.update(pk_enc);
        sh.update(msg);
        let mut buf = [0u8; 114];
        sh.finalize_xof().read(&mut buf);
        Scalar::decode_reduce(&buf)
    }

    fn H3(msg: &[u8]) -> Scalar {
        let mut sh = Shake256::default();
        sh.update(CONTEXT_STRING);
        sh.update(b"nonce");
        sh.update(msg);
        let mut buf = [0u8; 114];
        sh.finalize_xof().read(&mut buf);
        Scalar::decode_reduce(&buf)
    }

    fn H4(msg: &[u8]) -> [u8; 114] {
        let mut sh = Shake256::default();
        sh.update(CONTEXT_STRING);
        sh.update(b"msg");
        sh.update(msg);
        let mut r = [0u8; 114];
        sh.finalize_xof().read(&mut r);
        r
    }

    fn H5(msg: &[u8]) -> [u8; 114] {
        let mut sh = Shake256::default();
        sh.update(CONTEXT_STRING);
        sh.update(b"com");
        sh.update(msg);
        let mut r = [0u8; 114];
        sh.finalize_xof().read(&mut r);
        r
    }

    fn H6(pk_enc: &[u8], sk_enc: &[u8], seed: &[u8], msg: &[u8]) -> Scalar {
        let mut sh = Shake256::default();
        sh.update(CONTEXT_STRING);
        sh.update(b"single-signer");
        sh.update(pk_enc);
        sh.update(sk_enc);
        sh.update(seed);
        sh.update(msg);
        let mut buf = [0u8; 114];
        sh.finalize_xof().read(&mut buf);
        Scalar::decode_reduce(&buf)
    }

    #[cfg(test)]
    mod tests {

        static KAT_GROUP_SK: &str = "6298e1eef3c379392caaed061ed8a31033c9e9e3420726f23b404158a401cd9df24632adfe6b418dc942d8a091817dd8bd70e1c72ba52f3c00";
        static KAT_GROUP_PK: &str = "3832f82fda00ff5365b0376df705675b63d2a93c24c6e81d40801ba265632be10f443f95968fadb70d10786827f30dc001c8d0f9b7c1d1b000";
        static KAT_MSG: &str = "74657374";
        static KAT_PCOEFF: &str = "dbd7a514f7a731976620f0436bd135fe8dddc3fadd6e0d13dbd58a1981e587d377d48e0b7ce4e0092967c5e85884d0275a7a740b6abdcd0500";

        static KAT_SK1: &str = "4a2b2f5858a932ad3d3b18bd16e76ced3070d72fd79ae4402df201f525e754716a1bc1b87a502297f2a99d89ea054e0018eb55d39562fd0100";
        static KAT_SK2: &str = "2503d56c4f516444a45b080182b8a2ebbe4d9b2ab509f25308c88c0ea7ccdc44e2ef4fc4f63403a11b116372438a1e287265cadeff1fcb0700";
        static KAT_SK3: &str = "00db7a8146f995db0a7cf844ed89d8e94c2b5f259378ff66e39d172828b264185ac4decf7219e4aa4478285b9c0eef4fccdf3eea69dd980d00";

        static KAT_S1_NR: &str = "9cda90c98863ef3141b75f09375757286b4bc323dd61aeb45c07de45e4937bbd781bf4881ffe1aa06f9341a747179f07a49745f8cd37d4696f226aa065683c0a";
        static KAT_S1_HN: &str = "f922beb51a5ac88d1e862278d89e12c05263b945147db04b9566acb2b5b0f7422ccea4f9286f4f80e6b646e72143eeaecc0e5988f8b2b93100";
        static KAT_S1_BN: &str = "1890f16a120cdeac092df29955a29c7cf29c13f6f7be60e63d63f3824f2d37e9c3a002dfefc232972dc08658a8c37c3ec06a0c5dc146150500";
        static KAT_S1_HC: &str = "3518c2246c874569e54ab254cb1da666ca30f7879605cc43b4d2c47a521f8b5716080ab723d3a0cd04b7e41f3cc1d3031c94ccf3829b23fe80";
        static KAT_S1_BC: &str = "11b3d5220c57d02057497de3c4eebab384900206592d877059b0a5f1d5250d002682f0e22dff096c46bb81b46d60fcfe7752ed47cea76c3900";
        static KAT_S1_BF: &str = "71966390dfdbed73cf9b79486f3b70e23b243e6c40638fb55998642a60109daecbfcb879eed9fe7dbbed8d9e47317715a5740f772173342e00";

        static KAT_S3_NR: &str = "b3adf97ceea770e703ab295babf311d77e956a20d3452b4b3344aa89a828e6df81dbe7742b0920930299197322b255734e52bbb91f50cfe8ce689f56fadbce31";
        static KAT_S3_HN: &str = "ccb5c1e82f23e0a4b966b824dbc7b0ef1cc5f56eeac2a4126e2b2143c5f3a4d890c52d27803abcf94927faf3fc405c0b2123a57a93cefa3b00";
        static KAT_S3_BN: &str = "e089df9bf311cf711e2a24ea27af53e07b846d09692fe11035a1112f04d8b7462a62f34d8c01493a22b57a1cbf1f0a46c77d64d46449a90100";
        static KAT_S3_HC: &str = "1254546d7d104c04e4fbcf29e05747e2edd392f6787d05a6216f3713ef859efe573d180d291e48411e5e3006e9f90ee986ccc26b7a42490b80";
        static KAT_S3_BC: &str = "3ef0cec20be15e56b3ddcb6f7b956fca0c8f71990f45316b537b4f64c5e8763e6629d7262ff7cd0235d0781f23be97bf8fa8817643ea19cd00";
        static KAT_S3_BF: &str = "236a6f7239ac2019334bad21323ec93bef2fead37bd55114356419f3fc1fb59f797f44079f28b1a64f51dd0a113f90f2c3a1c27d2faa4f1300";

        static KAT_S1_SIG_SHARE: &str = "e1eb9bfbef792776b7103891032788406c070c5c315e3bf5d64acd46ea8855e85b53146150a09149665cbfec71626810b575e6f4dbe9ba3700";
        static KAT_S3_SIG_SHARE: &str = "815434eb0b9f9242d54b8baf2141fe28976cabe5f441ccfcd5ee7cdb4b52185b02b99e6de28e2ab086c7764068c5a01b5300986b9f084f3e00";

        static KAT_SIG: &str = "cd642cba59c449dad8e896a78a60e8edfcbd9040df524370891ff8077d47ce721d683874483795f0d85efcbd642c4510614328605a19c6ed806ffb773b6956419537cdfdb2b2a51948733de192dcc4b82dc31580a536db6d435e0cb3ce322fbcf9ec23362dda27092c08767e607bf2093600";

        define_frost_tests!{}

        #[test]
        fn interop_ed448() {
            // FROST signatures are supposed to be verifiable with a
            // plain RFC 8032 Ed448 verifier.
            use crate::ed448::PublicKey;

            let mut rng = DRNG::from_seed(b"interop_ed448");
            let msg = b"sample";
            let group_sk = GroupPrivateKey::generate(&mut rng);
            let esig = group_sk.sign(&mut rng, msg).encode();

            let group_pk = group_sk.get_public_key();
            let ed_pk = PublicKey::decode(&group_pk.encode()).unwrap();
            assert!(ed_pk.verify_raw(&esig, msg));
        }
    }
}

/// FROST(P-256, SHA-256)
#[cfg(feature = "p256")]
pub mod p256 {
    pub use crate::p256::{Point, Scalar};
    use sha2::{Sha256, Digest};

    define_frost_core!{}

    /// Decodes a point from bytes.
    fn point_decode(buf: &[u8]) -> Option<Point> {
        // As per draft-irtf-cfrg-frost-14, points use the compressed
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

    /// Convert a scalar to its minimal integer representative (in the 0
    /// to p-1 range, for a modulus p), in unsigned little-endian convention.
    fn scalar_encode_le(x: Scalar) -> [u8; 32] {
        x.encode()
    }

    const NE: usize = 33;
    const NS: usize = 32;

    const CONTEXT_STRING: &[u8] = b"FROST-P256-SHA256-v1";

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
        static KAT_PCOEFF: &str = "80f25e6c0709353e46bfbe882a11bdbb1f8097e46340eb8673b7e14556e6c3a4";

        static KAT_SK1: &str = "0c9c1a0fe806c184add50bbdcac913dda73e482daf95dcb9f35dbb0d8a9f7731";
        static KAT_SK2: &str = "8d8e787bef0ff6c2f494ca45f4dad198c6bee01212d6c84067159c52e1863ad5";
        static KAT_SK3: &str = "0e80d6e8f6192c003b5488ce1eec8f5429587d48cf001541e713b2d53c09d928";

        static KAT_S1_NR: &str = "ec4c891c85fee802a9d757a67d1252e7f4e5efb8a538991ac18fbd0e06fb6fd39334e29d09061223f69a09421715a347e4e6deba77444c8f42b0c833f80f4ef9";
        static KAT_S1_HN: &str = "9f0542a5ba879a58f255c09f06da7102ef6a2dec6279700c656d58394d8facd4";
        static KAT_S1_BN: &str = "6513dfe7429aa2fc972c69bb495b27118c45bbc6e654bb9dc9be55385b55c0d7";
        static KAT_S1_HC: &str = "0213b3e6298bf8ad46fd5e9389519a8665d63d98f4ec6a1fcca434e809d2d8070e";
        static KAT_S1_BC: &str = "02188ff1390bf69374d7b272e454b1878ef10a6b6ea3ff36f114b300b4dbd5233b";
        static KAT_S1_BF: &str = "7925f0d4693f204e6e59233e92227c7124664a99739d2c06b81cf64ddf90559e";

        static KAT_S3_NR: &str = "c0451c5a0a5480d6c1f860e5db7d655233dca2669fd90ff048454b8ce983367b2ba5f7793ae700e40e78937a82f407dd35e847e33d1e607b5c7eb6ed2a8ed799";
        static KAT_S3_HN: &str = "f73444a8972bcda9e506bbca3d2b1c083c10facdf4bb5d47fef7c2dc1d9f2a0d";
        static KAT_S3_BN: &str = "44c6a29075d6e7e4f8b97796205f9e22062e7835141470afe9417fd317c1c303";
        static KAT_S3_HC: &str = "033ac9a5fe4a8b57316ba1c34e8a6de453033b750e8984924a984eb67a11e73a3f";
        static KAT_S3_BC: &str = "03a7a2480ee16199262e648aea3acab628a53e9b8c1945078f2ddfbdc98b7df369";
        static KAT_S3_BF: &str = "e10d24a8a403723bcb6f9bb4c537f316593683b472f7a89f166630dde11822c4";

        static KAT_S1_SIG_SHARE: &str = "400308eaed7a2ddee02a265abe6a1cfe04d946ee8720768899619cfabe7a3aeb";
        static KAT_S3_SIG_SHARE: &str = "561da3c179edbb0502d941bb3e3ace3c37d122aaa46fb54499f15f3a3331de44";

        static KAT_SIG: &str = "026d8d434874f87bdb7bc0dfd239b2c00639044f9dcb195e9a04426f70bfa4b70d9620acac6767e8e3e3036815fca4eb3a3caa69992b902bcd3352fc34f1ac192f";

        define_frost_tests!{}
    }
}

/// FROST(secp256k1, SHA-256)
#[cfg(feature = "secp256k1")]
pub mod secp256k1 {
    pub use crate::secp256k1::{Point, Scalar};
    use sha2::{Sha256, Digest};

    define_frost_core!{}

    /// Decodes a point from bytes.
    fn point_decode(buf: &[u8]) -> Option<Point> {
        // As per draft-irtf-cfrg-frost-14, points use the compressed
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

    /// Convert a scalar to its minimal integer representative (in the 0
    /// to p-1 range, for a modulus p), in unsigned little-endian convention.
    fn scalar_encode_le(x: Scalar) -> [u8; 32] {
        x.encode()
    }

    const NE: usize = 33;
    const NS: usize = 32;

    const CONTEXT_STRING: &[u8] = b"FROST-secp256k1-SHA256-v1";

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
        static KAT_PCOEFF: &str = "fbf85eadae3058ea14f19148bb72b45e4399c0b16028acaf0395c9b03c823579";

        static KAT_SK1: &str = "08f89ffe80ac94dcb920c26f3f46140bfc7f95b493f8310f5fc1ea2b01f4254c";
        static KAT_SK2: &str = "04f0feac2edcedc6ce1253b7fab8c86b856a797f44d83d82a385554e6e401984";
        static KAT_SK3: &str = "00e95d59dd0d46b0e303e500b62b7ccb0e555d49f5b849f5e748c071da8c0dbc";

        static KAT_S1_NR: &str = "7ea5ed09af19f6ff21040c07ec2d2adbd35b759da5a401d4c99dd26b82391cb247acab018f116020c10cb9b9abdc7ac10aae1b48ca6e36dc15acb6ec9be5cdc5";
        static KAT_S1_HN: &str = "841d3a6450d7580b4da83c8e618414d0f024391f2aeb511d7579224420aa81f0";
        static KAT_S1_BN: &str = "8d2624f532af631377f33cf44b5ac5f849067cae2eacb88680a31e77c79b5a80";
        static KAT_S1_HC: &str = "03c699af97d26bb4d3f05232ec5e1938c12f1e6ae97643c8f8f11c9820303f1904";
        static KAT_S1_BC: &str = "02fa2aaccd51b948c9dc1a325d77226e98a5a3fe65fe9ba213761a60123040a45e";
        static KAT_S1_BF: &str = "3e08fe561e075c653cbfd46908a10e7637c70c74f0a77d5fd45d1a750c739ec6";

        static KAT_S3_NR: &str = "e6cc56ccbd0502b3f6f831d91e2ebd01c4de0479e0191b66895a4ffd9b68d5447203d55eb82a5ca0d7d83674541ab55f6e76f1b85391d2c13706a89a064fd5b9";
        static KAT_S3_HN: &str = "2b19b13f193f4ce83a399362a90cdc1e0ddcd83e57089a7af0bdca71d47869b2";
        static KAT_S3_BN: &str = "7a443bde83dc63ef52dda354005225ba0e553243402a4705ce28ffaafe0f5b98";
        static KAT_S3_HC: &str = "03077507ba327fc074d2793955ef3410ee3f03b82b4cdc2370f71d865beb926ef6";
        static KAT_S3_BC: &str = "02ad53031ddfbbacfc5fbda3d3b0c2445c8e3e99cbc4ca2db2aa283fa68525b135";
        static KAT_S3_BF: &str = "93f79041bb3fd266105be251adaeb5fd7f8b104fb554a4ba9a0becea48ddbfd7";

        static KAT_S1_SIG_SHARE: &str = "c4fce1775a1e141fb579944166eab0d65eefe7b98d480a569bbbfcb14f91c197";
        static KAT_S3_SIG_SHARE: &str = "0160fd0d388932f4826d2ebcd6b9eaba734f7c71cf25b4279a4ca2581e47b18d";

        static KAT_SIG: &str = "0205b6d04d3774c8929413e3c76024d54149c372d57aae62574ed74319b5ea14d0c65dde8492a7471437e6c2fe3da49b90d23f642b5c6dbe7e36089f096dd97324";

        define_frost_tests!{}
    }
}
