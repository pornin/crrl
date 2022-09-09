// This sample code shows how to use the crrl FROST implementation.

use crrl::frost::ristretto255::{
    GroupPrivateKey,
    SignerPrivateKeyShare,
    SignerPublicKey,
    KeySplitter,
    VSSElement,
    SignatureShare,
    Commitment,
    Nonce,
    Coordinator,
    };
use rand::RngCore;
use rand::rngs::OsRng;
use std::vec::Vec;

fn main() {

    // We want `max_signers` individual signers, such that a threshold
    // of `min_signers` of them is required to compute a signature.
    // Rules: 2 <= min_signers <= max_signers <= 65535
    let max_signers = 5;
    let min_signers = 3;

    // ====================================================================
    // KEY GENERATION
    //
    // This step happens once. A trusted dealer generates the group private
    // key and splits it into individual key shares. Each signer receives
    // one key share. The signers can verify a VSS commitment by the dealer
    // to validate that the split was performed correctly (though the trusted
    // dealer is still trusted with using a proper entropy source for the
    // private key, and not remembering any secret afterwards).

    // =========== trusted dealer ===========

    // Generate a group private key.
    let mut rng = OsRng::default();
    let group_sk = GroupPrivateKey::generate(&mut rng);

    // Split the key into individual signer key shares.
    let (sk_share, vss) = KeySplitter::trusted_split(
        &mut rng, group_sk, min_signers, max_signers);

    // Send its key share to each signer.
    // Optionally: also send the VSS commitment that allows each signer
    // to verify that the share was properly generated.
    let mut enc_sk_share: Vec<[u8; SignerPrivateKeyShare::ENC_LEN]> =
        Vec::new();
    for sks in sk_share.iter() {
        enc_sk_share.push(sks.encode());
    }
    let enc_vss = VSSElement::encode_list(&vss);

    // Also extract the group public key and each individual signer public
    // key; they should be "published" (everybody knows them).
    let group_pk = group_sk.get_public_key();
    let mut signer_pk: Vec<SignerPublicKey> = Vec::new();
    for sks in sk_share.iter() {
        signer_pk.push(sks.get_public_key());
    }

    // =========== signers ===========

    // Each signer receives its private key share, decodes it, and
    // optionally verifies the VSS commitment that demonstrates proper
    // generation of the share.
    // In this example code we simulate all signers in a loop.

    let mut signer_sk_share: Vec<SignerPrivateKeyShare> = Vec::new();
    for esks in enc_sk_share.iter() {
        // All decoding operations return Option<something> so that None
        // is obtained on decoding failure. In this example we use unwrap(),
        // but this is where some error handling should happen.
        let sks = SignerPrivateKeyShare::decode(esks).unwrap();

        // Verify the VSS commitment (optional; needed only if the dealing
        // process is such that accidental or malicious alteration of shares
        // may happen).
        let vss = VSSElement::decode_list(&enc_vss).unwrap();
        if !sks.verify_split(&vss) {
            panic!("invalid key share");
        }

        // The signer stores its private key share (securely! It's secret).
        // As shown above, it can be encoded and decoded, for storage in
        // a file or equivalent. In this example, we keep an in-RAM
        // structure.
        signer_sk_share.push(sks);
    }

    // ====================================================================
    // SIGNATURE GENERATION
    //
    // Whenever a signature must be computed, over a given message, a
    // two-round protocol happens:
    //
    //   Round 1: each signer generates a per-signature nonce and associated
    //   commitment; the commitments are sent to the coordinator. Each signer
    //   remembers its nonce and commitment.
    //
    //   Round 2: the coordinator selects enough signers (among received
    //   commitments) to meet the threshold. The corresponding list of
    //   commitments is sent to the signers, along with the message. Each
    //   signer computes and sends back to the coordinator a signature
    //   share. The coordinator assembles the signature shares into the
    //   signature value.

    // =========== signers ===========

    // Each signer generates a nonce and a commitment. The commitment is
    // sent to the coordinator.
    let mut signer_nonce: Vec<Nonce> = Vec::new();
    let mut signer_comm: Vec<Commitment> = Vec::new();
    let mut enc_signer_comm: Vec<[u8; Commitment::ENC_LEN]> = Vec::new();
    for sks in signer_sk_share.iter() {
        let (nonce, comm) = sks.commit(&mut rng);
        signer_nonce.push(nonce);
        signer_comm.push(comm);
        enc_signer_comm.push(comm.encode());
    }

    // =========== coordinator ===========

    // The coordinator knows the group public key and the signature
    // threshold.
    let coordinator = Coordinator::new(min_signers, group_pk).unwrap();

    // This is the message to sign.
    let msg: &[u8] = b"sample";

    // The coordinator receives _some_ commitments. The commitments may
    // be obtained in any order; some may missing; duplicates are tolerated
    // (they are automatically ignored).
    // In this example, we give apply a random permutation to the
    // encoded commitments to simulate some network-induced shuffling.
    for i in 0..enc_signer_comm.len() - 1 {
        let j = i + (rng.next_u64() as usize) % (enc_signer_comm.len() - i);
        if i != j {
            let t = enc_signer_comm[i];
            enc_signer_comm[i] = enc_signer_comm[j];
            enc_signer_comm[j] = t;
        }
    }

    // Decode the commitments and use them to select a proper subset.
    // The encoded commitments are sent to the selected signers (the
    // selected signers are identified by the 'ident' fields of the
    // commitments that have been chosen).
    let mut received_signer_comm: Vec<Commitment> = Vec::new();
    for esc in enc_signer_comm.iter() {
        let sc = Commitment::decode(esc).unwrap();
        received_signer_comm.push(sc);
    }
    let chosen_comm = coordinator.choose(&received_signer_comm).unwrap();
    let enc_chosen_comm = Commitment::encode_list(&chosen_comm);

    // =========== signers ===========

    // The selected signers receive the encoded commitments. The coordinator
    // may know who are the selected signers by looking at the identifiers
    // (the `Commitment`, `SignerPublicKey` and `SignerPrivateKeyShare`
    // all have matching public `ident` fields). Another option (which is
    // used below) is to send the encoded commitments to everybody and see
    // what they answer; only actually selected signers will respond.
    let mut enc_sig_share: Vec<[u8; SignatureShare::ENC_LEN]> = Vec::new();
    for (sks, (nonce, comm)) in signer_sk_share.iter().zip(
        signer_nonce.iter().zip(signer_comm))
    {
        // The signer knows its private key share (sks), nonce,
        // and commitment.
        // Note: the commitment could also be recomputed from the nonce,
        // using `nonce.get_commitment()`. Remembering the commitment
        // saves a few clock cycles.

        // Decode the received commitment list.
        let comm_list = Commitment::decode_list(&enc_chosen_comm).unwrap();

        // Compute the signature share from this signer. This may fail
        // if the commitment list is incorrect, but also if this signer
        // was not actually selected in the list.
        match sks.sign(*nonce, comm, msg, &comm_list) {
            Some(ss) => { enc_sig_share.push(ss.encode()); }
            None     => { }
        }
    }

    // =========== coordinator ===========

    // The coordinator receives the encoded signature shares (in any order),
    // decodes them, then assembles them into the signature. We again
    // (for this example) randomly shuffle the list of encoded shares.
    for i in 0..enc_sig_share.len() - 1 {
        let j = i + (rng.next_u64() as usize) % (enc_sig_share.len() - i);
        if i != j {
            let t = enc_sig_share[i];
            enc_sig_share[i] = enc_sig_share[j];
            enc_sig_share[j] = t;
        }
    }

    // Decode the encoded signature shares.
    let mut sig_share: Vec<SignatureShare> = Vec::new();
    for ess in enc_sig_share.iter() {
        sig_share.push(SignatureShare::decode(ess).unwrap());
    }

    // Assemble the signature. This also verifies each share, _and_ checks
    // that the assembled signature is valid.
    // The coordinator uses the known signer public keys (signer_pk list);
    // that list can be provided in any order and also contain public keys of
    // signers that were not selected for this signature generation.
    let sig = coordinator.assemble_signature(
        &sig_share, &chosen_comm, &signer_pk, msg).unwrap();

    // The signature can be encoded into bytes.
    let esig = sig.encode();

    // ====================================================================
    // SIGNATURE VERIFICATION

    // Generated signatures can be verified against the group public key.
    if !group_pk.verify_esig(&esig, msg) {
        panic!("signature verification failed");
    }
    println!("OK");
}
