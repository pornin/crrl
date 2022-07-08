use super::{addcarry_u32, subborrow_u32, umull_add2, sgnw};

// Given integers k and n, with 0 <= k < n < Nmax (with n prime),
// return signed integers c0 and c1 such that k = c0/c1 mod n. Integers
// are provided as arrays of 32-bit limbs in little-endian convention
// (least significant limb comes first). This function is NOT
// constant-time and MUST NOT be used with secret inputs.
//
// Limit Nmax is such that the solution always exists; its value is:
//   Nmax = floor(2^254 / (2/sqrt(3)))
//        = 0x376CF5D0B09954E764AE85AE0F17077124BB06998A7B48F318E414C90DC8B4DC
//
// If a larger n is provided as parameter, then the algorithm still
// terminates, but the real (c0, c1) may be larger than 128 bits, and thus
// only truncated results are returned.
pub(crate) fn lagrange253_vartime(k: &[u32; 8], n: &[u32; 8]) -> (i128, i128) {

    // IMPLEMENTATION NOTES:
    // =====================
    //
    // We follow the optimized binary algorithm described in:
    // https://eprint.iacr.org/2020/454
    //
    // We represent integers as arrays of 64-bit limbs, in little-endian
    // convention. If such an integer can be negative, then two's
    // complement is used.
    //
    // Initial temporary values are 510-bit integers; we represent them
    // over 8 limbs. When they have shrunk enough, we switch to 384-bit
    // representations (6 limbs). Since the results are guaranteed to
    // fit in 128 bits (with the sign bit), we can keep their accumulators
    // in plain i128 integers.
    //
    // Algorithm:
    //
    //   Init:
    //      u = [n, 0]
    //      v = [k, 1]
    //      nu = n^2        (squared L2 norm of vector u)
    //      nv = k^2 + 1    (squared L2 norm of vector v)
    //      sp = n*k        (scalar produced of u and v)
    //
    //   Loop:
    //      - if nu < nv then:
    //           (u, v) <- (v, u)
    //           (nu, nv) <- (nv, nu)
    //      - if bitlength(nv) <= 254 then:
    //           return (v0, v1)
    //      - s <- max(0, bitlength(sp) - bitlength(nv))
    //      - if sp > 0 then:
    //           u <- u - (v << s)
    //           nu <- nu + (nv << (2*s)) - (sp << (s+1))
    //           sp <- sp - (nv << s)
    //        else:
    //           u <- u + (v << s)
    //           nu <- nu + (nv << (2*s)) + (sp << (s+1))
    //           sp <- sp + (nv << s)
    //
    // (u, v) is a basis of a dimension-2 lattice; each operation is
    // reversible, and thus transforms (u, v) into another basis for the
    // same lattice.
    //
    // At all points, nu is the squared L2 norm of u, nv is the squared
    // L2 norm of v, and sp is the scalar product of u and v; whenever u
    // (or v) is modified, then nu (or nv) and sp is modified
    // accordingly. The two vectors gradually shrink; at the start of
    // each loop, the conditional swap ensures that v is the smallest of
    // the two.
    //
    // Algorithm stops when the smallest vector (v) has L2 norm at most
    // 2^254. This means that v0^2 + v1^2 < 2^254, which implies that v0
    // and v1 both fit on 127 bits in absolute value (hence on 128 bits
    // when including the sign bit). It was shown by Hermite and later
    // Minkowski that the smallest non-zero vector of a dimension-2
    // lattice has squared L2 norm at most (2/sqrt(3))*vol, where vol is
    // the volume of the lattice. Here, the lattice volume is equal to
    // n, which is why we can always get a vector v such that nv <=
    // (2/sqrt(3))*n < 2^254 (since n < 2^253). It can be shown that
    // the algorithm necessarily converges in at most 1008 iterations,
    // but in practice the algorithm is much faster, with an average
    // number of iterations slightly below 100.
    //
    //
    // Notes for a large modulus:
    //
    // If the source modulus is larger than Nmax, then we cannot always
    // get v such that N(v) < 2^254. For that case, we include a second
    // exit test. It can be shown (see the paper, section 2.4) that, as
    // long as the algorithm makes progress, the scalar product sp will
    // necessarily shrink by at least one bit every two iterations. We
    // can thus consider that we have reached the end of the algorithm
    // when sp ceases to shrink. In that case, we still have a shortest
    // vector in v, except that the coordinates are truncated (we stored
    // them in 128-bit integers). Thanks to the bound on nv, we know
    // that the largest value for a coordinate of that vector (in absolute
    // value) is sqrt((2/sqrt(3))*n), i.e. less than 1.075*2^128 for a
    // modulus n < 2^256.
    //
    // When the modulus is large (such that n^2 >= 2^511), some care
    // must be exercised for the first iteration, because the scalar
    // product sp may have size up to 512 bits: in that case, the top
    // bit could be 1, while the value is positive. We can handle this case
    // by noticing the following:
    //   - Initially, sp = k*n, which is never negative.
    //   - If sp has size 512 bits, then len(sp) >= len(nv), and
    //     the first iteration is in one of the first two cases of
    //     section 2.4, which implies that the first iteration will
    //     shrink sp by at least 1 bit.
    // Thus, after the first iteration, sp will be at most 511 bits in all
    // cases. We can thus simply consider sp nonnegative systematically for
    // the first iteration.

    // Multiply two nonnegative 256-bit integers together, result over
    // 512 bits.
    fn umul_256(a: &[u32; 8], b: &[u32; 8]) -> [u32; 16] {
        let mut d = [0u32; 16];
        for i in 0..8 {
            let mut cc = 0u32;
            for j in 0..8 {
                let (lo, hi) = umull_add2(a[i], b[j], cc, d[i + j]);
                d[i + j] = lo;
                cc = hi;
            }
            d[i + 8] = cc;
        }
        d
    }

    // Return true iff a < b (inputs must be nonnegative).
    fn cmplt_512(a: &[u32; 16], b: &[u32; 16]) -> bool {
        for i in (0..16).rev() {
            if a[i] < b[i] {
                return true;
            } else if a[i] > b[i] {
                return false;
            }
        }
        false
    }

    // Get the bit length of a signed integer.
    fn bitlength_512(a: &[u32; 16]) -> u32 {
        let m = sgnw(a[15]);
        for i in (0..16).rev() {
            let aw = a[i] ^ m;
            if aw != 0 {
                return 32 * (i as u32) + 32 - aw.leading_zeros();
            }
        }
        0
    }

    // Add b*2^s to a.
    fn add_shifted_512(a: &mut [u32; 16], b: &[u32; 16], s: u32) {
        if s < 32 {
            if s == 0 {
                let (d0, mut cc) = addcarry_u32(a[0], b[0], 0);
                a[0] = d0;
                for i in 1..16 {
                    let (dx, ee) = addcarry_u32(a[i], b[i], cc);
                    a[i] = dx;
                    cc = ee;
                }
            } else {
                let (d0, mut cc) = addcarry_u32(a[0], b[0] << s, 0);
                a[0] = d0;
                for i in 1..16 {
                    let bw = (b[i - 1] >> (32 - s)) | (b[i] << s);
                    let (dx, ee) = addcarry_u32(a[i], bw, cc);
                    a[i] = dx;
                    cc = ee;
                }
            }
        } else {
            let j = (s >> 5) as usize;
            let s = s & 31;
            if s == 0 {
                let (dj, mut cc) = addcarry_u32(a[j], b[0], 0);
                a[j] = dj;
                for i in (j + 1)..16 {
                    let (dx, ee) = addcarry_u32(a[i], b[i - j], cc);
                    a[i] = dx;
                    cc = ee;
                }
            } else {
                let (dj, mut cc) = addcarry_u32(a[j], b[0] << s, 0);
                a[j] = dj;
                for i in (j + 1)..16 {
                    let bw = (b[i - j - 1] >> (32 - s)) | (b[i - j] << s);
                    let (dx, ee) = addcarry_u32(a[i], bw, cc);
                    a[i] = dx;
                    cc = ee;
                }
            }
        }
    }

    // Subtract b*2^s from a.
    fn sub_shifted_512(a: &mut [u32; 16], b: &[u32; 16], s: u32) {
        if s < 32 {
            if s == 0 {
                let (d0, mut cc) = subborrow_u32(a[0], b[0], 0);
                a[0] = d0;
                for i in 1..16 {
                    let (dx, ee) = subborrow_u32(a[i], b[i], cc);
                    a[i] = dx;
                    cc = ee;
                }
            } else {
                let (d0, mut cc) = subborrow_u32(a[0], b[0] << s, 0);
                a[0] = d0;
                for i in 1..16 {
                    let bw = (b[i - 1] >> (32 - s)) | (b[i] << s);
                    let (dx, ee) = subborrow_u32(a[i], bw, cc);
                    a[i] = dx;
                    cc = ee;
                }
            }
        } else {
            let j = (s >> 5) as usize;
            let s = s & 31;
            if s == 0 {
                let (dj, mut cc) = subborrow_u32(a[j], b[0], 0);
                a[j] = dj;
                for i in (j + 1)..16 {
                    let (dx, ee) = subborrow_u32(a[i], b[i - j], cc);
                    a[i] = dx;
                    cc = ee;
                }
            } else {
                let (dj, mut cc) = subborrow_u32(a[j], b[0] << s, 0);
                a[j] = dj;
                for i in (j + 1)..16 {
                    let bw = (b[i - j - 1] >> (32 - s)) | (b[i - j] << s);
                    let (dx, ee) = subborrow_u32(a[i], bw, cc);
                    a[i] = dx;
                    cc = ee;
                }
            }
        }
    }

    // Return true iff a < b (inputs must be nonnegative).
    fn cmplt_384(a: &[u32; 16], b: &[u32; 16]) -> bool {
        for i in (0..12).rev() {
            if a[i] < b[i] {
                return true;
            } else if a[i] > b[i] {
                return false;
            }
        }
        false
    }

    // Get the bit length of a signed integer.
    fn bitlength_384(a: &[u32; 16]) -> u32 {
        let m = sgnw(a[11]);
        for i in (0..12).rev() {
            let aw = a[i] ^ m;
            if aw != 0 {
                return 32 * (i as u32) + 32 - aw.leading_zeros();
            }
        }
        0
    }

    // Add b*2^s to a.
    fn add_shifted_384(a: &mut [u32; 16], b: &[u32; 16], s: u32) {
        if s < 32 {
            if s == 0 {
                let (d0, mut cc) = addcarry_u32(a[0], b[0], 0);
                a[0] = d0;
                for i in 1..12 {
                    let (dx, ee) = addcarry_u32(a[i], b[i], cc);
                    a[i] = dx;
                    cc = ee;
                }
            } else {
                let (d0, mut cc) = addcarry_u32(a[0], b[0] << s, 0);
                a[0] = d0;
                for i in 1..12 {
                    let bw = (b[i - 1] >> (32 - s)) | (b[i] << s);
                    let (dx, ee) = addcarry_u32(a[i], bw, cc);
                    a[i] = dx;
                    cc = ee;
                }
            }
        } else {
            let j = (s >> 5) as usize;
            let s = s & 31;
            if s == 0 {
                let (dj, mut cc) = addcarry_u32(a[j], b[0], 0);
                a[j] = dj;
                for i in (j + 1)..12 {
                    let (dx, ee) = addcarry_u32(a[i], b[i - j], cc);
                    a[i] = dx;
                    cc = ee;
                }
            } else {
                let (dj, mut cc) = addcarry_u32(a[j], b[0] << s, 0);
                a[j] = dj;
                for i in (j + 1)..12 {
                    let bw = (b[i - j - 1] >> (32 - s)) | (b[i - j] << s);
                    let (dx, ee) = addcarry_u32(a[i], bw, cc);
                    a[i] = dx;
                    cc = ee;
                }
            }
        }
    }

    // Subtract b*2^s from a.
    fn sub_shifted_384(a: &mut [u32; 16], b: &[u32; 16], s: u32) {
        if s < 32 {
            if s == 0 {
                let (d0, mut cc) = subborrow_u32(a[0], b[0], 0);
                a[0] = d0;
                for i in 1..12 {
                    let (dx, ee) = subborrow_u32(a[i], b[i], cc);
                    a[i] = dx;
                    cc = ee;
                }
            } else {
                let (d0, mut cc) = subborrow_u32(a[0], b[0] << s, 0);
                a[0] = d0;
                for i in 1..12 {
                    let bw = (b[i - 1] >> (32 - s)) | (b[i] << s);
                    let (dx, ee) = subborrow_u32(a[i], bw, cc);
                    a[i] = dx;
                    cc = ee;
                }
            }
        } else {
            let j = (s >> 5) as usize;
            let s = s & 31;
            if s == 0 {
                let (dj, mut cc) = subborrow_u32(a[j], b[0], 0);
                a[j] = dj;
                for i in (j + 1)..12 {
                    let (dx, ee) = subborrow_u32(a[i], b[i - j], cc);
                    a[i] = dx;
                    cc = ee;
                }
            } else {
                let (dj, mut cc) = subborrow_u32(a[j], b[0] << s, 0);
                a[j] = dj;
                for i in (j + 1)..12 {
                    let bw = (b[i - j - 1] >> (32 - s)) | (b[i - j] << s);
                    let (dx, ee) = subborrow_u32(a[i], bw, cc);
                    a[i] = dx;
                    cc = ee;
                }
            }
        }
    }

    // Initialize:
    //    u <- [n, 0]
    //    v <- [k, 1]
    //    nu = u0^2 + u1^2 = n^2
    //    nv = v0^2 + v1^2 = k^2 + 1
    //    sp = u0*v0 + u1*v1 = n*k
    let mut u0 = n[0] as u128;
    let mut v0 = k[0] as u128;
    for i in 1..4 {
        u0 += (n[i] as u128) << (32 * i);
        v0 += (k[i] as u128) << (32 * i);
    }
    let mut u0 = u0 as i128;
    let mut u1 = 0i128;
    let mut v0 = v0 as i128;
    let mut v1 = 1i128;
    let mut nu = umul_256(&n, &n);
    let mut nv = umul_256(&k, &k);
    let (dx, mut cc) = addcarry_u32(nv[0], 1, 0);
    nv[0] = dx;
    for i in 1..16 {
        let (dx, ee) = addcarry_u32(nv[i], 0, cc);
        nv[i] = dx;
        cc = ee;
    }
    let mut sp = umul_256(&n, &k);

    let mut first = true;

    // First algorithm loop, to shrink values below 383 bits.
    loop {
        // If u is smaller than v, then swap u and v.
        if cmplt_512(&nu, &nv) {
            let t = u0;
            u0 = v0;
            v0 = t;
            let t = u1;
            u1 = v1;
            v1 = t;
            for i in 0..16 {
                let t = nu[i];
                nu[i] = nv[i];
                nv[i] = t;
            }
        }

        // If nu has shrunk to 383 bits, then we can switch to the
        // second loop (since v is smaller than u at this point).
        if (nu[12] | nu[13] | nu[14] | nu[15]) == 0 && nu[11] < 0x80000000 {
            break;
        }

        // If v is small enough, return it.
        let bl_nv = bitlength_512(&nv);
        if bl_nv <= 254 {
            return (v0, v1);
        }

        // Compute this amount s = len(sp) - len(nv)
        // (if s < 0, it is replaced with 0).
        let bl_sp = bitlength_512(&sp);
        let mut s = bl_sp.wrapping_sub(bl_nv);
        s &= !(((s as i32) >> 31) as u32);

        // Subtract or add v*2^s from/to u, depending on the sign of sp.
        if first || sp[15] < 0x80000000 {
            first = false;
            if s < 128 {
                u0 = u0.wrapping_sub(v0 << s);
                u1 = u1.wrapping_sub(v1 << s);
            }
            add_shifted_512(&mut nu, &nv, 2 * s);
            sub_shifted_512(&mut nu, &sp, s + 1);
            sub_shifted_512(&mut sp, &nv, s);
        } else {
            if s < 128 {
                u0 = u0.wrapping_add(v0 << s);
                u1 = u1.wrapping_add(v1 << s);
            }
            add_shifted_512(&mut nu, &nv, 2 * s);
            add_shifted_512(&mut nu, &sp, s + 1);
            add_shifted_512(&mut sp, &nv, s);
        }
    }

    // In the secondary loop, we need to check for the end condition,
    // which can be a "stuck" value of sp.
    let mut last_bl_sp = bitlength_384(&sp);
    let mut stuck = 0u32;

    // Second algorithm loop, once values have shrunk to 383 bits.
    loop {
        // If u is smaller than v, then swap u and v.
        if cmplt_384(&nu, &nv) {
            let t = u0;
            u0 = v0;
            v0 = t;
            let t = u1;
            u1 = v1;
            v1 = t;
            for i in 0..12 {
                let t = nu[i];
                nu[i] = nv[i];
                nv[i] = t;
            }
        }

        // If v is small enough, return it.
        let bl_nv = bitlength_384(&nv);
        if bl_nv <= 254 {
            return (v0, v1);
        }

        // As long as the algorithm makes progress, sp never grows, and it
        // gets smaller by at least 1 bit every two iterations. If it appears
        // to be stuck at a length for two long or to grow, then this means
        // that the source modulus n was larger than Nmax, and the shortest
        // vector's L2 norm is at least 2^254, which implies that its
        // coordinates _might_ not fit in 128 bits each (thus, we return
        // potentially truncated results).
        let bl_sp = bitlength_384(&sp);
        if bl_sp >= last_bl_sp {
            stuck += 1;
            if bl_sp > last_bl_sp || stuck > 3 {
                return (v0, v1);
            }
        } else {
            last_bl_sp = bl_sp;
            stuck = 0;
        }

        // Compute this amount s = len(sp) - len(nv)
        // (if s < 0, it is replaced with 0).
        let mut s = bl_sp.wrapping_sub(bl_nv);
        s &= !(((s as i32) >> 31) as u32);

        // Subtract or add v*2^s from/to u, depending on the sign of sp.
        if sp[11] < 0x80000000 {
            if s < 128 {
                u0 = u0.wrapping_sub(v0 << s);
                u1 = u1.wrapping_sub(v1 << s);
            }
            add_shifted_384(&mut nu, &nv, 2 * s);
            sub_shifted_384(&mut nu, &sp, s + 1);
            sub_shifted_384(&mut sp, &nv, s);
        } else {
            if s < 128 {
                u0 = u0.wrapping_add(v0 << s);
                u1 = u1.wrapping_add(v1 << s);
            }
            add_shifted_384(&mut nu, &nv, 2 * s);
            add_shifted_384(&mut nu, &sp, s + 1);
            add_shifted_384(&mut sp, &nv, s);
        }
    }
}

// ========================================================================
// Tests for this file are merged into ModInt256 tests.
