#! /usr/bin/env sage

# This Sage script computes and prints the UX_COMP[] array of precomputed
# values that are used to support efficient verification of truncated
# Ed25519 signatures.
#
# We work in the Montgomery domain of Curve25519: y^2 = x^3 + 486662*x^2 + x
# B = conventional generator
# For i = 0 to 16384, let U_i = i*(2^240)*B, and x_i = x coordinate of U_i.
# We consider x_i as an integer (with 0 <= x_i < 2^255-19) and define:
#   z_i = (x_i % 2^48)*2^16 + i  (as an unsigned 64-bit integer)
# The produced UX_COMP[] array contains the 16385 values z_i in ascending
# numerical order.

import importlib
import hashlib

def mkuxcomp():
    p = 2**255 - 19
    K = Zmod(p)
    E = EllipticCurve(K, [0, 486662, 0, 1, 0])
    B = E.point([9, 14781619447589544791020593568409986887264606134616475288964881837755586237401])
    tt = []
    T = E.point([0, 1, 0])
    P = (2**240)*B
    for i in range(0, 16385):
        if T.is_zero():
            x = K(0)
        else:
            x = T.xy()[0]
        tt.append(int(i) + ((int(x) % 2**48) << 16))
        T = T + P
    tt.sort()
    print('static UX_COMP: [u64; 16385] = [', end='')
    for i in range(0, len(tt)):
        if (i % 3) == 0:
            print()
            print('    ', end='')
        else:
            print(' ', end='')
        print('0x%016X,' % int(tt[i]), end='')
    print()
    print('];')

mkuxcomp()
