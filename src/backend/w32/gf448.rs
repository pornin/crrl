// TODO: make a dedicated GF448 implementation, leveraging the special
// modulus format. For now, we use the generic code.

use super::gfgen::{define_gfgen, define_gfgen_tests};

struct GF448Params;
impl GF448Params {

    const MODULUS: [u64; 7] = [
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFEFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
    ];
}

define_gfgen!(GF448, GF448Params, gf448mod, false);
define_gfgen_tests!(GF448, 7, test_gf448mod);
