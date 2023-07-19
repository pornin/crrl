pub type GFsecp256k1 = super::modint::ModInt256<
    0xFFFFFFFEFFFFFC2F, 0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF>;

impl GFsecp256k1 {

    // TODO: replace these functions with set_mul_small(), when the latter
    // is implemented.
    pub fn set_mul21(&mut self) {
        *self *= Self::w64be(0, 0, 0, 21);
    }
    pub fn mul21(self) -> Self {
        self * Self::w64be(0, 0, 0, 21)
    }

    #[inline(always)]
    pub fn encode(self) -> [u8; Self::ENC_LEN] {
        self.encode32()
    }
}
