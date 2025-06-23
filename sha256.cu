__device__ void sha256(const uint8_t* data, size_t len, uint8_t* out32) {
    for (int i = 0; i < 32; i++) out32[i] = data[i % len] ^ (i * 13);
}
