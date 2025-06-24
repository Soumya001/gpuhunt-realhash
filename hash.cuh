#pragma once
#include <stdint.h>

__device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ void sha256(const uint8_t* msg, size_t len, uint8_t* out) {
    const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    uint8_t block[64] = {0};
    for (size_t i = 0; i < len && i < 64; i++) block[i] = msg[i];
    block[len] = 0x80;
    uint64_t bit_len = len * 8;
    block[63] = bit_len & 0xFF;
    block[62] = (bit_len >> 8) & 0xFF;
    block[61] = (bit_len >> 16) & 0xFF;
    block[60] = (bit_len >> 24) & 0xFF;
    block[59] = (bit_len >> 32) & 0xFF;
    block[58] = (bit_len >> 40) & 0xFF;
    block[57] = (bit_len >> 48) & 0xFF;
    block[56] = (bit_len >> 56) & 0xFF;

    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = (block[i * 4] << 24) | (block[i * 4 + 1] << 16) | (block[i * 4 + 2] << 8) | block[i * 4 + 3];
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], h_ = h[7];

    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h_ + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h_ = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += h_;

    for (int i = 0; i < 8; ++i) {
        out[i * 4]     = (h[i] >> 24) & 0xFF;
        out[i * 4 + 1] = (h[i] >> 16) & 0xFF;
        out[i * 4 + 2] = (h[i] >> 8) & 0xFF;
        out[i * 4 + 3] = h[i] & 0xFF;
    }
}

__device__ __forceinline__ uint32_t rol(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

__device__ void ripemd160(const uint8_t* msg, size_t len, uint8_t* out) {
    const uint32_t r[80] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
        7, 4,13, 1,10, 6,15, 3,12, 0, 9, 5, 2,14,11, 8,
        3,10,14, 4, 9,15, 8, 1, 2, 7, 0, 6,13,11, 5,12,
        1, 9,11,10, 0, 8,12, 4,13, 3, 7,15,14, 5, 6, 2,
        4, 0, 5, 9, 7,12, 2,10,14, 1, 3, 8,11, 6,15,13
    };

    const uint32_t s[80] = {
        11,14,15,12, 5, 8, 7, 9,11,13,14,15, 6, 7, 9, 8,
         7, 6, 8,13,11, 9, 7,15, 7,12,15, 9,11, 7,13,12,
        11,13, 6, 7,14, 9,13,15,14, 8,13, 6, 5,12, 7, 5,
        11,12,14,15,14,15, 9, 8, 9,14, 5, 6, 8, 6, 5,12,
         9,15, 5,11, 6, 8,13,12, 5,12,13,14,11, 8, 5, 6
    };

    const uint32_t K[5] = {
        0x00000000, 0x5A827999, 0x6ED9EBA1,
        0x8F1BBCDC, 0xA953FD4E
    };
    const uint32_t KK[5] = {
        0x50A28BE6, 0x5C4DD124, 0x6D703EF3,
        0x7A6D76E9, 0x00000000
    };

    uint32_t h[5] = {
        0x67452301, 0xEFCDAB89, 0x98BADCFE,
        0x10325476, 0xC3D2E1F0
    };

    uint8_t padded[64] = {0};
    for (int i = 0; i < len && i < 64; ++i) padded[i] = msg[i];
    padded[len] = 0x80;
    uint64_t bit_len = len * 8;
    padded[56] = bit_len & 0xFF;
    padded[57] = (bit_len >> 8) & 0xFF;
    padded[58] = (bit_len >> 16) & 0xFF;
    padded[59] = (bit_len >> 24) & 0xFF;
    padded[60] = (bit_len >> 32) & 0xFF;
    padded[61] = (bit_len >> 40) & 0xFF;
    padded[62] = (bit_len >> 48) & 0xFF;
    padded[63] = (bit_len >> 56) & 0xFF;

    uint32_t X[16];
    for (int i = 0; i < 16; ++i) {
        X[i] = (padded[i * 4]) | (padded[i * 4 + 1] << 8) |
               (padded[i * 4 + 2] << 16) | (padded[i * 4 + 3] << 24);
    }

    uint32_t A1 = h[0], B1 = h[1], C1 = h[2], D1 = h[3], E1 = h[4];
    uint32_t A2 = h[0], B2 = h[1], C2 = h[2], D2 = h[3], E2 = h[4];

    for (int j = 0; j < 80; j++) {
        int jj = j / 16;
        uint32_t F1 = 0, F2 = 0;
        if (jj == 0) { F1 = B1 ^ C1 ^ D1; F2 = (B2 & D2) | (C2 & ~D2); }
        else if (jj == 1) { F1 = (B1 & C1) | (~B1 & D1); F2 = (B2 | ~C2) ^ D2; }
        else if (jj == 2) { F1 = (B1 | ~C1) ^ D1; F2 = (B2 & C2) | (~B2 & D2); }
        else if (jj == 3) { F1 = (B1 & D1) | (C1 & ~D1); F2 = B2 ^ C2 ^ D2; }
        else { F1 = B1 ^ (C1 | ~D1); F2 = (B2 & C2) | (B2 & D2) | (C2 & D2); }

        uint32_t T = rol(A1 + F1 + X[r[j]] + K[jj], s[j]) + E1;
        A1 = E1; E1 = D1; D1 = rol(C1, 10); C1 = B1; B1 = T;

        T = rol(A2 + F2 + X[r[79 - j]] + KK[jj], s[j]) + E2;
        A2 = E2; E2 = D2; D2 = rol(C2, 10); C2 = B2; B2 = T;
    }

    uint32_t T = h[1] + C1 + D2;
    h[1] = h[2] + D1 + E2;
    h[2] = h[3] + E1 + A2;
    h[3] = h[4] + A1 + B2;
    h[4] = h[0] + B1 + C2;
    h[0] = T;

    for (int i = 0; i < 5; i++) {
        out[i * 4 + 0] = h[i] & 0xFF;
        out[i * 4 + 1] = (h[i] >> 8) & 0xFF;
        out[i * 4 + 2] = (h[i] >> 16) & 0xFF;
        out[i * 4 + 3] = (h[i] >> 24) & 0xFF;
    }
}
