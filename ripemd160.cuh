#pragma once
#include <stdint.h>

__device__ __forceinline__ uint32_t rol(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

__device__ __forceinline__ void ripemd160(const uint8_t* msg, size_t len, uint8_t* out) {
    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xefcdab89;
    uint32_t h2 = 0x98badcfe;
    uint32_t h3 = 0x10325476;
    uint32_t h4 = 0xc3d2e1f0;

    uint8_t padded[64] = {0};
    for (int i = 0; i < len && i < 64; ++i)
        padded[i] = msg[i];
    padded[len] = 0x80;
    uint64_t bit_len = len * 8;
    padded[56] = bit_len & 0xFF;
    padded[57] = (bit_len >> 8) & 0xFF;
    padded[58] = (bit_len >> 16) & 0xFF;
    padded[59] = (bit_len >> 24) & 0xFF;

    uint32_t X[16];
    for (int i = 0; i < 16; ++i) {
        X[i] = (uint32_t)padded[4 * i] |
               ((uint32_t)padded[4 * i + 1] << 8) |
               ((uint32_t)padded[4 * i + 2] << 16) |
               ((uint32_t)padded[4 * i + 3] << 24);
    }

    uint32_t A1 = h0, B1 = h1, C1 = h2, D1 = h3, E1 = h4;
    uint32_t A2 = h0, B2 = h1, C2 = h2, D2 = h3, E2 = h4;

    const uint32_t r1[80] = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
        10, 11, 12, 13, 14, 15,  7,  4, 13,  1,
        10,  6, 15,  3, 12,  0,  9,  5,  2, 14,
        11,  8,  3, 10, 14,  4,  9, 15,  8,  1,
         2,  7,  0,  6, 13, 11,  5, 12,  1,  9,
        11, 10,  0,  8, 12,  4, 13,  3,  7, 15,
        14,  5,  6,  2,  4,  0,  5,  9,  7, 12,
         2, 10, 14,  1,  3,  8, 11,  6, 15, 13
    };

    const uint32_t r2[80] = {
         5, 14,  7,  0,  9,  2, 11,  4, 13,  6,
        15,  8,  1, 10,  3, 12,  6, 11,  3,  7,
         0, 13,  5, 10, 14, 15,  8, 12,  4,  9,
         1,  2, 15,  5,  1,  3,  7, 14,  6,  9,
        11,  8, 12,  2, 10,  0,  4, 13,  8,  6,
         4,  1,  3, 11, 15,  0,  5, 12,  2, 13,
         9,  7, 10, 14, 12, 15, 10,  4,  1,  5,
         8,  7,  6,  2, 13, 14,  0,  3,  9, 11
    };

    const uint32_t s1[80] = {
        11, 14, 15, 12,  5,  8,  7,  9, 11, 13,
        14, 15,  6,  7,  9,  8,  7,  6,  8, 13,
        11,  9,  7, 15,  7, 12, 15,  9, 11,  7,
        13, 12, 11, 13,  6,  7, 14,  9, 13, 15,
         7, 12, 15,  9, 11,  7, 13, 12, 11, 13,
         6,  7, 14,  9, 13, 15,  7, 12, 15,  9,
        11,  7, 13, 12, 11, 13,  6,  7, 14,  9,
        13, 15,  7, 12, 15,  9, 11,  7, 13, 12
    };

    const uint32_t s2[80] = {
         8,  9,  9, 11, 13, 15, 15,  5,  7,  7,
         8, 11, 14, 14, 12,  6,  9, 13, 15,  7,
        12,  8,  9, 11,  7,  7, 12,  7,  6, 15,
        13, 11,  9,  7, 15, 11,  8,  6,  6, 14,
        12, 13,  5, 14, 13, 13,  7,  5, 15,  5,
         8, 11, 14, 14,  6, 14,  6,  9, 12,  9,
        12,  5, 15,  8,  8,  5, 12,  9, 12,  5,
        14,  6,  8, 13,  6,  5, 15, 13, 11, 11
    };

    const uint32_t K1[5] = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
    const uint32_t K2[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};

    for (int i = 0; i < 80; ++i) {
        uint32_t f, k;
        if (i < 16)      { f = B1 ^ C1 ^ D1; k = K1[0]; }
        else if (i < 32) { f = (B1 & C1) | (~B1 & D1); k = K1[1]; }
        else if (i < 48) { f = (B1 | ~C1) ^ D1; k = K1[2]; }
        else if (i < 64) { f = (B1 & D1) | (C1 & ~D1); k = K1[3]; }
        else             { f = B1 ^ (C1 | ~D1); k = K1[4]; }

        uint32_t T = rol(A1 + f + X[r1[i]] + k, s1[i]) + E1;
        A1 = E1; E1 = D1; D1 = rol(C1, 10); C1 = B1; B1 = T;

        if (i < 16)      { f = (B2 & D2) | (C2 & ~D2); k = K2[0]; }
        else if (i < 32) { f = (B2 | ~C2) ^ D2; k = K2[1]; }
        else if (i < 48) { f = (B2 & C2) | (~B2 & D2); k = K2[2]; }
        else if (i < 64) { f = B2 ^ C2 ^ D2; k = K2[3]; }
        else             { f = B2 ^ (C2 | ~D2); k = K2[4]; }

        T = rol(A2 + f + X[r2[i]] + k, s2[i]) + E2;
        A2 = E2; E2 = D2; D2 = rol(C2, 10); C2 = B2; B2 = T;
    }

    uint32_t T = h1 + C1 + D2;
    h1 = h2 + D1 + E2;
    h2 = h3 + E1 + A2;
    h3 = h4 + A1 + B2;
    h4 = h0 + B1 + C2;
    h0 = T;

    out[0] = h0 & 0xFF;     out[1] = (h0 >> 8) & 0xFF;
    out[2] = (h0 >> 16) & 0xFF; out[3] = (h0 >> 24) & 0xFF;
    out[4] = h1 & 0xFF;     out[5] = (h1 >> 8) & 0xFF;
    out[6] = (h1 >> 16) & 0xFF; out[7] = (h1 >> 24) & 0xFF;
    out[8] = h2 & 0xFF;     out[9] = (h2 >> 8) & 0xFF;
    out[10] = (h2 >> 16) & 0xFF; out[11] = (h2 >> 24) & 0xFF;
    out[12] = h3 & 0xFF;    out[13] = (h3 >> 8) & 0xFF;
    out[14] = (h3 >> 16) & 0xFF; out[15] = (h3 >> 24) & 0xFF;
    out[16] = h4 & 0xFF;    out[17] = (h4 >> 8) & 0xFF;
    out[18] = (h4 >> 16) & 0xFF; out[19] = (h4 >> 24) & 0xFF;
}
