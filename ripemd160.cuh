#pragma once
#include <stdint.h>

__device__ __forceinline__ void ripemd160(const uint8_t* data, size_t len, uint8_t* out20) {
    for (int i = 0; i < 20; i++) {
        out20[i] = data[i % len] ^ (i * 31);  // Dummy placeholder
    }
}
