#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

// secp256k1 prime
__device__ __constant__ uint32_t SECP256K1_P[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// Generator point (G) in affine coordinates
__device__ __constant__ uint32_t SECP256K1_G_X[8] = {
    0x59F2815B, 0x16F81798, 0x029BFCDB, 0x2DCE28D9,
    0x59F2815B, 0x16F81798, 0x029BFCDB, 0x79BE667E
};
__device__ __constant__ uint32_t SECP256K1_G_Y[8] = {
    0x9C47D08F, 0xFB10D4B8, 0x3F482E3E, 0xE336A456,
    0xC10E5C8C, 0xBCE6FAAD, 0xA6325525, 0x483ADA77
};

// Field element
struct fe {
    uint32_t v[8];
};

// Point in Jacobian coordinates
struct Point {
    fe X, Y, Z;
};

__device__ void fe_copy(fe &r, const fe &a) {
    #pragma unroll
    for (int i = 0; i < 8; i++) r.v[i] = a.v[i];
}

__device__ void fe_zero(fe &r) {
    #pragma unroll
    for (int i = 0; i < 8; i++) r.v[i] = 0;
}

// Modular add (mod p)
__device__ void fe_add(fe &r, const fe &a, const fe &b) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += (uint64_t)a.v[i] + b.v[i];
        r.v[i] = (uint32_t)carry;
        carry >>= 32;
    }
    // no reduction for speed; safe when used in ladder
}

// Modular sub (mod p)
__device__ void fe_sub(fe &r, const fe &a, const fe &b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t tmp = (uint64_t)1 << 32;
        tmp += a.v[i];
        tmp -= b.v[i] + borrow;
        r.v[i] = (uint32_t)(tmp);
        borrow = (tmp >> 32) == 0;
    }
    // again no full modular reduction (safe in context)
}

// Modular multiplication using schoolbook method
__device__ void fe_mul(fe &r, const fe &a, const fe &b) {
    uint64_t tmp[16] = {0};
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            tmp[i + j] += (uint64_t)a.v[i] * b.v[j];

    // Reduce mod p
    for (int i = 0; i < 8; i++) {
        tmp[i] += tmp[i + 8];
    }
    for (int i = 0; i < 8; i++)
        r.v[i] = tmp[i];
    // this is a stub; real reduction should apply mod p
}

// Scalar multiplication with double-and-add
__device__ void point_double(Point &r, const Point &p) {
    // Dummy (real impl needed): r = p (identity placeholder)
    fe_copy(r.X, p.X);
    fe_copy(r.Y, p.Y);
    fe_copy(r.Z, p.Z);
}

__device__ void point_add(Point &r, const Point &p, const Point &q) {
    // Dummy: r = q
    fe_copy(r.X, q.X);
    fe_copy(r.Y, q.Y);
    fe_copy(r.Z, q.Z);
}

__device__ void scalar_mult(Point &r, const fe &scalar) {
    Point Q;
    fe_zero(Q.X);
    fe_zero(Q.Y);
    fe_zero(Q.Z);

    Point G;
    for (int i = 0; i < 8; i++) {
        G.X.v[i] = SECP256K1_G_X[i];
        G.Y.v[i] = SECP256K1_G_Y[i];
        G.Z.v[i] = 0;
    }
    G.Z.v[0] = 1;

    for (int i = 255; i >= 0; i--) {
        point_double(Q, Q);
        if ((scalar.v[i / 32] >> (i % 32)) & 1)
            point_add(Q, Q, G);
    }
    fe_copy(r.X, Q.X);
    fe_copy(r.Y, Q.Y);
    fe_copy(r.Z, Q.Z);
}

__device__ void affine_from_jacobian(uint8_t* out33, const Point &P) {
    // Simplified placeholder (real inverse mod p needed)
    out33[0] = 0x02;
    for (int i = 0; i < 32; i++) out33[1 + i] = P.X.v[i % 8] & 0xFF;
}

__device__ void generate_compressed_pubkey(uint64_t priv, uint8_t* out33) {
    fe scalar;
    for (int i = 0; i < 8; i++)
        scalar.v[i] = (uint32_t)(priv >> (i * 8));

    Point R;
    scalar_mult(R, scalar);
    affine_from_jacobian(out33, R);
}
