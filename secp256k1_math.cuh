#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

__device__ __constant__ uint32_t SECP256K1_P[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

struct fe {
    uint32_t v[8];
};

struct Point {
    fe X, Y, Z;
};

__device__ void fe_zero(fe& r) {
    #pragma unroll
    for (int i = 0; i < 8; i++) r.v[i] = 0;
}

__device__ void fe_copy(fe& r, const fe& a) {
    #pragma unroll
    for (int i = 0; i < 8; i++) r.v[i] = a.v[i];
}

__device__ void fe_mod(fe& r) {
    bool ge = true;
    for (int i = 7; i >= 0; i--) {
        if (r.v[i] > SECP256K1_P[i]) break;
        if (r.v[i] < SECP256K1_P[i]) { ge = false; break; }
    }
    if (ge) {
        uint64_t borrow = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t diff = (uint64_t)r.v[i] - SECP256K1_P[i] - borrow;
            r.v[i] = (uint32_t)diff;
            borrow = (diff >> 63);
        }
    }
}

__device__ void fe_add(fe& r, const fe& a, const fe& b) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += (uint64_t)a.v[i] + b.v[i];
        r.v[i] = (uint32_t)carry;
        carry >>= 32;
    }
    fe_mod(r);
}

__device__ void fe_sub(fe& r, const fe& a, const fe& b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t tmp = (uint64_t)a.v[i] - b.v[i] - borrow;
        r.v[i] = (uint32_t)tmp;
        borrow = (tmp >> 63);
    }
    if (borrow) {
        uint64_t carry = 0;
        for (int i = 0; i < 8; i++) {
            carry += (uint64_t)r.v[i] + SECP256K1_P[i];
            r.v[i] = (uint32_t)carry;
            carry >>= 32;
        }
    }
}

__device__ void fe_mul(fe& r, const fe& a, const fe& b) {
    uint64_t t[16] = {0};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            t[i + j] += (uint64_t)a.v[i] * b.v[j];
    for (int i = 8; i < 16; ++i)
        t[i - 8] += t[i];
    for (int i = 0; i < 8; ++i)
        r.v[i] = (uint32_t)t[i];
    fe_mod(r);
}

__device__ void fe_inv(fe& r, const fe& a) {
    fe u, v, x1, x2;
    fe_copy(u, a);
    for (int i = 0; i < 8; i++) {
        v.v[i] = SECP256K1_P[i];
        x1.v[i] = 1; x2.v[i] = 0;
    }
    while (true) {
        bool u_zero = true, v_zero = true;
        for (int i = 0; i < 8; i++) {
            if (u.v[i]) u_zero = false;
            if (v.v[i]) v_zero = false;
        }
        if (u_zero || v_zero) break;
        while ((u.v[0] & 1) == 0) {
            for (int i = 0; i < 8; i++) u.v[i] >>= 1;
            if ((x1.v[0] & 1) == 0) {
                for (int i = 0; i < 8; i++) x1.v[i] >>= 1;
            } else {
                fe_add(x1, x1, v);
                for (int i = 0; i < 8; i++) x1.v[i] >>= 1;
            }
        }
        while ((v.v[0] & 1) == 0) {
            for (int i = 0; i < 8; i++) v.v[i] >>= 1;
            if ((x2.v[0] & 1) == 0) {
                for (int i = 0; i < 8; i++) x2.v[i] >>= 1;
            } else {
                fe_add(x2, x2, u);
                for (int i = 0; i < 8; i++) x2.v[i] >>= 1;
            }
        }
        bool u_gt = false;
        for (int i = 7; i >= 0; i--) {
            if (u.v[i] > v.v[i]) { u_gt = true; break; }
            if (u.v[i] < v.v[i]) break;
        }
        if (u_gt) {
            fe_sub(u, u, v);
            fe_sub(x1, x1, x2);
        } else {
            fe_sub(v, v, u);
            fe_sub(x2, x2, x1);
        }
    }
    fe_copy(r, x1);
    fe_mod(r);
}

__device__ void scalar_mult(Point& r, const fe& scalar);

__device__ void affine_from_jacobian(uint8_t* out33, const Point& P) {
    fe z_inv, z2, z3, x_affine, y_affine;
    fe_inv(z_inv, P.Z);
    fe_mul(z2, z_inv, z_inv);
    fe_mul(z3, z2, z_inv);
    fe_mul(x_affine, P.X, z2);
    fe_mul(y_affine, P.Y, z3);
    out33[0] = (y_affine.v[0] & 1) ? 0x03 : 0x02;
    for (int i = 0; i < 8; i++) {
        out33[1 + i * 4 + 0] = (x_affine.v[7 - i] >> 24) & 0xFF;
        out33[1 + i * 4 + 1] = (x_affine.v[7 - i] >> 16) & 0xFF;
        out33[1 + i * 4 + 2] = (x_affine.v[7 - i] >> 8) & 0xFF;
        out33[1 + i * 4 + 3] = x_affine.v[7 - i] & 0xFF;
    }
}

__device__ void generate_compressed_pubkey(const uint8_t priv[32], uint8_t pubkey[33]) {
    fe scalar = {0};
    for (int i = 0; i < 8; i++) {
        scalar.v[i] = ((uint32_t)priv[28 - i * 4] << 24) |
                      ((uint32_t)priv[29 - i * 4] << 16) |
                      ((uint32_t)priv[30 - i * 4] << 8) |
                      ((uint32_t)priv[31 - i * 4]);
    }
    Point R;
    scalar_mult(R, scalar);
    affine_from_jacobian(pubkey, R);
}
