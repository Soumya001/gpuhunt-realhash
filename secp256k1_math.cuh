#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

__device__ __constant__ uint32_t SECP256K1_P[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

struct fe { uint32_t v[8]; };
struct Point { fe X, Y, Z; };

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
        x1.v[i] = (i == 0); x2.v[i] = 0;
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

__device__ void get_G(Point &P) {
    const uint32_t x[8] = {0x59F2815B, 0x16F81798, 0x483ADA77, 0xC10E5BF8,
                           0x334C74C7, 0x6B17D1F2, 0x7789C1E5, 0x79BE667E};
    const uint32_t y[8] = {0x9C47D08F, 0xFB10D4B8, 0xFD17B448, 0xA6855419,
                           0x5DA4FBFC, 0x59F2815B, 0x9C47D08F, 0x483ADA77};
    for (int i = 0; i < 8; i++) {
        P.X.v[i] = x[i];
        P.Y.v[i] = y[i];
        P.Z.v[i] = (i == 0); // Z = 1
    }
}

__device__ void point_double(Point &r, const Point &a) {
    fe S, M, T, tmp;
    fe_mul(S, a.X, a.X);
    fe_add(M, S, S);
    fe_add(M, M, S);
    fe_mul(T, a.Y, a.Y);
    fe_mul(tmp, T, a.X);
    fe_add(tmp, tmp, tmp);
    fe_add(tmp, tmp, tmp);
    fe_mul(r.X, M, M);
    fe_sub(r.X, r.X, tmp);
    fe_sub(r.X, r.X, tmp);
    fe_add(r.Y, tmp, tmp);
    fe_sub(r.Y, r.Y, r.X);
    fe_mul(r.Y, r.Y, M);
    fe_mul(tmp, T, T);
    fe_sub(r.Y, r.Y, tmp);
    fe_mul(r.Z, a.Y, a.Z);
    fe_add(r.Z, r.Z, r.Z);
}

__device__ void point_add(Point &r, const Point &a, const Point &b) {
    fe Z1Z1, Z2Z2, U1, U2, S1, S2, H, I, J, r_, V, tmp;
    fe_mul(Z1Z1, a.Z, a.Z);
    fe_mul(Z2Z2, b.Z, b.Z);
    fe_mul(U1, a.X, Z2Z2);
    fe_mul(U2, b.X, Z1Z1);
    fe_mul(S1, a.Y, b.Z); fe_mul(S1, S1, Z2Z2);
    fe_mul(S2, b.Y, a.Z); fe_mul(S2, S2, Z1Z1);
    fe_sub(H, U2, U1);
    fe_add(I, H, H); fe_mul(I, I, I);
    fe_mul(J, H, I);
    fe_sub(r_, S2, S1); fe_add(r_, r_, r_);
    fe_mul(V, U1, I);
    fe_mul(r.X, r_, r_); fe_sub(r.X, r.X, J); fe_sub(r.X, r.X, V); fe_sub(r.X, r.X, V);
    fe_sub(r.Y, V, r.X); fe_mul(r.Y, r.Y, r_); 
    fe_mul(tmp, S1, J); fe_add(tmp, tmp, tmp); fe_sub(r.Y, r.Y, tmp);
    fe_add(r.Z, a.Z, b.Z); fe_mul(r.Z, r.Z, r.Z); 
    fe_sub(r.Z, r.Z, Z1Z1); fe_sub(r.Z, r.Z, Z2Z2); fe_mul(r.Z, r.Z, H);
}

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

__device__ void scalar_mult(Point &r, const fe &scalar) {
    Point Q; get_G(Q);
    fe_zero(r.X); fe_zero(r.Y); fe_zero(r.Z);
    for (int i = 0; i < 8; i++) r.Z.v[i] = 0;
    for (int i = 255; i >= 0; i--) {
        point_double(r, r);
        int bit = (scalar.v[i / 32] >> (i % 32)) & 1;
        if (bit) point_add(r, r, Q);
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
    Point P;
    scalar_mult(P, scalar);
    affine_from_jacobian(pubkey, P);
}
