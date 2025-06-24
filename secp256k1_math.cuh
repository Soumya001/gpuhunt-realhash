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

// Field operations
__device__ void fe_zero(fe &r) {
    #pragma unroll
    for (int i = 0; i < 8; i++) r.v[i] = 0;
}

__device__ void fe_copy(fe &r, const fe &a) {
    #pragma unroll
    for (int i = 0; i < 8; i++) r.v[i] = a.v[i];
}

__device__ void fe_mod(fe &r) {
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

__device__ void fe_add(fe &r, const fe &a, const fe &b) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += (uint64_t)a.v[i] + b.v[i];
        r.v[i] = (uint32_t)carry;
        carry >>= 32;
    }
    fe_mod(r);
}

__device__ void fe_sub(fe &r, const fe &a, const fe &b) {
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

__device__ void fe_mul(fe &r, const fe &a, const fe &b) {
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

__device__ void fe_inv(fe &r, const fe &a) {
    fe u, v, x1, x2;
    fe_copy(u, a);
    for (int i = 0; i < 8; i++) {
        v.v[i] = SECP256K1_P[i];
        x1.v[i] = 1;
        x2.v[i] = 0;
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

// G (base point)
__device__ void get_G(Point &G) {
    G.X.v[0] = 0x59F2815B; G.X.v[1] = 0x16F81798; G.X.v[2] = 0x029BFCDB; G.X.v[3] = 0x2DCE28D9;
    G.X.v[4] = 0x029BFCDB; G.X.v[5] = 0x16F81798; G.X.v[6] = 0x59F2815B; G.X.v[7] = 0x79BE667E;

    G.Y.v[0] = 0x9C47D08F; G.Y.v[1] = 0xFB10D4B8; G.Y.v[2] = 0x3F482E3E; G.Y.v[3] = 0xE336A456;
    G.Y.v[4] = 0xA6325525; G.Y.v[5] = 0xBCE6FAAD; G.Y.v[6] = 0xC10E5C8C; G.Y.v[7] = 0x483ADA77;

    for (int i = 0; i < 8; i++) G.Z.v[i] = (i == 0) ? 1 : 0;
}

__device__ void point_double(Point &r, const Point &p) {
    fe S, M, T, U;

    fe_mul(S, p.X, p.X);           // S = X1^2
    fe_mul(M, S, fe{3});           // M = 3 * X1^2
    fe_mul(U, p.Y, p.Y);           // U = Y1^2
    fe_mul(U, U, p.X);             // U = X1 * Y1^2
    fe_add(U, U, U); fe_add(U, U, U); // U = 4 * X1 * Y1^2

    fe_mul(r.X, M, M);             // X3 = M^2
    fe_sub(r.X, r.X, U); fe_sub(r.X, r.X, U);

    fe_sub(T, U, r.X);
    fe_mul(T, T, M);
    fe_mul(U, p.Y, p.Y);
    fe_mul(U, U, U);
    fe_add(U, U, U);
    fe_add(U, U, U); // U = 8 * Y1^4

    fe_sub(r.Y, T, U);
    fe_mul(r.Z, p.Y, p.Z);
    fe_add(r.Z, r.Z, r.Z); // Z3 = 2 * Y1 * Z1
}

__device__ void point_add(Point &r, const Point &p, const Point &q) {
    if (q.Z.v[0] == 0 && q.Z.v[1] == 0) {
        fe_copy(r.X, p.X);
        fe_copy(r.Y, p.Y);
        fe_copy(r.Z, p.Z);
        return;
    }

    fe Z1Z1, Z2Z2, U1, U2, S1, S2, H, HH, I, J, r_, V;

    fe_mul(Z1Z1, p.Z, p.Z);
    fe_mul(Z2Z2, q.Z, q.Z);

    fe_mul(U1, p.X, Z2Z2);
    fe_mul(U2, q.X, Z1Z1);

    fe_mul(S1, p.Y, q.Z);
    fe_mul(S1, S1, Z2Z2);

    fe_mul(S2, q.Y, p.Z);
    fe_mul(S2, S2, Z1Z1);

    fe_sub(H, U2, U1);
    fe_add(I, H, H); fe_mul(I, I, I);

    fe_mul(J, H, I);
    fe_sub(r_, S2, S1); fe_add(r_, r_, r_);

    fe_mul(V, U1, I);

    fe_mul(r.X, r_, r_); fe_sub(r.X, r.X, J); fe_sub(r.X, r.X, V); fe_sub(r.X, r.X, V);

    fe_sub(r.Y, V, r.X); fe_mul(r.Y, r.Y, r_); fe_mul(S1, S1, J); fe_sub(r.Y, r.Y, S1); fe_sub(r.Y, r.Y, S1);

    fe_add(r.Z, p.Z, q.Z); fe_mul(r.Z, r.Z, r.Z); fe_sub(r.Z, r.Z, Z1Z1); fe_sub(r.Z, r.Z, Z2Z2);
    fe_mul(r.Z, r.Z, H);
}

// Scalar multiplication (naive double-and-add)
__device__ void scalar_mult(Point &r, const fe &scalar) {
    Point Q;
    get_G(Q);
    Point R;
    fe_zero(R.X); fe_zero(R.Y); fe_zero(R.Z); R.Z.v[0] = 0;

    for (int i = 255; i >= 0; i--) {
        point_double(R, R);
        int limb = (scalar.v[i / 32] >> (i % 32)) & 1;
        if (limb) point_add(R, R, Q);
    }

    fe_copy(r.X, R.X);
    fe_copy(r.Y, R.Y);
    fe_copy(r.Z, R.Z);
}

__device__ void affine_from_jacobian(uint8_t* out33, const Point &P) {
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
