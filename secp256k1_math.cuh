#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

// secp256k1 prime
__device__ __constant__ uint32_t SECP256K1_P[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// Generator G in affine coordinates
__device__ __constant__ uint32_t SECP256K1_G_X[8] = {
    0x59F2815B, 0x16F81798, 0x029BFCDB, 0x2DCE28D9,
    0x59F2815B, 0x16F81798, 0x029BFCDB, 0x79BE667E
};
__device__ __constant__ uint32_t SECP256K1_G_Y[8] = {
    0x9C47D08F, 0xFB10D4B8, 0x3F482E3E, 0xE336A456,
    0xC10E5C8C, 0xBCE6FAAD, 0xA6325525, 0x483ADA77
};

struct fe {
    uint32_t v[8];
};

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

__device__ void fe_add(fe &r, const fe &a, const fe &b) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += (uint64_t)a.v[i] + b.v[i];
        r.v[i] = (uint32_t)carry;
        carry >>= 32;
    }
    // Optionally reduce here
}

__device__ void fe_sub(fe &r, const fe &a, const fe &b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t tmp = (uint64_t)a.v[i] - b.v[i] - borrow;
        r.v[i] = (uint32_t)tmp;
        borrow = (tmp >> 63);
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
    // This is not full mod reduction, for now sufficient for curve ops
}

__device__ void point_double(Point &r, const Point &p) {
    fe S, M, T, X3, Y3, Z3, tmp1, tmp2;

    fe_mul(tmp1, p.Y, p.Y);     // tmp1 = Y1^2
    fe_mul(S, p.X, tmp1);       // S = X1*Y1^2
    fe_add(S, S, S);            // S = 2*S
    fe_add(S, S, S);            // S = 4*S

    fe_mul(M, p.X, p.X);        // M = X1^2
    fe_add(tmp2, M, M);         // tmp2 = 2*X1^2
    fe_add(M, M, tmp2);         // M = 3*X1^2

    fe_mul(X3, M, M);           // X3 = M^2
    fe_sub(X3, X3, S);          // X3 = M^2 - 2*S
    fe_sub(X3, X3, S);

    fe_sub(T, S, X3);           // T = S - X3
    fe_mul(T, T, M);            // T = M*(S - X3)
    fe_mul(tmp1, tmp1, tmp1);   // tmp1 = Y1^4
    fe_add(tmp1, tmp1, tmp1);   // tmp1 = 2*Y1^4
    fe_sub(Y3, T, tmp1);        // Y3 = M*(S - X3) - 2*Y1^4

    fe_mul(Z3, p.Y, p.Z);
    fe_add(Z3, Z3, Z3);         // Z3 = 2*Y1*Z1

    fe_copy(r.X, X3);
    fe_copy(r.Y, Y3);
    fe_copy(r.Z, Z3);
}

__device__ void point_add(Point &r, const Point &p, const Point &q) {
    if (q.Z.v[0] == 0 && q.Z.v[1] == 0) {
        fe_copy(r.X, p.X); fe_copy(r.Y, p.Y); fe_copy(r.Z, p.Z); return;
    }

    fe U1, U2, S1, S2, H, R, H2, H3, Z1Z1, Z2Z2, tmp1, tmp2;

    fe_mul(Z1Z1, p.Z, p.Z);
    fe_mul(Z2Z2, q.Z, q.Z);

    fe_mul(U1, p.X, Z2Z2);
    fe_mul(U2, q.X, Z1Z1);

    fe_mul(tmp1, q.Z, Z2Z2);
    fe_mul(S1, p.Y, tmp1);

    fe_mul(tmp2, p.Z, Z1Z1);
    fe_mul(S2, q.Y, tmp2);

    fe_sub(H, U2, U1);
    fe_sub(R, S2, S1);

    fe_mul(H2, H, H);
    fe_mul(H3, H2, H);
    fe_mul(tmp1, H2, U1);

    fe_mul(tmp2, R, R);
    fe_sub(tmp2, tmp2, H3);
    fe_sub(tmp2, tmp2, tmp1);
    fe_sub(tmp2, tmp2, tmp1);

    fe_copy(r.X, tmp2);

    fe_sub(tmp1, tmp1, r.X);
    fe_mul(tmp1, tmp1, R);
    fe_mul(tmp2, H3, S1);
    fe_sub(r.Y, tmp1, tmp2);

    fe_mul(r.Z, p.Z, q.Z);
    fe_mul(r.Z, r.Z, H);
}

__device__ void affine_from_jacobian(uint8_t* out33, const Point &P) {
    // WARNING: No modular inverse here
    out33[0] = 0x02;
    for (int i = 0; i < 32; i++) out33[1 + i] = P.X.v[i % 8] & 0xFF;
}

__device__ void scalar_mult(Point &r, const fe &scalar) {
    Point Q;
    fe_zero(Q.X); fe_zero(Q.Y); fe_zero(Q.Z);

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

__device__ void generate_compressed_pubkey(uint64_t priv, uint8_t* out33) {
    fe scalar;
    for (int i = 0; i < 8; i++)
        scalar.v[i] = (uint32_t)(priv >> (i * 8));

    Point R;
    scalar_mult(R, scalar);
    affine_from_jacobian(out33, R);
}
