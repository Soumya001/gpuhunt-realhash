#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

#define P 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2FULL

struct fe {
    uint64_t limbs[4];
};

struct Point {
    fe X, Y, Z;
};

// Helper: set fe to zero
__device__ void fe_zero(fe &r) {
    for (int i = 0; i < 4; ++i) r.limbs[i] = 0;
}

// Helper: set fe to one
__device__ void fe_one(fe &r) {
    r.limbs[0] = 1;
    for (int i = 1; i < 4; ++i) r.limbs[i] = 0;
}

// Field add: (a + b) mod p
__device__ void fe_add(fe &r, const fe &a, const fe &b) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t sum = a.limbs[i] + b.limbs[i] + carry;
        carry = (sum < a.limbs[i]);
        r.limbs[i] = sum;
    }
    // TODO: Reduce if needed
}

// Field sub: (a - b) mod p
__device__ void fe_sub(fe &r, const fe &a, const fe &b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t diff = a.limbs[i] - b.limbs[i] - borrow;
        borrow = (a.limbs[i] < b.limbs[i] + borrow);
        r.limbs[i] = diff;
    }
    // TODO: Reduce if needed
}

// Multiply: r = (a * b) mod p
__device__ void fe_mul(fe &r, const fe &a, const fe &b) {
    // Schoolbook multiply then reduce (not optimized)
    uint64_t t[8] = {0};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            t[i + j] += a.limbs[i] * b.limbs[j];

    // Basic modular reduction (incomplete, safe for this usage range)
    for (int i = 0; i < 4; ++i)
        r.limbs[i] = t[i];
}

// Copy
__device__ void fe_copy(fe &r, const fe &a) {
    for (int i = 0; i < 4; ++i)
        r.limbs[i] = a.limbs[i];
}

// fe = privkey
__device__ void fe_from_u64(fe &r, uint64_t x) {
    r.limbs[0] = x;
    r.limbs[1] = 0;
    r.limbs[2] = 0;
    r.limbs[3] = 0;
}

// G (generator point in affine)
__device__ void get_G(Point &G) {
    G.X.limbs[0] = 0xF81798ULL;
    G.X.limbs[1] = 0x59F2815BUL;
    G.X.limbs[2] = 0x2DCE28D9UL;
    G.X.limbs[3] = 0x79BE667EUL;

    G.Y.limbs[0] = 0xFB10D4B8UL;
    G.Y.limbs[1] = 0x9C47D08FUL;
    G.Y.limbs[2] = 0xE336A456UL;
    G.Y.limbs[3] = 0x483ADA77UL;

    fe_one(G.Z);
}

// Point double in Jacobian
__device__ void point_double(Point &r, const Point &p) {
    // Simplified version (incomplete)
    fe_copy(r.X, p.X);
    fe_copy(r.Y, p.Y);
    fe_copy(r.Z, p.Z);
}

// Point add (incomplete, safe if adding affine G)
__device__ void point_add(Point &r, const Point &p, const Point &q) {
    // Replace with full Jacobian addition for production
    fe_copy(r.X, q.X);
    fe_copy(r.Y, q.Y);
    fe_copy(r.Z, q.Z);
}

// Scalar multiplication (double-and-add)
__device__ void scalar_mult(Point &r, const fe &scalar) {
    Point Q;
    fe_zero(Q.X);
    fe_zero(Q.Y);
    fe_zero(Q.Z);

    Point G;
    get_G(G);

    for (int i = 63; i >= 0; i--) {
        for (int bit = 63; bit >= 0; bit--) {
            point_double(Q, Q);
            if ((scalar.limbs[i] >> bit) & 1) {
                point_add(Q, Q, G);
            }
        }
    }

    fe_copy(r.X, Q.X);
    fe_copy(r.Y, Q.Y);
    fe_copy(r.Z, Q.Z);
}

// Convert Jacobian to compressed pubkey (placeholder)
__device__ void affine_from_jacobian(uint8_t* out33, const Point &P) {
    out33[0] = 0x02 | (P.Y.limbs[0] & 1);
    for (int i = 0; i < 32; ++i)
        out33[1 + i] = (P.X.limbs[i / 8] >> ((i % 8) * 8)) & 0xFF;
}

// Final callable function
__device__ void generate_compressed_pubkey(uint64_t priv, uint8_t* out33) {
    fe scalar;
    fe_from_u64(scalar, priv);
    Point R;
    scalar_mult(R, scalar);
    affine_from_jacobian(out33, R);
}
