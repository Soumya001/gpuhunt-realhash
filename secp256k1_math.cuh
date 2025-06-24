#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

__device__ __constant__ uint32_t SECP256K1_P[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

__device__ __constant__ uint32_t SECP256K1_G_X[8] = {
    0x59F2815B, 0x16F81798, 0x029BFCDB, 0x2DCE28D9,
    0x029BFCDB, 0x16F81798, 0x59F2815B, 0x79BE667E
};
__device__ __constant__ uint32_t SECP256K1_G_Y[8] = {
    0x9C47D08F, 0xFB10D4B8, 0x3F482E3E, 0xE336A456,
    0xA6325525, 0xBCE6FAAD, 0xC10E5C8C, 0x483ADA77
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

__device__ void point_from_G(Point& P) {
    for (int i = 0; i < 8; i++) {
        P.X.v[i] = SECP256K1_G_X[i];
        P.Y.v[i] = SECP256K1_G_Y[i];
        P.Z.v[i] = (i == 0) ? 1 : 0;
    }
}

__device__ void point_double(Point& r, const Point& p) {
    fe t0, t1, t2, t3, t4;

    fe_mul(t0, p.Z, p.Z);             // Z^2
    fe_sub(t1, p.X, t0);              // X - Z^2
    fe_add(t2, p.X, t0);              // X + Z^2
    fe_mul(t3, t1, t2);               // (X - Z^2)(X + Z^2)
    fe_mul(t0, p.Y, p.Y);             // Y^2
    fe_mul(t1, p.Y, p.Z);             // Y*Z
    fe_add(t1, t1, t1);               // 2*Y*Z
    fe_add(t2, p.Y, p.Y);             // 2*Y
    fe_mul(t2, t2, t2);               // 4Y^2
    fe_mul(t2, t2, p.X);              // 4Y^2*X

    fe_add(t4, t0, t0);               // 2Y^2
    fe_add(t4, t4, t0);               // 3Y^2

    fe_add(r.X, t3, t3);              // 2*(...)
    fe_add(r.X, r.X, r.X);            // 4*(...)

    fe_sub(r.Y, t2, r.X);             // new Y
    fe_mul(r.Y, r.Y, t4);             // *3Y^2
    fe_sub(r.Y, r.Y, t2);             // -

    fe_mul(r.Z, t1, t1);              // Z3 = (2*Y*Z)^2
    fe_mul(r.X, t4, t4);              // X3 = (3Y^2)^2
    fe_sub(r.X, r.X, r.X);            // X3 - 2*X3 = 0 (placeholder)
}

__device__ void point_add(Point& r, const Point& p, const Point& q) {
    if (q.Z.v[0] == 0 && q.Z.v[1] == 0) {
        fe_copy(r.X, p.X);
        fe_copy(r.Y, p.Y);
        fe_copy(r.Z, p.Z);
        return;
    }

    fe z1z1, z2z2, u1, u2, s1, s2, h, i, j, r2, v;

    fe_mul(z1z1, p.Z, p.Z);
    fe_mul(z2z2, q.Z, q.Z);

    fe_mul(u1, p.X, z2z2);
    fe_mul(u2, q.X, z1z1);

    fe_mul(s1, p.Y, q.Z);
    fe_mul(s1, s1, z2z2);

    fe_mul(s2, q.Y, p.Z);
    fe_mul(s2, s2, z1z1);

    fe_sub(h, u2, u1);
    fe_sub(r2, s2, s1);

    fe_add(i, h, h);
    fe_mul(i, i, i);

    fe_mul(j, h, i);

    fe_mul(v, u1, i);

    fe_mul(r.X, r2, r2);
    fe_sub(r.X, r.X, j);
    fe_sub(r.X, r.X, v);
    fe_sub(r.X, r.X, v);

    fe_sub(r.Y, v, r.X);
    fe_mul(r.Y, r.Y, r2);
    fe_mul(s1, s1, j);
    fe_sub(r.Y, r.Y, s1);

    fe_add(r.Z, p.Z, q.Z);
    fe_mul(r.Z, r.Z, r.Z);
    fe_sub(r.Z, r.Z, z1z1);
    fe_sub(r.Z, r.Z, z2z2);
    fe_mul(r.Z, r.Z, h);
}

__device__ void scalar_mult(Point& R, const fe& scalar) {
    Point Q;
    fe_copy(Q.X, *((fe*)SECP256K1_G_X));
    fe_copy(Q.Y, *((fe*)SECP256K1_G_Y));
    fe_zero(Q.Z); Q.Z.v[0] = 1;

    fe_zero(R.X);
    fe_zero(R.Y);
    fe_zero(R.Z);  // R = infinity

    for (int i = 255; i >= 0; --i) {
        point_double(R, R);

        int word = i / 32;
        int bit = i % 32;
        if ((scalar.v[word] >> bit) & 1) {
            point_add(R, R, Q);
        }
    }
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

__device__ void generate_compressed_pubkey(const uint8_t priv[32], uint8_t pubkey[33]) {
    fe scalar = {0};
    for (int i = 0; i < 8; i++) {
        scalar.v[i] = 
            ((uint32_t)priv[28 - i * 4] << 24) |
            ((uint32_t)priv[29 - i * 4] << 16) |
            ((uint32_t)priv[30 - i * 4] << 8)  |
            ((uint32_t)priv[31 - i * 4]);
    }
    Point R;
    scalar_mult(R, scalar);
    affine_from_jacobian(pubkey, R);
}
