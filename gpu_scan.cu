#include "gpu_scan.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <iostream>

// Constants
#define THREADS_PER_BLOCK 256

// === Device Utility Functions (You should implement these!) ===
__device__ void generate_compressed_pubkey(uint64_t privkey, uint8_t* out33) {
    // 🔧 TODO: ECC secp256k1 scalar multiplication to get 33-byte compressed pubkey
    for (int i = 0; i < 33; i++) out33[i] = i + privkey % 256;  // Dummy pubkey
}

__device__ void sha256(const uint8_t* data, size_t len, uint8_t* out32) {
    for (int i = 0; i < 32; i++) out32[i] = (data[i % len] + i) % 256;  // Dummy SHA256
}

__device__ void ripemd160(const uint8_t* data, size_t len, uint8_t* out20) {
    for (int i = 0; i < 20; i++) out20[i] = (data[i % len] + i * 3) % 256;  // Dummy RIPEMD160
}

// === CUDA Kernel ===
__global__ void scan_kernel(uint64_t start, uint64_t total, const uint8_t* d_targets, size_t num_targets, uint64_t* d_matches, uint8_t* d_hashes, int* d_count) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    uint64_t key = start + idx;

    uint8_t pubkey[33], sha[32], h160[20];
    generate_compressed_pubkey(key, pubkey);
    sha256(pubkey, 33, sha);
    ripemd160(sha, 32, h160);

    for (int t = 0; t < num_targets; t++) {
        bool match = true;
        for (int j = 0; j < 20; j++) {
            if (h160[j] != d_targets[t * 20 + j]) {
                match = false;
                break;
            }
        }

        if (match) {
            int pos = atomicAdd(d_count, 1);
            d_matches[pos] = key;
            for (int j = 0; j < 20; j++) {
                d_hashes[pos * 20 + j] = h160[j];
            }
        }
    }
}

// === Host Wrapper Function ===
std::vector<std::pair<uint64_t, std::array<uint8_t, 20>>> scan_range_on_gpu_with_output(
    uint64_t start,
    uint64_t end,
    const std::vector<std::array<uint8_t, 20>>& targets
) {
    uint64_t total = end - start;
    size_t num_targets = targets.size();

    // Host output
    std::vector<std::pair<uint64_t, std::array<uint8_t, 20>>> results;

    // Flatten target list
    std::vector<uint8_t> flat_targets;
    for (auto& t : targets) flat_targets.insert(flat_targets.end(), t.begin(), t.end());

    // Allocate GPU memory
    uint8_t* d_targets;
    uint64_t* d_matches;
    uint8_t* d_hashes;
    int* d_count;
    cudaMalloc(&d_targets, flat_targets.size());
    cudaMalloc(&d_matches, total * sizeof(uint64_t));
    cudaMalloc(&d_hashes, total * 20);
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    cudaMemcpy(d_targets, flat_targets.data(), flat_targets.size(), cudaMemcpyHostToDevice);

    // Launch kernel
    int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    scan_kernel<<<blocks, THREADS_PER_BLOCK>>>(start, total, d_targets, num_targets, d_matches, d_hashes, d_count);
    cudaDeviceSynchronize();

    // Copy results back
    int h_count;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<uint64_t> h_keys(h_count);
    std::vector<uint8_t> h_hashes(h_count * 20);

    cudaMemcpy(h_keys.data(), d_matches, h_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hashes.data(), d_hashes, h_count * 20, cudaMemcpyDeviceToHost);

    for (int i = 0; i < h_count; i++) {
        std::array<uint8_t, 20> h160;
        std::copy_n(h_hashes.data() + i * 20, 20, h160.begin());
        results.emplace_back(h_keys[i], h160);
    }

    // Cleanup
    cudaFree(d_targets);
    cudaFree(d_matches);
    cudaFree(d_hashes);
    cudaFree(d_count);

    return results;
}
