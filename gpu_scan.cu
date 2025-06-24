#include "gpu_scan.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <iostream>
#include "hash.cuh"
#include "secp256k1_math.cuh"

__global__ void scan_kernel(uint64_t start, uint64_t total, const uint8_t* d_targets, size_t num_targets,
                            uint64_t* d_matches, uint8_t* d_hashes, int* d_count) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    uint64_t key = start + idx;

    uint8_t priv_bytes[32] = {0};
    for (int i = 0; i < 8; ++i)
        priv_bytes[24 + i] = (key >> ((7 - i) * 8)) & 0xFF;

    uint8_t pubkey[33], sha[32], h160[20];
    generate_compressed_pubkey(priv_bytes, pubkey);
    sha256(pubkey, 33, sha);
    ripemd160(sha, 32, h160);

    for (int t = 0; t < num_targets; t++) {
        const uint8_t* target = &d_targets[t * 20];
        bool match = true;
        #pragma unroll
        for (int j = 0; j < 20; j++) {
            if (h160[j] != target[j]) {
                match = false;
                break;
            }
        }

        if (match) {
            int pos = atomicAdd(d_count, 1);
            d_matches[pos] = key;
            #pragma unroll
            for (int j = 0; j < 20; j++)
                d_hashes[pos * 20 + j] = h160[j];
        }
    }
}

std::vector<std::pair<uint64_t, std::array<uint8_t, 20>>> scan_range_on_gpu_with_output(
    uint64_t start, uint64_t end, const std::vector<std::array<uint8_t, 20>>& targets
) {
    uint64_t total = end - start;
    std::vector<std::pair<uint64_t, std::array<uint8_t, 20>>> results;

    std::vector<uint8_t> flat_targets;
    flat_targets.reserve(targets.size() * 20);
    for (auto& t : targets) flat_targets.insert(flat_targets.end(), t.begin(), t.end());

    uint8_t* d_targets = nullptr;
    uint64_t* d_matches = nullptr;
    uint8_t* d_hashes = nullptr;
    int* d_count = nullptr;

    cudaMalloc(&d_targets, flat_targets.size());
    cudaMalloc(&d_matches, total * sizeof(uint64_t));
    cudaMalloc(&d_hashes, total * 20);
    cudaMalloc(&d_count, sizeof(int));

    cudaMemset(d_count, 0, sizeof(int));
    cudaMemcpy(d_targets, flat_targets.data(), flat_targets.size(), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    scan_kernel<<<blocks, threads>>>(start, total, d_targets, targets.size(), d_matches, d_hashes, d_count);
    cudaDeviceSynchronize();

    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_count > 0) {
        std::vector<uint64_t> h_keys(h_count);
        std::vector<uint8_t> h_h160(h_count * 20);
        cudaMemcpy(h_keys.data(), d_matches, h_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_h160.data(), d_hashes, h_count * 20, cudaMemcpyDeviceToHost);

        for (int i = 0; i < h_count; i++) {
            std::array<uint8_t, 20> hash;
            for (int j = 0; j < 20; ++j) {
                hash[j] = h_h160[i * 20 + j];
            }
            results.emplace_back(h_keys[i], hash);
        }
    }

    cudaFree(d_targets);
    cudaFree(d_matches);
    cudaFree(d_hashes);
    cudaFree(d_count);

    return results;
}
