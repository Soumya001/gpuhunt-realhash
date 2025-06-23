#include "gpu_scan.cuh"
#include "secp256k1_math.cu"
#include "sha256.cu"
#include "ripemd160.cu"
#include <cuda_runtime.h>
#include <vector>
#include <cstring>

__global__ void gpu_kernel(uint64_t start, uint64_t total, const uint8_t* d_targets, int num_targets, bool* d_flags, uint64_t* d_results) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    uint64_t key = start + idx;
    uint8_t pubkey[33];
    generate_compressed_pubkey(key, pubkey);

    uint8_t sha[32];
    sha256(pubkey, 33, sha);

    uint8_t h160[20];
    ripemd160(sha, 32, h160);

    for (int t = 0; t < num_targets; ++t) {
        bool match = true;
        for (int i = 0; i < 20; ++i) {
            if (h160[i] != d_targets[t * 20 + i]) {
                match = false;
                break;
            }
        }
        if (match) {
            d_flags[idx] = true;
            d_results[idx] = key;
        }
    }
}

std::vector<std::pair<uint64_t, std::array<uint8_t, 20>>> scan_range_on_gpu_with_output(uint64_t start, uint64_t end, const std::vector<std::array<uint8_t, 20>>& targets) {
    uint64_t total = end - start;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    std::vector<std::pair<uint64_t, std::array<uint8_t, 20>>> matches;

    bool* d_flags;
    uint64_t* d_results;
    uint8_t* d_targets;

    cudaMalloc(&d_flags, total * sizeof(bool));
    cudaMalloc(&d_results, total * sizeof(uint64_t));
    cudaMalloc(&d_targets, targets.size() * 20);

    std::vector<uint8_t> flat_targets(targets.size() * 20);
    for (size_t i = 0; i < targets.size(); ++i)
        memcpy(&flat_targets[i * 20], targets[i].data(), 20);

    cudaMemcpy(d_targets, flat_targets.data(), flat_targets.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_flags, 0, total * sizeof(bool));

    gpu_kernel<<<blocks, threads>>>(start, total, d_targets, targets.size(), d_flags, d_results);
    cudaDeviceSynchronize();

    std::vector<bool> h_flags(total);
    std::vector<uint64_t> h_results(total);
    cudaMemcpy(h_flags.data(), d_flags, total * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_results.data(), d_results, total * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    for (uint64_t i = 0; i < total; ++i) {
        if (h_flags[i]) {
            uint64_t key = h_results[i];
            uint8_t pubkey[33];
            generate_compressed_pubkey(key, pubkey);
            uint8_t sha[32], h160[20];
            sha256(pubkey, 33, sha);
            ripemd160(sha, 32, h160);
            std::array<uint8_t, 20> out;
            memcpy(out.data(), h160, 20);
            matches.emplace_back(key, out);
        }
    }

    cudaFree(d_flags);
    cudaFree(d_results);
    cudaFree(d_targets);

    return matches;
}