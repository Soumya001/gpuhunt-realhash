#pragma once
#include <vector>
#include <array>
std::vector<std::pair<uint64_t, std::array<uint8_t, 20>>> scan_range_on_gpu_with_output(
    uint64_t start,
    uint64_t end,
    const std::vector<std::array<uint8_t, 20>>& targets
);
