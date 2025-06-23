#pragma once
#include <vector>
#include <array>
#include <fstream>

inline bool load_hash160_bin(const std::string& filename, std::vector<std::array<uint8_t, 20>>& out) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return false;
    std::array<uint8_t, 20> buf;
    while (f.read((char*)buf.data(), 20)) out.push_back(buf);
    return true;
}
