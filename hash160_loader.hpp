#ifndef HASH160_LOADER_HPP
#define HASH160_LOADER_HPP
#include <vector>
#include <array>
#include <fstream>

bool load_hash160_bin(const std::string& filename, std::vector<std::array<uint8_t, 20>>& hashes) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return false;

    std::array<uint8_t, 20> buf;
    while (f.read(reinterpret_cast<char*>(buf.data()), 20)) {
        hashes.push_back(buf);
    }
    return true;
}

#endif