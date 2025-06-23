#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include "gpu_scan.cuh"
#include "hash160_loader.hpp"

void write_found_key(uint64_t key, const std::string& hash160) {
    std::ofstream fout("found.txt", std::ios::app);
    fout << "Key: 0x" << std::hex << key << std::dec << " -> Hash160: " << hash160 << "\n";
    fout.close();
}

int main() {
    std::vector<std::array<uint8_t, 20>> targets;
    if (!load_hash160_bin("puzzle71_hash160.bin", targets)) {
        std::cerr << "[!] Failed to load puzzle71_hash160.bin\n";
        return 1;
    }

    uint64_t start = 0x7000000000000000;
    uint64_t chunk_size = 0x10000000;

    while (start < 0x7FFFFFFFFFFFFFFF) {
        uint64_t end = start + chunk_size;
        std::cout << "[*] Scanning range: " << std::hex << start << " to " << end << std::dec << "\n";

        auto matches = scan_range_on_gpu_with_output(start, end, targets);

        for (const auto& match : matches) {
            std::ostringstream oss;
            for (uint8_t b : match.second)
                oss << std::hex << std::setw(2) << std::setfill('0') << (int)b;
            write_found_key(match.first, oss.str());
            std::cout << "[MATCH] Private key: 0x" << std::hex << match.first << std::dec << " -> " << oss.str() << "\n";
        }

        if (!matches.empty()) {
            std::cout << "[*] Pushing found.txt to GitHub..." << std::endl;
            (void)system("./upload_found.sh");
        }

        start = end;
    }

    return 0;
}
