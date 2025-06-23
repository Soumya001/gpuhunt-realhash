#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <sstream>
#include <sys/stat.h>
#include "gpu_scan.cuh"
#include "hash160_loader.hpp"

void write_found_key(uint64_t key, const std::string& hash160) {
    std::ofstream fout("found.txt", std::ios::app);
    fout << "Key: 0x" << std::hex << key << std::dec << " -> Hash160: " << hash160 << "\n";
    fout.close();
}

void save_last_range(uint64_t start) {
    std::ofstream out("range.txt");
    out << std::hex << start << std::endl;
}

uint64_t load_last_range_or_default(uint64_t fallback) {
    std::ifstream in("range.txt");
    uint64_t value;
    if (in >> std::hex >> value) {
        return value;
    }
    return fallback;
}

bool file_exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

void print_nvidia_smi() {
    std::cout << "\n[GPU] --- nvidia-smi snapshot ---" << std::endl;
    system("nvidia-smi --query-gpu=name,utilization.gpu,temperature.gpu,power.draw,memory.used --format=csv,noheader,nounits");
    std::cout << "-------------------------------\n" << std::endl;
}

int main() {
    std::vector<std::array<uint8_t, 20>> targets;
    if (!load_hash160_bin("puzzle71_hash160.bin", targets)) {
        std::cerr << "[!] Failed to load puzzle71_hash160.bin\n";
        return 1;
    }

    const uint64_t default_start = 0x7000014000000000;
    uint64_t start = load_last_range_or_default(default_start);
    const uint64_t chunk_size = 0x100000000000;
    const uint64_t max_end = 0x7FFFFFFFFFFFFFFF;
    int chunk_index = 1;

    while (start < max_end) {
        uint64_t end = start + chunk_size;
        if (end > max_end) end = max_end;

        auto chunk_start_time = std::chrono::high_resolution_clock::now();
        std::cout << "[*] Chunk " << std::setw(3) << chunk_index 
                  << " | Range: " << std::hex << start << " - " << end << std::dec << "\n";

        print_nvidia_smi(); // Live GPU stats

        auto matches = scan_range_on_gpu_with_output(start, end, targets);

        for (const auto& match : matches) {
            std::ostringstream oss;
            for (uint8_t b : match.second)
                oss << std::hex << std::setw(2) << std::setfill('0') << (int)b;
            write_found_key(match.first, oss.str());
            std::cout << "[MATCH] Private key: 0x" << std::hex << match.first << std::dec
                      << " -> " << oss.str() << "\n";
        }

        auto chunk_end_time = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(chunk_end_time - chunk_start_time).count();
        double speed_mkeys = static_cast<double>(chunk_size) / duration / 1e6;

        double remaining_chunks = static_cast<double>(max_end - end) / chunk_size;
        double eta_sec = remaining_chunks * duration;

        std::cout << "[✓] Chunk Done | Matches: " << matches.size()
                  << " | Time: " << std::fixed << std::setprecision(2) << duration << "s"
                  << " | Speed: " << speed_mkeys << " M keys/s"
                  << " | ETA: " << static_cast<int>(eta_sec / 60) << " min\n";

        // Save progress
        save_last_range(end);

        // Push to GitHub
        std::cout << "[*] Pushing found.txt & range.txt to GitHub...\n";
        (void)system("./upload_found.sh");

        start = end;
        chunk_index++;
    }

    return 0;
}
