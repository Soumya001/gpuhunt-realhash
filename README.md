# 🔍 GPUHunt RealHash - Bitcoin Puzzle GPU Scanner

A basic but fast GPU-accelerated brute-force scanner for solving Bitcoin hash160 puzzles (like Puzzle #71).  
Built with CUDA for real-time ECC key generation and `hash160` (SHA-256 + RIPEMD-160) hashing.

---

## ⚙️ Features

- 🔧 Custom GPU brute-forcer using CUDA
- ✅ Pluggable ECC and hash160 logic
- 🧠 Supports binary target file (`puzzle71_hash160.bin`)
- 🚀 Automatic GitHub push of `found.txt` when a match is found

---

## 🖥️ Requirements

- Ubuntu 18.04+ or 20.04+
- NVIDIA GPU with CUDA support
- CUDA Toolkit (e.g., `sudo apt install nvidia-cuda-toolkit`)
- Git & Make installed

---

## 🧱 Setup & Build

```bash
git clone https://github.com/YOUR_USERNAME/gpuhunt-realhash.git
cd gpuhunt-realhash
chmod +x upload_found.sh
make
