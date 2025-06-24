# GPU Bitcoin Puzzle Hunter

This tool performs high-performance brute-force scanning for Bitcoin Puzzle #71 using real secp256k1, SHA-256, and RIPEMD-160 implementations on CUDA-enabled GPUs.

## ⚙️ Requirements

- NVIDIA GPU with CUDA support
- Ubuntu/Debian-based system
- Internet connection

## 🛠 Installation

Run the following commands step-by-step in your terminal:

```bash
# 1. Update package lists
apt update

# 2. Install essential packages
apt install -y build-essential git nano curl unzip

# 3. Install CUDA toolkit
apt install -y nvidia-cuda-toolkit

# 4. Verify NVCC installation
nvcc --version

# 5. Clone the repository
git clone https://github.com/Soumya001/gpuhunt-realhash.git

# 6. Navigate to project directory
cd gpuhunt-realhash

# 7. Compile the project
nvcc main.cpp gpu_scan.cu -o gpuhunt -O2 -Wno-deprecated-gpu-targets

# 8. Make the helper script executable (optional if using upload_found.sh)
chmod +x upload_found.sh

# 9. Start the scanner
./gpuhunt
