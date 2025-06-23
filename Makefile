all:
	nvcc main.cpp gpu_scan.cu -o gpuhunt -O2