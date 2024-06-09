#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <vector>
#include <array>
#include <memory>
#include <iomanip>

// Device Code
__global__ void sum_array_gpu_timing(int* a, int* b, int* c, int size) {
	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	int block_offset = blockIdx.x * (blockDim.x * blockDim.y);
	int row_offset = blockIdx.y * (gridDim.x * blockDim.x * blockDim.y);
	int gid = tid + block_offset + row_offset;
	if (gid < size) {
		c[gid] = a[gid] + b[gid];
	}
}



// Host Code
void sum_array_cpu_timing(int* a, int * b, int * c, int size) {
	for(int i = 0; i < size; i++) c[i] = a[i] + b[i];
}


int mainTim() {

	const int size = 50000;
	int block_size = 128;
	int No_BYTES = size * sizeof(int);
	cudaError_t err;

	// Host arrays
	std::array<int, size> host_a;
	std::array<int, size> host_b;
	std::array<int, size> gpu_results = { 0 };
	std::array<int, size> cpu_results = { 0 };

	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; i++) {
		host_a[i] = static_cast<int>(rand() & 0xff);
		host_b[i] = static_cast<int>(rand() & 0xff);
	}

	// summation in CPU
	clock_t cpu_start, cpu_end;
	cpu_start = clock();
	sum_array_cpu_timing(host_a.data(), host_b.data(), cpu_results.data(), size);
	cpu_end = clock();


	// Device pointer
	int* device_a, * device_b, * device_c;
	// Allocate memory on GPU
	cudaMalloc((void**)&device_a, No_BYTES);	// To allocate memory on GPU
	cudaMalloc((void**)&device_b, No_BYTES);
	cudaMalloc((void**)&device_c, No_BYTES);

	clock_t HtoD_start, HtoD_end;
	HtoD_start = clock();
	cudaMemcpy(device_a, host_a.data(), No_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b.data(), No_BYTES, cudaMemcpyHostToDevice);
	HtoD_end = clock();

	dim3 block(block_size, 1, 1);
	dim3 grid(((size / block_size) + 1), 1, 1); // + 1 to insure there will be more thread than array size

	clock_t gpu_start, gpu_end;
	gpu_start = clock();
	sum_array_gpu_timing << < grid, block >> > (device_a, device_b, device_c, size);	// Asynchronous function call
	cudaDeviceSynchronize();	// Similar to .join() in std::thread but it is for all kernels that were launched
	gpu_end = clock();


	clock_t DtoH_start, DtoH_end;
	DtoH_start = clock();
	cudaMemcpy(gpu_results.data(), device_c, No_BYTES, cudaMemcpyDeviceToHost);
	DtoH_end = clock();

	//for (int i = 0; i < size; i++) { std::cout << "a = " << host_a[i] << " b = " << host_b[i] << " gpu = " << gpu_results[i] << std::endl; }

	std::cout << std::fixed << std::setprecision(10) << "Sum array CPU execution time: " << (double)((double)(cpu_end - cpu_start) / (CLOCKS_PER_SEC)) << std::endl;
	std::cout << std::fixed << std::setprecision(10) << "Sum array GPU execution time: " << (double)((double)(gpu_end - gpu_start) / (CLOCKS_PER_SEC)) << std::endl;
	std::cout << std::fixed << std::setprecision(10) << "Host To Device Transfer time: " << (double)((double)(HtoD_end - HtoD_start) / (CLOCKS_PER_SEC)) << std::endl;
	std::cout << std::fixed << std::setprecision(10) << "Device To Host Transfer time: " << (double)((double)(DtoH_end - DtoH_start) / (CLOCKS_PER_SEC)) << std::endl;

	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);
	cudaDeviceReset();

	return 0;
}