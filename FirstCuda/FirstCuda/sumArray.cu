#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <vector>
#include <array>
#include <memory>

// Device Code
__global__ void sum_array_gpu(int* a, int* b, int* c, int size) {
	//int tid = threadIdx.x + threadIdx.y * blockDim.x;
	//int block_offset = blockIdx.x * (blockDim.x * blockDim.y);
	//int row_offset = blockIdx.y * (gridDim.x * blockDim.x * blockDim.y);
	//int gid = tid + block_offset + row_offset;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < size) {
		c[gid] = a[gid] + b[gid];
	}
}

// Host Code
int mainSum() {
	const int size = 1000;
	int block_size = 128;
	int No_BYTES = size * sizeof(int);
	cudaError_t err;

	// Host arrays
	std::array<int,size> host_a;
	std::array<int,size> host_b;
	std::array<int, size> gpu_results = {0};
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; i++) {
		host_a[i] = static_cast<int>(rand() & 0xff);
		host_b[i] = static_cast<int>(rand() & 0xff);
	}

	// Device pointer
	int* device_a, * device_b, * device_c;
	// Allocate memory on GPU
	cudaMalloc((void**)&device_a, No_BYTES);	// To allocate memory on GPU
	cudaMalloc((void**)&device_b, No_BYTES);
	cudaMalloc((void**)&device_c, No_BYTES);	

	cudaMemcpy(device_a, host_a.data(), No_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b.data(), No_BYTES, cudaMemcpyHostToDevice);

	dim3 block(block_size, 1, 1);
	dim3 grid(((size / block_size) + 1), 1, 1); // + 1 to insure there will be more thread than array size
	sum_array_gpu << < grid, block >> > (device_a, device_b, device_c, size);	// Asynchronous function call
	// We we need to wait for the kernel function to be done we need to use
	cudaDeviceSynchronize();	// Similar to .join() in std::thread but it is for all kernels that were launched
	cudaMemcpy(gpu_results.data(), device_c, No_BYTES, cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; i++) { std::cout << "a = " << host_a[i] << " b = " << host_b[i] << " gpu = " << gpu_results[i] << std::endl;}
	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);
	cudaDeviceReset();
	return 0;
}