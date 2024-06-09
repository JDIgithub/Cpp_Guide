#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <vector>
#include <memory>

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char * file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// Device Code
__global__ void mem_transfer_test(int* input, int size) {

	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	int block_offset = blockIdx.x * (blockDim.x * blockDim.y);
	int row_offset = blockIdx.y * (gridDim.x * blockDim.x * blockDim.y);
	int gid = tid + block_offset + row_offset;
	if (gid < size) {
		printf("tid: %d, gid: %d, value: %d \n", tid, gid, input[gid]);
	}
}

// Host Code
int mainMem() {

	int array_size = 150;
	int array_byte_size = sizeof(int) * array_size;
	std::unique_ptr<int[]> host_input = std::make_unique<int[]>(array_size);

	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < array_size; i++) {
		host_input[i] = static_cast<int>(rand() & 0xff);
	}

	int* device_input = nullptr;

	cudaError_t err;

	// Allocate memory on GPU
	err = cudaMalloc((void**)&device_input,array_byte_size);	// To allocate memory on GPU
	if (err != cudaSuccess) {
		std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
		return -1;
	}

	// Alternative with Macro
	gpuErrorCheck(cudaMemcpy(device_input, host_input.get(), array_byte_size, cudaMemcpyHostToDevice));


	dim3 block(32, 1, 1);
	dim3 grid(5, 1, 1);
	mem_transfer_test << < grid, block >> > (device_input, array_size);	// Asynchronous function call

	// We we need to wait for the kernel function to be done we need to use
	err = cudaDeviceSynchronize();	// Similar to .join() in std::thread but it is for all kernels that were launched
	if (err != cudaSuccess) {
		std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << std::endl;
		cudaFree(device_input);
		return -1;
	}

	cudaFree(device_input);
	cudaDeviceReset();

	return 0;
}