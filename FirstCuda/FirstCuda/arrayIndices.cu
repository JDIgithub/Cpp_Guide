#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <vector>

// Device Code

__global__ void hello_cuda() {
	printf("Hello Cuda World\n"); // 64x times
	//std::cout << "Hello CUDA world \n";  Can not be used in device code
}

__global__ void print_details() {
	printf(	"blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d \n"
			"blockDim.x: %d, blockDim.y: %d, blockDim.z: %d \n"
			"gridDim.x: %d, gridDim.y: %d, gridDim.z: %d \n"
			, blockIdx.x, blockIdx.y, blockIdx.z
			, blockDim.x, blockDim.y, blockDim.z
			, gridDim.x, gridDim.y, gridDim.z
	);
}


__global__ void unique_gid_calc(int * input) {
	int tid = threadIdx.x;
	int offset = blockIdx.x * blockDim.x;
	int gid = tid + offset;
	printf("threadIdx: %d, gid: %d, value: %d \n", tid, gid, input[gid]);
}

__global__ void unique_gid_calc_2d(int* input) {
	//int tid = threadIdx.x;
	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	//int block_offset = blockIdx.x * blockDim.x;
	int block_offset = blockIdx.x * (blockDim.x * blockDim.y);
	int numThreadsInRow = gridDim.x * blockDim.x;	// Number of threads in one row
	//int row_offset = numThreadsInRow * blockIdx.y;	// Number of threads in one row * actual row
	int row_offset = blockIdx.y * (gridDim.x * blockDim.x * blockDim.y);
	int gid = tid + block_offset + row_offset;
	printf("threadIdx: %d, gid: %d, value: %d \n", tid, gid, input[gid]);
}


// Host Code

int mainIndices(){

	int array_size = 16;
	int array_byte_size = sizeof(int) * array_size;
	std::vector<int> v_data = { 23,9,4,53,65,12,1,33,41,48,49,47,45,46,44,43 };
	//int h_data[] = { 23,9,4,53,65,12,1,33,41,48,49,47,45,46,44,43 };

	int xThreadCount = 16;
	int yThreadCount = 16;
	int zThreadCount = 1;

	int* d_data = nullptr;
	std::vector<int> dd_data;
	cudaMalloc((void**)&d_data, sizeof(int)*v_data.size());
	cudaMemcpy(d_data, v_data.data(), sizeof(int) * v_data.size(), cudaMemcpyHostToDevice);



	dim3 block(2, 2, 1);
	dim3 grid(2,2,1); 		
	// 20 threads are launched
	unique_gid_calc_2d <<< grid,block >>>(d_data);	// Asynchronous function call

	// We we need to wait for the kernel function to be done we need to use
	cudaDeviceSynchronize();	// Similar to .join() in std::thread but it is for all kernels that were launched
	
	cudaDeviceReset();

	return 0;
}