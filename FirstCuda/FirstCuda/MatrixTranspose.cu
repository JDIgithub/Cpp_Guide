#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <vector>
#include <array>
#include <memory>
#include <ctime>

// Device Code
__global__ void transpose_mat_cuda (int * mat, int* transpose, int nx, int ny) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny) {
		transpose[ix * ny + iy] = mat[iy * nx + ix];
	}



	/*if (gid < size) {
		c[gid] = a[gid] + b[gid];
	}*/

	

}

/*
void tanspose_mat(float* out, float* in, const int nx, const int ny) {

	for (int iy = 0; iy < ny; ++iy) {
		for (int ix = 0; ix < nx; ++ix) {
			out[ix * ny + iy] = in[iy * nx + ix];
		}	
	}
}*/


// Host Code

int main() {
	
	int nx = 4;
	int ny = 3;
	int block_x = 32;
	int block_y = 3;

	int size = nx * ny;
	int byte_size = sizeof(int) * size;

	int* host_mat_array = new int[size];
	int* host_trans_array = new int[size];
	
	for (int x = 0; x < nx*ny; x++) {
		host_mat_array[x] = x;
	}

	int* device_mat_array;
	int* device_trans_array;

	cudaMalloc((void**)&device_mat_array, byte_size);
	cudaMalloc((void**)&device_trans_array, byte_size);

	cudaMemcpy(device_mat_array,host_mat_array, byte_size, cudaMemcpyHostToDevice);


	dim3 block(block_x, block_y, 1);
	dim3 grid((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, 1);

	clock_t gpu_start, gpu_end;
	gpu_start = clock();

	transpose_mat_cuda <<<grid,block>>> (device_mat_array, device_trans_array, nx, ny);

	cudaDeviceSynchronize();

	gpu_end = clock();

	cudaMemcpy(host_trans_array, device_trans_array, byte_size, cudaMemcpyDeviceToHost);


	int x = 49499;


	cudaDeviceReset();


	return 0;
}