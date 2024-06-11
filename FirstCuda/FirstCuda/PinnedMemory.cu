#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <vector>
#include <memory>



int main() {

	// Memory Size 128 MBs
	int isize = 1 << 25;
	int nbytes = isize * sizeof(float);

	// Allocate the Host Memory
	float * host_a = new float[isize];
	// Allocate the Pinned Memory
	float * host_a_pinned;
	cudaMallocHost((float **)&host_a_pinned, nbytes);

	// Allocate the Device Memory
	float * device_a = nullptr;
	cudaMalloc((float **)&device_a, nbytes);

	// Init the Host memory
	for (int i = 0; i < isize; i++) host_a[i] = 7;

	// Transfer data from the host to device
	cudaMemcpy(device_a, host_a, nbytes, cudaMemcpyHostToDevice);
	// Transfer  back from device to host
	cudaMemcpy(host_a, device_a, nbytes, cudaMemcpyDeviceToHost);

	// The same with Pinned memory
	// These transfers should be faster
	cudaMemcpy(device_a, host_a_pinned, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(host_a_pinned, device_a, nbytes, cudaMemcpyDeviceToHost);

	// free Memory
	delete[] host_a;
	cudaFreeHost(host_a_pinned);
	cudaFree(device_a);
	
	cudaDeviceReset();

	return 0;
}