#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <vector>
#include <array>
#include <memory>
#include <iomanip>

// Device Code




// Host Code
void query_device() {
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	
	int devNo = 0;
	cudaDeviceProp iProp;
	cudaGetDeviceProperties(&iProp, devNo);

	std::cout << "Number of Devices " << deviceCount << std::endl;
	std::cout << "Device " << devNo << " " << iProp.name << std::endl;
	std::cout << "Number of MultiProcessors " << iProp.multiProcessorCount << std::endl;
	std::cout << "Clock rate " << iProp.clockRate << std::endl;
	std::cout << "Compute capability " << iProp.major << "." << iProp.minor << std::endl;
	std::cout << "Global Memory [kB] " << iProp.totalGlobalMem/1024.0 << std::endl;
	std::cout << "Constant Memory [kB] " << iProp.totalConstMem/1024.0 << std::endl;
	std::cout << "Shared Memory per Block [kB] " << iProp.sharedMemPerBlock << std::endl;
	std::cout << "Shared Memory per MP [kB] " << iProp.sharedMemPerMultiprocessor << std::endl;
	std::cout << "Warp Size " << iProp.warpSize << std::endl;
	std::cout << "Max Threads per Block " << iProp.maxThreadsPerBlock << std::endl;
	std::cout << "Max Threads per MP " << iProp.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "Max Grid Size " << iProp.maxGridSize[0] << "," << iProp.maxGridSize[1] << "," << iProp.maxGridSize[2] << std::endl;
	std::cout << "Max Block Dim " << iProp.maxThreadsDim[0] << "," << iProp.maxThreadsDim[1] << "," << iProp.maxThreadsDim[2] << std::endl;

}


int mainProp() {

	query_device();

	cudaDeviceReset();

	return 0;
}