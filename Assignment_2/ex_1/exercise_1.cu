
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>

#define B 1
#define TPB 256

__global__ void executeKernel()
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Hello World! My threadId is %d\n", i);
}

int main()
{
	executeKernel << <B, TPB >> > ();
	cudaDeviceSynchronize();
	return 0;
}