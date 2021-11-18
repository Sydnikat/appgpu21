#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>
#include <math.h>

#define TPB 256
#define ARRAY_SIZE 100000
#define B (ARRAY_SIZE + TPB - 1) / TPB

typedef struct {
	float a;
	float x[ARRAY_SIZE];
	float y[ARRAY_SIZE];
} SAXPY;

void setup_data(SAXPY* data)
{
	data->a = 1.5f;
	for (size_t i = 0; i < ARRAY_SIZE; i++)
	{
		data->x[i] = 1;
		data->y[i] = 1;
	}
}

static uint64_t time_ns(void)
{
	struct timespec ts;

	if (timespec_get(&ts, TIME_UTC) != TIME_UTC)
	{
		fputs("timespec_get failed!", stderr);
		return 0;
	}
	return (uint64_t)1e9 * ts.tv_sec + ts.tv_nsec;
}

void execute_on_CPU(SAXPY* data)
{
	printf("Computing SAXPY on the CPU...\n");
	uint64_t start = time_ns();

	for (size_t i = 0; i < ARRAY_SIZE; i++)
	{
		data->y[i] += data->a * data->x[i];
	}

	uint64_t end = time_ns() - start;
	printf("Done! Time elapsed: (ns) = %d\t (s): %f\n\n", end, (float)end / 1e9);
}

__global__ void saxpyKernel(float a, float* d_x, float* d_y)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= ARRAY_SIZE) return;

	d_y[i] += a * d_x[i];
}

void execute_on_GPU(SAXPY* data)
{
	float* d_x = 0;
	float* d_y = 0;

	cudaMalloc(&d_x, ARRAY_SIZE * sizeof(float));
	cudaMalloc(&d_y, ARRAY_SIZE * sizeof(float));

	cudaMemcpy(d_x, data->x, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, data->y, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	printf("Computing SAXPY on the GPU...\n");
	uint64_t start = time_ns();

	saxpyKernel << <B, TPB >> > (data->a, d_x, d_y);

	uint64_t end = time_ns() - start;
	printf("Done! Time elapsed: (ns) = %d\t (s): %f\n\n", end, (float)end / 1e9);


	cudaDeviceSynchronize();

	cudaMemcpy(data->y, d_y, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_x);
	cudaFree(d_y);
}

int main()
{
	SAXPY* cpu_data = 0;
	SAXPY* gpu_data = 0;
	cpu_data = (SAXPY*)malloc(sizeof(SAXPY));
	gpu_data = (SAXPY*)malloc(sizeof(SAXPY));
	if (cpu_data == NULL || gpu_data == NULL) return -1;

	setup_data(cpu_data);
	execute_on_CPU(cpu_data);

	setup_data(gpu_data);
	execute_on_GPU(gpu_data);

	printf("Comparing the output for each implementation...");
	int same = 1;
	const float margin = 1e-6;
	for (size_t i = 0; i < ARRAY_SIZE; i++)
	{
		if (fabs(gpu_data->y[i] - cpu_data->y[i]) > margin)
		{
			same = 0;
			break;
		}
	}

	printf("\t%s!\n", same ? "Correct" : "Failed!");

	free(cpu_data);
	free(gpu_data);
	return 0;
}