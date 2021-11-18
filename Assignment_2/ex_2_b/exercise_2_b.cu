#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>

#define TPB (size_t)256

typedef struct {
	float a;
	float* x;
	float* y;
} SAXPY;

void setup_data(SAXPY* data, size_t array_size)
{
	data->a = 1.5f;
	data->x = (float*)malloc(array_size * sizeof(float));
	data->y = (float*)malloc(array_size * sizeof(float));
	for (size_t i = 0; i < array_size; i++)
	{
		data->x[i] = 1;
		data->y[i] = 1;
	}
}

void free_data(SAXPY* data)
{
	free(data->x);
	free(data->y);
	free(data);
}

static uint64_t time_ns(void)
{
	struct timespec ts;

	if (timespec_get(&ts, TIME_UTC) != TIME_UTC) return 0;
	return (uint64_t)1e9 * ts.tv_sec + ts.tv_nsec;
}

void execute_on_CPU(SAXPY* data, size_t array_size)
{
	uint64_t start = time_ns();

	for (size_t i = 0; i < array_size; i++)
	{
		data->y[i] += data->a * data->x[i];
	}

	uint64_t end = time_ns() - start;
	printf("%llu;", end);
}

__global__ void saxpy_kernel(float a, float* d_x, float* d_y, const int array_size)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= array_size) return;

	d_y[i] += a * d_x[i];
}

void execute_on_GPU(SAXPY* data, size_t array_size)
{
	float* d_x = 0;
	float* d_y = 0;

	uint64_t start = time_ns();

	cudaMalloc(&d_x, array_size * sizeof(float));
	cudaMalloc(&d_y, array_size * sizeof(float));

	cudaMemcpy(d_x, data->x, array_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, data->y, array_size * sizeof(float), cudaMemcpyHostToDevice);


	saxpy_kernel << <(array_size + TPB - 1) / TPB, TPB >> > (data->a, d_x, d_y, array_size);


	cudaMemcpy(data->y, d_y, array_size * sizeof(float), cudaMemcpyDeviceToHost);

	uint64_t end = time_ns() - start;
	printf("%llu\n", end);

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

	printf("array size;CPU time;GPU time\n");

	for (size_t i = 1e4; i <= 1e6; i+=1e4)
	{
		printf("%zd;", i);

		SAXPY* cpu_data = 0;
		cpu_data = (SAXPY*)malloc(sizeof(SAXPY));
		setup_data(cpu_data, i);
		execute_on_CPU(cpu_data, i);
		free_data(cpu_data);

		SAXPY* gpu_data = 0;
		gpu_data = (SAXPY*)malloc(sizeof(SAXPY));
		setup_data(gpu_data, i);
		execute_on_GPU(gpu_data, i);
		free_data(gpu_data);
	}
	return 0;
}