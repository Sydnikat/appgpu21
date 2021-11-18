#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <curand.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define NUM_ITER 10000000000
#define BLOCK_SIZE 64

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

__global__ void dot_double_kernel(int* d_res, curandState* states, time_t seed, size_t iterations) {
	const int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(seed, idx, 0, &states[idx]);

	for (size_t i = 0; i < iterations; i++)
	{
		double x = curand_uniform_double(&states[idx]);
		double y = curand_uniform_double(&states[idx]);
		double z = sqrt((x * x) + (y * y));

		d_res[idx] += (z <= 1.0) ? 1.0 : 0.0;
	}
}

__global__ void dot_single_kernel(int* d_res, curandState* states, time_t seed, size_t iterations) {
	const int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(seed, idx, 0, &states[idx]);

	for (size_t i = 0; i < iterations; i++)
	{
		float x = curand_uniform(&states[idx]);
		float y = curand_uniform(&states[idx]);
		float z = sqrtf((x * x) + (y * y));

		d_res[idx] += (z <= 1.0f) ? 1.0f : 0.0f;
	}
}

void execute_on_GPU(time_t seed, size_t number_of_iterations, size_t number_of_blocks, size_t block_size, bool is_sinlge)
{
	size_t iterations_per_thread = number_of_iterations / (number_of_blocks * block_size);

	int* block_partials;
	block_partials = (int*)malloc(number_of_blocks * block_size * sizeof(int));

	int* d_block_partials;
	cudaMalloc(&d_block_partials, number_of_blocks * block_size * sizeof(int));

	curandState* d_randoms;
	cudaMalloc((void**)&d_randoms, number_of_blocks * block_size * sizeof(curandState));

	cudaMemcpy(d_block_partials, block_partials, number_of_blocks * block_size * sizeof(curandState), cudaMemcpyHostToDevice);

	uint64_t start = time_ns();

	if (is_sinlge)
	{
		dot_single_kernel << <number_of_blocks, block_size >> > (d_block_partials, d_randoms, seed, iterations_per_thread);
	}
	else {
		dot_double_kernel << <number_of_blocks, block_size >> > (d_block_partials, d_randoms, seed, iterations_per_thread);
	}

	cudaDeviceSynchronize();

	uint64_t end = time_ns() - start;
	printf("Done! Time elapsed: (ns) = %llu\t (s): %f\n\n", end, (float)end / 1e9);

	cudaMemcpy(block_partials, d_block_partials, number_of_blocks * block_size * sizeof(int), cudaMemcpyDeviceToHost);

	uint64_t count = 0;
	for (int i = 0; i < number_of_blocks * block_size; i++) {
		count += block_partials[i];
	}

	double PI = acos(-1);

	if (is_sinlge)
	{
		float pi = ((float)count / (float)number_of_iterations) * 4.0;
		printf("Calculated the value of PI with the following result: %f.\n", pi);
	
		printf(" The difference between this and the predifined PI is: %f\n", fabs(PI - pi));
	}
	else {
		double pi = ((double)count / (double)number_of_iterations) * 4.0;
		printf("Calculated the value of PI with the following result: %f\n", pi);

		printf(" The difference between this and the predifined PI is: %f\n", fabs(PI - pi));
	}

	cudaFree(d_block_partials);
	cudaFree(d_randoms);
	free(block_partials);
}

int main(int argc, char* argv[])
{    
	// NVIDIA GeForce GTX 1650Ti number of threads that the GPU core may run simultaneously
	const size_t max_number_of_threads = 1024; 

	const time_t seed = time(NULL);
	const size_t num_of_iterations = (argc >= 2) ? strtoull(argv[1], NULL, 10) : NUM_ITER;
	const size_t block_size = (argc == 3) ? strtoull(argv[2], NULL, 10) : BLOCK_SIZE;
	const size_t num_of_blocks = max_number_of_threads / block_size;

	printf("Computing PI with %llu iterations on the GPU with grid size (1-dim): %llu and threads/block: %llu...\n\n", num_of_iterations, num_of_blocks, block_size);
	printf("With single precision...\n");
	execute_on_GPU(seed, num_of_iterations, 16, block_size, true);

	printf("\nWith double precision...\n");
	execute_on_GPU(seed, num_of_iterations, 16, block_size, false);

	return 0;
}