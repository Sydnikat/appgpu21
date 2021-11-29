#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <curand.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>
#include <ctime>


typedef struct {
	float3 position = float3{ 1, 1, 1 };
	float3 velocity = float3{ 1, 1, 1 };
} Particle;

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

void setup_data(Particle* list, size_t size, unsigned int seed)
{
	srand(seed);
	for (size_t i = 0; i < size; i++)
	{
		list[i].position = float3{
			rand() / (float)RAND_MAX,
			rand() / (float)RAND_MAX,
			rand() / (float)RAND_MAX
		};
		list[i].velocity = float3{
			rand() / (float)RAND_MAX,
			rand() / (float)RAND_MAX,
			rand() / (float)RAND_MAX
		};
	}
}

__device__ float3 update_position(float3 position, float3 velocity, float dt)
{
	position.x += velocity.x * dt;
	position.y += velocity.y * dt;
	position.z += velocity.z * dt;
	return position;
}

__device__ float3 update_velocity(float3 velocity, float dt, size_t iteration)
{
	float acc = (iteration % 100 == 0) ? 0.1f : 0.0f;
	velocity.x += acc;
	velocity.y += acc;
	velocity.z += acc;
	return velocity;
}

__global__ void simpleKernel(Particle* d_particles, size_t array_size, time_t seed, curandState* states, size_t iteration, size_t offset)
{
	const int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
	float dt = 1.0f;

	if (i >= array_size) return;

	d_particles[i].velocity = update_velocity(d_particles[i].velocity, dt, iteration);
	d_particles[i].position = update_position(d_particles[i].position, d_particles[i].velocity, dt);
}

void execute_on_GPU(Particle* particles, time_t seed, size_t number_of_iterations, size_t number_of_particles, size_t block_size, size_t number_of_streams)
{
	printf("Computing particles on the GPU...\n");
	uint64_t start = time_ns();

	Particle* d_particles = 0;
	cudaMalloc(&d_particles, number_of_particles * sizeof(Particle));

	curandState* d_randoms;
	cudaMalloc((void**)&d_randoms, number_of_particles * sizeof(curandState));

	const int stream_size = number_of_particles / number_of_streams;
	const int stream_bytes = stream_size * sizeof(Particle);

	cudaStream_t* streams = (cudaStream_t*)malloc(number_of_streams * sizeof(cudaStream_t));
	for (size_t i = 0; i < number_of_streams; i++)
		cudaStreamCreate(&streams[i]);

	for (size_t j = 0; j < number_of_iterations; j++)
	{
		for (size_t i = 0; i < number_of_streams; i++)
		{
			const int offset = i * stream_size;

			cudaMemcpyAsync(&d_particles[offset], &particles[offset], stream_bytes, cudaMemcpyHostToDevice, streams[i]);

			simpleKernel << <(number_of_particles + block_size - 1) / block_size, block_size, 0, streams[i] >> > (d_particles, number_of_particles, seed, d_randoms, j, offset);

			cudaMemcpyAsync(&particles[offset], &d_particles[offset], stream_bytes, cudaMemcpyDeviceToHost, streams[i]);
		}
	}

	cudaDeviceSynchronize();

	for (size_t i = 0; i < number_of_streams; i++)
		cudaStreamDestroy(streams[i]);

	uint64_t end = time_ns() - start;
	printf("Done! Time elapsed: (ns) = %llu\t (s): %f\n\n", end, (float)end / 1e9);

	cudaFree(d_particles);
	cudaFree(d_randoms);
}

int main(int argc, char* argv[])
{
	if (argc != 5)
	{
		printf("Not enough arguments <number of particles> <number of iterations> <block size (threads per block)>...\n");
		return -1;
	}
	size_t num_particles = strtoull(argv[1], NULL, 10);
	size_t num_iterations = strtoull(argv[2], NULL, 10);
	size_t block_size = strtoull(argv[3], NULL, 10);
	size_t number_of_streams = strtoull(argv[4], NULL, 10);

	printf("Execution in progress with setup:\n\tnumber of particles: %d, number of iterations: %d, block size (threads per block): %d...\n", num_particles, num_iterations, block_size);

	Particle* h_data = 0;

	cudaMallocHost(&h_data, num_particles * sizeof(Particle), cudaHostAllocDefault);

	time_t seed = time(NULL);
	setup_data(h_data, num_particles, seed);

	execute_on_GPU(h_data, seed, num_iterations, num_particles, block_size, number_of_streams);

	cudaFreeHost(h_data);

	return 0;
}