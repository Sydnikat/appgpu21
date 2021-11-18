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

__host__ __device__ float3 update_position(float3 position, float3 velocity, float dt)
{
	position.x += velocity.x * dt;
	position.y += velocity.y * dt;
	position.z += velocity.z * dt;
	return position;
}

__host__ __device__ float3 update_velocity(float3 velocity, float dt, size_t iteration)
{
	float acc = (iteration % 100 == 0) ? 0.1f : 0.0f;
	velocity.x += acc;
	velocity.y += acc;
	velocity.z += acc;
	return velocity;
}

void execute_on_CPU(Particle* particles, time_t seed, size_t number_of_iterations, size_t number_of_particles)
{
	uint64_t start = time_ns();
	float dt = 1.0f;

	for (size_t j = 0; j < number_of_iterations; j++)
	{
		for (size_t i = 0; i < number_of_particles; i++)
		{
			particles[i].velocity = update_velocity(particles[i].velocity, dt, j);
			particles[i].position = update_position(particles[i].position, particles[i].velocity, dt);
		}
	}

	uint64_t end = time_ns() - start;
	printf("%llu;", end);
}


__global__ void particleKernel(Particle* d_particles, size_t array_size, time_t seed, curandState* states, size_t number_of_iterations)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	float dt = 1.0f;

	if (i >= array_size) return;

	for (size_t j = 0; j < number_of_iterations; j++)
	{
		d_particles[i].velocity = update_velocity(d_particles[i].velocity, dt, j);
		d_particles[i].position = update_position(d_particles[i].position, d_particles[i].velocity, dt);
	}
}

__global__ void simpleKernel(Particle* d_particles, size_t array_size, time_t seed, curandState* states, size_t iteration)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	float dt = 1.0f;

	if (i >= array_size) return;

	d_particles[i].velocity = update_velocity(d_particles[i].velocity, dt, iteration);
	d_particles[i].position = update_position(d_particles[i].position, d_particles[i].velocity, dt);
}

void execute_all_on_GPU(Particle* particles, time_t seed, size_t number_of_iterations, size_t number_of_particles, size_t block_size)
{
	uint64_t start = time_ns();

	Particle* d_particles = 0;
	cudaMalloc(&d_particles, number_of_particles * sizeof(Particle));

	curandState* d_randoms;
	cudaMalloc((void**)&d_randoms, number_of_particles * sizeof(curandState));


	cudaMemcpy(d_particles, particles, number_of_particles * sizeof(Particle), cudaMemcpyHostToDevice);

	particleKernel << <(number_of_particles + block_size - 1) / block_size, block_size >> > (d_particles, number_of_particles, seed, d_randoms, number_of_iterations);

	cudaDeviceSynchronize();

	cudaMemcpy(particles, d_particles, number_of_particles * sizeof(Particle), cudaMemcpyDeviceToHost);

	uint64_t end = time_ns() - start;
	printf("%llu;", end);

	cudaFree(d_particles);
	cudaFree(d_randoms);
}

void execute_one_on_GPU(Particle* particles, time_t seed, size_t number_of_iterations, size_t number_of_particles, size_t block_size)
{
	uint64_t start = time_ns();

	Particle* d_particles = 0;
	cudaMalloc(&d_particles, number_of_particles * sizeof(Particle));

	curandState* d_randoms;
	cudaMalloc((void**)&d_randoms, number_of_particles * sizeof(curandState));

	for (size_t j = 0; j < number_of_iterations; j++)
	{
		cudaMemcpy(d_particles, particles, number_of_particles * sizeof(Particle), cudaMemcpyHostToDevice);

		simpleKernel << <(number_of_particles + block_size - 1) / block_size, block_size >> > (d_particles, number_of_particles, seed, d_randoms, j);

		cudaDeviceSynchronize();

		cudaMemcpy(particles, d_particles, number_of_particles * sizeof(Particle), cudaMemcpyDeviceToHost);
	}

	uint64_t end = time_ns() - start;
	printf("%llu;", end);

	cudaFree(d_particles);
	cudaFree(d_randoms);
}

void test_CPU()
{
	printf("particle size;time with 1000 iterations;time with 2000 iterations\n");

	for (size_t num_particles = 1e4; num_particles <= 1e5; num_particles += 1e4)
	{
		printf("%zd;", num_particles);
		for (size_t num_iterations = 1000; num_iterations <= 2000; num_iterations += 1000)
		{
			Particle* cpu_data = (Particle*)malloc(num_particles * sizeof(Particle));

			time_t seed = time(NULL);
			setup_data(cpu_data, num_particles, seed);

			execute_on_CPU(cpu_data, seed, num_iterations, num_particles);

			free(cpu_data);
		}
		printf("\n");
	}
	printf("\n\n");
}

void test_all_on_GPU()
{
	printf("particle size;16 block size;32 block size;64 block size;128 block size;256 block size;\n");
	size_t num_iterations = 5000;
	for (size_t num_particles = 1e4; num_particles <= 1e5; num_particles += 1e4)
	{
		printf("%zd;", num_particles);
		for (size_t block_size = 16; block_size <= 256; block_size *= 2)
		{
			Particle* gpu_data = (Particle*)malloc(num_particles * sizeof(Particle));

			time_t seed = time(NULL);
			setup_data(gpu_data, num_particles, seed);

			execute_all_on_GPU(gpu_data, seed, num_iterations, num_particles, block_size);

			free(gpu_data);
		}
		printf("\n");
	}
	printf("\n\n");
}

void test_one_on_GPU()
{
	printf("particle size;16 block size;32 block size;64 block size;128 block size;256 block size;\n");
	size_t num_iterations = 2000;
	for (size_t num_particles = 1e4; num_particles <= 1e5; num_particles += 1e4)
	{
		printf("%zd;", num_particles);
		for (size_t block_size = 16; block_size <= 256; block_size *= 2)
		{
			Particle* gpu_data = (Particle*)malloc(num_particles * sizeof(Particle));

			time_t seed = time(NULL);
			setup_data(gpu_data, num_particles, seed);

			execute_one_on_GPU(gpu_data, seed, num_iterations, num_particles, block_size);

			free(gpu_data);
		}
		printf("\n");
	}
	printf("\n\n");
}

int main()
{
	test_CPU();
	test_all_on_GPU();
	test_one_on_GPU();
	return 0;
}