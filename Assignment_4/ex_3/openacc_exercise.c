#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>
#include <math.h>

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


int main(int argc, char* argv) {
	if (argc != 2)
	{
		printf("Not enough arguments...\n");
		return -1;
	}
	size_t array_size = strtoull(argv[1], NULL, 10);

	float* x = 0;
	float* y = 0;
	float a = 1.5f;

	x = (float*)malloc(array_size * sizeof(float));
	y = (float*)malloc(array_size * sizeof(float));

	for (size_t i = 0; i < array_size; i++) {
		x[i] = (float)i;
		y[i] = 1.0f;
	}

	printf("Computing SAXPY sequentially...\n");
	uint64_t start = time_ns();

	for (size_t i = 0; i < array_size; i++)
	{
		y[i] += a * x[i];
	}

	uint64_t end = time_ns() - start;
	printf("Done! Time elapsed: (ns) = %I64u\t (s): %f\n\n", end, (float)end / 1e9);

	printf("Computing SAXPY with OepnACC...\n");
	uint64_t start = time_ns();

	#pragma acc parallel loop copyin(x[0:array_size], y[0:array_size]) copyout(y[0:array_size])
	for (size_t i = 0; i < array_size; i++)
	{
		y[i] += a * x[i];
	}

	uint64_t end = time_ns() - start;
	printf("Done! Time elapsed: (ns) = %I64u\t (s): %f\n\n", end, (float)end / 1e9);

	free(x);
	free(y);

	return 0;
}
