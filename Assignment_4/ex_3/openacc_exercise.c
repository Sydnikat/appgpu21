#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <sys/time.h>
#include <errno.h>
#include <unistd.h>

static double time_ns(void)
{
	struct timeval t0;

	gettimeofday(&t0, NULL);

	return (double)(t0.tv_sec) * 1e6 + (double)(t0.tv_usec);
}


int main(int argc, char* argv[]) {
	if (argc != 2)
	{
		printf("Not enough arguments...\n");
		return -1;
	}
	size_t array_size = atoi(argv[1]);

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
	double start = time_ns();

	for (size_t i = 0; i < array_size; i++)
	{
		y[i] += a * x[i];
	}

	double end = time_ns() - start;
	printf("Done! Time elapsed: (ms) = %d\t (s): %f\n\n", (int)end, end / 1e6);

	free(x);
	free(y);

	float* g_x = 0;
	float* g_y = 0;
	a = 1.6f;

	g_x = (float*)malloc(array_size * sizeof(float));
	g_y = (float*)malloc(array_size * sizeof(float));

	for (size_t i = 0; i < array_size; i++) {
		g_x[i] = (float)i;
		g_y[i] = 1.0f;
	}

	printf("Computing SAXPY with OepnACC...\n");
	start = time_ns();

	#pragma acc parallel loop copyin(g_x[0:array_size]) copyout(g_y[0:array_size])
	for (size_t i = 0; i < array_size; i++)
	{
		g_y[i] += a * g_x[i];
	}

	end = time_ns() - start;
	printf("Done! Time elapsed: (ms) = %d\t (s): %f\n\n", (int)end, end / 1e6);

	free(g_x);
	free(g_y);

	return 0;
}
