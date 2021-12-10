#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <time.h>
#include <inttypes.h>
#include <math.h>

// This is a macro for checking the error variable.
#define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr,"Error: %s\n",clGetErrorString(err));

// A errorCode to string converter (forward declaration)
const char* clGetErrorString(int);

#define TPB 256
#define ARRAY_SIZE 1000000
#define B (ARRAY_SIZE + TPB - 1) / TPB

typedef struct {
	float x, y, z;
} float3;

typedef struct {
	float3 position, velocity;
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
		list[i].position.x = rand() / (float)RAND_MAX;
		list[i].position.y = rand() / (float)RAND_MAX;
		list[i].position.z = rand() / (float)RAND_MAX;

		list[i].velocity.x = rand() / (float)RAND_MAX;
		list[i].velocity.y = rand() / (float)RAND_MAX;
		list[i].velocity.z = rand() / (float)RAND_MAX;
	}
}

void execute_on_CPU(Particle* particles, const size_t number_of_iterations, const size_t number_of_particles)
{
	printf("Computing particles on the CPU...\n");

	uint64_t start = time_ns();
	float dt = 1.0f;

	for (size_t j = 0; j < number_of_iterations; j++)
	{
		for (size_t i = 0; i < number_of_particles; i++)
		{
			float acc = (j % 100 == 0) ? 0.1f : 0.0f;
			particles[i].velocity.x += acc;
			particles[i].velocity.y += acc;
			particles[i].velocity.z += acc;

			particles[i].position.x += particles[i].velocity.x * dt;
			particles[i].position.y += particles[i].velocity.y * dt;
			particles[i].position.z += particles[i].velocity.z * dt;
		}
	}

	uint64_t end = time_ns() - start;
	printf("Done! Time elapsed: (ns) = %I64u\t (s): %f\n\n", end, (float)end / 1e9);
}

const char* execute_kernel = " \
typedef struct { \
float x, y, z; \
} _float3; \
 \
typedef struct {	\
	_float3 position, velocity; \
} Particle; \
			\
__kernel \
void saxpy(__global Particle* particles, int array_size, int number_of_iterations) { \
	int i = get_global_id(0);      \
	float dt = 1.0f;		       \
	if (i >= array_size) return;   \
								   \
	for (int j = 0; j < number_of_iterations; j++) \
	{ \
		float acc = (j % 100 == 0) ? 0.1f : 0.0f; \
		particles[i].velocity.x += acc; \
		particles[i].velocity.y += acc; \
		particles[i].velocity.z += acc; \
										\
		particles[i].position.x += particles[i].velocity.x * dt; \
		particles[i].position.y += particles[i].velocity.y * dt; \
		particles[i].position.z += particles[i].velocity.z * dt; \
	} \
} \
";

void execute_with_opencl(Particle* particles, const size_t number_of_iterations, const size_t number_of_particles, const size_t group_size)
{
	cl_platform_id* platforms; 
	cl_uint     n_platform;

	// Find OpenCL Platforms
	cl_int err = clGetPlatformIDs(0, NULL, &n_platform); CHK_ERROR(err);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * n_platform);
	err = clGetPlatformIDs(n_platform, platforms, NULL); CHK_ERROR(err);

	// Find and sort devices
	cl_device_id* device_list; cl_uint n_devices;
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &n_devices); CHK_ERROR(err);
	device_list = (cl_device_id*)malloc(sizeof(cl_device_id) * n_devices);
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL); CHK_ERROR(err);

	// Create and initialize an OpenCL context
	cl_context context = clCreateContext(NULL, n_devices, device_list, NULL, NULL, &err); CHK_ERROR(err);

	// Create a command queue
	cl_command_queue cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err); CHK_ERROR(err);

	printf("Computing SAXPY with OpenCL...\n");
	uint64_t start = time_ns();

	size_t array_buffer = number_of_particles * sizeof(Particle);

	cl_mem particles_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, array_buffer, NULL, &err); CHK_ERROR(err);

	err = clEnqueueWriteBuffer(cmd_queue, particles_dev, CL_TRUE, 0, array_buffer, particles, 0, NULL, NULL); CHK_ERROR(err);

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&execute_kernel, NULL, &err); CHK_ERROR(err);
	err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL); CHK_ERROR(err);

	cl_kernel kernel = clCreateKernel(program, "saxpy", &err); CHK_ERROR(err);

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&particles_dev); CHK_ERROR(err);
	err = clSetKernelArg(kernel, 1, sizeof(int), (void*)&((int)number_of_particles)); CHK_ERROR(err);
	err = clSetKernelArg(kernel, 2, sizeof(int), (void*)&((int)number_of_iterations)); CHK_ERROR(err);

	size_t workgroup_size = group_size;
	size_t n_workitem = (number_of_particles / group_size + 1) * group_size;

	err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &n_workitem, &workgroup_size, 0, NULL, NULL); CHK_ERROR(err);

	err = clEnqueueReadBuffer(cmd_queue, particles_dev, CL_TRUE, 0, array_buffer, particles, 0, NULL, NULL); CHK_ERROR(err);

	uint64_t end = time_ns() - start;
	printf("Done! Time elapsed: (ns) = %I64u\t (s): %f\n\n", end, (float)end / 1e9);

	err = clFlush(cmd_queue); CHK_ERROR(err);
	err = clFinish(cmd_queue); CHK_ERROR(err);

	// Finally, release all that we have allocated.
	err = clReleaseCommandQueue(cmd_queue); CHK_ERROR(err);
	err = clReleaseContext(context); CHK_ERROR(err);
	free(platforms);
	free(device_list);
}

const char* execute_timestamp_kernel = " \
typedef struct { \
float x, y, z; \
} _float3; \
 \
typedef struct {	\
	_float3 position, velocity; \
} Particle; \
			\
__kernel \
void saxpy(__global Particle* particles, int array_size, int iteration) { \
	int i = get_global_id(0);      \
	float dt = 1.0f;		       \
	if (i >= array_size) return;   \
								   \
	float acc = (iteration % 100 == 0) ? 0.1f : 0.0f; \
	particles[i].velocity.x += acc; \
	particles[i].velocity.y += acc; \
	particles[i].velocity.z += acc; \
									\
	particles[i].position.x += particles[i].velocity.x * dt; \
	particles[i].position.y += particles[i].velocity.y * dt; \
	particles[i].position.z += particles[i].velocity.z * dt; \
} \
";

void execute_with_opencl_with_repeat(Particle* particles, const size_t number_of_iterations, const size_t number_of_particles, const size_t group_size)
{
	cl_platform_id* platforms;
	cl_uint     n_platform;

	// Find OpenCL Platforms
	cl_int err = clGetPlatformIDs(0, NULL, &n_platform); CHK_ERROR(err);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * n_platform);
	err = clGetPlatformIDs(n_platform, platforms, NULL); CHK_ERROR(err);

	// Find and sort devices
	cl_device_id* device_list; cl_uint n_devices;
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &n_devices); CHK_ERROR(err);
	device_list = (cl_device_id*)malloc(sizeof(cl_device_id) * n_devices);
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL); CHK_ERROR(err);

	// Create and initialize an OpenCL context
	cl_context context = clCreateContext(NULL, n_devices, device_list, NULL, NULL, &err); CHK_ERROR(err);

	// Create a command queue
	cl_command_queue cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err); CHK_ERROR(err);

	printf("Computing SAXPY with OpenCL with copy repetition...\n");
	uint64_t start = time_ns();

	size_t array_buffer = number_of_particles * sizeof(Particle);

	cl_mem particles_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, array_buffer, NULL, &err); CHK_ERROR(err);

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&execute_timestamp_kernel, NULL, &err); CHK_ERROR(err);
	err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL); CHK_ERROR(err);

	cl_kernel kernel = clCreateKernel(program, "saxpy", &err); CHK_ERROR(err);

	for (size_t iteration = 0; iteration < number_of_iterations; iteration++)
	{
		err = clEnqueueWriteBuffer(cmd_queue, particles_dev, CL_TRUE, 0, array_buffer, particles, 0, NULL, NULL); CHK_ERROR(err);

		err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&particles_dev); CHK_ERROR(err);
		err = clSetKernelArg(kernel, 1, sizeof(int), (void*)&((int)number_of_particles)); CHK_ERROR(err);
		err = clSetKernelArg(kernel, 2, sizeof(int), (void*)&((int)iteration)); CHK_ERROR(err);

		size_t workgroup_size = group_size;
		size_t n_workitem = (number_of_particles / group_size + 1) * group_size;

		err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &n_workitem, &workgroup_size, 0, NULL, NULL); CHK_ERROR(err);

		err = clEnqueueReadBuffer(cmd_queue, particles_dev, CL_TRUE, 0, array_buffer, particles, 0, NULL, NULL); CHK_ERROR(err);
	}

	uint64_t end = time_ns() - start;
	printf("Done! Time elapsed: (ns) = %I64u\t (s): %f\n\n", end, (float)end / 1e9);

	err = clFlush(cmd_queue); CHK_ERROR(err);
	err = clFinish(cmd_queue); CHK_ERROR(err);

	// Finally, release all that we have allocated.
	err = clReleaseCommandQueue(cmd_queue); CHK_ERROR(err);
	err = clReleaseContext(context); CHK_ERROR(err);
	free(platforms);
	free(device_list);
}

int equals(const float3 a, const float3 b)
{
	const double margin = 1e-6;
	return fabs(a.x - b.x) < margin && fabs(a.y - b.y) < margin && fabs(a.z - b.z) < margin;
	//return a.x - b.x == 0 && a.y - b.y == 0 && a.z - b.z == 0;
}

int main(int argc, char* argv[]) {
	if (argc != 4)
	{
		printf("Not enough arguments <number of particles> <number of iterations> <block size (threads per block)>...\n");
		return -1;
	}
	size_t num_particles = strtoull(argv[1], NULL, 10);
	size_t num_iterations = strtoull(argv[2], NULL, 10);
	size_t block_size = strtoull(argv[3], NULL, 10);

	printf("Execution in progress with setup:\n\tnumber of particles: %I64u, number of iterations: %I64u, block size (threads per block): %I64u...\n", num_particles, num_iterations, block_size);

	Particle* cpu_data = (Particle*)malloc(num_particles * sizeof(Particle));
	Particle* gpu_data = (Particle*)malloc(num_particles * sizeof(Particle));
	Particle* gpu_data_simple = (Particle*)malloc(num_particles * sizeof(Particle));

	if (cpu_data == NULL || gpu_data == NULL) return -1;

	time_t seed = time(NULL);
	setup_data(cpu_data, num_particles, seed);
	setup_data(gpu_data, num_particles, seed);
	setup_data(gpu_data_simple, num_particles, seed);

	execute_on_CPU(cpu_data, num_iterations, num_particles);
	execute_with_opencl(gpu_data, num_iterations, num_particles, block_size);
	execute_with_opencl_with_repeat(gpu_data_simple, num_iterations, num_particles, block_size);

	int same = 1;
	for (size_t i = 0; i < num_particles; i++)
	{
		float3 c_v = cpu_data[i].velocity;
		float3 c_p = cpu_data[i].position;
		float3 g_v = gpu_data[i].velocity;
		float3 g_p = gpu_data[i].position;
		if (!equals(c_p, g_p) || !equals(c_v, g_v))
		{
			same = 0;
			break;
		}
	}

	printf("\t%s!\n", same ? "Correct" : "Failed!");

	free(cpu_data);
	free(gpu_data);
	free(gpu_data_simple);

	return 0;
}

// The source for this particular version is from: https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char* clGetErrorString(int errorCode) {
	switch (errorCode) {
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
	case -69: return "CL_INVALID_PIPE_SIZE";
	case -70: return "CL_INVALID_DEVICE_QUEUE";
	case -71: return "CL_INVALID_SPEC_ID";
	case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
	case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
	case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
	case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
	case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
	case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
	case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
	case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
	case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
	case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
	case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
	case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
	case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
	case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
	case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
	case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
	case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
	case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
	default: return "CL_UNKNOWN_ERROR";
	}
}