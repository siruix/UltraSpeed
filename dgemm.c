#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <CL/cl.h>

//#include "acml.h"

const int dimension = 2048;



char* readKernel(char *kernelFile);
int convert();
int begin();
int end(int *secs, int *nsecs);


int main(int argc, char *argv[])
{
	int i, j;
	cl_int err;
	cl_platform_id platform;
	cl_context_properties properties[3];
	cl_context context;
	size_t deviceListSize;
	cl_device_id *devices;
	cl_command_queue commandQueue;
	cl_mem bufferA, bufferB, bufferC;
	cl_program program;
	cl_kernel kernel;
	cl_double *A, *B, *C;
	cl_event event;
	char *source;
	char *fileName = "dgemm.cl";
	const size_t dimensions[2] = {(size_t)dimension, (size_t)dimension};
	char deviceExtensions[8192];

/* add global timer here */
	int g_sec, g_nanosec;
	int *g_secs = &g_sec;
	int *g_nsecs = &g_nanosec;
	g_begin();

	A = (cl_double*)malloc(dimension*dimension*sizeof(cl_double));
	B = (cl_double*)malloc(dimension*dimension*sizeof(cl_double));
	C = (cl_double*)malloc(dimension*dimension*sizeof(cl_double));

	/* change 9999 with last 4 digit of People Soft or CougarOne ID */
	srand(6935);

	for(i = 0; i < dimension; i++)
		for(j = 0; j < dimension; j++)
		{
			A[dimension*i+j] = (rand()/(RAND_MAX + 1.0));
			B[dimension*i+j] = (rand()/(RAND_MAX + 1.0));
			C[dimension*i+j] = 0.0;
		}

	err = clGetPlatformIDs(1, &platform, NULL);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clGetPlatformIDs failed\n");
		exit(EXIT_FAILURE);
	}

	properties[0] = (cl_context_properties)CL_CONTEXT_PLATFORM;
	properties[1] = (cl_context_properties)platform;
	properties[2] = (cl_context_properties)0;

	context = clCreateContextFromType(properties, CL_DEVICE_TYPE_CPU,
					  NULL, NULL, &err);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clCreateContextFromType failed\n");
		exit(EXIT_FAILURE);
	}

	err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL,
			       &deviceListSize);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clGetContextInfo failed\n");
		exit(EXIT_FAILURE);
	}

	devices = (cl_device_id*)malloc(deviceListSize);
	if(devices == NULL)
	{
		fprintf(stderr, "device list allocate failed\n");
		exit(EXIT_FAILURE);
	}

	err = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceListSize,
			       devices, NULL);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clGetContextInfo failed\n");
		exit(EXIT_FAILURE);
	}

	err = clGetDeviceInfo(devices[0], CL_DEVICE_EXTENSIONS,
			      sizeof(deviceExtensions), deviceExtensions,
			      NULL);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clGetDeviceInfo failed\n");
		exit(EXIT_FAILURE);
	}

	if(!strstr(deviceExtensions, "cl_khr_fp64"))
	{
		if(!strstr(deviceExtensions, "cl_amd_fp64"))
		{
			fprintf(stderr, "double precision not supported\n");
			exit(EXIT_FAILURE);
		}
	}
/* add CL_QUEUE_PROFILING_ENABLE for timing in GPU*/
	commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clCreateCommandQueue failed\n");
		exit(EXIT_FAILURE);
	}

	bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY,
				 dimension*dimension*sizeof(cl_double),
				 NULL, &err);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clCreateBuffer bufferA failed\n");
		exit(EXIT_FAILURE);
	}

	bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY,
				 dimension*dimension*sizeof(cl_double),
				 NULL, &err);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clCreateBuffer bufferB failed\n");
		exit(EXIT_FAILURE);
	}

	bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				 dimension*dimension*sizeof(cl_double),
				 NULL, &err);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clCreateBuffer bufferC failed\n");
		exit(EXIT_FAILURE);
	}
/*add timer for memory transfer*/
cl_event event_write2deviceA, event_write2deviceB;
cl_ulong startTime_w2d, endTime_w2d;
	err = clEnqueueWriteBuffer(commandQueue, bufferA, CL_TRUE, 0,
				   dimension*dimension*sizeof(cl_double), A,
				   0, NULL, &event_write2deviceA);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clEnqueueWriteBuffer A failed\n");
		exit(EXIT_FAILURE);
	}
	err = clEnqueueWriteBuffer(commandQueue, bufferB, CL_TRUE, 0,
				   dimension*dimension*sizeof(cl_double), B,
				   0, NULL, &event_write2deviceB);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clEnqueueWriteBuffer B failed\n");
		exit(EXIT_FAILURE);
	}
clFinish(commandQueue);
clGetEventProfilingInfo(event_write2deviceA, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime_w2d, NULL);
clWaitForEvents(1, &event_write2deviceA);
clWaitForEvents(1, &event_write2deviceB);

clGetEventProfilingInfo(event_write2deviceB, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime_w2d, NULL);
float elapsedTime_w2d_inMilisecond = (endTime_w2d - startTime_w2d) * 1e-6f;
printf("Time for write to device: %f ms\n", elapsedTime_w2d_inMilisecond);

	source = readKernel(fileName);

	program = clCreateProgramWithSource(context, 1, (const char**)&source,
					    NULL, &err);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clCreateProgramWithSource failed\n");
		exit(EXIT_FAILURE);
	}

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clBuildProgram failed\n");
		exit(EXIT_FAILURE);

	}

	kernel = clCreateKernel(program, "dgemm", &err);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clCreateKernel failed\n");
		exit(EXIT_FAILURE);
	}

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufferA);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clSetKernelArg A failed\n");
		exit(EXIT_FAILURE);
	}

	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufferB);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clSetKernelArg B failed\n");
		exit(EXIT_FAILURE);
	}

	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufferC);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clSetKernelArg C failed\n");
		exit(EXIT_FAILURE);
	}

	err = clSetKernelArg(kernel, 3, sizeof(int), (void *)&dimension);
	if(err != CL_SUCCESS)
	{       
		fprintf(stderr, "clSetKernelArg dimension failed\n");
		exit(EXIT_FAILURE);
	}
/* here add timing */
	int sec, nanosec;
	int *secs = &sec;
	int *nsecs = &nanosec;
	cl_ulong startTime, endTime;

	begin();
	err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, dimensions,
				     NULL, 0, NULL, &event);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clEnqueueNDRangeKernel failed\n");
		exit(EXIT_FAILURE);
	}
	clFinish(commandQueue);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
	err = clWaitForEvents(1, &event);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clEnqueueNDRangeKernel failed\n");
		exit(EXIT_FAILURE);
	}
	end(secs, nsecs);
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
	float elapsedTimeInMilisecond = (float)(endTime-startTime)* 1e-6f;
	printf("The GPU Timer is %f ms\n", elapsedTimeInMilisecond);

	printf("The CPU Timer is %d.%d s\n", sec, nanosec);


/* add timer for read from device */
cl_event event_readfromdevice;
cl_ulong startTime_readfromdevice, endTime_readfromdevice;
	err = clEnqueueReadBuffer(commandQueue, bufferC, CL_TRUE, 0,
				  dimension*dimension*sizeof(cl_double), C,
				  0, NULL, &event_readfromdevice);
clFinish(commandQueue);
clGetEventProfilingInfo(event_readfromdevice, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime_readfromdevice, NULL);
clWaitForEvents(1, &event_readfromdevice);
clGetEventProfilingInfo(event_readfromdevice, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime_readfromdevice, NULL);
float elapsedTime_readfromdevice = (float)(endTime_readfromdevice - startTime_readfromdevice) * 1e-6f;
printf("Time for read from device: %f ms\n", elapsedTime_readfromdevice);

	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "clEnqueueReadBuffer C failed\n");
		exit(EXIT_FAILURE);
	}

	/* spot checking -- not very accurate */
//	printf("%f\n", C[17]);
g_end(g_secs, g_nsecs);
printf("The global timer is %d.%d s\n", g_sec, g_nanosec);
	free(A);
	free(B);
	free(C);
	clReleaseContext(context);
	free(devices);
	clReleaseCommandQueue(commandQueue);
	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferB);
	clReleaseMemObject(bufferC);
	clReleaseProgram(program);
	clReleaseKernel(kernel);

	return 0;
}

