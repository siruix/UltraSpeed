
#if defined(cl_khr_fp64)
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
  #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void dgemm(__global double *A, __global double *B, __global double *C,
		    int dimension)
{
	int i;
	int x = get_global_id(0);
	int y = get_global_id(1);
	double tmp = 0.0;

	for (i = 0; i < dimension; i++)
	{
		tmp += A[dimension * y + i] * B[dimension * i + x];
	}

	C[dimension * y + x] = tmp;
}

