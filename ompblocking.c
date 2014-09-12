#include <stdlib.h>
#include <stdio.h>
#include "omp.h"

int convert();
int begin();
int end(int *secs, int *nsecs);

/* change dimension size as needed */
const int dimension = 2048;
const int block = 50;

void do_mult(int block_i, int block_j, int block_k, double *A, double *B,
	    double *C)
{
	int i, j, k;

	for (i=block_i; i < block_i+block; i++)
		for (j=block_j; j < block_j+block; j++)
			for (k=block_k; k < block_k+block; k++)
				C[dimension*i+j] += A[dimension*i+k] *
						    B[dimension*k+j];
} 

int main(int argc, char *argv[])
{
       int sec, nanosec;
       int *secs = &sec;
       int *nsecs = &nanosec;
        begin();


	int i, j, k;
	int block_i, block_j, block_k;
	double *A, *B, *C;
	int nr_blocks = dimension / block;

	A = (double*)malloc(dimension*dimension*sizeof(double));
	B = (double*)malloc(dimension*dimension*sizeof(double));
	C = (double*)malloc(dimension*dimension*sizeof(double));

	/* change 9999 with last 4 digit of People Soft or CougarOne ID */
	srand(6935);

	for(i = 0; i < dimension; i++)
		for(j = 0; j < dimension; j++)
		{
			A[dimension*i+j] = (rand()/(RAND_MAX + 1.0));
			B[dimension*i+j] = (rand()/(RAND_MAX + 1.0));
			C[dimension*i+j] = 0.0;
		}
double start = omp_get_wtime();
#pragma omp parallel for shared(A, B, C) private(i, j, k, block_i, block_j, block_k)
	for(i = 0; i < nr_blocks; i++)
	{
		block_i = i * block;
		for(j = 0; j < nr_blocks; j++)
		{
			block_j = j * block;
#pragma omp parallel for shared(A, B, C, block_i, block_j) private(k, block_k)
			for(k = 0; k < nr_blocks; k++)
			{
				block_k = k * block;
				do_mult(block_i, block_j, block_k, A, B, C);
			}
		}
	}

	/* spot checking -- not very accurate */
	printf("%f\n", C[17]);

	free(A);
	free(B);
	free(C);

        end(secs,nsecs);
	printf("Time is %d.%d\n",sec,nanosec);

	return 0;
}

