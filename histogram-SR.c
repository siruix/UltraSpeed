#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

const int dimension = 20000*20000;
const int K = 20;
void _debug(rank, str)
{
	printf("rank %d: at %s\n", rank, str);
}
void generateNumbers(int seed, double *A)
{
  int i, j;
  printf("message from function genearteNumbers\n");
  srand(seed);
  for(i = 0; i < dimension; i++)
      A[i] = (rand()/(RAND_MAX + 1.0));
}
void histogram(double *A, int *hist, int rank, int size)
{
	printf("rank %d: function histogram\n", rank);
	int i, j;
	for (i = rank*dimension/size; i < (double)(rank+1)*dimension/size; i++)
	{
		for (j = 0; j < K; j++)
		{
			if ((j*1.0/K <= A[i]) && (A[i] < (j+1)*1.0/K))
				hist[j]++;
		}
	}
	printf("rank %d: end function histogram\n", rank);
}

int main(int argc, char **argv)
{
  int rank, size;
  int i, j;
  double *A;
  double tick, tock, maxVal = -1.0;
  int root_hist[K], hist[K];
  A = (double*)malloc(dimension*sizeof(double));
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
//  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  for (i = 0; i < K; i++)
	hist[i] = 0;
  for(i = 0; i < K; i++)
 	root_hist[i] = 0;
  if(rank == 0)
  {
	  generateNumbers(6935, A);
  }
  if(rank == 0)
  {
  	tick = MPI_Wtime();
 	for (i = 1; i < size; i++)
  		MPI_Send(A, dimension, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
	tock = MPI_Wtime();
  }
  else if (rank != 0)
  {
	MPI_Recv(A, dimension, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	printf("proc %d recv\n", rank);
  }
  if(rank==0)
  printf("Processor %d, distribute to all proc: %f sec\n", rank, tock-tick);
  
  tick = MPI_Wtime();
  histogram(A, hist, rank, size);
  tock = MPI_Wtime();
  if(rank==0)
  printf("Processor %d, local histogram : %f sec\n", rank, tock-tick);
  // Reduction
  if (rank != 0)
  {
	MPI_Send(hist, K, MPI_INT, 0, 0,  MPI_COMM_WORLD);
  }
  else if (rank == 0)
  {
	  tick = MPI_Wtime();
  int hist[size][K];
	  for (i = 1; i < size; i++)
		  MPI_Recv(hist[i], K, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  for (j = 0; j < K; j++)
	  {
		  for (i = 1; i < size; i++)
			  root_hist[j] += hist[i][j];
	  }
   	 tock = MPI_Wtime();
	 printf("MPI_Reduce: %f sec\n", tock-tick);

  }

/*
   // Bcast 
     tick = MPI_Wtime();
    MPI_Bcast(root_hist, K, MPI_INT, 0, MPI_COMM_WORLD);
    tock = MPI_Wtime();
*/
  if(rank == 0)
  {
  	tick = MPI_Wtime();
 	for (i = 1; i < size; i++)
  		MPI_Send(root_hist, K, MPI_INT, i, 0, MPI_COMM_WORLD);
	tock = MPI_Wtime();
  }
  else if (rank != 0)
  {
	MPI_Recv(root_hist, K, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	printf("proc %d recv root_hist\n", rank);
  }
  if(rank == 0)
  {
    printf("MPI_Bcast took %f sec\n", tock-tick);
    for (i = 0; i < K; i++)
	  printf("Processor %d, hist[%d]: %d\n", rank, i, root_hist[i]);
  }
  MPI_Finalize();

  return EXIT_SUCCESS;
}

