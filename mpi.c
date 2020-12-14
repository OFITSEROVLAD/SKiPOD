#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (2*2*2*2*2*2+2)
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j;
double w = 0.5;
double eps;

double A [N][N];

void relax();
void init();
void verify(); 


 
int start_row, last_row, num_rows_per_block, cur_rank, num_procs;
MPI_Request req[4];
MPI_Status status[4];

int main(int argc, char *argv[])
{
    double start;
    double end;

    int it;

    MPI_Init(&argc, &argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); 
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_rank);

    start_row = ((N - 2) / num_procs) * cur_rank + 1;
    last_row = ((N - 2) / num_procs) * (cur_rank + 1) + 1;
    num_rows_per_block = last_row - start_row ;


    start = MPI_Wtime(); 
    init();

    for(it=1; it<=itmax; it++)
    {
        eps = 0.;
        relax();
        if (eps < maxeps) break;
    }

    // All processes have to reached this routine
    MPI_Barrier(MPI_COMM_WORLD) ;

    //Gather all
    if (cur_rank == 0) {
        MPI_Gather(MPI_IN_PLACE, num_rows_per_block * N, MPI_DOUBLE, A[1], num_rows_per_block * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gather(A[start_row], num_rows_per_block * N, MPI_DOUBLE, A[1], num_rows_per_block * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    end = MPI_Wtime();

    if (cur_rank == 0) {
        verify();
        printf("\n\n N: %6.d    Time: %10.4f \n", N, end - start);
    }
    MPI_Finalize();

    return 0;
}


void init()
{ 

    for (i = start_row; i < last_row; ++i) {
        if ( i == 0 || i == N - 1 ) {
            continue ;
        }

        for (j = 1; j <N- 1; ++j) {
            A[i][j] = (1. + i + j);
        }
    }
} 


// int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request * request)
// int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
void relax()
{

    double local_eps = 0.0;
    
    int dl = 0, dr = 0;
    if (cur_rank == 0) { 
        dl = 2;
    }

    if (cur_rank == num_procs - 1) {
        dr = 2; 
    }

    // Share row: 0 -> 1, 1 -> 2, ... , n-2 -> n-1
    if (cur_rank != 0) {
        MPI_Irecv(A[start_row - 1], N, MPI_DOUBLE, cur_rank - 1, 1215, MPI_COMM_WORLD, &req[0]);
    }
    if (cur_rank != num_procs - 1) {
        MPI_Isend(A[last_row - 1], N, MPI_DOUBLE, cur_rank + 1, 1215, MPI_COMM_WORLD, &req[2]); 
    }

    // Share row: n-1 -> n-2, n-2 -> n-3, ... , 1 -> 0
    if (cur_rank != num_procs - 1) {
        MPI_Irecv(A[last_row], N, MPI_DOUBLE, cur_rank + 1, 1216, MPI_COMM_WORLD, &req[3]);
    }   
    if (cur_rank != 0) {
        MPI_Isend(A[start_row], N, MPI_DOUBLE, cur_rank - 1, 1216, MPI_COMM_WORLD, &req[1]); 
    }
    // Waits for all given MPI Requests to complete
    MPI_Waitall(4 - dl - dr , &req[dl], status);

    for (i = start_row + dl; i < last_row - dr; ++i) {
        for ( j = 1 + i % 2 ; j <= N - 2 ; j += 2 ) { 
            double b ;
            b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
            A[i][j] =A[i][j] +b;
            local_eps = Max(fabs(b), local_eps);
        } 
    }

    // Share row: 0 -> 1, 1 -> 2, ... , n-2 -> n-1
    if (cur_rank != 0) {
        MPI_Irecv(A[start_row - 1], N, MPI_DOUBLE, cur_rank - 1, 1215, MPI_COMM_WORLD, &req[0]);
    }
    if (cur_rank != num_procs - 1) {
        MPI_Isend(A[last_row - 1], N, MPI_DOUBLE, cur_rank + 1, 1215, MPI_COMM_WORLD, &req[2]); 
    }

    // Share row: n-1 -> n-2, n-2 -> n-3, ... , 1 -> 0
    if (cur_rank != num_procs - 1) {
        MPI_Irecv(A[last_row], N, MPI_DOUBLE, cur_rank + 1, 1216, MPI_COMM_WORLD, &req[3]);
    }   
    if (cur_rank != 0) {
        MPI_Isend(A[start_row], N, MPI_DOUBLE, cur_rank - 1, 1216, MPI_COMM_WORLD, &req[1]); 
    }
    // Waits for all given MPI Requests to complete
    MPI_Waitall(4 - dl - dr , &req[dl], status);


    for (i = start_row + dl; i < last_row - dr; ++i) {
        for ( j = 1 + (i + 1) % 2 ; j <= N - 2 ; j += 2 ) { 
            double b ;
            b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
            A[i][j] =A[i][j] + b;
        } 
    }

    // All processes have to reached this routine
    MPI_Barrier(MPI_COMM_WORLD);
    // Reduce all local eps to one global
    MPI_Reduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    // Each proc have to know the real max eps, to break from main()
    MPI_Bcast(&eps , 1 , MPI_DOUBLE, 0 , MPI_COMM_WORLD);

}


void verify()
{ 
    double s;

    s=0.;
    for(i=0; i<=N-1; i++)
    for(j=0; j<=N-1; j++)
    {
        s=s+A[i][j]*(i+1)*(j+1)/(N*N);
    }
    printf("  S = %f\n",s);
}


