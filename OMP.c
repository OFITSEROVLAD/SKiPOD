#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (2*2*2*2*2*2*2*2*2*2*2*2*2*2*2*2+2)
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double w = 0.5;
double eps;

double A [N][N];

void relax();
void init();
void verify(); 

int main(int an, char **as)
{

    double timerOpenMp_start;
    double timerOpenMp_end;
    
    timerOpenMp_start = omp_get_wtime();

    int it;
    init();

    for(it=1; it<=itmax; it++)
    {
        eps = 0.;
        relax();
        if (eps < maxeps) break;
    }

    verify();

    timerOpenMp_end = omp_get_wtime();

    printf("(base) Threads: %d N: %d  Time: %10f   \n\n", omp_get_num_threads() , N, timerOpenMp_end  - timerOpenMp_start);

    return 0;
}

void init()
{ 
    for(j=0; j<=N-1; j++)
    for(i=0; i<=N-1; i++)
    {
        if(i==0 || i==N-1 || j==0 || j==N-1) A[i][j]= 0.;
        else A[i][j]= ( 1. + i + j) ;
    }
} 

void relax()
{

    for(j=1; j<=N-2; j++)
    for(i=1; i<=N-2; i++)
    if ((i + j) % 2 == 1)
    {
        double b;
        b = w*((A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4. - A[i][j]);
        eps =  Max(fabs(b),eps);
        A[i][j] = A[i][j] + b;
    }
    for(j=1; j<=N-2; j++)
    for(i=1; i<=N-2; i++)
    if ((i + j) % 2 == 0)
    {
        double b;
        b = w*((A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4. - A[i][j]);
        A[i][j] = A[i][j] + b;
    }
}

void verify()
{ 
    double s;

    s=0.;
    for(j=0; j<=N-1; j++)
    for(i=0; i<=N-1; i++)
    {
        s=s+A[i][j]*(i+1)*(j+1)/(N*N);
    }
    printf("  S = %f\n",s);
}
