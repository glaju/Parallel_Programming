#include <omp.h>
#include <stdio.h>
#define N 10000
main ()
{
int i;
float a[N], b[N], c[N];
/* Some initialisation */
for (i=0; i < N; i++)
a[i] = b[i] = i * 1.0;
double t1 = omp_get_wtime();
#pragma omp parallel for
for (i=0; i < N; i++)
c[i] = a[i] + b[i];
double t2 = omp_get_wtime();
printf("elapsed time: %g\n", t2-t1);
}
