#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

#include "nlao_ur.h"

#define NR_END 1
#define FREE_ARG char*
#define TINY 1.0e-20
#define PLUS 1.0
#define MINUS -1.0

// **************************
// Utility Routines nrutil.h
// Numerical Recipes
// Function definitions
// **************************

void nrerror(char error_text[])
// Numerical Recipes standard error handler 
{
	fprintf(stderr, "Numerical Recipes run-time error...\n");
	fprintf(stderr, "%s\n", error_text);
	fprintf(stderr, "...now exiting to system...\n");
	exit(1);
}

int *ivector(long nl, long nh)
// allocates an int vector with subscript range v[nl..nh]
{
	int *v;
	v = (int *)malloc((size_t) ((nh - nl + 1 + NR_END)*sizeof(int)));
	if (!v) nrerror("allocation failure in ivector()");
	return v - nl + NR_END;
}

void free_ivector(int *v, long nl, long nh)
// free an int vector allocated with ivector()
{
	free((FREE_ARG) (v + nl - NR_END));
}

double *dvector(long nl, long nh)
// allocates an double vector with subscript range v[nl..nh]
{
	double *v;
	v = (double *)malloc((size_t) ((nh - nl + 1 + NR_END)*sizeof(double)));
	if (!v) nrerror("allocation failure in dvector()");
	return v - nl + NR_END;
}

void free_dvector(double *v, long nl, long nh)
// free an double vector allocated with dvector()
{
	free((FREE_ARG) (v + nl - NR_END));
}

double *dmatrix(long nl, long nh)
// allocates an double matrix with subscript range m[nl..nh]
{
    long nh2 = nh * nh;
	double *m;
	m = (double *)malloc((size_t) ((nh2 - nl + 1 + NR_END)*sizeof(double)));
	if (!m) nrerror("allocation failure in dmatrix()");
	return m - nl + NR_END;
}

void free_dmatrix(double *m, long nl, long nh)
// free an double matrix allocated with dmatrix()
{
	free((FREE_ARG) (m + nl - NR_END));
}

/* Build Linear Equation System of test */

void matrix225(double *a, int dim)
// computes A matrix of Ax=b
{
	int matdim = dim * dim;
    for (int i = 1; i <= dim; i++)
    {
        int ctrlvar = dim * i - dim + i;
        for (int j = 1; j <= dim; j++)
        {
            int index = dim * i - dim + j;
            if (index == ctrlvar)
            {
                a[index] = 3.0;
                if (index == 1) 
                {
                    a[index+1] = -1.0;
                    j = index + 1;
                }
                else if (index == matdim) 
                {
                    a[index-1] = -1.0;
                }
                else 
                {
                    a[index-1] = -1.0;
                    a[index+1] = -1.0;
                    j = index + 1;
                }
            }
            else
            {
                a[index] = 0.0;
            }
        }
    }
    for (int i = 1; i <= dim; i++)
    {
        int index = dim * i - i + 1;
        if (a[index] == 0.0) a[index] = 0.5;
    }
}

/* computes b vector of Ax=b */
void vector225(double *b, int dim)
{
	int l = (int)floor(dim/2);

    b[1] = 2.5;
	b[dim] = 2.5;
	b[l] = 1.0;
	b[l+1] = 1.0;
    
    for (int i = 2; i <= l-1; i++)
    {
        b[i] = 1.5;
	}
    
    for (int i = l+2; i <= dim-1; i++)
    {
        b[i] = 1.5;
	}
}   

/* Matrices and vectors operations */

/* compute inverse of a diagonal matrix */
void invdiagmat(double *invd, double *d, int dim)
{
    int index;
    for (int i = 1; i <= dim; i ++)
    {
        index = dim * i - dim + i;
        invd[index] = 1/d[index];
    }
}

/* Computes A = D+L+U split */

/* obtain the diagonal of a matrix */
/* D <- diag(A) */
void diagmat(double *d, double *a, int dim)
{
    int index;
    for (int i = 1; i <= dim; i ++)
    {
        index = dim * i - dim + i;
        d[index] = a[index];
    }
}

/* obtain the portion lower triangular strict of a matrix */
/* L <- lowtrist(A) */
void lotrmat(double *l, double *a, int dim)
{
    for (int i = 1; i <= dim; i ++)
    {
        for (int j = 1; j <= i-1; j++)
        {
            int index = dim * i - dim + j;
            l[index] = a[index];
        }
    }
}

/* obtain the portion upper triangular strict of a matrix */
/* U <- uptrist(A) */
void uptrmat(double *u, double *a, int dim)
{
    for (int i = 1; i <= dim; i ++)
    {
        for (int j = i+1; j <= dim; j++)
        {
            int index = dim * i - dim + j;
            u[index] = a[index];
        }
    }
}

/* computes error of xs_method approximation */
/* error_method = ||xs-xe|| / || xs || */
double emethod(double *xe, double *xs, int dim)
{
    int n = dim;
    int incx = 1, incy = 1;
    double error_method = 0.0;
    double *error_vec = dvector(1, dim);

    cblas_dcopy(n, xs+1, incx, error_vec+1, incy);        // error_vec <-- xs 
    cblas_daxpy(n, MINUS, xe+1, incx, error_vec+1, incy);  // error_vec <-- error_vec - xe
    
    error_method = cblas_dnrm2(n, error_vec, incx) / cblas_dnrm2(n, xs, incx);

	free_dvector(error_vec, 1, dim);

    return error_method;
}

/************************************/
/*          Output routines         */
/************************************/

void iwritevector(int *a, int dim)
/* writes a int vector */
{
    for (int i = 1; i <= dim ; ++i)
    {
        printf("%d ", a[i]);
        printf("\n");
    }
    printf("\n");
}

void dwritevector(double *a, int dim)
/* writes a double vector */
{
    for (int i = 1; i <= dim ; ++i)
    {
        printf("%f ", a[i]);
        printf("\n");
    }
    printf("\n");
}

void dwritematrix(double *a, int rows, int cols)
/* writes a double matrix */
{
    for (int i = 1; i<=rows; ++i)
    {
        for (int j = 1; j<=cols; ++j)
            printf("%f ", a[rows * i - rows + j]);
	printf("\n");
    }
    printf("\n");
}

void info_method(double error_method, double time, int iter_max)
/* write the relative error, run time and entirely iterations of methods */
{
    printf("BiConjugate Gradient Stabilize Method with CBLAS and LAPACKE\n");
    printf("error: %.4e, time: %.2e s, iteration(s): %d\n", error_method, time, iter_max);
}