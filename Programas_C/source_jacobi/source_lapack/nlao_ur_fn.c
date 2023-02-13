#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "nlao_ur.h"

#define NR_END 1
#define FREE_ARG char*
#define TINY 1.0e-20

// **************************
// Utility Routines nrutil.h
// Numerical Recipes
// Function definitions
// **************************

void nrerror(char error_text[])
// Numerical Recipies standard error handler 
{
	fprintf(stderr, "Numerical Recipes run-time error...\n");
	fprintf(stderr, "%s\n", error_text);
	fprintf(stderr, "...now exiting to system...\n");
	exit(1);
}

int *ivector(long nl, long nh)
// allocates an int vector with subscrip range v[nl..nh]
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

float *fvector(long nl, long nh)
// allocates an float vector with subscrip range v[nl..nh]
{
	float *v;
	v = (float *)malloc((size_t) ((nh - nl + 1 + NR_END)*sizeof(float)));
	if (!v) nrerror("allocation failure in fvector()");
	return v - nl + NR_END;
}

void free_fvector(float *v, long nl, long nh)
// free an int vector allocated with ivector()
{
	free((FREE_ARG) (v + nl - NR_END));
}

float *fmatrix(long nl, long nh)
// allocates an float vector with subscrip range v[nl..nh]
{
    long nh2 = nh * nh;
	float *m;
	m = (float *)malloc((size_t) ((nh2 - nl + 1 + NR_END)*sizeof(float)));
	if (!m) nrerror("allocation failure in fmatrix()");
	return m - nl + NR_END;
}

void free_fmatrix(float *m, long nl, long nh)
// free an int vector allocated with fvector()
{
	free((FREE_ARG) (m + nl - NR_END));
}

/* Build Linear Equation System of test */

void matrix225(float *a, int dim)
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
void vector225(float *b, int dim)
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
void invdiagmat(float *invd, float *d, int dim)
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
void diagmat(float *d, float *a, int dim)
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
void lotrmat(float *l, float *a, int dim)
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
void uptrmat(float *u, float *a, int dim)
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

void fwritevector(float *a, int dim)
/* writes a float vector */
{
    for (int i = 1; i <= dim ; ++i)
    {
	    printf("%f ", a[i]);
	    printf("\n");
    }
    printf("\n");
}

void fwritematrix(float *a, int rows, int cols)
/* writes a float matrix */
{
    for (int i = 1; i<=rows; ++i)
    {
        for (int j = 1; j<=cols; ++j)
            printf("%f ", a[rows * i - rows + j]);
	printf("\n");
    }
    printf("\n");
}