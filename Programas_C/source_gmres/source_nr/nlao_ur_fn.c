#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "nlao_ur.h"

#define NR_END 1
#define FREE_ARG char*
#define TINY 1.0e-20
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

float *fvector(long nl, long nh)
// allocates an float vector with subscript range v[nl..nh]
{
	float *v;
	v = (float *)malloc((size_t) ((nh - nl + 1 + NR_END)*sizeof(float)));
	if (!v) nrerror("allocation failure in ivector()");
	return v - nl + NR_END;
}

void free_fvector(float *v, long nl, long nh)
// free an int vector allocated with ivector()
{
	free((FREE_ARG) (v + nl - NR_END));
}

float **fmatrix(long nrl, long nrh, long ncl, long nch)
/* allocates an int vector with subscript range v[nl..nh] */
{
	long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
	float **m;

	/* allocates pointers to rows */
	m = (float **) malloc ((size_t) ((nrow + NR_END) * sizeof(float *)));
	if (!m) nrerror("allocation failure 1 in matrix()");
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl] = (float *) malloc((size_t)((nrow * ncol + NR_END) * sizeof(float)));
	if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for (i = nrl + 1; i <= nrh; i++) m[i] = m[i-1] + ncol;
	
	/* return pointer to array of pointers to rows */
	return m;
}

void free_fmatrix(float **m, long nrl, long nrh, long ncl, long nch)
/* free an int matrix allocated with imatrix() */
{
	free((FREE_ARG) (m[nrl] + ncl - NR_END));
	free((FREE_ARG) (m + nrl - NR_END));

}

/* Build Linear Equation System of test */

void matrix225(float **a, int dim)
// computes A matrix of Ax=b
{
    for (int i = 1; i <= dim; i++)
    {
    	for (int j = 1; j <= dim; j++)
    	{
    		if ( i == j )
    		{
    			a[i][j] = 3.;
    			
    			if ( i >= 2 )
    			{
    				a[i-1][j] = -1.0;
    			}
    			if ( j >= 2 )
    			{
    				a[i][j-1] = -1.0;
    			}
    		}
    		else
    		{
    			a[i][j] = 0.0;
    		}

    		if (j == (dim-i+1) && a[i][j] == 0)
    		{
    			a[i][j] = 0.5;
    		}
    	}
    }
}

void vector225(float *b, int dim)
// computes b vector of Ax=b
{
	int l = (int)floor(dim/2);
    
    for (int i = 1; i <= dim; i++)
    {
    	b[i] = 1.5;
    	b[1] = 2.5;
	    b[dim] = 2.5;
	    b[l] = 1.0;
	    b[l+1] = 1.0;
	 }
}   

/* Matrices and vectors operations */

/* LU Decomposition routines */
// Given a matrix a[1...dim][1...dim], this routine replaces it by the LU
// decomposition of a row wise permutation of itself
void ludcmp(float **a, int dim , int *indx, float *d)
{
    int i, imax, j, k;
    float big, dum, sum, temp;
    float *vv;

    vv = fvector(1, dim);
    *d = 1.0;
    for (i = 1; i <= dim; i++)
    {
	    big = 0.0;
	    for (j = 1; j <= dim; j++)
            if ((temp = fabs(a[i][j])) > big ) big = temp;
	    if (big == 0.0) nrerror("Singular matrix in routine ludcmp");
	    vv[i] = 1.0/big;
    }
    for (j = 1; j <= dim; j++)
    {
	    for (i = 1; i < j; i++)
	    {
            sum = a[i][j];
	        for (k = 1; k < i ; k++) sum -= a[i][k]*a[k][j];
	        a[i][j] = sum;
	    }
	    big = 0.0;
	    for (i = j; i <= dim; i++)
	    {
	        sum = a[i][j];
	        for (k = 1; k < j ; k++)
	            sum -= a[i][k] * a[k][j];
	        a[i][j] = sum;
	        if ((dum = vv[i] * fabs(sum)) >= big)
	        {
	            big = dum;
		        imax = i;
	        }
	    }

    	if (j != imax)
	    {
            for (k = 1; k <= dim; k++)
	        {
	            dum = a[imax][k];
		        a[imax][k] = a[j][k];
		        a[j][k] = dum;
	        }
	        *d = -(*d);
	        vv[imax] = vv[j];
	    }
	    indx[j] = imax;
	    if (a[j][j] == 0.0) a[j][j] = TINY;
	    if (j != dim)
	    {
	        dum = 1.0/(a[j][j]);
	        for (i = j + 1; i <= dim; i++) a[i][j] *= dum;
	    }
    }	
    free_fvector(vv, 1, dim);
}

// Solves the set of dim linear equation Ax=B. Here a[1...dim][1...dim] is
// input, no as the matrix A but rather a is LU decomposition, determined
// by the routine ludcmp
void lubksb(float **a, int dim, int *indx, float b[])
{
    int ii = 0, ip, j;
    float sum;

    for (int i = 1; i <= dim; i++)
    {
	    ip = indx[i];
	    sum = b[ip];
        b[ip] = b[i];
	    if (ii)
	        for (j = ii; j <= i-1; j++) sum -= a[i][j] * b[j];
	    else if (sum) ii = i;
	    b[i] = sum;
    }
	
    for (int i = dim; i >= 1; i--)
    {
	    sum = b[i];
	    for (j = i + 1; j <= dim; j++) sum -= a[i][j] * b[j];
	    b[i] = sum/a[i][i];
    }
}


/* Solves using lu decomposition and backward substitution */
/* solve the set of n linear equation Ax=b */
void solve(float **a, float *b, int dim)
{
	float d;
	int *indx;
	indx = ivector(1,dim);
	ludcmp(a, dim, indx, &d);
	lubksb(a, dim, indx, b);
	//writevector(b, dim);
}

void inverse(float **a, float **y, int dim)
/* Finds the inverse of a matrix column by column */
/* matrix y will now contain the inverse of inverse of the */ 
/* original matrix a which will have been destroyed */
{
    float d, *col;
    int *indx;
    indx = ivector(1, dim);

    /* lu decomposition */ 
    ludcmp(a, dim, indx, &d);

    /* matriz inversion */
    col = fvector(1, dim);

    for (int j = 1; j <= dim; j++)
    {
	    for (int i = 1; i <= dim; i++) col[i] = 0.0;
	    col[j] = 1.0;
	    lubksb(a, dim, indx, col);
	    for (int i = 1; i <= dim; i++) y[i][j] = col[i];
    }
}

/* compute inverse of a diagonal matrix */
void invdiagmat(float **invd, float **d, int dim)
/* computes D matrix as a diagonal of A */
{
    for (int i = 1; i <= dim; i++)
    {
        invd[i][i] = 1/d[i][i];
    }
}

/* computes the 2-norm of a vector */
double normvec_2(float *a, int dim)
{
    float val;
    double sum = 0, norm;

    for (int i = 1; i <= dim; i++)
    {
	    val = a[i] * a[i];
	    sum += val;
    }

    norm = sqrt(sum);
    return norm;
}

/* computes the 2-norm of a matrix */
double normmat_2(float **a, int dim)
{
    double val, norm;
    double sum = 0;

    for (int i = 1; i <= dim; i++)
    {
	    for (int j = 1; j<= dim; j++)
	    {
            val = a[i][j] * a[i][j];
	        sum += val;
        }
    }

    norm = sqrt(sum);
    return norm;
}

/* copy a vectors to other vector */
/* y <- x */
void copyvec(float *y, float *x, int dim)
{
    for (int i = 1; i <= dim; i++)
    {
        y[i] = x[i];
    }
}

/* computes dot product of two vectors */
/* 〈x, y〉  */
float dotvec(float *x, float *y, int dim)
{
    float a = 0.0;
    for (int i = 1; i <= dim; i++)
    {
        a += x[i] * y[i];
    }
    return a;
}

/* computes vector addition */
/* z <- y + alpha*x */
void vecadd(float *z, float *y, float alpha, float *x, int dim)
{
    for (int i = 1; i <= dim; i++)
    {
        z[i] = y[i] + alpha*x[i];

    }
}

/* computes scalar-vector multiplication */
/* y <- alpha*x */
void scavecmul(float *y, float alpha, float *x, int dim)
{
    for (int i = 1; i <= dim; i++)
    {
        y[i] = alpha * x[i];
    }
}

/* computes matrix-vector multiplication */
/* y <- alpha*A*x */
void matvecmul(float *y, float alpha, float **a, float *x, int row, int column)
/* computes matrix-vector multiplication as y = A*x */
{
    for (int i = 1; i <= row; i++)
    {
        y[i] = 0.0;
        for (int j = 1; j<= column; j++)
	    {
            y[i] += alpha * a[i][j] * x[j];
	    }
    }
}

/* computes scalar-matrix multiplication */
/* B <- alpha*A */
void scamatmul(float **b, float alpha, float **a, int dim)
{
    for (int i = 1; i <= dim; i++)
    {
        for (int j = 1; j <= dim; j++)
        {
            b[i][j] = alpha * a[i][j];
        }
    }
}

/* computes matrix addition */
/* C <- B + alpha*A */
void matadd(float **c, float **b, float alpha, float **a, int dim)
{
    for (int i = 1; i <= dim; i++)
    {
        for (int j = 1; j<= dim; j++)
		{
            c[i][j] = b[i][j] + alpha*a[i][j];
		}
    }
}

/* computes error of xs_method approximation */
/* error_method = ||xs-xe|| / || xs || */
float emethod(float *xe, float *xs, int dim)
{
    float error_method = 0.0;
    float *error_vec = fvector(1, dim);

    vecadd(error_vec, xs, MINUS, xe, dim);
    error_method = normvec_2(error_vec, dim) / normvec_2(xs, dim);

	free_fvector(error_vec, 1, dim);

    return error_method;
}

/* Computes A = D+L+U split */

/* obtain the diagonal of a matrix */
/* D <- diag(A) */
void diagmat(float **d, float **a, int dim)
/* computes D matrix as a diagonal of A */
{
    for (int i = 1; i <= dim; i++)
    {
        d[i][i] = a[i][i];
    }
}

/* obtain the portion lower triangular strict of a matrix */
/* L <- lowtrist(A) */
void lotrmat(float **l, float **a, int dim)
{
    for (int i = 1; i <= dim; i++)
    {
        for (int j = 1; j<= i-1; j++)
		{
			l[i][j] = a[i][j];
		}
    }
}

/* obtain the portion upper triangular strict of a matrix */
/* U <- uptrist(A) */
void uptrmat(float **u, float **a, int dim)
{
    for (int i = 1; i <= dim; i++)
    {
        for (int j = i+1; j<= dim; j++)
		{
			u[i][j] = a[i][j];
		}
    }
}

void mat2vec(float **a, float *x, int index, int dim)
{
    for (int k = 1; k <= dim; k++) x[k] = a[k][index]; 
}

void vec2mat(float *x, float **a, int index, int dim)
{
    for (int k = 1; k <= dim; k++) a[k][index] = x[k]; 
}

void vec2vec(float *x, float *y, int index)
{
    for (int i = 1; i <= index; i++) y[i] = x[i]; 
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

void fwritematrix(float **a, int rows, int cols)
/* writes a float matrix */
{
    for (int i = 1; i <= rows; ++i)
    {
		for (int j = 1; j <= cols; ++j)
			printf("%f ", a[i][j]);
		printf("\n");
    }
    printf("\n");
}

void info_method(float error_method, float time, int iter_max)
{
    printf("GMRES Method\n");
    printf("error: %.4e, time: %.4e s, iter-max: %d\n", error_method, time, iter_max);
}