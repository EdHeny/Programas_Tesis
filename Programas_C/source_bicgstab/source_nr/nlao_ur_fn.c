#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "nlao_ur.h"

#define NR_END 1
#define FREE_ARG char*
#define TINY 1.0e-20
#define MINUS -1.0
#define PLUS 1.0

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
// free an float vector allocated with fvector()
{
	free((FREE_ARG) (v + nl - NR_END));
}

double *dvector(long nl, long nh)
// allocates an double vector with subscript range v[nl..nh]
{
	double *v;
	v = (double *)malloc((size_t) ((nh - nl + 1 + NR_END)*sizeof(double)));
	if (!v) nrerror("allocation failure in ivector()");
	return v - nl + NR_END;
}

void free_dvector(double *v, long nl, long nh)
// free an double vector allocated with dvector()
{
	free((FREE_ARG) (v + nl - NR_END));
}

float **fmatrix(long nrl, long nrh, long ncl, long nch)
/* allocates an float matrix with subscript range m[nrl..nrh][ncl..nch] */
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
/* free an float matrix allocated with fmatrix() */
{
	free((FREE_ARG) (m[nrl] + ncl - NR_END));
	free((FREE_ARG) (m + nrl - NR_END));

}

double **dmatrix(long nrl, long nrh, long ncl, long nch)
/* allocates an double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
	double **m;

	/* allocates pointers to rows */
	m = (double **) malloc ((size_t) ((nrow + NR_END) * sizeof(double *)));
	if (!m) nrerror("allocation failure 1 in matrix()");
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl] = (double *) malloc((size_t)((nrow * ncol + NR_END) * sizeof(double)));
	if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for (i = nrl + 1; i <= nrh; i++) m[i] = m[i-1] + ncol;
	
	/* return pointer to array of pointers to rows */
	return m;
}

void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
/* free an double matrix allocated with dmatrix() */
{
	free((FREE_ARG) (m[nrl] + ncl - NR_END));
	free((FREE_ARG) (m + nrl - NR_END));

}

/* Build Linear Equation System of test */

void matrix225(double **a, int dim)
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

void vector225(double *b, int dim)
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
void ludcmp(double **a, int dim , int *indx, double *d)
{
    int i, imax, j, k;
    double big, dum, sum, temp;
    double *vv;

    vv = dvector(1, dim);
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
    free_dvector(vv, 1, dim);
}

// Solves the set of dim linear equation Ax=B. Here a[1...dim][1...dim] is
// input, no as the matrix A but rather a is LU decomposition, determined
// by the routine ludcmp
void lubksb(double **a, int dim, int *indx, double b[])
{
    int ii = 0, ip, j;
    double sum;

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
void solve(double **a, double *b, int dim)
{
	double d;
	int *indx;
	indx = ivector(1,dim);
	ludcmp(a, dim, indx, &d);
	lubksb(a, dim, indx, b);
	//dwritevector(b, dim);
}

void inverse(double **a, double **y, int dim)
/* Finds the inverse of a matrix column by column */
/* matrix y will now contain the inverse of inverse of the */ 
/* original matrix a which will have been destroyed */
{
    double d, *col;
    int *indx;
    indx = ivector(1, dim);

    /* lu decomposition */ 
    ludcmp(a, dim, indx, &d);

    /* matrix inversion */
    col = dvector(1, dim);

    for (int j = 1; j <= dim; j++)
    {
		for (int i = 1; i <= dim; i++) col[i] = 0.0;
		col[j] = 1.0;
		lubksb(a, dim, indx, col);
		for (int i = 1; i <= dim; i++) y[i][j] = col[i];
    }
}

/* compute inverse of a diagonal matrix */
void invdiagmat(double **invd, double **d, int dim)
/* computes D matrix as a diagonal of A */
{
    for (int i = 1; i <= dim; i++)
    {
        invd[i][i] = 1/d[i][i];
    }
}

/* computes the 2-norm of a vector */
double normvec_2(double *a, int dim)
{
    double val;
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
double normmat_2(double **a, int dim)
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
void copyvec(double *y, double *x, int dim)
{
    for (int i = 1; i <= dim; i++)
    {
        y[i] = x[i];
    }
}

/* computes dot product of two vectors */
/* 〈x, y〉  */
double dotvec(double *x, double *y, int dim)
{
    double a = 0.0;
    for (int i = 1; i <= dim; i++)
    {
        a += x[i] * y[i];
    }
    return a;
}

/* computes vector addition */
/* z <- y + alpha*x */
void vecadd(double *z, double *y, double alpha, double *x, int dim)
{
    for (int i = 1; i <= dim; i++)
    {
        z[i] = y[i] + alpha*x[i];

    }
}

/* computes scalar-vector multiplication */
/* y <- alpha*x */
void scavecmul(double *y, double alpha, double *x, int dim)
{
    for (int i = 1; i <= dim; i++)
    {
        y[i] = alpha * x[i];
    }
}

/* computes matrix-vector multiplication */
/* y <- alpha*A*x */
void matvecmul(double *y, double alpha, double **a, double *x, int row, int column)
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
void scamatmul(double **b, double alpha, double **a, int dim)
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
void matadd(double **c, double **b, double alpha, double **a, int dim)
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
double emethod(double *xe, double *xs, int dim)
{
    double error_method = 0.0;
    double *error_vec = dvector(1, dim);

    vecadd(error_vec, xs, MINUS, xe, dim);
    error_method = normvec_2(error_vec, dim) / normvec_2(xs, dim);

	free_dvector(error_vec, 1, dim);

    return error_method;
}

/* Computes A = D+L+U split */

/* obtain the diagonal of a matrix */
/* D <- diag(A) */
void diagmat(double **d, double **a, int dim)
/* computes D matrix as a diagonal of A */
{
    for (int i = 1; i <= dim; i++)
    {
        d[i][i] = a[i][i];
    }
}

/* obtain the portion lower triangular strict of a matrix */
/* L <- lowtrist(A) */
void lotrmat(double **l, double **a, int dim)
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
void uptrmat(double **u, double **a, int dim)
{
    for (int i = 1; i <= dim; i++)
    {
        for (int j = i+1; j<= dim; j++)
		{
			u[i][j] = a[i][j];
		}
    }
}

/* Functions for GMRES Method */

/* get the ith column of a matrix and put in a vector */
void mat2vec(double **a, double *x, int index, int dim)
{
    for (int k = 1; k <= dim; k++) x[k] = a[k][index]; 
}

/* put the elements of a vector in the ith colum of a matrix */
void vec2mat(double *x, double **a, int index, int dim)
{
    for (int k = 1; k <= dim; k++) a[k][index] = x[k]; 
}

/* compute x tal que U*x = b, where U is a upper triangular matrix */
void backward_substitution(double **u, double *b, double *x, int dim)
{
    x[dim] = b[dim] / u[dim][dim];
    for (int i = dim-1; i >= 1; i--)
    {
        double s = 0.0;
        for (int j = i+1; j <= dim; j++) s += u[i][j]*x[j];
        x[i] = (b[i] -s) / u[i][i];
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

void dwritematrix(double **a, int rows, int cols)
/* writes a double matrix */
{
    for (int i = 1; i <= rows; ++i)
    {
		for (int j = 1; j <= cols; ++j)
			printf("%f ", a[i][j]);
		printf("\n");
    }
    printf("\n");
}

void info_method(double error_method, double time, int iter_max)
/* write the relative error, run time and entirely iterations of methods */
{
    printf("BiConjugate Gradient Stabilized Method\n");
    printf("error: %.4e, time: %.2e s, iteration(s): %d\n", error_method, time, iter_max);
}