/* ********************************************** */
/* *************** SOR Method ******************* */
/* ********************************************** */

#include <lapacke.h>
#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nlao_ur.h"

#define TOL 1.0e-10
#define ITER_MAX 600000
#define PLUS 1.0
#define MINUS -1.0

/* ******************************** */
/*          main function           */      
/* ******************************** */

int main(int argc, char const *argv[])
{

	/* Size of the linear equation system */

	int dim = atoi(argv[1]);

	/* allocate arbitrary-offset vector and matrix of arbitrary lenghts */

	double *a_mat, *b_vec, *xp_vec;

	a_mat = dmatrix(1, dim);
	matrix225(a_mat, dim);

	b_vec = dvector(1, dim);
	vector225(b_vec, dim);

    /* Initial vector */
    xp_vec = dvector(1, dim);

    /* Exact solution */
    double *xe_sol = dvector(1, dim);
    for (int i = 1; i <= dim; i++) xe_sol[i] = 1.0;

	/************************************************/
	/* Computes D, (L+U), T, matrices and f vector 	*/
	/************************************************/

	clock_t start, end;
    double cpu_time_used;

	/* star time count */
	start = clock();

    double omega = atof(argv[2]);
    
    /* matrices */
    double *d_mat, *u_mat, *l_mat, *t_mat;
    
    /* vectors */
	double *f_vec, *xs_vec, *v_vec, *w_vec, *y_vec, *z_vec, *err_vec;

    /* scalar value */
    double err_method = 0.0;
    int iter = 0;

    /* CBLAS parameters */
    // cblas_dcopy(n, x, incx, y, incy);
    // cblas_daxpy(n, alpha, x, incx, y, incy);
    // cblas_sscal(n, alpha, x, incx);
    // cblas_dnorm2(n, x, incx);
    // cblas_dgemv(cblas_layout, cblas_transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
    // cblas_dtrmv(cblas_layout, cblas_uplo, cblas_transa, cblas_diag, n, a, lda, x, incx);

    /* CBLAS variables */
    int n = dim, nmat = dim*dim, lda = dim;
    int incx = 1, incy = 1;

    /* LAPACKE parameters */
    // info = LAPACKE_dtrtri (matrix_layout, uplo, diag, n, a, lda);

    /* Lapacke variables */
    int info = 0;

    f_vec = dvector(1, dim); 
    xs_vec = dvector(1, dim);
    y_vec = dvector(1, dim); 
    z_vec = dvector(1, dim);
    v_vec = dvector(1, dim);
    w_vec = dvector(1, dim);
    err_vec = dvector(1, dim); 

    d_mat = dmatrix(1, dim);
    u_mat = dmatrix(1, dim);
    l_mat = dmatrix(1, dim);
    t_mat = dmatrix(1, dim);

    /* Computes D matrix as diagonal of A */
    diagmat(d_mat, a_mat, dim);

    /* compute U matrix of A matrix */
    uptrmat(u_mat, a_mat, dim);

    /* compute L matrix of A matrix */
    lotrmat(l_mat, a_mat, dim);

    /* compute omega*L matrix as l_mat <-- omega*l_mat */
    cblas_dscal(nmat, omega, l_mat+1, incx);

    /* Computes D+omega*L matrix as t_mat <-- d_mat + omega * l_mat */
    cblas_dcopy(nmat, d_mat+1, incx, t_mat+1, incy); /* t_mat <-- d_mat */
    cblas_daxpy(nmat, PLUS, l_mat+1, incx, t_mat+1, incy); /* t_mat <-- t_mat + l_mat */

    /* compute (1-omega)*D matrix as d_mat <-- (1-omega)*d_mat */
    cblas_dscal(nmat, 1-omega, d_mat+1, incx);

    /* compute omega*U matrix as u_mat <-- omega*u_mat */
    cblas_dscal(nmat, omega, u_mat+1, incx);

    /* Computes inv(D+omega*L) as t_mat <-- inv(t_mat) */
    info = LAPACKE_dtrtri (LAPACK_ROW_MAJOR, 'L', 'N', n, t_mat+1, lda);
    if (info > 0)
    {
        printf( "(D+omega*L)(%i,%i) is exactly zero. The triangular matrix\n", info, info );
        printf( "is singular and its inverse can not be computed.\n" );
        exit( 1 );
    }

    /* Computes omega*inv(D+omega*L)*b */
    cblas_dcopy(n, b_vec+1, incx, f_vec+1, incy);   /* f_vec <-- b_vec */
    cblas_dscal(n, omega, f_vec+1, incx);        /* f_vec <-- omega*f_vec */
    cblas_dtrmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                n, t_mat+1, lda, f_vec+1, incx);     /* f_vec <-- t_mat*f_vec */ 

	/* **************************************** */
    /*                SOR Method                */
    /* **************************************** */

    for (int k = 1; k <= ITER_MAX; k++)
    {

        /* Computes (1-omega)*D*x(k) */
        //cblas_sgemv(CblasRowMajor, CblasNoTrans, dim, dim, 1, d_mat+1, dim, xp_vec+1, 1, 0, y_vec+1, 1); 
        cblas_dcopy(n, xp_vec+1, incx, y_vec+1, incy);          /* y_vec <-- xp_vec */
        cblas_dtrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                n, d_mat+1, lda, y_vec+1, incx);             /* y_vec <-- d_mat*y_vec */

        /* Computes omega*U*x(k) */
        cblas_dcopy(n, xp_vec+1, incx, z_vec+1, incy);          /* z_vec <-- xp_vec */
        cblas_dtrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                n, u_mat+1, lda, z_vec+1, incx);             /* z_vec <-- u_mat*z_vec */
        
        /* Computes (1-omega)*D*x(k) - omega*U*x(k) */
        cblas_dcopy(n, y_vec+1, incx, v_vec+1, incy);           /* v_vec <-- y_vec */
        cblas_daxpy(n, MINUS, z_vec+1, incx, v_vec+1, incy);     /* v_vec <-- v_vec - z_vec */

        /* Computes inv(D+omega*L)*((1-omega)*D*x(k) - omega*U*x(k)) */
        cblas_dtrmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                n, t_mat+1, lda, v_vec+1, incx);             /* v_vec <-- t_mat*v_vec */

        /* Computes x(k+1) = inv(D+omega*L)*((1-omega)*D*x(k) - omega*U*x(k)) + omega*inv(D+omega*L)*b */
        cblas_dcopy(n, f_vec+1, incx, xs_vec+1, incy);          /* xs_vec <-- f_vec */ 
        cblas_daxpy(n, PLUS, v_vec+1, incx, xs_vec+1, incy);     /* xs_vec <-- v_vec + xs_vec */ 

        /* computes error of xs_method approximation */
        /* error_method = ||xs-xe|| / || xs || */
        err_method = emethod(xe_sol, xs_vec, dim);

        /* Stop criterion */
        if (err_method <= TOL) 
        {
            iter = k;
            break;
        }

        cblas_dcopy(n, xs_vec+1, incx, xp_vec+1, incy);         /* xp_vec <-- xs_vec */
    }

	/* finish time count */
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	/* Print final results */
    //dwritevector(xs_vec, dim);
	//printf("error: %f\n", err_method);
	//printf("time: %f\n", cpu_time_used);
	// printf("error: %.12f, time: %f, iter-max: %d\n", err_method, cpu_time_used, iter);
    info_method(err_method, cpu_time_used, iter);
        
    free_dmatrix(a_mat, 1, dim);
    free_dmatrix(d_mat, 1, dim);
    free_dmatrix(u_mat, 1, dim);
    free_dmatrix(l_mat, 1, dim);
    free_dmatrix(t_mat, 1, dim);

    free_dvector(b_vec, 1, dim);
    free_dvector(xp_vec, 1, dim);
    free_dvector(xs_vec, 1, dim);
    free_dvector(f_vec, 1, dim);
    free_dvector(y_vec, 1, dim);
    free_dvector(z_vec, 1, dim);
    free_dvector(w_vec, 1, dim);
    free_dvector(v_vec, 1, dim);
    free_dvector(err_vec, 1, dim);

	return 0;
}