/* ********************************************************** */
/* ********** BiConjugate Gradient Stabilize Method ********* */ 
/* ******************** LAPACKE and BLAS ******************** */
/* ********************************************************** */

#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nlao_ur.h"

#define TOL 1.0e-10
#define PLUS 1.0
#define MINUS -1.0

/* ******************************** */
/*          main function           */      
/* ******************************** */

int main(int argc, char const *argv[])
{
	/* Size of the linear equation system */

	int dim = atoi(argv[1]);

	/* allocate arbitrary-offset vector and matrix of arbitrary lengths */

	double *a_mat, *b_vec, *xp_vec;

	a_mat = dmatrix(1, dim);
	matrix225(a_mat, dim);

	b_vec = dvector(1, dim);
	vector225(b_vec, dim);

    /* Initial vector */
    xp_vec = dvector(1, dim);

    /* Exact solution of system test */
    double *xe_sol = dvector(1, dim);
    for (int i = 1; i <= dim; i++) xe_sol[i] = 1.0;
	
	/************************/
	/* Parameters Initials 	*/
	/************************/

	clock_t start, end;
    double cpu_time_used;

	/* star time count */
	start = clock();

	double *xs_method, *rp_vec, *rn_vec, *rast_vec, *dn_vec, *dp_vec; 
    double *s_vec, *adk_vec, *ask_vec, *y_vec, *z_vec; 
    double alpha = 0.0, beta = 0.0, omega = 0.0, err_method = 0.0;
    int iter = 0;

    xs_method = dvector(1, dim);
    rp_vec = dvector(1, dim);
    rn_vec = dvector(1, dim);
    rast_vec = dvector(1, dim);
    dp_vec = dvector(1, dim);
    dn_vec = dvector(1, dim);
    s_vec = dvector(1, dim);
    adk_vec = dvector(1, dim);
    ask_vec = dvector(1, dim);
    y_vec = dvector(1, dim);
    z_vec = dvector(1, dim);

    /* CBLAS parameters */
    // cblas_dcopy(n, x, incx, y, incy);
    // cblas_daxpy(n, alpha, x, incx, y, incy);
    // cblas_ddot (n, x, incx, y, incy);
    // cblas_dscal(n, alpha, x, incx);
    // cblas_dnorm2(n, x, incx);
    // cblas_dgemv(cblas_layout, cblas_transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
    // cblas_dtrmv(cblas_layout, cblas_uplo, cblas_transa, cblas_diag, n, a, lda, x, incx);

    int m = dim, n = dim, alpha_ = 1, lda = dim;
    int incx = 1, incy = 1, beta_ = 0;

    /* **************************************************** */
    /*        BiConjugate Gradient Stabilize Method         */
    /* **************************************************** */

    /* Compute d(0) = r(0) = b - A*x(0) */
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha_, 
            a_mat+1, lda, xp_vec+1, incx, beta_, y_vec+1, incy);      /* y_vec <-- a_mat * xp_vec */
    cblas_dcopy(n, b_vec+1, incx, rp_vec+1, incy);              /* rp_vec <-- b_vec */
    cblas_daxpy(n, MINUS, y_vec+1, incx, rp_vec+1, incy);       /* rp_vec <-- rp_vec - y_vec */ 
    cblas_dcopy(n, rp_vec+1, incx, rast_vec+1, incy);             /* dp_vec <-- rp_vec */
    cblas_dcopy(n, rp_vec+1, incx, dp_vec+1, incy);             /* dp_vec <-- rp_vec */
        
    for (int k = 1; k <= dim; k++)
    {
        /* Compute alpha(k) = <r(k), r*(0)> / <A*d(k), r*(0)> */
        cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha_, 
                a_mat+1, lda, dp_vec+1, incx, beta_, adk_vec+1, incy);  /* adk_vec <-- A*dp_vec */
        alpha = cblas_ddot(n, rp_vec+1, incx, rast_vec+1, incy) / cblas_ddot(n, rast_vec+1, incx, adk_vec+1, incy);       /* alpha <-- dot(rp_vec, rp_vec) / dot(dp_vec, y_vec) */

        /* s(k) <- r(k) - alpha*A*d(k) */
        cblas_dcopy(n, adk_vec+1, incx, y_vec+1, incy); // y_vec <- adk_vec
        cblas_dscal(n, alpha, y_vec+1, incx);
        cblas_dcopy(n, rp_vec+1, incx, s_vec+1, incy); // s_vec <- rp_vec
        cblas_daxpy(n, MINUS, y_vec+1, incx, s_vec+1, incy); // s_vec <- s_vec - y_vec

        /* Compute omega(k) = <A*s(k), s(k)> / <A*s(k), A*s(k)> */
        cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha_, 
                a_mat+1, lda, s_vec+1, incx, beta_, ask_vec+1, incy);  /* ask_vec <-- A*s_vec */
        omega = cblas_ddot(n, ask_vec+1, incx, s_vec+1, incy) / cblas_ddot(n, ask_vec+1, incx, ask_vec+1, incy);       /* omega <-- dot(ask_vec, s_vec) / dot(ask_vec, ask_vec) */

        /* Compute x(k+1) = x(k) + alpha(k)*d(k) + omega(k)*s(k) */
        cblas_dcopy(n, dp_vec+1, incx, y_vec+1, incy);          /* y_vec <-- dp_vec */
        cblas_dscal(n, alpha, y_vec+1, incx);                /* y_vec <-- alpha*y_vec */
        cblas_dcopy(n, s_vec+1, incx, z_vec+1, incy);          /* z_vec <-- s_vec */
        cblas_dscal(n, omega, z_vec+1, incx);                /* z_vec <-- omega*z_vec */
        cblas_dcopy(n, xp_vec+1, incx, xs_method+1, incy);      /* xs_method <-- xp_vec */
        cblas_daxpy(n, PLUS, y_vec+1, incx, xs_method+1, incy); /* xs_method <-- xs_method + y_vec */
        cblas_daxpy(n, PLUS, z_vec+1, incx, xs_method+1, incy); /* xs_method <-- xs_method + z_vec */

        /* computes error method and stop criterion */
        err_method = emethod(xe_sol, xs_method, dim);
        if (err_method <= TOL) 
        {
            iter = k;
            break;
        }

        /* Compute r(k+1) = s(k) - omega(k)*A*s(k) */
        cblas_dcopy(n, ask_vec+1, incx, z_vec+1, incy);           /* z_vec <-- ask_vec */
        cblas_dscal(n, omega, z_vec+1, incx);                /* z_vec <-- omega*z_vec */
        cblas_dcopy(n, s_vec+1, incx, rn_vec+1, incy);         /* rn_vec <-- s_vec */
        cblas_daxpy(n, MINUS, z_vec+1, incx, rn_vec+1, incy);   /* rn_vec <-- rn_vec - z_vec */

        /* Compute beta(k) = (<r(k+1), r*(0)> / <r(k), r*(0)>) * (alpha(k)/omega(k)) */
        beta = (cblas_ddot(n, rn_vec+1, incx, rast_vec+1, incy) / cblas_ddot(n, rp_vec+1, incx, rast_vec+1, incy)) * (alpha/omega);      /* beta <-- (dot(rn_vec, rast_vec) / dot(rp_vec, rast_vec)) * (alpha/omega) */

        /* Compute d(k+1) = r(k+1) + beta(k)*(d(k)-omega*A*d(k)) */
        cblas_dcopy(n, adk_vec+1, incx, z_vec+1, incy);          /* z_vec <-- adk_vec */
        cblas_dscal(n, omega, z_vec+1, incx);                 /* z_vec <-- omega*z_vec */
        cblas_dcopy(n, dp_vec+1, incx, y_vec+1, incy);          /* y_vec <-- dp_vec */
        cblas_daxpy(n, MINUS, z_vec+1, incx, y_vec+1, incy);    /* y_vec <-- y_vec - z_vec */
        cblas_dscal(n, beta, y_vec+1, incx);                 /* y_vec <-- beta*y_vec */
        cblas_dcopy(n, rn_vec+1, incx, dn_vec+1, incy);         /* dn_vec <-- rn_vec */
        cblas_daxpy(n, PLUS, y_vec+1, incx, dn_vec+1, incy);    /* dn_vec <-- dn_vec + z_vec */

        cblas_dcopy(n, rn_vec+1, incx, rp_vec+1, incy);         /* rp_vec <-- rn_vec */
        cblas_dcopy(n, dn_vec+1, incx, dp_vec+1, incy);         /* dp_vec <-- dn_vec */
        cblas_dcopy(n, xs_method+1, incx, xp_vec+1, incy);      /* xp_vec <-- xs_method */
    }

	/* finish time count */
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	
	/* Print final results */
	// dwritevector(xs_method, dim);
	//printf("error: %f\n", err_method );
	//printf("time: %f\n", cpu_time_used);
	// printf("error: %.12f, time: %f, iter-max: %d\n", err_method, cpu_time_used, iter);
    info_method(err_method, cpu_time_used, iter);

    free_dmatrix(a_mat, 1, dim);

    free_dvector(b_vec, 1, dim);
    free_dvector(xp_vec, 1, dim);
    free_dvector(xs_method, 1, dim);
    free_dvector(rp_vec, 1, dim);
    free_dvector(rn_vec, 1, dim);
    free_dvector(dp_vec, 1, dim);
    free_dvector(dn_vec, 1, dim);
    free_dvector(y_vec, 1, dim);
    free_dvector(z_vec, 1, dim);

	return 0;
}