/* ******************************************************** */
/* ************* Jacobi Method with BLAS ****************** */
/* ******************************************************** */

#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nlao_ur.h"

#define TOL 1.0e-10
#define ITER_MAX 600000

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

	/************************************************/
	/* Computes D, (L+U), T, matrices and f vector 	*/
	/************************************************/

	clock_t start, end;
        double cpu_time_used;

	/* star time count */
	start = clock();
        
        /* matrices */
        double *d_mat, *u_mat, *l_mat, *t_mat, *inv_d_mat;
        /* vectors */
	double *f_vec, *xs_vec, *y_vec, *z_vec, *err_vec;
        /* scalar value */
        double err_method = 0.0;
        int iter = 0;

        /* CBLAS parameters */
        // cblas_scopy(n, x, incx, y, incy);
        // cblas_saxpy(n, alpha, x, incx, y, incy);
        // cblas_snorm2(n, x, incx);
        // cblas_sgemv(cblas_layout, cblas_transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
        // cblas_strmv(cblas_layout, cblas_uplo, cblas_transa, cblas_diag, n, a, lda, x, incx);
        
        double plus = 1.0, minus = -1.0; // cblas parameters for add and subtraction in saxpy
        int n = dim, m = dim, nmat = dim*dim, lda = dim;
        int incx = 1, incy = 1;
        double alpha_ = 1.0, beta_ = 0.0;

        f_vec = dvector(1, dim); 
        xs_vec = dvector(1, dim);
        y_vec = dvector(1, dim); 
        z_vec = dvector(1, dim);
        err_vec = dvector(1, dim); 

        d_mat = dmatrix(1, dim);
        u_mat = dmatrix(1, dim);
        l_mat = dmatrix(1, dim);
        t_mat = dmatrix(1, dim);
        inv_d_mat = dmatrix(1, dim); 

        /* Computes D matrix as diagonal of A */
        diagmat(d_mat, a_mat, dim);

        /* compute U matrix of A matrix */
        uptrmat(u_mat, a_mat, dim);

        /* compute L matrix of A matrix */
        lotrmat(l_mat, a_mat, dim);

        /* Computes T = L+U matrix */
        cblas_dcopy(nmat, l_mat+1, incx, t_mat+1, incy);        // t_mat <-- l_mat 
        cblas_daxpy(nmat, plus, u_mat+1, incx, t_mat+1, incy);  // t_mat <-- t_mat + u_mat

        /* Computes D inverse */
        invdiagmat(inv_d_mat, d_mat, dim);                      // inv_d_mat <-- inv(d_mat) 

	/* **************************************** */
        /*              Jacobi Method               */
        /* **************************************** */

        for (int k = 1; k <= ITER_MAX; k++)
        {
                /* Computes y = (L+U) * x(k) */
                cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha_, t_mat+1, lda, xp_vec+1, incx, beta_, y_vec+1, incy);  // y_vec <-- t_mat * xp_vec 
                
                /* Computes f = b - (L+U) * x(k) */
                cblas_dcopy(n, b_vec+1, incx, f_vec+1, incy);               // f_vec <-- b_vec 
                cblas_daxpy(n, minus, y_vec+1, incx, f_vec+1, incy);        // f_vec <-- f_vec - y_vec
                
                /* Computes x(k+1) = inv(D)*(b - (L+U) * x(k)) */
                //cblas_sgemv(CblasRowMajor, CblasNoTrans, dim, dim, 1, inv_d_mat+1, dim, f_vec+1, 1, 0, xs_vec+1, 1); 
                cblas_dcopy(n, f_vec+1, incx, xs_vec+1, incy);              // xs_vec <-- f_vec
                cblas_dtrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, inv_d_mat+1, lda, xs_vec+1, incx);    // xs_vec <-- inv_d_mat*xs_vec

                /* Computes b - A * x(k+1) */
                // cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha_, a_mat+1, lda, xs_vec+1, incx, beta_, z_vec+1, incy); // z_vec <-- A * xs_vec
                // cblas_dcopy(n, b_vec+1, incx, err_vec+1, incy);             // err_vec <-- b_vec
                // cblas_daxpy(n, minus, z_vec+1, incx, err_vec+1, incy);      // err_vec <-- err_vec - z_vec

                /* Computes || b - A * x(k+1) || */
                // err_method = cblas_dnrm2(n, err_vec+1, incx);               // err_method = || err_vec ||

                /* computes error of xs_method approximation */
                /* error_method = ||xs-xe|| / || xs || */
                err_method = emethod(xe_sol, xs_vec, dim);

                /* Stop criterion */
                if (err_method <= TOL) 
                {
                        iter = k;
                        break;
                }

                cblas_dcopy(n, xs_vec+1, incx, xp_vec+1, incy);             // xp_vec <-- xs_vec
        }

	/* finish time count */
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	
	/* Print final results */
        // dwritevector(xs_vec, dim);
	//printf("error: %f\n", err_method );
	//printf("time: %f\n", cpu_time_used);
	// printf("error: %.12e, time: %e, iter-max: %d\n", err_method, cpu_time_used, iter);
        info_method(err_method, cpu_time_used, iter);
        
        free_dmatrix(a_mat, 1, dim);
        free_dmatrix(d_mat, 1, dim);
        free_dmatrix(u_mat, 1, dim);
        free_dmatrix(l_mat, 1, dim);
        free_dmatrix(t_mat, 1, dim);
        free_dmatrix(inv_d_mat, 1, dim);

        free_dvector(b_vec, 1, dim);
        free_dvector(xp_vec, 1, dim);
        free_dvector(xs_vec, 1, dim);
        free_dvector(f_vec, 1, dim);
        free_dvector(y_vec, 1, dim);
        free_dvector(z_vec, 1, dim);
        free_dvector(err_vec, 1, dim);

	return 0;
}