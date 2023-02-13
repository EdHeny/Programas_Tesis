/* ********************************************** */
/* ************* Jacobi Method ****************** */
/* ********************************************** */

#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nlao_ur.h"

#define epsilon 1.0e-5
#define itermax 600000

/* ******************************** */
/*          main function           */      
/* ******************************** */

int main(int argc, char const *argv[])
{

	/* Size of the linear equation system */

	int dim = atoi(argv[1]);

	/* allocate arbitrary-offset vector and matrix of arbitrary lenghts */

	float *a_mat, *b_vec, *xp_vec;

	a_mat = fmatrix(1, dim);
	matrix225(a_mat, dim);
	//fwritematrix(a_mat, dim, dim);

	b_vec = fvector(1, dim);
	vector225(b_vec, dim);
	//fwritevector(b_vec, dim);

        xp_vec = fvector(1, dim);
	//writevector(xp_vec, dim);

	/************************************************/
	/* Computes D, (L+U), T, matrices and f vector 	*/
	/************************************************/

	clock_t start, end;
        double cpu_time_used;

	/* star time count */
	start = clock();
        
        /* matrices */
        float *d_mat, *u_mat, *l_mat, *t_mat, *inv_d_mat;
        /* vectors */
	float *f_vec, *xs_vec, *y_vec, *z_vec, *err_vec;
        /* scalar value */
        double err_method = 0.0;
        int iter = 0;

        /* CBLAS parameters */
        // cblas_scopy(n, x, incx, y, incy);
        // cblas_saxpy(n, alpha, x, incx, y, incy);
        // cblas_snorm2(n, x, incx);
        // cblas_sgemv(cblas_layout, cblas_transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
        // cblas_strmv(cblas_layout, cblas_uplo, cblas_transa, cblas_diag, n, a, lda, x, incx);
        
        float plus = 1.0, minus = -1.0; // cblas parameters for add and sustraction in saxpy
        int n = dim, m = dim, nmat = dim*dim, lda = dim;
        int incx = 1, incy = 1;
        float alpha_ = 1.0, beta_ = 0.0;

        f_vec = fvector(1, dim); 
        xs_vec = fvector(1, dim);
        y_vec = fvector(1, dim); 
        z_vec = fvector(1, dim);
        err_vec = fvector(1, dim); 

        d_mat = fmatrix(1, dim);
        u_mat = fmatrix(1, dim);
        l_mat = fmatrix(1, dim);
        t_mat = fmatrix(1, dim);
        inv_d_mat = fmatrix(1, dim); 

        /* Computes D matrix as diagonal of A */
        diagmat(d_mat, a_mat, dim);

        /* compute U matrix of A matrix */
        uptrmat(u_mat, a_mat, dim);

        /* compute L matrix of A matrix */
        lotrmat(l_mat, a_mat, dim);

        /* Computes T = L+U matrix */
        cblas_scopy(nmat, l_mat+1, incx, t_mat+1, incy);        // t_mat <-- l_mat 
        cblas_saxpy(nmat, plus, u_mat+1, incx, t_mat+1, incy);  // t_mat <-- t_mat + u_mat

        /* Computes D inverse */
        invdiagmat(inv_d_mat, d_mat, dim);                      // inv_d_mat <-- inv(d_mat) 

	/* **************************************** */
        /*              Jacobi Method               */
        /* **************************************** */

        for (int k = 1; k <= itermax; k++)
        {
            /* Computes y = (L+U) * x(k) */
            cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha_, t_mat+1, lda, xp_vec+1, incx, beta_, y_vec+1, incy);  // y_vec <-- t_mat * xp_vec 
            
            /* Computes f = b - (L+U) * x(k) */
            cblas_scopy(n, b_vec+1, incx, f_vec+1, incy);               // f_vec <-- b_vec 
            cblas_saxpy(n, minus, y_vec+1, incx, f_vec+1, incy);        // f_vec <-- f_vec - y_vec
            
            /* Computes x(k+1) = inv(D)*(b - (L+U) * x(k)) */
            //cblas_sgemv(CblasRowMajor, CblasNoTrans, dim, dim, 1, inv_d_mat+1, dim, f_vec+1, 1, 0, xs_vec+1, 1); 
            cblas_scopy(n, f_vec+1, incx, xs_vec+1, incy);              // xs_vec <-- f_vec
            cblas_strmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, inv_d_mat+1, lda, xs_vec+1, incx);    // xs_vec <-- inv_d_mat*xs_vec

            /* Computes b - A * x(k+1) */
            cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha_, a_mat+1, lda, xs_vec+1, incx, beta_, z_vec+1, incy); // z_vec <-- A * xs_vec
            cblas_scopy(n, b_vec+1, incx, err_vec+1, incy);             // err_vec <-- b_vec
            cblas_saxpy(n, minus, z_vec+1, incx, err_vec+1, incy);      // err_vec <-- err_vec - z_vec

            /* Computes || b - A * x(k+1) || */
            err_method = cblas_snrm2(n, err_vec+1, incx);               // err_method = || err_vec ||

	    /* Stop criterion */
            if (err_method < epsilon) 
            {
                iter = k;
                break;
            }

            cblas_scopy(n, xs_vec+1, incx, xp_vec+1, incy);             // xp_vec <-- xs_vec
        }

	/* finish time count */
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	
	/* Print final results */
        //writevector(xs_vec, dim);
	//printf("error: %f\n", err_method );
	//printf("time: %f\n", cpu_time_used);
	printf("error: %.12f, time: %f, iter-max: %d\n", err_method, cpu_time_used, iter);
        
        free_fmatrix(a_mat, 1, dim);
        free_fmatrix(d_mat, 1, dim);
        free_fmatrix(u_mat, 1, dim);
        free_fmatrix(l_mat, 1, dim);
        free_fmatrix(t_mat, 1, dim);
        free_fmatrix(inv_d_mat, 1, dim);

        free_fvector(b_vec, 1, dim);
        free_fvector(xp_vec, 1, dim);
        free_fvector(xs_vec, 1, dim);
        free_fvector(f_vec, 1, dim);
        free_fvector(y_vec, 1, dim);
        free_fvector(z_vec, 1, dim);
        free_fvector(err_vec, 1, dim);

	return 0;
}