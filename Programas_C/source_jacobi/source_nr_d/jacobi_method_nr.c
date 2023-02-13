/* ********************************************** */
/* ************* Jacobi Method ****************** */
/* ********************************************** */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nlao_ur.h"

#define EPSILON 1.0e-10
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

	/* allocate arbitrary-offset vector and matrix of arbitrary lengths */

	double **a_mat, *b_vec, *xp_vec;

	a_mat = dmatrix(1, dim, 1, dim);
	matrix225(a_mat, dim);
	//dwritematrix(a_mat, dim);

	b_vec = dvector(1, dim);
	vector225(b_vec, dim);
	//dwritevector(b_vec, dim);

    xp_vec = dvector(1, dim);
	//dwritevector(xp_vec, dim);

	/* Exact solution of SEL */
	double *xe_sol = dvector(1, dim);
	for (int i = 1; i <= dim; i++)
	{
		xe_sol[i] = 1.0;
	}

	/************************************************/
	/* Computes D, (L+U), T, matrices and f vector 	*/
	/************************************************/

	clock_t start, end;
    double cpu_time_used;

	/* star time count */
	start = clock();

    double **d_mat, **invd_mat, **l_mat, **u_mat, **t_mat;
	double *f_vec, *xs_method, *y_vec;
    double err_method = 0.0;
    int iter = 0;

	d_mat = dmatrix(1, dim, 1, dim);
	invd_mat = dmatrix(1, dim, 1, dim);
	l_mat = dmatrix(1, dim, 1, dim);
	u_mat = dmatrix(1, dim, 1, dim);
	t_mat = dmatrix(1, dim, 1, dim);

	f_vec = dvector(1, dim);
	xs_method = dvector(1, dim);
	y_vec = dvector(1, dim);

	/* computes D matrix as diagonal of A */
	diagmat(d_mat, a_mat, dim);
	//dwritematrix(d_mat, dim);

	/* computes L matrix as a lower triangular part of A */
	lotrmat(l_mat, a_mat, dim);
	//dwritematrix(l_mat, dim);

	/* computes U matrix as a upper triangular part of A */
	uptrmat(u_mat, a_mat, dim);
	//dwritematrix(u_mat, dim);

    /* computes inv(D) matrix */
	invdiagmat(invd_mat,d_mat, dim);            /* invd_mat <-- inv(d_mat) */
	//dwritematrix(invd_mat, dim);

	/* computes (L+U) matrix */
    matadd(t_mat, l_mat, PLUS, u_mat, dim);      /* t_mat <-- l_mat + u_mat */
	//dwritematrix(t_mat, dim);

    /* **************************************** */
    /*              Jacobi Method               */
    /* **************************************** */

	for (int k = 1; k <= ITER_MAX; k++)
	{
		/* computes (L+U)*x(k) vector */
		matvecmul(y_vec, 1.0, t_mat, xp_vec, dim, dim);       /* y_vec <-- t_mat*xp_vec */ 

		/* computes b - (L+U)*x(k) vector */
		vecadd(f_vec, b_vec, MINUS, y_vec, dim);     /* f_vec = b_vec - y_vec */ 

		/* computes x(k+1) = inv(D)*(b - (L+U)*x(k)) vector */
		matvecmul(xs_method, 1.0, invd_mat, f_vec, dim, dim); /* xs_method = invd_mat*f_vec */

		/* computes error method and stop criterion */
		err_method = emethod(xe_sol, xs_method, dim);
		if (err_method < EPSILON) 
		{
			iter = k;
			break;
		}
		copyvec(xp_vec, xs_method, dim); /* xp_vec <-- xs_method */
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

	free_dmatrix(a_mat, 1, dim, 1, dim);
	free_dmatrix(d_mat, 1, dim, 1, dim);
	free_dmatrix(l_mat, 1, dim, 1, dim);
	free_dmatrix(u_mat, 1, dim, 1, dim);
	free_dmatrix(invd_mat, 1, dim, 1, dim);
	free_dmatrix(t_mat, 1, dim, 1, dim);
	
	free_dvector(b_vec, 1, dim);
	free_dvector(xp_vec, 1, dim);
	free_dvector(xe_sol, 1, dim);
	free_dvector(xs_method, 1, dim);
	free_dvector(f_vec, 1, dim);
	free_dvector(y_vec, 1, dim);

	return 0;
}