/* ********************************************** */
/* ********** Gauss-Seidel Method *************** */
/* ********************************************** */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nlao_ur.h"

#define EPSILON 1.0e-5
#define ITER_MAX 600000

/* ******************************** */
/*          main function           */      
/* ******************************** */

int main(int argc, char const *argv[])
{
	/* Operations */
	float plus = 1.0, minus = -1.0;

	/* Size of the linear equation system */

	int dim = atoi(argv[1]);

	/* allocate arbitrary-offset vector and matrix of arbitrary lenghts */

	float **a_mat, *b_vec, *xp_vec;

	a_mat = fmatrix(1, dim, 1, dim);
	matrix225(a_mat, dim);
	//writematrix(a_mat, dim);

	b_vec = fvector(1, dim);
	vector225(b_vec, dim);
	//writevector(b_vec, dim);

    xp_vec = fvector(1, dim);
	//writevector(xp_vec, dim);

	/************************************************/
	/* Computes D, (L+U), T, matrices and f vector 	*/
	/************************************************/

	clock_t start, end;
    double cpu_time_used;

	/* star time count */
	start = clock();

    float **d_mat, **l_mat, **u_mat, **t_mat, **invt_mat;
	float *f_vec, *y_vec, *xs_method;
	float err_method = 0.0;
	int iter = 0;

	d_mat = fmatrix(1, dim, 1, dim);
	l_mat = fmatrix(1, dim, 1, dim);
	u_mat = fmatrix(1, dim, 1, dim);
	t_mat = fmatrix(1, dim, 1, dim);
	invt_mat = fmatrix(1, dim, 1, dim);

	f_vec = fvector(1, dim);
	y_vec = fvector(1, dim);
	xs_method = fvector(1, dim);

	/* computes D matrix as diagonal of A */
	diagmat(d_mat, a_mat, dim);
	//writematrix(d_mat, dim);

	/* computes L matrix as a lower triangular part of A */
	lotrmat(l_mat, a_mat, dim);
	//writematrix(l_mat, dim);

	/* computes U matrix as a upper triangular part of A */
	uptrmat(u_mat, a_mat, dim);
	//writematrix(u_mat, dim);

	/* computes D+L matrix */
    matadd(t_mat, d_mat, plus, l_mat, dim);  /* t_mat <-- d_mat+l_mat */
	//writematrix(t_mat, dim);

	/* computes inv(D+L) matrix */
    inverse(t_mat, invt_mat, dim);          /* invt_mat <-- inv(t_mat) */
	//writematrix(egss_mat, dim);

	/* **************************************** */
	/*           Gauss-Seidel Method            */
	/* **************************************** */

	for (int k = 1; k <= ITER_MAX; k++)
	{
		/* computes U*x(k) */
		matvecmul(y_vec, 1.0, u_mat, xp_vec, dim); /* y_vec <-- u_mat*xp_vec */

		/* computes b - U*x(k) */
		vecadd(f_vec, b_vec, minus, y_vec, dim); /* f_vec <-- b - y_vec */

		/* computes x(k+1) = inv(D+L)*(b - U*x(k)) */
		matvecmul(xs_method, 1.0, invt_mat, f_vec, dim); /* xs_method <-- invt_mat*f_vec */

		/* computes error method and stop criterion */
		err_method = emethod(a_mat, b_vec, xs_method, dim);
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
    //writevector(xs_method, dim);
	//printf("error: %f\n", err_method );
	//printf("time: %f\n", cpu_time_used);
	printf("error: %.12f, time: %f, iter-max: %d\n", err_method, cpu_time_used, iter);

	free_fmatrix(a_mat, 1, dim, 1, dim);
	free_fmatrix(d_mat, 1, dim, 1, dim);
	free_fmatrix(u_mat, 1, dim, 1, dim);
	free_fmatrix(l_mat, 1, dim, 1, dim);
	free_fmatrix(t_mat, 1, dim, 1, dim);
	free_fmatrix(invt_mat, 1, dim, 1, dim);

	free_fvector(b_vec, 1, dim);
	free_fvector(xp_vec, 1, dim);
	free_fvector(f_vec, 1, dim);
	free_fvector(y_vec, 1, dim);
	free_fvector(xs_method, 1, dim);

	return 0;
}