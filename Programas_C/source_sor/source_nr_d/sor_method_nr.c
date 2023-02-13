/* ********************************************** */
/* ************** SOR Method ******************** */
/* ********************************************** */
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

	/* allocate arbitrary-offset vector and matrix of arbitrary lengths */

	double **a_mat, *b_vec, *xp_vec;

	a_mat = dmatrix(1, dim, 1, dim);
	matrix225(a_mat, dim);

	b_vec = dvector(1, dim);
	vector225(b_vec, dim);

    xp_vec = dvector(1, dim);

	/* Exact solution of system */
	double *xe_sol = dvector(1, dim);
	for (int i = 1; i <= dim; i++) xe_sol[i] = 1.0;

	/************************************************/
	/* Computes D, (L+U), T, matrices and f vector 	*/
	/************************************************/

	clock_t start, end;
    double cpu_time_used;

	/* star time count */
	start = clock();

    double **d_mat, **l_mat, **u_mat, **omegad_mat, **omegal_mat, **omegau_mat, **t_mat, **invt_mat;
	double *f_vec, *p_vec, *y_vec, *z_vec, *v_vec, *w_vec, *xs_method;
	double err_method = 0.0;
	int iter = 0.0;

	d_mat = dmatrix(1, dim, 1, dim);
	l_mat = dmatrix(1, dim, 1, dim);
	u_mat = dmatrix(1, dim, 1, dim);
	omegad_mat = dmatrix(1, dim, 1, dim);
	omegal_mat = dmatrix(1, dim, 1, dim);
	omegau_mat = dmatrix(1, dim, 1, dim);
	t_mat = dmatrix(1, dim, 1, dim);
	invt_mat = dmatrix(1, dim, 1, dim);

	f_vec = dvector(1, dim);
	p_vec = dvector(1, dim);
	y_vec = dvector(1, dim);
	z_vec = dvector(1, dim);
	v_vec = dvector(1, dim);
	w_vec = dvector(1, dim);
	xs_method = dvector(1, dim);

	/* computes D matrix as diagonal of A */
	diagmat(d_mat, a_mat, dim); 

	/* computes L matrix as a lower triangular part of A */
	lotrmat(l_mat, a_mat, dim);

	/* computes U matrix as a upper triangular part of A */
	uptrmat(u_mat, a_mat, dim);

	/* Relaxation parameter */
    double omega = atof(argv[2]);

	/* computes (1-omega)*D matrix */
    scamatmul(omegad_mat, 1-omega, d_mat, dim); /* omegad_mat <-- (1 - omega)*d_mat */

	/* computes omega*L matrix as */
    scamatmul(omegal_mat, omega, l_mat, dim);   /* omegal_mat <-- omega*l_mat */

	/* computes omega*U matrix */
    scamatmul(omegau_mat, omega, u_mat, dim);   /* omegau_mat <-- omega*u_mat */

	/* computes D+omega*L matrix */
    matadd(t_mat, d_mat, PLUS, omegal_mat, dim); /*t_mat <-- d_mat + omegal_mat */

	/* computes inv(D+omega*L) matrix */
    inverse(t_mat, invt_mat, dim);	            /* invt_mat <-- inv(t_mat) */

	/* computes omega*b vector */
    scavecmul(w_vec, omega, b_vec, dim);        /* w_vec <-- omega*b_vec */

	/* computes omega*(D+omega*L)*b vector */
    matvecmul(f_vec, 1.0, invt_mat, w_vec, dim, dim);     /* compute f_vec <-- invt_mat * w_vec */

	/* **************************************** */
	/*                SOR Method                */
	/* **************************************** */

	for (int k = 1; k <= ITER_MAX; k++)
	{
	/* computes (1-omega)*D*x(k) vector */
	matvecmul(y_vec, 1.0, omegad_mat, xp_vec, dim, dim);  /* y_vec = omegad * xp_vec */

	/* computes omega*U*x(k) vector */
	matvecmul(z_vec, 1.0, omegau_mat, xp_vec, dim, dim);  /* z_vec = omegau * xp_vec */

	/* computes (1-omega)*D*x(k) - omega*U*x(k) vector */
	vecadd(v_vec, y_vec, MINUS, z_vec, dim);     /* v_vec = y_vec - z_vec */

	/* computes inv(D+omega*L)*(1-omega)*D*x(k) - omega*U*x(k) vector */
	matvecmul(p_vec, 1.0, invt_mat, v_vec, dim, dim);     /* p_vec = invt * v_vec */

	/* computes x(k+1) = inv(D+omega*L)*((1-omega)*D*x(k) - omega*U*x(k)) + omega*(D+omega*L)*b vector */
	vecadd(xs_method, p_vec, PLUS, f_vec, dim);  /* xs_method = p_vec + f_vec */

	/* computes error method and stop criterion */
	err_method = emethod(xe_sol, xs_method, dim);
	if (err_method < TOL) 
	{
		iter = k;
		break;
	}
	copyvec(xp_vec, xs_method, dim);            /* xp_vec <-- xs_method */
	}

	/* finish time count */
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	/* Print final results */
	//dwritevector(xs_method, dim);
	//printf("error: %f\n", err_method );
	//printf("time: %f\n", cpu_time_used);
	// printf("error: %.12f, time: %f, iter-max: %d\n", err_method, cpu_time_used, iter);
	info_method(err_method, cpu_time_used, iter);

	free_dmatrix(a_mat, 1, dim, 1, dim);
	free_dmatrix(d_mat, 1, dim, 1, dim);
	free_dmatrix(u_mat, 1, dim, 1, dim);
	free_dmatrix(l_mat, 1, dim, 1, dim);
	free_dmatrix(omegad_mat, 1, dim, 1, dim);
	free_dmatrix(omegau_mat, 1, dim, 1, dim);
	free_dmatrix(omegal_mat, 1, dim, 1, dim);
	free_dmatrix(t_mat, 1, dim, 1, dim);
	free_dmatrix(invt_mat, 1, dim, 1, dim);

	free_dvector(b_vec, 1, dim);
	free_dvector(xp_vec, 1, dim);
	free_dvector(xs_method, 1, dim);
	free_dvector(y_vec, 1, dim);
	free_dvector(z_vec, 1, dim);
	free_dvector(v_vec, 1, dim);
	free_dvector(w_vec, 1, dim);
	free_dvector(f_vec, 1, dim);
	free_dvector(p_vec, 1, dim);

	return 0;
}