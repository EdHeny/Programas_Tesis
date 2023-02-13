/* ********************************************** */
/* ************** SOR Method ******************** */
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
	/* Arithmetic operations */
	float PLUS = 1.0, MINUS = -1.0;

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

    float **d_mat, **l_mat, **u_mat, **omegad_mat, **omegal_mat, **omegau_mat, **t_mat, **invt_mat;
	float *f_vec, *p_vec, *y_vec, *z_vec, *v_vec, *w_vec, *xs_method;
	float err_method = 0.0;
	int iter = 0.0;

	d_mat = fmatrix(1, dim, 1, dim);
	l_mat = fmatrix(1, dim, 1, dim);
	u_mat = fmatrix(1, dim, 1, dim);
	omegad_mat = fmatrix(1, dim, 1, dim);
	omegal_mat = fmatrix(1, dim, 1, dim);
	omegau_mat = fmatrix(1, dim, 1, dim);
	t_mat = fmatrix(1, dim, 1, dim);
	invt_mat = fmatrix(1, dim, 1, dim);

	f_vec = fvector(1, dim);
	p_vec = fvector(1, dim);
	y_vec = fvector(1, dim);
	z_vec = fvector(1, dim);
	v_vec = fvector(1, dim);
	w_vec = fvector(1, dim);
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

	/* Relaxation parameter */
    float omega = atof(argv[2]);

	/* computes (1-omega)*D matrix */
    scamatmul(omegad_mat, 1-omega, d_mat, dim); /* omegad_mat <-- (1 - omega)*d_mat */
	//writematrix(omegad_mat, dim);

	/* computes omega*L matrix as */
    scamatmul(omegal_mat, omega, l_mat, dim);   /* omegal_mat <-- omega*l_mat */
	//writematrix(omegal_mat, n);

	/* computes omega*U matrix */
    scamatmul(omegau_mat, omega, u_mat, dim);   /* omegau_mat <-- omega*u_mat */
	//writematrix(omegau_mat, dim);

	/* computes D+omega*L matrix */
    matadd(t_mat, d_mat, PLUS, omegal_mat, dim); /*t_mat <-- d_mat + omegal_mat */
	//writematrix(t_mat, n);

	/* computes inv(D+omega*L) matrix */
    inverse(t_mat, invt_mat, dim);	            /* invt_mat <-- inv(t_mat) */
	//writematrix(invt_mat, n);

	/* computes omega*b vector */
    scavecmul(w_vec, omega, b_vec, dim);        /* w_vec <-- omega*b_vec */
    //writevector(w_vec, dim);

	/* computes omega*(D+omega*L)*b vector */
    matvecmul(f_vec, 1.0, invt_mat, w_vec, dim);     /* compute f_vec <-- invt_mat * w_vec */
    //writevector(f_vec, dim);
        

	/* **************************************** */
	/*                SOR Method                */
	/* **************************************** */

	for (int k = 1; k <= ITER_MAX; k++)
	{
	/* computes (1-omega)*D*x(k) vector */
	matvecmul(y_vec, 1.0, omegad_mat, xp_vec, dim);  /* y_vec = omegad * xp_vec */

	/* computes omega*U*x(k) vector */
	matvecmul(z_vec, 1.0, omegau_mat, xp_vec, dim);  /* z_vec = omegau * xp_vec */

	/* computes (1-omega)*D*x(k) - omega*U*x(k) vector */
	vecadd(v_vec, y_vec, MINUS, z_vec, dim);     /* v_vec = y_vec - z_vec */

	/* computes inv(D+omega*L)*(1-omega)*D*x(k) - omega*U*x(k) vector */
	matvecmul(p_vec, 1.0, invt_mat, v_vec, dim);     /* p_vec = invt * v_vec */

	/* computes x(k+1) = inv(D+omega*L)*((1-omega)*D*x(k) - omega*U*x(k)) + omega*(D+omega*L)*b vector */
	vecadd(xs_method, p_vec, PLUS, f_vec, dim);  /* xs_method = p_vec + f_vec */

	/* computes error method and stop criterion */
	err_method = emethod(a_mat, b_vec, xs_method, dim);
	if (err_method < EPSILON) 
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
        //writevector(xs_method, dim);
	//printf("error: %f\n", err_method );
	//printf("time: %f\n", cpu_time_used);
	printf("error: %.12f, time: %f, iter-max: %d\n", err_method, cpu_time_used, iter);

	free_fmatrix(a_mat, 1, dim, 1, dim);
	free_fmatrix(d_mat, 1, dim, 1, dim);
	free_fmatrix(u_mat, 1, dim, 1, dim);
	free_fmatrix(l_mat, 1, dim, 1, dim);
	free_fmatrix(omegad_mat, 1, dim, 1, dim);
	free_fmatrix(omegau_mat, 1, dim, 1, dim);
	free_fmatrix(omegal_mat, 1, dim, 1, dim);
	free_fmatrix(t_mat, 1, dim, 1, dim);
	free_fmatrix(invt_mat, 1, dim, 1, dim);

	free_fvector(b_vec, 1, dim);
	free_fvector(xp_vec, 1, dim);
	free_fvector(xs_method, 1, dim);
	free_fvector(y_vec, 1, dim);
	free_fvector(z_vec, 1, dim);
	free_fvector(v_vec, 1, dim);
	free_fvector(w_vec, 1, dim);
	free_fvector(f_vec, 1, dim);
	free_fvector(p_vec, 1, dim);

	return 0;
}