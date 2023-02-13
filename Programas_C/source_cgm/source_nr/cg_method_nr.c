/* ********************************************** */
/* ********* Conjugate Gradient Method ********** */
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

	float *xs_method, *rp_vec, *rn_vec, *dn_vec, *dp_vec; 
    float *y_vec, *z_vec;
    float alpha, beta, err_method = 0.0;
    int iter = 0.0;

    xs_method = fvector(1, dim);
    rp_vec = fvector(1, dim);
    rn_vec = fvector(1, dim);
    dp_vec = fvector(1, dim);
    dn_vec = fvector(1, dim);
    y_vec = fvector(1, dim);
    z_vec = fvector(1, dim);

    /* **************************************** */
    /*        Conjugate Gradient Method         */
    /* **************************************** */

    matvecmul(y_vec, 1.0, a_mat, xp_vec, dim);
    vecadd(rp_vec, b_vec, MINUS, y_vec, dim);
    copyvec(dp_vec, rp_vec, dim);

    for (int k = 1; k <= ITER_MAX; k++)
    {
        /* computes error method and stop criterion */
        err_method = normvec_2(rp_vec, dim);
        if (err_method <= EPSILON) 
        {
            iter = k-1;
            break;
        }

        /* Compute alpha(k) = r^T(k)r(k) / d^T(k)Ad(k) */
        matvecmul(y_vec, 1.0, a_mat, dp_vec, dim);           /* y_vec <-- A*dp_vec */
        alpha = dotvec(rp_vec, rp_vec, dim) / dotvec(dp_vec, y_vec, dim);           /* alpha <-- dot(rp_vec, rp_vec) / dot(dp_vec, y_vec) */

        /* Compute x(k+1) = x(k) + alpha(k)*d(k) */
        scavecmul(z_vec, alpha, dp_vec, dim);           /* z_vec <-- alpha*dp_vec */
        vecadd(xs_method, xp_vec, PLUS, z_vec, dim);     /* xs_method <-- xp_vec + z_vec */  

        /* Compute r(k+1) = r(k) - alpha(k)*A*d(k) */
        scavecmul(z_vec, alpha, y_vec, dim);            /* z_vec <-- alpha*y_vec */
        vecadd(rn_vec, rp_vec, MINUS, z_vec, dim);       /* rn_vec <-- rp_vec - z_vec */

        /* Compute beta(k) = r^T(k+1)r(k+1) / r^T(k)r(k) */
        beta = dotvec(rn_vec, rn_vec, dim) / dotvec(rp_vec, rp_vec, dim);           /* beta <-- dot(rn_vec, rpnvec) / dot(rp_vec, rp_vec) */

        /* Compute d(k+1) = r(k+1) + beta(k)*d(k) */
        scavecmul(z_vec, beta, dp_vec, dim);            /* z_vec <-- beta*z_vec */
        vecadd(dn_vec, rn_vec, PLUS, z_vec, dim);        /* dn_vec <-- rn_vec + z_vec */

        copyvec(rp_vec, rn_vec, dim);                   /* rp_vec <-- rn_vec */
        copyvec(dp_vec, dn_vec, dim);                   /* dp_vec <-- dn_vec */
        copyvec(xp_vec, xs_method, dim);                /* xp_vec <-- xs_method */
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

    free_fvector(b_vec, 1, dim);
    free_fvector(xp_vec, 1, dim);
    free_fvector(xs_method, 1, dim);
    free_fvector(rp_vec, 1, dim);
    free_fvector(rn_vec, 1, dim);
    free_fvector(dp_vec, 1, dim);
    free_fvector(dn_vec, 1, dim);
    free_fvector(y_vec, 1, dim);
    free_fvector(z_vec, 1, dim);

	return 0;
}