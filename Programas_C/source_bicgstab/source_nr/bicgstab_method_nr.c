/* *********************************************************** */
/* ********* BiConjugate Gradient Stabilized Method ********** */
/* *********************************************************** */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nlao_ur.h"

#define TOL 1.0e-10
#define ITER_MAX 600000
#define MINUS -1.0
#define PLUS 1.0

/* ******************************** */
/*          main function           */      
/* ******************************** */

int main(int argc, char const *argv[])
{
	/* Size of the linear equation system */

	int dim = atoi(argv[1]);

	/* allocate arbitrary-offset vector and matrix of arbitrary lenghts */

	double **a_mat, *b_vec, *xp_vec;

	a_mat = dmatrix(1, dim, 1, dim);
	matrix225(a_mat, dim);

	b_vec = dvector(1, dim);
	vector225(b_vec, dim);

    /* Initial vector */
    xp_vec = dvector(1, dim);

    /* Exact solution of system test */
    double *xe_sol = dvector(1, dim);
    for (int i = 1; i <= dim; i++) xe_sol[i] = 1.0;

	//****************************
	/* Variables declarations 	*/
	//****************************

	clock_t start, end;
    double cpu_time_used;

	/* star time count */
	start = clock();

	double *xs_method, *rp_vec, *rn_vec, *pn_vec, *pp_vec, *s_vec, *rast_vec; 
    double *y_vec, *z_vec, *f_vec, *t1_vec, *t2_vec;
    double alpha = 0.0, beta = 0.0, omega = 0.0, err_method = 0.0;
    int iter = 0.0;

    xs_method = dvector(1, dim);
    rp_vec = dvector(1, dim);
    rn_vec = dvector(1, dim);
    pp_vec = dvector(1, dim);
    pn_vec = dvector(1, dim);
    s_vec = dvector(1, dim);
    rast_vec = dvector(1, dim);
    y_vec = dvector(1, dim);
    z_vec = dvector(1, dim);
    f_vec = dvector(1, dim);
    t1_vec = dvector(1, dim);
    t2_vec = dvector(1, dim);

    // *******************************************************
    /*        BiConjugate Gradient Stabilized Method         */
    // *******************************************************

    matvecmul(y_vec, 1.0, a_mat, xp_vec, dim, dim);
    vecadd(rp_vec, b_vec, -1.0, y_vec, dim);
    copyvec(pp_vec, rp_vec, dim);
    copyvec(rast_vec, rp_vec, dim);

    for (int k = 1; k <= ITER_MAX; k++)
    {
        /* Compute alpha(k) = <r(k), r*(0)> / <r*(0), Ap(k)> */ 
        matvecmul(y_vec, 1.0, a_mat, pp_vec, dim, dim);           /* y_vec <-- A*pp_vec */
        alpha = dotvec(rp_vec, rast_vec, dim) / dotvec(y_vec, rast_vec, dim);                   /* alpha <-- <rp_vec, rast_vec> / <rast_vec, y_vec> */

        /* Compute s(k) = r(k) - alpha(k)*A*p(k) */
        scavecmul(f_vec, alpha, y_vec, dim);            /* f_vec <-- alpha*y_vec */
        vecadd(s_vec, rp_vec, MINUS, f_vec, dim);        /* s_vec <-- rp_vec - f_vec */  
        
        /* Compute omega(k) = <A*s(k), s(k)> / <A*s(k), A*s(k)> */ 
        matvecmul(z_vec, 1.0, a_mat, s_vec, dim, dim);            /* z_vec <-- A*s_vec */
        omega = dotvec(z_vec, s_vec, dim) / dotvec(z_vec, z_vec, dim);                          /* omega <-- <z_vec, s_vec> / <z_vec, z_vec> */

        /* Compute x(k+1) = x(k) + alpha(k)*p(k) + omega(k)*s(k) */
        scavecmul(t1_vec, alpha, pp_vec, dim);          /* t1_vec <-- alpha*t1_vec */
        scavecmul(t2_vec, omega, s_vec, dim);           /* t2_vec <-- omega*t2_vec */
        vecadd(t1_vec, t1_vec, PLUS, t2_vec, dim);       /* t1_vec <-- t1_vec + t2_vec */  
        vecadd(xs_method, xp_vec, PLUS, t1_vec, dim);    /* xs_method <-- xp_vec + t1_vec */  

        /* computes error method and stop criterion */
        err_method = emethod(xe_sol, xs_method, dim);
        if (err_method <= TOL) 
        {
            iter = k;
            break;
        }

        /* Compute r(k+1) = s(k) - omega(k)*A*s(k) */
        scavecmul(f_vec, omega, z_vec, dim);            /* f_vec <-- omega*z_vec */
        vecadd(rn_vec, s_vec, MINUS, f_vec, dim);       /* rn_vec <-- rp_vec - z_vec */

        /* Compute beta(k) = <r(k+1), r*(0)>/<r(k), r*(0)> * alpha(k)/omega(k) */
        beta = dotvec(rn_vec, rast_vec, dim)/dotvec(rp_vec, rast_vec, dim) * alpha/omega;       /* beta <-- <rn_vec, rast_vec>/<rp_vec, rast_vec> * alpha/omega */

        /* Compute p(k+1) = r(k+1) + beta(k)*(p(k)-omega(k)*A*p(k)) */
        scavecmul(f_vec, omega, y_vec, dim);            /* f_vec <-- omega*y_vec */
        vecadd(t2_vec, pp_vec, MINUS, f_vec, dim);       /* t2_vec <-- pp_vec - f_vec */
        scavecmul(t2_vec, beta, t2_vec, dim);           /* t2_vec <-- omega*t2_vec */
        vecadd(pn_vec, rn_vec, PLUS, t2_vec, dim);       /* pn_vec <-- rn_vec + t2_vec */

        copyvec(rp_vec, rn_vec, dim);                   /* rp_vec <-- rn_vec */
        copyvec(pp_vec, pn_vec, dim);                   /* pp_vec <-- pn_vec */
        copyvec(xp_vec, xs_method, dim);                /* xp_vec <-- xs_method */
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

    free_dvector(b_vec, 1, dim);
    free_dvector(xp_vec, 1, dim);
    free_dvector(xs_method, 1, dim);
    free_dvector(rp_vec, 1, dim);
    free_dvector(rn_vec, 1, dim);
    free_dvector(pp_vec, 1, dim);
    free_dvector(pn_vec, 1, dim);
    free_dvector(s_vec, 1, dim);
    free_dvector(rast_vec, 1, dim);
    free_dvector(y_vec, 1, dim);
    free_dvector(z_vec, 1, dim);
    free_dvector(f_vec, 1, dim);
    free_dvector(t1_vec, 1, dim);
    free_dvector(t2_vec, 1, dim);

	return 0;
}