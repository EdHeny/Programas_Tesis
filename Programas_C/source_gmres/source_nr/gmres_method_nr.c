/* *********************************************** */
/* ********* Generalize Residual Method ********** */
/* *********************************************** */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "nlao_ur.h"
#define TOL 1e-8

/* ******************************** */
/*          main function           */      
/* ******************************** */

int main(int argc, char const *argv[])
{
	/* Size of the linear equation system */
	int dim = atoi(argv[1]);

	/* allocate arbitrary-offset vector and matrix of arbitrary lengths */
	float **a_mat, *b_vec, *xp_vec;

	a_mat = fmatrix(1, dim, 1, dim);
	matrix225(a_mat, dim);

	b_vec = fvector(1, dim);
	vector225(b_vec, dim);

    xp_vec = fvector(1, dim);

	/************************************************/
	/* Computes D, (L+U), T, matrices and f vector 	*/
	/************************************************/

	clock_t start, end;
    double cpu_time_used;

    /* parameters of addition and subtraction */
    float PLUS = 1.0, MINUS = -1.0;

    /* Variables definition */
    int k = dim;
    float beta = 0.0;
    float err_method = 0.0;
    int iter = 0.0;
    float norm_vj = 0.0;
    float kappa = 0.75;
    /* Matrix Q and H */
    float **q_mat, **h_mat;
    /* Residual vector, sine and cosine vectors */
    float *r_vec, *sn_vec, *cs_vec;
    /* Vector be1 = [beta,0,...,0], vector solution */
	float *be1_vec, *xs_method; 
    /* Auxiliaries vector */
	float  *vj_vec, *qj_vec; 
    float *y_vec, *aux_vec;

    q_mat = fmatrix(1, dim, 1, dim+1);
    h_mat = fmatrix(1, dim+1, 1, dim);

    r_vec = fvector(1, dim);
    be1_vec = fvector(1, dim+1);
    vj_vec = fvector(1, dim);
    qj_vec = fvector(1, dim);
    sn_vec = fvector(1, dim);
    cs_vec = fvector(1, dim);
    aux_vec = fvector(1, dim);
    y_vec = fvector(1, k);
    xs_method = fvector(1, dim);

	/* star time count */
	start = clock();

    /* ***************************************** */
    /*        Generalize Residual Method         */
    /* ***************************************** */

    matvecmul(r_vec, 1.0, a_mat, xp_vec, dim, dim);
    vecadd(r_vec, b_vec, MINUS, r_vec, dim);
    beta = normvec_2(r_vec, dim);
    scavecmul(vj_vec, 1/beta, r_vec, dim);
    
    /* Q[:,1] <- v1 */
    vec2mat(vj_vec, q_mat, 1, dim);

    /* be1 <- (beta,0,0,...,0) \in R^(k+1)*/
    be1_vec[1] = beta;

    for (int j = 1; j <= k; j++)
    {
        /* Arnoldi's Method */
        mat2vec(q_mat, qj_vec, j, dim);
        matvecmul(vj_vec, 1.0, a_mat, qj_vec, dim, dim);
        norm_vj = normvec_2(vj_vec, dim);
        for (int i = 1; i <= j; i++)
        {
            mat2vec(q_mat, qj_vec, i, dim);
            h_mat[i][j] = dotvec(qj_vec, vj_vec, dim);
            scavecmul(qj_vec, h_mat[i][j], qj_vec, dim);
            vecadd(vj_vec, vj_vec, MINUS, qj_vec, dim);
        }
        /* Test for loss of orthogonality and reorthogonalize if necessary */
        if (normvec_2(vj_vec, dim) / norm_vj <= kappa)
        {
            float p = 0.0;
            for (int i = 1; i <= j; i++)
            {
                mat2vec(q_mat, qj_vec, i, dim);
                p = dotvec(qj_vec, vj_vec, dim);
                scavecmul(qj_vec, p, qj_vec, dim);
                vecadd(vj_vec, vj_vec, MINUS, qj_vec, dim);
                h_mat[i][j] = h_mat[i][j] + p;
            }
        }
        h_mat[j+1][j] = normvec_2(vj_vec, dim);
        /* Breakdown */
        if (fabs(h_mat[j+1][j]) <= TOL)
        {
            iter = j;
            k = j;
            printf("successful\n");
            break;
        }

        scavecmul(vj_vec, 1/h_mat[j+1][j], vj_vec, dim);
        vec2mat(vj_vec, q_mat, j+1, dim);

        /* Givens Rotation to find R and be1 to solve Least Square Problem */
        /* compute the product of the previous Givens rotations to the jth column of H */
        float hij = 0.0;
        for (int i = 1; i <= j-1; i++)
        {
            hij = h_mat[i][j];
            h_mat[i][j] = cs_vec[i]*hij + sn_vec[i] * h_mat[i+1][j];
            h_mat[i+1][j] = -sn_vec[i]*hij + cs_vec[i] * h_mat[i+1][j];
        }
        /* compute and apply Givens rotation to put 0 in the subdiagonal */
        if (fabs(h_mat[j][j]) > fabs(h_mat[j+1][j]))
        {
            float t = 0.0, u = 0.0, sign = 0.0;
            t = h_mat[j+1][j] / h_mat[j][j];
            sign = copysign(1.0, h_mat[j][j]);
            u = sign * sqrt(1 + t*t);
            cs_vec[j] = 1 / u;
            sn_vec[j] = t * cs_vec[j];
        }
        else
        {
            float t = 0.0, u = 0.0, sign = 0.0;
            t = h_mat[j][j] / h_mat[j+1][j];
            sign = copysign(1.0, h_mat[j+1][j]);
            u = sign * sqrt(1 + t*t);
            sn_vec[j] = 1 / u;
            cs_vec[j] = t * sn_vec[j];
        }

        /* apply the rotation found (remember that be1[j+1]=0) and do H[j+1,j] = 0 */
        float hjj;
        hjj = h_mat[j][j];
        h_mat[j][j] = cs_vec[j]*hjj + sn_vec[j] * h_mat[j+1][j];
        h_mat[j+1][j] = -sn_vec[j]*hjj + cs_vec[j] * h_mat[j+1][j];
        be1_vec[j+1] = -sn_vec[j]*be1_vec[j];
        be1_vec[j] = cs_vec[j]*be1_vec[j];
    }

    /* compute last R column of QR givens */
    float hik = 0.0;
    for (int i = 1; i <= k-1; i++)
    {
        hik = h_mat[i][k];
        h_mat[i][k] = cs_vec[i]*hik + sn_vec[i] * h_mat[i+1][k];
        h_mat[i+1][k] = -sn_vec[i]*hik + cs_vec[i] * h_mat[i+1][k];
    }

    /* solve LS problem solving the system R*y = be1 */
    vec2vec(be1_vec, y_vec, k);
    solve(h_mat, y_vec, k);

    matvecmul(aux_vec, 1.0, q_mat, y_vec, dim, k);
    vecadd(xs_method, xp_vec, PLUS, aux_vec, dim);

	/* finish time count */
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    /* Exact solution of system */
    float *xe_sys = fvector(1, dim);
    for (int i = 1; i <= dim; i++)
    {
        xe_sys[i] = 1.0;
    }

    err_method = emethod(xe_sys, xs_method, dim);

	/* Print final results */
    // fwritevector(xs_method, dim);
	//printf("error: %f\n", err_method );
	// printf("time: %f\n", cpu_time_used);
    info_method(err_method, cpu_time_used, iter);

    free_fmatrix(h_mat, 1, dim, 1, dim);
    free_fmatrix(q_mat, 1, dim, 1, dim);
    free_fmatrix(a_mat, 1, dim, 1, dim);

    free_fvector(xe_sys, 1, dim);
    free_fvector(xs_method, 1, dim);
    free_fvector(y_vec, 1, k);
    free_fvector(aux_vec, 1, dim);
    free_fvector(cs_vec, 1, dim);
    free_fvector(sn_vec, 1, dim);
    free_fvector(qj_vec, 1, dim);
    free_fvector(vj_vec, 1, dim);
    free_fvector(be1_vec, 1, dim);
    free_fvector(r_vec, 1, dim);
    free_fvector(xp_vec, 1, dim);
    free_fvector(b_vec, 1, dim);

	return 0;
}