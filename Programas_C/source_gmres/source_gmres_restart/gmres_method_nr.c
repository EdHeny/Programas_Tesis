/* *************************************************************** */
/* ********* Generalize Minimum Residual Method Restart ********** */
/* *************************************************************** */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>

#include "nlao_ur.h"
#define TOL 1e-10
#define MINUS -1.0
#define PLUS 1.0


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
    double *xe_sys = dvector(1, dim);
    for (int i = 1; i <= dim; i++) xe_sys[i] = 1.0;

	/************************************************/
	/* Computes D, (L+U), T, matrices and f vector 	*/
	/************************************************/

	clock_t start, end;
    double cpu_time_used;

    /* Variables definition */
    double beta = 0.0;
    double err_method = 0.0;
    int restart = 0.0;
    double norm_vj = 0.0;
    double kappa = 0.75;
    /* Matrix Q and H */
    double **q_mat, **h_mat;
    /* Residual vector, sine and cosine vectors */
    double *r_vec, *sn_vec, *cs_vec;
    /* Vector be1 = [beta,0,...,0], vector solution */
	double *be1_vec, *xs_method; 
    /* Auxiliaries vector */
	double  *vj_vec, *qj_vec; 
    double *y_vec, *aux_vec;

    /* Restart */
    int k = atof(argv[2]);

    q_mat = dmatrix(1, dim, 1, k+1);
    h_mat = dmatrix(1, k+1, 1, k);

    r_vec = dvector(1, dim);
    be1_vec = dvector(1, k+1);
    vj_vec = dvector(1, dim);
    qj_vec = dvector(1, dim);
    sn_vec = dvector(1, k);
    cs_vec = dvector(1, k);
    aux_vec = dvector(1, dim);
    y_vec = dvector(1, k);
    xs_method = dvector(1, dim);

    /* Control variables of while */
    bool flag = true;
    int m = 0;

	/* star time count */
	start = clock();

    /* ************************************************* */
    /*        Generalize Minimum Residual Method         */
    /* ************************************************* */
    
    while (flag == true)
    {
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
                double p = 0.0;
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
                k = j;
                printf("successful\n");
                break;
            }

            scavecmul(vj_vec, 1/h_mat[j+1][j], vj_vec, dim);
            vec2mat(vj_vec, q_mat, j+1, dim);

            /* Givens Rotation to find R and be1 to solve Least Square Problem */
            /* compute the product of the previous Givens rotations to the jth column of H */
            double hij = 0.0;
            for (int i = 1; i <= j-1; i++)
            {
                hij = h_mat[i][j];
                h_mat[i][j] = cs_vec[i]*hij + sn_vec[i] * h_mat[i+1][j];
                h_mat[i+1][j] = -sn_vec[i]*hij + cs_vec[i] * h_mat[i+1][j];
            }
            /* compute and apply Givens rotation to put 0 in the subdiagonal */
            if (fabs(h_mat[j][j]) > fabs(h_mat[j+1][j]))
            {
                double t = 0.0, u = 0.0, sign = 0.0;
                t = h_mat[j+1][j] / h_mat[j][j];
                sign = copysign(1.0, h_mat[j][j]);
                u = sign * sqrt(1 + t*t);
                cs_vec[j] = 1 / u;
                sn_vec[j] = t * cs_vec[j];
            }
            else
            {
                double t = 0.0, u = 0.0, sign = 0.0;
                t = h_mat[j][j] / h_mat[j+1][j];
                sign = copysign(1.0, h_mat[j+1][j]);
                u = sign * sqrt(1 + t*t);
                sn_vec[j] = 1 / u;
                cs_vec[j] = t * sn_vec[j];
            }

            /* apply the rotation found (remember that be1[j+1]=0) and do H[j+1,j] = 0 */
            double hjj;
            hjj = h_mat[j][j];
            h_mat[j][j] = cs_vec[j]*hjj + sn_vec[j] * h_mat[j+1][j];
            h_mat[j+1][j] = -sn_vec[j]*hjj + cs_vec[j] * h_mat[j+1][j];
            be1_vec[j+1] = -sn_vec[j]*be1_vec[j];
            be1_vec[j] = cs_vec[j]*be1_vec[j];
        }

        /* compute last R column of QR givens */
        double hik = 0.0;
        for (int i = 1; i <= k-1; i++)
        {
            hik = h_mat[i][k];
            h_mat[i][k] = cs_vec[i]*hik + sn_vec[i] * h_mat[i+1][k];
            h_mat[i+1][k] = -sn_vec[i]*hik + cs_vec[i] * h_mat[i+1][k];
        }

        /* solve LS problem solving the system R*y = be1 */
        // copyvec(y_vec, be1_vec, k);
        // vec2vec(be1_vec, y_vec, k);
        // solve(h_mat, y_vec, k);
        backward_substitution(h_mat, be1_vec, y_vec, k);

        matvecmul(aux_vec, 1.0, q_mat, y_vec, dim, k);
        vecadd(xs_method, xp_vec, PLUS, aux_vec, dim);

        err_method = emethod(xe_sys, xs_method, dim);

        if (err_method <= TOL)
        {
            restart = m;
            break;
        }

        copyvec(xp_vec, xs_method, dim);
        m += 1;
    }

	/* finish time count */
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;


	/* Print final results */
    // dwritevector(xs_method, dim);
	//printf("error: %f\n", err_method );
	// printf("time: %f\n", cpu_time_used);
    info_method(err_method, cpu_time_used, restart);

    free_dmatrix(h_mat, 1, k+1, 1, k);
    free_dmatrix(q_mat, 1, dim, 1, k+1);
    free_dmatrix(a_mat, 1, dim, 1, dim);

    free_dvector(xe_sys, 1, dim);
    free_dvector(xs_method, 1, dim);
    free_dvector(y_vec, 1, k);
    free_dvector(aux_vec, 1, dim);
    free_dvector(cs_vec, 1, k);
    free_dvector(sn_vec, 1, k);
    free_dvector(qj_vec, 1, dim);
    free_dvector(vj_vec, 1, dim);
    free_dvector(be1_vec, 1, k+1);
    free_dvector(r_vec, 1, dim);
    free_dvector(xp_vec, 1, dim);
    free_dvector(b_vec, 1, dim);

	return 0;
}