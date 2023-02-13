/* ************************** */
/*      Utility Routines      */
/*     Numerical  Recipes     */
/* ************************** */

/* standard error handler */
void nrerror(char error_text[]);

/* allocates an float vector with subscript range v[nl..nh] */
int *ivector(long nl, long nh);

/* free an int vector allocated with ivector() */
void free_ivector(int *v, long ln, long nh);

/* allocates an float vector with subscript range v[nl..nh] */
float *fvector(long nl, long nh);

/* free an float vector allocated with fvector() */
void free_fvector(float *v, long ln, long nh);

/* allocates an double vector with subscript range v[nl..nh] */
double *dvector(long nl, long nh);

/* free an double vector allocated with dvector() */
void free_dvector(double *v, long ln, long nh);

/* allocates a float matrix with subscript range m[nrl..nrh][ncl..nch] */
float **fmatrix(long nrl, long nrh, long ncl, long nch);

/* free a float matrix allocated with fmatrix() */
void free_fmatrix(float **m, long nrl, long nrh, long ncl, long nch);

/* allocates a double matrix with range m[nrl..nrh][ncl..nch] */
double **dmatrix(long nrl, long nrh, long ncl, long nch);

/* free a double matrix allocated with dmatrix() */
void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch);

/* ******************************* */
/* Functions for build test linear */
/* equation system                 */
/* ******************************* */

/* computes A matrix */
void matrix225(double **a, int dim);

/* computes b vector */
void vector225(double *b, int dim);

/* *************************************** */
/*    Numerical linear algebra functions   */
/* *************************************** */

/* LU decomposition of a row wise permutation */
void ludcmp(double **a, int dim, int *indx, double *d);

/* LU decomposition and backward substitution */
void lubksb(double **a, int dim, int *indx, double b[]);

/* solve the linear set of equations Ax=b */
void solve(double **a, double *b, int dim);

/* compute inverse of a matrix */
void inverse(double **a, double **y, int dim);

/* compute inverse of a diagonal matrix */
void invdiagmat(double **invd, double **d, int dim);

/* compute 2-norm of a vector */
double normvec_2(double *a, int dim);

/* compute 2-norm of a matrix */
double normmat_2(double **a, int dim);

/* copy a vectors to other vector */
/* y <- x */
void copyvec(double *y, double *x, int dim);

/* compute dot product of two vectors */
/* 〈x, y〉  */
double dotvec(double *x, double *y, int dim);

/* computes vector addition */
/* z <- y + alpha*x */
void vecadd(double *z, double *y, double alpha, double *x, int dim);

/* computes scalar-vector multiplication */
/* y <- alpha*x */
void scavecmul(double *y, double alpha, double *x, int dim);

/* computes matrix-vector multiplication */
/* y <- alpha*A*x */
void matvecmul(double *y, double alpha, double **a, double *x, int row, int column);

/* computes scalar-matrix multiplication */
/* B <- alpha*A */
void scamatmul(double **b, double alpha, double **a, int dim);

/* computes matrix addition */
/* C <- B + alpha*A */
void matadd(double **c, double **b, double alpha, double **a, int dim);

/* computes error of xs_method approximation */
double emethod(double *xe, double *xs, int dim);

/* Computes A = D+L+U split */

/* obtain the diagonal of a matrix */
/* D <- diag(A) */
void diagmat(double **d, double **a, int dim);

/* obtain the portion lower triangular strict of a matrix */
/* L <- lowtrist(A) */
void lotrmat(double **l, double **a, int dim);

/* obtain the portion upper triangular strict of a matrix */
/* U <- uptrist(A) */
void uptrmat(double **u, double **a, int dim);

/* Functions for GMRES Method */

/* get the ith colum of a matrix and put in a vector */
void mat2vec(double **a, double *x, int index, int dim);

/* put the elements of a vector in the ith colum of a matrix */
void vec2mat(double *x, double **a, int index, int dim);

/* compute x tal que U*x = b, where U is a upper triangular matrix */
void backward_substitution(double **u, double *b, double *x, int dim);

/* Output routines */

/* write a int vector */
void iwritevector(int *a, int dim);

/* write a double vector */
void dwritevector(double *a, int dim);

/* write a double matrix */
void dwritematrix(double **a, int rows, int cols);

/* write the relative error, run time and entirely iterations of methods */
void info_method(double error_method, double time, int iter_max);