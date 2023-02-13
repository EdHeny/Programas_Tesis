/* ************************** */
/*      Utility Routines      */
/*     Numerical  Recipes     */
/* ************************** */

/* standard error handler */
void nrerror(char error_text[]);

/* allocates an float vector with subscrip range v[nl..nh] */
int *ivector(long nl, long nh);

/* free an int vector allocated with ivector() */
void free_ivector(int *v, long ln, long nh);

/* allocates an float vector with subscrip range v[nl..nh] */
float *fvector(long nl, long nh);

/* free an float vector allocated with ivector() */
void free_fvector(float *v, long ln, long nh);

/* allocates a float matrix with range [nrl..nrh][ncl..nch] */
float **fmatrix(long nrl, long nrh, long ncl, long nch);

/* free a float matrix */
void free_fmatrix(float **m, long nrl, long nrh, long ncl, long nch);

/* ******************************* */
/* Functions for build test linear */
/* equation system                 */
/* ******************************* */

/* computes A matrix */
void matrix225(float **a, int dim);

/* computes b vector */
void vector225(float *b, int dim);

/* *************************************** */
/*    Numerical linear algebra functions   */
/* *************************************** */

/* LU decomposition of a rowwise permutation */
void ludcmp(float **a, int dim, int *indx, float *d);

/* LU decomposition and backsubstitution */
void lubksb(float **a, int dim, int *indx, float b[]);

/* solve the linear set of equations Ax=b */
void solve(float **a, float *b, int dim);

/* compute inverse of a matrix */
void inverse(float **a, float **y, int dim);

/* compute inverse of a diagonal matrix */
void invdiagmat(float **invd, float **d, int dim);

/* compute 2-norm of a vector */
double normvec_2(float *a, int dim);

/* compute 2-norm of a matrix */
double normmat_2(float **a, int dim);

/* copy a vectors to other vector */
/* y <- x */
void copyvec(float *y, float *x, int dim);

/* compute dot product of two vectors */
/* 〈x, y〉  */
float dotvec(float *x, float *y, int dim);

/* computes vector addition */
/* z <- y + alpha*x */
void vecadd(float *z, float *y, float alpha, float *x, int dim);

/* computes scalar-vector multiplication */
/* y <- alpha*x */
void scavecmul(float *y, float alpha, float *x, int dim);

/* computes matrix-vector multiplication */
/* y <- alpha*A*x */
void matvecmul(float *y, float alpha, float **a, float *x, int dim);

/* computes scalar-matrix multiplication */
/* B <- alpha*A */
void scamatmul(float **b, float alpha, float **a, int dim);

/* computes matrix addition */
/* C <- B + alpha*A */
void matadd(float **c, float **b, float alpha, float **a, int dim);

/* computes error of xs_method approximation */
float emethod(float **a, float *b, float *xs, int dim);

/* Computes A = D+L+U split */

/* obtain the diagonal of a matrix */
/* D <- diag(A) */
void diagmat(float **d, float **a, int dim);

/* obtain the portion lower triangular strict of a matrix */
/* L <- lowtrist(A) */
void lotrmat(float **l, float **a, int dim);

/* obtain the portion upper triangular strict of a matrix */
/* U <- uptrist(A) */
void uptrmat(float **u, float **a, int dim);

/* Output routines */

/* write a int vector */
void iwritevector(int *a, int dim);

/* write a float vector */
void fwritevector(float *a, int dim);

/* write a float matrix */
void fwritematrix(float **a, int rows, int cols);