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
float *fmatrix(long nl, long nh);

/* free a float matrix */
void free_fmatrix(float *m, long ln, long nh);

/* ******************************* */
/* Functions for build test linear */
/* equation system                 */
/* ******************************* */

/* computes A matrix */
void matrix225(float *a, int dim);

/* computes b vector */
void vector225(float *b, int dim);

/* *************************************** */
/*    Numerical linear algebra functions   */
/* *************************************** */

/* compute inverse of a diagonal matrix */
void invdiagmat(float *invd, float *d, int dim);

/* Computes A = D+L+U split */

/* obtain the diagonal of a matrix */
/* D <- diag(A) */
void diagmat(float *d, float *a, int dim);

/* obtain the portion lower triangular strict of a matrix */
/* L <- lowtrist(A) */
void lotrmat(float *l, float *a, int dim);

/* obtain the portion upper triangular strict of a matrix */
/* U <- uptrist(A) */
void uptrmat(float *u, float *a, int dim);

/* Output routines */

/* write a int vector */
void iwritevector(int *a, int dim);

/* write a float vector */
void fwritevector(float *a, int dim);

/* write a float matrix */
void fwritematrix(float *a, int rows, int cols);