/* ************************** */
/*      Utility Routines      */
/*     Numerical  Recipes     */
/* ************************** */

/* standard error handler */
void nrerror(char error_text[]);

/* allocates an int vector with subscript range v[nl..nh] */
int *ivector(long nl, long nh);

/* free an int vector allocated with ivector() */
void free_ivector(int *v, long ln, long nh);

/* allocates an double vector with subscript range v[nl..nh] */
double *dvector(long nl, long nh);

/* free an double vector allocated with ivector() */
void free_dvector(double *v, long ln, long nh);

/* allocates a double matrix with range [nrl..nrh][ncl..nch] */
double *dmatrix(long nl, long nh);

/* free a double matrix */
void free_dmatrix(double *m, long ln, long nh);

/* ******************************* */
/* Functions for build test linear */
/* equation system                 */
/* ******************************* */

/* computes A matrix */
void matrix225(double *a, int dim);

/* computes b vector */
void vector225(double *b, int dim);

/* *************************************** */
/*    Numerical linear algebra functions   */
/* *************************************** */

/* compute inverse of a diagonal matrix */
void invdiagmat(double *invd, double *d, int dim);

/* Computes A = D+L+U split */

/* obtain the diagonal of a matrix */
/* D <- diag(A) */
void diagmat(double *d, double *a, int dim);

/* obtain the portion lower triangular strict of a matrix */
/* L <- lowtrist(A) */
void lotrmat(double *l, double *a, int dim);

/* obtain the portion upper triangular strict of a matrix */
/* U <- uptrist(A) */
void uptrmat(double *u, double *a, int dim);

/* computes error of xs_method approximation */
double emethod(double *xe, double *xs, int dim);

/* Output routines */

/* write a int vector */
void iwritevector(int *a, int dim);

/* write a double vector */
void dwritevector(double *a, int dim);

/* write a double matrix */
void dwritematrix(double *a, int rows, int cols);

/* write the relative error, run time and entirely iterations of methods */
void info_method(double error_method, double time, int iter_max);