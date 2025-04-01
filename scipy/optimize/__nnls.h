#ifndef __NNLS_H
#define __NNLS_H
#include <math.h>
#include "../_build_utils/src/npy_cblas.h"

void BLAS_FUNC(dlarf)(char* side, CBLAS_INT* m, CBLAS_INT* n, double* v, CBLAS_INT* incv, double* tau, double* c, CBLAS_INT* ldc, double* work);
void BLAS_FUNC(dlarfgp)(CBLAS_INT* n, double* alpha, double* x, CBLAS_INT* incx, double* tau);
void BLAS_FUNC(dlartgp)(double* f, double* g, double* cs, double* sn, double* r);
double BLAS_FUNC(dnrm2)(CBLAS_INT* n, double* x, CBLAS_INT* incx);

void
__nnls(const CBLAS_INT m, const CBLAS_INT n, double* restrict a, double* restrict b,
       double* restrict x, double* restrict w, double* restrict zz,
       CBLAS_INT* restrict indices, const CBLAS_INT maxiter, double* rnorm, CBLAS_INT* info);


#endif
