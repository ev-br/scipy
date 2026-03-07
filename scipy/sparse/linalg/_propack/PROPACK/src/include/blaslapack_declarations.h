#ifndef BLASLAPACK_DECLARATIONS_H
#define BLASLAPACK_DECLARATIONS_H

#include "npy_cblas.h"
#include "../include/propack/types.h"

// BLAS
void BLAS_FUNC(saxpy)(int* n, float* alpha, float* x, int* incx, float* y, int* incy);
void BLAS_FUNC(scopy)(int* n, float* x, int* incx, float* y, int* incy);
float BLAS_FUNC(sdot)(int* n, float* x, int* incx, float* y, int* incy);
void BLAS_FUNC(sgemm)(char* transa, char* transb, int* m, int* n, int* k, float* alpha, float* a, int* lda, float* b, int* ldb, float* beta, float* c, int* ldc);
void BLAS_FUNC(sgemv)(char* trans, int* m, int* n, float* alpha, float* a, int* lda, float* x, int* incx, float* beta, float* y, int* incy);
float BLAS_FUNC(snrm2)(int* n, float* x, int* incx);
void BLAS_FUNC(srot)(int* n, float* sx, int* incx, float* sy, int* incy, float* c, float* s);
void BLAS_FUNC(sscal)(int* n, float* alpha, float* x, int* incx);

void BLAS_FUNC(daxpy)(int* n, double* alpha, double* x, int* incx, double* y, int* incy);
void BLAS_FUNC(dcopy)(int* n, double* x, int* incx, double* y, int* incy);
double BLAS_FUNC(ddot)(int* n, double* x, int* incx, double* y, int* incy);
void BLAS_FUNC(dgemm)(char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc);
void BLAS_FUNC(dgemv)(char* trans, int* m, int* n, double* alpha, double* a, int* lda, double* x, int* incx, double* beta, double* y, int* incy);
double BLAS_FUNC(dnrm2)(int* n, double* x, int* incx);
void BLAS_FUNC(drot)(int* n, double* sx, int* incx, double* sy, int* incy, double* c, double* s);
void BLAS_FUNC(dscal)(int* n, double* alpha, double* x, int* incx);

void BLAS_FUNC(caxpy)(int* n, PROPACK_CPLXF_TYPE* alpha, PROPACK_CPLXF_TYPE* x, int* incx, PROPACK_CPLXF_TYPE* y, int* incy);
float BLAS_FUNC(scnrm2)(int* n, PROPACK_CPLXF_TYPE* x, int* incx);
void BLAS_FUNC(cgemm)(char* transa, char* transb, int* m, int* n, int* k, PROPACK_CPLXF_TYPE* alpha, PROPACK_CPLXF_TYPE* a, int* lda, PROPACK_CPLXF_TYPE* b, int* ldb, PROPACK_CPLXF_TYPE* beta, PROPACK_CPLXF_TYPE* c, int* ldc);
void BLAS_FUNC(cgemv)(char* trans, int* m, int* n, PROPACK_CPLXF_TYPE* alpha, PROPACK_CPLXF_TYPE* a, int* lda, PROPACK_CPLXF_TYPE* x, int* incx, PROPACK_CPLXF_TYPE* beta, PROPACK_CPLXF_TYPE* y, int* incy);
void BLAS_FUNC(csscal)(int* n, float* da, PROPACK_CPLXF_TYPE* zx, int* incx);

void BLAS_FUNC(zaxpy)(int* n, PROPACK_CPLX_TYPE* alpha, PROPACK_CPLX_TYPE* x, int* incx, PROPACK_CPLX_TYPE* y, int* incy);
double BLAS_FUNC(dznrm2)(int* n, PROPACK_CPLX_TYPE* x, int* incx);
void BLAS_FUNC(zgemm)(char* transa, char* transb, int* m, int* n, int* k, PROPACK_CPLX_TYPE* alpha, PROPACK_CPLX_TYPE* a, int* lda, PROPACK_CPLX_TYPE* b, int* ldb, PROPACK_CPLX_TYPE* beta, PROPACK_CPLX_TYPE* c, int* ldc);
void BLAS_FUNC(zgemv)(char* trans, int* m, int* n, PROPACK_CPLX_TYPE* alpha, PROPACK_CPLX_TYPE* a, int* lda, PROPACK_CPLX_TYPE* x, int* incx, PROPACK_CPLX_TYPE* beta, PROPACK_CPLX_TYPE* y, int* incy);
void BLAS_FUNC(zdscal)(int* n, double* da, PROPACK_CPLX_TYPE* zx, int* incx);

// LAPACK

void BLAS_FUNC(sbdsdc)(char* uplo, char* compq, int* n, float* d, float* e, float* u, int* ldu, float* vt, int* ldvt, float* q, int* iq, float* work, int* iwork, int* info);
void BLAS_FUNC(sbdsqr)(char* uplo, int* n, int* ncvt, int* nru, int* ncc, float* d, float* e, float* vt, int* ldvt, float* u, int* ldu, float* c, int* ldc, float* work, int* info);
void BLAS_FUNC(slartg)(float* f, float* g, float* c, float* s, float* r);
void BLAS_FUNC(slascl)(char* mtype, int* kl, int* ku, float* cfrom, float* cto, int* m, int* n, float* a, int* lda, int* info);
void BLAS_FUNC(slaset)(char* uplo, int* m, int* n, float* alpha, float* beta, float* a, int* lda);

void BLAS_FUNC(dbdsdc)(char* uplo, char* compq, int* n, double* d, double* e, double* u, int* ldu, double* vt, int* ldvt, double* q, int* iq, double* work, int* iwork, int* info);
void BLAS_FUNC(dbdsqr)(char* uplo, int* n, int* ncvt, int* nru, int* ncc, double* d, double* e, double* vt, int* ldvt, double* u, int* ldu, double* c, int* ldc, double* work, int* info);
void BLAS_FUNC(dlartg)(double* f, double* g, double* c, double* s, double* r);
void BLAS_FUNC(dlascl)(char* mtype, int* kl, int* ku, double* cfrom, double* cto, int* m, int* n, double* a, int* lda, int* info);
void BLAS_FUNC(dlaset)(char* uplo, int* m, int* n, double* alpha, double* beta, double* a, int* lda);

void BLAS_FUNC(clarfg)(int* n, PROPACK_CPLXF_TYPE* alpha, PROPACK_CPLXF_TYPE* x, int* incx, PROPACK_CPLXF_TYPE* tau);
void BLAS_FUNC(clascl)(char* mtype, int* kl, int* ku, float* cfrom, float* cto, int* m, int* n, PROPACK_CPLXF_TYPE* a, int* lda, int* info);

void BLAS_FUNC(zlarfg)(int* n, PROPACK_CPLX_TYPE* alpha, PROPACK_CPLX_TYPE* x, int* incx, PROPACK_CPLX_TYPE* tau);
void BLAS_FUNC(zlascl)(char* mtype, int* kl, int* ku, double* cfrom, double* cto, int* m, int* n, PROPACK_CPLX_TYPE* a, int* lda, int* info);


// (c,z)dotc is the complex conjugate dot product of two complex vectors.
// Due some historical reasons, this function can cause segfaults on some
// platforms. Hence implemented here instead of using the BLAS version.
static PROPACK_CPLXF_TYPE
BLAS_FUNC(cdotc)(const int* n, const PROPACK_CPLXF_TYPE* restrict x, const int* incx, const PROPACK_CPLXF_TYPE* restrict y, const int* incy)
{
    PROPACK_CPLXF_TYPE result = PROPACK_cplxf(0.0, 0.0);
#ifdef _MSC_VER
    PROPACK_CPLXF_TYPE temp = PROPACK_cplxf(0.0, 0.0);
#endif
    if (*n <= 0) { return result; }
    if ((*incx == 1) && (*incy == 1))
    {
        for (int i = 0; i < *n; i++)
        {
#ifdef _MSC_VER
            temp = _FCmulcc(conjf(x[i]), y[i]);
            result = PROPACK_cplxf(crealf(result) + crealf(temp), cimagf(result) + cimagf(temp));
#else
            result = result + (conjf(x[i]) * y[i]);
#endif
        }

    } else {

        for (int i = 0; i < *n; i++)
        {
#ifdef _MSC_VER
            temp = _FCmulcc(conjf(x[i * (*incx)]), y[i * (*incy)]);
            result = PROPACK_cplxf(crealf(result) + crealf(temp), cimagf(result) + cimagf(temp));
#else
            result = result + (conjf(x[i * (*incx)]) * y[i * (*incy)]);
#endif
        }
    }

    return result;
}


static PROPACK_CPLX_TYPE
BLAS_FUNC(zdotc)(const int* n, const PROPACK_CPLX_TYPE* restrict x, const int* incx, const PROPACK_CPLX_TYPE* restrict y, const int* incy)
{
    PROPACK_CPLX_TYPE result = PROPACK_cplx(0.0, 0.0);
#ifdef _MSC_VER
    PROPACK_CPLX_TYPE temp = PROPACK_cplx(0.0, 0.0);
#endif
    if (*n <= 0) { return result; }
    if ((*incx == 1) && (*incy == 1))
    {
        for (int i = 0; i < *n; i++)
        {
#ifdef _MSC_VER
            temp = _Cmulcc(conj(x[i]), y[i]);
            result = PROPACK_cplx(creal(result) + creal(temp), cimag(result) + cimag(temp));
#else
            result = result + (conj(x[i]) * y[i]);
#endif
        }

    } else {

        for (int i = 0; i < *n; i++)
        {
#ifdef _MSC_VER
            temp = _Cmulcc(conj(x[i * (*incx)]), y[i * (*incy)]);
            result = PROPACK_cplx(creal(result) + creal(temp), cimag(result) + cimag(temp));
#else
            result = result + (conj(x[i * (*incx)]) * y[i * (*incy)]);
#endif
        }
    }

    return result;
}

#endif
