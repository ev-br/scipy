#ifndef BLASLAPACK_DECLARATIONS_H
#define BLASLAPACK_DECLARATIONS_H

#include "arnaud/types.h"

#define saxpy_ BLAS_FUNC(saxpy)
#define scopy_ BLAS_FUNC(scopy)
#define sdot_ BLAS_FUNC(sdot)
#define sgemv_ BLAS_FUNC(sgemv)
#define sger_ BLAS_FUNC(sger)
#define snrm2_ BLAS_FUNC(snrm2)
#define srot_ BLAS_FUNC(srot)
#define sscal_ BLAS_FUNC(sscal)
#define sswap_ BLAS_FUNC(sswap)
#define strmm_ BLAS_FUNC(strmm)
#define daxpy_ BLAS_FUNC(daxpy)
#define dcopy_ BLAS_FUNC(dcopy)
#define ddot_ BLAS_FUNC(ddot)
#define dgemv_ BLAS_FUNC(dgemv)
#define dger_ BLAS_FUNC(dger)
#define dnrm2_ BLAS_FUNC(dnrm2)
#define drot_ BLAS_FUNC(drot)
#define dscal_ BLAS_FUNC(dscal)
#define dswap_ BLAS_FUNC(dswap)
#define dtrmm_ BLAS_FUNC(dtrmm)
#define caxpy_ BLAS_FUNC(caxpy)
#define ccopy_ BLAS_FUNC(ccopy)
#define cgeru_ BLAS_FUNC(cgeru)
#define scnrm2_ BLAS_FUNC(scnrm2)
#define cscal_ BLAS_FUNC(cscal)
#define csscal_ BLAS_FUNC(csscal)
#define cgemv_ BLAS_FUNC(cgemv)
#define crot_ BLAS_FUNC(crot)
#define ctrmm_ BLAS_FUNC(ctrmm)
#define zaxpy_ BLAS_FUNC(zaxpy)
#define zcopy_ BLAS_FUNC(zcopy)
#define zgeru_ BLAS_FUNC(zgeru)
#define dznrm2_ BLAS_FUNC(dznrm2)
#define zscal_ BLAS_FUNC(zscal)
#define zdscal_ BLAS_FUNC(zdscal)
#define zgemv_ BLAS_FUNC(zgemv)
#define zrot_ BLAS_FUNC(zrot)
#define ztrmm_ BLAS_FUNC(ztrmm)
#define sgeqr2_ BLAS_FUNC(sgeqr2)
#define slacpy_ BLAS_FUNC(slacpy)
#define slaev2_ BLAS_FUNC(slaev2)
#define slahqr_ BLAS_FUNC(slahqr)
#define slanhs_ BLAS_FUNC(slanhs)
#define slanst_ BLAS_FUNC(slanst)
#define slarf_ BLAS_FUNC(slarf)
#define slarfg_ BLAS_FUNC(slarfg)
#define slartg_ BLAS_FUNC(slartg)
#define slartgp_ BLAS_FUNC(slartgp)
#define slascl_ BLAS_FUNC(slascl)
#define slaset_ BLAS_FUNC(slaset)
#define slasr_ BLAS_FUNC(slasr)
#define sorm2r_ BLAS_FUNC(sorm2r)
#define ssteqr_ BLAS_FUNC(ssteqr)
#define strevc_ BLAS_FUNC(strevc)
#define strsen_ BLAS_FUNC(strsen)
#define dgeqr2_ BLAS_FUNC(dgeqr2)
#define dlacpy_ BLAS_FUNC(dlacpy)
#define dlaev2_ BLAS_FUNC(dlaev2)
#define dlahqr_ BLAS_FUNC(dlahqr)
#define dlanhs_ BLAS_FUNC(dlanhs)
#define dlanst_ BLAS_FUNC(dlanst)
#define dlarf_ BLAS_FUNC(dlarf)
#define dlarfg_ BLAS_FUNC(dlarfg)
#define dlartg_ BLAS_FUNC(dlartg)
#define dlartgp_ BLAS_FUNC(dlartgp)
#define dlascl_ BLAS_FUNC(dlascl)
#define dlaset_ BLAS_FUNC(dlaset)
#define dlasr_ BLAS_FUNC(dlasr)
#define dorm2r_ BLAS_FUNC(dorm2r)
#define dsteqr_ BLAS_FUNC(dsteqr)
#define dtrevc_ BLAS_FUNC(dtrevc)
#define dtrsen_ BLAS_FUNC(dtrsen)
#define cgeqr2_ BLAS_FUNC(cgeqr2)
#define clacpy_ BLAS_FUNC(clacpy)
#define clahqr_ BLAS_FUNC(clahqr)
#define clanhs_ BLAS_FUNC(clanhs)
#define clarf_ BLAS_FUNC(clarf)
#define clarfg_ BLAS_FUNC(clarfg)
#define clartg_ BLAS_FUNC(clartg)
#define clascl_ BLAS_FUNC(clascl)
#define claset_ BLAS_FUNC(claset)
#define ctrevc_ BLAS_FUNC(ctrevc)
#define ctrsen_ BLAS_FUNC(ctrsen)
#define cunm2r_ BLAS_FUNC(cunm2r)
#define zgeqr2_ BLAS_FUNC(zgeqr2)
#define zlacpy_ BLAS_FUNC(zlacpy)
#define zlahqr_ BLAS_FUNC(zlahqr)
#define zlanhs_ BLAS_FUNC(zlanhs)
#define zlarf_ BLAS_FUNC(zlarf)
#define zlarfg_ BLAS_FUNC(zlarfg)
#define zlartg_ BLAS_FUNC(zlartg)
#define zlascl_ BLAS_FUNC(zlascl)
#define zlaset_ BLAS_FUNC(zlaset)
#define ztrevc_ BLAS_FUNC(ztrevc)
#define ztrsen_ BLAS_FUNC(ztrsen)
#define zunm2r_ BLAS_FUNC(zunm2r)


// BLAS
void BLAS_FUNC(saxpy)(CBLAS_INT* n, float* alpha, float* x, CBLAS_INT* incx, float* y, CBLAS_INT* incy);
void BLAS_FUNC(scopy)(CBLAS_INT* n, float* x, CBLAS_INT* incx, float* y, CBLAS_INT* incy);
float BLAS_FUNC(sdot)(CBLAS_INT* n, float* x, CBLAS_INT* incx, float* y, CBLAS_INT* incy);
void BLAS_FUNC(sgemv)(char* trans, CBLAS_INT* m, CBLAS_INT* n, float* alpha, float* a, CBLAS_INT* lda, float* x, CBLAS_INT* incx, float* beta, float* y, CBLAS_INT* incy);
void BLAS_FUNC(sger)(CBLAS_INT* m, CBLAS_INT* n, float* alpha, float* x, CBLAS_INT* incx, float* y, CBLAS_INT* incy, float* a, CBLAS_INT* lda);
float BLAS_FUNC(snrm2)(CBLAS_INT* n, float* x, CBLAS_INT* incx);
void BLAS_FUNC(srot)(CBLAS_INT* n, float* sx, CBLAS_INT* incx, float* sy, CBLAS_INT* incy, float* c, float* s);
void BLAS_FUNC(sscal)(CBLAS_INT* n, float* alpha, float* x, CBLAS_INT* incx);
void BLAS_FUNC(sswap)(CBLAS_INT* n, float* x, CBLAS_INT* incx, float* y, CBLAS_INT* incy);
void BLAS_FUNC(strmm)(char* side, char* uplo, char* transa, char* diag, CBLAS_INT* m, CBLAS_INT* n, float* alpha, float* a, CBLAS_INT* lda, float* b, CBLAS_INT* ldb);


void BLAS_FUNC(daxpy)(CBLAS_INT* n, double* alpha, double* x, CBLAS_INT* incx, double* y, CBLAS_INT* incy);
void BLAS_FUNC(dcopy)(CBLAS_INT* n, double* x, CBLAS_INT* incx, double* y, CBLAS_INT* incy);
double BLAS_FUNC(ddot)(CBLAS_INT* n, double* x, CBLAS_INT* incx, double* y, CBLAS_INT* incy);
void BLAS_FUNC(dgemv)(char* trans, CBLAS_INT* m, CBLAS_INT* n, double* alpha, double* a, CBLAS_INT* lda, double* x, CBLAS_INT* incx, double* beta, double* y, CBLAS_INT* incy);
void BLAS_FUNC(dger)(CBLAS_INT* m, CBLAS_INT* n, double* alpha, double* x, CBLAS_INT* incx, double* y, CBLAS_INT* incy, double* a, CBLAS_INT* lda);
double BLAS_FUNC(dnrm2)(CBLAS_INT* n, double* x, CBLAS_INT* incx);
void BLAS_FUNC(drot)(CBLAS_INT* n, double* sx, CBLAS_INT* incx, double* sy, CBLAS_INT* incy, double* c, double* s);
void BLAS_FUNC(dscal)(CBLAS_INT* n, double* alpha, double* x, CBLAS_INT* incx);
void BLAS_FUNC(dswap)(CBLAS_INT* n, double* x, CBLAS_INT* incx, double* y, CBLAS_INT* incy);
void BLAS_FUNC(dtrmm)(char* side, char* uplo, char* transa, char* diag, CBLAS_INT* m, CBLAS_INT* n, double* alpha, double* a, CBLAS_INT* lda, double* b, CBLAS_INT* ldb);


void BLAS_FUNC(caxpy)(CBLAS_INT* n, ARNAUD_CPLXF_TYPE* alpha, ARNAUD_CPLXF_TYPE* x, CBLAS_INT* incx, ARNAUD_CPLXF_TYPE* y, CBLAS_INT* incy);
void BLAS_FUNC(ccopy)(CBLAS_INT* n, ARNAUD_CPLXF_TYPE* x, CBLAS_INT* incx, ARNAUD_CPLXF_TYPE* y, CBLAS_INT* incy);
void BLAS_FUNC(cgeru)(CBLAS_INT* m, CBLAS_INT* n, ARNAUD_CPLXF_TYPE* alpha, ARNAUD_CPLXF_TYPE* x, CBLAS_INT* incx, ARNAUD_CPLXF_TYPE* y, CBLAS_INT* incy, ARNAUD_CPLXF_TYPE* a, CBLAS_INT* lda);
float BLAS_FUNC(scnrm2)(CBLAS_INT* n, ARNAUD_CPLXF_TYPE* x, CBLAS_INT* incx);
void BLAS_FUNC(cscal)(CBLAS_INT* n, ARNAUD_CPLXF_TYPE* alpha, ARNAUD_CPLXF_TYPE* x, CBLAS_INT* incx);
void BLAS_FUNC(csscal)(CBLAS_INT* n, float* da, ARNAUD_CPLXF_TYPE* zx, CBLAS_INT* incx);
void BLAS_FUNC(cgemv)(char* trans, CBLAS_INT* m, CBLAS_INT* n, ARNAUD_CPLXF_TYPE* alpha, ARNAUD_CPLXF_TYPE* a, CBLAS_INT* lda, ARNAUD_CPLXF_TYPE* x, CBLAS_INT* incx, ARNAUD_CPLXF_TYPE* beta, ARNAUD_CPLXF_TYPE* y, CBLAS_INT* incy);
void BLAS_FUNC(crot)(CBLAS_INT* n, ARNAUD_CPLXF_TYPE* cx, CBLAS_INT* incx, ARNAUD_CPLXF_TYPE* cy, CBLAS_INT* incy, float* c, ARNAUD_CPLXF_TYPE* s);
void BLAS_FUNC(ctrmm)(char* side, char* uplo, char* transa, char* diag, CBLAS_INT* m, CBLAS_INT* n, ARNAUD_CPLXF_TYPE* alpha, ARNAUD_CPLXF_TYPE* a, CBLAS_INT* lda, ARNAUD_CPLXF_TYPE* b, CBLAS_INT* ldb);


void BLAS_FUNC(zaxpy)(CBLAS_INT* n, ARNAUD_CPLX_TYPE* alpha, ARNAUD_CPLX_TYPE* x, CBLAS_INT* incx, ARNAUD_CPLX_TYPE* y, CBLAS_INT* incy);
void BLAS_FUNC(zcopy)(CBLAS_INT* n, ARNAUD_CPLX_TYPE* x, CBLAS_INT* incx, ARNAUD_CPLX_TYPE* y, CBLAS_INT* incy);
void BLAS_FUNC(zgeru)(CBLAS_INT* m, CBLAS_INT* n, ARNAUD_CPLX_TYPE* alpha, ARNAUD_CPLX_TYPE* x, CBLAS_INT* incx, ARNAUD_CPLX_TYPE* y, CBLAS_INT* incy, ARNAUD_CPLX_TYPE* a, CBLAS_INT* lda);
double BLAS_FUNC(dznrm2)(CBLAS_INT* n, ARNAUD_CPLX_TYPE* x, CBLAS_INT* incx);
void BLAS_FUNC(zscal)(CBLAS_INT* n, ARNAUD_CPLX_TYPE* alpha, ARNAUD_CPLX_TYPE* x, CBLAS_INT* incx);
void BLAS_FUNC(zdscal)(CBLAS_INT* n, double* da, ARNAUD_CPLX_TYPE* zx, CBLAS_INT* incx);
void BLAS_FUNC(zgemv)(char* trans, CBLAS_INT* m, CBLAS_INT* n, ARNAUD_CPLX_TYPE* alpha, ARNAUD_CPLX_TYPE* a, CBLAS_INT* lda, ARNAUD_CPLX_TYPE* x, CBLAS_INT* incx, ARNAUD_CPLX_TYPE* beta, ARNAUD_CPLX_TYPE* y, CBLAS_INT* incy);
void BLAS_FUNC(zrot)(CBLAS_INT* n, ARNAUD_CPLX_TYPE* cx, CBLAS_INT* incx, ARNAUD_CPLX_TYPE* cy, CBLAS_INT* incy, double* c, ARNAUD_CPLX_TYPE* s);
void BLAS_FUNC(ztrmm)(char* side, char* uplo, char* transa, char* diag, CBLAS_INT* m, CBLAS_INT* n, ARNAUD_CPLX_TYPE* alpha, ARNAUD_CPLX_TYPE* a, CBLAS_INT* lda, ARNAUD_CPLX_TYPE* b, CBLAS_INT* ldb);



// LAPACK
void BLAS_FUNC(sgeqr2)(CBLAS_INT* m, CBLAS_INT* n, float* a, CBLAS_INT* lda, float* tau, float* work, CBLAS_INT* info);
void BLAS_FUNC(slacpy)(char* uplo, CBLAS_INT* m, CBLAS_INT* n, float* a, CBLAS_INT* lda, float* b, CBLAS_INT* ldb);
void BLAS_FUNC(slaev2)(float* a, float* b, float* c, float* rt1, float* rt2, float* cs1, float* sn1);
void BLAS_FUNC(slahqr)(CBLAS_INT* wantt, CBLAS_INT* wantz, CBLAS_INT* n, CBLAS_INT* ilo, CBLAS_INT* ihi, float* h, CBLAS_INT* ldh, float* wr, float* wi, CBLAS_INT* iloz, CBLAS_INT* ihiz, float* z, CBLAS_INT* ldz, CBLAS_INT* info );
float BLAS_FUNC(slanhs)(char* norm, CBLAS_INT* n, float* a, CBLAS_INT* lda, float* work);
float BLAS_FUNC(slanst)(char* norm, CBLAS_INT* n, float* d, float* e);
void BLAS_FUNC(slarf)(char* side, CBLAS_INT* m, CBLAS_INT* n, float* v, CBLAS_INT* incv, float* tau, float* c, CBLAS_INT* ldc, float* work);
void BLAS_FUNC(slarfg)(CBLAS_INT* n, float* alpha, float* x, CBLAS_INT* incx, float* tau);
void BLAS_FUNC(slartg)(float* f, float* g, float* c, float* s, float* r);
void BLAS_FUNC(slartgp)(float* f, float* g, float* c, float* s, float* r);
void BLAS_FUNC(slascl)(char* mtype, CBLAS_INT* kl, CBLAS_INT* ku, float* cfrom, float* cto, CBLAS_INT* m, CBLAS_INT* n, float* a, CBLAS_INT* lda, CBLAS_INT* info);
void BLAS_FUNC(slaset)(char* uplo, CBLAS_INT* m, CBLAS_INT* n, float* alpha, float* beta, float* a, CBLAS_INT* lda);
void BLAS_FUNC(slasr)(char* side, char* pivot, char* direct, CBLAS_INT* m, CBLAS_INT* n, float* c, float* s, float* a, CBLAS_INT* lda);
void BLAS_FUNC(sorm2r)(char* side, char* trans, CBLAS_INT* m, CBLAS_INT* n, CBLAS_INT* k, float* a, CBLAS_INT* lda, float* tau, float* c, CBLAS_INT* ldc, float* work, CBLAS_INT* info);
void BLAS_FUNC(ssteqr)(char* compz, CBLAS_INT* n, float* d, float* e, float* z, CBLAS_INT* ldz, float* work, CBLAS_INT* info);
void BLAS_FUNC(strevc)(char* side, char* howmny, CBLAS_INT* select, CBLAS_INT* n, float* t, CBLAS_INT* ldt, float* vl, CBLAS_INT* ldvl, float* vr, CBLAS_INT* ldvr, CBLAS_INT* mm, CBLAS_INT* m, float* work, CBLAS_INT* info);
void BLAS_FUNC(strsen)(char* job, char* compq, CBLAS_INT* select, CBLAS_INT* n, float* t, CBLAS_INT* ldt, float* q, CBLAS_INT* ldq, float* wr, float* wi, CBLAS_INT* m, float* s, float* sep, float* work, CBLAS_INT* lwork, CBLAS_INT* iwork, CBLAS_INT* liwork, CBLAS_INT* info);


void BLAS_FUNC(dgeqr2)(CBLAS_INT* m, CBLAS_INT* n, double* a, CBLAS_INT* lda, double* tau, double* work, CBLAS_INT* info);
void BLAS_FUNC(dlacpy)(char* uplo, CBLAS_INT* m, CBLAS_INT* n, double* a, CBLAS_INT* lda, double* b, CBLAS_INT* ldb);
void BLAS_FUNC(dlaev2)(double* a, double* b, double* c, double* rt1, double* rt2, double* cs1, double* sn1);
void BLAS_FUNC(dlahqr)(CBLAS_INT* wantt, CBLAS_INT* wantz, CBLAS_INT* n, CBLAS_INT* ilo, CBLAS_INT* ihi, double* h, CBLAS_INT* ldh, double* wr, double* wi, CBLAS_INT* iloz, CBLAS_INT* ihiz, double* z, CBLAS_INT* ldz, CBLAS_INT* info );
double BLAS_FUNC(dlanhs)(char* norm, CBLAS_INT* n, double* a, CBLAS_INT* lda, double* work);
double BLAS_FUNC(dlanst)(char* norm, CBLAS_INT* n, double* d, double* e);
void BLAS_FUNC(dlarf)(char* side, CBLAS_INT* m, CBLAS_INT* n, double* v, CBLAS_INT* incv, double* tau, double* c, CBLAS_INT* ldc, double* work);
void BLAS_FUNC(dlarfg)(CBLAS_INT* n, double* alpha, double* x, CBLAS_INT* incx, double* tau);
void BLAS_FUNC(dlartg)(double* f, double* g, double* c, double* s, double* r);
void BLAS_FUNC(dlartgp)(double* f, double* g, double* c, double* s, double* r);
void BLAS_FUNC(dlascl)(char* mtype, CBLAS_INT* kl, CBLAS_INT* ku, double* cfrom, double* cto, CBLAS_INT* m, CBLAS_INT* n, double* a, CBLAS_INT* lda, CBLAS_INT* info);
void BLAS_FUNC(dlaset)(char* uplo, CBLAS_INT* m, CBLAS_INT* n, double* alpha, double* beta, double* a, CBLAS_INT* lda);
void BLAS_FUNC(dlasr)(char* side, char* pivot, char* direct, CBLAS_INT* m, CBLAS_INT* n, double* c, double* s, double* a, CBLAS_INT* lda);
void BLAS_FUNC(dorm2r)(char* side, char* trans, CBLAS_INT* m, CBLAS_INT* n, CBLAS_INT* k, double* a, CBLAS_INT* lda, double* tau, double* c, CBLAS_INT* ldc, double* work, CBLAS_INT* info);
void BLAS_FUNC(dsteqr)(char* compz, CBLAS_INT* n, double* d, double* e, double* z, CBLAS_INT* ldz, double* work, CBLAS_INT* info);
void BLAS_FUNC(dtrevc)(char* side, char* howmny, CBLAS_INT* select, CBLAS_INT* n, double* t, CBLAS_INT* ldt, double* vl, CBLAS_INT* ldvl, double* vr, CBLAS_INT* ldvr, CBLAS_INT* mm, CBLAS_INT* m, double* work, CBLAS_INT* info);
void BLAS_FUNC(dtrsen)(char* job, char* compq, CBLAS_INT* select, CBLAS_INT* n, double* t, CBLAS_INT* ldt, double* q, CBLAS_INT* ldq, double* wr, double* wi, CBLAS_INT* m, double* s, double* sep, double* work, CBLAS_INT* lwork, CBLAS_INT* iwork, CBLAS_INT* liwork, CBLAS_INT* info);


void BLAS_FUNC(cgeqr2)(CBLAS_INT* m, CBLAS_INT* n, ARNAUD_CPLXF_TYPE* a, CBLAS_INT* lda, ARNAUD_CPLXF_TYPE* tau, ARNAUD_CPLXF_TYPE* work, CBLAS_INT* info);
void BLAS_FUNC(clacpy)(char* uplo, CBLAS_INT* m, CBLAS_INT* n, ARNAUD_CPLXF_TYPE* a, CBLAS_INT* lda, ARNAUD_CPLXF_TYPE* b, CBLAS_INT* ldb);
void BLAS_FUNC(clahqr)(CBLAS_INT* wantt, CBLAS_INT* wantz, CBLAS_INT* n, CBLAS_INT* ilo, CBLAS_INT* ihi, ARNAUD_CPLXF_TYPE* h, CBLAS_INT* ldh, ARNAUD_CPLXF_TYPE* w, CBLAS_INT* iloz, CBLAS_INT* ihiz, ARNAUD_CPLXF_TYPE* z, CBLAS_INT* ldz, CBLAS_INT* info );
float BLAS_FUNC(clanhs)(char* norm, CBLAS_INT* n, ARNAUD_CPLXF_TYPE* a, CBLAS_INT* lda, float* work);
void BLAS_FUNC(clarf)(char* side, CBLAS_INT* m, CBLAS_INT* n, ARNAUD_CPLXF_TYPE* v, CBLAS_INT* incv, ARNAUD_CPLXF_TYPE* tau, ARNAUD_CPLXF_TYPE* c, CBLAS_INT* ldc, ARNAUD_CPLXF_TYPE* work);
void BLAS_FUNC(clarfg)(CBLAS_INT* n, ARNAUD_CPLXF_TYPE* alpha, ARNAUD_CPLXF_TYPE* x, CBLAS_INT* incx, ARNAUD_CPLXF_TYPE* tau);
void BLAS_FUNC(clartg)(ARNAUD_CPLXF_TYPE* f, ARNAUD_CPLXF_TYPE* g, float* c, ARNAUD_CPLXF_TYPE* s, ARNAUD_CPLXF_TYPE* r);
void BLAS_FUNC(clascl)(char* mtype, CBLAS_INT* kl, CBLAS_INT* ku, float* cfrom, float* cto, CBLAS_INT* m, CBLAS_INT* n, ARNAUD_CPLXF_TYPE* a, CBLAS_INT* lda, CBLAS_INT* info);
void BLAS_FUNC(claset)(char* uplo, CBLAS_INT* m, CBLAS_INT* n, ARNAUD_CPLXF_TYPE* alpha, ARNAUD_CPLXF_TYPE* beta, ARNAUD_CPLXF_TYPE* a, CBLAS_INT* lda);
void BLAS_FUNC(ctrevc)(char* side, char* howmny, CBLAS_INT* select, CBLAS_INT* n, ARNAUD_CPLXF_TYPE* t, CBLAS_INT* ldt, ARNAUD_CPLXF_TYPE* vl, CBLAS_INT* ldvl, ARNAUD_CPLXF_TYPE* vr, CBLAS_INT* ldvr, CBLAS_INT* mm, CBLAS_INT* m, ARNAUD_CPLXF_TYPE* work, float* rwork, CBLAS_INT* info);
void BLAS_FUNC(ctrsen)(char* job, char* compq, CBLAS_INT* select, CBLAS_INT* n, ARNAUD_CPLXF_TYPE* t, CBLAS_INT* ldt, ARNAUD_CPLXF_TYPE* q, CBLAS_INT* ldq, ARNAUD_CPLXF_TYPE* w, CBLAS_INT* m, float* s, float* sep, ARNAUD_CPLXF_TYPE* work, CBLAS_INT* lwork, CBLAS_INT* info);
void BLAS_FUNC(cunm2r)(char* side, char* trans, CBLAS_INT* m, CBLAS_INT* n, CBLAS_INT* k, ARNAUD_CPLXF_TYPE* a, CBLAS_INT* lda, ARNAUD_CPLXF_TYPE* tau, ARNAUD_CPLXF_TYPE* c, CBLAS_INT* ldc, ARNAUD_CPLXF_TYPE* work, CBLAS_INT* info);


void BLAS_FUNC(zgeqr2)(CBLAS_INT* m, CBLAS_INT* n, ARNAUD_CPLX_TYPE* a, CBLAS_INT* lda, ARNAUD_CPLX_TYPE* tau, ARNAUD_CPLX_TYPE* work, CBLAS_INT* info);
void BLAS_FUNC(zlacpy)(char* uplo, CBLAS_INT* m, CBLAS_INT* n, ARNAUD_CPLX_TYPE* a, CBLAS_INT* lda, ARNAUD_CPLX_TYPE* b, CBLAS_INT* ldb);
void BLAS_FUNC(zlahqr)(CBLAS_INT* wantt, CBLAS_INT* wantz, CBLAS_INT* n, CBLAS_INT* ilo, CBLAS_INT* ihi, ARNAUD_CPLX_TYPE* h, CBLAS_INT* ldh, ARNAUD_CPLX_TYPE* w, CBLAS_INT* iloz, CBLAS_INT* ihiz, ARNAUD_CPLX_TYPE* z, CBLAS_INT* ldz, CBLAS_INT* info );
double BLAS_FUNC(zlanhs)(char* norm, CBLAS_INT* n, ARNAUD_CPLX_TYPE* a, CBLAS_INT* lda, double* work);
void BLAS_FUNC(zlarf)(char* side, CBLAS_INT* m, CBLAS_INT* n, ARNAUD_CPLX_TYPE* v, CBLAS_INT* incv, ARNAUD_CPLX_TYPE* tau, ARNAUD_CPLX_TYPE* c, CBLAS_INT* ldc, ARNAUD_CPLX_TYPE* work);
void BLAS_FUNC(zlarfg)(CBLAS_INT* n, ARNAUD_CPLX_TYPE* alpha, ARNAUD_CPLX_TYPE* x, CBLAS_INT* incx, ARNAUD_CPLX_TYPE* tau);
void BLAS_FUNC(zlartg)(ARNAUD_CPLX_TYPE* f, ARNAUD_CPLX_TYPE* g, double* c, ARNAUD_CPLX_TYPE* s, ARNAUD_CPLX_TYPE* r);
void BLAS_FUNC(zlascl)(char* mtype, CBLAS_INT* kl, CBLAS_INT* ku, double* cfrom, double* cto, CBLAS_INT* m, CBLAS_INT* n, ARNAUD_CPLX_TYPE* a, CBLAS_INT* lda, CBLAS_INT* info);
void BLAS_FUNC(zlaset)(char* uplo, CBLAS_INT* m, CBLAS_INT* n, ARNAUD_CPLX_TYPE* alpha, ARNAUD_CPLX_TYPE* beta, ARNAUD_CPLX_TYPE* a, CBLAS_INT* lda);
void BLAS_FUNC(ztrevc)(char* side, char* howmny, CBLAS_INT* select, CBLAS_INT* n, ARNAUD_CPLX_TYPE* t, CBLAS_INT* ldt, ARNAUD_CPLX_TYPE* vl, CBLAS_INT* ldvl, ARNAUD_CPLX_TYPE* vr, CBLAS_INT* ldvr, CBLAS_INT* mm, CBLAS_INT* m, ARNAUD_CPLX_TYPE* work, double* rwork, CBLAS_INT* info);
void BLAS_FUNC(ztrsen)(char* job, char* compq, CBLAS_INT* select, CBLAS_INT* n, ARNAUD_CPLX_TYPE* t, CBLAS_INT* ldt, ARNAUD_CPLX_TYPE* q, CBLAS_INT* ldq, ARNAUD_CPLX_TYPE* w, CBLAS_INT* m, double* s, double* sep, ARNAUD_CPLX_TYPE* work, CBLAS_INT* lwork, CBLAS_INT* info);
void BLAS_FUNC(zunm2r)(char* side, char* trans, CBLAS_INT* m, CBLAS_INT* n, CBLAS_INT* k, ARNAUD_CPLX_TYPE* a, CBLAS_INT* lda, ARNAUD_CPLX_TYPE* tau, ARNAUD_CPLX_TYPE* c, CBLAS_INT* ldc, ARNAUD_CPLX_TYPE* work, CBLAS_INT* info);


#endif
