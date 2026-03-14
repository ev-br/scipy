#ifndef PROPACK_LANBPRO_H
#define PROPACK_LANBPRO_H

#include "common.h"
#include "getu0.h"
#include "gs.h"
#include "blaslapack_declarations.h"


/**
 * Lanczos Bidiagonalization with Partial Reorthogonalization - Single precision
 *
 * Computes K steps of the Lanczos bidiagonalization (LBD) algorithm with partial
 * reorthogonalization (BPRO) with M-by-1 starting vector U(:,k0+1), producing a
 * lower bidiagonal K+1-by-K matrix B_k, an N-by-K matrix V_k, an M-by-K+1 matrix
 * U_{k+1} such that A*V_k = U_{k+1}*B_k.
 *
 * Partial reorthogonalization is used to keep the columns of V_K and U_k
 * semiorthogonal to a level prescribed in doption[0], i.e.
 * MAX(DIAG((EYE(K) - V_K'*V_K))) <= doption[0]
 * and
 * MAX(DIAG((EYE(K) - U_K'*U_K))) <= doption[0].
 *
 * If K0>0 and K>K0 an existing K0-step LBD stored in U, V and B is extended to a K-step LBD.
 *
 * @param m          Number of rows of A
 * @param n          Number of columns of A
 * @param k0         Dimension of previously computed Lanczos bidiagonalization stored in U, V, B
 * @param k          On entry: desired dimension of LBD. On exit: actual size computed
 * @param aprod      Matrix-vector operation callback function
 * @param U          Left Lanczos vectors (M x K+1, column-major)
 * @param ldu        Leading dimension of U (>= M)
 * @param V          Right Lanczos vectors (N x K, column-major)
 * @param ldv        Leading dimension of V (>= N)
 * @param B          Bidiagonal matrix (K x 2, column-major: diag in col 1, sub-diag in col 2)
 * @param ldb        Leading dimension of B (>= K)
 * @param rnorm      On entry: norm of starting vector. On exit: (K+1,K) element of B_k
 * @param doption    Real options array: [delta, eta, anorm]
 *                   doption[0] = delta: Level of orthogonality (negative for sqrt(eps/k) default)
 *                   doption[1] = eta: Reorthogonalization threshold (negative for eps^(3/4)/sqrt(k) default)
 *                   doption[2] = anorm: Estimate of ||A|| (negative for automatic estimation)
 * @param ioption    Integer options array: [cgs, elr]
 *                   ioption[0] = cgs: Orthogonalization method (0=MGS, 1=CGS)
 *                   ioption[1] = elr: Extended local reorthogonality (0=off, >0=number of ELR steps)
 * @param work       Workspace array of size >= 2*(m+n+k+1)
 * @param iwork      Integer workspace array of size >= 2*k+1
 * @param dparm      Real parameters for aprod
 * @param iparm      Integer parameters for aprod
 * @param ierr       Error status (0=success, <0=invariant subspace found, >0=near-invariant)
 * @param rng_state  Random number generator state (uint64_t[4])
 */
void slanbpro(PROPACK_INT m, PROPACK_INT n, PROPACK_INT k0, PROPACK_INT* k, PROPACK_aprod_s aprod,
              float* U, PROPACK_INT ldu, float* V, PROPACK_INT ldv, float* B, PROPACK_INT ldb,
              float* rnorm, float* doption, PROPACK_INT* ioption, float* work, PROPACK_INT* iwork,
              float* dparm, PROPACK_INT* iparm, PROPACK_INT* ierr, uint64_t* rng_state);


/**
 * Lanczos Bidiagonalization with Partial Reorthogonalization - Double precision
 *
 * Double precision version of slanbpro. See slanbpro documentation for details.
 *
 * @param m          Number of rows of A
 * @param n          Number of columns of A
 * @param k0         Dimension of previously computed Lanczos bidiagonalization stored in U, V, B
 * @param k          On entry: desired dimension of LBD. On exit: actual size computed
 * @param aprod      Matrix-vector operation callback function
 * @param U          Left Lanczos vectors (M x K+1, column-major)
 * @param ldu        Leading dimension of U (>= M)
 * @param V          Right Lanczos vectors (N x K, column-major)
 * @param ldv        Leading dimension of V (>= N)
 * @param B          Bidiagonal matrix (K x 2, column-major: diag in col 1, sub-diag in col 2)
 * @param ldb        Leading dimension of B (>= K)
 * @param rnorm      On entry: norm of starting vector. On exit: (K+1,K) element of B_k
 * @param doption    Real options array: [delta, eta, anorm]
 * @param ioption    Integer options array: [cgs, elr]
 * @param work       Workspace array of size >= 2*(m+n+k+1)
 * @param iwork      Integer workspace array of size >= 2*k+1
 * @param dparm      Real parameters for aprod
 * @param iparm      Integer parameters for aprod
 * @param ierr       Error status (0=success, <0=invariant subspace found, >0=near-invariant)
 * @param rng_state  Random number generator state (uint64_t[4])
 */
void dlanbpro(PROPACK_INT m, PROPACK_INT n, PROPACK_INT k0, PROPACK_INT* k, PROPACK_aprod_d aprod,
              double* U, PROPACK_INT ldu, double* V, PROPACK_INT ldv, double* B, PROPACK_INT ldb,
              double* rnorm, double* doption, PROPACK_INT* ioption, double* work, PROPACK_INT* iwork,
              double* dparm, PROPACK_INT* iparm, PROPACK_INT* ierr, uint64_t* rng_state);


/**
 * Lanczos Bidiagonalization with Partial Reorthogonalization - Single precision complex
 *
 * Complex single precision version of slanbpro. Computes K steps of the Lanczos
 * bidiagonalization (LBD) algorithm with partial reorthogonalization (BPRO) with
 * M-by-1 starting vector U(:,k0+1), producing a lower bidiagonal K+1-by-K matrix B_k,
 * an N-by-K matrix V_k, an M-by-K+1 matrix U_{k+1} such that A*V_k = U_{k+1}*B_k.
 *
 * Partial reorthogonalization is used to keep the columns of V_K and U_k
 * semiorthogonal to a level prescribed in doption[0], i.e.
 * MAX(DIAG((EYE(K) - V_K'*V_K))) <= doption[0]
 * and
 * MAX(DIAG((EYE(K) - U_K'*U_K))) <= doption[0].
 *
 * If K0>0 and K>K0 an existing K0-step LBD stored in U, V and B is extended to a K-step LBD.
 *
 * @param m          Number of rows of A
 * @param n          Number of columns of A
 * @param k0         Dimension of previously computed Lanczos bidiagonalization stored in U, V, B
 * @param k          On entry: desired dimension of LBD. On exit: actual size computed
 * @param aprod      Matrix-vector operation callback function
 * @param U          Left Lanczos vectors (M x K+1, column-major, complex)
 * @param ldu        Leading dimension of U (>= M)
 * @param V          Right Lanczos vectors (N x K, column-major, complex)
 * @param ldv        Leading dimension of V (>= N)
 * @param B          Bidiagonal matrix (K x 2, column-major: diag in col 1, sub-diag in col 2, real)
 * @param ldb        Leading dimension of B (>= K)
 * @param rnorm      On entry: norm of starting vector. On exit: (K+1,K) element of B_k
 * @param doption    Real options array: [delta, eta, anorm]
 * @param ioption    Integer options array: [cgs, elr]
 * @param swork      Real workspace array of size >= m+n+2*k+2+max(m,n)
 * @param cwork      Complex workspace array of size >= max(m,n)
 * @param iwork      Integer workspace array of size >= 2*k+1
 * @param cparm      Complex parameters for aprod
 * @param iparm      Integer parameters for aprod
 * @param ierr       Error status (0=success, <0=invariant subspace found, >0=near-invariant)
 * @param rng_state  Random number generator state (uint64_t[4])
 */
void clanbpro(
    PROPACK_INT m, PROPACK_INT n, PROPACK_INT k0, PROPACK_INT* k, PROPACK_aprod_c aprod, PROPACK_CPLXF_TYPE* U, PROPACK_INT ldu,
    PROPACK_CPLXF_TYPE* V, PROPACK_INT ldv, float* B, PROPACK_INT ldb, float* rnorm, float* soption,
    PROPACK_INT* ioption, float* swork, PROPACK_CPLXF_TYPE* cwork, PROPACK_INT* iwork, PROPACK_CPLXF_TYPE* cparm,
    PROPACK_INT* iparm, PROPACK_INT* ierr, uint64_t* rng_state);


/**
 * Lanczos Bidiagonalization with Partial Reorthogonalization - Double precision complex
 *
 * Complex double precision version of dlanbpro. Computes K steps of the Lanczos
 * bidiagonalization (LBD) algorithm with partial reorthogonalization (BPRO) with
 * M-by-1 starting vector U(:,k0+1), producing a lower bidiagonal K+1-by-K matrix B_k,
 * an N-by-K matrix V_k, an M-by-K+1 matrix U_{k+1} such that A*V_k = U_{k+1}*B_k.
 *
 * Partial reorthogonalization is used to keep the columns of V_K and U_k
 * semiorthogonal to a level prescribed in doption[0], i.e.
 * MAX(DIAG((EYE(K) - V_K'*V_K))) <= doption[0]
 * and
 * MAX(DIAG((EYE(K) - U_K'*U_K))) <= doption[0].
 *
 * If K0>0 and K>K0 an existing K0-step LBD stored in U, V and B is extended to a K-step LBD.
 *
 * @param m          Number of rows of A
 * @param n          Number of columns of A
 * @param k0         Dimension of previously computed Lanczos bidiagonalization stored in U, V, B
 * @param k          On entry: desired dimension of LBD. On exit: actual size computed
 * @param aprod      Matrix-vector operation callback function
 * @param U          Left Lanczos vectors (M x K+1, column-major, complex)
 * @param ldu        Leading dimension of U (>= M)
 * @param V          Right Lanczos vectors (N x K, column-major, complex)
 * @param ldv        Leading dimension of V (>= N)
 * @param B          Bidiagonal matrix (K x 2, column-major: diag in col 1, sub-diag in col 2, real)
 * @param ldb        Leading dimension of B (>= K)
 * @param rnorm      On entry: norm of starting vector. On exit: (K+1,K) element of B_k
 * @param doption    Real options array: [delta, eta, anorm]
 * @param ioption    Integer options array: [cgs, elr]
 * @param dwork      Real workspace array of size >= m+n+2*k+2+max(m,n)
 * @param zwork      Complex workspace array of size >= max(m,n)
 * @param iwork      Integer workspace array of size >= 2*k+1
 * @param zparm      Complex parameters for aprod
 * @param iparm      Integer parameters for aprod
 * @param ierr       Error status (0=success, <0=invariant subspace found, >0=near-invariant)
 * @param rng_state  Random number generator state (uint64_t[4])
 */
void zlanbpro(
    PROPACK_INT m, PROPACK_INT n, PROPACK_INT k0, PROPACK_INT* k, PROPACK_aprod_z aprod, PROPACK_CPLX_TYPE* U, PROPACK_INT ldu,
    PROPACK_CPLX_TYPE* V, PROPACK_INT ldv, double* B, PROPACK_INT ldb, double* rnorm, double* doption,
    PROPACK_INT* ioption, double* dwork, PROPACK_CPLX_TYPE* zwork, PROPACK_INT* iwork, PROPACK_CPLX_TYPE* zparm,
    PROPACK_INT* iparm, PROPACK_INT* ierr, uint64_t* rng_state);

#endif
