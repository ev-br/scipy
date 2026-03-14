#include "ritzvec.h"


void sritzvec(const PROPACK_INT which, const PROPACK_INT jobu, const PROPACK_INT jobv, const PROPACK_INT m, const PROPACK_INT n, const PROPACK_INT k, PROPACK_INT dim,
              float* restrict D, float* restrict E, float* restrict U, const PROPACK_INT ldu,
              float* restrict V, const PROPACK_INT ldv, float* restrict work, const PROPACK_INT in_lwrk, PROPACK_INT* restrict iwork)
{
    PROPACK_INT lwrk, mstart, ip, iqt, imt, iwrk, id[1], info;
    float c1, c2, dd[1];

    // The bidiagonal SVD is computed in a two-stage procedure:
    //
    // 1. Compute a QR-factorization M^T*B = [R; 0] of the (k+1)-by-k lower bidiagonal matrix B.
    // 2. Compute the SVD of the k-by-k upper bidiagonal matrix, R = P*S*Q^T. The SVD of B is then (M*P)*S*Q^T.

    // Set pointers into workspace array
    lwrk = in_lwrk;
    imt = 0;
    iqt = imt + (dim+1)*(dim+1);
    ip = iqt + dim*dim;
    iwrk = ip + dim*dim;
    lwrk = lwrk - iwrk;

    // Compute QR-factorization
    //   B = M * [R; 0]
    sbdqr((dim == (m < n ? m : n)), jobu, dim, D, E, &c1, &c2, &work[imt], dim+1);
    // Compute SVD of R using the Divide-and-conquer SVD: R = P * S * Q^T,
    BLAS_FUNC(sbdsdc)("U", "I", &dim, D, E, &work[ip], &dim, &work[iqt], &dim, dd, id, &work[iwrk], iwork, &info);
    // Compute left singular vectors for B, X = P^T * M^T
    sgemm_ovwr(1, dim, dim+1, dim, 1.0f, &work[ip], dim, 0.0f, &work[imt], dim+1, &work[iwrk], lwrk / dim);

    mstart = (which == 0) ? dim - k : 0;  // smallest : largest
    if (jobu)
    {
        // Form left Ritz-vectors, U = U * X^T
        sgemm_ovwr_left(1, m, k, dim+1, 1.0f, U, ldu, &work[imt + mstart], dim+1, &work[iwrk], lwrk / k);
    }

    if (jobv)
    {
        // Form right Ritz-vectors, V = V * Q
        sgemm_ovwr_left(1, n, k, dim, 1.0f, V, ldv, &work[iqt + mstart], dim, &work[iwrk], lwrk / k);
    }
}


void dritzvec(const PROPACK_INT which, const PROPACK_INT jobu, const PROPACK_INT jobv, const PROPACK_INT m, const PROPACK_INT n, const PROPACK_INT k, PROPACK_INT dim,
              double* restrict D, double* restrict E, double* restrict U, const PROPACK_INT ldu,
              double* restrict V, const PROPACK_INT ldv, double* restrict work, const PROPACK_INT in_lwrk, PROPACK_INT* restrict iwork)
{
    PROPACK_INT lwrk, mstart, ip, iqt, imt, iwrk, id[1], info;
    double c1, c2, dd[1];

    // The bidiagonal SVD is computed in a two-stage procedure:
    //
    // 1. Compute a QR-factorization M^T*B = [R; 0] of the (k+1)-by-k lower bidiagonal matrix B.
    // 2. Compute the SVD of the k-by-k upper bidiagonal matrix, R = P*S*Q^T. The SVD of B is then (M*P)*S*Q^T.

    // Set pointers into workspace array
    lwrk = in_lwrk;
    imt = 0;
    iqt = imt + (dim+1)*(dim+1);
    ip = iqt + dim*dim;
    iwrk = ip + dim*dim;
    lwrk = lwrk - iwrk;

    // Compute QR-factorization
    //   B = M * [R; 0]
    dbdqr((dim == (m < n ? m : n)), jobu, dim, D, E, &c1, &c2, &work[imt], dim+1);
    // Compute SVD of R using the Divide-and-conquer SVD: R = P * S * Q^T,
    BLAS_FUNC(dbdsdc)("U", "I", &dim, D, E, &work[ip], &dim, &work[iqt], &dim, dd, id, &work[iwrk], iwork, &info);
    // Compute left singular vectors for B, X = P^T * M^T
    dgemm_ovwr(1, dim, dim+1, dim, 1.0, &work[ip], dim, 0.0, &work[imt], dim+1, &work[iwrk], lwrk / dim);

    mstart = (which == 0) ? dim - k : 0;  // smallest : largest
    if (jobu)
    {
        // Form left Ritz-vectors, U = U * X^T
        dgemm_ovwr_left(1, m, k, dim+1, 1.0, U, ldu, &work[imt + mstart], dim+1, &work[iwrk], lwrk / k);
    }

    if (jobv)
    {
        // Form right Ritz-vectors, V = V * Q
        dgemm_ovwr_left(1, n, k, dim, 1.0, V, ldv, &work[iqt + mstart], dim, &work[iwrk], lwrk / k);
    }
}


void critzvec(const PROPACK_INT which, const PROPACK_INT jobu, const PROPACK_INT jobv, const PROPACK_INT m, const PROPACK_INT n, const PROPACK_INT k, PROPACK_INT dim,
              float* restrict D, float* restrict E, PROPACK_CPLXF_TYPE* restrict U, const PROPACK_INT ldu,
              PROPACK_CPLXF_TYPE* restrict V, const PROPACK_INT ldv, float* restrict work, const PROPACK_INT in_lwrk,
              PROPACK_CPLXF_TYPE* restrict cwork, const PROPACK_INT lcwrk, PROPACK_INT* restrict iwork)
{
    PROPACK_INT lwrk, mstart, ip, iqt, imt, iwrk, id[1], info;
    float c1, c2, dd[1];

    // The bidiagonal SVD is computed in a two-stage procedure:
    //
    // 1. Compute a QR-factorization M^T*B = [R; 0] of the (k+1)-by-k lower bidiagonal matrix B.
    // 2. Compute the SVD of the k-by-k upper bidiagonal matrix, R = P*S*Q^T. The SVD of B is then (M*P)*S*Q^T.

    // Set pointers into workspace array
    lwrk = in_lwrk;
    imt = 0;
    iqt = imt + (dim+1)*(dim+1);
    ip = iqt + dim*dim;
    iwrk = ip + dim*dim;
    lwrk = lwrk - iwrk;

    // Compute QR-factorization
    //   B = M * [R; 0]
    sbdqr((dim == (m < n ? m : n)), jobu, dim, D, E, &c1, &c2, &work[imt], dim+1);

    // Compute SVD of R using the Divide-and-conquer SVD: R = P * S * Q^T,
    BLAS_FUNC(sbdsdc)("U", "I", &dim, D, E, &work[ip], &dim, &work[iqt], &dim, dd, id, &work[iwrk], iwork, &info);

    // Compute left singular vectors for B, X = P^T * M^T
    sgemm_ovwr(1, dim, dim+1, dim, 1.0f, &work[ip], dim, 0.0f, &work[imt], dim+1, &work[iwrk], lwrk / dim);

    if (jobu)
    {
        mstart = (which == 0) ? dim - k : 0;  // smallest : largest
        // Form left Ritz-vectors, U = U * X^T
        csgemm_ovwr_left(1, m, k, dim+1, U, ldu, &work[imt + mstart], dim+1, cwork, lcwrk / k);
    }

    if (jobv)
    {
        mstart = (which == 0) ? dim - k : 0;  // smallest : largest
        // Form right Ritz-vectors, V = V * Q
        csgemm_ovwr_left(1, n, k, dim, V, ldv, &work[iqt + mstart], dim, cwork, lcwrk / k);
    }
}


void zritzvec(const PROPACK_INT which, const PROPACK_INT jobu, const PROPACK_INT jobv, const PROPACK_INT m, const PROPACK_INT n, const PROPACK_INT k, PROPACK_INT dim,
              double* restrict D, double* restrict E, PROPACK_CPLX_TYPE* restrict U, const PROPACK_INT ldu,
              PROPACK_CPLX_TYPE* restrict V, const PROPACK_INT ldv, double* restrict work, const PROPACK_INT in_lwrk,
              PROPACK_CPLX_TYPE* restrict zwork, const PROPACK_INT lzwrk, PROPACK_INT* restrict iwork)
{
    PROPACK_INT lwrk, mstart, ip, iqt, imt, iwrk, id[1], info;
    double c1, c2, dd[1];

    // The bidiagonal SVD is computed in a two-stage procedure:
    //
    // 1. Compute a QR-factorization M^T*B = [R; 0] of the (k+1)-by-k lower bidiagonal matrix B.
    // 2. Compute the SVD of the k-by-k upper bidiagonal matrix, R = P*S*Q^T. The SVD of B is then (M*P)*S*Q^T.

    // Set pointers into workspace array
    lwrk = in_lwrk;
    imt = 0;
    iqt = imt + (dim+1)*(dim+1);
    ip = iqt + dim*dim;
    iwrk = ip + dim*dim;
    lwrk = lwrk - iwrk;

    // Compute QR-factorization
    //   B = M * [R; 0]
    dbdqr((dim == (m < n ? m : n)), jobu, dim, D, E, &c1, &c2, &work[imt], dim+1);
    // Compute SVD of R using the Divide-and-conquer SVD: R = P * S * Q^T,
    BLAS_FUNC(dbdsdc)("U", "I", &dim, D, E, &work[ip], &dim, &work[iqt], &dim, dd, id, &work[iwrk], iwork, &info);
    // Compute left singular vectors for B, X = P^T * M^T
    dgemm_ovwr(1, dim, dim+1, dim, 1.0, &work[ip], dim, 0.0, &work[imt], dim+1, &work[iwrk], lwrk / dim);

    mstart = (which == 0) ? dim - k : 0;  // smallest : largest
    if (jobu)
    {
        // Form left Ritz-vectors, U = U * X^T
        zdgemm_ovwr_left(1, m, k, dim+1, U, ldu, &work[imt + mstart], dim+1, zwork, lzwrk / k);
    }

    if (jobv)
    {
        // Form right Ritz-vectors, V = V * Q
        zdgemm_ovwr_left(1, n, k, dim, V, ldv, &work[iqt + mstart], dim, zwork, lzwrk / k);
    }
}
