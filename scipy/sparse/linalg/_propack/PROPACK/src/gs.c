#include "gs.h"


void smgs(PROPACK_INT n, PROPACK_INT k, float* V, PROPACK_INT ldv, float* vnew, const PROPACK_INT* indices) {
    PROPACK_INT ione = 1;

    // Check for quick return
    if ((k < 0) || (n <= 0)) { return; }

    // Process each block specified in indices array
    PROPACK_INT idx = 0;
    PROPACK_INT start = indices[idx];
    PROPACK_INT end = indices[idx + 1];

    /**
     * PROPACK encodes sentinels in 1-index specific way.
     * Therefore, we have to guard for a few edge cases of 0-indexing.
     * TODO: Fix this by rewriting the indexing logic.
     */
    while ((start <= k) && ((start < end) || ((start == 0) && (end == 0) && (idx == 0)))) {
        // Orthogonalize against columns [start, end] (0-indexed)
        for (PROPACK_INT i = start; i <= end; i++) {
            // Compute projection coefficient: coef = V(:,i)' * vnew
            float coef = BLAS_FUNC(sdot)(&n, &V[i * ldv], &ione, vnew, &ione);

            // Orthogonalize: vnew = vnew - coef * V(:,i)
            float neg_coef = -coef;
            BLAS_FUNC(saxpy)(&n, &neg_coef, &V[i * ldv], &ione, vnew, &ione);
        }

        idx += 2;  // Move to next block
        start = indices[idx];
        end = indices[idx + 1];
    }
}


void dmgs(PROPACK_INT n, PROPACK_INT k, double* V, PROPACK_INT ldv, double* vnew, const PROPACK_INT* indices) {
    PROPACK_INT ione = 1;

    // Check for quick return
    if ((k < 0) || (n <= 0)) { return; }

    // Process each block specified in indices array
    PROPACK_INT idx = 0;
    PROPACK_INT start = indices[idx];
    PROPACK_INT end = indices[idx + 1];

    /**
     * PROPACK encodes sentinels in 1-index specific way.
     * Therefore, we have to guard for a few edge cases of 0-indexing.
     * TODO: Fix this by rewriting the indexing logic.
     */
    while ((start <= k) && ((start < end) || ((start == 0) && (end == 0) && (idx == 0)))) {
        // Orthogonalize against columns [start, end] (0-indexed)
        for (PROPACK_INT i = start; i <= end; i++) {
            // Compute projection coefficient: coef = V(:,i)' * vnew
            double coef = BLAS_FUNC(ddot)(&n, &V[i * ldv], &ione, vnew, &ione);

            // Orthogonalize: vnew = vnew - coef * V(:,i)
            double neg_coef = -coef;
            BLAS_FUNC(daxpy)(&n, &neg_coef, &V[i * ldv], &ione, vnew, &ione);
        }

        idx += 2;  // Move to next block
        start = indices[idx];
        end = indices[idx + 1];
    }
}


void cmgs(PROPACK_INT n, PROPACK_INT k, PROPACK_CPLXF_TYPE* V, PROPACK_INT ldv, PROPACK_CPLXF_TYPE* vnew, const PROPACK_INT* indices) {
    PROPACK_INT ione = 1;

    // Check for quick return
    if ((k < 0) || (n <= 0)) { return; }

    // Process each block specified in indices array
    PROPACK_INT idx = 0;
    PROPACK_INT start = indices[idx];
    PROPACK_INT end = indices[idx + 1];

    while ((start <= k) && ((start < end) || ((start == 0) && (end == 0) && (idx == 0)))) {
        // Orthogonalize against columns [start, end] (0-indexed)
        for (PROPACK_INT i = start; i <= end; i++) {
            // Compute projection coefficient: coef = V(:,i)^H * vnew (conjugate dot product)
            PROPACK_CPLXF_TYPE coef = cdotc_(&n, &V[i * ldv], &ione, vnew, &ione);

            // Orthogonalize: vnew = vnew - coef * V(:,i)
            PROPACK_CPLXF_TYPE neg_coef = PROPACK_cplxf(-crealf(coef), -cimagf(coef));
            BLAS_FUNC(caxpy)(&n, &neg_coef, &V[i * ldv], &ione, vnew, &ione);
        }

        idx += 2;  // Move to next block
        start = indices[idx];
        end = indices[idx + 1];
    }
}


void zmgs(PROPACK_INT n, PROPACK_INT k, PROPACK_CPLX_TYPE* V, PROPACK_INT ldv, PROPACK_CPLX_TYPE* vnew, const PROPACK_INT* indices) {
    PROPACK_INT ione = 1;

    // Check for quick return
    if ((k < 0) || (n <= 0)) { return; }

    // Process each block specified in indices array
    PROPACK_INT idx = 0;
    PROPACK_INT start = indices[idx];
    PROPACK_INT end = indices[idx + 1];

    while ((start <= k) && ((start < end) || ((start == 0) && (end == 0) && (idx == 0)))) {
        // Orthogonalize against columns [start, end] (0-indexed)
        for (PROPACK_INT i = start; i <= end; i++) {
            // Compute projection coefficient: coef = V(:,i)^H * vnew (conjugate dot product)
            PROPACK_CPLX_TYPE coef = zdotc_(&n, &V[i * ldv], &ione, vnew, &ione);

            // Orthogonalize: vnew = vnew - coef * V(:,i)
            PROPACK_CPLX_TYPE neg_coef = PROPACK_cplx(-creal(coef), -cimag(coef));
            BLAS_FUNC(zaxpy)(&n, &neg_coef, &V[i * ldv], &ione, vnew, &ione);
        }

        idx += 2;  // Move to next block
        start = indices[idx];
        end = indices[idx + 1];
    }
}


void scgs(PROPACK_INT n, PROPACK_INT k, float* V, PROPACK_INT ldv, float* vnew, const PROPACK_INT* indices, float* work) {
    PROPACK_INT ione = 1;
    float one = 1.0f;
    float zero = 0.0f;
    float neg_one = -1.0f;

    // Check for quick return
    if ((k < 0) || (n <= 0)) { return; }

    // Process each block specified in indices array
    PROPACK_INT idx = 0;
    PROPACK_INT start = indices[idx];
    PROPACK_INT end = indices[idx + 1];

    while (start < k && start >= 0 && start <= end) {
        // Block size
        PROPACK_INT block_size = end - start + 1;

        // Compute all projection coefficients for this block: work = V_block^T * vnew
        BLAS_FUNC(sgemv)("T", &n, &block_size, &one, &V[start * ldv], &ldv, vnew, &ione, &zero, work, &ione);

        // Orthogonalize: vnew = vnew - V_block * work
        BLAS_FUNC(sgemv)("N", &n, &block_size, &neg_one, &V[start * ldv], &ldv, work, &ione, &one, vnew, &ione);

        idx += 2;  // Move to next block
        start = indices[idx];
        end = indices[idx + 1];
    }
}


void dcgs(PROPACK_INT n, PROPACK_INT k, double* V, PROPACK_INT ldv, double* vnew, const PROPACK_INT* indices, double* work) {
    PROPACK_INT ione = 1;
    double one = 1.0;
    double zero = 0.0;
    double neg_one = -1.0;

    // Check for quick return
    if ((k < 0) || (n <= 0)) { return; }

    // Process each block specified in indices array
    PROPACK_INT idx = 0;
    PROPACK_INT start = indices[idx];
    PROPACK_INT end = indices[idx + 1];

    while (start < k && start >= 0 && start <= end) {
        // Block size
        PROPACK_INT block_size = end - start + 1;

        // Compute all projection coefficients for this block: work = V_block^T * vnew
        BLAS_FUNC(dgemv)("T", &n, &block_size, &one, &V[start * ldv], &ldv, vnew, &ione, &zero, work, &ione);

        // Orthogonalize: vnew = vnew - V_block * work
        BLAS_FUNC(dgemv)("N", &n, &block_size, &neg_one, &V[start * ldv], &ldv, work, &ione, &one, vnew, &ione);

        idx += 2;  // Move to next block
        start = indices[idx];
        end = indices[idx + 1];
    }
}


void ccgs(PROPACK_INT n, PROPACK_INT k, PROPACK_CPLXF_TYPE* V, PROPACK_INT ldv, PROPACK_CPLXF_TYPE* vnew, const PROPACK_INT* indices, PROPACK_CPLXF_TYPE* work) {
    PROPACK_INT ione = 1;
    PROPACK_CPLXF_TYPE one = PROPACK_cplxf(1.0f, 0.0f);
    PROPACK_CPLXF_TYPE zero = PROPACK_cplxf(0.0f, 0.0f);
    PROPACK_CPLXF_TYPE neg_one = PROPACK_cplxf(-1.0f, 0.0f);

    // Check for quick return
    if ((k < 0) || (n <= 0)) { return; }

    // Process each block specified in indices array
    PROPACK_INT idx = 0;
    PROPACK_INT start = indices[idx];
    PROPACK_INT end = indices[idx + 1];

    while (start < k && start >= 0 && start <= end) {
        // Block size
        PROPACK_INT block_size = end - start + 1;

        // Compute all projection coefficients for this block: work = V_block^H * vnew
        BLAS_FUNC(cgemv)("C", &n, &block_size, &one, &V[start * ldv], &ldv, vnew, &ione, &zero, work, &ione);

        // Orthogonalize: vnew = vnew - V_block * work
        BLAS_FUNC(cgemv)("N", &n, &block_size, &neg_one, &V[start * ldv], &ldv, work, &ione, &one, vnew, &ione);

        idx += 2;  // Move to next block
        start = indices[idx];
        end = indices[idx + 1];
    }
}


void zcgs(PROPACK_INT n, PROPACK_INT k, PROPACK_CPLX_TYPE* V, PROPACK_INT ldv, PROPACK_CPLX_TYPE* vnew, const PROPACK_INT* indices, PROPACK_CPLX_TYPE* work) {
    PROPACK_INT ione = 1;
    PROPACK_CPLX_TYPE one = PROPACK_cplx(1.0, 0.0);
    PROPACK_CPLX_TYPE zero = PROPACK_cplx(0.0, 0.0);
    PROPACK_CPLX_TYPE neg_one = PROPACK_cplx(-1.0, 0.0);

    // Check for quick return
    if ((k < 0) || (n <= 0)) { return; }

    // Process each block specified in indices array
    PROPACK_INT idx = 0;
    PROPACK_INT start = indices[idx];
    PROPACK_INT end = indices[idx + 1];

    while (start < k && start >= 0 && start <= end) {
        // Block size
        PROPACK_INT block_size = end - start + 1;

        // Compute all projection coefficients for this block: work = V_block^H * vnew
        BLAS_FUNC(zgemv)("C", &n, &block_size, &one, &V[start * ldv], &ldv, vnew, &ione, &zero, work, &ione);

        // Orthogonalize: vnew = vnew - V_block * work
        BLAS_FUNC(zgemv)("N", &n, &block_size, &neg_one, &V[start * ldv], &ldv, work, &ione, &one, vnew, &ione);

        idx += 2;  // Move to next block
        start = indices[idx];
        end = indices[idx + 1];
    }
}


void sreorth(PROPACK_INT n, PROPACK_INT k, float* V, PROPACK_INT ldv, float* vnew, float* normvnew, const PROPACK_INT* indices, float alpha, float* work, PROPACK_INT iflag) {
    const PROPACK_INT NTRY = 5;
    PROPACK_INT ione = 1;

    // Check for quick return
    if ((k < 0) || (n <= 0)) { return; }

    for (PROPACK_INT itry = 0; itry < NTRY; itry++)
    {
        float normvnew_0 = *normvnew;

        if (iflag == 1) {
            scgs(n, k, V, ldv, vnew, indices, work);
        } else {
            smgs(n, k, V, ldv, vnew, indices);
        }

        *normvnew = BLAS_FUNC(snrm2)(&n, vnew, &ione);
        if (*normvnew > alpha * normvnew_0) { return; }
    }

    // vnew is numerically in span(V) => return vnew = (0,0,...,0)^T
    *normvnew = 0.0f;
    for (PROPACK_INT i = 0; i < n; i++) { vnew[i] = 0.0f; }
}


void dreorth(PROPACK_INT n, PROPACK_INT k, double* V, PROPACK_INT ldv, double* vnew, double* normvnew, const PROPACK_INT* indices, double alpha, double* work, PROPACK_INT iflag) {
    const PROPACK_INT NTRY = 5;
    PROPACK_INT ione = 1;

    // Check for quick return
    if ((k < 0) || (n <= 0)) { return; }

    for (PROPACK_INT itry = 0; itry < NTRY; itry++)
    {
        double normvnew_0 = *normvnew;

        if (iflag == 1) {
            dcgs(n, k, V, ldv, vnew, indices, work);
        } else {
            dmgs(n, k, V, ldv, vnew, indices);
        }

        *normvnew = BLAS_FUNC(dnrm2)(&n, vnew, &ione);

        if (*normvnew > alpha * normvnew_0) { return; }
    }

    // vnew is numerically in span(V) => return vnew = (0,0,...,0)^T
    *normvnew = 0.0;
    for (PROPACK_INT i = 0; i < n; i++) { vnew[i] = 0.0; }
}


void creorth(PROPACK_INT n, PROPACK_INT k, PROPACK_CPLXF_TYPE* V, PROPACK_INT ldv, PROPACK_CPLXF_TYPE* vnew, float* normvnew, const PROPACK_INT* indices, float alpha, PROPACK_CPLXF_TYPE* work, PROPACK_INT iflag) {
    const PROPACK_INT NTRY = 5;
    PROPACK_INT ione = 1;

    // Check for quick return
    if ((k < 0) || (n <= 0)) { return; }

    for (PROPACK_INT itry = 0; itry < NTRY; itry++)
    {
        float normvnew_0 = *normvnew;

        if (iflag == 1) {
            ccgs(n, k, V, ldv, vnew, indices, work);
        } else {
            cmgs(n, k, V, ldv, vnew, indices);
        }

        *normvnew = BLAS_FUNC(scnrm2)(&n, vnew, &ione);

        if (*normvnew > alpha * normvnew_0) { return; }
    }

    // vnew is numerically in span(V) => return vnew = (0,0,...,0)^T
    *normvnew = 0.0f;
    for (PROPACK_INT i = 0; i < n; i++) { vnew[i] = PROPACK_cplxf(0.0f, 0.0f); }
}


void zreorth(PROPACK_INT n, PROPACK_INT k, PROPACK_CPLX_TYPE* V, PROPACK_INT ldv, PROPACK_CPLX_TYPE* vnew, double* normvnew, const PROPACK_INT* indices, double alpha, PROPACK_CPLX_TYPE* work, PROPACK_INT iflag) {
    const PROPACK_INT NTRY = 5;
    PROPACK_INT ione = 1;

    // Check for quick return
    if ((k < 0) || (n <= 0)) { return; }

    for (PROPACK_INT itry = 0; itry < NTRY; itry++)
    {
        double normvnew_0 = *normvnew;

        if (iflag == 1) {
            zcgs(n, k, V, ldv, vnew, indices, work);
        } else {
            zmgs(n, k, V, ldv, vnew, indices);
        }

        *normvnew = BLAS_FUNC(dznrm2)(&n, vnew, &ione);

        if (*normvnew > alpha * normvnew_0) { return; }
    }

    // vnew is numerically in span(V) => return vnew = (0,0,...,0)^T
    *normvnew = 0.0;
    for (PROPACK_INT i = 0; i < n; i++) { vnew[i] = PROPACK_cplx(0.0, 0.0); }
}
