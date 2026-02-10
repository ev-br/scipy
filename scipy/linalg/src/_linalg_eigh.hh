#pragma once
/*
 * Templated loops for linalg.eigh
 */


template<typename T>
int
_reg_eigh(PyArrayObject* ap_Am, PyArrayObject *ap_w, PyArrayObject *ap_v,
          char uplo, char jobz, SliceStatusVec& vec_status)
{
    using real_type = typename type_traits<T>::real_type;
    SliceStatus slice_status;

    // --------------------------------------------------------------------
    // Input Array Attributes
    // --------------------------------------------------------------------
    T* Am_data = (T *)PyArray_DATA(ap_Am);
    int ndim = PyArray_NDIM(ap_Am);
    npy_intp* shape = PyArray_SHAPE(ap_Am);
    npy_intp n = shape[ndim - 1];
    npy_intp* strides = PyArray_STRIDES(ap_Am);

    // Get the number of slices to traverse
    npy_intp outer_size = 1;
    if (ndim > 2) {
        for (int i = 0; i < ndim - 2; i++) { outer_size *= shape[i];}
    }

    // Output array pointers
    real_type *ptr_W = (real_type *)PyArray_DATA(ap_w);
    
    int compute_v = (jobz == 'V');
    T *ptr_v = compute_v ? (T *)PyArray_DATA(ap_v) : NULL;

    // --------------------------------------------------------------------
    // Workspace computation and allocation
    // --------------------------------------------------------------------
    CBLAS_INT intn = (CBLAS_INT)n, lwork = -1, liwork = -1, info;
    T tmp_work = numeric_limits<T>::zero;
    CBLAS_INT tmp_iwork = 0;

    CBLAS_INT lda = n;
    CBLAS_INT ldz = n;

    // For complex types, we need rwork
    real_type *rwork = NULL;
    real_type tmp_rwork = 0;
    CBLAS_INT lrwork = -1;
    
    real_type abstol = 0.0;  // Use default tolerance
    
    if constexpr (type_traits<T>::is_complex) {
        // query LWORK, LRWORK, LIWORK for complex types
        call_heevr(&jobz, "A", &uplo, &intn, NULL, &lda, NULL, NULL, NULL, NULL, &abstol, NULL, NULL, NULL, &ldz, NULL,
                   &tmp_work, &lwork, &tmp_rwork, &lrwork, &tmp_iwork, &liwork, &info);
    } else {
        // query LWORK, LIWORK for real types
        call_syevr(&jobz, "A", &uplo, &intn, NULL, &lda, NULL, NULL, NULL, NULL, &abstol, NULL, NULL, NULL, &ldz, NULL,
                   &tmp_work, &lwork, &tmp_iwork, &liwork, &info);
    }
    
    if (info != 0) { return -101; }

    lwork = _calc_lwork(tmp_work);
    if (lwork < 0) { return -102; }
    
    liwork = (CBLAS_INT)tmp_iwork;
    if (liwork < 0) { return -103; }

    // allocate workspace
    npy_intp bufsize = n*n + lwork + n*n;  // data + work + z (eigenvectors)
    T *buf = (T *)malloc(bufsize*sizeof(T));
    if (buf == NULL) { return -104; }

    CBLAS_INT *iwork = (CBLAS_INT *)malloc(liwork*sizeof(CBLAS_INT));
    if (iwork == NULL) { free(buf); return -105; }
    
    CBLAS_INT *isuppz = (CBLAS_INT *)malloc(2*n*sizeof(CBLAS_INT));
    if (isuppz == NULL) { free(buf); free(iwork); return -108; }

    if constexpr (type_traits<T>::is_complex) {
        lrwork = _calc_lwork(tmp_rwork);
        if (lrwork < 0) { free(buf); free(iwork); free(isuppz); return -106; }
        rwork = (real_type *)malloc(lrwork*sizeof(real_type));
        if (rwork == NULL) { free(buf); free(iwork); free(isuppz); return -107; }
    }

    // partition the workspace
    T *data = &buf[0];
    T *work = &buf[n*n];
    T *z_buf = &buf[n*n + lwork];  // eigenvectors buffer

    // --------------------------------------------------------------------
    // Main loop to traverse the slices
    // --------------------------------------------------------------------
    for (npy_intp idx = 0; idx < outer_size; idx++) {
        init_status(slice_status, idx, St::HER);

        // copy the slice to `data` in F order
        T *slice_ptr = compute_slice_ptr(idx, Am_data, ndim, shape, strides);
        copy_slice_F(data, slice_ptr, n, n, strides[ndim-2], strides[ndim-1]);

        // Prepare output
        real_type *w_buf = ptr_W + idx * n;
        CBLAS_INT m = 0;  // number of eigenvalues found

        // compute eigenvalues (and eigenvectors if requested) for the slice
        if constexpr (type_traits<T>::is_complex) {
            call_heevr(&jobz, "A", &uplo, &intn, data, &lda, NULL, NULL, NULL, NULL, &abstol, &m,
                       w_buf, z_buf, &ldz, isuppz, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
        } else {
            call_syevr(&jobz, "A", &uplo, &intn, data, &lda, NULL, NULL, NULL, NULL, &abstol, &m,
                       w_buf, z_buf, &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
        }

        if(info != 0) {
            slice_status.lapack_info = (Py_ssize_t)info;
            vec_status.push_back(slice_status);
            // cut it short on error in any slice
            goto done;
        }

        // copy eigenvectors if computed
        if (compute_v) {
            copy_slice_F_to_C(ptr_v + idx*n*n, z_buf, n, n, ldz);
        }
    }

 done:
    free(buf);
    free(iwork);
    free(isuppz);
    if (rwork) free(rwork);

    return 0;
}


template<typename T>
int
_gen_eigh(PyArrayObject* ap_Am, PyArrayObject *ap_Bm, PyArrayObject *ap_w, PyArrayObject *ap_v,
          char uplo, char jobz, CBLAS_INT itype, SliceStatusVec& vec_status)
{
    using real_type = typename type_traits<T>::real_type;
    SliceStatus slice_status;

    // --------------------------------------------------------------------
    // Input Array Attributes
    // --------------------------------------------------------------------
    T* Am_data = (T *)PyArray_DATA(ap_Am);
    int ndim = PyArray_NDIM(ap_Am);
    npy_intp* shape = PyArray_SHAPE(ap_Am);
    npy_intp n = shape[ndim - 1];
    npy_intp *strides_A = PyArray_STRIDES(ap_Am);

    T *Bm_data = (T *)PyArray_DATA(ap_Bm);
    npy_intp *strides_B = PyArray_STRIDES(ap_Bm);

    // Get the number of slices to traverse
    npy_intp outer_size = 1;
    if (ndim > 2) {
        for (int i = 0; i < ndim - 2; i++) { outer_size *= shape[i];}
    }

    // Output array pointers
    real_type *ptr_W = (real_type *)PyArray_DATA(ap_w);
    
    int compute_v = (jobz == 'V');
    T *ptr_v = compute_v ? (T *)PyArray_DATA(ap_v) : NULL;

    // --------------------------------------------------------------------
    // Workspace computation and allocation
    // --------------------------------------------------------------------
    CBLAS_INT intn = (CBLAS_INT)n, lwork = -1, liwork = -1, info;
    T tmp_work = numeric_limits<T>::zero;
    CBLAS_INT tmp_iwork = 0;

    CBLAS_INT lda = n;
    CBLAS_INT ldb = n;

    // For complex types, we need rwork
    real_type *rwork = NULL;
    real_type tmp_rwork = 0;
    CBLAS_INT lrwork = -1;
    
    if constexpr (type_traits<T>::is_complex) {
        // query LWORK, LRWORK, LIWORK for complex types
        call_hegvd(&itype, &jobz, &uplo, &intn, NULL, &lda, NULL, &ldb, NULL,
                   &tmp_work, &lwork, &tmp_rwork, &lrwork, &tmp_iwork, &liwork, &info);
    } else {
        // query LWORK, LIWORK for real types
        call_sygvd(&itype, &jobz, &uplo, &intn, NULL, &lda, NULL, &ldb, NULL,
                   &tmp_work, &lwork, &tmp_iwork, &liwork, &info);
    }
    
    if (info != 0) { return -101; }

    lwork = _calc_lwork(tmp_work);
    if (lwork < 0) { return -102; }
    
    liwork = (CBLAS_INT)tmp_iwork;
    if (liwork < 0) { return -103; }

    // allocate workspace
    npy_intp bufsize = 2*n*n + lwork;  // space for A and B matrices + work
    T *buf = (T *)malloc(bufsize*sizeof(T));
    if (buf == NULL) { return -104; }

    CBLAS_INT *iwork = (CBLAS_INT *)malloc(liwork*sizeof(CBLAS_INT));
    if (iwork == NULL) { free(buf); return -105; }

    if constexpr (type_traits<T>::is_complex) {
        lrwork = _calc_lwork(tmp_rwork);
        if (lrwork < 0) { free(buf); free(iwork); return -106; }
        rwork = (real_type *)malloc(lrwork*sizeof(real_type));
        if (rwork == NULL) { free(buf); free(iwork); return -107; }
    }

    // partition the workspace
    T *data_A = &buf[0];
    T *data_B = &buf[n*n];
    T *work = &buf[2*n*n];

    // --------------------------------------------------------------------
    // Main loop to traverse the slices
    // --------------------------------------------------------------------
    for (npy_intp idx = 0; idx < outer_size; idx++) {
        init_status(slice_status, idx, St::HER);

        // copy the slices to `data` in F order
        T *slice_ptr_A = compute_slice_ptr(idx, Am_data, ndim, shape, strides_A);
        copy_slice_F(data_A, slice_ptr_A, n, n, strides_A[ndim-2], strides_A[ndim-1]);

        T *slice_ptr_B = compute_slice_ptr(idx, Bm_data, ndim, shape, strides_B);
        copy_slice_F(data_B, slice_ptr_B, n, n, strides_B[ndim-2], strides_B[ndim-1]);

        // Prepare output
        real_type *w_buf = ptr_W + idx * n;

        // compute eigenvalues (and eigenvectors if requested) for the slice
        if constexpr (type_traits<T>::is_complex) {
            call_hegvd(&itype, &jobz, &uplo, &intn, data_A, &lda, data_B, &ldb, w_buf,
                       work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
        } else {
            call_sygvd(&itype, &jobz, &uplo, &intn, data_A, &lda, data_B, &ldb, w_buf,
                       work, &lwork, iwork, &liwork, &info);
        }

        if(info != 0) {
            slice_status.lapack_info = (Py_ssize_t)info;
            vec_status.push_back(slice_status);
            // cut it short on error in any slice
            goto done;
        }

        // copy eigenvectors if computed (stored in data_A)
        if (compute_v) {
            copy_slice_F_to_C(ptr_v + idx*n*n, data_A, n, n, lda);
        }
    }

 done:
    free(buf);
    free(iwork);
    if (rwork) free(rwork);

    return 0;
}


template<typename T>
int
_eigh(PyArrayObject* ap_Am, PyArrayObject *ap_Bm, PyArrayObject *ap_w, PyArrayObject *ap_v,
      char uplo, char jobz, CBLAS_INT itype, SliceStatusVec& vec_status)
{
    int info;
    if (ap_Bm == NULL) {
        // Standard eigenvalue problem
        info = _reg_eigh<T>(ap_Am, ap_w, ap_v, uplo, jobz, vec_status);
    }
    else {
        // Generalized eigenvalue problem
        info = _gen_eigh<T>(ap_Am, ap_Bm, ap_w, ap_v, uplo, jobz, itype, vec_status);
    }
    return info;
}
