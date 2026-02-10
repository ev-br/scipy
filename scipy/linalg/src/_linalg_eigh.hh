#pragma once
/*
 * Templated loops for linalg.eigh
 */


template<typename T>
int
_reg_eigh(PyArrayObject* ap_Am, PyArrayObject *ap_w, PyArrayObject *ap_v,
          char uplo, char jobz, 
          const char* driver,  // "ev", "evr", or "evx"
          char range,  // 'A' = all, 'V' = by value, 'I' = by index
          CBLAS_INT il, CBLAS_INT iu,  // index range (1-based)
          typename type_traits<T>::real_type vl, typename type_traits<T>::real_type vu,  // value range
          SliceStatusVec& vec_status)
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

    // Determine which driver we're using
    bool use_ev = (strcmp(driver, "ev") == 0);
    bool use_evr = (strcmp(driver, "evr") == 0);
    bool use_evx = (strcmp(driver, "evx") == 0);

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
    
    if (use_ev) {
        // Query workspace for syev/heev
        if constexpr (type_traits<T>::is_complex) {
            // query LWORK for complex types
            // Note: for heev, rwork size is fixed: max(1, 3*n-2), not returned by query
            call_heev(&jobz, &uplo, &intn, NULL, &lda, NULL, &tmp_work, &lwork, NULL, &info);
            lrwork = (3 * n > 2) ? (3 * n - 2) : 1;  // Fixed size for heev
        } else {
            // query LWORK for real types
            call_syev(&jobz, &uplo, &intn, NULL, &lda, NULL, &tmp_work, &lwork, &info);
        }
    } else if (use_evr) {
        // Query workspace for syevr/heevr
        if constexpr (type_traits<T>::is_complex) {
            // query LWORK, LRWORK, LIWORK for complex types
            call_heevr(&jobz, &range, &uplo, &intn, NULL, &lda, &vl, &vu, &il, &iu, &abstol, NULL, NULL, NULL, &ldz, NULL,
                       &tmp_work, &lwork, &tmp_rwork, &lrwork, &tmp_iwork, &liwork, &info);
            lrwork = _calc_lwork(tmp_rwork);
        } else {
            // query LWORK, LIWORK for real types
            call_syevr(&jobz, &range, &uplo, &intn, NULL, &lda, &vl, &vu, &il, &iu, &abstol, NULL, NULL, NULL, &ldz, NULL,
                       &tmp_work, &lwork, &tmp_iwork, &liwork, &info);
        }
    } else if (use_evx) {
        // Query workspace for syevx/heevx
        if constexpr (type_traits<T>::is_complex) {
            // query LWORK for complex types (no liwork for evx)
            call_heevx(&jobz, &range, &uplo, &intn, NULL, &lda, &vl, &vu, &il, &iu, &abstol, NULL, NULL, NULL, &ldz,
                       &tmp_work, &lwork, &tmp_rwork, NULL, NULL, &info);
            lrwork = _calc_lwork(tmp_rwork);
        } else {
            // query LWORK for real types
            call_syevx(&jobz, &range, &uplo, &intn, NULL, &lda, &vl, &vu, &il, &iu, &abstol, NULL, NULL, NULL, &ldz,
                       &tmp_work, &lwork, NULL, NULL, &info);
        }
        liwork = 5 * n;  // evx uses fixed iwork size
    } else {
        return -199;  // Unknown driver
    }
    
    if (info != 0) { return -101; }

    lwork = _calc_lwork(tmp_work);
    if (lwork < 0) { return -102; }
    
    if (use_ev) {
        liwork = 0;  // ev doesn't use iwork
    } else if (use_evx) {
        liwork = 5 * n;  // evx specific
    } else {
        liwork = (CBLAS_INT)tmp_iwork;
    }
    if (liwork < 0) { return -103; }

    // allocate workspace
    npy_intp bufsize = n*n + lwork + n*n;  // data + work + z (eigenvectors)
    T *buf = (T *)malloc(bufsize*sizeof(T));
    if (buf == NULL) { return -104; }

    CBLAS_INT *iwork = NULL;
    if (liwork > 0) {
        iwork = (CBLAS_INT *)malloc(liwork*sizeof(CBLAS_INT));
        if (iwork == NULL) { free(buf); return -105; }
    }
    
    // For evr, we need isuppz; for evx, we need ifail
    CBLAS_INT *isuppz = NULL;
    CBLAS_INT *ifail = NULL;
    if (use_evr) {
        isuppz = (CBLAS_INT *)malloc(2*n*sizeof(CBLAS_INT));
        if (isuppz == NULL) { free(buf); if(iwork) free(iwork); return -108; }
    } else if (use_evx) {
        ifail = (CBLAS_INT *)malloc(n*sizeof(CBLAS_INT));
        if (ifail == NULL) { free(buf); if(iwork) free(iwork); return -108; }
    }

    if constexpr (type_traits<T>::is_complex) {
        // lrwork should be set by workspace query for ev (line 68), evr (line 79), evx (line 93)
        if (lrwork < 0) { free(buf); if(iwork) free(iwork); if(isuppz) free(isuppz); if(ifail) free(ifail); return -106; }
        rwork = (real_type *)malloc(lrwork*sizeof(real_type));
        if (rwork == NULL) { free(buf); if(iwork) free(iwork); if(isuppz) free(isuppz); if(ifail) free(ifail); return -107; }
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
        if (use_ev) {
            // For ev driver, eigenvalues/eigenvectors are computed in-place
            if constexpr (type_traits<T>::is_complex) {
                call_heev(&jobz, &uplo, &intn, data, &lda, w_buf, work, &lwork, rwork, &info);
            } else {
                call_syev(&jobz, &uplo, &intn, data, &lda, w_buf, work, &lwork, &info);
            }
            m = n;  // ev computes all eigenvalues
            // For ev, eigenvectors are in data, need to copy to z_buf
            if (compute_v) {
                for (npy_intp i = 0; i < n*n; i++) {
                    z_buf[i] = data[i];
                }
            }
        } else if (use_evr) {
            if constexpr (type_traits<T>::is_complex) {
                call_heevr(&jobz, &range, &uplo, &intn, data, &lda, &vl, &vu, &il, &iu, &abstol, &m,
                           w_buf, z_buf, &ldz, isuppz, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
            } else {
                call_syevr(&jobz, &range, &uplo, &intn, data, &lda, &vl, &vu, &il, &iu, &abstol, &m,
                           w_buf, z_buf, &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
            }
        } else if (use_evx) {
            if constexpr (type_traits<T>::is_complex) {
                call_heevx(&jobz, &range, &uplo, &intn, data, &lda, &vl, &vu, &il, &iu, &abstol, &m,
                           w_buf, z_buf, &ldz, work, &lwork, rwork, iwork, ifail, &info);
            } else {
                call_syevx(&jobz, &range, &uplo, &intn, data, &lda, &vl, &vu, &il, &iu, &abstol, &m,
                           w_buf, z_buf, &ldz, work, &lwork, iwork, ifail, &info);
            }
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
    if (iwork) free(iwork);
    if (isuppz) free(isuppz);
    if (ifail) free(ifail);
    if (rwork) free(rwork);

    return 0;
}


template<typename T>
int
_gen_eigh(PyArrayObject* ap_Am, PyArrayObject *ap_Bm, PyArrayObject *ap_w, PyArrayObject *ap_v,
          char uplo, char jobz, CBLAS_INT itype,
          const char* driver,  // "gv", "gvd", or "gvx"
          char range,  // 'A' = all, 'V' = by value, 'I' = by index
          CBLAS_INT il, CBLAS_INT iu,  // index range (1-based)
          typename type_traits<T>::real_type vl, typename type_traits<T>::real_type vu,  // value range
          SliceStatusVec& vec_status)
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

    // Determine which driver we're using
    bool use_gv = (strcmp(driver, "gv") == 0);
    bool use_gvd = (strcmp(driver, "gvd") == 0);
    bool use_gvx = (strcmp(driver, "gvx") == 0);

    // --------------------------------------------------------------------
    // Workspace computation and allocation
    // --------------------------------------------------------------------
    CBLAS_INT intn = (CBLAS_INT)n, lwork = -1, liwork = -1, info;
    T tmp_work = numeric_limits<T>::zero;
    CBLAS_INT tmp_iwork = 0;

    CBLAS_INT lda = n;
    CBLAS_INT ldb = n;
    CBLAS_INT ldz = n;

    // For complex types, we need rwork
    real_type *rwork = NULL;
    real_type tmp_rwork = 0;
    CBLAS_INT lrwork = -1;
    
    real_type abstol = 0.0;  // Use default tolerance
    
    if (use_gv) {
        // Query workspace for sygv/hegv
        if constexpr (type_traits<T>::is_complex) {
            // query LWORK for complex types
            // Note: for hegv, rwork size is fixed: max(1, 3*n-2), not returned by query
            call_hegv(&itype, &jobz, &uplo, &intn, NULL, &lda, NULL, &ldb, NULL,
                      &tmp_work, &lwork, NULL, &info);
            lrwork = (3 * n > 2) ? (3 * n - 2) : 1;  // Fixed size for hegv
        } else {
            // query LWORK for real types
            call_sygv(&itype, &jobz, &uplo, &intn, NULL, &lda, NULL, &ldb, NULL,
                      &tmp_work, &lwork, &info);
        }
    } else if (use_gvd) {
        // Query workspace for sygvd/hegvd
        if constexpr (type_traits<T>::is_complex) {
            // query LWORK, LRWORK, LIWORK for complex types
            call_hegvd(&itype, &jobz, &uplo, &intn, NULL, &lda, NULL, &ldb, NULL,
                       &tmp_work, &lwork, &tmp_rwork, &lrwork, &tmp_iwork, &liwork, &info);
            lrwork = _calc_lwork(tmp_rwork);
        } else {
            // query LWORK, LIWORK for real types
            call_sygvd(&itype, &jobz, &uplo, &intn, NULL, &lda, NULL, &ldb, NULL,
                       &tmp_work, &lwork, &tmp_iwork, &liwork, &info);
        }
    } else if (use_gvx) {
        // Query workspace for sygvx/hegvx
        if constexpr (type_traits<T>::is_complex) {
            // query LWORK for complex types
            call_hegvx(&itype, &jobz, &range, &uplo, &intn, NULL, &lda, NULL, &ldb, &vl, &vu, &il, &iu, &abstol, NULL, NULL, NULL, &ldz,
                       &tmp_work, &lwork, &tmp_rwork, NULL, NULL, &info);
            lrwork = _calc_lwork(tmp_rwork);
        } else {
            // query LWORK for real types
            call_sygvx(&itype, &jobz, &range, &uplo, &intn, NULL, &lda, NULL, &ldb, &vl, &vu, &il, &iu, &abstol, NULL, NULL, NULL, &ldz,
                       &tmp_work, &lwork, NULL, NULL, &info);
        }
        liwork = 5 * n;  // gvx uses fixed iwork size
    } else {
        return -199;  // Unknown driver
    }
    
    if (info != 0) { return -101; }

    lwork = _calc_lwork(tmp_work);
    if (lwork < 0) { return -102; }
    
    if (use_gv) {
        liwork = 0;  // gv doesn't use iwork
    } else if (use_gvx) {
        liwork = 5 * n;  // gvx specific
    } else {
        liwork = (CBLAS_INT)tmp_iwork;
    }
    if (liwork < 0) { return -103; }

    // allocate workspace
    npy_intp bufsize = 2*n*n + lwork + (use_gvx ? n*n : 0);  // space for A and B matrices + work + (z for gvx)
    T *buf = (T *)malloc(bufsize*sizeof(T));
    if (buf == NULL) { return -104; }

    CBLAS_INT *iwork = NULL;
    if (liwork > 0) {
        iwork = (CBLAS_INT *)malloc(liwork*sizeof(CBLAS_INT));
        if (iwork == NULL) { free(buf); return -105; }
    }
    
    // For gvx, we need ifail
    CBLAS_INT *ifail = NULL;
    if (use_gvx) {
        ifail = (CBLAS_INT *)malloc(n*sizeof(CBLAS_INT));
        if (ifail == NULL) { free(buf); if(iwork) free(iwork); return -108; }
    }

    if constexpr (type_traits<T>::is_complex) {
        if (use_gv || use_gvd) {
            lrwork = _calc_lwork(tmp_rwork);
        }
        if (lrwork < 0) { free(buf); if(iwork) free(iwork); if(ifail) free(ifail); return -106; }
        rwork = (real_type *)malloc(lrwork*sizeof(real_type));
        if (rwork == NULL) { free(buf); if(iwork) free(iwork); if(ifail) free(ifail); return -107; }
    }

    // partition the workspace
    T *data_A = &buf[0];
    T *data_B = &buf[n*n];
    T *work = &buf[2*n*n];
    T *z_buf = use_gvx ? &buf[2*n*n + lwork] : NULL;  // eigenvectors buffer for gvx

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
        CBLAS_INT m = 0;  // number of eigenvalues found (for gvx)

        // compute eigenvalues (and eigenvectors if requested) for the slice
        if (use_gv) {
            // For gv driver, eigenvalues/eigenvectors are computed in-place
            if constexpr (type_traits<T>::is_complex) {
                call_hegv(&itype, &jobz, &uplo, &intn, data_A, &lda, data_B, &ldb, w_buf,
                          work, &lwork, rwork, &info);
            } else {
                call_sygv(&itype, &jobz, &uplo, &intn, data_A, &lda, data_B, &ldb, w_buf,
                          work, &lwork, &info);
            }
            m = n;  // gv computes all eigenvalues
            
            // copy eigenvectors if computed (stored in data_A for gv)
            if (compute_v) {
                copy_slice_F_to_C(ptr_v + idx*n*n, data_A, n, n, lda);
            }
        } else if (use_gvd) {
            if constexpr (type_traits<T>::is_complex) {
                call_hegvd(&itype, &jobz, &uplo, &intn, data_A, &lda, data_B, &ldb, w_buf,
                           work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
            } else {
                call_sygvd(&itype, &jobz, &uplo, &intn, data_A, &lda, data_B, &ldb, w_buf,
                           work, &lwork, iwork, &liwork, &info);
            }
            
            // copy eigenvectors if computed (stored in data_A for gvd)
            if (compute_v) {
                copy_slice_F_to_C(ptr_v + idx*n*n, data_A, n, n, lda);
            }
        } else if (use_gvx) {
            if constexpr (type_traits<T>::is_complex) {
                call_hegvx(&itype, &jobz, &range, &uplo, &intn, data_A, &lda, data_B, &ldb, &vl, &vu, &il, &iu, &abstol, &m,
                           w_buf, z_buf, &ldz, work, &lwork, rwork, iwork, ifail, &info);
            } else {
                call_sygvx(&itype, &jobz, &range, &uplo, &intn, data_A, &lda, data_B, &ldb, &vl, &vu, &il, &iu, &abstol, &m,
                           w_buf, z_buf, &ldz, work, &lwork, iwork, ifail, &info);
            }
            
            // copy eigenvectors if computed (stored in z_buf for gvx)
            if (compute_v) {
                copy_slice_F_to_C(ptr_v + idx*n*n, z_buf, n, n, ldz);
            }
        }

        if(info != 0) {
            slice_status.lapack_info = (Py_ssize_t)info;
            vec_status.push_back(slice_status);
            // cut it short on error in any slice
            goto done;
        }
    }

 done:
    free(buf);
    if (iwork) free(iwork);
    if (ifail) free(ifail);
    if (rwork) free(rwork);

    return 0;
}


template<typename T>
int
_eigh(PyArrayObject* ap_Am, PyArrayObject *ap_Bm, PyArrayObject *ap_w, PyArrayObject *ap_v,
      char uplo, char jobz, CBLAS_INT itype,
      const char* driver,  // "evr", "evx", "gvd", or "gvx"
      char range,  // 'A' = all, 'V' = by value, 'I' = by index
      CBLAS_INT il, CBLAS_INT iu,  // index range (1-based)
      typename type_traits<T>::real_type vl, typename type_traits<T>::real_type vu,  // value range
      SliceStatusVec& vec_status)
{
    int info;
    if (ap_Bm == NULL) {
        // Standard eigenvalue problem
        info = _reg_eigh<T>(ap_Am, ap_w, ap_v, uplo, jobz, driver, range, il, iu, vl, vu, vec_status);
    }
    else {
        // Generalized eigenvalue problem
        info = _gen_eigh<T>(ap_Am, ap_Bm, ap_w, ap_v, uplo, jobz, itype, driver, range, il, iu, vl, vu, vec_status);
    }
    return info;
}
