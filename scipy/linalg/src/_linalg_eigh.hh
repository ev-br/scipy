#pragma once
#include <limits>
/*
 * Templated loops for linalg.eigh
 */

enum EighDriver : int {
    EV = 1,
    EVD = 2,
    EVR = 3,
    EVX = 4,
    GV = 5,
    GVD = 6,
    GVX = 7
};

enum EighSubsetKind : int {
    SUBSET_NONE = 0,
    SUBSET_INDEX = 1,
    SUBSET_VALUE = 2
};


inline bool
_checked_cast_cblas_int(npy_intp value, CBLAS_INT *out, const char *name)
{
    if (value < static_cast<npy_intp>(std::numeric_limits<CBLAS_INT>::min())
            || value > static_cast<npy_intp>(std::numeric_limits<CBLAS_INT>::max())) {
        PyErr_Format(PyExc_OverflowError,
                     "`%s` is too large for the LAPACK integer type.", name);
        return false;
    }
    *out = static_cast<CBLAS_INT>(value);
    return true;
}


template<typename T>
int
_std_eigh_ev(
    PyArrayObject* ap_Am, PyArrayObject* ap_w, PyArrayObject* ap_v, PyArrayObject* ap_m,
    int lower, int eigvals_only, int overwrite_a,
    SliceStatusVec& vec_status
) {
    using real_type = typename sp_type_traits<T>::real_type;
    SliceStatus slice_status;

    T* Am_data = (T *)PyArray_DATA(ap_Am);
    int ndim = PyArray_NDIM(ap_Am);
    npy_intp* shape = PyArray_SHAPE(ap_Am);
    npy_intp* strides = PyArray_STRIDES(ap_Am);
    npy_intp n = shape[ndim - 1];

    npy_intp outer_size = 1;
    for (int i = 0; i < ndim - 2; i++) {
        outer_size *= shape[i];
    }

    CBLAS_INT *ptr_m = (CBLAS_INT *)PyArray_DATA(ap_m);
    real_type *ptr_w = (real_type *)PyArray_DATA(ap_w);
    T *ptr_v = ap_v == NULL ? NULL : (T *)PyArray_DATA(ap_v);

    CBLAS_INT intn = 0, c_lwork = 0;
    CBLAS_INT info = 0;
    char jobz = eigvals_only ? 'N' : 'V';
    char uplo = lower ? 'L' : 'U';

    npy_intp data_size = overwrite_a ? 0 : n*n;
    npy_intp lwork = std::max<npy_intp>(sp_type_traits<T>::is_complex ? 2*n - 1 : 3*n - 1, 1);
    npy_intp rwork_size = sp_type_traits<T>::is_complex ? std::max<npy_intp>(3*n - 2, 1) : 0;
    npy_intp bufsize = data_size + lwork;

    if (!_checked_cast_cblas_int(n, &intn, "n")
            || !_checked_cast_cblas_int(lwork, &c_lwork, "lwork")) {
        return -90;
    }

    T *buf = (T *)malloc(bufsize*sizeof(T));
    real_type *rwork = NULL;
    if (buf == NULL) {
        return -90;
    }
    if constexpr (sp_type_traits<T>::is_complex) {
        rwork = (real_type *)malloc(rwork_size*sizeof(real_type));
        if (rwork == NULL) {
            free(buf);
            return -91;
        }
    }

    T *work = &buf[0];
    T *data = overwrite_a ? (T *)Am_data : &buf[lwork];

    for (npy_intp idx = 0; idx < outer_size; idx++) {
        init_status(slice_status, idx, sp_type_traits<T>::is_complex ? St::HER : St::SYM);

        if (!overwrite_a) {
            T *slice_ptr = compute_slice_ptr(idx, Am_data, ndim, shape, strides);
            copy_slice_F(data, slice_ptr, n, n, strides[ndim-2], strides[ndim-1]);
        }

        call_ev(&jobz, &uplo, &intn, data, &intn, ptr_w, work, &c_lwork, rwork, &info);

        if (info != 0) {
            slice_status.lapack_info = (Py_ssize_t)info;
            vec_status.push_back(slice_status);
            goto done;
        }

        ptr_w += n;
        *ptr_m++ = intn;

        if (!eigvals_only) {
            copy_slice_F_to_C(ptr_v, data, n, n, intn);
            ptr_v += n*n;
        }
    }

done:
    free(buf);
    free(rwork);
    return info < 0 ? (int)info : 0;
}


template<typename T>
int
_std_eigh_evd(
    PyArrayObject* ap_Am, PyArrayObject* ap_w, PyArrayObject* ap_v, PyArrayObject* ap_m,
    int lower, int eigvals_only, int overwrite_a,
    SliceStatusVec& vec_status
) {
    using real_type = typename sp_type_traits<T>::real_type;
    SliceStatus slice_status;

    T* Am_data = (T *)PyArray_DATA(ap_Am);
    int ndim = PyArray_NDIM(ap_Am);
    npy_intp* shape = PyArray_SHAPE(ap_Am);
    npy_intp* strides = PyArray_STRIDES(ap_Am);
    npy_intp n = shape[ndim - 1];

    npy_intp outer_size = 1;
    for (int i = 0; i < ndim - 2; i++) {
        outer_size *= shape[i];
    }

    CBLAS_INT *ptr_m = (CBLAS_INT *)PyArray_DATA(ap_m);
    real_type *ptr_w = (real_type *)PyArray_DATA(ap_w);
    T *ptr_v = ap_v == NULL ? NULL : (T *)PyArray_DATA(ap_v);

    CBLAS_INT intn = 0, c_lwork = 0, c_lrwork = 0, c_liwork = 0;
    CBLAS_INT info = 0;
    char jobz = eigvals_only ? 'N' : 'V';
    char uplo = lower ? 'L' : 'U';

    npy_intp data_size = overwrite_a ? 0 : n*n;
    npy_intp lwork = 0, lrwork = 0, liwork = 0;
    if constexpr (sp_type_traits<T>::is_complex) {
        lwork = std::max<npy_intp>(eigvals_only ? n + 1 : n*(n + 2), 1);
        lrwork = std::max<npy_intp>(eigvals_only ? n : 2*n*n + 5*n + 1, 1);
        liwork = eigvals_only ? 1 : 5*n + 3;
    } else {
        lwork = std::max<npy_intp>(eigvals_only ? 2*n + 1 : 1 + 6*n + 2*n*n, 1);
        liwork = eigvals_only ? 1 : 5*n + 3;
    }

    npy_intp bufsize = data_size + lwork;
    if (!_checked_cast_cblas_int(n, &intn, "n")
            || !_checked_cast_cblas_int(lwork, &c_lwork, "lwork")
            || !_checked_cast_cblas_int(lrwork, &c_lrwork, "lrwork")
            || !_checked_cast_cblas_int(liwork, &c_liwork, "liwork")) {
        return -92;
    }

    T *buf = (T *)malloc(bufsize*sizeof(T));
    if (buf == NULL) {
        return -92;
    }

    real_type *rwork = NULL;
    if constexpr (sp_type_traits<T>::is_complex) {
        rwork = (real_type *)malloc(lrwork*sizeof(real_type));
        if (rwork == NULL) {
            free(buf);
            return -93;
        }
    }

    CBLAS_INT *iwork = (CBLAS_INT *)malloc(liwork*sizeof(CBLAS_INT));
    if (iwork == NULL) {
        free(buf);
        free(rwork);
        return -94;
    }

    T *work = &buf[0];
    T *data = overwrite_a ? (T *)Am_data : &buf[lwork];

    for (npy_intp idx = 0; idx < outer_size; idx++) {
        init_status(slice_status, idx, sp_type_traits<T>::is_complex ? St::HER : St::SYM);

        if (!overwrite_a) {
            T *slice_ptr = compute_slice_ptr(idx, Am_data, ndim, shape, strides);
            copy_slice_F(data, slice_ptr, n, n, strides[ndim-2], strides[ndim-1]);
        }

        call_evd(
            &jobz, &uplo, &intn, data, &intn, ptr_w, work, &c_lwork,
            rwork, &c_lrwork, iwork, &c_liwork, &info
        );

        if (info != 0) {
            slice_status.lapack_info = (Py_ssize_t)info;
            vec_status.push_back(slice_status);
            goto done;
        }

        ptr_w += n;
        *ptr_m++ = intn;

        if (!eigvals_only) {
            copy_slice_F_to_C(ptr_v, data, n, n, intn);
            ptr_v += n*n;
        }
    }

done:
    free(buf);
    free(rwork);
    free(iwork);
    return info < 0 ? (int)info : 0;
}


template<typename T>
int
_std_eigh_evr(
    PyArrayObject* ap_Am, PyArrayObject* ap_w, PyArrayObject* ap_v, PyArrayObject* ap_m,
    int lower, int eigvals_only, int overwrite_a,
    int subset_kind, CBLAS_INT il, CBLAS_INT iu,
    typename sp_type_traits<T>::real_type vl_in,
    typename sp_type_traits<T>::real_type vu_in,
    SliceStatusVec& vec_status
) {
    using real_type = typename sp_type_traits<T>::real_type;
    SliceStatus slice_status;

    T* Am_data = (T *)PyArray_DATA(ap_Am);
    int ndim = PyArray_NDIM(ap_Am);
    npy_intp* shape = PyArray_SHAPE(ap_Am);
    npy_intp* strides = PyArray_STRIDES(ap_Am);
    npy_intp n = shape[ndim - 1];

    npy_intp outer_size = 1;
    for (int i = 0; i < ndim - 2; i++) {
        outer_size *= shape[i];
    }

    CBLAS_INT *ptr_m = (CBLAS_INT *)PyArray_DATA(ap_m);
    real_type *ptr_w = (real_type *)PyArray_DATA(ap_w);
    T *ptr_v = ap_v == NULL ? NULL : (T *)PyArray_DATA(ap_v);

    CBLAS_INT intn = 0, ldz = 0, c_lwork = 0, c_liwork = 0, c_lrwork = 0;
    CBLAS_INT m_expected = 0;
    CBLAS_INT m_found = 0;
    CBLAS_INT info = 0;
    char jobz = eigvals_only ? 'N' : 'V';
    char range = subset_kind == SUBSET_INDEX ? 'I' : (subset_kind == SUBSET_VALUE ? 'V' : 'A');
    char uplo = lower ? 'L' : 'U';
    real_type vl = vl_in, vu = vu_in, abstol = 0;

    npy_intp data_size = overwrite_a ? 0 : n*n;
    npy_intp z_size = 0;
    npy_intp lwork = std::max<npy_intp>(sp_type_traits<T>::is_complex ? 2*n : 26*n, 1);
    npy_intp liwork = std::max<npy_intp>(10*n, 1);
    npy_intp lrwork = sp_type_traits<T>::is_complex ? std::max<npy_intp>(24*n, 1) : 0;
    npy_intp isuppz_size = eigvals_only ? 0 : 2*std::max<npy_intp>(n, 1);
    npy_intp bufsize = data_size + z_size + lwork;

    if (!_checked_cast_cblas_int(n, &intn, "n")
            || !_checked_cast_cblas_int(n, &ldz, "ldz")
            || !_checked_cast_cblas_int(lwork, &c_lwork, "lwork")
            || !_checked_cast_cblas_int(liwork, &c_liwork, "liwork")
            || !_checked_cast_cblas_int(lrwork, &c_lrwork, "lrwork")) {
        return -100;
    }
    m_expected = subset_kind == SUBSET_INDEX ? iu - il + 1 : intn;
    z_size = eigvals_only ? 0 : n*(subset_kind == SUBSET_INDEX ? m_expected : intn);
    bufsize = data_size + z_size + lwork;

    T *buf = (T *)malloc(bufsize*sizeof(T));
    if (buf == NULL) {
        return -100;
    }

    real_type *w = (real_type *)malloc(std::max<npy_intp>(n, 1)*sizeof(real_type));
    real_type *rwork = NULL;
    CBLAS_INT *iwork = (CBLAS_INT *)malloc(liwork*sizeof(CBLAS_INT));
    CBLAS_INT *isuppz = isuppz_size > 0 ? (CBLAS_INT *)malloc(isuppz_size*sizeof(CBLAS_INT)) : NULL;
    if (w == NULL || iwork == NULL || (isuppz_size > 0 && isuppz == NULL)) {
        free(buf);
        free(w);
        free(iwork);
        free(isuppz);
        return -101;
    }

    if constexpr (sp_type_traits<T>::is_complex) {
        rwork = (real_type *)malloc(lrwork*sizeof(real_type));
        if (rwork == NULL) {
            free(buf);
            free(w);
            free(iwork);
            free(isuppz);
            return -102;
        }
    }

    T *work = &buf[0];
    T *data = overwrite_a ? (T *)Am_data : &buf[lwork];
    T *z = eigvals_only ? NULL : &buf[lwork + data_size];

    for (npy_intp idx = 0; idx < outer_size; idx++) {
        init_status(slice_status, idx, sp_type_traits<T>::is_complex ? St::HER : St::SYM);

        if (!overwrite_a) {
            T *slice_ptr = compute_slice_ptr(idx, Am_data, ndim, shape, strides);
            copy_slice_F(data, slice_ptr, n, n, strides[ndim-2], strides[ndim-1]);
        }

        call_evr(
            &jobz, &range, &uplo, &intn, data, &intn, &vl, &vu, &il, &iu,
            &abstol, &m_found, w, z, &ldz, isuppz, work, &c_lwork,
            rwork, &c_lrwork, iwork, &c_liwork, &info
        );

        if (info != 0) {
            slice_status.lapack_info = (Py_ssize_t)info;
            vec_status.push_back(slice_status);
            goto done;
        }

        if (subset_kind != SUBSET_VALUE && m_found != m_expected) {
            info = -103;
            goto done;
        }

        memcpy(ptr_w, w, m_found*sizeof(real_type));
        ptr_w += subset_kind == SUBSET_INDEX ? m_expected : intn;
        *ptr_m++ = m_found;

        if (!eigvals_only) {
            copy_slice_F_to_C(ptr_v, z, n, m_found, ldz);
            ptr_v += n*(subset_kind == SUBSET_INDEX ? m_expected : intn);
        }
    }

done:
    free(buf);
    free(w);
    free(rwork);
    free(iwork);
    free(isuppz);
    return info < 0 ? (int)info : 0;
}


template<typename T>
int
_std_eigh_evx(
    PyArrayObject* ap_Am, PyArrayObject* ap_w, PyArrayObject* ap_v, PyArrayObject* ap_m,
    int lower, int eigvals_only, int overwrite_a,
    int subset_kind, CBLAS_INT il, CBLAS_INT iu,
    typename sp_type_traits<T>::real_type vl_in,
    typename sp_type_traits<T>::real_type vu_in,
    SliceStatusVec& vec_status
) {
    using real_type = typename sp_type_traits<T>::real_type;
    SliceStatus slice_status;

    T* Am_data = (T *)PyArray_DATA(ap_Am);
    int ndim = PyArray_NDIM(ap_Am);
    npy_intp* shape = PyArray_SHAPE(ap_Am);
    npy_intp* strides = PyArray_STRIDES(ap_Am);
    npy_intp n = shape[ndim - 1];

    npy_intp outer_size = 1;
    for (int i = 0; i < ndim - 2; i++) {
        outer_size *= shape[i];
    }

    CBLAS_INT *ptr_m = (CBLAS_INT *)PyArray_DATA(ap_m);
    real_type *ptr_w = (real_type *)PyArray_DATA(ap_w);
    T *ptr_v = ap_v == NULL ? NULL : (T *)PyArray_DATA(ap_v);

    CBLAS_INT intn = 0, ldz = 0, c_lwork = 0;
    CBLAS_INT m_expected = subset_kind == SUBSET_INDEX ? iu - il + 1 : intn;
    CBLAS_INT m_found = 0;
    CBLAS_INT info = 0;
    char jobz = eigvals_only ? 'N' : 'V';
    char range = subset_kind == SUBSET_INDEX ? 'I' : (subset_kind == SUBSET_VALUE ? 'V' : 'A');
    char uplo = lower ? 'L' : 'U';
    real_type vl = vl_in, vu = vu_in, abstol = 0;

    npy_intp data_size = overwrite_a ? 0 : n*n;
    npy_intp z_size = 0;
    npy_intp lwork = std::max<npy_intp>(sp_type_traits<T>::is_complex ? 2*n : 8*n, 1);
    npy_intp iwork_size = std::max<npy_intp>(5*n, 1);
    npy_intp rwork_size = sp_type_traits<T>::is_complex ? std::max<npy_intp>(7*n, 1) : 0;
    npy_intp ifail_size = eigvals_only ? 0 : n;
    npy_intp bufsize = data_size + z_size + lwork;

    if (!_checked_cast_cblas_int(n, &intn, "n")
            || !_checked_cast_cblas_int(n, &ldz, "ldz")
            || !_checked_cast_cblas_int(lwork, &c_lwork, "lwork")) {
        return -104;
    }
    m_expected = subset_kind == SUBSET_INDEX ? iu - il + 1 : intn;
    z_size = eigvals_only ? 0 : n*(subset_kind == SUBSET_INDEX ? m_expected : intn);
    bufsize = data_size + z_size + lwork;

    T *buf = (T *)malloc(bufsize*sizeof(T));
    if (buf == NULL) {
        return -104;
    }

    real_type *w = (real_type *)malloc(std::max<npy_intp>(n, 1)*sizeof(real_type));
    real_type *rwork = NULL;
    CBLAS_INT *iwork = (CBLAS_INT *)malloc(iwork_size*sizeof(CBLAS_INT));
    CBLAS_INT *ifail = ifail_size > 0 ? (CBLAS_INT *)malloc(ifail_size*sizeof(CBLAS_INT)) : NULL;
    if (w == NULL || iwork == NULL || (ifail_size > 0 && ifail == NULL)) {
        free(buf);
        free(w);
        free(iwork);
        free(ifail);
        return -105;
    }

    if constexpr (sp_type_traits<T>::is_complex) {
        rwork = (real_type *)malloc(rwork_size*sizeof(real_type));
        if (rwork == NULL) {
            free(buf);
            free(w);
            free(iwork);
            free(ifail);
            return -106;
        }
    }

    T *work = &buf[0];
    T *data = overwrite_a ? (T *)Am_data : &buf[lwork];
    T *z = eigvals_only ? NULL : &buf[lwork + data_size];

    for (npy_intp idx = 0; idx < outer_size; idx++) {
        init_status(slice_status, idx, sp_type_traits<T>::is_complex ? St::HER : St::SYM);

        if (!overwrite_a) {
            T *slice_ptr = compute_slice_ptr(idx, Am_data, ndim, shape, strides);
            copy_slice_F(data, slice_ptr, n, n, strides[ndim-2], strides[ndim-1]);
        }

        call_evx(
            &jobz, &range, &uplo, &intn, data, &intn, &vl, &vu, &il, &iu,
            &abstol, &m_found, w, z, &ldz, work, &c_lwork, rwork,
            iwork, ifail, &info
        );

        if (info != 0) {
            slice_status.lapack_info = (Py_ssize_t)info;
            vec_status.push_back(slice_status);
            goto done;
        }

        if (subset_kind != SUBSET_VALUE && m_found != m_expected) {
            info = -107;
            goto done;
        }

        memcpy(ptr_w, w, m_found*sizeof(real_type));
        ptr_w += subset_kind == SUBSET_INDEX ? m_expected : intn;
        *ptr_m++ = m_found;

        if (!eigvals_only) {
            copy_slice_F_to_C(ptr_v, z, n, m_found, ldz);
            ptr_v += n*(subset_kind == SUBSET_INDEX ? m_expected : intn);
        }
    }

done:
    free(buf);
    free(w);
    free(rwork);
    free(iwork);
    free(ifail);
    return info < 0 ? (int)info : 0;
}


template<typename T>
int
_gen_eigh_gv(
    PyArrayObject* ap_Am, PyArrayObject* ap_Bm, PyArrayObject* ap_w, PyArrayObject* ap_v, PyArrayObject* ap_m,
    int lower, int eigvals_only, int overwrite_a, int overwrite_b, int itype,
    SliceStatusVec& vec_status
) {
    using real_type = typename sp_type_traits<T>::real_type;
    SliceStatus slice_status;

    T* Am_data = (T *)PyArray_DATA(ap_Am);
    T* Bm_data = (T *)PyArray_DATA(ap_Bm);
    int ndim = PyArray_NDIM(ap_Am);
    npy_intp* shape = PyArray_SHAPE(ap_Am);
    npy_intp* strides_A = PyArray_STRIDES(ap_Am);
    npy_intp* strides_B = PyArray_STRIDES(ap_Bm);
    npy_intp n = shape[ndim - 1];

    npy_intp outer_size = 1;
    for (int i = 0; i < ndim - 2; i++) {
        outer_size *= shape[i];
    }

    CBLAS_INT *ptr_m = (CBLAS_INT *)PyArray_DATA(ap_m);
    real_type *ptr_w = (real_type *)PyArray_DATA(ap_w);
    T *ptr_v = ap_v == NULL ? NULL : (T *)PyArray_DATA(ap_v);

    CBLAS_INT intn = 0, c_lwork = 0, c_itype = 0;
    CBLAS_INT info = 0;
    char jobz = eigvals_only ? 'N' : 'V';
    char uplo = lower ? 'L' : 'U';

    npy_intp A_size = overwrite_a ? 0 : n*n;
    npy_intp B_size = overwrite_b ? 0 : n*n;
    npy_intp lwork = std::max<npy_intp>(sp_type_traits<T>::is_complex ? 2*n - 1 : 3*n - 1, 1);
    npy_intp rwork_size = sp_type_traits<T>::is_complex ? std::max<npy_intp>(3*n - 2, 1) : 0;
    npy_intp bufsize = A_size + B_size + lwork;

    if (!_checked_cast_cblas_int(n, &intn, "n")
            || !_checked_cast_cblas_int(lwork, &c_lwork, "lwork")
            || !_checked_cast_cblas_int(static_cast<npy_intp>(itype), &c_itype, "itype")) {
        return -104;
    }

    T *buf = (T *)malloc(bufsize*sizeof(T));
    real_type *rwork = NULL;
    if (buf == NULL) {
        return -104;
    }
    if constexpr (sp_type_traits<T>::is_complex) {
        rwork = (real_type *)malloc(rwork_size*sizeof(real_type));
        if (rwork == NULL) {
            free(buf);
            return -105;
        }
    }

    T *work = &buf[0];
    T *data_A = overwrite_a ? (T *)Am_data : &buf[lwork];
    T *data_B = overwrite_b ? (T *)Bm_data : &buf[lwork + A_size];

    for (npy_intp idx = 0; idx < outer_size; idx++) {
        init_status(slice_status, idx, sp_type_traits<T>::is_complex ? St::HER : St::SYM);

        if (!overwrite_a) {
            T *slice_ptr_A = compute_slice_ptr(idx, Am_data, ndim, shape, strides_A);
            copy_slice_F(data_A, slice_ptr_A, n, n, strides_A[ndim-2], strides_A[ndim-1]);
        }
        if (!overwrite_b) {
            T *slice_ptr_B = compute_slice_ptr(idx, Bm_data, ndim, shape, strides_B);
            copy_slice_F(data_B, slice_ptr_B, n, n, strides_B[ndim-2], strides_B[ndim-1]);
        }

        call_gv(&c_itype, &jobz, &uplo, &intn, data_A, &intn, data_B, &intn,
                ptr_w, work, &c_lwork, rwork, &info);

        if (info != 0) {
            slice_status.lapack_info = (Py_ssize_t)info;
            vec_status.push_back(slice_status);
            goto done;
        }

        ptr_w += n;
        *ptr_m++ = intn;

        if (!eigvals_only) {
            copy_slice_F_to_C(ptr_v, data_A, n, n, intn);
            ptr_v += n*n;
        }
    }

done:
    free(buf);
    free(rwork);
    return info < 0 ? (int)info : 0;
}


template<typename T>
int
_gen_eigh_gvd(
    PyArrayObject* ap_Am, PyArrayObject* ap_Bm, PyArrayObject* ap_w, PyArrayObject* ap_v, PyArrayObject* ap_m,
    int lower, int eigvals_only, int overwrite_a, int overwrite_b, int itype,
    SliceStatusVec& vec_status
) {
    using real_type = typename sp_type_traits<T>::real_type;
    SliceStatus slice_status;

    T* Am_data = (T *)PyArray_DATA(ap_Am);
    T* Bm_data = (T *)PyArray_DATA(ap_Bm);
    int ndim = PyArray_NDIM(ap_Am);
    npy_intp* shape = PyArray_SHAPE(ap_Am);
    npy_intp* strides_A = PyArray_STRIDES(ap_Am);
    npy_intp* strides_B = PyArray_STRIDES(ap_Bm);
    npy_intp n = shape[ndim - 1];

    npy_intp outer_size = 1;
    for (int i = 0; i < ndim - 2; i++) {
        outer_size *= shape[i];
    }

    CBLAS_INT *ptr_m = (CBLAS_INT *)PyArray_DATA(ap_m);
    real_type *ptr_w = (real_type *)PyArray_DATA(ap_w);
    T *ptr_v = ap_v == NULL ? NULL : (T *)PyArray_DATA(ap_v);

    CBLAS_INT intn = 0, c_lwork = 0, c_lrwork = 0, c_liwork = 0, c_itype = 0;
    CBLAS_INT info = 0;
    char jobz = eigvals_only ? 'N' : 'V';
    char uplo = lower ? 'L' : 'U';

    npy_intp A_size = overwrite_a ? 0 : n*n;
    npy_intp B_size = overwrite_b ? 0 : n*n;
    npy_intp lwork = 0, lrwork = 0, liwork = 0;
    if constexpr (sp_type_traits<T>::is_complex) {
        lwork = std::max<npy_intp>(eigvals_only ? n + 1 : n*(n + 2), 1);
        lrwork = std::max<npy_intp>(eigvals_only ? n : 2*n*n + 5*n + 1, 1);
        liwork = eigvals_only ? 1 : 5*n + 3;
    } else {
        lwork = std::max<npy_intp>(eigvals_only ? 2*n + 1 : 1 + 6*n + 2*n*n, 1);
        liwork = eigvals_only ? 1 : 5*n + 3;
    }

    npy_intp bufsize = A_size + B_size + lwork;
    if (!_checked_cast_cblas_int(n, &intn, "n")
            || !_checked_cast_cblas_int(lwork, &c_lwork, "lwork")
            || !_checked_cast_cblas_int(lrwork, &c_lrwork, "lrwork")
            || !_checked_cast_cblas_int(liwork, &c_liwork, "liwork")
            || !_checked_cast_cblas_int(static_cast<npy_intp>(itype), &c_itype, "itype")) {
        return -110;
    }

    T *buf = (T *)malloc(bufsize*sizeof(T));
    if (buf == NULL) {
        return -110;
    }

    real_type *rwork = NULL;
    if constexpr (sp_type_traits<T>::is_complex) {
        rwork = (real_type *)malloc(lrwork*sizeof(real_type));
        if (rwork == NULL) {
            free(buf);
            return -111;
        }
    }

    CBLAS_INT *iwork = (CBLAS_INT *)malloc(liwork*sizeof(CBLAS_INT));
    if (iwork == NULL) {
        free(buf);
        free(rwork);
        return -112;
    }

    T *work = &buf[0];
    T *data_A = overwrite_a ? (T *)Am_data : &buf[lwork];
    T *data_B = overwrite_b ? (T *)Bm_data : &buf[lwork + A_size];

    for (npy_intp idx = 0; idx < outer_size; idx++) {
        init_status(slice_status, idx, sp_type_traits<T>::is_complex ? St::HER : St::SYM);

        if (!overwrite_a) {
            T *slice_ptr_A = compute_slice_ptr(idx, Am_data, ndim, shape, strides_A);
            copy_slice_F(data_A, slice_ptr_A, n, n, strides_A[ndim-2], strides_A[ndim-1]);
        }
        if (!overwrite_b) {
            T *slice_ptr_B = compute_slice_ptr(idx, Bm_data, ndim, shape, strides_B);
            copy_slice_F(data_B, slice_ptr_B, n, n, strides_B[ndim-2], strides_B[ndim-1]);
        }

        call_gvd(
            &c_itype, &jobz, &uplo, &intn, data_A, &intn, data_B, &intn,
            ptr_w, work, &c_lwork, rwork, &c_lrwork,
            iwork, &c_liwork, &info
        );

        if (info != 0) {
            slice_status.lapack_info = (Py_ssize_t)info;
            vec_status.push_back(slice_status);
            goto done;
        }

        ptr_w += n;
        *ptr_m++ = intn;

        if (!eigvals_only) {
            copy_slice_F_to_C(ptr_v, data_A, n, n, intn);
            ptr_v += n*n;
        }
    }

done:
    free(buf);
    free(rwork);
    free(iwork);
    return info < 0 ? (int)info : 0;
}


template<typename T>
int
_gen_eigh_gvx(
    PyArrayObject* ap_Am, PyArrayObject* ap_Bm, PyArrayObject* ap_w, PyArrayObject* ap_v, PyArrayObject* ap_m,
    int lower, int eigvals_only, int overwrite_a, int overwrite_b, int itype,
    int subset_kind, CBLAS_INT il, CBLAS_INT iu,
    typename sp_type_traits<T>::real_type vl_in,
    typename sp_type_traits<T>::real_type vu_in,
    SliceStatusVec& vec_status
) {
    using real_type = typename sp_type_traits<T>::real_type;
    SliceStatus slice_status;

    T* Am_data = (T *)PyArray_DATA(ap_Am);
    T* Bm_data = (T *)PyArray_DATA(ap_Bm);
    int ndim = PyArray_NDIM(ap_Am);
    npy_intp* shape = PyArray_SHAPE(ap_Am);
    npy_intp* strides_A = PyArray_STRIDES(ap_Am);
    npy_intp* strides_B = PyArray_STRIDES(ap_Bm);
    npy_intp n = shape[ndim - 1];

    npy_intp outer_size = 1;
    for (int i = 0; i < ndim - 2; i++) {
        outer_size *= shape[i];
    }

    CBLAS_INT *ptr_m = (CBLAS_INT *)PyArray_DATA(ap_m);
    real_type *ptr_w = (real_type *)PyArray_DATA(ap_w);
    T *ptr_v = ap_v == NULL ? NULL : (T *)PyArray_DATA(ap_v);

    CBLAS_INT intn = 0, ldz = 0, c_lwork = 0, c_itype = 0;
    CBLAS_INT m_expected = 0;
    CBLAS_INT m_found = 0;
    CBLAS_INT info = 0;
    char jobz = eigvals_only ? 'N' : 'V';
    char range = subset_kind == SUBSET_INDEX ? 'I' : (subset_kind == SUBSET_VALUE ? 'V' : 'A');
    char uplo = lower ? 'L' : 'U';
    real_type vl = vl_in, vu = vu_in, abstol = 0;

    npy_intp A_size = overwrite_a ? 0 : n*n;
    npy_intp B_size = overwrite_b ? 0 : n*n;
    npy_intp z_size = 0;
    npy_intp lwork = std::max<npy_intp>(sp_type_traits<T>::is_complex ? 2*n : 8*n, 1);
    npy_intp iwork_size = 5*n;
    npy_intp rwork_size = sp_type_traits<T>::is_complex ? 7*n : 0;
    npy_intp ifail_size = eigvals_only ? 0 : n;
    npy_intp bufsize = A_size + B_size + z_size + lwork;

    if (!_checked_cast_cblas_int(n, &intn, "n")
            || !_checked_cast_cblas_int(n, &ldz, "ldz")
            || !_checked_cast_cblas_int(lwork, &c_lwork, "lwork")
            || !_checked_cast_cblas_int(static_cast<npy_intp>(itype), &c_itype, "itype")) {
        return -120;
    }
    m_expected = subset_kind == SUBSET_INDEX ? iu - il + 1 : intn;
    z_size = eigvals_only ? 0 : n*(subset_kind == SUBSET_INDEX ? m_expected : intn);
    bufsize = A_size + B_size + z_size + lwork;

    T *buf = (T *)malloc(bufsize*sizeof(T));
    if (buf == NULL) {
        return -120;
    }

    real_type *w = (real_type *)malloc(std::max<npy_intp>(n, 1)*sizeof(real_type));
    real_type *rwork = NULL;
    CBLAS_INT *iwork = (CBLAS_INT *)malloc(iwork_size*sizeof(CBLAS_INT));
    CBLAS_INT *ifail = ifail_size > 0 ? (CBLAS_INT *)malloc(ifail_size*sizeof(CBLAS_INT)) : NULL;
    if (w == NULL || iwork == NULL || (ifail_size > 0 && ifail == NULL)) {
        free(buf);
        free(w);
        free(iwork);
        free(ifail);
        return -121;
    }

    if constexpr (sp_type_traits<T>::is_complex) {
        rwork = (real_type *)malloc(rwork_size*sizeof(real_type));
        if (rwork == NULL) {
            free(buf);
            free(w);
            free(iwork);
            free(ifail);
            return -122;
        }
    }

    T *work = &buf[0];
    T *data_A = overwrite_a ? (T *)Am_data : &buf[lwork];
    T *data_B = overwrite_b ? (T *)Bm_data : &buf[lwork + A_size];
    T *z = eigvals_only ? NULL : &buf[lwork + A_size + B_size];

    for (npy_intp idx = 0; idx < outer_size; idx++) {
        init_status(slice_status, idx, sp_type_traits<T>::is_complex ? St::HER : St::SYM);

        if (!overwrite_a) {
            T *slice_ptr_A = compute_slice_ptr(idx, Am_data, ndim, shape, strides_A);
            copy_slice_F(data_A, slice_ptr_A, n, n, strides_A[ndim-2], strides_A[ndim-1]);
        }
        if (!overwrite_b) {
            T *slice_ptr_B = compute_slice_ptr(idx, Bm_data, ndim, shape, strides_B);
            copy_slice_F(data_B, slice_ptr_B, n, n, strides_B[ndim-2], strides_B[ndim-1]);
        }

        call_gvx(
            &c_itype, &jobz, &range, &uplo, &intn, data_A, &intn, data_B, &intn,
            &vl, &vu, &il, &iu, &abstol, &m_found, w, z, &ldz, work,
            &c_lwork, rwork, iwork, ifail, &info
        );

        if (info != 0) {
            slice_status.lapack_info = (Py_ssize_t)info;
            vec_status.push_back(slice_status);
            goto done;
        }

        if (subset_kind != SUBSET_VALUE && m_found != m_expected) {
            info = -123;
            goto done;
        }

        memcpy(ptr_w, w, m_found*sizeof(real_type));
        ptr_w += subset_kind == SUBSET_INDEX ? m_expected : intn;
        *ptr_m++ = m_found;

        if (!eigvals_only) {
            copy_slice_F_to_C(ptr_v, z, n, m_found, ldz);
            ptr_v += n*(subset_kind == SUBSET_INDEX ? m_expected : intn);
        }
    }

done:
    free(buf);
    free(w);
    free(rwork);
    free(iwork);
    free(ifail);
    return info < 0 ? (int)info : 0;
}


template<typename T>
int
_eigh(
    PyArrayObject* ap_Am, PyArrayObject* ap_Bm,
    PyArrayObject* ap_w, PyArrayObject* ap_v, PyArrayObject* ap_m,
    int lower, int eigvals_only, int overwrite_a, int overwrite_b,
    int itype, int subset_kind, CBLAS_INT il, CBLAS_INT iu,
    typename sp_type_traits<T>::real_type vl,
    typename sp_type_traits<T>::real_type vu,
    int driver,
    SliceStatusVec& vec_status
) {
    if (ap_Bm == NULL) {
        if (driver == EV) {
            return _std_eigh_ev<T>(
                ap_Am, ap_w, ap_v, ap_m, lower, eigvals_only, overwrite_a, vec_status
            );
        }
        if (driver == EVD) {
            return _std_eigh_evd<T>(
                ap_Am, ap_w, ap_v, ap_m, lower, eigvals_only, overwrite_a, vec_status
            );
        }
        if (driver == EVR) {
            return _std_eigh_evr<T>(
                ap_Am, ap_w, ap_v, ap_m, lower, eigvals_only, overwrite_a,
                subset_kind, il, iu, vl, vu, vec_status
            );
        }
        if (driver == EVX) {
            return _std_eigh_evx<T>(
                ap_Am, ap_w, ap_v, ap_m, lower, eigvals_only, overwrite_a,
                subset_kind, il, iu, vl, vu, vec_status
            );
        }
        return -200;
    }

    if (driver == GV) {
        return _gen_eigh_gv<T>(
            ap_Am, ap_Bm, ap_w, ap_v, ap_m, lower, eigvals_only, overwrite_a, overwrite_b,
            itype, vec_status
        );
    }

    if (driver == GVD) {
        return _gen_eigh_gvd<T>(
            ap_Am, ap_Bm, ap_w, ap_v, ap_m, lower, eigvals_only, overwrite_a, overwrite_b,
            itype, vec_status
        );
    }
    if (driver == GVX) {
        return _gen_eigh_gvx<T>(
            ap_Am, ap_Bm, ap_w, ap_v, ap_m, lower, eigvals_only, overwrite_a, overwrite_b,
            itype, subset_kind, il, iu, vl, vu, vec_status
        );
    }
    return -201;
}
