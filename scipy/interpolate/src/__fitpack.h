#include <iostream>
#include <tuple>
#include "../linalg/fortran_defs.h"

#define DLARTG F_FUNC(dlartg, DLARTG)

extern "C" {
void DLARTG(double *f, double *g, double *cs, double *sn, double *r);
}


namespace fitpack {

/*
 * B-spline evaluation routine.
 */

static inline void
_deBoor_D(const double *t, double x, int k, int ell, int m, double *result) {
    /*
     * On completion the result array stores
     * the k+1 non-zero values of beta^(m)_i,k(x):  for i=ell, ell-1, ell-2, ell-k.
     * Where t[ell] <= x < t[ell+1].
     */
    /*
     * Implements a recursive algorithm similar to the original algorithm of
     * deBoor.
     */
    double *hh = result + k + 1;
    double *h = result;
    double xb, xa, w;
    int ind, j, n;

    /*
     * Perform k-m "standard" deBoor iterations
     * so that h contains the k+1 non-zero values of beta_{ell,k-m}(x)
     * needed to calculate the remaining derivatives.
     */
    result[0] = 1.0;
    for (j = 1; j <= k - m; j++) {
        memcpy(hh, h, j*sizeof(double));
        h[0] = 0.0;
        for (n = 1; n <= j; n++) {
            ind = ell + n;
            xb = t[ind];
            xa = t[ind - j];
            if (xb == xa) {
                h[n] = 0.0;
                continue;
            }
            w = hh[n - 1]/(xb - xa);
            h[n - 1] += w*(xb - x);
            h[n] = w*(x - xa);
        }
    }

    /*
     * Now do m "derivative" recursions
     * to convert the values of beta into the mth derivative
     */
    for (j = k - m + 1; j <= k; j++) {
        memcpy(hh, h, j*sizeof(double));
        h[0] = 0.0;
        for (n = 1; n <= j; n++) {
            ind = ell + n;
            xb = t[ind];
            xa = t[ind - j];
            if (xb == xa) {
                h[m] = 0.0;
                continue;
            }
            w = j*hh[n - 1]/(xb - xa);
            h[n - 1] -= w;
            h[n] = w;
        }
    }
}


/* TODO:
 * 2. std::cout << array
 * 4. rationalize explicit double vs templates
 * 5. naming: __qr @ C++, _qr @ cython, _qr_py @ python tests
 * 7. checks: contiguity etc: @ cython; y.shape[0] == m
 * 8. complex
 * 9. () or [] --- runtime penalty of a tuple?
 * 10. flip boundscheck to False
 */


///////////// Bounds checking
template<bool boundscheck> void _bcheck(ssize_t index, ssize_t size, ssize_t dim);

template<>
void _bcheck<true>(ssize_t index, ssize_t size, ssize_t dim) {
    if (!((0 <= index) && (index < size))){
        auto mesg = "Out of bounds with index = " + std::to_string(index) + " of size = ";
        mesg = mesg + std::to_string(size) + " in dimension = " + std::to_string(dim);
        throw(std::runtime_error(mesg) );
    }
}

template<>
void _bcheck<false>(ssize_t index, ssize_t size, ssize_t dim) { /* noop*/ }
////////////////////////////


template<typename T, bool boundscheck=true>
struct Array1D
{
    T* data;
    ssize_t nelem;
    T& operator()(const ssize_t i) {
        _bcheck<boundscheck>(i, nelem, 0);
        return *(data + i);
    }
    Array1D(T *ptr, ssize_t num_elem) : data(ptr), nelem(num_elem) {};
  //  Array1D(const T *ptr, ssize_t num_elem) : data((const T*)ptr), nelem(num_elem) {};
};



// XXX ndim != 2, if needed
template<typename T, bool boundscheck=true>
struct Array2D
{
    T* data;
    ssize_t nrows;
    ssize_t ncols;
    T& operator()(const ssize_t i, const ssize_t j) {
        _bcheck<boundscheck>(i, nrows, 0);
        _bcheck<boundscheck>(j, ncols, 1);
        return *(data + ncols*i + j);
    }
    Array2D(T *ptr, ssize_t num_rows, ssize_t num_columns) : data(ptr), nrows(num_rows), ncols(num_columns) {};
};


// Flip boundschecking on/off here
typedef Array2D<double, true> RealArray2D;
typedef Array1D<double, true> RealArray1D;
typedef Array1D<const double, true> ConstRealArray1D;
typedef Array2D<const double, true> ConstRealArray2D;


/*
 *  Find an interval such that t[interval] <= xval < t[interval+1].
 */
inline
ssize_t
__find_interval(const double* tptr, ssize_t len_t,
                int k,
                double xval,
                ssize_t prev_l,
                int extrapolate)
{
    ConstRealArray1D t = ConstRealArray1D(tptr, len_t);

    ssize_t n = t.nelem - k - 1;
    double tb = t(k);
    double te = t(n);

    if (xval != xval) {
        // nan
        return -1;
    }

    if (((xval < tb) || (xval > te)) && !extrapolate) {
        return -1;
    }
    ssize_t l = (k < prev_l) && (prev_l < n) ? prev_l : k;

    // xval is in support, search for interval s.t. t[interval] <= xval < t[l+1]
    while ((xval < t(l)) && (l != k)) {
        l -= 1;
    }

    l += 1;
    while ((xval >= t(l)) && (l != n)) {
        l += 1;
    }

    return l-1;
}


/*
 * Fill the (m, k+1) matrix of non-zero b-splines. A row gives b-splines
 * which are non-zero at the corresponding value of `x`.
 * Also for each row store the `offset`: with full matrices, the non-zero elements
 * in row `i` start at `offset[i]`. IOW,
 * A_full[i, offset[i]: offset[i] + k + 1] = A_packed[i, :].
 *
 * What we construct here is `A_packed` and `offset` arrays.
 *
 * We also take into account possible weights for each `x` value: they
 * multiply the rows of the data matrix.
 *
 * To reconstruct the full dense matrix, `A_full`, we would need to know the
 * number of its columns, `nc`. So we return it, too.
 */
inline
void __data_matrix(const double *xptr, ssize_t m,
                           const double *tptr, ssize_t len_t,
                           int k,
                           const double *wptr,   // NB: len(w) == len(x), not checked
                           double *Aptr,         // outputs
                           ssize_t *offset_ptr,
                           ssize_t *nc,
                           double *wrk)         // work array
{
    auto x = ConstRealArray1D(xptr, m);
    auto t = ConstRealArray1D(tptr, len_t);
    auto w = ConstRealArray1D(wptr, m);
    auto A = RealArray2D(Aptr, m, k+1);
    auto offset = Array1D<ssize_t, false>(offset_ptr, m);

    ssize_t ind = k;
    for (int i=0; i < m; ++i) { 
        double xval = x(i);

        // find the interval  
        ind = __find_interval(t.data, len_t, k, xval, ind, 0);
        if (ind < 0){
            // should not happen here, validation is expected on the python side
            throw std::runtime_error("find_interval: out of bounds with x = " + std::to_string(xval));
        }
        offset(i) = ind - k;

        // compute non-zero b-splines
        _deBoor_D(t.data, xval, k, ind, 0, wrk);

        for (ssize_t j=0; j < k+1; ++j) {
            A(i, j) = wrk[j] * w(i);
        }
    }

    *nc = len_t - k - 1;
}



/*
 * Linear algebra: banded QR via Givens transforms.
 */
template<typename T>
inline
std::tuple<T, T> fprota(T c, T s, T a, T b)
{
    return std::make_tuple(
        c*a + s*b,
       -s*a + c*b
    );
}



/*
    Solve the LSQ problem ||y - A@c||^2 via QR factorization.

    QR factorization follows FITPACK: we reduce A row-by-row by Givens rotations.

    To zero out the lower triangle, we use in the row `i` and column `j < i`,
    the diagonal element in that column. That way, the sequence is
    (here `[x]` are the pair of elements to Givens-rotate)

     [x] x x x       x  x  x x      x  x  x x      x x  x  x      x x x x
     [x] x x x  ->   0 [x] x x  ->  0 [x] x x  ->  0 x  x  x  ->  0 x x x
      0  x x x       0 [x] x x      0  0  x x      0 0 [x] x      0 0 x x
      0  x x x       0  x  x x      0 [x] x x      0 0 [x] x      0 0 0 x

    The matrix A has a special structure: each row has at most (k+1) non-zeros, so
    is encoded as a PackedMatrix instance.

    On exit, the return matrix, also of shape (m, k+1), contains
    elements of the upper triangular matrix `R[i, i: i + k + 1]`.
    When we process the element (i, j), we store the rotated row in R[i, :],
    and *shift it to the left*, so that the the diagonal element is always in the
    zero-th place. This way, the process above becomes


     [x] x x x       x  x x x       x  x x x       x  x x x      x x x x
     [x] x x x  ->  [x] x x -  ->  [x] x x -   ->  x  x x -  ->  x x x -
      x  x x -      [x] x x -       x  x - -      [x] x - -      x x - -
      x  x x -       x  x x -      [x] x x -      [x] x - -      x - - -

    The most confusing part is that when rotating the row `i` with a row `j`
    above it, the offsets differ: for the upper row  `j`, `R[j, 0]` is the diagonal
    element, while for the row `i`, `R[i, 0]` is the element being annihilated.

    NB. This row-by-row Givens reduction process follows FITPACK:
    https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpcurf.f#L112-L161
    A possibly more efficient way could be to note that all data points which
    lie between two knots all have the same offset: if `t[i] < x_1 .... x_s < t[i+1]`,
    the `s-1` corresponding rows form an `(s-1, k+1)`-sized "block".
    Then a blocked QR implementation could look like
    https://people.sc.fsu.edu/~jburkardt/f77_src/band_qr/band_qr.f

    The `startrow` optional argument accounts for the scenatio with a two-step
    factorization. Namely, the preceding rows are assumend to be already
    processed and are skipped.
    This is to account for the scenario where we append new rows to an already
    triangularized matrix.
*/
void __qr_reduce(double *aptr, const ssize_t m, const ssize_t nz,    // a
                 ssize_t *offset,
                 const ssize_t nc,
                 double *yptr, ssize_t ydim1,                        // y
                 ssize_t startrow=1
)
{
    RealArray2D R = RealArray2D(aptr, m, nz);
    RealArray2D y = RealArray2D(yptr, m, ydim1);

    for (ssize_t i=startrow; i < m; ++i) {
        ssize_t oi = offset[i];
        for (ssize_t j=oi; j < nc; ++j) {

            // rotate only the lower diagonal
            if (j >= std::min(i, nc)) {
                break;
            }

            // in dense format: diag a1[j, j] vs a1[i, j]
            double c, s, r;
            DLARTG(&R(j, 0), &R(i, 0), &c, &s, &r);

            // rotate l.h.s.
            R(j, 0) = r;
            for (ssize_t l=1; l < R.ncols; ++l) {
                std::tie(R(j, l), R(i, l-1)) = fprota(c, s, R(j, l), R(i, l));
            }
            R(i, R.ncols-1) = 0.0;

            // rotate r.h.s.
            for (ssize_t l=0; l < y.ncols; ++l) {
                std::tie(y(j, l), y(i, l)) = fprota(c, s, y(j, l), y(i, l));
            }
        }
        if (i < nc) {
            offset[i] = i;
        }

    } // for(i = ...
}


void __fpback(const double *Rptr, ssize_t m, ssize_t nz,    // R(m, nz), packed
              ssize_t nc,
              const double *yptr, ssize_t ydim2,            // y(m, ydim2)
              double *cptr)
{
    auto R = ConstRealArray2D(Rptr, m, nz);
    auto y = ConstRealArray2D(yptr, m, ydim2);
    auto c = RealArray2D(cptr, nc, ydim2);

    // c[nc-1, ...] = y[nc-1] / R[nc-1, 0]
    for (ssize_t l=0; l < ydim2; ++l) {
        c(nc - 1, l) = y(nc - 1, l) / R(nc-1, 0); 
    }

    //for i in range(nc-2, -1, -1):
    //    nel = min(nz, nc-i)
    //    c[i, ...] = ( y[i] - (R[i, 1:nel, None] * c[i+1:i+nel, ...]).sum(axis=0) ) / R[i, 0]
    for (ssize_t i=nc-2; i >= 0; --i) {
        ssize_t nel = std::min(nz, nc - i);
        for (ssize_t l=0; l < ydim2; ++l){
            double ssum = y(i, l);
            for (ssize_t j=1; j < nel; ++j) {
                ssum -= R(i, j) * c(i + j, l);
            }
            ssum /= R(i, 0);
            c(i, l) = ssum;
        }
    }
}


} // namespace fitpack
