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


// XXX ndim != 2, if needed
template<typename T, bool boundscheck=true>
struct Array
{
    T* data;
    ssize_t nrows;
    ssize_t ncols;
    T& operator()(const ssize_t i, const ssize_t j) {
        _bcheck<boundscheck>(i, nrows, 0);
        _bcheck<boundscheck>(j, ncols, 1);
        return *(data + ncols*i + j);
    }
    Array(T *ptr, ssize_t num_rows, ssize_t num_columns) : data(ptr), nrows(num_rows), ncols(num_columns) {};
};


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
    Array<double, true> R = Array<double, true>(aptr, m, nz);
    Array<double, true> y = Array<double, true>(yptr, m, ydim1);

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


} // namespace fitpack
