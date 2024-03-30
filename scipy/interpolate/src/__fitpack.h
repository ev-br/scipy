#pragma once
#include <iostream>
#include <tuple>
#include <vector>
#include <array>
#include "../_build_utils/src/fortran_defs.h"
#include "../_build_utils/src/mdspan.h"

#define DLARTG F_FUNC(dlartg, DLARTG)


namespace fitpack {


/*
 * 1D and 2D array wrappers, with and without boundschecking
 */

template<typename T>
using array_1D_t = std::mdspan<T, std::dextents<ssize_t, 1>, std::layout_stride> ;

template<typename T>
using array_2D_t = std::mdspan<T, std::dextents<ssize_t, 2>, std::layout_stride> ;


template<typename T> inline array_1D_t<T>
wrap_1D(T *ptr, ssize_t sz) {
    std::array<ssize_t, 1> exts = {sz};
    std::array<ssize_t, 1> strides = {1};
    return array_1D_t<T>(ptr, {exts, strides});
}

template<typename T> inline array_2D_t<T>
wrap_2D(T *ptr, ssize_t sz1, ssize_t sz2) {
    std::array<ssize_t, 2> exts = {sz1, sz2};
    std::array<ssize_t, 2> strides = {sz2, 1};
    return array_2D_t<T>(ptr, {exts, strides});
}


/*
 * LAPACK Givens rotation
 */
extern "C" {
void DLARTG(double *f, double *g, double *cs, double *sn, double *r);
}


/*
 * Apply a Givens transform.
 */
template<typename T>
inline
std::tuple<T, T>
fprota(T c, T s, T a, T b)
{
    return std::make_tuple(
        c*a + s*b,
       -s*a + c*b
    );
}


/*
 * B-spline evaluation routine.
 */
void
_deBoor_D(const double *t, double x, int k, int ell, int m, double *result);


/*
 *  Find an interval such that t[interval] <= xval < t[interval+1].
 */
ssize_t
_find_interval(const double* tptr, ssize_t len_t,
               int k,
               double xval,
               ssize_t prev_l,
               int extrapolate);



/*
 * Fill the (m, k+1) matrix of non-zero b-splines.
 */
void
data_matrix(/* inputs */
            const double *xptr, ssize_t m,      // x, shape (m,)
            const double *tptr, ssize_t len_t,  // t, shape (len_t,)
            int k,
            const double *wptr,                 // weights, shape (m,) // NB: len(w) == len(x), not checked
            /* outputs */
            double *Aptr,                       // A, shape(m, k+1)
            ssize_t *offset_ptr,                // offset, shape (m,)
            ssize_t *nc,                        // the number of coefficient
            /* work array*/
            double *wrk                         // work, shape (2k+2)
);


/*
    Solve the LSQ problem ||y - A@c||^2 via QR factorization.
    This routine MODIFIES `a` & `y` in-place.
*/
void
qr_reduce(double *aptr, const ssize_t m, const ssize_t nz, // a(m, nz), packed
          ssize_t *offset,                                 // offset(m)
          const ssize_t nc,                                // dense would be a(m, nc)
          double *yptr, const ssize_t ydim1,               // y(m, ydim2)
          const ssize_t startrow=1
);


/*
 * Back substitution solve of `R @ c = y` with an upper triangular R.
 */
void
fpback( /* inputs*/
       const double *Rptr, ssize_t m, ssize_t nz,    // R(m, nz), packed
       ssize_t nc,                                   // dense R would be (m, nc)
       const double *yptr, ssize_t ydim2,            // y(m, ydim2)
        /* output */
       double *cptr                                 // c(nc, ydim2)
);


/*
 * A helper for _fpknot:
 * Split the `x` array into knot "runs" and sum the residuals per "run".
 */
typedef std::tuple<std::vector<double>, std::vector<ssize_t>> pair_t;

pair_t
_split(array_1D_t<const double> x,
       array_1D_t<const double> t,
       int k,
       array_1D_t<const double> residuals);


/*
 * Find a position for a new knot, a la FITPACK
 */
double
fpknot(const double *x_ptr, ssize_t m,
       const double *t_ptr, ssize_t len_t,
       int k,
       const double *residuals_ptr);


} // namespace fitpack
