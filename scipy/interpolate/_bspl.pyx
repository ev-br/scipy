"""
Routines for evaluating and manipulating B-splines.

"""

import numpy as np
cimport numpy as cnp

cimport cython

include "_find_interval.pxi"

cdef double nan = np.nan

cimport libc.stdlib
cimport libc.math

ctypedef double complex double_complex

ctypedef fused double_or_complex:
    double
    double complex


#------------------------------------------------------------------------------
# B-splines
#------------------------------------------------------------------------------

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def evaluate_spline(double[::1] t,
             double[:,::1] c,
             int k,
             double[::1] xp,
             int der,
             int extrapolate,
             double[:,::1] out):
    """
    Evaluate a spline in the B-spline basis.

    Parameters
    ----------
    t : ndarray, shape (n+k+1)
        knots
    c : ndarray, shape (n, m)
        B-spline coefficients
    xp : ndarray, shape (s,)
        Points to evaluate the spline at.
    der : int
        Order of derivative to evaluate.
    extrapolate : int, optional
        Whether to extrapolate to ouf-of-bounds points, or to return NaNs.
    out : ndarray, shape (s, m)
        Computed values of the spline at each of the input points.
        This argument is modified in-place.

    """

    cdef int ip, jp, n, a
    cdef int i, interval
    cdef double xval

    # shape checks
    if out.shape[0] != xp.shape[0]:
        raise ValueError("out and xp have incompatible shapes")
    if out.shape[1] != c.shape[1]:
        raise ValueError("out and c have incompatible shapes")

    # check derivative order
    if der < 0:
        raise NotImplementedError("Cannot do derivative order %s." % der)

    n = c.shape[0]
    cdef double[::1] work = np.empty(k+2, dtype=np.float_)

    # evaluate
    interval = 0
    for ip in range(len(xp)):
        xval = xp[ip]

        # Find correct interval
        i = find_interval(t[k:t.shape[0]-k], xval, interval, extrapolate)

        if i < 0:
            # xval was nan etc
            for jp in range(c.shape[1]):
                out[ip, jp] = nan
            continue
        else:
            interval = i + k

        # Evaluate (k+1) b-splines which are non-zero on the interval
        # returns work = B_{m-k},..., B_{m}, 0
        evaluate_bspl(t, k, xval, interval, der, work)

        # Form linear combinations
        for jp in range(c.shape[1]):
            out[ip, jp] = 0.
            for a in range(k+1):
                out[ip, jp] += c[interval + a - k, jp] * work[a]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void evaluate_bspl(double[::1] t,
                       int k, 
                       double xval,
                       int m,
                       int nu, 
                       double [::1] bbb) nogil:
    """ Evaluate k+1 B-splines which are non-zero on interval `m`.

    On exit, the `bbb` array contains `[B_{m-k}(x), ..., B_{m}(x), 0]`
    (a zero is prepended to avoid an out-of-bounds access).

    Notes
    -----

    Is basically equivalent to Dierckx's `fpbspl` routine.

    Implements algorithm 2.21 of [1_] for nu=0, and algorithm 3.18 for nu>0.

    References
    ----------
    [1]_ Tom Lyche and Knut Morken, Spline Methods,
        http://www.uio.no/studier/emner/matnat/ifi/INF-MAT5340/v05/undervisningsmateriale/    

    """
    cdef int i, j, deg
    cdef double w0, w1
    cdef double queue[2]

    for i in range(k+2):
        bbb[i] = 0.

    if k - nu + 1 > 0:
        bbb[0] = 1.

        if k != 0:
            for deg in range(1, k+1):
                # build all k+1 B-splines of degree k
                queue[0] = 0.
                queue[1] = bbb[0]
                for j in range(m-deg, m+1):
                    w0 = t[j + deg] - t[j]
                    if w0 != 0:
                        if deg > k - nu:
                            w0 = 1. * deg / w0    # derivative order nu
                        else:
                            w0 = (xval - t[j]) / w0

                    w1 = t[j+deg+1] - t[j+1]
                    if w1 != 0:
                        if deg > k - nu:
                            w1 = -1. * deg / w1    # derivative order nu
                        else:
                            w1 = (t[j+deg+1] - xval) / w1

                    bbb[j-m+deg] = w0 * queue[0] + w1 * queue[1]
                    queue[0] = queue[1]
                    queue[1] = bbb[j - m + deg + 1]

