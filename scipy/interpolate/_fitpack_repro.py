""" Replicate FITPACK's logic, in bits and pieces.
"""


import warnings
import operator
import math
import numpy as np

from scipy.interpolate._bsplines import (
    _not_a_knot, make_lsq_spline, make_interp_spline, BSpline,
    fpcheck, PackedMatrix, _qr_reduce, fpback, _lsq_solve_qr
)

#    cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#    c  part 1: determination of the number of knots and their position     c
#    c  **************************************************************      c
#
# https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpcurf.f#L31

# Hardcoded in curfit.f
TOL = 0.001
MAXIT = 20


def _get_residuals(x, y, t, k, w):
 #   from scipy.interpolate._bsplines import make_lsq_spline
    # FITPACK has (w*(spl(x)-y))**2; make_lsq_spline has w*(spl(x)-y)**2

    w2 = w**2
    # XXX: reuse the QR solver from gh-19753
    spl = make_lsq_spline(x, y, w=w2, t=t, k=k)
    residuals = w2 * (spl(x) - y)**2
    return residuals


def _add_knot(x, y, t, k, w):
    """Add a new knot.

    (Approximately) replicate FITPACK's logic:
      1. split the `x` array into knot intervals, ``t(j+k) <= x(i) <= t(j+k+1)``
      2. find the interval with the maximum sum of residuals
      3. insert a new knot into the middle of that interval.

    NB: a new knot is in fact an `x` value at the middle of the interval.
    So *the knots are a subset of `x`*.

    This routine is an analog of
    https://github.com/scipy/scipy/blob/v1.11.4/scipy/interpolate/fitpack/fpcurf.f#L190-L215
    (cf _split function)

    and https://github.com/scipy/scipy/blob/v1.11.4/scipy/interpolate/fitpack/fpknot.f
    """
    fparts, ix = _split(x, y, t, k, w)

    ### redo
    assert all(ix == np.searchsorted(x, t[k:-k]))

    # find the interval with max fparts and non-zero number of x values inside
    idx_max = -101
    fpart_max = -1e100
    for i in range(len(fparts)):
        if ix[i+1] - ix[i] > 1 and fparts[i] > fpart_max:
            idx_max = i
            fpart_max = fparts[i]

    if idx_max == -101:
        raise ValueError("Internal error, please report it to SciPy developers.")

    # round up, like Dierckx does? This is really arbitrary though.
    idx_newknot = (ix[idx_max] + ix[idx_max+1] + 1) // 2
    new_knot = x[idx_newknot]

    idx_t = np.searchsorted(t, new_knot)
    t_new = np.r_[t[:idx_t], new_knot, t[idx_t:]]

    return t_new


def _split(x, y, t, k, w):
    """Split the `x` array into knot intervals and compute the residuals.

    The intervals are `t(j+k) <= x(i) <= t(j+k+1)`. Return the arrays of
    the start and end `x` indices of the intervals and the sums of residuals:
    `fparts[i]` corresponds to the interval
    `x_intervals[i] <= xvalue <= x_intervals[i+1]]`.

    This routine is a (best-effort) translation of
    https://github.com/scipy/scipy/blob/v1.11.4/scipy/interpolate/fitpack/fpcurf.f#L190-L215
    """
# c  search for knot interval t(number+k) <= x <= t(number+k+1) where
# c  fpint(number) is maximal on the condition that nrdata(number)
# c  not equals zero.
    residuals = _get_residuals(x, y, t, k, w)

    interval = k+1
    x_intervals = [0]
    fparts = []
    fpart = 0.0
    for it in range(len(x)):
        xv, rv = x[it], residuals[it]
        fpart += rv

        if (xv >= t[interval]) and interval < len(t) - k - 1:
            # end of the current t interval: split the weight at xv by 1/2
            # between two intervals
            carry = rv / 2.0
            fpart -= carry
            fparts.append(fpart)

            fpart = carry
            interval += 1

            x_intervals.append(it)

    x_intervals.append(len(x)-1)
    fparts.append(fpart)

    return fparts, x_intervals

    '''
    The whole _split routine is basically this:

    ix = np.searchsorted(x, t[k:-k])
    # sum half-open intervals
    fparts = [residuals[ix[i]:ix[i+1]].sum() for i in range(len(ix)-1)]
    carries = residuals[x[ix[1:-1]]]

    for i in range(len(carries)):     # split residuals at internal knots
        carry = carries[i] / 2
        fparts[i] += carry
        fparts[i+1] -= carry

    fparts[-1] += residuals[-1]       # add the contribution of the last knot

    assert sum(fparts) == sum(residuals)
    '''



def _validate_inputs(x, y, w, k, s, xb, xe):
    """Common input validations for generate_knots and make_splrep.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise NotImplementedError(f"{y.ndim = } not implemented yet.")

    if w is None:
        w = np.ones_like(y, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
        if (w < 0).any():
            raise ValueError("Weights must be non-negative")
    if w.ndim != 1 or w.shape[0] != x.shape[0]:
        raise ValueError(f"Weights is incompatible: {w.shape =} != {x.shape}.")

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Data is incompatible: {x.shape = } and {y.shape = }.")
    if (x[1:] < x[:-1]).any():
        raise ValueError("Expect `x` to be an ordered 1D sequence.")

    k = operator.index(k)

    if s < 0:
        raise ValueError(f"`s` must be non-negative. Got {s = }")

    if xb is None:
        xb = min(x)
    if xe is None:
        xe = max(x)

    return x, y, w, k, s, xb, xe


def generate_knots(x, y, k=3, *, s=0, w=None, nest=None, xb=None, xe=None):
    """Replicate FITPACK's constructing the knot vector.

    Parameters
    ----------
    x, y : array_like
        The data points defining the curve ``y = f(x)``.
    k : int, optional
        The spline degree. Default is cubic, ``k = 3``.
    s : float, optional
        The smoothing factor.
    w : array_like, optional
        Weights.
    nest : int, optional
        Stop when at least this many knots are placed.
    xb : float, optional
        The boundary of the approximation interval. If None (default),
        is set to ``x[0]``.
    xe : float, optional
        The boundary of the approximation interval. If None (default),
        is set to ``x[-1]``.

    Yields
    ------
    t : ndarray
        Knot vectors with an increasing number of knots.
        The generator is finite: it stops when the smoothing critetion is
        satisfied, or when then number of knots exceeds the maximum value:
        the user-provided `nest` or `x.size + k + 1` --- which is the knot vector
        for the interpolating spline.

    Examples
    --------
    Generate some noisy data and fit a sequence of LSQ splines:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import make_lsq_spline, generate_knots
    >>> rng = np.random.default_rng(12345)
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * rng.standard_normal(size=50)

    >>> knots = list(generate_knots(x, y, s=1e-10))
    >>> for t in knots[::3]:
    ...     spl = make_lsq_spline(x, y, t)
    ...     xs = xs = np.linspace(-3, 3, 201)
    ...     plt.plot(xs, spl(xs), '-', label=f'n = {len(t)}', lw=3, alpha=0.7)
    >>> plt.plot(x, y, 'o', label='data')
    >>> plt.plot(xs, np.exp(-xs**2), '--')
    >>> plt.legend()

    Note that increasing the number of knots make the result follow the data
    more and more closely.

    Also note that a step of the generator may add multiple knots:

    >>> [len(t) for t in knots]
    [8, 9, 10, 12, 16, 24, 40, 48, 52]

    Notes
    -----
    The routine generates successive knots vectors of increasing length, starting
    from ``2*(k+1)`` to ``len(x) + k + 1``, trying to make knots more dense
    in the regions where the deviation of the LSQ spline from data is large.

    When the maximum number of knots, ``len(x) + k + 1`` is reached
    (this happens when ``s`` is small and ``nest`` is large), the generator
    stops, and the last output is the knots for interpolation with the
    not-a-knot boundary condition.

    Knots are located at data sites, unless ``k`` is even and the number of knots
    is ``len(x) + k + 1``. In that case, the last output of the generator
    has internal knots at Greville sites, ``(x[1:] + x[:-1]) / 2``.
    """
    if s == 0:
        if nest is not None or w is not None:
            raise ValueError("s == 0 is interpolation only")
        t = _not_a_knot(x, k)
        yield t
        return

    x, y, w, k, s, xb, xe = _validate_inputs(x, y, w, k, s, xb, xe)

    acc = s * TOL
    m = x.size    # the number of data points

    if nest is None:
        # the max number of knots. This is set in _fitpack_impl.py line 274
        # and fitpack.pyf line 198
        nest = max(m + k + 1, 2*k + 3)
    else:
        if nest < 2*(k+1):
            raise ValueError(f"`nest` too small: {nest = } < 2*(k+1) = {2*(k+1)}.")

    nmin = 2*(k+1)    # the number of knots for an LSQ polynomial approximation
    nmax = m + k + 1  # the number of knots for the spline interpolation

    # start from no internal knots
    t = np.asarray([xb]*(k+1) + [xe]*(k+1), dtype=float)
    n = t.shape[0]
    fp = 0.0
    fpold = 0.0

    # c  main loop for the different sets of knots. m is a safe upper bound
    # c  for the number of trials.
    for _ in range(m):

     #   if _ > 3:
     #       breakpoint()

        yield t

        # construct the LSQ spline with this set of knots
        fpold = fp
        residuals = _get_residuals(x, y, t, k, w=w)
        fp = residuals.sum()
        fpms = fp - s

        # c  test whether the approximation sinf(x) is an acceptable solution.
        # c  if f(p=inf) < s accept the choice of knots.
        if (abs(fpms) < acc) or (fpms < 0):
            return

        # ### c  increase the number of knots. ###

        # c  determine the number of knots nplus we are going to add.
        if n == nmin:
            # the first iteration
            nplus = 1
        else:
            delta = fpold - fp
            npl1 = int(nplus * fpms / delta) if delta > acc else nplus*2
            nplus = min(nplus*2, max(npl1, nplus//2, 1))

        # actually add knots
        for j in range(nplus):
            t = _add_knot(x, y, t, k, w)

            # check if we have enough knots already

            n = t.shape[0]
            # c  if n = nmax, sinf(x) is an interpolating spline.
            # c  if n=nmax we locate the knots as for interpolation.
            if (n >= nmax):
                t = _not_a_knot(x, k)
                yield t
                return

            # c  if n=nest we cannot increase the number of knots because of
            # c  the storage capacity limitation.
            if (n >= nest):
                yield t
                return

    # this should never be reached
    return


def construct_knot_vector(x, y, *, s, k=3, w=None, nest=None, xb=None, xe=None):
    # return the last value generated
    # XXX: needed? vs list(generate_knots(...))[-1]
    for t in generate_knots(x, y, k=k, s=s, w=w, nest=nest, xb=xb, xe=xe):
        pass
    return t


#   cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#   c  part 2: determination of the smoothing spline sp(x).                c
#   c  ***************************************************                 c
#   c  we have determined the number of knots and their position.          c
#   c  we now compute the b-spline coefficients of the smoothing spline    c
#   c  sp(x). the observation matrix a is extended by the rows of matrix   c
#   c  b expressing that the kth derivative discontinuities of sp(x) at    c
#   c  the interior knots t(k+2),...t(n-k-1) must be zero. the corres-     c
#   c  ponding weights of these additional rows are set to 1/p.            c
#   c  iteratively we then have to determine the value of p such that      c
#   c  f(p)=sum((w(i)*(y(i)-sp(x(i))))**2) be = s. we already know that    c
#   c  the least-squares kth degree polynomial corresponds to p=0, and     c
#   c  that the least-squares spline corresponds to p=infinity. the        c
#   c  iteration process which is proposed here, makes use of rational     c
#   c  interpolation. since f(p) is a convex and strictly decreasing       c
#   c  function of p, it can be approximated by a rational function        c
#   c  r(p) = (u*p+v)/(p+w). three values of p(p1,p2,p3) with correspond-  c
#   c  ing values of f(p) (f1=f(p1)-s,f2=f(p2)-s,f3=f(p3)-s) are used      c
#   c  to calculate the new value of p such that r(p)=s. convergence is    c
#   c  guaranteed by taking f1>0 and f3<0.                                 c
#   cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


def prodd(t, i, j, k):
    res = 1.0
    for s in range(k+2):
        if i + s != j:
            res *= (t[j] - t[i+s])
    return res


def disc(t, k):
    """Discontinuity matrix: jumps of k-th derivatives of b-splines at internal knots.

    See Eqs. (9)-(10) of Ref. [1], or, equivalently, Eq. (3.43) of Ref. [2].

    This routine assumes internal knots are all simple (have multiplicity =1).

    Parameters
    ----------
    t : ndarray, 1D, shape(n,)
        Knots.
    k : int
        The spline degree

    Returns
    -------
    disc : ndarray, shape(n-2*k-1, k+2)
        The jumps of the k-th derivatives of b-splines at internal knots,
        ``t[k+1], ...., t[n-k-1]``.

    Notes
    -----

    The normalization here follows FITPACK:
    (https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpdisc.f#L36)

    The k-th derivative jumps are multiplied by a factor::

        (delta / nrint)**k / k!

    where ``delta`` is the length of the interval spanned by internal knots, and
    ``nrint`` is one less the number of internal knots (i.e., the number of
    subintervals between them).

    References
    ----------
    .. [1] Paul Dierckx, Algorithms for smoothing data with periodic and parametric
           splines, Computer Graphics and Image Processing, vol. 20, p. 171 (1982).
           :doi:`10.1016/0146-664X(82)90043-0`

    .. [2] Tom Lyche and Knut Morken, Spline methods,
        http://www.uio.no/studier/emner/matnat/ifi/INF-MAT5340/v05/undervisningsmateriale/

    """
    n = t.shape[0]

    # the length of the base interval spanned by internal knots & the number
    # of subintervas between these internal knots
    delta = t[n - k - 1] - t[k]
    nrint = n - 2*k - 1

    matr = np.empty((nrint-1, k+2), dtype=float)
    for jj in range(nrint-1):
        j = jj + k + 1
        for ii in range(k+2):
            i = jj + ii
            matr[jj, ii] = (t[i + k + 1] - t[i]) / prodd(t, i, j, k)
        # XXX: debug
        # row = [(t[i + k + 1] - t[i]) / prodd(t, i, j, k) for i in range(j-k-1, j+1)]
        # assert (matr[j-k-1, :] == row).all()

    # follow FITPACK
    matr *= (delta/ nrint)**k

    # make it packed
    offset = np.array([i for i in range(nrint-1)])
    nc = n - k - 1
    return PackedMatrix(matr, offset, nc)


def disc_naive(t, k):
    """Straitforward way to compute the discontinuity matrix. For testing ONLY.

    This routine returns a dense matrix, while `disc` returns a packed one.
    """
    n = t.shape[0]

    delta = t[n - k - 1] - t[k]
    nrint = n - 2*k - 1

    ti = t[k+1:n-k-1]   # internal knots
    tii = np.repeat(ti, 2)
    tii[::2] += 1e-10
    tii[1::2] -= 1e-10
    m = BSpline.design_matrix(tii, t, k, nu=k).todense()

    matr = np.empty((nrint-1, m.shape[1]), dtype=float)
    for i in range(0, m.shape[0], 2):
        matr[i//2, :] = m[i, :] - m[i+1, :]

    matr *= (delta/nrint)**k / math.factorial(k)
    return matr


def _lsq_solve_a(x, y, t, k, w):
    """Solve for the LSQ spline coeffs given x, y and knots.
    """
    assert y.ndim == 1

    y = y[:, None]
    R, qTy, c =  _lsq_solve_qr(x, y, t, k, w)
    c = c.reshape((c.shape[0],) + y.shape[1:])
    return R, qTy, c


class F_dense:
    """ The r.h.s. of ``f(p) = s``. Uses full matrices, so is for tests only.
    """
    def __init__(self, x, y, t, k, s, w=None):
        self.x = x
        self.y = y
        self.t = t
        self.k = k
        self.w = np.ones_like(y) if w is None else w
        assert self.w.ndim == 1

        # lhs
        a_csr = BSpline.design_matrix(x, t, k)
        self.a_w = (a_csr * self.w[:, None]).tocsr()
        self.b = disc(t, k)

        self.a_dense = (a_csr * self.w[:, None]).todense()
        self.b_dense = disc(t, k).todense()

        # rhs
        assert y.ndim == 1
        yy = y * self.w
        self.yy = np.r_[yy, np.zeros(self.b.shape[0])]    # XXX: weights ignored


  #      breakpoint()

        self.s = s

    def __call__(self, p):
        ab = np.vstack((self.a_dense, self.b_dense / p))

        # LSQ solution of ab @ c = yy
        from scipy.linalg import qr, solve
        q, r = qr(ab, mode='economic')

        qy = q.T @ self.yy

        nc = r.shape[1]
        c = solve(r[:nc, :nc], qy[:nc])

        spl = BSpline(self.t, c, self.k)
        fp = np.sum(self.w**2 * (spl(self.x) - self.y)**2)

        self.spl = spl   # store it

        print(">>>> dense,", p, fp)

        ### replicate, packed ###
        R, Y, _ = _lsq_solve_a(self.x, self.y, self.t, self.k, self.w)

        # combine R & disc matrix
        # m = self.x.shape[0]
        nz = R.a.shape[1]
        assert nz == self.k+1

        AA = np.zeros((nc + self.b.shape[0], self.k+2), dtype=float)
        AA[:nc, :R.a.shape[1]] = R.a
        AA[nc:, :] = self.b.a / p
        offs = np.r_[R.offset, self.b.offset]
        AB = PackedMatrix(AA, offs, nc)

        YY = np.r_[Y[:nc], np.zeros((self.b.shape[0], Y.shape[1]))]

        # solve for the coefficients
        RR, QY = _qr_reduce(AB, YY)
        cc = fpback(RR, QY[:nc])
        cc = cc.reshape((nc,) + self.yy.shape[1:])

        from numpy.testing import assert_allclose


        assert_allclose(cc, c, atol=1e-15)        

        s_spl = BSpline(self.t, cc, self.k)

        from numpy.testing import assert_allclose
        assert_allclose(s_spl.c, spl.c, atol=1e-14)

        return fp - self.s


class F:
    """ The r.h.s. of ``f(p) = s``.

    Given scalar `p`, we solve the system of equations in the LSQ sense:

        | A     |  @ | c | = | y |
        | B / p |    | 0 |   | 0 |

    where `A` is the matrix of b-splines and `b` is the discontinuity matrix
    (the jumps of the k-th derivatives of b-spline basis elements at knots).

    Since we do that repeatedly while minimizing over `p`, we QR-factorize
    `A` only once and update the QR factorization only of the `B` rows of the
    augmented matrix |A, B/p|.

    The system of equations is Eq. (15) Ref. [1]_, the strategy and implementation
    follows that of FITPACK, see specific links below.

    References
    ----------
    [1] P. Dierckx, Algorithms for Smoothing Data with Periodic and Parametric Splines,
        COMPUTER GRAPHICS AND IMAGE PROCESSING vol. 20, pp 171-184 (1982.)
        https://doi.org/10.1016/0146-664X(82)90043-0

    """
    def __init__(self, x, y, t, k, s, w=None, *, R=None, Y=None):
        self.x = x
        self.y = y
        self.t = t
        self.k = k
        w = np.ones_like(y) if w is None else w
        assert w.ndim == 1
        assert y.ndim == 1
        self.w = w

        self.s = s

        # ### precompute what we can ###

        # https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpcurf.f#L250
        # c  evaluate the discontinuity jump of the kth derivative of the
        # c  b-splines at the knots t(l),l=k+2,...n-k-1 and store in b.
        b = disc(t, k)

        # the QR factorization of the data matrix, if not provided
        # NB: otherwise, must be consistent with x,y & s, but this is not checked
        if R is None and Y is None:
            R, Y, _ = _lsq_solve_a(x, y, t, k, w)

        # prepare to combine R and the discontinuity matrix (AB); also r.h.s. (YY)
        # https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpcurf.f#L269
        # c  the rows of matrix b with weight 1/p are rotated into the
        # c  triangularised observation matrix a which is stored in g.
        nc, nz = R.a.shape
        assert nz == k + 1

        # r.h.s. of the augmented system
        z = np.zeros((b.shape[0], Y.shape[1]), dtype=float)
        self.YY = np.r_[Y[:nc], z]

        # l.h.s. of the augmented system
        AA = np.zeros((nc + b.shape[0], self.k+2), dtype=float)
        AA[:nc, :nz] = R.a
        # AA[nc:, :] = b.a / p  # done in __call__(self, p)
        offs = np.r_[R.offset, b.offset]
        self.AB = PackedMatrix(AA, offs, nc)

        self.nc = nc
        self.b = b

    def __call__(self, p):
        nc = self.nc

        # https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpcurf.f#L279
        # c  the row of matrix b is rotated into triangle by givens transformation
        self.AB.a[nc:, :] = self.b.a / p
        RR, QY = _qr_reduce(self.AB, self.YY, startrow=nc)

        # solve for the coefficients
        c = fpback(RR, QY[:nc])
        c = c.reshape((nc,) + self.y.shape[1:])

        spl = BSpline(self.t, c, self.k)
        fp = np.sum(self.w**2 * (spl(self.x) - self.y)**2)

        self.spl = spl   # store it

        return fp - self.s


def fprati(p1, f1, p2, f2, p3, f3):
    """The root of r(p) = (u*p + v) / (p + w) given three points and values,
    (p1, f2), (p2, f2) and (p3, f3).

    The FITPACK analog adjusts the bounds, and we do not
    https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fprati.f

    NB: FITPACK uses p < 0 to encode p=infinity. We just use the infinity itself.
    Since the bracket is ``p1 <= p2 <= p3``, ``p3`` can be infinite (in fact,
    this is what the minimizer starts with, ``p3=inf``).
    """
    h1 = f1 * (f2 - f3)
    h2 = f2 * (f3 - f1)
    h3 = f3 * (f1 - f2)
    if p3 == np.inf:
        return -(p2*h1 + p1*h2) / h3
    return -(p1*p2*h3 + p2*p3*h1 + p1*p3*h2) / (p1*h1 + p2*h2 + p3*h3)


class Bunch:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


_iermesg = {
2: """error. a theoretically impossible result was found during
the iteration process for finding a smoothing spline with
fp = s. probably causes : s too small.
there is an approximation returned but the corresponding
weighted sum of squared residuals does not satisfy the
condition abs(fp-s)/s < tol.
""",
3: """error. the maximal number of iterations maxit (set to 20
by the program) allowed for finding a smoothing spline
with fp=s has been reached. probably causes : s too small
there is an approximation returned but the corresponding
weighted sum of squared residuals does not satisfy the
condition abs(fp-s)/s < tol.
"""
}


def root_rati(f, p0, bracket, acc):
    """Solve `f(p) = 0` using a rational function approximation.

    https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpcurf.f#L229
    """
    (p1, f1), (p3, f3)  = bracket
    p = p0

    # Magic values from
    # https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpcurf.f#L27
    con1 = 0.1
    con9 = 0.9
    con4 = 0.04

    for it in range(MAXIT):
        fp = f(p)
        #print(f"{it = },   {p = }")

        # c  test whether the approximation sp(x) is an acceptable solution.
        if abs(fp) < acc:
            ier, converged = 0, True
            break
        # c  carry out one more step of the iteration process.
        p2, f2 = p, f(p)

        if f2 - f3 < acc:
            # c  our initial choice of p is too large.
            p3 = p2
            f3 = f2
            p = p*con4
            if p <= p1:
                 p = p1*con9 + p2*con1
            continue

        if f1 - f2 < acc:
            # c  our initial choice of p is too small
            p1 = p2
            f1 = f2
            p = p/con4
            if p3 != np.inf and p <= p3:
                 p = p2*con1 + p3*con9
            continue

        # c  test whether the iteration process proceeds as theoretically expected.
        # [f(p) should be monotonically decreasing]
        if f1 <= f2 or f2 <= f3:
            ier, converged = 2, False
            break

        p = fprati(p1, f1, p2, f2, p3, f3)

        # c  adjust the value of p1,f1,p3 and f3 such that f1 > 0 and f3 < 0.
        if f2 < 0:
            p3, f3 = p2, f2
        else:
            p1, f1 = p2, f2

    else:
        # not converged in MAXIT iterations
        ier, converged = 3, False

    if ier != 0:
        warnings.warn(RuntimeWarning(_iermesg[ier]), stacklevel=2)

    return Bunch(converged=converged, root=p, iterations=it, ier=ier)


def make_splrep(x, y, *, t=None, w=None, k=3, s=0, xb=None, xe=None, nest=None):
    """splrep replacement
    """
    if s == 0:
        if t is not None or w is not None or nest is not None:
            raise ValueError("s==0 is for interpolation only")
        return make_interp_spline(x, y, k=k)

    x, y, w, k, s, xb, xe = _validate_inputs(x, y, w, k, s, xb, xe)

    acc = s * TOL
    m = x.size    # the number of data points

    if nest is None:
        # the max number of knots. This is set in _fitpack_impl.py line 274
        # and fitpack.pyf line 198
        nest = max(m + k + 1, 2*k + 3)
    else:
        if nest < 2*(k+1):
            raise ValueError(f"`nest` too small: {nest = } < 2*(k+1) = {2*(k+1)}.")    

    if t is None:
        t = list(generate_knots(x, y, w=w, k=k, s=s, xb=xb, xe=xe, nest=nest))[-1]
    else:
        fpcheck(x, t, k)

    if t.shape[0] == 2 * (k + 1):
        # nothing to optimize
        _, _, c = _lsq_solve_a(x, y, t, k, w)
        c = c.squeeze()  # was (nc, 1)
        return BSpline.construct_fast(t, c, k)

    ### solve ###

    # c  initial value for p.
    # https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpcurf.f#L253
    R, Y, _ = _lsq_solve_a(x, y, t, k=k, w=w)
    nc = t.shape[0] -k -1
    p = nc / R.a[:, 0].sum()

    # ### bespoke solver ####
    # initial conditions
    # f(p=inf) : LSQ spline with knots t   (XXX: resuse R, c)
    residuals = _get_residuals(x, y, t, k, w=w)
    fp = residuals.sum()
    fpinf = fp - s

    # f(p=0): LSQ spline without internal knots
    residuals = _get_residuals(x, y, np.array([xb]*(k+1) + [xe]*(k+1)), k, w)
    fp0 = residuals.sum()
    fp0 = fp0 - s

    # solve
    bracket = (0, fp0), (np.inf, fpinf)
    f = F(x, y, t, k=k, s=s, w=w, R=R, Y=Y)
    _ = root_rati(f, p, bracket, acc)

    # solve ALTERNATIVE
 #   f = F(x, y, t, k=k, s=s, w=w, R=R, Y=Y)
 #   from scipy.optimize import root_scalar
 #   res_ = root_scalar(f, x0=p, rtol=acc)

 #   assert res_.converged
    # f.spl is the spline corresponding to the found `p` value
    return f.spl
