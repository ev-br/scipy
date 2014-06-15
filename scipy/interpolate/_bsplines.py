from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.linalg import solve_banded, inv
from . import _bspl

__all__ = ["BSpline", "make_interp_spline", "make_interp_periodic_spline"]


class BSpline(object):
    r"""Univariate spline in the B-spline basis.

    .. math::

        S(x) = \sum_{j=0}^{n-1} c_j  B_{j, k; t}(x)

    where :math:`B_{j, k; t}` are B-spline basis functions of the order `k`
    and knots `t`.

    Parameters
    ----------
    t : ndarray, shape (n+k+1,)
        knots
    c : ndarray, shape (>=n, ...)
        spline coefficients
    k : int
        B-spline order
    extrapolate : bool, optional
        whether to extrapolate beyond the basic interval, ``t[k] .. t[n]``,
        or to return nans. Default is True.

    Methods
    -------
    __call__
    basis_element

    Notes
    -----
    B-spline basis elements are defined via

    .. math::

        S(x) = \sum_{j=0}^{n-1} c_{j} B_{j, k; t}(x)
        \\
        B_{i, 0}(x) = 1, \textrm{if $t_i \le x < t_{i+1}$, otherwise $0$,}
        \\
        B_{i, k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)
                 + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)

    Or, in terms of Python code:

    .. code-block::

    def B(x, k, i, t):
        if k == 0:
           return 1.0 if t[i] <= x < t[i+1] else 0.0
        if t[i+k] == t[i]:
           c1 = 0.0
        else:
           c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
        if t[i+k+1] == t[i+1]:
           c2 = 0.0
        else:
           c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
        return c1 + c2

    def bspline(x, t, c, k):
        n = len(t) - (k+1)
        assert (n >= k+1) and (len(c) >= n)
        return sum(c[i] * B(x, k, i, t) for i in range(n))

    Note that this is an inefficient (if straightforward) way to
    evaluate B-splines --- this spline class does it in a more
    efficient way.

    >>> spl = BSpline(t=[0, 1, 2, 3, 4, 5], c=[2, 3], k=2)
    >>> spl(2.1)
    array(2.58)
    >>> bspline(2.1, spl.t, spl.c, spl.k)
    2.58

    ***Implementation details***
    * At least ``k+1`` coefficients are required for a spline of degree `k`,
      so that ``n >= k+1``. Additional coefficients, ``c[j]`` with
      ``j > n``, are ignored.
    * B-spline basis elements of degree `k` form a partition of unity on the
    __base interval__, ``t[k] <= x <= t[n] ``. The behavior for ``x > t[n]``
    and ``x < t[k]`` is controlled by the `extrapolate` parameter.
    * The base interval is closed, so that the spline is right-continuous
    at ``x == t[n]``.

    References
    ----------
    .. [1]_ Tom Lyche and Knut Morken, Spline methods,
        http://www.uio.no/studier/emner/matnat/ifi/INF-MAT5340/v05/undervisningsmateriale/
    .. [2]_ Carl de Boor, A practical guide to splines, Springer, 2001.

    """
    def __init__(self, t, c, k, extrapolate=True):
        super(BSpline, self).__init__()

        self.k = int(k)
        self.c = np.asarray(c)
        self.t = np.ascontiguousarray(t, dtype=np.float64)
        self.extrapolate = bool(extrapolate)

        n = self.t.shape[0] - self.k - 1

        if k < 0:
            raise ValueError("Spline order cannot be negative.")
        if int(k) != k:
            raise ValueError("Spline order must be integer.")
        if self.t.ndim != 1:
            raise ValueError("Knot vector must be one-dimensional.")
        if n < self.k + 1:
            raise ValueError("Need at least %d knots for degree %d" %
                    (2*k + 2, k))
        if (np.diff(self.t) < 0).any():
            raise ValueError("Knots must be in a non-decreasing order.")
        if len(np.unique(self.t[k:n+1])) < 2:
            raise ValueError("Need at least two internal knots.")
        if not np.isfinite(self.t).all():
            raise ValueError("Knots should not have nans or infs.")
        if self.c.ndim < 1:
            raise ValueError("Coefficients must be at least 1-dimensional.")
        if self.c.shape[0] < n:
            raise ValueError("Knots, coefficients and degree are inconsistent.")

        dt = self._get_dtype(self.c.dtype)
        self.c = np.ascontiguousarray(self.c, dtype=dt)

    def _get_dtype(self, dtype):
        if np.issubdtype(dtype, np.complexfloating):
            return np.complex_
        else:
            return np.float_

    @classmethod
    def basis_element(cls, t, extrapolate=True):
        """Return a B-spline basis element ``B(x | t[0], ..., t[k+1])``.
        """
        k = len(t) - 2
        t = np.r_[(t[0]-1,) * k, t, (t[-1]+1,) * k]
        c = np.zeros_like(t)
        c[k] = 1.
        return cls(t, c, k, extrapolate)

    def __call__(self, x, nu=0, extrapolate=None):
        """
        Evaluate a spline function.

        Parameters
        ----------
        x : array_like
            points to evaluate the spline at.
        nu: int, optional
            derivative to evaluate (default = 0).
        extrapolate : bool, optional
            whether to extrapolate based on the first and last intervals
            or return nans. Default is `self.extrapolate`.

        Returns
        -------
        y : array_like
            Shape is determined by replacing the interpolation axis
            in the coefficient array with the shape of `x`.

        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = np.asarray(x)
        x_shape = x.shape
        x = np.ascontiguousarray(x.ravel(), dtype=np.float_)
        out = np.empty((len(x), int(np.prod(self.c.shape[1:]))),
                dtype=self.c.dtype)
        self._ensure_c_contiguous()
        self._evaluate(x, nu, extrapolate, out)
        return out.reshape(x_shape + self.c.shape[1:])

    def _evaluate(self, xp, nu, extrapolate, out):
        _bspl.evaluate_spline(self.t, self.c.reshape(self.c.shape[0], -1),
                self.k, xp, nu, extrapolate, out)

    def _ensure_c_contiguous(self):
        """
        c and t may be modified by the user. The Cython code expects
        that they are C contiguous.

        """
        if not self.t.flags.c_contiguous:
            self.x = self.x.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()


#################################
#  Interpolating spline helpers #
#################################

def _check_xytk(x, y, t, k):
    if x.ndim != 1 or np.any(x[1:] - x[:-1] <= 0):
        raise ValueError("Expect x to be a 1-D sorted array_like.")
    if x.shape[0] < k+1:
        raise("Need more x points.")
    if k <= 0:
        raise ValueError("Expect positive k.")
    if t.ndim != 1 or np.any(t[1:] - t[:-1] < 0):
        raise ValueError("Expect t to be a 1-D sorted array_like.")
    if x.size != y.shape[0]:
        raise ValueError('x & y are incompatible.')
    if t.size < x.size + k + 1:
        raise ValueError('Got %d knots, need at least %d.' % (t.size, x.size + k + 1))
    if np.any((x < t[k]) | (x > t[-k])):
        raise ValueError('Out of bounds w/ x = %s.' % x)


def make_interp_spline(x, y, t, k, deriv_l=None, deriv_r=None,
                       check_finite=True):
    """Compute the (coefficients of) interpolating B-spline.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Abscissas.
    y : ndarray, shape (n, ...)
        Ordinatas.
    t : ndarray, shape (nt + k + 1,)
        Knots.
        The number of knots need to agree with the number of datapoints and
        the number of derivatives at the edges. Specifically, ``nt - n`` must
        equal ``len(deriv_l) + len(deriv_r)``.
    k : int
        B-spline degree
    deriv_l : iterable of pairs (int, float) or None
        Derivatives known at ``x[0]`` (order, value)
        Default is None.
    deriv_r : iterable of pairs (int, float) or None
        Derivatives known at ``x[-1]``.
        Default is None.
    check_finite : bool
        This forwards directly to the lin. algebra routines [solve_banded].
        Default is True.

    Returns
    -------
    c : ndarray, shape(n, ...)
        The coefficients of the B-spline of degree `k` with knots `t`, which
        interpolates `x` and `y`.

    Examples
    --------
    >>> c = make_interp_spline(x, y, t, k)
    >>> b = BSpline(t, c, k)
    >>> assert b(x) == y

    """
    x, y, t = map(np.asarray, (x, y, t))
    k = int(k)

    _check_xytk(x, y, t, k)

    # Here : deriv_l, r = [(nu, value), ...]
    if deriv_l is not None:
        deriv_l_ords, deriv_l_vals = zip(*deriv_l)
    else:
        deriv_l_ords, deriv_l_vals = [], []
    deriv_l_ords, deriv_l_vals = map(np.atleast_1d, (deriv_l_ords, deriv_l_vals))
    nleft = deriv_l_ords.shape[0]

    if deriv_r is not None:
        deriv_r_ords, deriv_r_vals = zip(*deriv_r)
    else:
        deriv_r_ords, deriv_r_vals = [], []
    deriv_r_ords, deriv_r_vals = map(np.atleast_1d, (deriv_r_ords, deriv_r_vals))
    nright = deriv_r_ords.shape[0]

    # have `n` conditions for `nt` coefficients; need nt-n derivatives
    n = x.size
    nt = t.size - k - 1

    if nt - n != nleft + nright:
        raise ValueError("number of derivatives at boundaries.")

    # set up the LHS: the collocation matrix + derivatives @boundaries
    Ab, kl_ku = _bspl._colloc(x, t, k, offset=nleft)
    if nleft > 0:
        _bspl._handle_lhs_derivatives(t, k, x[0], Ab, kl_ku, deriv_l_ords)
    if nright > 0:
        _bspl._handle_lhs_derivatives(t, k, x[-1], Ab, kl_ku, deriv_r_ords,
                                offset=nt-nright)

    # RHS
    extradim = int(np.prod(y.shape[1:]))
    rhs = np.empty((nt, extradim))
    if nleft > 0:
        rhs[:nleft] = deriv_l_vals.reshape(-1, extradim)
    rhs[nleft:nt - nright] = y.reshape(-1, extradim)
    if nright > 0:
        rhs[nt - nright:] = deriv_r_vals.reshape(-1, extradim)

    c = solve_banded(kl_ku, Ab, rhs, overwrite_ab=True,
                     overwrite_b=True, check_finite=check_finite)
    return c.reshape((nt,) + y.shape[1:])


def make_interp_periodic_spline(x, y, t, k, alpha=1., check_finite=True):
    """Compute the (coefficients of) interpolating B-spline. 
 
    Contrains the values of the first k-1 derivatives at ``x[0]..x[-1]`` via

    ..math::

        y^{j}(x_0) = alpha * y^{j}(x_1)

    where y(x) is the B-spline interpolant, and ``j = 1..k-1``

    Parameters
    ----------
    x : ndarray, shape (n,)
        Abscissas.
    y : ndarray, shape (n, ...)
        Ordinatas.
    t : ndarray, shape (nt + k + 1,)
        Knots.
        The number of knots need to agree with the number of datapoints:
        ``nt - n`` must be equal to `k`-1.
    k : int
        B-spline degree
    alpha : float
        Periodicity factor: alpha = 1 means periodic boundary condition, 
        alpha = -1 means antiperiodic etc.
        Default is 1. 
    check_finite : bool, optional
        This forwards directly to the lin. algebra routines [solve_banded].
        Default is True.

    Returns
    -------
    c : ndarray, shape(n, ...)
        The coefficients of the B-spline of degree `k` with knots `t`, which
        interpolates `x` and `y`.

    Examples
    --------
    >>> c = make_interp_periodic_spline(x, y, t, k)
    >>> b = BSpline(t, c, k)
    >>> assert b(x) == y

    Notes
    -----

    To enforce periodicity we require that k-1 derivatives match at x[0] and x[-1].

    This leads to the *almost* banded linear system for the spline coefficients:
    the lhs matrix has extra elements in the top right corner. To handle this, we
    write the collocation matrix as

        A + u @ v.T,

    where A is banded, and use the Woodbury formula to handle the u @ v.T
    correction (see eg NR book, Chap 2.7.3.)

    Specifically, use the fact that

        inv(A + u @ v.T) = inv(A) - inv(A) @ u @ inv(Z) v.T @ inv(A)

    where Z = 1 + v.T @ inv(A) @ u

    """
    x, y, t = map(np.asarray, (x, y, t))
    k = int(k)
    _check_xytk(x, y, t, k)

    # have `n` conditions for `nt` coefficients; need nt-n derivatives
    n = x.size
    nt = t.size - k - 1

    if nt - n != k - 1:
        raise ValueError("knots not consistent with data.")

    # LHS: the collocation matrix + derivatives @edges
    Ab, (kl, ku) = _bspl._colloc(x, t, k, offset=k-1)

    # derivatives @ x[0] fit into the banded part of the matrix:
    _bspl._handle_lhs_derivatives(t, k, x[0], Ab, (kl, ku),
                            deriv_ords=np.arange(1, k))

    # derivatives @ x[-1]: These form a Woodbury formula correction matrix `U`,
    # while V is simply a rectangular unit matrix, whose job is to only bring
    # the elements of `U` into proper parts of the lhs matrix:
    # explicitly, its transpose is vT = np.eye(k+1, nt, k=nt-k-1).
    # We don't actually allocate it since we can get the results of
    # multiplications with vT by just slicing, see below.
    u = np.zeros((nt, k+1), dtype=np.float_)
    _bspl._fill_woodbury_U(t, k, x[-1], alpha, u)

    # RHS
    extradim = int(np.prod(y.shape[1:]))
    rhs = np.empty((nt, extradim))
    rhs[:k-1] = 0.
    rhs[k-1:] = y.reshape(-1, extradim)

    # Solve the linear system:
    # (2.7.17) + (2.7.19)
    z = solve_banded((kl, ku), Ab, u, check_finite=check_finite)
    h = np.eye(k+1, dtype=np.float_) + z[-k-1:, -k-1:]  # was np.dot(vT, z)
    h = inv(h)

    yy = solve_banded((kl, ku), Ab, rhs, check_finite=check_finite)   # (2.7.20)
    #c = yy - np.dot(z, np.dot(h, np.dot(vT, yy)))
    c = yy - np.dot(z, np.dot(h, yy[-k-1:]))   # was np.dot(vT, yy)

    return c.reshape((nt,) + y.shape[1:])
