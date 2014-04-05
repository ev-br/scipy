from __future__ import division, print_function, absolute_import

import numpy as np
from . import _ppoly

__all__ = ["BSpline"]


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
        or to return nans. Default is False.

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
    def __init__(self, t, c, k, extrapolate=False):
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
                    (2*k+2, k))
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
    def basis_element(cls, t, extrapolate=False):
        """Return a B-spline basis element ``B(x|t[0], ..., t[len(t)+2])``.

        Is equivalent to ``BSpline(t, c=[1.], k=len(t)-2)``.

        # FIXME

        """
        t = np.asarray(t)
        return cls(t, c=[1.], k=t.size-2, extrapolate=extrapolate)

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
        _ppoly.evaluate_spline(self.t, self.c.reshape(self.c.shape[0], -1),
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


