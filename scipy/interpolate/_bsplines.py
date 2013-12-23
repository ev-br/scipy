from __future__ import division, print_function, absolute_import

import numpy as np
from . import _ppoly

__all__ = ["BSpline"]


class BSpline(object):
    r"""Univariate spline in the B-spline basis.

        S = sum(c_j * B_{j, k; t} for j in range(N))

    where ``B_{j, k; t}`` are the B-spline basis functions of the order `k`
    and knots `t`.

    Parameters
    ----------
    t : ndarray, shape (n+k+1,)
        knots
    c : ndarray, shape (n, ...)
        spline coefficients
    k : int
        B-spline order
    extrapolate : bool, optional
        whether to extrapolate based on the first and last intervals
        or return nans. Default is `False`.

    Methods
    -------
    __call__
    basis_element
    from_fitpack_tck
    get_fitpack_tck
    knots

    Notes
    -----

    B-spline basis elements are defined via

    .. math::

        s(x) = \sum_{j=0}^N c_{j} B^k_{j}(x)
        \\
        B^0_i(x) = 1, \textrm{if $t_i \le x < t_{i+1}$, otherwise $0$,}
        \\
        B^k_i(x) = \frac{x - t_i}{t_{i+k} - t_i} B^{k-1}_i(x)
                 + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B^{k-1}_{i+1}(x)

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
        assert len(t) == len(c) + k + 1
        return sum(ci * B(x, k, i, t) for (i, ci) in enumerate(c))

    Note that this is an inefficient (if straightforward) way to
    evaluate B-splines --- this spline class does it in a more
    efficient way:

    >>> spl = BSpline(t=[0, 1, 2, 3, 4], c=[2, 3], k=2)
    >>> spl(2.1)
    array(2.58)
    >>> bspline(2.1, spl.knots, spl.c, spl.k)
    2.58

    Implementation details
    ----------------------
    To simplify the implementation, knots are augmented so that an instance 
    attribute `t` holds both internal knots (accessible via `knots` 
    attribute), and `2k` 'boundary' knots:

    >>> spl = BSpline(t=[0, 1, 2, 3, 4], c=[2, 3], k=2)
    >>> assert_equal(spl.knots, [0, 1, 2, 3, 4])
    >>> assert_equal(len(spl.t), len(spl.knots) + 2*spl.k)

    References
    ----------
    [1]_ Tom Lyche and Knut Morken, Spline Methods,
        http://www.uio.no/studier/emner/matnat/ifi/INF-MAT5340/v05/undervisningsmateriale/ 

    """
    def __init__(self, t, c, k, extrapolate=False):
        super(BSpline, self).__init__()

        self.k = int(k)
        self.c = np.asarray(c)
        self.t = np.asarray(t, dtype=np.float64)
        self.extrapolate = bool(extrapolate)

        if k < 0:
            raise ValueError("Spline order cannot be negative.")
        if self.t.ndim != 1:
            raise ValueError("Knot vector must be one-dimensional.")
        if self.t.size < 2:
            raise ValueError("At least two knots are required.")
        if (np.diff(self.t) < 0).any():
            raise ValueError("Knots must be in a non-decreasing order.")
        if not np.isfinite(self.t).all():
            raise ValueError("Knots should not have nans or infs.")
        if self.c.ndim < 1:
            raise ValueError("Coefficients must be at least 1-dimensional.")
        if self.t.shape[0] != self.c.shape[0] + self.k + 1:
            raise ValueError("Knots, coefficients and degree are inconsistent.")

        # augment the knots
        self.t = np.r_[(t[0]-1,)*self.k, self.t, (t[-1]+1,)*self.k]
        self.t = np.ascontiguousarray(self.t, dtype=np.float64)

        dt = self._get_dtype(self.c.dtype)
        self.c = np.ascontiguousarray(self.c, dtype=dt)

    def _get_dtype(self, dtype):
        if np.issubdtype(dtype, np.complexfloating) \
               or np.issubdtype(self.c.dtype, np.complexfloating):
            return np.complex_
        else:
            return np.float_

    @classmethod
    def basis_element(cls, t, extrapolate=False):
        """Return a B-spline basis element ``B(x|t[0], ..., t[len(t)+2])``.

        Is equivalent to ``BSpline(t, c=[1.], k=len(t)-2)``.

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

    def get_fitpack_tck(self):
        """Return (tck) compatible with fitpack's `splev`.

        Example
        -------
        >>> xx = np.linspace(0, 4, 10)
        >>> b = BSpline(t=[0, 1, 2, 3, 4], c=[2., 3.], k=2)
        >>> np.allclose(b(xx), splev(xx, b.get_fitpack_tck()))
        True

        """
        # `splev` (more specifically, splev.f) expects
        # len(t) == len(c), and starts using coefficients from the first one
        # Hence:
        # 1) append k+1 zeros to c
        # 2) augment the result with k zeros from left and k from right
        #    to match the length of self.t
        # Ref: http://www.netlib.org/dierckx/
        assert self.c.ndim == 1
        cc = np.r_[(0.,)*self.k, self.c, (0.,)*(2*self.k+1)]
        return self.t, cc, self.k

    @classmethod
    def from_fitpack_tck(cls, tck, extrapolate=False):
        """Construct a BSpline from fitpack-style tck.

        Example
        -------
        >>> x, y = np.random.random((2, 30))
        >>> x.sort()
        >>> tck = splrep(x, y)
        >>> b = BSpline.from_fitpack_tck(tck)
        >>> xx = np.linspace(-0.5, 0.5, 100)
        >>> np.allclose(b(xx, extrapolate=True), splev(xx, tck))
        True

        """
        t, c, k = tck
        return cls(t=np.r_[t, (t[-1],)*(k+1)], c=c, k=k, extrapolate=extrapolate)

    @property
    def knots(self):
        """A (read-only) getter for the internal knots.

        >>> b = BSpline.basis_element(t=[0, 1, 1], k=1)
        >>> assert_equal(b.knots, [0, 1, 1])
        True

        """
        return self.t[self.k:-self.k]
