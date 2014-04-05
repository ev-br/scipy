import numpy as np
from numpy.testing import (run_module_suite, TestCase, assert_equal,
        assert_allclose, assert_raises, assert_)
from numpy.testing.decorators import skipif

from scipy.interpolate import BSpline, splev, splrep, BPoly, PPoly

class TestBSpline(TestCase):

    def test_ctor(self):
        # knots should be an ordered 1D array of finite real numbers
        assert_raises(TypeError, BSpline, **dict(t=[1, 1.j], c=[1.], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[1, np.nan], c=[1.], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[1, np.inf], c=[1.], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[1, -1], c=[1.], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[[1], [1]], c=[1.], k=0))

        # for n+k+1 knots and degree k need at least n coefficients
        assert_raises(ValueError, BSpline, **dict(t=[0, 1, 2], c=[1], k=0))
        assert_raises(ValueError, BSpline,
                **dict(t=[0, 1, 2, 3, 4], c=[1., 1.], k=2))

        # non-integer orders
        assert_raises(ValueError, BSpline,
                **dict(t=[0., 0., 1., 2., 3., 4.], c=[1., 1., 1.], k="cubic"))
        assert_raises(ValueError, BSpline,
                **dict(t=[0., 0., 1., 2., 3., 4.], c=[1., 1., 1.], k=2.5))

        # basic inteval cannot have measure zero (here: [1..1])
        assert_raises(ValueError, BSpline,
                **dict(t=[0., 0, 1, 1, 2, 3], c=[1., 1, 1], k=2))

        # tck vs self.tck
        b, t, c, k = self._make_random_spline()
        assert_allclose(t, b.t)
        assert_allclose(c, b.c)
        assert_equal(k, b.k)

    def _make_random_spline(self, n=35, k=3):
        np.random.seed(123)
        t = np.sort(np.random.random(n+k+1))
        c = np.random.random(n)
        return BSpline(t, c, k), t, c, k

    def _order_0(self):
        xx = np.linspace(0, 1, 10)

        b = BSpline(t=[0, 1], c=[3.], k=0)
        assert_allclose(b(xx), 3)

        b = BSpline(t=[0, 0.35, 1], c=[3, 4], k=0)
        assert_allclose(b(xx), np.where(xx< 0.35, 3, 4))

    def test_order_1(self):
        t = [0, 1, 2, 3, 4]
        c = [1, 2, 3]
        k = 1
        b = BSpline(t, c, k)

        x = np.linspace(1, 3, 50)
        assert_allclose(c[0]*B_012(x) + c[1]*B_012(x-1) + c[2]*B_012(x-2),
                        b(x))

    @skipif(True)
    def test_order_2(self):
        xx = np.linspace(0, 3, 20, endpoint=False)
        conds = [xx < 1, (xx > 1) & (xx < 2), xx > 2]
        funcs = [lambda x: x*x/2., 
                 lambda x: 3./4 - (x-3./2)**2, 
                 lambda x: (3.-x)**2 / 2]
        pieces = np.piecewise(xx, conds, funcs)
        b = BSpline.basis_element(t=[0, 1, 2, 3])
        assert_allclose(b(xx), pieces)

        b = BSpline.basis_element(t=[0, 1, 1, 2])
        xx = np.linspace(0, 2, 10, endpoint=False)
        assert_allclose(b(xx),
                np.where(xx < 1, xx*xx, (2.-xx)**2))

    def test_bernstein(self):
        # a special knot vector: Bernstein polynomials
        k = 3
        t = np.asarray([0]*(k+1) + [1]*(k+1))
        c = np.asarray([1., 2., 3., 4.])
        bp = BPoly(c.reshape(-1, 1), [0, 1])
        bspl = BSpline(t, c, k)

        xx = np.linspace(-1., 2., 100)
        assert_allclose(bp(xx, extrapolate=True),
                        bspl(xx, extrapolate=True))

    def test_rndm_naive_eval(self):
        b, t, c, k = self._make_random_spline()
        xx = np.linspace(t[k], t[-k-1], 50)
        y_b = b(xx)

        y_n = [_naive_eval(x, t, c, k) for x in xx]
        assert_allclose(y_b, y_n)

        y_n2 = [_naive_eval_2(x, t, c, k) for x in xx]
        assert_allclose(y_b, y_n2)

    def test_rndm_naive_eval_multiple_knots(self):
        b, t, c, k = self._make_random_spline()
        b.t[9] = b.t[8]
        b.t[15:17] = b.t[15]

        xx = np.linspace(b.t[k], b.t[-k-1], 50)
        y_b = b(xx)
##        y_b = splev(xx, (t, c, k), ext=2)

        y_n = [_naive_eval(x, b.t, b.c, b.k) for x in xx]
        assert_allclose(y_b, y_n)

    def test_rndm_splev(self):
        b, t, c, k = self._make_random_spline()
        xx = np.linspace(t[k], t[-k-1], 50)
        assert_allclose(b(xx), splev(xx, (t, c, k), ext=2))

    def test_rndm_splrep(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(20))
        y = np.random.random(20)

        tck = splrep(x, y)
        b = BSpline(*tck)

        xx = np.linspace(b.t[b.k], b.t[-b.k-1], 80)
        assert_allclose(b(xx), splev(xx, tck))

    def test_rndm_unity(self):
        b, t, c, k = self._make_random_spline()
        b.c = np.ones_like(b.c)
        xx = np.linspace(t[k], t[-k-1], 100)
        assert_allclose(b(xx), 1.)

    def test_vectorization(self):
        n = 22
        _, t, _, k = self._make_random_spline(n=n)
        c = np.random.random(size=(n, 6, 7))
        b = BSpline(t, c, k)
        tm, tp = t[k], t[-k-1]
        xx = tm + (tp - tm) * np.random.random((3, 4, 5))
        assert_equal(b(xx).shape, (3, 4, 5, 6, 7))

    def test_len_c(self):
        # for n+k+1 knots, only first n coefs are used.
        b, t, c, k = self._make_random_spline()
        dt = t[-1] - t[0]
        xx = np.linspace(t[0] - dt, t[-1] + dt, 50)
        mask = (xx > t[k]) & (xx < t[-k-1])

        yy = b(xx, extrapolate=True)
        for _ in range(k+5):
            b.c = np.r_[b.c, np.random.random()]
            assert_allclose(yy[mask], b(xx[mask], extrapolate=False))
            assert_allclose(yy, b(xx, extrapolate=True))

    def test_endpoints(self):
        # base interval is closed
        b, t, c, k = self._make_random_spline()
        tm, tp = t[k], t[-k-1]
        for extrap in (True, False):
            assert_allclose(b([tm, tp], extrap),
                            b([tm + 1e-10, tp - 1e-10], extrap))

    def test_continuity(self):
        # assert continuity @ internal knots
        b, t, c, k = self._make_random_spline()
        for x in t[k+1:-k-2]:
            assert_allclose(b(x - 1e-10), b(x + 1e-10))

        # repeat with multiple knots
        b.t[9] = b.t[8]
        b.t[15:17] = b.t[15]
        for x in t[k+1:-k-2]:
            assert_allclose(b(x - 1e-10), b(x + 1e-10))

    def test_extrap(self):
        b, t, c, k = self._make_random_spline()
        dt = t[-1] - t[0]
        xx = np.linspace(t[k] - dt, t[-k-1] + dt, 50)
        mask = (t[k] < xx) & (xx < t[-k-1])

        # extrap has no effect within the base interval
        assert_allclose(b(xx[mask], extrapolate=True),
                        b(xx[mask], extrapolate=False))

        # extrapolated values agree with fitpack
        assert_allclose(b(xx, extrapolate=True),
                splev(xx, (t, c, k), ext=0))

        # repeat with multiple knots
        b.t[9] = b.t[8]
        b.t[15:17] = b.t[15]
        assert_allclose(b(xx, extrapolate=True),
                splev(xx, (b.t, b.c, b.k), ext=0))

        # and with multiple boundary knots
        b.t[:k+2] = b.t[0]
        assert_allclose(b(xx, extrapolate=True),
                splev(xx, (b.t, b.c, b.k), ext=0))

    def test_ppoly(self):
        b, t, c, k = self._make_random_spline()
        bp = PPoly.from_spline((t, c, k))

        xx = np.linspace(t[0], t[-1], 100)
        assert_allclose(b(xx, extrapolate=True), bp(xx, extrapolate=True))

########################################################################

    @skipif(True)
    def test_derivative(self):
        b = BSpline.basis_element(t=[0, 1, 1, 2])
        xx = np.linspace(0, 2, 10, endpoint=False)
        assert_allclose(b(xx, nu=1),
                np.where(xx < 1, 2.*xx, 2.*(xx-2.)))
        assert_allclose(b(xx, nu=2), 2.)
        assert_allclose(b(xx, nu=3), 0.)

    @skipif(True)
    def test_derivative_2(self):
        # 3rd derivative jumps @ a triple knot
        b = BSpline.basis_element(t=[0, 1, 1, 1, 2])
        xx = np.linspace(0, 2, 10, endpoint=False)
        assert_allclose(b(xx),
                np.where(xx < 1, xx**3, (2.-xx)**3))
        assert_allclose(b(xx, nu=1),
                np.where(xx < 1, 3.*xx**2, -3.*(2.-xx)**2))
        assert_allclose(b(xx, nu=2),
                np.where(xx < 1, 6.*xx, 6.*(2.-xx)))
        assert_allclose(b(xx, nu=3),
                np.where(xx < 1, 6., -6.))

    @skipif(True)
    def test_basis_element(self):
        b, t, c, k, n = self._make_random_spline(n=1, k=3)
        b.c = np.array([1.])
        bb = BSpline.basis_element(t, k)
        xx = np.linspace(t[0], t[-1], 20, endpoint=False)
        assert_allclose(b(xx), bb(xx))

    @skipif(True)
    def test_continuity(self):
        np.random.seed(1234)
        t = np.r_[np.random.random(5),
                1.1, 1.1,
                np.random.random(5) + 1.2,
                2.3, 2.3, 2.3,
                np.random.random(5) + 2.4]
        t.sort()
        n, k = t.size - 4, 3
        c = np.random.random(n)
        b = BSpline(t, c, k)
        t_p, t_m = t[1:-1] - 1e-13, t[1:-1] + 1e-13

        # the spline is continuous
        assert_allclose(b(t_m), b(t_p), atol=1e-12, rtol=1e-12)

        # triple knot @ x=2.3: 1st derivative jumps
        delta = b(t_m, nu=1) - b(t_p, nu=1)
        mask = t[1:-1] != 2.3
        assert_allclose(delta[mask], 0, atol=1e-10, rtol=1e-10)
        assert_(not np.allclose(delta[~mask], 0))

        # double knot @ x=1.1: 2nd derivative jumps
        delta = b(t_m, nu=2) - b(t_p, nu=2)
        mask = mask & (t[1:-1] != 1.1)
        assert_allclose(delta[mask], 0., atol=1e-8, rtol=1e-8)
        assert_(not np.allclose(delta[~mask], 0.))


### stolen from pv, verbatim
def _naive_B(x, k, i, t):
    """
    Naive way to compute B-spline basis functions. Useful only for testing!
    computes B(x; t[i],..., t[i+k+1])
    """
    if k == 0:
        return 1.0 if t[i] <= x < t[i+1] else 0.0
    if t[i+k] == t[i]:
        c1 = 0.0
    else:
        c1 = (x - t[i])/(t[i+k] - t[i]) * _naive_B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
        c2 = 0.0
    else:
        c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * _naive_B(x, k-1, i+1, t)
    return (c1 + c2)


### stolen from pv, verbatim
def _naive_eval(x, t, c, k):
    """
    Naive B-spline evaluation. Useful only for testing!
    """
    if x == t[k]:
        i = k
    else:
        i = np.searchsorted(t, x) - 1
    assert t[i] <= x <= t[i+1]
    assert i >= k and i < len(t) - k
    return sum(c[i-j] * _naive_B(x, k, i-j, t) for j in range(0, k+1))

def _naive_eval_2(x, t, c, k):
    """Naive B-spline evaluation, another way."""
    n = len(t) - (k+1)
    assert n >= k+1
    assert len(c) >= n
    assert t[k] <= x <= t[n]
    return sum(c[i] * _naive_B(x, k, i, t) for i in range(n))

def B_012(x):
    """ A linear B-spline function B(x | 0, 1, 2)"""
    x = np.atleast_1d(x)
    return np.piecewise(x, [(x < 0) | (x > 2), 
                            (x >= 0) & (x < 1), 
                            (x >= 1) & (x <= 2)],
                           [lambda x: 0., lambda x: x, lambda x: 2.-x])

if __name__ == "__main__":
    run_module_suite()
