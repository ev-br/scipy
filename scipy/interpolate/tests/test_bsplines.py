import numpy as np
from numpy.testing import (run_module_suite, TestCase, assert_equal,
        assert_allclose, assert_raises, assert_)

from scipy.interpolate import BSpline, splev, splrep, PPoly

class TestBSpline(TestCase):

    def test_ctor(self):
        # knots should be an ordered 1D array of finite real numbers
        assert_raises(TypeError, BSpline, **dict(t=[1, 1.j], c=[1.], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[1, np.nan], c=[1.], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[1, np.inf], c=[1.], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[1, -1], c=[1.], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[[1]], c=[1], k=0))

        # for `n+k+1` knots and order `k` need exactly `n` coefficients
        assert_raises(ValueError, BSpline, **dict(t=[0, 1, 2], c=[1], k=0))

        # non-integer orders
        assert_raises(ValueError, BSpline, **dict(t=[0., 1.], c=[1.], k='cubic'))

    def _order_0(self):
        xx = np.linspace(0, 1, 10)

        b = BSpline(t=[0, 1], c=[3.], k=0)
        assert_allclose(b(xx), 3)

        b = BSpline(t=[0, 0.35, 1], c=[3, 4], k=0)
        assert_allclose(b(xx), np.where(xx< 0.35, 3, 4))

    def test_order_1(self):
        xx = np.linspace(0, 1, 10, endpoint=False)

        b = BSpline.basis_element(t=[0, 0, 1])
        assert_allclose(b(xx), 1.-xx)

        b = BSpline.basis_element(t=[0, 1, 1])
        assert_allclose(b(xx), xx)

        b = BSpline.basis_element(t=[0, 1, 2])
        assert_allclose(b(xx), np.where(xx < 1, xx, 2. - xx))

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

    def test_derivative(self):
        b = BSpline.basis_element(t=[0, 1, 1, 2])
        xx = np.linspace(0, 2, 10, endpoint=False)
        assert_allclose(b(xx, nu=1),
                np.where(xx < 1, 2.*xx, 2.*(xx-2.)))
        assert_allclose(b(xx, nu=2), 2.)
        assert_allclose(b(xx, nu=3), 0.)

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

    def test_basis_element(self):
        b, t, c, k, n = self._make_random_spline(n=1, k=3)
        b.c = np.array([1.])
        bb = BSpline.basis_element(t, k)
        xx = np.linspace(t[0], t[-1], 20, endpoint=False)
        assert_allclose(b(xx), bb(xx))

    def _make_random_spline(self, n=35, k=3):
        np.random.seed(123)
        t = np.sort(np.random.random(n+k+1))
        c = np.random.random(n)
        return BSpline(t, c, k), t, c, k, n

    def test_rndm_unity(self):
        # B-splines from a partition of unity
        # NB: internal intervals *only*, x\in [t[k], t[n])
        b, t, c, k, n = self._make_random_spline()
        b.c = np.ones_like(b.c)
        xx = np.linspace(t[k], t[n], 100)
        assert_allclose(b(xx), np.ones_like(xx))

    def test_rndm_naive(self):
        b, t, c, k, n = self._make_random_spline()
        xx = np.linspace(b.t[k+1], b.t[n-k-1], 90)

        # pad `c` w/zeros to suppress the contributions from
        # B-splines starting at 'augmenting' knots
        cc = np.r_[[0.]*k, b.c, [0.]*k]
        naive = [_naive_eval(x, b.t, cc, b.k) for x in xx]
        assert_allclose(b(xx), naive)

    def test_rndm_naive_multiple_knots(self):
        np.random.seed(123)
        t = np.sort(np.random.random(21))*3
        t = np.r_[t[:5], (t[5],)*3,
                  t[6:9], (t[9],)*2,
                  t[10:17], (t[17],)*5,
                  t[18:]]
        n, k = t.size - 4, 3 
        c = np.random.random(n)
        b = BSpline(t, c, k)

        xx = np.linspace(b.t[k+1], b.t[n-k-1], 20)
        cc = np.r_[(0.,)*k, b.c, (0.,)*k]   # see test_rndm_naive
        naive = [_naive_eval(x, b.t, cc, b.k) for x in xx]
        assert_allclose(b(xx), naive)

    def test_augmenting(self):
        # make sure nothing depends on the values of leading& trailing knots
        b, t, c, k, n = self._make_random_spline()
        b1 = BSpline(t, c, k)
        b1.t[:k] -= 3.
        b1.t[-k:] += 3.

        xx = np.random.random(size=30) * (t[-1] - t[0]) + t[0]
        assert_allclose(b(xx), b1(xx))

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

    def test_vectorization(self):
        _, t, _, k, n = self._make_random_spline()
        c = np.random.random(size=(n, 6, 7))
        b = BSpline(t, c, k)
        xx = np.random.random((3, 4, 5))*(t[-1] - t[0]) + t[0]
        assert_equal(b(xx).shape, (3, 4, 5, 6, 7))

    def test_endpoints(self):
        # first and last intervals are closed
        b, t, c, k, n = self._make_random_spline()
        x = [t[0], t[-1]]
        xe = [t[0] + 1e-10, t[-1] - 1e-10]
        for extrap in (True, False):
            assert_allclose(b(x, extrapolate=extrap),
                    b(xe, extrapolate=extrap), atol=1e-9, rtol=0)

        # repeat w/ single basis element
        xx = np.array([0., 1.])

        bb = [(BSpline.basis_element(t=[0, 0, 1]), lambda x: 1.-x),
              (BSpline.basis_element(t=[0, 1, 1]), lambda x: x)]
        for b, f in bb:
            for extrap in (False, True):
                assert_allclose(b(xx, extrapolate=extrap), f(xx),
                        atol=1e-14, rtol=0)

    def test_out_of_bounds(self):
        xx = np.array([-5., -1., 2., 5.])
        bb = [(BSpline.basis_element(t=[0, 0, 1]), lambda x: 1.-x),
              (BSpline.basis_element(t=[0, 1, 1]), lambda x: x)]

        for b, f in bb:
            y_nope = b(xx, extrapolate=False)
            assert_equal(y_nope, 0.)
            y_extr = b(xx, extrapolate=True)
            assert_allclose(y_extr, f(xx), atol=1e-14, rtol=0)

        # repeat with a random spline
        b, t, c, k, n = self._make_random_spline()
        xx = [t[0]-8., t[0]-0.5, t[-1]+0.5, t[-1]+8.]
        assert_equal(b(xx), 0)

    def test_to_fitpack(self):
        b, t, _, _, _ = self._make_random_spline()
        tck = b.get_fitpack_tck()
        
        xx = np.linspace(t[0], t[-1], 30)
        assert_allclose(b(xx), splev(xx, tck))

        # also make sure PPoly understands these tck
        p = PPoly.from_spline(tck)
        assert_allclose(b(xx), p(xx))

    def test_from_fitpack(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(20))
        y = np.random.random(20)

        tck = splrep(x, y)
        b = BSpline.from_fitpack_tck(tck)

        xx = np.linspace(-0.5, 1.5, 80)
        assert_allclose(b(xx, extrapolate=True),
                        splev(xx, tck))

    def test_knots(self):
        b, t, c, k, n = self._make_random_spline()
        assert_equal(b.knots, t)
        assert_equal(b.t.size, b.knots.size + 2*k)
        with assert_raises(AttributeError):
            b.knots = 101

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
    i = np.searchsorted(t, x) - 1
    assert t[i] <= x <= t[i+1]
    assert i >= k and i < len(t) - k
    return sum(c[i-j] * _naive_B(x, k, i-j, t) for j in range(0, k+1))


if __name__ == "__main__":
    run_module_suite()
