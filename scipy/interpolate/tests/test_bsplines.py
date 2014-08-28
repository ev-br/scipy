import numpy as np
from numpy.testing import (run_module_suite, TestCase, assert_equal,
        assert_allclose, assert_raises, assert_)
from numpy.testing.decorators import skipif, knownfailureif

from scipy.interpolate import (BSpline, splev, splrep, BPoly, PPoly,
        make_interp_spline, _bspl)
from scipy.interpolate._bsplines import _not_a_knot, _augknt
import scipy.linalg as sl


class TestBSpline(TestCase):

    def test_ctor(self):
        # knots should be an ordered 1D array of finite real numbers
        assert_raises((TypeError, ValueError), BSpline,
                **dict(t=[1, 1.j], c=[1.], k=0))
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
        assert_allclose(b(xx), np.where(xx < 0.35, 3, 4))

    def test_order_1(self):
        t = [0, 1, 2, 3, 4]
        c = [1, 2, 3]
        k = 1
        b = BSpline(t, c, k)

        x = np.linspace(1, 3, 50)
        assert_allclose(c[0]*B_012(x) + c[1]*B_012(x-1) + c[2]*B_012(x-2),
                        b(x))

    def test_bernstein(self):
        # a special knot vector: Bernstein polynomials
        k = 3
        t = np.asarray([0]*(k+1) + [1]*(k+1))
        c = np.asarray([1., 2., 3., 4.])
        bp = BPoly(c.reshape(-1, 1), [0, 1])
        bspl = BSpline(t, c, k)

        xx = np.linspace(-1., 2., 100)
        assert_allclose(bp(xx, extrapolate=True),
                        bspl(xx, extrapolate=True), atol=1e-14)

    def test_rndm_naive_eval(self):
        b, t, c, k = self._make_random_spline()
        xx = np.linspace(t[k], t[-k-1], 50)
        y_b = b(xx)

        y_n = [_naive_eval(x, t, c, k) for x in xx]
        assert_allclose(y_b, y_n)

        y_n2 = [_naive_eval_2(x, t, c, k) for x in xx]
        assert_allclose(y_b, y_n2)

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

    def test_rndm_splev_multiple_knots(self):
        b, t, c, k = self._make_random_spline()
        for b1 in _make_multiples(b):
            xx = np.linspace(b1.t[0]-0.3, b1.t[-1]+0.3, 50)
            xx = np.r_[xx, b1.t]
            assert_allclose(b1(xx, extrapolate=True),
                    splev(xx, (b1.t, b1.c, b1.k)))

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
        assert_allclose(b(t[k+1:-k-1] - 1e-10), b(t[k+1:-k-1] + 1e-10))

        # repeat with multiple knots
        for b1 in _make_multiples(b):
            x = np.unique(b1.t[b1.k + 1:-b1.k - 1])
            m = (x != b1.t[0]) & (x != b1.t[-1])
            assert_allclose(b1(x[m] - 1e-14), b1(x[m] + 1e-14), atol=1e-10)

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
        for b1 in _make_multiples(b):
            assert_allclose(b1(xx, extrapolate=True),
                    splev(xx, (b1.t, b1.c, b1.k), ext=0))

    def test_default_extrap(self):
        # BSpline defaults to extrapolate=True
        b, t, c, k = self._make_random_spline()
        xx = [t[0] - 1, t[-1] + 1]
        yy = b(xx)
        assert_(not np.all(np.isnan(yy)))

    def test_ppoly(self):
        b, t, c, k = self._make_random_spline()
        pp = PPoly.from_spline((t, c, k))

        xx = np.linspace(t[k], t[-k-1], 100)
        assert_allclose(b(xx), pp(xx), atol=1e-14, rtol=1e-14)

    def test_derivative_rndm(self):
        b, t, c, k = self._make_random_spline()
        xx = np.linspace(t[0], t[-10], 50)
        xx = np.r_[xx, t]

        for der in range(1, k):
            yd = splev(xx, (t, c, k), der=der)
            assert_allclose(yd, b(xx, nu=der))

            # repeat with multiple knots
            for b1 in _make_multiples(b):
                assert_allclose(b1(xx, nu=der),
                        splev(xx, (b1.t, b1.c, b1.k), der=der))

    def test_derivative_jumps(self):
        # example from de Boor, Chap IX, example (24)
        # NB: knots augmented & corresp coefs are zeroed out
        # in agreement with the convention (29)
        k = 2
        t = [-1, -1, 0, 1, 1, 3, 4, 6, 6, 6, 7, 7]
        np.random.seed(1234)
        c = np.r_[0, 0, np.random.random(5), 0, 0]
        b = BSpline(t, c, k)

        # b is continuous at x != 6 (triple knot)
        x = np.asarray([1, 3, 4, 6])
        assert_allclose(b(x[x != 6] - 1e-10),
                        b(x[x != 6] + 1e-10))
        assert_(not np.allclose(b(6.-1e-10), b(6+1e-10)))

        # 1st derivative jumps at double knots, 1 & 6:
        x0 = np.asarray([3, 4])
        assert_allclose(b(x0 - 1e-10, nu=1),
                        b(x0 + 1e-10, nu=1))
        x1 = np.asarray([1, 6])
        assert_(not np.all(np.allclose(b(x1 - 1e-10, nu=1),
                                       b(x1 + 1e-10, nu=1))))

        # 2nd derivative is not guaranteed to be continuous
        assert_(not np.all(np.allclose(b(x - 1e-10, nu=2),
                                       b(x + 1e-10, nu=2))))

    def test_basis_element_quadratic(self):
        xx = np.linspace(-1, 4, 20)
        conds = [xx < 1, (xx > 1) & (xx < 2), xx > 2]
        funcs = [lambda x: x*x/2.,
                 lambda x: 3./4 - (x-3./2)**2,
                 lambda x: (3.-x)**2 / 2]
        pieces = np.piecewise(xx, conds, funcs)
        b = BSpline.basis_element(t=[0, 1, 2, 3])
        assert_allclose(b(xx), pieces)

        b = BSpline.basis_element(t=[0, 1, 1, 2])
        xx = np.linspace(0, 2, 10)
        assert_allclose(b(xx),
                np.where(xx < 1, xx*xx, (2.-xx)**2))

    def test_basis_element_rndm(self):
        b, t, c, k = self._make_random_spline()
        xx = np.linspace(t[k], t[-k-1], 20)
        assert_allclose(b(xx), _sum_basis_elements(xx, t, c, k))

    def test_cmplx(self):
        b, t, c, k = self._make_random_spline()
        cc = c * (1. + 3.j)

        b = BSpline(t, cc, k)
        b_re = BSpline(t, b.c.real, k)
        b_im = BSpline(t, b.c.imag, k)

        xx = np.linspace(t[k], t[-k-1], 20)
        assert_allclose(b(xx).real, b_re(xx))
        assert_allclose(b(xx).imag, b_im(xx))

    def test_derivative(self):
        b, t, c, k = self._make_random_spline(k=5)
        b0 = BSpline(t, c, k)
        xx = np.linspace(t[k], t[-k-1], 20)
        for j in range(1, k):
            b = b.derivative()
            assert_allclose(b0(xx, j), b(xx), atol=1e-12, rtol=1e-12)

    def test_antiderivative(self):
        b, t, c, k = self._make_random_spline()
        xx = np.linspace(t[k], t[-k-1], 20)
        assert_allclose(b.antiderivative().derivative()(xx),
                        b(xx), atol=1e-14, rtol=1e-14)

    def test_integral(self):
        b = BSpline.basis_element([0, 1, 1, 2])  # x**2 for x < 1 else (x-2)**2
        assert_allclose(b.integrate(0, 1), 1./3)
        assert_allclose(b.integrate(1, 0), -1./3)

        # extrapolate or zeros outside of [0, 2]; default is yes
        assert_allclose(b.integrate(-1, 3), 4./3)
        assert_allclose(b.integrate(-1, 3, extrapolate=True), 4./3)
        assert_allclose(b.integrate(-1, 3, extrapolate=False), 2./3)


### stolen from @pv, verbatim
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


### stolen from @pv, verbatim
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


def _sum_basis_elements(x, t, c, k):
    n = len(t) - (k+1)
    assert n >= k+1
    assert len(c) >= n
    s = 0.
    for i in range(n):
        b = BSpline.basis_element(t[i:i+k+2], extrapolate=False)(x)
        s += c[i] * np.nan_to_num(b)   # zero out out-of-bounds elements
    return s


def B_012(x):
    """ A linear B-spline function B(x | 0, 1, 2)"""
    x = np.atleast_1d(x)
    return np.piecewise(x, [(x < 0) | (x > 2),
                            (x >= 0) & (x < 1),
                            (x >= 1) & (x <= 2)],
                           [lambda x: 0., lambda x: x, lambda x: 2.-x])


def _make_multiples(b):
    """Increase knot multiplicity."""
    c, k = b.c, b.k

    t1 = b.t.copy()
    t1[17:19] = t1[17]
    t1[22] = t1[21]
    yield BSpline(t1, c, k)

    t1 = b.t.copy()
    t1[:k+1] = t1[0]
    yield BSpline(t1, c, k)

    t1 = b.t.copy()
    t1[:2*k + 2] = t1[0]
    yield BSpline(t1, c, k)

    t1 = b.t.copy()
    t1[-k-1:] = t1[-1]
    yield BSpline(t1, c, k)

    t1 = b.t.copy()
    t1[-2*k-2:] = t1[-1]
    yield BSpline(t1, c, k)


class TestInterp(TestCase):
    #
    # Test basic ways of constructing interpolating splines.
    #
    xx = np.linspace(0., 2.*np.pi)
    yy = np.sin(xx)

    def test_order_0(self):
        tck = make_interp_spline(self.xx, self.yy, k=0)
        b = BSpline(*tck)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    def test_linear(self):
        tck = make_interp_spline(self.xx, self.yy, k=1)
        b = BSpline(*tck)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    def test_not_a_knot(self):
        for k in [3, 5]:
            tck = make_interp_spline(self.xx, self.yy, k)
            b = BSpline(*tck)
            assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    def test_quadratic_deriv(self):
        der = [(1, 8.)]  # order, value: f'(x) = 8.

        # derivative @ right-hand edge
        tck = make_interp_spline(self.xx, self.yy, k=2, deriv_r=der)
        b = BSpline(*tck)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        assert_allclose(b(self.xx[-1], 1), der[0][1], atol=1e-14, rtol=1e-14)

        # derivative @ left-hand edge
        tck = make_interp_spline(self.xx, self.yy, k=2, deriv_l=der)
        b = BSpline(*tck)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        assert_allclose(b(self.xx[0], 1), der[0][1], atol=1e-14, rtol=1e-14)

    def test_cubic_deriv(self):
        k = 3

        # first derivatives @ left & right edges:
        der_l, der_r = [(1, 3.)], [(1, 4.)]
        tck = make_interp_spline(self.xx, self.yy, k,
                deriv_l=der_l, deriv_r=der_r)
        b = BSpline(*tck)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        assert_allclose([b(self.xx[0], 1), b(self.xx[-1], 1)],
                        [der_l[0][1], der_r[0][1]], atol=1e-14, rtol=1e-14)

        # 'natural' cubic spline, zero out 2nd derivatives @ the boundaries
        der_l, der_r = [(2, 0)], [(2, 0)]
        tck = make_interp_spline(self.xx, self.yy, k,
                deriv_l=der_l, deriv_r=der_r)
        b = BSpline(*tck)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    def test_quintic_derivs(self):
        k, n = 5, 7
        x = np.arange(n).astype(np.float_)
        y = np.sin(x)
        der_l = [(1, -12.), (2, 1)]
        der_r = [(1, 8.), (2, 3.)]
        tck = make_interp_spline(x, y, k=5, deriv_l=der_l, deriv_r=der_r)
        b = BSpline(*tck)
        assert_allclose(b(x), y, atol=1e-14, rtol=1e-14)
        assert_allclose([b(x[0], 1), b(x[0], 2)],
                        [val for (nu, val) in der_l])
        assert_allclose([b(x[-1], 1), b(x[-1], 2)],
                        [val for (nu, val) in der_r])

    @knownfailureif(True, 'unstable')
    def test_cubic_deriv_unstable(self):
        # 1st and 2nd derivative @ x[0], no derivative information @ x[-1]
        # The problem is not that it fails [who would use this anyway],
        # the problem is that it fails *silently*, and I've no idea
        # how to detect this sort of instability.
        k = 3
        t = _augknt(self.xx, k)

        der_l = [(1, 3.), (2, 4.)]
        tck = make_interp_spline(self.xx, self.yy, k, t, deriv_l=der_l)
        b = BSpline(*tck)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    def test_knots_not_data_sites(self):
        # Knots need not coincide with the data sites.
        # use a quadratic spline, knots are @ data averages,
        # two additional constraints are zero 2nd derivs @ edges
        k, n = 2, 8
        t = np.r_[(self.xx[0],)*(k+1),
                  (self.xx[1:] + self.xx[:-1]) / 2.,
                  (self.xx[-1],)*(k+1)]
        tck = make_interp_spline(self.xx, self.yy, k, t,
                deriv_l=[(2, 0)], deriv_r=[(2, 0)])
        b = BSpline(*tck)

        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        assert_allclose([b(self.xx[0], 2), b(self.xx[-1], 2)], [0., 0.],
                atol=1e-14)

    def test_complex(self):
        k = 3
        xx = self.xx
        yy = self.yy + 1.j*self.yy

        # first derivatives @ left & right edges:
        der_l, der_r = [(1, 3.j)], [(1, 4.+2.j)]
        tck = make_interp_spline(xx, yy, k,
                deriv_l=der_l, deriv_r=der_r)
        b = BSpline(*tck)
        assert_allclose(b(xx), yy, atol=1e-14, rtol=1e-14)
        assert_allclose([b(xx[0], 1), b(xx[-1], 1)],
                        [der_l[0][1], der_r[0][1]], atol=1e-14, rtol=1e-14)

    def test_int_xy(self):
        x = np.arange(10).astype(np.int_)
        y = np.arange(10).astype(np.int_)

        # cython chokes on "buffer type mismatch"
        tck = make_interp_spline(x, y, k=1)

    def test_sliced_input(self):
        # cython code chokes on non C contiguous arrays
        xx = np.linspace(-1, 1, 100)

        x = xx[::5]
        y = xx[::5]

        tck = make_interp_spline(x, y, k=1)

    def test_multiple_rhs(self):
        yy = np.c_[np.sin(self.xx), np.cos(self.xx)]
        der_l = [(1, [1., 2.])]
        der_r = [(1, [3., 4.])]

        tck = make_interp_spline(self.xx, yy, k=3, deriv_l=der_l, deriv_r=der_r)
        b = BSpline(*tck)
        assert_allclose(b(self.xx), yy, atol=1e-14, rtol=1e-14)
        assert_allclose(b(self.xx[0], 1), der_l[0][1], atol=1e-14, rtol=1e-14)
        assert_allclose(b(self.xx[-1], 1), der_r[0][1], atol=1e-14, rtol=1e-14)

    def test_shapes(self):
        np.random.seed(1234)
        k, n = 3, 22
        x = np.sort(np.random.random(size=n))
        y = np.random.random(size=(n, 5, 6, 7))

        t, c, k = make_interp_spline(x, y, k)
        assert_equal(c.shape, (n, 5, 6, 7))

        # now throw in some derivatives
        d_l = [(1, np.random.random((5, 6, 7)))]
        d_r = [(1, np.random.random((5, 6, 7)))]
        t, c, k = make_interp_spline(x, y, k, deriv_l=d_l, deriv_r=d_r)
        assert_equal(c.shape, (n + k - 1, 5, 6, 7))

    def test_full_matrix(self):
        np.random.seed(1234)
        k, n = 3, 7
        x = np.sort(np.random.random(size=n))
        y = np.random.random(size=n)
        t = _not_a_knot(x, k)

        _, cb, _ = make_interp_spline(x, y, k, t)
        cf = make_interp_full_matr(x, y, t, k)
        assert_allclose(cb, cf, atol=1e-14, rtol=1e-14)


def make_interp_full_matr(x, y, t, k):
    """Assemble an spline order k with knots t to interpolate
    y(x) using full matrices.
    Not-a-knot BC only.

    This routine is here for testing only (even though it's functional).
    """
    assert x.size == y.size
    assert t.size == x.size + k + 1
    n = x.size

    A = np.zeros((n, n), dtype=np.float_)

    for j in range(n):
        xval = x[j]
        if xval == t[k]:
            left = k
        else:
            left = np.searchsorted(t, xval) - 1

        # fill a row
        bb = _bspl.evaluate_all_bspl(t, k, xval, left)
        A[j, left-k:left+1] = bb

    c = sl.solve(A, y)
    return c


### XXX: 'periodic' interp spline using full matrices
def make_interp_per_full_matr(x, y, t, k):
    x, y, t = map(np.asarray, (x, y, t))

    n = x.size
    nt = t.size - k - 1

    # have `n` conditions for `nt` coefficients; need nt-n derivatives
    assert nt - n == k - 1

    # LHS: the collocation matrix + derivatives @edges
    A = np.zeros((nt, nt), dtype=np.float_)

    # derivatives @ x[0]:
    offset = 0

    if x[0] == t[k]:
        left = k
    else:
        left = np.searchsorted(t, x[0]) - 1

    if x[-1] == t[k]:
        left2 = k
    else:
        left2 = np.searchsorted(t, x[-1]) - 1

    for i in range(k-1):
        bb = _bspl.evaluate_all_bspl(t, k, x[0], left, nu=i+1)
        A[i, left-k:left+1] = bb
        bb = _bspl.evaluate_all_bspl(t, k, x[-1], left2, nu=i+1)
        A[i, left2-k:left2+1] = -bb
        offset += 1

    # RHS
    y = np.r_[[0]*(k-1), y]

    # collocation matrix
    for j in range(n):
        xval = x[j]
        # find interval
        if xval == t[k]:
            left = k
        else:
            left = np.searchsorted(t, xval) - 1

        # fill a row
        bb = _bspl.evaluate_all_bspl(t, k, xval, left)
        A[j + offset, left-k:left+1] = bb

    c = sl.solve(A, y)
    return c


if __name__ == "__main__":
    run_module_suite()
