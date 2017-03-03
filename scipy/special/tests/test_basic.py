# this program corresponds to special.py

### Means test is not done yet
# E   Means test is giving error (E)
# F   Means test is failing (F)
# EF  Means test is giving error and Failing
#!   Means test is segfaulting
# 8   Means test runs forever

###  test_besselpoly
###  test_mathieu_a
###  test_mathieu_even_coef
###  test_mathieu_odd_coef
###  test_modfresnelp
###  test_modfresnelm
#    test_pbdv_seq
###  test_pbvv_seq
###  test_sph_harm
#    test_sph_in
#    test_sph_jn
#    test_sph_kn

from __future__ import division, print_function, absolute_import

import itertools
import warnings

import numpy as np
from numpy import (array, isnan, r_, arange, finfo, pi, sin, cos, tan, exp,
        log, zeros, sqrt, asarray, inf, nan_to_num, real, arctan, float_)

from numpy.testing import (assert_equal, assert_almost_equal,
        assert_array_equal, assert_array_almost_equal, assert_approx_equal,
        assert_, dec, TestCase, run_module_suite, assert_allclose,
        assert_raises, assert_array_almost_equal_nulp)

from scipy import special
import scipy.special._ufuncs as cephes
from scipy.special import ellipk, zeta

from scipy.special._testutils import assert_tol_equal, with_special_errors, \
     assert_func_equal

from scipy._lib._version import NumpyVersion

import math


class TestCephes(TestCase):
    def test_airy(self):
        cephes.airy(0)

    def test_airye(self):
        cephes.airye(0)

    def test_binom(self):
        n = np.array([0.264, 4, 5.2, 17])
        k = np.array([2, 0.4, 7, 3.3])
        nk = np.array(np.broadcast_arrays(n[:,None], k[None,:])
                      ).reshape(2, -1).T
        rknown = np.array([[-0.097152, 0.9263051596159367, 0.01858423645695389,
            -0.007581020651518199],[6, 2.0214389119675666, 0, 2.9827344527963846],
            [10.92, 2.22993515861399, -0.00585728, 10.468891352063146],
            [136, 3.5252179590758828, 19448, 1024.5526916174495]])
        assert_func_equal(cephes.binom, rknown.ravel(), nk, rtol=1e-13)

        # Test branches in implementation
        np.random.seed(1234)
        n = np.r_[np.arange(-7, 30), 1000*np.random.rand(30) - 500]
        k = np.arange(0, 102)
        nk = np.array(np.broadcast_arrays(n[:,None], k[None,:])
                      ).reshape(2, -1).T

        assert_func_equal(cephes.binom,
                          cephes.binom(nk[:,0], nk[:,1] * (1 + 1e-15)),
                          nk,
                          atol=1e-10, rtol=1e-10)

    def test_binom_2(self):
        # Test branches in implementation
        np.random.seed(1234)
        n = np.r_[np.logspace(1, 300, 20)]
        k = np.arange(0, 102)
        nk = np.array(np.broadcast_arrays(n[:,None], k[None,:])
                      ).reshape(2, -1).T

        assert_func_equal(cephes.binom,
                          cephes.binom(nk[:,0], nk[:,1] * (1 + 1e-15)),
                          nk,
                          atol=1e-10, rtol=1e-10)

    def test_binom_exact(self):
        @np.vectorize
        def binom_int(n, k):
            n = int(n)
            k = int(k)
            num = int(1)
            den = int(1)
            for i in range(1, k+1):
                num *= i + n - k
                den *= i
            return float(num/den)

        np.random.seed(1234)
        n = np.arange(1, 15)
        k = np.arange(0, 15)
        nk = np.array(np.broadcast_arrays(n[:,None], k[None,:])
                      ).reshape(2, -1).T
        nk = nk[nk[:,0] >= nk[:,1]]
        assert_func_equal(cephes.binom,
                          binom_int(nk[:,0], nk[:,1]),
                          nk,
                          atol=0, rtol=0)

    def test_bdtr(self):
        assert_equal(cephes.bdtr(1,1,0.5),1.0)

    def test_bdtri(self):
        assert_equal(cephes.bdtri(1,3,0.5),0.5)

    def test_bdtrc(self):
        assert_equal(cephes.bdtrc(1,3,0.5),0.5)

    def test_bdtrin(self):
        assert_equal(cephes.bdtrin(1,0,1),5.0)

    def test_bdtrik(self):
        cephes.bdtrik(1,3,0.5)

    def test_bei(self):
        assert_equal(cephes.bei(0),0.0)

    def test_beip(self):
        assert_equal(cephes.beip(0),0.0)

    def test_ber(self):
        assert_equal(cephes.ber(0),1.0)

    def test_berp(self):
        assert_equal(cephes.berp(0),0.0)

    def test_besselpoly(self):
        assert_equal(cephes.besselpoly(0,0,0),1.0)

    def test_beta(self):
        assert_equal(cephes.beta(1,1),1.0)
        assert_allclose(cephes.beta(-100.3, 1e-200), cephes.gamma(1e-200))
        assert_allclose(cephes.beta(0.0342, 171), 24.070498359873497,
                        rtol=1e-13, atol=0)

    def test_betainc(self):
        assert_equal(cephes.betainc(1,1,1),1.0)
        assert_allclose(cephes.betainc(0.0342, 171, 1e-10), 0.55269916901806648)

    def test_betaln(self):
        assert_equal(cephes.betaln(1,1),0.0)
        assert_allclose(cephes.betaln(-100.3, 1e-200), cephes._gammaln(1e-200))
        assert_allclose(cephes.betaln(0.0342, 170), 3.1811881124242447,
                        rtol=1e-14, atol=0)

    def test_betaincinv(self):
        assert_equal(cephes.betaincinv(1,1,1),1.0)
        assert_allclose(cephes.betaincinv(0.0342, 171, 0.25),
                        8.4231316935498957e-21, rtol=3e-12, atol=0)

    def test_beta_inf(self):
        assert_(np.isinf(special.beta(-1, 2)))

    def test_btdtr(self):
        assert_equal(cephes.btdtr(1,1,1),1.0)

    def test_btdtri(self):
        assert_equal(cephes.btdtri(1,1,1),1.0)

    def test_btdtria(self):
        assert_equal(cephes.btdtria(1,1,1),5.0)

    def test_btdtrib(self):
        assert_equal(cephes.btdtrib(1,1,1),5.0)

    def test_cbrt(self):
        assert_approx_equal(cephes.cbrt(1),1.0)

    def test_chdtr(self):
        assert_equal(cephes.chdtr(1,0),0.0)

    def test_chdtrc(self):
        assert_equal(cephes.chdtrc(1,0),1.0)

    def test_chdtri(self):
        assert_equal(cephes.chdtri(1,1),0.0)

    def test_chdtriv(self):
        assert_equal(cephes.chdtriv(0,0),5.0)

    def test_chndtr(self):
        assert_equal(cephes.chndtr(0,1,0),0.0)
        p = cephes.chndtr(np.linspace(20, 25, 5), 2, 1.07458615e+02)
        assert_allclose(p, [1.21805009e-09, 2.81979982e-09, 6.25652736e-09,
                            1.33520017e-08, 2.74909967e-08],
                        rtol=1e-6, atol=0)
        assert_almost_equal(cephes.chndtr(np.inf, np.inf, 0), 2.0)
        assert_almost_equal(cephes.chndtr(2, 1, np.inf), 0.0)
        assert_(np.isnan(cephes.chndtr(np.nan, 1, 2)))
        assert_(np.isnan(cephes.chndtr(5, np.nan, 2)))
        assert_(np.isnan(cephes.chndtr(5, 1, np.nan)))

    def test_chndtridf(self):
        assert_equal(cephes.chndtridf(0,0,1),5.0)

    def test_chndtrinc(self):
        assert_equal(cephes.chndtrinc(0,1,0),5.0)

    def test_chndtrix(self):
        assert_equal(cephes.chndtrix(0,1,0),0.0)

    def test_cosdg(self):
        assert_equal(cephes.cosdg(0),1.0)

    def test_cosm1(self):
        assert_equal(cephes.cosm1(0),0.0)

    def test_cotdg(self):
        assert_almost_equal(cephes.cotdg(45),1.0)

    def test_dawsn(self):
        assert_equal(cephes.dawsn(0),0.0)
        assert_allclose(cephes.dawsn(1.23), 0.50053727749081767)

    def test_diric(self):
        # Test behavior near multiples of 2pi.  Regression test for issue
        # described in gh-4001.
        n_odd = [1, 5, 25]
        x = np.array(2*np.pi + 5e-5).astype(np.float32)
        assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=7)
        x = np.array(2*np.pi + 1e-9).astype(np.float64)
        assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=15)
        x = np.array(2*np.pi + 1e-15).astype(np.float64)
        assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=15)
        if hasattr(np, 'float128'):
            # No float128 available in 32-bit numpy
            x = np.array(2*np.pi + 1e-12).astype(np.float128)
            assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=19)

        n_even = [2, 4, 24]
        x = np.array(2*np.pi + 1e-9).astype(np.float64)
        assert_almost_equal(special.diric(x, n_even), -1.0, decimal=15)

        # Test at some values not near a multiple of pi
        x = np.arange(0.2*np.pi, 1.0*np.pi, 0.2*np.pi)
        octave_result = [0.872677996249965, 0.539344662916632,
                         0.127322003750035, -0.206011329583298]
        assert_almost_equal(special.diric(x, 3), octave_result, decimal=15)

    def test_diric_broadcasting(self):
        x = np.arange(5)
        n = np.array([1, 3, 7])
        assert_(special.diric(x[:, np.newaxis], n).shape == (x.size, n.size))

    def test_ellipe(self):
        assert_equal(cephes.ellipe(1),1.0)

    def test_ellipeinc(self):
        assert_equal(cephes.ellipeinc(0,1),0.0)

    def test_ellipj(self):
        cephes.ellipj(0,1)

    def test_ellipk(self):
        assert_allclose(ellipk(0), pi/2)

    def test_ellipkinc(self):
        assert_equal(cephes.ellipkinc(0,0),0.0)

    def test_erf(self):
        assert_equal(cephes.erf(0),0.0)

    def test_erfc(self):
        assert_equal(cephes.erfc(0),1.0)

    def test_exp1(self):
        cephes.exp1(1)

    def test_expi(self):
        cephes.expi(1)

    def test_expn(self):
        cephes.expn(1,1)

    def test_exp1_reg(self):
        # Regression for #834
        a = cephes.exp1(-complex(19.9999990))
        b = cephes.exp1(-complex(19.9999991))
        assert_array_almost_equal(a.imag, b.imag)

    def test_exp10(self):
        assert_approx_equal(cephes.exp10(2),100.0)

    def test_exp2(self):
        assert_equal(cephes.exp2(2),4.0)

    def test_expm1(self):
        assert_equal(cephes.expm1(0),0.0)
        assert_equal(cephes.expm1(np.inf), np.inf)
        assert_equal(cephes.expm1(-np.inf), -1)
        assert_equal(cephes.expm1(np.nan), np.nan)

    # Earlier numpy version don't guarantee that npy_cexp conforms to C99.
    @dec.skipif(NumpyVersion(np.__version__) < '1.9.0')
    def test_expm1_complex(self):
        expm1 = cephes.expm1
        assert_equal(expm1(0 + 0j), 0 + 0j)
        assert_equal(expm1(complex(np.inf, 0)), complex(np.inf, 0))
        assert_equal(expm1(complex(np.inf, 1)), complex(np.inf, np.inf))
        assert_equal(expm1(complex(np.inf, 2)), complex(-np.inf, np.inf))
        assert_equal(expm1(complex(np.inf, 4)), complex(-np.inf, -np.inf))
        assert_equal(expm1(complex(np.inf, 5)), complex(np.inf, -np.inf))
        assert_equal(expm1(complex(1, np.inf)), complex(np.nan, np.nan))
        assert_equal(expm1(complex(0, np.inf)), complex(np.nan, np.nan))
        assert_equal(expm1(complex(np.inf, np.inf)), complex(np.inf, np.nan))
        assert_equal(expm1(complex(-np.inf, np.inf)), complex(-1, 0))
        assert_equal(expm1(complex(-np.inf, np.nan)), complex(-1, 0))
        assert_equal(expm1(complex(np.inf, np.nan)), complex(np.inf, np.nan))
        assert_equal(expm1(complex(0, np.nan)), complex(np.nan, np.nan))
        assert_equal(expm1(complex(1, np.nan)), complex(np.nan, np.nan))
        assert_equal(expm1(complex(np.nan, 1)), complex(np.nan, np.nan))
        assert_equal(expm1(complex(np.nan, np.nan)), complex(np.nan, np.nan))

    @dec.knownfailureif(True, 'The real part of expm1(z) bad at these points')
    def test_expm1_complex_hard(self):
        # The real part of this function is difficult to evaluate when
        # z.real = -log(cos(z.imag)).
        y = np.array([0.1, 0.2, 0.3, 5, 11, 20])
        x = -np.log(np.cos(y))
        z = x + 1j*y

        # evaluate using mpmath.expm1 with dps=1000
        expected = np.array([-5.5507901846769623e-17+0.10033467208545054j,
                              2.4289354732893695e-18+0.20271003550867248j,
                              4.5235500262585768e-17+0.30933624960962319j,
                              7.8234305217489006e-17-3.3805150062465863j,
                             -1.3685191953697676e-16-225.95084645419513j,
                              8.7175620481291045e-17+2.2371609442247422j])
        found = cephes.expm1(z)
        # this passes.
        assert_array_almost_equal_nulp(found.imag, expected.imag, 3)
        # this fails.
        assert_array_almost_equal_nulp(found.real, expected.real, 20)

    def test_fdtr(self):
        assert_equal(cephes.fdtr(1,1,0),0.0)

    def test_fdtrc(self):
        assert_equal(cephes.fdtrc(1,1,0),1.0)

    def test_fdtri(self):
        # cephes.fdtri(1,1,0.5)  #BUG: gives NaN, should be 1
        assert_allclose(cephes.fdtri(1, 1, [0.499, 0.501]),
                        array([0.9937365, 1.00630298]), rtol=1e-6)

    def test_fdtridfd(self):
        assert_equal(cephes.fdtridfd(1,0,0),5.0)

    def test_fresnel(self):
        assert_equal(cephes.fresnel(0),(0.0,0.0))

    def test_gamma(self):
        assert_equal(cephes.gamma(5),24.0)

    def test_gammainc(self):
        assert_equal(cephes.gammainc(5,0),0.0)

    def test_gammaincc(self):
        assert_equal(cephes.gammaincc(5,0),1.0)

    def test_gammainccinv(self):
        assert_equal(cephes.gammainccinv(5,1),0.0)

    def test_gammaln(self):
        cephes._gammaln(10)

    def test_gammasgn(self):
        vals = np.array([-4, -3.5, -2.3, 1, 4.2], np.float64)
        assert_array_equal(cephes.gammasgn(vals), np.sign(cephes.rgamma(vals)))

    def test_gdtr(self):
        assert_equal(cephes.gdtr(1,1,0),0.0)

    def test_gdtr_inf(self):
        assert_equal(cephes.gdtr(1,1,np.inf),1.0)

    def test_gdtrc(self):
        assert_equal(cephes.gdtrc(1,1,0),1.0)

    def test_gdtria(self):
        assert_equal(cephes.gdtria(0,1,1),0.0)

    def test_gdtrib(self):
        cephes.gdtrib(1,0,1)
        # assert_equal(cephes.gdtrib(1,0,1),5.0)

    def test_gdtrix(self):
        cephes.gdtrix(1,1,.1)

    def test_hankel1(self):
        cephes.hankel1(1,1)

    def test_hankel1e(self):
        cephes.hankel1e(1,1)

    def test_hankel2(self):
        cephes.hankel2(1,1)

    def test_hankel2e(self):
        cephes.hankel2e(1,1)

    def test_hyp1f1(self):
        assert_approx_equal(cephes.hyp1f1(1,1,1), exp(1.0))
        assert_approx_equal(cephes.hyp1f1(3,4,-6), 0.026056422099537251095)
        cephes.hyp1f1(1,1,1)

    def test_hyp1f2(self):
        cephes.hyp1f2(1,1,1,1)

    def test_hyp2f0(self):
        cephes.hyp2f0(1,1,1,1)

    def test_hyp2f1(self):
        assert_equal(cephes.hyp2f1(1,1,1,0),1.0)

    def test_hyp3f0(self):
        assert_equal(cephes.hyp3f0(1,1,1,0),(1.0,0.0))

    def test_hyperu(self):
        assert_equal(cephes.hyperu(0,1,1),1.0)

    def test_i0(self):
        assert_equal(cephes.i0(0),1.0)

    def test_i0e(self):
        assert_equal(cephes.i0e(0),1.0)

    def test_i1(self):
        assert_equal(cephes.i1(0),0.0)

    def test_i1e(self):
        assert_equal(cephes.i1e(0),0.0)

    def test_it2i0k0(self):
        cephes.it2i0k0(1)

    def test_it2j0y0(self):
        cephes.it2j0y0(1)

    def test_it2struve0(self):
        cephes.it2struve0(1)

    def test_itairy(self):
        cephes.itairy(1)

    def test_iti0k0(self):
        assert_equal(cephes.iti0k0(0),(0.0,0.0))

    def test_itj0y0(self):
        assert_equal(cephes.itj0y0(0),(0.0,0.0))

    def test_itmodstruve0(self):
        assert_equal(cephes.itmodstruve0(0),0.0)

    def test_itstruve0(self):
        assert_equal(cephes.itstruve0(0),0.0)

    def test_iv(self):
        assert_equal(cephes.iv(1,0),0.0)

    def _check_ive(self):
        assert_equal(cephes.ive(1,0),0.0)

    def test_j0(self):
        assert_equal(cephes.j0(0),1.0)

    def test_j1(self):
        assert_equal(cephes.j1(0),0.0)

    def test_jn(self):
        assert_equal(cephes.jn(0,0),1.0)

    def test_jv(self):
        assert_equal(cephes.jv(0,0),1.0)

    def _check_jve(self):
        assert_equal(cephes.jve(0,0),1.0)

    def test_k0(self):
        cephes.k0(2)

    def test_k0e(self):
        cephes.k0e(2)

    def test_k1(self):
        cephes.k1(2)

    def test_k1e(self):
        cephes.k1e(2)

    def test_kei(self):
        cephes.kei(2)

    def test_keip(self):
        assert_equal(cephes.keip(0),0.0)

    def test_ker(self):
        cephes.ker(2)

    def test_kerp(self):
        cephes.kerp(2)

    def _check_kelvin(self):
        cephes.kelvin(2)

    def test_kn(self):
        cephes.kn(1,1)

    def test_kolmogi(self):
        assert_equal(cephes.kolmogi(1),0.0)
        assert_(np.isnan(cephes.kolmogi(np.nan)))

    def test_kolmogorov(self):
        assert_equal(cephes.kolmogorov(0),1.0)

    def _check_kv(self):
        cephes.kv(1,1)

    def _check_kve(self):
        cephes.kve(1,1)

    def test_log1p(self):
        log1p = cephes.log1p
        assert_equal(log1p(0), 0.0)
        assert_equal(log1p(-1), -np.inf)
        assert_equal(log1p(-2), np.nan)
        assert_equal(log1p(np.inf), np.inf)

    # earlier numpy version don't guarantee that npy_clog conforms to C99
    @dec.skipif(NumpyVersion(np.__version__) < '1.9.0')
    def test_log1p_complex(self):
        log1p = cephes.log1p
        c = complex
        assert_equal(log1p(0 + 0j), 0 + 0j)
        assert_equal(log1p(c(-1, 0)), c(-np.inf, 0))
        assert_allclose(log1p(c(1, np.inf)), c(np.inf, np.pi/2))
        assert_equal(log1p(c(1, np.nan)), c(np.nan, np.nan))
        assert_allclose(log1p(c(-np.inf, 1)), c(np.inf, np.pi))
        assert_equal(log1p(c(np.inf, 1)), c(np.inf, 0))
        assert_allclose(log1p(c(-np.inf, np.inf)), c(np.inf, 3*np.pi/4))
        assert_allclose(log1p(c(np.inf, np.inf)), c(np.inf, np.pi/4))
        assert_equal(log1p(c(np.inf, np.nan)), c(np.inf, np.nan))
        assert_equal(log1p(c(-np.inf, np.nan)), c(np.inf, np.nan))
        assert_equal(log1p(c(np.nan, np.inf)), c(np.inf, np.nan))
        assert_equal(log1p(c(np.nan, 1)), c(np.nan, np.nan))
        assert_equal(log1p(c(np.nan, np.nan)), c(np.nan, np.nan))

    def test_lpmv(self):
        assert_equal(cephes.lpmv(0,0,1),1.0)

    def test_mathieu_a(self):
        assert_equal(cephes.mathieu_a(1,0),1.0)

    def test_mathieu_b(self):
        assert_equal(cephes.mathieu_b(1,0),1.0)

    def test_mathieu_cem(self):
        assert_equal(cephes.mathieu_cem(1,0,0),(1.0,0.0))

        # Test AMS 20.2.27
        @np.vectorize
        def ce_smallq(m, q, z):
            z *= np.pi/180
            if m == 0:
                return 2**(-0.5) * (1 - .5*q*cos(2*z))  # + O(q^2)
            elif m == 1:
                return cos(z) - q/8 * cos(3*z)  # + O(q^2)
            elif m == 2:
                return cos(2*z) - q*(cos(4*z)/12 - 1/4)  # + O(q^2)
            else:
                return cos(m*z) - q*(cos((m+2)*z)/(4*(m+1)) - cos((m-2)*z)/(4*(m-1)))  # + O(q^2)
        m = np.arange(0, 100)
        q = np.r_[0, np.logspace(-30, -9, 10)]
        assert_allclose(cephes.mathieu_cem(m[:,None], q[None,:], 0.123)[0],
                        ce_smallq(m[:,None], q[None,:], 0.123),
                        rtol=1e-14, atol=0)

    def test_mathieu_sem(self):
        assert_equal(cephes.mathieu_sem(1,0,0),(0.0,1.0))

        # Test AMS 20.2.27
        @np.vectorize
        def se_smallq(m, q, z):
            z *= np.pi/180
            if m == 1:
                return sin(z) - q/8 * sin(3*z)  # + O(q^2)
            elif m == 2:
                return sin(2*z) - q*sin(4*z)/12  # + O(q^2)
            else:
                return sin(m*z) - q*(sin((m+2)*z)/(4*(m+1)) - sin((m-2)*z)/(4*(m-1)))  # + O(q^2)
        m = np.arange(1, 100)
        q = np.r_[0, np.logspace(-30, -9, 10)]
        assert_allclose(cephes.mathieu_sem(m[:,None], q[None,:], 0.123)[0],
                        se_smallq(m[:,None], q[None,:], 0.123),
                        rtol=1e-14, atol=0)

    def test_mathieu_modcem1(self):
        assert_equal(cephes.mathieu_modcem1(1,0,0),(0.0,0.0))

    def test_mathieu_modcem2(self):
        cephes.mathieu_modcem2(1,1,1)

        # Test reflection relation AMS 20.6.19
        m = np.arange(0, 4)[:,None,None]
        q = np.r_[np.logspace(-2, 2, 10)][None,:,None]
        z = np.linspace(0, 1, 7)[None,None,:]

        y1 = cephes.mathieu_modcem2(m, q, -z)[0]

        fr = -cephes.mathieu_modcem2(m, q, 0)[0] / cephes.mathieu_modcem1(m, q, 0)[0]
        y2 = -cephes.mathieu_modcem2(m, q, z)[0] - 2*fr*cephes.mathieu_modcem1(m, q, z)[0]

        assert_allclose(y1, y2, rtol=1e-10)

    def test_mathieu_modsem1(self):
        assert_equal(cephes.mathieu_modsem1(1,0,0),(0.0,0.0))

    def test_mathieu_modsem2(self):
        cephes.mathieu_modsem2(1,1,1)

        # Test reflection relation AMS 20.6.20
        m = np.arange(1, 4)[:,None,None]
        q = np.r_[np.logspace(-2, 2, 10)][None,:,None]
        z = np.linspace(0, 1, 7)[None,None,:]

        y1 = cephes.mathieu_modsem2(m, q, -z)[0]
        fr = cephes.mathieu_modsem2(m, q, 0)[1] / cephes.mathieu_modsem1(m, q, 0)[1]
        y2 = cephes.mathieu_modsem2(m, q, z)[0] - 2*fr*cephes.mathieu_modsem1(m, q, z)[0]
        assert_allclose(y1, y2, rtol=1e-10)

    def test_mathieu_overflow(self):
        # Check that these return NaNs instead of causing a SEGV
        assert_equal(cephes.mathieu_cem(10000, 0, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_sem(10000, 0, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_cem(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_sem(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_modcem1(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_modsem1(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_modcem2(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_modsem2(10000, 1.5, 1.3), (np.nan, np.nan))

    def test_mathieu_ticket_1847(self):
        # Regression test --- this call had some out-of-bounds access
        # and could return nan occasionally
        for k in range(60):
            v = cephes.mathieu_modsem2(2, 100, -1)
            # Values from ACM TOMS 804 (derivate by numerical differentiation)
            assert_allclose(v[0], 0.1431742913063671074347, rtol=1e-10)
            assert_allclose(v[1], 0.9017807375832909144719, rtol=1e-4)

    def test_modfresnelm(self):
        cephes.modfresnelm(0)

    def test_modfresnelp(self):
        cephes.modfresnelp(0)

    def _check_modstruve(self):
        assert_equal(cephes.modstruve(1,0),0.0)

    def test_nbdtr(self):
        assert_equal(cephes.nbdtr(1,1,1),1.0)

    def test_nbdtrc(self):
        assert_equal(cephes.nbdtrc(1,1,1),0.0)

    def test_nbdtri(self):
        assert_equal(cephes.nbdtri(1,1,1),1.0)

    def __check_nbdtrik(self):
        cephes.nbdtrik(1,.4,.5)

    def test_nbdtrin(self):
        assert_equal(cephes.nbdtrin(1,0,0),5.0)

    def test_ncfdtr(self):
        assert_equal(cephes.ncfdtr(1,1,1,0),0.0)

    def test_ncfdtri(self):
        assert_equal(cephes.ncfdtri(1,1,1,0),0.0)

    def test_ncfdtridfd(self):
        cephes.ncfdtridfd(1,0.5,0,1)

    def __check_ncfdtridfn(self):
        cephes.ncfdtridfn(1,0.5,0,1)

    def __check_ncfdtrinc(self):
        cephes.ncfdtrinc(1,0.5,0,1)

    def test_nctdtr(self):
        assert_equal(cephes.nctdtr(1,0,0),0.5)
        assert_equal(cephes.nctdtr(9, 65536, 45), 0.0)

        assert_approx_equal(cephes.nctdtr(np.inf, 1., 1.), 0.5, 5)
        assert_(np.isnan(cephes.nctdtr(2., np.inf, 10.)))
        assert_approx_equal(cephes.nctdtr(2., 1., np.inf), 1.)

        assert_(np.isnan(cephes.nctdtr(np.nan, 1., 1.)))
        assert_(np.isnan(cephes.nctdtr(2., np.nan, 1.)))
        assert_(np.isnan(cephes.nctdtr(2., 1., np.nan)))

    def __check_nctdtridf(self):
        cephes.nctdtridf(1,0.5,0)

    def test_nctdtrinc(self):
        cephes.nctdtrinc(1,0,0)

    def test_nctdtrit(self):
        cephes.nctdtrit(.1,0.2,.5)

    def test_ndtr(self):
        assert_equal(cephes.ndtr(0), 0.5)
        assert_almost_equal(cephes.ndtr(1), 0.84134474606)

    def test_ndtri(self):
        assert_equal(cephes.ndtri(0.5),0.0)

    def test_nrdtrimn(self):
        assert_approx_equal(cephes.nrdtrimn(0.5,1,1),1.0)

    def test_nrdtrisd(self):
        assert_tol_equal(cephes.nrdtrisd(0.5,0.5,0.5), 0.0,
                         atol=0, rtol=0)

    def test_obl_ang1(self):
        cephes.obl_ang1(1,1,1,0)

    def test_obl_ang1_cv(self):
        result = cephes.obl_ang1_cv(1,1,1,1,0)
        assert_almost_equal(result[0],1.0)
        assert_almost_equal(result[1],0.0)

    def _check_obl_cv(self):
        assert_equal(cephes.obl_cv(1,1,0),2.0)

    def test_obl_rad1(self):
        cephes.obl_rad1(1,1,1,0)

    def test_obl_rad1_cv(self):
        cephes.obl_rad1_cv(1,1,1,1,0)

    def test_obl_rad2(self):
        cephes.obl_rad2(1,1,1,0)

    def test_obl_rad2_cv(self):
        cephes.obl_rad2_cv(1,1,1,1,0)

    def test_pbdv(self):
        assert_equal(cephes.pbdv(1,0),(0.0,1.0))

    def test_pbvv(self):
        cephes.pbvv(1,0)

    def test_pbwa(self):
        cephes.pbwa(1,0)

    def test_pdtr(self):
        val = cephes.pdtr(0, 1)
        assert_almost_equal(val, np.exp(-1))
        # Edge case: m = 0.
        val = cephes.pdtr([0, 1, 2], 0.0)
        assert_array_equal(val, [1, 1, 1])

    def test_pdtrc(self):
        val = cephes.pdtrc(0, 1)
        assert_almost_equal(val, 1 - np.exp(-1))
        # Edge case: m = 0.
        val = cephes.pdtrc([0, 1, 2], 0.0)
        assert_array_equal(val, [0, 0, 0])

    def test_pdtri(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cephes.pdtri(0.5,0.5)

    def test_pdtrik(self):
        k = cephes.pdtrik(0.5, 1)
        assert_almost_equal(cephes.gammaincc(k + 1, 1), 0.5)
        # Edge case: m = 0 or very small.
        k = cephes.pdtrik([[0], [0.25], [0.95]], [0, 1e-20, 1e-6])
        assert_array_equal(k, np.zeros((3, 3)))

    def test_pro_ang1(self):
        cephes.pro_ang1(1,1,1,0)

    def test_pro_ang1_cv(self):
        assert_array_almost_equal(cephes.pro_ang1_cv(1,1,1,1,0),
                                  array((1.0,0.0)))

    def _check_pro_cv(self):
        assert_equal(cephes.pro_cv(1,1,0),2.0)

    def test_pro_rad1(self):
        cephes.pro_rad1(1,1,1,0.1)

    def test_pro_rad1_cv(self):
        cephes.pro_rad1_cv(1,1,1,1,0)

    def test_pro_rad2(self):
        cephes.pro_rad2(1,1,1,0)

    def test_pro_rad2_cv(self):
        cephes.pro_rad2_cv(1,1,1,1,0)

    def test_psi(self):
        cephes.psi(1)

    def test_radian(self):
        assert_equal(cephes.radian(0,0,0),0)

    def test_rgamma(self):
        assert_equal(cephes.rgamma(1),1.0)

    def test_round(self):
        assert_equal(cephes.round(3.4),3.0)
        assert_equal(cephes.round(-3.4),-3.0)
        assert_equal(cephes.round(3.6),4.0)
        assert_equal(cephes.round(-3.6),-4.0)
        assert_equal(cephes.round(3.5),4.0)
        assert_equal(cephes.round(-3.5),-4.0)

    def test_shichi(self):
        cephes.shichi(1)

    def test_sici(self):
        cephes.sici(1)

        s, c = cephes.sici(np.inf)
        assert_almost_equal(s, np.pi * 0.5)
        assert_almost_equal(c, 0)

        s, c = cephes.sici(-np.inf)
        assert_almost_equal(s, -np.pi * 0.5)
        assert_(np.isnan(c), "cosine integral(-inf) is not nan")

    def test_sindg(self):
        assert_equal(cephes.sindg(90),1.0)

    def test_smirnov(self):
        assert_equal(cephes.smirnov(1,.1),0.9)
        assert_(np.isnan(cephes.smirnov(1,np.nan)))

    def test_smirnovi(self):
        assert_almost_equal(cephes.smirnov(1,cephes.smirnovi(1,0.4)),0.4)
        assert_almost_equal(cephes.smirnov(1,cephes.smirnovi(1,0.6)),0.6)
        assert_(np.isnan(cephes.smirnovi(1,np.nan)))

    def test_spence(self):
        assert_equal(cephes.spence(1),0.0)

    def test_stdtr(self):
        assert_equal(cephes.stdtr(1,0),0.5)
        assert_almost_equal(cephes.stdtr(1,1), 0.75)
        assert_almost_equal(cephes.stdtr(1,2), 0.852416382349)

    def test_stdtridf(self):
        cephes.stdtridf(0.7,1)

    def test_stdtrit(self):
        cephes.stdtrit(1,0.7)

    def test_struve(self):
        assert_equal(cephes.struve(0,0),0.0)

    def test_tandg(self):
        assert_equal(cephes.tandg(45),1.0)

    def test_tklmbda(self):
        assert_almost_equal(cephes.tklmbda(1,1),1.0)

    def test_y0(self):
        cephes.y0(1)

    def test_y1(self):
        cephes.y1(1)

    def test_yn(self):
        cephes.yn(1,1)

    def test_yv(self):
        cephes.yv(1,1)

    def _check_yve(self):
        cephes.yve(1,1)

    def test_zeta(self):
        assert_allclose(zeta(2,2), pi**2/6 - 1, rtol=1e-12)

    def test_zetac(self):
        assert_equal(cephes.zetac(0),-1.5)

    def test_zeta_1arg(self):
        assert_allclose(zeta(2), pi**2/6, rtol=1e-12)
        assert_allclose(zeta(4), pi**4/90, rtol=1e-12)

    def test_wofz(self):
        z = [complex(624.2,-0.26123), complex(-0.4,3.), complex(0.6,2.),
             complex(-1.,1.), complex(-1.,-9.), complex(-1.,9.),
             complex(-0.0000000234545,1.1234), complex(-3.,5.1),
             complex(-53,30.1), complex(0.0,0.12345),
             complex(11,1), complex(-22,-2), complex(9,-28),
             complex(21,-33), complex(1e5,1e5), complex(1e14,1e14)
             ]
        w = [
            complex(-3.78270245518980507452677445620103199303131110e-7,
                    0.000903861276433172057331093754199933411710053155),
            complex(0.1764906227004816847297495349730234591778719532788,
                    -0.02146550539468457616788719893991501311573031095617),
            complex(0.2410250715772692146133539023007113781272362309451,
                    0.06087579663428089745895459735240964093522265589350),
            complex(0.30474420525691259245713884106959496013413834051768,
                    -0.20821893820283162728743734725471561394145872072738),
            complex(7.317131068972378096865595229600561710140617977e34,
                    8.321873499714402777186848353320412813066170427e34),
            complex(0.0615698507236323685519612934241429530190806818395,
                    -0.00676005783716575013073036218018565206070072304635),
            complex(0.3960793007699874918961319170187598400134746631,
                    -5.593152259116644920546186222529802777409274656e-9),
            complex(0.08217199226739447943295069917990417630675021771804,
                    -0.04701291087643609891018366143118110965272615832184),
            complex(0.00457246000350281640952328010227885008541748668738,
                    -0.00804900791411691821818731763401840373998654987934),
            complex(0.8746342859608052666092782112565360755791467973338452,
                    0.),
            complex(0.00468190164965444174367477874864366058339647648741,
                    0.0510735563901306197993676329845149741675029197050),
            complex(-0.0023193175200187620902125853834909543869428763219,
                    -0.025460054739731556004902057663500272721780776336),
            complex(9.11463368405637174660562096516414499772662584e304,
                    3.97101807145263333769664875189354358563218932e305),
            complex(-4.4927207857715598976165541011143706155432296e281,
                    -2.8019591213423077494444700357168707775769028e281),
            complex(2.820947917809305132678577516325951485807107151e-6,
                    2.820947917668257736791638444590253942253354058e-6),
            complex(2.82094791773878143474039725787438662716372268e-15,
                    2.82094791773878143474039725773333923127678361e-15)
        ]
        assert_func_equal(cephes.wofz, w, z, rtol=1e-13)


class TestAiry(TestCase):
    def test_airy(self):
        # This tests the airy function to ensure 8 place accuracy in computation

        x = special.airy(.99)
        assert_array_almost_equal(x,array([0.13689066,-0.16050153,1.19815925,0.92046818]),8)
        x = special.airy(.41)
        assert_array_almost_equal(x,array([0.25238916,-.23480512,0.80686202,0.51053919]),8)
        x = special.airy(-.36)
        assert_array_almost_equal(x,array([0.44508477,-0.23186773,0.44939534,0.48105354]),8)

    def test_airye(self):
        a = special.airye(0.01)
        b = special.airy(0.01)
        b1 = [None]*4
        for n in range(2):
            b1[n] = b[n]*exp(2.0/3.0*0.01*sqrt(0.01))
        for n in range(2,4):
            b1[n] = b[n]*exp(-abs(real(2.0/3.0*0.01*sqrt(0.01))))
        assert_array_almost_equal(a,b1,6)

    def test_bi_zeros(self):
        bi = special.bi_zeros(2)
        bia = (array([-1.17371322, -3.2710930]),
               array([-2.29443968, -4.07315509]),
               array([-0.45494438, 0.39652284]),
               array([0.60195789, -0.76031014]))
        assert_array_almost_equal(bi,bia,4)

        bi = special.bi_zeros(5)
        assert_array_almost_equal(bi[0],array([-1.173713222709127,
                                               -3.271093302836352,
                                               -4.830737841662016,
                                               -6.169852128310251,
                                               -7.376762079367764]),11)

        assert_array_almost_equal(bi[1],array([-2.294439682614122,
                                               -4.073155089071828,
                                               -5.512395729663599,
                                               -6.781294445990305,
                                               -7.940178689168587]),10)

        assert_array_almost_equal(bi[2],array([-0.454944383639657,
                                               0.396522836094465,
                                               -0.367969161486959,
                                               0.349499116831805,
                                               -0.336026240133662]),11)

        assert_array_almost_equal(bi[3],array([0.601957887976239,
                                               -0.760310141492801,
                                               0.836991012619261,
                                               -0.88947990142654,
                                               0.929983638568022]),10)

    def test_ai_zeros(self):
        ai = special.ai_zeros(1)
        assert_array_almost_equal(ai,(array([-2.33810741]),
                                     array([-1.01879297]),
                                     array([0.5357]),
                                     array([0.7012])),4)

    def test_ai_zeros_big(self):
        z, zp, ai_zpx, aip_zx = special.ai_zeros(50000)
        ai_z, aip_z, _, _ = special.airy(z)
        ai_zp, aip_zp, _, _ = special.airy(zp)

        ai_envelope = 1/abs(z)**(1./4)
        aip_envelope = abs(zp)**(1./4)

        # Check values
        assert_allclose(ai_zpx, ai_zp, rtol=1e-10)
        assert_allclose(aip_zx, aip_z, rtol=1e-10)

        # Check they are zeros
        assert_allclose(ai_z/ai_envelope, 0, atol=1e-10, rtol=0)
        assert_allclose(aip_zp/aip_envelope, 0, atol=1e-10, rtol=0)

        # Check first zeros, DLMF 9.9.1
        assert_allclose(z[:6],
            [-2.3381074105, -4.0879494441, -5.5205598281,
             -6.7867080901, -7.9441335871, -9.0226508533], rtol=1e-10)
        assert_allclose(zp[:6],
            [-1.0187929716, -3.2481975822, -4.8200992112,
             -6.1633073556, -7.3721772550, -8.4884867340], rtol=1e-10)

    def test_bi_zeros_big(self):
        z, zp, bi_zpx, bip_zx = special.bi_zeros(50000)
        _, _, bi_z, bip_z = special.airy(z)
        _, _, bi_zp, bip_zp = special.airy(zp)

        bi_envelope = 1/abs(z)**(1./4)
        bip_envelope = abs(zp)**(1./4)

        # Check values
        assert_allclose(bi_zpx, bi_zp, rtol=1e-10)
        assert_allclose(bip_zx, bip_z, rtol=1e-10)

        # Check they are zeros
        assert_allclose(bi_z/bi_envelope, 0, atol=1e-10, rtol=0)
        assert_allclose(bip_zp/bip_envelope, 0, atol=1e-10, rtol=0)

        # Check first zeros, DLMF 9.9.2
        assert_allclose(z[:6],
            [-1.1737132227, -3.2710933028, -4.8307378417,
             -6.1698521283, -7.3767620794, -8.4919488465], rtol=1e-10)
        assert_allclose(zp[:6],
            [-2.2944396826, -4.0731550891, -5.5123957297,
             -6.7812944460, -7.9401786892, -9.0195833588], rtol=1e-10)


class TestAssocLaguerre(TestCase):
    def test_assoc_laguerre(self):
        a1 = special.genlaguerre(11,1)
        a2 = special.assoc_laguerre(.2,11,1)
        assert_array_almost_equal(a2,a1(.2),8)
        a2 = special.assoc_laguerre(1,11,1)
        assert_array_almost_equal(a2,a1(1),8)


class TestBesselpoly(TestCase):
    def test_besselpoly(self):
        pass


class TestKelvin(TestCase):
    def test_bei(self):
        mbei = special.bei(2)
        assert_almost_equal(mbei, 0.9722916273066613,5)  # this may not be exact

    def test_beip(self):
        mbeip = special.beip(2)
        assert_almost_equal(mbeip,0.91701361338403631,5)  # this may not be exact

    def test_ber(self):
        mber = special.ber(2)
        assert_almost_equal(mber,0.75173418271380821,5)  # this may not be exact

    def test_berp(self):
        mberp = special.berp(2)
        assert_almost_equal(mberp,-0.49306712470943909,5)  # this may not be exact

    def test_bei_zeros(self):
        # Abramowitz & Stegun, Table 9.12
        bi = special.bei_zeros(5)
        assert_array_almost_equal(bi,array([5.02622,
                                            9.45541,
                                            13.89349,
                                            18.33398,
                                            22.77544]),4)

    def test_beip_zeros(self):
        bip = special.beip_zeros(5)
        assert_array_almost_equal(bip,array([3.772673304934953,
                                               8.280987849760042,
                                               12.742147523633703,
                                               17.193431752512542,
                                               21.641143941167325]),8)

    def test_ber_zeros(self):
        ber = special.ber_zeros(5)
        assert_array_almost_equal(ber,array([2.84892,
                                             7.23883,
                                             11.67396,
                                             16.11356,
                                             20.55463]),4)

    def test_berp_zeros(self):
        brp = special.berp_zeros(5)
        assert_array_almost_equal(brp,array([6.03871,
                                             10.51364,
                                             14.96844,
                                             19.41758,
                                             23.86430]),4)

    def test_kelvin(self):
        mkelv = special.kelvin(2)
        assert_array_almost_equal(mkelv,(special.ber(2) + special.bei(2)*1j,
                                         special.ker(2) + special.kei(2)*1j,
                                         special.berp(2) + special.beip(2)*1j,
                                         special.kerp(2) + special.keip(2)*1j),8)

    def test_kei(self):
        mkei = special.kei(2)
        assert_almost_equal(mkei,-0.20240006776470432,5)

    def test_keip(self):
        mkeip = special.keip(2)
        assert_almost_equal(mkeip,0.21980790991960536,5)

    def test_ker(self):
        mker = special.ker(2)
        assert_almost_equal(mker,-0.041664513991509472,5)

    def test_kerp(self):
        mkerp = special.kerp(2)
        assert_almost_equal(mkerp,-0.10660096588105264,5)

    def test_kei_zeros(self):
        kei = special.kei_zeros(5)
        assert_array_almost_equal(kei,array([3.91467,
                                              8.34422,
                                              12.78256,
                                              17.22314,
                                              21.66464]),4)

    def test_keip_zeros(self):
        keip = special.keip_zeros(5)
        assert_array_almost_equal(keip,array([4.93181,
                                                9.40405,
                                                13.85827,
                                                18.30717,
                                                22.75379]),4)

    # numbers come from 9.9 of A&S pg. 381
    def test_kelvin_zeros(self):
        tmp = special.kelvin_zeros(5)
        berz,beiz,kerz,keiz,berpz,beipz,kerpz,keipz = tmp
        assert_array_almost_equal(berz,array([2.84892,
                                               7.23883,
                                               11.67396,
                                               16.11356,
                                               20.55463]),4)
        assert_array_almost_equal(beiz,array([5.02622,
                                               9.45541,
                                               13.89349,
                                               18.33398,
                                               22.77544]),4)
        assert_array_almost_equal(kerz,array([1.71854,
                                               6.12728,
                                               10.56294,
                                               15.00269,
                                               19.44382]),4)
        assert_array_almost_equal(keiz,array([3.91467,
                                               8.34422,
                                               12.78256,
                                               17.22314,
                                               21.66464]),4)
        assert_array_almost_equal(berpz,array([6.03871,
                                                10.51364,
                                                14.96844,
                                                19.41758,
                                                23.86430]),4)
        assert_array_almost_equal(beipz,array([3.77267,
                 # table from 1927 had 3.77320
                 #  but this is more accurate
                                                8.28099,
                                                12.74215,
                                                17.19343,
                                                21.64114]),4)
        assert_array_almost_equal(kerpz,array([2.66584,
                                                7.17212,
                                                11.63218,
                                                16.08312,
                                                20.53068]),4)
        assert_array_almost_equal(keipz,array([4.93181,
                                                9.40405,
                                                13.85827,
                                                18.30717,
                                                22.75379]),4)

    def test_ker_zeros(self):
        ker = special.ker_zeros(5)
        assert_array_almost_equal(ker,array([1.71854,
                                               6.12728,
                                               10.56294,
                                               15.00269,
                                               19.44381]),4)

    def test_kerp_zeros(self):
        kerp = special.kerp_zeros(5)
        assert_array_almost_equal(kerp,array([2.66584,
                                                7.17212,
                                                11.63218,
                                                16.08312,
                                                20.53068]),4)


class TestBernoulli(TestCase):
    def test_bernoulli(self):
        brn = special.bernoulli(5)
        assert_array_almost_equal(brn,array([1.0000,
                                             -0.5000,
                                             0.1667,
                                             0.0000,
                                             -0.0333,
                                             0.0000]),4)


class TestBeta(TestCase):
    def test_beta(self):
        bet = special.beta(2,4)
        betg = (special.gamma(2)*special.gamma(4))/special.gamma(6)
        assert_almost_equal(bet,betg,8)

    def test_betaln(self):
        betln = special.betaln(2,4)
        bet = log(abs(special.beta(2,4)))
        assert_almost_equal(betln,bet,8)

    def test_betainc(self):
        btinc = special.betainc(1,1,.2)
        assert_almost_equal(btinc,0.2,8)

    def test_betaincinv(self):
        y = special.betaincinv(2,4,.5)
        comp = special.betainc(2,4,y)
        assert_almost_equal(comp,.5,5)


class TestCombinatorics(TestCase):
    def test_comb(self):
        assert_array_almost_equal(special.comb([10, 10], [3, 4]), [120., 210.])
        assert_almost_equal(special.comb(10, 3), 120.)
        assert_equal(special.comb(10, 3, exact=True), 120)
        assert_equal(special.comb(10, 3, exact=True, repetition=True), 220)

        assert_allclose([special.comb(20, k, exact=True) for k in range(21)],
                        special.comb(20, list(range(21))), atol=1e-15)

        ii = np.iinfo(int).max + 1
        assert_equal(special.comb(ii, ii-1, exact=True), ii)

        expected = 100891344545564193334812497256
        assert_equal(special.comb(100, 50, exact=True), expected)

    def test_comb_with_np_int64(self):
        n = 70
        k = 30
        np_n = np.int64(n)
        np_k = np.int64(k)
        assert_equal(special.comb(np_n, np_k, exact=True),
                     special.comb(n, k, exact=True))

    def test_comb_zeros(self):
        assert_equal(special.comb(2, 3, exact=True), 0)
        assert_equal(special.comb(-1, 3, exact=True), 0)
        assert_equal(special.comb(2, -1, exact=True), 0)
        assert_equal(special.comb(2, -1, exact=False), 0)
        assert_array_almost_equal(special.comb([2, -1, 2, 10], [3, 3, -1, 3]),
                [0., 0., 0., 120.])

    def test_perm(self):
        assert_array_almost_equal(special.perm([10, 10], [3, 4]), [720., 5040.])
        assert_almost_equal(special.perm(10, 3), 720.)
        assert_equal(special.perm(10, 3, exact=True), 720)

    def test_perm_zeros(self):
        assert_equal(special.perm(2, 3, exact=True), 0)
        assert_equal(special.perm(-1, 3, exact=True), 0)
        assert_equal(special.perm(2, -1, exact=True), 0)
        assert_equal(special.perm(2, -1, exact=False), 0)
        assert_array_almost_equal(special.perm([2, -1, 2, 10], [3, 3, -1, 3]),
                [0., 0., 0., 720.])


class TestTrigonometric(TestCase):
    def test_cbrt(self):
        cb = special.cbrt(27)
        cbrl = 27**(1.0/3.0)
        assert_approx_equal(cb,cbrl)

    def test_cbrtmore(self):
        cb1 = special.cbrt(27.9)
        cbrl1 = 27.9**(1.0/3.0)
        assert_almost_equal(cb1,cbrl1,8)

    def test_cosdg(self):
        cdg = special.cosdg(90)
        cdgrl = cos(pi/2.0)
        assert_almost_equal(cdg,cdgrl,8)

    def test_cosdgmore(self):
        cdgm = special.cosdg(30)
        cdgmrl = cos(pi/6.0)
        assert_almost_equal(cdgm,cdgmrl,8)

    def test_cosm1(self):
        cs = (special.cosm1(0),special.cosm1(.3),special.cosm1(pi/10))
        csrl = (cos(0)-1,cos(.3)-1,cos(pi/10)-1)
        assert_array_almost_equal(cs,csrl,8)

    def test_cotdg(self):
        ct = special.cotdg(30)
        ctrl = tan(pi/6.0)**(-1)
        assert_almost_equal(ct,ctrl,8)

    def test_cotdgmore(self):
        ct1 = special.cotdg(45)
        ctrl1 = tan(pi/4.0)**(-1)
        assert_almost_equal(ct1,ctrl1,8)

    def test_specialpoints(self):
        assert_almost_equal(special.cotdg(45), 1.0, 14)
        assert_almost_equal(special.cotdg(-45), -1.0, 14)
        assert_almost_equal(special.cotdg(90), 0.0, 14)
        assert_almost_equal(special.cotdg(-90), 0.0, 14)
        assert_almost_equal(special.cotdg(135), -1.0, 14)
        assert_almost_equal(special.cotdg(-135), 1.0, 14)
        assert_almost_equal(special.cotdg(225), 1.0, 14)
        assert_almost_equal(special.cotdg(-225), -1.0, 14)
        assert_almost_equal(special.cotdg(270), 0.0, 14)
        assert_almost_equal(special.cotdg(-270), 0.0, 14)
        assert_almost_equal(special.cotdg(315), -1.0, 14)
        assert_almost_equal(special.cotdg(-315), 1.0, 14)
        assert_almost_equal(special.cotdg(765), 1.0, 14)

    def test_sinc(self):
        # the sinc implementation and more extensive sinc tests are in numpy
        assert_array_equal(special.sinc([0]), 1)
        assert_equal(special.sinc(0.0), 1.0)

    def test_sindg(self):
        sn = special.sindg(90)
        assert_equal(sn,1.0)

    def test_sindgmore(self):
        snm = special.sindg(30)
        snmrl = sin(pi/6.0)
        assert_almost_equal(snm,snmrl,8)
        snm1 = special.sindg(45)
        snmrl1 = sin(pi/4.0)
        assert_almost_equal(snm1,snmrl1,8)


class TestTandg(TestCase):

    def test_tandg(self):
        tn = special.tandg(30)
        tnrl = tan(pi/6.0)
        assert_almost_equal(tn,tnrl,8)

    def test_tandgmore(self):
        tnm = special.tandg(45)
        tnmrl = tan(pi/4.0)
        assert_almost_equal(tnm,tnmrl,8)
        tnm1 = special.tandg(60)
        tnmrl1 = tan(pi/3.0)
        assert_almost_equal(tnm1,tnmrl1,8)

    def test_specialpoints(self):
        assert_almost_equal(special.tandg(0), 0.0, 14)
        assert_almost_equal(special.tandg(45), 1.0, 14)
        assert_almost_equal(special.tandg(-45), -1.0, 14)
        assert_almost_equal(special.tandg(135), -1.0, 14)
        assert_almost_equal(special.tandg(-135), 1.0, 14)
        assert_almost_equal(special.tandg(180), 0.0, 14)
        assert_almost_equal(special.tandg(-180), 0.0, 14)
        assert_almost_equal(special.tandg(225), 1.0, 14)
        assert_almost_equal(special.tandg(-225), -1.0, 14)
        assert_almost_equal(special.tandg(315), -1.0, 14)
        assert_almost_equal(special.tandg(-315), 1.0, 14)


class TestEllip(TestCase):
    def test_ellipj_nan(self):
        """Regression test for #912."""
        special.ellipj(0.5, np.nan)

    def test_ellipj(self):
        el = special.ellipj(0.2,0)
        rel = [sin(0.2),cos(0.2),1.0,0.20]
        assert_array_almost_equal(el,rel,13)

    def test_ellipk(self):
        elk = special.ellipk(.2)
        assert_almost_equal(elk,1.659623598610528,11)

        assert_equal(special.ellipkm1(0.0), np.inf)
        assert_equal(special.ellipkm1(1.0), pi/2)
        assert_equal(special.ellipkm1(np.inf), 0.0)
        assert_equal(special.ellipkm1(np.nan), np.nan)
        assert_equal(special.ellipkm1(-1), np.nan)
        assert_allclose(special.ellipk(-10), 0.7908718902387385)

    def test_ellipkinc(self):
        elkinc = special.ellipkinc(pi/2,.2)
        elk = special.ellipk(0.2)
        assert_almost_equal(elkinc,elk,15)
        alpha = 20*pi/180
        phi = 45*pi/180
        m = sin(alpha)**2
        elkinc = special.ellipkinc(phi,m)
        assert_almost_equal(elkinc,0.79398143,8)
        # From pg. 614 of A & S

        assert_equal(special.ellipkinc(pi/2, 0.0), pi/2)
        assert_equal(special.ellipkinc(pi/2, 1.0), np.inf)
        assert_equal(special.ellipkinc(pi/2, -np.inf), 0.0)
        assert_equal(special.ellipkinc(pi/2, np.nan), np.nan)
        assert_equal(special.ellipkinc(pi/2, 2), np.nan)
        assert_equal(special.ellipkinc(0, 0.5), 0.0)
        assert_equal(special.ellipkinc(np.inf, 0.5), np.inf)
        assert_equal(special.ellipkinc(-np.inf, 0.5), -np.inf)
        assert_equal(special.ellipkinc(np.inf, np.inf), np.nan)
        assert_equal(special.ellipkinc(np.inf, -np.inf), np.nan)
        assert_equal(special.ellipkinc(-np.inf, -np.inf), np.nan)
        assert_equal(special.ellipkinc(-np.inf, np.inf), np.nan)
        assert_equal(special.ellipkinc(np.nan, 0.5), np.nan)
        assert_equal(special.ellipkinc(np.nan, np.nan), np.nan)

        assert_allclose(special.ellipkinc(0.38974112035318718, 1), 0.4, rtol=1e-14)
        assert_allclose(special.ellipkinc(1.5707, -10), 0.79084284661724946)

    def test_ellipkinc_2(self):
        # Regression test for gh-3550
        # ellipkinc(phi, mbad) was NaN and mvals[2:6] were twice the correct value
        mbad = 0.68359375000000011
        phi = 0.9272952180016123
        m = np.nextafter(mbad, 0)
        mvals = []
        for j in range(10):
            mvals.append(m)
            m = np.nextafter(m, 1)
        f = special.ellipkinc(phi, mvals)
        assert_array_almost_equal_nulp(f, 1.0259330100195334 * np.ones_like(f), 1)
        # this bug also appears at phi + n * pi for at least small n
        f1 = special.ellipkinc(phi + pi, mvals)
        assert_array_almost_equal_nulp(f1, 5.1296650500976675 * np.ones_like(f1), 2)

    def test_ellipkinc_singular(self):
        # ellipkinc(phi, 1) has closed form and is finite only for phi in (-pi/2, pi/2)
        xlog = np.logspace(-300, -17, 25)
        xlin = np.linspace(1e-17, 0.1, 25)
        xlin2 = np.linspace(0.1, pi/2, 25, endpoint=False)

        assert_allclose(special.ellipkinc(xlog, 1), np.arcsinh(np.tan(xlog)), rtol=1e14)
        assert_allclose(special.ellipkinc(xlin, 1), np.arcsinh(np.tan(xlin)), rtol=1e14)
        assert_allclose(special.ellipkinc(xlin2, 1), np.arcsinh(np.tan(xlin2)), rtol=1e14)
        assert_equal(special.ellipkinc(np.pi/2, 1), np.inf)
        assert_allclose(special.ellipkinc(-xlog, 1), np.arcsinh(np.tan(-xlog)), rtol=1e14)
        assert_allclose(special.ellipkinc(-xlin, 1), np.arcsinh(np.tan(-xlin)), rtol=1e14)
        assert_allclose(special.ellipkinc(-xlin2, 1), np.arcsinh(np.tan(-xlin2)), rtol=1e14)
        assert_equal(special.ellipkinc(-np.pi/2, 1), np.inf)

    def test_ellipe(self):
        ele = special.ellipe(.2)
        assert_almost_equal(ele,1.4890350580958529,8)

        assert_equal(special.ellipe(0.0), pi/2)
        assert_equal(special.ellipe(1.0), 1.0)
        assert_equal(special.ellipe(-np.inf), np.inf)
        assert_equal(special.ellipe(np.nan), np.nan)
        assert_equal(special.ellipe(2), np.nan)
        assert_allclose(special.ellipe(-10), 3.6391380384177689)

    def test_ellipeinc(self):
        eleinc = special.ellipeinc(pi/2,.2)
        ele = special.ellipe(0.2)
        assert_almost_equal(eleinc,ele,14)
        # pg 617 of A & S
        alpha, phi = 52*pi/180,35*pi/180
        m = sin(alpha)**2
        eleinc = special.ellipeinc(phi,m)
        assert_almost_equal(eleinc, 0.58823065, 8)

        assert_equal(special.ellipeinc(pi/2, 0.0), pi/2)
        assert_equal(special.ellipeinc(pi/2, 1.0), 1.0)
        assert_equal(special.ellipeinc(pi/2, -np.inf), np.inf)
        assert_equal(special.ellipeinc(pi/2, np.nan), np.nan)
        assert_equal(special.ellipeinc(pi/2, 2), np.nan)
        assert_equal(special.ellipeinc(0, 0.5), 0.0)
        assert_equal(special.ellipeinc(np.inf, 0.5), np.inf)
        assert_equal(special.ellipeinc(-np.inf, 0.5), -np.inf)
        assert_equal(special.ellipeinc(np.inf, -np.inf), np.inf)
        assert_equal(special.ellipeinc(-np.inf, -np.inf), -np.inf)
        assert_equal(special.ellipeinc(np.inf, np.inf), np.nan)
        assert_equal(special.ellipeinc(-np.inf, np.inf), np.nan)
        assert_equal(special.ellipeinc(np.nan, 0.5), np.nan)
        assert_equal(special.ellipeinc(np.nan, np.nan), np.nan)
        assert_allclose(special.ellipeinc(1.5707, -10), 3.6388185585822876)

    def test_ellipeinc_2(self):
        # Regression test for gh-3550
        # ellipeinc(phi, mbad) was NaN and mvals[2:6] were twice the correct value
        mbad = 0.68359375000000011
        phi = 0.9272952180016123
        m = np.nextafter(mbad, 0)
        mvals = []
        for j in range(10):
            mvals.append(m)
            m = np.nextafter(m, 1)
        f = special.ellipeinc(phi, mvals)
        assert_array_almost_equal_nulp(f, 0.84442884574781019 * np.ones_like(f), 2)
        # this bug also appears at phi + n * pi for at least small n
        f1 = special.ellipeinc(phi + pi, mvals)
        assert_array_almost_equal_nulp(f1, 3.3471442287390509 * np.ones_like(f1), 4)


class TestErf(TestCase):

    def test_erf(self):
        er = special.erf(.25)
        assert_almost_equal(er,0.2763263902,8)

    def test_erf_zeros(self):
        erz = special.erf_zeros(5)
        erzr = array([1.45061616+1.88094300j,
                     2.24465928+2.61657514j,
                     2.83974105+3.17562810j,
                     3.33546074+3.64617438j,
                     3.76900557+4.06069723j])
        assert_array_almost_equal(erz,erzr,4)

    def _check_variant_func(self, func, other_func, rtol, atol=0):
        np.random.seed(1234)
        n = 10000
        x = np.random.pareto(0.02, n) * (2*np.random.randint(0, 2, n) - 1)
        y = np.random.pareto(0.02, n) * (2*np.random.randint(0, 2, n) - 1)
        z = x + 1j*y

        old_errors = np.seterr(all='ignore')
        try:
            w = other_func(z)
            w_real = other_func(x).real

            mask = np.isfinite(w)
            w = w[mask]
            z = z[mask]

            mask = np.isfinite(w_real)
            w_real = w_real[mask]
            x = x[mask]

            # test both real and complex variants
            assert_func_equal(func, w, z, rtol=rtol, atol=atol)
            assert_func_equal(func, w_real, x, rtol=rtol, atol=atol)
        finally:
            np.seterr(**old_errors)

    def test_erfc_consistent(self):
        self._check_variant_func(
            cephes.erfc,
            lambda z: 1 - cephes.erf(z),
            rtol=1e-12,
            atol=1e-14  # <- the test function loses precision
            )

    def test_erfcx_consistent(self):
        self._check_variant_func(
            cephes.erfcx,
            lambda z: np.exp(z*z) * cephes.erfc(z),
            rtol=1e-12
            )

    def test_erfi_consistent(self):
        self._check_variant_func(
            cephes.erfi,
            lambda z: -1j * cephes.erf(1j*z),
            rtol=1e-12
            )

    def test_dawsn_consistent(self):
        self._check_variant_func(
            cephes.dawsn,
            lambda z: sqrt(pi)/2 * np.exp(-z*z) * cephes.erfi(z),
            rtol=1e-12
            )

    def test_erfcinv(self):
        i = special.erfcinv(1)
        # Use assert_array_equal instead of assert_equal, so the comparsion
        # of -0.0 and 0.0 doesn't fail.
        assert_array_equal(i, 0)

    def test_erfinv(self):
        i = special.erfinv(0)
        assert_equal(i,0)

    def test_errprint(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = special.errprint()
            b = 1-a  # a is the state 1-a inverts state
            c = special.errprint(b)  # returns last state 'a'
            assert_equal(a,c)
            d = special.errprint(a)  # returns to original state
            assert_equal(d,b)  # makes sure state was returned
            # assert_equal(d,1-a)

    def test_erf_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, -1, 1]
        assert_allclose(special.erf(vals), expected, rtol=1e-15)

    def test_erfc_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, 2, 0]
        assert_allclose(special.erfc(vals), expected, rtol=1e-15)

    def test_erfcx_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, np.inf, 0]
        assert_allclose(special.erfcx(vals), expected, rtol=1e-15)

    def test_erfi_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, -np.inf, np.inf]
        assert_allclose(special.erfi(vals), expected, rtol=1e-15)

    def test_dawsn_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, -0.0, 0.0]
        assert_allclose(special.dawsn(vals), expected, rtol=1e-15)

    def test_wofz_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan + np.nan * 1.j, 0.-0.j, 0.+0.j]
        assert_allclose(special.wofz(vals), expected, rtol=1e-15)


class TestEuler(TestCase):
    def test_euler(self):
        eu0 = special.euler(0)
        eu1 = special.euler(1)
        eu2 = special.euler(2)   # just checking segfaults
        assert_allclose(eu0, [1], rtol=1e-15)
        assert_allclose(eu1, [1, 0], rtol=1e-15)
        assert_allclose(eu2, [1, 0, -1], rtol=1e-15)
        eu24 = special.euler(24)
        mathworld = [1,1,5,61,1385,50521,2702765,199360981,
                     19391512145,2404879675441,
                     370371188237525,69348874393137901,
                     15514534163557086905]
        correct = zeros((25,),'d')
        for k in range(0,13):
            if (k % 2):
                correct[2*k] = -float(mathworld[k])
            else:
                correct[2*k] = float(mathworld[k])
        olderr = np.seterr(all='ignore')
        try:
            err = nan_to_num((eu24-correct)/correct)
            errmax = max(err)
        finally:
            np.seterr(**olderr)
        assert_almost_equal(errmax, 0.0, 14)


class TestExp(TestCase):
    def test_exp2(self):
        ex = special.exp2(2)
        exrl = 2**2
        assert_equal(ex,exrl)

    def test_exp2more(self):
        exm = special.exp2(2.5)
        exmrl = 2**(2.5)
        assert_almost_equal(exm,exmrl,8)

    def test_exp10(self):
        ex = special.exp10(2)
        exrl = 10**2
        assert_approx_equal(ex,exrl)

    def test_exp10more(self):
        exm = special.exp10(2.5)
        exmrl = 10**(2.5)
        assert_almost_equal(exm,exmrl,8)

    def test_expm1(self):
        ex = (special.expm1(2),special.expm1(3),special.expm1(4))
        exrl = (exp(2)-1,exp(3)-1,exp(4)-1)
        assert_array_almost_equal(ex,exrl,8)

    def test_expm1more(self):
        ex1 = (special.expm1(2),special.expm1(2.1),special.expm1(2.2))
        exrl1 = (exp(2)-1,exp(2.1)-1,exp(2.2)-1)
        assert_array_almost_equal(ex1,exrl1,8)


class TestFactorialFunctions(TestCase):
    def test_factorial(self):
        # Some known values, float math
        assert_array_almost_equal(special.factorial(0), 1)
        assert_array_almost_equal(special.factorial(1), 1)
        assert_array_almost_equal(special.factorial(2), 2)
        assert_array_almost_equal([6., 24., 120.],
                                  special.factorial([3, 4, 5], exact=False))
        assert_array_almost_equal(special.factorial([[5, 3], [4, 3]]),
                                  [[120, 6], [24, 6]])

        # Some known values, integer math
        assert_equal(special.factorial(0, exact=True), 1)
        assert_equal(special.factorial(1, exact=True), 1)
        assert_equal(special.factorial(2, exact=True), 2)
        assert_equal(special.factorial(5, exact=True), 120)
        assert_equal(special.factorial(15, exact=True), 1307674368000)

        # ndarray shape is maintained
        assert_equal(special.factorial([7, 4, 15, 10], exact=True),
                     [5040, 24, 1307674368000, 3628800])

        assert_equal(special.factorial([[5, 3], [4, 3]], True),
                     [[120, 6], [24, 6]])

        # object arrays
        assert_equal(special.factorial(np.arange(-3, 22), True),
                     special.factorial(np.arange(-3, 22), False))

        # int64 array
        assert_equal(special.factorial(np.arange(-3, 15), True),
                     special.factorial(np.arange(-3, 15), False))

        # int32 array
        assert_equal(special.factorial(np.arange(-3, 5), True),
                     special.factorial(np.arange(-3, 5), False))

        # Consistent output for n < 0
        for exact in (True, False):
            assert_array_equal(0, special.factorial(-3, exact))
            assert_array_equal([1, 2, 0, 0],
                               special.factorial([1, 2, -5, -4], exact))

        for n in range(0, 22):
            # Compare all with math.factorial
            correct = math.factorial(n)
            assert_array_equal(correct, special.factorial(n, True))
            assert_array_equal(correct, special.factorial([n], True)[0])

            assert_allclose(float(correct), special.factorial(n, False))
            assert_allclose(float(correct), special.factorial([n], False)[0])

            # Compare exact=True vs False, scalar vs array
            assert_array_equal(special.factorial(n, True),
                               special.factorial(n, False))

            assert_array_equal(special.factorial([n], True),
                               special.factorial([n], False))

    def test_factorial2(self):
        assert_array_almost_equal([105., 384., 945.],
                                  special.factorial2([7, 8, 9], exact=False))
        assert_equal(special.factorial2(7, exact=True), 105)

    def test_factorialk(self):
        assert_equal(special.factorialk(5, 1, exact=True), 120)
        assert_equal(special.factorialk(5, 3, exact=True), 10)


class TestFresnel(TestCase):
    def test_fresnel(self):
        frs = array(special.fresnel(.5))
        assert_array_almost_equal(frs,array([0.064732432859999287, 0.49234422587144644]),8)

    def test_fresnel_inf1(self):
        frs = special.fresnel(np.inf)
        assert_equal(frs, (0.5, 0.5))

    def test_fresnel_inf2(self):
        frs = special.fresnel(-np.inf)
        assert_equal(frs, (-0.5, -0.5))

    # values from pg 329  Table 7.11 of A & S
    #  slightly corrected in 4th decimal place
    def test_fresnel_zeros(self):
        szo, czo = special.fresnel_zeros(5)
        assert_array_almost_equal(szo,
                                  array([2.0093+0.2885j,
                                          2.8335+0.2443j,
                                          3.4675+0.2185j,
                                          4.0026+0.2009j,
                                          4.4742+0.1877j]),3)
        assert_array_almost_equal(czo,
                                  array([1.7437+0.3057j,
                                          2.6515+0.2529j,
                                          3.3204+0.2240j,
                                          3.8757+0.2047j,
                                          4.3611+0.1907j]),3)
        vals1 = special.fresnel(szo)[0]
        vals2 = special.fresnel(czo)[1]
        assert_array_almost_equal(vals1,0,14)
        assert_array_almost_equal(vals2,0,14)

    def test_fresnelc_zeros(self):
        szo, czo = special.fresnel_zeros(6)
        frc = special.fresnelc_zeros(6)
        assert_array_almost_equal(frc,czo,12)

    def test_fresnels_zeros(self):
        szo, czo = special.fresnel_zeros(5)
        frs = special.fresnels_zeros(5)
        assert_array_almost_equal(frs,szo,12)


class TestGamma(TestCase):
    def test_gamma(self):
        gam = special.gamma(5)
        assert_equal(gam,24.0)

    def test_gammaln(self):
        gamln = special.gammaln(3)
        lngam = log(special.gamma(3))
        assert_almost_equal(gamln,lngam,8)

    def test_gammainc(self):
        gama = special.gammainc(.5,.5)
        assert_almost_equal(gama,.7,1)

    def test_gammaincnan(self):
        gama = special.gammainc(-1,1)
        assert_(isnan(gama))

    def test_gammainczero(self):
        # bad arg but zero integration limit
        gama = special.gammainc(-1,0)
        assert_equal(gama,0.0)

    def test_gammaincinf(self):
        gama = special.gammainc(0.5, np.inf)
        assert_equal(gama,1.0)

    def test_gammaincc(self):
        gicc = special.gammaincc(.5,.5)
        greal = 1 - special.gammainc(.5,.5)
        assert_almost_equal(gicc,greal,8)

    def test_gammainccnan(self):
        gama = special.gammaincc(-1,1)
        assert_(isnan(gama))

    def test_gammainccinf(self):
        gama = special.gammaincc(0.5,np.inf)
        assert_equal(gama,0.0)

    def test_gammainccinv(self):
        gccinv = special.gammainccinv(.5,.5)
        gcinv = special.gammaincinv(.5,.5)
        assert_almost_equal(gccinv,gcinv,8)

    @with_special_errors
    def test_gammaincinv(self):
        y = special.gammaincinv(.4,.4)
        x = special.gammainc(.4,y)
        assert_almost_equal(x,0.4,1)
        y = special.gammainc(10, 0.05)
        x = special.gammaincinv(10, 2.5715803516000736e-20)
        assert_almost_equal(0.05, x, decimal=10)
        assert_almost_equal(y, 2.5715803516000736e-20, decimal=10)
        x = special.gammaincinv(50, 8.20754777388471303050299243573393e-18)
        assert_almost_equal(11.0, x, decimal=10)

    @with_special_errors
    def test_975(self):
        # Regression test for ticket #975 -- switch point in algorithm
        # check that things work OK at the point, immediately next floats
        # around it, and a bit further away
        pts = [0.25,
               np.nextafter(0.25, 0), 0.25 - 1e-12,
               np.nextafter(0.25, 1), 0.25 + 1e-12]
        for xp in pts:
            y = special.gammaincinv(.4, xp)
            x = special.gammainc(0.4, y)
            assert_tol_equal(x, xp, rtol=1e-12)

    def test_rgamma(self):
        rgam = special.rgamma(8)
        rlgam = 1/special.gamma(8)
        assert_almost_equal(rgam,rlgam,8)

    def test_infinity(self):
        assert_(np.isinf(special.gamma(-1)))
        assert_equal(special.rgamma(-1), 0)


class TestHankel(TestCase):

    def test_negv1(self):
        assert_almost_equal(special.hankel1(-3,2), -special.hankel1(3,2), 14)

    def test_hankel1(self):
        hank1 = special.hankel1(1,.1)
        hankrl = (special.jv(1,.1) + special.yv(1,.1)*1j)
        assert_almost_equal(hank1,hankrl,8)

    def test_negv1e(self):
        assert_almost_equal(special.hankel1e(-3,2), -special.hankel1e(3,2), 14)

    def test_hankel1e(self):
        hank1e = special.hankel1e(1,.1)
        hankrle = special.hankel1(1,.1)*exp(-.1j)
        assert_almost_equal(hank1e,hankrle,8)

    def test_negv2(self):
        assert_almost_equal(special.hankel2(-3,2), -special.hankel2(3,2), 14)

    def test_hankel2(self):
        hank2 = special.hankel2(1,.1)
        hankrl2 = (special.jv(1,.1) - special.yv(1,.1)*1j)
        assert_almost_equal(hank2,hankrl2,8)

    def test_neg2e(self):
        assert_almost_equal(special.hankel2e(-3,2), -special.hankel2e(3,2), 14)

    def test_hankl2e(self):
        hank2e = special.hankel2e(1,.1)
        hankrl2e = special.hankel2e(1,.1)
        assert_almost_equal(hank2e,hankrl2e,8)


class TestHyper(TestCase):
    def test_h1vp(self):
        h1 = special.h1vp(1,.1)
        h1real = (special.jvp(1,.1) + special.yvp(1,.1)*1j)
        assert_almost_equal(h1,h1real,8)

    def test_h2vp(self):
        h2 = special.h2vp(1,.1)
        h2real = (special.jvp(1,.1) - special.yvp(1,.1)*1j)
        assert_almost_equal(h2,h2real,8)

    def test_hyp0f1(self):
        # scalar input
        assert_allclose(special.hyp0f1(2.5, 0.5), 1.21482702689997, rtol=1e-12)
        assert_allclose(special.hyp0f1(2.5, 0), 1.0, rtol=1e-15)

        # float input, expected values match mpmath
        x = special.hyp0f1(3.0, [-1.5, -1, 0, 1, 1.5])
        expected = np.array([0.58493659229143, 0.70566805723127, 1.0,
                             1.37789689539747, 1.60373685288480])
        assert_allclose(x, expected, rtol=1e-12)

        # complex input
        x = special.hyp0f1(3.0, np.array([-1.5, -1, 0, 1, 1.5]) + 0.j)
        assert_allclose(x, expected.astype(complex), rtol=1e-12)

        # test broadcasting
        x1 = [0.5, 1.5, 2.5]
        x2 = [0, 1, 0.5]
        x = special.hyp0f1(x1, x2)
        expected = [1.0, 1.8134302039235093, 1.21482702689997]
        assert_allclose(x, expected, rtol=1e-12)
        x = special.hyp0f1(np.row_stack([x1] * 2), x2)
        assert_allclose(x, np.row_stack([expected] * 2), rtol=1e-12)
        assert_raises(ValueError, special.hyp0f1,
                      np.row_stack([x1] * 3), [0, 1])

    def test_hyp0f1_gh5764(self):
        # Just checks the point that failed; there's a more systematic
        # test in test_mpmath
        res = special.hyp0f1(0.8, 0.5 + 0.5*1J)
        # The expected value was generated using mpmath
        assert_almost_equal(res, 1.6139719776441115 + 1J*0.80893054061790665)

    def test_hyp1f1(self):
        hyp1 = special.hyp1f1(.1,.1,.3)
        assert_almost_equal(hyp1, 1.3498588075760032,7)

        # test contributed by Moritz Deger (2008-05-29)
        # http://projects.scipy.org/scipy/scipy/ticket/659

        # reference data obtained from mathematica [ a, b, x, m(a,b,x)]:
        # produced with test_hyp1f1.nb
        ref_data = array([[-8.38132975e+00, -1.28436461e+01, -2.91081397e+01, 1.04178330e+04],
                          [2.91076882e+00, -6.35234333e+00, -1.27083993e+01, 6.68132725e+00],
                          [-1.42938258e+01, 1.80869131e-01, 1.90038728e+01, 1.01385897e+05],
                          [5.84069088e+00, 1.33187908e+01, 2.91290106e+01, 1.59469411e+08],
                          [-2.70433202e+01, -1.16274873e+01, -2.89582384e+01, 1.39900152e+24],
                          [4.26344966e+00, -2.32701773e+01, 1.91635759e+01, 6.13816915e+21],
                          [1.20514340e+01, -3.40260240e+00, 7.26832235e+00, 1.17696112e+13],
                          [2.77372955e+01, -1.99424687e+00, 3.61332246e+00, 3.07419615e+13],
                          [1.50310939e+01, -2.91198675e+01, -1.53581080e+01, -3.79166033e+02],
                          [1.43995827e+01, 9.84311196e+00, 1.93204553e+01, 2.55836264e+10],
                          [-4.08759686e+00, 1.34437025e+01, -1.42072843e+01, 1.70778449e+01],
                          [8.05595738e+00, -1.31019838e+01, 1.52180721e+01, 3.06233294e+21],
                          [1.81815804e+01, -1.42908793e+01, 9.57868793e+00, -2.84771348e+20],
                          [-2.49671396e+01, 1.25082843e+01, -1.71562286e+01, 2.36290426e+07],
                          [2.67277673e+01, 1.70315414e+01, 6.12701450e+00, 7.77917232e+03],
                          [2.49565476e+01, 2.91694684e+01, 6.29622660e+00, 2.35300027e+02],
                          [6.11924542e+00, -1.59943768e+00, 9.57009289e+00, 1.32906326e+11],
                          [-1.47863653e+01, 2.41691301e+01, -1.89981821e+01, 2.73064953e+03],
                          [2.24070483e+01, -2.93647433e+00, 8.19281432e+00, -6.42000372e+17],
                          [8.04042600e-01, 1.82710085e+01, -1.97814534e+01, 5.48372441e-01],
                          [1.39590390e+01, 1.97318686e+01, 2.37606635e+00, 5.51923681e+00],
                          [-4.66640483e+00, -2.00237930e+01, 7.40365095e+00, 4.50310752e+00],
                          [2.76821999e+01, -6.36563968e+00, 1.11533984e+01, -9.28725179e+23],
                          [-2.56764457e+01, 1.24544906e+00, 1.06407572e+01, 1.25922076e+01],
                          [3.20447808e+00, 1.30874383e+01, 2.26098014e+01, 2.03202059e+04],
                          [-1.24809647e+01, 4.15137113e+00, -2.92265700e+01, 2.39621411e+08],
                          [2.14778108e+01, -2.35162960e+00, -1.13758664e+01, 4.46882152e-01],
                          [-9.85469168e+00, -3.28157680e+00, 1.67447548e+01, -1.07342390e+07],
                          [1.08122310e+01, -2.47353236e+01, -1.15622349e+01, -2.91733796e+03],
                          [-2.67933347e+01, -3.39100709e+00, 2.56006986e+01, -5.29275382e+09],
                          [-8.60066776e+00, -8.02200924e+00, 1.07231926e+01, 1.33548320e+06],
                          [-1.01724238e-01, -1.18479709e+01, -2.55407104e+01, 1.55436570e+00],
                          [-3.93356771e+00, 2.11106818e+01, -2.57598485e+01, 2.13467840e+01],
                          [3.74750503e+00, 1.55687633e+01, -2.92841720e+01, 1.43873509e-02],
                          [6.99726781e+00, 2.69855571e+01, -1.63707771e+01, 3.08098673e-02],
                          [-2.31996011e+01, 3.47631054e+00, 9.75119815e-01, 1.79971073e-02],
                          [2.38951044e+01, -2.91460190e+01, -2.50774708e+00, 9.56934814e+00],
                          [1.52730825e+01, 5.77062507e+00, 1.21922003e+01, 1.32345307e+09],
                          [1.74673917e+01, 1.89723426e+01, 4.94903250e+00, 9.90859484e+01],
                          [1.88971241e+01, 2.86255413e+01, 5.52360109e-01, 1.44165360e+00],
                          [1.02002319e+01, -1.66855152e+01, -2.55426235e+01, 6.56481554e+02],
                          [-1.79474153e+01, 1.22210200e+01, -1.84058212e+01, 8.24041812e+05],
                          [-1.36147103e+01, 1.32365492e+00, -7.22375200e+00, 9.92446491e+05],
                          [7.57407832e+00, 2.59738234e+01, -1.34139168e+01, 3.64037761e-02],
                          [2.21110169e+00, 1.28012666e+01, 1.62529102e+01, 1.33433085e+02],
                          [-2.64297569e+01, -1.63176658e+01, -1.11642006e+01, -2.44797251e+13],
                          [-2.46622944e+01, -3.02147372e+00, 8.29159315e+00, -3.21799070e+05],
                          [-1.37215095e+01, -1.96680183e+01, 2.91940118e+01, 3.21457520e+12],
                          [-5.45566105e+00, 2.81292086e+01, 1.72548215e-01, 9.66973000e-01],
                          [-1.55751298e+00, -8.65703373e+00, 2.68622026e+01, -3.17190834e+16],
                          [2.45393609e+01, -2.70571903e+01, 1.96815505e+01, 1.80708004e+37],
                          [5.77482829e+00, 1.53203143e+01, 2.50534322e+01, 1.14304242e+06],
                          [-1.02626819e+01, 2.36887658e+01, -2.32152102e+01, 7.28965646e+02],
                          [-1.30833446e+00, -1.28310210e+01, 1.87275544e+01, -9.33487904e+12],
                          [5.83024676e+00, -1.49279672e+01, 2.44957538e+01, -7.61083070e+27],
                          [-2.03130747e+01, 2.59641715e+01, -2.06174328e+01, 4.54744859e+04],
                          [1.97684551e+01, -2.21410519e+01, -2.26728740e+01, 3.53113026e+06],
                          [2.73673444e+01, 2.64491725e+01, 1.57599882e+01, 1.07385118e+07],
                          [5.73287971e+00, 1.21111904e+01, 1.33080171e+01, 2.63220467e+03],
                          [-2.82751072e+01, 2.08605881e+01, 9.09838900e+00, -6.60957033e-07],
                          [1.87270691e+01, -1.74437016e+01, 1.52413599e+01, 6.59572851e+27],
                          [6.60681457e+00, -2.69449855e+00, 9.78972047e+00, -2.38587870e+12],
                          [1.20895561e+01, -2.51355765e+01, 2.30096101e+01, 7.58739886e+32],
                          [-2.44682278e+01, 2.10673441e+01, -1.36705538e+01, 4.54213550e+04],
                          [-4.50665152e+00, 3.72292059e+00, -4.83403707e+00, 2.68938214e+01],
                          [-7.46540049e+00, -1.08422222e+01, -1.72203805e+01, -2.09402162e+02],
                          [-2.00307551e+01, -7.50604431e+00, -2.78640020e+01, 4.15985444e+19],
                          [1.99890876e+01, 2.20677419e+01, -2.51301778e+01, 1.23840297e-09],
                          [2.03183823e+01, -7.66942559e+00, 2.10340070e+01, 1.46285095e+31],
                          [-2.90315825e+00, -2.55785967e+01, -9.58779316e+00, 2.65714264e-01],
                          [2.73960829e+01, -1.80097203e+01, -2.03070131e+00, 2.52908999e+02],
                          [-2.11708058e+01, -2.70304032e+01, 2.48257944e+01, 3.09027527e+08],
                          [2.21959758e+01, 4.00258675e+00, -1.62853977e+01, -9.16280090e-09],
                          [1.61661840e+01, -2.26845150e+01, 2.17226940e+01, -8.24774394e+33],
                          [-3.35030306e+00, 1.32670581e+00, 9.39711214e+00, -1.47303163e+01],
                          [7.23720726e+00, -2.29763909e+01, 2.34709682e+01, -9.20711735e+29],
                          [2.71013568e+01, 1.61951087e+01, -7.11388906e-01, 2.98750911e-01],
                          [8.40057933e+00, -7.49665220e+00, 2.95587388e+01, 6.59465635e+29],
                          [-1.51603423e+01, 1.94032322e+01, -7.60044357e+00, 1.05186941e+02],
                          [-8.83788031e+00, -2.72018313e+01, 1.88269907e+00, 1.81687019e+00],
                          [-1.87283712e+01, 5.87479570e+00, -1.91210203e+01, 2.52235612e+08],
                          [-5.61338513e-01, 2.69490237e+01, 1.16660111e-01, 9.97567783e-01],
                          [-5.44354025e+00, -1.26721408e+01, -4.66831036e+00, 1.06660735e-01],
                          [-2.18846497e+00, 2.33299566e+01, 9.62564397e+00, 3.03842061e-01],
                          [6.65661299e+00, -2.39048713e+01, 1.04191807e+01, 4.73700451e+13],
                          [-2.57298921e+01, -2.60811296e+01, 2.74398110e+01, -5.32566307e+11],
                          [-1.11431826e+01, -1.59420160e+01, -1.84880553e+01, -1.01514747e+02],
                          [6.50301931e+00, 2.59859051e+01, -2.33270137e+01, 1.22760500e-02],
                          [-1.94987891e+01, -2.62123262e+01, 3.90323225e+00, 1.71658894e+01],
                          [7.26164601e+00, -1.41469402e+01, 2.81499763e+01, -2.50068329e+31],
                          [-1.52424040e+01, 2.99719005e+01, -2.85753678e+01, 1.31906693e+04],
                          [5.24149291e+00, -1.72807223e+01, 2.22129493e+01, 2.50748475e+25],
                          [3.63207230e-01, -9.54120862e-02, -2.83874044e+01, 9.43854939e-01],
                          [-2.11326457e+00, -1.25707023e+01, 1.17172130e+00, 1.20812698e+00],
                          [2.48513582e+00, 1.03652647e+01, -1.84625148e+01, 6.47910997e-02],
                          [2.65395942e+01, 2.74794672e+01, 1.29413428e+01, 2.89306132e+05],
                          [-9.49445460e+00, 1.59930921e+01, -1.49596331e+01, 3.27574841e+02],
                          [-5.89173945e+00, 9.96742426e+00, 2.60318889e+01, -3.15842908e-01],
                          [-1.15387239e+01, -2.21433107e+01, -2.17686413e+01, 1.56724718e-01],
                          [-5.30592244e+00, -2.42752190e+01, 1.29734035e+00, 1.31985534e+00]])

        for a,b,c,expected in ref_data:
            result = special.hyp1f1(a,b,c)
            assert_(abs(expected - result)/expected < 1e-4)

    def test_hyp1f1_gh2957(self):
        hyp1 = special.hyp1f1(0.5, 1.5, -709.7827128933)
        hyp2 = special.hyp1f1(0.5, 1.5, -709.7827128934)
        assert_almost_equal(hyp1, hyp2, 12)

    def test_hyp1f1_gh2282(self):
        hyp = special.hyp1f1(0.5, 1.5, -1000)
        assert_almost_equal(hyp, 0.028024956081989643, 12)

    def test_hyp1f2(self):
        pass

    def test_hyp2f0(self):
        pass

    def test_hyp2f1(self):
        # a collection of special cases taken from AMS 55
        values = [[0.5, 1, 1.5, 0.2**2, 0.5/0.2*log((1+0.2)/(1-0.2))],
                  [0.5, 1, 1.5, -0.2**2, 1./0.2*arctan(0.2)],
                  [1, 1, 2, 0.2, -1/0.2*log(1-0.2)],
                  [3, 3.5, 1.5, 0.2**2,
                      0.5/0.2/(-5)*((1+0.2)**(-5)-(1-0.2)**(-5))],
                  [-3, 3, 0.5, sin(0.2)**2, cos(2*3*0.2)],
                  [3, 4, 8, 1, special.gamma(8)*special.gamma(8-4-3)/special.gamma(8-3)/special.gamma(8-4)],
                  [3, 2, 3-2+1, -1, 1./2**3*sqrt(pi) *
                      special.gamma(1+3-2)/special.gamma(1+0.5*3-2)/special.gamma(0.5+0.5*3)],
                  [5, 2, 5-2+1, -1, 1./2**5*sqrt(pi) *
                      special.gamma(1+5-2)/special.gamma(1+0.5*5-2)/special.gamma(0.5+0.5*5)],
                  [4, 0.5+4, 1.5-2*4, -1./3, (8./9)**(-2*4)*special.gamma(4./3) *
                      special.gamma(1.5-2*4)/special.gamma(3./2)/special.gamma(4./3-2*4)],
                  # and some others
                  # ticket #424
                  [1.5, -0.5, 1.0, -10.0, 4.1300097765277476484],
                  # negative integer a or b, with c-a-b integer and x > 0.9
                  [-2,3,1,0.95,0.715],
                  [2,-3,1,0.95,-0.007],
                  [-6,3,1,0.95,0.0000810625],
                  [2,-5,1,0.95,-0.000029375],
                  # huge negative integers
                  (10, -900, 10.5, 0.99, 1.91853705796607664803709475658e-24),
                  (10, -900, -10.5, 0.99, 3.54279200040355710199058559155e-18),
                  ]
        for i, (a, b, c, x, v) in enumerate(values):
            cv = special.hyp2f1(a, b, c, x)
            assert_almost_equal(cv, v, 8, err_msg='test #%d' % i)

    def test_hyp3f0(self):
        pass

    def test_hyperu(self):
        val1 = special.hyperu(1,0.1,100)
        assert_almost_equal(val1,0.0098153,7)
        a,b = [0.3,0.6,1.2,-2.7],[1.5,3.2,-0.4,-3.2]
        a,b = asarray(a), asarray(b)
        z = 0.5
        hypu = special.hyperu(a,b,z)
        hprl = (pi/sin(pi*b))*(special.hyp1f1(a,b,z) /
                               (special.gamma(1+a-b)*special.gamma(b)) -
                               z**(1-b)*special.hyp1f1(1+a-b,2-b,z)
                               / (special.gamma(a)*special.gamma(2-b)))
        assert_array_almost_equal(hypu,hprl,12)

    def test_hyperu_gh2287(self):
        assert_almost_equal(special.hyperu(1, 1.5, 20.2),
                            0.048360918656699191, 12)


class TestBessel(TestCase):
    def test_itj0y0(self):
        it0 = array(special.itj0y0(.2))
        assert_array_almost_equal(it0,array([0.19933433254006822, -0.34570883800412566]),8)

    def test_it2j0y0(self):
        it2 = array(special.it2j0y0(.2))
        assert_array_almost_equal(it2,array([0.0049937546274601858, -0.43423067011231614]),8)

    def test_negv_iv(self):
        assert_equal(special.iv(3,2), special.iv(-3,2))

    def test_j0(self):
        oz = special.j0(.1)
        ozr = special.jn(0,.1)
        assert_almost_equal(oz,ozr,8)

    def test_j1(self):
        o1 = special.j1(.1)
        o1r = special.jn(1,.1)
        assert_almost_equal(o1,o1r,8)

    def test_jn(self):
        jnnr = special.jn(1,.2)
        assert_almost_equal(jnnr,0.099500832639235995,8)

    def test_negv_jv(self):
        assert_almost_equal(special.jv(-3,2), -special.jv(3,2), 14)

    def test_jv(self):
        values = [[0, 0.1, 0.99750156206604002],
                  [2./3, 1e-8, 0.3239028506761532e-5],
                  [2./3, 1e-10, 0.1503423854873779e-6],
                  [3.1, 1e-10, 0.1711956265409013e-32],
                  [2./3, 4.0, -0.2325440850267039],
                  ]
        for i, (v, x, y) in enumerate(values):
            yc = special.jv(v, x)
            assert_almost_equal(yc, y, 8, err_msg='test #%d' % i)

    def test_negv_jve(self):
        assert_almost_equal(special.jve(-3,2), -special.jve(3,2), 14)

    def test_jve(self):
        jvexp = special.jve(1,.2)
        assert_almost_equal(jvexp,0.099500832639235995,8)
        jvexp1 = special.jve(1,.2+1j)
        z = .2+1j
        jvexpr = special.jv(1,z)*exp(-abs(z.imag))
        assert_almost_equal(jvexp1,jvexpr,8)

    def test_jn_zeros(self):
        jn0 = special.jn_zeros(0,5)
        jn1 = special.jn_zeros(1,5)
        assert_array_almost_equal(jn0,array([2.4048255577,
                                              5.5200781103,
                                              8.6537279129,
                                              11.7915344391,
                                              14.9309177086]),4)
        assert_array_almost_equal(jn1,array([3.83171,
                                              7.01559,
                                              10.17347,
                                              13.32369,
                                              16.47063]),4)

        jn102 = special.jn_zeros(102,5)
        assert_tol_equal(jn102, array([110.89174935992040343,
                                       117.83464175788308398,
                                       123.70194191713507279,
                                       129.02417238949092824,
                                       134.00114761868422559]), rtol=1e-13)

        jn301 = special.jn_zeros(301,5)
        assert_tol_equal(jn301, array([313.59097866698830153,
                                       323.21549776096288280,
                                       331.22338738656748796,
                                       338.39676338872084500,
                                       345.03284233056064157]), rtol=1e-13)

    def test_jn_zeros_slow(self):
        jn0 = special.jn_zeros(0, 300)
        assert_tol_equal(jn0[260-1], 816.02884495068867280, rtol=1e-13)
        assert_tol_equal(jn0[280-1], 878.86068707124422606, rtol=1e-13)
        assert_tol_equal(jn0[300-1], 941.69253065317954064, rtol=1e-13)

        jn10 = special.jn_zeros(10, 300)
        assert_tol_equal(jn10[260-1], 831.67668514305631151, rtol=1e-13)
        assert_tol_equal(jn10[280-1], 894.51275095371316931, rtol=1e-13)
        assert_tol_equal(jn10[300-1], 957.34826370866539775, rtol=1e-13)

        jn3010 = special.jn_zeros(3010,5)
        assert_tol_equal(jn3010, array([3036.86590780927,
                                        3057.06598526482,
                                        3073.66360690272,
                                        3088.37736494778,
                                        3101.86438139042]), rtol=1e-8)

    def test_jnjnp_zeros(self):
        jn = special.jn

        def jnp(n, x):
            return (jn(n-1,x) - jn(n+1,x))/2
        for nt in range(1, 30):
            z, n, m, t = special.jnjnp_zeros(nt)
            for zz, nn, tt in zip(z, n, t):
                if tt == 0:
                    assert_allclose(jn(nn, zz), 0, atol=1e-6)
                elif tt == 1:
                    assert_allclose(jnp(nn, zz), 0, atol=1e-6)
                else:
                    raise AssertionError("Invalid t return for nt=%d" % nt)

    def test_jnp_zeros(self):
        jnp = special.jnp_zeros(1,5)
        assert_array_almost_equal(jnp, array([1.84118,
                                                5.33144,
                                                8.53632,
                                                11.70600,
                                                14.86359]),4)
        jnp = special.jnp_zeros(443,5)
        assert_tol_equal(special.jvp(443, jnp), 0, atol=1e-15)

    def test_jnyn_zeros(self):
        jnz = special.jnyn_zeros(1,5)
        assert_array_almost_equal(jnz,(array([3.83171,
                                                7.01559,
                                                10.17347,
                                                13.32369,
                                                16.47063]),
                                       array([1.84118,
                                                5.33144,
                                                8.53632,
                                                11.70600,
                                                14.86359]),
                                       array([2.19714,
                                                5.42968,
                                                8.59601,
                                                11.74915,
                                                14.89744]),
                                       array([3.68302,
                                                6.94150,
                                                10.12340,
                                                13.28576,
                                                16.44006])),5)

    def test_jvp(self):
        jvprim = special.jvp(2,2)
        jv0 = (special.jv(1,2)-special.jv(3,2))/2
        assert_almost_equal(jvprim,jv0,10)

    def test_k0(self):
        ozk = special.k0(.1)
        ozkr = special.kv(0,.1)
        assert_almost_equal(ozk,ozkr,8)

    def test_k0e(self):
        ozke = special.k0e(.1)
        ozker = special.kve(0,.1)
        assert_almost_equal(ozke,ozker,8)

    def test_k1(self):
        o1k = special.k1(.1)
        o1kr = special.kv(1,.1)
        assert_almost_equal(o1k,o1kr,8)

    def test_k1e(self):
        o1ke = special.k1e(.1)
        o1ker = special.kve(1,.1)
        assert_almost_equal(o1ke,o1ker,8)

    def test_jacobi(self):
        a = 5*np.random.random() - 1
        b = 5*np.random.random() - 1
        P0 = special.jacobi(0,a,b)
        P1 = special.jacobi(1,a,b)
        P2 = special.jacobi(2,a,b)
        P3 = special.jacobi(3,a,b)

        assert_array_almost_equal(P0.c,[1],13)
        assert_array_almost_equal(P1.c,array([a+b+2,a-b])/2.0,13)
        cp = [(a+b+3)*(a+b+4), 4*(a+b+3)*(a+2), 4*(a+1)*(a+2)]
        p2c = [cp[0],cp[1]-2*cp[0],cp[2]-cp[1]+cp[0]]
        assert_array_almost_equal(P2.c,array(p2c)/8.0,13)
        cp = [(a+b+4)*(a+b+5)*(a+b+6),6*(a+b+4)*(a+b+5)*(a+3),
              12*(a+b+4)*(a+2)*(a+3),8*(a+1)*(a+2)*(a+3)]
        p3c = [cp[0],cp[1]-3*cp[0],cp[2]-2*cp[1]+3*cp[0],cp[3]-cp[2]+cp[1]-cp[0]]
        assert_array_almost_equal(P3.c,array(p3c)/48.0,13)

    def test_kn(self):
        kn1 = special.kn(0,.2)
        assert_almost_equal(kn1,1.7527038555281462,8)

    def test_negv_kv(self):
        assert_equal(special.kv(3.0, 2.2), special.kv(-3.0, 2.2))

    def test_kv0(self):
        kv0 = special.kv(0,.2)
        assert_almost_equal(kv0, 1.7527038555281462, 10)

    def test_kv1(self):
        kv1 = special.kv(1,0.2)
        assert_almost_equal(kv1, 4.775972543220472, 10)

    def test_kv2(self):
        kv2 = special.kv(2,0.2)
        assert_almost_equal(kv2, 49.51242928773287, 10)

    def test_kn_largeorder(self):
        assert_allclose(special.kn(32, 1), 1.7516596664574289e+43)

    def test_kv_largearg(self):
        assert_equal(special.kv(0, 1e19), 0)

    def test_negv_kve(self):
        assert_equal(special.kve(3.0, 2.2), special.kve(-3.0, 2.2))

    def test_kve(self):
        kve1 = special.kve(0,.2)
        kv1 = special.kv(0,.2)*exp(.2)
        assert_almost_equal(kve1,kv1,8)
        z = .2+1j
        kve2 = special.kve(0,z)
        kv2 = special.kv(0,z)*exp(z)
        assert_almost_equal(kve2,kv2,8)

    def test_kvp_v0n1(self):
        z = 2.2
        assert_almost_equal(-special.kv(1,z), special.kvp(0,z, n=1), 10)

    def test_kvp_n1(self):
        v = 3.
        z = 2.2
        xc = -special.kv(v+1,z) + v/z*special.kv(v,z)
        x = special.kvp(v,z, n=1)
        assert_almost_equal(xc, x, 10)   # this function (kvp) is broken

    def test_kvp_n2(self):
        v = 3.
        z = 2.2
        xc = (z**2+v**2-v)/z**2 * special.kv(v,z) + special.kv(v+1,z)/z
        x = special.kvp(v, z, n=2)
        assert_almost_equal(xc, x, 10)

    def test_y0(self):
        oz = special.y0(.1)
        ozr = special.yn(0,.1)
        assert_almost_equal(oz,ozr,8)

    def test_y1(self):
        o1 = special.y1(.1)
        o1r = special.yn(1,.1)
        assert_almost_equal(o1,o1r,8)

    def test_y0_zeros(self):
        yo,ypo = special.y0_zeros(2)
        zo,zpo = special.y0_zeros(2,complex=1)
        all = r_[yo,zo]
        allval = r_[ypo,zpo]
        assert_array_almost_equal(abs(special.yv(0.0,all)),0.0,11)
        assert_array_almost_equal(abs(special.yv(1,all)-allval),0.0,11)

    def test_y1_zeros(self):
        y1 = special.y1_zeros(1)
        assert_array_almost_equal(y1,(array([2.19714]),array([0.52079])),5)

    def test_y1p_zeros(self):
        y1p = special.y1p_zeros(1,complex=1)
        assert_array_almost_equal(y1p,(array([0.5768+0.904j]), array([-0.7635+0.5892j])),3)

    def test_yn_zeros(self):
        an = special.yn_zeros(4,2)
        assert_array_almost_equal(an,array([5.64515, 9.36162]),5)
        an = special.yn_zeros(443,5)
        assert_tol_equal(an, [450.13573091578090314, 463.05692376675001542,
                              472.80651546418663566, 481.27353184725625838,
                              488.98055964441374646], rtol=1e-15)

    def test_ynp_zeros(self):
        ao = special.ynp_zeros(0,2)
        assert_array_almost_equal(ao,array([2.19714133, 5.42968104]),6)
        ao = special.ynp_zeros(43,5)
        assert_tol_equal(special.yvp(43, ao), 0, atol=1e-15)
        ao = special.ynp_zeros(443,5)
        assert_tol_equal(special.yvp(443, ao), 0, atol=1e-9)

    def test_ynp_zeros_large_order(self):
        ao = special.ynp_zeros(443,5)
        assert_tol_equal(special.yvp(443, ao), 0, atol=1e-14)

    def test_yn(self):
        yn2n = special.yn(1,.2)
        assert_almost_equal(yn2n,-3.3238249881118471,8)

    def test_negv_yv(self):
        assert_almost_equal(special.yv(-3,2), -special.yv(3,2), 14)

    def test_yv(self):
        yv2 = special.yv(1,.2)
        assert_almost_equal(yv2,-3.3238249881118471,8)

    def test_negv_yve(self):
        assert_almost_equal(special.yve(-3,2), -special.yve(3,2), 14)

    def test_yve(self):
        yve2 = special.yve(1,.2)
        assert_almost_equal(yve2,-3.3238249881118471,8)
        yve2r = special.yv(1,.2+1j)*exp(-1)
        yve22 = special.yve(1,.2+1j)
        assert_almost_equal(yve22,yve2r,8)

    def test_yvp(self):
        yvpr = (special.yv(1,.2) - special.yv(3,.2))/2.0
        yvp1 = special.yvp(2,.2)
        assert_array_almost_equal(yvp1,yvpr,10)

    def _cephes_vs_amos_points(self):
        """Yield points at which to compare Cephes implementation to AMOS"""
        # check several points, including large-amplitude ones
        for v in [-120, -100.3, -20., -10., -1., -.5,
                  0., 1., 12.49, 120., 301]:
            for z in [-1300, -11, -10, -1, 1., 10., 200.5, 401., 600.5,
                      700.6, 1300, 10003]:
                yield v, z

        # check half-integers; these are problematic points at least
        # for cephes/iv
        for v in 0.5 + arange(-60, 60):
            yield v, 3.5

    def check_cephes_vs_amos(self, f1, f2, rtol=1e-11, atol=0, skip=None):
        for v, z in self._cephes_vs_amos_points():
            if skip is not None and skip(v, z):
                continue
            c1, c2, c3 = f1(v, z), f1(v,z+0j), f2(int(v), z)
            if np.isinf(c1):
                assert_(np.abs(c2) >= 1e300, (v, z))
            elif np.isnan(c1):
                assert_(c2.imag != 0, (v, z))
            else:
                assert_tol_equal(c1, c2, err_msg=(v, z), rtol=rtol, atol=atol)
                if v == int(v):
                    assert_tol_equal(c3, c2, err_msg=(v, z),
                                     rtol=rtol, atol=atol)

    def test_jv_cephes_vs_amos(self):
        self.check_cephes_vs_amos(special.jv, special.jn, rtol=1e-10, atol=1e-305)

    def test_yv_cephes_vs_amos(self):
        self.check_cephes_vs_amos(special.yv, special.yn, rtol=1e-11, atol=1e-305)

    def test_yv_cephes_vs_amos_only_small_orders(self):
        skipper = lambda v, z: (abs(v) > 50)
        self.check_cephes_vs_amos(special.yv, special.yn, rtol=1e-11, atol=1e-305, skip=skipper)

    def test_iv_cephes_vs_amos(self):
        olderr = np.seterr(all='ignore')
        try:
            self.check_cephes_vs_amos(special.iv, special.iv, rtol=5e-9, atol=1e-305)
        finally:
            np.seterr(**olderr)

    @dec.slow
    def test_iv_cephes_vs_amos_mass_test(self):
        N = 1000000
        np.random.seed(1)
        v = np.random.pareto(0.5, N) * (-1)**np.random.randint(2, size=N)
        x = np.random.pareto(0.2, N) * (-1)**np.random.randint(2, size=N)

        imsk = (np.random.randint(8, size=N) == 0)
        v[imsk] = v[imsk].astype(int)

        old_err = np.seterr(all='ignore')
        try:
            c1 = special.iv(v, x)
            c2 = special.iv(v, x+0j)

            # deal with differences in the inf and zero cutoffs
            c1[abs(c1) > 1e300] = np.inf
            c2[abs(c2) > 1e300] = np.inf
            c1[abs(c1) < 1e-300] = 0
            c2[abs(c2) < 1e-300] = 0

            dc = abs(c1/c2 - 1)
            dc[np.isnan(dc)] = 0
        finally:
            np.seterr(**old_err)

        k = np.argmax(dc)

        # Most error apparently comes from AMOS and not our implementation;
        # there are some problems near integer orders there
        assert_(dc[k] < 2e-7, (v[k], x[k], special.iv(v[k], x[k]), special.iv(v[k], x[k]+0j)))

    def test_kv_cephes_vs_amos(self):
        self.check_cephes_vs_amos(special.kv, special.kn, rtol=1e-9, atol=1e-305)
        self.check_cephes_vs_amos(special.kv, special.kv, rtol=1e-9, atol=1e-305)

    def test_ticket_623(self):
        assert_tol_equal(special.jv(3, 4), 0.43017147387562193)
        assert_tol_equal(special.jv(301, 1300), 0.0183487151115275)
        assert_tol_equal(special.jv(301, 1296.0682), -0.0224174325312048)

    def test_ticket_853(self):
        """Negative-order Bessels"""
        # cephes
        assert_tol_equal(special.jv(-1, 1), -0.4400505857449335)
        assert_tol_equal(special.jv(-2, 1), 0.1149034849319005)
        assert_tol_equal(special.yv(-1, 1), 0.7812128213002887)
        assert_tol_equal(special.yv(-2, 1), -1.650682606816255)
        assert_tol_equal(special.iv(-1, 1), 0.5651591039924851)
        assert_tol_equal(special.iv(-2, 1), 0.1357476697670383)
        assert_tol_equal(special.kv(-1, 1), 0.6019072301972347)
        assert_tol_equal(special.kv(-2, 1), 1.624838898635178)
        assert_tol_equal(special.jv(-0.5, 1), 0.43109886801837607952)
        assert_tol_equal(special.yv(-0.5, 1), 0.6713967071418031)
        assert_tol_equal(special.iv(-0.5, 1), 1.231200214592967)
        assert_tol_equal(special.kv(-0.5, 1), 0.4610685044478945)
        # amos
        assert_tol_equal(special.jv(-1, 1+0j), -0.4400505857449335)
        assert_tol_equal(special.jv(-2, 1+0j), 0.1149034849319005)
        assert_tol_equal(special.yv(-1, 1+0j), 0.7812128213002887)
        assert_tol_equal(special.yv(-2, 1+0j), -1.650682606816255)

        assert_tol_equal(special.iv(-1, 1+0j), 0.5651591039924851)
        assert_tol_equal(special.iv(-2, 1+0j), 0.1357476697670383)
        assert_tol_equal(special.kv(-1, 1+0j), 0.6019072301972347)
        assert_tol_equal(special.kv(-2, 1+0j), 1.624838898635178)

        assert_tol_equal(special.jv(-0.5, 1+0j), 0.43109886801837607952)
        assert_tol_equal(special.jv(-0.5, 1+1j), 0.2628946385649065-0.827050182040562j)
        assert_tol_equal(special.yv(-0.5, 1+0j), 0.6713967071418031)
        assert_tol_equal(special.yv(-0.5, 1+1j), 0.967901282890131+0.0602046062142816j)

        assert_tol_equal(special.iv(-0.5, 1+0j), 1.231200214592967)
        assert_tol_equal(special.iv(-0.5, 1+1j), 0.77070737376928+0.39891821043561j)
        assert_tol_equal(special.kv(-0.5, 1+0j), 0.4610685044478945)
        assert_tol_equal(special.kv(-0.5, 1+1j), 0.06868578341999-0.38157825981268j)

        assert_tol_equal(special.jve(-0.5,1+0.3j), special.jv(-0.5, 1+0.3j)*exp(-0.3))
        assert_tol_equal(special.yve(-0.5,1+0.3j), special.yv(-0.5, 1+0.3j)*exp(-0.3))
        assert_tol_equal(special.ive(-0.5,0.3+1j), special.iv(-0.5, 0.3+1j)*exp(-0.3))
        assert_tol_equal(special.kve(-0.5,0.3+1j), special.kv(-0.5, 0.3+1j)*exp(0.3+1j))

        assert_tol_equal(special.hankel1(-0.5, 1+1j), special.jv(-0.5, 1+1j) + 1j*special.yv(-0.5,1+1j))
        assert_tol_equal(special.hankel2(-0.5, 1+1j), special.jv(-0.5, 1+1j) - 1j*special.yv(-0.5,1+1j))

    def test_ticket_854(self):
        """Real-valued Bessel domains"""
        assert_(isnan(special.jv(0.5, -1)))
        assert_(isnan(special.iv(0.5, -1)))
        assert_(isnan(special.yv(0.5, -1)))
        assert_(isnan(special.yv(1, -1)))
        assert_(isnan(special.kv(0.5, -1)))
        assert_(isnan(special.kv(1, -1)))
        assert_(isnan(special.jve(0.5, -1)))
        assert_(isnan(special.ive(0.5, -1)))
        assert_(isnan(special.yve(0.5, -1)))
        assert_(isnan(special.yve(1, -1)))
        assert_(isnan(special.kve(0.5, -1)))
        assert_(isnan(special.kve(1, -1)))
        assert_(isnan(special.airye(-1)[0:2]).all(), special.airye(-1))
        assert_(not isnan(special.airye(-1)[2:4]).any(), special.airye(-1))

    def test_ticket_503(self):
        """Real-valued Bessel I overflow"""
        assert_tol_equal(special.iv(1, 700), 1.528500390233901e302)
        assert_tol_equal(special.iv(1000, 1120), 1.301564549405821e301)

    def test_iv_hyperg_poles(self):
        assert_tol_equal(special.iv(-0.5, 1), 1.231200214592967)

    def iv_series(self, v, z, n=200):
        k = arange(0, n).astype(float_)
        r = (v+2*k)*log(.5*z) - special.gammaln(k+1) - special.gammaln(v+k+1)
        r[isnan(r)] = inf
        r = exp(r)
        err = abs(r).max() * finfo(float_).eps * n + abs(r[-1])*10
        return r.sum(), err

    def test_i0_series(self):
        for z in [1., 10., 200.5]:
            value, err = self.iv_series(0, z)
            assert_tol_equal(special.i0(z), value, atol=err, err_msg=z)

    def test_i1_series(self):
        for z in [1., 10., 200.5]:
            value, err = self.iv_series(1, z)
            assert_tol_equal(special.i1(z), value, atol=err, err_msg=z)

    def test_iv_series(self):
        for v in [-20., -10., -1., 0., 1., 12.49, 120.]:
            for z in [1., 10., 200.5, -1+2j]:
                value, err = self.iv_series(v, z)
                assert_tol_equal(special.iv(v, z), value, atol=err, err_msg=(v, z))

    def test_i0(self):
        values = [[0.0, 1.0],
                  [1e-10, 1.0],
                  [0.1, 0.9071009258],
                  [0.5, 0.6450352706],
                  [1.0, 0.4657596077],
                  [2.5, 0.2700464416],
                  [5.0, 0.1835408126],
                  [20.0, 0.0897803119],
                  ]
        for i, (x, v) in enumerate(values):
            cv = special.i0(x) * exp(-x)
            assert_almost_equal(cv, v, 8, err_msg='test #%d' % i)

    def test_i0e(self):
        oize = special.i0e(.1)
        oizer = special.ive(0,.1)
        assert_almost_equal(oize,oizer,8)

    def test_i1(self):
        values = [[0.0, 0.0],
                  [1e-10, 0.4999999999500000e-10],
                  [0.1, 0.0452984468],
                  [0.5, 0.1564208032],
                  [1.0, 0.2079104154],
                  [5.0, 0.1639722669],
                  [20.0, 0.0875062222],
                  ]
        for i, (x, v) in enumerate(values):
            cv = special.i1(x) * exp(-x)
            assert_almost_equal(cv, v, 8, err_msg='test #%d' % i)

    def test_i1e(self):
        oi1e = special.i1e(.1)
        oi1er = special.ive(1,.1)
        assert_almost_equal(oi1e,oi1er,8)

    def test_iti0k0(self):
        iti0 = array(special.iti0k0(5))
        assert_array_almost_equal(iti0,array([31.848667776169801, 1.5673873907283657]),5)

    def test_it2i0k0(self):
        it2k = special.it2i0k0(.1)
        assert_array_almost_equal(it2k,array([0.0012503906973464409, 3.3309450354686687]),6)

    def test_iv(self):
        iv1 = special.iv(0,.1)*exp(-.1)
        assert_almost_equal(iv1,0.90710092578230106,10)

    def test_negv_ive(self):
        assert_equal(special.ive(3,2), special.ive(-3,2))

    def test_ive(self):
        ive1 = special.ive(0,.1)
        iv1 = special.iv(0,.1)*exp(-.1)
        assert_almost_equal(ive1,iv1,10)

    def test_ivp0(self):
        assert_almost_equal(special.iv(1,2), special.ivp(0,2), 10)

    def test_ivp(self):
        y = (special.iv(0,2) + special.iv(2,2))/2
        x = special.ivp(1,2)
        assert_almost_equal(x,y,10)


class TestLaguerre(TestCase):
    def test_laguerre(self):
        lag0 = special.laguerre(0)
        lag1 = special.laguerre(1)
        lag2 = special.laguerre(2)
        lag3 = special.laguerre(3)
        lag4 = special.laguerre(4)
        lag5 = special.laguerre(5)
        assert_array_almost_equal(lag0.c,[1],13)
        assert_array_almost_equal(lag1.c,[-1,1],13)
        assert_array_almost_equal(lag2.c,array([1,-4,2])/2.0,13)
        assert_array_almost_equal(lag3.c,array([-1,9,-18,6])/6.0,13)
        assert_array_almost_equal(lag4.c,array([1,-16,72,-96,24])/24.0,13)
        assert_array_almost_equal(lag5.c,array([-1,25,-200,600,-600,120])/120.0,13)

    def test_genlaguerre(self):
        k = 5*np.random.random() - 0.9
        lag0 = special.genlaguerre(0,k)
        lag1 = special.genlaguerre(1,k)
        lag2 = special.genlaguerre(2,k)
        lag3 = special.genlaguerre(3,k)
        assert_equal(lag0.c,[1])
        assert_equal(lag1.c,[-1,k+1])
        assert_almost_equal(lag2.c,array([1,-2*(k+2),(k+1.)*(k+2.)])/2.0)
        assert_almost_equal(lag3.c,array([-1,3*(k+3),-3*(k+2)*(k+3),(k+1)*(k+2)*(k+3)])/6.0)


# Base polynomials come from Abrahmowitz and Stegan
class TestLegendre(TestCase):
    def test_legendre(self):
        leg0 = special.legendre(0)
        leg1 = special.legendre(1)
        leg2 = special.legendre(2)
        leg3 = special.legendre(3)
        leg4 = special.legendre(4)
        leg5 = special.legendre(5)
        assert_equal(leg0.c, [1])
        assert_equal(leg1.c, [1,0])
        assert_almost_equal(leg2.c, array([3,0,-1])/2.0, decimal=13)
        assert_almost_equal(leg3.c, array([5,0,-3,0])/2.0)
        assert_almost_equal(leg4.c, array([35,0,-30,0,3])/8.0)
        assert_almost_equal(leg5.c, array([63,0,-70,0,15,0])/8.0)


class TestLambda(TestCase):
    def test_lmbda(self):
        lam = special.lmbda(1,.1)
        lamr = (array([special.jn(0,.1), 2*special.jn(1,.1)/.1]),
                array([special.jvp(0,.1), -2*special.jv(1,.1)/.01 + 2*special.jvp(1,.1)/.1]))
        assert_array_almost_equal(lam,lamr,8)


class TestLog1p(TestCase):
    def test_log1p(self):
        l1p = (special.log1p(10), special.log1p(11), special.log1p(12))
        l1prl = (log(11), log(12), log(13))
        assert_array_almost_equal(l1p,l1prl,8)

    def test_log1pmore(self):
        l1pm = (special.log1p(1), special.log1p(1.1), special.log1p(1.2))
        l1pmrl = (log(2),log(2.1),log(2.2))
        assert_array_almost_equal(l1pm,l1pmrl,8)


class TestLegendreFunctions(TestCase):
    def test_clpmn(self):
        z = 0.5+0.3j
        clp = special.clpmn(2, 2, z, 3)
        assert_array_almost_equal(clp,
                   (array([[1.0000, z, 0.5*(3*z*z-1)],
                           [0.0000, sqrt(z*z-1), 3*z*sqrt(z*z-1)],
                           [0.0000, 0.0000, 3*(z*z-1)]]),
                    array([[0.0000, 1.0000, 3*z],
                           [0.0000, z/sqrt(z*z-1), 3*(2*z*z-1)/sqrt(z*z-1)],
                           [0.0000, 0.0000, 6*z]])),
                    7)

    def test_clpmn_close_to_real_2(self):
        eps = 1e-10
        m = 1
        n = 3
        x = 0.5
        clp_plus = special.clpmn(m, n, x+1j*eps, 2)[0][m, n]
        clp_minus = special.clpmn(m, n, x-1j*eps, 2)[0][m, n]
        assert_array_almost_equal(array([clp_plus, clp_minus]),
                                  array([special.lpmv(m, n, x),
                                         special.lpmv(m, n, x)]),
                                  7)

    def test_clpmn_close_to_real_3(self):
        eps = 1e-10
        m = 1
        n = 3
        x = 0.5
        clp_plus = special.clpmn(m, n, x+1j*eps, 3)[0][m, n]
        clp_minus = special.clpmn(m, n, x-1j*eps, 3)[0][m, n]
        assert_array_almost_equal(array([clp_plus, clp_minus]),
                                  array([special.lpmv(m, n, x)*np.exp(-0.5j*m*np.pi),
                                         special.lpmv(m, n, x)*np.exp(0.5j*m*np.pi)]),
                                  7)

    def test_clpmn_across_unit_circle(self):
        eps = 1e-7
        m = 1
        n = 1
        x = 1j
        for type in [2, 3]:
            assert_almost_equal(special.clpmn(m, n, x+1j*eps, type)[0][m, n],
                            special.clpmn(m, n, x-1j*eps, type)[0][m, n], 6)

    def test_inf(self):
        for z in (1, -1):
            for n in range(4):
                for m in range(1, n):
                    lp = special.clpmn(m, n, z)
                    assert_(np.isinf(lp[1][1,1:]).all())
                    lp = special.lpmn(m, n, z)
                    assert_(np.isinf(lp[1][1,1:]).all())

    def test_deriv_clpmn(self):
        # data inside and outside of the unit circle
        zvals = [0.5+0.5j, -0.5+0.5j, -0.5-0.5j, 0.5-0.5j,
                 1+1j, -1+1j, -1-1j, 1-1j]
        m = 2
        n = 3
        for type in [2, 3]:
            for z in zvals:
                for h in [1e-3, 1e-3j]:
                    approx_derivative = (special.clpmn(m, n, z+0.5*h, type)[0]
                                         - special.clpmn(m, n, z-0.5*h, type)[0])/h
                    assert_allclose(special.clpmn(m, n, z, type)[1],
                                    approx_derivative,
                                    rtol=1e-4)

    def test_lpmn(self):
        lp = special.lpmn(0,2,.5)
        assert_array_almost_equal(lp,(array([[1.00000,
                                                      0.50000,
                                                      -0.12500]]),
                                      array([[0.00000,
                                                      1.00000,
                                                      1.50000]])),4)

    def test_lpn(self):
        lpnf = special.lpn(2,.5)
        assert_array_almost_equal(lpnf,(array([1.00000,
                                                        0.50000,
                                                        -0.12500]),
                                      array([0.00000,
                                                      1.00000,
                                                      1.50000])),4)

    def test_lpmv(self):
        lp = special.lpmv(0,2,.5)
        assert_almost_equal(lp,-0.125,7)
        lp = special.lpmv(0,40,.001)
        assert_almost_equal(lp,0.1252678976534484,7)

        # XXX: this is outside the domain of the current implementation,
        #      so ensure it returns a NaN rather than a wrong answer.
        olderr = np.seterr(all='ignore')
        try:
            lp = special.lpmv(-1,-1,.001)
        finally:
            np.seterr(**olderr)
        assert_(lp != 0 or np.isnan(lp))

    def test_lqmn(self):
        lqmnf = special.lqmn(0,2,.5)
        lqf = special.lqn(2,.5)
        assert_array_almost_equal(lqmnf[0][0],lqf[0],4)
        assert_array_almost_equal(lqmnf[1][0],lqf[1],4)

    def test_lqmn_gt1(self):
        """algorithm for real arguments changes at 1.0001
           test against analytical result for m=2, n=1
        """
        x0 = 1.0001
        delta = 0.00002
        for x in (x0-delta, x0+delta):
            lq = special.lqmn(2, 1, x)[0][-1, -1]
            expected = 2/(x*x-1)
            assert_almost_equal(lq, expected)

    def test_lqmn_shape(self):
        a, b = special.lqmn(4, 4, 1.1)
        assert_equal(a.shape, (5, 5))
        assert_equal(b.shape, (5, 5))

        a, b = special.lqmn(4, 0, 1.1)
        assert_equal(a.shape, (5, 1))
        assert_equal(b.shape, (5, 1))

    def test_lqn(self):
        lqf = special.lqn(2,.5)
        assert_array_almost_equal(lqf,(array([0.5493, -0.7253, -0.8187]),
                                       array([1.3333, 1.216, -0.8427])),4)


class TestMathieu(TestCase):

    def test_mathieu_a(self):
        pass

    def test_mathieu_even_coef(self):
        mc = special.mathieu_even_coef(2,5)
        # Q not defined broken and cannot figure out proper reporting order

    def test_mathieu_odd_coef(self):
        # same problem as above
        pass


class TestFresnelIntegral(TestCase):

    def test_modfresnelp(self):
        pass

    def test_modfresnelm(self):
        pass


class TestOblCvSeq(TestCase):
    def test_obl_cv_seq(self):
        obl = special.obl_cv_seq(0,3,1)
        assert_array_almost_equal(obl,array([-0.348602,
                                              1.393206,
                                              5.486800,
                                              11.492120]),5)


class TestParabolicCylinder(TestCase):
    def test_pbdn_seq(self):
        pb = special.pbdn_seq(1,.1)
        assert_array_almost_equal(pb,(array([0.9975,
                                              0.0998]),
                                      array([-0.0499,
                                             0.9925])),4)

    def test_pbdv(self):
        pbv = special.pbdv(1,.2)
        derrl = 1/2*(.2)*special.pbdv(1,.2)[0] - special.pbdv(0,.2)[0]

    def test_pbdv_seq(self):
        pbn = special.pbdn_seq(1,.1)
        pbv = special.pbdv_seq(1,.1)
        assert_array_almost_equal(pbv,(real(pbn[0]),real(pbn[1])),4)

    def test_pbdv_points(self):
        # simple case
        eta = np.linspace(-10, 10, 5)
        z = 2**(eta/2)*np.sqrt(np.pi)/special.gamma(.5-.5*eta)
        assert_tol_equal(special.pbdv(eta, 0.)[0], z, rtol=1e-14, atol=1e-14)

        # some points
        assert_tol_equal(special.pbdv(10.34, 20.44)[0], 1.3731383034455e-32, rtol=1e-12)
        assert_tol_equal(special.pbdv(-9.53, 3.44)[0], 3.166735001119246e-8, rtol=1e-12)

    def test_pbdv_gradient(self):
        x = np.linspace(-4, 4, 8)[:,None]
        eta = np.linspace(-10, 10, 5)[None,:]

        p = special.pbdv(eta, x)
        eps = 1e-7 + 1e-7*abs(x)
        dp = (special.pbdv(eta, x + eps)[0] - special.pbdv(eta, x - eps)[0]) / eps / 2.
        assert_tol_equal(p[1], dp, rtol=1e-6, atol=1e-6)

    def test_pbvv_gradient(self):
        x = np.linspace(-4, 4, 8)[:,None]
        eta = np.linspace(-10, 10, 5)[None,:]

        p = special.pbvv(eta, x)
        eps = 1e-7 + 1e-7*abs(x)
        dp = (special.pbvv(eta, x + eps)[0] - special.pbvv(eta, x - eps)[0]) / eps / 2.
        assert_tol_equal(p[1], dp, rtol=1e-6, atol=1e-6)


class TestPolygamma(TestCase):
    # from Table 6.2 (pg. 271) of A&S
    def test_polygamma(self):
        poly2 = special.polygamma(2,1)
        poly3 = special.polygamma(3,1)
        assert_almost_equal(poly2,-2.4041138063,10)
        assert_almost_equal(poly3,6.4939394023,10)

        # Test polygamma(0, x) == psi(x)
        x = [2, 3, 1.1e14]
        assert_almost_equal(special.polygamma(0, x), special.psi(x))

        # Test broadcasting
        n = [0, 1, 2]
        x = [0.5, 1.5, 2.5]
        expected = [-1.9635100260214238, 0.93480220054467933,
                    -0.23620405164172739]
        assert_almost_equal(special.polygamma(n, x), expected)
        expected = np.row_stack([expected]*2)
        assert_almost_equal(special.polygamma(n, np.row_stack([x]*2)),
                            expected)
        assert_almost_equal(special.polygamma(np.row_stack([n]*2), x),
                            expected)


class TestProCvSeq(TestCase):
    def test_pro_cv_seq(self):
        prol = special.pro_cv_seq(0,3,1)
        assert_array_almost_equal(prol,array([0.319000,
                                               2.593084,
                                               6.533471,
                                               12.514462]),5)


class TestPsi(TestCase):
    def test_psi(self):
        ps = special.psi(1)
        assert_almost_equal(ps,-0.57721566490153287,8)


class TestRadian(TestCase):
    def test_radian(self):
        rad = special.radian(90,0,0)
        assert_almost_equal(rad,pi/2.0,5)

    def test_radianmore(self):
        rad1 = special.radian(90,1,60)
        assert_almost_equal(rad1,pi/2+0.0005816135199345904,5)


class TestRiccati(TestCase):
    def test_riccati_jn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            jnrl = (special.sph_jn(1,.2)[0]*.2,special.sph_jn(1,.2)[0]+special.sph_jn(1,.2)[1]*.2)
        ricjn = special.riccati_jn(1,.2)
        assert_array_almost_equal(ricjn,jnrl,8)

    def test_riccati_yn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ynrl = (special.sph_yn(1,.2)[0]*.2,special.sph_yn(1,.2)[0]+special.sph_yn(1,.2)[1]*.2)
        ricyn = special.riccati_yn(1,.2)
        assert_array_almost_equal(ricyn,ynrl,8)


class TestRound(TestCase):
    def test_round(self):
        rnd = list(map(int,(special.round(10.1),special.round(10.4),special.round(10.5),special.round(10.6))))

        # Note: According to the documentation, scipy.special.round is
        # supposed to round to the nearest even number if the fractional
        # part is exactly 0.5. On some platforms, this does not appear
        # to work and thus this test may fail. However, this unit test is
        # correctly written.
        rndrl = (10,10,10,11)
        assert_array_equal(rnd,rndrl)


def test_sph_harm():
    # Tests derived from tables in
    # http://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    sh = special.sph_harm
    pi = np.pi
    exp = np.exp
    sqrt = np.sqrt
    sin = np.sin
    cos = np.cos
    yield (assert_array_almost_equal, sh(0,0,0,0),
           0.5/sqrt(pi))
    yield (assert_array_almost_equal, sh(-2,2,0.,pi/4),
           0.25*sqrt(15./(2.*pi)) *
           (sin(pi/4))**2.)
    yield (assert_array_almost_equal, sh(-2,2,0.,pi/2),
           0.25*sqrt(15./(2.*pi)))
    yield (assert_array_almost_equal, sh(2,2,pi,pi/2),
           0.25*sqrt(15/(2.*pi)) *
           exp(0+2.*pi*1j)*sin(pi/2.)**2.)
    yield (assert_array_almost_equal, sh(2,4,pi/4.,pi/3.),
           (3./8.)*sqrt(5./(2.*pi)) *
           exp(0+2.*pi/4.*1j) *
           sin(pi/3.)**2. *
           (7.*cos(pi/3.)**2.-1))
    yield (assert_array_almost_equal, sh(4,4,pi/8.,pi/6.),
           (3./16.)*sqrt(35./(2.*pi)) *
           exp(0+4.*pi/8.*1j)*sin(pi/6.)**4.)


def test_sph_harm_ufunc_loop_selection():
    # see https://github.com/scipy/scipy/issues/4895
    dt = np.dtype(np.complex128)
    assert_equal(special.sph_harm(0, 0, 0, 0).dtype, dt)
    assert_equal(special.sph_harm([0], 0, 0, 0).dtype, dt)
    assert_equal(special.sph_harm(0, [0], 0, 0).dtype, dt)
    assert_equal(special.sph_harm(0, 0, [0], 0).dtype, dt)
    assert_equal(special.sph_harm(0, 0, 0, [0]).dtype, dt)
    assert_equal(special.sph_harm([0], [0], [0], [0]).dtype, dt)


class TestSpherical(TestCase):
    def test_sph_harm(self):
        # see test_sph_harm function
        pass

    def test_sph_in(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            i1n = special.sph_in(1,.2)
        inp0 = (i1n[0][1])
        inp1 = (i1n[0][0] - 2.0/0.2 * i1n[0][1])
        assert_array_almost_equal(i1n[0],array([1.0066800127054699381,
                                                0.066933714568029540839]),12)
        assert_array_almost_equal(i1n[1],[inp0,inp1],12)

    def test_sph_inkn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            spikn = r_[special.sph_in(1,.2) + special.sph_kn(1,.2)]
            inkn = r_[special.sph_inkn(1,.2)]
        assert_array_almost_equal(inkn,spikn,10)

    def test_sph_in_kn_order0(self):
        x = 1.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sph_i0 = special.sph_in(0, x)
            sph_i0_expected = np.array([np.sinh(x)/x,
                                        np.cosh(x)/x-np.sinh(x)/x**2])
            assert_array_almost_equal(r_[sph_i0], sph_i0_expected)
            sph_k0 = special.sph_kn(0, x)
            sph_k0_expected = np.array([0.5*pi*exp(-x)/x,
                                        -0.5*pi*exp(-x)*(1/x+1/x**2)])
            assert_array_almost_equal(r_[sph_k0], sph_k0_expected)
            sph_i0k0 = special.sph_inkn(0, x)
            assert_array_almost_equal(r_[sph_i0+sph_k0],
                                      r_[sph_i0k0],
                                      10)

    def test_sph_jn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            s1 = special.sph_jn(2,.2)
        s10 = -s1[0][1]
        s11 = s1[0][0]-2.0/0.2*s1[0][1]
        s12 = s1[0][1]-3.0/0.2*s1[0][2]
        assert_array_almost_equal(s1[0],[0.99334665397530607731,
                                      0.066400380670322230863,
                                      0.0026590560795273856680],12)
        assert_array_almost_equal(s1[1],[s10,s11,s12],12)

    def test_sph_jnyn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            jnyn = r_[special.sph_jn(1,.2) + special.sph_yn(1,.2)]  # tuple addition
            jnyn1 = r_[special.sph_jnyn(1,.2)]
        assert_array_almost_equal(jnyn1,jnyn,9)

    def test_sph_kn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            kn = special.sph_kn(2,.2)
        kn0 = -kn[0][1]
        kn1 = -kn[0][0]-2.0/0.2*kn[0][1]
        kn2 = -kn[0][1]-3.0/0.2*kn[0][2]
        assert_array_almost_equal(kn[0],[6.4302962978445670140,
                                         38.581777787067402086,
                                         585.15696310385559829],12)
        assert_array_almost_equal(kn[1],[kn0,kn1,kn2],9)

    def test_sph_yn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sy1 = special.sph_yn(2,.2)[0][2]
            sy2 = special.sph_yn(0,.2)[0][0]
            sy3 = special.sph_yn(1,.2)[1][1]
            sphpy = (special.sph_yn(1,.2)[0][0]-2*special.sph_yn(2,.2)[0][2])/3  # correct derivative value
        assert_almost_equal(sy1,-377.52483,5)  # previous values in the system
        assert_almost_equal(sy2,-4.9003329,5)
        assert_almost_equal(sy3,sphpy,4)  # compare correct derivative val. (correct =-system val).


class TestStruve(object):
    def _series(self, v, z, n=100):
        """Compute Struve function & error estimate from its power series."""
        k = arange(0, n)
        r = (-1)**k * (.5*z)**(2*k+v+1)/special.gamma(k+1.5)/special.gamma(k+v+1.5)
        err = abs(r).max() * finfo(float_).eps * n
        return r.sum(), err

    def test_vs_series(self):
        """Check Struve function versus its power series"""
        for v in [-20, -10, -7.99, -3.4, -1, 0, 1, 3.4, 12.49, 16]:
            for z in [1, 10, 19, 21, 30]:
                value, err = self._series(v, z)
                assert_tol_equal(special.struve(v, z), value, rtol=0, atol=err), (v, z)

    def test_some_values(self):
        assert_tol_equal(special.struve(-7.99, 21), 0.0467547614113, rtol=1e-7)
        assert_tol_equal(special.struve(-8.01, 21), 0.0398716951023, rtol=1e-8)
        assert_tol_equal(special.struve(-3.0, 200), 0.0142134427432, rtol=1e-12)
        assert_tol_equal(special.struve(-8.0, -41), 0.0192469727846, rtol=1e-11)
        assert_equal(special.struve(-12, -41), -special.struve(-12, 41))
        assert_equal(special.struve(+12, -41), -special.struve(+12, 41))
        assert_equal(special.struve(-11, -41), +special.struve(-11, 41))
        assert_equal(special.struve(+11, -41), +special.struve(+11, 41))

        assert_(isnan(special.struve(-7.1, -1)))
        assert_(isnan(special.struve(-10.1, -1)))

    def test_regression_679(self):
        """Regression test for #679"""
        assert_tol_equal(special.struve(-1.0, 20 - 1e-8), special.struve(-1.0, 20 + 1e-8))
        assert_tol_equal(special.struve(-2.0, 20 - 1e-8), special.struve(-2.0, 20 + 1e-8))
        assert_tol_equal(special.struve(-4.3, 20 - 1e-8), special.struve(-4.3, 20 + 1e-8))


def test_chi2_smalldf():
    assert_almost_equal(special.chdtr(0.6,3), 0.957890536704110)


def test_ch2_inf():
    assert_equal(special.chdtr(0.7,np.inf), 1.0)


def test_chi2c_smalldf():
    assert_almost_equal(special.chdtrc(0.6,3), 1-0.957890536704110)


def test_chi2_inv_smalldf():
    assert_almost_equal(special.chdtri(0.6,1-0.957890536704110), 3)


def test_agm_simple():
    assert_allclose(special.agm(24, 6), 13.4581714817)
    assert_allclose(special.agm(1e30, 1), 2.2292230559453832047768593e28)


def test_legacy():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # Legacy behavior: truncating arguments to integers
        assert_equal(special.bdtrc(1, 2, 0.3), special.bdtrc(1.8, 2.8, 0.3))
        assert_equal(special.bdtr(1, 2, 0.3), special.bdtr(1.8, 2.8, 0.3))
        assert_equal(special.bdtri(1, 2, 0.3), special.bdtri(1.8, 2.8, 0.3))
        assert_equal(special.expn(1, 0.3), special.expn(1.8, 0.3))
        assert_equal(special.hyp2f0(1, 2, 0.3, 1), special.hyp2f0(1, 2, 0.3, 1.8))
        assert_equal(special.nbdtrc(1, 2, 0.3), special.nbdtrc(1.8, 2.8, 0.3))
        assert_equal(special.nbdtr(1, 2, 0.3), special.nbdtr(1.8, 2.8, 0.3))
        assert_equal(special.nbdtri(1, 2, 0.3), special.nbdtri(1.8, 2.8, 0.3))
        assert_equal(special.pdtrc(1, 0.3), special.pdtrc(1.8, 0.3))
        assert_equal(special.pdtr(1, 0.3), special.pdtr(1.8, 0.3))
        assert_equal(special.pdtri(1, 0.3), special.pdtri(1.8, 0.3))
        assert_equal(special.kn(1, 0.3), special.kn(1.8, 0.3))
        assert_equal(special.yn(1, 0.3), special.yn(1.8, 0.3))
        assert_equal(special.smirnov(1, 0.3), special.smirnov(1.8, 0.3))
        assert_equal(special.smirnovi(1, 0.3), special.smirnovi(1.8, 0.3))


@with_special_errors
def test_error_raising():
    assert_raises(special.SpecialFunctionError, special.iv, 1, 1e99j)


def test_xlogy():
    def xfunc(x, y):
        if x == 0 and not np.isnan(y):
            return x
        else:
            return x*np.log(y)

    z1 = np.asarray([(0,0), (0, np.nan), (0, np.inf), (1.0, 2.0)], dtype=float)
    z2 = np.r_[z1, [(0, 1j), (1, 1j)]]

    w1 = np.vectorize(xfunc)(z1[:,0], z1[:,1])
    assert_func_equal(special.xlogy, w1, z1, rtol=1e-13, atol=1e-13)
    w2 = np.vectorize(xfunc)(z2[:,0], z2[:,1])
    assert_func_equal(special.xlogy, w2, z2, rtol=1e-13, atol=1e-13)


def test_xlog1py():
    def xfunc(x, y):
        if x == 0 and not np.isnan(y):
            return x
        else:
            return x * np.log1p(y)

    z1 = np.asarray([(0,0), (0, np.nan), (0, np.inf), (1.0, 2.0),
                     (1, 1e-30)], dtype=float)
    w1 = np.vectorize(xfunc)(z1[:,0], z1[:,1])
    assert_func_equal(special.xlog1py, w1, z1, rtol=1e-13, atol=1e-13)


def test_entr():
    def xfunc(x):
        if x < 0:
            return -np.inf
        else:
            return -special.xlogy(x, x)
    values = (0, 0.5, 1.0, np.inf)
    signs = [-1, 1]
    arr = []
    for sgn, v in itertools.product(signs, values):
        arr.append(sgn * v)
    z = np.array(arr, dtype=float)
    w = np.vectorize(xfunc, otypes=[np.float64])(z)
    assert_func_equal(special.entr, w, z, rtol=1e-13, atol=1e-13)


def test_kl_div():
    def xfunc(x, y):
        if x < 0 or y < 0 or (y == 0 and x != 0):
            # extension of natural domain to preserve convexity
            return np.inf
        elif np.isposinf(x) or np.isposinf(y):
            # limits within the natural domain
            return np.inf
        elif x == 0:
            return y
        else:
            return special.xlogy(x, x/y) - x + y
    values = (0, 0.5, 1.0)
    signs = [-1, 1]
    arr = []
    for sgna, va, sgnb, vb in itertools.product(signs, values, signs, values):
        arr.append((sgna*va, sgnb*vb))
    z = np.array(arr, dtype=float)
    w = np.vectorize(xfunc, otypes=[np.float64])(z[:,0], z[:,1])
    assert_func_equal(special.kl_div, w, z, rtol=1e-13, atol=1e-13)


def test_rel_entr():
    def xfunc(x, y):
        if x > 0 and y > 0:
            return special.xlogy(x, x/y)
        elif x == 0 and y >= 0:
            return 0
        else:
            return np.inf
    values = (0, 0.5, 1.0)
    signs = [-1, 1]
    arr = []
    for sgna, va, sgnb, vb in itertools.product(signs, values, signs, values):
        arr.append((sgna*va, sgnb*vb))
    z = np.array(arr, dtype=float)
    w = np.vectorize(xfunc, otypes=[np.float64])(z[:,0], z[:,1])
    assert_func_equal(special.rel_entr, w, z, rtol=1e-13, atol=1e-13)


def test_huber():
    assert_equal(special.huber(-1, 1.5), np.inf)
    assert_allclose(special.huber(2, 1.5), 0.5 * np.square(1.5))
    assert_allclose(special.huber(2, 2.5), 2 * (2.5 - 0.5 * 2))

    def xfunc(delta, r):
        if delta < 0:
            return np.inf
        elif np.abs(r) < delta:
            return 0.5 * np.square(r)
        else:
            return delta * (np.abs(r) - 0.5 * delta)

    z = np.random.randn(10, 2)
    w = np.vectorize(xfunc, otypes=[np.float64])(z[:,0], z[:,1])
    assert_func_equal(special.huber, w, z, rtol=1e-13, atol=1e-13)


def test_pseudo_huber():
    def xfunc(delta, r):
        if delta < 0:
            return np.inf
        elif (not delta) or (not r):
            return 0
        else:
            return delta**2 * (np.sqrt(1 + (r/delta)**2) - 1)

    z = np.array(np.random.randn(10, 2).tolist() + [[0, 0.5], [0.5, 0]])
    w = np.vectorize(xfunc, otypes=[np.float64])(z[:,0], z[:,1])
    assert_func_equal(special.pseudo_huber, w, z, rtol=1e-13, atol=1e-13)


owenst_dt = [[(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (2.2942365980155351117754771526378292708740231264960539316842063696200884829628237039895484898529192823e-03)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9680322689154555657058017082736661495213677947212098762543622722763825563827711568131850564334565506e-03)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.1597654248037660387449520543820166351754668510505531823382492970457382200077558044792256804713386129e-03)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.3232953135233836153944327583670440589476418747370390302105974330237550687887327299360596774532791101e-03)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.0100550889644229746912561152386170907268949874219104978095620802138249447202878889126283209658742296e-03)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (6.1497543339940146251856615610147485246423382395107105527248575888794201708943383385544040841893167983e-03)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (6.7002307780595028311848002054681870716943983498717211440016321757838493256869247382910637200137362515e-03)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.0087628098924846553420452991323128540455257919474718546084030038511341392097627616477463678020520866e-02)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.0091132928374177373986149936548570045661635017241761865456752105413011145950758665503631981469962091e-02)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.0865842961288026187893734175479235962324504288027096290482416005557459196544605743966959957549259912e-02)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.1928879205559397161128411536383665386049060100190891105612351797899324440156577849032776806264950516e-02)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.2007404932089150629318599721043926151201252590201116565157838995615594563567446592356739043515060955e-02)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.2234231937902890618076422454559854736034641208834938001805660215261543653258044507100832640227607205e-02)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.2254641258057330150164973433843452155232009681583050905550399458557126941497853232806334014703269603e-02)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.2360475720884925881580539738830093572012024162485856485613974712653546627243019666702904836835836072e-02)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.2376176601227851086722830167717452814836846810367599158215995631280835335070487840750421239628452270e-02)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.2382005445939933843659988418503656684093356069262317993257524956338275590041101050569009428193352371e-02)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.2384419169235315605535269562348458006973333200556008698543822798769741450130359222839946856017776574e-02)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.2431007739121873243331263486394085638096884194270101674004097829060031548768693191305694357247777346e-02)],
             [(1.9508080482482910156250000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.2437508801776841503931807680295142938639719448531279010032010195899482047937342387190771017427394090e-02)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (6.0892598894871013831450310453855577024940199295839498702191855220058998766969059376364952526465284021e-04)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (7.8552477503066093316653283632946784434348752481643564867235309304982560891773412550717031568541075705e-04)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (8.3547481001847470306205437554767680175926460100401487634312607482107121403542105369607597913620486228e-04)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.1349538807683529766754432045661620867376550973436541105753149427481686354171267899610423696075448149e-03)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.3081149058186992317938095276814421293879284969724144515996421560499787699936613174360343582962295740e-03)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.5878048364081975081251891304217167013312099142666376506423788580505618186158232170549422219934783478e-03)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7187428864034126218093390684607934848342548100346115458866508939353555304950555823644799745834099854e-03)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4341415275418323804965250950312187178896787958041667325921433680168800975724752491267519263702206171e-03)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4347703667517973497589772453954356420155380457362917154615271851092638128618543252129405995381120823e-03)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.5654934061729800585071531711732581992665178577119680937850556665600001416830920923998758996250562572e-03)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7102222130256680778082229506054624600298089640008649867810816174099490266589359223735451239228867695e-03)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7187366890979789886994424919757198431063606086845524871086681826444014144546460885653698813171873483e-03)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7410410391314317327235243808230003063641913370466248328606151566540037142368309640948307733089157819e-03)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7428627024087233594744881207353300006000606524950207103958084731385200535788238988081899365708614181e-03)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7517497927491308313460905871910944558439108965006558155874933342489541271005569828949406095203595635e-03)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7529832238099327981448373487368565869894878389079013255254799787854521106511736100337800130550804734e-03)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7534351959075352602930158286289964102877576582018338135138282398495212502293273181665468958443704470e-03)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7536214058405533351145775811408280041522270716058519266403220051159633550918364954505859587344152318e-03)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7571031476501931637301192852028739334678122250513760833401608355268943614593156536569593782544447721e-03)],
             [(2.5397357940673828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7575714835398346014438296247850916003189300937943203722758437029746167717384652465698091284467773426e-03)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (3.8940745658072282978586957911412357454682765821749814836616564829173079872316409867511611976849736342e-04)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.0186385594150413563720938913340836408702098697405640386356529741412088880349476855118149225684014609e-04)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.3360688311466125005252448740101520538894144350836290391278541005037227186243743866548835757888360652e-04)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (7.2315251382848029096683072707480453985912558126336386527573914363988514243246183770185987113387535199e-04)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (8.3199073192490712799823192285677346980855542442974978897968476665958917076394761827303266241868465945e-04)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.0061931119630268484487805082491564032326296367388733517020254570427572736254423687590778649658249096e-03)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.0868916378637619572097460390852201239462222301832643217934940452073593117086045031650946336775855746e-03)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.5110157901412780741426775042567283935747358445902234201572377288834365228433712285342965544034552748e-03)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.5113682118315831201987371760486649534073487093876992748945139050826988248442989305885337043132393817e-03)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.5831528976658829836997327286231731322089686134330528543291733473197710170079359571431768003494423563e-03)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.6572169749096491924163109159921676881944978277595840159403244566950242974512385079665033273622688991e-03)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.6612439279967586957140138359964883821502208376509603557655255441681481125629988563423959385188161538e-03)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.6714556308336217753171381866996443084243736381286418638347445677213682261251532342583937566992717151e-03)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.6722624778808367977995147046858444258418194435715990872903875603765262594625732015411127293228438280e-03)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.6761194157103371816977416998275297565936901461474020543117311175561531089768156284177404752935904542e-03)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.6766427142893238870847770149294261771614928728592535910825570178997635000494936535762378417100148802e-03)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.6768336273317476132692430228896823497998904505689956143241080664792267056877546782871997727242873305e-03)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.6769121473395352511439294614727282817917740041746229730216052226350101174826166507927773552375821411e-03)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.6783647772686101175352741425631004867886590069983287539588393296574776707132752277341266060793888350e-03)],
             [(2.7095394134521484375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.6785577567585352522054366157450458673679852116453227468817677148275864001413569620692407110648965204e-03)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (1.2518193788012443276795851413120299200269753997134839357414725807078453807852895413177074062574459983e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.6017131192524860484653082904755463144319056578755543254049199604315320256409581129889188220100675332e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.6989418601481680794687088928083346446849308182951652497815914609838435509719352355142254677347425136e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.2617448922082963500482262139429377748232234486484760599464489111515664263171638244401828016950900851e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.5680868969244219676880291885841322486944409686726018103997641845899952216815685193158880402629881057e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.0254184588420069376985146070357611482226487762093208257139496170242814911605619352622304008788192493e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2207273623191260761088765135977300240322665707700512664514826740445391367247755658749793012689910504e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.9980013151843328316784780978726240749247498721094099058633257828017096023953684879039103434727008834e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.9984119657785781958597459132238262869844521352087962029561547500368164131441225118862815429710046814e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.0703260862041728370659576245908686787340052179469221347721760937789064179808343467245853254198249600e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.1147106923546465821832629249598115340084824540694339394996905180128533667920775236850142274275546847e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.1159840387915100980276614980911990292210811285718215208881956050949671986950287955450032701627341956e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.1185183534259483672043200518429630599245739717501612364496270816510674841376889686418335119606276445e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.1186723241864493000461681837415605766631698783237824272352823729491004303200599388172978998493572651e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.1193039962933989775526962502866835653430470157954273763944875994151168345750596821315210645837667418e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.1193757269936849748788484329732735888257788787846146823686095516612698454403755737711021450244241599e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.1194010233859503504247168257429313033415928042666688891493941858088030867029591583372085048334685616e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.1194112907232908025377411561252030769570977187989123896650298150849929960956301508451077920663478632e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.1195865360805797845274899238563356388625040862289059993505673216344813926110513226705057878442105470e-05)],
             [(3.7676391601562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.1196076674394984851538474558472743623860206363509657114025490053760777584315419851187233731464042438e-05)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (8.5662690242005774274767130763707987993528614801742630335496566146370209271417514083126669616045461378e-07)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.0900281236140833756897189028063017301014220942723434005933613489872783427002407037974019771608384060e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.1540959777806153242771392362703918124972741485847217929394945593368156829224819270759240498193571458e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.5161312090736233561747852036937116557541672344057323423670271062124905049026637976919709047052458961e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7051270424905783025169542049149536395219935839363294071812407155551077040466668316457303522859329106e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.9724678715769244037482092123571540306374663788479049674123607978555309734130726868523015755392533938e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.0795817040191398043424263567962191757037472557255519022522204349902753651852146850591820170442500692e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4307191833605178385606229862116130668615521090413565873903877255305352710538700254685484649102589455e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4308464823678394633503600746862040097797404288873150122619431488682018956825321826951515224453585976e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4507909664589498684163894464363182048040475915401762307161708089639852502457384047804102187882115633e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4592943951673925464215111613277430692495068574835217790794700282048468965909763402584862560218606037e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4594371774547568713304228368508918341624751042093476031145648482511337600502368516401315321516733267e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4596745545140070090072972474469645578592472707194982133265553909128070361671871607799023388858245632e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4596861896172713045944422749671444283555256313260790032383140759138793198269327615599681117232709029e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4597287426499747302875201525892595033234741617042217106132812745025669056583827069526914083405987171e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4597329298883793432147846277285219262094085625441668149457820171724700380728198669500608258527473911e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4597343678116512840267468713261767008946781648893257725200499732688382416798885811132552634329289869e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4597349454353282329105313301767035084970452160206328314378456326037338751004126875652195750901574600e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4597442176920724873959364740982546817016926822287351949691166593634226454830661965132715446192496862e-06)],
             [(4.4206809997558593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4597452528400781200727773347148578394859756646410141125881596127193754432887855670848518267254914405e-06)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (2.7030512493692108520404961144346910482757936596144325548434739193859124095999160447647324248732281266e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.4001961204257077999560568337022275896359092582376928409153038184100873098181155743535515656399122892e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.5865763174010895102088743570945583459801930368357514234365897586603706536239416127639747276368786983e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.5873127634009519966375741174518731947450227924230517140592882611344586923274579883469815721582054606e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.0647891760575599440059235748049403601867666499386019462653800297362340840531051544072513487623576858e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6671405064280408628729977242101837372670118691707019408685080423381390900852164262010873472586715739e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.8773637797154382587904788259943932410704647830705622500186312401651071993804376441937718444542611799e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (6.3586007261371926967730185115181189940789484942140015284401912413759568350541640858830390375625930385e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (6.3586740707374372332973485569090234306280222147671022431220366003790811096796422704344814287495220930e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (6.3678929343758782503016458536052454711146918306294165123428725788945122243573543299366548879178552445e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (6.3697976709206138002086123543340622638372997423028603881457519715496080959825644914170216326503727486e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (6.3698069585593568539264402781860053497105855090925791129085334814190289235696195845243511226314456404e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (6.3698177186121812705152138547119870229643429824637830255849426970233652888315459061488269710390158602e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (6.3698180423290976796441462551201687428763714400203651359954410145327220849923125039677699250278297860e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (6.3698189792530360249383671785674018906755144165043960816578411052673281929176576333478636456259463444e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (6.3698190461805331284389517966235336271958880314168464031037590439186878677082251760405152644565050685e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (6.3698190678811146685407344722258200632425754152246640780982319547357149024559736417057200890976142053e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (6.3698190764065844417399072451975022629094360953832047319459489227517739396875795934462159833958319471e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (6.3698191968944703931621011311534948947853449248283107099042811237939152556708133948113594583286703269e-09)],
             [(5.5699634552001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (6.3698192082456047804471390740510529275942966670459458767792869159634717710567195690244816393281939508e-09)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (8.2307962385446293147850071046224826587132805852272143242479728881172896794860781740349908020844215370e-11)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.0283963277416623201804463973999049414822083348785938125216625919323195393721257543686987937457990009e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.0824132060565853827927990910290994241030457383639324717044353255719802049291298967872069397063967779e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.3637067306879723939356778531945205359591090948707105880052796210528321965355760135823812756446476091e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4907748280031041047512037945282691074168838901476394291803963888017198136855953365863977193964269365e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.6404519162603652163113937567948520472254042578292347991876153369420867813166210608579270223329187334e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.6884629354129612089797429254177374994941601208968768885889005201941965541207489302434302995065797557e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7796463538981782326836339213416487884550654811220357414306858150337863696312639069477301953675244354e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7796543171997010745715805469702186607600802190015449414975905153654849316574711276946688373745954426e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7805404152885821942717099292030784173211902572777571416821782925492529084124531349890902977798741421e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7806623964662454781566924743963930703690998426985131423071504138353834413966206045880875052663629228e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7806626655328540939904279113617568641571526302204491552850403802907480189686813981348689238242752520e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7806629198364844052218637865246853247568449159852061884943363990597331551623161776219554471086443472e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7806629254333125009698672806382167409242163977335864576955477879632901157548742771990285337169366691e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7806629396040104810604532023373989467501438077268692272228398307156674023660836550476652663147212698e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7806629404308586263942384503594964844454580671918598688533381484306094913189987064909013337926319257e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7806629406897213940841350377220148560626236108563851482593678883488715356765508143041510611388964432e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7806629407900543504128069412555720511088916750234168183965682839046874403544178165644184639642311920e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7806629421064262663864058821730696684245317502298186670580356801987156960562796239048798389148023474e-10)],
             [(6.1633396148681640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7806629422179965403723872056775657985224162290477294409748651656860788796972983716944107641204085024e-10)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (1.3669580781844789231318859205091550249740997959753468373446376846248034028456101047437628548246315436e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.5967404486401372797243100260300464158116672635998236429276238874434114738891006150057090087842762702e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.6466760156070684009586269670997596457359323913134978082307922944799545694295895416327570976727357475e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.8321698375844716587008674501164810771773002399055662523502104333023832160916362232464642547970836413e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.8759307737335383399758254031783456791445055801623120234544157960027995573629541976984260396973903010e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.9000570285676015845774377685858438216430943496464835165911732699742559550842111985119555069841751269e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.9028328137378034294263372079340678788666407032030921934259038487347727626372463455064970154351662443e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.9041294978357685290296060622704927756521069133160611338008330578234961679827745190311024858929051122e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.9041294979092973099744857901960622861433317986525710260036208201617850109781518413611874450280708082e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.9041295010550777553751610875065340427437674185310904860169370609863543615660882943795753155486999482e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.9041295010613532817057465452782045710389272370980258343573032510766833159600291118924483360667745084e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.9041295010613532822082452419995266717796119985756107063433937158086204054926729875059835438063538195e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.9041295010613532822828394723339866704841271912806161599337665415177575107487418905738218626912120766e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.9041295010613532822828624914906367279334733981245329141705851890712504833122027489034276619817854068e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.9041295010613532822828798355835259833399566220863644887505845105889160804407154431517877325100181398e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.9041295010613532822828799017042867403567299966784415039105261955855847165921058462530917692733181598e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.9041295010613532822828799152534296333230515004707762706168184614444934735630878169704521494371056818e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.9041295010613532822828799197266779612029779770443204510974618023406248826920962336126781808963586846e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.9041295010613532822828799481835905183520600674347944730883508731563567413646854435472526750350521376e-28)],
             [(1.0937629699707031250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.9041295010613532822828799487875263361351187074625316521313683355995917038576424068701364663344970730e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (1.2689362214738440975744230530813840642017366438214402549707342179362193768665974030615174818332323960e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4820871600688099148496946304860385617692694663435417704398233765308489356931528766928937886837462035e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.5283921217768466688204835884488635140293079747833136961817886487209690019793121346558124445849406329e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7002997885618094418663112283329944302429865243066011745243168463416099919353280213603139074058673097e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7408071415337177747588439840708744441378680021821638872932429990102170826062194946119094954764327183e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7631104043423429890544379537658830829055705877173893405249570800747837325370893065434293959679789472e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7656715289867008983986996838808664018779825214200147275358852745822570395579512155006746539433841575e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7668656499493656799882818426696713620038319132706932057386922564340989854089202167716899922351664784e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7668656500161361789749132514906392224046356981068760227531902585314778436793359639723285274067281188e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7668656528693694874065304221576121157303085121774231388970112986094444501422229592182099774009597556e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7668656528750190206996954456753185240753546405104277431208289524793813700716503716617014163629850580e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7668656528750190211434551044647249184652301621094400273847851601893100236448690271725318553257222813e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7668656528750190212091423973686090102685720495510842139024421254477095600992298619794850256173045604e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7668656528750190212091624988508285806112294085042034469009590489519751071885764887670280618301614988e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7668656528750190212091776179522859054167907673770134325639481170756610993075399638966531954712314258e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7668656528750190212091776752802319450371695791946295951560695147016725890345373773298652708030690968e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7668656528750190212091776870182983774808826534329109140254154030437666593932220225631381605110486071e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7668656528750190212091776908924783113143170712256301664805226802265094764245696413126656359622158170e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7668656528750190212091777155114107479637861421743767225564898980292242558409899940725130730652208333e-28)],
             [(1.0944412231445312500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.7668656528750190212091777160324679577601142852684519223834218056774899852747538639850087655326840009e-28)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (2.2770505103391758119559932793619204322213404907603559304691035183460381049418519786519552498913947220e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.5913310373742305854948973099826412902412422243181917527791986098943406758550748924825052947700591709e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.6533562728906664559656758411974864988422347826034622961209557772614749840151711670498839384876937964e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.8510556422268561141235918740979993338644689486362556771311320700208128632710296762348131844674085466e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.8847214733539339240197490164986845175474033070134399303881639952603323077315692549692458103361595423e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.8978342028077556742204431797715236201890511484459045899348690693300848043699847668815692980931691601e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.8987240504798525250156023922448342887934728092135408244863927293881972387101672243386486978472905447e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.8989802694371870955782034832436956396943073208445904319505185439385041850901217431220210518603326440e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.8989802694374968792321190934511709843700963511595603337229671845229934162325653424198528314502062460e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.8989802694474820292552333686080161459465445552533728944481007903625406727249808867761465178167447887e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.8989802694474846506777458089680506753425813092829929114165843341746642846218708304267190145698278379e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.8989802694474846506777469082879060067801608384996798279004447573208223395610478778654041717263413239e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.8989802694474846506777469857848132561940230361461603024093752181331444098117843122658139081376261736e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.8989802694474846506777469857871726770292766823377376460131099419982133630007388503158425203497459471e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.8989802694474846506777469857883154105936448281784198334785811781141030186489038831911605263613270295e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.8989802694474846506777469857883163793407739745716870170215314065360782442866947737558685724959424131e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.8989802694474846506777469857883165389985145983053088223895299881908771166937953150338200695632808431e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.8989802694474846506777469857883165876350007268614373699756833303491402495835015345459783476689786687e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.8989802694474846506777469857883168230383038892430310577849615145810436821539378957311285467240642390e-37)],
             [(1.2647182464599609375000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.8989802694474846506777469857883168253616188739390762067648898153841901648303957152481569127187620087e-37)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (2.4019827109851080539564115082482784330129765638075052820901304287067752708102440756805399404365634434e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.5986953689988862538052972437743366730756169715711716065325969045294065991953865842876073638881177006e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.6286826789856318753559793827573539310556706086243080911069749176759339093606285740471711896222364768e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.6952518518698465470647958517151551062898479146931531016736093915914392819292438166183134711144202947e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7000145299021305892101281855033088829831777925781401728188126884053573689205260566551293122892350094e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7008122062540901975400176745806389733303556547024291631774255617256215859107971683890208816509621443e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7008251386433437818787796493908228514410011338412569558394939053912490454944147481651521382805485111e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7008264001838319221907854857278497200793714239224307286405852457195979759591596650677391300303121243e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7008264001838319222420312087476725992185317610809049207531281957880689197553004936257817723097406771e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7008264001838319232388285365148916194703641481091676978109011476243165787990842422908375428072821384e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7008264001838319232388298390750736263275415332924648258219594982616661509753455193847960431740101396e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7008264001838319232388298390750736263280566386956254824529685549179941283630933059075956645315497583e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7008264001838319232388298390750736263280625033470452015421064159591341412106638427352250772166209881e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7008264001838319232388298390750736263280625033473795743733983618608468224192996716796941951652578282e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7008264001838319232388298390750736263280625033474421536184942614673350519057927723367982448142594553e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7008264001838319232388298390750736263280625033474421544626888102677111140311789076405181980788895016e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7008264001838319232388298390750736263280625033474421545394077237515246336371967532276179457799793929e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7008264001838319232388298390750736263280625033474421545582675590954463106222861399197285845322149076e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7008264001838319232388298390750736263280625033474421546112686203106573838122860950191865642365348159e-60)],
             [(1.6294471740722656250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.7008264001838319232388298390750736263280625033474421546113283869190062024059478981786014037447271842e-60)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (2.9339321969508820194824162410957321245475751276095254284275486684952702631733123300565451654876560286e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.1587028105600101555380088396000772591825405244281735906006754493251111737478103995487015276176939325e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.1918433632703015481141253344660898224848253746827650953219731886118277912211195912494801869869034860e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2623744743650812448406117886504232160257887720395925880032425421007367043430595169007652998842911985e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2668896346970722766209517924464926753262777793899281916655167273259740080375591578482407262514875140e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2675727314784888623134673458878177516507241985447145304880663806909124950225882757205385350077933998e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2675819320678273051929079409573860828709541978682920090244257259207276397171764876723227853224430995e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2675827221443955788429727047768398972633832465223960687469233384042118226596964999356837310249249421e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2675827221443955788515456999253713386691661279437214926747508919416830299739309876144334876468342482e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2675827221443955790102325732514717605057478132145455205102411416195258515682319106760535853525195689e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2675827221443955790102326791168090830898033189882044739341478649365757224279864467719634996190409554e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2675827221443955790102326791168090830898104919086455115386558410697826337345174209644856720074571280e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2675827221443955790102326791168090830898105570374832874769079690541854972947308291009263110325930413e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2675827221443955790102326791168090830898105570374849420753064022612813407656148154795396086299554848e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2675827221443955790102326791168090830898105570374852198412593742539359670771572549989865121761644544e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2675827221443955790102326791168090830898105570374852198434499967719174390345116976100080334632929478e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2675827221443955790102326791168090830898105570374852198436341534198622242800239351934884440811759396e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2675827221443955790102326791168090830898105570374852198436781928668884596449357570610107817507727354e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2675827221443955790102326791168090830898105570374852198437951783778012443711385134649162178349604948e-63)],
             [(1.6700164794921875000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.2675827221443955790102326791168090830898105570374852198437952773856096479591634871178503416984805049e-63)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (5.5191097659413416120390603603717530958881654160231344923494077130522749174111168055204932246176318447e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.8495563676428042732652229494624460257384969660002337832733437605704915962706516648486344099195679848e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.8925976298439032475208897274848135814122454692724348652734131471380544131423602275343325142702966336e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.9715331370584007247150936906788737328563085358766224392207244315035665385452927231177318228876162757e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.9748822028408810986854506841911127381352642464684912821545808401953346157337564461258280528516262083e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.9752333070684903040184169931416751967100427340190275608621374937679061584135487093747373493558092314e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.9752356839769495177195290813393642680754002003223215693314532791688635802102775166637066883893428949e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.9752358129013488147841069380452158567573785850829814817698351909514857324486545049777484533814220234e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.9752358129013488147841175899503380678541384074644219060692753922324076141500474128271804708158440820e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.9752358129013488147842847933266096144697959593162905141885533398900847126829894445610080004030737558e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.9752358129013488147842847933359938393250458726426983913744216050593101628626574979183629036802766871e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.9752358129013488147842847933359938393250458726436552868715071711210268450358266018308085600765709899e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.9752358129013488147842847933359938393250458726436590760691136190552400127208494766719876030837725551e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.9752358129013488147842847933359938393250458726436590760691184622479196716803922941917532539941455104e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.9752358129013488147842847933359938393250458726436590760691190142354255597210224783224792858614728141e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.9752358129013488147842847933359938393250458726436590760691190142360210440034821006755320193689189313e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.9752358129013488147842847933359938393250458726436590760691190142360584540717206293361955532932882926e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.9752358129013488147842847933359938393250458726436590760691190142360665301500237332177667486724085048e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.9752358129013488147842847933359938393250458726436590760691190142360842654418267485063928057015374736e-74)],
             [(1.8115837097167968750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.9752358129013488147842847933359938393250458726436590760691190142360842705780732804935986650912837289e-74)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (3.4757541366541816087530043985108154809261469072672370276708713846190319704881986921471511392021329473e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.6781773605153348367166316756518684268994345067914801675490837737786153676205200691218643675222354366e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7041770174749299149158235232218706487195712551348399576202288711877707974285631430131188063968820309e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7510994137965218018155250289561790732739962552665520667511661088859562674051101627390829091617767549e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7530002886430463959575876930643074110998409234681366942099052009402027215825745952058162630720569941e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7531916776192058002137778773832104570237939538214740762802404044639614290819898441197291804798644328e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7531928767081717242777146788924588254597241446560147238267793354459039112504528311925943642698301688e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7531929385342995283939172793874315345890002972093747943813795116181090887154103443540762328223120070e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7531929385342995283939202336539492411838485441190143345976330614422050096048254268951089817544952958e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7531929385342995283939658271491430282784731466212600679568203321898699118009215764282210105817341589e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7531929385342995283939658271510825240437323496722368022407716304233137534633200617421755252883180386e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7531929385342995283939658271510825240437323496723323103652504435486817003573719094939210316130109783e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7531929385342995283939658271510825240437323496723326550855718281243537828144681285040311466080277900e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7531929385342995283939658271510825240437323496723326550855721430654504871140990453427367161504176686e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7531929385342995283939658271510825240437323496723326550855721774725362073409183283531564640752643518e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7531929385342995283939658271510825240437323496723326550855721774725658838182999300371458405492332771e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7531929385342995283939658271510825240437323496723326550855721774725676876138082093136394246206244164e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7531929385342995283939658271510825240437323496723326550855721774725680725523918995765964719099269161e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7531929385342995283939658271510825240437323496723326550855721774725689012720962563542884707611803837e-75)],
             [(1.8267517089843750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (3.7531929385342995283939658271510825240437323496723326550855721774725689014846625313142358304379682386e-75)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (2.2661704720764430974809869872537009183690664631966473566021853503794936093284495902662257582880935544e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3779690202261065171908418765533295276499181379212493489185457387274481971131380442059557310535424631e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3911691636232112832565826237055836301379850030421759523484907911117240823563979226621182333274448186e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4128605292089238358464595759233388498565149619919670031020852909946357381061671588281677140613635326e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4135264499849360643108146181530849284983277473372998454957612577782473640811118419954268667424587882e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4135792197319428368845170341234404329321871138818078807445367622455528845931554432697333007096025116e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4135794273865709858864781593891826184127634115252233405318381185560066669709719251364473259195938843e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4135794353127537415575320338487908809611811843447900175898278286558724592052961769599158917517806361e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4135794353127537415575320480709468962812682969751190235679919359561670267288804127501307705933914664e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4135794353127537415575322474310893550758897463628304819827402077773972984210322872278476448645463200e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4135794353127537415575322474310909698664070368971185945227897238166774454032470776318510018614355563e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4135794353127537415575322474310909698664070368971185955425608296490048150424224830731357766776519227e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4135794353127537415575322474310909698664070368971185955446756159834200959381172789599298592971001458e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4135794353127537415575322474310909698664070368971185955446756159836782643541759441951003421558975161e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4135794353127537415575322474310909698664070368971185955446756159837002336761682996580954821583130590e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4135794353127537415575322474310909698664070368971185955446756159837002336811182297126761165386938067e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4135794353127537415575322474310909698664070368971185955446756159837002336813647320662401872367189868e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4135794353127537415575322474310909698664070368971185955446756159837002336814138142215745094050174683e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4135794353127537415575322474310909698664070368971185955446756159837002336815081326895436377879463431e-82)],
             [(1.9150131225585937500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.4135794353127537415575322474310909698664070368971185955446756159837002336815081443354785255124518831e-82)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (1.3191198940585145448581309174518681090051764289807221927708058677248338549641302125264669522357775190e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.3823639515808246053080935915351637779086536894797282072388013030658820997983370793847889339670612116e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.3897233624946945956483009264900533964243722314365082184999045555548795970863161081157275114904993536e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4016283278298765543709141098595984521280239546946313459052974901416203319425657117997156673106683370e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4019767591196865638692758494267009596030139649888562924049835323876662452756940643980904472248048814e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4020032664018558896713654724416555987943423778321901265922273760008339933790517742849551009757067669e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4020033626749832112055433990456565935418431804898311500858256663463497038484070211736088546674516379e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4020033661659135489906039772914664768213205100652410030305851196090064712264964027203233196416711536e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4020033661659135489906039808518827504624362365801346125780841233377854813276177901585903478991039558e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4020033661659135489906040299829828181690863861662643260243009561308998561154120696284537429024387395e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4020033661659135489906040299829831174172117625751056596284269666908835044048190697934609873688543754e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4020033661659135489906040299829831174172117625751056597177989958512360809462582893304451972833973374e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4020033661659135489906040299829831174172117625751056597179675090061924267512353239466051743901542572e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4020033661659135489906040299829831174172117625751056597179675090062069747141037879585683359028579291e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4020033661659135489906040299829831174172117625751056597179675090062081612819095485790147441154724248e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4020033661659135489906040299829831174172117625751056597179675090062081612821217035463935811252829624e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4020033661659135489906040299829831174172117625751056597179675090062081612821319100734608610312292906e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4020033661659135489906040299829831174172117625751056597179675090062081612821339181670850046185308834e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4020033661659135489906040299829831174172117625751056597179675090062081612821377055108762236212803659e-83)],
             [(1.9297767639160156250000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (1.4020033661659135489906040299829831174172117625751056597179675090062081612821377059229951251545745005e-83)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (4.4490393956719026030424824478246800350837744280218290910180597278869150379342865442615594358929646523e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.6600273789204641649803094220052639551915788712820451267116468007867395437058695117794482315674579148e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.6844427427068621216032668985854229014392861610372711611872070206684973773440138791317811209977696523e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.7237030283665899246764140174407580424806024108863172604235560724552924602953196055672061711842974909e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.7248312896491306864998980527199673933841808550598423958906332204905475403120192161958923477722602312e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.7249157978283023663928615368836783505092015309052729307940032878083778409457174890572739349589092501e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.7249160954910707798394202462405978434875086134477969406649339579916384495990395980060843448849083144e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.7249161060751634884465882718461512044401772026732334807183013324654929814180626685532914518705846189e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.7249161060751634884465882805441194749716910849204709764102168782981538112639127570624737937426674967e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.7249161060751634884465883998588687550735336402643048045424606345263232453375273923998030478480680653e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.7249161060751634884465883998588694067791889850573198778488217878470089531437804153550454829371039629e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.7249161060751634884465883998588694067791889850573198779950130821260070025031930459267864046961719810e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.7249161060751634884465883998588694067791889850573198779952788843985217051654150388533122791975375536e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.7249161060751634884465883998588694067791889850573198779952788843985418054487383657663529579342519320e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.7249161060751634884465883998588694067791889850573198779952788843985434185693964071187683583562031625e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.7249161060751634884465883998588694067791889850573198779952788843985434185696604235040993536822658340e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.7249161060751634884465883998588694067791889850573198779952788843985434185696729582223222091828400472e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.7249161060751634884465883998588694067791889850573198779952788843985434185696754131157006320411133980e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.7249161060751634884465883998588694067791889850573198779952788843985434185696800104369383645196363960e-84)],
             [(1.9353897094726562500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (4.7249161060751634884465883998588694067791889850573198779952788843985434185696800109135789609793731663e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (2.8225136662191486900863465575916865952448280708241267714474912083204296123805108858681506777806378759e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9557556195357063262512619003514625185274064353472423178852901024452656999875624280416702137634510411e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9711383594166934108442295450013310087042386864109339727024865071317022776707090728884446523962787645e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9958123636720670973175150415830455711860298251347131063221017373645805033650407448114562509323599727e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9965160415736516114878224347652616441248570285776481929165777330223006625884208523539045865887558752e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9965684058224165658161286419653486321755854347243722330668087364377287599082208927609190552770092082e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9965685879119483199246815080233994718401425290544426432012588216816363985450917188567201782535543745e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9965685943337075842128708134324770240552814009291817788672782289495375208444995930791831628871692958e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9965685943337075842128708182535373738479552224447024588312191943603493971261067796621953246822802129e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9965685943337075842128708842230283305376995684464705428352619057921842285607957649468157621467100551e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9965685943337075842128708842230286747995771257955557983986067545054455535764305523146436193245438972e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9965685943337075842128708842230286747995771257955557984671106045775646675393816309229880068844981389e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9965685943337075842128708842230286747995771257955557984672332748137646619043980790696360140601523723e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9965685943337075842128708842230286747995771257955557984672332748137734378693387067492189594857428585e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9965685943337075842128708842230286747995771257955557984672332748137741374220592630824583003278594369e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9965685943337075842128708842230286747995771257955557984672332748137741374221695961308557179059573598e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9965685943337075842128708842230286747995771257955557984672332748137741374221748054871592455334798295e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9965685943337075842128708842230286747995771257955557984672332748137741374221758237725405813638158493e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9965685943337075842128708842230286747995771257955557984672332748137741374221777251000447793495163427e-84)],
             [(1.9377349853515625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.9965685943337075842128708842230286747995771257955557984672332748137741374221777252932176829112981831e-84)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (2.2438222857362642328721666762700670180512280358418254668975233507745332654116410436011453686965583320e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3401498546424510733553760760713089220194411374612244753632571325026025609278414016278426420435941406e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3507442475443941769350296619070755923978988367103893932068275511001366418117131564067482107360261256e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3668902899384734364002740136245246666651297141478626576353623717333057222726571690487782253713815225e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3672831316167632657724475029929234130566491646454665814596085474916464739333870332873695536029518534e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3673086849227167312500038271366885158855725283973352461790290342204550385443187547817889150752632821e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3673087529822476518694748438059077395092889307169444313706384386840282580125934385201938096902673118e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3673087550081705781915964023061439759020768926901365903615834775454723918691011603336032390489329245e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3673087550081705781915964025388629094045959345392418522712094481659886036526141496643680018703855331e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3673087550081705781915964055675774880697034171474958911013883381096078358602101190347072844155446614e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3673087550081705781915964055675774942048518999361543210821948933436165328793679693511805322280998218e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3673087550081705781915964055675774942048518999361543210822965157609037677948877689026250186946015566e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3673087550081705781915964055675774942048518999361543210822966484600396400657620575715736093645266310e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3673087550081705781915964055675774942048518999361543210822966484600396430678144795337231850694662522e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3673087550081705781915964055675774942048518999361543210822966484600396432759227887237016634150400426e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3673087550081705781915964055675774942048518999361543210822966484600396432759228039403737187181468982e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3673087550081705781915964055675774942048518999361543210822966484600396432759228045805603597866088641e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3673087550081705781915964055675774942048518999361543210822966484600396432759228047008093771242206679e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3673087550081705781915964055675774942048518999361543210822966484600396432759228049122323413146794407e-88)],
             [(1.9857620239257812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (2.3673087550081705781915964055675774942048518999361543210822966484600396432759228049122464370425725085e-88)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.7540378570556640625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-02), (5.3852419933543575619020334768361375203594870664698871979558343050954133295912121894695215466342458640e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.2698680162429809570312500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6131327913383104785242885720800568183822747879375527761193112141461299087908931132220004173820217320e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.3547700643539428710937500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6380132109203215953682341142006484163028723876320845280985884437990702987939134369984972274801723921e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (1.8838196992874145507812500000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6756432225141755716181254364578838398260084711471148446038558395555151245520302395698427479909090188e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.2103404998779296875000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6765370329975020887578693197006018491279730057642519497656614145617760862881464201058169472650265027e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (2.7849817276000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6765940072383066005097652539757437078433621689726193895722238525223712942947787019057651330175067344e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (3.0816698074340820312500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6765941529819362163555178434780673589908418376638967134553680595361500546239249492182079654295352261e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4688143730163574218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6765941572108102929182392355720017900893213514144857837052426624327588242025916320562922691572615536e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (5.4722046852111816406250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6765941572108102929182392359376746516400752338011814009982639136654163659015328286598007640951121062e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (6.3235926628112792968750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6765941572108102929182392406617201965462427239939549761682717880384292426173011499264309557470010809e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.1472349166870117187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6765941572108102929182392406617202048399346202690008425364176650068311070333103178533019023731342326e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (8.3500838279724121093750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6765941572108102929182392406617202048399346202690008425365120016320727925714718349788376825053245493e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.0579175949096679687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6765941572108102929182392406617202048399346202690008425365121190758925890311477970813572027755455370e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.1337585449218750000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6765941572108102929182392406617202048399346202690008425365121190758925912633751741867242391943135847e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.5750665664672851562500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6765941572108102929182392406617202048399346202690008425365121190758925914149018322974493002246798279e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6488833427429199218750000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6765941572108102929182392406617202048399346202690008425365121190758925914149018421601995655864249226e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6769475936889648437500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6765941572108102929182392406617202048399346202690008425365121190758925914149018425679363648779892917e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.6886777877807617187500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6765941572108102929182392406617202048399346202690008425365121190758925914149018426440620859348118110e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9288129806518554687500000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6765941572108102929182392406617202048399346202690008425365121190758925914149018427767218487038675738e-89)],
             [(1.9929222106933593750000000000000000000000000000000000000000000000000000000000000000000000000000000000e+01), (9.9646115303039550781250000000000000000000000000000000000000000000000000000000000000000000000000000000e-01), (5.6765941572108102929182392406617202048399346202690008425365121190758925914149018427767301460804161903e-89)]]


def test_owens_t():
    for data in owenst_dt:
        assert_almost_equal(special.owens_t(data[0], data[1]), data[2], decimal=1)


if __name__ == "__main__":
    run_module_suite()
