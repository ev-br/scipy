from __future__ import division

import numpy as np
from numpy.testing import (assert_raises, assert_allclose, assert_equal,
                           assert_, TestCase, run_module_suite, dec,
                           assert_almost_equal)
from scipy.optimize import approx_derivatives, check_derivatives
from scipy.optimize._numdiff import (_adjust_step_1st_order,
                                     _adjust_step_2nd_order)


class TestAdjustStep1stOrder:
    def test_no_bounds(self):
        x0 = np.zeros(3)
        h = np.ones(3) * 1e-2
        h_adjusted = _adjust_step_1st_order(x0, h, (None, None))
        assert_allclose(h_adjusted, h)

    def test_single_bound(self):
        x0 = np.zeros(3)
        h = np.ones(3) * 1e-1
        lb = -np.ones(3)
        ub = np.ones(3)
        h_adjusted = _adjust_step_1st_order(x0, h, (lb, None))
        assert_equal(h_adjusted, h)
        h_adjusted = _adjust_step_1st_order(x0, h, (None, ub))
        assert_allclose(h_adjusted, -h)

    def test_general(self):
        lb = np.array([-1, -1, -0.05])
        ub = np.array([1, 1, 0.05])
        x0 = np.array([-0.95, 0.95, 0.01])
        h = np.array([0.1, 0.1, 0.1])
        h_adjusted = _adjust_step_1st_order(x0, h, (lb, ub))
        assert_allclose(h_adjusted, np.array([0.1, -0.1, -0.06]))

    def test_scalar_case(self):
        lb = -1.0
        ub = 1.0
        x0 = np.array(0.95)
        h = np.array(0.1)
        h_adjusted = _adjust_step_1st_order(x0, h, (lb, ub))
        assert_almost_equal(h_adjusted, -0.1)


class TestAdjustStep2ndOrder:
    def test_no_bounds(self):
        x0 = np.zeros(3)
        h = np.ones(3) * 1e-2
        h_adjusted, scheme = _adjust_step_2nd_order(x0, h, (None, None))
        assert_allclose(h_adjusted, h)
        assert_equal(scheme, np.zeros_like(h, dtype=int))

    def test_single_bound(self):
        x0 = np.array([0.0, 0.95, -0.95])
        h = np.ones(3) * 1e-1
        lb = -np.ones(3)
        ub = np.ones(3)
        h_adjusted, scheme = _adjust_step_2nd_order(x0, h, (lb, None))
        assert_allclose(h_adjusted, h)
        assert_equal(scheme, np.array([0, 0, 1]))

        h_adjusted, scheme = _adjust_step_2nd_order(x0, h, (None, ub))
        assert_allclose(h_adjusted, np.array([1e-1, -1e-1, 1e-1]))
        assert_equal(scheme, np.array([0, 1, 0]))

    def test_general(self):
        x0 = np.array([0.0, 0.95, -0.95, 0.01])
        h = np.ones(4) * 1e-1
        lb = np.array([-1.0, -1.0, -1.0, -0.05])
        ub = np.array([1.0, 1.0, 1.0, 0.05])
        h_adjusted, scheme = _adjust_step_2nd_order(x0, h, (lb, ub))
        assert_allclose(h_adjusted, np.array([1e-1, -1e-1, 1e-1, 0.04]))
        assert_equal(scheme, np.array([0, 1, 1, 0]))

    def test_scalar(self):
        h = np.array(0.1)
        x0 = np.array(0.95)
        lb = -1
        ub = 1
        h_adjusted, scheme = _adjust_step_2nd_order(x0, h, (None, None))
        assert_allclose(h_adjusted, 0.1)
        assert_equal(scheme, 0)
        h_adjusted, scheme = _adjust_step_2nd_order(x0, h, (lb, None))
        assert_allclose(h_adjusted, 0.1)
        assert_equal(scheme, 0)
        h_adjusted, scheme = _adjust_step_2nd_order(x0, h, (None, ub))
        assert_allclose(h_adjusted, -0.1)
        assert_equal(scheme, 1)
        lb = -0.05
        ub = 0.05
        x0 = np.array(-0.01)
        h_adjusted, scheme = _adjust_step_2nd_order(x0, h, (lb, ub))
        assert_allclose(h_adjusted, 0.04)
        assert_equal(scheme, 0)


class TestApproxDerivatives(object):
    def fun_scalar_scalar(self, x):
        return np.sinh(x)

    def jac_scalar_scalar(self, x):
        return np.cosh(x)

    def fun_scalar_vector(self, x):
        return np.array([x**2, np.tan(x), np.exp(x)])

    def jac_scalar_vector(self, x):
        return np.array([2 * x, np.cos(x) ** -2, np.exp(x)])

    def fun_vector_scalar(self, x):
        return np.sin(x[0] * x[1]) * np.log(x[0])

    def jac_vector_scalar(self, x):
        return np.array([
            x[1] * np.cos(x[0] * x[1]) * np.log(x[0]) +
            np.sin(x[0] * x[1]) / x[0],
            x[0] * np.cos(x[0] * x[1]) * np.log(x[0])
        ])

    def fun_vector_vector(self, x):
        return np.array([
            x[0] * np.sin(x[1]),
            x[1] * np.cos(x[0]),
            x[0] ** 3 * x[1] ** -0.5
        ])

    def jac_vector_vector(self, x):
        return np.array([
            [np.sin(x[1]), x[0] * np.cos(x[1])],
            [-x[1] * np.sin(x[0]), np.cos(x[0])],
            [3 * x[0] ** 2 * x[1] ** -0.5, -0.5 * x[0] ** 3 * x[1] ** -1.5]
        ])

    def fun_parametrized(self, x, c0, c1):
        return np.array([np.exp(c0 * x[0]), np.exp(c1 * x[1])])

    def jac_parametrized(self, x, c0, c1):
        return np.array([
            [c0 * np.exp(c0 * x[0]), 0],
            [0, c1 * np.exp(c1 * x[1])]
        ])

    def fun_zero_jacobian(self, x):
        return np.array([x[0] * x[1], np.cos(x[0] * x[1])])

    def jac_zero_jacobian(self, x):
        return np.array([
            [x[1], x[0]],
            [-x[1] * np.sin(x[0] * x[1]), -x[0] * np.sin(x[0] * x[1])]
        ])

    def test_scalar_scalar(self):
        x0 = 1.0
        jac_diff_f = approx_derivatives(self.fun_scalar_scalar, x0,
                                        method='forward')
        jac_diff_c = approx_derivatives(self.fun_scalar_scalar, x0)
        jac_true = self.jac_scalar_scalar(x0)
        assert_allclose(jac_diff_f, jac_true, rtol=1e-6)
        assert_allclose(jac_diff_c, jac_true, rtol=1e-9)

    def test_scalar_vector(self):
        x0 = 0.5
        jac_diff_f = approx_derivatives(self.fun_scalar_vector, x0,
                                        method='forward')
        jac_diff_c = approx_derivatives(self.fun_scalar_vector, x0)
        jac_true = self.jac_scalar_vector(x0)
        assert_allclose(jac_diff_f, jac_true, rtol=1e-6)
        assert_allclose(jac_diff_c, jac_true, rtol=1e-9)

    def test_vector_scalar(self):
        x0 = np.array([100.0, -0.5])
        jac_diff_f = approx_derivatives(self.fun_vector_scalar, x0,
                                        method='forward')
        jac_diff_c = approx_derivatives(self.fun_vector_scalar, x0)
        jac_true = self.jac_vector_scalar(x0)
        assert_allclose(jac_diff_f, jac_true, rtol=1e-6)
        assert_allclose(jac_diff_c, jac_true, rtol=1e-7)

    def test_vector_vector(self):
        x0 = np.array([-100.0, 0.2])
        jac_diff_f = approx_derivatives(self.fun_vector_vector, x0,
                                        method='forward')
        jac_diff_c = approx_derivatives(self.fun_vector_vector, x0)
        jac_true = self.jac_vector_vector(x0)
        assert_allclose(jac_diff_f, jac_true, rtol=1e-5)
        assert_allclose(jac_diff_c, jac_true, rtol=1e-6)

    def test_custom_rel_step(self):
        x0 = np.array([-0.1, 0.1])
        jac_diff_f = approx_derivatives(self.fun_vector_vector, x0,
                                        method='forward', rel_step=1e-4)
        jac_diff_c = approx_derivatives(self.fun_vector_vector, x0,
                                        rel_step=1e-4)
        jac_true = self.jac_vector_vector(x0)
        assert_allclose(jac_diff_f, jac_true, rtol=1e-3)
        assert_allclose(jac_diff_c, jac_true, rtol=1e-4)

    def test_with_args(self):
        x0 = np.array([1.0, 1.0])
        c0 = -1.0
        c1 = 1.0
        f0 = self.fun_parametrized(x0, c0, c1)
        jac_diff_f = approx_derivatives(self.fun_parametrized, x0,
                                        method='forward', f0=f0, args=(c0, c1))
        jac_diff_c = approx_derivatives(self.fun_parametrized, x0,
                                        f0=f0, args=(c0, c1))
        jac_true = self.jac_parametrized(x0, c0, c1)
        assert_allclose(jac_diff_f, jac_true, rtol=1e-6)
        assert_allclose(jac_diff_c, jac_true, rtol=1e-9)

    def test_with_bounds_forward(self):
        lb = -np.ones(2)
        ub = np.ones(2)

        x0 = np.array([-2.0, 0.2])
        assert_raises(ValueError, approx_derivatives,
                      self.fun_vector_vector, x0, bounds=(lb, ub))

        x0 = np.array([-1.0, 1.0])
        jac_diff = approx_derivatives(self.fun_vector_vector, x0,
                                      method='forward', bounds=(lb, ub))
        jac_true = self.jac_vector_vector(x0)
        assert_allclose(jac_diff, jac_true, rtol=1e-6)

    def test_with_bounds_central(self):
        lb = np.array([1.0, 1.0])
        ub = np.array([2.0, 2.0])

        x0 = np.array([1.0, 2.0])
        jac_true = self.jac_vector_vector(x0)

        jac_diff = approx_derivatives(self.fun_vector_vector, x0)
        assert_allclose(jac_diff, jac_true, rtol=1e-9)

        jac_diff = approx_derivatives(self.fun_vector_vector, x0,
                                      bounds=(lb, None))
        assert_allclose(jac_diff, jac_true, rtol=1e-9)

        jac_diff = approx_derivatives(self.fun_vector_vector, x0,
                                      bounds=(None, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-9)

        jac_diff = approx_derivatives(self.fun_vector_vector, x0,
                                      bounds=(lb, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-9)

        lb = x0 - 1e-8
        ub = x0 + 1e-6
        jac_diff = approx_derivatives(self.fun_vector_vector, x0,
                                      bounds=(lb, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-6)

    def test_check_derivatives(self):
        x0 = np.array([-10.0, 10])
        accuracy = check_derivatives(self.fun_vector_vector,
                                     self.jac_vector_vector, x0)
        assert_(accuracy < 1e-9)
        accuracy = check_derivatives(self.fun_vector_vector,
                                     self.jac_vector_vector, x0,
                                     method='forward')
        assert_(accuracy < 1e-6)

        x0 = np.array([0.0, 0.0])
        accuracy = check_derivatives(self.fun_zero_jacobian,
                                     self.jac_zero_jacobian, x0)
        assert_(accuracy == 0)
        accuracy = check_derivatives(self.fun_zero_jacobian,
                                     self.jac_zero_jacobian, x0,
                                     method='forward')
        assert_(accuracy == 0)


if __name__ == '__main__':
    run_module_suite()
