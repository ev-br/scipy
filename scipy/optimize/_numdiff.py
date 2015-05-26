"""Routines for numerical differentiation."""
from __future__ import division

import numpy as np

__all__ = ['approx_derivatives', 'check_derivatives']


def _adjust_step_1st_order(x0, h, bounds):
    """Adjust a step for the forward finite difference derivative estimation,
    such that the perturbed point is inside the bounds.

    This function tries to select -h[i] or +h[i] as a step for i-th variable.
    If both such steps move the point out of the boundaries, then the largest
    allowed step is selected with positive or negative sign.

    Parameters
    ----------
    x0 : ndarray
        Initial point. Assumed to lie inside the bounds, no checks of that
        are done.
    h : ndarray
        Desired positive step sizes. Might be decreased in extreme cases.
    bounds : tuple of array_like
        Lower and upper bounds on the variables. None means no bounds.

    Returns
    -------
    h_adjusted : ndarray
        Adjusted steps to use in derivative estimation.
    """
    lb, ub = bounds
    if ub is None:
        return h
    if lb is None:
        return -h

    lb = np.asarray(lb)
    ub = np.asarray(ub)

    lower_dist = x0 - lb
    upper_dist = ub - x0
    forward = upper_dist > lower_dist

    h_adjusted = np.empty_like(h)
    h_adjusted[forward] = np.minimum(upper_dist[forward], h[forward])
    h_adjusted[~forward] = -np.minimum(lower_dist[~forward], h[~forward])

    return h_adjusted


def _adjust_step_2nd_order(x0, h, bounds):
    """Select a finite difference scheme and adjust the step size for the
    first derivative estimation with the second order accuracy, such that
    perturbed points are inside the bounds.

    This function chooses between central finite difference and 3-point
    forward (backward) finite difference. Both methods provide second order
    accuracy.Central difference scheme is chosen with higher priority.

    Parameters
    ----------
    x0 : ndarray
        Initial point. Assumed to lie inside the bounds, no checks of that
        are done.
    h : ndarray
        Desired positive step sizes. Might be decreased in extreme cases.
    bounds : tuple of array_like
        Lower and upper bounds on the variables. None means no bounds.

    Returns
    -------
    h_adjusted : ndarray
        Adjusted steps to use in derivative estimation.
    scheme : ndarray
        Array containing 0 and 1. If 0 then use central finite difference,
        if 1 then use 3-point forward finite difference.
    """
    lb, ub = bounds
    if lb is None and ub is None:
        return h, np.zeros_like(h, dtype=int)

    if lb is None:
        lb = np.empty_like(h)
        lb.fill(-np.inf)

    if ub is None:
        ub = np.empty_like(h)
        ub.fill(np.inf)

    scheme = np.zeros_like(h, dtype=int)
    h_adjusted = h.copy()

    lower_dist = x0 - lb
    upper_dist = ub - x0

    central = (lower_dist >= h) & (upper_dist >= h)

    forward = ~central & (upper_dist >= 2 * h)
    scheme[forward] = 1
    h_adjusted[forward] = h[forward]

    backward = ~central & (lower_dist >= 2 * h)
    scheme[backward] = 1
    h_adjusted[backward] = -h[backward]

    central_adjusted = ~central & ~backward & ~forward
    h_adjusted[central_adjusted] = np.minimum(lower_dist[central_adjusted],
                                              upper_dist[central_adjusted])
    return h_adjusted, scheme


def _approx_forward(f, x0, h, f0, args, bounds):
    h = _adjust_step_1st_order(x0, h, bounds)

    if f0 is None:
        f0 = f(x0, *args)

    if x0.ndim == 0:
        x = x0 + h
        dx = x - x0
        df = f(x, *args) - f0
        J = df / dx
    else:
        J = []
        for i in range(x0.size):
            x = np.copy(x0)
            x[i] += h[i]
            dx = x[i] - x0[i]
            df = f(x, *args) - f0
            J.append(df / dx)
        J = np.array(J).T
    return J


def _approx_central(f, x0, h, f0, args, bounds):
    h, scheme = _adjust_step_2nd_order(x0, h, bounds)

    if np.any(scheme == 1) and f0 is None:
        f0 = f(x0, *args)

    if x0.ndim == 0:
        if scheme == 0:
            x1 = x0 - h
            x2 = x0 + h
            dx = x2 - x1
            f1 = f(x1, *args)
            f2 = f(x2, *args)
            J = (f2 - f1) / dx
        else:
            x1 = x0 + h
            x2 = x0 + 2 * h
            dx = x2 - x0
            f1 = f(x1, *args)
            f2 = f(x2 * h, *args)
            J = (-3 * f0 + 4 * f1 - f2) / dx
    else:
        J = []
        for i in range(x0.size):
            x1 = np.copy(x0)
            x2 = np.copy(x0)
            if scheme[i] == 0:
                x1[i] -= h[i]
                x2[i] += h[i]
                dx = x2[i] - x1[i]
                f1 = f(x1, *args)
                f2 = f(x2, *args)
                J.append((f2 - f1) / dx)
            else:
                x1[i] += h[i]
                x2[i] += 2 * h[i]
                f1 = f(x1, *args)
                f2 = f(x2, *args)
                dx = x2[i] - x0[i]
                J.append((-3 * f0 + 4 * f1 - f2) / dx)
        J = np.array(J).T

    return J


def approx_derivatives(f, x0, method='central', rel_step=None,
                       f0=None, args=(), bounds=(None, None)):
    """Compute a finite difference approximation of the derivatives of a
    vector-valued function.

    If a function maps from R^n to R^m, its derivatives form m x n matrix
    called Jacobian, where an element (i, j) is equal to the partial derivative
    of f[i] with respect to x[j].

    Parameters
    ----------
    f : callable
        The function of which to estimate the derivatives. Takes and returns
        array_like.
    x0 : array_like
        The point at which to estimate the derivatives.
    method : {'central', 'forward'}, optional
        Finite difference scheme to use.
    rel_step : None or array_like, optional
        Relative step size. The actual step size is computed as
        ``h[i] = rel_step[i] * max(1, abs(x[i])``. If None, it is set to
        eps**0.5 for `method`="center" and eps**(1/3) for `method`="forward",
        where eps is the machine epsilon for double precision floating point
        numbers.
    f0 : None or array_like, optional
        If not None it is assumed to be equal to f(x0), in this case the
        f(x0) is not called.
    args : tuple, optional
        Additional arguments passed to `f`.
    bounds : 2-tuple of array_like or None, optional
        Lower and upper bounds on components of `x`. None means that no bounds
        are imposed.

    Returns
    -------
    J : ndarray
        Finite difference approximation of derivatives in the form of Jacobian
        matrix. If n > 1 and m > 1 then it's a 2d-array. If n = 1 and m = 1
        then it's a scalar. Otherwise it's a 1d-array.

    See Also
    --------
    check_derivatives : Function for checking if your implementations of
                        function and its derivatives conform with each other.
    """

    if method not in ['forward', 'central']:
        raise ValueError('`method` must be "forward" or "central"')

    x0 = np.asarray(x0)
    lb, ub = bounds
    if (lb is not None and np.any(x0 < lb) or
            ub is not None and np.any(x0 > ub)):
        raise ValueError("`x0` violates bound constraints.")

    if rel_step is None:
        if method == 'forward':
            rel_step = np.sqrt(np.finfo(np.float64).eps)
        elif method == 'central':
            rel_step = np.finfo(np.float64).eps ** (1 / 3)

    if np.any(rel_step <= 0):
        raise ValueError("`rel_step` must have positive components")

    h = rel_step * np.maximum(1, np.abs(x0))

    if method == 'forward':
        return _approx_forward(f, x0, h, f0, args, bounds)
    elif method == 'central':
        return _approx_central(f, x0, h, f0, args, bounds)


def check_derivatives(f, jac, x0, method='central', rel_step=None,
                      args=(), bounds=(None, None)):
    """Check the correctness of a function computing derivatives (Jacobian or
    gradient) by comparison with finite difference approximation.

    Parameters
    ----------
    f : callable
        Function of which to test the derivatives.
    jac : callable
        Function which computes Jacobian matrix of `f`.
    x0 : array_like
        Point at which to estimate the derivatives.
    method : {'central', 'forward'}, optional
        Finite difference scheme to use.
    rel_step : None or array_like, optional
        Relative step size. The actual step size is computed as
        ``h[i] = rel_step[i] * max(1, abs(x[i])``. If None, it is set to
        eps**0.5 for `method`="center" and eps**(1/3) for `method`="forward",
        where eps is the machine epsilon for double precision floating point
        numbers.
    args : tuple, optional
        Additional arguments passed to `f`.
    bounds : 2-tuple of array_like or None, optional
        Lower and upper bounds on components of `x`. None means that no bounds
        are imposed.

    Returns
    -------
    accuracy : float
        The maximum among all relative errors for elements with absolute values
        higher than 1 and absolute errors for elements with absolute values
        less or equal than 1. If `accuracy` is on the order of 1e-6 or lower,
        then it is likely that your `jac` implementation is correct.

    See Also
    --------
    approx_derivatives : Function computing finite difference approximation of
                         derivatives.
    """
    jac_computed = jac(x0, *args)
    jac_diff = approx_derivatives(f, x0, method=method, rel_step=rel_step,
                                  args=args, bounds=bounds)
    abs_err = np.abs(jac_computed - jac_diff)
    return np.max(abs_err / np.maximum(1, np.abs(jac_diff)))
