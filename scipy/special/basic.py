#
# Author:  Travis Oliphant, 2002
#

from __future__ import division, print_function, absolute_import

import warnings

import numpy as np
import math
from scipy._lib.six import xrange
from numpy import (pi, asarray, floor, isscalar, iscomplex, real,
                   imag, sqrt, where, mgrid, sin, place, issubdtype,
                   extract, less, inexact, nan, zeros, sinc)
from . import _ufuncs as ufuncs
from ._ufuncs import (ellipkm1, mathieu_a, mathieu_b, iv, jv, gamma,
                      psi, _zeta, hankel1, hankel2, yv, kv, _gammaln,
                      ndtri, poch, binom, hyp0f1, ndtr, erf, erfc)
from . import specfun
from . import orthogonal
from ._comb import _comb_int


__all__ = ['agm', 'ai_zeros', 'assoc_laguerre', 'bei_zeros', 'beip_zeros',
           'ber_zeros', 'bernoulli', 'berp_zeros', 'bessel_diff_formula',
           'bi_zeros', 'clpmn', 'comb', 'digamma', 'diric', 'ellipk',
           'erf_zeros', 'erfcinv', 'erfinv', 'euler', 'factorial',
           'factorialk', 'factorial2', 'fresnel_zeros',
           'fresnelc_zeros', 'fresnels_zeros', 'gamma', 'gammaln', 'h1vp',
           'h2vp', 'hankel1', 'hankel2', 'hyp0f1', 'iv', 'ivp', 'jn_zeros',
           'jnjnp_zeros', 'jnp_zeros', 'jnyn_zeros', 'jv', 'jvp', 'kei_zeros',
           'keip_zeros', 'kelvin_zeros', 'ker_zeros', 'kerp_zeros', 'kv',
           'kvp', 'lmbda', 'lpmn', 'lpn', 'lqmn', 'lqn', 'mathieu_a',
           'mathieu_b', 'mathieu_even_coef', 'mathieu_odd_coef', 'ndtri',
           'obl_cv_seq', 'owens_t', 'pbdn_seq', 'pbdv_seq', 'pbvv_seq', 'perm',
           'polygamma', 'pro_cv_seq', 'psi', 'riccati_jn', 'riccati_yn',
           'sinc', 'sph_in', 'sph_inkn',
           'sph_jn', 'sph_jnyn', 'sph_kn', 'sph_yn', 'y0_zeros', 'y1_zeros',
           'y1p_zeros', 'yn_zeros', 'ynp_zeros', 'yv', 'yvp', 'zeta']


def diric(x, n):
    """Periodic sinc function, also called the Dirichlet function.

    The Dirichlet function is defined as::

        diric(x) = sin(x * n/2) / (n * sin(x / 2)),

    where `n` is a positive integer.

    Parameters
    ----------
    x : array_like
        Input data
    n : int
        Integer defining the periodicity.

    Returns
    -------
    diric : ndarray

    Examples
    --------
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(-8*np.pi, 8*np.pi, num=201)
    >>> plt.figure(figsize=(8, 8));
    >>> for idx, n in enumerate([2, 3, 4, 9]):
    ...     plt.subplot(2, 2, idx+1)
    ...     plt.plot(x, special.diric(x, n))
    ...     plt.title('diric, n={}'.format(n))
    >>> plt.show()

    The following example demonstrates that `diric` gives the magnitudes
    (modulo the sign and scaling) of the Fourier coefficients of a
    rectangular pulse.

    Suppress output of values that are effectively 0:

    >>> np.set_printoptions(suppress=True)

    Create a signal `x` of length `m` with `k` ones:

    >>> m = 8
    >>> k = 3
    >>> x = np.zeros(m)
    >>> x[:k] = 1

    Use the FFT to compute the Fourier transform of `x`, and
    inspect the magnitudes of the coefficients:

    >>> np.abs(np.fft.fft(x))
    array([ 3.        ,  2.41421356,  1.        ,  0.41421356,  1.        ,
            0.41421356,  1.        ,  2.41421356])

    Now find the same values (up to sign) using `diric`.  We multiply
    by `k` to account for the different scaling conventions of
    `numpy.fft.fft` and `diric`:

    >>> theta = np.linspace(0, 2*np.pi, m, endpoint=False)
    >>> k * special.diric(theta, k)
    array([ 3.        ,  2.41421356,  1.        , -0.41421356, -1.        ,
           -0.41421356,  1.        ,  2.41421356])
    """
    x, n = asarray(x), asarray(n)
    n = asarray(n + (x-x))
    x = asarray(x + (n-n))
    if issubdtype(x.dtype, inexact):
        ytype = x.dtype
    else:
        ytype = float
    y = zeros(x.shape, ytype)

    # empirical minval for 32, 64 or 128 bit float computations
    # where sin(x/2) < minval, result is fixed at +1 or -1
    if np.finfo(ytype).eps < 1e-18:
        minval = 1e-11
    elif np.finfo(ytype).eps < 1e-15:
        minval = 1e-7
    else:
        minval = 1e-3

    mask1 = (n <= 0) | (n != floor(n))
    place(y, mask1, nan)

    x = x / 2
    denom = sin(x)
    mask2 = (1-mask1) & (abs(denom) < minval)
    xsub = extract(mask2, x)
    nsub = extract(mask2, n)
    zsub = xsub / pi
    place(y, mask2, pow(-1, np.round(zsub)*(nsub-1)))

    mask = (1-mask1) & (1-mask2)
    xsub = extract(mask, x)
    nsub = extract(mask, n)
    dsub = extract(mask, denom)
    place(y, mask, sin(nsub*xsub)/(nsub*dsub))
    return y


def gammaln(x):
    """
    Logarithm of the absolute value of the Gamma function for real inputs.

    Parameters
    ----------
    x : array-like
        Values on the real line at which to compute ``gammaln``

    Returns
    -------
    gammaln : ndarray
        Values of ``gammaln`` at x.

    See Also
    --------
    gammasgn : sign of the gamma function
    loggamma : principal branch of the logarithm of the gamma function

    Notes
    -----
    When used in conjunction with `gammasgn`, this function is useful
    for working in logspace on the real axis without having to deal with
    complex numbers, via the relation ``exp(gammaln(x)) = gammasgn(x)*gamma(x)``.

    Note that `gammaln` currently accepts complex-valued inputs, but it is not
    the same function as for real-valued inputs, and the branch is not
    well-defined --- using `gammaln` with complex is deprecated and will be
    disallowed in future Scipy versions.

    For complex-valued log-gamma, use `loggamma` instead of `gammaln`.

    """
    if np.iscomplexobj(x):
        warnings.warn(("Use of gammaln for complex arguments is "
                       "deprecated as of scipy 0.18.0. Use "
                       "scipy.special.loggamma instead."),
                      DeprecationWarning)
    return _gammaln(x)


def jnjnp_zeros(nt):
    """Compute zeros of integer-order Bessel functions Jn and Jn'.

    Results are arranged in order of the magnitudes of the zeros.

    Parameters
    ----------
    nt : int
        Number (<=1200) of zeros to compute

    Returns
    -------
    zo[l-1] : ndarray
        Value of the lth zero of Jn(x) and Jn'(x). Of length `nt`.
    n[l-1] : ndarray
        Order of the Jn(x) or Jn'(x) associated with lth zero. Of length `nt`.
    m[l-1] : ndarray
        Serial number of the zeros of Jn(x) or Jn'(x) associated
        with lth zero. Of length `nt`.
    t[l-1] : ndarray
        0 if lth zero in zo is zero of Jn(x), 1 if it is a zero of Jn'(x). Of
        length `nt`.

    See Also
    --------
    jn_zeros, jnp_zeros : to get separated arrays of zeros.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt > 1200):
        raise ValueError("Number must be integer <= 1200.")
    nt = int(nt)
    n, m, t, zo = specfun.jdzo(nt)
    return zo[1:nt+1], n[:nt], m[:nt], t[:nt]


def jnyn_zeros(n, nt):
    """Compute nt zeros of Bessel functions Jn(x), Jn'(x), Yn(x), and Yn'(x).

    Returns 4 arrays of length `nt`, corresponding to the first `nt` zeros of
    Jn(x), Jn'(x), Yn(x), and Yn'(x), respectively.

    Parameters
    ----------
    n : int
        Order of the Bessel functions
    nt : int
        Number (<=1200) of zeros to compute

    See jn_zeros, jnp_zeros, yn_zeros, ynp_zeros to get separate arrays.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not (isscalar(nt) and isscalar(n)):
        raise ValueError("Arguments must be scalars.")
    if (floor(n) != n) or (floor(nt) != nt):
        raise ValueError("Arguments must be integers.")
    if (nt <= 0):
        raise ValueError("nt > 0")
    return specfun.jyzo(abs(n), nt)


def jn_zeros(n, nt):
    """Compute zeros of integer-order Bessel function Jn(x).

    Parameters
    ----------
    n : int
        Order of Bessel function
    nt : int
        Number of zeros to return

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           http://jin.ece.illinois.edu/specfunc.html

    """
    return jnyn_zeros(n, nt)[0]


def jnp_zeros(n, nt):
    """Compute zeros of integer-order Bessel function derivative Jn'(x).

    Parameters
    ----------
    n : int
        Order of Bessel function
    nt : int
        Number of zeros to return

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           http://jin.ece.illinois.edu/specfunc.html

    """
    return jnyn_zeros(n, nt)[1]


def yn_zeros(n, nt):
    """Compute zeros of integer-order Bessel function Yn(x).

    Parameters
    ----------
    n : int
        Order of Bessel function
    nt : int
        Number of zeros to return

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           http://jin.ece.illinois.edu/specfunc.html

    """
    return jnyn_zeros(n, nt)[2]


def ynp_zeros(n, nt):
    """Compute zeros of integer-order Bessel function derivative Yn'(x).

    Parameters
    ----------
    n : int
        Order of Bessel function
    nt : int
        Number of zeros to return

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           http://jin.ece.illinois.edu/specfunc.html

    """
    return jnyn_zeros(n, nt)[3]


def y0_zeros(nt, complex=False):
    """Compute nt zeros of Bessel function Y0(z), and derivative at each zero.

    The derivatives are given by Y0'(z0) = -Y1(z0) at each zero z0.

    Parameters
    ----------
    nt : int
        Number of zeros to return
    complex : bool, default False
        Set to False to return only the real zeros; set to True to return only
        the complex zeros with negative real part and positive imaginary part.
        Note that the complex conjugates of the latter are also zeros of the
        function, but are not returned by this routine.

    Returns
    -------
    z0n : ndarray
        Location of nth zero of Y0(z)
    y0pz0n : ndarray
        Value of derivative Y0'(z0) for nth zero

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("Arguments must be scalar positive integer.")
    kf = 0
    kc = not complex
    return specfun.cyzo(nt, kf, kc)


def y1_zeros(nt, complex=False):
    """Compute nt zeros of Bessel function Y1(z), and derivative at each zero.

    The derivatives are given by Y1'(z1) = Y0(z1) at each zero z1.

    Parameters
    ----------
    nt : int
        Number of zeros to return
    complex : bool, default False
        Set to False to return only the real zeros; set to True to return only
        the complex zeros with negative real part and positive imaginary part.
        Note that the complex conjugates of the latter are also zeros of the
        function, but are not returned by this routine.

    Returns
    -------
    z1n : ndarray
        Location of nth zero of Y1(z)
    y1pz1n : ndarray
        Value of derivative Y1'(z1) for nth zero

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("Arguments must be scalar positive integer.")
    kf = 1
    kc = not complex
    return specfun.cyzo(nt, kf, kc)


def y1p_zeros(nt, complex=False):
    """Compute nt zeros of Bessel derivative Y1'(z), and value at each zero.

    The values are given by Y1(z1) at each z1 where Y1'(z1)=0.

    Parameters
    ----------
    nt : int
        Number of zeros to return
    complex : bool, default False
        Set to False to return only the real zeros; set to True to return only
        the complex zeros with negative real part and positive imaginary part.
        Note that the complex conjugates of the latter are also zeros of the
        function, but are not returned by this routine.

    Returns
    -------
    z1pn : ndarray
        Location of nth zero of Y1'(z)
    y1z1pn : ndarray
        Value of derivative Y1(z1) for nth zero

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("Arguments must be scalar positive integer.")
    kf = 2
    kc = not complex
    return specfun.cyzo(nt, kf, kc)


def _bessel_diff_formula(v, z, n, L, phase):
    # from AMS55.
    # L(v, z) = J(v, z), Y(v, z), H1(v, z), H2(v, z), phase = -1
    # L(v, z) = I(v, z) or exp(v*pi*i)K(v, z), phase = 1
    # For K, you can pull out the exp((v-k)*pi*i) into the caller
    v = asarray(v)
    p = 1.0
    s = L(v-n, z)
    for i in xrange(1, n+1):
        p = phase * (p * (n-i+1)) / i   # = choose(k, i)
        s += p*L(v-n + i*2, z)
    return s / (2.**n)


bessel_diff_formula = np.deprecate(_bessel_diff_formula,
    message="bessel_diff_formula is a private function, do not use it!")


def jvp(v, z, n=1):
    """Compute nth derivative of Bessel function Jv(z) with respect to `z`.

    Parameters
    ----------
    v : float
        Order of Bessel function
    z : complex
        Argument at which to evaluate the derivative
    n : int, default 1
        Order of derivative

    Notes
    -----
    The derivative is computed using the relation DLFM 10.6.7 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           http://jin.ece.illinois.edu/specfunc.html
    .. [2] NIST Digital Library of Mathematical Functions.
           http://dlmf.nist.gov/10.6.E7

    """
    if not isinstance(n, int) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if n == 0:
        return jv(v, z)
    else:
        return _bessel_diff_formula(v, z, n, jv, -1)


def yvp(v, z, n=1):
    """Compute nth derivative of Bessel function Yv(z) with respect to `z`.

    Parameters
    ----------
    v : float
        Order of Bessel function
    z : complex
        Argument at which to evaluate the derivative
    n : int, default 1
        Order of derivative

    Notes
    -----
    The derivative is computed using the relation DLFM 10.6.7 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           http://jin.ece.illinois.edu/specfunc.html
    .. [2] NIST Digital Library of Mathematical Functions.
           http://dlmf.nist.gov/10.6.E7

    """
    if not isinstance(n, int) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if n == 0:
        return yv(v, z)
    else:
        return _bessel_diff_formula(v, z, n, yv, -1)


def kvp(v, z, n=1):
    """Compute nth derivative of real-order modified Bessel function Kv(z)

    Kv(z) is the modified Bessel function of the second kind.
    Derivative is calculated with respect to `z`.

    Parameters
    ----------
    v : array_like of float
        Order of Bessel function
    z : array_like of complex
        Argument at which to evaluate the derivative
    n : int
        Order of derivative.  Default is first derivative.

    Returns
    -------
    out : ndarray
        The results

    Examples
    --------
    Calculate multiple values at order 5:

    >>> from scipy.special import kvp
    >>> kvp(5, (1, 2, 3+5j))
    array([-1849.0354+0.j    ,   -25.7735+0.j    ,    -0.0307+0.0875j])

    Calculate for a single value at multiple orders:

    >>> kvp((4, 4.5, 5), 1)
    array([ -184.0309,  -568.9585, -1849.0354])

    Notes
    -----
    The derivative is computed using the relation DLFM 10.29.5 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 6.
           http://jin.ece.illinois.edu/specfunc.html
    .. [2] NIST Digital Library of Mathematical Functions.
           http://dlmf.nist.gov/10.29.E5

    """
    if not isinstance(n, int) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if n == 0:
        return kv(v, z)
    else:
        return (-1)**n * _bessel_diff_formula(v, z, n, kv, 1)


def ivp(v, z, n=1):
    """Compute nth derivative of modified Bessel function Iv(z) with respect
    to `z`.

    Parameters
    ----------
    v : array_like of float
        Order of Bessel function
    z : array_like of complex
        Argument at which to evaluate the derivative
    n : int, default 1
        Order of derivative

    Notes
    -----
    The derivative is computed using the relation DLFM 10.29.5 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 6.
           http://jin.ece.illinois.edu/specfunc.html
    .. [2] NIST Digital Library of Mathematical Functions.
           http://dlmf.nist.gov/10.29.E5

    """
    if not isinstance(n, int) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if n == 0:
        return iv(v, z)
    else:
        return _bessel_diff_formula(v, z, n, iv, 1)


def h1vp(v, z, n=1):
    """Compute nth derivative of Hankel function H1v(z) with respect to `z`.

    Parameters
    ----------
    v : float
        Order of Hankel function
    z : complex
        Argument at which to evaluate the derivative
    n : int, default 1
        Order of derivative

    Notes
    -----
    The derivative is computed using the relation DLFM 10.6.7 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           http://jin.ece.illinois.edu/specfunc.html
    .. [2] NIST Digital Library of Mathematical Functions.
           http://dlmf.nist.gov/10.6.E7

    """
    if not isinstance(n, int) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if n == 0:
        return hankel1(v, z)
    else:
        return _bessel_diff_formula(v, z, n, hankel1, -1)


def h2vp(v, z, n=1):
    """Compute nth derivative of Hankel function H2v(z) with respect to `z`.

    Parameters
    ----------
    v : float
        Order of Hankel function
    z : complex
        Argument at which to evaluate the derivative
    n : int, default 1
        Order of derivative

    Notes
    -----
    The derivative is computed using the relation DLFM 10.6.7 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           http://jin.ece.illinois.edu/specfunc.html
    .. [2] NIST Digital Library of Mathematical Functions.
           http://dlmf.nist.gov/10.6.E7

    """
    if not isinstance(n, int) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if n == 0:
        return hankel2(v, z)
    else:
        return _bessel_diff_formula(v, z, n, hankel2, -1)


@np.deprecate(message="scipy.special.sph_jn is deprecated in scipy 0.18.0. "
                      "Use scipy.special.spherical_jn instead. "
                      "Note that the new function has a different signature.")
def sph_jn(n, z):
    """Compute spherical Bessel function jn(z) and derivative.

    This function computes the value and first derivative of jn(z) for all
    orders up to and including n.

    Parameters
    ----------
    n : int
        Maximum order of jn to compute
    z : complex
        Argument at which to evaluate

    Returns
    -------
    jn : ndarray
        Value of j0(z), ..., jn(z)
    jnp : ndarray
        First derivative j0'(z), ..., jn'(z)

    See also
    --------
    spherical_jn

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 8.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not (isscalar(n) and isscalar(z)):
        raise ValueError("arguments must be scalars.")
    if (n != floor(n)) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    if iscomplex(z):
        nm, jn, jnp, yn, ynp = specfun.csphjy(n1, z)
    else:
        nm, jn, jnp = specfun.sphj(n1, z)
    return jn[:(n+1)], jnp[:(n+1)]


@np.deprecate(message="scipy.special.sph_yn is deprecated in scipy 0.18.0. "
                      "Use scipy.special.spherical_yn instead. "
                      "Note that the new function has a different signature.")
def sph_yn(n, z):
    """Compute spherical Bessel function yn(z) and derivative.

    This function computes the value and first derivative of yn(z) for all
    orders up to and including n.

    Parameters
    ----------
    n : int
        Maximum order of yn to compute
    z : complex
        Argument at which to evaluate

    Returns
    -------
    yn : ndarray
        Value of y0(z), ..., yn(z)
    ynp : ndarray
        First derivative y0'(z), ..., yn'(z)

    See also
    --------
    spherical_yn

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 8.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not (isscalar(n) and isscalar(z)):
        raise ValueError("arguments must be scalars.")
    if (n != floor(n)) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    if iscomplex(z) or less(z, 0):
        nm, jn, jnp, yn, ynp = specfun.csphjy(n1, z)
    else:
        nm, yn, ynp = specfun.sphy(n1, z)
    return yn[:(n+1)], ynp[:(n+1)]


@np.deprecate(message="scipy.special.sph_jnyn is deprecated in scipy 0.18.0. "
                      "Use scipy.special.spherical_jn and "
                      "scipy.special.spherical_yn instead. "
                      "Note that the new function has a different signature.")
def sph_jnyn(n, z):
    """Compute spherical Bessel functions jn(z) and yn(z) and derivatives.

    This function computes the value and first derivative of jn(z) and yn(z)
    for all orders up to and including n.

    Parameters
    ----------
    n : int
        Maximum order of jn and yn to compute
    z : complex
        Argument at which to evaluate

    Returns
    -------
    jn : ndarray
        Value of j0(z), ..., jn(z)
    jnp : ndarray
        First derivative j0'(z), ..., jn'(z)
    yn : ndarray
        Value of y0(z), ..., yn(z)
    ynp : ndarray
        First derivative y0'(z), ..., yn'(z)

    See also
    --------
    spherical_jn
    spherical_yn

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 8.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not (isscalar(n) and isscalar(z)):
        raise ValueError("arguments must be scalars.")
    if (n != floor(n)) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    if iscomplex(z) or less(z, 0):
        nm, jn, jnp, yn, ynp = specfun.csphjy(n1, z)
    else:
        nm, yn, ynp = specfun.sphy(n1, z)
        nm, jn, jnp = specfun.sphj(n1, z)
    return jn[:(n+1)], jnp[:(n+1)], yn[:(n+1)], ynp[:(n+1)]


@np.deprecate(message="scipy.special.sph_in is deprecated in scipy 0.18.0. "
                      "Use scipy.special.spherical_in instead. "
                      "Note that the new function has a different signature.")
def sph_in(n, z):
    """Compute spherical Bessel function in(z) and derivative.

    This function computes the value and first derivative of in(z) for all
    orders up to and including n.

    Parameters
    ----------
    n : int
        Maximum order of in to compute
    z : complex
        Argument at which to evaluate

    Returns
    -------
    in : ndarray
        Value of i0(z), ..., in(z)
    inp : ndarray
        First derivative i0'(z), ..., in'(z)

    See also
    --------
    spherical_in

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 8.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not (isscalar(n) and isscalar(z)):
        raise ValueError("arguments must be scalars.")
    if (n != floor(n)) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    if iscomplex(z):
        nm, In, Inp, kn, knp = specfun.csphik(n1, z)
    else:
        nm, In, Inp = specfun.sphi(n1, z)
    return In[:(n+1)], Inp[:(n+1)]


@np.deprecate(message="scipy.special.sph_kn is deprecated in scipy 0.18.0. "
                      "Use scipy.special.spherical_kn instead. "
                      "Note that the new function has a different signature.")
def sph_kn(n, z):
    """Compute spherical Bessel function kn(z) and derivative.

    This function computes the value and first derivative of kn(z) for all
    orders up to and including n.

    Parameters
    ----------
    n : int
        Maximum order of kn to compute
    z : complex
        Argument at which to evaluate

    Returns
    -------
    kn : ndarray
        Value of k0(z), ..., kn(z)
    knp : ndarray
        First derivative k0'(z), ..., kn'(z)

    See also
    --------
    spherical_kn

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 8.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not (isscalar(n) and isscalar(z)):
        raise ValueError("arguments must be scalars.")
    if (n != floor(n)) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    if iscomplex(z) or less(z, 0):
        nm, In, Inp, kn, knp = specfun.csphik(n1, z)
    else:
        nm, kn, knp = specfun.sphk(n1, z)
    return kn[:(n+1)], knp[:(n+1)]


@np.deprecate(message="scipy.special.sph_inkn is deprecated in scipy 0.18.0. "
                      "Use scipy.special.spherical_in and "
                      "scipy.special.spherical_kn instead. "
                      "Note that the new function has a different signature.")
def sph_inkn(n, z):
    """Compute spherical Bessel functions in(z), kn(z), and derivatives.

    This function computes the value and first derivative of in(z) and kn(z)
    for all orders up to and including n.

    Parameters
    ----------
    n : int
        Maximum order of in and kn to compute
    z : complex
        Argument at which to evaluate

    Returns
    -------
    in : ndarray
        Value of i0(z), ..., in(z)
    inp : ndarray
        First derivative i0'(z), ..., in'(z)
    kn : ndarray
        Value of k0(z), ..., kn(z)
    knp : ndarray
        First derivative k0'(z), ..., kn'(z)

    See also
    --------
    spherical_in
    spherical_kn

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 8.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not (isscalar(n) and isscalar(z)):
        raise ValueError("arguments must be scalars.")
    if (n != floor(n)) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    if iscomplex(z) or less(z, 0):
        nm, In, Inp, kn, knp = specfun.csphik(n1, z)
    else:
        nm, In, Inp = specfun.sphi(n1, z)
        nm, kn, knp = specfun.sphk(n1, z)
    return In[:(n+1)], Inp[:(n+1)], kn[:(n+1)], knp[:(n+1)]


def riccati_jn(n, x):
    r"""Compute Ricatti-Bessel function of the first kind and its derivative.

    The Ricatti-Bessel function of the first kind is defined as :math:`x
    j_n(x)`, where :math:`j_n` is the spherical Bessel function of the first
    kind of order :math:`n`.

    This function computes the value and first derivative of the
    Ricatti-Bessel function for all orders up to and including `n`.

    Parameters
    ----------
    n : int
        Maximum order of function to compute
    x : float
        Argument at which to evaluate

    Returns
    -------
    jn : ndarray
        Value of j0(x), ..., jn(x)
    jnp : ndarray
        First derivative j0'(x), ..., jn'(x)

    Notes
    -----
    The computation is carried out via backward recurrence, using the
    relation DLMF 10.51.1 [2]_.

    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
    Jin [1]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html
    .. [2] NIST Digital Library of Mathematical Functions.
           http://dlmf.nist.gov/10.51.E1

    """
    if not (isscalar(n) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    if (n != floor(n)) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if (n == 0):
        n1 = 1
    else:
        n1 = n
    nm, jn, jnp = specfun.rctj(n1, x)
    return jn[:(n+1)], jnp[:(n+1)]


def riccati_yn(n, x):
    """Compute Ricatti-Bessel function of the second kind and its derivative.

    The Ricatti-Bessel function of the second kind is defined as :math:`x
    y_n(x)`, where :math:`y_n` is the spherical Bessel function of the second
    kind of order :math:`n`.

    This function computes the value and first derivative of the function for
    all orders up to and including `n`.

    Parameters
    ----------
    n : int
        Maximum order of function to compute
    x : float
        Argument at which to evaluate

    Returns
    -------
    yn : ndarray
        Value of y0(x), ..., yn(x)
    ynp : ndarray
        First derivative y0'(x), ..., yn'(x)

    Notes
    -----
    The computation is carried out via ascending recurrence, using the
    relation DLMF 10.51.1 [2]_.

    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
    Jin [1]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html
    .. [2] NIST Digital Library of Mathematical Functions.
           http://dlmf.nist.gov/10.51.E1

    """
    if not (isscalar(n) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    if (n != floor(n)) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if (n == 0):
        n1 = 1
    else:
        n1 = n
    nm, jn, jnp = specfun.rcty(n1, x)
    return jn[:(n+1)], jnp[:(n+1)]


def erfinv(y):
    """Inverse function for erf.
    """
    return ndtri((y+1)/2.0)/sqrt(2)


def erfcinv(y):
    """Inverse function for erfc.
    """
    return -ndtri(0.5*y)/sqrt(2)


def erf_zeros(nt):
    """Compute nt complex zeros of error function erf(z).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if (floor(nt) != nt) or (nt <= 0) or not isscalar(nt):
        raise ValueError("Argument must be positive scalar integer.")
    return specfun.cerzo(nt)


def fresnelc_zeros(nt):
    """Compute nt complex zeros of cosine Fresnel integral C(z).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if (floor(nt) != nt) or (nt <= 0) or not isscalar(nt):
        raise ValueError("Argument must be positive scalar integer.")
    return specfun.fcszo(1, nt)


def fresnels_zeros(nt):
    """Compute nt complex zeros of sine Fresnel integral S(z).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if (floor(nt) != nt) or (nt <= 0) or not isscalar(nt):
        raise ValueError("Argument must be positive scalar integer.")
    return specfun.fcszo(2, nt)


def fresnel_zeros(nt):
    """Compute nt complex zeros of sine and cosine Fresnel integrals S(z) and C(z).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if (floor(nt) != nt) or (nt <= 0) or not isscalar(nt):
        raise ValueError("Argument must be positive scalar integer.")
    return specfun.fcszo(2, nt), specfun.fcszo(1, nt)


def assoc_laguerre(x, n, k=0.0):
    """Compute the generalized (associated) Laguerre polynomial of degree n and order k.

    The polynomial :math:`L^{(k)}_n(x)` is orthogonal over ``[0, inf)``,
    with weighting function ``exp(-x) * x**k`` with ``k > -1``.

    Notes
    -----
    `assoc_laguerre` is a simple wrapper around `eval_genlaguerre`, with
    reversed argument order ``(x, n, k=0.0) --> (n, k, x)``.

    """
    return orthogonal.eval_genlaguerre(n, k, x)

digamma = psi


def polygamma(n, x):
    """Polygamma function n.

    This is the nth derivative of the digamma (psi) function.

    Parameters
    ----------
    n : array_like of int
        The order of the derivative of `psi`.
    x : array_like
        Where to evaluate the polygamma function.

    Returns
    -------
    polygamma : ndarray
        The result.

    Examples
    --------
    >>> from scipy import special
    >>> x = [2, 3, 25.5]
    >>> special.polygamma(1, x)
    array([ 0.64493407,  0.39493407,  0.03999467])
    >>> special.polygamma(0, x) == special.psi(x)
    array([ True,  True,  True], dtype=bool)

    """
    n, x = asarray(n), asarray(x)
    fac2 = (-1.0)**(n+1) * gamma(n+1.0) * zeta(n+1, x)
    return where(n == 0, psi(x), fac2)


def mathieu_even_coef(m, q):
    r"""Fourier coefficients for even Mathieu and modified Mathieu functions.

    The Fourier series of the even solutions of the Mathieu differential
    equation are of the form

    .. math:: \mathrm{ce}_{2n}(z, q) = \sum_{k=0}^{\infty} A_{(2n)}^{(2k)} \cos 2kz

    .. math:: \mathrm{ce}_{2n+1}(z, q) = \sum_{k=0}^{\infty} A_{(2n+1)}^{(2k+1)} \cos (2k+1)z

    This function returns the coefficients :math:`A_{(2n)}^{(2k)}` for even
    input m=2n, and the coefficients :math:`A_{(2n+1)}^{(2k+1)}` for odd input
    m=2n+1.

    Parameters
    ----------
    m : int
        Order of Mathieu functions.  Must be non-negative.
    q : float (>=0)
        Parameter of Mathieu functions.  Must be non-negative.

    Returns
    -------
    Ak : ndarray
        Even or odd Fourier coefficients, corresponding to even or odd m.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html
    .. [2] NIST Digital Library of Mathematical Functions
           http://dlmf.nist.gov/28.4#i

    """
    if not (isscalar(m) and isscalar(q)):
        raise ValueError("m and q must be scalars.")
    if (q < 0):
        raise ValueError("q >=0")
    if (m != floor(m)) or (m < 0):
        raise ValueError("m must be an integer >=0.")

    if (q <= 1):
        qm = 7.5 + 56.1*sqrt(q) - 134.7*q + 90.7*sqrt(q)*q
    else:
        qm = 17.0 + 3.1*sqrt(q) - .126*q + .0037*sqrt(q)*q
    km = int(qm + 0.5*m)
    if km > 251:
        print("Warning, too many predicted coefficients.")
    kd = 1
    m = int(floor(m))
    if m % 2:
        kd = 2

    a = mathieu_a(m, q)
    fc = specfun.fcoef(kd, m, q, a)
    return fc[:km]


def mathieu_odd_coef(m, q):
    r"""Fourier coefficients for even Mathieu and modified Mathieu functions.

    The Fourier series of the odd solutions of the Mathieu differential
    equation are of the form

    .. math:: \mathrm{se}_{2n+1}(z, q) = \sum_{k=0}^{\infty} B_{(2n+1)}^{(2k+1)} \sin (2k+1)z

    .. math:: \mathrm{se}_{2n+2}(z, q) = \sum_{k=0}^{\infty} B_{(2n+2)}^{(2k+2)} \sin (2k+2)z

    This function returns the coefficients :math:`B_{(2n+2)}^{(2k+2)}` for even
    input m=2n+2, and the coefficients :math:`B_{(2n+1)}^{(2k+1)}` for odd
    input m=2n+1.

    Parameters
    ----------
    m : int
        Order of Mathieu functions.  Must be non-negative.
    q : float (>=0)
        Parameter of Mathieu functions.  Must be non-negative.

    Returns
    -------
    Bk : ndarray
        Even or odd Fourier coefficients, corresponding to even or odd m.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not (isscalar(m) and isscalar(q)):
        raise ValueError("m and q must be scalars.")
    if (q < 0):
        raise ValueError("q >=0")
    if (m != floor(m)) or (m <= 0):
        raise ValueError("m must be an integer > 0")

    if (q <= 1):
        qm = 7.5 + 56.1*sqrt(q) - 134.7*q + 90.7*sqrt(q)*q
    else:
        qm = 17.0 + 3.1*sqrt(q) - .126*q + .0037*sqrt(q)*q
    km = int(qm + 0.5*m)
    if km > 251:
        print("Warning, too many predicted coefficients.")
    kd = 4
    m = int(floor(m))
    if m % 2:
        kd = 3

    b = mathieu_b(m, q)
    fc = specfun.fcoef(kd, m, q, b)
    return fc[:km]


def lpmn(m, n, z):
    """Sequence of associated Legendre functions of the first kind.

    Computes the associated Legendre function of the first kind of order m and
    degree n, ``Pmn(z)`` = :math:`P_n^m(z)`, and its derivative, ``Pmn'(z)``.
    Returns two arrays of size ``(m+1, n+1)`` containing ``Pmn(z)`` and
    ``Pmn'(z)`` for all orders from ``0..m`` and degrees from ``0..n``.

    This function takes a real argument ``z``. For complex arguments ``z``
    use clpmn instead.

    Parameters
    ----------
    m : int
       ``|m| <= n``; the order of the Legendre function.
    n : int
       where ``n >= 0``; the degree of the Legendre function.  Often
       called ``l`` (lower case L) in descriptions of the associated
       Legendre function
    z : float
        Input value.

    Returns
    -------
    Pmn_z : (m+1, n+1) array
       Values for all orders 0..m and degrees 0..n
    Pmn_d_z : (m+1, n+1) array
       Derivatives for all orders 0..m and degrees 0..n

    See Also
    --------
    clpmn: associated Legendre functions of the first kind for complex z

    Notes
    -----
    In the interval (-1, 1), Ferrer's function of the first kind is
    returned. The phase convention used for the intervals (1, inf)
    and (-inf, -1) is such that the result is always real.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html
    .. [2] NIST Digital Library of Mathematical Functions
           http://dlmf.nist.gov/14.3

    """
    if not isscalar(m) or (abs(m) > n):
        raise ValueError("m must be <= n.")
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if not isscalar(z):
        raise ValueError("z must be scalar.")
    if iscomplex(z):
        raise ValueError("Argument must be real. Use clpmn instead.")
    if (m < 0):
        mp = -m
        mf, nf = mgrid[0:mp+1, 0:n+1]
        with ufuncs.errstate(all='ignore'):
            if abs(z) < 1:
                # Ferrer function; DLMF 14.9.3
                fixarr = where(mf > nf, 0.0,
                               (-1)**mf * gamma(nf-mf+1) / gamma(nf+mf+1))
            else:
                # Match to clpmn; DLMF 14.9.13
                fixarr = where(mf > nf, 0.0, gamma(nf-mf+1) / gamma(nf+mf+1))
    else:
        mp = m
    p, pd = specfun.lpmn(mp, n, z)
    if (m < 0):
        p = p * fixarr
        pd = pd * fixarr
    return p, pd


def clpmn(m, n, z, type=3):
    """Associated Legendre function of the first kind for complex arguments.

    Computes the associated Legendre function of the first kind of order m and
    degree n, ``Pmn(z)`` = :math:`P_n^m(z)`, and its derivative, ``Pmn'(z)``.
    Returns two arrays of size ``(m+1, n+1)`` containing ``Pmn(z)`` and
    ``Pmn'(z)`` for all orders from ``0..m`` and degrees from ``0..n``.

    Parameters
    ----------
    m : int
       ``|m| <= n``; the order of the Legendre function.
    n : int
       where ``n >= 0``; the degree of the Legendre function.  Often
       called ``l`` (lower case L) in descriptions of the associated
       Legendre function
    z : float or complex
        Input value.
    type : int, optional
       takes values 2 or 3
       2: cut on the real axis ``|x| > 1``
       3: cut on the real axis ``-1 < x < 1`` (default)

    Returns
    -------
    Pmn_z : (m+1, n+1) array
       Values for all orders ``0..m`` and degrees ``0..n``
    Pmn_d_z : (m+1, n+1) array
       Derivatives for all orders ``0..m`` and degrees ``0..n``

    See Also
    --------
    lpmn: associated Legendre functions of the first kind for real z

    Notes
    -----
    By default, i.e. for ``type=3``, phase conventions are chosen according
    to [1]_ such that the function is analytic. The cut lies on the interval
    (-1, 1). Approaching the cut from above or below in general yields a phase
    factor with respect to Ferrer's function of the first kind
    (cf. `lpmn`).

    For ``type=2`` a cut at ``|x| > 1`` is chosen. Approaching the real values
    on the interval (-1, 1) in the complex plane yields Ferrer's function
    of the first kind.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html
    .. [2] NIST Digital Library of Mathematical Functions
           http://dlmf.nist.gov/14.21

    """
    if not isscalar(m) or (abs(m) > n):
        raise ValueError("m must be <= n.")
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if not isscalar(z):
        raise ValueError("z must be scalar.")
    if not(type == 2 or type == 3):
        raise ValueError("type must be either 2 or 3.")
    if (m < 0):
        mp = -m
        mf, nf = mgrid[0:mp+1, 0:n+1]
        with ufuncs.errstate(all='ignore'):
            if type == 2:
                fixarr = where(mf > nf, 0.0,
                               (-1)**mf * gamma(nf-mf+1) / gamma(nf+mf+1))
            else:
                fixarr = where(mf > nf, 0.0, gamma(nf-mf+1) / gamma(nf+mf+1))
    else:
        mp = m
    p, pd = specfun.clpmn(mp, n, real(z), imag(z), type)
    if (m < 0):
        p = p * fixarr
        pd = pd * fixarr
    return p, pd


def lqmn(m, n, z):
    """Sequence of associated Legendre functions of the second kind.

    Computes the associated Legendre function of the second kind of order m and
    degree n, ``Qmn(z)`` = :math:`Q_n^m(z)`, and its derivative, ``Qmn'(z)``.
    Returns two arrays of size ``(m+1, n+1)`` containing ``Qmn(z)`` and
    ``Qmn'(z)`` for all orders from ``0..m`` and degrees from ``0..n``.

    Parameters
    ----------
    m : int
       ``|m| <= n``; the order of the Legendre function.
    n : int
       where ``n >= 0``; the degree of the Legendre function.  Often
       called ``l`` (lower case L) in descriptions of the associated
       Legendre function
    z : complex
        Input value.

    Returns
    -------
    Qmn_z : (m+1, n+1) array
       Values for all orders 0..m and degrees 0..n
    Qmn_d_z : (m+1, n+1) array
       Derivatives for all orders 0..m and degrees 0..n

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not isscalar(m) or (m < 0):
        raise ValueError("m must be a non-negative integer.")
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if not isscalar(z):
        raise ValueError("z must be scalar.")
    m = int(m)
    n = int(n)

    # Ensure neither m nor n == 0
    mm = max(1, m)
    nn = max(1, n)

    if iscomplex(z):
        q, qd = specfun.clqmn(mm, nn, z)
    else:
        q, qd = specfun.lqmn(mm, nn, z)
    return q[:(m+1), :(n+1)], qd[:(m+1), :(n+1)]


def bernoulli(n):
    """Bernoulli numbers B0..Bn (inclusive).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    n = int(n)
    if (n < 2):
        n1 = 2
    else:
        n1 = n
    return specfun.bernob(int(n1))[:(n+1)]


def euler(n):
    """Euler numbers E0..En (inclusive).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    n = int(n)
    if (n < 2):
        n1 = 2
    else:
        n1 = n
    return specfun.eulerb(n1)[:(n+1)]


def lpn(n, z):
    """Legendre function of the first kind.

    Compute sequence of Legendre functions of the first kind (polynomials),
    Pn(z) and derivatives for all degrees from 0 to n (inclusive).

    See also special.legendre for polynomial class.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not (isscalar(n) and isscalar(z)):
        raise ValueError("arguments must be scalars.")
    if (n != floor(n)) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    if iscomplex(z):
        pn, pd = specfun.clpn(n1, z)
    else:
        pn, pd = specfun.lpn(n1, z)
    return pn[:(n+1)], pd[:(n+1)]


def lqn(n, z):
    """Legendre function of the second kind.

    Compute sequence of Legendre functions of the second kind, Qn(z) and
    derivatives for all degrees from 0 to n (inclusive).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not (isscalar(n) and isscalar(z)):
        raise ValueError("arguments must be scalars.")
    if (n != floor(n)) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    if iscomplex(z):
        qn, qd = specfun.clqn(n1, z)
    else:
        qn, qd = specfun.lqnb(n1, z)
    return qn[:(n+1)], qd[:(n+1)]


def ai_zeros(nt):
    """
    Compute `nt` zeros and values of the Airy function Ai and its derivative.

    Computes the first `nt` zeros, `a`, of the Airy function Ai(x);
    first `nt` zeros, `ap`, of the derivative of the Airy function Ai'(x);
    the corresponding values Ai(a');
    and the corresponding values Ai'(a).

    Parameters
    ----------
    nt : int
        Number of zeros to compute

    Returns
    -------
    a : ndarray
        First `nt` zeros of Ai(x)
    ap : ndarray
        First `nt` zeros of Ai'(x)
    ai : ndarray
        Values of Ai(x) evaluated at first `nt` zeros of Ai'(x)
    aip : ndarray
        Values of Ai'(x) evaluated at first `nt` zeros of Ai(x)

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    kf = 1
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be a positive integer scalar.")
    return specfun.airyzo(nt, kf)


def bi_zeros(nt):
    """
    Compute `nt` zeros and values of the Airy function Bi and its derivative.

    Computes the first `nt` zeros, b, of the Airy function Bi(x);
    first `nt` zeros, b', of the derivative of the Airy function Bi'(x);
    the corresponding values Bi(b');
    and the corresponding values Bi'(b).

    Parameters
    ----------
    nt : int
        Number of zeros to compute

    Returns
    -------
    b : ndarray
        First `nt` zeros of Bi(x)
    bp : ndarray
        First `nt` zeros of Bi'(x)
    bi : ndarray
        Values of Bi(x) evaluated at first `nt` zeros of Bi'(x)
    bip : ndarray
        Values of Bi'(x) evaluated at first `nt` zeros of Bi(x)

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    kf = 2
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be a positive integer scalar.")
    return specfun.airyzo(nt, kf)


def lmbda(v, x):
    r"""Jahnke-Emden Lambda function, Lambdav(x).

    This function is defined as [2]_,

    .. math:: \Lambda_v(x) = \Gamma(v+1) \frac{J_v(x)}{(x/2)^v},

    where :math:`\Gamma` is the gamma function and :math:`J_v` is the
    Bessel function of the first kind.

    Parameters
    ----------
    v : float
        Order of the Lambda function
    x : float
        Value at which to evaluate the function and derivatives

    Returns
    -------
    vl : ndarray
        Values of Lambda_vi(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.
    dl : ndarray
        Derivatives Lambda_vi'(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html
    .. [2] Jahnke, E. and Emde, F. "Tables of Functions with Formulae and
           Curves" (4th ed.), Dover, 1945
    """
    if not (isscalar(v) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    if (v < 0):
        raise ValueError("argument must be > 0.")
    n = int(v)
    v0 = v - n
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    v1 = n1 + v0
    if (v != floor(v)):
        vm, vl, dl = specfun.lamv(v1, x)
    else:
        vm, vl, dl = specfun.lamn(v1, x)
    return vl[:(n+1)], dl[:(n+1)]


def pbdv_seq(v, x):
    """Parabolic cylinder functions Dv(x) and derivatives.

    Parameters
    ----------
    v : float
        Order of the parabolic cylinder function
    x : float
        Value at which to evaluate the function and derivatives

    Returns
    -------
    dv : ndarray
        Values of D_vi(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.
    dp : ndarray
        Derivatives D_vi'(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 13.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not (isscalar(v) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    n = int(v)
    v0 = v-n
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    v1 = n1 + v0
    dv, dp, pdf, pdd = specfun.pbdv(v1, x)
    return dv[:n1+1], dp[:n1+1]


def pbvv_seq(v, x):
    """Parabolic cylinder functions Vv(x) and derivatives.

    Parameters
    ----------
    v : float
        Order of the parabolic cylinder function
    x : float
        Value at which to evaluate the function and derivatives

    Returns
    -------
    dv : ndarray
        Values of V_vi(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.
    dp : ndarray
        Derivatives V_vi'(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 13.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not (isscalar(v) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    n = int(v)
    v0 = v-n
    if (n <= 1):
        n1 = 1
    else:
        n1 = n
    v1 = n1 + v0
    dv, dp, pdf, pdd = specfun.pbvv(v1, x)
    return dv[:n1+1], dp[:n1+1]


def pbdn_seq(n, z):
    """Parabolic cylinder functions Dn(z) and derivatives.

    Parameters
    ----------
    n : int
        Order of the parabolic cylinder function
    z : complex
        Value at which to evaluate the function and derivatives

    Returns
    -------
    dv : ndarray
        Values of D_i(z), for i=0, ..., i=n.
    dp : ndarray
        Derivatives D_i'(z), for i=0, ..., i=n.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 13.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not (isscalar(n) and isscalar(z)):
        raise ValueError("arguments must be scalars.")
    if (floor(n) != n):
        raise ValueError("n must be an integer.")
    if (abs(n) <= 1):
        n1 = 1
    else:
        n1 = n
    cpb, cpd = specfun.cpbdn(n1, z)
    return cpb[:n1+1], cpd[:n1+1]


def ber_zeros(nt):
    """Compute nt zeros of the Kelvin function ber(x).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return specfun.klvnzo(nt, 1)


def bei_zeros(nt):
    """Compute nt zeros of the Kelvin function bei(x).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return specfun.klvnzo(nt, 2)


def ker_zeros(nt):
    """Compute nt zeros of the Kelvin function ker(x).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return specfun.klvnzo(nt, 3)


def kei_zeros(nt):
    """Compute nt zeros of the Kelvin function kei(x).
    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return specfun.klvnzo(nt, 4)


def berp_zeros(nt):
    """Compute nt zeros of the Kelvin function ber'(x).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return specfun.klvnzo(nt, 5)


def beip_zeros(nt):
    """Compute nt zeros of the Kelvin function bei'(x).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return specfun.klvnzo(nt, 6)


def kerp_zeros(nt):
    """Compute nt zeros of the Kelvin function ker'(x).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return specfun.klvnzo(nt, 7)


def keip_zeros(nt):
    """Compute nt zeros of the Kelvin function kei'(x).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return specfun.klvnzo(nt, 8)


def kelvin_zeros(nt):
    """Compute nt zeros of all Kelvin functions.

    Returned in a length-8 tuple of arrays of length nt.  The tuple contains
    the arrays of zeros of (ber, bei, ker, kei, ber', bei', ker', kei').

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return (specfun.klvnzo(nt, 1),
            specfun.klvnzo(nt, 2),
            specfun.klvnzo(nt, 3),
            specfun.klvnzo(nt, 4),
            specfun.klvnzo(nt, 5),
            specfun.klvnzo(nt, 6),
            specfun.klvnzo(nt, 7),
            specfun.klvnzo(nt, 8))


def pro_cv_seq(m, n, c):
    """Characteristic values for prolate spheroidal wave functions.

    Compute a sequence of characteristic values for the prolate
    spheroidal wave functions for mode m and n'=m..n and spheroidal
    parameter c.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not (isscalar(m) and isscalar(n) and isscalar(c)):
        raise ValueError("Arguments must be scalars.")
    if (n != floor(n)) or (m != floor(m)):
        raise ValueError("Modes must be integers.")
    if (n-m > 199):
        raise ValueError("Difference between n and m is too large.")
    maxL = n-m+1
    return specfun.segv(m, n, c, 1)[1][:maxL]


def obl_cv_seq(m, n, c):
    """Characteristic values for oblate spheroidal wave functions.

    Compute a sequence of characteristic values for the oblate
    spheroidal wave functions for mode m and n'=m..n and spheroidal
    parameter c.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           http://jin.ece.illinois.edu/specfunc.html

    """
    if not (isscalar(m) and isscalar(n) and isscalar(c)):
        raise ValueError("Arguments must be scalars.")
    if (n != floor(n)) or (m != floor(m)):
        raise ValueError("Modes must be integers.")
    if (n-m > 199):
        raise ValueError("Difference between n and m is too large.")
    maxL = n-m+1
    return specfun.segv(m, n, c, -1)[1][:maxL]


def ellipk(m):
    """Complete elliptic integral of the first kind.

    This function is defined as

    .. math:: K(m) = \\int_0^{\\pi/2} [1 - m \\sin(t)^2]^{-1/2} dt

    Parameters
    ----------
    m : array_like
        The parameter of the elliptic integral.

    Returns
    -------
    K : array_like
        Value of the elliptic integral.

    Notes
    -----
    For more precision around point m = 1, use `ellipkm1`, which this
    function calls.

    See Also
    --------
    ellipkm1 : Complete elliptic integral of the first kind around m = 1
    ellipkinc : Incomplete elliptic integral of the first kind
    ellipe : Complete elliptic integral of the second kind
    ellipeinc : Incomplete elliptic integral of the second kind


    """
    return ellipkm1(1 - asarray(m))


def agm(a, b):
    """Arithmetic, Geometric Mean.

    Start with a_0=a and b_0=b and iteratively compute

    a_{n+1} = (a_n+b_n)/2
    b_{n+1} = sqrt(a_n*b_n)

    until a_n=b_n.   The result is agm(a, b)

    agm(a, b)=agm(b, a)
    agm(a, a) = a
    min(a, b) < agm(a, b) < max(a, b)
    """
    s = a + b + 0.0
    return (pi / 4) * s / ellipkm1(4 * a * b / s ** 2)


def comb(N, k, exact=False, repetition=False):
    """The number of combinations of N things taken k at a time.

    This is often expressed as "N choose k".

    Parameters
    ----------
    N : int, ndarray
        Number of things.
    k : int, ndarray
        Number of elements taken.
    exact : bool, optional
        If `exact` is False, then floating point precision is used, otherwise
        exact long integer is computed.
    repetition : bool, optional
        If `repetition` is True, then the number of combinations with
        repetition is computed.

    Returns
    -------
    val : int, ndarray
        The total number of combinations.

    See Also
    --------
    binom : Binomial coefficient ufunc

    Notes
    -----
    - Array arguments accepted only for exact=False case.
    - If k > N, N < 0, or k < 0, then a 0 is returned.

    Examples
    --------
    >>> from scipy.special import comb
    >>> k = np.array([3, 4])
    >>> n = np.array([10, 10])
    >>> comb(n, k, exact=False)
    array([ 120.,  210.])
    >>> comb(10, 3, exact=True)
    120L
    >>> comb(10, 3, exact=True, repetition=True)
    220L

    """
    if repetition:
        return comb(N + k - 1, k, exact)
    if exact:
        return _comb_int(N, k)
    else:
        k, N = asarray(k), asarray(N)
        cond = (k <= N) & (N >= 0) & (k >= 0)
        vals = binom(N, k)
        if isinstance(vals, np.ndarray):
            vals[~cond] = 0
        elif not cond:
            vals = np.float64(0)
        return vals


def perm(N, k, exact=False):
    """Permutations of N things taken k at a time, i.e., k-permutations of N.

    It's also known as "partial permutations".

    Parameters
    ----------
    N : int, ndarray
        Number of things.
    k : int, ndarray
        Number of elements taken.
    exact : bool, optional
        If `exact` is False, then floating point precision is used, otherwise
        exact long integer is computed.

    Returns
    -------
    val : int, ndarray
        The number of k-permutations of N.

    Notes
    -----
    - Array arguments accepted only for exact=False case.
    - If k > N, N < 0, or k < 0, then a 0 is returned.

    Examples
    --------
    >>> from scipy.special import perm
    >>> k = np.array([3, 4])
    >>> n = np.array([10, 10])
    >>> perm(n, k)
    array([  720.,  5040.])
    >>> perm(10, 3, exact=True)
    720

    """
    if exact:
        if (k > N) or (N < 0) or (k < 0):
            return 0
        val = 1
        for i in xrange(N - k + 1, N + 1):
            val *= i
        return val
    else:
        k, N = asarray(k), asarray(N)
        cond = (k <= N) & (N >= 0) & (k >= 0)
        vals = poch(N - k + 1, k)
        if isinstance(vals, np.ndarray):
            vals[~cond] = 0
        elif not cond:
            vals = np.float64(0)
        return vals


# http://stackoverflow.com/a/16327037/125507
def _range_prod(lo, hi):
    """
    Product of a range of numbers.

    Returns the product of
    lo * (lo+1) * (lo+2) * ... * (hi-2) * (hi-1) * hi
    = hi! / (lo-1)!

    Breaks into smaller products first for speed:
    _range_prod(2, 9) = ((2*3)*(4*5))*((6*7)*(8*9))
    """
    if lo + 1 < hi:
        mid = (hi + lo) // 2
        return _range_prod(lo, mid) * _range_prod(mid + 1, hi)
    if lo == hi:
        return lo
    return lo * hi


def factorial(n, exact=False):
    """
    The factorial of a number or array of numbers.

    The factorial of non-negative integer `n` is the product of all
    positive integers less than or equal to `n`::

        n! = n * (n - 1) * (n - 2) * ... * 1

    Parameters
    ----------
    n : int or array_like of ints
        Input values.  If ``n < 0``, the return value is 0.
    exact : bool, optional
        If True, calculate the answer exactly using long integer arithmetic.
        If False, result is approximated in floating point rapidly using the
        `gamma` function.
        Default is False.

    Returns
    -------
    nf : float or int or ndarray
        Factorial of `n`, as integer or float depending on `exact`.

    Notes
    -----
    For arrays with ``exact=True``, the factorial is computed only once, for
    the largest input, with each other result computed in the process.
    The output dtype is increased to ``int64`` or ``object`` if necessary.

    With ``exact=False`` the factorial is approximated using the gamma
    function:

    .. math:: n! = \\Gamma(n+1)

    Examples
    --------
    >>> from scipy.special import factorial
    >>> arr = np.array([3, 4, 5])
    >>> factorial(arr, exact=False)
    array([   6.,   24.,  120.])
    >>> factorial(arr, exact=True)
    array([  6,  24, 120])
    >>> factorial(5, exact=True)
    120L

    """
    if exact:
        if np.ndim(n) == 0:
            return 0 if n < 0 else math.factorial(n)
        else:
            n = asarray(n)
            un = np.unique(n).astype(object)

            # Convert to object array of long ints if np.int can't handle size
            if un[-1] > 20:
                dt = object
            elif un[-1] > 12:
                dt = np.int64
            else:
                dt = np.int

            out = np.empty_like(n, dtype=dt)

            # Handle invalid/trivial values
            un = un[un > 1]
            out[n < 2] = 1
            out[n < 0] = 0

            # Calculate products of each range of numbers
            if un.size:
                val = math.factorial(un[0])
                out[n == un[0]] = val
                for i in xrange(len(un) - 1):
                    prev = un[i] + 1
                    current = un[i + 1]
                    val *= _range_prod(prev, current)
                    out[n == current] = val
            return out
    else:
        n = asarray(n)
        vals = gamma(n + 1)
        return where(n >= 0, vals, 0)


def factorial2(n, exact=False):
    """Double factorial.

    This is the factorial with every second value skipped.  E.g., ``7!! = 7 * 5
    * 3 * 1``.  It can be approximated numerically as::

      n!! = special.gamma(n/2+1)*2**((m+1)/2)/sqrt(pi)  n odd
          = 2**(n/2) * (n/2)!                           n even

    Parameters
    ----------
    n : int or array_like
        Calculate ``n!!``.  Arrays are only supported with `exact` set
        to False.  If ``n < 0``, the return value is 0.
    exact : bool, optional
        The result can be approximated rapidly using the gamma-formula
        above (default).  If `exact` is set to True, calculate the
        answer exactly using integer arithmetic.

    Returns
    -------
    nff : float or int
        Double factorial of `n`, as an int or a float depending on
        `exact`.

    Examples
    --------
    >>> from scipy.special import factorial2
    >>> factorial2(7, exact=False)
    array(105.00000000000001)
    >>> factorial2(7, exact=True)
    105L

    """
    if exact:
        if n < -1:
            return 0
        if n <= 0:
            return 1
        val = 1
        for k in xrange(n, 0, -2):
            val *= k
        return val
    else:
        n = asarray(n)
        vals = zeros(n.shape, 'd')
        cond1 = (n % 2) & (n >= -1)
        cond2 = (1-(n % 2)) & (n >= -1)
        oddn = extract(cond1, n)
        evenn = extract(cond2, n)
        nd2o = oddn / 2.0
        nd2e = evenn / 2.0
        place(vals, cond1, gamma(nd2o + 1) / sqrt(pi) * pow(2.0, nd2o + 0.5))
        place(vals, cond2, gamma(nd2e + 1) * pow(2.0, nd2e))
        return vals


def factorialk(n, k, exact=True):
    """Multifactorial of n of order k, n(!!...!).

    This is the multifactorial of n skipping k values.  For example,

      factorialk(17, 4) = 17!!!! = 17 * 13 * 9 * 5 * 1

    In particular, for any integer ``n``, we have

      factorialk(n, 1) = factorial(n)

      factorialk(n, 2) = factorial2(n)

    Parameters
    ----------
    n : int
        Calculate multifactorial. If `n` < 0, the return value is 0.
    k : int
        Order of multifactorial.
    exact : bool, optional
        If exact is set to True, calculate the answer exactly using
        integer arithmetic.

    Returns
    -------
    val : int
        Multifactorial of `n`.

    Raises
    ------
    NotImplementedError
        Raises when exact is False

    Examples
    --------
    >>> from scipy.special import factorialk
    >>> factorialk(5, 1, exact=True)
    120L
    >>> factorialk(5, 3, exact=True)
    10L

    """
    if exact:
        if n < 1-k:
            return 0
        if n <= 0:
            return 1
        val = 1
        for j in xrange(n, 0, -k):
            val = val*j
        return val
    else:
        raise NotImplementedError


def zeta(x, q=None, out=None):
    r"""
    Riemann zeta function.

    The two-argument version is the Hurwitz zeta function:

    .. math:: \zeta(x, q) = \sum_{k=0}^{\infty} \frac{1}{(k + q)^x},

    Riemann zeta function corresponds to ``q = 1``.

    See also
    --------
    zetac

    """
    if q is None:
        q = 1
    return _zeta(x, q, out)


def _get_method(h, a):
    select = [1, 1, 2, 13, 13, 13, 13, 13, 13, 13, 13, 16, 16, 16, 9,
              1, 2, 2, 3, 3, 5, 5, 14, 14, 15, 15, 16, 16, 16, 9,
              2, 2, 3, 3, 3, 5, 5, 15, 15, 15, 15, 16, 16, 16, 10,
              2, 2, 3, 5, 5, 5, 5, 7, 7, 16, 16, 16, 16, 16, 10,
              2, 3, 3, 5, 5, 6, 6, 8, 8, 17, 17, 17, 12, 12, 11,
              2, 3, 5, 5, 5, 6, 6, 8, 8, 17, 17, 17, 12, 12, 12,
              2, 3, 4, 4, 6, 6, 8, 8, 17, 17, 17, 17, 17, 12, 12,
              2, 3, 4, 4, 6, 6, 18, 18, 18, 18, 17, 17, 17, 12, 12]

    hrange = [0.2, 0.06, 0.09, 0.125, 0.26, 0.4, 0.6, 1.6,
              1.7, 2.33, 2.4, 3.36, 3.4, 4.8]

    arange = [0.025, 0.09, 0.15, 0.36, 0.5, 0.9, 0.99999]

    ihint = 14
    iaint = 7

    for i in xrange(len(hrange)):
        if h <= hrange[i]:
            ihint = i
            break

    for i in xrange(len(arange)):
        if a < arange[i]:
            iaint = i
            break

    return select[iaint * 15 + ihint] - 1


def _get_order(index):
    ord = [2, 3, 4, 5, 7, 10, 12, 18, 10, 20, 30, 0, 4, 7, 8, 20, 0, 0]

    return ord[index]


def _owens_t_norm1(x):
    return erf(x / np.sqrt(2)) / 2


def _owens_t_norm2(x):
    return erfc(x / np.sqrt(2)) / 2


def _owensT1(h, a, m):

    a_sq = a * a
    h_sq = -1 * h * h / 2
    h_exp = np.exp(h_sq)

    div = 1
    aj = a / (2 * np.pi)
    dj = np.expm1(h_sq)
    gj = h_sq * h_exp

    result = np.arctan(a) * (1 / (2 * np.pi))

    for j in xrange(1, int(m) + 1):
        result += dj * aj / div
        div += 2
        aj *= a_sq
        dj = gj - dj
        gj *= h_sq / j

    return result


def _owensT2(h, a, m):

    h_sq = h * h
    ah_sq = -0.5 * h * h * a * a
    pi_sq = np.power(2 * np.pi, -0.5)

    vi = a * np.exp(ah_sq) * pi_sq
    z = (ndtr(a * h) - 0.5) / h

    result = 0

    for i in xrange(int(2 * m) + 1):
        result += np.power(-1, i - 1) * z
        z = ((2 * i - 1) * z - (pi_sq * np.power(a, 2 * i - 1) *
             np.exp(ah_sq))) / (h_sq)

    result *= -pi_sq * np.exp(-0.5 * h_sq)

    return result


def _owensT3(h, a, m):

    c = [0.99999999999999999999999729978162447266851932041876728736094298092917625009873,
         -0.99999999999999999999467056379678391810626533251885323416799874878563998732905968,
         0.99999999999999999824849349313270659391127814689133077036298754586814091034842536,
         -0.9999999999999997703859616213643405880166422891953033591551179153879839440241685,
         0.99999999999998394883415238173334565554173013941245103172035286759201504179038147,
         -0.9999999999993063616095509371081203145247992197457263066869044528823599399470977,
         0.9999999999797336340409464429599229870590160411238245275855903767652432017766116267,
         -0.999999999574958412069046680119051639753412378037565521359444170241346845522403274,
         0.9999999933226234193375324943920160947158239076786103108097456617750134812033362048,
         -0.9999999188923242461073033481053037468263536806742737922476636768006622772762168467,
         0.9999992195143483674402853783549420883055129680082932629160081128947764415749728967,
         -0.999993935137206712830997921913316971472227199741857386575097250553105958772041501,
         0.99996135597690552745362392866517133091672395614263398912807169603795088421057688716,
         -0.99979556366513946026406788969630293820987757758641211293079784585126692672425362469,
         0.999092789629617100153486251423850590051366661947344315423226082520411961968929483,
         -0.996593837411918202119308620432614600338157335862888580671450938858935084316004769854,
         0.98910017138386127038463510314625339359073956513420458166238478926511821146316469589567,
         -0.970078558040693314521331982203762771512160168582494513347846407314584943870399016019,
         0.92911438683263187495758525500033707204091967947532160289872782771388170647150321633673,
         -0.8542058695956156057286980736842905011429254735181323743367879525470479126968822863,
         0.73796526033030091233118357742803709382964420335559408722681794195743240930748630755,
         -0.58523469882837394570128599003785154144164680587615878645171632791404210655891158,
         0.415997776145676306165661663581868460503874205343014196580122174949645271353372263,
         -0.2588210875241943574388730510317252236407805082485246378222935376279663808416534365,
         0.1375535825163892648504646951500265585055789019410617565727090346559210218472356689,
         -0.0607952766325955730493900985022020434830339794955745989150270485056436844239206648,
         0.0216337683299871528059836483840390514275488679530797294557060229266785853764115,
         -0.00593405693455186729876995814181203900550014220428843483927218267309209471516256,
         0.0011743414818332946510474576182739210553333860106811865963485870668929503649964142,
         -1.489155613350368934073453260689881330166342484405529981510694514036264969925132E-4,
         9.072354320794357587710929507988814669454281514268844884841547607134260303118208E-6]

    a_sq = a * a
    h_sq = h * h
    y = 1 / h_sq

    vi = a * np.exp(- a_sq * h_sq / 2) / np.sqrt(2 * np.pi)
    zi = _owens_t_norm1(a * h) / h
    val = 0

    for i in xrange(int(m) + 1):
        val += zi * c[i]
        zi = y * ((2 * i + 1) * zi - vi)
        vi *= a_sq

    val *= np.exp(-h_sq / 2) / np.sqrt(2 * np.pi)

    return val


def _owensT4(h, a, m):
    maxi = 2 * m + 1
    h_sq = h * h
    a_sq = -a * a

    i = 1
    ai = a * np.exp(-h_sq * (1 - a_sq) / 2) / (2 * np.pi)
    yi = 1
    result = 0

    while True:
        result += ai * yi

        if maxi <= i:
            break

        i += 2
        yi = (1 - h_sq * yi) / i
        ai *= a_sq

    return result


def _owensT5(h, a, m):

    pts = [0.35082039676451715489E-02, 0.31279042338030753740E-01,
           0.85266826283219451090E-01, 0.16245071730812277011E+00,
           0.25851196049125434828E+00, 0.36807553840697533536E+00,
           0.48501092905604697475E+00, 0.60277514152618576821E+00,
           0.71477884217753226516E+00, 0.81475510988760098605E+00,
           0.89711029755948965867E+00, 0.95723808085944261843E+00,
           0.99178832974629703586E+00]

    wts = [0.18831438115323502887E-01, 0.18567086243977649478E-01,
           0.18042093461223385584E-01, 0.17263829606398753364E-01,
           0.16243219975989856730E-01, 0.14994592034116704829E-01,
           0.13535474469662088392E-01, 0.11886351605820165233E-01,
           0.10070377242777431897E-01, 0.81130545742299586629E-02,
           0.60419009528470238773E-02, 0.38862217010742057883E-02,
           0.16793031084546090448E-02]

    result = 0

    a_sq = a * a
    h_sq = -0.5 * h * h

    for i in range(1, 14):
        r = 1 + a_sq * pts[i - 1]
        result += wts[i - 1] * np.exp(h_sq * r) / r

    result *= a

    return result


def _owensT6(h, a, m):

    result = (0.5 * (ndtr(h)) * (1 - ndtr(h)) - np.arctan((1 - a) / (1 + a)) *
              np.exp(-0.5 * (1 - a) * h * h / np.arctan((1 - a) / (1 + a))) /
              (2 * np.pi))

    return result


def owens_t(h, a):
    r"""
    Owen's T function.

    Parameters
    ----------
    h: scalar
        Input value.
    a: float
        Input value.
        If ``a == 0``, the return value is 0.

    Returns
    -------
    t: float
        Probability of the event (X > h and 0 < Y < a * X),
        where X and Y are independent standard normal random variables.

    Examples
    --------
    >>> from scipy import special

    >>> a = 3.5
    >>> h = 0.78
    >>> special.owens_t(h, a)
    0.11317292784451626

    References
    ----------
    .. [1] M. Patefield and D. Tandy, "Fast and accurate calculation of 
           Owen’st function", Statistical Software vol. 5, pp. 1-25, 2000.

    """

    if h < 0:
        return owens_t(-h, a)

    if a < 0:
        return -owens_t(h, a)
    elif a > 1:
        ncdf_h = ndtr(h)
        ncdf_ah = ndtr(a * h)
        return (0.5 * (ncdf_h + ncdf_ah) - ncdf_h * ncdf_ah -
                owens_t(a * h, 1. / a))

    if a == 0:
        return 0

    if h == 0:
        return 1 * np.arctan(a) / (2 * np.pi)

    if a == 1:
        return _owens_t_norm2(-h) * _owens_t_norm2(h) / 2

    meth = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6]

    index = _get_method(h, a)
    m = _get_order(index)
    meth_code = meth[index - 1]

    if meth_code == 1:
        result = _owensT1(h, a, m)
    elif meth_code == 2:
        result = _owensT2(h, a, m)
    elif meth_code == 3:
        result = _owensT3(h, a, m)
    elif meth_code == 4:
        result = _owensT4(h, a, m)
    elif meth_code == 5:
        result = _owensT5(h, a, m)
    elif meth_code == 6:
        result = _owensT6(h, a, m)

    return result
