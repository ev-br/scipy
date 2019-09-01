"""
Low-level LAPACK functions (:mod:`scipy.linalg.lapack`)
=======================================================

This module contains low-level functions from the LAPACK library.

The `*gegv` family of routines have been removed from LAPACK 3.6.0
and have been deprecated in SciPy 0.17.0. They will be removed in
a future release.

.. versionadded:: 0.12.0

.. note::

    The common ``overwrite_<>`` option in many routines, allows the
    input arrays to be overwritten to avoid extra memory allocation.
    However this requires the array to satisfy two conditions
    which are memory order and the data type to match exactly the
    order and the type expected by the routine.

    As an example, if you pass a double precision float array to any
    ``S....`` routine which expects single precision arguments, f2py
    will create an intermediate array to match the argument types and
    overwriting will be performed on that intermediate array.

    Similarly, if a C-contiguous array is passed, f2py will pass a
    FORTRAN-contiguous array internally. Please make sure that these
    details are satisfied. More information can be found in the f2py
    documentation.

.. warning::

   These functions do little to no error checking.
   It is possible to cause crashes by mis-using them,
   so prefer using the higher-level routines in `scipy.linalg`.

Finding functions
-----------------

.. autosummary::
   :toctree: generated/

   get_lapack_funcs

All functions
-------------

.. autosummary::
   :toctree: generated/


   sgbsv
   dgbsv
   cgbsv
   zgbsv

   sgbtrf
   dgbtrf
   cgbtrf
   zgbtrf

   sgbtrs
   dgbtrs
   cgbtrs
   zgbtrs

   sgebal
   dgebal
   cgebal
   zgebal

   sgecon
   dgecon
   cgecon
   zgecon

   sgees
   dgees
   cgees
   zgees

   sgeev
   dgeev
   cgeev
   zgeev

   sgeev_lwork
   dgeev_lwork
   cgeev_lwork
   zgeev_lwork

   sgehrd
   dgehrd
   cgehrd
   zgehrd

   sgehrd_lwork
   dgehrd_lwork
   cgehrd_lwork
   zgehrd_lwork

   sgels
   dgels
   cgels
   zgels

   sgels_lwork
   dgels_lwork
   cgels_lwork
   zgels_lwork

   sgelsd
   dgelsd
   cgelsd
   zgelsd

   sgelsd_lwork
   dgelsd_lwork
   cgelsd_lwork
   zgelsd_lwork

   sgelss
   dgelss
   cgelss
   zgelss

   sgelss_lwork
   dgelss_lwork
   cgelss_lwork
   zgelss_lwork

   sgelsy
   dgelsy
   cgelsy
   zgelsy

   sgelsy_lwork
   dgelsy_lwork
   cgelsy_lwork
   zgelsy_lwork

   sgeqp3
   dgeqp3
   cgeqp3
   zgeqp3

   sgeqrf
   dgeqrf
   cgeqrf
   zgeqrf

   sgerqf
   dgerqf
   cgerqf
   zgerqf

   sgesdd
   dgesdd
   cgesdd
   zgesdd

   sgesdd_lwork
   dgesdd_lwork
   cgesdd_lwork
   zgesdd_lwork

   sgesv
   dgesv
   cgesv
   zgesv

   sgesvd
   dgesvd
   cgesvd
   zgesvd

   sgesvd_lwork
   dgesvd_lwork
   cgesvd_lwork
   zgesvd_lwork

   sgesvx
   dgesvx
   cgesvx
   zgesvx

   sgetrf
   dgetrf
   cgetrf
   zgetrf

   sgetri
   dgetri
   cgetri
   zgetri

   sgetri_lwork
   dgetri_lwork
   cgetri_lwork
   zgetri_lwork

   sgetrs
   dgetrs
   cgetrs
   zgetrs

   sgges
   dgges
   cgges
   zgges

   sggev
   dggev
   cggev
   zggev

   sgglse
   dgglse
   cgglse
   zgglse

   sgglse_lwork
   dgglse_lwork
   cgglse_lwork
   zgglse_lwork

   sgtsv
   dgtsv
   cgtsv
   zgtsv

   chbevd
   zhbevd

   chbevx
   zhbevx

   checon
   zhecon

   cheev
   zheev

   cheevd
   zheevd

   cheevr
   zheevr

   chegst
   zhegst

   chegv
   zhegv

   chegvd
   zhegvd

   chegvx
   zhegvx

   chesv
   zhesv

   chesv_lwork
   zhesv_lwork

   chesvx
   zhesvx

   chesvx_lwork
   zhesvx_lwork

   chetrd
   zhetrd

   chetrd_lwork
   zhetrd_lwork

   chetrf
   zhetrf

   chetrf_lwork
   zhetrf_lwork

   chfrk
   zhfrk

   slamch
   dlamch

   slange
   dlange
   clange
   zlange

   slarf
   dlarf
   clarf
   zlarf

   slarfg
   dlarfg
   clarfg
   zlarfg

   slartg
   dlartg
   clartg
   zlartg

   slasd4
   dlasd4

   slaswp
   dlaswp
   claswp
   zlaswp

   slauum
   dlauum
   clauum
   zlauum

   sorghr
   dorghr
   sorghr_lwork
   dorghr_lwork

   sorgqr
   dorgqr

   sorgrq
   dorgrq

   sormqr
   dormqr

   sormrz
   dormrz

   sormrz_lwork
   dormrz_lwork

   spbsv
   dpbsv
   cpbsv
   zpbsv

   spbtrf
   dpbtrf
   cpbtrf
   zpbtrf

   spbtrs
   dpbtrs
   cpbtrs
   zpbtrs

   spftrf
   dpftrf
   cpftrf
   zpftrf

   spftri
   dpftri
   cpftri
   zpftri

   spftrs
   dpftrs
   cpftrs
   zpftrs

   spocon
   dpocon
   cpocon
   zpocon

   sposv
   dposv
   cposv
   zposv

   sposvx
   dposvx
   cposvx
   zposvx

   spotrf
   dpotrf
   cpotrf
   zpotrf

   spotri
   dpotri
   cpotri
   zpotri

   spotrs
   dpotrs
   cpotrs
   zpotrs

   sptsv
   dptsv
   cptsv
   zptsv

   crot
   zrot

   ssbev
   dsbev

   ssbevd
   dsbevd

   ssbevx
   dsbevx

   ssfrk
   dsfrk

   sstebz
   dstebz

   sstein
   dstein

   sstemr
   dstemr

   sstemr_lwork
   dstemr_lwork

   ssterf
   dsterf

   sstev
   dstev

   ssycon
   dsycon
   csycon
   zsycon

   ssyconv
   dsyconv
   csyconv
   zsyconv

   ssyev
   dsyev

   ssyevd
   dsyevd

   ssyevr
   dsyevr

   ssygst
   dsygst

   ssygv
   dsygv

   ssygvd
   dsygvd

   ssygvx
   dsygvx

   ssysv
   dsysv
   csysv
   zsysv

   ssysv_lwork
   dsysv_lwork
   csysv_lwork
   zsysv_lwork

   ssysvx
   dsysvx
   csysvx
   zsysvx

   ssysvx_lwork
   dsysvx_lwork
   csysvx_lwork
   zsysvx_lwork

   ssytf2
   dsytf2
   csytf2
   zsytf2

   ssytrd
   dsytrd

   ssytrd_lwork
   dsytrd_lwork

   ssytrf
   dsytrf
   csytrf
   zsytrf

   ssytrf_lwork
   dsytrf_lwork
   csytrf_lwork
   zsytrf_lwork

   stfsm
   dtfsm
   ctfsm
   ztfsm

   stfttp
   dtfttp
   ctfttp
   ztfttp

   stfttr
   dtfttr
   ctfttr
   ztfttr

   stgsen
   dtgsen
   ctgsen
   ztgsen

   stpttf
   dtpttf
   ctpttf
   ztpttf

   stpttr
   dtpttr
   ctpttr
   ztpttr

   strsyl
   dtrsyl
   ctrsyl
   ztrsyl

   strtri
   dtrtri
   ctrtri
   ztrtri

   strtrs
   dtrtrs
   ctrtrs
   ztrtrs

   strttf
   dtrttf
   ctrttf
   ztrttf

   strttp
   dtrttp
   ctrttp
   ztrttp

   stzrzf
   dtzrzf
   ctzrzf
   ztzrzf

   stzrzf_lwork
   dtzrzf_lwork
   ctzrzf_lwork
   ztzrzf_lwork

   cunghr
   zunghr

   cunghr_lwork
   zunghr_lwork

   cungqr
   zungqr

   cungrq
   zungrq

   cunmqr
   zunmqr

   sgeqrt
   dgeqrt
   cgeqrt
   zgeqrt

   sgemqrt
   dgemqrt
   cgemqrt
   zgemqrt

   stpqrt
   dtpqrt
   ctpqrt
   ztpqrt

   stpmqrt
   dtpmqrt
   ctpmqrt
   ztpmqrt

   cunmrz
   zunmrz

   cunmrz_lwork
   zunmrz_lwork

   ilaver

"""
#
# Author: Pearu Peterson, March 2002
#

from __future__ import division, print_function, absolute_import
import numpy as _np
from .blas import _get_funcs, _memoize_get_funcs
from scipy.linalg import _flapack
try:
    from scipy.linalg import _clapack
except ImportError:
    _clapack = None

# Backward compatibility
from .blas import find_best_blas_type as find_best_lapack_type
from scipy._lib._util import DeprecatedImport as _DeprecatedImport
clapack = _DeprecatedImport("scipy.linalg.blas.clapack", "scipy.linalg.lapack")
flapack = _DeprecatedImport("scipy.linalg.blas.flapack", "scipy.linalg.lapack")

# Expose all functions (only flapack --- clapack is an implementation detail)
empty_module = None
from scipy.linalg._flapack import *
del empty_module

__all__ = ['get_lapack_funcs',
           'cgbsv', 'cgbtrf', 'cgbtrs', 'cgebal', 'cgecon', 'cgees',
           'cgeev_lwork', 'cgegv', 'cgehrd', 'cgehrd_lwork', 'cgels',
           'cgelsd', 'cgelsd_lwork', 'cgelss', 'cgelss_lwork', 'cgelsy',
           'cgemqrt', 'cgeqp3', 'cgeqrf', 'cgeqrt', 'cgerqf', 'cgesdd',
           'cgesv', 'cgesvd', 'cgesvd_lwork', 'cgesvx', 'cgetrf', 'cgetri',
           'cgetrs', 'cgges', 'cggev', 'cgglse', 'cgglse_lwork', 'cgtsv',
           'chbevx', 'checon', 'cheev', 'cheevd', 'cheevr', 'chegst',
           'chegvd', 'chegvx', 'chesv', 'chesv_lwork', 'chesvx',
           'chetrd', 'chetrd_lwork', 'chetrf', 'chetrf_lwork', 'chfrk',
           'clarf', 'clarfg', 'clartg', 'claswp', 'clauum', 'cpbsv',
           'cpbtrs', 'cpftrf', 'cpftri', 'cpftrs', 'cpocon', 'cposv',
           'cpotrf', 'cpotri', 'cpotrs', 'cptsv', 'crot', 'csycon',
           'csysv', 'csysv_lwork', 'csysvx', 'csysvx_lwork', 'csytf2',
           'csytrf_lwork', 'ctfsm', 'ctfttp', 'ctfttr', 'ctgsen', 'ctpmqrt',
           'ctpttf', 'ctpttr', 'ctrsyl', 'ctrtri', 'ctrtrs', 'ctrttf',
           'ctzrzf', 'ctzrzf_lwork', 'cunghr', 'cunghr_lwork', 'cungqr',
           'cunmqr', 'cunmrz', 'cunmrz_lwork', 'dgbsv', 'dgbtrf', 'dgbtrs',
           'dgecon', 'dgees', 'dgeev', 'dgeev_lwork', 'dgegv', 'dgehrd',
           'dgels', 'dgels_lwork', 'dgelsd', 'dgelsd_lwork', 'dgelss',
           'dgelsy', 'dgelsy_lwork', 'dgemqrt', 'dgeqp3', 'dgeqrf',
           'dgerqf', 'dgesdd', 'dgesdd_lwork', 'dgesv', 'dgesvd',
           'dgesvx', 'dgetrf', 'dgetri', 'dgetri_lwork', 'dgetrs', 'dgges',
           'dgglse', 'dgglse_lwork', 'dgtsv', 'dlamch', 'dlange', 'dlarf',
           'dlartg', 'dlasd4', 'dlaswp', 'dlauum', 'dorghr', 'dorghr_lwork',
           'dorgrq', 'dormqr', 'dormrz', 'dormrz_lwork', 'dpbsv', 'dpbtrf',
           'dpftrf', 'dpftri', 'dpftrs', 'dpocon', 'dposv', 'dposvx',
           'dpotri', 'dpotrs', 'dptsv', 'dsbev', 'dsbevd', 'dsbevx',
           'dstebz', 'dstein', 'dstemr', 'dstemr_lwork', 'dsterf', 'dstev',
           'dsyconv', 'dsyev', 'dsyevd', 'dsyevr', 'dsygst', 'dsygv',
           'dsygvx', 'dsysv', 'dsysv_lwork', 'dsysvx', 'dsysvx_lwork',
           'dsytrd', 'dsytrd_lwork', 'dsytrf', 'dsytrf_lwork', 'dtfsm',
           'dtfttr', 'dtgsen', 'dtpmqrt', 'dtpqrt', 'dtpttf', 'dtpttr',
           'dtrtri', 'dtrtrs', 'dtrttf', 'dtrttp', 'dtzrzf', 'dtzrzf_lwork',
           'sgbsv', 'sgbtrf', 'sgbtrs', 'sgebal', 'sgecon', 'sgees',
           'sgeev_lwork', 'sgegv', 'sgehrd', 'sgehrd_lwork', 'sgels',
           'sgelsd', 'sgelsd_lwork', 'sgelss', 'sgelss_lwork', 'sgelsy',
           'sgemqrt', 'sgeqp3', 'sgeqrf', 'sgeqrt', 'sgerqf', 'sgesdd',
           'sgesv', 'sgesvd', 'sgesvd_lwork', 'sgesvx', 'sgetrf', 'sgetri',
           'sgetrs', 'sgges', 'sggev', 'sgglse', 'sgglse_lwork', 'sgtsv',
           'slange', 'slarf', 'slarfg', 'slartg', 'slasd4', 'slaswp',
           'sorghr', 'sorghr_lwork', 'sorgqr', 'sorgrq', 'sormqr', 'sormrz',
           'spbsv', 'spbtrf', 'spbtrs', 'spftrf', 'spftri', 'spftrs',
           'sposv', 'sposvx', 'spotrf', 'spotri', 'spotrs', 'sptsv',
           'ssbevd', 'ssbevx', 'ssfrk', 'sstebz', 'sstein', 'sstemr',
           'ssterf', 'sstev', 'ssycon', 'ssyconv', 'ssyev', 'ssyevd',
           'ssygst', 'ssygv', 'ssygvd', 'ssygvx', 'ssysv', 'ssysv_lwork',
           'ssysvx_lwork', 'ssytf2', 'ssytrd', 'ssytrd_lwork', 'ssytrf',
           'stfsm', 'stfttp', 'stfttr', 'stgsen', 'stpmqrt', 'stpqrt',
           'stpttr', 'strsyl', 'strtri', 'strtrs', 'strttf', 'strttp',
           'stzrzf_lwork', 'zgbsv', 'zgbtrf', 'zgbtrs', 'zgebal', 'zgecon',
           'zgeev', 'zgeev_lwork', 'zgegv', 'zgehrd', 'zgehrd_lwork',
           'zgels_lwork', 'zgelsd', 'zgelsd_lwork', 'zgelss',
           'zgelsy', 'zgelsy_lwork', 'zgemqrt', 'zgeqp3', 'zgeqrf',
           'zgerqf', 'zgesdd', 'zgesdd_lwork', 'zgesv', 'zgesvd',
           'zgesvx', 'zgetrf', 'zgetri', 'zgetri_lwork', 'zgetrs', 'zgges',
           'zgglse', 'zgglse_lwork', 'zgtsv', 'zhbevd', 'zhbevx', 'zhecon',
           'zheevd', 'zheevr', 'zhegst', 'zhegv', 'zhegvd', 'zhegvx',
           'zhesv_lwork', 'zhesvx', 'zhesvx_lwork', 'zhetrd',
           'zhetrf', 'zhetrf_lwork', 'zhfrk', 'zlange', 'zlarf', 'zlarfg',
           'zlaswp', 'zlauum', 'zpbsv', 'zpbtrf', 'zpbtrs', 'zpftrf',
           'zpftrs', 'zpocon', 'zposv', 'zposvx', 'zpotrf', 'zpotri',
           'zptsv', 'zrot', 'zsycon', 'zsyconv', 'zsysv', 'zsysv_lwork',
           'zsysvx_lwork', 'zsytf2', 'zsytrf', 'zsytrf_lwork', 'ztfsm',
           'ztfttr', 'ztgsen', 'ztpmqrt', 'ztpqrt', 'ztpttf', 'ztpttr',
           'ztrtri', 'ztrtrs', 'ztrttf', 'ztrttp', 'ztzrzf', 'ztzrzf_lwork',
           'zunghr_lwork', 'zungqr', 'zungrq', 'zunmqr', 'zunmrz',
           'cgeev', 'cgels_lwork', 'cgelsy_lwork', 'cgesdd_lwork',
           'chbevd', 'chegv', 'chesvx_lwork', 'clange', 'cpbtrf', 'cposvx',
           'csytrf', 'ctpqrt', 'ctrttp', 'cungrq', 'dgebal', 'dgehrd_lwork',
           'dgeqrt', 'dgesvd_lwork', 'dggev', 'dlarfg', 'dorgqr', 'dpbtrs',
           'dsfrk', 'dsycon', 'dsygvd', 'dsytf2', 'dtfttp', 'dtrsyl',
           'sgeev', 'sgels_lwork', 'sgelsy_lwork', 'sgesdd_lwork',
           'slamch', 'slauum', 'sormrz_lwork', 'spocon', 'ssbev',
           'ssyevr', 'ssysvx', 'ssytrf_lwork', 'stpttf', 'stzrzf', 'zgees',
           'zgelss_lwork', 'zgeqrt', 'zgesvd_lwork', 'zggev', 'zheev',
           'zhetrd_lwork', 'zlartg', 'zpftri', 'zpotrs', 'zsysvx', 'ztfttp',
           'zunghr', 'zunmrz_lwork', 'cgetri_lwork', 'csyconv',
           'dgelss_lwork', 'dpotrf', 'ilaver', 'sgetri_lwork', 'sstemr_lwork',
           'zgels', 'zhesv', 'ztrsyl',
]

_dep_message = """The `*gegv` family of routines has been deprecated in
LAPACK 3.6.0 in favor of the `*ggev` family of routines.
The corresponding wrappers will be removed from SciPy in
a future release."""

cgegv = _np.deprecate(cgegv, old_name='cgegv', message=_dep_message)
dgegv = _np.deprecate(dgegv, old_name='dgegv', message=_dep_message)
sgegv = _np.deprecate(sgegv, old_name='sgegv', message=_dep_message)
zgegv = _np.deprecate(zgegv, old_name='zgegv', message=_dep_message)

# Modyfy _flapack in this scope so the deprecation warnings apply to
# functions returned by get_lapack_funcs.
_flapack.cgegv = cgegv
_flapack.dgegv = dgegv
_flapack.sgegv = sgegv
_flapack.zgegv = zgegv

# some convenience alias for complex functions
_lapack_alias = {
    'corghr': 'cunghr', 'zorghr': 'zunghr',
    'corghr_lwork': 'cunghr_lwork', 'zorghr_lwork': 'zunghr_lwork',
    'corgqr': 'cungqr', 'zorgqr': 'zungqr',
    'cormqr': 'cunmqr', 'zormqr': 'zunmqr',
    'corgrq': 'cungrq', 'zorgrq': 'zungrq',
}


@_memoize_get_funcs
def get_lapack_funcs(names, arrays=(), dtype=None):
    """Return available LAPACK function objects from names.

    Arrays are used to determine the optimal prefix of LAPACK routines.

    Parameters
    ----------
    names : str or sequence of str
        Name(s) of LAPACK functions without type prefix.

    arrays : sequence of ndarrays, optional
        Arrays can be given to determine optimal prefix of LAPACK
        routines. If not given, double-precision routines will be
        used, otherwise the most generic type in arrays will be used.

    dtype : str or dtype, optional
        Data-type specifier. Not used if `arrays` is non-empty.

    Returns
    -------
    funcs : list
        List containing the found function(s).

    Notes
    -----
    This routine automatically chooses between Fortran/C
    interfaces. Fortran code is used whenever possible for arrays with
    column major order. In all other cases, C code is preferred.

    In LAPACK, the naming convention is that all functions start with a
    type prefix, which depends on the type of the principal
    matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
    types {float32, float64, complex64, complex128} respectively, and
    are stored in attribute ``typecode`` of the returned functions.

    Examples
    --------
    Suppose we would like to use '?lange' routine which computes the selected
    norm of an array. We pass our array in order to get the correct 'lange'
    flavor.

    >>> import scipy.linalg as LA
    >>> a = np.random.rand(3,2)
    >>> x_lange = LA.get_lapack_funcs('lange', (a,))
    >>> x_lange.typecode
    'd'
    >>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))
    >>> x_lange.typecode
    'z'

    Several LAPACK routines work best when its internal WORK array has
    the optimal size (big enough for fast computation and small enough to
    avoid waste of memory). This size is determined also by a dedicated query
    to the function which is often wrapped as a standalone function and
    commonly denoted as ``###_lwork``. Below is an example for ``?sysv``

    >>> import scipy.linalg as LA
    >>> a = np.random.rand(1000,1000)
    >>> b = np.random.rand(1000,1)*1j
    >>> # We pick up zsysv and zsysv_lwork due to b array
    ... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))
    >>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix
    >>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))

    """
    return _get_funcs(names, arrays, dtype,
                      "LAPACK", _flapack, _clapack,
                      "flapack", "clapack", _lapack_alias)


_int32_max = _np.iinfo(_np.int32).max


def _compute_lwork(routine, *args, **kwargs):
    """
    Round floating-point lwork returned by lapack to integer.

    Several LAPACK routines compute optimal values for LWORK, which
    they return in a floating-point variable. However, for large
    values of LWORK, single-precision floating point is not sufficient
    to hold the exact value --- some LAPACK versions (<= 3.5.0 at
    least) truncate the returned integer to single precision and in
    some cases this can be smaller than the required value.

    Examples
    --------
    >>> from scipy.linalg import lapack
    >>> n = 5000
    >>> s_r, s_lw = lapack.get_lapack_funcs(('sysvx', 'sysvx_lwork'))
    >>> lwork = lapack._compute_lwork(s_lw, n)
    >>> lwork
    32000

    """
    dtype = getattr(routine, 'dtype', None)
    ret = routine(*args, **kwargs)
    if ret[-1] != 0:
        raise ValueError("Internal work array size computation failed: "
                         "%d" % (ret[-1],))

    if len(ret) == 2:
        return _check_work_float(ret[0].real, dtype)
    else:
        return tuple(_check_work_float(x.real, dtype) for x in ret[:-1])


def _check_work_float(value, dtype):
    """
    Convert LAPACK-returned work array size float to integer,
    carefully for single-precision types.
    """

    if dtype == _np.float32 or dtype == _np.complex64:
        # Single-precision routine -- take next fp value to work
        # around possible truncation in LAPACK code
        value = _np.nextafter(value, _np.inf, dtype=_np.float32)

    value = int(value)
    if value < 0 or value > _int32_max:
        raise ValueError("Too large work array required -- computation cannot "
                         "be performed with standard 32-bit LAPACK.")
    return value
