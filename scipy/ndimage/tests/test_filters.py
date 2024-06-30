''' Some tests for filters '''
import functools
import itertools
import math

from scipy._lib._array_api import (
    xp_assert_equal, xp_assert_close,
    assert_array_almost_equal,
    assert_almost_equal,
)
from scipy._lib._array_api import is_cupy

from numpy.testing import suppress_warnings
import pytest
from pytest import raises as assert_raises

from scipy import ndimage
from scipy.ndimage._filters import _gaussian_kernel1d

from . import types, float_types, complex_types

from scipy.conftest import array_api_compatible
pytestmark = [array_api_compatible, pytest.mark.usefixtures("skip_xp_backends")]
skip_xp_backends = pytest.mark.skip_xp_backends


def sumsq(a, b):
    return math.sqrt(((a - b)**2).sum())


def _complex_correlate(xp, array, kernel, real_dtype, convolve=False,
                       mode="reflect", cval=0, ):
    """Utility to perform a reference complex-valued convolutions.

    When convolve==False, correlation is performed instead
    """
    array = xp.asarray(array)
    kernel = xp.asarray(kernel)
    complex_array = array.dtype.kind == 'c'
    complex_kernel = kernel.dtype.kind == 'c'
    if array.ndim == 1:
        func = ndimage.convolve1d if convolve else ndimage.correlate1d
    else:
        func = ndimage.convolve if convolve else ndimage.correlate
    if not convolve:
        kernel = kernel.conj()
    if complex_array and complex_kernel:
        # use: real(cval) for array.real component
        #      imag(cval) for array.imag component
        re_cval = xp.real(cval)
        re_cval = re_cval.item() if isinstance(re_cval, xp.ndarray) else re_cval
        im_cval = xp.imag(cval)
        im_cval = im_cval.item() if isinstance(im_cval, xp.ndarray) else im_cval

        output = (
            func(array.real, kernel.real, output=real_dtype,
                 mode=mode, cval=re_cval) -
            func(array.imag, kernel.imag, output=real_dtype,
                 mode=mode, cval=im_cval) +
            1j * func(array.imag, kernel.real, output=real_dtype,
                      mode=mode, cval=im_cval) +
            1j * func(array.real, kernel.imag, output=real_dtype,
                      mode=mode, cval=re_cval)
        )
    elif complex_array:
        re_cval = xp.real(cval)
        re_cval = re_cval.item() if isinstance(re_cval, xp.ndarray) else re_cval
        im_cval = xp.imag(cval)
        im_cval = im_cval.item() if isinstance(im_cval, xp.ndarray) else im_cval

        output = (
            func(array.real, kernel, output=real_dtype, mode=mode,
                 cval=re_cval) +
            1j * func(array.imag, kernel, output=real_dtype, mode=mode,
                      cval=im_cval)
        )
    elif complex_kernel:
        # real array so cval is real too
        output = (
            func(array, kernel.real, output=real_dtype, mode=mode, cval=cval) +
            1j * func(array, kernel.imag, output=real_dtype, mode=mode,
                      cval=cval)
        )
    return output


def _cases_axes_tuple_length_mismatch():
    # Generate combinations of filter function, valid kwargs, and
    # keyword-value pairs for which the value will become with mismatched
    # (invalid) size
    filter_func = ndimage.gaussian_filter
    kwargs = dict(radius=3, mode='constant', sigma=1.0, order=0)
    for key, val in kwargs.items():
        yield filter_func, kwargs, key, val

    filter_funcs = [ndimage.uniform_filter, ndimage.minimum_filter,
                    ndimage.maximum_filter]
    kwargs = dict(size=3, mode='constant', origin=0)
    for filter_func in filter_funcs:
        for key, val in kwargs.items():
            yield filter_func, kwargs, key, val


class TestNdimageFilters:

    def _validate_complex(self, xp, array, kernel, type2, mode='reflect', cval=0):
        # utility for validating complex-valued correlations
        real_dtype = xp.asarray([], dtype=type2).real.dtype
        expected = _complex_correlate(
            xp, array, kernel, real_dtype, convolve=False, mode=mode, cval=cval
        )

        if array.ndim == 1:
            correlate = functools.partial(ndimage.correlate1d, axis=-1,
                                          mode=mode, cval=cval)
            convolve = functools.partial(ndimage.convolve1d, axis=-1,
                                         mode=mode, cval=cval)
        else:
            correlate = functools.partial(ndimage.correlate, mode=mode,
                                          cval=cval)
            convolve = functools.partial(ndimage.convolve, mode=mode,
                                          cval=cval)

        # test correlate output dtype
        output = correlate(array, kernel, output=type2)
        assert_array_almost_equal(expected, output)
        assert output.dtype.type == type2

        # test correlate with pre-allocated output
        output = xp.zeros_like(array, dtype=type2)
        correlate(array, kernel, output=output)
        assert_array_almost_equal(expected, output)

        # test convolve output dtype
        output = convolve(array, kernel, output=type2)
        expected = _complex_correlate(
            xp, array, kernel, real_dtype, convolve=True, mode=mode, cval=cval,
        )
        assert_array_almost_equal(expected, output)
        assert output.dtype.type == type2

        # convolve with pre-allocated output
        convolve(array, kernel, output=output)
        assert_array_almost_equal(expected, output)
        assert output.dtype.type == type2

        # warns if the output is not a complex dtype
        with pytest.warns(UserWarning,
                          match="promoting specified output dtype to complex"):
            correlate(array, kernel, output=real_dtype)

        with pytest.warns(UserWarning,
                          match="promoting specified output dtype to complex"):
            convolve(array, kernel, output=real_dtype)

        # raises if output array is provided, but is not complex-valued
        output_real = xp.zeros_like(array, dtype=real_dtype)
        with assert_raises(RuntimeError):
            correlate(array, kernel, output=output_real)

        with assert_raises(RuntimeError):
            convolve(array, kernel, output=output_real)

    def test_correlate01(self, xp):
        array = xp.array([1, 2])
        weights = xp.array([2])
        expected = xp.array([2, 4])

        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, expected)

    def test_correlate01_overlap(self, xp):
        if is_cupy(xp):
            pytest.xfail("Differs by a factor of two?")

        array = xp.arange(256).reshape(16, 16)
        weights = xp.array([2])
        expected = 2 * array

        ndimage.correlate1d(array, weights, output=array)
        assert_array_almost_equal(array, expected)

    def test_correlate02(self, xp):
        array = xp.array([1, 2, 3])
        kernel = xp.array([1])

        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(array, output)

        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(array, output)

        output = ndimage.correlate1d(array, kernel)
        assert_array_almost_equal(array, output)

        output = ndimage.convolve1d(array, kernel)
        assert_array_almost_equal(array, output)

    def test_correlate03(self, xp):
        array = xp.array([1])
        weights = xp.array([1, 1])
        expected = xp.asarray([2])

        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, expected)

    def test_correlate04(self, xp):
        array = xp.asarray([1, 2])
        tcor = xp.asarray([2, 3])
        tcov = xp.asarray([3, 4])
        weights = xp.asarray([1, 1])
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, tcov)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, tcov)

    def test_correlate05(self, xp):
        array = xp.asarray([1, 2, 3])
        tcor = xp.asarray([2, 3, 5])
        tcov = xp.asarray([3, 5, 6])
        kernel = xp.asarray([1, 1])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(tcor, output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(tcov, output)
        output = ndimage.correlate1d(array, kernel)
        assert_array_almost_equal(tcor, output)
        output = ndimage.convolve1d(array, kernel)
        assert_array_almost_equal(tcov, output)

    def test_correlate06(self, xp):
        array = xp.asarray([1, 2, 3])
        tcor = xp.asarray([9, 14, 17])
        tcov = xp.asarray([7, 10, 15])
        weights = xp.asarray([1, 2, 3])
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, tcov)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, tcov)

    def test_correlate07(self, xp):
        array = xp.asarray([1, 2, 3])
        expected = xp.asarray([5, 8, 11])
        weights = xp.asarray([1, 2, 1])
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, expected)

    def test_correlate08(self, xp):
        array = xp.asarray([1, 2, 3])
        tcor = xp.asarray([1, 2, 5])
        tcov = xp.asarray([3, 6, 7])
        weights = xp.asarray([1, 2, -1])
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, tcov)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, tcov)

    def test_correlate09(self, xp):
        array = xp.asarray([])
        kernel = xp.asarray([1, 1])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.correlate1d(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.convolve1d(array, kernel)
        assert_array_almost_equal(array, output)

    def test_correlate10(self, xp):
        array = xp.asarray([[]])
        kernel = xp.array([[1, 1]])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(array, output)

    def test_correlate11(self, xp):
        array = xp.array([[1, 2, 3],
                          [4, 5, 6]])
        kernel = xp.array([[1, 1],
                           [1, 1]])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(xp.asarray([[4, 6, 10], [10, 12, 16]]), output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(xp.asarray([[12, 16, 18], [18, 22, 24]]), output)

    def test_correlate12(self, xp):
        array = xp.asarray([[1, 2, 3],
                          [4, 5, 6]])
        kernel = xp.asarray([[1, 0],
                           [0, 1]])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(xp.asarray([[2, 3, 5], [5, 6, 8]]), output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(xp.asarray([[6, 8, 9], [9, 11, 12]]), output)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_kernel', types)
    def test_correlate13(self, dtype_array, dtype_kernel, xp):
        kernel = xp.asarray([[1, 0],
                              [0, 1]])
        array = xp.asarray([[1, 2, 3],
                          [4, 5, 6]], dtype_array)
        output = ndimage.correlate(array, kernel, output=dtype_kernel)
        assert_array_almost_equal(xp.asarray([[2, 3, 5], [5, 6, 8]]), output)
        assert output.dtype.type == dtype_kernel

        output = ndimage.convolve(array, kernel,
                                  output=dtype_kernel)
        assert_array_almost_equal(xp.asarray([[6, 8, 9], [9, 11, 12]]), output)
        assert output.dtype.type == dtype_kernel

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate14(self, dtype_array, dtype_output, xp):
        kernel = xp.asarray([[1, 0],
                           [0, 1]])
        array = xp.asarray([[1, 2, 3],
                          [4, 5, 6]], dtype_array)
        output = xp.zeros(array.shape, dtype_output)
        ndimage.correlate(array, kernel, output=output)
        assert_array_almost_equal(xp.asarray([[2, 3, 5], [5, 6, 8]]), output)
        assert output.dtype.type == dtype_output

        ndimage.convolve(array, kernel, output=output)
        assert_array_almost_equal(xp.asarray([[6, 8, 9], [9, 11, 12]]), output)
        assert output.dtype.type == dtype_output

    @pytest.mark.parametrize('dtype_array', types)
    def test_correlate15(self, dtype_array, xp):
        kernel = xp.asarray([[1, 0],
                           [0, 1]])
        array = xp.asarray([[1, 2, 3],
                          [4, 5, 6]], dtype_array)
        output = ndimage.correlate(array, kernel, output=xp.float32)
        assert_array_almost_equal(xp.asarray([[2, 3, 5], [5, 6, 8]]), output)
        assert output.dtype.type == xp.float32

        output = ndimage.convolve(array, kernel, output=xp.float32)
        assert_array_almost_equal(xp.asarray([[6, 8, 9], [9, 11, 12]]), output)
        assert output.dtype.type == xp.float32

    @pytest.mark.parametrize('dtype_array', types)
    def test_correlate16(self, dtype_array, xp):
        kernel = xp.asarray([[0.5, 0],
                           [0, 0.5]])
        array = xp.asarray([[1, 2, 3], [4, 5, 6]], dtype_array)
        output = ndimage.correlate(array, kernel, output=xp.float32)
        assert_array_almost_equal(xp.asarray([[1, 1.5, 2.5], [2.5, 3, 4]]), output)
        assert output.dtype.type == xp.float32

        output = ndimage.convolve(array, kernel, output=xp.float32)
        assert_array_almost_equal(xp.asarray([[3, 4, 4.5], [4.5, 5.5, 6]]), output)
        assert output.dtype.type == xp.float32

    def test_correlate17(self, xp):
        array = xp.asarray([1, 2, 3])
        tcor = xp.asarray([3, 5, 6])
        tcov = xp.asarray([2, 3, 5])
        kernel = xp.asarray([1, 1])
        output = ndimage.correlate(array, kernel, origin=-1)
        assert_array_almost_equal(tcor, output)
        output = ndimage.convolve(array, kernel, origin=-1)
        assert_array_almost_equal(tcov, output)
        output = ndimage.correlate1d(array, kernel, origin=-1)
        assert_array_almost_equal(tcor, output)
        output = ndimage.convolve1d(array, kernel, origin=-1)
        assert_array_almost_equal(tcov, output)

    @pytest.mark.parametrize('dtype_array', types)
    def test_correlate18(self, dtype_array, xp):
        kernel = xp.asarray([[1, 0],
                           [0, 1]])
        array = xp.asarray([[1, 2, 3],
                          [4, 5, 6]], dtype_array)
        output = ndimage.correlate(array, kernel,
                                   output=xp.float32,
                                   mode='nearest', origin=-1)
        assert_array_almost_equal(xp.asarray([[6, 8, 9], [9, 11, 12]]), output)
        assert output.dtype.type == xp.float32

        output = ndimage.convolve(array, kernel,
                                  output=xp.float32,
                                  mode='nearest', origin=-1)
        assert_array_almost_equal(xp.asarray([[2, 3, 5], [5, 6, 8]]), output)
        assert output.dtype.type == xp.float32

    def test_correlate_mode_sequence(self, xp):
        kernel = xp.ones((2, 2))
        array = xp.ones((3, 3), float)
        with assert_raises(RuntimeError):
            ndimage.correlate(array, kernel, mode=['nearest', 'reflect'])
        with assert_raises(RuntimeError):
            ndimage.convolve(array, kernel, mode=['nearest', 'reflect'])

    @pytest.mark.parametrize('dtype_array', types)
    def test_correlate19(self, dtype_array, xp):
        kernel = xp.asarray([[1, 0],
                           [0, 1]])
        array = xp.asarray([[1, 2, 3],
                          [4, 5, 6]], dtype_array)
        output = ndimage.correlate(array, kernel,
                                   output=xp.float32,
                                   mode='nearest', origin=[-1, 0])
        assert_array_almost_equal(xp.asarray([[5, 6, 8], [8, 9, 11]]), output)
        assert output.dtype.type == xp.float32

        output = ndimage.convolve(array, kernel,
                                  output=xp.float32,
                                  mode='nearest', origin=[-1, 0])
        assert_array_almost_equal(xp.asarray([[3, 5, 6], [6, 8, 9]]), output)
        assert output.dtype.type == xp.float32

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate20(self, dtype_array, dtype_output, xp):
        weights = xp.asarray([1, 2, 1])
        expected = xp.asarray([[5, 10, 15], [7, 14, 21]])
        array = xp.asarray([[1, 2, 3],
                          [2, 4, 6]], dtype_array)
        output = xp.zeros((2, 3), dtype_output)
        ndimage.correlate1d(array, weights, axis=0, output=output)
        assert_array_almost_equal(output, expected)
        ndimage.convolve1d(array, weights, axis=0, output=output)
        assert_array_almost_equal(output, expected)

    def test_correlate21(self, xp):
        array = xp.asarray([[1, 2, 3],
                          [2, 4, 6]])
        expected = xp.asarray([[5, 10, 15], [7, 14, 21]])
        weights = xp.asarray([1, 2, 1])
        output = ndimage.correlate1d(array, weights, axis=0)
        assert_array_almost_equal(output, expected)
        output = ndimage.convolve1d(array, weights, axis=0)
        assert_array_almost_equal(output, expected)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate22(self, dtype_array, dtype_output, xp):
        weights = xp.asarray([1, 2, 1])
        expected = xp.asarray([[6, 12, 18], [6, 12, 18]])
        array = xp.asarray([[1, 2, 3],
                          [2, 4, 6]], dtype_array)
        output = xp.zeros((2, 3), dtype_output)
        ndimage.correlate1d(array, weights, axis=0,
                            mode='wrap', output=output)
        assert_array_almost_equal(output, expected)
        ndimage.convolve1d(array, weights, axis=0,
                           mode='wrap', output=output)
        assert_array_almost_equal(output, expected)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate23(self, dtype_array, dtype_output, xp):
        weights = xp.asarray([1, 2, 1])
        expected = xp.asarray([[5, 10, 15], [7, 14, 21]])
        array = xp.asarray([[1, 2, 3],
                          [2, 4, 6]], dtype_array)
        output = xp.zeros((2, 3), dtype_output)
        ndimage.correlate1d(array, weights, axis=0,
                            mode='nearest', output=output)
        assert_array_almost_equal(output, expected)
        ndimage.convolve1d(array, weights, axis=0,
                           mode='nearest', output=output)
        assert_array_almost_equal(output, expected)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate24(self, dtype_array, dtype_output, xp):
        weights = xp.asarray([1, 2, 1])
        tcor = xp.asarray([[7, 14, 21], [8, 16, 24]])
        tcov = xp.asarray([[4, 8, 12], [5, 10, 15]])
        array = xp.asarray([[1, 2, 3],
                          [2, 4, 6]], dtype_array)
        output = xp.zeros((2, 3), dtype_output)
        ndimage.correlate1d(array, weights, axis=0,
                            mode='nearest', output=output, origin=-1)
        assert_array_almost_equal(output, tcor)
        ndimage.convolve1d(array, weights, axis=0,
                           mode='nearest', output=output, origin=-1)
        assert_array_almost_equal(output, tcov)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate25(self, dtype_array, dtype_output, xp):
        weights = xp.asarray([1, 2, 1])
        tcor = xp.asarray([[4, 8, 12], [5, 10, 15]])
        tcov = xp.asarray([[7, 14, 21], [8, 16, 24]])
        array = xp.asarray([[1, 2, 3],
                          [2, 4, 6]], dtype_array)
        output = xp.zeros((2, 3), dtype_output)
        ndimage.correlate1d(array, weights, axis=0,
                            mode='nearest', output=output, origin=1)
        assert_array_almost_equal(output, tcor)
        ndimage.convolve1d(array, weights, axis=0,
                           mode='nearest', output=output, origin=1)
        assert_array_almost_equal(output, tcov)

    def test_correlate26(self, xp):
        # test fix for gh-11661 (mirror extension of a length 1 signal)
        y = ndimage.convolve1d(xp.ones(1), xp.ones(5), mode='mirror')
        xp_assert_equal(y, xp.asarray([5.]))

        y = ndimage.correlate1d(xp.ones(1), xp.ones(5), mode='mirror')
        xp_assert_equal(y, xp.asarray([5.]))

    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate_complex_kernel(self, dtype_input, dtype_kernel,
                                      dtype_output, xp):
        kernel = xp.asarray([[1, 0],
                           [0, 1 + 1j]], dtype_kernel)
        array = xp.asarray([[1, 2, 3],
                          [4, 5, 6]], dtype_input)
        self._validate_complex(xp, array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    @pytest.mark.parametrize('mode', ['grid-constant', 'constant'])
    def test_correlate_complex_kernel_cval(self, dtype_input, dtype_kernel,
                                           dtype_output, mode, xp):
        if is_cupy(xp) and mode == 'grid-constant':
            pytest.xfail('https://github.com/cupy/cupy/issues/8404')

        # test use of non-zero cval with complex inputs
        # also verifies that mode 'grid-constant' does not segfault
        kernel = xp.asarray([[1, 0],
                           [0, 1 + 1j]], dtype_kernel)
        array = xp.asarray([[1, 2, 3],
                          [4, 5, 6]], dtype_input)
        self._validate_complex(xp, array, kernel, dtype_output, mode=mode,
                               cval=5.0)

    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    def test_correlate_complex_kernel_invalid_cval(self, dtype_input,
                                                   dtype_kernel, xp):
        if is_cupy(xp):
            pytest.xfail("https://github.com/cupy/cupy/issues/8405")

        # cannot give complex cval with a real image
        kernel = xp.asarray([[1, 0],
                           [0, 1 + 1j]], dtype_kernel)
        array = xp.asarray([[1, 2, 3],
                          [4, 5, 6]], dtype_input)
        for func in [ndimage.convolve, ndimage.correlate, ndimage.convolve1d,
                     ndimage.correlate1d]:
            with pytest.raises((ValueError, TypeError)):
                func(array, kernel, mode='constant', cval=5.0 + 1.0j,
                     output=xp.complex64)

    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_kernel(self, dtype_input, dtype_kernel,
                                        dtype_output, xp):
        kernel = xp.asarray([1, 1 + 1j], dtype_kernel)
        array = xp.asarray([1, 2, 3, 4, 5, 6], dtype_input)
        self._validate_complex(xp, array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_kernel_cval(self, dtype_input, dtype_kernel,
                                             dtype_output, xp):
        kernel = xp.asarray([1, 1 + 1j], dtype_kernel)
        array = xp.asarray([1, 2, 3, 4, 5, 6], dtype_input)
        self._validate_complex(xp, array, kernel, dtype_output, mode='constant',
                               cval=5.0)

    @pytest.mark.parametrize('dtype_kernel', types)
    @pytest.mark.parametrize('dtype_input', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate_complex_input(self, dtype_input, dtype_kernel,
                                     dtype_output, xp):
        kernel = xp.asarray([[1, 0],
                           [0, 1]], dtype_kernel)
        array = xp.asarray([[1, 2j, 3],
                          [1 + 4j, 5, 6j]], dtype_input)
        self._validate_complex(xp, array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype_kernel', types)
    @pytest.mark.parametrize('dtype_input', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_input(self, dtype_input, dtype_kernel,
                                       dtype_output, xp):
        kernel = xp.asarray([1, 0, 1], dtype_kernel)
        array = xp.asarray([1, 2j, 3, 1 + 4j, 5, 6j], dtype_input)
        self._validate_complex(xp, array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype_kernel', types)
    @pytest.mark.parametrize('dtype_input', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_input_cval(self, dtype_input, dtype_kernel,
                                            dtype_output, xp):
        if is_cupy(xp):
            pytest.xfail("https://github.com/cupy/cupy/issues/8405")

        kernel = xp.asarray([1, 0, 1], dtype_kernel)
        array = xp.asarray([1, 2j, 3, 1 + 4j, 5, 6j], dtype_input)
        self._validate_complex(xp, array, kernel, dtype_output, mode='constant',
                               cval=5 - 3j)

    @pytest.mark.parametrize('dtype', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate_complex_input_and_kernel(self, dtype, dtype_output, xp):
        kernel = xp.asarray([[1, 0],
                           [0, 1 + 1j]], dtype)
        array = xp.asarray([[1, 2j, 3],
                          [1 + 4j, 5, 6j]], dtype)
        self._validate_complex(xp, array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate_complex_input_and_kernel_cval(self, dtype,
                                                     dtype_output, xp):
        if is_cupy(xp):
            pytest.xfail("https://github.com/cupy/cupy/issues/8405")

        kernel = xp.asarray([[1, 0],
                           [0, 1 + 1j]], dtype)
        array = xp.asarray([[1, 2, 3],
                          [4, 5, 6]], dtype)
        self._validate_complex(xp, array, kernel, dtype_output, mode='constant',
                               cval=5.0 + 2.0j)

    @pytest.mark.parametrize('dtype', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_input_and_kernel(self, dtype, dtype_output, xp):
        kernel = xp.asarray([1, 1 + 1j], dtype)
        array = xp.asarray([1, 2j, 3, 1 + 4j, 5, 6j], dtype)
        self._validate_complex(xp, array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_input_and_kernel_cval(self, dtype,
                                                       dtype_output, xp):
        if is_cupy(xp):
            pytest.xfail("https://github.com/cupy/cupy/issues/8405")

        kernel = xp.asarray([1, 1 + 1j], dtype)
        array = xp.asarray([1, 2j, 3, 1 + 4j, 5, 6j], dtype)
        self._validate_complex(xp, array, kernel, dtype_output, mode='constant',
                               cval=5.0 + 2.0j)

    def test_gauss01(self, xp):
        input = xp.asarray([[1, 2, 3],
                          [2, 4, 6]], xp.float32)
        output = ndimage.gaussian_filter(input, 0)
        assert_array_almost_equal(output, input)

    def test_gauss02(self, xp):
        input = xp.asarray([[1, 2, 3],
                          [2, 4, 6]], xp.float32)
        output = ndimage.gaussian_filter(input, 1.0)
        assert input.dtype == output.dtype
        assert input.shape == output.shape

    def test_gauss03(self, xp):
        if is_cupy(xp):
            pytest.xfail("https://github.com/cupy/cupy/issues/8403")

        # single precision data
        input = xp.arange(100 * 100).astype(xp.float32)
        input.shape = (100, 100)
        output = ndimage.gaussian_filter(input, [1.0, 1.0])

        assert input.dtype == output.dtype
        assert input.shape == output.shape

        # input.sum() is 49995000.0.  With single precision floats, we can't
        # expect more than 8 digits of accuracy, so use decimal=0 in this test.
        assert_almost_equal(output.sum(dtype='d'), input.sum(dtype='d'),
                            decimal=0)
        assert sumsq(input, output) > 1.0

    def test_gauss04(self, xp):
        input = xp.arange(100 * 100).astype(xp.float32)
        input.shape = (100, 100)
        otype = xp.float64
        output = ndimage.gaussian_filter(input, [1.0, 1.0], output=otype)
        assert output.dtype.type == xp.float64
        assert input.shape == output.shape
        assert sumsq(input, output) > 1.0

    def test_gauss05(self, xp):
        input = xp.arange(100 * 100).astype(xp.float32)
        input.shape = (100, 100)
        otype = xp.float64
        output = ndimage.gaussian_filter(input, [1.0, 1.0],
                                         order=1, output=otype)
        assert output.dtype.type == xp.float64
        assert input.shape == output.shape
        assert sumsq(input, output) > 1.0

    def test_gauss06(self, xp):
        input = xp.arange(100 * 100).astype(xp.float32)
        input.shape = (100, 100)
        otype = xp.float64
        output1 = ndimage.gaussian_filter(input, [1.0, 1.0], output=otype)
        output2 = ndimage.gaussian_filter(input, 1.0, output=otype)
        assert_array_almost_equal(output1, output2)

    def test_gauss_memory_overlap(self, xp):
        input = xp.arange(100 * 100).astype(xp.float32)
        input.shape = (100, 100)
        output1 = ndimage.gaussian_filter(input, 1.0)
        ndimage.gaussian_filter(input, 1.0, output=input)
        assert_array_almost_equal(output1, input)

    @pytest.mark.parametrize(('filter_func', 'extra_args', 'size0', 'size'),
                             [(ndimage.gaussian_filter, (), 0, 1.0),
                              (ndimage.uniform_filter, (), 1, 3),
                              (ndimage.minimum_filter, (), 1, 3),
                              (ndimage.maximum_filter, (), 1, 3),
                              (ndimage.median_filter, (), 1, 3),
                              (ndimage.rank_filter, (1,), 1, 3),
                              (ndimage.percentile_filter, (40,), 1, 3)])
    @pytest.mark.parametrize(
        'axes',
        tuple(itertools.combinations(range(-3, 3), 1))
        + tuple(itertools.combinations(range(-3, 3), 2))
        + ((0, 1, 2),))
    def test_filter_axes(self, filter_func, extra_args, size0, size, axes, xp):
        if is_cupy(xp):
            pytest.xfail("https://github.com/cupy/cupy/pull/8339")

        # Note: `size` is called `sigma` in `gaussian_filter`
        array = xp.arange(6 * 8 * 12, dtype=xp.float64).reshape(6, 8, 12)

        if len(set(ax % array.ndim for ax in axes)) != len(axes):
            # parametrized cases with duplicate axes raise an error
            with pytest.raises(ValueError, match="axes must be unique"):
                filter_func(array, *extra_args, size, axes=axes)
            return
        output = filter_func(array, *extra_args, size, axes=axes)

        # result should be equivalent to sigma=0.0/size=1 on unfiltered axes
        axes = xp.asarray(axes)
        all_sizes = (size if ax in (axes % array.ndim) else size0
                     for ax in range(array.ndim))
        expected = filter_func(array, *extra_args, all_sizes)
        xp_assert_close(output, expected)

    kwargs_gauss = dict(radius=[4, 2, 3], order=[0, 1, 2],
                        mode=['reflect', 'nearest', 'constant'])
    kwargs_other = dict(origin=(-1, 0, 1),
                        mode=['reflect', 'nearest', 'constant'])
    kwargs_rank = dict(origin=(-1, 0, 1))

    @pytest.mark.parametrize("filter_func, size0, size, kwargs",
                             [(ndimage.gaussian_filter, 0, 1.0, kwargs_gauss),
                              (ndimage.uniform_filter, 1, 3, kwargs_other),
                              (ndimage.maximum_filter, 1, 3, kwargs_other),
                              (ndimage.minimum_filter, 1, 3, kwargs_other),
                              (ndimage.median_filter, 1, 3, kwargs_rank),
                              (ndimage.rank_filter, 1, 3, kwargs_rank),
                              (ndimage.percentile_filter, 1, 3, kwargs_rank)])
    @pytest.mark.parametrize('axes', itertools.combinations(range(-3, 3), 2))
    def test_filter_axes_kwargs(self, filter_func, size0, size, kwargs, axes, xp):
        array = xp.arange(6 * 8 * 12, dtype=xp.float64).reshape(6, 8, 12)

        if is_cupy(xp):
            pytest.xfail("https://github.com/cupy/cupy/pull/8339")

        kwargs = {key: xp.asarray(val)  for key, val in kwargs.items()}
        axes = xp.asarray(axes)
        n_axes = axes.size

        if filter_func == ndimage.rank_filter:
            args = (2,)  # (rank,)
        elif filter_func == ndimage.percentile_filter:
            args = (30,)  # (percentile,)
        else:
            args = ()

        # form kwargs that specify only the axes in `axes`
        reduced_kwargs = {key: val[axes] for key, val in kwargs.items()}
        if len(set(axes % array.ndim)) != len(axes):
            # parametrized cases with duplicate axes raise an error
            with pytest.raises(ValueError, match="axes must be unique"):
                filter_func(array, *args, [size]*n_axes, axes=axes,
                            **reduced_kwargs)
            return

        output = filter_func(array, *args, [size]*n_axes, axes=axes,
                             **reduced_kwargs)

        # result should be equivalent to sigma=0.0/size=1 on unfiltered axes
        size_3d = xp.full(array.ndim, fill_value=size0)
        size_3d[axes] = size
        if 'origin' in kwargs:
            # origin should be zero on the axis that has size 0
            origin = xp.asarray([0, 0, 0])
            origin[axes] = reduced_kwargs['origin']
            kwargs['origin'] = origin
        expected = filter_func(array, *args, size_3d, **kwargs)
        xp_assert_close(output, expected)

    @pytest.mark.parametrize("filter_func, kwargs",
                             [(ndimage.minimum_filter, {}),
                              (ndimage.maximum_filter, {}),
                              (ndimage.median_filter, {}),
                              (ndimage.rank_filter, {"rank": 1}),
                              (ndimage.percentile_filter, {"percentile": 30})])
    def test_filter_weights_subset_axes_origins(self, filter_func, kwargs, xp):
        if is_cupy(xp):
            pytest.xfail("https://github.com/cupy/cupy/pull/8339")

        axes = (-2, -1)
        origins = (0, 1)
        array = xp.arange(6 * 8 * 12, dtype=xp.float64).reshape(6, 8, 12)
        axes = xp.asarray(axes)

        # weights with ndim matching len(axes)
        footprint = xp.ones((3, 5), dtype=bool)
        footprint[0, 1] = 0  # make non-separable

        output = filter_func(
            array, footprint=footprint, axes=axes, origin=origins, **kwargs)

        output0 = filter_func(
            array, footprint=footprint, axes=axes, origin=0, **kwargs)

        # output has origin shift on last axis relative to output0, so
        # expect shifted arrays to be equal.
        xp_assert_equal(output[:, :, 1:], output0[:, :, :-1])

    @pytest.mark.parametrize(
        'filter_func, args',
        [(ndimage.gaussian_filter, (1.0,)),      # args = (sigma,)
         (ndimage.uniform_filter, (3,)),         # args = (size,)
         (ndimage.minimum_filter, (3,)),         # args = (size,)
         (ndimage.maximum_filter, (3,)),         # args = (size,)
         (ndimage.median_filter, (3,)),          # args = (size,)
         (ndimage.rank_filter, (2, 3)),          # args = (rank, size)
         (ndimage.percentile_filter, (30, 3))])  # args = (percentile, size)
    @pytest.mark.parametrize(
        'axes', [(1.5,), (0, 1, 2, 3), (3,), (-4,)]
    )
    def test_filter_invalid_axes(self, filter_func, args, axes, xp):
        if is_cupy(xp):
            pytest.xfail("https://github.com/cupy/cupy/pull/8339")

        array = xp.arange(6 * 8 * 12, dtype=xp.float64).reshape(6, 8, 12)
        if any(isinstance(ax, float) for ax in axes):
            error_class = TypeError
            match = "cannot be interpreted as an integer"
        else:
            error_class = ValueError
            match = "out of range"
        with pytest.raises(error_class, match=match):
            filter_func(array, *args, axes=axes)

    @pytest.mark.parametrize(
        'filter_func, kwargs',
        [(ndimage.minimum_filter, {}),
         (ndimage.maximum_filter, {}),
         (ndimage.median_filter, {}),
         (ndimage.rank_filter, dict(rank=3)),
         (ndimage.percentile_filter, dict(percentile=30))])
    @pytest.mark.parametrize(
        'axes', [(0, ), (1, 2), (0, 1, 2)]
    )
    @pytest.mark.parametrize('separable_footprint', [False, True])
    def test_filter_invalid_footprint_ndim(self, filter_func, kwargs, axes,
                                           separable_footprint, xp):
        if is_cupy(xp):
            pytest.xfail("https://github.com/cupy/cupy/pull/8339")

        array = xp.arange(6 * 8 * 12, dtype=xp.float64).reshape(6, 8, 12)
        # create a footprint with one too many dimensions
        footprint = xp.ones((3,) * (len(axes) + 1))
        if not separable_footprint:
            footprint[(0,) * footprint.ndim] = 0
        if (filter_func in [ndimage.minimum_filter, ndimage.maximum_filter]
            and separable_footprint):
            match = "sequence argument must have length equal to input rank"
        else:
            match = "footprint array has incorrect shape"
        with pytest.raises(RuntimeError, match=match):
            filter_func(array, **kwargs, footprint=footprint, axes=axes)

    @pytest.mark.parametrize('n_mismatch', [1, 3])
    @pytest.mark.parametrize('filter_func, kwargs, key, val',
                             _cases_axes_tuple_length_mismatch())
    def test_filter_tuple_length_mismatch(self, n_mismatch, filter_func,
                                          kwargs, key, val, xp):
        if is_cupy(xp):
            pytest.xfail("https://github.com/cupy/cupy/pull/8339")

        # Test for the intended RuntimeError when a kwargs has an invalid size
        array = xp.arange(6 * 8 * 12, dtype=xp.float64).reshape(6, 8, 12)
        kwargs = dict(**kwargs, axes=(0, 1))
        kwargs[key] = (val,) * n_mismatch
        err_msg = "sequence argument must have length equal to input rank"
        with pytest.raises(RuntimeError, match=err_msg):
            filter_func(array, **kwargs)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_prewitt01(self, dtype, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        t = ndimage.correlate1d(array, xp.asarray([-1.0, 0.0, 1.0]), 0)
        t = ndimage.correlate1d(t, xp.asarray([1.0, 1.0, 1.0]), 1)
        output = ndimage.prewitt(array, 0)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_prewitt02(self, dtype, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        t = ndimage.correlate1d(array, xp.asarray([-1.0, 0.0, 1.0]), 0)
        t = ndimage.correlate1d(t, xp.asarray([1.0, 1.0, 1.0]), 1)
        output = xp.zeros(array.shape, dtype)
        ndimage.prewitt(array, 0, output)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_prewitt03(self, dtype, xp):
        if is_cupy(xp) and dtype in [xp.uint32, xp.uint64]:
            pytest.xfail("uint UB? XXX")

        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        t = ndimage.correlate1d(array, xp.asarray([-1.0, 0.0, 1.0]), 1)
        t = ndimage.correlate1d(t, xp.asarray([1.0, 1.0, 1.0]), 0)
        output = ndimage.prewitt(array, 1)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_prewitt04(self, dtype, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        t = ndimage.prewitt(array, -1)
        output = ndimage.prewitt(array, 1)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_sobel01(self, dtype, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        t = ndimage.correlate1d(array, xp.asarray([-1.0, 0.0, 1.0]), 0)
        t = ndimage.correlate1d(t, xp.asarray([1.0, 2.0, 1.0]), 1)
        output = ndimage.sobel(array, 0)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_sobel02(self, dtype, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        t = ndimage.correlate1d(array, xp.asarray([-1.0, 0.0, 1.0]), 0)
        t = ndimage.correlate1d(t, xp.asarray([1.0, 2.0, 1.0]), 1)
        output = xp.zeros(array.shape, dtype)
        ndimage.sobel(array, 0, output)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_sobel03(self, dtype, xp):
        if is_cupy(xp) and dtype in [xp.uint32, xp.uint64]:
            pytest.xfail("uint UB? XXX")

        array = xp.asarray([[3, 2, 5, 1, 4],
                            [5, 8, 3, 7, 1],
                            [5, 6, 9, 3, 5]], dtype=dtype)
        t = ndimage.correlate1d(array, xp.asarray([-1.0, 0.0, 1.0]), 1)
        t = ndimage.correlate1d(t, xp.asarray([1.0, 2.0, 1.0]), 0)
        output = xp.zeros(array.shape, dtype=dtype)
        output = ndimage.sobel(array, 1)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_sobel04(self, dtype, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        t = ndimage.sobel(array, -1)
        output = ndimage.sobel(array, 1)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype',
                             [np.int32, np.float32, np.float64,
                              np.complex64, np.complex128])
    def test_laplace01(self, dtype, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.correlate1d(array, xp.asarray([1, -2, 1]), 0)
        tmp2 = ndimage.correlate1d(array, xp.asarray([1, -2, 1]), 1)
        output = ndimage.laplace(array)
        assert_array_almost_equal(tmp1 + tmp2, output)

    @pytest.mark.parametrize('dtype',
                             [np.int32, np.float32, np.float64,
                              np.complex64, np.complex128])
    def test_laplace02(self, dtype, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.correlate1d(array, xp.asarray([1, -2, 1]), 0)
        tmp2 = ndimage.correlate1d(array, xp.asarray([1, -2, 1]), 1)
        output = xp.zeros(array.shape, dtype)
        ndimage.laplace(array, output=output)
        assert_array_almost_equal(tmp1 + tmp2, output)

    @pytest.mark.parametrize('dtype',
                             [np.int32, np.float32, np.float64,
                              np.complex64, np.complex128])
    def test_gaussian_laplace01(self, dtype, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.gaussian_filter(array, 1.0, [2, 0])
        tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 2])
        output = ndimage.gaussian_laplace(array, 1.0)
        assert_array_almost_equal(tmp1 + tmp2, output)

    @pytest.mark.parametrize('dtype',
                             [np.int32, np.float32, np.float64,
                              np.complex64, np.complex128])
    def test_gaussian_laplace02(self, dtype, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.gaussian_filter(array, 1.0, [2, 0])
        tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 2])
        output = xp.zeros(array.shape, dtype)
        ndimage.gaussian_laplace(array, 1.0, output)
        assert_array_almost_equal(tmp1 + tmp2, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_generic_laplace01(self, dtype, xp):
        def derivative2(input, axis, output, mode, cval, a, b):
            sigma = [a, b / 2.0]
            input = xp.asarray(input)
            order = [0] * input.ndim
            order[axis] = 2
            return ndimage.gaussian_filter(input, sigma, order,
                                           output, mode, cval)
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        output = xp.zeros(array.shape, dtype)
        tmp = ndimage.generic_laplace(array, derivative2,
                                      extra_arguments=(1.0,),
                                      extra_keywords={'b': 2.0})
        ndimage.gaussian_laplace(array, 1.0, output)
        assert_array_almost_equal(tmp, output)

    @pytest.mark.parametrize('dtype',
                             [np.int32, np.float32, np.float64,
                              np.complex64, np.complex128])
    def test_gaussian_gradient_magnitude01(self, dtype, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.gaussian_filter(array, 1.0, [1, 0])
        tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 1])
        output = ndimage.gaussian_gradient_magnitude(array, 1.0)
        expected = tmp1 * tmp1 + tmp2 * tmp2
        expected = xp.sqrt(expected).astype(dtype)
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype',
                             [np.int32, np.float32, np.float64,
                              np.complex64, np.complex128])
    def test_gaussian_gradient_magnitude02(self, dtype, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.gaussian_filter(array, 1.0, [1, 0])
        tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 1])
        output = xp.zeros(array.shape, dtype)
        ndimage.gaussian_gradient_magnitude(array, 1.0, output)
        expected = tmp1 * tmp1 + tmp2 * tmp2
        expected = xp.sqrt(expected).astype(dtype)
        assert_array_almost_equal(expected, output)

    def test_generic_gradient_magnitude01(self, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], xp.float64)

        def derivative(input, axis, output, mode, cval, a, b):
            sigma = [a, b / 2.0]
            input = xp.asarray(input)
            order = [0] * input.ndim
            order[axis] = 1
            return ndimage.gaussian_filter(input, sigma, order,
                                           output, mode, cval)
        tmp1 = ndimage.gaussian_gradient_magnitude(array, 1.0)
        tmp2 = ndimage.generic_gradient_magnitude(
            array, derivative, extra_arguments=(1.0,),
            extra_keywords={'b': 2.0})
        assert_array_almost_equal(tmp1, tmp2)

    def test_uniform01(self, xp):
        array = xp.asarray([2, 4, 6])
        size = 2
        output = ndimage.uniform_filter1d(array, size, origin=-1)
        assert_array_almost_equal(xp.asarray([3, 5, 6]), output)

    def test_uniform01_complex(self, xp):
        array = xp.asarray([2 + 1j, 4 + 2j, 6 + 3j], dtype=xp.complex128)
        size = 2
        output = ndimage.uniform_filter1d(array, size, origin=-1)
        assert_array_almost_equal(xp.asarray([3, 5, 6]), output.real)
        assert_array_almost_equal(xp.asarray([1.5, 2.5, 3]), output.imag)

    def test_uniform02(self, xp):
        array = xp.asarray([1, 2, 3])
        filter_shape = [0]
        output = ndimage.uniform_filter(array, filter_shape)
        assert_array_almost_equal(array, output)

    def test_uniform03(self, xp):
        array = xp.asarray([1, 2, 3])
        filter_shape = [1]
        output = ndimage.uniform_filter(array, filter_shape)
        assert_array_almost_equal(array, output)

    def test_uniform04(self, xp):
        array = xp.asarray([2, 4, 6])
        filter_shape = [2]
        output = ndimage.uniform_filter(array, filter_shape)
        assert_array_almost_equal(xp.asarray([2, 3, 5]), output)

    def test_uniform05(self, xp):
        array = xp.asarray([])
        filter_shape = [1]
        output = ndimage.uniform_filter(array, filter_shape)
        assert_array_almost_equal(xp.asarray([]), output)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_uniform06(self, dtype_array, dtype_output, xp):
        filter_shape = [2, 2]
        array = xp.asarray([[4, 8, 12],
                          [16, 20, 24]], dtype_array)
        output = ndimage.uniform_filter(
            array, filter_shape, output=dtype_output)
        assert_array_almost_equal(xp.asarray([[4, 6, 10], [10, 12, 16]]), output)
        assert output.dtype.type == dtype_output

    @pytest.mark.parametrize('dtype_array', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_uniform06_complex(self, dtype_array, dtype_output, xp):
        filter_shape = [2, 2]
        array = xp.asarray([[4, 8 + 5j, 12],
                          [16, 20, 24]], dtype_array)
        output = ndimage.uniform_filter(
            array, filter_shape, output=dtype_output)
        assert_array_almost_equal(xp.asarray([[4, 6, 10], [10, 12, 16]]), output.real)
        assert output.dtype.type == dtype_output

    def test_minimum_filter01(self, xp):
        array = xp.asarray([1, 2, 3, 4, 5])
        filter_shape = xp.asarray([2])
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal(xp.asarray([1, 1, 2, 3, 4]), output)

    def test_minimum_filter02(self, xp):
        array = xp.asarray([1, 2, 3, 4, 5])
        filter_shape = xp.asarray([3])
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal(xp.asarray([1, 1, 2, 3, 4]), output)

    def test_minimum_filter03(self, xp):
        array = xp.asarray([3, 2, 5, 1, 4])
        filter_shape = xp.asarray([2])
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal(xp.asarray([3, 2, 2, 1, 1]), output)

    def test_minimum_filter04(self, xp):
        array = xp.asarray([3, 2, 5, 1, 4])
        filter_shape = xp.asarray([3])
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal(xp.asarray([2, 2, 1, 1, 1]), output)

    def test_minimum_filter05(self, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        filter_shape = xp.asarray([2, 3])
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal(xp.asarray([[2, 2, 1, 1, 1],
                                              [2, 2, 1, 1, 1],
                                              [5, 3, 3, 1, 1]]), output)

    def test_minimum_filter05_overlap(self, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        filter_shape = xp.asarray([2, 3])
        ndimage.minimum_filter(array, filter_shape, output=array)
        assert_array_almost_equal(xp.asarray([[2, 2, 1, 1, 1],
                                              [2, 2, 1, 1, 1],
                                              [5, 3, 3, 1, 1]]), array)

    def test_minimum_filter06(self, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        footprint = xp.asarray([[1, 1, 1], [1, 1, 1]])
        output = ndimage.minimum_filter(array, footprint=footprint)
        assert_array_almost_equal(xp.asarray([[2, 2, 1, 1, 1],
                                              [2, 2, 1, 1, 1],
                                              [5, 3, 3, 1, 1]]), output)
        # separable footprint should allow mode sequence
        output2 = ndimage.minimum_filter(array, footprint=footprint,
                                         mode=['reflect', 'reflect'])
        assert_array_almost_equal(output2, output)

    def test_minimum_filter07(self, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        footprint = xp.asarray([[1, 0, 1], [1, 1, 0]])
        output = ndimage.minimum_filter(array, footprint=footprint)
        assert_array_almost_equal(xp.asarray([[2, 2, 1, 1, 1],
                                              [2, 3, 1, 3, 1],
                                              [5, 5, 3, 3, 1]]), output)
        with assert_raises(RuntimeError):
            ndimage.minimum_filter(array, footprint=footprint,
                                   mode=['reflect', 'constant'])

    def test_minimum_filter08(self, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        footprint = xp.asarray([[1, 0, 1], [1, 1, 0]])
        output = ndimage.minimum_filter(array, footprint=footprint, origin=-1)
        assert_array_almost_equal(xp.asarray([[3, 1, 3, 1, 1],
                                              [5, 3, 3, 1, 1],
                                              [3, 3, 1, 1, 1]]), output)

    def test_minimum_filter09(self, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        footprint = xp.asarray([[1, 0, 1], [1, 1, 0]])
        output = ndimage.minimum_filter(array, footprint=footprint,
                                        origin=[-1, 0])
        assert_array_almost_equal(xp.asarray([[2, 3, 1, 3, 1],
                                              [5, 5, 3, 3, 1],
                                              [5, 3, 3, 1, 1]]), output)

    def test_maximum_filter01(self, xp):
        array = xp.asarray([1, 2, 3, 4, 5])
        filter_shape = xp.asarray([2])
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal(xp.asarray([1, 2, 3, 4, 5]), output)

    def test_maximum_filter02(self, xp):
        array = xp.asarray([1, 2, 3, 4, 5])
        filter_shape = xp.asarray([3])
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal(xp.asarray([2, 3, 4, 5, 5]), output)

    def test_maximum_filter03(self, xp):
        array = xp.asarray([3, 2, 5, 1, 4])
        filter_shape = xp.asarray([2])
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal(xp.asarray([3, 3, 5, 5, 4]), output)

    def test_maximum_filter04(self, xp):
        array = xp.asarray([3, 2, 5, 1, 4])
        filter_shape = xp.asarray([3])
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal(xp.asarray([3, 5, 5, 5, 4]), output)

    def test_maximum_filter05(self, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        filter_shape = xp.asarray([2, 3])
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal(xp.asarray([[3, 5, 5, 5, 4],
                                              [7, 9, 9, 9, 5],
                                              [8, 9, 9, 9, 7]]), output)

    def test_maximum_filter06(self, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        footprint = xp.asarray([[1, 1, 1], [1, 1, 1]])
        output = ndimage.maximum_filter(array, footprint=footprint)
        assert_array_almost_equal(xp.asarray([[3, 5, 5, 5, 4],
                                              [7, 9, 9, 9, 5],
                                              [8, 9, 9, 9, 7]]), output)
        # separable footprint should allow mode sequence
        output2 = ndimage.maximum_filter(array, footprint=footprint,
                                         mode=['reflect', 'reflect'])
        assert_array_almost_equal(output2, output)

    def test_maximum_filter07(self, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        footprint = xp.asarray([[1, 0, 1], [1, 1, 0]])
        output = ndimage.maximum_filter(array, footprint=footprint)
        assert_array_almost_equal(xp.asarray([[3, 5, 5, 5, 4],
                                              [7, 7, 9, 9, 5],
                                              [7, 9, 8, 9, 7]]), output)
        # non-separable footprint should not allow mode sequence
        with assert_raises(RuntimeError):
            ndimage.maximum_filter(array, footprint=footprint,
                                   mode=['reflect', 'reflect'])

    def test_maximum_filter08(self, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        footprint = xp.asarray([[1, 0, 1], [1, 1, 0]])
        output = ndimage.maximum_filter(array, footprint=footprint, origin=-1)
        assert_array_almost_equal(xp.asarray([[7, 9, 9, 5, 5],
                                              [9, 8, 9, 7, 5],
                                              [8, 8, 7, 7, 7]]), output)

    def test_maximum_filter09(self, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        footprint = xp.asarray([[1, 0, 1], [1, 1, 0]])
        output = ndimage.maximum_filter(array, footprint=footprint,
                                        origin=[-1, 0])
        assert_array_almost_equal(xp.asarray([[7, 7, 9, 9, 5],
                                             [7, 9, 8, 9, 7],
                                             [8, 8, 8, 7, 7]]), output)

    @pytest.mark.parametrize(
        'axes', tuple(itertools.combinations(range(-3, 3), 2))
    )
    @pytest.mark.parametrize(
        'filter_func, kwargs',
        [(ndimage.minimum_filter, {}),
         (ndimage.maximum_filter, {}),
         (ndimage.median_filter, {}),
         (ndimage.rank_filter, dict(rank=3)),
         (ndimage.percentile_filter, dict(percentile=60))]
    )
    def test_minmax_nonseparable_axes(self, filter_func, axes, kwargs, xp):
        if is_cupy(xp):
            pytest.xfail("https://github.com/cupy/cupy/pull/8339")

        array = xp.arange(6 * 8 * 12, dtype=xp.float32).reshape(6, 8, 12)
        # use 2D triangular footprint because it is non-separable
        footprint = xp.tri(5)
        axes = xp.asarray(axes)

        if len(set(axes % array.ndim)) != len(axes):
            # parametrized cases with duplicate axes raise an error
            with pytest.raises(ValueError):
                filter_func(array, footprint=footprint, axes=axes, **kwargs)
            return
        output = filter_func(array, footprint=footprint, axes=axes, **kwargs)

        missing_axis = tuple(set(range(3)) - set(axes % array.ndim))[0]
        footprint_3d = xp.expand_dims(footprint, missing_axis)
        expected = filter_func(array, footprint=footprint_3d, **kwargs)
        xp_assert_close(output, expected)

    def test_rank01(self, xp):
        array = xp.asarray([1, 2, 3, 4, 5])
        output = ndimage.rank_filter(array, 1, size=2)
        assert_array_almost_equal(array, output)
        output = ndimage.percentile_filter(array, 100, size=2)
        assert_array_almost_equal(array, output)
        output = ndimage.median_filter(array, 2)
        assert_array_almost_equal(array, output)

    def test_rank02(self, xp):
        array = xp.asarray([1, 2, 3, 4, 5])
        output = ndimage.rank_filter(array, 1, size=[3])
        assert_array_almost_equal(array, output)
        output = ndimage.percentile_filter(array, 50, size=3)
        assert_array_almost_equal(array, output)
        output = ndimage.median_filter(array, (3,))
        assert_array_almost_equal(array, output)

    def test_rank03(self, xp):
        array = xp.asarray([3, 2, 5, 1, 4])
        output = ndimage.rank_filter(array, 1, size=[2])
        assert_array_almost_equal(xp.asarray([3, 3, 5, 5, 4]), output)
        output = ndimage.percentile_filter(array, 100, size=2)
        assert_array_almost_equal(xp.asarray([3, 3, 5, 5, 4]), output)

    def test_rank04(self, xp):
        array = xp.asarray([3, 2, 5, 1, 4])
        expected = xp.asarray([3, 3, 2, 4, 4])
        output = ndimage.rank_filter(array, 1, size=3)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 50, size=3)
        assert_array_almost_equal(expected, output)
        output = ndimage.median_filter(array, size=3)
        assert_array_almost_equal(expected, output)

    def test_rank05(self, xp):
        array = xp.asarray([3, 2, 5, 1, 4])
        expected = xp.asarray([3, 3, 2, 4, 4])
        output = ndimage.rank_filter(array, -2, size=3)
        assert_array_almost_equal(expected, output)

    def test_rank06(self, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]])
        expected = [[2, 2, 1, 1, 1],
                    [3, 3, 2, 1, 1],
                    [5, 5, 3, 3, 1]]
        expected = xp.asarray(expected)
        output = ndimage.rank_filter(array, 1, size=[2, 3])
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 17, size=(2, 3))
        assert_array_almost_equal(expected, output)

    def test_rank06_overlap(self, xp):
        if is_cupy(xp):
            pytest.xfail("https://github.com/cupy/cupy/issues/8406")

        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]])
        array_copy = array.copy()
        expected = [[2, 2, 1, 1, 1],
                    [3, 3, 2, 1, 1],
                    [5, 5, 3, 3, 1]]
        expected = xp.asarray(expected)
        ndimage.rank_filter(array, 1, size=[2, 3], output=array)
        assert_array_almost_equal(expected, array)

        ndimage.percentile_filter(array_copy, 17, size=(2, 3),
                                  output=array_copy)
        assert_array_almost_equal(expected, array_copy)

    def test_rank07(self, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]])
        expected = [[3, 5, 5, 5, 4],
                    [5, 5, 7, 5, 4],
                    [6, 8, 8, 7, 5]]
        expected = xp.asarray(expected)
        output = ndimage.rank_filter(array, -2, size=[2, 3])
        assert_array_almost_equal(expected, output)

    def test_rank08(self, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]])
        expected = [[3, 3, 2, 4, 4],
                    [5, 5, 5, 4, 4],
                    [5, 6, 7, 5, 5]]
        expected = xp.asarray(expected)
        output = ndimage.percentile_filter(array, 50.0, size=(2, 3))
        assert_array_almost_equal(expected, output)
        output = ndimage.rank_filter(array, 3, size=(2, 3))
        assert_array_almost_equal(expected, output)
        output = ndimage.median_filter(array, size=(2, 3))
        assert_array_almost_equal(expected, output)

        # non-separable: does not allow mode sequence
        with assert_raises(RuntimeError):
            ndimage.percentile_filter(array, 50.0, size=(2, 3),
                                      mode=['reflect', 'constant'])
        with assert_raises(RuntimeError):
            ndimage.rank_filter(array, 3, size=(2, 3), mode=['reflect']*2)
        with assert_raises(RuntimeError):
            ndimage.median_filter(array, size=(2, 3), mode=['reflect']*2)

    @pytest.mark.parametrize('dtype', types)
    def test_rank09(self, dtype, xp):
        expected = [[3, 3, 2, 4, 4],
                    [3, 5, 2, 5, 1],
                    [5, 5, 8, 3, 5]]
        expected = xp.asarray(expected)
        footprint = xp.asarray([[1, 0, 1], [0, 1, 0]])
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        output = ndimage.rank_filter(array, 1, footprint=footprint)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 35, footprint=footprint)
        assert_array_almost_equal(expected, output)

    def test_rank10(self, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        expected = [[2, 2, 1, 1, 1],
                    [2, 3, 1, 3, 1],
                    [5, 5, 3, 3, 1]]
        expected = xp.asarray(expected)
        footprint = xp.asarray([[1, 0, 1], [1, 1, 0]])
        output = ndimage.rank_filter(array, 0, footprint=footprint)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 0.0, footprint=footprint)
        assert_array_almost_equal(expected, output)

    def test_rank11(self, xp):
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        expected = [[3, 5, 5, 5, 4],
                    [7, 7, 9, 9, 5],
                    [7, 9, 8, 9, 7]]
        expected = xp.asarray(expected)
        footprint = xp.asarray([[1, 0, 1], [1, 1, 0]])
        output = ndimage.rank_filter(array, -1, footprint=footprint)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 100.0, footprint=footprint)
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    def test_rank12(self, dtype, xp):
        expected = [[3, 3, 2, 4, 4],
                    [3, 5, 2, 5, 1],
                    [5, 5, 8, 3, 5]]
        expected = xp.asarray(expected, dtype=dtype)
        footprint = xp.asarray([[1, 0, 1], [0, 1, 0]])
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        output = ndimage.rank_filter(array, 1, footprint=footprint)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 50.0,
                                           footprint=footprint)
        assert_array_almost_equal(expected, output)
        output = ndimage.median_filter(array, footprint=footprint)
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    def test_rank13(self, dtype, xp):
        expected = [[5, 2, 5, 1, 1],
                    [5, 8, 3, 5, 5],
                    [6, 6, 5, 5, 5]]
        expected = xp.asarray(expected, dtype=dtype)
        footprint = xp.asarray([[1, 0, 1], [0, 1, 0]])
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        output = ndimage.rank_filter(array, 1, footprint=footprint,
                                     origin=-1)
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    def test_rank14(self, dtype, xp):
        expected = [[3, 5, 2, 5, 1],
                    [5, 5, 8, 3, 5],
                    [5, 6, 6, 5, 5]]
        expected = xp.asarray(expected, dtype=dtype)
        footprint = xp.asarray([[1, 0, 1], [0, 1, 0]])
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        output = ndimage.rank_filter(array, 1, footprint=footprint,
                                     origin=[-1, 0])
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    def test_rank15(self, dtype, xp):
        expected = [[2, 3, 1, 4, 1],
                    [5, 3, 7, 1, 1],
                    [5, 5, 3, 3, 3]]
        expected = xp.asarray(expected, dtype=dtype)
        footprint = xp.asarray([[1, 0, 1], [0, 1, 0]])
        array = xp.asarray([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        output = ndimage.rank_filter(array, 0, footprint=footprint,
                                     origin=[-1, 0])
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    def test_generic_filter1d01(self, dtype, xp):
        weights = xp.asarray([1.1, 2.2, 3.3])

        if is_cupy(xp):
            pytest.xfail("CuPy does not support extra_arguments")


        def _filter_func(input, output, fltr, total):
            fltr = fltr / total
            for ii in range(input.shape[0] - 2):
                output[ii] = input[ii] * fltr[0]
                output[ii] += input[ii + 1] * fltr[1]
                output[ii] += input[ii + 2] * fltr[2]
        a = xp.arange(12, dtype=dtype)
        a.shape = (3, 4)
        r1 = ndimage.correlate1d(a, weights / weights.sum(), 0, origin=-1)
        r2 = ndimage.generic_filter1d(
            a, _filter_func, 3, axis=0, origin=-1,
            extra_arguments=(weights,),
            extra_keywords={'total': weights.sum()})
        assert_array_almost_equal(r1, r2)

    @pytest.mark.parametrize('dtype', types)
    def test_generic_filter01(self, dtype, xp):
        if is_cupy(xp):
            pytest.xfail("CuPy does not support extra_arguments")

        filter_ = xp.asarray([[1.0, 2.0], [3.0, 4.0]])
        footprint = xp.asarray([[1, 0], [0, 1]])
        cf = xp.asarray([1., 4.])

        def _filter_func(buffer, weights, total=1.0):
            weights = cf / total
            return (buffer * weights).sum()

        a = xp.arange(12, dtype=dtype)
        a.shape = (3, 4)
        r1 = ndimage.correlate(a, filter_ * footprint)
        if dtype in float_types:
            r1 /= 5
        else:
            r1 //= 5
        r2 = ndimage.generic_filter(
            a, _filter_func, footprint=footprint, extra_arguments=(cf,),
            extra_keywords={'total': cf.sum()})
        assert_array_almost_equal(r1, r2)

        # generic_filter doesn't allow mode sequence
        with assert_raises(RuntimeError):
            r2 = ndimage.generic_filter(
                a, _filter_func, mode=['reflect', 'reflect'],
                footprint=footprint, extra_arguments=(cf,),
                extra_keywords={'total': cf.sum()})

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [1, 1, 2]),
         ('wrap', [3, 1, 2]),
         ('reflect', [1, 1, 2]),
         ('mirror', [2, 1, 2]),
         ('constant', [0, 1, 2])]
    )
    def test_extend01(self, mode, expected_value, xp):
        array = xp.asarray([1, 2, 3])
        weights = xp.asarray([1, 0])
        output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
        expected_value = xp.asarray(expected_value)
        xp_assert_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [1, 1, 1]),
         ('wrap', [3, 1, 2]),
         ('reflect', [3, 3, 2]),
         ('mirror', [1, 2, 3]),
         ('constant', [0, 0, 0])]
    )
    def test_extend02(self, mode, expected_value, xp):
        array = xp.asarray([1, 2, 3])
        weights = xp.asarray([1, 0, 0, 0, 0, 0, 0, 0])
        output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
        expected_value = xp.asarray(expected_value)
        xp_assert_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [2, 3, 3]),
         ('wrap', [2, 3, 1]),
         ('reflect', [2, 3, 3]),
         ('mirror', [2, 3, 2]),
         ('constant', [2, 3, 0])]
    )
    def test_extend03(self, mode, expected_value, xp):
        array = xp.asarray([1, 2, 3])
        weights = xp.asarray([0, 0, 1])
        output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
        expected_value = xp.asarray(expected_value)
        xp_assert_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [3, 3, 3]),
         ('wrap', [2, 3, 1]),
         ('reflect', [2, 1, 1]),
         ('mirror', [1, 2, 3]),
         ('constant', [0, 0, 0])]
    )
    def test_extend04(self, mode, expected_value, xp):
        array = xp.asarray([1, 2, 3])
        weights = xp.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1])
        output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
        expected_value = xp.asarray(expected_value)
        xp_assert_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [[1, 1, 2], [1, 1, 2], [4, 4, 5]]),
         ('wrap', [[9, 7, 8], [3, 1, 2], [6, 4, 5]]),
         ('reflect', [[1, 1, 2], [1, 1, 2], [4, 4, 5]]),
         ('mirror', [[5, 4, 5], [2, 1, 2], [5, 4, 5]]),
         ('constant', [[0, 0, 0], [0, 1, 2], [0, 4, 5]])]
    )
    def test_extend05(self, mode, expected_value, xp):
        array = xp.asarray([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
        weights = xp.asarray([[1, 0], [0, 0]])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        expected_value = xp.asarray(expected_value)
        xp_assert_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [[5, 6, 6], [8, 9, 9], [8, 9, 9]]),
         ('wrap', [[5, 6, 4], [8, 9, 7], [2, 3, 1]]),
         ('reflect', [[5, 6, 6], [8, 9, 9], [8, 9, 9]]),
         ('mirror', [[5, 6, 5], [8, 9, 8], [5, 6, 5]]),
         ('constant', [[5, 6, 0], [8, 9, 0], [0, 0, 0]])]
    )
    def test_extend06(self, mode, expected_value, xp):
        array = xp.asarray([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
        weights = xp.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        expected_value = xp.asarray(expected_value)
        xp_assert_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [3, 3, 3]),
         ('wrap', [2, 3, 1]),
         ('reflect', [2, 1, 1]),
         ('mirror', [1, 2, 3]),
         ('constant', [0, 0, 0])]
    )
    def test_extend07(self, mode, expected_value, xp):
        array = xp.asarray([1, 2, 3])
        weights = xp.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        expected_value = xp.asarray(expected_value)
        xp_assert_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [[3], [3], [3]]),
         ('wrap', [[2], [3], [1]]),
         ('reflect', [[2], [1], [1]]),
         ('mirror', [[1], [2], [3]]),
         ('constant', [[0], [0], [0]])]
    )
    def test_extend08(self, mode, expected_value, xp):
        array = xp.asarray([[1], [2], [3]])
        weights = xp.asarray([[0], [0], [0], [0], [0], [0], [0], [0], [1]])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        expected_value = xp.asarray(expected_value)
        xp_assert_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [3, 3, 3]),
         ('wrap', [2, 3, 1]),
         ('reflect', [2, 1, 1]),
         ('mirror', [1, 2, 3]),
         ('constant', [0, 0, 0])]
    )
    def test_extend09(self, mode, expected_value, xp):
        array = xp.asarray([1, 2, 3])
        weights = xp.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        expected_value = xp.asarray(expected_value)
        xp_assert_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [[3], [3], [3]]),
         ('wrap', [[2], [3], [1]]),
         ('reflect', [[2], [1], [1]]),
         ('mirror', [[1], [2], [3]]),
         ('constant', [[0], [0], [0]])]
    )
    def test_extend10(self, mode, expected_value, xp):
        array = xp.asarray([[1], [2], [3]])
        weights = xp.asarray([[0], [0], [0], [0], [0], [0], [0], [0], [1]])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        expected_value = xp.asarray(expected_value)
        xp_assert_equal(output, expected_value)


def test_ticket_701(xp):
    if is_cupy(xp):
        pytest.xfail("CuPy raises a TypeError.")

    # Test generic filter sizes
    arr = xp.arange(4).reshape((2, 2))
    def func(x):
        return xp.min(x)
    res = ndimage.generic_filter(arr, func, size=(1, 1))
    # The following raises an error unless ticket 701 is fixed
    res2 = ndimage.generic_filter(arr, func, size=1)
    xp_assert_equal(res, res2)


def test_gh_5430(xp):
    # At least one of these raises an error unless gh-5430 is
    # fixed. In py2k an int is implemented using a C long, so
    # which one fails depends on your system. In py3k there is only
    # one arbitrary precision integer type, so both should fail.
    sigma = xp.int32(1)
    out = ndimage._ni_support._normalize_sequence(sigma, 1)
    assert out == [sigma]
    sigma = xp.int64(1)
    out = ndimage._ni_support._normalize_sequence(sigma, 1)
    assert out == [sigma]
    # This worked before; make sure it still works
    sigma = 1
    out = ndimage._ni_support._normalize_sequence(sigma, 1)
    assert out == [sigma]
    # This worked before; make sure it still works
    sigma = [1, 1]
    out = ndimage._ni_support._normalize_sequence(sigma, 2)
    assert out == sigma
    # Also include the OPs original example to make sure we fixed the issue
    x = xp.random.normal(size=(256, 256))
    perlin = xp.zeros_like(x)
    for i in 2**xp.arange(6):
        perlin += ndimage.gaussian_filter(x, i, mode="wrap") * i**2
    # This also fixes gh-4106, show that the OPs example now runs.
    x = xp.int64(21)
    ndimage._ni_support._normalize_sequence(x, 0)


def test_gaussian_kernel1d(xp):
    if is_cupy(xp):
        pytest.skip("This test tests a private scipy utility.")
    radius = 10
    sigma = 2
    sigma2 = sigma * sigma
    x = xp.arange(-radius, radius + 1, dtype=xp.double)
    phi_x = xp.exp(-0.5 * x * x / sigma2)
    phi_x /= phi_x.sum()
    xp_assert_close(phi_x, _gaussian_kernel1d(sigma, 0, radius))
    xp_assert_close(-phi_x * x / sigma2, _gaussian_kernel1d(sigma, 1, radius))
    xp_assert_close(phi_x * (x * x / sigma2 - 1) / sigma2,
                    _gaussian_kernel1d(sigma, 2, radius))
    xp_assert_close(phi_x * (3 - x * x / sigma2) * x / (sigma2 * sigma2),
                    _gaussian_kernel1d(sigma, 3, radius))


def test_orders_gauss(xp):
    # Check order inputs to Gaussians
    arr = xp.zeros((1,))
    xp_assert_equal(ndimage.gaussian_filter(arr, 1, order=0), xp.asarray([0.]))
    xp_assert_equal(ndimage.gaussian_filter(arr, 1, order=3), xp.asarray([0.]))
    assert_raises(ValueError, ndimage.gaussian_filter, arr, 1, -1)
    xp_assert_equal(ndimage.gaussian_filter1d(arr, 1, axis=-1, order=0), xp.asarray([0.]))
    xp_assert_equal(ndimage.gaussian_filter1d(arr, 1, axis=-1, order=3), xp.asarray([0.]))
    assert_raises(ValueError, ndimage.gaussian_filter1d, arr, 1, -1, -1)


def test_valid_origins(xp):
    """Regression test for #1311."""
    if is_cupy(xp):
        pytest.xfail("CuPy raises a TypeError.")

    def func(x):
        return xp.mean(x)
    data = xp.asarray([1, 2, 3, 4, 5], dtype=xp.float64)
    assert_raises(ValueError, ndimage.generic_filter, data, func, size=3,
                  origin=2)
    assert_raises(ValueError, ndimage.generic_filter1d, data, func,
                  filter_size=3, origin=2)
    assert_raises(ValueError, ndimage.percentile_filter, data, 0.2, size=3,
                  origin=2)

    for filter in [ndimage.uniform_filter, ndimage.minimum_filter,
                   ndimage.maximum_filter, ndimage.maximum_filter1d,
                   ndimage.median_filter, ndimage.minimum_filter1d]:
        # This should work, since for size == 3, the valid range for origin is
        # -1 to 1.
        list(filter(data, 3, origin=-1))
        list(filter(data, 3, origin=1))
        # Just check this raises an error instead of silently accepting or
        # segfaulting.
        assert_raises(ValueError, filter, data, 3, origin=2)


def test_bad_convolve_and_correlate_origins(xp):
    """Regression test for gh-822."""
    # Before gh-822 was fixed, these would generate seg. faults or
    # other crashes on many system.
    assert_raises(ValueError, ndimage.correlate1d,
                  [0, 1, 2, 3, 4, 5], [1, 1, 2, 0], origin=2)
    assert_raises(ValueError, ndimage.correlate,
                  [0, 1, 2, 3, 4, 5], [0, 1, 2], origin=[2])
    assert_raises(ValueError, ndimage.correlate,
                  xp.ones((3, 5)), xp.ones((2, 2)), origin=[0, 1])

    assert_raises(ValueError, ndimage.convolve1d,
                  xp.arange(10), xp.ones(3), origin=-2)
    assert_raises(ValueError, ndimage.convolve,
                  xp.arange(10), xp.ones(3), origin=[-2])
    assert_raises(ValueError, ndimage.convolve,
                  xp.ones((3, 5)), xp.ones((2, 2)), origin=[0, -2])


def test_multiple_modes(xp):
    # Test that the filters with multiple mode cababilities for different
    # dimensions give the same result as applying a single mode.
    arr = xp.asarray([[1., 0., 0.],
                       [1., 1., 0.],
                       [0., 0., 0.]])

    mode1 = 'reflect'
    mode2 = ['reflect', 'reflect']

    xp_assert_equal(ndimage.gaussian_filter(arr, 1, mode=mode1),
                 ndimage.gaussian_filter(arr, 1, mode=mode2))
    xp_assert_equal(ndimage.prewitt(arr, mode=mode1),
                 ndimage.prewitt(arr, mode=mode2))
    xp_assert_equal(ndimage.sobel(arr, mode=mode1),
                 ndimage.sobel(arr, mode=mode2))
    xp_assert_equal(ndimage.laplace(arr, mode=mode1),
                 ndimage.laplace(arr, mode=mode2))
    xp_assert_equal(ndimage.gaussian_laplace(arr, 1, mode=mode1),
                 ndimage.gaussian_laplace(arr, 1, mode=mode2))
    xp_assert_equal(ndimage.maximum_filter(arr, size=5, mode=mode1),
                 ndimage.maximum_filter(arr, size=5, mode=mode2))
    xp_assert_equal(ndimage.minimum_filter(arr, size=5, mode=mode1),
                 ndimage.minimum_filter(arr, size=5, mode=mode2))
    xp_assert_equal(ndimage.gaussian_gradient_magnitude(arr, 1, mode=mode1),
                 ndimage.gaussian_gradient_magnitude(arr, 1, mode=mode2))
    xp_assert_equal(ndimage.uniform_filter(arr, 5, mode=mode1),
                 ndimage.uniform_filter(arr, 5, mode=mode2))


def test_multiple_modes_sequentially(xp):
    # Test that the filters with multiple mode cababilities for different
    # dimensions give the same result as applying the filters with
    # different modes sequentially
    arr = xp.asarray([[1., 0., 0.],
                    [1., 1., 0.],
                    [0., 0., 0.]])

    modes = ['reflect', 'wrap']

    expected = ndimage.gaussian_filter1d(arr, 1, axis=0, mode=modes[0])
    expected = ndimage.gaussian_filter1d(expected, 1, axis=1, mode=modes[1])
    xp_assert_equal(expected,
                 ndimage.gaussian_filter(arr, 1, mode=modes))

    expected = ndimage.uniform_filter1d(arr, 5, axis=0, mode=modes[0])
    expected = ndimage.uniform_filter1d(expected, 5, axis=1, mode=modes[1])
    xp_assert_equal(expected,
                 ndimage.uniform_filter(arr, 5, mode=modes))

    expected = ndimage.maximum_filter1d(arr, size=5, axis=0, mode=modes[0])
    expected = ndimage.maximum_filter1d(expected, size=5, axis=1,
                                        mode=modes[1])
    xp_assert_equal(expected,
                 ndimage.maximum_filter(arr, size=5, mode=modes))

    expected = ndimage.minimum_filter1d(arr, size=5, axis=0, mode=modes[0])
    expected = ndimage.minimum_filter1d(expected, size=5, axis=1,
                                        mode=modes[1])
    xp_assert_equal(expected,
                 ndimage.minimum_filter(arr, size=5, mode=modes))


def test_multiple_modes_prewitt(xp):
    # Test prewitt filter for multiple extrapolation modes
    arr = xp.asarray([[1., 0., 0.],
                    [1., 1., 0.],
                    [0., 0., 0.]])

    expected = xp.asarray([[1., -3., 2.],
                         [1., -2., 1.],
                         [1., -1., 0.]])

    modes = ['reflect', 'wrap']

    xp_assert_equal(expected,
                 ndimage.prewitt(arr, mode=modes))


def test_multiple_modes_sobel(xp):
    # Test sobel filter for multiple extrapolation modes
    arr = xp.asarray([[1., 0., 0.],
                    [1., 1., 0.],
                    [0., 0., 0.]])

    expected = xp.asarray([[1., -4., 3.],
                         [2., -3., 1.],
                         [1., -1., 0.]])

    modes = ['reflect', 'wrap']

    xp_assert_equal(expected,
                 ndimage.sobel(arr, mode=modes))


def test_multiple_modes_laplace(xp):
    # Test laplace filter for multiple extrapolation modes
    arr = xp.asarray([[1., 0., 0.],
                    [1., 1., 0.],
                    [0., 0., 0.]])

    expected = xp.asarray([[-2., 2., 1.],
                         [-2., -3., 2.],
                         [1., 1., 0.]])

    modes = ['reflect', 'wrap']

    xp_assert_equal(expected,
                 ndimage.laplace(arr, mode=modes))


def test_multiple_modes_gaussian_laplace(xp):
    # Test gaussian_laplace filter for multiple extrapolation modes
    arr = xp.asarray([[1., 0., 0.],
                    [1., 1., 0.],
                    [0., 0., 0.]])

    expected = xp.asarray([[-0.28438687, 0.01559809, 0.19773499],
                         [-0.36630503, -0.20069774, 0.07483620],
                         [0.15849176, 0.18495566, 0.21934094]])

    modes = ['reflect', 'wrap']

    assert_almost_equal(expected,
                        ndimage.gaussian_laplace(arr, 1, mode=modes))


def test_multiple_modes_gaussian_gradient_magnitude(xp):
    # Test gaussian_gradient_magnitude filter for multiple
    # extrapolation modes
    arr = xp.asarray([[1., 0., 0.],
                    [1., 1., 0.],
                    [0., 0., 0.]])

    expected = xp.asarray([[0.04928965, 0.09745625, 0.06405368],
                         [0.23056905, 0.14025305, 0.04550846],
                         [0.19894369, 0.14950060, 0.06796850]])

    modes = ['reflect', 'wrap']

    calculated = ndimage.gaussian_gradient_magnitude(arr, 1, mode=modes)

    assert_almost_equal(expected, calculated)


def test_multiple_modes_uniform(xp):
    # Test uniform filter for multiple extrapolation modes
    arr = xp.asarray([[1., 0., 0.],
                    [1., 1., 0.],
                    [0., 0., 0.]])

    expected = xp.asarray([[0.32, 0.40, 0.48],
                         [0.20, 0.28, 0.32],
                         [0.28, 0.32, 0.40]])

    modes = ['reflect', 'wrap']

    assert_almost_equal(expected,
                        ndimage.uniform_filter(arr, 5, mode=modes))


def test_gaussian_truncate(xp):
    # Test that Gaussian filters can be truncated at different widths.
    # These tests only check that the result has the expected number
    # of nonzero elements.
    arr = xp.zeros((100, 100), float)
    arr[50, 50] = 1
    num_nonzeros_2 = (ndimage.gaussian_filter(arr, 5, truncate=2) > 0).sum()
    assert num_nonzeros_2 == 21**2
    num_nonzeros_5 = (ndimage.gaussian_filter(arr, 5, truncate=5) > 0).sum()
    assert num_nonzeros_5 == 51**2

    # Test truncate when sigma is a sequence.
    f = ndimage.gaussian_filter(arr, [0.5, 2.5], truncate=3.5)
    fpos = f > 0
    n0 = fpos.any(axis=0).sum()
    # n0 should be 2*int(2.5*3.5 + 0.5) + 1
    assert n0 == 19
    n1 = fpos.any(axis=1).sum()
    # n1 should be 2*int(0.5*3.5 + 0.5) + 1
    assert n1 == 5

    # Test gaussian_filter1d.
    x = xp.zeros(51)
    x[25] = 1
    f = ndimage.gaussian_filter1d(x, sigma=2, truncate=3.5)
    n = (f > 0).sum()
    assert n == 15

    # Test gaussian_laplace
    y = ndimage.gaussian_laplace(x, sigma=2, truncate=3.5)
    nonzero_indices = xp.nonzero(y != 0)[0]
    n = xp.ptp(nonzero_indices) + 1
    assert n == 15

    # Test gaussian_gradient_magnitude
    y = ndimage.gaussian_gradient_magnitude(x, sigma=2, truncate=3.5)
    nonzero_indices = xp.nonzero(y != 0)[0]
    n = xp.ptp(nonzero_indices) + 1
    assert n == 15


def test_gaussian_radius(xp):
    if is_cupy(xp):
        pytest.xfail("https://github.com/cupy/cupy/issues/8402")

    # Test that Gaussian filters with radius argument produce the same
    # results as the filters with corresponding truncate argument.
    # radius = int(truncate * sigma + 0.5)
    # Test gaussian_filter1d
    x = xp.zeros(7)
    x[3] = 1
    f1 = ndimage.gaussian_filter1d(x, sigma=2, truncate=1.5)
    f2 = ndimage.gaussian_filter1d(x, sigma=2, radius=3)
    xp_assert_equal(f1, f2)

    # Test gaussian_filter when sigma is a number.
    a = xp.zeros((9, 9))
    a[4, 4] = 1
    f1 = ndimage.gaussian_filter(a, sigma=0.5, truncate=3.5)
    f2 = ndimage.gaussian_filter(a, sigma=0.5, radius=2)
    xp_assert_equal(f1, f2)

    # Test gaussian_filter when sigma is a sequence.
    a = xp.zeros((50, 50))
    a[25, 25] = 1
    f1 = ndimage.gaussian_filter(a, sigma=[0.5, 2.5], truncate=3.5)
    f2 = ndimage.gaussian_filter(a, sigma=[0.5, 2.5], radius=[2, 9])
    xp_assert_equal(f1, f2)


def test_gaussian_radius_invalid(xp):
    if is_cupy(xp):
        pytest.xfail("https://github.com/cupy/cupy/issues/8402")

    # radius must be a nonnegative integer
    with assert_raises(ValueError):
        ndimage.gaussian_filter1d(xp.zeros(8), sigma=1, radius=-1)
    with assert_raises(ValueError):
        ndimage.gaussian_filter1d(xp.zeros(8), sigma=1, radius=1.1)


class TestThreading:
    def check_func_thread(self, n, fun, args, out):
        from threading import Thread
        thrds = [Thread(target=fun, args=args, kwargs={'output': out[x]})
                 for x in range(n)]
        [t.start() for t in thrds]
        [t.join() for t in thrds]

    def check_func_serial(self, n, fun, args, out):
        for i in range(n):
            fun(*args, output=out[i])

    def test_correlate1d(self, xp):
        if is_cupy(xp):
            pytest.xfail("XXX thread exception; cannot repro outside of pytest")

        d = xp.random.randn(5000)
        os = xp.empty((4, d.size))
        ot = xp.empty_like(os)
        k = xp.arange(5)
        self.check_func_serial(4, ndimage.correlate1d, (d, k), os)
        self.check_func_thread(4, ndimage.correlate1d, (d, k), ot)
        xp_assert_equal(os, ot)

    def test_correlate(self, xp):
        if is_cupy(xp):
            pytest.xfail("XXX thread exception; cannot repro outside of pytest")

        d = xp.random.randn(500, 500)
        k = xp.random.randn(10, 10)
        os = xp.empty([4] + list(d.shape))
        ot = xp.empty_like(os)
        self.check_func_serial(4, ndimage.correlate, (d, k), os)
        self.check_func_thread(4, ndimage.correlate, (d, k), ot)
        xp_assert_equal(os, ot)

    def test_median_filter(self, xp):
        if is_cupy(xp):
            pytest.xfail("XXX thread exception; cannot repro outside of pytest")

        d = xp.random.randn(500, 500)
        os = xp.empty([4] + list(d.shape))
        ot = xp.empty_like(os)
        self.check_func_serial(4, ndimage.median_filter, (d, 3), os)
        self.check_func_thread(4, ndimage.median_filter, (d, 3), ot)
        xp_assert_equal(os, ot)

    def test_uniform_filter1d(self, xp):
        if is_cupy(xp):
            pytest.xfail("XXX thread exception; cannot repro outside of pytest")

        d = xp.random.randn(5000)
        os = xp.empty((4, d.size))
        ot = xp.empty_like(os)
        self.check_func_serial(4, ndimage.uniform_filter1d, (d, 5), os)
        self.check_func_thread(4, ndimage.uniform_filter1d, (d, 5), ot)
        xp_assert_equal(os, ot)

    def test_minmax_filter(self, xp):
        if is_cupy(xp):
            pytest.xfail("XXX thread exception; cannot repro outside of pytest")

        d = xp.random.randn(500, 500)
        os = xp.empty([4] + list(d.shape))
        ot = xp.empty_like(os)
        self.check_func_serial(4, ndimage.maximum_filter, (d, 3), os)
        self.check_func_thread(4, ndimage.maximum_filter, (d, 3), ot)
        xp_assert_equal(os, ot)
        self.check_func_serial(4, ndimage.minimum_filter, (d, 3), os)
        self.check_func_thread(4, ndimage.minimum_filter, (d, 3), ot)
        xp_assert_equal(os, ot)


def test_minmaximum_filter1d(xp):
    # Regression gh-3898
    in_ = xp.arange(10)
    out = ndimage.minimum_filter1d(in_, 1)
    xp_assert_equal(in_, out)
    out = ndimage.maximum_filter1d(in_, 1)
    xp_assert_equal(in_, out)
    # Test reflect
    out = ndimage.minimum_filter1d(in_, 5, mode='reflect')
    xp_assert_equal(xp.asarray([0, 0, 0, 1, 2, 3, 4, 5, 6, 7]), out)
    out = ndimage.maximum_filter1d(in_, 5, mode='reflect')
    xp_assert_equal(xp.asarray([2, 3, 4, 5, 6, 7, 8, 9, 9, 9]), out)
    # Test constant
    out = ndimage.minimum_filter1d(in_, 5, mode='constant', cval=-1)
    xp_assert_equal(xp.asarray([-1, -1, 0, 1, 2, 3, 4, 5, -1, -1]), out)
    out = ndimage.maximum_filter1d(in_, 5, mode='constant', cval=10)
    xp_assert_equal(xp.asarray([10, 10, 4, 5, 6, 7, 8, 9, 10, 10]), out)
    # Test nearest
    out = ndimage.minimum_filter1d(in_, 5, mode='nearest')
    xp_assert_equal(xp.asarray([0, 0, 0, 1, 2, 3, 4, 5, 6, 7]), out)
    out = ndimage.maximum_filter1d(in_, 5, mode='nearest')
    xp_assert_equal(xp.asarray([2, 3, 4, 5, 6, 7, 8, 9, 9, 9]), out)
    # Test wrap
    out = ndimage.minimum_filter1d(in_, 5, mode='wrap')
    xp_assert_equal(xp.asarray([0, 0, 0, 1, 2, 3, 4, 5, 0, 0]), out)
    out = ndimage.maximum_filter1d(in_, 5, mode='wrap')
    xp_assert_equal(xp.asarray([9, 9, 4, 5, 6, 7, 8, 9, 9, 9]), out)


def test_uniform_filter1d_roundoff_errors(xp):
    if is_cupy(xp):
        pytest.xfail("https://github.com/cupy/cupy/issues/8401")
    # gh-6930
    in_ = xp.repeat(xp.asarray([0, 1, 0]), [9, 9, 9])
    for filter_size in range(3, 10):
        out = ndimage.uniform_filter1d(in_, filter_size)
        xp_assert_equal(out.sum(), xp.asarray(10 - filter_size))


def test_footprint_all_zeros(xp):
    # regression test for gh-6876: footprint of all zeros segfaults
    arr = xp.random.randint(0, 100, (100, 100))
    kernel = xp.zeros((3, 3), bool)
    with assert_raises(ValueError):
        ndimage.maximum_filter(arr, footprint=kernel)


def test_gaussian_filter(xp):
    if is_cupy(xp):
        pytest.xfail("CuPy does not raise")

    # Test gaussian filter with xp.float16
    # gh-8207
    data = xp.asarray([1], dtype=xp.float16)
    sigma = 1.0
    with assert_raises(RuntimeError):
        ndimage.gaussian_filter(data, sigma)


def test_rank_filter_noninteger_rank(xp):
    if is_cupy(xp):
        pytest.xfail("CuPy does not raise")

    # regression test for issue 9388: ValueError for
    # non integer rank when performing rank_filter
    arr = xp.random.random((10, 20, 30))
    assert_raises(TypeError, ndimage.rank_filter, arr, 0.5,
                  footprint=xp.ones((1, 1, 10), dtype=bool))


def test_size_footprint_both_set(xp):
    # test for input validation, expect user warning when
    # size and footprint is set
    with suppress_warnings() as sup:
        sup.filter(UserWarning,
                   "ignoring size because footprint is set")
        arr = xp.random.random((10, 20, 30))
        ndimage.rank_filter(arr, 5, size=2, footprint=xp.ones((1, 1, 10), dtype=bool))


def test_byte_order_median(xp):
    """Regression test for #413: median_filter does not handle bytes orders."""
    if is_cupy(xp):
        pytest.xfail("CuPy does not handle byteorders")

    a = xp.arange(9, dtype='<f4').reshape(3, 3)
    ref = ndimage.median_filter(a, (3, 3))
    b = xp.arange(9, dtype='>f4').reshape(3, 3)
    t = ndimage.median_filter(b, (3, 3))
    assert_array_almost_equal(ref, t)
