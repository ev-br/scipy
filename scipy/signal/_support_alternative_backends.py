import sys
import os
import functools
from scipy._lib._array_api import array_namespace, is_cupy, is_jax, scipy_namespace_for

from ._signaltools import convolve, convolve2d


MODULE_NAME = 'signal'
_SCIPY_ARRAY_API = os.environ.get("SCIPY_ARRAY_API", False)


def dispatch_cupy(dispatcher, module_name):
    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            xp, can_dispatch = dispatcher(*args, **kwds)

            # try delegating to a cupyx/jax namesake
            if can_dispatch and is_cupy(xp):
                # https://github.com/cupy/cupy/issues/8336
                import importlib
                cupyx_module = importlib.import_module(f"cupyx.scipy.{module_name}")
                cupyx_func = getattr(cupyx_module, func.__name__)
                return cupyx_func(*args, **kwds)
            elif can_dispatch and is_jax(xp):
                spx = scipy_namespace_for(xp)
                jax_module = getattr(spx, module_name)
                jax_func = getattr(jax_module, func.__name__)
                return jax_func(*args, **kwds)
            else:
                # the original function
                return func(*args, **kwds)
        return wrapper
    return inner



# X_dispatcher signature must match the signature of X

def convolve_dispatcher(in1, in2, mode='full', method='auto'):
    """ Deduce the array namespace and whether we can dispatch to the cupy namesake.

    The signature must agree with that of convolve itself.
    """
    xp = array_namespace(in1, in2)
    if is_cupy(xp):
        # Conditions are accurate as of CuPy 13.x
        if method == "auto":
            can_dispatch = in1.ndim == 1 and in2.ndim == 1
        else:
            can_dispatch = True

        # inputs are cupy arrays and cupyx.scipy.convolve cannot be used
        if not can_dispatch:
            raise ValueError(
                f"Cannot dispatch to {xp.__name__} for {in1.ndim = }, "
                f" {in2.ndim = }, {mode = } and {method = }."
            )

    elif is_jax(xp):
        can_dispatch = True
    else:
        can_dispatch = False

    return xp, can_dispatch

def convolve2d_dispatcher(in1, in2, mode='full', boundary='fill', fillvalue=0):
    """ Deduce the array namespace and whether we can dispatch to the cupy namesake.

    The signature must agree with that of convolve2d itself.
    """
    xp = array_namespace(in1, in2)
    if is_jax(xp):
        can_dispatch = boundary == 'fill' and fillvalue == 0

        if not can_dispatch:
            raise ValueError(
                f"Cannot dispatch to {xp.__name__} for {in1.ndim = }, "
                f" {in2.ndim = }, {boundary = } and {fillvalue = }."
            )
    else:
        can_dispatch = True

    return xp, can_dispatch



# functions we patch for dispatch
_func_map = {
    convolve: convolve_dispatcher,
    convolve2d: convolve2d_dispatcher,
}


# ### decorate ###
for func in _func_map:
    f = (dispatch_cupy(_func_map[func], MODULE_NAME)(func)
         if _SCIPY_ARRAY_API
         else func)
    #f = dispatch_cupy(_func_map[func], MODULE_NAME)(func)
    sys.modules[__name__].__dict__[func.__name__] = f


__all__ = [f.__name__ for f in _func_map]
