import numpy as np
from numpy.testing import assert_equal, run_module_suite
from scipy import special


def test_owens_t():
    data = np.random.rand(100, 2)

    for h, a in data:
        assert_equal(special.owens_t(h, a), special.owens_t(-h, a))
        assert_equal(special.owens_t(h, a), -special.owens_t(h, -a))


if __name__ == "__main__":
    run_module_suite()
