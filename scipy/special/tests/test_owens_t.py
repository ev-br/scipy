import numpy as np
from numpy.testing import (TestCase, assert_equal, run_module_suite,
                           assert_allclose)
from scipy import special


class TestOwensT(TestCase):
    def test_symmetries(self):
        data = np.random.rand(100, 2)

        for h, a in data:
            assert_equal(special.owens_t(h, a), special.owens_t(-h, a))
            assert_equal(special.owens_t(h, a), -special.owens_t(h, -a))

    def test_zeros(self):
        assert_equal(special.owens_t(5, 0), 0)
        assert_allclose(special.owens_t(0, 5),
                        0.2185835209054994081, rtol=5e-14)
        assert_allclose(special.owens_t(5, 1),
                        1.4332574485503512543e-07, rtol=5e-14)

    def test_owensT1(self):
        assert_allclose(special.owens_t(0.00001, 0.98085),
                        0.1234614068559199053, rtol=5e-14)

    def test_owensT2(self):
        assert_allclose(special.owens_t(15, 7),
                        1.8354830996563754428e-51, rtol=5e-14)

    def test_owensT1_accelerated(self):
        assert_allclose(special.owens_t(0.05, 900),
                        0.2400305970808137692, rtol=5e-14)

    def test_owensT2_accelerated(self):
        assert_allclose(special.owens_t(20, 700),
                        1.3768120593031168275e-89, rtol=5e-14)

    def test_nans(self):
        assert_equal(special.owens_t(20, np.nan), np.nan)
        assert_equal(special.owens_t(np.nan, 20), np.nan)
        assert_equal(special.owens_t(np.nan, np.nan), np.nan)


if __name__ == "__main__":
    run_module_suite()
