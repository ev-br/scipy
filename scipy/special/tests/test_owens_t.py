import numpy as np
from numpy.testing import assert_equal, run_module_suite, assert_allclose
from scipy import special


def test_owens_t():
    data = np.random.rand(100, 2)

    # symmetries test
    for h, a in data:
        assert_equal(special.owens_t(h, a), special.owens_t(-h, a))
        assert_equal(special.owens_t(h, a), -special.owens_t(h, -a))

    # test specific values: a = 0, a = 1, h = 0
    assert_equal(special.owens_t(5, 0), 0)
    assert_allclose(special.owens_t(0, 5),
                    0.2185835209054994081, rtol=5e-14)
    assert_allclose(special.owens_t(5, 1),
                    1.4332574485503512543e-07, rtol=5e-14)

    # Owens T1 test
    assert_allclose(special.owens_t(0.00001, 0.98085),
                    0.1234614068559199053, rtol=5e-14)

    # Owens T2 test
    assert_allclose(special.owens_t(15, 7),
                    1.8354830996563754428e-51, rtol=5e-14)

    # Owens T1 accelerated test
    assert_allclose(special.owens_t(0.05, 900),
                    0.2400305970808137692, rtol=5e-14)

    # Owens T2 accelerated test
    assert_allclose(special.owens_t(20, 700),
                    1.3768120593031168275e-89, rtol=5e-14)


if __name__ == "__main__":
    run_module_suite()
