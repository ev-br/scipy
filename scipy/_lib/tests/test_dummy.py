import pytest

from scipy.conftest import array_api_compatible
skip_xp_backends = pytest.mark.skip_xp_backends

pytestmark = [array_api_compatible, pytest.mark.usefixtures("skip_xp_backends")]

#@array_api_compatible

@skip_xp_backends(np_only=True)
@pytest.mark.parametrize('i', [1, 2, 3])
@skip_xp_backends('numpy', reasons=['a numpy reason'])
#@skip_xp_backends('array_api_strict')
def test_dummy(xp, i):
    print('>>', xp.__name__)



i = 4


@pytest.mark.skipif(i==4, reason='exact value')
@pytest.mark.skipif(i%2 == 0, reason='is even')
def test_two_skips():
    assert True
