import pytest

from scipy.conftest import array_api_compatible
skip_xp_backends = pytest.mark.skip_xp_backends

pytestmark = [array_api_compatible, pytest.mark.usefixtures("skip_xp_backends")]

#@array_api_compatible
@skip_xp_backends('numpy')
@pytest.mark.parametrize('i', [1, 2, 3])
@skip_xp_backends('array_api_strict')
def test_dummy(xp, i):
    print('>>', xp.__name__)



