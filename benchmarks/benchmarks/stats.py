from __future__ import division, absolute_import, print_function

import numpy as np

try:
    from scipy.stats import moment
except ImportError:
    pass

from .common import Benchmark


class BenchMoment(Benchmark):
    params = [
        [1, 2, 3, 8],
        [100, 1000, 10000],
    ]
    param_names = ["order", "size"]
    
    def setup(self, order, size):
        np.random.random(1234)
        self.x = np.random.random(size)

    def time_moment(self, order, size):
        moment(self.x, order)
