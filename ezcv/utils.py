import functools
from typing import Any

import numpy as np


def is_image(data: Any):
    return (
        isinstance(data, np.ndarray) and
        (data.ndim == 2 or data.ndim == 3) and
        (data.ndim == 2 or data.shape[2] == 3) and
        functools.reduce(lambda a, b: a * b, data.shape) > 0
    )
