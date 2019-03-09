import functools
from typing import Any, Tuple, Callable

import numpy as np
import pytest


def build_img(size: Tuple[int, ...], kind='random') -> np.ndarray:
    if kind == 'random':
        return np.random.randint(0, 256, size=size, dtype='uint8')
    elif kind == 'black':
        return np.zeros(size, dtype='uint8')
    elif kind == 'white':
        return np.full(size, 255, dtype='uint8')


def is_image(data: Any):
    return (
        isinstance(data, np.ndarray) and
        (data.ndim == 2 or data.ndim == 3) and
        (data.ndim == 2 or data.shape[2] == 3)
    )


def parametrize_img(func: Callable = None, include_valid: bool = True, include_invalid: bool = False,
                    gray_only=False, kind='random'):
    shapes = []
    if include_valid:
        shapes += [
            (100, 100),
            (1, 1),
            (3, 3),
            (720, 1280),
            (3, 1280),
            (720, 3)
        ]
        if not gray_only:
            shapes += [
                (100, 100, 3),
                (1, 1, 3),
                (3, 3, 3),
                (720, 1280, 3),
                (3, 1280, 3),
                (720, 3, 3)
            ]

    if include_invalid:
        shapes += [
            (0, 0),
            (100, 0),
            (0, 100),
        ]
        if not gray_only:
            shapes += [
                (0, 0, 3),
                (100, 0, 3),
                (0, 100, 3)
            ]
    wrapper = pytest.mark.parametrize('img', map(functools.partial(build_img, kind=kind), shapes))
    if func is None:
        return wrapper
    else:
        return wrapper(func)
