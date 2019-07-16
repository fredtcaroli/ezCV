import functools
from typing import Tuple, Callable, List

import numpy as np
import pytest


def build_img(size: Tuple[int, int], kind='random', rgb=False) -> np.ndarray:
    assert len(size) == 2
    if rgb:
        size = size + (3,)
    if kind == 'random':
        return np.random.randint(0, 256, size=size, dtype='uint8')
    elif kind == 'black':
        return np.zeros(size, dtype='uint8')
    elif kind == 'white':
        return np.full(size, 255, dtype='uint8')


def parametrize_img(func: Callable = None, include_valid: bool = True, include_invalid: bool = False,
                    gray_only=False, kind='random'):
    shapes = []
    valid_sizes = [
        (100, 100),
        (1, 1),
        (3, 3),
        (720, 1280),
        (3, 1280),
        (720, 3)
    ]
    invalid_sizes = [
        (0, 0),
        (100, 0),
        (0, 100),
    ]
    if include_valid:
        shapes += valid_sizes

    if include_invalid:
        shapes += invalid_sizes

    imgs = list()
    imgs.extend(map(functools.partial(build_img, kind=kind, rgb=False), shapes))
    if not gray_only:
        imgs.extend(map(functools.partial(build_img, kind=kind, rgb=True), shapes))

    wrapper = pytest.mark.parametrize('img', imgs)
    if func is None:
        return wrapper
    else:
        return wrapper(func)


def assert_terms_in_exception(e: "ExceptionInfo", terms: List[str]):
    msg = str(e.value).lower()
    for term in terms:
        assert term.lower().strip() in msg
