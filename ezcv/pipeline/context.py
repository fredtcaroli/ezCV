from contextlib import contextmanager
from typing import ContextManager

import numpy as np

from ezcv.utils import is_image


class PipelineContext(object):
    def __init__(self, original_img: np.ndarray):
        if not is_image(original_img):
            raise ValueError('Invalid original image')
        self._original_img = original_img
        self._serving_original_img = None
        self._scopes = list()

    @property
    def original_img(self):
        if self._serving_original_img is None:
            self._serving_original_img = self._original_img.copy()
        return self._serving_original_img

    @contextmanager
    def scope(self, name: str) -> ContextManager:
        self._scopes.append(name)
        last_serving_original_img = self._serving_original_img
        self._serving_original_img = None
        yield
        assert self._scopes[-1] == name
        self._serving_original_img = last_serving_original_img
        self._scopes.pop()
