from collections import OrderedDict
from contextlib import contextmanager
from typing import ContextManager, Any

import numpy as np

from ezcv.utils import is_image


class PipelineContext(object):
    def __init__(self, original_img: np.ndarray):
        if not is_image(original_img):
            raise ValueError('Invalid original image')
        self.original_img = original_img.copy()
        self.original_img.flags.writeable = False
        self._serving_original_img = None
        self._scopes = list()
        self.info = OrderedDict()

    @contextmanager
    def scope(self, name: str) -> ContextManager:
        self._scopes.append(name)
        self._create_info_path()
        yield
        assert self._scopes[-1] == name
        self._scopes.pop()

    def _create_info_path(self):
        last = self.info
        for scope_name in self._scopes:
            last = last.setdefault(scope_name, dict())

    def add_info(self, name: str, info_value: Any):
        if name in self.info:
            raise ValueError('Trying to add a duplicated info name: "%s"' % name)
        if not isinstance(name, str) or len(name) == 0 or '/' in name:
            raise ValueError('Invalid name %s' % str(name))
        self._insert_info(name, info_value)

    def _insert_info(self, name: str, info_value: Any):
        last = self.info
        for scope_name in self._scopes:
            last = last[scope_name]
        last[name] = info_value