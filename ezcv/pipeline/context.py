from contextlib import contextmanager
from typing import ContextManager, Any

from ezcv.typing import Image
from ezcv.utils import is_image


class PipelineContext(object):
    def __init__(self, original_img: Image):
        if not is_image(original_img):
            raise ValueError('Invalid original image')
        self.original_img = original_img.copy()
        self.original_img.flags.writeable = False
        self._scopes = list()
        self.info = dict()

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
        scoped_info = self._get_info_in_scope()
        if name in scoped_info:
            raise ValueError('Trying to add a duplicated info name: "%s"' % name)
        if not isinstance(name, str) or len(name) == 0 or '/' in name:
            raise ValueError('Invalid name %s' % str(name))
        self._insert_info(name, info_value)

    def _get_info_in_scope(self) -> dict:
        last = self.info
        for scope_name in self._scopes:
            last = last[scope_name]
        return last

    def _insert_info(self, name: str, info_value: Any):
        scoped_info = self._get_info_in_scope()
        scoped_info[name] = info_value
