from typing import Type

from .core import *


_OPERATORS = list()


def get_available_operators() -> List[Type[Operator]]:
    return _OPERATORS.copy()


def register_operator(cls: Type[Operator]) -> Type[Operator]:
    if not issubclass(cls, Operator):
        raise ValueError("%s is not an Operator" % str(cls))
    _OPERATORS.append(cls)
    return cls


from . import implementations
