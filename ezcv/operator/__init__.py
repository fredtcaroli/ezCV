import functools
import importlib
import inspect
import pkgutil
from typing import Set

from .core import *
from . import implementations


def __cache_once(wrapped):
    r = wrapped()

    @functools.wraps(wrapped)
    def wrapper():
        return r
    return wrapper


@__cache_once
def get_available_operators() -> Set[Operator]:
    collected = set()
    for importer, modname, ispkg in pkgutil.iter_modules(implementations.__path__):
        if not ispkg:
            module = importlib.import_module('ezcv.operator.implementations.%s' % modname)
            ops = inspect.getmembers(module, lambda member: inspect.isclass(member) and issubclass(member, Operator))
            for op in ops:
                collected.add(op[1])
        else:
            raise ValueError('Packages inside ezcv.operator.implementations are not supported yet')
    if Operator in collected:
        collected.remove(Operator)
    return collected
