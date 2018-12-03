from typing import Type, TypeVar, Generic

from .parameter import Parameter, IntegerParameter, NumberParameter


__ADAPTERS = {}


def adapts(parameter: Parameter):
    """ Decorator for assigning a ParameterAdapter to some Parameter class
    """
    if not isinstance(parameter, Parameter):
        raise ValueError('You should specify which parameter this adapter will be assigned to')
    global __ADAPTERS

    def decorator(cls: Type[ParameterAdapter]) -> Type[ParameterAdapter]:
        __ADAPTERS[parameter] = cls()
        return cls

    return decorator


def get_adapter(parameter: Parameter):
    """ Returns the adapter assigned to some parameter
    """
    global __ADAPTERS
    if parameter not in __ADAPTERS:
        raise ValueError('Adapter for parameter %s not found' % str(type(parameter)))
    return __ADAPTERS[parameter]


T = TypeVar('T')


class ParameterAdapter(Generic[T]):
    def accept(self, value_str: str) -> bool:
        raise NotImplementedError()

    def parse(self, value_str: str) -> T:
        raise NotImplementedError()

    def serialize(self, value: T) -> str:
        raise NotImplementedError()


@adapts(IntegerParameter)
class IntegerAdapter(ParameterAdapter[int]):
    def accept(self, value_str: str) -> bool:
        return value_str.isnumeric()

    def parse(self, value_str: str) -> int:
        return int(value_str)

    def serialize(self, value: int) -> str:
        return str(value)


@adapts(NumberParameter)
class NumberAdapter(ParameterAdapter[float]):
    def accept(self, value_str: str) -> bool:
        try:
            float(value_str)
        except ValueError:
            return False
        return True

    def parse(self, value_str: str) -> float:
        return float(value_str)

    def serialize(self, value: float) -> str:
        return str(value)
