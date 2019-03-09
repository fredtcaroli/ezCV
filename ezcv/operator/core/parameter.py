from typing import Generic, TypeVar, Any

T = TypeVar('T')


class Parameter(Generic[T]):
    def from_config(self, config: Any) -> T:
        raise NotImplementedError()

    def to_config(self, value: T) -> Any:
        raise NotImplementedError()

    def __get__(self, instance, owner) -> T:
        """ Implementing __get__ so IDEs don't get confused with the values injection
        """
        return self


class IntegerParameter(Parameter[int]):
    def from_config(self, config: Any) -> int:
        assert isinstance(config, int)
        return config

    def to_config(self, value: int) -> Any:
        assert isinstance(value, int)
        return value


class NumberParameter(Parameter[float]):
    def from_config(self, config: Any) -> float:
        assert isinstance(config, (float, int))
        return config

    def to_config(self, value: float) -> Any:
        assert isinstance(value, (float, int))
        return float(value)
