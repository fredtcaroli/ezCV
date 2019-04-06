from typing import Generic, TypeVar, Any, List

T = TypeVar('T')


class Parameter(Generic[T]):
    def __init__(self, default_value: T):
        self.default_value = default_value

    def from_config(self, config: Any) -> T:
        raise NotImplementedError()

    def to_config(self, value: T) -> Any:
        raise NotImplementedError()

    def __get__(self, instance, owner) -> T:
        if instance is None:
            return self
        return self.default_value


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

    def to_config(self, value: float) -> float:
        assert isinstance(value, (float, int))
        return float(value)


class EnumParameter(Parameter[str]):
    def __init__(self, possible_values: List[str], default_value: str):
        if default_value not in possible_values:
            raise ValueError('Invalid default value: "%s". Possible values are: %s' %
                             (str(default_value), str(possible_values)))
        super().__init__(default_value=default_value)
        self.possible_values = possible_values

    def from_config(self, config: Any) -> str:
        assert isinstance(config, str) and config in self.possible_values
        return config

    def to_config(self, value: str) -> str:
        assert isinstance(value, str) and value in self.possible_values
        return value

