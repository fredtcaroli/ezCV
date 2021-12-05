from typing import Generic, TypeVar, Any, List

T = TypeVar('T')


class ParameterSpec(Generic[T]):
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


class IntegerParameter(ParameterSpec[int]):
    def __init__(self, default_value: int, lower: int, upper: int, step_size: int = 1):
        if not isinstance(default_value, int):
            raise ValueError('Invalid default_value: %s' % str(default_value))
        super().__init__(default_value)
        if not isinstance(lower, int):
            raise ValueError('Invalid integer lower limit: %s' % str(lower))
        if not isinstance(upper, int):
            raise ValueError('Invalid integer upper limit: %s' % str(upper))
        if not isinstance(step_size, int) or step_size <= 0:
            raise ValueError('Invalid integer step_size: %s' % str(step_size))
        if lower >= upper:
            raise ValueError('Invalid lower and upper limits: (%d, %d)' % (lower, upper))
        self.lower = lower
        self.upper = upper
        self.step_size = step_size

    def from_config(self, config: Any) -> int:
        assert isinstance(config, int)
        return config

    def to_config(self, value: int) -> Any:
        assert isinstance(value, int)
        return value


class DoubleParameter(ParameterSpec[float]):
    def __init__(self, default_value: float, lower: float, upper: float, step_size: float = 0.1):
        if not isinstance(default_value, (float, int)):
            raise ValueError('Invalid default_value: %s' % str(default_value))
        super().__init__(default_value)
        if not isinstance(lower, (float, int)):
            raise ValueError('Invalid number lower limit: %s' % str(lower))
        if not isinstance(upper, (float, int)):
            raise ValueError('Invalid number upper limit: %s' % str(upper))
        if not isinstance(step_size, (float, int)) or step_size <= 0:
            raise ValueError('Invalid integer step_size: %s' % str(step_size))
        if lower >= upper:
            raise ValueError('Invalid lower and upper limits: (%d, %d)' % (lower, upper))
        self.lower = lower
        self.upper = upper
        self.step_size = step_size

    def from_config(self, config: Any) -> float:
        assert isinstance(config, (float, int))
        return config

    def to_config(self, value: float) -> float:
        assert isinstance(value, (float, int))
        return float(value)


class EnumParameter(ParameterSpec[str]):
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

