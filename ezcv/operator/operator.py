from typing import Type, Dict, List

from .parameter import ParameterSpec
from .settings import OperatorSettingsMixin
from ezcv.pipeline import PipelineContext
from ezcv.typing import Image


_OPERATORS = list()


def get_available_operators() -> List[Type['Operator']]:
    return _OPERATORS.copy()


def register_operator(cls: Type['Operator']) -> Type['Operator']:
    if not issubclass(cls, Operator):
        raise ValueError("%s is not an Operator" % str(cls))
    _OPERATORS.append(cls)
    return cls


class Operator(OperatorSettingsMixin):
    """ Class representing an image operator

    Extend this class to implement new functionality
    """
    def run(self, img: Image, ctx: PipelineContext) -> Image:
        raise NotImplementedError()

    @classmethod
    def get_parameters_specs(cls: Type['Operator']) -> Dict[str, ParameterSpec]:
        return {name: value for name, value in cls.__dict__.items() if isinstance(value, ParameterSpec)}
