from typing import Type, Dict

from .parameter import ParameterSpec
from .settings import OperatorSettingsMixin
from ezcv.pipeline import PipelineContext
from ezcv.typing import Image


class Operator(OperatorSettingsMixin):
    """ Class representing an image operator

    Extend this class to implement new functionality
    """
    def run(self, img: Image, ctx: PipelineContext) -> Image:
        raise NotImplementedError()

    @classmethod
    def get_parameters_specs(cls: Type['Operator']) -> Dict[str, ParameterSpec]:
        return {name: value for name, value in cls.__dict__.items() if isinstance(value, ParameterSpec)}
