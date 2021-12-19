from typing import Type, Dict

from .parameter import ParameterSpec
from ezcv.pipeline import PipelineContext
from ezcv.typing import Image


class Operator(object):
    """ Class representing an image operator

    Extend this class to implement new functionality
    """

    only_gray: bool = False

    def run(self, img: Image, ctx: PipelineContext) -> Image:
        raise NotImplementedError()

    @classmethod
    def get_parameters_specs(cls: Type['Operator']) -> Dict[str, ParameterSpec]:
        return {name: value for name, value in cls.__dict__.items() if isinstance(value, ParameterSpec)}
