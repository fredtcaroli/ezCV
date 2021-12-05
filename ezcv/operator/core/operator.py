import numpy as np

from ezcv.pipeline import PipelineContext
from ezcv.typing import Image


class Operator(object):
    """ Class representing an image operator

    Extend this class to implement new functionality
    """

    only_gray: bool = False

    def run(self, img: Image, ctx: PipelineContext) -> Image:
        raise NotImplementedError()
