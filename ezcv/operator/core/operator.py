import numpy as np

from ezcv.pipeline import PipelineContext
from ezcv.typing import Image


class Operator(object):
    """ Class representing an image operator

    Extend this class to implement new functionality
    """
    def run(self, img: Image, ctx: PipelineContext) -> Image:
        raise NotImplementedError()
