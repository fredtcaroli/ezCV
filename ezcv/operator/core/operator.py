import numpy as np

from ezcv.runner.context import PipelineContext


class Operator(object):
    """ Class representing an image operator

    Extend this class to implement new functionality
    """
    def run(self, img: np.ndarray, ctx: PipelineContext) -> np.ndarray:
        raise NotImplementedError()
