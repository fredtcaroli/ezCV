import numpy as np


class Operator(object):
    """ Class representing an image operator

    Extend this class to implement new functionality
    """
    def run(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
