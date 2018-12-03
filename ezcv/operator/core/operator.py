import numpy as np


class Operator(object):
    """ Class representing an image operator

    Extend this class to implement new functionality
    """
    def __init__(self, name: str):
        self.name = name

    def run(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
