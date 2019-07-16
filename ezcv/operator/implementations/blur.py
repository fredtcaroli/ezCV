import cv2
import numpy as np

from ezcv.operator import register_operator
from ezcv.operator.core import Operator, IntegerParameter, NumberParameter
from ezcv.pipeline import PipelineContext


@register_operator
class GaussianBlur(Operator):
    """ Applies a Gaussian blur to the image

    Parameters:
        - kernel_size: Gaussian kernel size
    """
    kernel_size = IntegerParameter(default_value=3)
    sigma = NumberParameter(default_value=1.5)

    def run(self, img: np.ndarray, ctx: PipelineContext):
        odd_kernel_size = self.kernel_size - ((self.kernel_size + 1) % 2)
        return cv2.GaussianBlur(img, (odd_kernel_size, odd_kernel_size), self.sigma)
