import cv2

from ezcv.operator import register_operator
from ezcv.operator.core import Operator, IntegerParameter, DoubleParameter
from ezcv.pipeline import PipelineContext
from ezcv.typing import Image


@register_operator
class GaussianBlur(Operator):
    """ Applies a Gaussian blur to the image

    Parameters:
        - kernel_size: Gaussian kernel size
    """
    kernel_size = IntegerParameter(default_value=3, lower=3, upper=25)
    sigma = DoubleParameter(default_value=1.5, lower=0, upper=15)

    def run(self, img: Image, ctx: PipelineContext) -> Image:
        odd_kernel_size = self.kernel_size - ((self.kernel_size + 1) % 2)
        return cv2.GaussianBlur(img, (odd_kernel_size, odd_kernel_size), self.sigma)
