import cv2

from ezcv.operator.core import Operator, IntegerParameter, NumberParameter


class GaussianBlur(Operator):
    """ Applies a Gaussian blur to the image

    Parameters:
        - kernel_size: Gaussian kernel size
    """
    kernel_size = IntegerParameter()
    sigma = NumberParameter()

    def run(self, img):
        odd_kernel_size = self.kernel_size - ((self.kernel_size + 1) % 2)
        return cv2.GaussianBlur(img, (odd_kernel_size, odd_kernel_size), self.sigma)
