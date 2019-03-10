import numpy as np

from ezcv.utils import is_image


class PipelineContext(object):
    def __init__(self, original_img: np.ndarray):
        if not is_image(original_img):
            raise ValueError('Invalid original image')
        self._original_img = original_img

    def original_img(self):
        return self._original_img.copy()
