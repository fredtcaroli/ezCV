import cv2

from ezcv.operator import register_operator, Operator, IntegerParameter, DoubleParameter
from ezcv.pipeline import PipelineContext
from ezcv.typing import Image


@register_operator
class CLAHE(Operator):
    clip_limit = DoubleParameter(default_value=2, lower=0.1, upper=10)
    tile_grid_size = IntegerParameter(default_value=8, lower=1, upper=50)

    def run(self, img: Image, ctx: PipelineContext) -> Image:
        # TODO: Use a color_space parameter to determine the right color space
        if img.ndim == 3:
            return self._run_clahe_bgr(img)
        else:
            return self._run_clahe_gray(img)

    def _run_clahe_bgr(self, img: Image) -> Image:
        clahe = self._create_clahe()
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        light_channel = lab_img[..., 0]
        lab_img[..., 0] = clahe.apply(light_channel)
        return cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

    def _run_clahe_gray(self, img: Image) -> Image:
        clahe = self._create_clahe()
        return clahe.apply(img)

    def _create_clahe(self) -> cv2.CLAHE:
        return cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile_grid_size, self.tile_grid_size))
