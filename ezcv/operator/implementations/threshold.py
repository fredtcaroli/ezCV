import cv2

from ezcv.operator import register_operator, Operator, EnumParameter, IntegerParameter, BooleanParameter
from ezcv.pipeline import PipelineContext
from ezcv.typing import Image


threshold_types = [
    'THRESH_BINARY',
    'THRESH_BINARY_INV',
    'THRESH_TRUNC',
    'THRESH_TOZERO',
    'THRESH_TOZERO_INV'
]


@register_operator
class SimpleThreshold(Operator):

    only_gray = True

    threshold_type = EnumParameter(
        possible_values=threshold_types,
        default_value=threshold_types[0]
    )
    otsu = BooleanParameter(default_value=False)
    threshold_value = IntegerParameter(default_value=127, lower=0, upper=255)
    max_value = IntegerParameter(default_value=255, lower=0, upper=255)

    def run(self, img: Image, ctx: PipelineContext) -> Image:
        type_flag = getattr(cv2, self.threshold_type)
        if self.otsu:
            type_flag += cv2.THRESH_OTSU
        _, thresh = cv2.threshold(img, self.threshold_value, self.max_value, type_flag)
        return thresh


@register_operator
class AdaptiveThreshold(Operator):

    only_gray = True

    threshold_type = EnumParameter(
        possible_values=threshold_types,
        default_value=threshold_types[0]
    )
    adaptive_method = EnumParameter(
        possible_values=[
            "ADAPTIVE_THRESH_MEAN_C",
            "ADAPTIVE_THRESH_GAUSSIAN_C"
        ],
        default_value="ADAPTIVE_THRESH_MEAN_C"
    )
    max_value = IntegerParameter(default_value=255, lower=0, upper=255)
    block_size = IntegerParameter(default_value=9, lower=1, upper=50, step_size=2)
    C = IntegerParameter(default_value=5, lower=1, upper=50)

    def run(self, img: Image, ctx: PipelineContext) -> Image:
        type_flag = getattr(cv2, self.threshold_type)
        method_flag = getattr(cv2, self.adaptive_method)
        thresh = cv2.adaptiveThreshold(img, self.max_value, method_flag, type_flag, self.block_size, self.C)
        return thresh
