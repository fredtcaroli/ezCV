import functools
from typing import Callable

import cv2
import numpy as np

from ezcv.operator import Operator, EnumParameter, register_operator
from ezcv.pipeline_context import PipelineContext


def _space_transformer(keys) -> Callable[[np.ndarray], np.ndarray]:
    if not isinstance(keys, list):
        keys = [keys]

    transforms = list()
    for cvt_key in keys:
        if cvt_key is None:
            transforms.append(lambda _: _)
        else:
            transforms.append(functools.partial(cv2.cvtColor, code=cvt_key))

    def all_transforms(img: np.ndarray) -> np.ndarray:
        last = img
        for transform in transforms:
            last = transform(last)
        return last

    return all_transforms


@register_operator
class ColorSpaceChange(Operator):
    """ Changes the image's color space

    Parameters:
        - src: The color space before the change
        - target: Target color space
    """

    src = EnumParameter(
        possible_values=['RGB', 'BGR', 'GRAY', 'HSV'],
        default_value='BGR'
    )

    target = EnumParameter(
        possible_values=['RGB', 'BGR', 'GRAY', 'HSV'],
        default_value='GRAY'
    )

    _space_transformer_dict = {
        'RGB': {
            'RGB': _space_transformer(None),
            'BGR': _space_transformer(cv2.COLOR_RGB2BGR),
            'GRAY': _space_transformer(cv2.COLOR_RGB2GRAY),
            'HSV': _space_transformer(cv2.COLOR_RGB2HSV)
        },
        'BGR': {
            'RGB': _space_transformer(cv2.COLOR_BGR2RGB),
            'BGR': _space_transformer(None),
            'GRAY': _space_transformer(cv2.COLOR_BGR2GRAY),
            'HSV': _space_transformer(cv2.COLOR_BGR2HSV)
        },
        'GRAY': {
            'RGB': _space_transformer(cv2.COLOR_GRAY2RGB),
            'BGR': _space_transformer(cv2.COLOR_GRAY2BGR),
            'GRAY': _space_transformer(None),
            'HSV': _space_transformer([cv2.COLOR_GRAY2BGR, cv2.COLOR_BGR2HSV])
        },
        'HSV': {
            'RGB': _space_transformer(cv2.COLOR_HSV2RGB),
            'BGR': _space_transformer(cv2.COLOR_HSV2BGR),
            'GRAY': _space_transformer([cv2.COLOR_HSV2BGR, cv2.COLOR_BGR2GRAY]),
            'HSV': _space_transformer(None)
        }
    }

    def run(self, img: np.ndarray, ctx: PipelineContext) -> np.ndarray:
        return self._space_transformer_dict[self.src][self.target](img)
