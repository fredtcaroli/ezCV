from ezcv.exceptions import OperatorFailedError
from ezcv.operator import Operator
from ezcv.operator.core import settings
from ezcv.pipeline import PipelineContext
from ezcv.typing import Image


class PipelineHook:
    def before_operator(self, operator: Operator, img: Image, ctx: PipelineContext):
        pass

    def after_operator(self, operator: Operator, img: Image, ctx: PipelineContext):
        pass

    def before_pipeline(self, ctx: PipelineContext):
        pass

    def after_pipeline(self, img: Image, ctx: PipelineContext):
        pass


class GrayOnlyHook(PipelineHook):
    """ Makes sure the GRAY_ONLY setting is being followed
    """
    def before_operator(self, operator: Operator, img: Image, ctx: PipelineContext):
        if operator.get(settings.GRAY_ONLY) is True and img.ndim > 2:
            raise OperatorFailedError(f'Operator {operator.__class__.__name__} expects a gray image')
