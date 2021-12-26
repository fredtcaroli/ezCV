from ezcv.operator import get_available_operators
from ezcv.operator.core import settings
from ezcv.pipeline import PipelineContext
from ezcv.test_utils import build_img
from ezcv.utils import is_image


def test_implementations_returns():
    """ All implementations are expected to return a valid image when all parameters are set to default
    """
    operators_classes = get_available_operators()
    for operator_class in operators_classes:
        rgb = not operator_class.get(settings.GRAY_ONLY)
        test_img = build_img(size=(50, 50), kind='black', rgb=rgb)
        operator = operator_class()
        test_context = PipelineContext(test_img)
        return_img = operator.run(test_img, test_context)
        assert is_image(return_img)
