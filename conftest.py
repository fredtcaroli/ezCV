import pytest

from ezcv.pipeline import PipelineContext
from ezcv.test_utils import build_img


@pytest.fixture
def ctx():
    return PipelineContext(build_img((16, 16)))
