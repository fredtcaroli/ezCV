import numpy as np
import pytest

from ezcv.pipeline.context import PipelineContext
from tests.utils import build_img, parametrize_img


@pytest.fixture
def original_img():
    return build_img((16, 16))


@pytest.fixture
def ctx(original_img):
    return PipelineContext(original_img)


@parametrize_img(include_valid=False, include_invalid=True)
def test_pipeline_context_invalid_img(img):
    with pytest.raises(ValueError) as e:
        PipelineContext(img)

    msg = str(e).lower()
    assert 'invalid' in msg and 'image' in msg


def test_pipeline_context_original_img(original_img, ctx):
    assert np.all(original_img == ctx.original_img())


def test_pipeline_context_protect_original_img(original_img):
    before = original_img.copy()
    ctx = PipelineContext(original_img)
    img = ctx.original_img()
    img[10, ...] = 255
    after = ctx.original_img()

    assert np.all(before == after)
