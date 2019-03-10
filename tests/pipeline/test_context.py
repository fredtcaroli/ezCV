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
    assert np.all(original_img == ctx.original_img)


def test_pipeline_context_keeps_changes_to_original_img_inside_default_scope(ctx):
    reference = ctx.original_img.copy()
    new = build_img(ctx.original_img.shape[:2])
    ctx.original_img[:] = new
    assert not np.array_equal(reference, ctx.original_img) and np.array_equal(ctx.original_img, new)


def test_pipeline_context_keeps_changes_to_original_img_inside_same_scope(ctx):
    with ctx.scope('test_scope'):
        new = build_img(ctx.original_img.shape[:2])
        ctx.original_img[:] = new
        assert np.array_equal(ctx.original_img, new)


def test_pipeline_context_separates_original_imgs_per_scope(ctx):
    scope_img = build_img(ctx.original_img.shape[:2])
    reference = ctx.original_img.copy()

    with ctx.scope('test_scope'):
        ctx.original_img[:] = scope_img

    with ctx.scope('different_test_scope'):
        assert np.array_equal(ctx.original_img, reference)

    assert np.array_equal(ctx.original_img, reference)


def test_pipeline_context_scoped_original_img_handles_nested_calls(ctx):
    outer_img = build_img(ctx.original_img.shape[:2])
    inner_img = build_img(ctx.original_img.shape[:2])
    reference = ctx.original_img.copy()
    with ctx.scope('outer'):
        ctx.original_img[:] = outer_img
        with ctx.scope('inner'):
            assert np.array_equal(ctx.original_img, reference)
            ctx.original_img[:] = inner_img
        assert np.array_equal(ctx.original_img, outer_img)


def test_pipeline_context_scope_runs(ctx):
    ctx.scope('test_scope')


def test_pipeline_context_scope_is_a_context_manage(ctx):
    with ctx.scope('test_scope'):
        pass
