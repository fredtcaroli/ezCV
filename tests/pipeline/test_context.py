import numpy as np
import pytest

from ezcv.pipeline.context import PipelineContext
from tests.utils import build_img, parametrize_img, assert_terms_in_exception


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


def test_pipeline_context_scope_is_a_context_manager(ctx):
    with ctx.scope('test_scope'):
        pass


def test_pipeline_context_add_info_runs(ctx):
    info_object = object()
    ctx.add_info('info_name', info_object)


def test_pipeline_context_add_info_repeated_names(ctx):
    info_object = object()
    name = 'info_name'
    ctx.add_info(name, info_object)
    with pytest.raises(ValueError) as e:
        another_object = object()
        ctx.add_info(name, another_object)

    assert_terms_in_exception(e, ['duplicated', 'name'])


@pytest.mark.parametrize('invalid_name', [
    None,
    '',
    123,
    12.3,
    object(),
    'no/slashes'
])
def test_pipeline_context_invalid_name(ctx, invalid_name):
    with pytest.raises(ValueError) as e:
        ctx.add_info(invalid_name, object())

    assert_terms_in_exception(e, ['invalid', 'name'])


def test_pipeline_context_add_info_reflects_info(ctx):
    name = 'info_name'
    info_object = object
    ctx.add_info(name, info_object)
    assert name in ctx.info
    assert ctx.info[name] is info_object


def test_pipeline_context_add_info_inside_scope(ctx):
    info_name = 'info_name'
    scope_name = 'scope_name'
    info_object = object()
    with ctx.scope(scope_name):
        ctx.add_info(info_name, info_object)
    assert scope_name in ctx.info and info_name in ctx.info[scope_name]
    assert ctx.info[scope_name][info_name] is info_object


def test_pipeline_context_add_info_nested_scope(ctx):
    info_name = 'info_name'
    outer_scope_name = 'outer_scope'
    inner_scope_name_1 = 'inner_scope_1'
    inner_scope_name_2 = 'inner_scope_2'
    info_object = object()
    with ctx.scope(outer_scope_name):
        with ctx.scope(inner_scope_name_1):
            ctx.add_info(info_name, info_object)
        with ctx.scope(inner_scope_name_2):
            ctx.add_info(info_name, info_object)

    assert ctx.info == {
        outer_scope_name: {
            inner_scope_name_1: {
                info_name: info_object
            },
            inner_scope_name_2: {
                info_name: info_object
            }
        }
    }


