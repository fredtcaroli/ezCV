import numpy as np
import pytest

from ezcv.operator.implementations.blur import GaussianBlur
from ezcv.runner.context import PipelineContext
from tests.utils import parametrize_img, build_img
from ezcv.utils import is_image


@pytest.fixture
def blur_op():
    blur = GaussianBlur()
    blur.kernel_size = 3
    blur.sigma = 1.5
    return blur


@pytest.fixture
def ctx():
    return PipelineContext(build_img((16, 16)))


@parametrize_img
def test_gaussian_blur_run_return_img(blur_op, img, ctx):
    r = blur_op.run(img, ctx)
    assert is_image(r)


@parametrize_img
def test_gaussian_blur_run_return_same_shape(blur_op, img, ctx):
    r = blur_op.run(img, ctx)
    assert r.shape == img.shape


@parametrize_img
def test_gaussian_blur_run_leave_input_untouched(blur_op, img, ctx):
    before_run = img.copy()
    _ = blur_op.run(img, ctx)
    assert np.array_equal(before_run, img)


def test_gaussian_blur_run_indicator_that_works(blur_op, ctx):
    img = build_img((301, 301), rgb=False, kind='black')
    img[150, ...] = 255
    blurred = blur_op.run(img, ctx)
    assert np.all(blurred[150, ...] < 255)
    assert np.all(blurred[149, ...] > 0) and np.all(blurred[151, ...] > 0)


@parametrize_img
def test_gaussian_blur_run_normalize_kernel_size(blur_op, img, ctx):
    blur_op.kernel_size = 3
    result_odd = blur_op.run(img, ctx)
    blur_op.kernel_size = 4
    result_even = blur_op.run(img, ctx)
    assert np.all(result_even == result_odd)
