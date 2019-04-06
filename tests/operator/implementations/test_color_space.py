import numpy as np
import pytest

from ezcv.operator.implementations.color_space import ColorSpaceChange
from tests.utils import build_img


@pytest.mark.parametrize('target', ColorSpaceChange.target.possible_values)
@pytest.mark.parametrize('src', ColorSpaceChange.src.possible_values)
def test_color_space_runs_all_possible_values(ctx, src, target):
    if src == 'GRAY':
        rgb = False
    else:
        rgb = True
    img = build_img((16, 16), rgb=rgb, kind='black')
    op = ColorSpaceChange()
    op.src = src
    op.target = target
    _ = op.run(img, ctx)


@pytest.mark.parametrize('target', [
    ('BGR', np.asarray([[[50, 100, 150]]], dtype='uint8')),
    ('GRAY', np.asarray([[109]], dtype='uint8')),
    ('RGB', np.asarray([[[150, 100, 50]]], dtype='uint8')),
    ('HSV', np.asarray([[[15, 170, 150]]], dtype='uint8'))
])
@pytest.mark.parametrize('src', [
    ('BGR', np.asarray([[[50, 100, 150]]], dtype='uint8')),
    ('RGB', np.asarray([[[150, 100, 50]]], dtype='uint8')),
    ('HSV', np.asarray([[[15, 170, 150]]], dtype='uint8'))
])
def test_color_cvt_mapping(src, target, ctx):
    src_space = src[0]
    img_src = src[1]
    target_space = target[0]
    img_target = target[1]
    op = ColorSpaceChange()
    op.src = src_space
    op.target = target_space
    outp = op.run(img_src, ctx)
    assert np.all(np.isclose(outp, img_target))


@pytest.mark.parametrize('target', [
    ('BGR', np.asarray([[[109, 109, 109]]], dtype='uint8')),
    ('GRAY', np.asarray([[109]], dtype='uint8')),
    ('RGB', np.asarray([[[109, 109, 109]]], dtype='uint8')),
    ('HSV', np.asarray([[[0, 0, 109]]], dtype='uint8'))
])
def test_color_cvt_mapping_from_gray(target, ctx):
    src_space = 'GRAY'
    img_src = np.asarray([[109]], dtype='uint8')
    target_space = target[0]
    img_target = target[1]
    op = ColorSpaceChange()
    op.src = src_space
    op.target = target_space
    outp = op.run(img_src, ctx)
    assert np.all(np.isclose(outp, img_target))
