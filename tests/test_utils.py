from ezcv.utils import is_image
from ezcv.test_utils import parametrize_img


@parametrize_img
def test_is_image_valid(img):
    assert is_image(img)


@parametrize_img(include_invalid=True, include_valid=False)
def test_is_image_invalid(img):
    assert not is_image(img)


def test_is_image_none():
    assert not is_image(None)


def test_is_image_random_object():
    assert not is_image(object())
