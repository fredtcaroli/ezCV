import pytest

from ezcv.operator.core.settings import OperatorSettingsMixin, OperatorSetting

TEST_SETTING: OperatorSetting[int] = OperatorSetting('TEST_SETTING', 42)


@pytest.fixture(scope='function')
def test_class():
    class Foo(OperatorSettingsMixin):
        pass
    return Foo


class TestOperatorSettingsMixin:
    def test_happy_path(self, test_class):
        test_class.set(TEST_SETTING, 10)
        assert test_class.get(TEST_SETTING) == 10

    def test_set_twice(self, test_class):
        test_class.set(TEST_SETTING, 10)
        test_class.set(TEST_SETTING, 20)
        assert test_class.get(TEST_SETTING) == 20

    def test_get_default(self, test_class):
        assert test_class.get(TEST_SETTING) == 42

    def test_decorator_set(self):
        @TEST_SETTING(30)
        class Foo(OperatorSettingsMixin):
            pass
        assert Foo.get(TEST_SETTING) == 30

    def test_decorator_set_without_mixin(self):
        with pytest.raises(ValueError):
            @TEST_SETTING(30)
            class Foo(object):
                pass
