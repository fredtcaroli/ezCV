from typing import Any

import pytest

from ezcv.operator import ParameterSpec, IntegerParameter, DoubleParameter, EnumParameter
from ezcv.test_utils import assert_terms_in_exception


@pytest.fixture(scope='module')
def integer_param():
    return IntegerParameter(default_value=10, lower=5, upper=15)


@pytest.fixture(scope='module')
def double_param():
    return DoubleParameter(default_value=2.5, lower=1, upper=10)


def test_parameter_not_implemented_from_config():
    param = ParameterSpec(default_value=None)

    with pytest.raises(NotImplementedError):
        param.from_config(None)


def test_parameter_not_implemented_to_config():
    param = ParameterSpec(default_value=None)

    with pytest.raises(NotImplementedError):
        param.to_config(None)


@pytest.mark.parametrize('config', [
    '',
    'not-a-number',
    None,
    '2.5',
    '10',
    'ten',
    tuple(),
    list(),
    2.5
])
def test_integer_parameter_from_config_invalid_config(config, integer_param):
    with pytest.raises(AssertionError):
        integer_param.from_config(config)


@pytest.mark.parametrize('config', [
    0,
    -1,
    1,
    2147483648,  # 2**31
    -2147483648,  # -2**31
    2147483647,  # 2**31 - 1
    -2147483647,  # -2**31 + 1
    10,
    -10
])
def test_integer_parameter_from_config_valid_config(config, integer_param):
    assert config == integer_param.from_config(config)


@pytest.mark.parametrize('value', [
    '',
    'not-a-number',
    None,
    '2.5',
    '10',
    'ten',
    tuple(),
    list(),
    2.5
])
def test_integer_parameter_to_config_invalid_value(value, integer_param):
    with pytest.raises(AssertionError):
        integer_param.to_config(value)


@pytest.mark.parametrize('value', [
    0,
    -1,
    1,
    2147483648,  # 2**31
    -2147483648,  # -2**31
    2147483647,  # 2**31 - 1
    -2147483647,  # -2**31 + 1
    9223372036854775808,  # old sys.maxint + 1,
    -9223372036854775808,
    10,
    -10
])
def test_integer_parameter_to_config_valid_value(value, integer_param):
    assert value == integer_param.to_config(value)


@pytest.mark.parametrize('limits', [
    (0, 10),
    (2, 5),
    (-10, 10),
    (-10, -5)
])
def test_integer_parameter_valid_lower_upper_limits(limits):
    IntegerParameter(default_value=5, lower=limits[0], upper=limits[1])


@pytest.mark.parametrize('limits', [
    (1, 1),
    (5, 4),
    ('not a number', 5),
    (5, 'not a number')
])
def test_integer_parameter_invalid_lower_higher_limits(limits):
    with pytest.raises(ValueError) as e:
        IntegerParameter(default_value=5, lower=limits[0], upper=limits[1])
    assert_terms_in_exception(e, ['invalid'])


@pytest.mark.parametrize('default_value', [
    0,
    -1,
    1,
    2147483648,  # 2**31
    -2147483648,  # -2**31
    2147483647,  # 2**31 - 1
    -2147483647,  # -2**31 + 1
    9223372036854775808,  # old sys.maxint + 1,
    -9223372036854775808
])
def test_integer_parameter_valid_default_value(default_value):
    IntegerParameter(default_value=default_value, lower=0, upper=10)


@pytest.mark.parametrize('default_value', [
    '',
    'not-a-number',
    None,
    '2.5',
    '10',
    'ten',
    tuple(),
    list()
])
def test_integer_parameter_invalid_default_value(default_value):
    with pytest.raises(ValueError) as e:
        IntegerParameter(default_value=default_value, lower=0, upper=10)
    assert_terms_in_exception(e, ['invalid', 'default'])


@pytest.mark.parametrize('config', [
    '',
    'not-a-number',
    None,
    '2.5',
    '10',
    'ten',
    tuple(),
    list()
])
def test_double_parameter_from_config_invalid_config(config, double_param):
    with pytest.raises(AssertionError):
        double_param.from_config(config)


@pytest.mark.parametrize('config', [
    0,
    -1,
    1,
    2147483648,  # 2**31
    -2147483648,  # -2**31
    2147483647,  # 2**31 - 1
    -2147483647,  # -2**31 + 1
    10,
    -10,
    0.0,
    -3.0,
    3.0,
    2147483648.0,  # 2**31
    -2147483648.0,  # -2**31
    2147483647.0,  # 2**31 - 1
    -2147483647.0,  # -2**31 + 1
    10.0,
    -10.0,
    2.5,
    float('inf'),
    float('-inf')
])
def test_double_parameter_from_config_valid_config(config, double_param):
    assert config == double_param.from_config(config)


@pytest.mark.parametrize('value', [
    '',
    'not-a-number',
    None,
    '2.5',
    '10',
    'ten',
    tuple(),
    list()
])
def test_double_parameter_to_config_invalid_config(value, double_param):
    with pytest.raises(AssertionError):
        double_param.to_config(value)


@pytest.mark.parametrize('value', [
    0,
    -1,
    1,
    2147483648,  # 2**31
    -2147483648,  # -2**31
    2147483647,  # 2**31 - 1
    -2147483647,  # -2**31 + 1
    10,
    -10,
    0.0,
    -3.0,
    3.0,
    2147483648.0,  # 2**31
    -2147483648.0,  # -2**31
    2147483647.0,  # 2**31 - 1
    -2147483647.0,  # -2**31 + 1
    10.0,
    -10.0,
    2.5,
    float('inf'),
    float('-inf')
])
def test_double_parameter_to_config_valid_config(value, double_param):
    assert value == double_param.to_config(value)


@pytest.mark.parametrize('limits', [
    (0, 10),
    (-10, 10),
    (-10, -5),
    (2.3, 5.4),
    (-10.2, 2.4)
])
def test_double_parameter_valid_upper_lower_limits(limits):
    DoubleParameter(default_value=5, lower=limits[0], upper=limits[1])


@pytest.mark.parametrize('limits', [
    (3.0, 3.0),
    (5, 4),
    ('not a number', 5),
    (5, 'not a number')
])
def test_double_parameter_invalid_lower_upper_limits(limits):
    with pytest.raises(ValueError) as e:
        DoubleParameter(default_value=5, lower=limits[0], upper=limits[1])

    assert_terms_in_exception(e, ['invalid'])


@pytest.mark.parametrize('value', [
    10,
    -10,
    0.0,
    -3.0,
    3.0,
    2147483648.0,  # 2**31
    -2147483648.0,  # -2**31
    2147483647.0,  # 2**31 - 1
    -2147483647.0,  # -2**31 + 1
    10.0,
    -10.0,
    2.5,
    float('inf'),
    float('-inf')
])
def test_double_parameter_valid_default_value(value):
    DoubleParameter(default_value=value, lower=0, upper=5)


@pytest.mark.parametrize('value', [
    '',
    'not-a-number',
    None,
    '2.5',
    '10',
    'ten',
    tuple(),
    list()
])
def test_double_parameter_invalid_default_value(value):
    with pytest.raises(ValueError) as e:
        DoubleParameter(default_value=value, lower=0, upper=5)
    assert_terms_in_exception(e, ['invalid', 'default'])


def TestDefaultParameterHelper(value: Any):
    class SomeClass(object):
        param = ParameterSpec(default_value=value)
    return SomeClass()


@pytest.mark.parametrize('value', [
    None,
    object(),
    object,
    10,
    2.5,
    'foo'
])
def test_parameter_default_value_return(value):
    obj = TestDefaultParameterHelper(value)
    assert obj.param == value


def test_parameter_default_value_not_returned_when_there_is_no_instance():
    default_value = object()

    class SomeClass(object):
        param = ParameterSpec(default_value=default_value)

    assert isinstance(SomeClass.param, ParameterSpec) and SomeClass.param != default_value


def test_parameter_can_get_default_value():
    default_value = object()
    param = ParameterSpec(default_value=default_value)
    assert param.default_value is default_value


@pytest.mark.parametrize('value', [
    '1',
    '2',
    '3'
])
def test_enum_parameter_from_config_valid(value):
    param = EnumParameter(['1', '2', '3'], '1')
    assert param.from_config(value) == value


@pytest.mark.parametrize('value', [
    1,
    2,
    3,
    '0',
    '4',
    None,
    object(),
    object
])
def test_enum_invalid_from_config_invalid(value):
    param = EnumParameter(['1', '2', '3'], default_value='1')
    with pytest.raises(AssertionError):
        param.from_config(value)


@pytest.mark.parametrize('value', [
    '1',
    '2',
    '3'
])
def test_enum_parameter_to_config_valid(value):
    param = EnumParameter(['1', '2', '3'], '1')
    assert param.to_config(value) == value


@pytest.mark.parametrize('value', [
    1,
    2,
    3,
    '0',
    '4',
    None,
    object(),
    object
])
def test_enum_invalid_to_config_invalid(value):
    param = EnumParameter(['1', '2', '3'], default_value='1')
    with pytest.raises(AssertionError):
        param.to_config(value)


@pytest.mark.parametrize('value', [
    1,
    2,
    3,
    '0',
    '4',
    None,
    object(),
    object
])
def test_enum_invalid_default_param(value):
    with pytest.raises(ValueError) as e:
        _ = EnumParameter(['1', '2', '3'], default_value=value)

    assert_terms_in_exception(e, ['invalid', 'default', 'value'])


def test_enum_exposes_possible_values():
    possible_values = ['1', '2', '3']
    param = EnumParameter(possible_values=possible_values, default_value='1')
    assert param.possible_values == possible_values
