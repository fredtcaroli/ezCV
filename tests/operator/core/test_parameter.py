from typing import Any

import pytest

from ezcv.operator import Parameter, IntegerParameter, NumberParameter


@pytest.fixture(scope='module')
def integer_param():
    return IntegerParameter(default_value=10)


@pytest.fixture(scope='module')
def number_param():
    return NumberParameter(default_value=2.5)


def test_parameter_not_implemented_from_config():
    param = Parameter(default_value=None)

    with pytest.raises(NotImplementedError):
        param.from_config(None)


def test_parameter_not_implemented_to_config():
    param = Parameter(default_value=None)

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
    9223372036854775808, # old sys.maxint + 1,
    -9223372036854775808,
    10,
    -10
])
def test_integer_parameter_to_config_valid_value(value, integer_param):
    assert value == integer_param.to_config(value)


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
def test_number_parameter_from_config_invalid_config(config, number_param):
    with pytest.raises(AssertionError):
        number_param.from_config(config)


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
def test_number_parameter_from_config_valid_config(config, number_param):
    assert config == number_param.from_config(config)


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
def test_number_parameter_to_config_invalid_config(value, number_param):
    with pytest.raises(AssertionError):
        number_param.to_config(value)


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
def test_number_parameter_to_config_valid_config(value, number_param):
    assert value == number_param.to_config(value)


def TestDefaultParameterHelper(value: Any):
    class SomeClass(object):
        param = Parameter(default_value=value)
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
        param = Parameter(default_value=default_value)

    assert isinstance(SomeClass.param, Parameter) and SomeClass.param != default_value


def test_parameter_can_get_default_value():
    default_value = object()
    param = Parameter(default_value=default_value)
    assert param.default_value is default_value
