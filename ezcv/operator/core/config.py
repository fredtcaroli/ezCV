import inspect
from typing import Dict, Type

from ezcv.classpath import class_from_fully_qualified_name, fully_qualified_name
from ezcv.operator.core.operator import Operator
from ezcv.operator.core.parameter import ParameterSpec


def create_operator(config: dict) -> Operator:
    fqn = config['implementation']
    try:
        cls = class_from_fully_qualified_name(fqn)
    except (ModuleNotFoundError, ValueError):
        raise ValueError('Invalid implementation: "%s"' % fqn)

    if not issubclass(cls, Operator):
        raise ValueError("%s is not an Operator" % fqn)

    parameters = get_parameters_specs(cls)

    op = cls()
    for name, param_config in config['params'].items():
        try:
            parsed_value = parameters[name].from_config(param_config)
        except KeyError:
            raise ValueError('Invalid param specified trying to instantiate %s: "%s"' % (fqn, name))
        except AssertionError:
            raise ValueError('%s failed to parse value \'%s\'' % (str(type(parameters[name])), str(param_config)))
        setattr(op, name, parsed_value)

    return op


def get_parameters_specs(op_cls: Type[Operator]) -> Dict[str, ParameterSpec]:
    return {name: value for name, value in inspect.getmembers(op_cls) if isinstance(value, ParameterSpec)}


def get_operator_config(op: Operator) -> dict:
    config = dict()
    config['implementation'] = fully_qualified_name(type(op))

    params_objs = get_parameters_specs(type(op))
    params = dict()
    for name, param_obj in params_objs.items():
        value = getattr(op, name)
        value_config = param_obj.to_config(value)
        params[name] = value_config
    config['params'] = params

    return config
