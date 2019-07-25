import inspect

from ezcv.classpath import class_from_fully_qualified_name, fully_qualified_name
from .operator import Operator
from .parameter import Parameter


def create_operator(config: dict) -> Operator:
    fqn = config['implementation']
    try:
        cls = class_from_fully_qualified_name(fqn)
    except (ModuleNotFoundError, ValueError):
        raise ValueError('Invalid implementation: "%s"' % fqn)

    if not issubclass(cls, Operator):
        raise ValueError("%s is not an Operator" % fqn)

    parameters = _get_parameters_objects(cls)

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


def _get_parameters_objects(cls: type) -> dict:
    return {name: value for name, value in inspect.getmembers(cls) if isinstance(value, Parameter)}


def get_operator_config(op: Operator) -> dict:
    config = dict()
    config['implementation'] = fully_qualified_name(type(op))

    params_objs = _get_parameters_objects(type(op))
    params = dict()
    for name, _ in params_objs.items():
        params[name] = getattr(op, name)
    config['params'] = params

    return config
