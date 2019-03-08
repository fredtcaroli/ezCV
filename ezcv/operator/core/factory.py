import inspect
from typing import Iterator

from ezcv.classpath import class_from_fully_qualified_name
from ezcv.operator.core.operator import Operator
from ezcv.operator.core.parameter import Parameter


def create_operator(config: dict) -> Operator:
    fqn = config['implementation']
    try:
        cls = class_from_fully_qualified_name(fqn)
    except (ModuleNotFoundError, ValueError):
        raise ValueError('Invalid implementation: "%s"' % fqn)

    if not issubclass(cls, Operator):
        raise ValueError("%s is not an Operator" % fqn)

    parameters = {name: value for name, value in inspect.getmembers(cls) if isinstance(value, Parameter)}
    _check_missing_params(config, list(parameters.keys()), fqn)

    op = cls()
    for name, param_config in config['params'].items():
        try:
            parsed_value = parameters[name].from_config(param_config)
        except KeyError:
            raise ValueError('Invalid param specified trying to instantiate %s: "%s"' % (fqn, name))
        setattr(op, name, parsed_value)

    return op


def _check_missing_params(config: dict, parameters: Iterator[str], fqn: str):
    specified_params = set(param_name for param_name in config['params'].keys())
    missing_params = [param_name for param_name in parameters if param_name not in specified_params]
    if len(missing_params) > 0:
        raise ValueError('Some parameters appear to be missing while instantiating %s. The parameters are: %s' % (
            fqn, str(missing_params)
        ))
