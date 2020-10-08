import inspect
from typing import Dict, Type, Optional

from cerberus import Validator

from ezcv import CompVizPipeline
from ezcv.classpath import class_from_fully_qualified_name, fully_qualified_name
from ezcv.operator import Operator, ParameterSpec


ConfigSchema = Dict
Config = Dict

operator_config_schema: ConfigSchema = {
    'implementation': {
        'type': 'string',
        'required': True,
    },
    'params': {
        'type': 'dict',
        'required': True,
        'allow_unknown': True,
    }
}


pipeline_schema: ConfigSchema = {
    'version': {
        'type': 'string',
        'required': True,
    },
    'pipeline': {
        'type': 'list',
        'required': True,
        'schema': {
            'type': 'dict',
            'schema': {
                'name': {
                    'type': 'string',
                    'required': True,
                },
                'config': {
                    'type': 'dict',
                    'required': True,
                    'schema': operator_config_schema,
                }
            }
        }
    }
}


class ConfigParsingError(Exception):
    pass


def _perform_validation(config: Config, schema: ConfigSchema):
    validator = Validator(schema)
    if not validator.validate(config):
        errors = validator.errors
        raise ConfigParsingError('Failed to parse configuration', errors)


def create_pipeline(pipeline_config: Config, validate: Optional[bool] = True) -> CompVizPipeline:
    if validate:
        _perform_validation(pipeline_config, pipeline_schema)
    runner = CompVizPipeline()
    for op_config in pipeline_config['pipeline']:
        operator = create_operator(op_config['config'], validate=False)
        runner.add_operator(op_config['name'], operator)
    return runner


def create_operator(operator_config: Config, validate: Optional[bool] = True) -> Operator:
    if validate:
        _perform_validation(operator_config, operator_config_schema)

    fqn = operator_config['implementation']
    try:
        cls = class_from_fully_qualified_name(fqn)
    except (ModuleNotFoundError, ValueError) as e:
        raise ConfigParsingError(f'Invalid implementation: "{fqn}"') from e

    if not issubclass(cls, Operator):
        raise ConfigParsingError(f"{fqn} is not an Operator")

    parameters = get_parameters_specs(cls)

    op = cls()
    for name, param_config in operator_config['params'].items():
        try:
            parsed_value = parameters[name].from_config(param_config)
        except KeyError:
            raise ConfigParsingError(f'Invalid param specified trying to instantiate {fqn}: "{name}"')
        except AssertionError:
            raise ConfigParsingError(f'{type(parameters[name])} failed to parse value \'{param_config}\'')
        setattr(op, name, parsed_value)

    return op


def get_parameters_specs(op_cls: Type[Operator]) -> Dict[str, ParameterSpec]:
    return {name: value for name, value in inspect.getmembers(op_cls) if isinstance(value, ParameterSpec)}


def get_pipeline_config(pipeline: CompVizPipeline) -> Config:
    config = dict()
    config['version'] = '0.0'
    pipeline_config = list()
    for operator_name, operator in pipeline.operators.items():
        stage_config = dict()
        stage_config['name'] = operator_name
        stage_config['config'] = get_operator_config(operator)
        pipeline_config.append(stage_config)
    config['pipeline'] = pipeline_config
    return config


def get_operator_config(operator: Operator) -> Config:
    config = dict()
    config['implementation'] = fully_qualified_name(type(operator))

    params_objs = get_parameters_specs(type(operator))
    params = dict()
    for name, param_obj in params_objs.items():
        value = getattr(operator, name)
        value_config = param_obj.to_config(value)
        params[name] = value_config
    config['params'] = params

    return config
