from dataclasses import dataclass
from typing import Type, Dict, Generic, TypeVar, Optional, Callable

_T = TypeVar('_T')


@dataclass(frozen=True)
class OperatorSetting(Generic[_T]):
    name: str
    default_value: _T

    def __call__(self, value: Optional[_T] = None) \
            -> Callable[[Type['OperatorSettingsMixin']], Type['OperatorSettingsMixin']]:

        def decorator(cls: Type[OperatorSettingsMixin]) -> Type[OperatorSettingsMixin]:
            if not issubclass(cls, OperatorSettingsMixin):
                raise ValueError('Can only set settings on classes that inherit from OperatorSettingsMixin')
            cls.set(self, value)
            return cls

        return decorator


class OperatorSettingsMixin:
    __settings: Dict[OperatorSetting[_T], _T]

    @classmethod
    def get(cls, setting: OperatorSetting[_T]) -> _T:
        if not cls.__is_settings_initialized() or setting not in cls.__settings:
            print('Returning default', cls.__dict__)
            return setting.default_value
        return cls.__settings[setting]

    @classmethod
    def set(cls, setting: OperatorSetting[_T], value: _T):
        if not cls.__is_settings_initialized():
            cls.__init_settings()
        cls.__settings[setting] = value

    @classmethod
    def __is_settings_initialized(cls) -> bool:
        return hasattr(cls, '_OperatorSettingsMixin__settings')

    @classmethod
    def __init_settings(cls):
        cls.__settings = dict()


GRAY_ONLY = OperatorSetting('GRAY_ONLY', False)
