from importlib import import_module


def fully_qualified_name(cls: type) -> str:
    """ Returns the fully qualified name of a class
    """
    return cls.__module__ + '.' + cls.__qualname__


def class_from_fully_qualified_name(fqn: str) -> type:
    """ Returns the class specified by its fully qualified name
    """
    parts = fqn.rsplit('.', 1)
    if len(parts) < 2:
        raise ValueError('Invalid FQN "%s". Specify the class name like so: "package.module.class"' % fqn)
    module, cls = parts
    cls = getattr(import_module(module), cls)
    return cls
