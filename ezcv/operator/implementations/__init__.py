import importlib
import pkgutil

# We need to import all submodules so all operators are registered
# TODO: Support subpackages
for loader, name, is_pkg in pkgutil.walk_packages(__path__):
    full_name = __name__ + '.' + name
    importlib.import_module(full_name)
