"""Generic discovery helpers for plugin-style sub-packages.

Provides ``discover_subclasses`` and ``get_subclass`` which scan a Python
package for concrete subclasses of a given base class.  Used by both the
inference and evaluation modules to avoid duplicating discovery logic.
"""
import importlib
import inspect
import pkgutil
from types import ModuleType


def discover_subclasses(package: ModuleType, base_class: type) -> dict[str, type]:
    """Discover all concrete subclasses of *base_class* in *package*.

    Scans every module in *package*, imports it, and collects all classes
    that are subclasses of *base_class* (excluding *base_class* itself and
    any abstract classes).

    Args:
        package: The package to scan (must have a ``__path__`` attribute).
        base_class: The abstract base class to search for subclasses of.

    Returns:
        A dict mapping class name to class for each discovered subclass.

    Raises:
        ValueError: If two modules define a subclass with the same name.
    """
    found: dict[str, type] = {}
    for _importer, modname, _ispkg in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(
            f".{modname}", package=package.__name__
        )
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, base_class)
                and obj is not base_class
                and not inspect.isabstract(obj)
            ):
                if name in found and found[name] is not obj:
                    raise ValueError(
                        f"Duplicate class name '{name}' found in module "
                        f"'{module.__name__}' (already registered from "
                        f"'{found[name].__module__}')"
                    )
                found[name] = obj
    return found


def get_subclass(name: str, package: ModuleType, base_class: type, label: str = "class") -> type:
    """Look up a single concrete subclass by name.

    Args:
        name: The class name to look up.
        package: The package to scan.
        base_class: The abstract base class.
        label: A human-readable label for error messages (e.g. "strategy").

    Returns:
        The class matching the given name.

    Raises:
        KeyError: If no subclass with the given name is found.
    """
    registry = discover_subclasses(package, base_class)
    if name not in registry:
        available = ", ".join(sorted(registry.keys())) or "(none)"
        raise KeyError(
            f"No {label} named '{name}'. Available: {available}"
        )
    return registry[name]
