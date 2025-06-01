import os
import importlib
import pathlib

from .core import *


PLUGINS_PATH = pathlib.Path(__file__).parent  / "plugins"


def import_plugin_by_path(path: str):
    """import plugin by path, Only names from __all__ are taken into account"""
    module = importlib.import_module(path)
    for name in getattr(module, '__all__', []):
        globals()[name] = getattr(module, name)

def import_plugins_dir(path: str):
    """import every plugin by directiory path"""
    for item in pathlib.Path(path).iterdir():
        if item.is_dir() and (item / "__init__.py").exists():
            import_plugin_by_path(f"{__name__}.plugins.{item.name}")


def load_plugins(*args: str):
    """Loads plugins from 'plugins' folder. If no arguments given — loads all plugins."""
    if args:
        for name in args:
            if (PLUGINS_PATH / name).exists():
                import_plugin_by_path(f"{__name__}.plugins.{name}")
            else:
                raise ImportError(f"'{name}' plugin is not defined")
    else:
        import_plugins_dir(PLUGINS_PATH)