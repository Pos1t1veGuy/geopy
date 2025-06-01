import os
import importlib
import pathlib

from .core import *

PLUGINS_PATH = pathlib.Path(__file__).parent  / "plugins"

for item in PLUGINS_PATH.iterdir():
    if item.is_dir() and (item / "__init__.py").exists():
        module = importlib.import_module(f"{__name__}.plugins.{item.name}")

        for name in getattr(module, '__all__', []):
            globals()[name] = getattr(module, name)