"""Every `aspd.cli` entry point and `aspd.calibrate` imports (their lab imports resolve)."""

import importlib
import pkgutil

import pytest

import aspd.cli

MODULES = ["aspd.calibrate"] + [f"aspd.cli.{m.name}" for m in pkgutil.iter_modules(aspd.cli.__path__)]


@pytest.mark.parametrize("module", MODULES)
def test_entry_point_imports(module: str):
    importlib.import_module(module)
