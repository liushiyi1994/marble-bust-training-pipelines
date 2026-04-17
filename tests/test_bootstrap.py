import importlib


def test_core_and_backends_packages_import():
    assert importlib.import_module("core").__name__ == "core"
    assert importlib.import_module("backends").__name__ == "backends"
