from types import ModuleType
from unittest.mock import patch

from scripts.check_environment import module_version
from importlib.metadata import PackageNotFoundError


def test_lazy_package_version_does_not_import_fake_submodule():
    obj = ModuleType("utils3d")

    def lazy_import(name):
        raise ModuleNotFoundError(f"No module named 'utils3d.{name}'")

    obj.__getattr__ = lazy_import
    with patch("scripts.check_environment.importlib.metadata.version", return_value="0.0.2"):
        assert module_version(obj, "utils3d") == "0.0.2"


def test_source_package_without_metadata_has_unknown_version():
    with patch("scripts.check_environment.importlib.metadata.version", side_effect=PackageNotFoundError):
        assert module_version(ModuleType("local_package"), "local_package") == "unknown"


def test_explicit_version_preserves_cuda_suffix():
    obj = ModuleType("torch")
    obj.__version__ = "2.11.0+cu128"
    assert module_version(obj, "torch") == "2.11.0+cu128"
