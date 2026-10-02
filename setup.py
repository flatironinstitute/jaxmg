# setup.py
from setuptools import setup
from setuptools.dist import Distribution

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools < 70.1
    from wheel.bdist_wheel import bdist_wheel


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        # Tell wheel/setuptools: this project contains native (non-pure) code
        return True

    def is_pure(self):
        # Explicitly say it's not a pure Python package
        return False


class PlatformWheel(bdist_wheel):
    def get_tag(self):
        # The native backend is loaded with ctypes and does not use the Python
        # C API, so a single py3-none-<platform> wheel serves every Python.
        _, _, plat = super().get_tag()
        return "py3", "none", plat


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": PlatformWheel})
