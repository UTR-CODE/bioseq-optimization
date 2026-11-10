"""Optional native kernels: installations without a C compiler retain Python fallbacks."""
from setuptools import Extension, setup

setup(ext_modules=[Extension(
    "sequence_optimization._sequence_kernels",
    ["sequence_optimization/_sequence_kernels.c"],
    optional=True,
)])
