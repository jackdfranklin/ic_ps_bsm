from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "transport_solver",
        ["src/transport_solver.cpp"],
    ),
]
