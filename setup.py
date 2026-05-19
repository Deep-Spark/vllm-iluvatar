"""Setuptools shim.

All real metadata lives in ``pyproject.toml``. This file exists so
``pip install -e .`` works on older toolchains and because some CI / build
images still call ``python setup.py develop`` directly.
"""

from setuptools import setup

setup()
