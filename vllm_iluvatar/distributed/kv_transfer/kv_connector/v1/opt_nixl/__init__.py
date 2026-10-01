"""Refactored Iluvatar NIXL push connector.

Keep the package import lightweight: topology and validation unit tests should
not require the optional NIXL Python extension to be installed.
"""

from typing import Any

__all__ = ["IluNixlPushConnector", "OptNixlConnector"]


def __getattr__(name: str) -> Any:
    if name == "IluNixlPushConnector":
        from .connector import IluNixlPushConnector

        return IluNixlPushConnector
    if name == "OptNixlConnector":
        from .connector import OptNixlConnector

        return OptNixlConnector
    raise AttributeError(name)
