# SPDX-License-Identifier: Apache-2.0
"""Registry for model-specific Iluvatar quantization methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import overload

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

logger = init_logger(__name__)

QuantMethodClass = type[QuantizeMethodBase]


@dataclass
class _IluvatarQuantRegistry:
    methods: dict[str, QuantMethodClass] = field(default_factory=dict)

    @overload
    def register_method(self, method_cls: QuantMethodClass, /) -> QuantMethodClass: ...

    @overload
    def register_method(
        self, method_name: str, method_cls: QuantMethodClass, /
    ) -> QuantMethodClass: ...

    @overload
    def register_method(
        self, method_name: str | None = None, /, *, name: str | None = None
    ): ...

    def register_method(
        self,
        method_name: str | QuantMethodClass | None = None,
        method_cls: QuantMethodClass | None = None,
        *,
        name: str | None = None,
    ):
        """Register a quantization method directly or as a decorator.

        Supported forms::

            @IluvatarQuantRegistry.register_method
            @IluvatarQuantRegistry.register_method()
            @IluvatarQuantRegistry.register_method("custom_name")
            @IluvatarQuantRegistry.register_method(name="custom_name")
            IluvatarQuantRegistry.register_method("custom_name", Method)
        """
        if name is not None and isinstance(method_name, str):
            raise TypeError("Specify the method name either positionally or by keyword")

        if isinstance(method_name, type):
            if method_cls is not None:
                raise TypeError("A method class was provided more than once")
            return self._register(name or method_name.__name__, method_name)

        if method_name is not None and not isinstance(method_name, str):
            raise TypeError("Method name must be a string")

        registration_name = name or method_name

        if method_cls is not None:
            if registration_name is None:
                registration_name = method_cls.__name__
            return self._register(registration_name, method_cls)

        def decorator(decorated_cls: QuantMethodClass) -> QuantMethodClass:
            return self._register(
                registration_name or decorated_cls.__name__, decorated_cls
            )

        return decorator

    def _register(
        self, method_name: str, method_cls: QuantMethodClass
    ) -> QuantMethodClass:
        if not isinstance(method_cls, type) or not issubclass(
            method_cls, QuantizeMethodBase
        ):
            raise TypeError(
                "`method_cls` must be a QuantizeMethodBase subclass, "
                f"not {method_cls!r}"
            )

        if method_name in self.methods:
            logger.debug(
                "Quantization method %s is already registered and will be "
                "overwritten by %s.",
                method_name,
                method_cls,
            )
        self.methods[method_name] = method_cls
        return method_cls

    def get_method(self, method_name: str) -> QuantMethodClass | None:
        if not isinstance(method_name, str):
            raise TypeError("Method name must be a string")
        return self.methods.get(method_name)

    def create_runtime_method(
        self, method_name: str, context: object
    ) -> QuantizeMethodBase:
        """Instantiate a method through the runtime-conversion contract."""
        method_cls = self.get_method(method_name)
        if method_cls is None:
            raise KeyError(
                f"Iluvatar quantization method {method_name!r} is not "
                f"registered; registered methods: {sorted(self.methods)}"
            )
        factory = getattr(method_cls, "from_ilu_runtime_config", None)
        if not callable(factory):
            raise TypeError(
                f"Iluvatar quantization method {method_name!r} does not "
                "implement from_ilu_runtime_config(context)"
            )
        method = factory(context)
        if not isinstance(method, method_cls):
            raise TypeError(
                f"{method_name}.from_ilu_runtime_config returned "
                f"{type(method).__name__}, expected {method_cls.__name__}"
            )
        return method


IluvatarQuantRegistry = _IluvatarQuantRegistry()
