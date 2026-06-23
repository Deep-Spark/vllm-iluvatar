# SPDX-License-Identifier: Apache-2.0
"""Iluvatar IR kernel providers (mirrors vllm/kernels/).

Importing this package registers the ``"iluvatar"`` IR op implementations via
the decorators in ``ilu_ops``. ``IluPlatform.import_ir_kernels`` imports it so
the provider is available before ``ir_op_priority`` is validated/applied.
"""

from . import ilu_ops

__all__ = ["ilu_ops"]
