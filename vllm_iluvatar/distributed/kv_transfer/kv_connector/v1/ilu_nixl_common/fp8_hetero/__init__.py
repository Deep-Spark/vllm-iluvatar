# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P BF16 -> D FP8 heterogeneous-dtype push support for IluNixlPushConnector.

Kept import-light on purpose: KV cache planning imports ``wire_layout`` from the
engine-core process, which must not drag in triton kernels or CUDA staging.
"""
