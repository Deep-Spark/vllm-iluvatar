# SPDX-License-Identifier: Apache-2.0
"""Mirror of ``vllm.model_executor`` for Iluvatar OOT layer/model implementations.

Subpackages mirror the upstream tree (``layers/`` -> ``vllm.model_executor.layers``,
``models/`` -> ``vllm.model_executor.models``) so out-of-tree CustomOp / model
overrides can be diffed against the community source file-by-file.  Registration
is driven by the dedicated OOT layer (see ``vllm_iluvatar.ilu_register``), not the
``ilu_patch`` general-patch framework.
"""
