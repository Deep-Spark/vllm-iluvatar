# SPDX-License-Identifier: Apache-2.0
"""Step5 MTP draft positions and FULL padded token-count metadata.

Target      : vllm.v1.worker.gpu.spec_decode.speculator.DraftModelSpeculator.
              _build_attn_metadata
Baseline    : vLLM v0.31.0; function SHA256 is BASELINE_SHA256 in the override.
Why         : draft attention builders require current device positions, and
              FULL graphs require the padded token count in their metadata.
How         : replace only this method after signature and fingerprint checks;
              the override retains upstream flow with the metadata delta.
Scope       : registered by the Iluvatar general plugin for draft execution.
Remove      : when upstream supplies equivalent draft positions and padded
              metadata through the supported attention-builder interface.
Fingerprint : vllm_iluvatar.v1.worker.gpu.spec_decode.speculator.BASELINE_SHA256;
              checked against inspect.getsource before replacing the method.
"""

import inspect
from types import SimpleNamespace

from vllm_iluvatar.ilu_patch import general_patch as p


def register():
    p.register_once(__name__, _apply, detail="MTP draft positions and padded metadata")


def _apply():
    from vllm.v1.worker.gpu.spec_decode import speculator

    from vllm_iluvatar.v1.worker.gpu.spec_decode import speculator as ilu_speculator

    cls = speculator.DraftModelSpeculator
    name = "_build_attn_metadata"
    p.check_fingerprint(
        name,
        inspect.getsource(getattr(cls, name)),
        ilu_speculator.BASELINE_SHA256,
        baseline_is_hash=True,
        resync_hint="Rebase MTP draft metadata override.",
    )
    p.check_signatures(
        cls,
        SimpleNamespace(**{name: getattr(ilu_speculator, name)}),
        [name],
        strict=True,
    )
    setattr(cls, name, getattr(ilu_speculator, name))
