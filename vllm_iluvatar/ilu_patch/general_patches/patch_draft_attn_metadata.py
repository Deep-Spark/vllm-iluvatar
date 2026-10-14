"""Supply draft positions and capture sequence upper bounds on Iluvatar."""

import inspect
from types import SimpleNamespace

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__, _apply, detail="draft positions and DFlash capture metadata"
    )


def _apply() -> None:
    from vllm.v1.worker.gpu.spec_decode import speculator
    from vllm.v1.worker.gpu.spec_decode.dflash import cudagraph

    from vllm_iluvatar.v1.worker.gpu.spec_decode import cudagraph as ilu_cudagraph
    from vllm_iluvatar.v1.worker.gpu.spec_decode import metadata as ilu_metadata

    cls = speculator.DraftModelSpeculator
    for target, override, name in (
        (cls, ilu_metadata, "_build_attn_metadata"),
        (cudagraph, ilu_cudagraph, "_prepare_dflash_inputs_to_capture"),
    ):
        p.check_fingerprint(
            name,
            inspect.getsource(getattr(target, name)),
            override.BASELINE,
            resync_hint="Rebase draft metadata override.",
        )
        p.check_signatures(
            target,
            SimpleNamespace(**{name: getattr(override, name)}),
            [name],
            strict=True,
        )
        setattr(target, name, getattr(override, name))
