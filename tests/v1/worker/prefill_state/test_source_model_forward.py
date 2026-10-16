"""Exercise the actual HostSourceWorker binding without weights or CUDA."""

from types import SimpleNamespace

import pytest
import torch
from ixformer.inference.functions.deepseek.quantized_kv_format import I4G32, I8G64
from vllm.sequence import IntermediateTensors

from vllm_iluvatar.models.deepseek_v41 import model as model_module
from vllm_iluvatar.models.deepseek_v41.compressed_kv import WorkspacePool
from vllm_iluvatar.v1.worker.gpu_worker import IluWorker
from vllm_iluvatar.v1.worker.prefill_state import host_source_worker


class Layer(torch.nn.Module):
    def __init__(self, workspace, records, consumer=False):
        super().__init__()
        self.attn = SimpleNamespace(
            compressed_kv_workspace=workspace,
            accept_remote_kv=lambda *args, **kwargs: records["received"].append(
                kwargs.get("main_scale_rows")
            ),
        )
        self.workspace, self.records, self.consumer = workspace, records, consumer

    def forward(self, hidden, positions, ids, pre, post, residual_mix, residual):
        if self.consumer and self.workspace is not None:
            pool = self.workspace
            with pool.read_scope():
                self.records["epochs"].append(pool.restore.step_epoch)
                if pool.restore.lookup("source2/index2") is None:
                    ticket = pool.restore.begin_prepare("source2/index2")
                    selection = object()
                    pool.restore.publish(ticket, selection)
                    self.records["prepares"].append(selection)
        return hidden, residual, post, residual_mix, pre


def bind_source_worker(monkeypatch, tmp_path, fmt, pp, *, workspace=None):
    """Only base model loading, layer math and groups are isolated boundaries."""
    import vllm.distributed as distributed

    group = SimpleNamespace(rank_in_group=pp, is_first_rank=pp == 0, is_last_rank=False)
    monkeypatch.setattr(distributed, "get_pp_group", lambda: group)
    monkeypatch.setattr(
        distributed, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    monkeypatch.setattr(model_module, "get_pp_group", lambda: group)
    monkeypatch.setattr(model_module.mhc, "hc_post", lambda hidden, *unused: hidden)
    monkeypatch.setattr(IluWorker, "load_model", lambda *args, **kwargs: "loaded")
    monkeypatch.setattr(
        host_source_worker, "local_paths", lambda engine: (tmp_path, "source-run")
    )
    records = {"received": [], "prepares": [], "epochs": []}
    model = model_module.IluDeepseekV41Model.__new__(model_module.IluDeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.hc_mult = 4
    model.start_layer, model.end_layer = pp * 5, (pp + 1) * 5
    model.aux_hidden_state_layers = set()
    model.layers = torch.nn.ModuleList(
        [Layer(workspace, records, consumer=2 <= index < 5) for index in range(10)]
    )
    model.pp_plan = SimpleNamespace(
        sends_kv=True,
        sends_topk=False,
        sends_candidates=False,
        recvs_kv=pp > 0,
        recvs_topk=False,
        recvs_candidates=False,
        forwarder=model.start_layer if pp else None,
    )
    model.relays_kv = pp > 0
    model.main_kv_scale_width = fmt.scale_groups(512) if fmt else 0
    width = fmt.payload_bytes(512) if fmt else 512
    model.main_kv_rows = (
        torch.arange(3 * width)
        .reshape(3, width)
        .to(torch.uint8 if fmt else torch.bfloat16)
    )
    model.main_kv_scale_rows = (
        (
            torch.arange(3 * model.main_kv_scale_width)
            .to(torch.float16)
            .reshape(3, model.main_kv_scale_width)
        )
        if fmt
        else None
    )
    model.indexer_kv_rows = torch.arange(3 * 132).to(torch.uint8).reshape(3, 132)
    model._prepare_engram = lambda *unused: None
    runner = SimpleNamespace(
        get_model=lambda: model,
        req_states=SimpleNamespace(add_request=object(), remove_request=object()),
        prepare_inputs=object(),
        execute_model=object(),
        finish_requests=object(),
    )
    worker = host_source_worker.HostSourceWorker.__new__(
        host_source_worker.HostSourceWorker
    )
    worker.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(engine_id="source"),
        scheduler_config=SimpleNamespace(max_num_seqs=32),
    )
    worker.model_runner = runner
    assert worker.load_model() == "loaded"
    return model, records


def incoming_packet(model, fmt):
    tensors = {
        "hidden_states": torch.zeros(3, 4, 5120, dtype=torch.bfloat16),
        "pre_mix": torch.ones(3, 4, dtype=torch.float32),
        "main_kv": model.main_kv_rows.clone(),
        "indexer_kv": model.indexer_kv_rows.clone(),
    }
    if fmt:
        tensors["main_kv_scales"] = model.main_kv_scale_rows.clone()
    return IntermediateTensors(tensors)


@pytest.mark.parametrize("fmt", [None, I8G64, I4G32])
@pytest.mark.parametrize("pp", [0, 1])
def test_host_source_binding_preserves_public_wrapper_and_paired_pp_records(
    monkeypatch, tmp_path, fmt, pp
):
    model, _ = bind_source_worker(monkeypatch, tmp_path, fmt, pp)
    positions, ids = torch.arange(3), torch.arange(3, dtype=torch.int32)
    packet = incoming_packet(model, fmt) if pp else None
    output = model(
        ids, positions, packet, inputs_embeds=torch.zeros(3, 5120, dtype=torch.bfloat16)
    )
    if fmt:
        assert "main_kv_scales" in output.tensors
        assert torch.equal(output["main_kv_scales"], model.main_kv_scale_rows)
    else:
        assert "main_kv_scales" not in output.tensors

    # The same actual receiver method used on the next PP rank consumes/relays it.
    received = []
    receiver = SimpleNamespace(
        pp_plan=SimpleNamespace(recvs_kv=True, recvs_topk=False, forwarder=0),
        main_kv_scale_width=model.main_kv_scale_width,
        relays_kv=True,
        main_kv_rows=torch.empty_like(model.main_kv_rows),
        main_kv_scale_rows=(
            torch.empty_like(model.main_kv_scale_rows) if fmt else None
        ),
        indexer_kv_rows=torch.empty_like(model.indexer_kv_rows),
        layers=[
            SimpleNamespace(
                attn=SimpleNamespace(
                    accept_remote_kv=lambda *args, **kwargs: received.append(kwargs)
                )
            )
        ],
    )
    model_module.IluDeepseekV41Model._accept_remote_kv(receiver, output, positions)
    assert torch.equal(receiver.main_kv_rows, model.main_kv_rows)
    assert torch.equal(receiver.indexer_kv_rows, model.indexer_kv_rows)
    if fmt:
        assert received[0]["main_scale_rows"] is output["main_kv_scales"]
        assert torch.equal(receiver.main_kv_scale_rows, model.main_kv_scale_rows)
    else:
        assert received[0]["main_scale_rows"] is None


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
def test_source_forward_shares_one_real_family_scope_for_three_consumers(
    monkeypatch, tmp_path, fmt
):
    pool = WorkspacePool(torch.empty(4096, dtype=torch.uint8))
    model, records = bind_source_worker(monkeypatch, tmp_path, fmt, 0, workspace=pool)
    for step in (1, 2):
        # The actual worker replay lease is distinct from the model scope.
        with pool.acquire():
            model(
                torch.arange(3, dtype=torch.int32),
                torch.arange(3),
                None,
                inputs_embeds=torch.zeros(3, 5120, dtype=torch.bfloat16),
            )
            assert records["epochs"][-3:] == [step] * 3
            assert pool._forward_depth == 0
            assert pool.restore.lookup("source2/index2") is None
        assert len(records["prepares"]) == step
    assert records["prepares"][0] is not records["prepares"][1]
    assert model.forward.__func__ is model_module.IluDeepseekV41Model.forward
