# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/model_executor/model_loader/default_loader.py.

Target: DefaultModelLoader; baseline v0.31.0 (db9527a468).
Why: block-FP8 and MXFP4 conversion need source pairs and reusable upload storage.
How: registered subclass, routing deferred sources through existing model loaders.
Scope: block-FP8 and MXFP4 parameters opting into paired loading.
Remove: when upstream provides deferred checkpoint sources and upload sessions.
"""

import json
import sys
import time
from pathlib import Path

import torch
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    _natural_sort_key,
    _prefetch_all_checkpoints,
)

from vllm_iluvatar.model_executor.layers.quantization.common.paired_moe_loader import (
    ExpertUploadBuffer,
)
from vllm_iluvatar.model_executor.model_loader.ilu_paired_weights import (
    DeferredCheckpointTensor,
    SafetensorsReaders,
    checkpoint_references,
)

logger = init_logger(__name__)


class PairedLoadSession:
    """Collect actual parameter routing, then read and convert complete pairs."""

    def __init__(self, loaders):
        self.loaders = loaders
        self.readers = SafetensorsReaders()
        self.buffers = {}
        self.pairs = {}
        self.collecting = True
        self.complete = False

    def bind(self):
        for loader in self.loaders:
            if loader._load_session is not None or loader._upload_buffer is not None:
                raise RuntimeError("Paired expert load session is already bound")
            if loader._loaded or loader._finalized or loader._failed:
                raise RuntimeError(
                    "Paired expert load session requires fresh parameters"
                )
        for loader in self.loaders:
            loader._load_session = self

    def add(self, loader, group, is_scale, tensor, shard, global_expert, local_expert):
        if not isinstance(tensor, DeferredCheckpointTensor):
            raise ValueError(
                "Paired expert loading requires a deferred checkpoint source; "
                f"unsupported data transformation before callback for {self.loaders[loader]} "
                f"expert={global_expert} shard={shard} scale={is_scale}"
            )
        if tensor.readers is not self.readers:
            raise ValueError(
                "Paired expert source belongs to a different loading session"
            )
        key = (loader, local_expert, shard)
        pair = self.pairs.setdefault(key, {})
        if is_scale in pair:
            raise ValueError(
                f"Duplicate {loader.source_name} target: {tensor.reference.source.name}"
            )
        pair[is_scale] = (tensor.reference, group, global_expert)

    def finish(self):
        # Include loaders with no callbacks: an entirely missing pair or layer
        # must not disappear from the expected set.
        for loader, name in self.loaders.items():
            for expert, shard in loader.expected_pairs():
                if set(self.pairs.get((loader, expert, shard), {})) != {False, True}:
                    source = loader.source_name
                    raise ValueError(
                        f"Missing {source} pair: {name} local_expert={expert} shard={shard}"
                    )
        self.collecting = False
        capacities = {}
        conversion_capacities = {}
        for loader in self.loaders:
            device = loader._layer.w13_weight.device
            capacities[device] = max(capacities.get(device, 0), loader.staging_bytes)
            conversion_capacities[device] = max(
                conversion_capacities.get(device, 0), loader.conversion_bytes
            )
        for device, capacity in capacities.items():
            self.buffers[device] = ExpertUploadBuffer(
                capacity, device, conversion_capacity=conversion_capacities[device]
            )
            self._log_memory("staging_ready", device)
        for loader in self.loaders:
            loader.bind_upload_buffer(self.buffers[loader._layer.w13_weight.device])
        for (loader, _, shard), pair in self.pairs.items():
            for role in (False, True):
                reference, group, expert = pair[role]
                # One source is consumed before reading the next. Only the
                # shared pinned slot retains the first half of the pair.
                loader.load_shard(
                    group, role, reference.read(self.readers), shard, expert
                )
        for device in capacities:
            self._log_memory("weights_copied", device)
        self.complete = True
        logger.info(
            "Expert paired loading: pairs=%d opens=%d reads=%d staging_bytes=%s conversion_bytes=%s",
            len(self.pairs),
            self.readers.opens,
            self.readers.reads,
            list(capacities.values()),
            list(conversion_capacities.values()),
        )

    def close(self):
        errors = []
        try:
            self.readers.close()
        except Exception as error:
            errors.append(error)
        for buffer in self.buffers.values():
            try:
                buffer.close()
            except Exception as error:
                errors.append(error)
        for loader in self.loaders:
            loader._load_session = None
            loader._upload_buffer = None
            loader._pending = None
            if not self.complete or errors:
                loader._failed = True
        self.buffers.clear()
        self.pairs.clear()
        if errors:
            raise errors[0]

    @staticmethod
    def _log_memory(phase, device):
        if device.type == "cuda":
            stats = torch.cuda.memory_stats(device)
            logger.info(
                "Paired expert upload memory: phase=%s allocated=%d reserved=%d segments=%d "
                "peak_allocated=%d peak_reserved=%d",
                phase,
                stats["allocated_bytes.all.current"],
                stats["reserved_bytes.all.current"],
                stats["segment.all.current"],
                stats["allocated_bytes.all.peak"],
                stats["reserved_bytes.all.peak"],
            )


class IluDefaultModelLoader(DefaultModelLoader):
    def _prepare_weights(self, *args, **kwargs):
        prepared = super()._prepare_weights(*args, **kwargs)
        bind = getattr(self, "_bind_checkpoint_source", None)
        if bind is not None:
            directory, _, safetensors, index = prepared
            if not safetensors:
                raise ValueError("This checkpoint requires indexed safetensors")
            bind(directory, index)
        return prepared

    def load_weights(self, model, model_config):
        self._bind_checkpoint_source = getattr(model, "bind_checkpoint_source", None)
        try:
            return self._load_paired_or_original(model, model_config)
        finally:
            self._bind_checkpoint_source = None

    def _load_paired_or_original(self, model, model_config):
        loaders = {}
        for name, parameter in model.named_parameters():
            callback = getattr(parameter, "weight_loader", None)
            loader = getattr(callback, "paired_weight_loader", None)
            if loader is not None:
                loaders.setdefault(loader, name)
        if not loaders:
            return super().load_weights(model, model_config)
        cfg = self.load_config
        if cfg.model_loader_extra_config.get("enable_multithread_load"):
            raise ValueError(
                "Paired expert loading does not support enable_multithread_load"
            )
        if cfg.safetensors_load_strategy not in (None, "lazy", "prefetch"):
            raise ValueError(
                "Paired expert loading supports lazy/prefetch safetensors only"
            )
        if getattr(model, "secondary_weights", ()):
            raise ValueError(
                "Paired expert loading requires a single checkpoint source"
            )
        session = PairedLoadSession(loaders)
        session.bind()
        self._paired_session = session
        started = time.perf_counter()
        try:
            result = super().load_weights(model, model_config)
            session.finish()
            self.counter_after_loading_weights = time.perf_counter()
            logger.info(
                "Paired expert weight loading took %.3f seconds (routing, reads and conversion)",
                self.counter_after_loading_weights - started,
            )
            return result
        finally:
            failing = sys.exc_info()[0] is not None
            self._paired_session = None
            try:
                session.close()
            except Exception:
                if not failing:
                    raise
                logger.exception(
                    "Paired expert cleanup failed while handling a loading error"
                )

    def _get_weights_iterator(self, source):
        if getattr(self, "_paired_session", None) is None:
            return super()._get_weights_iterator(source)
        return self._deferred_iterator(source)

    def _deferred_iterator(self, source):
        self.counter_before_loading_weights = time.perf_counter()
        directory, files, safetensors, index = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        if not safetensors:
            raise ValueError("Paired expert loading requires a safetensors checkpoint")
        files = sorted(files, key=_natural_sort_key)
        index_path = Path(directory, index)
        weight_map = None
        if index_path.is_file():
            with index_path.open(encoding="utf-8") as file:
                weight_map = {
                    name: Path(directory, shard)
                    for name, shard in json.load(file)["weight_map"].items()
                }
        references = checkpoint_references(files, weight_map=weight_map)
        logger.info(
            "Paired expert checkpoint metadata: files=%d tensors=%d time=%.3fs",
            len(files),
            len(references),
            time.perf_counter() - self.counter_before_loading_weights,
        )
        if self.load_config.safetensors_load_strategy == "prefetch":
            _prefetch_all_checkpoints(
                files,
                num_prefetch_threads=self.load_config.safetensors_prefetch_num_threads,
                block_size=self.load_config.safetensors_prefetch_block_size,
            )
        # Model/parameter routing owns PP/EP filtering. It operates on metadata,
        # so even custom draft expert numbering does not trigger payload reads.
        for ref in references:
            yield (
                source.prefix + ref.source.name,
                DeferredCheckpointTensor(ref, self._paired_session.readers),
            )
