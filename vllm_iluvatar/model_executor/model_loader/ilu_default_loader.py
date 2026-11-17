# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/model_executor/model_loader/default_loader.py.

Target: DefaultModelLoader; baseline v0.31.0 (db9527a468).
Why: MXFP4 conversion needs source pairs and reusable upload storage.
How: registered subclass, routing deferred sources through existing model loaders.
Scope: parameters using Mxfp4WeightLoader; other models delegate unchanged.
Remove: when upstream provides deferred checkpoint sources and upload sessions.
"""

import sys
import time

import torch
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    _natural_sort_key,
    _prefetch_all_checkpoints,
)

from vllm_iluvatar.model_executor.layers.quantization.common.mxfp4_loader import (
    Mxfp4UploadBuffer,
    Mxfp4WeightLoader,
)
from vllm_iluvatar.model_executor.model_loader.ilu_paired_weights import (
    DeferredCheckpointTensor,
    SafetensorsReaders,
    checkpoint_references,
)

logger = init_logger(__name__)


class Mxfp4LoadSession:
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
                raise RuntimeError("MXFP4 load session is already bound")
            if loader._loaded or loader._finalized or loader._failed:
                raise RuntimeError("MXFP4 load session requires fresh parameters")
        for loader in self.loaders:
            loader._load_session = self

    def add(self, loader, group, is_scale, tensor, shard, global_expert, local_expert):
        if not isinstance(tensor, DeferredCheckpointTensor):
            raise ValueError(
                "MXFP4 paired loading requires a deferred checkpoint source; "
                f"unsupported data transformation before callback for {self.loaders[loader]} "
                f"expert={global_expert} shard={shard} scale={is_scale}"
            )
        if tensor.readers is not self.readers:
            raise ValueError("MXFP4 source belongs to a different loading session")
        key = (loader, local_expert, shard)
        pair = self.pairs.setdefault(key, {})
        if is_scale in pair:
            raise ValueError(f"Duplicate MXFP4 target: {tensor.reference.source.name}")
        pair[is_scale] = (tensor.reference, group, global_expert)

    def finish(self):
        # Include loaders with no callbacks: an entirely missing pair or layer
        # must not disappear from the expected set.
        for loader, name in self.loaders.items():
            for expert in range(loader.load_spec.experts):
                for shard in ("w1", "w2", "w3"):
                    if set(self.pairs.get((loader, expert, shard), {})) != {
                        False,
                        True,
                    }:
                        raise ValueError(
                            f"Missing MXFP4 pair: {name} local_expert={expert} shard={shard}"
                        )
        self.collecting = False
        capacities = {}
        for loader in self.loaders:
            device = loader._layer.w13_weight.device
            capacities[device] = max(capacities.get(device, 0), loader.staging_bytes)
        for device, capacity in capacities.items():
            self.buffers[device] = Mxfp4UploadBuffer(capacity, device)
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
            "MXFP4 paired loading: pairs=%d opens=%d reads=%d staging_bytes=%s",
            len(self.pairs),
            self.readers.opens,
            self.readers.reads,
            list(capacities.values()),
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
                "MXFP4 upload memory: phase=%s allocated=%d reserved=%d segments=%d",
                phase,
                stats["allocated_bytes.all.current"],
                stats["reserved_bytes.all.current"],
                stats["segment.all.current"],
            )


class IluDefaultModelLoader(DefaultModelLoader):
    def load_weights(self, model, model_config):
        loaders = {}
        for name, parameter in model.named_parameters():
            callback = getattr(parameter, "weight_loader", None)
            loader = getattr(callback, "mxfp4_loader", None)
            if isinstance(loader, Mxfp4WeightLoader):
                loaders.setdefault(loader, name)
        if not loaders:
            return super().load_weights(model, model_config)
        cfg = self.load_config
        if cfg.model_loader_extra_config.get("enable_multithread_load"):
            raise ValueError(
                "MXFP4 paired loading does not support enable_multithread_load"
            )
        if cfg.safetensors_load_strategy not in (None, "lazy", "prefetch"):
            raise ValueError(
                "MXFP4 paired loading supports lazy/prefetch safetensors only"
            )
        if getattr(model, "secondary_weights", ()):
            raise ValueError("MXFP4 paired loading requires a single checkpoint source")
        session = Mxfp4LoadSession(loaders)
        session.bind()
        self._paired_session = session
        started = time.perf_counter()
        try:
            result = super().load_weights(model, model_config)
            session.finish()
            self.counter_after_loading_weights = time.perf_counter()
            logger.info(
                "MXFP4 complete weight loading took %.3f seconds (routing, reads and conversion)",
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
                logger.exception("MXFP4 cleanup failed while handling a loading error")

    def _get_weights_iterator(self, source):
        if getattr(self, "_paired_session", None) is None:
            return super()._get_weights_iterator(source)
        return self._deferred_iterator(source)

    def _deferred_iterator(self, source):
        self.counter_before_loading_weights = time.perf_counter()
        _, files, safetensors, _ = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        if not safetensors:
            raise ValueError("MXFP4 paired loading requires a safetensors checkpoint")
        files = sorted(files, key=_natural_sort_key)
        references = checkpoint_references(files)
        logger.info(
            "MXFP4 checkpoint metadata: files=%d tensors=%d time=%.3fs",
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
