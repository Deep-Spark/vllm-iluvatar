# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3.1 DSpark: dense MiniMax decoder and shared vLLM DSpark protocol.

Reuse the context-cache/proposal interface, Markov head and confidence head;
the decoder preserves MiniMax's Gemma norms, partial RoPE and SwiGLU-OAI.
"""

import ixformer.inference.functions as ixf_ops
import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3Attention,
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    _resolve_layer_attention,
)
from vllm.model_executor.models.qwen3_dspark import (
    Qwen3DSparkForCausalLM,
    Qwen3DSparkModel,
)

from .mxfp8.attention import MiniMAXGemmaRMSNorm
from .mxfp8.model import MiniMaxM3MLP

logger = init_logger(__name__)


class IluMiniMaxDSparkDecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config,
        *,
        config,
        layer_idx,
        cache_config=None,
        quant_config=None,
        prefix="",
    ):
        torch.nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        window, causal = _resolve_layer_attention(config, layer_idx)
        self.self_attn = DFlashQwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_parameters=config.rope_parameters,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            sliding_window=window,
            causal=causal,
            is_neox_style=True,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn.q_norm = MiniMAXGemmaRMSNorm(
            config.head_dim, config.rms_norm_eps
        )
        self.self_attn.k_norm = MiniMAXGemmaRMSNorm(
            config.head_dim, config.rms_norm_eps
        )
        self.mlp = MiniMaxM3MLP(
            config,
            config.dense_intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = MiniMAXGemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = MiniMAXGemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )


class IluMiniMaxDSparkModel(Qwen3DSparkModel):
    decoder_layer_cls = IluMiniMaxDSparkDecoderLayer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_norm = MiniMAXGemmaRMSNorm(
            self.config.hidden_size, self.config.rms_norm_eps
        )
        self.norm = MiniMAXGemmaRMSNorm(
            self.config.hidden_size, self.config.rms_norm_eps
        )
        self._context_kv_weights = None
        self._context_kv_scales = None

    def _build_context_kv_buffers(self, layers_attn, has_bias):
        super()._build_context_kv_buffers(layers_attn, has_bias)
        if has_bias or any(
            proj.weight.dtype is not torch.float8_e4m3fn
            or not hasattr(proj, "weight_scale")
            or proj.weight_scale.ndim not in (1, 2)
            for proj in self._context_qkv_projs
        ):
            # Keep the existing projection path for other quantization methods.
            logger.info("MiniMax DSpark context KV-only MXFP8 path unavailable")
            return

        # This hook runs before the QKV layer's normal post-load requant. Copy
        # only KV rows, then apply that same block-to-channel requant to the
        # copy. The decoder's full QKV weight is processed independently later.
        kv_weights = []
        kv_scales = []
        for attn, proj in zip(layers_attn, self._context_qkv_projs, strict=True):
            start = attn.q_size
            assert proj.weight.shape[0] - start == 2 * attn.kv_size
            weight = proj.weight.detach()[start:].clone()
            scale = proj.weight_scale.detach()[start:]
            assert scale.shape[0] == weight.shape[0]
            if scale.ndim == 2:
                assert scale.shape[1] == weight.shape[1] // 32
                assert scale.dtype in (torch.uint8, torch.float8_e8m0fnu)
                block_scale = (
                    scale
                    if scale.dtype is torch.float8_e8m0fnu
                    else scale.view(torch.float8_e8m0fnu)
                )
                weight, scale = ixf_ops.fp8_weight_block_to_channel_requant(
                    weight,
                    block_scale,
                    1,
                    32,
                    fp8_output=weight,
                )
            else:
                scale = scale.clone()
            kv_weights.append(weight)
            kv_scales.append(scale)
        self._context_kv_weights = tuple(kv_weights)
        self._context_kv_scales = tuple(kv_scales)
        self._context_k_norm_weights = torch.stack(
            [layer.self_attn.k_norm.weight.detach() for layer in self.layers]
        )
        logger.info(
            "MiniMax DSpark context uses %d KV-only MXFP8 projections with shared input quantization",
            len(kv_weights),
        )

    def _project_context_kv(
        self, context_states, num_ctx, num_layers, num_kv_heads, head_dim
    ):
        context = self.hidden_norm(context_states)
        if self._context_kv_weights is None:
            values = []
            for proj, q_size in zip(
                self._context_qkv_projs, self._context_q_sizes, strict=True
            ):
                qkv, _ = proj(context)
                values.append(qkv[:, q_size:].view(num_ctx, 2, num_kv_heads, head_dim))
            kv = torch.stack(values, dim=1).permute(2, 1, 0, 3, 4).contiguous()
            return kv[0], kv[1]
        else:
            # K/V are views; the cache writer accepts V's token stride.
            packed = torch.empty(
                (num_layers, num_ctx, 2, num_kv_heads, head_dim),
                device=context.device,
                dtype=context.dtype,
            )
            if num_ctx == 0:
                return packed[:, :, 0], packed[:, :, 1]
            qcontext, scale_context = ixf_ops.fp8_per_token_dynamic_scaled(
                context.reshape(-1, context.shape[-1])
            )
            for i, (weight, scale_weight) in enumerate(
                zip(self._context_kv_weights, self._context_kv_scales, strict=True)
            ):
                output = packed[i].view(num_ctx, 2 * num_kv_heads * head_dim)
                if num_ctx <= 16 and weight.shape[0] % 32 == 0:
                    ixf_ops.fp8_per_channel_gemv(
                        qcontext,
                        weight,
                        scale_context,
                        scale_weight,
                        output,
                        format="TN",
                        backend="ixmm",
                    )
                else:
                    ixf_ops.fp8_per_channel_mm(
                        qcontext,
                        weight,
                        scale_context,
                        scale_weight,
                        output,
                        backend="ixmm",
                    )
        return packed[:, :, 0], packed[:, :, 1]

    def _normalize_context_k(self, all_k):
        if self._context_kv_weights is not None:
            from vllm_iluvatar.model_executor.layers.layernorm import (
                batched_gemma_rms_norm,
            )

            return batched_gemma_rms_norm(
                all_k,
                self._context_k_norm_weights,
                self.layers[0].self_attn.k_norm.variance_epsilon,
            )
        return torch.stack(
            [layer.self_attn.k_norm(all_k[i]) for i, layer in enumerate(self.layers)],
            dim=0,
        )


class IluMiniMaxDSparkForCausalLM(Qwen3DSparkForCausalLM):
    model_cls = IluMiniMaxDSparkModel

    def __init__(self, *, vllm_config, prefix=""):
        # The DFlash constructor honors model_cls; Qwen DSpark hardcodes its
        # decoder. All DSpark sampling methods operate on the common interface.
        DFlashQwen3ForCausalLM.__init__(self, vllm_config=vllm_config, prefix=prefix)
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()

    def try_bind_target_context(self, target_model, output: torch.Tensor) -> bool:
        """Opt in only when the target can publish our projected auxiliary states.

        The runner guarantees sequential PP1/CP1 execution and owns ``output``
        until proposal finishes. Targets without this capability use the usual
        auxiliary-state export and community draft projection.
        """
        language = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        target = getattr(language, "model", None)
        bind = getattr(target, "bind_draft_context", None)
        return bool(bind(self.model.fc, output)) if callable(bind) else False

    def _finish_local_logits(self, logits: torch.Tensor) -> torch.Tensor:
        soft_cap = self.logits_processor.soft_cap
        if soft_cap is not None:
            logits = torch.tanh(logits / soft_cap) * soft_cap
        scale = self.logits_processor.scale
        if scale != 1.0:
            logits = logits * scale
        return logits

    def compute_draft_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor._apply_head(self.lm_head, hidden_states, None)
        return self._finish_local_logits(logits)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

    def get_draft_vocab_shard(self) -> tuple[int, int]:
        indices = self.lm_head.shard_indices
        return indices.org_vocab_start_index, indices.num_org_vocab_padding

    def markov_bias_local(self, markov_embed: torch.Tensor) -> torch.Tensor:
        markov_w2 = self.model.markov_head.markov_w2
        indices = self.lm_head.shard_indices
        start = indices.org_vocab_start_index
        local_size = self.lm_head.num_embeddings_per_partition
        weight = markov_w2.weight
        if weight.shape[0] != local_size:
            if start + local_size > weight.shape[0]:
                raise ValueError(
                    f"MiniMax DSpark cannot select Markov rows [{start}:{start + local_size}] "
                    f"from weight shape {tuple(weight.shape)}"
                )
            weight = weight.narrow(0, start, local_size)

        head_dtype = self.logits_processor.head_dtype
        if head_dtype is None or head_dtype == markov_embed.dtype:
            logits = F.linear(markov_embed, weight)
        elif (
            head_dtype == torch.float32
            and markov_embed.is_cuda
            and markov_embed.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
        ):
            flat = markov_embed.reshape(-1, markov_embed.shape[-1])
            logits = ixf_ops.gemm_bf16_bf16_fp32(flat, weight, format="TN")
            logits = logits.reshape(*markov_embed.shape[:-1], -1)
        else:
            logits = F.linear(markov_embed.to(head_dtype), weight.to(head_dtype))
        return self._finish_local_logits(logits)

    def load_weights(self, weights):
        expected = {
            "fc.weight",
            "hidden_norm.weight",
            "norm.weight",
            "markov_head.markov_w1.weight",
            "markov_head.markov_w2.weight",
            "confidence_head.proj.weight",
            "confidence_head.proj.bias",
        }
        for i in range(self.config.num_hidden_layers):
            stem = f"layers.{i}."
            expected.update(
                stem + name + ".weight"
                for name in (
                    "input_layernorm",
                    "post_attention_layernorm",
                    "self_attn.q_norm",
                    "self_attn.k_norm",
                )
            )
            expected.update(
                stem + name + suffix
                for name in (
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                )
                for suffix in (".weight", ".weight_scale")
            )
        seen = set()

        def remap():
            for name, weight in weights:
                prefix = "language_model.model.dspark."
                if not name.startswith(prefix):
                    raise ValueError(
                        f"Unexpected MiniMax DSpark checkpoint tensor: {name}"
                    )
                name = name.removeprefix(prefix).replace(".decoder_layer.", ".")
                if name.startswith("final_norm."):
                    name = "norm." + name.removeprefix("final_norm.")
                name = name.replace("weight_scale_inv", "weight_scale")
                if name in seen or name not in expected:
                    raise ValueError(
                        f"Duplicate/unexpected MiniMax DSpark tensor: {name}"
                    )
                seen.add(name)
                yield name, weight

        result = super().load_weights(remap())
        if seen != expected:
            raise ValueError(
                f"Missing MiniMax DSpark weights: {sorted(expected - seen)}"
            )
        logger.info(
            "MiniMax DSpark loaded %d/%d checkpoint tensors; target layers=%s",
            len(seen),
            len(expected),
            self.config.dspark_target_layer_ids,
        )
        return result


def register_model():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "DSparkMiniMaxDraftModel",
        "vllm_iluvatar.models.minimax_m3.dspark:IluMiniMaxDSparkForCausalLM",
    )
