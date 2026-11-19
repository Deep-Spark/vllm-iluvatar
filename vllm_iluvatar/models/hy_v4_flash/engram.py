# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
from torch import nn

from .engram_lookup import _hash_rows, _write_residual, lookup_normalized_rows


class FlashEngram(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.layer = layer + 1
        self.seed = config.engram_seed
        self.pad = config.engram_pad_id
        self.round_fp8 = config.engram_fp8_embedding
        vocab = config.engram_layer_head_vocab_sizes
        groups = vocab.get(str(self.layer), vocab.get(self.layer))
        sizes = [s for group in groups for s in group]
        self.register_buffer(
            "sizes", torch.tensor(sizes, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "offsets",
            torch.tensor(np.cumsum([0] + sizes[:-1]), dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "multipliers", torch.empty(3, dtype=torch.int64), persistent=False
        )
        self.register_buffer("lookup_table", None, persistent=False)
        self.owned_table = None
        h = config.hidden_size
        self.lookup_per_head_norms = nn.ModuleList(nn.RMSNorm(96) for _ in sizes)
        self.key_projs = nn.ModuleList(nn.Linear(1536, h) for _ in range(4))
        self.value_proj = nn.Linear(1536, h)
        self.norm_query = nn.ModuleList(nn.RMSNorm(h) for _ in range(4))
        self.norm_key = nn.ModuleList(nn.RMSNorm(h) for _ in range(4))
        self.unified_pre_gamma = nn.Parameter(torch.empty(h))
        self.unified_per_hc_scales = nn.ParameterList(
            nn.Parameter(torch.empty(h)) for _ in range(4)
        )
        self.value_norm = nn.RMSNorm(h, elementwise_affine=False)

    def load_map(self, value):
        compressed = int(value.max()) + 1
        bound = max(1, np.iinfo(np.int64).max // compressed // 2)
        rng = np.random.default_rng(self.seed + 10007 * self.layer)
        self.multipliers.copy_(
            torch.from_numpy(2 * rng.integers(0, bound, 3, dtype=np.int64) + 1)
        )
        self.lookup_table = value.to(self.sizes.device)

    def forward(self, hidden, triplets):
        ids = torch.empty(
            (hidden.shape[0], 16), dtype=torch.int32, device=hidden.device
        )
        _hash_rows[(hidden.shape[0],)](
            triplets,
            self.lookup_table,
            self.multipliers,
            self.sizes,
            self.offsets,
            ids,
            self.lookup_table.numel(),
            self.pad,
        )
        out = hidden.new_empty((hidden.shape[0], 16, 96))
        lookup_normalized_rows(
            self.owned_table[1], ids, self.gamma, out, round_fp8=self.round_fp8
        )
        projected = torch.nn.functional.linear(
            out.flatten(1), self.proj_weight, self.proj_bias
        )
        result = torch.empty_like(hidden)
        _write_residual[(hidden.shape[0], 4)](
            hidden,
            projected,
            self.key_gamma,
            self.query_gamma,
            self.unified_pre_gamma,
            self.branch_scales,
            triplets,
            result,
            self.pad,
        )
        return result

    def finalize(self):
        self.register_buffer(
            "gamma",
            torch.stack([x.weight for x in self.lookup_per_head_norms]),
            persistent=False,
        )
        projections = [*self.key_projs, self.value_proj]
        self.register_buffer(
            "proj_weight", torch.cat([x.weight for x in projections]), persistent=False
        )
        self.register_buffer(
            "proj_bias", torch.cat([x.bias for x in projections]), persistent=False
        )
        self.register_buffer(
            "key_gamma",
            torch.stack([x.weight for x in self.norm_key]),
            persistent=False,
        )
        self.register_buffer(
            "query_gamma",
            torch.stack([x.weight for x in self.norm_query]),
            persistent=False,
        )
        self.register_buffer(
            "branch_scales",
            torch.stack(list(self.unified_per_hc_scales)),
            persistent=False,
        )
