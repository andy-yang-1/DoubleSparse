# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from triton_kernels.channel import get_label_tensor
from triton_kernels.sparse import fwd_sparse, torch_fwd_sparse, fwd_sparse_no_mask
# from triton_kernels.heavy import get_heavy
from triton_kernels.bgemv import bgemv
from triton_kernels.bgemv_int8 import bgemv_int8


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 16384
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 40000 # TODO: add config for vicuna-16k
    norm_eps: float = 1e-5
    heavy_const: int = 256
    heavy_channel_num: int = 32


    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config in str(name).upper() or config in str(name)]
        assert len(config) == 1, name
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, heavy_channel_num, dtype=torch.bfloat16):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        cache_shape = (max_batch_size, max_seq_length, n_heads, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('k_label', torch.zeros((max_batch_size, max_seq_length, n_heads, heavy_channel_num), dtype=dtype))

        # store qk tmp label while decoding
        # self.register_buffer('tmp_label', torch.zeros((max_batch_size * 2, n_heads, heavy_channel_num), dtype=dtype))
        # store tmp label scores while decoding
        # self.register_buffer('label_scores', torch.zeros((max_batch_size, n_heads, max_seq_length), dtype=dtype))
        # store tmp attn output while decoding
        self.register_buffer('attn_out', torch.zeros((max_batch_size, n_heads, head_dim), dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, S, H, D]
        assert input_pos.shape[0] == k_val.shape[1]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, input_pos] = k_val
        v_out[:, input_pos] = v_val
        return k_out, v_out

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, self.config.heavy_channel_num)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base)
        self.prefill_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

        # TODO: change 16 to 32
        self.label_mask = torch.zeros(self.max_seq_length, self.max_seq_length, dtype=torch.float16)
        self.label_mask = self.label_mask.masked_fill(self.prefill_mask == False, float('-inf'))

        # TODO: change 16 to 32
        self.attn_mask = torch.zeros(self.max_seq_length, self.config.heavy_const, dtype=torch.float16)
        self.attn_mask = self.attn_mask.masked_fill(torch.tril(torch.ones(self.max_seq_length, self.config.heavy_const, dtype=torch.bool)) == False, float('-inf'))



    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"

        # is_prefill = input_pos.shape[-1] > 1

        # if is_prefill:
        #     mask1 = self.prefill_mask[None, None, input_pos] # [B, H, S, S]
        #     mask2 = None
        # else:
        #     # TODO: this is a shortcut, the mask broadcast should be rewritten
        #     mask1 = self.label_mask[None, input_pos] # [1, 1, S]
        #     mask2 = self.attn_mask[input_pos] # [1, HEAVY_CONST] 

        mask1 = self.label_mask[None, None, input_pos] # [B, H, S, S]
        mask2 = torch.zeros(1, self.config.heavy_const, dtype=torch.float16).cuda() # [1, HEAVY_CONST]

        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask1, mask2)
        x = self.norm(x)
        logits = self.output(x)
        return logits
    
    def sparse_forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"

        mask1 = self.label_mask[None, None, input_pos]
        mask2 = torch.zeros(1, self.config.heavy_const, dtype=torch.float16).cuda()

        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer.sparse_forward(x, input_pos, freqs_cis, mask1, mask2)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask1: Tensor, mask2: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask1, mask2, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def sparse_forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask1: Tensor, mask2: Tensor) -> Tensor:
        h = x + self.attention.sparse_forward(self.attention_norm(x), freqs_cis, mask1, mask2, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

        # channel config
        self.sorted_channel = None

        # heavy const
        self.heavy_const = config.heavy_const
        self.heavy_channel_num = config.heavy_channel_num

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, mask1: Tensor, mask2: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        tmp_labels = torch.empty((bsz * seqlen, self.n_head, self.heavy_channel_num), dtype=self.kv_cache.k_label.dtype, device='cuda')
        get_label_tensor(k.view(bsz * seqlen, self.n_local_heads, self.head_dim), self.sorted_channel, tmp_labels, self.heavy_channel_num)


        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)
            self.kv_cache.k_label[:, input_pos] = tmp_labels.view(bsz, seqlen, self.n_head, self.heavy_channel_num)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights += mask1
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        y = torch.matmul(attn_weights, v)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)

        return y

    def sparse_forward(self, x: Tensor, freqs_cis: Tensor, mask1: Tensor, mask2: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        tmp_labels = torch.empty((bsz * seqlen, self.n_head, self.heavy_channel_num), dtype=self.kv_cache.k_label.dtype, device='cuda')
        get_label_tensor(k.view(bsz * seqlen, self.n_local_heads, self.head_dim), self.sorted_channel, tmp_labels, self.heavy_channel_num)


        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)
            self.kv_cache.k_label[:, input_pos] = tmp_labels.view(bsz, seqlen, self.n_head, self.heavy_channel_num)

        get_label_tensor(q.view(bsz, self.n_head, self.head_dim), self.sorted_channel, tmp_labels, self.heavy_channel_num)
        label_scores = torch.matmul(tmp_labels.view(bsz, 1, self.n_head, self.heavy_channel_num).transpose(1,2), self.kv_cache.k_label.view(bsz, -1, self.n_head, self.heavy_channel_num).transpose(1,2).transpose(2, 3)).view(bsz, self.n_head, 1, -1)
        label_scores += mask1
        _, label_index = torch.topk(label_scores, self.heavy_const, dim=-1)
        # fwd_sparse(q.view(-1, self.n_head, self.head_dim), k.view(-1, self.n_local_heads, self.head_dim), v.view(-1, self.n_local_heads, self.head_dim), self.kv_cache.attn_out, label_index.view(bsz, self.n_head, -1), mask2)
        fwd_sparse_no_mask(q.view(-1, self.n_head, self.head_dim), k.view(-1, self.n_local_heads, self.head_dim), v.view(-1, self.n_local_heads, self.head_dim), self.kv_cache.attn_out, label_index.view(bsz, self.n_head, -1))
        y = self.wo(self.kv_cache.attn_out.view(bsz, seqlen, self.dim))

        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def init_model_channel_config(model, channel_config, selected_channel="k"):

    selected_channel = "." + selected_channel + "_proj"

    for name, module in model.named_modules():

        if isinstance(module, Attention):
            
            layer_idx = int(name.split(".")[1])
            key = "model.layers." + str(layer_idx) + ".self_attn" + selected_channel
            
            module.sorted_channel = permute_channel_config(torch.tensor(channel_config[key]))[:,:module.heavy_channel_num].contiguous().cuda()

    return model

def permute_channel_config(sorted_channel):
    head_num = sorted_channel.shape[0]
    head_dim = sorted_channel.shape[1]
    return (sorted_channel * 2) % head_dim + (sorted_channel * 2) // head_dim

