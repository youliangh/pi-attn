"""Dense-page-table PI attention for vLLM V1; no changes to the numerical program."""
from dataclasses import dataclass, replace
import math
import os

import torch
from vllm.v1.attention.backend import (
    AttentionBackend, AttentionImpl, AttentionMetadataBuilder, AttentionCGSupport, AttentionType,
)
from . import gemm_module

OBSERVED = set()


@dataclass
class Metadata:
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    decode_workspace: tuple
    prefill_workspace: tuple


class MetadataBuilder(AttentionMetadataBuilder[Metadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    supports_update_block_table = True
    requires_block_table_width = True

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, block_table_width):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        mc, pc, sc = vllm_config.model_config, vllm_config.parallel_config, vllm_config.scheduler_config
        assert pc.world_size == 1 and not pc.use_ubatching
        assert vllm_config.speculative_config is None
        heads = mc.get_num_attention_heads(pc)
        groups = math.ceil(block_table_width * kv_cache_spec.block_size / 512)
        rows = sc.max_num_seqs * groups
        kw = dict(device=device, dtype=torch.float32)
        self.decode_workspace = (torch.empty(rows, heads, 128, **kw), torch.empty(rows, heads, 8, **kw),
                                 torch.zeros(rows, heads, 4, device=device, dtype=torch.int32))
        self.prefill_workspace = (torch.empty(sc.max_num_batched_tokens, heads, groups, 128, **kw),
                                  torch.empty(sc.max_num_batched_tokens, heads, groups, 2, **kw))

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        c = common_attn_metadata
        assert c.causal is True and common_prefix_len == 0
        if os.environ.get("PI_VLLM_AUDIT") == "1":
            OBSERVED.add((c.num_reqs, c.num_actual_tokens, c.max_query_len))
        return Metadata(c.num_reqs, c.num_actual_tokens, c.max_query_len, c.max_seq_len,
                        c.query_start_loc, c.seq_lens, c.block_table_tensor, c.slot_mapping,
                        self.decode_workspace, self.prefill_workspace)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        m = self.build(0, common_attn_metadata)
        m.seq_lens.fill_(1)
        return m

    def update_block_table(self, metadata, blk_table, slot_mapping):
        return replace(metadata, block_table=blk_table, slot_mapping=slot_mapping)


class PIAttentionBackend(AttentionBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "bfloat16"]
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return Implementation

    @staticmethod
    def get_builder_cls():
        return MetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16, 32]

    @classmethod
    def supports_block_size(cls, block_size):
        return block_size in (None, 16, 32)

    @classmethod
    def get_supported_head_sizes(cls):
        return [128]

    @classmethod
    def supports_batch_invariance(cls):
        return True

    @classmethod
    def supports_kv_connector(cls):
        return False

    @classmethod
    def supports_compute_capability(cls, capability):
        return (capability.major, capability.minor) == (12, 0)

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
        assert block_size in (16, 32) and head_size == 128 and cache_dtype_str in ("auto", "bfloat16")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        return tuple(range(6 if include_num_layers_dimension else 5))

    @classmethod
    def indexes_kv_by_block_stride(cls):
        return False

    @staticmethod
    def use_cascade_attention(*args, **kwargs):
        return False


class Implementation(AttentionImpl[Metadata]):
    supports_dcp = False

    def __init__(self, num_heads, head_size, scale, num_kv_heads=None, alibi_slopes=None,
                 sliding_window=None, kv_cache_dtype="auto", logits_soft_cap=None,
                 attn_type=AttentionType.DECODER, kv_sharing_target_layer_name=None, **kwargs):
        assert head_size == 128 and num_kv_heads and num_heads % num_kv_heads == 0
        assert num_heads // num_kv_heads <= 16
        assert not alibi_slopes and sliding_window is None and not logits_soft_cap
        assert attn_type == AttentionType.DECODER and kv_sharing_target_layer_name is None
        assert kv_cache_dtype in ("auto", "bfloat16")
        assert not any(v is not None and v is not False for v in kwargs.values()), kwargs
        self.scale, self.kv_cache_dtype = scale, kv_cache_dtype
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self._ext = gemm_module().ext()

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if kv_cache.numel() == 0:
            return
        from vllm import _custom_ops as ops
        ops.reshape_and_cache_flash(key, value, kv_cache[0], kv_cache[1], slot_mapping,
                                    self.kv_cache_dtype, layer._k_scale, layer._v_scale)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        assert output_scale is None and output_block_scale is None
        if attn_metadata is None:  # vLLM memory profiling, no real request/cache
            return output.zero_()
        m = attn_metadata
        q = query[:m.num_actual_tokens].contiguous()
        out = output[:m.num_actual_tokens]
        table, lens = m.block_table[:m.num_reqs], m.seq_lens[:m.num_reqs]
        k, v = kv_cache[0], kv_cache[1]
        if m.max_query_len == 1:
            assert q.shape[0] == m.num_reqs
            self._ext.decode_out(q, k, v, table, lens, out, *m.decode_workspace, self.scale)
        else:
            self._ext.prefill_out(q, k, v, m.query_start_loc[:m.num_reqs + 1], table, lens,
                                 out, *m.prefill_workspace, m.max_query_len, m.max_seq_len, self.scale, 0)
        return output
