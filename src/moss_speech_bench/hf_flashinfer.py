"""Independent FlashInfer attention for the official HF model's public registry."""

from collections import Counter

import flashinfer
import torch
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, eager_mask


class FlashInferAttention:
    """Unpadded B1 HF reference; HF retains projections, RoPE and DynamicCache."""

    def __init__(self, device):
        self.workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.prefill = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace, "NHD", backend="fa2"
        )
        self.decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace, "NHD", backend="fa2", use_tensor_cores=True
        )
        self.last_plan = None
        self.calls = Counter()

    def __call__(
        self,
        module,
        query,
        key,
        value,
        attention_mask,
        *,
        dropout=0.0,
        scaling,
        sliding_window=None,
        **kwargs,
    ):
        if module.training or dropout != 0 or sliding_window is not None:
            raise ValueError(
                "HF FlashInfer reference requires eval, zero dropout and full attention"
            )
        if query.shape[0] != 1 or key.shape[0] != 1 or value.shape != key.shape:
            raise ValueError("HF FlashInfer reference requires unpadded B1 attention")
        if query.dtype != torch.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValueError("HF FlashInfer reference requires BF16 Q/K/V")
        query_length, key_length = query.shape[-2], key.shape[-2]
        allowed = torch.arange(key_length, device=query.device)[None, :] <= (
            torch.arange(query_length, device=query.device)[:, None] + key_length - query_length
        )
        if attention_mask is None:
            raise ValueError("Registered causal mask required; do not silently drop HF masking")
        actual = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
        if actual.shape != (1, 1, query_length, key_length) or not torch.equal(
            actual[0, 0], allowed
        ):
            raise ValueError("Only the unpadded bottom-right causal mask is qualified")
        query = query[0].transpose(0, 1).contiguous()
        key = key[0].transpose(0, 1).contiguous()
        value = value[0].transpose(0, 1).contiguous()
        geometry = (query_length, key_length, query.shape[1], key.shape[1], query.shape[2], scaling)
        if geometry != self.last_plan:
            key_indptr = torch.tensor([0, key_length], dtype=torch.int32, device=query.device)
            if query_length == 1:
                self.decode.plan(
                    key_indptr,
                    torch.arange(key_length, dtype=torch.int32, device=query.device),
                    torch.ones(1, dtype=torch.int32, device=query.device),
                    query.shape[1],
                    key.shape[1],
                    query.shape[2],
                    1,
                    q_data_type=query.dtype,
                    kv_data_type=key.dtype,
                    sm_scale=scaling,
                )
            else:
                self.prefill.plan(
                    torch.tensor([0, query_length], dtype=torch.int32, device=query.device),
                    key_indptr,
                    query.shape[1],
                    key.shape[1],
                    query.shape[2],
                    causal=True,
                    q_data_type=query.dtype,
                    kv_data_type=key.dtype,
                    sm_scale=scaling,
                )
            self.last_plan = geometry
        if query_length == 1:
            output = self.decode.run(query, (key.unsqueeze(1), value.unsqueeze(1)))
            phase = "decode"
        else:
            output = self.prefill.run(query, key, value)
            phase = "prefill"
        self.calls[f"{phase}/q{query_length}/kv{key_length}"] += 1
        return output.unsqueeze(0), None

    def register(self):
        AttentionInterface.register("flashinfer", self)
        AttentionMaskInterface.register("flashinfer", eager_mask)

    def report(self):
        assert any(k.startswith("prefill/") for k in self.calls)
        assert any(k.startswith("decode/") for k in self.calls)
        return dict(
            version=flashinfer.__version__,
            prefill="BatchPrefillWithRaggedKVCacheWrapper",
            decode="BatchDecodeWithPagedKVCacheWrapper",
            backend="fa2",
            page_size=1,
            use_tensor_cores=True,
            mask="validated-unpadded-bottom-right-causal",
            calls=dict(self.calls),
        )
