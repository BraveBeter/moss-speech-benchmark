"""CUDA checks for the independent HF attention hook, including mask rejection."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flashinfer")
pytest.importorskip("transformers")

FlashInferAttention = pytest.importorskip("moss_speech_bench.hf_flashinfer").FlashInferAttention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("query_length,key_length", [(5, 5), (1, 6), (3, 8)])
def test_flashinfer_matches_independent_sdpa_with_bottom_right_causality(query_length, key_length):
    torch.manual_seed(42)
    attention = FlashInferAttention(torch.device("cuda:0"))
    query = torch.randn(1, 32, query_length, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 8, key_length, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    mask = torch.arange(key_length, device="cuda")[None, :] <= (
        torch.arange(query_length, device="cuda")[:, None] + key_length - query_length
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask[None, None], enable_gqa=True
    ).transpose(1, 2)
    actual, weights = attention(
        SimpleNamespace(training=False), query, key, value, mask[None, None], scaling=128**-0.5
    )
    assert weights is None
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    repeated, _ = attention(
        SimpleNamespace(training=False), query, key, value, mask[None, None], scaling=128**-0.5
    )
    assert torch.equal(actual, repeated)


def test_flashinfer_rejects_unregistered_or_noncausal_masks():
    attention = FlashInferAttention(torch.device("cuda:0"))
    query = torch.zeros(1, 32, 3, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.zeros(1, 8, 3, 128, device="cuda", dtype=torch.bfloat16)
    for mask in (None, torch.ones(1, 1, 3, 3, device="cuda", dtype=torch.bool)):
        with pytest.raises(ValueError, match="mask"):
            attention(SimpleNamespace(training=False), query, key, key, mask, scaling=128**-0.5)


def test_flashinfer_rejects_training_and_unsupported_batches():
    attention = FlashInferAttention(torch.device("cuda:0"))
    query = torch.zeros(2, 32, 3, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.zeros(2, 8, 3, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="eval"):
        attention(SimpleNamespace(training=True), query, key, key, None, scaling=128**-0.5)
    with pytest.raises(ValueError, match="B1"):
        attention(SimpleNamespace(training=False), query, key, key, None, scaling=128**-0.5)
