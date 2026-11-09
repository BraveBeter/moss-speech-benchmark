"""The optional model worker freezes random inputs, never random algorithms."""

import pytest

torch = pytest.importorskip("torch")
from moss_speech_bench.noise_tape import NoiseTape  # noqa: E402


def test_noise_replay_restores_functions_and_rejects_changed_trace():
    original = torch.randn
    frozen = {}
    with NoiseTape(frozen, "test") as recorded:
        expected = torch.randn(2, 3)
        torch.rand_like(expected)
    assert torch.randn is original
    torch.manual_seed(981)
    with NoiseTape(frozen, "test", recorded.calls):
        assert torch.equal(torch.randn(2, 3), expected)
        torch.rand_like(expected)
    assert torch.randn is original
    with pytest.raises(AssertionError, match="contract changed"):
        with NoiseTape(frozen, "test", recorded.calls):
            torch.randn(3, 2)
    assert torch.randn is original
    with pytest.raises(AssertionError, match="full fixed-noise"):
        with NoiseTape(frozen, "test", recorded.calls):
            torch.randn(2, 3)
    assert torch.randn is original
