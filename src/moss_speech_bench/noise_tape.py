"""Freeze decoder stochastic inputs across framework versions without changing layers."""

import torch


class NoiseTape:
    def __init__(self, tensors, prefix, expected=None):
        self.tensors, self.prefix, self.expected = tensors, prefix, expected
        self.calls = []
        self.originals = {}

    def __enter__(self):
        for name in ("rand", "randn", "rand_like", "randn_like"):
            original = getattr(torch, name)
            self.originals[name] = original

            def draw(*args, function=name, original=original, **kwargs):
                actual = original(*args, **kwargs)
                index = len(self.calls)
                call = dict(
                    function=function,
                    shape=list(actual.shape),
                    dtype=str(actual.dtype),
                    device=actual.device.type,
                )
                self.calls.append(call)
                key = f"noise/{self.prefix}/{index}"
                if self.expected is None:
                    self.tensors[key] = actual.detach().cpu().clone().contiguous()
                    return actual
                assert index < len(self.expected) and call == self.expected[index], (
                    "Decoder random input contract changed"
                )
                return self.tensors[key].to(device=actual.device, dtype=actual.dtype).clone()

            setattr(torch, name, draw)
        return self

    def __exit__(self, exception_type, exception, traceback):
        for name, original in self.originals.items():
            setattr(torch, name, original)
        if exception_type is None and self.expected is not None:
            assert len(self.calls) == len(self.expected), (
                "Decoder did not consume the full fixed-noise trace"
            )
        return False
