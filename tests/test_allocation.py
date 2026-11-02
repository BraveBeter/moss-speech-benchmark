import pytest

from moss_speech_bench.providers.allocation import local_devices, require_gpu_allocation


def test_local_allocation_is_explicit_and_matches_visible_devices(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("MOSS_LOCAL_GPU_DEVICES", raising=False)
    monkeypatch.delenv("MOSS_LOCAL_GPU_AUTHORIZATION", raising=False)
    assert local_devices({}) is None
    with pytest.raises(AssertionError, match="explicit"):
        require_gpu_allocation()
    declared = {"devices": [1, 2], "authorization": "Operator assigned idle GPUs for this run"}
    assert local_devices({"local_gpu_allocation": declared}) == "1,2"
    monkeypatch.setenv("MOSS_LOCAL_GPU_DEVICES", "1,2")
    monkeypatch.setenv("MOSS_LOCAL_GPU_AUTHORIZATION", declared["authorization"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(AssertionError):
        require_gpu_allocation()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
    require_gpu_allocation()
    for devices in ([], [1, 1], [True], [-1], "1"):
        with pytest.raises(ValueError):
            local_devices({"local_gpu_allocation": dict(declared, devices=devices)})
    with pytest.raises(ValueError):
        local_devices({"local_gpu_allocation": dict(declared, authorization="")})
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("MOSS_LOCAL_GPU_DEVICES")
    require_gpu_allocation()
