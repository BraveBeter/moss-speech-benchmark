"""Explicit GPU allocation declarations for Slurm and operator-owned local devices."""

import os
import re


def local_devices(config):
    allocation = config.get("local_gpu_allocation")
    if allocation is None:
        return None
    if (
        not isinstance(allocation, dict)
        or set(allocation) != {"devices", "authorization"}
        or not isinstance(allocation["authorization"], str)
        or not allocation["authorization"].strip()
        or not isinstance(allocation["devices"], list)
        or not allocation["devices"]
        or any(type(device) is not int or device < 0 for device in allocation["devices"])
        or len(set(allocation["devices"])) != len(allocation["devices"])
    ):
        raise ValueError("local_gpu_allocation requires unique device indices and authorization")
    return ",".join(map(str, allocation["devices"]))


def require_gpu_allocation():
    if os.environ.get("SLURM_JOB_ID"):
        return
    devices = os.environ.get("MOSS_LOCAL_GPU_DEVICES", "")
    assert (
        re.fullmatch(r"\d+(,\d+)*", devices)
        and devices == os.environ.get("CUDA_VISIBLE_DEVICES")
        and os.environ.get("MOSS_LOCAL_GPU_AUTHORIZATION", "").strip()
    ), "CUDA providers require Slurm or an explicit operator-owned local GPU allocation"
