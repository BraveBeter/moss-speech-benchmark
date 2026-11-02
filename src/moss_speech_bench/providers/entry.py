"""Run only in the explicitly selected candidate environment, never in the client."""

import importlib
import json
import os
import runpy
import sys
from pathlib import Path

from allocation import require_gpu_allocation
from identity import collect_origins
from selection import expected_tests

core = Path(os.environ["MOSS_CORE_ROOT"]).resolve()
sys.path.insert(0, str(core))
output = Path(sys.argv[1])
mode = sys.argv[2]
arguments = sys.argv[3:]


def origins():
    return collect_origins(core)


status = 1
if mode in ("C01", "C02"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
else:
    os.environ.pop("SGLANG_USE_CPU_ENGINE", None)
if mode in ("C03", "C04", "C05", "C06"):
    require_gpu_allocation()
    import torch

    assert torch.cuda.is_available(), "Allocated CUDA devices are unavailable"
    (output / "gpu-allocation.json").write_text(
        json.dumps(
            {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "local_devices": os.environ.get("MOSS_LOCAL_GPU_DEVICES"),
                "authorization": os.environ.get("MOSS_LOCAL_GPU_AUTHORIZATION"),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "devices": [
                    str(torch.cuda.get_device_properties(i))
                    for i in range(torch.cuda.device_count())
                ],
            },
            indent=2,
        )
        + "\n"
    )
try:
    importlib.import_module("sglang_omni")
    origins()
    if mode in ("C01", "C03"):
        import pytest

        selection = json.loads(Path(__file__).with_name("selection.json").read_text())
        expected = expected_tests(
            selection,
            mode,
            os.environ.get("MOSS_CORRECTNESS_PROFILE", "hf-reference"),
            os.environ.get("MOSS_CONTEXT_REMAINING") == "1",
            os.environ.get("MOSS_CORRECTNESS_TEST_SUITE", "legacy"),
            json.loads(os.environ.get("MOSS_AR_OPTIMIZATIONS", "[]")),
        )

        class Coverage:
            def __init__(self):
                self.collected, self.outcomes = [], {}

            def pytest_collection_finish(self, session):
                self.collected = [i.nodeid for i in session.items]
                if set(self.collected) != set(expected) or len(self.collected) != len(expected):
                    raise pytest.UsageError("Collected node IDs differ from fixed selection")

            def pytest_runtest_logreport(self, report):
                if report.failed or report.skipped or report.when == "call":
                    previous = self.outcomes.get(report.nodeid)
                    if previous != "failed":
                        self.outcomes[report.nodeid] = report.outcome

        plugin = Coverage()
        status = int(
            pytest.main(
                [*expected, "-q", "--junitxml=" + str(output / "junit.xml")], plugins=[plugin]
            )
        )
        data = {"expected": expected, "collected": plugin.collected, "outcomes": plugin.outcomes}
        (output / "coverage.json").write_text(json.dumps(data, indent=2) + "\n")
        if set(plugin.outcomes) != set(expected) or any(
            v != "passed" for v in plugin.outcomes.values()
        ):
            status = 1
        if (
            status == 0
            and mode == "C03"
            and os.environ.get("MOSS_CORRECTNESS_PROFILE") == "native-tail-research"
        ):
            from native_tail import run

            run(output)
    else:
        script = Path(__file__).with_name(
            {"C02": "processor.py", "C04": "ar.py", "C05": "encoder.py", "C06": "codec.py"}[mode]
        )
        sys.argv = [str(script), *arguments]
        try:
            runpy.run_path(str(script), run_name="__main__")
            status = 0
        except SystemExit as exc:
            status = exc.code or 0
finally:
    try:
        (output / "origins.json").write_text(json.dumps(origins(), indent=2) + "\n")
    except Exception:
        status = 1
        raise
raise SystemExit(status)
