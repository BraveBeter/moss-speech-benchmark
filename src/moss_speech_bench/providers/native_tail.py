"""Current-header ABI and actual legacy Lt algorithm research, never a service certificate."""

import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
from types import ModuleType


def run(output):
    import torch

    output = Path(output)
    source = Path(os.environ["MOSS_NATIVE_TAIL_SOURCE"])
    library = ctypes.CDLL(os.environ["MOSS_NATIVE_TAIL_ABI"])
    names = (
        "algo_size",
        "heuristic_size",
        "workspace_offset",
        "state_offset",
        "waves_offset",
        "reserved_offset",
        "header_version",
    )
    header = {}
    for name in names:
        function = getattr(library, name)
        function.argtypes, function.restype = [], ctypes.c_int
        header[name] = function()
    original = source.read_text()
    module = ModuleType("legacy_direct_lt_reference")
    exec(compile(original, str(source), "exec"), module.__dict__)
    torch.cuda.set_device(0)
    torch.empty((16, 16), device="cuda") @ torch.empty((16, 16), device="cuda")
    loaded, path = module._open_loaded_library()
    module._bind_library(loaded)
    version = int(loaded.cublasLtGetVersion())
    actual = dict(
        algo_size=ctypes.sizeof(module._Algo), heuristic_size=ctypes.sizeof(module._HeuristicResult)
    )
    for name, field in (
        ("workspace_offset", "workspaceSize"),
        ("state_offset", "state"),
        ("waves_offset", "wavesCount"),
        ("reserved_offset", "reserved"),
    ):
        actual[name] = getattr(module._HeuristicResult, field).offset
    assert all(header[k] == v for k, v in actual.items()), "Current header and ctypes ABI differ"
    try:
        context = module.DirectLtContext(torch.device("cuda:0"))
    except RuntimeError as error:
        rejected = str(error)
        assert "12.8.4" in rejected, rejected
    else:
        context.close()
        raise AssertionError("Expected original version guard to reject current runtime")
    tree = ast.parse(original)
    changes = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Attribute)
            and node.left.attr == "version"
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value == 120804
        ):
            node.comparators[0].value = version
            changes.append(node.lineno)
    assert len(changes) == 1
    diagnostic = ModuleType("diagnostic_direct_lt_version_override")
    exec(compile(tree, "<diagnostic-version-override>", "exec"), diagnostic.__dict__)
    state = torch.load(
        Path(os.environ["MOSS_CODEC_PATH"]) / "flow/flow.pt", map_location="cpu", weights_only=True
    )
    if "state_dict" in state:
        state = state["state_dict"]
    cases = []
    before_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.inference_mode():
            for profile, width, features, rows_set in (
                ("output", 512, 256, (224, 240, 288, 440, 2784, 2800, 2848, 3000)),
                ("output", 1024, 256, (224, 240, 288, 440, 2784, 2800, 2848, 3000)),
                ("ff_first", 256, 1024, (224, 240, 288, 440, 2784, 2800, 2848, 3000)),
                ("qkv_1392", 256, 512, (224, 2784)),
                ("qkv_1400", 256, 512, (240, 2800)),
                ("qkv_1424", 256, 512, (288, 2848)),
                ("qkv_1500", 256, 512, (440, 3000)),
            ):
                biased = profile in ("output", "ff_first")
                matches = [
                    (name, value)
                    for name, value in state.items()
                    if isinstance(value, torch.Tensor)
                    and name.endswith(".weight")
                    and tuple(value.shape) == (features, width)
                    and ((name[:-6] + "bias" in state) == biased)
                ]
                assert matches, (profile, width, features)
                name, tensor = matches[0]
                weight = tensor.to(device="cuda", dtype=torch.float32)
                bias = (
                    state[name[:-6] + "bias"].to(device="cuda", dtype=torch.float32)
                    if biased
                    else None
                )
                for rows in rows_set:
                    result = dict(profile=profile, rows=rows, weight_name=name, k=width, n=features)
                    plan = None
                    try:
                        plan = diagnostic.DirectLtProjection(weight, bias, rows, profile=profile)
                        generator = torch.Generator(device="cuda").manual_seed(20261031 + rows)
                        value = torch.randn(
                            (2, rows // 2, width),
                            generator=generator,
                            device="cuda",
                            dtype=torch.float32,
                        )
                        reference = torch.nn.functional.linear(value, weight, bias)
                        candidate = plan.run(value).clone()
                        torch.cuda.synchronize()
                        result.update(
                            api_success=True,
                            exact=bool(
                                torch.equal(
                                    candidate.view(torch.uint8), reference.view(torch.uint8)
                                )
                            ),
                            differing_values=int((candidate != reference).sum()),
                            maximum_absolute_error=float((candidate - reference).abs().max()),
                            plan=plan.describe(),
                        )
                    except (RuntimeError, ValueError) as error:
                        result.update(api_success=False, exact=False, error=repr(error))
                    finally:
                        if plan is not None:
                            plan.close()
                            result["closed"] = plan.status()["closed"]
                    cases.append(result)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before_tf32
    report = dict(
        status="unsupported",
        reason="Original guarded provider rejects this runtime; diagnostic override is not qualified complete full/tail CFM or a service implementation",
        header=header,
        ctypes_layout=actual,
        loaded_library=str(path),
        loaded_version=version,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        original_guard_rejection=rejected,
        diagnostic_version_override_lines=changes,
        cases=cases,
        exact_cases=sum(c["exact"] for c in cases),
        total_cases=len(cases),
        tf32_restored=torch.backends.cuda.matmul.allow_tf32 == before_tf32,
        limits="Actual flow checkpoint affine weights with synthetic inputs; not full CFM/PCM or end-to-end performance. No original source/version guard was modified.",
    )
    (output / "native-tail-research.json").write_text(json.dumps(report, indent=2) + "\n")
