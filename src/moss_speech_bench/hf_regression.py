"""Client-owned HF regression policy and isolated, versioned backend workers."""

import asyncio
import os
import shutil
import signal
import subprocess
from pathlib import Path

from safetensors.numpy import load_file

from .hf_contract import (
    GATE_NAMES,
    PROTOCOL,
    compare,
    deployment_binding,
    file_hash,
    load_config,
    oracle_binding,
    preflight,
    reference,
    reference_files,
    validate_assets,
)
from .providers.allocation import local_devices
from .storage import digest, read_json, source_identity, write_json

MODULE = Path(__file__).parent


def prepare_config(path, output):
    """Pin existing local assets before any model execution; preserve interpreter symlinks."""
    path = Path(path).resolve()
    config = read_json(path)
    workspace = (path.parent / config["workspace_root"]).resolve()
    assets = []
    for kind, field in (
        ("model", "model_path"),
        ("codec", "codec_path"),
        ("source", "hf_source_root"),
    ):
        directory = (workspace / config[field]).resolve()
        for asset in sorted(directory.rglob("*")):
            if (
                not asset.is_file()
                or "__pycache__" in asset.parts
                or ".git" in asset.parts
                or (kind == "source" and asset.suffix != ".py")
            ):
                continue
            assets.append(
                dict(
                    id=kind + "/" + str(asset.relative_to(directory)),
                    path=os.path.relpath(asset, workspace),
                    sha256=file_hash(asset),
                )
            )
    voice = (workspace / config["voice_wav"]).resolve()
    assets.append(dict(id="voice", path=os.path.relpath(voice, workspace), sha256=file_hash(voice)))
    config["assets"] = assets
    if not config.get("core_sha"):
        config["core_sha"] = subprocess.check_output(
            ["git", "-C", str(workspace / config["core_root"]), "rev-parse", "HEAD"], text=True
        ).strip()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preparation requires a fresh execution directory")
    output.mkdir(parents=True, exist_ok=True)
    config["workspace_root"] = os.path.relpath(workspace, output)
    write_json(output / "correctness.json", config)
    resolved = load_config(output / "correctness.json")
    write_json(
        output / "deployment.expected.json",
        {
            "server": {"commit": resolved["core_sha"], "dirty": False},
            "hf_regression": deployment_binding(resolved),
        },
    )
    return dict(status="complete", config=str(output / "correctness.json"), assets=len(assets))


async def worker(config, kind, phase, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "config.json"
    write_json(resolved, config)
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    env.pop("SGLANG_USE_CPU_ENGINE", None)
    visible = local_devices(config)
    if visible is not None:
        env.update(
            CUDA_VISIBLE_DEVICES=visible,
            MOSS_LOCAL_GPU_DEVICES=visible,
            MOSS_LOCAL_GPU_AUTHORIZATION=config["local_gpu_allocation"]["authorization"],
        )
    elif not env.get("SLURM_JOB_ID"):
        raise ValueError("Explicit GPU allocation required for numerical workers")
    env["PYTHONPATH"] = str(MODULE.parent) + (
        os.pathsep + config["core_root"] if kind == "candidate" else ""
    )
    script = MODULE / ("native_probe.py" if kind == "candidate" else "hf_oracle.py")
    argv = [
        config["candidate_python" if kind == "candidate" else "reference_python"],
        str(script),
        phase,
        str(resolved),
        str(output),
    ]
    write_json(output / "command.json", dict(argv=argv, visible_devices=visible))
    with (output / "worker.log").open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            *argv,
            env=env,
            cwd=config["workspace_root"],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            await asyncio.wait_for(process.wait(), config.get("worker_timeout", 1200))
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), 10)
                except asyncio.TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
    assert process.returncode == 0, f"{kind} {phase} worker failed; inspect worker.log"
    report = read_json(output / (phase + ".json"))
    assert report["status"] == "complete"
    return report


def numerical(gate, config, root, candidate):
    bundle = reference(config)
    checks = []
    if gate in ("C01", "C02", "C03", "C04"):
        report = read_json(candidate / "ar.json")
        expected = read_json(bundle / "ar.json")
        if config.get("reference_attention_backend") == "flashinfer":
            assert report["torch"] == expected["torch"], "HF/SG Torch runtime differs"
            assert report["flashinfer"] == expected["flashinfer"]["version"], (
                "HF/SG FlashInfer runtime differs"
            )
        by_id = {case["id"]: case for case in expected["cases"]}
        assert set(by_id) == {c["id"] for c in config["cases"]}
        if gate == "C01":
            for case in report["cases"]:
                if case["group"] == "teacher-b1":
                    checks.append(
                        dict(
                            id=case["case_id"],
                            status="pass"
                            if case["prompt_rows"] == by_id[case["case_id"]]["prompt_rows"]
                            else "fail",
                        )
                    )
            assert len(checks) == len(by_id)
        elif gate == "C04":
            for case in report["cases"]:
                if case["group"].startswith("greedy-"):
                    checks.append(
                        dict(
                            id=case["group"] + "/" + case["case_id"],
                            status="pass"
                            if case["rows"] == by_id[case["case_id"]]["greedy_rows"]
                            else "fail",
                            actual_rows=case["rows"],
                            expected_rows=by_id[case["case_id"]]["greedy_rows"],
                        )
                    )
            assert len(checks) == len(by_id) * 2
        else:
            actual = load_file(candidate / "ar.safetensors")
            expected = load_file(bundle / "ar.safetensors")
            for size in [1] if gate == "C02" else config["batch_sizes"]:
                for case in config["cases"]:
                    for step in [0] if gate == "C02" else range(1, len(config["teacher_rows"]) + 1):
                        for head in ("text", "audio"):
                            key = f"teacher-b{size}/{case['id']}/{step}/{head}"
                            checks.append(
                                compare(
                                    actual[key],
                                    expected[f"teacher/{case['id']}/{step}/{head}"],
                                    0.25,
                                    0.02,
                                    top1=True,
                                    id=key,
                                )
                            )
            if gate == "C03":
                for size in config["batch_sizes"]:
                    active = any(
                        r["group"] == f"teacher-b{size}" and r["decode"] and r["active"] == size
                        for r in report["observations"]
                    )
                    replay = any(r["group"] == f"teacher-b{size}" for r in report["graph_replays"])
                    checks.append(
                        dict(
                            id=f"declared-decode-path-B{size}",
                            status="pass"
                            if active and replay == (not config["runtime"]["disable_cuda_graph"])
                            else "fail",
                            active_batch_observed=active,
                            replay_observed=replay,
                        )
                    )
                assert (
                    report["runtime"]["attention_backend"] == config["runtime"]["attention_backend"]
                )
                assert (
                    report["runtime"]["disable_cuda_graph"]
                    == config["runtime"]["disable_cuda_graph"]
                )
    else:
        if config.get("reference_attention_backend") == "flashinfer":
            report = read_json(candidate / "codec.json")
            expected_report = read_json(bundle / "codec.json")
            for name in ("torch", "torchaudio", "onnxruntime"):
                assert report[name] == expected_report[name], "HF/SG Codec runtime differs: " + name
        actual = load_file(candidate / "codec.safetensors")
        expected = load_file(bundle / "codec.safetensors")
        keys = (
            ["voice/prompt_token", "voice/prompt_feat", "voice/embedding"]
            if gate == "C05"
            else [f"pcm/{n}" for n in config["codec_code_counts"]]
        )
        for key in keys:
            checks.append(compare(actual[key], expected[key], 1e-5, 1e-5, id=key))
    assert checks
    result = dict(
        status="pass" if all(c["status"] == "pass" for c in checks) else "fail", checks=checks
    )
    write_json(root / "comparisons.json", result)
    return result


async def export_reference(path, output):
    config = load_config(path)
    output = Path(output)
    if output.exists() and any(p.name != "execution" for p in output.iterdir()):
        raise ValueError("HF reference requires a fresh output directory")
    validate_assets(config)
    output.mkdir(parents=True, exist_ok=True)
    for phase in ("ar", "codec"):
        folder = output / "execution" / phase
        await worker(config, "reference", phase, folder)
        for suffix in (".json", ".safetensors"):
            shutil.copyfile(folder / (phase + suffix), output / (phase + suffix))
    shutil.copyfile(MODULE / "hf_oracle.py", output / "execution/hf_oracle.py")
    shutil.copyfile(MODULE / "noise_tape.py", output / "execution/noise_tape.py")
    if config.get("reference_attention_backend") == "flashinfer":
        shutil.copyfile(MODULE / "hf_flashinfer.py", output / "execution/hf_flashinfer.py")
    files = {name: file_hash(output / name) for name in reference_files(config)}
    result = dict(
        protocol=PROTOCOL,
        status="complete",
        binding=oracle_binding(config),
        files=files,
        client=source_identity(),
    )
    write_json(output / "reference.json", result)
    from .reporting import code_identity_lines

    identity_lines = "\n".join(code_identity_lines({"client": result["client"], "contract": {}}))
    (output / "summary.md").write_text(
        "# Independent HF reference\n\n"
        + identity_lines
        + "\n\n"
        + "Protocol: `"
        + PROTOCOL
        + "`. Local official HF sources, "
        + config.get("reference_attention_backend", "sdpa")
        + "/BF16 AR and FP32 Codec; no candidate imports. Frozen data is reusable only with the identical asset/input/policy binding.\n\n## 正确性测试\n\nReference export complete; this does not qualify a candidate.\n\n| 检查项 | 通过/全部 | 状态 |\n|---|---:|---|\n\n## 性能测试\n\nNot executed.\n\n| 数据集 | 指标 | 结果 |\n|---|---|---|\n\n## 质量测试\n\nNot executed.\n\n| 数据集 | 指标 | 结果 |\n|---|---|---|\n"
    )
    return result


async def run(config, output, selected=None, resume=False):
    from .correctness import read_correctness, render_correctness
    from .hf_service import run_http

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = set(selected.split(",")) if selected else set(GATE_NAMES)
    if not selected <= set(GATE_NAMES):
        raise ValueError("Unknown gate ID")
    suite = dict(
        schema_version=2,
        protocol=PROTOCOL,
        gate_names=GATE_NAMES,
        config=config,
        config_hash=digest(config),
        client=source_identity(),
        status="incomplete",
        gates={gate: dict(status="not_run", reason="Not selected") for gate in GATE_NAMES},
    )
    if resume:
        suite = read_correctness(output)
        assert (
            suite["config_hash"] == digest(config)
            and suite["client"]["source_hash"] == source_identity()["source_hash"]
        ), "Regression resume identity mismatch"
        if suite["gates"]["C00"]["status"] == "pass":
            assert preflight(config)["status"] == "pass"
    elif any(output.iterdir()):
        raise ValueError("Use a fresh correctness directory")
    try:
        for gate in GATE_NAMES:
            if gate not in selected or suite["gates"][gate]["status"] == "pass":
                continue
            previous = suite["gates"][gate]
            if previous["status"] != "not_run" or previous.get("artifacts"):
                suite.setdefault("history", []).append(dict(gate=gate, result=previous))
            attempt = 1 + sum(h["gate"] == gate for h in suite.get("history", []))
            folder = output / (gate if attempt == 1 else f"{gate}-attempt-{attempt:02d}")
            folder.mkdir(exist_ok=False)
            result = dict(status="not_run", reason="Execution interrupted")
            try:
                if gate == "C00":
                    result = preflight(config)
                elif suite["gates"]["C00"]["status"] != "pass":
                    result = dict(
                        status="not_run", reason="Requires valid C00 identity/assets/reference"
                    )
                elif gate in ("C07", "C08"):
                    result = await run_http(gate, config, folder)
                else:
                    phase = "ar" if gate in ("C01", "C02", "C03", "C04") else "codec"
                    producer = "C01" if phase == "ar" else "C05"
                    artifact = next(
                        (
                            k
                            for k in suite["gates"][producer].get("artifacts", {})
                            if k.endswith("/" + phase + ".json")
                        ),
                        None,
                    )
                    candidate = output / Path(artifact).parent if artifact else folder / "candidate"
                    if artifact is None and gate != producer:
                        raise RuntimeError("Required backend worker did not complete: " + producer)
                    if artifact is None:
                        await worker(config, "candidate", phase, candidate)
                    result = numerical(gate, config, folder, candidate)
            except asyncio.CancelledError:
                result = dict(status="not_run", reason="Interrupted; rerun whole gate")
                raise
            except Exception as error:
                result = dict(status="fail", reason=type(error).__name__ + ": " + str(error))
            finally:
                suite["gates"][gate] = result
                suite["gates"][gate]["artifacts"] = {
                    **(previous.get("artifacts", {}) if gate in ("C01", "C05") else {}),
                    **{
                        str(p.relative_to(output)): file_hash(p)
                        for p in folder.rglob("*")
                        if p.is_file()
                    },
                }
                states = [v["status"] for v in suite["gates"].values()]
                suite["status"] = (
                    "fail"
                    if "fail" in states
                    else "pass"
                    if all(s == "pass" for s in states)
                    else "incomplete"
                )
                write_json(output / "correctness.json", suite)
    finally:
        render_correctness(output)
    return suite
