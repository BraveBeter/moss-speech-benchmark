"""Candidate correctness against independent HF references and service invariants."""

import asyncio
import hashlib
import os
import re
import signal
import subprocess
from pathlib import Path

from .correctness_http import run_http
from .storage import digest, read_json, source_identity, write_json

GATES = tuple("C%02d" % i for i in range(9))
PROVIDERS = Path(__file__).with_name("providers")


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def load_suite(path):
    path = Path(path).resolve()
    config = read_json(path)
    workspace = (path.parent / config["workspace_root"]).resolve()
    config["workspace_root"] = str(workspace)

    def resolve(value):
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if isinstance(value, str):
            return value.replace("{workspace_root}", str(workspace))
        return value

    config = resolve(config)
    if not re.fullmatch("[0-9a-f]{40}", config.get("core_sha", "")):
        raise ValueError("Correctness requires an explicit full candidate SHA")
    profile = config.get("profile")
    if profile in ("hf-reference", "eager-base"):
        # eager-base is a legacy spelling of the reference compute adapter, not a Git pin.
        if config.get("native_norm_library") is not None:
            raise ValueError("Use the native-norm profile to qualify a native norm library")
    elif profile == "native-tail-research":
        if not all(config.get(k) for k in ("native_tail_source", "native_tail_abi")):
            raise ValueError("native-tail research requires explicit source/header ABI library")
    elif profile == "native-norm":
        if not config.get("native_norm_library"):
            raise ValueError("native-norm requires an explicit native library")
    else:
        raise ValueError("Unknown correctness profile")
    return config


def preflight(config):
    core = Path(config["core_root"]).resolve()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(core), *args], text=True).strip()

    assert git("rev-parse", "HEAD") == config["core_sha"], "Candidate SHA mismatch"
    assert not git("status", "--porcelain"), "Candidate worktree must be clean"
    verified = []
    assert config["assets"], "Asset/golden allowlist is required"
    for asset in config["assets"]:
        assert asset["level"] == "fresh_full_hash", "This implementation requires full asset hashes"
        actual = file_hash(asset["path"])
        assert actual == asset["sha256"], "Asset hash mismatch: " + asset["id"]
        verified.append(dict(asset, verified_sha256=actual))
    listed = {Path(a["path"]).resolve() for a in config["assets"]}
    required = [
        config[k]
        for k in ("model_path", "codec_path", "audio_en", "audio_zh", "voice_wav", "code_profile")
    ]
    assert all(Path(p).exists() for p in required)
    for name in ("audio_en", "audio_zh", "voice_wav", "code_profile"):
        assert Path(config[name]).resolve() in listed, "Unpinned input: " + name
    for gate in ("C02", "C04", "C05", "C06"):
        golden = config["golden"][gate]
        assert all(Path(p).resolve() in listed for p in golden.values()), "Unpinned golden"
        reference = read_json(golden["report"])
        assert reference["status"] == "passed"
        if gate == "C02":
            assert reference["serving_core_modules_imported"] == [] and len(reference["cases"]) == 8
        assert reference.get("serving_core_modules_imported") in ([], False), (
            "HF golden imported candidate modules"
        )
        if gate == "C04":
            assert reference["model_revision"] == "cff025bb41d8459d59abac0b5e44aba7f659ec9e"
            assert reference["determinism"]["seed"] == 20261027
            assert reference["requested_profile"] == {
                "attention_backend": "sdpa",
                "logits_to_keep": 0,
            }
        if "tensors" in golden:
            assert file_hash(golden["tensors"]) == reference["tensor_artifact_sha256"]
    model = Path(config["model_path"])
    index = read_json(model / "model.safetensors.index.json")
    assert all((model / name).resolve() in listed for name in set(index["weight_map"].values())), (
        "Unpinned AR shard"
    )
    codec = Path(config["codec_path"])
    assert all(
        (codec / name).resolve() in listed
        for name in ("model.safetensors", "flow/flow.pt", "flow/hift.pt", "flow/campplus.onnx")
    ), "Unpinned codec weights"
    ar = read_json(config["golden"]["C04"]["report"])
    assert ar["inputs_json_sha256"] == file_hash(config["golden"]["C02"]["report"])
    encoder = read_json(config["golden"]["C05"]["report"])
    for name, key in (("en", "audio_en"), ("cn", "audio_zh"), ("voice", "voice_wav")):
        assert encoder["audio_inputs"][name]["sha256"] == file_hash(config[key]), (
            "Golden audio identity mismatch"
        )
    full = read_json(config["golden"]["C06"]["report"])
    assert full["code_profile_sha256"] == file_hash(config["code_profile"])
    assert full["voice_conditioning"]["voice_input_sha256"] == file_hash(config["voice_wav"])
    if config["profile"] == "native-tail-research":
        assert all(
            Path(config[k]).resolve() in listed for k in ("native_tail_source", "native_tail_abi")
        ), "Unpinned native-tail research inputs"
    if config["profile"] == "native-norm":
        library = Path(config["native_norm_library"]).resolve()
        assert library in listed, "Native library must be in the pinned asset allowlist"
        assert config["deployment"].get("native_norm_library_sha256") == file_hash(library)
    deployment = config.get("deployment", {})
    assert deployment.get("core_sha") == config["core_sha"], "Deployment candidate SHA mismatch"
    gpu_count = deployment.get("gpu_count")
    assert type(gpu_count) is int and gpu_count > 0, "Invalid deployment GPU count"
    assert deployment.get("voice_sha256") == file_hash(config["voice_wav"])
    placement = deployment.get("placement", {})
    assert set(placement) == {"ar", "encoder", "vocoder"}, "Incomplete deployment placement"
    assert all(type(device) is int and 0 <= device < gpu_count for device in placement.values()), (
        "Deployment placement is outside allocated devices"
    )
    return dict(
        status="pass",
        checks=[dict(id="identity-assets-golden", status="pass")],
        identity=dict(
            core_sha=config["core_sha"],
            core_branch=git("branch", "--show-current"),
            core_dirty=False,
            core_root=str(core),
            client=source_identity(),
        ),
        assets=verified,
        deployment=deployment,
        lineage="Independent HF golden reuse; candidate results are new executions",
    )


def numerical_checks(gate, report, golden):
    assert report.get("status") == "passed", report.get("error", report.get("status"))
    checks = {}
    if gate == "C02":
        assert set(report["text_comparisons"]) == set(golden["text_encodings"])
        assert set(report["left_padding_comparisons"]) == set(golden["cases"])
        assert set(report["batch_comparisons"]) == {
            "case_order",
            "input_ids",
            "attention_mask",
            "shape",
            "padding_side",
        }
        for key in ("defaults_comparison", "pad_id_comparison"):
            checks[key] = report[key]["equal"]
        for key in ("text_comparisons", "canonical_text_comparisons", "batch_comparisons"):
            checks.update({key + "/" + name: v["equal"] for name, v in report[key].items()})
        assert set(report["cases"]) == set(golden["cases"])
        for name, case in report["cases"].items():
            for key in (
                "input_ids_comparison",
                "attention_mask_comparison",
                "prompt_len_comparison",
            ):
                checks[name + "/" + key] = case[key]["equal"]
            for key in ("grid", "mask"):
                checks[name + "/padding/" + key] = report["left_padding_comparisons"][name][key][
                    "equal"
                ]
        assert len(checks) == 76 and report["samples_unchanged"]
    elif gate == "C04":
        assert len(report["cases"]) == 8 and set(report["cases"]) == set(golden["cases"])
        for name, case in report["cases"].items():
            assert set(case["heads"]) == {"text", "audio"}
            for head, value in case["heads"].items():
                c = value["comparison"]
                checks[name + "/" + head + "/tolerance"] = c["finite"] and c["allclose"]
                checks[name + "/" + head + "/raw-top1"] = c["top1_equal"]
    elif gate == "C05":
        assert set(report["comparisons"]) == {
            "en.codes",
            "cn.codes",
            "voice.prompt_token",
            "voice.prompt_feat",
            "voice.embedding",
        }
        checks = {name: value["passed"] for name, value in report["comparisons"].items()}
    else:
        assert set(report["cases"]) == {"codes24", "codes32", "codes511", "codes512"}
        for name, case in report["cases"].items():
            checks[name] = case["status"] == "passed"
        assert report["numerical_comparison_complete"]
    assert checks and all(checks.values()), "Numerical/greedy assertions failed"
    return [dict(id=k, status="pass" if v else "fail") for k, v in checks.items()]


async def provider(gate, config, output):
    # These adapters compare the declared candidate path, never a silent eager fallback.
    supported = (
        {"ar_dtype": "bfloat16", "attention": "torch_native", "ar_tp": 1, "ar_graph": False}
        if gate == "C04"
        else {"codec_dtype": "float32"}
        if gate in ("C05", "C06")
        else {}
    )
    differences = {
        key: {"declared": config.get("deployment", {}).get(key), "supported": value}
        for key, value in supported.items()
        if config.get("deployment", {}).get(key) != value
    }
    if differences:
        return dict(
            status="unsupported",
            reason=f"{gate} adapter does not cover the declared compute path: {differences}",
        )
    output = Path(output).resolve()
    spec = config.get("providers", {}).get(gate, {})
    python = spec.get("python", config["candidate_python"])
    if not Path(python).is_file():
        return dict(status="unsupported", reason="Missing candidate interpreter: " + python)
    env = dict(
        os.environ,
        MOSS_CORE_ROOT=config["core_root"],
        MOSS_MODEL_PATH=config["model_path"],
        MOSS_AUDIO_EN=config["audio_en"],
        MOSS_AUDIO_CN=config["audio_zh"],
        MOSS_AUDIO_VOICE=config["voice_wav"],
        MOSS_CODE_PROFILE=config["code_profile"],
        MOSS_CORRECTNESS_PROFILE=config["profile"],
        MOSS_NATIVE_NORM_LIBRARY=config.get("native_norm_library", ""),
        MOSS_NATIVE_TAIL_SOURCE=config.get("native_tail_source", ""),
        MOSS_NATIVE_TAIL_ABI=config.get("native_tail_abi", ""),
        MOSS_CODEC_PATH=config["codec_path"],
        PYTHONPATH=config["core_root"],
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    if gate in ("C01", "C02"):
        env.update(SGLANG_USE_CPU_ENGINE="1", CUDA_VISIBLE_DEVICES="")
    else:
        env.pop("SGLANG_USE_CPU_ENGINE", None)
    argv = [*spec.get("launcher", []), python, str(PROVIDERS / "entry.py"), str(output), gate]
    filenames = {
        "C02": "processor.json",
        "C04": "core-weight-reference.json",
        "C05": "core-encoder-features.json",
        "C06": "core-full-codec-comparison.json",
    }
    if gate == "C02":
        argv += [
            "--reference-json",
            config["golden"][gate]["report"],
            "--output-json",
            str(output / "processor.json"),
        ]
    elif gate == "C04":
        argv += [
            "--model-path",
            config["model_path"],
            "--inputs-json",
            config["golden"]["C02"]["report"],
            "--hf-output-dir",
            str(Path(config["golden"][gate]["report"]).parent),
            "--policy-json",
            str(PROVIDERS / "policy.json"),
            "--output-dir",
            str(output),
        ]
    elif gate in ("C05", "C06"):
        argv += [
            "--codec-path",
            config["codec_path"],
            "--reference-json",
            config["golden"][gate]["report"],
            "--reference-tensors",
            config["golden"][gate]["tensors"],
            "--output-dir",
            str(output),
        ]
    write_json(
        output / "command.json",
        dict(
            argv=argv,
            cwd=config["core_root"],
            env={
                k: env[k]
                for k in ("PYTHONPATH", "MOSS_CORE_ROOT", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
            },
        ),
    )
    with (output / "provider.log").open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            *argv, cwd=config["core_root"], env=env, stdout=log, stderr=log, start_new_session=True
        )
        try:
            await asyncio.wait_for(process.wait(), spec.get("timeout", 1000))
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), 10)
                except asyncio.TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
    assert process.returncode == 0, "Provider failed; see provider.log"
    assert read_json(output / "origins.json"), "Provider import origins missing"
    if gate in ("C01", "C03"):
        report = read_json(output / "coverage.json")
        selection = read_json(PROVIDERS / "selection.json")
        expected = selection[gate] + (
            selection[gate + "-native-norm"] if config["profile"] == "native-norm" else []
        )
        assert set(report["collected"]) == set(expected) and set(report["outcomes"]) == set(
            expected
        )
        checks = [
            dict(id=n, status="pass" if report["outcomes"][n] == "passed" else "fail")
            for n in expected
        ]
        assert all(c["status"] == "pass" for c in checks), "Mandatory tests skipped or failed"
    else:
        checks = numerical_checks(
            gate, read_json(output / filenames[gate]), read_json(config["golden"][gate]["report"])
        )
    if gate == "C03" and config["profile"] == "native-tail-research":
        research = read_json(output / "native-tail-research.json")
        assert research["status"] == "unsupported" and research["total_cases"] == 32
        return dict(
            status="unsupported", reason=research["reason"], checks=checks, research=research
        )
    return dict(status="pass", checks=checks, exit_code=process.returncode)


def read_correctness(root):
    """Validate frozen correctness evidence without rewriting its report."""
    root = Path(root)
    manifest = read_json(root / "correctness.json")
    assert set(manifest["gates"]) == set(GATES), "Required gate allowlist changed"
    assert manifest["config_hash"] == digest(manifest["config"]), (
        "Suite configuration identity changed"
    )
    for gate in [*manifest["gates"].values(), *(h["result"] for h in manifest.get("history", []))]:
        assert gate["status"] in {"pass", "fail", "skip", "not_run", "unsupported"}
        if gate["status"] == "pass":
            assert gate.get("checks") and all(c["status"] == "pass" for c in gate["checks"])
        for relative, expected in gate.get("artifacts", {}).items():
            path = (root / relative).resolve()
            assert root.resolve() in path.parents and file_hash(path) == expected, (
                "Correctness artifact hash mismatch"
            )
    states = [g["status"] for g in manifest["gates"].values()]
    status = "fail" if "fail" in states else "pass" if states == ["pass"] * 9 else "incomplete"
    assert manifest["status"] == status, "Summary inconsistent with required gates"
    return manifest


def render_correctness(root):
    root = Path(root)
    manifest = read_correctness(root)
    status = manifest["status"]
    lines = [
        "# MOSS correctness",
        "",
        f"Status: {status}",
        "",
        f"Candidate commit: `{manifest['config'].get('core_sha', '未记录')}`; "
        f"branch: `{manifest['gates']['C00'].get('identity', {}).get('core_branch', '未记录')}`.",
        "",
        "Numerical oracle: independent pinned HF outputs. Git ancestry and performance baselines are not correctness criteria.",
        "",
        "Historical golden is reference evidence; only new provider/HTTP executions count.",
        "Quality and performance are separate, not evaluated by this suite.",
        "",
        "| Gate | Status | Executed assertions | Reason |",
        "|---|---|---:|---|",
    ]
    for name, gate in manifest["gates"].items():
        lines.append(
            f"| {name} | {gate['status']} | {len(gate.get('checks', []))} | {gate.get('reason', '')} |"
        )
    (root / "correctness.md").write_text("\n".join(lines) + "\n")
    return manifest


async def run_correctness(path, output, selected=None, resume=False):
    config = load_suite(path)
    output = Path(output)
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh correctness output directory")
    output.mkdir(parents=True, exist_ok=True)
    selected = set(selected.split(",")) if selected else set(GATES)
    if not selected <= set(GATES):
        raise ValueError("Unknown gate ID")
    suite = dict(
        schema_version=1,
        config=config,
        config_hash=digest(config),
        client=source_identity(),
        gates={g: dict(status="not_run", reason="Not selected") for g in GATES},
        status="incomplete",
    )
    suite["provider_data"] = {p.name: file_hash(p) for p in PROVIDERS.glob("*.json")}
    if resume:
        previous = render_correctness(output)
        if (
            previous["config_hash"] != digest(config)
            or previous["client"]["source_hash"] != suite["client"]["source_hash"]
            or previous["provider_data"] != suite["provider_data"]
        ):
            raise ValueError("Correctness resume identity mismatch; use a fresh directory")
        if previous["gates"]["C00"]["status"] == "pass":
            preflight(config)  # Revalidate external inputs/weights, not only saved output files.
        suite = previous
    failed = []
    try:
        for gate in GATES:
            old = suite["gates"][gate]
            if gate not in selected:
                if old["status"] in ("fail", "skip", "unsupported"):
                    failed.append(gate)
                continue
            if old["status"] == "pass":
                continue
            if old.get("artifacts") or old["status"] != "not_run":
                suite.setdefault("history", []).append(dict(gate=gate, result=old))
            attempt = 1 + sum(h["gate"] == gate for h in suite.get("history", []))
            folder = output / (gate if attempt == 1 else f"{gate}-attempt-{attempt:02d}")
            folder.mkdir(exist_ok=False)
            if failed:
                suite["gates"][gate] = dict(
                    status="not_run", reason="Blocked by " + ",".join(failed)
                )
                continue
            try:
                if gate == "C00":
                    result = preflight(config)
                elif gate in ("C07", "C08"):
                    result = await run_http(gate, config, folder)
                else:
                    result = await provider(gate, config, folder)
                suite["gates"][gate] = result
            except asyncio.CancelledError:
                suite["gates"][gate] = dict(
                    status="not_run",
                    reason="Execution interrupted; rerun whole gate in a fresh directory",
                )
                raise
            except Exception as exc:
                suite["gates"][gate] = dict(
                    status="fail", reason=type(exc).__name__ + ": " + str(exc)
                )
            finally:
                coverage_file = folder / "coverage.json"
                if (
                    gate in ("C01", "C03")
                    and coverage_file.exists()
                    and "checks" not in suite["gates"][gate]
                ):
                    outcomes = read_json(coverage_file)["outcomes"]
                    suite["gates"][gate]["checks"] = [
                        dict(
                            id=n,
                            status={"passed": "pass", "failed": "fail", "skipped": "skip"}.get(
                                v, "not_run"
                            ),
                        )
                        for n, v in outcomes.items()
                    ]
                suite["gates"][gate]["artifacts"] = {
                    str(p.relative_to(output)): file_hash(p)
                    for p in folder.rglob("*")
                    if p.is_file()
                }
                if suite["gates"][gate]["status"] != "pass":
                    failed.append(gate)
                write_json(output / "correctness.json", suite)
    finally:
        states = [g["status"] for g in suite["gates"].values()]
        suite["status"] = (
            "fail" if "fail" in states else "pass" if states == ["pass"] * 9 else "incomplete"
        )
        suite["coverage"] = {
            s: sum(g["status"] == s for g in suite["gates"].values())
            for s in ("pass", "fail", "skip", "not_run", "unsupported")
        }

        def group(keys):
            states = [suite["gates"][k]["status"] for k in keys]
            return (
                "fail"
                if "fail" in states
                else "pass"
                if all(s == "pass" for s in states)
                else "incomplete"
            )

        suite["summary"] = dict(
            correctness=group(("C00", "C01", "C03", "C07", "C08")),
            numerical_alignment=group(("C02", "C04", "C05", "C06")),
            greedy_equivalence=suite["gates"]["C04"]["status"],
            quality="not_evaluated",
            performance="not_evaluated",
        )
        write_json(output / "correctness.json", suite)
        render_correctness(output)
    return suite
