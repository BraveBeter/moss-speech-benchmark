"""Fail closed when the new decode path or its independent oracle is absent."""

import copy
import json

import pytest

from moss_speech_bench.correctness import load_suite
from moss_speech_bench.providers.ar_combination import validate_activation
from moss_speech_bench.providers.batched_rope import validate_qualification
from moss_speech_bench.providers.selection import selected_nodes


def configuration(tmp_path, **changes):
    config = dict(
        workspace_root=".",
        core_sha="a" * 40,
        profile="ar-batched-rope",
        context_remaining=True,
        deployment={"decode_rope_reuse": True},
        service_activation_log=str(tmp_path / "server.log"),
    )
    config.update(changes)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def proof():
    return dict(
        batch_sizes=[2, 4],
        executions={"2": 4, "4": 2},
        complete_heads_exact=True,
        all_kv_exact=True,
        kv_layers=40,
        cases={
            str(b): {
                str(n): {
                    head: dict(finite=True, allclose=True, top1_equal=True)
                    for head in ("text", "audio")
                }
                for n in range(8)
            }
            for b in (2, 4)
        },
    )


def test_explicit_profile_and_fixed_nodes(tmp_path):
    assert load_suite(configuration(tmp_path))["profile"] == "ar-batched-rope"
    for gate in ("C01", "C03"):
        nodes = selected_nodes(gate, "ar-batched-rope", True)
        assert any("decode_rope_reuse" in node for node in nodes)
        baseline = set(selected_nodes(gate, "hf-reference", True))
        factory = "tests/unit_test/moss_speech/test_config.py::test_ar_factory_preflights_and_forwards_placement_budget"
        if gate == "C01":
            baseline.remove(factory)
            assert factory + "[False]" in nodes and factory + "[True]" in nodes
        assert baseline <= set(nodes)


@pytest.mark.parametrize(
    "change",
    [
        {"deployment": {}},
        {"deployment": {"decode_rope_reuse": 1}},
        {"context_remaining": False},
        {"service_activation_log": ""},
        {"service_activation_required": []},
        {"service_activation_offset": -1},
        {"deployment": {"decode_rope_reuse": True, "ar_graph": True}},
        {"deployment": {"decode_rope_reuse": True, "use_host_attention_metadata": True}},
        {"native_norm_library": "/not-qualified.so"},
        {"profile": "hf-reference"},
    ],
)
def test_profile_rejects_missing_or_additional_paths(tmp_path, change):
    with pytest.raises(ValueError):
        load_suite(configuration(tmp_path, **change))


def test_complete_proof_and_all_hf_heads_required():
    original = proof()
    validate_qualification(original, [str(n) for n in range(8)])
    for key, value in (
        ("batch_sizes", [2]),
        ("executions", {"2": 1, "4": 0}),
        ("executions", {"2": True, "4": 1}),
        ("all_kv_exact", False),
        ("kv_layers", 39),
    ):
        broken = dict(original, **{key: value})
        with pytest.raises(AssertionError):
            validate_qualification(broken, [str(n) for n in range(8)])
    for field in ("finite", "allclose", "top1_equal"):
        broken = copy.deepcopy(original)
        broken["cases"]["4"]["7"]["audio"][field] = False
        with pytest.raises(AssertionError):
            validate_qualification(broken, [str(n) for n in range(8)])
    broken = copy.deepcopy(original)
    del broken["cases"]["2"]["7"]
    with pytest.raises(AssertionError):
        validate_qualification(broken, [str(n) for n in range(8)])


def test_actual_service_marker_required(tmp_path):
    config = load_suite(configuration(tmp_path))
    path = tmp_path / "server.log"
    path.write_text("decode_rope_reuse=True: startup only\n")
    with pytest.raises(AssertionError):
        validate_activation(config, tmp_path / "missing")
    path.write_text("batched RoPE reuse active batch=2\n")
    assert validate_activation(config, tmp_path / "actual")["status"] == "pass"
    config["service_activation_offset"] = path.stat().st_size
    with pytest.raises(AssertionError):
        validate_activation(config, tmp_path / "stale")


@pytest.mark.parametrize("corrupt", [None, "wrong-head", "missing-case"])
def test_real_provider_compares_batch_vectors_to_independent_oracle(tmp_path, monkeypatch, corrupt):
    import ast
    import hashlib
    import sys
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from moss_speech_bench.correctness import PROVIDERS

    tree = ast.parse((PROVIDERS / "ar.py").read_text())
    run = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    block = next(
        node
        for node in ast.walk(run)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "PROFILE['profile'] == 'ar-batched-rope'"
    )
    comparison = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "compare_logits"
    )
    names = [str(n) for n in range(8)]
    oracle = {
        f"{n}.{h}_last_logits": torch.arange(16).float() for n in names for h in ("text", "audio")
    }
    outputs = {
        str(b): {
            n: {
                h: oracle[f"{n}.{h}_last_logits"].to(torch.bfloat16).clone()
                for h in ("text", "audio")
            }
            for n in names
        }
        for b in (2, 4)
    }
    if corrupt == "wrong-head":
        outputs["4"]["7"]["audio"][0] = 100
    if corrupt == "missing-case":
        del outputs["2"]["7"]
    report = {
        "cases": {n: {"input_ids": [[1, 0], [2, 0]]} for n in names},
        "policy": {"absolute_tolerance": 0, "relative_tolerance": 0},
    }
    monkeypatch.setitem(
        sys.modules,
        "tests.unit_test.fixtures.moss_speech_model",
        SimpleNamespace(qualify_batched_rope=lambda model, grids: (proof(), outputs)),
    )
    scope = dict(
        torch=torch,
        PROFILE={"profile": "ar-batched-rope"},
        model=object(),
        device="cpu",
        report=report,
        expected_vectors=oracle,
        tensor_path=tmp_path / "serial.safetensors",
        save_file=save_file,
        digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
        write_json=lambda p, d: p.write_text(json.dumps(d)),
        report_path=tmp_path / "report.json",
        validate_batched_rope=validate_qualification,
    )
    exec(
        compile(ast.Module(body=[comparison], type_ignores=[]), "provider-comparison", "exec"),
        scope,
    )
    code = compile(ast.Module(body=[block], type_ignores=[]), "provider-batched-block", "exec")
    if corrupt:
        with pytest.raises(AssertionError):
            exec(code, scope)
    else:
        exec(code, scope)
        validate_qualification(report["batched_rope"], names)
        assert (tmp_path / "batched-rope-logits.safetensors").is_file()


def test_c04_dispatch_does_not_require_unrelated_combination_proof():
    import ast
    import inspect

    from moss_speech_bench import correctness

    tree = ast.parse(inspect.getsource(correctness.provider))
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "validate_qualification"
            for statement in node.body
            for call in ast.walk(statement)
        )
    ]
    assert len(nodes) == 1
    for profile, expected in (("ar-combination", True), ("ar-batched-rope", False)):
        expression = compile(ast.Expression(nodes[0].test), "qualification-dispatch", "eval")
        assert eval(expression, {"gate": "C04", "config": {"profile": profile}}) is expected


def test_provider_imports_as_package_and_isolated_script(monkeypatch):
    import importlib
    import runpy

    from moss_speech_bench.correctness import PROVIDERS

    module = importlib.import_module("moss_speech_bench.providers.ar")
    assert callable(module.validate_batched_rope)
    monkeypatch.syspath_prepend(str(PROVIDERS))
    script = runpy.run_path(str(PROVIDERS / "ar.py"))
    assert callable(script["validate_batched_rope"])
