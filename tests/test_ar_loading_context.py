"""The real provider loading block must retain versioned Graph parameters."""

import ast

import pytest

from moss_speech_bench.correctness import PROVIDERS


@pytest.mark.parametrize("profile", ["ar-combination", "ar-projection-graph", "hf-reference"])
def test_model_loading_keeps_combo_weights_versioned_and_restores_inference(profile):
    torch = pytest.importorskip("torch")
    source = ast.parse((PROVIDERS / "ar.py").read_text())
    run = next(
        node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    contexts = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "inference_mode"
            and item.context_expr.args
            for item in node.items
        )
    ]
    assert len(contexts) == 1, "A distinct parameter-loading context is required"

    class TinyModel(torch.nn.Module):
        total_attention_layers = 40

        def __init__(self, configuration, init_device):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.zeros(2, 2, device=init_device, dtype=torch.bfloat16)
            )

        def attention_layer_ids(self):
            return list(range(40))

        def load_weights(self, weights):
            self.weight.copy_(torch.ones_like(self.weight))

        def assert_heads_independent(self):
            pass

    scope = dict(
        torch=torch,
        PROFILE={"profile": profile},
        MossSpeechSGLangModel=TinyModel,
        configuration=None,
        device=torch.device("cpu"),
        model_path=None,
        stream_weights=lambda *args: (),
        report={},
    )
    loading = compile(
        ast.Module(body=contexts, type_ignores=[]), "<provider-loading-context>", "exec"
    )
    with torch.inference_mode():
        exec(loading, scope)
        parameter = scope["model"].weight
        assert torch.is_inference_mode_enabled()
        assert parameter.is_inference() is (profile != "ar-combination")
        if profile == "ar-combination":
            before = parameter._version
            parameter.add_(1)
            assert parameter._version == before + 1
        else:
            with pytest.raises(RuntimeError, match="version"):
                _ = parameter._version
