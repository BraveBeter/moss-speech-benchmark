"""Fixed model test selection shared by the client and isolated provider."""

import json
from pathlib import Path


def selected_nodes(gate, profile):
    selection = json.loads(Path(__file__).with_name("selection.json").read_text())
    expected = list(selection[gate])
    if profile.startswith("context-") and gate == "C01":
        expected = [selection["context-window-replacements"].get(n, n) for n in expected]
        expected += selection["C01-context-window"] + selection["C01-context-remaining"]
    if profile == "context-prefill-graph" and gate == "C03":
        return list(selection["C03-context-prefill-graph"])
    if profile == "native-norm" or (
        profile in ("context-small-first", "context-startup-schedule") and gate == "C01"
    ):
        expected += selection.get(gate + "-" + profile, [])
    return expected
