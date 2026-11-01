"""Compose fixed correctness node IDs for explicitly qualified candidate profiles."""

import json
from pathlib import Path


def expected_tests(selection, gate, profile, context_remaining=False):
    expected = list(selection[gate])
    if (context_remaining or profile.startswith("context-")) and gate == "C01":
        expected = [selection["context-window-replacements"].get(n, n) for n in expected]
        expected += selection["C01-context-window"] + selection["C01-context-remaining"]
    if profile == "mixed-modalities" and gate == "C01":
        replacements = selection["C01-mixed-modalities-replacements"]
        expected = [replacement for n in expected for replacement in replacements.get(n, [n])]
    if profile == "context-prefill-graph" and gate == "C03":
        expected = list(selection["C03-context-prefill-graph"])
    elif profile != "context-remaining":
        expected += selection.get(gate + "-" + profile, [])
    if len(expected) != len(set(expected)):
        raise ValueError("Duplicate fixed correctness node IDs")
    return expected


def selected_nodes(gate, profile, context_remaining=False):
    selection = json.loads(Path(__file__).with_name("selection.json").read_text())
    return expected_tests(selection, gate, profile, context_remaining)
