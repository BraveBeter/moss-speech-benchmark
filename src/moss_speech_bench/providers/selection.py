"""Compose explicit fixed allowlists for the declared candidate path."""

import json
from pathlib import Path


def expected_tests(selection, gate, profile, context_remaining=False):
    expected = list(selection[gate])
    context_remaining = context_remaining or profile.startswith("context-")
    if context_remaining and gate == "C01":
        expected = [selection["context-window-replacements"].get(n, n) for n in expected]
        expected += selection["C01-context-window"] + selection["C01-context-remaining"]
    if profile == "context-prefill-graph" and gate == "C03":
        return list(selection["C03-context-prefill-graph"])
    replacements = selection.get(gate + "-" + profile + "-replacements", {})
    expected = [replacements.get(node, node) for node in expected]
    if profile not in ("context-remaining", "context-window"):
        expected += selection.get(gate + "-" + profile, [])
    if len(expected) != len(set(expected)):
        raise ValueError("Duplicate fixed correctness node IDs")
    return expected


def selected_nodes(gate, profile, context_remaining=False):
    selection = json.loads(Path(__file__).with_name("selection.json").read_text())
    return expected_tests(selection, gate, profile, context_remaining)
