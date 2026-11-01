"""Compose fixed correctness node IDs for explicitly qualified candidate profiles."""

import json
from pathlib import Path


def expected_tests(selection, gate, profile, context_remaining=False, test_suite="legacy"):
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
    if test_suite != "legacy":
        if test_suite not in selection["test_suites"]:
            raise ValueError("Unknown correctness test suite: " + test_suite)
        if profile != "context-remaining":
            raise ValueError("P4 test suites require the context-remaining profile")
        changes = selection["test_suites"][test_suite].get(gate, {})
        replacements = changes.get("replace", {})
        if not set(replacements) <= set(expected):
            raise ValueError("Test suite replaces an absent regression node")
        expected = [replacements.get(node, node) for node in expected] + changes.get("add", [])
    if len(expected) != len(set(expected)):
        raise ValueError("Duplicate fixed correctness node IDs")
    return expected


def selected_nodes(gate, profile, context_remaining=False, test_suite="legacy"):
    selection = json.loads(Path(__file__).with_name("selection.json").read_text())
    return expected_tests(selection, gate, profile, context_remaining, test_suite)
