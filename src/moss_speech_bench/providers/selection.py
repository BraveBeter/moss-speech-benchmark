"""Compose fixed correctness node IDs for explicitly qualified candidate profiles."""

import json
from pathlib import Path


def expected_tests(
    selection,
    gate,
    profile,
    context_remaining=False,
    test_suite="legacy",
    optimizations=(),
    cumulative_api=None,
):
    if profile == "cumulative":
        if not context_remaining or test_suite != "eager-bugs":
            raise ValueError("Cumulative selection requires the new eager contract")
        if __package__:
            from .cumulative import regression_nodes
        else:
            from cumulative import regression_nodes
        return regression_nodes(selection, gate, optimizations, expected_tests, cumulative_api)
    if profile == "ar-combination":
        strategies = selection["ar-combination-strategies"]
        if (
            not optimizations
            or len(optimizations) != len(set(optimizations))
            or not set(optimizations) <= set(strategies)
        ):
            raise ValueError("Missing, unknown or duplicate AR combination strategies")
        expected = expected_tests(selection, gate, "hf-reference", context_remaining, test_suite)
        replacements = {}
        for feature in strategies:
            for old, new in selection.get(gate + "-" + feature + "-replacements", {}).items():
                if old in replacements and replacements[old] != new:
                    override = selection.get(gate + "-ar-combination-replacements", {}).get(old)
                    if override is None:
                        raise ValueError("Conflicting strategy replacement: " + old)
                    replacements[old] = override
                else:
                    replacements[old] = new
        replacements.update(selection.get(gate + "-ar-combination-replacements", {}))
        for old, new in replacements.items():
            if old not in expected:
                raise ValueError("Combination replaces an absent fixed node")
            index = expected.index(old)
            expected[index : index + 1] = new if isinstance(new, list) else [new]
        for feature in optimizations:
            base = set(selection[gate])
            for node in selection.get(gate + "-" + feature, []):
                mapped = selection.get(gate + "-ar-combination-node-replacements", {}).get(
                    node, node
                )
                for actual in mapped if isinstance(mapped, list) else [mapped]:
                    if actual not in base and actual not in expected:
                        expected.append(actual)
        expected += selection.get(gate + "-ar-combination", [])
        if len(expected) != len(set(expected)):
            raise ValueError("Duplicate combination correctness node IDs")
        return expected
    expected = list(selection[gate])
    if (context_remaining or profile.startswith("context-")) and gate == "C01":
        expected = [selection["context-window-replacements"].get(n, n) for n in expected]
        expected += selection["C01-context-window"] + selection["C01-context-remaining"]
    replacements = selection.get(gate + "-" + profile + "-replacements", {})
    if not set(replacements) <= set(expected):
        raise ValueError("Strategy replaces an absent fixed node")
    expected = [
        replacement
        for node in expected
        for replacement in (
            replacements.get(node, [node])
            if isinstance(replacements.get(node, [node]), list)
            else [replacements[node]]
        )
    ]
    if profile == "context-prefill-graph" and gate == "C03":
        expected = list(selection["C03-context-prefill-graph"])
    elif profile != "context-remaining":
        key = gate + "-" + profile
        if (
            profile
            in (
                "ar-batched-rope",
                "ar-serial-decode-graph",
                "codec-bias-broadcast",
                "ar-prefix-cache",
                "ar-sampling-history",
                "ar-host-metadata",
                "direct-lt",
                "codec-cfm-graph",
                "codec-fairness",
                "codec-score-release",
            )
            and key not in selection
        ):
            raise ValueError("Missing fixed strategy qualification: " + key)
        expected += selection.get(key, [])
    if test_suite != "legacy":
        if test_suite not in selection["test_suites"]:
            raise ValueError("Unknown correctness test suite: " + test_suite)
        if test_suite == "eager-bugs" and not (context_remaining or profile == "context-remaining"):
            raise ValueError("Eager feedback suite requires the remaining-context contract")
        if test_suite != "eager-bugs" and profile != "context-remaining":
            raise ValueError("P4 test suites require the context-remaining profile")
        changes = selection["test_suites"][test_suite].get(gate, {})
        replacements = changes.get("replace", {})
        if not set(replacements) <= set(expected):
            raise ValueError("Test suite replaces an absent regression node")
        expected = [replacements.get(node, node) for node in expected] + changes.get("add", [])
    if len(expected) != len(set(expected)):
        raise ValueError("Duplicate fixed correctness node IDs")
    return expected


def selected_nodes(
    gate,
    profile,
    context_remaining=False,
    test_suite="legacy",
    optimizations=(),
    cumulative_api=None,
):
    selection = json.loads(Path(__file__).with_name("selection.json").read_text())
    return expected_tests(
        selection, gate, profile, context_remaining, test_suite, optimizations, cumulative_api
    )
