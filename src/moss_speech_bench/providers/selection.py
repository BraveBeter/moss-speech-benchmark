"""Compose fixed correctness node IDs for explicitly qualified candidate profiles."""


def expected_tests(selection, gate, profile, context_remaining=False):
    expected = list(selection[gate])
    if context_remaining and gate == "C01":
        expected = [selection["context-window-replacements"].get(n, n) for n in expected]
        expected += selection["C01-context-window"] + selection["C01-context-remaining"]
    if profile == "mixed-modalities" and gate == "C01":
        replacements = selection["C01-mixed-modalities-replacements"]
        expected = [replacement for n in expected for replacement in replacements.get(n, [n])]
    expected += selection.get(gate + "-" + profile, [])
    if len(expected) != len(set(expected)):
        raise ValueError("Duplicate fixed correctness node IDs")
    return expected
