"""Compose explicit fixed allowlists for the declared candidate path."""


def expected_tests(selection, gate, profile, context_remaining=False):
    expected = list(selection[gate])
    if context_remaining and gate == "C01":
        expected = [selection["context-window-replacements"].get(n, n) for n in expected]
        expected += selection["C01-context-window"] + selection["C01-context-remaining"]
    expected += selection.get(gate + "-" + profile, [])
    if len(expected) != len(set(expected)):
        raise ValueError("Duplicate fixed correctness node IDs")
    return expected
