"""Candidate module-origin audit, including namespace packages."""

import sys
import types
from pathlib import Path


def collect_origins(core, modules=None):
    core = Path(core).resolve()
    found = {}
    for name, module in tuple((sys.modules if modules is None else modules).items()):
        if not (
            name == "sglang_omni"
            or name.startswith(("sglang_omni.", "tests.unit_test.fixtures.moss_speech"))
        ):
            continue
        if not isinstance(module, types.ModuleType):
            raise RuntimeError(f"Non-module candidate import: {name}")
        data = vars(module)
        paths = [data["__file__"]] if data.get("__file__") else list(data.get("__path__", []))
        assert paths, f"Candidate module has no source origin: {name}"
        resolved = [Path(p).resolve() for p in paths]
        assert all(core in p.parents for p in resolved), (name, paths, core)
        found[name] = [str(p) for p in resolved]
    assert "sglang_omni" in found, "Candidate package was not imported"
    return found
