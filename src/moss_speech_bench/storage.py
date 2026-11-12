"""Small, deterministic artifact and identity helpers."""

import hashlib
import json
import subprocess
from pathlib import Path


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def source_identity():
    package = Path(__file__).parent
    hashes = {
        str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*.py"))
    }
    identity = {"source_hash": digest(hashes)}
    try:
        for key, command in {
            "commit": ["rev-parse", "HEAD"],
            "branch": ["branch", "--show-current"],
            "dirty": ["status", "--porcelain"],
        }.items():
            identity[key] = subprocess.check_output(
                ["git", "-C", str(package), *command], text=True, stderr=subprocess.DEVNULL
            ).strip()
    except (OSError, subprocess.CalledProcessError):
        identity["commit"] = None
    return identity
