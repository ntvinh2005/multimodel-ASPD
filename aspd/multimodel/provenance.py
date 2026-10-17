"""Small reproducibility helpers shared by training and experiment evaluations."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", *args], cwd=repo, stderr=subprocess.DEVNULL
    )


def source_state(repo: str | Path = ".") -> dict[str, Any]:
    """Fingerprint committed, modified, and untracked source used by an expensive run."""

    repo = Path(repo).resolve()
    try:
        root = Path(_git(repo, "rev-parse", "--show-toplevel").decode().strip())
        commit = _git(root, "rev-parse", "HEAD").decode().strip()
        status = _git(root, "status", "--porcelain", "--untracked-files=all").decode().splitlines()
        digest = hashlib.sha256()
        digest.update(f"commit\0{commit}\0".encode())
        digest.update(_git(root, "diff", "--binary", "HEAD", "--"))
        untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        for raw_path in sorted(path for path in untracked if path):
            path = root / raw_path.decode()
            digest.update(b"untracked\0" + raw_path + b"\0")
            digest.update(path.read_bytes())
        return {
            "git_commit": commit,
            "git_dirty": bool(status),
            "git_status": status,
            "source_state_sha256": digest.hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        return {
            "git_commit": None,
            "git_dirty": None,
            "git_status": None,
            "source_state_sha256": None,
        }
