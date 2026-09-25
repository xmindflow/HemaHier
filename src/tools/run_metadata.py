"""Attach git hash, command line, and dataset metadata to run outputs."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def git_commit_hash(repo_root: Optional[Path] = None) -> str:
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_run_metadata(
    exp_id: str,
    cfg: dict,
    args_namespace: Any,
    dataset: Optional[str] = None,
    fold: Optional[int] = None,
    split_type: str = "cell-level stratified CV",
) -> Dict[str, Any]:
    cmd = " ".join(sys.argv)
    ns = vars(args_namespace) if hasattr(args_namespace, "__dict__") else dict(args_namespace)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "command": cmd,
        "exp_id": exp_id,
        "dataset": dataset or ns.get("single_dataset"),
        "fold": fold if fold is not None else ns.get("fold"),
        "split_type": split_type,
        "backbone": ns.get("backbone"),
        "config": cfg,
        "args": ns,
    }


def write_run_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))
