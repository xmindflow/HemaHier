"""Load configs/config.yaml and resolve experiment entries."""

from __future__ import annotations

import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
# Names kept so cv_splits.py / older imports still resolve.
DEFAULT_DATA_CONFIG_PATH = DEFAULT_CONFIG_PATH
DEFAULT_PROJECT_CONFIG_PATH = DEFAULT_CONFIG_PATH


def _deep_merge(base: dict, overlay: Mapping[str, Any]) -> dict:
    out = deepcopy(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = deepcopy(val)
    return out


def _load_file(path: Path) -> dict:
    if not path.is_file():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML required; pip install pyyaml")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _off_chain_from_unhealthy(unhealthy: Mapping[str, Any]) -> tuple[list[str], dict[str, str]]:
    identities: list[str] = []
    lineage_map: dict[str, str] = {}
    for group in ("dysplastic", "pathologic"):
        for name, meta in (unhealthy.get(group) or {}).items():
            if not isinstance(meta, dict):
                continue
            identities.append(name)
            if meta.get("lineage"):
                lineage_map[name] = str(meta["lineage"])
    return identities, lineage_map


def _normalize_data_raw(raw: Mapping[str, Any]) -> dict:
    """Pack taxonomy keys into the `ontology` namespace the rest of the code reads."""
    if not raw:
        return {}
    unhealthy = raw.get("unhealthy") or {}
    artifacts = raw.get("artifacts") or {}
    harmonization = raw.get("harmonization") or {}
    off_ids, off_lin = _off_chain_from_unhealthy(unhealthy)
    out: dict[str, Any] = {"paths": dict(raw.get("paths") or {})}
    out["ontology"] = {
        "canonical_aliases": dict(harmonization.get("canonical_aliases") or {}),
        "training_excluded_canonical": list(artifacts.get("training_excluded_canonical") or []),
        "artifact_canonical": list(artifacts.get("artifact_canonical") or []),
        "progenitor": dict(raw.get("progenitor") or {}),
        "unhealthy": deepcopy(dict(unhealthy)),
        "eval_distance": dict(raw.get("eval_distance") or {}),
        "branches": deepcopy(dict(raw.get("branches") or {})),
        "off_chain_identities": off_ids,
        "off_chain_lineage": off_lin,
    }
    datasets: dict[str, Any] = {}
    for name, ds in (raw.get("datasets") or {}).items():
        if not isinstance(ds, dict):
            continue
        flat = dict(ds)
        cv = flat.pop("cv", None) or {}
        if isinstance(cv, dict):
            flat.update(cv)
        datasets[name] = flat
    out["datasets"] = datasets
    return out


@lru_cache(maxsize=4)
def load_config(path: Optional[str] = None) -> dict:
    cfg_path = Path(path or os.environ.get("HEMAHIER_CONFIG", DEFAULT_CONFIG_PATH))
    raw = _load_file(cfg_path)
    cfg = _normalize_data_raw(raw)
    for key in ("model", "training", "evaluation", "experiments", "paths"):
        if key in raw:
            cfg = _deep_merge(cfg, {key: raw[key]})
    return cfg


def ns(namespace: str, *, path: Optional[str] = None) -> dict:
    cur: Any = load_config(path)
    for part in namespace.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return {}
        cur = cur[part]
    return dict(cur) if isinstance(cur, dict) else {}


def cfg_paths() -> dict:
    return ns("paths")


def cfg_ontology() -> dict:
    return ns("ontology")


def cfg_model() -> dict:
    return ns("model")


def cfg_training() -> dict:
    return ns("training")


def dataset_config(name: str, *, path: Optional[str] = None) -> dict:
    return dict(ns("datasets", path=path).get(name, {}))


def experiment_configs(*, path: Optional[str] = None) -> Dict[str, dict]:
    raw = ns("experiments", path=path)
    base = {k: dict(v) for k, v in raw.items() if isinstance(v, dict)}
    resolved: Dict[str, dict] = {}

    def resolve(name: str, seen: set[str]) -> dict:
        if name in resolved:
            return resolved[name]
        if name not in base:
            raise KeyError(f"Unknown experiment: {name}")
        if name in seen:
            raise ValueError(f"Circular same_as involving {name!r}")
        cfg = dict(base[name])
        parent = cfg.pop("same_as", None)
        if parent:
            seen.add(name)
            merged = dict(resolve(str(parent), seen))
            merged.update(cfg)
            cfg = merged
        resolved[name] = cfg
        return cfg

    for name in base:
        resolve(name, set())
    return resolved


EXP_CONFIGS: Dict[str, dict] = experiment_configs()
