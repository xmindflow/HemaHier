"""
Bone-marrow datasets: registry, sample scanning, PyTorch loaders.

- Registry: source folder → canonical class (labels_v2.json)
- Sample scan: used when building CV splits (collect_dataset_samples)
- HierHematologyDataset: training / validation image loading with augmentation
"""

from __future__ import annotations

# --- registry --------------------------------------------------------------


import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from data.ontology import artifact_canonical, canonical_aliases, training_excluded_canonical

TRAINING_EXCLUDED_CANONICAL = training_excluded_canonical()
ARTIFACT_CANONICAL = artifact_canonical()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_JSON_PATH = REPO_ROOT / "configs" / "labels_v2.json"

# labels_v2.json dataset key -> dataset name used by the loader.
_LABELS_DATASET_KEYS = {"mll_v1": "mllv1"}

CANONICAL_ALIASES: Dict[str, str] = canonical_aliases()


def _labels_path() -> Path:
    try:
        from data.ontology import is_v2_labels, labels_v2_path
        if is_v2_labels():
            return labels_v2_path()
    except Exception:
        pass
    try:
        from config.settings import cfg_paths
        rel = cfg_paths().get("labels_file", "labels.json")
        p = REPO_ROOT / str(rel)
        if p.is_file():
            return p
    except Exception:
        pass
    return LABELS_JSON_PATH


def _use_labels_v2() -> bool:
    try:
        from data.ontology import is_v2_labels
        return is_v2_labels()
    except Exception:
        return False


def _canonical_classes():
    from data.ontology import CANONICAL_CLASSES
    return CANONICAL_CLASSES


@dataclass
class DatasetInfo:
    name: str
    source: str
    path_env_var: str
    role: str
    notes: str = ""


DATASETS: Dict[str, DatasetInfo] = {
    "mllv1": DatasetInfo(
        name="mllv1", source="bone_marrow",
        path_env_var="MLLV1_ROOT", role="train",
        notes="MLL Helmholtz / Fraunhofer bone marrow v1.",
    ),
}


def resolve_canonical(
    name: Optional[str],
    *,
    include_artifacts: bool = False,
) -> Optional[str]:
    if name is None:
        return None
    if name in TRAINING_EXCLUDED_CANONICAL:
        return None
    classes = _canonical_classes()
    seen = set()
    cur = name
    while cur in CANONICAL_ALIASES:
        if cur in seen:
            return None
        seen.add(cur)
        cur = CANONICAL_ALIASES[cur]
    if cur in classes:
        return cur
    if include_artifacts and cur in ARTIFACT_CANONICAL:
        return cur
    return None


def output_class_order(
    kept: set[str],
    *,
    include_artifacts: bool = False,
) -> List[str]:
    """Stable label order for CV manifests and flat classifiers."""
    order = [c for c in _canonical_classes() if c in kept]
    if include_artifacts:
        order.extend(sorted(kept - set(order)))
    return order


def build_mapping(
    dataset_name: str,
    *,
    include_artifacts: bool = False,
) -> Dict[str, Optional[str]]:
    if _use_labels_v2():
        from data.ontology import build_mapping as v2_build_mapping
        return v2_build_mapping(dataset_name, include_artifacts=include_artifacts)

    raise FileNotFoundError("labels_v2.json required; set paths.labels_file in configs/config.yaml")


def find_label_dir(root: Path, src_label: str) -> Optional[Path]:
    if not root.is_dir():
        return None
    direct = root / src_label
    if direct.is_dir():
        return direct
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == src_label.lower():
            return child
    return None


def is_image_path(p: Path) -> bool:
    """True for real image files; skips macOS/Windows metadata junk."""
    if not p.is_file():
        return False
    if p.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    name = p.name
    if name.startswith(".") or name.startswith("._"):
        return False
    low = name.lower()
    if low in ("thumbs.db", "desktop.ini"):
        return False
    return True


def iter_label_images(label_dir: Path, recursive: bool = True) -> Iterator[Path]:
    paths = label_dir.rglob("*") if recursive else label_dir.iterdir()
    for p in paths:
        if is_image_path(p):
            yield p


def resolve_root(dataset_name: str, override: Optional[str] = None) -> str:
    if override:
        return override
    env = DATASETS[dataset_name].path_env_var
    root = os.environ.get(env)
    if not root:
        raise ValueError(f"{env} is not set and no root override provided.")
    return root


def scan_label_counts(
    dataset_name: str,
    root: Optional[str] = None,
    exclude_artifacts: bool = True,
) -> Dict:
    """
    Count images on disk per source label and per resolved canonical class.
    """
    root_path = Path(resolve_root(dataset_name, root))
    mapping = build_mapping(dataset_name, include_artifacts=not exclude_artifacts)

    source_counts: Dict[str, int] = {}
    canonical_counts: Dict[str, int] = {}
    skipped: Dict[str, str] = {}

    for src_label, canonical in mapping.items():
        label_dir = find_label_dir(root_path, src_label)
        if label_dir is None:
            skipped[src_label] = "directory not found"
            source_counts[src_label] = 0
            continue

        n = sum(1 for _ in iter_label_images(label_dir))
        source_counts[src_label] = n

        if canonical is None:
            skipped[src_label] = "excluded mapping"
            continue
        if exclude_artifacts and canonical in {
            "unlabeled", "cell_artifact", "smudge_cell", "nuclear_debris",
            "apoptotic_cell", "mitotic_cell",
        }:
            skipped[src_label] = f"artifact ({canonical})"
            continue

        canonical_counts[canonical] = canonical_counts.get(canonical, 0) + n

    return {
        "dataset": dataset_name,
        "root": str(root_path),
        "source_counts": source_counts,
        "canonical_counts": canonical_counts,
        "skipped": skipped,
        "total_images": sum(source_counts.values()),
        "trainable_images": sum(canonical_counts.values()),
    }


def format_class_count_report(
    counts: Dict,
    min_samples_per_class: int = 0,
) -> str:
    """Human-readable per-class count table."""
    lines = [
        f"Dataset: {counts['dataset']}  ({counts['root']})",
        f"Total images on disk: {counts['total_images']}",
        f"Trainable (mapped canonical): {counts['trainable_images']}",
        "",
        "Per source label:",
    ]
    for src, n in sorted(counts["source_counts"].items(), key=lambda x: (-x[1], x[0])):
        reason = counts["skipped"].get(src)
        suffix = f"  [{reason}]" if reason else ""
        lines.append(f"  {n:5d}  {src}{suffix}")

    lines.append("")
    lines.append("Per canonical class (merged):")
    for canon, n in sorted(counts["canonical_counts"].items(), key=lambda x: (-x[1], x[0])):
        flag = ""
        if min_samples_per_class > 0 and n < min_samples_per_class:
            flag = f"  ** below min ({min_samples_per_class}), excluded **"
        lines.append(f"  {n:5d}  {canon}{flag}")

    if min_samples_per_class > 0:
        thin = [
            c for c, n in counts["canonical_counts"].items()
            if n < min_samples_per_class
        ]
        if thin:
            lines.append("")
            lines.append(
                f"Classes excluded by min_samples={min_samples_per_class}: "
                + ", ".join(sorted(thin))
            )
    return "\n".join(lines)

# --- sample manifest (CV split generation) ---------------------------------


import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


# labels_v2.json dataset key -> dataset name used by the loader.
_LABELS_DATASET_KEYS = {"mll_v1": "mllv1"}

BIOLOGIC_FILTERS = ("hematopoietic", "healthy_only", "all_mapped")


@dataclass
class SampleRecord:
    path: str              # absolute path on disk
    rel_path: str          # relative to dataset root
    source_label: str
    canonical: str
    biologic_status: str
    fine_idx: int

    def to_dict(self) -> dict:
        return asdict(self)


def _labels_key(dataset_name: str) -> str:
    return next(
        (k for k, v in _LABELS_DATASET_KEYS.items() if v == dataset_name),
        dataset_name,
    )


def build_source_metadata(dataset_name: str) -> Dict[str, dict]:
    """source_label -> labels row fields for one dataset."""
    key = _labels_key(dataset_name)
    rows = json.loads(_labels_path().read_text())
    out: Dict[str, dict] = {}
    for row in rows:
        if row.get("dataset") != key:
            continue
        out[row["source_label"]] = row
    return out


def biologic_status_allowed(
    status: str,
    mode: str,
    *,
    canonical: str = "",
    health_category: str = "",
) -> bool:
    """
    hematopoietic  : healthy + dysplastic/pathologic hematopoietic cells (default for CV)
    healthy_only   : healthy cells only
    all_mapped     : any mapped canonical (still drops artifacts separately)
    """
    healthy = status in (
        "healthy_or_non_neoplastic", "healthy_normal",
    )
    abnormal = status in (
        "non_healthy_or_pathologic", "dysplastic", "pathologic",
    )
    # Progenitor root: "unspecified" is a naming convention, not unknown biology.
    if canonical == "blast_unspecified" and health_category == "healthy":
        healthy = True
    if mode == "hematopoietic":
        return healthy or abnormal
    if mode == "healthy_only":
        return healthy
    if mode == "all_mapped":
        return True
    raise ValueError(f"Unknown biologic_filter: {mode!r}")


def collect_dataset_samples(
    dataset_name: str,
    root: Optional[str] = None,
    *,
    min_samples_per_class: int = 5,
    biologic_filter: str = "hematopoietic",
    exclude_artifacts: bool = True,
    use_source_labels: bool = False,
) -> tuple[List[SampleRecord], dict]:
    """
    Scan the corpus and return trainable samples plus a summary dict.

    Steps:
      1. Map source folders via labels.json
      2. Apply biologic_filter (healthy / pathologic / …)
      3. Drop artifact & stromal exclusions
      4. Drop canonical classes with < min_samples_per_class images
    """
    from data.ontology import FINE_TO_IDX

    root_path = Path(resolve_root(dataset_name, root)).resolve()
    include_artifacts = not exclude_artifacts
    mapping = build_mapping(dataset_name, include_artifacts=include_artifacts)
    meta = build_source_metadata(dataset_name)

    raw: List[SampleRecord] = []
    skipped: Dict[str, str] = {}

    def _append_sample(src_label: str, class_key: str, bio: str, p: Path) -> None:
        p = p.resolve()
        raw.append(SampleRecord(
            path=str(p),
            rel_path=str(p.relative_to(root_path)),
            source_label=src_label,
            canonical=class_key,
            biologic_status=bio,
            fine_idx=-1,
        ))

    if use_source_labels:
        for src_label, row in sorted(meta.items()):
            bio = row.get("biologic_status", "uncertain")
            hc = row.get("health_category", "")
            cid = row.get("canonical_id")
            if exclude_artifacts and (
                cid in ARTIFACT_CANONICAL
                or row.get("is_artifact", False)
                or row.get("annotation_mode") == "artifact_or_background"
            ):
                skipped[src_label] = "artifact"
                continue
            if not biologic_status_allowed(
                bio, biologic_filter,
                canonical=str(cid or ""),
                health_category=str(hc),
            ):
                skipped[src_label] = f"biologic_filter ({bio})"
                continue
            label_dir = find_label_dir(root_path, src_label)
            if label_dir is None:
                skipped[src_label] = "directory not found"
                continue
            for p in iter_label_images(label_dir, recursive=True):
                _append_sample(src_label, src_label, bio, p)
    else:
        for src_label, canonical in mapping.items():
            row = meta.get(src_label, {})
            bio = row.get("biologic_status", "uncertain")
            hc = row.get("health_category", "")

            if canonical is None:
                skipped[src_label] = "excluded mapping"
                continue
            canonical = resolve_canonical(canonical, include_artifacts=include_artifacts)
            if canonical is None:
                skipped[src_label] = "unresolved canonical"
                continue
            if canonical in TRAINING_EXCLUDED_CANONICAL:
                skipped[src_label] = f"excluded canonical ({canonical})"
                continue
            if exclude_artifacts and (
                canonical in ARTIFACT_CANONICAL
                or row.get("is_artifact", False)
                or row.get("annotation_mode") == "artifact_or_background"
            ):
                skipped[src_label] = f"artifact ({canonical})"
                continue
            if not biologic_status_allowed(
                bio,
                biologic_filter,
                canonical=canonical or "",
                health_category=str(hc),
            ):
                skipped[src_label] = f"biologic_filter ({bio})"
                continue

            label_dir = find_label_dir(root_path, src_label)
            if label_dir is None:
                skipped[src_label] = "directory not found"
                continue

            for p in iter_label_images(label_dir, recursive=True):
                p = p.resolve()
                raw.append(SampleRecord(
                    path=str(p),
                    rel_path=str(p.relative_to(root_path)),
                    source_label=src_label,
                    canonical=canonical,
                    biologic_status=bio,
                    fine_idx=-1,
                ))

    canon_counts = Counter(r.canonical for r in raw)
    if min_samples_per_class > 0:
        allowed = {c for c, n in canon_counts.items() if n >= min_samples_per_class}
        dropped = {c: n for c, n in canon_counts.items() if n < min_samples_per_class}
        samples = [r for r in raw if r.canonical in allowed]
    else:
        dropped = {}
        samples = raw

    if samples and not use_source_labels:
        class_order = output_class_order(
            {r.canonical for r in samples},
            include_artifacts=include_artifacts,
        )
        local_idx = {c: i for i, c in enumerate(class_order)}
        samples = [
            SampleRecord(
                path=r.path,
                rel_path=r.rel_path,
                source_label=r.source_label,
                canonical=r.canonical,
                biologic_status=r.biologic_status,
                fine_idx=local_idx[r.canonical],
            )
            for r in samples
        ]
    elif samples and use_source_labels:
        class_order = sorted({r.canonical for r in samples})
        local_idx = {c: i for i, c in enumerate(class_order)}
        samples = [
            SampleRecord(
                path=r.path,
                rel_path=r.rel_path,
                source_label=r.source_label,
                canonical=r.canonical,
                biologic_status=r.biologic_status,
                fine_idx=local_idx[r.canonical],
            )
            for r in samples
        ]

    summary = {
        "dataset": dataset_name,
        "root": str(root_path),
        "biologic_filter": biologic_filter,
        "min_samples_per_class": min_samples_per_class,
        "n_raw": len(raw),
        "n_kept": len(samples),
        "n_classes": len({r.canonical for r in samples}),
        "use_source_labels": use_source_labels,
        "class_order": "source_labels" if use_source_labels else "CANONICAL_CLASSES",
        "canonical_counts": dict(sorted(canon_counts.items())),
        "dropped_classes": dropped,
        "skipped_source_labels": skipped,
        "biologic_counts": dict(Counter(r.biologic_status for r in samples)),
    }
    return samples, summary

# --- PyTorch dataset + dataloaders -----------------------------------------


import os
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from data.ontology import (
    CANONICAL_CLASSES, FINE_TO_IDX, FINE_IDX_TO_COARSE_IDX, LINEAGES, MATURATION_TABLE,
)
from tools.reproducibility import dataloader_generators


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(360),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_train_augment_transform() -> transforms.Compose:
    """Augment + tensorize after images are already resized (uint8 preload path)."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(360),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_tensor_transform() -> transforms.Compose:
    """Tensorize + normalize after images are already resized (uint8 preload path)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Retry limit when a path is corrupt (should be rare after is_image_path filtering).
_MAX_LOAD_RETRIES = 8


def _load_rgb_with_opencv(path: str) -> Image.Image:
    import cv2

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"OpenCV could not decode {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_rgb_image(path: str) -> Image.Image:
    """Open an image as RGB; fall back to OpenCV when PIL cannot decode."""
    try:
        with Image.open(path) as img:
            img.load()
            return img.convert("RGB")
    except (OSError, Image.UnidentifiedImageError):
        return _load_rgb_with_opencv(path)


def _path_readable(path: str) -> bool:
    try:
        load_rgb_image(path)
        return True
    except Exception:
        return False


def drop_unreadable_samples(
    samples: List[SampleRecord],
    *,
    max_workers: int = 8,
) -> tuple[List[SampleRecord], List[str]]:
    """Remove corrupt images so training does not retry them every epoch."""
    if not samples:
        return [], []

    paths = [rec.path for rec in samples]
    readable: List[bool]
    if len(samples) >= 2000 and max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            readable = list(pool.map(_path_readable, paths))
    else:
        readable = [_path_readable(p) for p in paths]

    kept = [rec for rec, ok in zip(samples, readable) if ok]
    dropped = [p for p, ok in zip(paths, readable) if not ok]
    return kept, dropped


class HierHematologyDataset(Dataset):
    """
    Loads one dataset and returns an image + multi-label tuple.

    Splits 'all' / 'train' / 'val' use a deterministic per-dataset shuffle,
    unless ``fold`` is set (loads a precomputed CV split from splits/<dataset>/).
    """

    def __init__(
        self,
        dataset_name: str,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        split: str = "all",
        val_fraction: float = 0.15,
        seed: int = 42,
        exclude_artifacts: bool = True,
        dataset_id: int = 0,
        allowed_classes: Optional[List[str]] = None,
        data_fraction: float = 1.0,
        max_samples: Optional[int] = None,
        preload_images: bool = False,
        img_size: int = 224,
        min_samples_per_class: int = 5,
        print_class_counts: bool = False,
        fold: Optional[int] = None,
        splits_dir: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.info = DATASETS[dataset_name]
        self.transform = transform
        self.split = split
        self.dataset_id = dataset_id
        self.fold = fold
        self._warned_bad_paths: set[str] = set()
        self.img_size = img_size
        self._preloaded_uint8: Optional[np.ndarray] = None

        root_path = Path(resolve_root(dataset_name, root))
        self.root = root_path

        if fold is not None:
            if split not in ("train", "val"):
                raise ValueError("CV mode requires split='train' or split='val'")
            from data.cv_splits import load_fold_split
            fold_doc = load_fold_split(
                dataset_name, fold,
                splits_root=Path(splits_dir) if splits_dir else None,
            )
            self.num_fine_classes = int(
                fold_doc.get("n_output_classes", len(fold_doc.get("classes", [])))
            )
            part = fold_doc["train"] if split == "train" else fold_doc["val"]
            # Split manifests retain the original absolute path for provenance,
            # but experiments must remain relocatable across machines.  Prefer
            # the path relative to the explicitly resolved dataset root and
            # fall back to the legacy absolute path for old manifests.
            self.samples = []
            for item in part["items"]:
                rel_path = item.get("rel_path")
                relocated = root_path / rel_path if rel_path else None
                path = relocated if relocated is not None and relocated.exists() else Path(item["path"])
                self.samples.append((
                    str(path),
                    item["canonical"],
                    int(item["fine_idx"]),
                ))
        else:
            self.num_fine_classes = len(CANONICAL_CLASSES)
            mapping = build_mapping(dataset_name, include_artifacts=not exclude_artifacts)

            samples: List[Tuple[str, str]] = []
            for src_label, canonical in mapping.items():
                if canonical is None:
                    continue
                if exclude_artifacts and canonical in ARTIFACT_CANONICAL:
                    continue
                if allowed_classes is not None and canonical not in allowed_classes:
                    continue
                label_dir = find_label_dir(root_path, src_label)
                if label_dir is None:
                    continue
                for p in iter_label_images(label_dir, recursive=True):
                    samples.append((str(p), canonical))

            # Drop canonical classes with too few images (unstable splits / CV).
            if min_samples_per_class > 0 and samples:
                canon_totals = Counter(c for _, c in samples)
                allowed = {
                    c for c, n in canon_totals.items()
                    if n >= min_samples_per_class
                }
                dropped = {
                    c: n for c, n in canon_totals.items()
                    if n < min_samples_per_class
                }
                if dropped:
                    parts = ", ".join(f"{c}({n})" for c, n in sorted(dropped.items()))
                    print(
                        f"[{dataset_name}] excluding {len(dropped)} class(es) with "
                        f"<{min_samples_per_class} samples: {parts}"
                    )
                samples = [(p, c) for p, c in samples if c in allowed]

            rng = random.Random(seed)
            rng.shuffle(samples)
            if split == "all":
                self.samples = samples
            else:
                n_val = max(1, int(len(samples) * val_fraction))
                self.samples = samples[:n_val] if split == "val" else samples[n_val:]

        if data_fraction < 1.0:
            n_keep = max(1, int(len(self.samples) * data_fraction))
            self.samples = self.samples[:n_keep]
        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        self._preloaded: Optional[List[torch.Tensor]] = None
        if preload_images:
            self._preload_uint8_images()
            self.transform = (
                get_train_augment_transform()
                if split == "train"
                else get_eval_tensor_transform()
            )

        # Class counts (used by balanced sampler).
        self._class_counts: Dict[str, int] = {}
        for sample in self.samples:
            c = sample[1]
            self._class_counts[c] = self._class_counts.get(c, 0) + 1

        print(f"[{dataset_name}/{split}] {len(self.samples)} samples, "
              f"{len(self._class_counts)} fine classes"
              + (f"  (fold {fold})" if fold is not None else ""))

        if print_class_counts and split == "all":
            report = scan_label_counts(
                dataset_name, root=str(root_path), exclude_artifacts=exclude_artifacts,
            )
            print(format_class_count_report(report, min_samples_per_class=min_samples_per_class))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def _preload_uint8_images(self) -> None:
        """Decode + resize once; keep RGB uint8 in RAM (augment per batch)."""
        n = len(self.samples)
        print(
            f"[{self.dataset_name}/{self.split}] preloading {n} images "
            f"as uint8 {self.img_size}x{self.img_size} …"
        )
        resize = transforms.Resize((self.img_size, self.img_size))
        arrs: List[np.ndarray] = []
        for sample in tqdm(self.samples, desc="preload", leave=False):
            path = sample[0]
            img = load_rgb_image(path)
            img = resize(img)
            arrs.append(np.asarray(img, dtype=np.uint8))
        self._preloaded_uint8 = np.stack(arrs, axis=0)
        gib = self._preloaded_uint8.nbytes / (1024 ** 3)
        print(
            f"[{self.dataset_name}/{self.split}] preload done "
            f"({n} images, {gib:.2f} GiB uint8)"
        )

    def _load_tensor_at(self, idx: int):
        sample = self.samples[idx]
        path, canonical = sample[0], sample[1]
        if len(sample) == 3:
            # CV split manifests store the *global* canonical fine index, so
            # the lineage and maturation targets are recoverable from it. The
            # old code hardcoded coarse=0 / mat_mask=0 here, which silently
            # disabled lineage supervision and masked all maturation for every
            # single-dataset CV run.
            fine_idx = sample[2]
            coarse_idx = FINE_IDX_TO_COARSE_IDX[fine_idx]
            chain_id, mat_pos = MATURATION_TABLE[fine_idx]
            mat_mask = float(chain_id >= 0)
        else:
            canonical = resolve_canonical(canonical) or canonical
            fine_idx = FINE_TO_IDX[canonical]
            coarse_idx = FINE_IDX_TO_COARSE_IDX[fine_idx]
            chain_id, mat_pos = MATURATION_TABLE[fine_idx]
            mat_mask = float(chain_id >= 0)

        if self._preloaded_uint8 is not None:
            img = Image.fromarray(self._preloaded_uint8[idx], mode="RGB")
            if self.transform:
                img = self.transform(img)
        elif self._preloaded is not None:
            img = self._preloaded[idx]
        else:
            img = load_rgb_image(path)
            if self.transform:
                img = self.transform(img)

        return img, fine_idx, coarse_idx, mat_pos, mat_mask

    def __getitem__(self, idx: int):
        n = len(self.samples)
        last_err = None
        for attempt in range(_MAX_LOAD_RETRIES):
            try:
                img, fine_idx, coarse_idx, mat_pos, mat_mask = self._load_tensor_at(
                    (idx + attempt) % n,
                )
                return (
                    img,
                    fine_idx,
                    coarse_idx,
                    float(mat_pos),
                    mat_mask,
                    self.dataset_id,
                )
            except (OSError, Image.UnidentifiedImageError) as exc:
                bad = self.samples[(idx + attempt) % n][0]
                if bad not in self._warned_bad_paths:
                    self._warned_bad_paths.add(bad)
                    print(f"[{self.dataset_name}] skip corrupt image: {bad} ({exc})")
                last_err = exc
        raise RuntimeError(
            f"[{self.dataset_name}] failed to load sample after {_MAX_LOAD_RETRIES} "
            f"retries near index {idx}: {last_err}",
        )

    # ------------------------------------------------------------------
    def class_weights_per_sample(self) -> torch.Tensor:
        w = torch.zeros(len(self.samples))
        for i, sample in enumerate(self.samples):
            w[i] = 1.0 / self._class_counts[sample[1]]
        return w

    def balanced_sampler(self, generator: Optional[torch.Generator] = None) -> WeightedRandomSampler:
        w = self.class_weights_per_sample()
        return WeightedRandomSampler(
            w, num_samples=len(w), replacement=True, generator=generator,
        )


# ---------------------------------------------------------------------------
# Concatenation across train datasets
# ---------------------------------------------------------------------------

class HierMultiDataset(Dataset):
    def __init__(self, datasets: List[HierHematologyDataset]):
        self.datasets = datasets
        self.offsets = [0]
        for ds in datasets:
            self.offsets.append(self.offsets[-1] + len(ds))
        self.total = self.offsets[-1]

        weights = [ds.class_weights_per_sample() for ds in datasets]
        self._sample_weights = torch.cat(weights)

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        for i, ds in enumerate(self.datasets):
            if idx < self.offsets[i + 1]:
                return ds[idx - self.offsets[i]]
        raise IndexError(idx)

    def balanced_sampler(self, generator: Optional[torch.Generator] = None) -> WeightedRandomSampler:
        return WeightedRandomSampler(
            self._sample_weights, num_samples=self.total, replacement=True,
            generator=generator,
        )


# ---------------------------------------------------------------------------
# Factory used by train.py
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_datasets: List[str],
    dataset_roots: Dict[str, str],
    batch_size: int = 64,
    img_size: int = 224,
    num_workers: int = 8,
    use_balanced_sampler: bool = True,
    seed: int = 42,
    data_fraction: float = 1.0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    preload_images: bool = False,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    min_samples_per_class: int = 5,
) -> Tuple[DataLoader, DataLoader, Union[HierMultiDataset, HierHematologyDataset],
           Union[HierMultiDataset, HierHematologyDataset]]:
    train_list, val_list = [], []
    for i, name in enumerate(train_datasets):
        root = dataset_roots.get(name) or os.environ.get(
            DATASETS[name].path_env_var,
        )
        train_list.append(HierHematologyDataset(
            name, root=root, transform=get_train_transform(img_size),
            split="train", seed=seed, dataset_id=i,
            data_fraction=data_fraction,
            max_samples=max_train_samples,
            preload_images=preload_images,
            img_size=img_size,
            min_samples_per_class=min_samples_per_class,
        ))
        val_list.append(HierHematologyDataset(
            name, root=root, transform=get_eval_transform(img_size),
            split="val", seed=seed, dataset_id=i,
            data_fraction=data_fraction,
            max_samples=max_val_samples,
            preload_images=preload_images,
            img_size=img_size,
            min_samples_per_class=min_samples_per_class,
        ))

    train_ds = HierMultiDataset(train_list) if len(train_list) > 1 else train_list[0]
    val_ds   = HierMultiDataset(val_list)   if len(val_list)   > 1 else val_list[0]

    # Preload + workers>0 duplicates RAM; feature cache uses its own loader.
    workers = num_workers

    sampler_gen, loader_gen, worker_init = dataloader_generators(seed)
    sampler = train_ds.balanced_sampler(sampler_gen) if use_balanced_sampler else None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=workers, pin_memory=True,
        drop_last=True, generator=loader_gen,
        worker_init_fn=worker_init if workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
        worker_init_fn=worker_init if workers > 0 else None,
    )
    if max_train_batches is not None:
        train_loader = _limit_batches(train_loader, max_train_batches)
    if max_val_batches is not None:
        val_loader = _limit_batches(val_loader, max_val_batches)
    return train_loader, val_loader, train_ds, val_ds


def build_cv_dataloaders(
    dataset_name: str,
    fold: int,
    dataset_root: Optional[str] = None,
    splits_dir: Optional[str] = None,
    batch_size: int = 64,
    img_size: int = 224,
    num_workers: int = 8,
    use_balanced_sampler: bool = True,
    preload_images: bool = False,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, HierHematologyDataset, HierHematologyDataset]:
    """
    Train / val loaders for ONE dataset and ONE precomputed CV fold.

    Requires split files under ``splits/<dataset>/fold_<k>.json``.
    Image size and filtering are read from ``splits/<dataset>/config.json``
    when present; ``img_size`` / ``batch_size`` args override the config.
    """
    from data.cv_splits import DEFAULT_SPLITS_DIR, splits_dir_for

    splits_root = Path(splits_dir) if splits_dir else DEFAULT_SPLITS_DIR
    cfg_path = splits_dir_for(dataset_name, splits_root) / "config.json"
    cfg = {}
    if cfg_path.exists():
        import json
        cfg = json.loads(cfg_path.read_text())
    if img_size == 224 and cfg.get("img_size"):
        img_size = int(cfg["img_size"])
    if batch_size == 64 and cfg.get("batch_size"):
        batch_size = int(cfg["batch_size"])

    root = dataset_root or cfg.get("root")
    kw = dict(
        dataset_name=dataset_name,
        root=root,
        fold=fold,
        splits_dir=str(splits_root),
        dataset_id=0,
        preload_images=preload_images,
        img_size=img_size,
    )
    train_ds = HierHematologyDataset(
        **kw, transform=get_train_transform(img_size), split="train",
    )
    val_ds = HierHematologyDataset(
        **kw, transform=get_eval_transform(img_size), split="val",
    )

    workers = num_workers
    sampler_gen, loader_gen, worker_init = dataloader_generators(seed)
    sampler = train_ds.balanced_sampler(sampler_gen) if use_balanced_sampler else None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=workers, pin_memory=True,
        drop_last=True, generator=loader_gen,
        worker_init_fn=worker_init if workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
        worker_init_fn=worker_init if workers > 0 else None,
    )
    if max_train_batches is not None:
        train_loader = _limit_batches(train_loader, max_train_batches)
    if max_val_batches is not None:
        val_loader = _limit_batches(val_loader, max_val_batches)
    return train_loader, val_loader, train_ds, val_ds


class _LimitedLoader:
    """Wrap a DataLoader and stop after `max_batches` batches per epoch."""

    def __init__(self, loader: DataLoader, max_batches: int):
        self.loader = loader
        self.max_batches = max_batches
        self.batch_size = loader.batch_size
        self.dataset = loader.dataset

    def __len__(self) -> int:
        return min(len(self.loader), self.max_batches)

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.max_batches:
                break
            yield batch


def _limit_batches(loader: DataLoader, max_batches: int) -> _LimitedLoader:
    return _LimitedLoader(loader, max_batches)
