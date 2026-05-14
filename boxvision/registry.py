"""
Dataset registry — resolve dataset names to concrete paths via datasets.yaml.

The registry decouples training code from filesystem layouts. Any dataset
(Roboflow, native COCO, YOLO TXT, custom) is registered once in datasets.yaml
and then referenced by name across the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class SplitSpec:
    """Resolved paths for a single split (train/val/test)."""

    images: str
    annotations: Optional[str] = None  # COCO format
    labels: Optional[str] = None       # YOLO TXT format


@dataclass
class DatasetSpec:
    """A registered dataset, with all paths resolved to absolute strings."""

    name: str
    format: str                # "coco" or "yolo"
    root: str                  # absolute path
    class_agnostic: bool
    description: str
    license: str
    source: str
    train: SplitSpec
    val: SplitSpec
    test: Optional[SplitSpec] = None


def _project_root() -> Path:
    """Locate the project root by walking up until datasets.yaml is found."""
    cur = Path(__file__).resolve().parent
    for _ in range(6):
        if (cur / "datasets.yaml").exists():
            return cur
        cur = cur.parent
    raise FileNotFoundError(
        "Could not locate datasets.yaml — expected at project root, "
        "starting search from boxvision package directory."
    )


def _resolve(root: Path, relative: Optional[str]) -> Optional[str]:
    if relative is None:
        return None
    return str((root / relative).resolve())


def load_registry(yaml_path: Optional[str] = None) -> dict:
    """Load and return the raw registry dictionary from datasets.yaml."""
    if yaml_path is None:
        yaml_path = str(_project_root() / "datasets.yaml")
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f) or {}


def load_dataset(name: str, yaml_path: Optional[str] = None) -> DatasetSpec:
    """
    Resolve a dataset name to a DatasetSpec with absolute paths.

    Args:
        name: Dataset name as registered in datasets.yaml (e.g. "road-signs")
        yaml_path: Optional override for the registry YAML location

    Raises:
        KeyError: if the dataset name is not registered
        ValueError: if the entry is malformed
    """
    registry = load_registry(yaml_path)
    if name not in registry:
        available = ", ".join(sorted(registry.keys())) or "(none)"
        raise KeyError(
            f"Dataset '{name}' not found in registry. Available: {available}"
        )

    entry = registry[name]
    project_root = _project_root()
    root_path = (project_root / entry["root"]).resolve()

    fmt = entry.get("format", "coco")
    if fmt not in ("coco", "yolo"):
        raise ValueError(f"Dataset '{name}': format must be 'coco' or 'yolo', got '{fmt}'")

    splits = entry.get("splits", {})

    def parse_split(key: str, required: bool) -> Optional[SplitSpec]:
        if key not in splits:
            if required:
                raise ValueError(f"Dataset '{name}': missing required split '{key}'")
            return None
        s = splits[key]
        return SplitSpec(
            images=_resolve(root_path, s.get("images")),
            annotations=_resolve(root_path, s.get("annotations")),
            labels=_resolve(root_path, s.get("labels")),
        )

    return DatasetSpec(
        name=name,
        format=fmt,
        root=str(root_path),
        class_agnostic=entry.get("class_agnostic", True),
        description=entry.get("description", ""),
        license=entry.get("license", ""),
        source=entry.get("source", ""),
        train=parse_split("train", required=True),
        val=parse_split("val", required=True),
        test=parse_split("test", required=False),
    )


def list_datasets(yaml_path: Optional[str] = None) -> list[dict]:
    """Return a summary of all registered datasets — used by `boxvision datasets` CLI."""
    registry = load_registry(yaml_path)
    return [
        {
            "name": name,
            "format": entry.get("format", "coco"),
            "description": entry.get("description", ""),
            "license": entry.get("license", ""),
        }
        for name, entry in sorted(registry.items())
    ]


def verify_dataset(spec: DatasetSpec) -> list[str]:
    """
    Check that all paths in a DatasetSpec exist on disk.

    Returns a list of error messages (empty if everything is present).
    """
    errors: list[str] = []
    for split_name, split in [("train", spec.train), ("val", spec.val), ("test", spec.test)]:
        if split is None:
            continue
        if not os.path.isdir(split.images):
            errors.append(f"{split_name}: images directory not found: {split.images}")
        if spec.format == "coco":
            if split.annotations is None:
                errors.append(f"{split_name}: COCO format requires 'annotations' path")
            elif not os.path.isfile(split.annotations):
                errors.append(f"{split_name}: annotation file not found: {split.annotations}")
        elif spec.format == "yolo":
            if split.labels is None:
                errors.append(f"{split_name}: YOLO format requires 'labels' path")
            elif not os.path.isdir(split.labels):
                errors.append(f"{split_name}: labels directory not found: {split.labels}")
    return errors
