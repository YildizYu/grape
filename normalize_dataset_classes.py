#!/usr/bin/env python3
"""Normalize a YOLO dataset to the standard grape/stem class schema.

Reads the source ``data.yaml`` (old class ID -> old name) and the
``class_aliases.yaml`` config, then writes a normalized copy of the dataset
under ``--output-dir`` (``datasets/normalized/`` by default) in which every
label's class-ID column is mapped to the standard schema (``grapes=0``,
``stem=1``). Images are linked (symlink) or copied, a new ``data.yaml`` is
generated, and mapping/report files are written to ``reports/dataset/``.

Class names that the alias config cannot resolve are handled according to
``--unknown-policy``: ``error`` aborts the whole run (after pre-scanning,
so nothing is written), ``skip`` drops just those bounding-box lines.

Original data is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grape_stem.class_normalizer import ClassNormalizer  # noqa: E402

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "normalized"
REPORT_DIR = PROJECT_ROOT / "reports" / "dataset"

logger = logging.getLogger("normalize_dataset_classes")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Normalize a YOLO dataset's class labels to the standard "
                    "grape/stem schema.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the original YOLO data.yaml.",
    )
    parser.add_argument(
        "--aliases",
        type=Path,
        default=PROJECT_ROOT / "configs" / "class_aliases.yaml",
        help="Path to the class-aliases YAML (default: configs/class_aliases.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the normalized dataset is written "
             "(default: datasets/normalized/).",
    )
    parser.add_argument(
        "--unknown-policy",
        choices=("error", "skip"),
        default="error",
        help="How to handle classes not present in the aliases config "
             "(default: error).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy image files instead of creating symlinks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    return parser.parse_args()


def setup_logging() -> Path:
    """Configure logging to the terminal and a file in reports/dataset/."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = REPORT_DIR / "normalize_classes.log"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )
    return log_file


def load_yaml(path: Path, what: str) -> Dict[str, Any]:
    """Load a YAML file as a dict, with a helpful error on failure."""
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"failed to parse {what} {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"{what} {path}: expected a YAML mapping")
    return cfg


def extract_names(names_field: Any) -> List[str]:
    """Return class names indexed by class ID (list or dict format)."""
    if isinstance(names_field, dict):
        try:
            return [name for _id, name in sorted(
                names_field.items(), key=lambda kv: int(kv[0]))]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"names dict has non-integer keys: {names_field!r}") from exc
    if isinstance(names_field, list):
        return list(names_field)
    raise ValueError(f"names must be a list or dict, got {type(names_field).__name__}")


def resolve_split_dir(data_yaml: Path, cfg: Dict[str, Any], key: str) -> Optional[Path]:
    """Resolve a split's images directory (absolute or relative path).

    Relative paths are resolved against the data.yaml location with a few
    common fallbacks (Roboflow exports sometimes sit one level above the
    split dirs, and stale ``../`` prefixes are retried without the parent
    hop), mirroring ``inspect_dataset.py``.
    """
    raw = cfg.get(key)
    if raw is None:
        return None
    raw_str = str(raw)
    p = Path(raw_str)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(data_yaml.parent / p)                  # relative to yaml
        candidates.append(data_yaml.parent.parent / p)           # yaml one level down
        stripped = re.sub(r"^(?:\.\./)+", "", raw_str)            # stale ../ prefix
        if stripped != raw_str:
            candidates.append(data_yaml.parent / stripped)       # relative to yaml root
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            if (resolved / "images").is_dir():
                resolved = resolved / "images"
            if candidate is not candidates[0]:
                logger.warning("split '%s': %r did not resolve at %s; using %s",
                               key, raw_str, candidates[0].resolve(), resolved)
            return resolved
    raise FileNotFoundError(
        f"split '{key}' directory from {data_yaml} does not exist "
        f"(tried: {', '.join(str(c.resolve()) for c in candidates)})"
    )


def find_labels_dir(images_dir: Path) -> Optional[Path]:
    """Locate the labels directory paired with an images directory."""
    for candidate in (images_dir.parent / "labels", images_dir / "labels"):
        if candidate.is_dir():
            return candidate
    return None


def pre_scan_unknowns(
    normalizer: ClassNormalizer,
    old_names: List[str],
    splits: List[Tuple[str, Optional[Path]]],
) -> List[Dict[str, Any]]:
    """Count occurrences of classes the normalizer cannot resolve.

    Scans every label file without modifying anything, so an ``error``
    policy run can abort before any data is written.

    Returns:
        List of ``{old_name, old_class_id, occurrences}`` dicts.
    """
    unknown: Dict[Tuple[str, int], int] = {}
    for _split_name, images_dir in splits:
        if images_dir is None:
            continue
        labels_dir = find_labels_dir(images_dir)
        if labels_dir is None:
            continue
        for label_path in sorted(labels_dir.glob("*.txt")):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    class_id = -1
                old_name = old_names[class_id] if 0 <= class_id < len(old_names) else f"<id={class_id}>"
                try:
                    normalizer.normalize_name(old_name)
                except ValueError:
                    unknown[(old_name, class_id)] = unknown.get((old_name, class_id), 0) + 1
    return [
        {"old_name": name, "old_class_id": class_id, "occurrences": count}
        for (name, class_id), count in sorted(unknown.items())
    ]


def link_or_copy_image(src: Path, dst: Path, use_copy: bool) -> str:
    """Link (symlink) or copy an image into the normalized dataset.

    Returns:
        The action actually performed: ``"linked"`` or ``"copied"``.
    """
    if use_copy:
        shutil.copy2(src, dst)
        return "copied"
    try:
        dst.symlink_to(src)
    except OSError:
        logger.warning("symlink failed for %s, falling back to copy", src)
        shutil.copy2(src, dst)
        return "copied"
    return "linked"


def write_unknowns_csv(unknowns: List[Dict[str, Any]]) -> Path:
    """Write unknown_classes.csv to reports/dataset/ (header-only if none)."""
    csv_path = REPORT_DIR / "unknown_classes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["old_name", "old_class_id", "occurrences"])
        for row in unknowns:
            writer.writerow([row["old_name"], row["old_class_id"], row["occurrences"]])
    return csv_path


def write_new_data_yaml(output_dir: Path, split_dirs: Dict[str, Path],
                        standard_names: List[str]) -> Path:
    """Generate the normalized data.yaml with relative split paths."""
    cfg: Dict[str, Any] = {"nc": len(standard_names), "names": standard_names}
    for key, split_dir in split_dirs.items():
        split_path = split_dir.relative_to(output_dir)
        if (split_dir / "images").is_dir():
            split_path = split_path / "images"
        cfg[key] = split_path.as_posix()
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(cfg, default_flow_style=False, allow_unicode=True,
                       sort_keys=False),
        encoding="utf-8",
    )
    return data_yaml


def process_split(
    split_name: str,
    images_dir: Path,
    output_split: Path,
    old_names: List[str],
    normalizer: ClassNormalizer,
    use_copy: bool,
    skip_unknown: bool,
    unknown_counts: Dict[Tuple[str, int], int],
) -> Dict[str, Any]:
    """Normalize one split: link images, rewrite labels, return statistics."""
    labels_dir = find_labels_dir(images_dir)
    out_images = output_split / "images"
    out_labels = output_split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_dir.iterdir() if p.is_file())
    labels_by_stem: Dict[str, Path] = {}
    if labels_dir is not None:
        labels_by_stem = {p.stem: p for p in labels_dir.glob("*.txt")}

    stats: Dict[str, Any] = {
        "images": len(images),
        "labels_written": 0,
        "boxes_normalized": 0,
        "boxes_skipped": 0,
        "missing_labels": 0,
    }
    for image_path in images:
        link_or_copy_image(image_path, out_images / image_path.name, use_copy)
        label_path = labels_by_stem.get(image_path.stem)
        if label_path is None:
            stats["missing_labels"] += 1
            continue
        out_label = out_labels / label_path.name
        if skip_unknown:
            # Write only the lines whose class the normalizer can resolve.
            kept: List[str] = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    class_id = -1
                old_name = (old_names[class_id]
                            if 0 <= class_id < len(old_names) else f"<id={class_id}>")
                try:
                    _new_name, new_id = normalizer.normalize_name(old_name)
                except ValueError:
                    unknown_counts[(old_name, class_id)] = (
                        unknown_counts.get((old_name, class_id), 0) + 1)
                    stats["boxes_skipped"] += 1
                    continue
                kept.append(f"{new_id} {' '.join(parts[1:])}")
            out_label.write_text("\n".join(kept) + ("\n" if kept else ""),
                                 encoding="utf-8")
            stats["boxes_normalized"] += len(kept)
        else:
            normalizer.normalize_label_file(label_path, old_names, out_label)
            stats["boxes_normalized"] += sum(
                1 for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip())
        stats["labels_written"] += 1
    logger.info("%s: %d images, %d labels, %d boxes normalized, %d skipped",
                split_name, stats["images"], stats["labels_written"],
                stats["boxes_normalized"], stats["boxes_skipped"])
    return stats


def main() -> None:
    """Entry point: normalize the dataset classes."""
    args = parse_args()
    log_file = setup_logging()
    logger.info("Starting dataset class normalization")
    logger.info("data.yaml: %s", args.data.resolve())
    logger.info("aliases:   %s", args.aliases.resolve())
    logger.info("output:    %s", args.output_dir.resolve())
    logger.info("policy:    %s", args.unknown_policy)
    started = datetime.now(timezone.utc)

    data_cfg = load_yaml(args.data, "data.yaml")
    aliases_cfg = load_yaml(args.aliases, "aliases config")
    old_names = extract_names(data_cfg.get("names"))
    logger.info("Old classes (%d): %s", len(old_names), old_names)
    if not old_names:
        raise ValueError(f"{args.data}: 'names' is empty")

    normalizer = ClassNormalizer(aliases_cfg)
    logger.info("Standard classes: %s", normalizer.standard_names)

    splits: List[Tuple[str, Optional[Path]]] = [
        (key, resolve_split_dir(args.data, data_cfg, key))
        for key in ("train", "val", "test")
    ]

    # Pre-scan so an 'error' policy aborts before anything is written.
    unknown = pre_scan_unknowns(normalizer, old_names, splits)
    if unknown and args.unknown_policy == "error":
        write_unknowns_csv(unknown)
        logger.error("Aborting: %d unknown class(es) found (use "
                     "--unknown-policy skip to drop those boxes):",
                     len(unknown))
        for row in unknown:
            logger.error("  %s (class id %s): %d occurrence(s)",
                         row["old_name"], row["old_class_id"], row["occurrences"])
        logger.error("No files were written. Report: %s", REPORT_DIR / "unknown_classes.csv")
        sys.exit(1)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.force:
            logger.error("Output directory already exists and is not empty: %s "
                         "(use --force to overwrite)", args.output_dir)
            sys.exit(1)
        logger.warning("Overwriting existing output directory %s", args.output_dir)
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    skip_unknown = args.unknown_policy == "skip"
    unknown_counts: Dict[Tuple[str, int], int] = {
        (row["old_name"], row["old_class_id"]): row["occurrences"] for row in unknown
    }
    split_dirs: Dict[str, Path] = {}
    per_split: Dict[str, Any] = {}
    for split_name, images_dir in splits:
        if images_dir is None:
            logger.info("split '%s' not defined in data.yaml; skipped", split_name)
            continue
        output_split = args.output_dir / split_name
        per_split[split_name] = process_split(
            split_name, images_dir, output_split, old_names,
            normalizer, args.copy, skip_unknown, unknown_counts,
        )
        split_dirs[split_name] = output_split

    data_yaml_path = write_new_data_yaml(
        args.output_dir, split_dirs, normalizer.standard_names)
    logger.info("wrote %s", data_yaml_path)

    # ---- Reports ------------------------------------------------------------
    mapping: Dict[str, Dict[str, Any]] = {}
    for old_name in old_names:
        try:
            new_name, new_id = normalizer.normalize_name(old_name)
        except ValueError:
            mapping[old_name] = {"new_name": None, "new_id": None, "skipped": True}
        else:
            mapping[old_name] = {"new_name": new_name, "new_id": new_id}
    class_mapping: Dict[str, Any] = {
        "schema": {name: new_id for name, new_id in normalizer.standard_ids.items()},
        "mapping": mapping,
    }
    class_mapping_path = REPORT_DIR / "class_mapping.json"
    class_mapping_path.write_text(
        json.dumps(class_mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %s", class_mapping_path)

    unknown_rows = [
        {"old_name": name, "old_class_id": class_id, "occurrences": count}
        for (name, class_id), count in sorted(unknown_counts.items())
    ]
    unknowns_path = write_unknowns_csv(unknown_rows)
    logger.info("wrote %s", unknowns_path)

    report: Dict[str, Any] = {
        "source": {
            "data_yaml": str(args.data.resolve()),
            "aliases_yaml": str(args.aliases.resolve()),
        },
        "unknown_policy": args.unknown_policy,
        "image_strategy": "copy" if args.copy else "symlink",
        "old_classes": old_names,
        "standard_classes": normalizer.standard_names,
        "per_split": per_split,
        "totals": {
            key: sum(split.get(key, 0) for split in per_split.values())
            for key in ("images", "labels_written", "boxes_normalized", "boxes_skipped")
        },
        "unknown_classes": unknown_rows,
        "output_dir": str(args.output_dir.resolve()),
        "data_yaml_output": str(data_yaml_path),
        "log_file": str(log_file),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = REPORT_DIR / "normalization_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %s", report_path)

    logger.info("Done in %.1f s. Normalized dataset: %s",
                (datetime.now(timezone.utc) - started).total_seconds(),
                args.output_dir)
    if unknown_rows:
        logger.warning("%d unknown class(es) skipped: %d boxes dropped",
                       len(unknown_rows), report["totals"]["boxes_skipped"])


if __name__ == "__main__":
    main()
