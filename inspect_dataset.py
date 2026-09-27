#!/usr/bin/env python3
"""Inspect a YOLO-format grape/stem dataset and report data-quality issues.

Reads a YOLO ``data.yaml``, scans every split (train/val/test), validates
label files against a set of consistency rules, and writes an inspection
report (JSON + CSV) with per-class bounding-box statistics. The script is
strictly read-only: it never modifies the dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:  # Pillow is only needed for pixel-space bbox statistics.
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover
    Image = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Tokens treated as "grape" instances for pixel bbox statistics.
GRAPE_TOKENS = frozenset({
    "grape", "grapes", "grape_bunch", "grape bunch", "grape-bunch",
    "bunch", "grapes_green", "grapes_purple", "grapes_unripe",
    "green", "purple", "unripe",
})

#: Tokens treated as "stem" instances for pixel bbox statistics.
STEM_TOKENS = frozenset({
    "stem", "grape_stem", "grape stem", "grape-stem",
    "peduncle", "picking",
})

#: Number of whitespace-separated fields in a valid YOLO label line.
YOLO_FIELDS = 5


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect a YOLO dataset and report data-quality issues "
                    "(read-only, never modifies the data).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the YOLO data.yaml file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "dataset",
        help="Directory where the inspection reports are written "
             "(default: reports/dataset/).",
    )
    return parser.parse_args()


def load_data_yaml(data_yaml: Path) -> Dict[str, Any]:
    """Load and validate the YOLO ``data.yaml`` file.

    Args:
        data_yaml: Path to the YAML file.

    Returns:
        The parsed mapping.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML content is not a mapping or lacks names.
    """
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    with data_yaml.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"{data_yaml}: expected a YAML mapping, got {type(cfg).__name__}")
    if "names" not in cfg:
        raise ValueError(f"{data_yaml}: missing required 'names' key")
    return cfg


def extract_names(names_field: Any, nc: Optional[int] = None) -> List[str]:
    """Normalize the ``names`` field to an ID-indexed list of class names.

    Supports both YOLO formats: a list (``['grape', 'stem']``) and a dict
    (``{0: 'grape', 1: 'stem'}`` or ``{'0': 'grape', '1': 'stem'}``).

    Args:
        names_field: The raw ``names`` value from the YAML.
        nc: Optional declared number of classes; used for validation only.

    Returns:
        Class name per class ID (index = class ID).

    Raises:
        ValueError: If the field is neither a list nor a dict.
    """
    if isinstance(names_field, dict):
        try:
            names = [name for _, name in sorted(
                names_field.items(), key=lambda kv: int(kv[0]))]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"names dict has non-integer keys: {names_field!r}") from exc
    elif isinstance(names_field, list):
        names = list(names_field)
    else:
        raise ValueError(f"names must be a list or dict, got {type(names_field).__name__}")
    if not names:
        raise ValueError("names is empty")
    if nc is not None and nc > 0 and len(names) < nc:
        print(f"  [warn] names has {len(names)} entries but nc={nc}", file=sys.stderr)
    return names


def resolve_split_dir(data_yaml: Path, cfg: Dict[str, Any], key: str) -> Optional[Path]:
    """Resolve a split directory path from the data.yaml.

    Handles absolute and relative paths, and both ``<split>/images`` and
    ``<split>`` pointing directly at the images directory. Relative paths
    are resolved against the data.yaml location with a few common fallbacks
    (Roboflow exports sometimes sit one level above the split dirs, and
    stale ``../`` prefixes are retried without the parent hop), so a
    slightly off path is reported instead of crashing the scan.

    Args:
        data_yaml: Location of the data.yaml (base for relative paths).
        cfg: Parsed YAML mapping.
        key: Split key (``train``, ``val`` or ``test``).

    Returns:
        Resolved images directory, or None if the key is absent or the
        directory cannot be located.
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
        candidates.append(data_yaml.parent / p)                 # relative to yaml
        candidates.append(data_yaml.parent.parent / p)          # yaml one level down
        stripped = re.sub(r"^(?:\.\./)+", "", raw_str)           # stale ../ prefix
        if stripped != raw_str:
            candidates.append(data_yaml.parent / stripped)      # relative to yaml root
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            if (resolved / "images").is_dir():
                resolved = resolved / "images"
            if resolved is not candidates[0].resolve() or candidate != candidates[0]:
                print(f"  [warn] split '{key}': {raw_str!r} did not resolve at "
                      f"{candidates[0].resolve()}; using {resolved}",
                      file=sys.stderr)
            return resolved
    print(f"  [issue] split '{key}' directory not found "
          f"(tried: {', '.join(str(c.resolve()) for c in candidates)})",
          file=sys.stderr)
    return None


def find_labels_dir(images_dir: Path) -> Optional[Path]:
    """Locate the labels directory paired with an images directory."""
    candidates = [
        images_dir.parent / "labels",
        images_dir / "labels",
        images_dir.parent / "label",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _parse_label_line(line: str, label_path: Path, lineno: int) -> Optional[Tuple[int, List[float]]]:
    """Parse one YOLO label line.

    Args:
        line: Raw line from the label file.
        label_path: Label file path (for error reporting).
        lineno: 1-based line number (for error reporting).

    Returns:
        ``(class_id, [cx, cy, w, h])`` or None if the line is malformed.
    """
    parts = line.split()
    if len(parts) != YOLO_FIELDS:
        print(f"    [issue] {label_path}:{lineno}: expected {YOLO_FIELDS} fields, "
              f"got {len(parts)}", file=sys.stderr)
        return None
    try:
        numeric = [float(part) for part in parts]
    except ValueError:
        print(f"    [issue] {label_path}:{lineno}: non-numeric value in {line.strip()!r}",
              file=sys.stderr)
        return None
    cls, cx, cy, w, h = numeric
    if cls != int(cls):
        print(f"    [issue] {label_path}:{lineno}: class ID {cls!r} is not an integer",
              file=sys.stderr)
        return None
    return int(cls), [cx, cy, w, h]


def inspect_label_file(label_path: Path, image_size: Optional[Tuple[int, int]]) -> Dict[str, Any]:
    """Validate a single YOLO label file.

    Args:
        label_path: Path to the ``.txt`` label file.
        image_size: ``(width, height)`` of the paired image, or None if unknown.

    Returns:
        Dict with keys: ``bboxes`` (list of ``(class_id, w, h, w_px, h_px)``),
        ``issues`` (list of dicts) and ``n_boxes``.
    """
    bboxes: List[Tuple[int, float, float, Optional[float], Optional[float]]] = []
    issues: List[Dict[str, Any]] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        issues.append({"type": "unreadable", "detail": str(exc)})
        return {"bboxes": [], "issues": issues, "n_boxes": 0}

    img_w = image_size[0] if image_size else None
    img_h = image_size[1] if image_size else None
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            issues.append({"type": "empty_line", "line": lineno})
            continue
        parsed = _parse_label_line(line, label_path, lineno)
        if parsed is None:
            issues.append({"type": "illegal", "line": lineno, "detail": line.strip()})
            continue
        cls, (cx, cy, w, h) = parsed
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            issues.append({
                "type": "center_out_of_range", "line": lineno,
                "detail": f"cx={cx}, cy={cy}", "class_id": cls,
            })
        if not (w > 0.0 and h > 0.0):
            issues.append({
                "type": "non_positive_size", "line": lineno,
                "detail": f"w={w}, h={h}", "class_id": cls,
            })
        elif not (w <= 1.0 and h <= 1.0):
            issues.append({
                "type": "size_out_of_range", "line": lineno,
                "detail": f"w={w}, h={h}", "class_id": cls,
            })
        w_px = w * img_w if img_w else None
        h_px = h * img_h if img_h else None
        bboxes.append((cls, w, h, w_px, h_px))
    return {"bboxes": bboxes, "issues": issues, "n_boxes": len(bboxes)}


def image_size_of(path: Path, cache: Dict[str, Optional[Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
    """Return the pixel size of an image, using PIL with a per-path cache."""
    key = str(path)
    if key in cache:
        return cache[key]
    if Image is None:
        cache[key] = None
        return None
    try:
        with Image.open(path) as img:
            size = img.size  # type: ignore[attr-defined]
    except Exception:
        size = None
    cache[key] = size
    return size


def describe(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics (min/max/mean/std/median/p5/p95) of values."""
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)
    p5 = ordered[max(0, int(round(0.05 * (n - 1))))]
    p95 = ordered[min(n - 1, int(round(0.95 * (n - 1))))]
    stats: Dict[str, float] = {
        "count": n,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "stdev": statistics.pstdev(ordered),
        "p5": p5,
        "p95": p95,
    }
    return stats


def is_grape_name(name: str) -> bool:
    """Return True if a class name denotes a grape instance (case-insensitive)."""
    return name.strip().lower() in GRAPE_TOKENS


def is_stem_name(name: str) -> bool:
    """Return True if a class name denotes a stem instance (case-insensitive)."""
    return name.strip().lower() in STEM_TOKENS


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    """Write a CSV report file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    """Entry point: inspect the dataset and write reports."""
    args = parse_args()

    print(f"Inspecting dataset defined by {args.data}")
    cfg = load_data_yaml(args.data)
    names = extract_names(cfg.get("names"), cfg.get("nc"))
    print(f"  classes ({len(names)}): {names}")

    splits: List[Tuple[str, Optional[Path]]] = []
    issues: List[Dict[str, Any]] = []
    for key in ("train", "val", "test"):
        split_dir = resolve_split_dir(args.data, cfg, key)
        splits.append((key, split_dir))
        if split_dir is not None:
            status = str(split_dir)
        elif key in cfg and cfg.get(key) is not None:
            status = "NOT FOUND"
            issues.append({
                "type": "split_dir_missing",
                "split": key,
                "detail": str(cfg.get(key)),
            })
        else:
            status = "not defined"
        print(f"  {key}: {status}")
    if all(d is None for _, d in splits):
        raise ValueError(f"{args.data}: no train/val/test paths defined")

    # ---- Aggregate counters -------------------------------------------------
    per_name: Dict[str, Dict[str, Any]] = {}
    per_id: Dict[int, int] = {}
    total_bboxes = 0
    total_images = 0
    total_labels = 0
    split_summary: Dict[str, Dict[str, int]] = {}
    image_to_split: Dict[str, str] = {}
    grape_sizes: List[Tuple[float, float]] = []  # (w_px, h_px)
    stem_sizes: List[Tuple[float, float]] = []
    image_cache: Dict[str, Optional[Tuple[int, int]]] = {}

    for split_name, images_dir in splits:
        if images_dir is None:
            continue
        labels_dir = find_labels_dir(images_dir)
        images = sorted(p for p in images_dir.iterdir() if p.is_file())
        labels = sorted(p for p in labels_dir.iterdir() if p.is_file()) if labels_dir else []
        label_by_stem = {p.stem: p for p in labels}
        print(f"  scanning {split_name}: {len(images)} images, {len(labels)} labels")
        split_summary[split_name] = {"images": len(images), "labels": len(labels)}
        total_images += len(images)
        total_labels += len(labels)

        for image_path in images:
            resolved = str(image_path.resolve())
            if resolved in image_to_split:
                issues.append({
                    "type": "image_in_multiple_splits",
                    "image": resolved,
                    "splits": [image_to_split[resolved], split_name],
                })
            image_to_split[resolved] = split_name

            size = image_size_of(image_path, image_cache)
            label_path = label_by_stem.get(image_path.stem)
            if label_path is None:
                issues.append({
                    "type": "missing_label",
                    "image": str(image_path),
                    "split": split_name,
                })
                continue
            del label_by_stem[image_path.stem]

            result = inspect_label_file(label_path, size)
            if result["issues"]:
                for issue in result["issues"]:
                    issue.update({"label": str(label_path), "split": split_name})
                    issues.append(issue)
            n_boxes = result["n_boxes"]
            total_bboxes += n_boxes
            for cls, _w, _h, w_px, h_px in result["bboxes"]:
                name = names[cls] if 0 <= cls < len(names) else f"<id={cls}>"
                entry = per_name.setdefault(name, {"class_id": cls, "instances": 0, "bboxes": 0})
                entry["instances"] += 1
                entry["bboxes"] += 1
                per_id[cls] = per_id.get(cls, 0) + 1
                if w_px is not None and h_px is not None:
                    if is_grape_name(name):
                        grape_sizes.append((w_px, h_px))
                    elif is_stem_name(name):
                        stem_sizes.append((w_px, h_px))

        # Remaining labels have no image.
        for label_path in label_by_stem.values():
            issues.append({
                "type": "missing_image",
                "label": str(label_path),
                "split": split_name,
            })

    # ---- Build pixel-space distribution stats -------------------------------
    def pixel_stats(sizes: List[Tuple[float, float]]) -> Dict[str, Dict[str, float]]:
        widths = [w for w, _h in sizes]
        heights = [h for _w, h in sizes]
        return {"width": describe(widths), "height": describe(heights)}

    distributions: Dict[str, Dict[str, Any]] = {
        "grape_bbox_pixel_stats": pixel_stats(grape_sizes),
        "stem_bbox_pixel_stats": pixel_stats(stem_sizes),
        "raw_widths_heights": {
            "grape": grape_sizes,
            "stem": stem_sizes,
        },
    }

    summary: Dict[str, Any] = {
        "data_yaml": str(args.data.resolve()),
        "class_names": names,
        "nc": len(names),
        "totals": {
            "images": total_images,
            "labels": total_labels,
            "bboxes": total_bboxes,
        },
        "per_split": split_summary,
        "per_class": {
            name: dict(info, name=name) for name, info in sorted(per_name.items())
        },
        "per_class_id": {str(k): v for k, v in sorted(per_id.items())},
        "bbox_pixel_stats": distributions,
        "issue_count": len(issues),
        "issues": issues,
    }

    # ---- Write reports ------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "dataset_inspection.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"  wrote {json_path}")

    class_rows = [
        [info["class_id"], name, info["instances"], info["bboxes"]]
        for name, info in sorted(per_name.items())
    ]
    write_csv(
        args.output_dir / "dataset_class_summary.csv",
        ["class_id", "class_name", "instances", "bboxes"],
        class_rows,
    )

    stat_rows: List[List[Any]] = []
    for label, stats in (("grape", distributions["grape_bbox_pixel_stats"]),
                         ("stem", distributions["stem_bbox_pixel_stats"])):
        for key in ("width", "height"):
            stat_rows.append([label, key] + [stats.get(key, {}).get(field) for field in
                                             ("count", "min", "max", "mean", "median", "stdev", "p5", "p95")])
    write_csv(
        args.output_dir / "dataset_bbox_pixel_stats.csv",
        ["class", "dimension", "count", "min", "max", "mean", "median", "stdev", "p5", "p95"],
        stat_rows,
    )
    print(f"  wrote {args.output_dir / 'dataset_class_summary.csv'}")
    print(f"  wrote {args.output_dir / 'dataset_bbox_pixel_stats.csv'}")

    # ---- Print summary and issues ------------------------------------------
    print(f"\nTotal images: {total_images}, labels: {total_labels}, bboxes: {total_bboxes}")
    for name, info in sorted(per_name.items()):
        print(f"  class {info['class_id']} '{name}': {info['instances']} instances / {info['bboxes']} bboxes")
    print(f"Grape bbox pixel stats (n={len(grape_sizes)}): {distributions['grape_bbox_pixel_stats']}")
    print(f"Stem bbox pixel stats (n={len(stem_sizes)}): {distributions['stem_bbox_pixel_stats']}")

    if issues:
        print(f"\n{len(issues)} potential data issue(s):")
        for issue in issues[:50]:
            location = issue.get("image") or issue.get("label") or ""
            detail = f" {issue.get('detail')}" if issue.get("detail") else ""
            print(f"  [{issue['type']}] {location}{detail}")
        if len(issues) > 50:
            print(f"  ... and {len(issues) - 50} more (full list in {json_path})")
    else:
        print("\nNo data issues found.")


if __name__ == "__main__":
    main()
