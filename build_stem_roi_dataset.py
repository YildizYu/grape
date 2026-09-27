#!/usr/bin/env python3
"""Build the stage-2 stem ROI dataset from a normalized YOLO dataset.

Run with the project's YOLO environment, e.g.::

    /home/user/miniconda3/envs/sam3/bin/python scripts/build_stem_roi_dataset.py \\
        --data data/Grape/data.yaml \\
        --output-dir datasets/stage2_stem_roi \\
        --config configs/pipeline.yaml \\
        --splits train val test

For every split the script:

1. reads all images and YOLO labels,
2. separates grape bboxes (class 0) from stem bboxes (class 1),
3. expands each grape bbox into an ROI, matches stems to grapes and
   converts the stem boxes to ROI-local YOLO coordinates,
4. applies jittered ROI variants (train split only),
5. generates negative samples (empty label files) at the configured ratio,
6. saves ROI crops to ``datasets/stage2_stem_roi/{split}/images`` and YOLO
   labels to ``datasets/stage2_stem_roi/{split}/labels``,
7. writes a ``data.yaml`` with a single class ``stem``,
8. writes ``metadata.jsonl`` per split,
9. prints per-split statistics (ROI counts, positive/negative ratio and
   stem pixel sizes).
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from grape_stem.config import load_yaml  # noqa: E402
from grape_stem.roi_builder import ROIBuilder  # noqa: E402


def resolve_dataset_dir(data_yaml_path: Path) -> Path:
    """Derive the dataset root from a normalized ``data.yaml``.

    The root is the directory that contains the ``{split}/images`` folders.
    Several layouts are supported:

    * absolute paths in the yaml (root = two levels above the image dir),
    * relative paths resolved against the yaml location (e.g. ``data.yaml``
      inside ``data/Grape/`` with ``train: ../train/images`` or
      ``train: train/images``).

    Candidate roots are probed in order and the first one containing a
    ``train`` / ``valid`` / ``val`` image directory wins.
    """
    data_yaml_path = Path(data_yaml_path).resolve()
    cfg = load_yaml(data_yaml_path)
    train_key = cfg.get("train") or cfg.get("val")
    if not train_key:
        raise ValueError(f"No train/val path found in {data_yaml_path}")
    ref = Path(str(train_key))
    if not ref.is_absolute():
        ref = data_yaml_path.parent / ref
    ref = ref.resolve()

    candidates: List[Path] = []
    for d in (data_yaml_path.parent,
              data_yaml_path.parent.parent,
              data_yaml_path.parent.parent.parent,
              ref.parents[1] if len(ref.parents) > 1 else ref.parent):
        if d not in candidates:
            candidates.append(d)

    for cand in candidates:
        if any((cand / s / "images").is_dir() for s in ("train", "valid", "val")):
            return cand
    raise ValueError(
        "Could not locate the dataset root (directory containing "
        f"train/valid images) from {data_yaml_path}; tried: "
        f"{[str(c) for c in candidates]}"
    )


def write_data_yaml(output_dir: Path, splits: List[str]) -> None:
    """Write the stage-2 ``data.yaml`` (single class: stem)."""
    data: Dict[str, Any] = {"nc": 1, "names": ["stem"]}
    for token in ("train", "val", "test"):
        if token in splits:
            data[token] = str((output_dir / token / "images").resolve())
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(data, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


def print_stats(all_stats: List[Dict[str, Any]]) -> None:
    """Print a per-split statistics table."""
    header = (
        f"{'split':<6} {'imgs':>5} {'grapes':>7} {'stems':>6} {'matched':>8} "
        f"{'pos':>8} {'neg':>7} {'total':>7} {'neg%':>6}  stem px (w x h)"
    )
    print(header)
    print("-" * len(header))
    for s in all_stats:
        sizes = s.get("stem_sizes", {}) or {}
        w_mean = sizes.get("width_px", {}).get("mean", 0.0)
        h_mean = sizes.get("height_px", {}).get("mean", 0.0)
        print(
            f"{s['split']:<6} {s['images_total']:>5} {s['grapes_total']:>7} "
            f"{s['stems_total']:>6} {s['stems_matched']:>8} "
            f"{s['positive_rois']:>8} {s['negative_rois']:>7} "
            f"{s['rois_total']:>7} {s['negative_ratio'] * 100:>5.1f}%  "
            f"{w_mean:.0f} x {h_mean:.0f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a stage-2 stem ROI dataset from a normalized YOLO dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to the normalized data.yaml (grape=class 0, stem=class 1).")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "datasets" / "stage2_stem_roi",
                        help="Output root: crops under {split}/images, labels under "
                             "{split}/labels, plus data.yaml and per-split metadata.jsonl.")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs" / "pipeline.yaml",
                        help="Pipeline YAML config (roi / roi_jitter / negative_samples).")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Splits to process (a 'val' split falls back to 'valid').")
    parser.add_argument("--grape-class", type=int, default=None,
                        help="Override the grape class id (default: from config).")
    parser.add_argument("--stem-class", type=int, default=None,
                        help="Override the stem class id (default: from config).")
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"data.yaml not found: {args.data}")
    if not args.config.is_file():
        parser.error(f"config file not found: {args.config}")

    config = load_yaml(args.config)
    if args.grape_class is not None or args.stem_class is not None:
        config.setdefault("classes", {})
        if args.grape_class is not None:
            config["classes"]["grapes"] = args.grape_class
        if args.stem_class is not None:
            config["classes"]["stem"] = args.stem_class

    dataset_dir = resolve_dataset_dir(args.data)
    output_dir = args.output_dir.resolve()
    builder = ROIBuilder(config)

    print(f"Dataset root : {dataset_dir}")
    print(f"Output dir   : {output_dir}")
    print()

    all_stats: List[Dict[str, Any]] = []
    for split in args.splits:
        print(f"[build] split={split}")
        stats = builder.build_roi_dataset(dataset_dir, split, output_dir)
        all_stats.append(stats)

    if not all_stats:
        print("No splits processed. Aborting.")
        return 1

    write_data_yaml(output_dir, [s["split"] for s in all_stats])

    print()
    print_stats(all_stats)

    total_rois = sum(s["rois_total"] for s in all_stats)
    total_neg = sum(s["negative_rois"] for s in all_stats)
    print(f"\nTotal ROIs: {total_rois} "
          f"(positive {total_rois - total_neg}, negative {total_neg})")
    print(f"data.yaml  : {(output_dir / 'data.yaml').resolve()}")
    for s in all_stats:
        if s["negative_shortfall"]:
            print(f"[warn] split {s['split']}: only {s['negative_rois']} of "
                  f"{s['negative_rois'] + s['negative_shortfall']} target "
                  f"negative samples generated")
    print("metadata.jsonl per split:")
    for s in all_stats:
        print(f"  {s['metadata_file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
