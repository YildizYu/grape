#!/usr/bin/env python3
"""生成葡萄模型数据集：只保留 grapes 类别，删除所有 stem 标签"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.grape_stem.dataset_utils import read_yolo_labels, write_yolo_labels


def main():
    parser = argparse.ArgumentParser(description="Build grapes-only dataset")
    parser.add_argument("--data", required=True, help="Path to normalized data.yaml")
    parser.add_argument("--output-dir", default="datasets/stage1_grapes")
    args = parser.parse_args()

    data_path = Path(args.data)
    base_dir = data_path.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_path) as f:
        data_cfg = yaml.safe_load(f)

    # 类别映射：grapes=0, stem=1 → grapes-only: grapes=0
    print("Building grapes-only dataset...")

    for split_key in ["train", "val", "test"]:
        if split_key not in data_cfg:
            continue

        split_path = resolve_path(data_cfg[split_key], base_dir)
        label_dir = split_path.parent / "labels"

        img_out = output_dir / split_key / "images"
        lbl_out = output_dir / split_key / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        grape_count = 0
        img_count = 0
        for img_path in split_path.glob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            # 复制图像
            shutil.copy2(img_path, img_out / img_path.name)

            # 过滤标签：只保留 class 0 (grapes)
            lbl_path = label_dir / (img_path.stem + ".txt")
            labels = read_yolo_labels(lbl_path)
            grape_labels = [l for l in labels if l[0] == 0]

            write_yolo_labels(lbl_out / (img_path.stem + ".txt"), grape_labels)
            grape_count += len(grape_labels)
            img_count += 1

        print(f"  {split_key}: {img_count} images, {grape_count} grape instances")

    # 生成 data.yaml
    new_yaml = {
        "train": str((output_dir / "train" / "images").resolve()),
        "val": str((output_dir / "val" / "images").resolve()),
        "test": str((output_dir / "test" / "images").resolve()),
        "nc": 1,
        "names": ["grapes"],
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(new_yaml, f, default_flow_style=False)

    print(f"Grapes dataset saved to {output_dir}")


def resolve_path(p: str, base_dir: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base_dir / path).resolve()


if __name__ == "__main__":
    main()
