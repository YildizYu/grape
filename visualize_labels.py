#!/usr/bin/env python3
"""可视化标签：生成 QA 图片"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.grape_stem.dataset_utils import read_yolo_labels, yolo_to_xyxy
from src.grape_stem.visualization import (
    COLOR_GRAPE, COLOR_STEM, COLOR_ROI, COLOR_CENTROID,
    draw_bbox, visualize_grape_stem, visualize_roi, save_qa_image,
)


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO labels")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--output-dir", default="reports/qa_images")
    parser.add_argument("--num-train", type=int, default=100)
    parser.add_argument("--num-val", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    data_path = Path(args.data)
    base_dir = data_path.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_path) as f:
        data_cfg = yaml.safe_load(f)

    is_roi_dataset = data_cfg.get("nc") == 1 and data_cfg.get("names") == ["stem"]

    for split_key in ["train", "val", "test"]:
        if split_key not in data_cfg:
            continue

        num_samples = args.num_train if split_key == "train" else args.num_val
        split_path = resolve_path(data_cfg[split_key], base_dir)
        label_dir = split_path.parent / "labels" if (split_path.parent / "labels").exists() else split_path / "labels"

        if not label_dir.exists():
            label_dir = split_path.parent.parent / "labels"

        images = sorted([p for p in split_path.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

        if not images:
            print(f"  {split_key}: no images found")
            continue

        if len(images) > num_samples:
            images = random.sample(images, num_samples)

        print(f"  {split_key}: generating {len(images)} QA images...")

        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            h, w = image.shape[:2]
            img_name = img_path.stem

            # 读取标签
            lbl_path = None
            for ext in [".txt"]:
                candidate = label_dir / (img_name + ext)
                if candidate.exists():
                    lbl_path = candidate
                    break

            if lbl_path is None:
                continue

            labels = read_yolo_labels(lbl_path)

            # 绘制
            grape_bboxes = []
            stem_bboxes = []
            for lbl in labels:
                cls_id, xc, yc, bw, bh = lbl
                xyxy = yolo_to_xyxy(xc, yc, bw, bh, w, h)
                if cls_id == 0:
                    grape_bboxes.append(xyxy)
                elif cls_id == 1:
                    stem_bboxes.append(xyxy)

            annotated = visualize_grape_stem(image, grape_bboxes, stem_bboxes)
            save_qa_image(annotated, output_dir / f"{split_key}_{img_name}_labels.jpg")


def resolve_path(p: str, base_dir: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base_dir / path).resolve()


if __name__ == "__main__":
    main()
