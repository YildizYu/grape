#!/usr/bin/env python3
"""数据集划分脚本：按 train/val/test 比例随机划分"""

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_file_hash(filepath: Path) -> str:
    """计算文件 SHA256 哈希。"""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Split YOLO dataset")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--output-dir", default="datasets/split")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-regex", help="Regex to group images (e.g. for video frames)")
    args = parser.parse_args()

    random.seed(args.seed)

    # 验证比例
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    # 读取原始数据
    data_path = Path(args.data)
    with open(data_path) as f:
        data_cfg = yaml.safe_load(f)

    base_dir = data_path.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有图像
    all_images = []
    for split_key in ["train", "val", "test"]:
        if split_key not in data_cfg:
            continue
        split_dir = resolve_path(data_cfg[split_key], base_dir)
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist, skipping")
            continue
        for img_path in split_dir.glob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                all_images.append(img_path)

    all_images = sorted(set(all_images))
    print(f"Found {len(all_images)} total images")

    # 随机打乱
    random.shuffle(all_images)

    # 划分
    n_train = int(len(all_images) * args.train_ratio)
    n_val = int(len(all_images) * args.val_ratio)
    n_test = len(all_images) - n_train - n_val

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:],
    }

    # 检查重复（按哈希）
    print("Checking for duplicate images by hash...")
    hash_to_paths: Dict[str, List[Path]] = {}
    for img_path in all_images:
        h = get_file_hash(img_path)
        hash_to_paths.setdefault(h, []).append(img_path)
    duplicates = {h: ps for h, ps in hash_to_paths.items() if len(ps) > 1}
    if duplicates:
        print(f"WARNING: Found {len(duplicates)} duplicate hashes!")

    # 复制到输出目录
    for split_name, images in splits.items():
        img_out = output_dir / split_name / "images"
        lbl_out = output_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            # 复制图像
            shutil.copy2(img_path, img_out / img_path.name)

            # 复制标签
            for lbl_ext in [".txt"]:
                lbl_path = img_path.parent.parent / "labels" / (img_path.stem + lbl_ext)
                if lbl_path.exists():
                    shutil.copy2(lbl_path, lbl_out / lbl_path.name)
                break

        print(f"  {split_name}: {len(images)} images")

    # 生成 data.yaml
    new_yaml = {
        "train": str((output_dir / "train" / "images").resolve()),
        "val": str((output_dir / "val" / "images").resolve()),
        "test": str((output_dir / "test" / "images").resolve()),
        "nc": data_cfg.get("nc", 2),
        "names": data_cfg.get("names", ["grapes", "stem"]),
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(new_yaml, f, default_flow_style=False)

    # 保存划分报告
    report = {
        "total_images": len(all_images),
        "train_count": n_train,
        "val_count": n_val,
        "test_count": n_test,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "duplicate_hashes": len(duplicates),
    }
    report_path = output_dir / "split_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Split complete. Data saved to {output_dir}")


def resolve_path(p: str, base_dir: Path) -> Path:
    """Resolve a path relative to base_dir."""
    path = Path(p)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


if __name__ == "__main__":
    main()
