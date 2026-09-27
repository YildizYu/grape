#!/usr/bin/env python3
"""YOLO11 果梗检测模型训练脚本

用法示例:
    python scripts/train_stem_yolo.py --data datasets/stage2_stem_roi/data.yaml
    python scripts/train_stem_yolo.py --data datasets/stage2_stem_roi/data.yaml --resume
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import yaml
from ultralytics import YOLO

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_split_dirs(split_path: Path) -> Tuple[Path, Path]:
    """把 data.yaml 中的 split 路径解析为 (images_dir, labels_dir)。

    兼容两种目录约定:
      - split 路径指向 split 目录本身 (含 images/ 和 labels/ 子目录)
      - split 路径直接指向 images 目录 (标签在兄弟目录 labels/)
    其它情况把 split 路径本身当作 images 目录, 标签目录取父目录下的 labels/。
    """
    p = split_path
    if (p / "images").is_dir():
        return p / "images", p / "labels"
    if p.name == "images":
        return p, p.parent / "labels"
    if p.name == "labels":
        return p.parent / "images", p
    return p, p.parent / "labels"


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 for stem detection")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--translate", type=float, default=0.08)
    parser.add_argument("--scale", type=float, default=0.30)
    parser.add_argument("--mosaic", type=float, default=0.20)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/train_stem")
    parser.add_argument("--name", default="stem_detect")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    # ── 1. 校验数据集 ─────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    if not isinstance(data_cfg, dict):
        raise ValueError(f"data.yaml 内容无效 (应为 YAML 字典): {data_path}")
    assert data_cfg["nc"] == 1, f"Expected nc=1, got {data_cfg['nc']}"
    assert data_cfg["names"] == ["stem"], f"Expected ['stem'], got {data_cfg['names']}"

    # 检查 train/val 目录存在且非空 (路径兼容相对与绝对写法)
    split_info: Dict[str, Dict] = {}
    for split_key in ("train", "val"):
        raw_path = data_cfg.get(split_key)
        if not raw_path:
            raise ValueError(f"data.yaml 缺少 '{split_key}' 键: {data_path}")
        split_path = Path(str(raw_path))
        if not split_path.is_absolute():
            split_path = (data_path.parent / split_path).resolve()
        images_dir, labels_dir = resolve_split_dirs(split_path)
        if not images_dir.is_dir():
            raise FileNotFoundError(f"找不到 {split_key} 图像目录: {images_dir}")
        images = sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise ValueError(f"{split_key} 图像目录为空: {images_dir}")

        n_labels, n_missing = 0, 0
        if labels_dir.is_dir():
            for img in images:
                if (labels_dir / (img.stem + ".txt")).exists():
                    n_labels += 1
                else:
                    n_missing += 1
        else:
            n_missing = len(images)

        split_info[split_key] = {
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir) if labels_dir.is_dir() else None,
            "n_images": len(images),
            "n_labels": n_labels,
            "n_missing_labels": n_missing,
        }
        print(f"  [{split_key}] images={len(images)}  labels={n_labels}  "
              f"missing_labels={n_missing}")
        if n_missing and split_key == "train":
            print(f"  [警告] train 中有 {n_missing} 张图像缺少标注文件, "
                  "会被当作负样本处理")

    # ── 2. 打印全部参数 ───────────────────────────────────────────
    print(f"Training config: {json.dumps(vars(args), indent=2, default=str)}")

    # ── 3. 保存参数快照 ───────────────────────────────────────────
    snapshot_path = Path(args.project) / args.name / "train_config.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = dict(vars(args))
    snapshot["timestamp"] = datetime.now().isoformat(timespec="seconds")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"Config snapshot saved to {snapshot_path}")

    # ── 4. 训练 ───────────────────────────────────────────────────
    try:
        model = YOLO(args.model)
    except Exception as exc:  # 模型文件缺失 / 下载失败 / 权重损坏
        raise RuntimeError(f"无法加载模型 {args.model}: {exc}") from exc

    print(f"\n开始训练: {args.model} on {data_path.resolve()}")
    try:
        results = model.train(
            data=str(data_path.resolve()),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0=args.lr0,
            weight_decay=args.weight_decay,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            mosaic=args.mosaic,
            mixup=args.mixup,
            close_mosaic=args.close_mosaic,
            seed=args.seed,
            project=args.project,
            name=args.name,
            resume=args.resume,
            cache=args.cache,
            exist_ok=True,
        )
    except Exception as exc:
        raise RuntimeError(f"训练失败: {exc}") from exc

    # ── 5. 复制 best / last 权重到统一目录 ───────────────────────
    best_src = Path(args.project) / args.name / "weights" / "best.pt"
    last_src = Path(args.project) / args.name / "weights" / "last.pt"
    weights_dir = get_project_root() / "weights" / "stem"
    weights_dir.mkdir(parents=True, exist_ok=True)
    if best_src.exists():
        shutil.copy2(best_src, weights_dir / "best.pt")
        print(f"  copied best.pt  -> {weights_dir / 'best.pt'}")
    else:
        print(f"  [警告] 未找到 best.pt: {best_src}")
    if last_src.exists():
        shutil.copy2(last_src, weights_dir / "last.pt")
        print(f"  copied last.pt  -> {weights_dir / 'last.pt'}")

    print(f"Training complete. Weights saved to {weights_dir}")


if __name__ == "__main__":
    main()
