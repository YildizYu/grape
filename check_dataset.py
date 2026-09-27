#!/usr/bin/env python3
"""数据集质量检查脚本

检查内容:
  1. 图像存在且可读 (非空文件, 可解码)
  2. 标注文件存在 (缺失标为 warn); 空标注文件视为负样本, 允许
  3. 类别 ID 合法 (0 <= cls < nc)
  4. 归一化坐标在 [0, 1] 内; 无 NaN; 宽高 > 0
  5. ROI 非空: 有内容的标注文件必须至少解析出 1 个有效框
  6. 果梗未被错误裁剪: 框贴合图像边缘 (border_touch) 记为警告
  7. 无跨划分重复图像 (按文件内容 MD5)
  8. 无重复果梗框 (文件内 / 跨文件完全相同的框)

统计输出:
  - 各类别实例数, 正/负样本比
  - 缩放到 --imgsz 后的果梗像素尺寸分布: % <8px / <16px / <32px
  - 报告文件: {output_dir}/stem_roi_dataset_report.json 和 .csv

退出码: 0=通过  1=存在失败级问题  2=data.yaml 无法读取/无效
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EPS = 1e-6          # 坐标范围检查容差
BORDER_EPS = 1e-4   # 贴边检测容差 (归一化坐标)
SPLIT_KEYS = ("train", "val", "test")


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_split_dirs(split_path: Path) -> Tuple[Path, Path]:
    """把 data.yaml 中的 split 路径解析为 (images_dir, labels_dir)。"""
    p = split_path
    if (p / "images").is_dir():
        return p / "images", p / "labels"
    if p.name == "images":
        return p, p.parent / "labels"
    if p.name == "labels":
        return p.parent / "images", p
    return p, p.parent / "labels"


def md5_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_label_line(line: str) -> Optional[List[float]]:
    """解析一行 YOLO 标签; 非法返回 None。"""
    parts = line.split()
    if not parts:
        return None
    if len(parts) != 5:
        return None
    try:
        return [float(v) for v in parts]
    except ValueError:
        return None


def check_label_file(label_path: Path, split: str, image_name: str,
                     nc: int, names: List[str],
                     issues: List[Issue]) -> Tuple[int, Dict[str, int],
                                                   List[Tuple[float, ...]]]:
    """检查单个标注文件, 返回 (有效框数, 类别计数, 有效框列表)。"""
    n_valid = 0
    class_counts: Dict[str, int] = {}
    valid_boxes: List[Tuple[float, ...]] = []
    has_content = False

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                has_content = True
            vals = parse_label_line(line)
            if vals is None:
                if line.strip():
                    issues.append(Issue(split, image_name, label_path.name,
                                        "malformed_line", line.strip()[:80], "fail"))
                continue

            cls = int(vals[0])
            cx, cy, w, h = vals[1], vals[2], vals[3], vals[4]
            label = label_path.name

            # 类别 ID
            if cls < 0 or cls >= nc:
                issues.append(Issue(split, image_name, label, "invalid_class",
                                    f"cls={cls} (nc={nc})", "fail"))
                continue
            # NaN / 坐标范围 / 宽高
            if any(np.isnan(v) for v in vals[1:]):
                issues.append(Issue(split, image_name, label, "nan_value",
                                    str(vals), "fail"))
                continue
            if not (-EPS <= cx <= 1 + EPS and -EPS <= cy <= 1 + EPS
                    and -EPS <= w <= 1 + EPS and -EPS <= h <= 1 + EPS):
                issues.append(Issue(split, image_name, label, "out_of_range_coord",
                                    f"cx={cx} cy={cy} w={w} h={h}", "fail"))
                continue
            if w <= 0.0 or h <= 0.0:
                issues.append(Issue(split, image_name, label,
                                    "nonpositive_dimension",
                                    f"w={w} h={h}", "fail"))
                continue

            # 果梗被裁剪: 框贴合图像边缘 (归一化坐标下)
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            if (x1 <= BORDER_EPS or x2 >= 1 - BORDER_EPS
                    or y1 <= BORDER_EPS or y2 >= 1 - BORDER_EPS):
                name = names[cls] if cls < len(names) else f"class{cls}"
                issues.append(Issue(split, image_name, label, "border_touch",
                                    f"{name}: x1={x1:.4f} y1={y1:.4f} "
                                    f"x2={x2:.4f} y2={y2:.4f} 可能被裁剪", "warn"))

            n_valid += 1
            cls_name = names[cls] if cls < len(names) else f"class{cls}"
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            valid_boxes.append((cls, cx, cy, w, h))

    # ROI 非空: 有内容但没有一个有效框
    if has_content and n_valid == 0:
        issues.append(Issue(split, image_name, label_path.name, "no_valid_boxes",
                            "标注文件有内容但无有效框", "fail"))
    return n_valid, class_counts, valid_boxes


@dataclass
class Issue:
    """一条检查问题记录。"""
    split: str
    image: str
    label: str
    issue_type: str
    detail: str = ""
    severity: str = "warn"   # "fail" | "warn"


@dataclass
class SplitStats:
    name: str
    images_dir: Path
    labels_dir: Path
    n_images: int = 0
    n_label_files: int = 0
    n_missing_labels: int = 0
    n_positive_images: int = 0      # 有 >=1 有效框
    n_negative_images: int = 0      # 标注文件存在且为空
    n_boxes: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dataset quality checker for stem ROI dataset")
    parser.add_argument("--data", default="datasets/stage2_stem_roi/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--output-dir", default="reports/dataset",
                        help="Where to save reports")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="Target image size for stem pixel-size estimation")
    args = parser.parse_args()

    project_root = get_project_root()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (project_root / data_path).resolve()
    if not data_path.is_file():
        print(f"[错误] data.yaml 不存在: {data_path}", file=sys.stderr)
        return 2

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[错误] 无法解析 data.yaml: {exc}", file=sys.stderr)
        return 2
    if not isinstance(cfg, dict):
        print("[错误] data.yaml 内容无效 (应为 YAML 字典)", file=sys.stderr)
        return 2

    nc = cfg.get("nc")
    names: List[str] = cfg.get("names") or []
    if not isinstance(nc, int) or nc <= 0:
        print(f"[错误] data.yaml 中 nc 无效: {nc}", file=sys.stderr)
        return 2
    if not isinstance(names, list) or len(names) != nc:
        print(f"[错误] data.yaml 中 names 无效 (应为 {nc} 项): {names}",
              file=sys.stderr)
        return 2

    # ── 遍历各划分 ───────────────────────────────────────────────
    issues: List[Issue] = []
    split_stats: List[SplitStats] = []
    seen_boxes: Dict[Tuple[float, ...], str] = {}   # 跨文件重复框 (split/图片定位)
    dup_in_file: Dict[Tuple[float, ...], str] = {}  # 文件内重复框

    for split_key in SPLIT_KEYS:
        raw = cfg.get(split_key)
        if not raw:
            continue
        split_path = Path(str(raw))
        if not split_path.is_absolute():
            split_path = (data_path.parent / split_path).resolve()
        images_dir, labels_dir = resolve_split_dirs(split_path)
        stats = SplitStats(split_key, images_dir, labels_dir)
        split_stats.append(stats)

        if not images_dir.is_dir():
            issues.append(Issue(split_key, "-", "-", "missing_images_dir",
                                f"目录不存在: {images_dir}", "fail"))
            continue

        images = sorted(p for p in images_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        stats.n_images = len(images)
        if not images:
            issues.append(Issue(split_key, "-", "-", "empty_images_dir",
                                f"图像目录为空: {images_dir}", "warn"))

        labels_ok = labels_dir.is_dir()
        if not labels_ok:
            issues.append(Issue(split_key, "-", "-", "missing_labels_dir",
                                f"标签目录不存在: {labels_dir}", "fail"))
        label_files = {p.stem: p for p in labels_dir.glob("*.txt")} if labels_ok else {}
        stats.n_label_files = len(label_files)

        img_md5s: Dict[str, str] = {}   # md5 -> "split/image"

        for img in images:
            image_name = img.name
            size = img.stat().st_size
            if size == 0:
                issues.append(Issue(split_key, image_name, "-", "empty_image_file",
                                    "图像文件为 0 字节", "fail"))
                continue
            img_arr = cv2.imread(str(img))
            if img_arr is None:
                issues.append(Issue(split_key, image_name, "-", "unreadable_image",
                                    "无法解码图像", "fail"))
                continue

            # 跨划分重复图像 (内容 MD5)
            md5 = md5_of_file(img)
            prev = img_md5s.get(md5)
            loc = f"{split_key}/{image_name}"
            if prev is not None:
                issues.append(Issue(split_key, image_name, "-",
                                    "duplicate_image_across_splits",
                                    f"与 {prev} 内容完全相同 (md5={md5[:12]})", "fail"))
            else:
                img_md5s[md5] = loc

            # 标注文件
            lbl = label_files.get(img.stem)
            if lbl is None:
                stats.n_missing_labels += 1
                issues.append(Issue(split_key, image_name, "-", "missing_label",
                                    "图像缺少标注文件 (若为负样本请提供空 .txt)", "warn"))
                continue

            n_valid, class_counts, valid_boxes = check_label_file(
                lbl, split_key, image_name, nc, names, issues)
            if n_valid > 0:
                stats.n_positive_images += 1
            else:
                stats.n_negative_images += 1
            stats.n_boxes += n_valid
            for k, v in class_counts.items():
                stats.class_counts[k] = stats.class_counts.get(k, 0) + v

            # 重复框: 文件内 / 跨文件 (完全相同的归一化框)
            for b in valid_boxes:
                box = (round(float(b[1]), 9), round(float(b[2]), 9),
                       round(float(b[3]), 9), round(float(b[4]), 9))
                if dup_in_file.get(box) == image_name:
                    issues.append(Issue(split_key, image_name, lbl.name,
                                        "duplicate_box_in_file",
                                        f"同一文件内重复框 {box}", "warn"))
                dup_in_file[box] = image_name
                if box in seen_boxes:
                    issues.append(Issue(split_key, image_name, lbl.name,
                                        "duplicate_box_across_files",
                                        f"框 {box} 与 {seen_boxes[box]} 完全相同", "warn"))
                else:
                    seen_boxes[box] = loc

        # 孤儿标签 (标签文件没有对应图像)
        if labels_ok:
            img_stems = {p.stem for p in images}
            for lbl in label_files.values():
                if lbl.stem not in img_stems:
                    issues.append(Issue(split_key, "-", lbl.name, "orphan_label",
                                        "标签文件没有对应图像", "warn"))

    # ── 统计汇总 ─────────────────────────────────────────────────
    total_images = sum(s.n_images for s in split_stats)
    total_boxes = sum(s.n_boxes for s in split_stats)
    total_pos = sum(s.n_positive_images for s in split_stats)
    total_neg = sum(s.n_negative_images for s in split_stats)
    total_class_counts: Dict[str, int] = {}
    for s in split_stats:
        for k, v in s.class_counts.items():
            total_class_counts[k] = total_class_counts.get(k, 0) + v

    # ── 果梗像素尺寸估计 (缩放到 --imgsz) ───────────────────────
    stem_cls_name = "stem" if "stem" in names else None
    min_dims: List[float] = []
    widths: List[float] = []
    areas: List[float] = []
    n_stem_boxes = 0
    for s in split_stats:
        if not s.labels_dir.is_dir():
            continue
        for lbl in sorted(s.labels_dir.glob("*.txt")):
            if lbl.stat().st_size == 0:
                continue
            with open(lbl, "r", encoding="utf-8") as f:
                for line in f:
                    vals = parse_label_line(line)
                    if vals is None:
                        continue
                    cls = int(vals[0])
                    if stem_cls_name is not None and (cls >= len(names)
                                                      or names[cls] != stem_cls_name):
                        continue
                    _, _, _, w, h = vals
                    if w <= 0 or h <= 0:
                        continue
                    pw, ph = w * args.imgsz, h * args.imgsz
                    n_stem_boxes += 1
                    min_dims.append(min(pw, ph))
                    widths.append(pw)
                    areas.append(pw * ph)

    size_summary: Dict[str, Any] = {
        "scaled_imgsz": args.imgsz,
        "n_stem_boxes": n_stem_boxes,
        "pct_min_dim_lt8px": (round(100 * sum(1 for d in min_dims if d < 8)
                                    / n_stem_boxes, 2) if n_stem_boxes else None),
        "pct_min_dim_lt16px": (round(100 * sum(1 for d in min_dims if d < 16)
                                     / n_stem_boxes, 2) if n_stem_boxes else None),
        "pct_min_dim_lt32px": (round(100 * sum(1 for d in min_dims if d < 32)
                                     / n_stem_boxes, 2) if n_stem_boxes else None),
        "pct_width_lt8px": (round(100 * sum(1 for w in widths if w < 8)
                                  / n_stem_boxes, 2) if n_stem_boxes else None),
        "pct_width_lt16px": (round(100 * sum(1 for w in widths if w < 16)
                                   / n_stem_boxes, 2) if n_stem_boxes else None),
        "pct_width_lt32px": (round(100 * sum(1 for w in widths if w < 32)
                                   / n_stem_boxes, 2) if n_stem_boxes else None),
        "mean_min_dim_px": (round(float(np.mean(min_dims)), 2) if n_stem_boxes else None),
        "median_min_dim_px": (round(float(np.median(min_dims)), 2) if n_stem_boxes else None),
        "mean_area_px2": (round(float(np.mean(areas)), 1) if n_stem_boxes else None),
    }

    # ── 检查结果汇总 ─────────────────────────────────────────────
    n_fail = sum(1 for i in issues if i.severity == "fail")
    n_warn = sum(1 for i in issues if i.severity == "warn")
    overall = "fail" if n_fail else ("warn" if n_warn else "pass")

    report: Dict[str, Any] = {
        "general": {
            "data_yaml": str(data_path),
            "nc": nc,
            "names": names,
            "imgsz": args.imgsz,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "summary": {
            "overall_status": overall,
            "n_fail_issues": n_fail,
            "n_warn_issues": n_warn,
            "total_images": total_images,
            "total_boxes": total_boxes,
            "total_positive_images": total_pos,
            "total_negative_images": total_neg,
            "positive_negative_ratio": (round(total_pos / total_neg, 3)
                                        if total_neg else None),
            "negative_ratio": (round(total_neg / total_images, 3)
                               if total_images else None),
            "class_counts": total_class_counts,
        },
        "splits": [
            {
                "split": s.name,
                "images_dir": str(s.images_dir),
                "labels_dir": str(s.labels_dir) if s.labels_dir.is_dir() else None,
                "n_images": s.n_images,
                "n_label_files": s.n_label_files,
                "n_missing_labels": s.n_missing_labels,
                "n_positive_images": s.n_positive_images,
                "n_negative_images": s.n_negative_images,
                "n_boxes": s.n_boxes,
                "class_counts": s.class_counts,
            }
            for s in split_stats
        ],
        "size_estimate": size_summary,
        "checks": [
            {"name": "images_exist_and_readable",
             "status": ("pass" if not any(i.issue_type in ("empty_image_file",
                                                           "unreadable_image")
                                          for i in issues) else "fail"),
             "detail": "所有图像文件非空且可解码"},
            {"name": "labels_exist",
             "status": ("warn" if any(i.issue_type == "missing_label"
                                      for i in issues) else "pass"),
             "detail": "缺失标注文件已记录; 空标注文件作为负样本允许"},
            {"name": "class_ids_valid",
             "status": ("pass" if not any(i.issue_type == "invalid_class"
                                          for i in issues) else "fail"),
             "detail": f"类别 ID 必须在 [0, {nc}) 内"},
            {"name": "coords_in_unit_range",
             "status": ("pass" if not any(i.issue_type == "out_of_range_coord"
                                          for i in issues) else "fail"),
             "detail": "归一化坐标应在 [0, 1] 内"},
            {"name": "no_nan_and_positive_dims",
             "status": ("pass" if not any(i.issue_type in ("nan_value",
                                                           "nonpositive_dimension")
                                          for i in issues) else "fail"),
             "detail": "无 NaN, 宽高 > 0"},
            {"name": "roi_non_empty",
             "status": ("pass" if not any(i.issue_type == "no_valid_boxes"
                                          for i in issues) else "fail"),
             "detail": "有内容的标注文件必须包含至少 1 个有效框"},
            {"name": "stems_not_wrongly_cropped",
             "status": ("warn" if any(i.issue_type == "border_touch"
                                      for i in issues) else "pass"),
             "detail": "贴合边缘的框可能被 ROI 裁剪截断"},
            {"name": "no_duplicate_images_across_splits",
             "status": ("pass" if not any(i.issue_type == "duplicate_image_across_splits"
                                          for i in issues) else "fail"),
             "detail": "按 MD5 检测内容重复的图像"},
            {"name": "no_duplicate_stem_assignments",
             "status": ("warn" if any(i.issue_type.startswith("duplicate_box")
                                      for i in issues) else "pass"),
             "detail": "文件内/跨文件完全相同的框; 跨裁剪块重复的果梗无法仅凭标签检测"},
        ],
        "issues": [
            {"split": i.split, "image": i.image, "label": i.label,
             "issue": i.issue_type, "severity": i.severity, "detail": i.detail}
            for i in issues
        ],
    }

    # ── 输出报告 ─────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stem_roi_dataset_report.json"
    csv_path = output_dir / "stem_roi_dataset_report.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "key", "value"])
        for sec in ("general", "summary", "size_estimate"):
            for k, v in report[sec].items():
                writer.writerow([sec, k, v if not isinstance(v, (dict, list))
                                 else json.dumps(v, ensure_ascii=False)])
        writer.writerow([])
        writer.writerow(["issue", "split", "image", "label", "severity", "detail"])
        for i in issues:
            writer.writerow([i.issue_type, i.split, i.image, i.label,
                             i.severity, i.detail])

    # ── 终端输出 ─────────────────────────────────────────────────
    print(f"\n数据集检查完成: {data_path}")
    print(f"  nc={nc} names={names}")
    print(f"  图像总数={total_images}  框总数={total_boxes}  类别分布={total_class_counts}")
    print(f"  正样本图像={total_pos}  负样本图像={total_neg}  "
          f"正:负={report['summary']['positive_negative_ratio']}")
    if n_stem_boxes:
        print(f"  果梗像素尺寸 (@{args.imgsz}px): "
              f"min边 <8px={size_summary['pct_min_dim_lt8px']}%  "
              f"<16px={size_summary['pct_min_dim_lt16px']}%  "
              f"<32px={size_summary['pct_min_dim_lt32px']}%")
    print(f"  检查结果: {overall}  (fail={n_fail} warn={n_warn})")
    print(f"  报告: {json_path}")
    print(f"        {csv_path}")
    if n_fail:
        print("\n[失败级问题]")
        for i in issues:
            if i.severity == "fail":
                print(f"  - [{i.split}] {i.image}: {i.issue_type} - {i.detail}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
