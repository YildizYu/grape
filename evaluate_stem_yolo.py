#!/usr/bin/env python3
"""YOLO11 果梗检测模型评估脚本

两种 ROI 来源模式:
  --roi-source ground_truth : 使用人工标注的葡萄框 (GT) 作为 ROI
  --roi-source predicted    : 使用葡萄检测模型的预测框作为 ROI

流程:
  1. 加载果梗 YOLO 模型
  2. 对每张测试图像: 获取葡萄框 → 扩展 ROI → 裁剪 → 果梗检测
  3. 与 GT 果梗比对, 计算 Precision / Recall / F1 / mAP@50 / mAP@50-95 / FP / FN
  4. 置信度阈值扫描 [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
     找出 最佳F1 / 召回优先 / 精度优先 三个阈值
  5. 失败样本 (FN / FP / 低置信度 TP) 保存到 reports/failures/
  6. 输出评估报告 JSON 与 CSV 到 reports/evaluation/

用法示例:
  python scripts/evaluate_stem_yolo.py --roi-source ground_truth
  python scripts/evaluate_stem_yolo.py --roi-source predicted \
      --grapes-weights weights/grapes/best.pt

默认评估数据: datasets/normalized/data.yaml (nc=2: grapes, stem,
全图坐标, 同时包含 GT 葡萄框与 GT 果梗框)。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CONF_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
AP_IOU_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
# 与 configs/pipeline.yaml 中 roi 配置保持一致
DEFAULT_ROI_EXPAND = {"left": 0.15, "right": 0.15, "top": 0.50, "bottom": 0.10}

# 失败样本绘制颜色 (BGR)
COLOR_GT = (0, 0, 255)        # 红色: GT 果梗
COLOR_FP = (0, 255, 255)      # 黄色: 误检
COLOR_LOW = (0, 165, 255)     # 橙色: 低置信度命中
COLOR_GRAPE = (255, 0, 0)     # 蓝色: 葡萄 ROI 框


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ─────────────────────────── 数据结构 ───────────────────────────

@dataclass
class Box:
    """像素坐标系下的轴对齐框 (x1,y1 左上, x2,y2 右下)。"""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def iou(self, other: "Box") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0

    def to_int(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)),
                int(round(self.x2)), int(round(self.y2)))


@dataclass
class GtStem:
    """一个 GT 果梗, 已指派到某个 crop (crop_id), 或未指派。"""
    img: str
    box: Box
    crop_id: Optional[int] = None
    matched: bool = field(default=False)


@dataclass
class CropInfo:
    """一张图像内的一个 ROI 裁剪块。"""
    img: str
    crop_id: int
    roi: Box            # 裁剪块在整图中的像素框
    grape: Box          # 对应的葡萄框 (整图坐标)


@dataclass
class Detection:
    """一个果梗检测结果 (整图坐标系)。"""
    img: str
    crop_id: int
    box: Box
    conf: float


@dataclass
class ImageRecord:
    """一张测试图像的全部评估中间信息 (用于失败样本可视化)。"""
    name: str
    path: Path
    grapes: List[Box]
    crops: List[CropInfo]
    gt_stems: List[GtStem]


# ─────────────────────────── 数据加载 ───────────────────────────

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_split_dirs(split_path: Path) -> Tuple[Path, Path]:
    p = split_path
    if (p / "images").is_dir():
        return p / "images", p / "labels"
    if p.name == "images":
        return p, p.parent / "labels"
    if p.name == "labels":
        return p.parent / "images", p
    return p, p.parent / "labels"


def parse_labels(path: Path) -> List[Tuple[int, float, float, float, float]]:
    """解析 YOLO 格式标签, 返回 [(cls, cx, cy, w, h), ...]; 无效行跳过。"""
    out: List[Tuple[int, float, float, float, float]] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                vals = [float(v) for v in parts]
                if len(vals) != 5:
                    continue
                cls = int(vals[0])
            except ValueError:
                continue
            out.append((cls, vals[1], vals[2], vals[3], vals[4]))
    return out


def norm_to_pixel(cx: float, cy: float, w: float, h: float,
                  img_w: int, img_h: int) -> Box:
    x1 = (cx - w / 2.0) * img_w
    x2 = (cx + w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    y2 = (cy + h / 2.0) * img_h
    return Box(x1, y1, x2, y2)


# ─────────────────────────── ROI 与匹配 ───────────────────────────

def build_roi(grape: Box, img_w: int, img_h: int,
              expand: Dict[str, float]) -> Optional[Box]:
    """按葡萄框尺寸的比例向外扩展得到 ROI, 并裁剪到图像边界。"""
    w, h = grape.w, grape.h
    if w <= 0 or h <= 0:
        return None
    x1 = grape.x1 - expand["left"] * w
    x2 = grape.x2 + expand["right"] * w
    y1 = grape.y1 - expand["top"] * h
    y2 = grape.y2 + expand["bottom"] * h
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(img_w), x2)
    y2 = min(float(img_h), y2)
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return Box(x1, y1, x2, y2)


def assign_gt_stems(gt_stems: Sequence[GtStem], crops: Sequence[CropInfo]) -> int:
    """把 GT 果梗指派到包含其中心的 ROI (多个时取 IoU 最大者);
    无 ROI 包含中心时, 回退到 IoU 最大的 ROI。返回未指派数量。"""
    n_unassigned = 0
    for stem in gt_stems:
        cx, cy = stem.box.center
        candidates = [c for c in crops if c.roi.x1 <= cx <= c.roi.x2
                      and c.roi.y1 <= cy <= c.roi.y2]
        if not candidates:
            candidates = sorted(crops, key=lambda c: c.roi.iou(stem.box),
                                reverse=True)
            if not candidates or candidates[0].roi.iou(stem.box) <= 0.0:
                stem.crop_id = None
                n_unassigned += 1
                continue
        best = max(candidates, key=lambda c: c.roi.iou(stem.box))
        stem.crop_id = best.crop_id
    return n_unassigned


def _greedy_match(dets_sorted: Sequence[Detection],
                  gt_by_key: Dict[Tuple[str, int], List[GtStem]],
                  iou_thresh: float) -> List[bool]:
    """对已按置信度降序排列的检测做贪心 IoU 匹配 (每个 crop 内部独立匹配)。
    命中时设置对应 GtStem.matched=True; 返回与输入顺序一致的 TP 标志列表。
    调用前需先重置全部 g.matched=False。"""
    tp_flags: List[bool] = []
    for d in dets_sorted:
        gts = gt_by_key.get((d.img, d.crop_id), [])
        best_i = -1
        best_iou = iou_thresh
        for i, g in enumerate(gts):
            if g.matched:
                continue
            iou = d.box.iou(g.box)
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if best_i >= 0:
            gts[best_i].matched = True
            tp_flags.append(True)
        else:
            tp_flags.append(False)
    return tp_flags


def match_detections(dets: Sequence[Detection],
                     gt_by_key: Dict[Tuple[str, int], List[GtStem]],
                     conf_thresh: float, iou_thresh: float,
                     n_gt: int) -> Tuple[int, int, int]:
    """按置信度降序做贪心 IoU 匹配 (每个 crop 内部匹配)。
    返回 (tp, fp, fn)。fn 基于全部 GT 果梗数 n_gt (含未指派, 未指派恒为 FN)。"""
    sorted_dets = sorted((d for d in dets if d.conf >= conf_thresh),
                         key=lambda d: -d.conf)
    tp = sum(_greedy_match(sorted_dets, gt_by_key, iou_thresh))
    fp = len(sorted_dets) - tp
    fn = n_gt - tp
    return tp, fp, fn


def compute_ap(dets: Sequence[Detection],
               gt_by_key: Dict[Tuple[str, int], List[GtStem]],
               iou_thresh: float, n_gt: int) -> Optional[float]:
    """对单个 IoU 阈值计算 AP (PR 曲线下面积, 连续积分)。
    n_gt 为全部 GT 果梗数 (含未指派), 未指派者永远无法被匹配。"""
    if n_gt == 0:
        return None
    if not dets:
        return 0.0
    sorted_dets = sorted(dets, key=lambda d: -d.conf)
    tp_flags = _greedy_match(sorted_dets, gt_by_key, iou_thresh)

    tp = np.cumsum(np.array([1.0 if f else 0.0 for f in tp_flags]))
    fp = np.cumsum(np.array([0.0 if f else 1.0 for f in tp_flags]))
    rec = tp / float(n_gt)
    prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(mpre.size - 2, -1, -1):  # 右侧最大包络, 单调化
        mpre[i] = max(mpre[i], mpre[i + 1])
    # continuous 积分 (与 ultralytics metrics 的 continuous 模式一致):
    # 沿 recall 变化的折点处求 PR 曲线下面积, 无端点插值偏差
    i = np.where(mrec[1:] != mrec[:-1])[0]
    if i.size == 0:
        return 0.0
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def prf_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """由 tp/fp/fn 计算 precision/recall/f1 (空集合时取平凡值)。"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4)}


def pick_thresholds(sweep_rows: Sequence[Dict[str, Any]]) -> Tuple[Dict, Dict, Dict]:
    """从扫描表选出: 最佳F1 / 召回优先 / 精度优先 三个阈值。
    召回/精度优先定义为: 在 F1 >= 0.95*最佳F1 的候选里取最高召回/最高精度。"""
    best = max(sweep_rows, key=lambda r: (r["f1"], r["precision"]))
    if best["f1"] <= 0.0:
        rec_prior = max(sweep_rows, key=lambda r: (r["recall"], r["precision"]))
        prec_prior = max(sweep_rows, key=lambda r: (r["precision"], r["recall"]))
        return best, rec_prior, prec_prior
    eligible = [r for r in sweep_rows if r["f1"] >= 0.95 * best["f1"]]
    rec_prior = max(eligible, key=lambda r: (r["recall"], r["precision"]))
    prec_prior = max(eligible, key=lambda r: (r["precision"], r["recall"]))
    return best, rec_prior, prec_prior


# ─────────────────────────── 失败样本保存 ───────────────────────────

def _draw_box(img: np.ndarray, box: Box, color: Tuple[int, int, int],
              label: Optional[str] = None, thickness: int = 2) -> None:
    x1, y1, x2, y2 = box.to_int()
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, max(10, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def save_failure_crops(records: Sequence[ImageRecord], mode: str,
                       gt_by_key: Dict[Tuple[str, int], List[GtStem]],
                       dets: Sequence[Detection], best_thresh: float,
                       failures_dir: Path, max_per_category: int,
                       match_iou: float = 0.5) -> Dict[str, int]:
    """在 best-F1 阈值下保存三类失败样本: FN / FP / 低置信度 TP。"""
    # 重置 matched 状态后做一次贪心匹配, 得到每个检测/GT 的命中情况
    for gts in gt_by_key.values():
        for g in gts:
            g.matched = False
    sorted_dets = sorted((d for d in dets if d.conf >= best_thresh),
                         key=lambda d: -d.conf)
    tp_flags = _greedy_match(sorted_dets, gt_by_key, match_iou)
    tp_ids = {id(d) for d, f in zip(sorted_dets, tp_flags) if f}

    fn_dir = failures_dir / "false_negative"
    fp_dir = failures_dir / "false_positive"
    low_dir = failures_dir / "low_confidence"
    for d in (fn_dir, fp_dir, low_dir):
        d.mkdir(parents=True, exist_ok=True)

    counts = {"false_negative": 0, "false_positive": 0, "low_confidence": 0}
    index: List[Dict[str, Any]] = []
    image_cache: Dict[str, np.ndarray] = {}

    det_by_key: Dict[Tuple[str, int], List[Detection]] = {}
    for d in dets:
        det_by_key.setdefault((d.img, d.crop_id), []).append(d)

    def get_image(img_name: str) -> Optional[np.ndarray]:
        if img_name not in image_cache:
            for rec in records:
                if rec.name == img_name:
                    img = cv2.imread(str(rec.path))
                    image_cache[img_name] = img
                    break
            else:
                image_cache[img_name] = None
        return image_cache[img_name]

    for rec in records:
        for crop in rec.crops:
            key = (rec.name, crop.crop_id)
            gts = gt_by_key.get(key, [])
            preds = det_by_key.get(key, [])
            if not gts and not preds:
                continue
            img = get_image(rec.name)
            if img is None:
                continue
            x1, y1, x2, y2 = crop.roi.to_int()
            crop_img = img[y1:y2, x1:x2].copy()
            to_crop = lambda b: Box(b.x1 - x1, b.y1 - y1, b.x2 - x1, b.y2 - y1)

            fn_boxes = [g for g in gts if not g.matched]       # FN: 未命中 GT
            for g in fn_boxes:
                _draw_box(crop_img, to_crop(g.box), COLOR_GT, "GT")
            if fn_boxes and counts["false_negative"] < max_per_category:
                out = fn_dir / f"fn_{mode}_{rec.name}_{crop.crop_id}.png"
                cv2.imwrite(str(out), crop_img)
                counts["false_negative"] += 1
                index.append({"category": "false_negative", "file": out.name,
                              "image": rec.name, "crop": crop.crop_id})

            fp_boxes = [d for d in preds
                        if d.conf >= best_thresh and id(d) not in tp_ids]
            if fp_boxes and counts["false_positive"] < max_per_category:
                for d in fp_boxes:
                    _draw_box(crop_img, to_crop(d.box), COLOR_FP, f"{d.conf:.2f}")
                out = fp_dir / f"fp_{mode}_{rec.name}_{crop.crop_id}.png"
                cv2.imwrite(str(out), crop_img)
                counts["false_positive"] += 1
                index.append({"category": "false_positive", "file": out.name,
                              "image": rec.name, "crop": crop.crop_id})

            low_boxes = [d for d in preds
                         if d.conf < best_thresh and id(d) in tp_ids]
            if low_boxes and counts["low_confidence"] < max_per_category:
                for d in low_boxes:
                    _draw_box(crop_img, to_crop(d.box), COLOR_LOW, f"{d.conf:.2f}")
                out = low_dir / f"low_{mode}_{rec.name}_{crop.crop_id}.png"
                cv2.imwrite(str(out), crop_img)
                counts["low_confidence"] += 1
                index.append({"category": "low_confidence", "file": out.name,
                              "image": rec.name, "crop": crop.crop_id})

    if index:
        with open(failures_dir / f"failures_index_{mode}.json", "w",
                  encoding="utf-8") as f:
            json.dump(index, f, indent=2)
    return counts


# ─────────────────────────── 报告输出 ───────────────────────────

def write_report(report: Dict[str, Any], output_dir: Path,
                 mode: str) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"stem_eval_{mode}.json"
    csv_path = output_dir / f"stem_eval_{mode}.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "tp", "fp", "fn", "precision", "recall", "f1"])
        for row in report["sweep"]:
            writer.writerow([row["threshold"], row["tp"], row["fp"], row["fn"],
                             row["precision"], row["recall"], row["f1"]])
        writer.writerow([])
        writer.writerow(["metric", "value"])
        for key, value in report["summary"].items():
            if isinstance(value, dict):
                for k2, v2 in value.items():
                    writer.writerow([f"summary.{key}.{k2}", v2])
            else:
                writer.writerow([f"summary.{key}", value])
    return json_path, csv_path


# ─────────────────────────── 主流程 ───────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO11 stem detection with ROI crops")
    parser.add_argument("--data", default="datasets/normalized/data.yaml",
                        help="数据集 data.yaml (默认 normalized: grapes+stem 全图标注)")
    parser.add_argument("--split", choices=["auto", "test", "val"], default="auto",
                        help="评估用划分 (auto: 优先 test, 没有则用 val)")
    parser.add_argument("--roi-source", choices=["ground_truth", "predicted"],
                        required=True, help="葡萄框 ROI 来源")
    parser.add_argument("--stem-weights", default="weights/stem/best.pt",
                        help="果梗检测模型权重")
    parser.add_argument("--grapes-weights", default="weights/grapes/best.pt",
                        help="葡萄检测模型权重 (predicted 模式必需)")
    parser.add_argument("--imgsz", type=int, default=960, help="果梗模型输入尺寸")
    parser.add_argument("--infer-conf", type=float, default=0.05,
                        help="果梗推理置信度下限 (需 <= 扫描最小阈值)")
    parser.add_argument("--infer-iou", type=float, default=0.5, help="果梗推理 NMS IoU")
    parser.add_argument("--match-iou", type=float, default=0.5,
                        help="P/R/F1 扫描使用的匹配 IoU")
    parser.add_argument("--grape-conf", type=float, default=0.25,
                        help="葡萄模型置信度 (predicted 模式)")
    parser.add_argument("--grape-iou", type=float, default=0.50,
                        help="葡萄模型 NMS IoU (predicted 模式)")
    parser.add_argument("--grape-imgsz", type=int, default=640,
                        help="葡萄模型输入尺寸 (predicted 模式)")
    parser.add_argument("--roi-expand-left", type=float, default=DEFAULT_ROI_EXPAND["left"])
    parser.add_argument("--roi-expand-right", type=float, default=DEFAULT_ROI_EXPAND["right"])
    parser.add_argument("--roi-expand-top", type=float, default=DEFAULT_ROI_EXPAND["top"])
    parser.add_argument("--roi-expand-bottom", type=float, default=DEFAULT_ROI_EXPAND["bottom"])
    parser.add_argument("--min-roi", type=int, default=32,
                        help="ROI 最小边长 (像素), 更小则跳过该葡萄")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", default="reports/evaluation",
                        help="评估报告输出目录")
    parser.add_argument("--failures-dir", default="reports/failures",
                        help="失败样本输出目录")
    parser.add_argument("--max-failures", type=int, default=50,
                        help="每类失败样本最多保存数量")
    parser.add_argument("--limit", type=int, default=None,
                        help="仅处理前 N 张图像 (调试用)")
    args = parser.parse_args()

    project_root = get_project_root()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (project_root / data_path).resolve()
    if not data_path.exists():
        print(f"[错误] data.yaml 不存在: {data_path}", file=sys.stderr)
        return 2
    cfg = load_yaml(data_path)
    names: List[str] = cfg.get("names") or ["grape", "stem", "picking"]
    stem_cls = names.index("stem") if "stem" in names else 1
    grape_cls = names.index("grape") if "grape" in names else 0

    # 选择评估划分
    split_key = args.split
    if split_key == "auto":
        split_key = "test" if cfg.get("test") else "val"
    raw_split = cfg.get(split_key)
    if not raw_split:
        print(f"[错误] data.yaml 缺少 '{split_key}' 划分", file=sys.stderr)
        return 2
    split_path = Path(str(raw_split))
    if not split_path.is_absolute():
        split_path = (data_path.parent / split_path).resolve()
    images_dir, labels_dir = resolve_split_dirs(split_path)
    if not images_dir.is_dir():
        print(f"[错误] 找不到测试图像目录: {images_dir}", file=sys.stderr)
        return 2
    images = sorted(p for p in images_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        images = images[:args.limit]
    if not images:
        print(f"[错误] 测试图像目录为空: {images_dir}", file=sys.stderr)
        return 2

    # 加载果梗模型
    stem_weights = Path(args.stem_weights)
    if not stem_weights.is_absolute():
        stem_weights = (project_root / stem_weights).resolve()
    if not stem_weights.exists():
        print(f"[错误] 果梗模型权重不存在: {stem_weights}\n"
              "请先运行 scripts/train_stem.sh 训练果梗模型", file=sys.stderr)
        return 2
    try:
        stem_model = YOLO(str(stem_weights))
    except Exception as exc:
        print(f"[错误] 无法加载果梗模型 {stem_weights}: {exc}", file=sys.stderr)
        return 2

    grape_model: Optional[YOLO] = None
    resolved_grapes_weights: Optional[Path] = None
    if args.roi_source == "predicted":
        grape_weights = Path(args.grapes_weights)
        if not grape_weights.is_absolute():
            grape_weights = (project_root / grape_weights).resolve()
        resolved_grapes_weights = grape_weights
        if not grape_weights.exists():
            print(f"[错误] 葡萄模型权重不存在: {grape_weights}", file=sys.stderr)
            return 2
        try:
            grape_model = YOLO(str(grape_weights))
        except Exception as exc:
            print(f"[错误] 无法加载葡萄模型 {grape_weights}: {exc}", file=sys.stderr)
            return 2

    expand = {"left": args.roi_expand_left, "right": args.roi_expand_right,
              "top": args.roi_expand_top, "bottom": args.roi_expand_bottom}

    # ── 逐图处理 ───────────────────────────────────────────────
    gt_by_key: Dict[Tuple[str, int], List[GtStem]] = {}
    dets: List[Detection] = []
    records: List[ImageRecord] = []
    all_gt_stems: List[GtStem] = []
    n_gt_grapes, n_pred_grapes, n_crops, n_skipped = 0, 0, 0, 0
    n_missing_labels = 0

    for idx, img_path in enumerate(images):
        img_name = img_path.name
        if (idx + 1) % 25 == 0 or idx == len(images) - 1:
            print(f"  处理中 {idx + 1}/{len(images)}  ({img_name})", flush=True)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [警告] 无法读取图像, 跳过: {img_path}", file=sys.stderr)
            continue
        img_h, img_w = img.shape[:2]

        # GT 标签
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            n_missing_labels += 1
            label_path = None
        gt_rows = parse_labels(label_path) if label_path else []
        gt_stems_img = [GtStem(img_name, norm_to_pixel(cx, cy, w, h, img_w, img_h))
                        for cls, cx, cy, w, h in gt_rows if cls == stem_cls]
        gt_grapes = [norm_to_pixel(cx, cy, w, h, img_w, img_h)
                     for cls, cx, cy, w, h in gt_rows if cls == grape_cls]
        all_gt_stems.extend(gt_stems_img)
        n_gt_grapes += len(gt_grapes)

        # 葡萄框来源
        if args.roi_source == "ground_truth":
            grapes = gt_grapes
        else:
            assert grape_model is not None
            res = grape_model.predict(
                img, imgsz=args.grape_imgsz, conf=args.grape_conf,
                iou=args.grape_iou, device=args.device, verbose=False)[0]
            grapes = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    box = Box(max(0.0, float(x1)), max(0.0, float(y1)),
                              min(float(img_w), float(x2)), min(float(img_h), float(y2)))
                    if box.w > 0 and box.h > 0:
                        grapes.append(box)
            n_pred_grapes += len(grapes)

        # 构建 ROI 裁剪块
        crops: List[CropInfo] = []
        for gi, g in enumerate(grapes):
            roi = build_roi(g, img_w, img_h, expand)
            if roi is None or roi.w < args.min_roi or roi.h < args.min_roi:
                n_skipped += 1
                continue
            crops.append(CropInfo(img_name, len(crops), roi, g))
        n_crops += len(crops)

        # 指派 GT 果梗 (未指派者计入 FN, 由 n_gt_total 口径统一统计)
        assign_gt_stems(gt_stems_img, crops)
        for stem in gt_stems_img:
            if stem.crop_id is not None:
                gt_by_key.setdefault((img_name, stem.crop_id), []).append(stem)

        # 果梗检测 (批量推理该图所有裁剪块)
        if crops:
            crop_arrays = [img[int(c.roi.y1):int(c.roi.y2),
                               int(c.roi.x1):int(c.roi.x2)] for c in crops]
            try:
                results = stem_model.predict(
                    crop_arrays, imgsz=args.imgsz, conf=args.infer_conf,
                    iou=args.infer_iou, device=args.device, verbose=False)
            except Exception as exc:
                print(f"[警告] 果梗推理失败 {img_name}: {exc}", file=sys.stderr)
                results = [None] * len(crops)
            for ci, res in enumerate(results):
                if res is None or res.boxes is None or len(res.boxes) == 0:
                    continue
                ox, oy = float(crops[ci].roi.x1), float(crops[ci].roi.y1)
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    dets.append(Detection(
                        img_name, crops[ci].crop_id,
                        Box(float(x1) + ox, float(y1) + oy,
                            float(x2) + ox, float(y2) + oy),
                        float(confs[i])))

        records.append(ImageRecord(img_name, img_path, grapes, crops, gt_stems_img))

    n_gt_total = len(all_gt_stems)
    print(f"\n统计: images={len(images)} gt_stems={n_gt_total} "
          f"gt_grapes={n_gt_grapes} pred_grapes={n_pred_grapes} "
          f"crops={n_crops} skipped_roi={n_skipped} "
          f"missing_labels={n_missing_labels}")

    # ── 阈值扫描 ───────────────────────────────────────────────
    sweep_rows: List[Dict[str, Any]] = []
    for t in CONF_SWEEP:
        for gts in gt_by_key.values():
            for g in gts:
                g.matched = False
        tp, fp, fn = match_detections(dets, gt_by_key, t, args.match_iou,
                                      n_gt_total)
        row = prf_metrics(tp, fp, fn)
        row["threshold"] = t
        sweep_rows.append(row)

    # ── mAP ─────────────────────────────────────────────────────
    ap_by_iou: Dict[str, Optional[float]] = {}
    for iou_t in AP_IOU_THRESHOLDS:
        for gts in gt_by_key.values():
            for g in gts:
                g.matched = False
        ap_by_iou[f"{iou_t:.2f}"] = compute_ap(dets, gt_by_key, iou_t,
                                               n_gt_total)
    ap_values = [v for v in ap_by_iou.values() if v is not None]
    map50 = ap_by_iou["0.50"]
    map50_95 = round(float(np.mean(ap_values)), 4) if ap_values else None

    best, rec_prior, prec_prior = pick_thresholds(sweep_rows)

    # ── 失败样本 (best-F1 阈值) ─────────────────────────────────
    failures_dir = Path(args.failures_dir)
    if not failures_dir.is_absolute():
        failures_dir = project_root / failures_dir
    fail_counts = save_failure_crops(
        records, args.roi_source, gt_by_key, dets, best["threshold"],
        failures_dir, args.max_failures, args.match_iou)

    # ── 报告 ────────────────────────────────────────────────────
    report: Dict[str, Any] = {
        "metadata": {
            "roi_source": args.roi_source,
            "data": str(data_path),
            "split": split_key,
            "stem_weights": str(stem_weights),
            "grapes_weights": (str(resolved_grapes_weights)
                               if resolved_grapes_weights else None),
            "imgsz": args.imgsz,
            "match_iou": args.match_iou,
            "roi_expand": expand,
            "min_roi": args.min_roi,
            "infer_conf": args.infer_conf,
            "conf_sweep": CONF_SWEEP,
        },
        "summary": {
            "total_images": len(images),
            "total_gt_stems": n_gt_total,
            "total_gt_grapes": n_gt_grapes,
            "total_pred_grapes": n_pred_grapes,
            "total_crops": n_crops,
            "skipped_roi_crops": n_skipped,
            "missing_gt_labels": n_missing_labels,
            "mAP50": map50,
            "mAP50_95": map50_95,
            "best_f1_threshold": best,
            "recall_priority_threshold": rec_prior,
            "precision_priority_threshold": prec_prior,
            "failures": fail_counts,
        },
        "ap_iou_breakdown": ap_by_iou,
        "sweep": sweep_rows,
    }
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    json_path, csv_path = write_report(report, output_dir, args.roi_source)

    # ── 打印摘要 ────────────────────────────────────────────────
    print("\n================ 评估结果 ================")
    print(f"模式: {args.roi_source}   图像: {len(images)}   GT 果梗: {n_gt_total}")
    print(f"mAP@50 = {map50}   mAP@50-95 = {map50_95}")
    print(f"最佳F1阈值:  conf={best['threshold']}  "
          f"P={best['precision']} R={best['recall']} F1={best['f1']} "
          f"(TP={best['tp']} FP={best['fp']} FN={best['fn']})")
    print(f"召回优先:    conf={rec_prior['threshold']}  "
          f"P={rec_prior['precision']} R={rec_prior['recall']} F1={rec_prior['f1']}")
    print(f"精度优先:    conf={prec_prior['threshold']}  "
          f"P={prec_prior['precision']} R={prec_prior['recall']} F1={prec_prior['f1']}")
    print(f"失败样本: {fail_counts}")
    print(f"报告: {json_path}")
    print(f"      {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
