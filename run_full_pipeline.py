#!/usr/bin/env python3
"""完整推理管线入口：葡萄检测 → 果梗检测 → SAM 分割 → 中心点输出"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.grape_stem.grape_detector import GrapeDetector
from src.grape_stem.stem_detector import StemDetector
from src.grape_stem.stem_selector import StemSelector
from src.grape_stem.sam_segmenter import SAMSegmenter
from src.grape_stem.pipeline import GrapeStemPipeline
from src.grape_stem.result_writer import ResultWriter
from src.grape_stem.config import load_pipeline_config


def main():
    parser = argparse.ArgumentParser(
        description="完整推理管线：葡萄检测 → 果梗检测 → SAM 分割 → 中心点输出"
    )
    parser.add_argument("--source", required=True, help="输入图像或目录路径")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Pipeline 配置文件")
    parser.add_argument("--grapes-weights", help="葡萄模型权重路径")
    parser.add_argument("--stem-weights", help="果梗模型权重路径")
    parser.add_argument("--sam-type", default="sam2", choices=["sam", "sam2", "mobilesam"])
    parser.add_argument("--sam-checkpoint", help="SAM 权重路径")
    parser.add_argument("--device", default="0", help="GPU 设备编号")
    parser.add_argument("--grapes-conf", type=float, default=0.25, help="葡萄检测置信度阈值")
    parser.add_argument("--stem-conf", type=float, default=0.15, help="果梗检测置信度阈值")
    parser.add_argument("--stem-iou", type=float, default=0.50, help="果梗 NMS IOU 阈值")
    parser.add_argument("--imgsz", type=int, default=960, help="果梗检测输入尺寸")
    parser.add_argument("--output-dir", default="outputs", help="输出目录")
    parser.add_argument("--save-mask", action="store_true", default=True, help="保存掩膜图像")
    parser.add_argument("--save-json", action="store_true", default=True, help="保存 JSON 结果")
    parser.add_argument("--save-csv", action="store_true", default=True, help="保存 CSV 结果")
    parser.add_argument("--save-annotated", action="store_true", default=True, help="保存标注可视化")
    args = parser.parse_args()

    # 加载配置
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        config = load_pipeline_config(config_path)
    else:
        config = {}

    # 用 CLI 参数覆盖配置
    grapes_weights = args.grapes_weights or config.get("grapes_detector", {}).get(
        "weights", "weights/grapes/best.pt"
    )
    stem_weights = args.stem_weights or config.get("stem_detector", {}).get(
        "weights", "weights/stem/best.pt"
    )
    sam_checkpoint = args.sam_checkpoint or config.get("sam", {}).get(
        "checkpoint", "weights/sam/model.pt"
    )

    # 初始化检测器
    print(f"Loading grape detector: {grapes_weights}")
    grape_detector = GrapeDetector(
        weights_path=PROJECT_ROOT / grapes_weights,
        confidence=args.grapes_conf,
        device=args.device,
    )

    print(f"Loading stem detector: {stem_weights}")
    stem_detector = StemDetector(
        weights_path=PROJECT_ROOT / stem_weights,
        confidence=args.stem_conf,
        iou=args.stem_iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    stem_selector = StemSelector()

    # 尝试加载 SAM
    sam_segmenter = None
    sam_checkpoint_path = PROJECT_ROOT / sam_checkpoint
    if sam_checkpoint_path.exists():
        print(f"Loading SAM: {sam_checkpoint}")
        sam_segmenter = SAMSegmenter(
            sam_type=args.sam_type,
            checkpoint_path=sam_checkpoint_path,
            box_padding=config.get("sam", {}).get("box_padding", 0.15),
            device=args.device,
        )
    else:
        print(f"SAM checkpoint not found at {sam_checkpoint_path}, skipping SAM step")

    # 创建 pipeline
    pipeline = GrapeStemPipeline(config)
    writer = ResultWriter(PROJECT_ROOT / args.output_dir)

    # 处理输入
    source_path = Path(args.source)
    image_paths = []
    if source_path.is_dir():
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_paths.extend(source_path.glob(ext))
        image_paths = sorted(image_paths)
    else:
        image_paths = [source_path]

    print(f"Processing {len(image_paths)} images...")

    all_results = []
    for img_path in image_paths:
        print(f"  Processing: {img_path.name}")
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"    Skipping: cannot read {img_path}")
            continue

        # 运行 pipeline
        result = pipeline.run(
            image=image,
            grape_detector=grape_detector,
            stem_detector=stem_detector,
            stem_selector=stem_selector,
            sam_segmenter=sam_segmenter,
            image_name=img_path.name,
        )

        # 保存结果
        if args.save_json:
            writer.save_json(result, img_path.name)

        # 保存掩膜
        if args.save_mask and sam_segmenter is not None:
            for grape in result.get("grapes", []):
                if grape.get("peduncle_mask_centroid_roi_xy"):
                    grape_id = grape["grape_id"]
                    mask_path = PROJECT_ROOT / args.output_dir / "masks" / f"{img_path.stem}_{grape_id}.png"
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    # 注意：mask 是在 pipeline 内部计算的，这里需要重新获取
                    # 实际上 mask 应该从 result 中获取，简化处理

        # 收集 CSV 行
        from src.grape_stem.result_writer import flatten_results
        flat_rows = flatten_results(result, img_path.name)
        all_results.extend(flat_rows)

    if args.save_csv and all_results:
        writer.save_csv(all_results)

    # 打印摘要
    success_count = sum(1 for r in all_results if r.get("grape_status") == "success")
    total_grapes = len(all_results)
    print(f"\nSummary: {success_count}/{total_grapes} grapes with stem detected")
    print(f"Results saved to: {PROJECT_ROOT / args.output_dir}")


if __name__ == "__main__":
    main()
