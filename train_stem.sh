#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source /home/user/miniconda3/envs/sam3/bin/activate 2>/dev/null || true

echo "============================================"
echo "  果梗 YOLO11 模型训练"
echo "============================================"

python scripts/train_stem_yolo.py \
  --data datasets/stage2_stem_roi/data.yaml \
  --model yolo11s.pt \
  --epochs 200 \
  --batch 8 \
  --imgsz 960 \
  --device 0 \
  --workers 8 \
  --patience 50 \
  --optimizer AdamW \
  --lr0 0.001 \
  --weight-decay 0.0005 \
  --project runs/train_stem \
  --name yolo11s_roi_960
