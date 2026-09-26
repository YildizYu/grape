#!/bin/bash
set -euo pipefail

# ============================================================
# 数据集准备脚本
# 依次执行：检查原始数据 → 类别标准化 → 划分 → 葡萄数据集 → 果梗ROI → 检查 → 可视化
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source /home/user/miniconda3/envs/sam3/bin/activate 2>/dev/null || true

echo "============================================"
echo "  数据集准备 Pipeline"
echo "============================================"

# Step 1: 检查原始数据
echo ""
echo "[1/7] 检查原始数据..."
python scripts/inspect_dataset.py \
    --data datasets/original/data.yaml \
    --output-dir reports/dataset/

# Step 2: 类别标准化
echo ""
echo "[2/7] 类别标准化..."
python scripts/normalize_dataset_classes.py \
    --data datasets/original/data.yaml \
    --aliases configs/class_aliases.yaml \
    --output-dir datasets/normalized/ \
    --unknown-policy error

# Step 3: 数据集划分（如果尚未划分）
echo ""
echo "[3/7] 检查/执行数据集划分..."
if [ -f datasets/split/data.yaml ]; then
    echo "  已存在划分，跳过。"
else
    python scripts/split_dataset.py \
        --data datasets/normalized/data.yaml \
        --output-dir datasets/split/ \
        --train-ratio 0.70 \
        --val-ratio 0.20 \
        --test-ratio 0.10 \
        --seed 42
fi

# Step 4: 生成葡萄数据集
echo ""
echo "[4/7] 生成葡萄数据集..."
python scripts/build_grapes_dataset.py \
    --data datasets/normalized/data.yaml \
    --output-dir datasets/stage1_grapes/

# Step 5: 生成果梗 ROI 数据集
echo ""
echo "[5/7] 生成果梗 ROI 数据集..."
python scripts/build_stem_roi_dataset.py \
    --data datasets/normalized/data.yaml \
    --output-dir datasets/stage2_stem_roi/ \
    --config configs/pipeline.yaml

# Step 6: 数据集质量检查
echo ""
echo "[6/7] 数据集质量检查..."
python scripts/check_dataset.py \
    --data datasets/stage2_stem_roi/data.yaml \
    --imgsz 960 \
    --output-dir reports/dataset/

# Step 7: 标签可视化
echo ""
echo "[7/7] 生成 QA 图片..."
python scripts/visualize_labels.py \
    --data datasets/stage2_stem_roi/data.yaml \
    --output-dir reports/qa_images/ \
    --num-train 100 \
    --num-val 50

echo ""
echo "============================================"
echo "  数据集准备完成！"
echo "============================================"
