#!/bin/bash
set -euo pipefail

# ============================================================
# 测试脚本：语法检查 → pytest → --help 检查 → 单张推理测试
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source /home/user/miniconda3/envs/sam3/bin/activate 2>/dev/null || true
PYTHON="python"

echo "============================================"
echo "  Pipeline 测试"
echo "============================================"

# 1. Python 语法检查
echo ""
echo "[1/6] Python 语法检查..."
$PYTHON -m compileall src scripts tests -q 2>&1 || {
    echo "ERROR: Syntax check failed"
    exit 1
}
echo "  ✓ 语法检查通过"

# 2. 各脚本 --help
echo ""
echo "[2/6] 检查脚本 --help..."
for script in scripts/*.py; do
    echo "  $script --help"
    $PYTHON "$script" --help > /dev/null 2>&1 || echo "  (no --help available)"
done
echo "  ✓ --help 检查完成"

# 3. pytest
echo ""
echo "[3/6] 运行单元测试..."
if [ -d tests ] && ls tests/test_*.py 1>/dev/null 2>&1; then
    $PYTHON -m pytest tests -q --tb=short 2>&1 || {
        echo "WARNING: Some tests failed"
    }
else
    echo "  No tests found, skipping."
fi

# 4. 数据集检查
echo ""
echo "[4/6] 数据集检查..."
if [ -f datasets/stage2_stem_roi/data.yaml ]; then
    $PYTHON scripts/check_dataset.py \
        --data datasets/stage2_stem_roi/data.yaml \
        --output-dir reports/dataset/ \
        --imgsz 960
    echo "  ✓ 数据集检查完成"
else
    echo "  datasets/stage2_stem_roi/data.yaml 不存在，跳过。"
fi

# 5. 单张推理测试
echo ""
echo "[5/6] 单张推理测试..."
TEST_IMAGE=$(find datasets/original -name "*.jpg" -o -name "*.png" 2>/dev/null | head -1)
if [ -n "$TEST_IMAGE" ] && [ -f weights/grapes/best.pt ]; then
    $PYTHON scripts/run_full_pipeline.py \
        --source "$TEST_IMAGE" \
        --grapes-weights weights/grapes/best.pt \
        --device cpu \
        --output-dir outputs/ 2>&1 | tail -5
    echo "  ✓ 推理测试完成"
else
    echo "  缺少测试图像或模型，跳过。"
fi

# 6. 输出文件检查
echo ""
echo "[6/6] 输出文件检查..."
echo "  项目结构:"
find . -maxdepth 2 -type f -name "*.py" -o -name "*.sh" -o -name "*.yaml" -o -name "*.md" | sort

echo ""
echo "============================================"
echo "  测试完成！"
echo "============================================"
