#!/usr/bin/env python3
"""UNet 果梗三点推理 worker — GPU (torch) 版, 替代 stem_keypoint_infer.py。

与原 TF 版协议完全一致（stdin/stdout JSON-lines），仅模型加载/推理替换为
ONNX→onnx2torch→CUDA，用于解决 TensorFlow 纯 CPU 推理太慢的问题。

依赖: torch (CUDA) / onnx / onnx2torch / opencv / scikit-image / numpy
（运行于 sam3 环境）

用法:
    /home/user/miniconda3/envs/sam3/bin/python3 stem_keypoint_infer_torch.py \
        --model weights/unet_mobilenetv2.onnx [--threshold 0.93] ...
"""

import os
import sys

# 本文件位于 grape_stem_3d 包目录内, 该目录下的 types.py 会遮蔽 stdlib
# `types` 模块, 导致 py3.10 启动时 enum→types 循环导入崩溃。
# 必须在任何其他 import 之前移除脚本目录的 sys.path 优先级
# (worker 不依赖包内模块)。
if sys.path and sys.path[0] == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)

import argparse
import heapq
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

MODEL_SIZE = 224
NEIGHBORS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def load_model(model_path):
    """加载 ONNX 并转换为 torch 模块 (CUDA)。"""
    import onnx
    import onnx2torch
    import torch

    onnx_model = onnx.load(str(model_path))
    module = onnx2torch.convert(onnx_model)
    if torch.cuda.is_available():
        module = module.cuda()
    module.eval()
    return module


def torch_predict(model, image_bgr_224):
    """BGR uint8 (224,224,3) → (224,224) float32 分数图。"""
    import torch

    x = image_bgr_224.astype(np.float32) / 255.0
    tensor = torch.from_numpy(x).unsqueeze(0)  # (1,224,224,3) NHWC, 与 ONNX 输入一致
    if next(model.parameters()).is_cuda:
        tensor = tensor.cuda()
    with torch.no_grad():
        out = model(tensor)
    return out.detach().cpu().numpy().squeeze().astype(np.float32)


# ── 以下骨架图与最长路径逻辑与原 TF 版一致 ──────────────
def build_skeleton_graph(skeleton):
    pixels = [tuple(map(int, point)) for point in np.argwhere(skeleton)]
    pixel_set = set(pixels)
    graph = {}

    for y, x in pixels:
        neighbors = []
        for dy, dx in NEIGHBORS_8:
            neighbor = (y + dy, x + dx)
            if neighbor in pixel_set:
                weight = math.sqrt(2.0) if dy != 0 and dx != 0 else 1.0
                neighbors.append((neighbor, weight))
        graph[(y, x)] = sorted(neighbors)

    return graph


def dijkstra(graph, start):
    distances = {start: 0.0}
    previous = {}
    queue = [(0.0, start)]

    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances.get(node, float("inf")):
            continue

        for neighbor, weight in graph[node]:
            candidate = distance + weight
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))

    return distances, previous


def farthest_reachable(distances, candidates):
    reachable = [node for node in candidates if node in distances]
    if not reachable:
        return None
    return max(reachable, key=lambda node: (distances[node], node))


def reconstruct_path(previous, start, end):
    path = [end]
    current = end

    while current != start:
        if current not in previous:
            return []
        current = previous[current]
        path.append(current)

    path.reverse()
    return path


def longest_skeleton_path(skeleton):
    graph = build_skeleton_graph(skeleton)
    if len(graph) < 2:
        return [], 0.0

    nodes = sorted(graph)
    endpoints = sorted(node for node in nodes if len(graph[node]) == 1)
    targets = endpoints if len(endpoints) >= 2 else nodes
    initial = targets[0]

    first_distances, _ = dijkstra(graph, initial)
    first_end = farthest_reachable(first_distances, targets)
    if first_end is None:
        return [], 0.0

    second_distances, previous = dijkstra(graph, first_end)
    second_end = farthest_reachable(second_distances, targets)
    if second_end is None or second_end == first_end:
        return [], 0.0

    path = reconstruct_path(previous, first_end, second_end)
    return path, float(second_distances[second_end])


def select_candidate_path(binary_mask, min_component_area):
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )

    best_path = []
    best_length = 0.0

    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue

        component = labels == component_index
        skeleton = skeletonize(component)
        path, length = longest_skeleton_path(skeleton)

        if length > best_length:
            best_path = path
            best_length = length

    return best_path, best_length


def cumulative_path_lengths(path):
    cumulative = [0.0]
    for index in range(1, len(path)):
        y0, x0 = path[index - 1]
        y1, x1 = path[index]
        cumulative.append(cumulative[-1] + math.hypot(y1 - y0, x1 - x0))
    return np.asarray(cumulative, dtype=np.float32)


def nearest_path_point(path, cumulative, target_distance):
    index = int(np.argmin(np.abs(cumulative - target_distance)))
    return path[index]


def select_three_points(path, pose_offset):
    cumulative = cumulative_path_lengths(path)
    total_length = float(cumulative[-1])
    midpoint_distance = total_length / 2.0
    effective_offset = min(float(pose_offset), total_length / 4.0)

    point_1 = nearest_path_point(path, cumulative, midpoint_distance - effective_offset)
    cut_point = nearest_path_point(path, cumulative, midpoint_distance)
    point_3 = nearest_path_point(path, cumulative, midpoint_distance + effective_offset)

    if len({point_1, cut_point, point_3}) < 3:
        return None

    if point_1 > point_3:
        point_1, point_3 = point_3, point_1

    return point_1, cut_point, point_3


def to_original_xy(point_yx, original_width, original_height):
    y, x = point_yx
    x_scale = (original_width - 1) / float(MODEL_SIZE - 1)
    y_scale = (original_height - 1) / float(MODEL_SIZE - 1)
    original_x = int(round(x * x_scale))
    original_y = int(round(y * y_scale))
    original_x = int(np.clip(original_x, 0, original_width - 1))
    original_y = int(np.clip(original_y, 0, original_height - 1))
    return original_x, original_y


def infer_image(model, image_path, threshold, min_component_area,
                min_path_length, pose_offset):
    """推理单张裁剪图, 返回三点结果 dict / None / {"error": ...}。"""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "unreadable image"}

    height, width = image.shape[:2]
    resized = cv2.resize(image, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)

    raw_map = torch_predict(model, resized)
    if raw_map.shape != (MODEL_SIZE, MODEL_SIZE):
        return {"error": f"unexpected output shape {raw_map.shape}"}

    raw_map = np.nan_to_num(raw_map, nan=0.0, posinf=1.0, neginf=0.0)
    raw_score = np.clip(raw_map, 0.0, 1.0)

    binary_mask = raw_score > threshold
    candidate_path, path_length = select_candidate_path(binary_mask, min_component_area)

    if not candidate_path:
        return {"error": "no connected skeleton"}
    if path_length < min_path_length:
        return {"error": f"path too short ({path_length:.1f})"}

    three_points = select_three_points(candidate_path, pose_offset)
    if three_points is None:
        return {"error": "cannot separate three points"}

    point_1, cut_point, point_3 = three_points
    p1_xy = to_original_xy(point_1, width, height)
    cut_xy = to_original_xy(cut_point, width, height)
    p3_xy = to_original_xy(point_3, width, height)

    return {
        "point_1": [int(p1_xy[0]), int(p1_xy[1])],
        "cut_point": [int(cut_xy[0]), int(cut_xy[1])],
        "point_3": [int(p3_xy[0]), int(p3_xy[1])],
        "path_length_224": round(path_length, 3),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="UNet grape-stem keypoint worker (torch/GPU)")
    parser.add_argument("--model", required=True, help="UNet ONNX path")
    parser.add_argument("--threshold", type=float, default=0.93)
    parser.add_argument("--pose-offset", type=float, default=12.0)
    parser.add_argument("--min-component-area", type=int, default=12)
    parser.add_argument("--min-path-length", type=float, default=20.0)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        print(json.dumps({"fatal": f"model not found: {model_path}"}), flush=True)
        return 2

    started = time.perf_counter()
    model = load_model(model_path)
    load_s = time.perf_counter() - started

    # 预热一次, 确保 cuBLAS 上下文就绪后再报 READY
    dummy = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
    torch_predict(model, dummy)
    warm_s = time.perf_counter() - started
    print(f"READY (model {model_path.name} loaded in {load_s:.1f}s, "
          f"warmed in {warm_s:.1f}s, threshold > {args.threshold:.3f})", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"error": "invalid json line"}), flush=True)
            continue

        job_id = job.get("job_id")
        paths = job.get("paths") or []
        results = {}
        for path in paths:
            try:
                results[path] = infer_image(
                    model, path, args.threshold, args.min_component_area,
                    args.min_path_length, args.pose_offset,
                )
            except Exception as e:  # 单张失败不影响批内其他图
                results[path] = {"error": f"{type(e).__name__}: {e}"}

        print(json.dumps({"job_id": job_id, "results": results}), flush=True)


if __name__ == "__main__":
    main()
