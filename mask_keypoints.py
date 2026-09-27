"""Shared mask-to-keypoint post-processing for SAM3 and UNet.

Both segmentation backends must call this module so a comparison changes only
the mask-producing model. Coordinates returned by :func:`mask_to_keypoints`
are local to the source ROI supplied with the mask.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


MODEL_SIZE = 224
NEIGHBORS_8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def build_skeleton_graph(skeleton: np.ndarray) -> Dict[Tuple[int, int], list]:
    pixels = [tuple(map(int, point)) for point in np.argwhere(skeleton)]
    pixel_set = set(pixels)
    graph = {}
    for y, x in pixels:
        neighbors = []
        for dy, dx in NEIGHBORS_8:
            neighbor = (y + dy, x + dx)
            if neighbor in pixel_set:
                weight = math.sqrt(2.0) if dy and dx else 1.0
                neighbors.append((neighbor, weight))
        graph[(y, x)] = sorted(neighbors)
    return graph


def _dijkstra(graph, start):
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


def _farthest_reachable(distances, candidates):
    reachable = [node for node in candidates if node in distances]
    if not reachable:
        return None
    return max(reachable, key=lambda node: (distances[node], node))


def _reconstruct_path(previous, start, end):
    path = [end]
    current = end
    while current != start:
        if current not in previous:
            return []
        current = previous[current]
        path.append(current)
    path.reverse()
    return path


def longest_skeleton_path(skeleton: np.ndarray):
    graph = build_skeleton_graph(skeleton)
    if len(graph) < 2:
        return [], 0.0
    nodes = sorted(graph)
    endpoints = sorted(node for node in nodes if len(graph[node]) == 1)
    targets = endpoints if len(endpoints) >= 2 else nodes
    first_distances, _ = _dijkstra(graph, targets[0])
    first_end = _farthest_reachable(first_distances, targets)
    if first_end is None:
        return [], 0.0
    second_distances, previous = _dijkstra(graph, first_end)
    second_end = _farthest_reachable(second_distances, targets)
    if second_end is None or second_end == first_end:
        return [], 0.0
    return (
        _reconstruct_path(previous, first_end, second_end),
        float(second_distances[second_end]),
    )


def select_candidate_path(binary_mask: np.ndarray, min_component_area: int):
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
        path, length = longest_skeleton_path(skeletonize(component))
        if length > best_length:
            best_path = path
            best_length = length
    return best_path, best_length


def _cumulative_path_lengths(path) -> np.ndarray:
    cumulative = [0.0]
    for index in range(1, len(path)):
        y0, x0 = path[index - 1]
        y1, x1 = path[index]
        cumulative.append(cumulative[-1] + math.hypot(y1 - y0, x1 - x0))
    return np.asarray(cumulative, dtype=np.float32)


def _nearest_path_point(path, cumulative, target_distance):
    index = int(np.argmin(np.abs(cumulative - target_distance)))
    return path[index]


def select_three_points(path, pose_offset: float):
    cumulative = _cumulative_path_lengths(path)
    total_length = float(cumulative[-1])
    midpoint_distance = total_length / 2.0
    effective_offset = min(float(pose_offset), total_length / 4.0)
    point_1 = _nearest_path_point(
        path, cumulative, midpoint_distance - effective_offset
    )
    cut_point = _nearest_path_point(path, cumulative, midpoint_distance)
    point_3 = _nearest_path_point(
        path, cumulative, midpoint_distance + effective_offset
    )
    if len({point_1, cut_point, point_3}) < 3:
        return None
    if point_1 > point_3:
        point_1, point_3 = point_3, point_1
    return point_1, cut_point, point_3


def _to_source_xy(point_yx, source_width: int, source_height: int):
    y, x = point_yx
    x_scale = (source_width - 1) / float(MODEL_SIZE - 1)
    y_scale = (source_height - 1) / float(MODEL_SIZE - 1)
    source_x = int(round(x * x_scale))
    source_y = int(round(y * y_scale))
    return (
        int(np.clip(source_x, 0, source_width - 1)),
        int(np.clip(source_y, 0, source_height - 1)),
    )


def _expanded_box_mask(
    shape: Tuple[int, int],
    bbox_xyxy: Optional[Sequence[float]],
    source_width: int,
    source_height: int,
    padding: float,
) -> np.ndarray:
    allowed = np.ones(shape, dtype=bool)
    if bbox_xyxy is None:
        return allowed
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    x1 = max(0.0, x1 - padding * width)
    y1 = max(0.0, y1 - padding * height)
    x2 = min(float(source_width), x2 + padding * width)
    y2 = min(float(source_height), y2 + padding * height)
    sx = MODEL_SIZE / float(source_width)
    sy = MODEL_SIZE / float(source_height)
    ix1 = int(np.clip(math.floor(x1 * sx), 0, MODEL_SIZE))
    iy1 = int(np.clip(math.floor(y1 * sy), 0, MODEL_SIZE))
    ix2 = int(np.clip(math.ceil(x2 * sx), 0, MODEL_SIZE))
    iy2 = int(np.clip(math.ceil(y2 * sy), 0, MODEL_SIZE))
    allowed[:] = False
    allowed[iy1:iy2, ix1:ix2] = True
    return allowed


def mask_to_keypoints(
    mask: np.ndarray,
    source_width: int,
    source_height: int,
    *,
    stem_bbox_xyxy: Optional[Sequence[float]],
    box_padding: float,
    min_component_area: int,
    min_path_length: float,
    pose_offset: float,
) -> dict:
    """Convert either backend's mask to the same P1/CUT/P3 representation.

    The selected YOLO stem box is applied identically to both masks. This gives
    the second detector the same role in both comparison branches.
    """
    if mask is None or source_width <= 0 or source_height <= 0:
        return {"error": "empty mask", "mask_224": None}
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2 or mask_array.size == 0:
        return {"error": "invalid mask shape", "mask_224": None}
    binary = cv2.resize(
        (mask_array > 0).astype(np.uint8),
        (MODEL_SIZE, MODEL_SIZE),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    binary &= _expanded_box_mask(
        binary.shape,
        stem_bbox_xyxy,
        source_width,
        source_height,
        float(box_padding),
    )
    mask_224 = binary.astype(np.uint8) * 255
    if not np.any(binary):
        return {"error": "empty mask after shared YOLO-box constraint", "mask_224": mask_224}

    path, path_length = select_candidate_path(binary, int(min_component_area))
    if not path:
        return {"error": "no connected skeleton", "mask_224": mask_224}
    if path_length < float(min_path_length):
        return {
            "error": f"path too short ({path_length:.1f})",
            "path_length_224": round(path_length, 3),
            "mask_224": mask_224,
        }
    points = select_three_points(path, float(pose_offset))
    if points is None:
        return {
            "error": "cannot separate three points",
            "path_length_224": round(path_length, 3),
            "mask_224": mask_224,
        }
    point_1, cut_point, point_3 = points
    return {
        "point_1": list(_to_source_xy(point_1, source_width, source_height)),
        "cut_point": list(_to_source_xy(cut_point, source_width, source_height)),
        "point_3": list(_to_source_xy(point_3, source_width, source_height)),
        "path_length_224": round(path_length, 3),
        "mask_224": mask_224,
    }
