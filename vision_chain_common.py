"""Geometry and post-processing shared by production and comparison chains."""

from __future__ import annotations

from typing import Mapping, Sequence

from .mask_keypoints import mask_to_keypoints


def expanded_grape_roi(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
    roi_config: Mapping,
):
    """Expand a grape detection using the configured production ROI policy."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return (
        max(0, int(x1 - float(roi_config.get("expand_left", 0.15)) * width)),
        max(0, int(y1 - float(roi_config.get("expand_top", 0.50)) * height)),
        min(
            image_width,
            int(x2 + float(roi_config.get("expand_right", 0.15)) * width),
        ),
        min(
            image_height,
            int(y2 + float(roi_config.get("expand_bottom", 0.10)) * height),
        ),
    )


def postprocess_segmentation_mask(
    mask,
    source_width: int,
    source_height: int,
    stem_bbox_xyxy: Sequence[float],
    config: Mapping,
):
    """Run the exact mask-to-P1/CUT/P3 policy used by both entry points."""
    postprocess = config.get("mask_postprocess", config.get("keypoint", {}))
    sam = config.get("sam", {})
    return mask_to_keypoints(
        mask,
        source_width=source_width,
        source_height=source_height,
        stem_bbox_xyxy=stem_bbox_xyxy,
        box_padding=float(sam.get("box_padding", 0.15)),
        min_component_area=int(postprocess.get("min_component_area", 12)),
        min_path_length=float(postprocess.get("min_path_length", 20.0)),
        pose_offset=float(postprocess.get("pose_offset", 12.0)),
    )

