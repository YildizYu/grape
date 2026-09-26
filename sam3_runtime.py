"""Build the production SAM3 worker from the deployment configuration."""

from pathlib import Path

from .sam3_mask_predictor import Sam3MaskPredictor


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else project_root / path


def build_sam3_predictor(
    project_root: Path,
    config: dict,
    *,
    checkpoint_override=None,
    device_override=None,
):
    models = config.get("models", {})
    sam_cfg = config.get("sam", {})
    runtime_cfg = config.get("sam3_runtime", config.get("comparison", {}))
    checkpoint = _resolve(
        project_root,
        checkpoint_override or models.get("sam_checkpoint", "weights/sam/model.pt"),
    )
    worker_python = _resolve(
        project_root,
        runtime_cfg.get(
            "worker_python",
            runtime_cfg.get("sam3_worker_python", ""),
        ),
    )
    worker_script = _resolve(
        project_root,
        runtime_cfg.get(
            "worker_script",
            runtime_cfg.get("sam3_worker_script", "src/grape_stem_3d/sam3_mask_worker.py"),
        ),
    )
    # SAM3 代码目录: 优化/微调后的代码 (默认 ../sam3), 传给 worker 子进程的
    # sys.path/PYTHONPATH, 使其导入该目录下的 sam3 包而非 third_party 原版
    sam3_source_dir = _resolve(
        project_root,
        runtime_cfg.get("sam3_source_dir", "../sam3"),
    ).resolve()
    required = {
        "SAM3 checkpoint": checkpoint,
        "SAM3 worker Python": worker_python,
        "SAM3 worker script": worker_script,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("SAM3 production chain is incomplete:\n  " + "\n  ".join(missing))
    if not sam3_source_dir.is_dir():
        raise FileNotFoundError(
            f"SAM3 代码目录不存在: {sam3_source_dir} "
            "(检查 fusion_pipeline.yaml 的 sam3_runtime.sam3_source_dir)"
        )
    predictor = Sam3MaskPredictor(
        worker_python=str(worker_python),
        worker_script=str(worker_script),
        checkpoint=str(checkpoint),
        device=str(device_override or models.get("device", "0")),
        box_padding=float(sam_cfg.get("box_padding", 0.15)),
        sam3_source_dir=str(sam3_source_dir),
        ready_timeout_s=float(runtime_cfg.get("ready_timeout_s", runtime_cfg.get("sam3_ready_timeout_s", 300.0))),
        job_timeout_s=float(runtime_cfg.get("job_timeout_s", runtime_cfg.get("sam3_job_timeout_s", 120.0))),
    )
    return predictor, checkpoint, worker_python

