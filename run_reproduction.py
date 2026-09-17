#!/usr/bin/env python3
"""Reproduce the synthetic Star and Fracture C2F Hash-DIC benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_FIGURES_DIR = ROOT / "figures"
CRACK_TIP_X = 600
CRACK_CENTER_Y = 250
MAX_QUERY_CHUNK_SIZE = 131072


@dataclass(frozen=True)
class CaseConfig:
    name: str
    reference: str
    deformed: str


@dataclass(frozen=True)
class ReproductionSettings:
    iterations: int = 1200
    batch_size: int = 2**16
    n_levels: int = 32
    features_per_level: int = 2
    log2_hashmap_size: int = 17
    base_resolution: int = 8
    max_resolution: int = 5120
    warmup_iterations: int = 600
    encoder_learning_rate: float = 1.0e-3
    decoder_learning_rate: float = 1.0e-3
    learning_rate_milestone: int = 1000
    learning_rate_factor: float = 0.5
    seed: int = 20260903
    query_chunk_size: int = 131072


CASES = {
    "star": CaseConfig("star", "star_reference.bmp", "star_deformed.bmp"),
    "fracture": CaseConfig(
        "fracture", "fracture_reference.bmp", "fracture_deformed.bmp"
    ),
}


def build_ground_truth(
    case: str, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return analytical displacements in right/down-positive pixel units."""
    if case not in CASES:
        raise ValueError(f"Unknown benchmark case: {case}")
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"Invalid image shape: {shape}")

    height, width = shape
    u_true = np.zeros(shape, dtype=np.float32)
    if case == "star":
        x_1based, y_1based = np.meshgrid(
            np.arange(1, width + 1, dtype=np.float64),
            np.arange(1, height + 1, dtype=np.float64),
        )
        wavelength = 30.0 + 0.2 * x_1based
        v_true = 3.0 * np.cos(
            2.0 * np.pi * (y_1based - (height + 1) / 2.0) / wavelength
        )
    else:
        x = np.arange(width, dtype=np.float32)
        opening = 5.0 * np.sqrt(np.clip(1.0 - x / CRACK_TIP_X, 0.0, 1.0))
        sign = np.where(
            np.arange(height, dtype=np.int32) < CRACK_CENTER_Y, -1.0, 1.0
        ).astype(np.float32)
        v_true = sign[:, None] * opening[None, :]
    return u_true, np.asarray(v_true, dtype=np.float32)


def _validate_matching_fields(*arrays: np.ndarray) -> tuple[int, int]:
    shapes = {np.asarray(array).shape for array in arrays}
    if len(shapes) != 1 or any(len(shape) != 2 for shape in shapes):
        raise ValueError("All displacement and truth arrays must have the same shape")
    return next(iter(shapes))


def calculate_metrics(
    case: str,
    u: np.ndarray,
    v: np.ndarray,
    u_true: np.ndarray,
    v_true: np.ndarray,
) -> dict[str, float | int | str]:
    """Calculate traceable full-field error metrics for one benchmark."""
    height, width = _validate_matching_fields(u, v, u_true, v_true)
    if case not in CASES:
        raise ValueError(f"Unknown benchmark case: {case}")

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u_true = np.asarray(u_true, dtype=np.float64)
    v_true = np.asarray(v_true, dtype=np.float64)
    valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(u_true) & np.isfinite(v_true)
    if not np.any(valid):
        raise ValueError("No finite displacement pixels are available for evaluation")

    du = u[valid] - u_true[valid]
    dv = v[valid] - v_true[valid]
    epe = np.hypot(du, dv)
    metrics: dict[str, float | int | str] = {
        "case": case,
        "n_valid_pixels": int(valid.sum()),
        "valid_coverage": float(valid.mean()),
        "u_mae_px": float(np.mean(np.abs(du))),
        "u_rmse_px": float(np.sqrt(np.mean(np.square(du)))),
        "v_mae_px": float(np.mean(np.abs(dv))),
        "v_rmse_px": float(np.sqrt(np.mean(np.square(dv)))),
        "mean_epe_px": float(np.mean(epe)),
        "p95_epe_px": float(np.percentile(epe, 95.0)),
        "max_abs_u_error_px": float(np.max(np.abs(du))),
        "max_abs_v_error_px": float(np.max(np.abs(dv))),
    }
    if case == "fracture":
        upper = CRACK_CENTER_Y - 10
        lower = CRACK_CENTER_Y + 10
        if lower >= height:
            raise ValueError(
                f"Fracture field height {height} does not contain jump-evaluation rows"
            )
        x_stop = min(CRACK_TIP_X + 1, width)
        predicted_jump = v[lower, :x_stop] - v[upper, :x_stop]
        true_jump = v_true[lower, :x_stop] - v_true[upper, :x_stop]
        jump_valid = np.isfinite(predicted_jump) & np.isfinite(true_jump)
        if not np.any(jump_valid):
            raise ValueError("No finite fracture jump samples are available")
        metrics["jump_rmse_px"] = float(
            np.sqrt(np.mean(np.square(predicted_jump[jump_valid] - true_jump[jump_valid])))
        )
    return metrics


def write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    """Write a deterministic one-row metrics table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)


def sha256_file(path: Path) -> str:
    """Return a lower-case SHA-256 digest without loading the whole file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_grayscale(path: Path) -> np.ndarray:
    """Load an 8/16-bit image as finite float32 grayscale in [0, 1]."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required to read benchmark images; install opencv-python."
        ) from exc

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark image: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read a two-dimensional grayscale image: {path}")
    if image.ndim == 3 and image.shape[2] in (3, 4):
        color_channels = image[..., :3]
        if not (
            np.array_equal(color_channels[..., 0], color_channels[..., 1])
            and np.array_equal(color_channels[..., 0], color_channels[..., 2])
        ):
            raise ValueError(f"Input must be grayscale, but color channels differ: {path}")
        image = color_channels[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Cannot read a two-dimensional grayscale image: {path}")
    if image.dtype == np.uint8:
        normalized = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        normalized = image.astype(np.float32) / 65535.0
    else:
        normalized = image.astype(np.float32)
        minimum = float(np.nanmin(normalized))
        maximum = float(np.nanmax(normalized))
        normalized = (normalized - minimum) / max(maximum - minimum, 1.0e-8)
    if not np.isfinite(normalized).all():
        raise ValueError(f"Image contains non-finite values: {path}")
    return normalized


def save_result(
    path: Path,
    u: np.ndarray,
    v: np.ndarray,
    u_true: np.ndarray,
    v_true: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Persist fields and JSON metadata in a portable compressed archive."""
    _validate_matching_fields(u, v, u_true, v_true)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        u=np.asarray(u, dtype=np.float32),
        v=np.asarray(v, dtype=np.float32),
        u_true=np.asarray(u_true, dtype=np.float32),
        v_true=np.asarray(v_true, dtype=np.float32),
        metadata=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    )


def build_matched_pyramid(
    image_tensor,
    *,
    n_levels: int,
    base_resolution: int,
    max_resolution: int,
):
    """Build the Gaussian image sequence matched to HashGrid cell sizes."""
    import torchvision.transforms.functional as vision_functional

    if image_tensor.ndim != 4 or image_tensor.shape[:2] != (1, 1):
        raise ValueError("image_tensor must have shape (1, 1, H, W)")
    if n_levels < 2 or base_resolution <= 0 or max_resolution <= base_resolution:
        raise ValueError("Invalid pyramid resolution settings")

    image_size = max(int(image_tensor.shape[-2]), int(image_tensor.shape[-1]))
    per_level_scale = float(
        np.exp(np.log(max_resolution / base_resolution) / (n_levels - 1))
    )
    pyramid = []
    for level in range(n_levels):
        resolution = base_resolution * per_level_scale**level
        sigma = image_size / resolution / 2.0
        if sigma < 0.5:
            pyramid.append(image_tensor.clone())
            continue
        kernel_size = max(3, int(sigma * 4.0))
        if kernel_size % 2 == 0:
            kernel_size += 1
        maximum_kernel = 2 * min(image_tensor.shape[-2:]) - 1
        kernel_size = min(kernel_size, maximum_kernel)
        pyramid.append(
            vision_functional.gaussian_blur(
                image_tensor,
                kernel_size=[kernel_size, kernel_size],
                sigma=[sigma, sigma],
            )
        )
    return pyramid


def dense_query(model, coordinates, *, cursor: float, chunk_size: int):
    """Query a coordinate model in bounded chunks and concatenate on-device."""
    import torch

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_size > MAX_QUERY_CHUNK_SIZE:
        raise ValueError(
            f"chunk_size must be at most {MAX_QUERY_CHUNK_SIZE} for bounded memory"
        )
    chunks = []
    granularity = int(getattr(model, "batch_size_granularity", 1))
    if granularity <= 0:
        raise ValueError("model batch_size_granularity must be positive")
    with torch.no_grad():
        for start in range(0, int(coordinates.shape[0]), chunk_size):
            coordinate_chunk = coordinates[start : start + chunk_size]
            original_size = int(coordinate_chunk.shape[0])
            remainder = original_size % granularity
            if remainder:
                padding = granularity - remainder
                coordinate_chunk = torch.cat(
                    [coordinate_chunk, coordinate_chunk[-1:].expand(padding, -1)], dim=0
                )
            prediction = model(coordinate_chunk, cursor)
            chunks.append(prediction[:original_size])
    if not chunks:
        return torch.empty((0, 2), device=coordinates.device, dtype=coordinates.dtype)
    return torch.cat(chunks, dim=0)


def require_gpu_stack():
    """Import and validate the CUDA-only dependencies at execution time."""
    try:
        import torch
        import tinycudann as tcnn
    except ImportError as exc:
        raise RuntimeError(
            "Complete reproduction requires PyTorch with CUDA and the "
            "tiny-cuda-nn Python bindings. See README.md for installation."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA-capable NVIDIA GPU is required for complete reproduction."
        )
    return torch, tcnn


def _ensure_writable_directory(path: Path) -> None:
    """Create a directory and verify that a small temporary file can be written."""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".write-check-", dir=path, delete=True
        ) as handle:
            handle.write(b"ok")
            handle.flush()
    except OSError as exc:
        raise RuntimeError(f"Output directory is not writable: {path}") from exc


def preflight_environment(output_root: Path) -> dict[str, str | None]:
    """Validate runtime dependencies and output locations before optimization."""
    torch, tcnn = require_gpu_stack()
    try:
        import cv2

        configure_matplotlib_cache(
            Path(tempfile.gettempdir()) / "c2f-hash-dic-matplotlib-cache"
        )
        import matplotlib
        import torchvision
    except ImportError as exc:
        raise RuntimeError(
            "Complete reproduction requires OpenCV, Matplotlib, and torchvision. "
            "See README.md for installation."
        ) from exc

    output_root = Path(output_root)
    _ensure_writable_directory(output_root)
    _ensure_writable_directory(output_root / "results")
    _ensure_writable_directory(output_root / "figures")
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "tinycudann": getattr(tcnn, "__version__", "not reported"),
    }


def set_random_seed(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_solver(
    reference: np.ndarray,
    deformed: np.ndarray,
    settings: ReproductionSettings,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Optimize one reference/deformed pair with synchronized C2F Hash-DIC."""
    torch, tcnn = require_gpu_stack()
    import torch.nn.functional as torch_functional

    if reference.shape != deformed.shape or reference.ndim != 2:
        raise ValueError("Reference and deformed images must have the same 2-D shape")
    if not np.isfinite(reference).all() or not np.isfinite(deformed).all():
        raise ValueError("Input images must contain only finite values")
    if settings.batch_size % 128:
        raise ValueError("batch_size must be a multiple of 128 for tiny-cuda-nn")

    set_random_seed(settings.seed, torch)
    device = torch.device("cuda")
    height, width = reference.shape

    reference_tensor = torch.from_numpy(reference).to(device).view(1, 1, height, width)
    deformed_tensor = torch.from_numpy(deformed).to(device).view(1, 1, height, width)
    reference_pyramid = build_matched_pyramid(
        reference_tensor,
        n_levels=settings.n_levels,
        base_resolution=settings.base_resolution,
        max_resolution=settings.max_resolution,
    )
    deformed_pyramid = build_matched_pyramid(
        deformed_tensor,
        n_levels=settings.n_levels,
        base_resolution=settings.base_resolution,
        max_resolution=settings.max_resolution,
    )

    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    pixel_coordinates = torch.stack([x, y], dim=-1).reshape(-1, 2).float()
    normalized_coordinates = pixel_coordinates.clone()
    normalized_coordinates[:, 0] /= width - 1
    normalized_coordinates[:, 1] /= height - 1

    class C2FHashDIC(torch.nn.Module):
        batch_size_granularity = 128

        def __init__(self) -> None:
            super().__init__()
            level_scale = float(
                np.exp(
                    np.log(settings.max_resolution / settings.base_resolution)
                    / (settings.n_levels - 1)
                )
            )
            self.encoder = tcnn.Encoding(
                n_input_dims=2,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": settings.n_levels,
                    "n_features_per_level": settings.features_per_level,
                    "log2_hashmap_size": settings.log2_hashmap_size,
                    "base_resolution": settings.base_resolution,
                    "per_level_scale": level_scale,
                },
            )
            self.decoder = tcnn.Network(
                n_input_dims=settings.n_levels * settings.features_per_level,
                n_output_dims=2,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

        def forward(self, coordinates, cursor: float):
            features = self.encoder(coordinates)
            levels = torch.arange(
                settings.n_levels, device=features.device, dtype=torch.float32
            )
            weights = (cursor - levels + 1.0).clamp(0.0, 1.0)
            feature_mask = weights.repeat_interleave(
                settings.features_per_level
            ).unsqueeze(0)
            return self.decoder(features * feature_mask)

    model = C2FHashDIC().to(device)
    optimizer = torch.optim.Adam(
        [
            {
                "params": model.encoder.parameters(),
                "lr": settings.encoder_learning_rate,
            },
            {
                "params": model.decoder.parameters(),
                "lr": settings.decoder_learning_rate,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[settings.learning_rate_milestone],
        gamma=settings.learning_rate_factor,
    )

    point_count = int(pixel_coordinates.shape[0])
    start_time = time.perf_counter()
    final_loss = float("nan")
    for iteration in range(1, settings.iterations + 1):
        progress = min((iteration - 1) / settings.warmup_iterations, 1.0)
        cursor = progress * (settings.n_levels - 1)
        lower_level = int(math.floor(cursor))
        upper_level = min(lower_level + 1, settings.n_levels - 1)
        alpha = cursor - lower_level

        sample_indices = torch.randint(
            0, point_count, (settings.batch_size,), device=device
        )
        sampled_normalized = normalized_coordinates[sample_indices]
        sampled_pixels = pixel_coordinates[sample_indices]
        displacement = model(sampled_normalized, cursor)
        warped = sampled_pixels + displacement
        grid = warped.view(1, 1, -1, 2).clone()
        grid[..., 0] = 2.0 * grid[..., 0] / (width - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (height - 1) - 1.0

        target_lower = reference_pyramid[lower_level].view(-1, 1)[sample_indices]
        target_upper = reference_pyramid[upper_level].view(-1, 1)[sample_indices]
        target = (1.0 - alpha) * target_lower + alpha * target_upper
        sampled_lower = torch_functional.grid_sample(
            deformed_pyramid[lower_level],
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        ).reshape(-1, 1)
        sampled_upper = torch_functional.grid_sample(
            deformed_pyramid[upper_level],
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        ).reshape(-1, 1)
        sampled = (1.0 - alpha) * sampled_lower + alpha * sampled_upper
        loss = torch.mean((sampled - target) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite photometric loss at iteration {iteration}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss = float(loss.detach().cpu())
        if iteration == 1 or iteration % 100 == 0 or iteration == settings.iterations:
            print(
                f"  iteration {iteration:4d}/{settings.iterations}: "
                f"cursor={cursor:5.2f}, photometric_loss={final_loss:.7g}",
                flush=True,
            )

    elapsed_seconds = time.perf_counter() - start_time
    dense = dense_query(
        model,
        normalized_coordinates,
        cursor=float(settings.n_levels - 1),
        chunk_size=settings.query_chunk_size,
    ).detach().float().cpu().numpy()
    u = dense[:, 0].reshape(height, width).astype(np.float32)
    v = dense[:, 1].reshape(height, width).astype(np.float32)
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise RuntimeError("Dense displacement query produced non-finite values")

    solver_metadata = {
        "device": torch.cuda.get_device_name(0),
        "elapsed_seconds": elapsed_seconds,
        "final_photometric_loss": final_loss,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tinycudann_version": getattr(tcnn, "__version__", "not reported"),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    del model, optimizer, reference_pyramid, deformed_pyramid
    torch.cuda.empty_cache()
    return u, v, solver_metadata


def configure_matplotlib_cache(cache_directory: Path) -> None:
    """Point Matplotlib at a writable cache before importing pyplot."""
    cache_directory = Path(cache_directory)
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_directory))


def plot_displacement_field(
    path: Path,
    *,
    field: np.ndarray,
    title: str,
    field_limit: float,
) -> None:
    """Export one signed vertical-displacement field with a fixed color scale."""
    configure_matplotlib_cache(
        Path(tempfile.gettempdir()) / "c2f-hash-dic-matplotlib-cache"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    field = np.asarray(field)
    if field.ndim != 2 or not np.isfinite(field).all():
        raise ValueError("Displacement field must be a finite two-dimensional array")
    if field_limit <= 0:
        raise ValueError("field_limit must be positive")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.8,
        }
    )
    figure, axis = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    shown = axis.imshow(
        field,
        cmap="coolwarm",
        vmin=-field_limit,
        vmax=field_limit,
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.set_xlabel("x / px")
    axis.set_ylabel("y / px")
    colorbar = figure.colorbar(shown, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label(r"Vertical displacement $v$ / px")
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def persist_case_artifacts(
    *,
    output_root: Path,
    case: str,
    u: np.ndarray,
    v: np.ndarray,
    u_true: np.ndarray,
    v_true: np.ndarray,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    """Stage a complete artifact set, then publish it to the output folders."""
    output_root = Path(output_root)
    results_directory = output_root / "results"
    figures_directory = output_root / "figures"
    results_directory.mkdir(parents=True, exist_ok=True)
    figures_directory.mkdir(parents=True, exist_ok=True)
    result_path = results_directory / f"{case}_result.npz"
    metrics_path = results_directory / f"{case}_metrics.csv"
    computed_path = figures_directory / f"{case}_computed.png"
    theoretical_path = figures_directory / f"{case}_theoretical.png"
    field_limit = 3.0 if case == "star" else 5.0

    with tempfile.TemporaryDirectory(prefix=".c2f-stage-", dir=output_root) as directory:
        stage = Path(directory)
        staged_result = stage / result_path.name
        staged_metrics = stage / metrics_path.name
        staged_computed = stage / computed_path.name
        staged_theoretical = stage / theoretical_path.name
        save_result(staged_result, u, v, u_true, v_true, metadata)
        write_metrics_csv(staged_metrics, metrics)
        plot_displacement_field(
            staged_computed,
            field=v,
            title=f"{case.capitalize()}: C2F Hash-DIC computed field",
            field_limit=field_limit,
        )
        plot_displacement_field(
            staged_theoretical,
            field=v_true,
            title=f"{case.capitalize()}: theoretical field",
            field_limit=field_limit,
        )
        os.replace(staged_result, result_path)
        os.replace(staged_metrics, metrics_path)
        os.replace(staged_computed, computed_path)
        os.replace(staged_theoretical, theoretical_path)
    return result_path, metrics_path, computed_path, theoretical_path


def run_case(
    case: str,
    settings: ReproductionSettings,
    output_root: Path,
) -> dict[str, Any]:
    """Run, evaluate, and persist one complete benchmark."""
    config = CASES[case]
    output_root = Path(output_root)
    library_versions = preflight_environment(output_root)
    reference_path = DATA_DIR / config.reference
    deformed_path = DATA_DIR / config.deformed
    reference = load_grayscale(reference_path)
    deformed = load_grayscale(deformed_path)
    if reference.shape != deformed.shape:
        raise ValueError(
            f"{case} input shapes differ: {reference.shape} and {deformed.shape}"
        )

    print(f"Running {case}: {reference.shape[1]} x {reference.shape[0]} pixels")
    u, v, solver_metadata = run_solver(reference, deformed, settings)
    u_true, v_true = build_ground_truth(case, reference.shape)
    metrics = calculate_metrics(case, u, v, u_true, v_true)
    metadata = {
        "case": case,
        "reference_file": f"data/{config.reference}",
        "deformed_file": f"data/{config.deformed}",
        "reference_sha256": sha256_file(reference_path),
        "deformed_sha256": sha256_file(deformed_path),
        "coordinate_convention": "u right-positive, v down-positive, pixels",
        "settings": asdict(settings),
        "library_versions": library_versions,
        **solver_metadata,
    }
    result_path, metrics_path, computed_path, theoretical_path = persist_case_artifacts(
        output_root=output_root,
        case=case,
        u=u,
        v=v,
        u_true=u_true,
        v_true=v_true,
        metadata=metadata,
        metrics=metrics,
    )
    print(f"  result:  {result_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  computed field:    {computed_path}")
    print(f"  theoretical field: {theoretical_path}")
    return metrics


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def bounded_query_chunk_size(value: str) -> int:
    parsed = positive_int(value)
    if parsed > MAX_QUERY_CHUNK_SIZE:
        raise argparse.ArgumentTypeError(
            f"value must be at most {MAX_QUERY_CHUNK_SIZE}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fully recompute the synthetic C2F Hash-DIC benchmarks."
    )
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--iterations", type=positive_int, default=1200)
    parser.add_argument("--batch-size", type=positive_int, default=2**16)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--query-chunk-size",
        type=bounded_query_chunk_size,
        default=MAX_QUERY_CHUNK_SIZE,
    )
    parser.add_argument("--output-root", type=Path, default=ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = ReproductionSettings(
        iterations=args.iterations,
        batch_size=args.batch_size,
        seed=args.seed,
        query_chunk_size=args.query_chunk_size,
    )
    selected_cases = tuple(CASES) if args.case == "all" else (args.case,)
    print(
        "C2F Hash-DIC complete reproduction\n"
        f"Python: {sys.version.split()[0]}\n"
        f"Cases: {', '.join(selected_cases)}\n"
        f"Settings: {json.dumps(asdict(settings), sort_keys=True)}"
    )
    for case in selected_cases:
        metrics = run_case(case, settings, args.output_root)
        print(
            f"Completed {case}: mean EPE={float(metrics['mean_epe_px']):.6f} px, "
            f"v RMSE={float(metrics['v_rmse_px']):.6f} px"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
