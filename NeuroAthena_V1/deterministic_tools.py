"""Deterministic plotting tools for trained MR-AIV/PINN MATLAB results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


TOOL_VERSION = "1.0.0"
REQUIRED_KEYS = {"txyz_smm", "u_mm_s", "v_mm_s", "w_mm_s", "K_m2"}
EPS = np.finfo(np.float64).tiny
PLOT_CMAP = "plasma"


def get_colors_plot(cmap: str = PLOT_CMAP, n_colors: int = 8) -> list[Any]:
    """Sample distinct colors from the outer quarters of a colormap."""
    if n_colors < 1:
        raise ValueError("n_colors must be positive")
    sampled_count = n_colors * 2
    color_map = plt.get_cmap(cmap)
    sampled = [color_map(value) for value in np.linspace(0.0, 1.0, sampled_count)]
    quarter = sampled_count // 4
    return sampled[:quarter] + sampled[3 * quarter :]


def configure_plot_style() -> None:
    """Apply project-wide typography without imposing a global color cycle."""
    matplotlib.rcParams.update(
        {
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "font.size": 14,
        }
    )


configure_plot_style()


@dataclass(frozen=True)
class PINNResult:
    source_file: Path
    x_mm: np.ndarray
    y_mm: np.ndarray
    z_mm: np.ndarray
    u_um_s: np.ndarray
    v_um_s: np.ndarray
    w_um_s: np.ndarray
    speed_um_s: np.ndarray
    permeability_mm2: np.ndarray


def load_pinn_result(path: Path) -> PINNResult:
    """Load and validate the trained-result schema, applying explicit SI conversions."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = loadmat(path)
    missing = sorted(REQUIRED_KEYS.difference(raw))
    if missing:
        raise KeyError(f"Missing required MATLAB fields: {', '.join(missing)}")
    coordinates = np.asarray(raw["txyz_smm"], dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 4:
        raise ValueError(f"txyz_smm must have shape (n, >=4), got {coordinates.shape}")

    fields = {key: np.asarray(raw[key], dtype=np.float64).reshape(-1) for key in REQUIRED_KEYS - {"txyz_smm"}}
    n = coordinates.shape[0]
    bad_lengths = {key: values.size for key, values in fields.items() if values.size != n}
    if bad_lengths:
        raise ValueError(f"Field lengths do not match txyz_smm ({n} rows): {bad_lengths}")

    # Source velocities are mm/s; 1 mm = 1000 µm.
    u = fields["u_mm_s"] * 1_000.0
    v = fields["v_mm_s"] * 1_000.0
    w = fields["w_mm_s"] * 1_000.0
    speed = np.sqrt(u * u + v * v + w * w)
    # 1 m² = 10^6 mm². No normalization is applied, so physical units are retained.
    permeability = fields["K_m2"] * 1_000_000.0
    finite = np.isfinite(coordinates[:, 1:4]).all(axis=1)
    for values in (u, v, w, speed, permeability):
        finite &= np.isfinite(values)
    if not finite.any():
        raise ValueError("No finite samples remain after validation")
    return PINNResult(
        source_file=path,
        x_mm=coordinates[finite, 1],
        y_mm=coordinates[finite, 2],
        z_mm=coordinates[finite, 3],
        u_um_s=u[finite],
        v_um_s=v[finite],
        w_um_s=w[finite],
        speed_um_s=speed[finite],
        permeability_mm2=permeability[finite],
    )


def dataset_metadata(path: Path) -> dict[str, Any]:
    data = load_pinn_result(path)
    return {
        "source_file": str(data.source_file),
        "sample_count": int(data.x_mm.size),
        "required_keys": sorted(REQUIRED_KEYS),
        "coordinate_bounds_mm": {
            "x": [float(data.x_mm.min()), float(data.x_mm.max())],
            "y": [float(data.y_mm.min()), float(data.y_mm.max())],
            "z": [float(data.z_mm.min()), float(data.z_mm.max())],
        },
        "units": {"coordinates": "mm", "velocity": "um/s", "permeability": "mm^2"},
    }


def _log10_abs(values: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(np.abs(values), EPS))


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "p01": float(np.percentile(values, 1)),
        "p99": float(np.percentile(values, 99)),
    }


def plot_velocity_distributions(path: Path, output: Path, bins: int = 80) -> dict[str, Any]:
    data = load_pinn_result(path)
    fields = [data.speed_um_s, data.u_um_s, data.v_um_s, data.w_um_s]
    labels = [r"$\|\mathbf{u}\|$", r"$|u|$", r"$|v|$", r"$|w|$"]
    logs = [_log10_abs(values) for values in fields]
    bin_edges = np.linspace(-5.0, 2.0, bins + 1)
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True, constrained_layout=True)
    for axis, values, label in zip(axes, logs, labels):
        hist, edges = np.histogram(values, bins=bin_edges, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        axis.plot(centers, hist, color="#3264a8", linewidth=2)
        axis.set_xlabel(f"{label} [µm/s]")
        axis.set_xlim(-5, 2)
        ticks = np.arange(-5, 3, 2)
        axis.set_xticks(ticks, [rf"$10^{{{int(t)}}}$" for t in ticks])
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Probability density")
    fig.suptitle("PINN velocity distributions")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure_path": str(output.resolve()),
        "bins": bins,
        "log10_range": [-5.0, 2.0],
        "units": "um/s",
        "statistics": {name: _summary(values) for name, values in zip(("speed", "u", "v", "w"), fields)},
    }


def plot_permeability_distribution(path: Path, output: Path, bins: int = 80) -> dict[str, Any]:
    data = load_pinn_result(path)
    log_k = np.log10(np.maximum(data.permeability_mm2, EPS))
    edges = np.linspace(-10.5, -6.5, bins + 1)
    hist, edges = np.histogram(log_k, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    positive = hist > 0
    fig, axis = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    axis.plot(centers[positive], hist[positive], color="#7b3294", linewidth=2)
    axis.set_yscale("log")
    axis.set_xlim(-10.5, -6.5)
    ticks = np.arange(-10, -5, 1)
    axis.set_xticks(ticks, [rf"$10^{{{int(t)}}}$" for t in ticks])
    axis.set_xlabel(r"Permeability $\kappa$ [mm$^2$]")
    axis.set_ylabel("Probability density")
    axis.set_title("PINN permeability distribution")
    axis.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure_path": str(output.resolve()),
        "bins": bins,
        "log10_range": [-10.5, -6.5],
        "units": "mm^2",
        "statistics": _summary(data.permeability_mm2),
    }


def _central_plane(values: np.ndarray) -> tuple[float, np.ndarray]:
    unique = np.unique(values)
    center = float(np.median(unique))
    selected = float(unique[np.argmin(np.abs(unique - center))])
    return selected, np.isclose(values, selected, rtol=0.0, atol=1e-10)


def _plot_midplanes(
    data: PINNResult,
    values: np.ndarray,
    output: Path,
    *,
    title: str,
    colorbar_label: str,
    cmap: str,
    color_limits: tuple[float, float],
) -> dict[str, Any]:
    x_plane, x_mask = _central_plane(data.x_mm)
    y_plane, y_mask = _central_plane(data.y_mm)
    z_plane, z_mask = _central_plane(data.z_mm)
    panels = [
        (data.z_mm[x_mask], data.y_mm[x_mask], values[x_mask], f"Sagittal: x={x_plane:.2f} mm"),
        (data.x_mm[y_mask], data.z_mm[y_mask], values[y_mask], f"Coronal: y={y_plane:.2f} mm"),
        (data.x_mm[z_mask], data.y_mm[z_mask], values[z_mask], f"Axial: z={z_plane:.2f} mm"),
    ]
    if any(panel[0].size == 0 for panel in panels):
        raise ValueError("At least one selected central plane contains no samples")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    shown = None
    for axis, (horizontal, vertical, color, panel_title) in zip(axes, panels):
        shown = axis.scatter(
            horizontal,
            vertical,
            c=color,
            cmap=cmap,
            s=5,
            vmin=color_limits[0],
            vmax=color_limits[1],
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(panel_title)
        axis.set_aspect("equal")
        axis.set_axis_off()
    fig.colorbar(shown, ax=axes, shrink=0.78, label=colorbar_label)
    fig.suptitle(title, y=0.93)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure_path": str(output.resolve()),
        "plane_locations_mm": {"x": x_plane, "y": y_plane, "z": z_plane},
        "plane_sample_counts": {"x": int(x_mask.sum()), "y": int(y_mask.sum()), "z": int(z_mask.sum())},
        "color_limits": list(color_limits),
    }


def plot_speed_slices(path: Path, output: Path) -> dict[str, Any]:
    data = load_pinn_result(path)
    log_speed = np.log10(np.maximum(data.speed_um_s, EPS))
    result = _plot_midplanes(
        data,
        log_speed,
        output,
        title="PINN speed — central anatomical planes",
        colorbar_label=r"$\log_{10}\|\mathbf{u}\|$ [µm/s]",
        cmap="magma",
        color_limits=(-3.0, 2.0),
    )
    result.update({"units": "um/s", "statistics": _summary(data.speed_um_s)})
    return result


def plot_permeability_slices(path: Path, output: Path) -> dict[str, Any]:
    data = load_pinn_result(path)
    log_k = np.log10(np.maximum(data.permeability_mm2, EPS))
    result = _plot_midplanes(
        data,
        log_k,
        output,
        title="PINN permeability — central anatomical planes",
        colorbar_label=r"$\log_{10}\kappa$ [mm$^2$]",
        cmap="viridis",
        color_limits=(-10.5, -6.5),
    )
    result.update({"units": "mm^2", "statistics": _summary(data.permeability_mm2)})
    return result
