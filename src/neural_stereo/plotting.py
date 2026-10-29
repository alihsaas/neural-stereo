from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .config import SCENES
from .reporting import StudyRow

SCENE_COLORS = {"Motorcycle": "#2a78d6", "Piano": "#eb6834", "Pipes": "#1baf7a"}
SECONDARY_INK = "#52514e"
KILOBYTES_TO_MBPS = 8 * 10 / 1000


def depth_colormap():
    colormap = colormaps["viridis"].copy()
    colormap.set_bad("#ffffff")
    return colormap


def error_colormap():
    colormap = colormaps["magma"].copy()
    colormap.set_bad("#ffffff")
    return colormap


def plot_payload_vs_error(rows: list[StudyRow], destination: Path) -> None:
    figure = Figure(figsize=(8.0, 5.0), dpi=150)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    for scene in SCENES:
        scene_rows = sorted(
            (row for row in rows if row.scene == scene), key=lambda row: row.bytes_pair
        )
        kilobytes = [row.bytes_pair / 1000 for row in scene_rows]
        errors = [row.median_abs_depth_error_common_cm for row in scene_rows]
        axes.plot(
            kilobytes,
            errors,
            marker="o",
            markersize=5,
            linewidth=2,
            color=SCENE_COLORS[scene],
            label=scene,
        )
        for row, x, y in zip(scene_rows, kilobytes, errors, strict=True):
            axes.annotate(
                row.condition,
                (x, y),
                textcoords="offset points",
                xytext=(0, 7),
                fontsize=7,
                ha="center",
                color=SECONDARY_INK,
            )
    axes.set_xscale("log")
    axes.set_xlabel("payload per stereo pair (KB)")
    axes.set_ylabel("median abs depth error on common pixels (cm)")
    secondary = axes.secondary_xaxis(
        "top",
        functions=(
            lambda kilobytes: kilobytes * KILOBYTES_TO_MBPS,
            lambda mbps: mbps / KILOBYTES_TO_MBPS,
        ),
    )
    secondary.set_xlabel("estimated image rate at 10 pairs/s (Mbps)")
    axes.grid(True, which="both", linewidth=0.3, alpha=0.4)
    axes.legend(frameon=False)
    figure.savefig(destination, bbox_inches="tight")


def plot_payload_vs_error_comparison(
    labeled_rows: list[tuple[str, list[StudyRow]]], destination: Path
) -> None:
    linestyles = ("-", "--", ":")
    figure = Figure(figsize=(8.0, 5.0), dpi=150)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    for (_, rows), linestyle in zip(labeled_rows, linestyles, strict=False):
        for scene in SCENES:
            scene_rows = sorted(
                (row for row in rows if row.scene == scene), key=lambda row: row.bytes_pair
            )
            kilobytes = [row.bytes_pair / 1000 for row in scene_rows]
            errors = [row.median_abs_depth_error_common_cm for row in scene_rows]
            axes.plot(
                kilobytes,
                errors,
                linestyle=linestyle,
                marker="o",
                markersize=4,
                linewidth=1.8,
                color=SCENE_COLORS[scene],
            )
    handles = [
        Line2D([], [], color=SCENE_COLORS[scene], linewidth=2, label=scene) for scene in SCENES
    ]
    handles += [
        Line2D([], [], color=SECONDARY_INK, linewidth=2, linestyle=linestyle, label=label)
        for (label, _), linestyle in zip(labeled_rows, linestyles, strict=False)
    ]
    axes.set_xscale("log")
    axes.set_xlabel("payload per stereo pair (KB)")
    axes.set_ylabel("median abs depth error on common pixels (cm)")
    secondary = axes.secondary_xaxis(
        "top",
        functions=(
            lambda kilobytes: kilobytes * KILOBYTES_TO_MBPS,
            lambda mbps: mbps / KILOBYTES_TO_MBPS,
        ),
    )
    secondary.set_xlabel("estimated image rate at 10 pairs/s (Mbps)")
    axes.grid(True, which="both", linewidth=0.3, alpha=0.4)
    axes.legend(handles=handles, frameon=False)
    figure.savefig(destination, bbox_inches="tight")


def plot_scene_grid(
    scene: str,
    original_rgb: np.ndarray,
    gt_depth_cm: np.ndarray,
    condition_panels: list[tuple[str, np.ndarray, np.ndarray]],
    depth_limits_cm: tuple[float, float],
    error_limit_cm: float,
    destination: Path,
) -> None:
    row_count = 1 + len(condition_panels)
    figure = Figure(figsize=(7.6, 2.5 * row_count), dpi=150)
    FigureCanvasAgg(figure)
    axes_grid = figure.subplots(row_count, 2)
    depth_vmin, depth_vmax = depth_limits_cm
    depth_map = None
    error_map = None
    axes_grid[0][0].imshow(original_rgb)
    axes_grid[0][0].set_title(f"{scene}: original im0", fontsize=9)
    depth_map = axes_grid[0][1].imshow(
        gt_depth_cm, cmap=depth_colormap(), vmin=depth_vmin, vmax=depth_vmax
    )
    axes_grid[0][1].set_title("ground-truth depth (cm)", fontsize=9)
    for row_index, (label, est_depth_cm, abs_error_cm) in enumerate(condition_panels, start=1):
        depth_map = axes_grid[row_index][0].imshow(
            est_depth_cm, cmap=depth_colormap(), vmin=depth_vmin, vmax=depth_vmax
        )
        axes_grid[row_index][0].set_ylabel(label, fontsize=10)
        error_map = axes_grid[row_index][1].imshow(
            abs_error_cm, cmap=error_colormap(), vmin=0.0, vmax=error_limit_cm
        )
        if row_index == 1:
            axes_grid[row_index][0].set_title("estimated depth (cm)", fontsize=9)
            axes_grid[row_index][1].set_title("abs depth error (cm)", fontsize=9)
    for axes_row in axes_grid:
        for axes in axes_row:
            axes.set_xticks([])
            axes.set_yticks([])
    figure.colorbar(
        depth_map,
        ax=[axes_row[0] for axes_row in axes_grid],
        location="bottom",
        fraction=0.012,
        pad=0.01,
        label="depth (cm)",
    )
    figure.colorbar(
        error_map,
        ax=[axes_row[1] for axes_row in axes_grid],
        location="bottom",
        fraction=0.012,
        pad=0.01,
        label="abs error (cm)",
    )
    figure.savefig(destination, bbox_inches="tight")
