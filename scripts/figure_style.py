"""Shared publication style for the manuscript graph review bundle."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import matplotlib as mpl


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green": "#679F67",
    "red_strong": "#B64342",
    "teal": "#42949E",
    "violet": "#7A4E8C",
    "neutral": "#767676",
    "light": "#D8D8D8",
    "dark": "#272727",
}


def apply_publication_style(font_size: float = 8.6) -> None:
    """Apply a compact, print-safe adaptation of the figures4papers style."""

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 0.2,
            "axes.titleweight": "bold",
            "axes.linewidth": 1.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": font_size - 0.5,
            "ytick.labelsize": font_size - 0.5,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "legend.fontsize": font_size - 0.4,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 4.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(axis, *, grid: bool = True) -> None:
    axis.spines["left"].set_color(PALETTE["dark"])
    axis.spines["bottom"].set_color(PALETTE["dark"])
    axis.tick_params(direction="out", length=3.0, color=PALETTE["dark"])
    if grid:
        axis.grid(
            axis="y",
            color=PALETTE["light"],
            linewidth=0.55,
            linestyle=":",
            zorder=0,
        )


def save_figure(figure, output_stem: Path, *, dpi: int = 400) -> list[Path]:
    """Export vector PDF and high-resolution PNG with deterministic PDF dates."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    created = []
    fixed_date = datetime(2026, 9, 27, tzinfo=UTC)
    for suffix in ("pdf", "png"):
        path = output_stem.with_suffix(f".{suffix}")
        metadata = {
            "Creator": "Causal-State-Matching manuscript graph review bundle",
            "Title": output_stem.name,
        }
        if suffix == "pdf":
            metadata.update(CreationDate=fixed_date, ModDate=fixed_date)
        figure.savefig(
            path,
            dpi=dpi if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.035,
            metadata=metadata,
        )
        created.append(path)
    return created
