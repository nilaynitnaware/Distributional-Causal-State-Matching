"""Shared output-directory guard for public reproduction entry points."""

from __future__ import annotations

from pathlib import Path


def prepare_output_directory(
    output: Path, *, overwrite: bool = False, protected: tuple[Path, ...] = ()
) -> Path:
    """Resolve and create an output directory without touching protected inputs."""
    destination = output.resolve()
    for item in protected:
        source = item.resolve()
        if destination == source or destination in source.parents or source in destination.parents:
            raise ValueError(f"output must be separate from frozen input: {source}")
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {destination}; pass --overwrite to reuse it"
        )
    destination.mkdir(parents=True, exist_ok=True)
    return destination
