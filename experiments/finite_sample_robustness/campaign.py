"""Resumable task-level orchestration and immutable manifest handling."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .design import CampaignTask, load_config, task_specs
from .execution import run_task

ROOT = Path(__file__).resolve().parents[2]


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    return value


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _git(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hardware_payload() -> dict[str, object]:
    try:
        import torch

        torch_version = torch.__version__
        cuda_runtime = torch.version.cuda
        cuda_available = torch.cuda.is_available()
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ] if cuda_available else []
    except ImportError:
        torch_version = None
        cuda_runtime = None
        cuda_available = False
        gpu_names = []
    try:
        import importlib.util

        cupy_available = importlib.util.find_spec("cupy") is not None
    except (ImportError, ValueError):
        cupy_available = False
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "torch": torch_version,
        "torch_cuda_runtime": cuda_runtime,
        "torch_cuda_available": cuda_available,
        "gpu_names": gpu_names,
        "cupy_available": cupy_available,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }


def manifest_payload(
    *,
    config_path: Path,
    config: dict[str, object],
    tasks: tuple[CampaignTask, ...],
    mode: Literal["pilot", "full"],
    seed: int,
    output_dir: Path,
    workers: int,
) -> dict[str, object]:
    return {
        "schema_version": "finite_sample_robustness_manifest_v1",
        "experiment_version": config["experiment_version"],
        "protocol_version": config["protocol_version"],
        "mode": mode,
        "status": "RUNNING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": _file_hash(config_path),
        "seed": seed,
        "workers": workers,
        "backend": config["backend"],
        "device": config["device"],
        "execution_backend": config["execution_backend"],
        "bootstrap_replicates": (
            config["pilot_bootstrap_replicates"]
            if mode == "pilot"
            else config["bootstrap_replicates"]
        ),
        "expected_task_count": len(tasks),
        "output_dir": str(output_dir.resolve()),
        "result_directory": str((output_dir / "results").resolve()),
        "failure_policy": "Every task writes one row, including explicit failures; no failed row is dropped.",
        "pairing_policy": "Stage-B source ID plus deterministic sample-prefix seeds; never row order.",
        "hidden_information_policy": (
            "Hidden SCM/DAG/target metadata is construction/evaluation-only. "
            "Only controlled orientation corruption uses the hidden DAG to create a labeled wrong direction."
        ),
        "production_algorithm_semantics_modified": False,
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "status_short": (_git("status", "--short") or "").splitlines(),
        },
        "hardware": hardware_payload(),
        "frozen_configuration": config,
        "tasks": [asdict(task) for task in tasks],
    }


def _worker(config: dict[str, object], payload: dict[str, object], bootstraps: int):
    return run_task(config, CampaignTask(**payload), bootstrap_replicates=bootstraps)


def _task_order(task: CampaignTask) -> tuple[object, ...]:
    return (
        task.campaign,
        task.n,
        task.graph_index,
        task.replicate,
        task.target_family,
        task.graph_condition,
        task.N_obs,
        task.N_target,
        task.N_int,
        task.corruption_type or "",
        task.corruption_severity or "",
    )


def run_campaign(
    *,
    config_path: Path,
    output_dir: Path,
    mode: Literal["pilot", "full"],
    seed: int,
    workers: int,
    resume: bool,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    config = load_config(config_path)
    tasks = task_specs(config, mode=mode, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    expected_manifest = manifest_payload(
        config_path=config_path,
        config=config,
        tasks=tasks,
        mode=mode,
        seed=seed,
        output_dir=output_dir,
        workers=workers,
    )
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        invariant_keys = (
            "schema_version", "experiment_version", "mode", "config_sha256",
            "seed", "expected_task_count", "tasks",
        )
        if any(prior.get(key) != expected_manifest.get(key) for key in invariant_keys):
            raise RuntimeError("existing output manifest is incompatible with this run")
        if not resume:
            raise FileExistsError("output manifest exists; pass --resume for the same campaign")
        manifest = prior
        manifest["status"] = "RUNNING"
        manifest["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["workers"] = workers
    else:
        manifest = expected_manifest
    atomic_json(manifest_path, manifest)

    completed = {
        path.stem for path in (output_dir / "results").glob("robust-*.json")
    }
    pending = [task for task in sorted(tasks, key=_task_order) if task.task_id not in completed]
    bootstraps = int(
        config["pilot_bootstrap_replicates"]
        if mode == "pilot"
        else config["bootstrap_replicates"]
    )
    print(
        f"{mode}: {len(tasks)} expected, {len(completed)} existing, "
        f"{len(pending)} pending, workers={workers}, bootstraps={bootstraps}",
        flush=True,
    )

    def persist(row: dict[str, object], index: int) -> None:
        task_id = str(row["task_id"])
        atomic_json(output_dir / "results" / f"{task_id}.json", row)
        print(
            f"[{index}/{len(pending)}] {task_id} {row.get('status')}",
            flush=True,
        )

    if workers == 1:
        for index, task in enumerate(pending, start=1):
            persist(run_task(config, task, bootstrap_replicates=bootstraps), index)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_worker, config, asdict(task), bootstraps): task
                for task in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    row = future.result()
                except Exception as error:  # noqa: BLE001
                    row = {
                        **asdict(task),
                        "schema_version": "finite_sample_robustness_row_v1",
                        "status": "ALGORITHM_FAILED",
                        "stage": "worker_process",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "catastrophic_failure": True,
                    }
                persist(row, index)

    observed = list((output_dir / "results").glob("robust-*.json"))
    statuses: dict[str, int] = {}
    for path in observed:
        status = str(json.loads(path.read_text(encoding="utf-8")).get("status"))
        statuses[status] = statuses.get(status, 0) + 1
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["observed_task_count"] = len(observed)
    manifest["status_counts"] = statuses
    manifest["status"] = "COMPLETE" if len(observed) == len(tasks) else "INCOMPLETE"
    atomic_json(manifest_path, manifest)
    atomic_json(
        output_dir / "run_summary.json",
        {
            "mode": mode,
            "expected_task_count": len(tasks),
            "observed_task_count": len(observed),
            "status_counts": statuses,
            "complete": len(observed) == len(tasks),
        },
    )
    return manifest
