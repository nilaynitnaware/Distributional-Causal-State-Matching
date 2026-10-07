"""Frozen one-factor design and deterministic identifiers for robustness v1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/finite_sample_robustness_v1.json"

CampaignName = Literal[
    "campaign_a_n_obs",
    "campaign_b_n_target",
    "campaign_c_estimated_cpdag",
    "campaign_d_controlled_corruption",
    "campaign_e_nonrealisable",
    "campaign_f_large_n",
]


def stable_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def derive_seed(base_seed: int, *parts: object) -> int:
    """Match the Stage-B SHA-256/little-endian seed derivation."""
    material = "|".join(map(str, (base_seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


@dataclass(frozen=True)
class CampaignTask:
    task_id: str
    campaign: CampaignName
    source_problem_id: str
    seed: int
    n: int
    graph_index: int
    graph_family: str
    replicate: int
    target_family: str
    graph_condition: str
    N_obs: int
    N_target: int
    N_int: int
    N_graph: int | None = None
    corruption_type: str | None = None
    corruption_severity: str | None = None
    graph_estimation_seed: int | None = None
    corruption_seed: int | None = None
    method: str = "finite_sample_algorithm1"
    policy: str = "first_unresolved"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "finite_sample_robustness_config_v1":
        raise ValueError("finite-sample robustness configuration is stale")
    return config


def graph_family_label(regime: dict[str, object]) -> str:
    if "edge_probability" in regime:
        return f"{regime['family']}_{regime['edge_probability']}"
    if "attachment_edges" in regime:
        return f"{regime['family']}_m{regime['attachment_edges']}"
    return str(regime["family"])


def _make_task(
    config: dict[str, object],
    *,
    campaign: CampaignName,
    n: int,
    graph_index: int,
    replicate: int,
    target_family: str,
    graph_condition: str,
    N_obs: int,
    N_target: int,
    N_int: int,
    N_graph: int | None = None,
    corruption_type: str | None = None,
    corruption_severity: str | None = None,
) -> CampaignTask:
    base_seed = int(config["_active_seed"])
    regime = config["graph_regimes"][graph_index]
    source_id = f"stageb-v2:seed{base_seed}:n{n}:g{graph_index}:r{replicate}"
    graph_seed = (
        derive_seed(base_seed, n, graph_index, replicate, "source_samples")
        if N_graph is not None
        else None
    )
    corruption_seed = (
        derive_seed(base_seed, source_id, "corruption", corruption_type)
        if corruption_type is not None
        else None
    )
    scientific = {
        "version": config["experiment_version"],
        "campaign": campaign,
        "source_problem_id": source_id,
        "target_family": target_family,
        "graph_condition": graph_condition,
        "N_obs": N_obs,
        "N_target": N_target,
        "N_int": N_int,
        "N_graph": N_graph,
        "corruption_type": corruption_type,
        "corruption_severity": corruption_severity,
        "method": config["method"],
        "policy": config["policy"],
    }
    return CampaignTask(
        task_id="robust-" + stable_digest(scientific)[:20],
        campaign=campaign,
        source_problem_id=source_id,
        seed=base_seed,
        n=n,
        graph_index=graph_index,
        graph_family=graph_family_label(regime),
        replicate=replicate,
        target_family=target_family,
        graph_condition=graph_condition,
        N_obs=N_obs,
        N_target=N_target,
        N_int=N_int,
        N_graph=N_graph,
        corruption_type=corruption_type,
        corruption_severity=corruption_severity,
        graph_estimation_seed=graph_seed,
        corruption_seed=corruption_seed,
        method=str(config["method"]),
        policy=str(config["policy"]),
    )


def _full_specs(config: dict[str, object]) -> list[CampaignTask]:
    full = config["full"]
    reps = range(int(full["replicates"]))
    tasks: list[CampaignTask] = []

    panel = full["campaign_a"]
    for n in panel["n"]:
        for graph_index in panel["graph_regime_indices"]:
            for replicate in reps:
                for count in panel["N_obs"]:
                    tasks.append(_make_task(
                        config, campaign="campaign_a_n_obs", n=n,
                        graph_index=graph_index, replicate=replicate,
                        target_family="realisable", graph_condition="exact_cpdag",
                        N_obs=count, N_target=panel["N_target"], N_int=panel["N_int"],
                    ))

    panel = full["campaign_b"]
    for n in panel["n"]:
        for graph_index in panel["graph_regime_indices"]:
            for replicate in reps:
                for count in panel["N_target"]:
                    tasks.append(_make_task(
                        config, campaign="campaign_b_n_target", n=n,
                        graph_index=graph_index, replicate=replicate,
                        target_family="realisable", graph_condition="exact_cpdag",
                        N_obs=panel["N_obs"], N_target=count, N_int=panel["N_int"],
                    ))

    panel = full["campaign_c"]
    for n in panel["n"]:
        for graph_index in panel["graph_regime_indices"]:
            for replicate in reps:
                for count in panel["N_graph_and_N_obs"]:
                    tasks.append(_make_task(
                        config, campaign="campaign_c_estimated_cpdag", n=n,
                        graph_index=graph_index, replicate=replicate,
                        target_family="realisable", graph_condition="estimated_pc_cpdag",
                        N_obs=count, N_target=panel["N_target"], N_int=panel["N_int"],
                        N_graph=count,
                    ))

    panel = full["campaign_d"]
    for n in panel["n"]:
        for graph_index in panel["graph_regime_indices"]:
            for replicate in reps:
                for corruption_type in panel["corruption_types"]:
                    for severity in panel["severity_levels"]:
                        tasks.append(_make_task(
                            config, campaign="campaign_d_controlled_corruption", n=n,
                            graph_index=graph_index, replicate=replicate,
                            target_family="realisable",
                            graph_condition=f"controlled_{corruption_type}",
                            N_obs=panel["N_obs"], N_target=panel["N_target"],
                            N_int=panel["N_int"], corruption_type=corruption_type,
                            corruption_severity=severity,
                        ))

    panel = full["campaign_e"]
    for n in panel["n"]:
        for graph_index in panel["graph_regime_indices"]:
            for replicate in reps:
                for family in panel["target_families"]:
                    for count in panel["N_target"]:
                        tasks.append(_make_task(
                            config, campaign="campaign_e_nonrealisable", n=n,
                            graph_index=graph_index, replicate=replicate,
                            target_family=family, graph_condition="exact_cpdag",
                            N_obs=panel["N_obs"], N_target=count, N_int=panel["N_int"],
                        ))

    panel = full["campaign_f"]
    for n in panel["n"]:
        for graph_index in panel["graph_regime_indices"]:
            for replicate in range(int(panel["replicates"])):
                tasks.append(_make_task(
                    config, campaign="campaign_f_large_n", n=n,
                    graph_index=graph_index, replicate=replicate,
                    target_family="realisable", graph_condition="exact_cpdag",
                    N_obs=panel["N_obs"], N_target=panel["N_target"], N_int=panel["N_int"],
                ))
    return tasks


def _pilot_specs(config: dict[str, object]) -> list[CampaignTask]:
    # Explicit, small coverage grid. Large-n full-bootstrap gates are benchmarked
    # separately and are not disguised as these 16-replicate smoke tasks.
    rows = [
        ("campaign_a_n_obs", 10, 0, 0, "realisable", "exact_cpdag", 250, 10000, 10000, None, None, None),
        ("campaign_a_n_obs", 10, 0, 0, "realisable", "exact_cpdag", 10000, 10000, 10000, None, None, None),
        ("campaign_b_n_target", 10, 0, 0, "realisable", "exact_cpdag", 10000, 250, 10000, None, None, None),
        ("campaign_b_n_target", 10, 0, 0, "realisable", "exact_cpdag", 10000, 10000, 10000, None, None, None),
        ("campaign_c_estimated_cpdag", 10, 2, 0, "realisable", "estimated_pc_cpdag", 250, 10000, 10000, 250, None, None),
        ("campaign_c_estimated_cpdag", 10, 2, 0, "realisable", "estimated_pc_cpdag", 10000, 10000, 10000, 10000, None, None),
        ("campaign_d_controlled_corruption", 20, 3, 0, "realisable", "controlled_edge_deletion", 10000, 10000, 10000, None, "edge_deletion", "one"),
        ("campaign_d_controlled_corruption", 20, 3, 0, "realisable", "controlled_edge_addition", 10000, 10000, 10000, None, "edge_addition", "five_percent"),
        ("campaign_d_controlled_corruption", 20, 3, 1, "realisable", "controlled_orientation_error", 10000, 10000, 10000, None, "orientation_error", "one"),
        ("campaign_e_nonrealisable", 20, 2, 0, "realisable", "exact_cpdag", 10000, 500, 10000, None, None, None),
        ("campaign_e_nonrealisable", 20, 2, 0, "nonproportional_structural_coefficients", "exact_cpdag", 10000, 500, 10000, None, None, None),
        ("campaign_e_nonrealisable", 20, 2, 0, "external_spd_gaussian", "exact_cpdag", 10000, 500, 10000, None, None, None),
        ("campaign_e_nonrealisable", 20, 2, 0, "nonproportional_structural_coefficients", "exact_cpdag", 10000, 10000, 10000, None, None, None),
        ("campaign_e_nonrealisable", 20, 2, 0, "external_spd_gaussian", "exact_cpdag", 10000, 10000, 10000, None, None, None),
        ("campaign_f_large_n", 80, 0, 0, "realisable", "exact_cpdag", 10000, 10000, 10000, None, None, None),
        ("campaign_f_large_n", 100, 3, 0, "realisable", "exact_cpdag", 10000, 10000, 10000, None, None, None),
    ]
    return [
        _make_task(
            config, campaign=campaign, n=n, graph_index=graph_index,
            replicate=replicate, target_family=target_family,
            graph_condition=graph_condition, N_obs=n_obs, N_target=n_target,
            N_int=n_int, N_graph=n_graph, corruption_type=corruption_type,
            corruption_severity=severity,
        )
        for (
            campaign, n, graph_index, replicate, target_family, graph_condition,
            n_obs, n_target, n_int, n_graph, corruption_type, severity,
        ) in rows
    ]


def task_specs(
    config: dict[str, object], *, mode: Literal["pilot", "full"], seed: int
) -> tuple[CampaignTask, ...]:
    active = dict(config)
    active["_active_seed"] = int(seed)
    tasks = _pilot_specs(active) if mode == "pilot" else _full_specs(active)
    if len({task.task_id for task in tasks}) != len(tasks):
        raise RuntimeError("campaign task identifiers are not unique")
    if mode == "full" and len(tasks) != int(config["expected_full_tasks"]):
        raise RuntimeError(
            f"frozen full task count changed: {len(tasks)} != {config['expected_full_tasks']}"
        )
    return tuple(sorted(tasks, key=lambda task: task.task_id))


def task_payloads(tasks: tuple[CampaignTask, ...]) -> list[dict[str, object]]:
    return [asdict(task) for task in tasks]
