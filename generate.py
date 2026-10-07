"""Generate reproducible shift-scale causal matching problem instances."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from graphs import (
    DEFAULT_SUPPORT_ATOL,
    GRAPH_GENERATORS,
    AlgorithmSeeds,
    GeneratedProblem,
    LinearGaussianSCM,
    PublicProblemView,
    ShiftScaleIntervention,
    SimulationGroundTruth,
    SimulationSeeds,
    derive_noise_variance_seed,
    generate_graph,
    generate_noise_variances,
    generate_realisable_target,
    sample_coefficients,
    validate_noise_variance_regime,
    validate_support_atol,
)
from graphs.DAG_gen import DEFAULT_ERDOS_RENYI_EDGE_PROBABILITY

LEGACY_FORMAT_VERSION = 1
SPLIT_FORMAT_VERSION = 2
FORMAT_VERSION = 3
_PUBLIC_PART = "public"
_GROUND_TRUTH_PART = "ground_truth"
_PRIVATE_GENERATION_CONFIG_KEYS = frozenset({"seed", "target_interventions_path"})
_REPRODUCIBILITY_KIND = "semantic"


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for a reproducible collection of generated problems."""

    graph_type: str
    n_nodes: int
    target_support_size: int
    dataset_size: int
    n_observational_samples: int
    seed: int
    output_dir: Path
    target_interventions_path: Path
    edge_probability: float = DEFAULT_ERDOS_RENYI_EDGE_PROBABILITY
    attachment_edges: int = 2
    rooted_tree_root: int | None = None
    permute_path: bool = False
    clique_tree_degree: int = 3
    min_clique_size: int = 3
    max_clique_size: int = 5
    coefficient_min_abs: float = 0.25
    coefficient_max_abs: float = 1.0
    support_atol: float = DEFAULT_SUPPORT_ATOL
    noise_variance_regime: str = "homoskedastic"
    noise_variance_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate generation-wide values before any filesystem writes."""
        if self.graph_type not in GRAPH_GENERATORS:
            supported = ", ".join(GRAPH_GENERATORS)
            raise ValueError(
                f"Unsupported graph type {self.graph_type!r}. "
                f"Supported graph types: {supported}."
            )

        positive_counts = {
            "n_nodes": self.n_nodes,
            "dataset_size": self.dataset_size,
        }
        for name, value in positive_counts.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 1:
                raise ValueError(f"{name} must be positive.")

        nonnegative_counts = {
            "target_support_size": self.target_support_size,
            "n_observational_samples": self.n_observational_samples,
            "seed": self.seed,
        }
        for name, value in nonnegative_counts.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")
        if self.target_support_size > self.n_nodes:
            raise ValueError("target_support_size must not exceed n_nodes.")

        if not isinstance(self.output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path.")
        if not isinstance(self.target_interventions_path, Path):
            raise TypeError("target_interventions_path must be a pathlib.Path.")
        validate_support_atol(self.support_atol)
        validate_noise_variance_regime(self.noise_variance_regime)
        if self.noise_variance_seed is not None:
            if isinstance(self.noise_variance_seed, bool) or not isinstance(
                self.noise_variance_seed, int
            ):
                raise TypeError("noise_variance_seed must be an integer or None.")
            if self.noise_variance_seed < 0:
                raise ValueError("noise_variance_seed must be nonnegative.")


def load_target_interventions(
    config: GenerationConfig,
) -> tuple[ShiftScaleIntervention, ...]:
    """Load explicit hidden target interventions without inventing a prior.

    The NPZ archive must contain real arrays named ``shift`` and ``scale``.
    For one problem, each array may have shape ``(n_nodes,)``. Otherwise both
    arrays must have shape ``(dataset_size, n_nodes)``.
    """
    if not isinstance(config, GenerationConfig):
        raise TypeError("config must be a GenerationConfig.")

    try:
        with np.load(config.target_interventions_path, allow_pickle=False) as archive:
            missing = {"shift", "scale"} - set(archive.files)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(
                    f"target intervention archive is missing array(s): {names}."
                )
            shifts = np.array(archive["shift"], copy=True)
            scales = np.array(archive["scale"], copy=True)
    except OSError as error:
        raise ValueError(
            f"could not read target intervention archive "
            f"{config.target_interventions_path}."
        ) from error

    if shifts.ndim == 1 and scales.ndim == 1 and config.dataset_size == 1:
        shifts = shifts[np.newaxis, :]
        scales = scales[np.newaxis, :]

    expected_shape = (config.dataset_size, config.n_nodes)
    if shifts.shape != expected_shape or scales.shape != expected_shape:
        raise ValueError(
            "target shift and scale arrays must both have shape "
            f"{expected_shape}; received {shifts.shape} and {scales.shape}."
        )

    interventions = tuple(
        ShiftScaleIntervention(shifts[index], scales[index])
        for index in range(config.dataset_size)
    )
    for index, intervention in enumerate(interventions):
        target_nodes = intervention.targeted_nodes(config.support_atol)
        if len(target_nodes) != config.target_support_size:
            raise ValueError(
                f"target intervention {index} has support size "
                f"{len(target_nodes)}, expected "
                f"{config.target_support_size}."
            )
    return interventions


def _graph_parameters(config: GenerationConfig) -> dict[str, object]:
    """Return only parameters accepted by the selected graph family."""
    if config.graph_type == "path":
        return {"permute": config.permute_path}
    if config.graph_type == "rooted_tree":
        return {"root": config.rooted_tree_root}
    if config.graph_type in {"erdos_renyi", "moralized_erdos_renyi"}:
        return {"edge_probability": config.edge_probability}
    if config.graph_type == "barabasi_albert":
        return {"attachment_edges": config.attachment_edges}
    if config.graph_type == "complete":
        return {"node_order": None}
    if config.graph_type == "tree_of_cliques":
        return {
            "branching_degree": config.clique_tree_degree,
            "min_clique_size": config.min_clique_size,
            "max_clique_size": config.max_clique_size,
        }
    return {}


def _derive_problem_seeds(
    root_seed: int,
) -> tuple[SimulationSeeds, AlgorithmSeeds]:
    """Derive independent, recorded component seeds from one instance seed."""
    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise TypeError("instance_seed must be an integer.")
    if root_seed < 0:
        raise ValueError("instance_seed must be nonnegative.")

    child_sequences = np.random.SeedSequence(root_seed).spawn(7)
    derived_seeds = tuple(
        int(sequence.generate_state(1, dtype=np.uint64)[0])
        for sequence in child_sequences
    )
    simulation_seeds = SimulationSeeds(
        root_seed=root_seed,
        graph_generation_seed=derived_seeds[0],
        coefficient_generation_seed=derived_seeds[1],
        observational_sampling_seed=derived_seeds[2],
        target_intervention_seed=derived_seeds[3],
        target_sampling_seed=derived_seeds[4],
    )
    algorithm_seeds = AlgorithmSeeds(
        exploration_seed=derived_seeds[5],
        candidate_evaluation_seed=derived_seeds[6],
    )
    return simulation_seeds, algorithm_seeds


def generate_problem(
    config: GenerationConfig,
    target_intervention: ShiftScaleIntervention,
    instance_seed: int,
) -> GeneratedProblem:
    """Generate one problem with independent component RNG streams."""
    if not isinstance(config, GenerationConfig):
        raise TypeError("config must be a GenerationConfig.")
    if not isinstance(target_intervention, ShiftScaleIntervention):
        raise TypeError("target_intervention must be a ShiftScaleIntervention.")

    simulation_seeds, algorithm_seeds = _derive_problem_seeds(instance_seed)
    noise_variance_seed = (
        config.noise_variance_seed
        if config.noise_variance_seed is not None
        else derive_noise_variance_seed(instance_seed)
    )
    simulation_seeds = replace(
        simulation_seeds,
        noise_variance_seed=noise_variance_seed,
    )
    dag = generate_graph(
        config.graph_type,
        config.n_nodes,
        np.random.default_rng(simulation_seeds.graph_generation_seed),
        **_graph_parameters(config),
    )
    coefficients = sample_coefficients(
        dag,
        np.random.default_rng(simulation_seeds.coefficient_generation_seed),
        config.coefficient_min_abs,
        config.coefficient_max_abs,
    )
    environment = LinearGaussianSCM(
        dag,
        coefficients,
        noise_variances=generate_noise_variances(
            config.n_nodes,
            config.noise_variance_regime,
            noise_variance_seed,
        ),
    )
    target = generate_realisable_target(environment, target_intervention)
    observational_samples = environment.sample_observational(
        config.n_observational_samples,
        np.random.default_rng(simulation_seeds.observational_sampling_seed),
    )
    return GeneratedProblem(
        public=PublicProblemView(
            observational_samples=observational_samples,
            target_mean=target.mean,
            target_covariance=target.covariance,
            current_intervention=ShiftScaleIntervention.identity(config.n_nodes),
            algorithm_seeds=algorithm_seeds,
        ),
        ground_truth=SimulationGroundTruth(
            scm=environment,
            generating_intervention=target_intervention,
            seeds=simulation_seeds,
        ),
    )


def _configuration_metadata(config: GenerationConfig) -> dict[str, object]:
    """Convert generation configuration to JSON-compatible values."""
    values = asdict(config)
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in values.items()
    }


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    """Write deterministic JSON metadata."""
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_metadata(metadata_path: Path) -> dict[str, object]:
    """Load metadata shared by legacy and split problem formats."""
    if not isinstance(metadata_path, Path):
        raise TypeError("metadata_path must be a pathlib.Path.")
    metadata: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("problem metadata must be a JSON object.")
    version = metadata.get("format_version")
    if version not in {
        LEGACY_FORMAT_VERSION,
        SPLIT_FORMAT_VERSION,
        FORMAT_VERSION,
    }:
        raise ValueError(f"unsupported format version {version!r}.")
    return metadata


def _load_arrays(
    metadata_path: Path,
    metadata: dict[str, object],
    required_arrays: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Load only explicitly requested arrays from one NPZ archive."""
    arrays_file = metadata.get("arrays_file")
    if not isinstance(arrays_file, str):
        raise TypeError("problem metadata must contain an arrays_file string.")

    with np.load(metadata_path.parent / arrays_file, allow_pickle=False) as archive:
        missing = set(required_arrays) - set(archive.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"problem archive is missing array(s): {names}.")
        return {name: np.array(archive[name], copy=True) for name in required_arrays}


def _load_generation_config(metadata: dict[str, object]) -> dict[str, object]:
    """Validate and return serialized generation configuration."""
    generation_config = metadata.get("generation_config")
    if not isinstance(generation_config, dict):
        raise TypeError("problem metadata must contain a generation_config object.")
    return generation_config


def _load_algorithm_seeds(metadata: dict[str, object]) -> AlgorithmSeeds:
    """Load public algorithm RNG streams, including legacy derivation."""
    if metadata.get("format_version") == LEGACY_FORMAT_VERSION:
        _, algorithm_seeds = _derive_problem_seeds(metadata.get("random_seed"))
        return algorithm_seeds

    serialized = metadata.get("algorithm_seeds")
    if not isinstance(serialized, dict):
        raise TypeError("public metadata must contain an algorithm_seeds object.")
    try:
        return AlgorithmSeeds(**serialized)
    except TypeError as error:
        raise TypeError(
            "algorithm_seeds must contain the expected seed fields."
        ) from error


def _load_simulation_seeds(metadata: dict[str, object]) -> SimulationSeeds:
    """Load private simulation RNG streams, including legacy derivation."""
    if metadata.get("format_version") == LEGACY_FORMAT_VERSION:
        simulation_seeds, _ = _derive_problem_seeds(metadata.get("random_seed"))
        return simulation_seeds

    serialized = metadata.get("simulation_seeds")
    if not isinstance(serialized, dict):
        raise TypeError("ground-truth metadata must contain a simulation_seeds object.")
    try:
        return SimulationSeeds(**serialized)
    except TypeError as error:
        raise TypeError(
            "simulation_seeds must contain the expected seed fields."
        ) from error


def _public_generation_config(
    generation_config: dict[str, object],
) -> dict[str, object]:
    """Remove values that can reconstruct or locate hidden simulation state."""
    return {
        name: value
        for name, value in generation_config.items()
        if name not in _PRIVATE_GENERATION_CONFIG_KEYS
    }


def save_problem(
    problem: GeneratedProblem,
    output_dir: Path,
    instance_index: int,
    generation_config: dict[str, object],
) -> tuple[Path, Path]:
    """Save public and oracle-only state in separate format-version-2 files.

    The returned paths identify the public NPZ and JSON files. Ground-truth
    paths use the same stem with a ``ground_truth`` part name and are never
    referenced by public metadata.
    """
    if not isinstance(problem, GeneratedProblem):
        raise TypeError("problem must be a GeneratedProblem.")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path.")
    if isinstance(instance_index, bool) or not isinstance(instance_index, int):
        raise TypeError("instance_index must be an integer.")
    if instance_index < 0:
        raise ValueError("instance_index must be nonnegative.")
    if not isinstance(generation_config, dict):
        raise TypeError("generation_config must be a dictionary.")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"problem_{instance_index:06d}"
    public_arrays_path = output_dir / f"{stem}.{_PUBLIC_PART}.npz"
    public_metadata_path = output_dir / f"{stem}.{_PUBLIC_PART}.json"
    ground_truth_arrays_path = output_dir / f"{stem}.{_GROUND_TRUTH_PART}.npz"
    ground_truth_metadata_path = output_dir / f"{stem}.{_GROUND_TRUTH_PART}.json"
    output_paths = (
        public_arrays_path,
        public_metadata_path,
        ground_truth_arrays_path,
        ground_truth_metadata_path,
    )
    if any(path.exists() for path in output_paths):
        raise FileExistsError(f"refusing to overwrite existing problem {stem}.")

    public = problem.public
    truth = problem.ground_truth
    edges = np.asarray(sorted(truth.scm.dag.edges), dtype=np.int64).reshape(-1, 2)

    np.savez_compressed(
        public_arrays_path,
        observational_samples=public.observational_samples,
        target_mean=public.target_mean,
        target_covariance=public.target_covariance,
        current_shift=public.current_intervention.shift,
        current_scale=public.current_intervention.scale,
    )
    np.savez_compressed(
        ground_truth_arrays_path,
        edges=edges,
        coefficients=truth.scm.coefficients,
        noise_variances=truth.scm.noise_variances,
        generating_shift=truth.generating_intervention.shift,
        generating_scale=truth.generating_intervention.scale,
    )

    public_metadata = {
        "format_version": FORMAT_VERSION,
        "problem_part": _PUBLIC_PART,
        "instance_index": instance_index,
        "arrays_file": public_arrays_path.name,
        "algorithm_seeds": asdict(public.algorithm_seeds),
        "reproducibility": _REPRODUCIBILITY_KIND,
        "generation_config": _public_generation_config(generation_config),
    }
    ground_truth_metadata = {
        "format_version": FORMAT_VERSION,
        "problem_part": _GROUND_TRUTH_PART,
        "instance_index": instance_index,
        "arrays_file": ground_truth_arrays_path.name,
        "public_metadata_file": public_metadata_path.name,
        "is_realisable": True,
        "simulation_seeds": asdict(truth.seeds),
        "reproducibility": _REPRODUCIBILITY_KIND,
        "generation_config": generation_config,
    }
    _write_metadata(public_metadata_path, public_metadata)
    _write_metadata(ground_truth_metadata_path, ground_truth_metadata)
    return public_arrays_path, public_metadata_path


def load_public_problem(
    metadata_path: Path,
) -> tuple[PublicProblemView, dict[str, object]]:
    """Load public problem state without reading hidden arrays.

    Format-version-1 archives are supported for compatibility. Only their
    public array members are accessed, despite public and hidden members sharing
    one legacy NPZ file.
    """
    metadata = _load_metadata(metadata_path)
    if (
        metadata.get("format_version") in {SPLIT_FORMAT_VERSION, FORMAT_VERSION}
        and metadata.get("problem_part") != _PUBLIC_PART
    ):
        raise ValueError("format-version-2 public metadata has an invalid part.")

    arrays = _load_arrays(
        metadata_path,
        metadata,
        (
            "observational_samples",
            "target_mean",
            "target_covariance",
            "current_shift",
            "current_scale",
        ),
    )
    public = PublicProblemView(
        observational_samples=arrays["observational_samples"],
        target_mean=arrays["target_mean"],
        target_covariance=arrays["target_covariance"],
        current_intervention=ShiftScaleIntervention(
            arrays["current_shift"],
            arrays["current_scale"],
        ),
        algorithm_seeds=_load_algorithm_seeds(metadata),
    )
    return public, _public_generation_config(_load_generation_config(metadata))


def _default_ground_truth_metadata_path(public_metadata_path: Path) -> Path:
    """Derive a private sibling path without recording it in public metadata."""
    public_suffix = f".{_PUBLIC_PART}.json"
    if not public_metadata_path.name.endswith(public_suffix):
        raise ValueError(
            "split-format public metadata filename must end with "
            f"{public_suffix!r}."
        )
    stem = public_metadata_path.name[: -len(public_suffix)]
    return public_metadata_path.with_name(f"{stem}.{_GROUND_TRUTH_PART}.json")


def _build_ground_truth(
    public: PublicProblemView,
    arrays: dict[str, np.ndarray],
    seeds: SimulationSeeds,
) -> SimulationGroundTruth:
    """Reconstruct validated oracle state from hidden serialized arrays."""
    coefficients = arrays["coefficients"]
    if coefficients.ndim != 2 or coefficients.shape != (
        public.n_nodes,
        public.n_nodes,
    ):
        raise ValueError("saved coefficients must match the public dimension.")
    edges = arrays["edges"]
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("saved edges must have shape (n_edges, 2).")

    dag = nx.DiGraph()
    dag.add_nodes_from(range(public.n_nodes))
    dag.add_edges_from((int(parent), int(child)) for parent, child in edges)
    return SimulationGroundTruth(
        scm=LinearGaussianSCM(
            dag,
            coefficients,
            noise_variances=arrays.get(
                "noise_variances",
                np.ones(public.n_nodes, dtype=float),
            ),
        ),
        generating_intervention=ShiftScaleIntervention(
            arrays["generating_shift"],
            arrays["generating_scale"],
        ),
        seeds=seeds,
    )


def load_problem_with_ground_truth(
    metadata_path: Path,
    ground_truth_metadata_path: Path | None = None,
) -> tuple[GeneratedProblem, dict[str, object]]:
    """Load a combined problem through an explicitly oracle-only interface."""
    public, _ = load_public_problem(metadata_path)
    public_metadata = _load_metadata(metadata_path)
    version = public_metadata["format_version"]

    if version == LEGACY_FORMAT_VERSION:
        if ground_truth_metadata_path is not None:
            raise ValueError(
                "legacy format stores ground truth in the public metadata archive."
            )
        ground_truth_arrays = _load_arrays(
            metadata_path,
            public_metadata,
            (
                "edges",
                "coefficients",
                "generating_shift",
                "generating_scale",
            ),
        )
        private_metadata = public_metadata
    else:
        if ground_truth_metadata_path is None:
            ground_truth_metadata_path = _default_ground_truth_metadata_path(
                metadata_path
            )
        elif not isinstance(ground_truth_metadata_path, Path):
            raise TypeError("ground_truth_metadata_path must be a pathlib.Path.")

        private_metadata = _load_metadata(ground_truth_metadata_path)
        if private_metadata.get("format_version") != version:
            raise ValueError(
                "ground-truth metadata version must match public metadata."
            )
        if private_metadata.get("problem_part") != _GROUND_TRUTH_PART:
            raise ValueError("ground-truth metadata has an invalid part.")
        if private_metadata.get("instance_index") != public_metadata.get(
            "instance_index"
        ):
            raise ValueError("public and ground-truth instance indices must agree.")
        if private_metadata.get("public_metadata_file") != metadata_path.name:
            raise ValueError(
                "ground-truth metadata references a different public file."
            )
        required_ground_truth_arrays = (
            "edges",
            "coefficients",
            "generating_shift",
            "generating_scale",
        )
        if version == FORMAT_VERSION:
            required_ground_truth_arrays += ("noise_variances",)
        ground_truth_arrays = _load_arrays(
            ground_truth_metadata_path,
            private_metadata,
            required_ground_truth_arrays,
        )

    generation_config = _load_generation_config(private_metadata)
    simulation_seeds = _load_simulation_seeds(private_metadata)
    problem = GeneratedProblem(
        public=public,
        ground_truth=_build_ground_truth(
            public,
            ground_truth_arrays,
            simulation_seeds,
        ),
    )
    return problem, generation_config


def load_problem(
    metadata_path: Path,
) -> tuple[GeneratedProblem, dict[str, object]]:
    """Backward-compatible name for the explicit oracle loader."""
    return load_problem_with_ground_truth(metadata_path)


def generate_dataset(config: GenerationConfig) -> tuple[Path, ...]:
    """Generate and save all configured problem instances."""
    if not isinstance(config, GenerationConfig):
        raise TypeError("config must be a GenerationConfig.")

    target_interventions = load_target_interventions(config)
    if config.output_dir.exists():
        if not config.output_dir.is_dir():
            raise NotADirectoryError(f"{config.output_dir} is not a directory.")
        if any(config.output_dir.iterdir()):
            raise FileExistsError(
                f"output directory {config.output_dir} must be empty."
            )
    else:
        config.output_dir.mkdir(parents=True)

    child_sequences = np.random.SeedSequence(config.seed).spawn(config.dataset_size)
    instance_seeds = [
        int(sequence.generate_state(1, dtype=np.uint64)[0])
        for sequence in child_sequences
    ]
    metadata_config = _configuration_metadata(config)
    metadata_paths = []
    for index, (target_intervention, instance_seed) in enumerate(
        zip(target_interventions, instance_seeds, strict=True)
    ):
        problem = generate_problem(config, target_intervention, instance_seed)
        _, metadata_path = save_problem(
            problem,
            config.output_dir,
            index,
            metadata_config,
        )
        metadata_paths.append(metadata_path)
    return tuple(metadata_paths)


def build_parser() -> argparse.ArgumentParser:
    """Build the descriptive generation command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate shift-scale causal matching problems."
    )
    parser.add_argument("--graph-type", choices=tuple(GRAPH_GENERATORS), required=True)
    parser.add_argument("--n-nodes", type=int, required=True)
    parser.add_argument("--edge-probability", type=float, default=0.2)
    parser.add_argument("--attachment-edges", type=int, default=2)
    parser.add_argument("--rooted-tree-root", type=int)
    parser.add_argument(
        "--permute-path",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--clique-tree-degree", type=int, default=3)
    parser.add_argument("--min-clique-size", type=int, default=3)
    parser.add_argument("--max-clique-size", type=int, default=5)
    parser.add_argument("--coefficient-min-abs", type=float, default=0.25)
    parser.add_argument("--coefficient-max-abs", type=float, default=1.0)
    parser.add_argument("--support-atol", type=float, default=DEFAULT_SUPPORT_ATOL)
    parser.add_argument("--target-support-size", type=int, required=True)
    parser.add_argument(
        "--target-interventions",
        dest="target_interventions_path",
        type=Path,
        required=True,
        help="NPZ containing explicit shift and scale arrays.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1,
        help="Number of independent problem instances.",
    )
    parser.add_argument(
        "--n-observational-samples",
        type=int,
        default=10_000,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise-variance-regime",
        choices=(
            "homoskedastic",
            "heterogeneous_moderate",
            "heterogeneous_strong",
        ),
        default="homoskedastic",
    )
    parser.add_argument("--noise-variance-seed", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the generation CLI."""
    namespace = build_parser().parse_args(argv)
    config = GenerationConfig(**vars(namespace))
    metadata_paths = generate_dataset(config)
    print(
        f"Generated {len(metadata_paths)} problem(s) in {config.output_dir} "
        f"using format version {FORMAT_VERSION}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
