"""Explicitly oracle-only research baselines."""

from .analytic_gaussian_kl import OracleAnalyticGaussianKLLoss
from .batching import (
    NumericalBatchRecord,
    OracleBatchProblem,
    OracleInterventionBatchEvaluator,
    OracleInterventionBatchResult,
)
from .comparator import (
    NonRealisableGapMetrics,
    OracleComparatorResult,
    OracleComparatorSettings,
    best_found_oracle_comparator,
    nonrealisable_gap_metrics,
)
from .known_dag_baseline import (
    EmpiricalSCMEnvironment,
    OracleAnalyticSCMEnvironment,
    OracleKnownDAGKnowledge,
    OracleSCMEnvironment,
    OracleSourceProbePolicy,
    run_known_dag_oracle_baseline,
    run_known_dag_unknown_support_oracle_baseline,
    run_known_support_oracle_baseline,
)
from .local_gaussian import OracleCandidateMomentsProvider
from .mean_mismatch import OracleMeanMismatchProvider
from .noiseless_probe_environment import NoiselessOracleProbeEnvironment
from .observational_cpdag import cpdag_from_true_dag_for_simulation
from .oracle_optimizer import OracleActiveCoordinateOptimizer
from .structure_provider import OracleExactCPDAGProvider

__all__ = [
    "EmpiricalSCMEnvironment",
    "NoiselessOracleProbeEnvironment",
    "NonRealisableGapMetrics",
    "NumericalBatchRecord",
    "OracleActiveCoordinateOptimizer",
    "OracleAnalyticGaussianKLLoss",
    "OracleAnalyticSCMEnvironment",
    "OracleBatchProblem",
    "OracleCandidateMomentsProvider",
    "OracleComparatorResult",
    "OracleComparatorSettings",
    "OracleExactCPDAGProvider",
    "OracleInterventionBatchEvaluator",
    "OracleInterventionBatchResult",
    "OracleKnownDAGKnowledge",
    "OracleMeanMismatchProvider",
    "OracleSCMEnvironment",
    "OracleSourceProbePolicy",
    "best_found_oracle_comparator",
    "cpdag_from_true_dag_for_simulation",
    "nonrealisable_gap_metrics",
    "run_known_dag_oracle_baseline",
    "run_known_dag_unknown_support_oracle_baseline",
    "run_known_support_oracle_baseline",
]
