"""Random legal exploration over public unresolved graph components."""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from matching.probes import ShiftProbe
from matching.scheduler import AlgorithmStateView, BudgetView
from matching.unresolved_state import UnresolvedStateKnowledge


@dataclass(frozen=True)
class RandomExplorationPolicy:
    """Sample a bounded pure-shift probe from one top unresolved component."""

    max_probe_targets: int
    shift_magnitude: float

    def __post_init__(self) -> None:
        """Validate the explicit probe-design configuration."""
        if isinstance(self.max_probe_targets, bool) or not isinstance(
            self.max_probe_targets,
            int,
        ):
            raise TypeError("max_probe_targets must be an integer.")
        if self.max_probe_targets <= 0:
            raise ValueError("max_probe_targets must be positive.")
        if isinstance(self.shift_magnitude, bool) or not isinstance(
            self.shift_magnitude,
            Real,
        ):
            raise TypeError("shift_magnitude must be a real number.")
        magnitude = float(self.shift_magnitude)
        if not np.isfinite(magnitude):
            raise ValueError("shift_magnitude must be finite.")
        if magnitude == 0.0:
            raise ValueError("shift_magnitude must be nonzero.")
        object.__setattr__(self, "shift_magnitude", magnitude)

    def propose_probe(
        self,
        public_state: AlgorithmStateView,
        knowledge: UnresolvedStateKnowledge,
        budget: BudgetView,
        rng: np.random.Generator,
        *,
        allow_when_actionable: bool = False,
    ) -> ShiftProbe | None:
        """Return a random legal probe, or ``None`` when exploration is gated."""
        if not isinstance(public_state, AlgorithmStateView):
            raise TypeError("public_state must be an AlgorithmStateView.")
        if not isinstance(knowledge, UnresolvedStateKnowledge):
            raise TypeError("knowledge must be an UnresolvedStateKnowledge.")
        if not isinstance(budget, BudgetView):
            raise TypeError("budget must be a BudgetView.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        if not isinstance(allow_when_actionable, bool):
            raise TypeError("allow_when_actionable must be a boolean.")

        n_nodes = public_state.problem_view.n_nodes
        if any(node >= n_nodes for node in knowledge.unresolved_nodes):
            raise ValueError("knowledge nodes must fit the public problem dimension.")
        if budget.remaining_experiment_trials == 0:
            return None
        if not knowledge.unresolved_nodes:
            return None
        if knowledge.certified_actionable_sources and not allow_when_actionable:
            return None

        components = knowledge.top_unresolved_chain_components
        if not components:
            return None
        component_position = int(rng.integers(len(components)))
        component = components[component_position]
        component_nodes = np.asarray(sorted(component), dtype=int)
        target_count = min(self.max_probe_targets, component_nodes.size)
        selected = rng.choice(
            component_nodes,
            size=target_count,
            replace=False,
        )
        targets = frozenset(int(node) for node in np.atleast_1d(selected))

        increment = np.zeros(n_nodes, dtype=float)
        increment[list(targets)] = self.shift_magnitude
        component_label = ",".join(str(node) for node in sorted(component))
        target_label = ",".join(str(node) for node in sorted(targets))
        return ShiftProbe(
            shift_increment=increment,
            targets=targets,
            reason=(
                "random-legal-exploration:"
                f"component={component_label};targets={target_label}"
            ),
        )
