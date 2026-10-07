"""Adapters connecting public unknown-graph evidence to the generic scheduler."""

from dataclasses import dataclass, field

import numpy as np

from graphs.interventions import ShiftScaleIntervention
from graphs.pdag import PartiallyDirectedGraph
from graphs.problem_gen import PublicProblemView

from .graph_updates import update_graph_from_probe
from .local_losses import CompositeGaussianMismatchProvider
from .losses import InterventionLoss
from .probe_outcomes import ProbeEnvironment, ProbeOutcome
from .scheduler import (
    AlgorithmStateView,
    ExplorationObservation,
    MatchingEnvironment,
)
from .types import InterventionBackground, ParameterCoordinate, ShiftProbe
from .unresolved_state import (
    MismatchProvider,
    UnresolvedStateKnowledge,
    analyze_unresolved_state,
)

_StateToken = tuple[int, int, bytes, bytes]


@dataclass(frozen=True)
class GraphEvidenceRecord:
    """Auditable graph evidence tied to its accepted intervention background."""

    background: InterventionBackground
    proposal: ShiftProbe
    outcome: ProbeOutcome


@dataclass
class UnknownGraphMatchingKnowledge:
    """Maintain public PDAG knowledge across scheduler state transitions.

    One immutable :class:`UnresolvedStateKnowledge` snapshot is cached per
    scheduler state. This prevents a stateful mismatch provider from advancing
    separately for the unresolved, actionable, and policy queries made during
    one transition.

    Each certified unresolved source exposes a shift coordinate. It exposes a
    log-scale coordinate only after the public PDAG contains a directed parent.
    """

    initial_graph: PartiallyDirectedGraph
    mismatch_provider: MismatchProvider
    _graph: PartiallyDirectedGraph = field(init=False, repr=False)
    _graph_history: list[PartiallyDirectedGraph] = field(
        init=False,
        repr=False,
    )
    _cached_token: _StateToken | None = field(default=None, init=False, repr=False)
    _cached_snapshot: UnresolvedStateKnowledge | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _evidence_history: list[GraphEvidenceRecord] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate dependencies and initialize immutable graph history."""
        if not isinstance(self.initial_graph, PartiallyDirectedGraph):
            raise TypeError("initial_graph must be a PartiallyDirectedGraph.")
        if not isinstance(self.mismatch_provider, MismatchProvider):
            raise TypeError("mismatch_provider must implement MismatchProvider.")
        self._graph = self.initial_graph.copy()
        self._graph_history = [self._graph]

    @property
    def graph(self) -> PartiallyDirectedGraph:
        """Return the current immutable public PDAG snapshot."""
        return self._graph

    @property
    def graph_history(self) -> tuple[PartiallyDirectedGraph, ...]:
        """Return every committed public PDAG snapshot in order."""
        return tuple(self._graph_history)

    @property
    def evidence_history(self) -> tuple[GraphEvidenceRecord, ...]:
        """Return all accepted graph evidence with collection backgrounds."""
        return tuple(self._evidence_history)

    def current_background_evidence(
        self,
        candidate: ShiftScaleIntervention,
    ) -> tuple[GraphEvidenceRecord, ...]:
        """Return numerical evidence collected under exactly ``candidate``."""
        identifier = InterventionBackground.from_intervention(candidate).identifier
        return tuple(
            record
            for record in self._evidence_history
            if record.background.identifier == identifier
        )

    @property
    def current_snapshot(self) -> UnresolvedStateKnowledge | None:
        """Return the last computed unresolved snapshot, if one is cached."""
        return self._cached_snapshot

    def unresolved_variables_remain(self, state: AlgorithmStateView) -> bool:
        """Return whether the current public mismatch set is nonempty."""
        return bool(self._snapshot(state).unresolved_nodes)

    def actionable_coordinates(
        self,
        state: AlgorithmStateView,
    ) -> tuple[ParameterCoordinate, ...]:
        """Return graph-valid coordinates for certified unresolved sources."""
        snapshot = self._snapshot(state)
        return tuple(
            ParameterCoordinate(node, kind)
            for node in sorted(snapshot.certified_actionable_sources)
            for kind in (
                ("shift", "log_scale") if self._graph.parents_of(node) else ("shift",)
            )
        )

    def proposal_local_loss(
        self,
        coordinate: ParameterCoordinate,
        candidate: ShiftScaleIntervention,
        state: AlgorithmStateView,
    ) -> float:
        """Score one proposal through the configured composite local loss."""
        if not isinstance(self.mismatch_provider, CompositeGaussianMismatchProvider):
            return state.current_loss
        return self.mismatch_provider.local_loss.coordinate_loss(
            coordinate,
            state.problem_view,
            candidate,
            self._graph,
        )

    def accepted_background_changed(
        self,
        previous: ShiftScaleIntervention,
        current: ShiftScaleIntervention,
    ) -> None:
        """Invalidate numerical caches while retaining graph orientations."""
        self._cached_token = None
        self._cached_snapshot = None
        if isinstance(self.mismatch_provider, CompositeGaussianMismatchProvider):
            self.mismatch_provider.local_loss.invalidate_background(previous, current)

    def exploration_knowledge(
        self,
        state: AlgorithmStateView,
    ) -> UnresolvedStateKnowledge:
        """Expose the immutable unresolved snapshot expected by graph policies."""
        return self._snapshot(state)

    def update_from_probe(
        self,
        observation: ExplorationObservation,
        state: AlgorithmStateView,
    ) -> None:
        """Commit one transactional PDAG update from a public probe outcome."""
        if not isinstance(observation, ExplorationObservation):
            raise TypeError("observation must be an ExplorationObservation.")
        token = self._state_token(state)
        if self._cached_token != token or self._cached_snapshot is None:
            raise ValueError(
                "probe state must match the most recent unresolved snapshot."
            )
        if observation.outcome is None:
            raise ValueError("unknown-graph knowledge requires a public ProbeOutcome.")
        if not observation.proposal.targets <= self._cached_snapshot.unresolved_nodes:
            raise ValueError("probe targets must belong to the unresolved snapshot.")

        expected_intervention = observation.proposal.apply_to(
            state.current_candidate,
            max_probe_targets=len(observation.proposal.targets),
        )
        if not np.array_equal(
            observation.intervention.shift,
            expected_intervention.shift,
        ) or not np.array_equal(
            observation.intervention.scale,
            expected_intervention.scale,
        ):
            raise ValueError(
                "probe observation must use the current accepted candidate "
                "as its temporary background."
            )

        updated = update_graph_from_probe(self._graph, observation.outcome)
        if not self._graph.directed_edges <= updated.directed_edges:
            raise RuntimeError("a graph update must not remove directed evidence.")
        if not updated.undirected_edges <= self._graph.undirected_edges:
            raise RuntimeError("a graph update must not add undirected evidence.")
        if observation.background is None:
            raise RuntimeError("probe observation background was not initialized.")
        self._evidence_history.append(
            GraphEvidenceRecord(
                observation.background,
                observation.proposal,
                observation.outcome,
            )
        )
        self._graph = updated
        self._graph_history.append(updated)
        self._cached_token = None
        self._cached_snapshot = None

    def _snapshot(self, state: AlgorithmStateView) -> UnresolvedStateKnowledge:
        """Return one cached, internally consistent public state analysis."""
        token = self._state_token(state)
        if token != self._cached_token:
            if isinstance(
                self.mismatch_provider,
                CompositeGaussianMismatchProvider,
            ):
                unresolved = self.mismatch_provider.unresolved_nodes_for_graph(
                    state.problem_view,
                    state.current_candidate,
                    self._graph,
                )
            else:
                unresolved = self.mismatch_provider.unresolved_nodes(
                    state.problem_view,
                    state.current_candidate,
                )
            self._cached_snapshot = analyze_unresolved_state(
                self._graph,
                unresolved,
            )
            self._cached_token = token
        if self._cached_snapshot is None:
            raise RuntimeError("unresolved-state cache was not initialized.")
        return self._cached_snapshot

    def _state_token(self, state: AlgorithmStateView) -> _StateToken:
        """Validate a scheduler state and derive its cache identity."""
        if not isinstance(state, AlgorithmStateView):
            raise TypeError("state must be an AlgorithmStateView.")
        expected_nodes = frozenset(range(state.problem_view.n_nodes))
        if self._graph.nodes != expected_nodes:
            raise ValueError(
                "public graph nodes must equal the public problem node indices."
            )
        return (
            id(state.problem_view),
            state.step,
            state.current_candidate.shift.tobytes(),
            state.current_candidate.scale.tobytes(),
        )


@dataclass(frozen=True)
class ProbeOutcomeMatchingEnvironment:
    """Combine a public-outcome probe boundary with candidate evaluation.

    The adapter never inspects a hidden DAG or coefficient matrix. It delegates
    probe execution to ``ProbeEnvironment`` and exposes only its validated
    :class:`ProbeOutcome` to the scheduler. Candidate evaluation remains a
    separate injected environment operation.
    """

    probe_environment: ProbeEnvironment
    candidate_environment: MatchingEnvironment
    candidate_evaluation_requires_experiment: bool = field(init=False)

    def __post_init__(self) -> None:
        """Validate both independent execution dependencies."""
        if not isinstance(self.probe_environment, ProbeEnvironment):
            raise TypeError("probe_environment must implement ProbeEnvironment.")
        if not isinstance(self.candidate_environment, MatchingEnvironment):
            raise TypeError("candidate_environment must implement MatchingEnvironment.")
        requires_experiment = (
            self.candidate_environment.candidate_evaluation_requires_experiment
        )
        if not isinstance(requires_experiment, bool):
            raise TypeError(
                "candidate_evaluation_requires_experiment must be a boolean."
            )
        object.__setattr__(
            self,
            "candidate_evaluation_requires_experiment",
            requires_experiment,
        )

    def execute_probe(
        self,
        current_candidate: ShiftScaleIntervention,
        proposal: ShiftProbe,
        problem_view: PublicProblemView,
        rng: np.random.Generator,
    ) -> ProbeOutcome:
        """Execute one exact graph probe and return only its public outcome."""
        if not isinstance(current_candidate, ShiftScaleIntervention):
            raise TypeError("current_candidate must be a ShiftScaleIntervention.")
        if not isinstance(proposal, ShiftProbe):
            raise TypeError("proposal must be a ShiftProbe.")
        if not isinstance(problem_view, PublicProblemView):
            raise TypeError("problem_view must be a PublicProblemView.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        if current_candidate.shift.shape != (problem_view.n_nodes,):
            raise ValueError(
                "current candidate dimension must agree with the public problem."
            )
        if proposal.shift_increment.shape != (problem_view.n_nodes,):
            raise ValueError("probe dimension must agree with the public problem.")

        outcome = self.probe_environment.execute_probe(
            current_candidate,
            proposal,
        )
        if not isinstance(outcome, ProbeOutcome):
            raise TypeError("probe_environment must return a ProbeOutcome.")
        if outcome.targets != proposal.targets:
            raise ValueError("probe outcome targets must match the proposal.")
        return outcome

    def evaluate_candidate(
        self,
        candidate: ShiftScaleIntervention,
        problem_view: PublicProblemView,
        loss: InterventionLoss,
        rng: np.random.Generator,
    ) -> float:
        """Delegate candidate evaluation without coupling it to graph evidence."""
        return self.candidate_environment.evaluate_candidate(
            candidate,
            problem_view,
            loss,
            rng,
        )
