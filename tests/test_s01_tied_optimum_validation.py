"""Focused S01-B3 adversarial tests frozen in frozen_cases.json."""

from __future__ import annotations

import numpy as np
import pytest

from research.matching_query_complexity import infer_outward_tree_root
from research.s01_tree_population import (
    CertificateCache,
    enumerate_tree_root_class,
    paired_policy_path,
    select_one_step,
)
from outputs.s01_b3_tied_optimum_validation_v1.tied_optimum_reference import (
    evaluate_case,
    load_frozen,
)


FROZEN = load_frozen()
CASES = {case["id"]: case for case in FROZEN["cases"]}
CONVENTIONS = FROZEN["numerical_conventions"]


def _assert_reference_equals_production(case_id: str):
    case = CASES[case_id]
    candidates, _, reference, production, certificate = evaluate_case(
        case, CONVENTIONS
    )
    for candidate in candidates:
        root = infer_outward_tree_root(candidate)
        actual = production[candidate.id]
        expected = reference[root]
        assert actual.optimal_kl == pytest.approx(expected.optimal_kl, abs=2e-12)
        assert len(actual.actions) == len(expected.actions)
        for action in expected.actions:
            assert any(
                np.allclose(action.shift, item.shift, atol=1e-12, rtol=0.0)
                and np.allclose(action.scale, item.scale, atol=1e-12, rtol=0.0)
                for item in actual.actions
            )
    assert (certificate is not None) is bool(case["expected_common"])
    if certificate is not None:
        assert certificate.commanded_support <= int(CONVENTIONS["budget"])
        assert len(certificate.candidate_losses) == len(candidates)
        assert all(
            item.absolute_residual <= item.tolerance
            for item in certificate.candidate_losses
        )
    return candidates, production, certificate


def test_realisable_nonroot_shift_scale_tie_is_complete_and_certified() -> None:
    _, production, certificate = _assert_reference_equals_production(
        "r1_two_dag_nonroot_shift_scale_tie"
    )
    assert certificate is not None
    assert certificate.command.shift == pytest.approx((0.0, 1.0))
    assert certificate.command.scale == pytest.approx((1.0, 1.0))
    assert sorted(len(item.actions) for item in production.values()) == [2, 2]


def test_predetermined_perturbation_removes_tie_and_global_witness() -> None:
    _, production, certificate = _assert_reference_equals_production(
        "r2_predetermined_tie_breaking_perturbation"
    )
    assert certificate is None
    root_zero = next(
        item
        for item in production.values()
        if infer_outward_tree_root(item.scm.candidate) == 0
    )
    assert len(root_zero.actions) == 1
    assert root_zero.actions[0].scale == pytest.approx((1.0, 2.0))


@pytest.mark.parametrize(
    ("case_id", "has_witness"),
    [
        ("r3_three_dag_simultaneous_negative", False),
        ("r4_three_dag_simultaneous_positive", True),
    ],
)
def test_three_candidate_intersections_are_simultaneous(
    case_id: str, has_witness: bool
) -> None:
    candidates, _, certificate = _assert_reference_equals_production(case_id)
    assert len(candidates) == 3
    assert (certificate is not None) is has_witness
    if certificate is not None:
        assert len(certificate.candidate_losses) == 3
        assert certificate.command.shift == pytest.approx((1.0, 0.0, 0.0))


def test_certificate_caps_are_inconclusive_then_recover_without_stopping() -> None:
    candidates, production, _ = _assert_reference_equals_production(
        "r1_two_dag_nonroot_shift_scale_tie"
    )
    root_class = enumerate_tree_root_class(__import__("networkx").path_graph(2))
    assert set(root_class.candidates) == {candidate.id for candidate in candidates}
    state = root_class.initial_state

    low = CertificateCache(production, maximum_tied_family_combinations=3)
    low_result = low.check(state)
    assert low_result.status == "inconclusive"
    assert low_result.tied_family_combinations == 0
    assert low.tied_family_combinations == 0
    assert low.cap_hits[0]["required_upper_bound"] == 4

    no_checks = CertificateCache(production, maximum_uncached_checks=0)
    assert no_checks.check(state).status == "inconclusive"
    assert no_checks.uncached_checks == 0

    scoring = CertificateCache(production, maximum_tied_family_combinations=0)
    selection = select_one_step(root_class, state, frozenset(), scoring)
    assert selection.child_statuses == ("inconclusive", "inconclusive")
    assert selection.score[0] == 1
    path = paired_policy_path(
        policy="one_step_matching_aware",
        root_class=root_class,
        true_root=0,
        certificates=scoring,
    )
    assert path["matching_query_count"] == path["dag_query_count"] == 1
    assert path["fallback_to_full_recovery"] is True

    recovered = CertificateCache(production, maximum_tied_family_combinations=4)
    recovered_result = recovered.check(state)
    assert recovered_result.status == "certified"
    assert recovered_result.certificate is not None
    assert recovered_result.tied_family_combinations == 4
    assert recovered.tied_family_combinations == 4

