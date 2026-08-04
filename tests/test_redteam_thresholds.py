"""The CI gate and cross-run trend arithmetic."""

from __future__ import annotations

import pytest

from redteam.schemas import (
    AttackAttempt,
    AttackCategory,
    CategorySummary,
    Outcome,
    RunSummary,
)
from redteam.store import InMemoryRunStore, compute_trend, rates_from_rows
from redteam.thresholds import EXIT_ERROR, EXIT_OK, EXIT_THRESHOLD, check_thresholds


def summary_with(**rates: float) -> RunSummary:
    """Build a run whose categories land on exactly the given block rates."""
    summary = RunSummary(target="http://localhost:8000/pipeline/run")
    for name, rate in rates.items():
        category = AttackCategory(name)
        total = 10
        blocked = round(rate * total)
        summary.categories.append(
            CategorySummary(
                category=category,
                total=total,
                blocked=blocked,
                refused=0,
                leaked=total - blocked,
                errors=0,
            )
        )
        summary.attempts.extend(
            AttackAttempt(
                category=category,
                objective=f"{name} objective {i}",
                outcome=Outcome.BLOCKED if i < blocked else Outcome.LEAKED,
            )
            for i in range(total)
        )
    return summary


# --- the gate --------------------------------------------------------------


def test_all_categories_above_threshold_exits_zero():
    result = check_thresholds(summary_with(jailbreak=1.0, xpia=0.95), minimum=0.9)

    assert result.exit_code == EXIT_OK
    assert result.passed
    assert not result.failures


def test_a_category_below_threshold_exits_non_zero():
    """Acceptance: deliberately drop one category under the bar."""
    result = check_thresholds(summary_with(jailbreak=1.0, xpia=0.4), minimum=0.9)

    assert result.exit_code == EXIT_THRESHOLD
    assert result.exit_code != 0
    assert not result.passed
    assert [v.category for v in result.failures] == [AttackCategory.XPIA]
    assert "FAIL" in result.report() and "xpia" in result.report()


def test_one_bad_category_fails_the_run_even_if_the_average_is_fine():
    """The gate is per-category on purpose: a strong average must not mask a hole."""
    summary = summary_with(jailbreak=1.0, xpia=1.0, crescendo=1.0, skeleton_key=0.5)

    assert summary.block_rate > 0.85          # the average looks healthy
    assert check_thresholds(summary, 0.9).exit_code == EXIT_THRESHOLD


def test_threshold_is_inclusive_at_the_boundary():
    assert check_thresholds(summary_with(jailbreak=0.9), 0.9).exit_code == EXIT_OK
    assert check_thresholds(summary_with(jailbreak=0.8), 0.9).exit_code == EXIT_THRESHOLD


def test_per_category_override_is_honoured():
    summary = summary_with(jailbreak=1.0, crescendo=0.6)
    overrides = {AttackCategory.CRESCENDO: 0.5}

    assert check_thresholds(summary, 0.9, overrides).exit_code == EXIT_OK
    assert check_thresholds(summary, 0.9).exit_code == EXIT_THRESHOLD


def test_a_category_with_no_scoreable_attempts_fails_rather_than_passing_vacuously():
    """No evidence is not the same as a pass -- that would be the worst failure mode."""
    summary = RunSummary()
    summary.categories.append(
        CategorySummary(category=AttackCategory.CRESCENDO, total=2, errors=2)
    )

    result = check_thresholds(summary, 0.9)

    assert result.exit_code == EXIT_THRESHOLD
    assert "no scoreable attempts" in result.failures[0].reason


def test_a_run_with_no_categories_is_an_error_not_a_pass():
    assert check_thresholds(RunSummary(), 0.9).exit_code == EXIT_ERROR


# --- persistence + trend ---------------------------------------------------


async def test_second_run_compares_against_the_first():
    """Acceptance: persist two runs, the second reports a trend against the first."""
    store = InMemoryRunStore()
    target = "http://localhost:8000/pipeline/run"

    first = summary_with(jailbreak=1.0, xpia=0.9)
    first.target = target
    await store.save(first)

    second = summary_with(jailbreak=0.7, xpia=0.9)
    second.target = target
    previous_id, previous_rates = await store.previous_run(target, second.run_id)
    await store.save(second)

    assert previous_id == first.run_id
    trend = compute_trend(second, previous_rates, previous_id)

    by_category = {t.category: t for t in trend.categories}
    assert by_category[AttackCategory.JAILBREAK].delta == pytest.approx(-0.3)
    assert by_category[AttackCategory.JAILBREAK].direction == "regressed"
    assert by_category[AttackCategory.XPIA].direction == "flat"
    assert [t.category for t in trend.regressions] == [AttackCategory.JAILBREAK]


async def test_first_ever_run_reports_new_rather_than_a_bogus_delta():
    store = InMemoryRunStore()
    summary = summary_with(jailbreak=1.0)
    summary.target = "http://localhost:8000/pipeline/run"

    previous_id, previous_rates = await store.previous_run(summary.target, summary.run_id)
    trend = compute_trend(summary, previous_rates, previous_id)

    assert previous_id is None
    assert trend.categories[0].previous is None
    assert trend.categories[0].delta is None
    assert trend.categories[0].direction == "new"


async def test_trend_only_compares_runs_against_the_same_target():
    store = InMemoryRunStore()
    other = summary_with(jailbreak=0.1)
    other.target = "http://other.test/pipeline/run"
    await store.save(other)

    current = summary_with(jailbreak=1.0)
    current.target = "http://localhost:8000/pipeline/run"
    previous_id, _ = await store.previous_run(current.target, current.run_id)

    assert previous_id is None, "a different target's history is not a baseline"


async def test_attempts_round_trip_through_the_store():
    store = InMemoryRunStore()
    summary = summary_with(jailbreak=0.5)
    await store.save(summary)

    stored = await store.attempts_for(summary.run_id)
    assert len(stored) == 10
    assert {a["outcome"] for a in stored} == {"blocked", "leaked"}

    runs = await store.recent_runs()
    assert runs[0]["run_id"] == summary.run_id
    assert rates_from_rows(runs[0]["categories"])["jailbreak"] == pytest.approx(0.5)
