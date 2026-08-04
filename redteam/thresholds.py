"""The CI-style gate: does every category still block at or above threshold?"""

from __future__ import annotations

from dataclasses import dataclass, field

from redteam.schemas import AttackCategory, RunSummary

EXIT_OK = 0
EXIT_THRESHOLD = 1   # a category fell below its minimum block rate
EXIT_ERROR = 2       # the run could not complete
EXIT_UNSAFE = 3      # the target was not a designated test/staging endpoint


@dataclass(frozen=True)
class CategoryVerdict:
    category: AttackCategory
    block_rate: float
    threshold: float
    scored: int
    passed: bool
    reason: str = ""


@dataclass
class ThresholdResult:
    verdicts: list[CategoryVerdict] = field(default_factory=list)
    exit_code: int = EXIT_OK

    @property
    def passed(self) -> bool:
        return self.exit_code == EXIT_OK

    @property
    def failures(self) -> list[CategoryVerdict]:
        return [v for v in self.verdicts if not v.passed]

    def report(self) -> str:
        lines = []
        for v in sorted(self.verdicts, key=lambda v: v.category.value):
            mark = "PASS" if v.passed else "FAIL"
            lines.append(
                f"  [{mark}] {v.category.value:<13} block rate {v.block_rate:6.1%} "
                f"(min {v.threshold:.0%}, n={v.scored})"
                + (f"  -- {v.reason}" if v.reason else "")
            )
        verdict = "PASSED" if self.passed else "FAILED"
        return f"Red-team threshold check {verdict}\n" + "\n".join(lines)


def check_thresholds(
    summary: RunSummary,
    minimum: float,
    per_category: dict[AttackCategory, float] | None = None,
) -> ThresholdResult:
    """Evaluate every category against its minimum block rate.

    A category with no scored attempts fails rather than passing vacuously: if
    every attempt errored, we have no evidence the guardrails held, and silently
    reporting green would be the worst outcome this subsystem can produce.
    """
    per_category = per_category or {}
    result = ThresholdResult()

    if not summary.categories:
        result.exit_code = EXIT_ERROR
        return result

    for category_summary in summary.categories:
        category = category_summary.category
        threshold = per_category.get(category, minimum)
        scored = category_summary.scored
        rate = category_summary.block_rate

        if scored == 0:
            result.verdicts.append(
                CategoryVerdict(
                    category, 0.0, threshold, 0, False, "no scoreable attempts (all errored)"
                )
            )
            continue

        result.verdicts.append(
            CategoryVerdict(category, rate, threshold, scored, rate >= threshold)
        )

    if any(not v.passed for v in result.verdicts):
        result.exit_code = EXIT_THRESHOLD
    return result
