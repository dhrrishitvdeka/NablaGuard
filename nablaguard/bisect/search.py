"""Generic monotonic first-bad-step binary search."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, TypeVar

T = TypeVar("T")
ProbeOutcome = Literal["GOOD", "BAD"]


@dataclass(frozen=True, slots=True)
class SearchProbe:
    """One evaluated midpoint."""

    step: int
    outcome: ProbeOutcome


@dataclass(frozen=True, slots=True)
class SearchResult:
    """First bad boundary under a monotonic predicate assumption."""

    first_bad_step: int
    probes: tuple[SearchProbe, ...]
    monotonicity_violations: tuple[int, ...] = ()

    @property
    def monotonicity_ok(self) -> bool:
        """Whether sampled probes were consistent with a single false-to-true transition."""

        return not self.monotonicity_violations


def first_bad(
    known_good: int,
    known_bad: int,
    evaluate: Callable[[int], bool],
    *,
    verify_probes: bool = True,
) -> SearchResult:
    """Binary-search a false-to-true transition.

    ``evaluate`` must be monotonic in the chosen interval. Endpoints are checked
    before the search. When ``verify_probes`` is true, every midpoint outcome is
    checked against the current good/bad bracket; inconsistent outcomes are
    recorded as ``monotonicity_violations`` (the search still returns the
    bracket implied by the binary-search decisions).
    """

    if known_good < 0 or known_bad <= known_good:
        raise ValueError("bisect requires 0 <= known_good < known_bad")
    if evaluate(known_good):
        raise ValueError("known_good satisfies the failure predicate")
    if not evaluate(known_bad):
        raise ValueError("known_bad does not satisfy the failure predicate")
    probes: list[SearchProbe] = []
    outcomes: dict[int, bool] = {known_good: False, known_bad: True}
    good = known_good
    bad = known_bad
    while bad - good > 1:
        midpoint = good + (bad - good) // 2
        failed = evaluate(midpoint)
        outcomes[midpoint] = failed
        probes.append(SearchProbe(midpoint, "BAD" if failed else "GOOD"))
        if failed:
            bad = midpoint
        else:
            good = midpoint
    violations: list[int] = []
    if verify_probes:
        # Binary search alone cannot prove monotonicity. Re-check every step in
        # small intervals, otherwise spot-check the claimed good/bad regions.
        check_steps: Iterable[int]
        if known_bad - known_good <= 32:
            check_steps = range(known_good, known_bad + 1)
        else:
            check_steps = {
                known_good,
                max(known_good, bad - 1),
                bad,
                known_bad,
                good + (bad - good) // 2,
            }
        for step in check_steps:
            if step in outcomes:
                failed = outcomes[step]
            else:
                failed = evaluate(step)
                outcomes[step] = failed
            expected = step >= bad
            if failed != expected:
                violations.append(step)
    return SearchResult(bad, tuple(probes), tuple(sorted(set(violations))))
