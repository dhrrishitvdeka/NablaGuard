"""Generic monotonic first-bad-step binary search."""

from __future__ import annotations

from collections.abc import Callable
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


def first_bad(
    known_good: int,
    known_bad: int,
    evaluate: Callable[[int], bool],
) -> SearchResult:
    """Binary-search a false-to-true transition.

    ``evaluate`` must be monotonic in the chosen interval. Endpoints are checked
    before the search; monotonicity of unobserved steps cannot be proven by a
    logarithmic search.
    """

    if known_good < 0 or known_bad <= known_good:
        raise ValueError("bisect requires 0 <= known_good < known_bad")
    if evaluate(known_good):
        raise ValueError("known_good satisfies the failure predicate")
    if not evaluate(known_bad):
        raise ValueError("known_bad does not satisfy the failure predicate")
    probes: list[SearchProbe] = []
    good = known_good
    bad = known_bad
    while bad - good > 1:
        midpoint = good + (bad - good) // 2
        failed = evaluate(midpoint)
        probes.append(SearchProbe(midpoint, "BAD" if failed else "GOOD"))
        if failed:
            bad = midpoint
        else:
            good = midpoint
    return SearchResult(bad, tuple(probes))
