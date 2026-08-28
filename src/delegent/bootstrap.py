"""Deterministic paired question-level bootstrap intervals."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections.abc import Sequence
from typing import Any


def deterministic_seed(*parts: str, namespace: str = "delegent-bootstrap-v1") -> int:
    digest = hashlib.sha256("\0".join((namespace, *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def nearest_rank_percentile(
    sorted_values: Sequence[float], probability: float
) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    index = min(
        len(sorted_values) - 1, max(0, math.ceil(probability * len(sorted_values)) - 1)
    )
    return float(sorted_values[index])


def paired_bootstrap(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    repetitions: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap the mean paired delta ``candidate - reference``."""

    if len(reference) != len(candidate) or not reference:
        raise ValueError("paired inputs must be non-empty and have equal length")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    deltas = [
        float(candidate_value) - float(reference_value)
        for reference_value, candidate_value in zip(reference, candidate)
    ]
    if any(not math.isfinite(value) for value in deltas):
        raise ValueError("paired values must be finite")
    rng = random.Random(seed)
    count = len(deltas)
    sampled = sorted(
        statistics.fmean(deltas[rng.randrange(count)] for _ in range(count))
        for _ in range(repetitions)
    )
    tolerance = 1e-15
    return {
        "reference_mean": statistics.fmean(float(value) for value in reference),
        "candidate_mean": statistics.fmean(float(value) for value in candidate),
        "mean_delta": statistics.fmean(deltas),
        "paired_bootstrap_95ci": [
            nearest_rank_percentile(sampled, 0.025),
            nearest_rank_percentile(sampled, 0.975),
        ],
        "worse_equal_better": {
            "worse": sum(value < -tolerance for value in deltas),
            "equal": sum(abs(value) <= tolerance for value in deltas),
            "better": sum(value > tolerance for value in deltas),
        },
        "repetitions": repetitions,
        "seed": seed,
        "paired": True,
    }
