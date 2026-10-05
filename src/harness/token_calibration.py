"""Conservative calibration from privacy-safe token-count observations."""

from __future__ import annotations

import math


def calibration_report(samples, *, min_samples=5, max_multiplier=1.5):
    """Return a bounded p90 adjustment from (raw, actual, safety-margin) rows."""
    if (
        isinstance(min_samples, bool)
        or not isinstance(min_samples, int)
        or min_samples < 1
    ):
        raise ValueError("minimum calibration samples must be a positive integer")
    if (
        isinstance(max_multiplier, bool)
        or not isinstance(max_multiplier, (int, float))
        or not math.isfinite(max_multiplier)
        or not 1 <= max_multiplier <= 2
    ):
        raise ValueError("maximum calibration multiplier must be between 1 and 2")

    ratios = []
    for sample in samples:
        if not isinstance(sample, (list, tuple)) or len(sample) != 3:
            continue
        raw_tokens, actual_tokens, margin = sample
        if (
            isinstance(raw_tokens, bool)
            or not isinstance(raw_tokens, int)
            or raw_tokens <= 0
            or isinstance(actual_tokens, bool)
            or not isinstance(actual_tokens, int)
            or actual_tokens < 0
            or isinstance(margin, bool)
            or not isinstance(margin, int)
            or not 0 <= margin <= 100
        ):
            continue
        baseline = math.ceil(raw_tokens * (100 + margin) / 100)
        ratios.append(actual_tokens / baseline)

    count = len(ratios)
    if count < min_samples:
        return {"sample_count": count, "calibrated": False, "multiplier": 1.0}

    ratios.sort()
    p90 = ratios[math.ceil(0.9 * count) - 1]
    return {
        "sample_count": count,
        "calibrated": True,
        "multiplier": min(max_multiplier, max(1.0, p90)),
    }
