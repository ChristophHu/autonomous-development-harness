import pytest

from harness.token_calibration import calibration_report


def test_calibration_uses_bounded_nearest_rank_p90_after_minimum_samples():
    samples = [(100, actual, 20) for actual in (80, 100, 120, 140, 160)]

    result = calibration_report(samples, min_samples=5, max_multiplier=1.5)

    assert result == {
        "sample_count": 5,
        "calibrated": True,
        "multiplier": 1.3333333333333333,
    }


@pytest.mark.parametrize("samples", [[], [(0, 5, 20)], [(-1, 2, 10)], [(10, -1, 0)]])
def test_calibration_falls_back_when_samples_are_unusable(samples):
    assert calibration_report(samples, min_samples=1, max_multiplier=1.5) == {
        "sample_count": 0,
        "calibrated": False,
        "multiplier": 1.0,
    }


def test_calibration_ignores_invalid_samples_and_requires_minimum_count():
    samples = [(10, 100, 0), ("invalid", 12, 0), (10, 12, 0)]
    assert calibration_report(samples, min_samples=3, max_multiplier=1.5) == {
        "sample_count": 2,
        "calibrated": False,
        "multiplier": 1.0,
    }


def test_calibration_ignores_samples_with_invalid_shape():
    assert calibration_report([None, (10, 12), "not-a-row", [10, 12, 20, 1]]) == {
        "sample_count": 0,
        "calibrated": False,
        "multiplier": 1.0,
    }


def test_calibration_is_never_less_than_baseline_and_caps_outliers():
    under = calibration_report([(100, 50, 20)] * 5, min_samples=5, max_multiplier=1.5)
    over = calibration_report([(100, 1000, 0)] * 5, min_samples=5, max_multiplier=1.25)

    assert under["multiplier"] == 1.0
    assert over["multiplier"] == 1.25


@pytest.mark.parametrize(
    ("min_samples", "max_multiplier"),
    [(0, 1.5), (True, 1.5), (5, 0.9), (5, float("nan")), (5, True)],
)
def test_calibration_rejects_invalid_limits(min_samples, max_multiplier):
    with pytest.raises(ValueError):
        calibration_report([], min_samples=min_samples, max_multiplier=max_multiplier)
