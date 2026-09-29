"""Prevent silent row-update failures and inconsistent progress counts."""

from pathlib import Path


def test_matrix_has_all_requirements_and_consistent_counts():
    text = (Path(__file__).resolve().parents[1] / "GAP_MATRIX.md").read_text()
    rows = [
        line.split("|")
        for line in text.splitlines()
        if line.startswith("| ") and line.split("|")[1].strip().isdigit()
    ]
    assert [int(row[1]) for row in rows] == list(range(1, 110))
    statuses = [row[3].strip() for row in rows]
    assert set(statuses) <= {"Erfüllt", "Teilweise", "Offen"}
    for status in ("Erfüllt", "Teilweise", "Offen"):
        assert f"{status}: {statuses.count(status)}/109" in text
    by_id = {int(row[1]): row for row in rows}
    for number in (
        6,
        12,
        14,
        16,
        52,
        53,
        55,
        57,
        58,
        59,
        60,
        61,
        62,
        80,
        90,
        92,
        98,
    ):
        assert by_id[number][3].strip() == "Erfüllt"
    for number in (44, 56, 63, 87, 108):
        assert by_id[number][3].strip() == "Teilweise"
    assert "Hashvektoren entfernt" in by_id[20][4]
    assert "Migration v3" in by_id[17][4]
    assert "Restziel-Scope" in by_id[52][4]
    assert "SIGKILL" in by_id[98][4]
    assert "Grant-Replays" in by_id[62][4]
    assert "persistierte Commit-Absicht" in by_id[57][4]
    assert "Zielbranchvalidierung" in by_id[57][4]
