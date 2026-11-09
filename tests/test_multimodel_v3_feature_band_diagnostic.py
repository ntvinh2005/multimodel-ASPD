from __future__ import annotations

import pytest

from aspd.multimodel.v3.feature_band_diagnostic import analyze_feature_activity_rows


def _rows() -> list[dict[str, str]]:
    rows = []
    for feature in range(8192):
        fires = 5 if feature < 512 else 1
        rows.append(
            {
                "feature": str(feature),
                "heldout_fires": str(fires),
                "matrix_component_events": str(fires * 2),
                "matrix_contribution_l2_mean": "3.0",
                "model": "model-a",
                "site": "pre",
            }
        )
    return rows


def test_feature_band_diagnostic_preserves_matryoshka_boundaries() -> None:
    result = analyze_feature_activity_rows(_rows())
    bands = result["model-a"]["pre"]["bands"]

    assert [(band["feature_lower"], band["feature_upper"]) for band in bands] == [
        (0, 512),
        (512, 1024),
        (1024, 2048),
        (2048, 4096),
        (4096, 8192),
    ]
    assert [band["matryoshka_prefix_loss_opportunities"] for band in bands] == [
        5,
        4,
        3,
        2,
        1,
    ]
    assert sum(band["firing_share"] for band in bands) == pytest.approx(1.0)
    assert sum(band["matrix_contribution_l2_sum_share"] for band in bands) == pytest.approx(1.0)
    assert bands[0]["firing_density_vs_uniform"] > bands[-1]["firing_density_vs_uniform"]


def test_feature_band_diagnostic_rejects_duplicate_features() -> None:
    rows = _rows()
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match="duplicate feature"):
        analyze_feature_activity_rows(rows)
