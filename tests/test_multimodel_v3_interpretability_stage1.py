"""Tests for deterministic ASPD v3.1 Stage-1 Intruder task freezing."""

from __future__ import annotations

from aspd.multimodel.cli.prepare_aspd_v3_interpretability_stage1_tasks import (
    _donor_candidates,
    _sample_trial,
    _trial_seed,
)


def _row(key: str, site: str, density: float, base: int) -> dict:
    return {
        "component_key": key,
        "site": site,
        "firing_density": density,
        "contexts": [
            {
                "anchor_id": f"a{base + index}",
                "document_id": f"d{base + index}",
                "source_group": "test",
                "activation": float(index + 1),
            }
            for index in range(6)
        ],
    }


def test_donor_candidates_are_density_matched_and_site_local() -> None:
    target = _row("pre/0001", "pre", 0.10, 0)
    same_site = _row("pre/0002", "pre", 0.14, 10)
    other_site = _row("mid/0002", "mid", 0.10, 20)
    too_far = _row("pre/0003", "pre", 0.16, 30)
    assert _donor_candidates(target, [target, same_site, other_site, too_far], 0.05) == [
        same_site
    ]


def test_trial_sampling_is_reproducible_and_records_natural_examples() -> None:
    target = _row("pre/0001", "pre", 0.10, 0)
    donor = _row("pre/0002", "pre", 0.12, 10)
    seed = _trial_seed(20261115, "pre/0001", 0)
    first = _sample_trial(
        target=target, donor_candidates=[donor], n_real=4, seed=seed
    )
    second = _sample_trial(
        target=target, donor_candidates=[donor], n_real=4, seed=seed
    )
    assert first == second
    assert len({row["anchor_id"] for row in first["target_contexts"]}) == 4
    assert 0 <= first["intruder_position_zero_based"] <= 4
