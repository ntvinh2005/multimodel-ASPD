"""Generate the read-only ASPD v3 dictionary-concentration evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from matplotlib import pyplot as plt

from aspd.multimodel.v3.dictionary_concentration import (
    FEATURE_BANDS,
    assert_shared_gate_counts,
    band_count_rows,
    count_summary,
    feature_band,
)

plt.switch_backend("Agg")


SITES = ("pre", "mid")
MODELS = ("gemma", "smol", "granite")
REQUESTED_STEPS = (0, 1, 10, 50, 80, 128, 256, 500, 800, 1000, 1221)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"inconsistent table columns for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _canonical_metrics(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["step"])] = row
    if sorted(rows) != list(range(1, max(rows) + 1)):
        raise ValueError("canonicalized metrics contain missing optimizer steps")
    return rows


def _load_gate_state(checkpoint: Path, rank: int) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for site in SITES:
        prefix = f"model.gates.{site}"
        state[f"{prefix}.feature_fire_counts"] = torch.empty(8192, dtype=torch.long)
        state[f"{prefix}.accepted_anchor_count"] = torch.empty((), dtype=torch.long)
        state[f"{prefix}.n_pools_not_active"] = torch.empty(8192, dtype=torch.float32)
    dcp.load(state, checkpoint_id=checkpoint / f"rank-{rank:05d}" / "dcp")
    return state


def _verified_gate_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    states = [_load_gate_state(checkpoint, rank) for rank in range(3)]
    for key in states[0]:
        if any(not torch.equal(states[0][key], state[key]) for state in states[1:]):
            raise ValueError(f"replicated gate state differs across ranks: {key}")
    return states[0]


def _read_feature_activity(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = 8192 * len(MODELS) * len(SITES)
    if len(rows) != expected:
        raise ValueError(f"feature activity has {len(rows)} rows; expected {expected}")
    return rows


def _heldout_counts(rows: Sequence[Mapping[str, str]], site: str) -> torch.Tensor:
    by_model: list[torch.Tensor] = []
    for model in MODELS:
        values = sorted(
            (row for row in rows if row["site"] == site and row["model"] == model),
            key=lambda row: int(row["feature"]),
        )
        by_model.append(torch.tensor([int(row["heldout_fires"]) for row in values]))
    return assert_shared_gate_counts(by_model)


def _contribution_energy(row: Mapping[str, str]) -> float:
    return int(row["matrix_component_events"]) * float(row["matrix_contribution_l2_rms"]) ** 2


def _heldout_band_tables(
    rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gate_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    for site in SITES:
        counts = _heldout_counts(rows, site)
        for row in band_count_rows(counts):
            gate_rows.append({"site": site, **row})
        for model in MODELS:
            selected = [row for row in rows if row["site"] == site and row["model"] == model]
            total = sum(_contribution_energy(row) for row in selected)
            for band in FEATURE_BANDS:
                band_rows = [
                    row for row in selected if band.lower <= int(row["feature"]) < band.upper
                ]
                energy = sum(_contribution_energy(row) for row in band_rows)
                magnitude = sum(
                    int(row["matrix_component_events"])
                    * float(row["matrix_contribution_l2_mean"])
                    for row in band_rows
                )
                contribution_rows.append(
                    {
                        "model": model,
                        "site": site,
                        "band": band.name,
                        "matrix_contribution_energy_proxy": energy,
                        "matrix_contribution_energy_share": energy / total,
                        "matrix_contribution_l2_magnitude_sum": magnitude,
                    }
                )
    return gate_rows, contribution_rows


def _top_feature_tables(
    rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    features: list[dict[str, Any]] = []
    cumulative: list[dict[str, Any]] = []
    band_counts: list[dict[str, Any]] = []
    for site in SITES:
        canonical = {
            int(row["feature"]): row
            for row in rows
            if row["site"] == site and row["model"] == "gemma"
        }
        ordered = sorted(canonical, key=lambda feature: (-int(canonical[feature]["heldout_fires"]), feature))
        total = sum(int(row["heldout_fires"]) for row in canonical.values())
        top = ordered[:82]
        for rank, feature in enumerate(top, start=1):
            row = canonical[feature]
            model_rows = [
                candidate
                for candidate in rows
                if candidate["site"] == site and int(candidate["feature"]) == feature
            ]
            features.append(
                {
                    "site": site,
                    "rank": rank,
                    "feature": feature,
                    "band": feature_band(feature).name,
                    "heldout_fires": int(row["heldout_fires"]),
                    "heldout_firing_probability": float(row["heldout_firing_rate"]),
                    "share_of_firing_events": int(row["heldout_fires"]) / total,
                    "positive_gate_amplitude_mean": float(row["activation_mean_when_fired"]),
                    "positive_gate_amplitude_rms": float(row["activation_rms_when_fired"]),
                    "positive_gate_amplitude_max": float(row["activation_max"]),
                    "matrix_contribution_energy_proxy_all_models": sum(
                        _contribution_energy(candidate) for candidate in model_rows
                    ),
                }
            )
        for count in (10, 25, 50, 82):
            cumulative.append(
                {
                    "site": site,
                    "top_feature_count": count,
                    "cumulative_firing_share": sum(
                        int(canonical[feature]["heldout_fires"]) for feature in top[:count]
                    )
                    / total,
                }
            )
        for band in FEATURE_BANDS:
            band_counts.append(
                {
                    "site": site,
                    "band": band.name,
                    "top_82_feature_count": sum(feature_band(feature) == band for feature in top),
                }
            )
    return features, cumulative, band_counts


def _training_count_tables(
    run_dir: Path,
    activity: Sequence[Mapping[str, str]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    states = {
        step: _verified_gate_state(run_dir / "checkpoints" / f"step-{step:09d}")
        for step in (1000, 1221)
    }
    summaries: list[dict[str, Any]] = []
    bands: list[dict[str, Any]] = []
    revival: list[dict[str, Any]] = []
    replica = {"steps": [1000, 1221], "ranks_verified_equal": [0, 1, 2]}
    for site in SITES:
        key = f"model.gates.{site}.feature_fire_counts"
        anchor_key = f"model.gates.{site}.accepted_anchor_count"
        dead_key = f"model.gates.{site}.n_pools_not_active"
        intervals = (
            ("cumulative_to_1000", states[1000][key], int(states[1000][anchor_key]), 1000),
            ("cumulative_to_1221", states[1221][key], int(states[1221][anchor_key]), 1221),
            (
                "interval_1001_1221",
                states[1221][key] - states[1000][key],
                int(states[1221][anchor_key] - states[1000][anchor_key]),
                1221,
            ),
        )
        for label, counts, anchors, endpoint in intervals:
            if torch.any(counts < 0):
                raise ValueError("feature counts decreased between checkpoints")
            dead = int((states[endpoint][dead_key] >= 320).sum())
            summaries.append(
                {
                    "site": site,
                    "measurement": label,
                    "accepted_anchors": anchors,
                    "positive_firing_events_per_anchor": int(counts.sum()) / anchors,
                    "endpoint_recently_dead_features": dead,
                    **count_summary(counts),
                }
            )
            for row in band_count_rows(counts):
                bands.append({"site": site, "measurement": label, **row})
        interval = states[1221][key] - states[1000][key]
        dead_at_1000 = states[1000][dead_key] >= 320
        dead_at_1221 = states[1221][dead_key] >= 320
        fired_after_dead = dead_at_1000 & (interval > 0)
        candidate_sustained = fired_after_dead & ~dead_at_1221
        transient = fired_after_dead & dead_at_1221
        newly_dead = ~dead_at_1000 & dead_at_1221
        heldout = _heldout_counts(activity, site)
        for label, mask in (
            ("dead_at_1000_fired_in_interval", fired_after_dead),
            ("candidate_sustained_not_dead_at_1221", candidate_sustained),
            ("transient_fired_but_dead_at_1221", transient),
            ("newly_dead_by_1221", newly_dead),
        ):
            revival.append(
                {
                    "site": site,
                    "classification": label,
                    "feature_count": int(mask.sum()),
                    "interval_firing_events": int(interval[mask].sum()),
                    "interval_firing_share": float(interval[mask].sum() / interval.sum()),
                    "heldout_features_firing": int(((heldout > 0) & mask).sum()),
                    "heldout_firing_events": int(heldout[mask].sum()),
                    "heldout_firing_share": float(heldout[mask].sum() / heldout.sum()),
                }
            )
    return summaries, bands, revival, replica


def _gate_trajectory(
    metrics: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    available = []
    for step in REQUESTED_STEPS:
        metric = metrics.get(step)
        if metric is None:
            continue
        loss = metric["loss"]
        if not any(key.startswith("diagnostic/gate/") for key in loss):
            continue
        available.append(step)
        for site in SITES:
            prefix = f"diagnostic/gate/{site}/gemma/"
            rows.append(
                {
                    "step": step,
                    "accepted_exposures": int(metric["accepted_exposures"]),
                    "site": site,
                    "positive_firings_per_anchor": loss[prefix + "positive_firings_per_anchor"],
                    "active_features_in_current_pool": loss[prefix + "active_features_in_pool"],
                    "recently_dead_features": loss[prefix + "dead_features"],
                    "raw_logit_negative_fraction": loss[prefix + "raw_logit_negative_fraction"],
                    "mean_positive_gate_amplitude": loss[prefix + "positive_gate_amplitude_mean"],
                    "p95_positive_gate_amplitude": "UNAVAILABLE",
                    "p99_positive_gate_amplitude": "UNAVAILABLE",
                    "positive_score_q90_sampled": loss[prefix + "positive_score_q90_sampled"],
                    "positive_score_q99_sampled": loss[prefix + "positive_score_q99_sampled"],
                    "auxk_loss": loss["auxk"],
                }
            )
    nonzero = [step for step, row in metrics.items() if float(row["loss"]["auxk"]) != 0.0]
    summary = {
        "requested_steps": list(REQUESTED_STEPS),
        "available_diagnostic_steps": available,
        "missing_or_uninstrumented_requested_steps": sorted(set(REQUESTED_STEPS) - set(available)),
        "canonical_optimizer_steps": len(metrics),
        "auxk_nonzero_steps": len(nonzero),
        "auxk_nonzero_fraction": len(nonzero) / len(metrics),
        "auxk_first_nonzero_step": min(nonzero),
        "auxk_last_nonzero_step": max(nonzero),
        "feature_level_auxk_selection_logged": False,
        "feature_level_revival_logged": False,
    }
    return rows, summary


def _feature_group_contributions(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for site in SITES:
        counts = _heldout_counts(rows, site)
        active = (counts > 0).nonzero(as_tuple=True)[0].tolist()
        ordered = sorted(active, key=lambda feature: (-int(counts[feature]), feature))
        groups = {"top_82": set(ordered[:82]), "rare_active_82": set(ordered[-82:])}
        for model in MODELS:
            model_rows = {
                int(row["feature"]): row
                for row in rows
                if row["site"] == site and row["model"] == model
            }
            for group, selected in groups.items():
                energy = [_contribution_energy(model_rows[feature]) for feature in selected]
                per_event_rms = [
                    float(model_rows[feature]["matrix_contribution_l2_rms"])
                    for feature in selected
                    if int(model_rows[feature]["matrix_component_events"]) > 0
                ]
                result.append(
                    {
                        "model": model,
                        "site": site,
                        "feature_group": group,
                        "feature_count": len(selected),
                        "total_matrix_contribution_energy_proxy": sum(energy),
                        "mean_per_event_matrix_contribution_rms": sum(per_event_rms)
                        / len(per_event_rms),
                    }
                )
    return result


def _plots(output: Path, band_rows: Sequence[Mapping[str, Any]], trajectory: Sequence[Mapping[str, Any]]) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, site in zip(axes, SITES, strict=True):
        rows = [row for row in band_rows if row["site"] == site]
        axis.bar([row["band"] for row in rows], [100 * row["firing_share"] for row in rows])
        axis.plot(
            [row["band"] for row in rows],
            [100 * row["capacity_share"] for row in rows],
            marker="o",
            label="capacity share",
        )
        axis.set_title(site.upper())
        axis.set_xlabel("Matryoshka band")
        axis.set_ylabel("Share (%)")
        axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "heldout_band_firing_share.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for site, color in (("pre", "tab:blue"), ("mid", "tab:orange")):
        rows = [row for row in trajectory if row["site"] == site]
        x = [row["step"] for row in rows]
        axes[0, 0].plot(x, [row["active_features_in_current_pool"] for row in rows], marker="o", label=site, color=color)
        axes[0, 1].plot(x, [row["recently_dead_features"] for row in rows], marker="o", label=site, color=color)
        axes[1, 0].plot(x, [row["raw_logit_negative_fraction"] for row in rows], marker="o", label=site, color=color)
        axes[1, 1].plot(x, [row["mean_positive_gate_amplitude"] for row in rows], marker="o", label=site, color=color)
    titles = ("Active features in current pool", "Recently dead features", "Negative-logit fraction", "Mean selected amplitude")
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title)
        axis.set_xlabel("Optimizer step")
        axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "training_gate_concentration_trajectory.png", dpi=180)
    plt.close(fig)


def _provenance(run_dir: Path, heldout_dir: Path) -> dict[str, Any]:
    run_manifest_path = run_dir / "run_manifest.json"
    checkpoint_manifest_path = run_dir / "checkpoints" / "step-000001221" / "manifest.json"
    evaluation_manifest_path = heldout_dir / "experiment_manifest.json"
    run = json.loads(run_manifest_path.read_text())
    checkpoint = json.loads(checkpoint_manifest_path.read_text())
    evaluation = json.loads(evaluation_manifest_path.read_text())
    if len(
        {
            run["config_fingerprint"],
            checkpoint["config_fingerprint"],
            evaluation["config_fingerprint"],
        }
    ) != 1:
        raise ValueError("config fingerprint mismatch")
    if checkpoint["step"] != evaluation["runtime_state_step"]:
        raise ValueError("evaluation/checkpoint step mismatch")
    return {
        "training_job_id": run["slurm_job_id"],
        "evaluation_job_id": evaluation["slurm_job_id"],
        "training_git_sha": run["git_sha"],
        "diagnostic_git_sha": evaluation["diagnostic_git_sha"],
        "config_fingerprint": run["config_fingerprint"],
        "config_file_sha256": run["config_file_sha256"],
        "dataset_manifest_sha256": run["dataset_manifest_sha256"],
        "alignment_manifest_sha256": run["alignment_manifest_sha256"],
        "checkpoint_manifest_sha256": _sha256(checkpoint_manifest_path),
        "recorded_checkpoint_manifest_sha256": evaluation["checkpoint_manifest_sha256"],
        "evaluation_contract_sha256": evaluation["evaluation_contract_sha256"],
        "heldout_feature_activity_sha256": _sha256(heldout_dir / "feature_activity.csv"),
        "checkpoint_step": checkpoint["step"],
        "accepted_training_exposures": 10_002_432,
        "accepted_heldout_anchors": evaluation["accepted_anchors"],
        "model_order": run["model_order"],
        "activation_target_normalization": evaluation["activation_target_normalization"],
        "activation_decoder_normalization": "none",
        "checkpoint_manifest_experiment_metadata": checkpoint["experiment_metadata"],
        "source_paths": {
            "run_dir": str(run_dir.resolve()),
            "heldout_dir": str(heldout_dir.resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    activity = _read_feature_activity(args.heldout_dir / "feature_activity.csv")
    heldout_bands, contribution_bands = _heldout_band_tables(activity)
    top_features, top_summary, top_bands = _top_feature_tables(activity)
    training_summary, training_bands, revival, replica = _training_count_tables(
        args.run_dir, activity
    )
    metrics = _canonical_metrics(args.run_dir / "metrics.jsonl")
    trajectory, auxk = _gate_trajectory(metrics)
    group_contributions = _feature_group_contributions(activity)
    provenance = _provenance(args.run_dir, args.heldout_dir)
    if provenance["checkpoint_manifest_sha256"] != provenance["recorded_checkpoint_manifest_sha256"]:
        raise ValueError("checkpoint manifest SHA-256 does not match held-out provenance")

    _write_csv(args.output_dir / "heldout_band_metrics.csv", heldout_bands)
    _write_csv(args.output_dir / "heldout_contribution_band_metrics.csv", contribution_bands)
    _write_csv(args.output_dir / "top_82_features.csv", top_features)
    _write_csv(args.output_dir / "top_feature_cumulative_shares.csv", top_summary)
    _write_csv(args.output_dir / "top_82_band_counts.csv", top_bands)
    _write_csv(args.output_dir / "training_count_summary.csv", training_summary)
    _write_csv(args.output_dir / "training_count_band_metrics.csv", training_bands)
    _write_csv(args.output_dir / "auxk_revival_checkpoint_bounds.csv", revival)
    _write_csv(args.output_dir / "training_gate_trajectory.csv", trajectory)
    _write_csv(args.output_dir / "feature_group_contribution_summary.csv", group_contributions)
    _plots(args.output_dir, heldout_bands, trajectory)

    payload = {
        "schema_version": 1,
        "status": "READ_ONLY_DICTIONARY_CONCENTRATION_AUDIT_COMPLETE",
        "provenance": provenance,
        "checkpoint_gate_replica_verification": replica,
        "auxk": auxk,
        "measurement_boundaries": {
            "heldout_counts": "fixed validation anchors; shared gate counts verified equal across models",
            "training_counts": "cumulative checkpoint buffers; interval is an exact checkpoint difference",
            "training_gate_trajectory": "instantaneous per-step pool diagnostics averaged over four pools",
            "contribution_energy": "observational sum(events * RMS_L2^2), not marginal residual reduction",
            "auxk_revival": "feature-level selected/revived identities were not logged",
        },
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
