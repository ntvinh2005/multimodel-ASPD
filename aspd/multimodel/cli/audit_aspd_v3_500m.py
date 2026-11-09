"""Run the CPU-only source/provenance and training-trajectory audit for ASPD v3 500M."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from aspd.multimodel.v3.diagnostic import (
    DiagnosticContract,
    compact_metric_row,
    iter_canonical_metric_file,
    summarize_training_trajectory,
    validate_frozen_provenance,
)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _loss_audit(provenance: dict[str, Any]) -> str:
    return f"""# ASPD v3 500M loss implementation audit

Status: **IMPLEMENTATION_AUDITED; HELD-OUT MEASUREMENT NOT RUN**

Frozen source commit: `{provenance['source_git_sha']}`
Checkpoint step: `{provenance['checkpoint_step']}`

## Activation reconstruction

For one model `n`, site `s` (PRE or MID), and selected layer `l`, the continuous
BatchTopK code `g_s` is decoded by a model/layer-specific linear decoder with bias.
With Matryoshka boundaries `0=C_0<...<C_5=8192`, the implementation constructs

`r_hat_m = b_dec + sum_{{c < C_m}} g_{{s,c}} d_{{n,s,l,c}}`,

plus the empty `m=0` bias-only reconstruction.  Each decoder direction is constrained
to unit norm.  The reported layer loss is

`(1/6) sum_{{m=0}}^5 SSE(r_hat_m,r) / sum ||r-mean_anchor(r)||^2`.

The denominator centers every activation coordinate across accepted anchors, in FP32.
Therefore the logged value is an average of six prefix FVUs, **not** the FVU of the
full 8192-feature reconstruction.  It is dimensionless and algebraically comparable
across models, but comparison can still reflect different target geometry and how useful
the shared feature ordering is at early Matryoshka prefixes.  Source:
`aspd/multimodel/v3/objective.py:26-36,67-133` and
`aspd/multimodel/v3/reconstruction.py:144-180,216-242`.

The global activation objective gives equal total weight to PRE and MID, then equal
weight to all three models and all eight selected layers:
`coefficient = 0.5/(3*8)` per model/site/layer.  Source:
`aspd/multimodel/v3/model_owned.py:261-303`.

## Internal matrix-output reconstruction

For logical matrix `j` and its PRE/MID binary support `1[g_s>0]`, the prediction is

`y_hat_j = b_native_j + sum_c 1[g_{{s,c}}>0] (x_j^T v_{{j,c}}) u_{{j,c}}`.

The real physical matrix input `x_j` and output `y_j` are captured by pre/post hooks.
Native physical bias is preserved; no learned per-component `b_dec` or `b_out` is used.
The loss is exact global FP32 FVU over accepted anchors:

`SSE(y_hat_j,y_j) / sum ||y_j-mean_anchor(y_j)||^2`.

Source: `aspd/multimodel/v3/reconstruction.py:245-353`,
`aspd/multimodel/components.py:92-127`, and
`aspd/multimodel/v3/objective.py:39-51`.

PRE contains q/k/v (72 equally weighted targets globally); MID contains
o/gate/up/down (96 equally weighted targets globally). PRE and MID each receive weight
0.5, so every model contributes one third of the global internal objective, but matrix
families do not receive equal family-level weight: they receive equal **target** weight.
Source: `aspd/multimodel/v3/model_owned.py:309-348`.

## AuxK

Features inactive for 320 consecutive 2048-anchor pools are marked dead.  For each
anchor, the top `min(2048,n_dead)` dead preactivations decode the detached residual
`r-r_hat_full`.  Its MSE is divided by the same coordinate-centered target variance per
scalar, and enters the total objective with coefficient `1/32`.  Source:
`aspd/multimodel/v3/gating.py:52-88` and
`aspd/multimodel/v3/objective.py:136-201`.

## BatchTopK and gradient routes

The PRE and MID shared encoders each produce non-negative scores. BatchTopK chooses
exactly the approved pooled support over 2048 aligned anchors. Activation decoding uses
continuous selected values; internal reconstruction uses detached binary support. Thus
internal loss updates only the rank-one matrix factors, whereas activation/Matryoshka
loss updates packing, shared encoders, readout and activation decoders. Source:
`aspd/multimodel/v3/model_owned.py:233-259` and
`aspd/multimodel/v3/gating.py:90-106`.

## Model/site mapping audit

The registry has 56 targets per model and maps q/k/v to PRE and o/gate/up/down to MID.
Qwen uses layers 10-17, Gemma 9-16, and SmolLM2 8-15. Gemma MID is the input of
`pre_feedforward_layernorm`, i.e. the real post-attention residual after Gemma2's
post-attention normalization and residual add. Matrix inputs/outputs are captured at the
physical projection itself. Source: `aspd/multimodel/v3/registry.py:84-190,284-319`
and `aspd/multimodel/v3/training_capture.py:92-190`.

## Audit conclusion

The code confirms that the quoted activation values are normalized and comparable as
**Matryoshka mean FVU**, but they cannot answer whether Gemma's full reconstruction FVU
is similarly poor. No scaling or aggregation bug is established by source inspection.
The held-out evaluator must separately report every prefix, the full reconstruction,
target RMS/variance, and zero/mean baselines before assigning a root cause.
"""


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stride = max(1, len(rows) // 2000)
    sampled = rows[::stride]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot([row["step"] for row in sampled], [row["loss"]["total"] for row in sampled])
    axes[0, 0].set(title="Global training objective", xlabel="optimizer step", ylabel="loss")
    for model in ("qwen", "gemma", "smol"):
        axes[0, 1].plot(
            [row["step"] for row in sampled],
            [row["loss"][f"activation/{model}/pre"] for row in sampled],
            label=model,
        )
        axes[1, 0].plot(
            [row["step"] for row in sampled],
            [row["loss"][f"activation/{model}/mid"] for row in sampled],
            label=model,
        )
    axes[0, 1].set(title="PRE Matryoshka mean FVU", xlabel="optimizer step")
    axes[1, 0].set(title="MID Matryoshka mean FVU", xlabel="optimizer step")
    axes[0, 1].legend()
    axes[1, 0].legend()
    axes[1, 1].plot(
        [row["step"] for row in sampled],
        [row["loss"]["auxk"] for row in sampled],
        color="tab:purple",
    )
    axes[1, 1].set(title="AuxK objective", xlabel="optimizer step")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    contract = DiagnosticContract.load(args.contract)
    provenance = validate_frozen_provenance(contract, source_root=args.source_root)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    metrics_path = args.source_root / contract.source_run / "metrics.jsonl"
    iterator, accounting = iter_canonical_metric_file(metrics_path)
    rows = [compact_metric_row(row) for row in iterator]
    if accounting["final_step"] != 61036 or len(rows) != 61036:
        raise ValueError("training trajectory does not end at the frozen checkpoint step")
    trajectory = summarize_training_trajectory(rows)
    trajectory["accounting"] = accounting
    _write_json(args.output_dir / "training_trajectory.json", trajectory)

    fields = [
        "step",
        "accepted_exposures",
        "seconds",
        "anchors_per_second",
        *sorted(rows[0]["loss"]),
    ]
    with (args.output_dir / "training_trajectory_compact.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row[key] for key in fields[:4]}, **row["loss"]})
    _plot(rows, args.output_dir / "loss_trajectories.png")
    (args.output_dir / "loss_audit.md").write_text(_loss_audit(provenance), encoding="utf-8")
    _write_json(
        args.output_dir / "phase_a_manifest.json",
        {
            "schema_version": 1,
            "status": "PHASE_A_COMPLETE",
            "diagnostic_git_sha": _git_sha(),
            "contract_sha256": contract.fingerprint(),
            "frozen_provenance": provenance,
            "metrics_accounting": accounting,
            "elapsed_seconds": time.perf_counter() - started,
            "heldout_evaluation": "NOT_RUN",
        },
    )
    print(json.dumps({"status": "PHASE_A_COMPLETE", "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
