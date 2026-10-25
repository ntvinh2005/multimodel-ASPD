"""Export human-readable scientific results and the P6 handoff package."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from aspd.multimodel.config import MultiModelExperimentConfig
from aspd.multimodel.original_recipe import sha256_file, validate_original_recipe_config


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _value(tensors: dict[str, Tensor], key: str, index: int) -> float | int | None:
    value = tensors.get(key)
    if value is None:
        return None
    item = value[index].item()
    return int(item) if value.dtype in {torch.int8, torch.int16, torch.int32, torch.int64} else float(item)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite result table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _component_rows(
    cfg: MultiModelExperimentConfig,
    evaluation: dict[str, Any],
    tensors: dict[str, Tensor],
) -> list[dict[str, Any]]:
    tokens = int(evaluation["groups"]["all"]["tokens"])
    threshold = float(evaluation["sparsity"]["threshold"])
    rows = []
    optional = (
        "risk_fire_rate",
        "sports_control_fire_rate",
        "neutral_fire_rate",
        "risk_activation_strength",
        "sports_control_activation_strength",
        "neutral_activation_strength",
        "paired_risk_minus_control_fire_rate",
        "paired_risk_minus_control_activation_strength",
        "paired_risk_minus_control_base_contribution_rms",
        "paired_risk_minus_control_target_contribution_rms",
        "risk_base_contribution_rms",
        "risk_target_contribution_rms",
        "sports_control_base_contribution_rms",
        "sports_control_target_contribution_rms",
        "neutral_base_contribution_rms",
        "neutral_target_contribution_rms",
    )
    for component_id in range(cfg.sparsity.n_features):
        base_contribution = float(
            tensors["base_contribution_rms_per_validation_token"][component_id]
        )
        target_contribution = float(
            tensors["target_contribution_rms_per_validation_token"][component_id]
        )
        row: dict[str, Any] = {
            "run_id": "qwen25_aspd_original_100m",
            "component_id": component_id,
            "aspd_checkpoint_sha256": evaluation["checkpoint"]["rank0_sha256"],
            "base_model_id": cfg.models[0].model_id,
            "base_model_revision": cfg.models[0].revision,
            "target_model_id": cfg.models[1].model_id,
            "target_model_revision": cfg.models[1].revision,
            "layer": 14,
            "matrix": "model.layers.14.self_attn.o_proj",
            "matrix_scope": "matrix_local",
            "gate_policy": "clean_shared_learned_JumpReLU_threshold",
            "learned_threshold": threshold,
            "held_out_fire_count": _value(tensors, "fire_counts", component_id),
            "held_out_firing_frequency": float(tensors["fire_counts"][component_id]) / max(tokens, 1),
            "base_read_norm": _value(tensors, "base_read_norm", component_id),
            "base_write_norm": _value(tensors, "base_write_norm", component_id),
            "base_mechanism_mass": _value(tensors, "base_mechanism_mass", component_id),
            "target_read_norm": _value(tensors, "target_read_norm", component_id),
            "target_write_norm": _value(tensors, "target_write_norm", component_id),
            "target_mechanism_mass": _value(tensors, "target_mechanism_mass", component_id),
            "base_contribution_rms_per_validation_token": base_contribution,
            "target_contribution_rms_per_validation_token": target_contribution,
            "target_minus_base_contribution_rms_per_validation_token": (
                target_contribution - base_contribution
            ),
            "risk_screening_status": evaluation["behavioral_screening"]["status"],
            "candidate_reason": "Not ranked in the all-component table; use the P6 shortlist files.",
            "evidence_limitation": (
                "Observational association and standalone component contribution; "
                "causal effect requires a separate P6 intervention."
            ),
        }
        for key in optional:
            row[key] = _value(tensors, key, component_id)
        rows.append(row)
    return rows


def _plots(run: Path, evaluation_dir: Path, evaluation: dict[str, Any], tensors: dict[str, Tensor]) -> list[Path]:
    plot_dir = evaluation_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    metric_rows = [
        json.loads(line)
        for line in (run / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    loss_path = plot_dir / "training_loss.png"
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot([row["step"] for row in metric_rows], [row["train/loss"] for row in metric_rows])
    axis.set(xlabel="Optimizer step", ylabel="Training loss", title="ASPD training loss")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(loss_path, dpi=160)
    plt.close(fig)
    outputs.append(loss_path)

    reconstruction_path = plot_dir / "held_out_reconstruction_fvu.png"
    all_group = evaluation["groups"]["all"]
    labels = ["Residual Base", "Residual Target", "O Base", "O Target", "ΔY parameter"]
    values = [
        all_group["residual_activation"]["base"]["fvu"],
        all_group["residual_activation"]["target"]["fvu"],
        all_group["o_proj_computation"]["base"]["fvu"],
        all_group["o_proj_computation"]["target"]["fvu"],
        all_group["o_proj_computation"]["delta_parameter"]["normalized_squared_error"],
    ]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, values)
    axis.set(ylabel="FVU / NSE", title="Held-out reconstruction error")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(reconstruction_path, dpi=160)
    plt.close(fig)
    outputs.append(reconstruction_path)

    utilization_path = plot_dir / "component_firing_frequency.png"
    frequency = tensors["fire_counts"].double() / max(int(all_group["tokens"]), 1)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(frequency.numpy(), bins=80)
    axis.set(xlabel="Held-out firing frequency", ylabel="Components", title="Component utilization")
    fig.tight_layout()
    fig.savefig(utilization_path, dpi=160)
    plt.close(fig)
    outputs.append(utilization_path)

    contribution_path = plot_dir / "component_contribution_distribution.png"
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(
        [
            tensors["base_contribution_rms_per_validation_token"].numpy(),
            tensors["target_contribution_rms_per_validation_token"].numpy(),
        ],
        bins=80,
        label=["Base", "Target"],
        alpha=0.65,
    )
    axis.set(
        xlabel="Standalone contribution RMS per held-out token",
        ylabel="Components",
        title="Measured component computation contribution",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(contribution_path, dpi=160)
    plt.close(fig)
    outputs.append(contribution_path)
    return outputs


def _format_metric(value: float | None) -> str:
    return "không xác định" if value is None else f"{value:.6g}"


def _scientific_summary_vi(
    evaluation: dict[str, Any],
    training: dict[str, Any],
    top_rows: list[dict[str, Any]],
) -> str:
    group = evaluation["groups"]["all"]
    delta = group["o_proj_computation"]["delta_parameter"]
    sparsity = evaluation["sparsity"]
    feature_count = int(sparsity["component_count"])
    active_count = int(sparsity["active_components"])
    lines = [
        "# Kết quả two-model ASPD 100M — Qwen2.5 L14 `o_proj`",
        "",
        "## 1. Trạng thái và đối tượng được huấn luyện",
        "",
        f"Training đạt `{training['final_step']:,}` bước và `{training['total_token_exposures']:,}` token exposures; trạng thái là `{training['status']}`. Hai LLM nguồn được đóng băng. Phần được học là shared gate từ `R_mid`, hai activation decoder và các cặp tham số hạng một `u_c^(n), v_c^(n)` riêng cho Base/Target của duy nhất ma trận L14 `o_proj`.",
        "",
        "## 2. Reconstruction trên held-out general corpus",
        "",
        f"- Residual FVU — Base: `{_format_metric(group['residual_activation']['base']['fvu'])}`; Target: `{_format_metric(group['residual_activation']['target']['fvu'])}`.",
        f"- L14 `o_proj` output FVU — Base: `{_format_metric(group['o_proj_computation']['base']['fvu'])}`; Target: `{_format_metric(group['o_proj_computation']['target']['fvu'])}`.",
        f"- Parameter-only ΔY NSE: `{_format_metric(delta['normalized_squared_error'])}`; true RMS `{delta['actual_output_rms']:.6g}`, predicted RMS `{delta['reconstructed_output_rms']:.6g}`, error RMS `{delta['rmse']:.6g}`, cosine agreement `{_format_metric(delta['cosine_similarity'])}`.",
        f"- Numerical identity `ΔY_total = ΔY_parameter + ΔY_input`: relative RMS residual `{group['decomposition_identity']['true']['relative_rms_residual']:.6g}` cho phép tính thật và `{group['decomposition_identity']['aspd']['relative_rms_residual']:.6g}` cho reconstruction.",
        "",
        "FVU là phần phương sai chưa giải thích: ví dụ FVU 0.10 nghĩa là squared reconstruction error bằng 10% phương sai của tín hiệu; 0 là hoàn hảo. NSE dùng năng lượng của ΔY thật thay cho phương sai: NSE 0.20 nghĩa là squared error bằng 20% năng lượng ΔY. `Parameter-only ΔY=(W_Target-W_Base)X_Base` chỉ đo thay đổi do trọng số; `ΔY_input=W_Target(X_Target-X_Base)` được báo cáo riêng.",
        "",
        "## 3. Mức sử dụng component",
        "",
        f"- Có `{active_count:,}/{feature_count:,}` component fire ít nhất một lần (`{sparsity['active_fraction']:.3%}`); `{feature_count - active_count:,}` component không fire trên held-out set.",
        f"- Trung bình learned JumpReLU chọn `{sparsity['jump_relu_l0']:.6g}` component/token, còn BatchTopK tham chiếu chọn `{sparsity['batch_topk_l0']:.6g}`; support Jaccard là `{sparsity['support_jaccard']:.6g}`.",
        f"- Learned inference threshold là `{sparsity['threshold']:.6g}`; effective utilization là `{sparsity['effective_utilization']:.6g}` component. Effective utilization thấp hơn nhiều số component active cho thấy activation tập trung vào một nhóm nhỏ.",
        "",
        "BatchTopK ép đúng K activation lớn nhất trong mỗi pool khi train. Learned threshold là ngưỡng JumpReLU dùng lúc inference; ví dụ preactivation 0.8 chỉ fire nếu threshold nhỏ hơn 0.8. Component mass `||u||·||v||` là độ lớn Frobenius của toán tử hạng một, không phải causal effect. Contribution ở bảng dưới còn dùng input và gate đo được nên hữu ích hơn việc xếp hạng chỉ bằng norm.",
        "",
        "## 4. Behavioral screening và candidate",
        "",
        f"Behavioral screening có trạng thái `{evaluation['behavioral_screening']['status']}` trên `{evaluation['behavioral_screening'].get('matched_pair_count', 0)}` cặp risk/control. Đây là clean-forward observational screening, chưa phải can thiệp nhân quả.",
        "",
        "| Rank | Component | Base contribution RMS | Target contribution RMS | Lý do |",
        "|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(top_rows[:20], start=1):
        lines.append(
            f"| {rank} | {row['component_id']} | {float(row['base_contribution_rms_per_validation_token']):.6g} | "
            f"{float(row['target_contribution_rms_per_validation_token']):.6g} | {row['candidate_reason']} |"
        )
    lines.extend(
        [
            "",
            "## 5. Kết luận được hỗ trợ và điều chưa biết",
            "",
            "Các số trên hỗ trợ kết luận về fidelity, utilization và đóng góp tính toán độc lập của component trong clean forward pass. Chúng không chứng minh component nào gây ra hành vi chấp nhận rủi ro. P6 phải dùng gate sạch đã đóng băng, gate-off, matched-active, orthogonal và random same-L2 nhiều seed, đồng thời ghi actual perturbation norm, trước khi đưa ra kết luận causal.",
            "",
            "Các file `component_metrics.csv`, `delta_computation_metrics.json` và thư mục `plots/` chứa bảng/biểu đồ đầy đủ; checkpoint rank 0 là nguồn sự thật duy nhất cho vector `u`, `v`, bias và threshold.",
        ]
    )
    return "\n".join(lines) + "\n"


def export_original_results(
    cfg: MultiModelExperimentConfig,
    run_dir: str | Path,
    evaluation_dir: str | Path,
) -> dict[str, Any]:
    """Create the final CSV/plots/Vietnamese report and P6 candidate package."""

    validate_original_recipe_config(cfg)
    run = Path(run_dir)
    evaluation_root = Path(evaluation_dir)
    evaluation = _load_json(evaluation_root / "held_out_evaluation.json")
    training = _load_json(run / "training_summary.json")
    verification = _load_json(run / "checkpoint_verification_report.json")
    if training.get("status") != "100M_TRAINING_COMPLETE":
        raise ValueError("result export requires a completed 100M training summary")
    if verification.get("status") != "100M_TRAINING_COMPLETE":
        raise ValueError("result export requires a verified final checkpoint pair")
    if not evaluation.get("validation_scope", {}).get("full_frozen_validation"):
        raise ValueError("scientific result export requires the full frozen validation split")
    if evaluation.get("checkpoint", {}).get("step") != cfg.training.max_steps:
        raise ValueError("scientific result export requires the final 24,414-step checkpoint")
    tensors = torch.load(
        evaluation_root / "component_metrics.pt", map_location="cpu", weights_only=True
    )
    rows = _component_rows(cfg, evaluation, tensors)
    _write_csv(evaluation_root / "component_metrics.csv", rows)
    delta = {
        "schema_version": 1,
        "parameter_only": evaluation["groups"]["all"]["o_proj_computation"]["delta_parameter"],
        "total_base_target": evaluation["groups"]["all"]["o_proj_computation"]["delta_total"],
        "input_change": evaluation["groups"]["all"]["o_proj_computation"]["delta_input"],
        "identity": evaluation["groups"]["all"]["decomposition_identity"],
    }
    delta_path = evaluation_root / "delta_computation_metrics.json"
    delta_path.write_text(json.dumps(delta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_paths = _plots(run, evaluation_root, evaluation, tensors)

    computational = sorted(
        rows,
        key=lambda row: (
            float(row["base_contribution_rms_per_validation_token"])
            + float(row["target_contribution_rms_per_validation_token"])
        ),
        reverse=True,
    )[:100]
    computational_rows = []
    for rank, row in enumerate(computational, start=1):
        computational_rows.append(
            {
                **row,
                "rank": rank,
                "candidate_reason": "High measured standalone computation contribution with held-out utilization.",
            }
        )
    p6 = run / "p6_handoff"
    p6.mkdir(parents=True, exist_ok=False)
    _write_csv(p6 / "candidates_computational.csv", computational_rows)

    risk_rows: list[dict[str, Any]] = []
    if evaluation["behavioral_screening"]["status"] == "VERIFIED":
        supported = [
            row
            for row in rows
            if row["paired_risk_minus_control_activation_strength"] is not None
        ]
        supported.sort(
            key=lambda row: abs(float(row["paired_risk_minus_control_activation_strength"]))
            * (
                abs(
                    float(row["risk_target_contribution_rms"])
                    - float(row["risk_base_contribution_rms"])
                )
                + 1e-12
            ),
            reverse=True,
        )
        for rank, row in enumerate(supported[:100], start=1):
            risk_rows.append(
                {
                    **row,
                    "rank": rank,
                    "candidate_reason": (
                        "Observational paired risk-control activation contrast combined with "
                        "a measured Base/Target risk-prompt contribution difference; P6 remains required."
                    ),
                }
            )
        _write_csv(p6 / "candidates_risk_screening.csv", risk_rows)

    summary_path = evaluation_root / "summary_vi.md"
    summary_path.write_text(
        _scientific_summary_vi(evaluation, training, computational_rows), encoding="utf-8"
    )
    parameter_index = {
        "schema_version": 1,
        "authoritative_checkpoint_relative_to_run": (
            f"checkpoint_{cfg.training.max_steps:08d}.rank0.pt"
        ),
        "authoritative_checkpoint_sha256": evaluation["checkpoint"]["rank0_sha256"],
        "component_count": cfg.sparsity.n_features,
        "state_dict_paths": {
            "base_u": "components.0.o_proj.U",
            "base_v": "components.0.o_proj.V",
            "base_b_dec": "components.0.o_proj.b_dec",
            "base_b_out": "components.0.o_proj.b_out",
            "target_u": "components.1.o_proj.U",
            "target_v": "components.1.o_proj.V",
            "target_b_dec": "components.1.o_proj.b_dec",
            "target_b_out": "components.1.o_proj.b_out",
            "shared_gate_threshold": "sparsity_controller.threshold",
            "rank_local_dead_counters": "sparsity_controller.n_pools_not_active",
            "activation_centers": "encoder.centers.{model_index}",
            "activation_decoders": "activation_decoders.full.{model_index}",
        },
        "loading_helper": (
            "aspd.multimodel.original_steering.extract_original_component_for_p6"
        ),
    }
    parameter_index_path = p6 / "component_parameter_index.json"
    parameter_index_path.write_text(
        json.dumps(parameter_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract = f"""# P6 component-loading contract

- Canonical config: `{run / 'config_frozen.yaml'}`.
- Canonical checkpoint relative to this run directory: `checkpoint_{cfg.training.max_steps:08d}.rank0.pt` with SHA-256 `{evaluation['checkpoint']['rank0_sha256']}`.
- Component IDs are local to this 36,864-component dictionary and must not be equated with historical IDs.
- Load with `load_original_decomposition_for_p6`; obtain one copied, validated record with `extract_original_component_for_p6(decomposition, component_id)`.
- `b_dec` centers the actual L14 `o_proj` input `X_O`; `b_out` is added once by the matrix reconstruction.
- Gate input is the clean input of `model.layers.14.post_attention_layernorm`; matrix input is the separate input of `model.layers.14.self_attn.o_proj`.
- Freeze the clean shared learned-threshold gate before intervention. The learned JumpReLU threshold is `{evaluation['sparsity']['threshold']}`.
- Preserve Base `{cfg.models[0].model_id}@{cfg.models[0].revision}` and Target `{cfg.models[1].model_id}@{cfg.models[1].revision}`.
- Future controls: gate-off, matched active, orthogonal, and random same-L2 write direction with multiple seeds; record actual perturbation L2/RMS.
- Risk attribution and experiment PASS thresholds are separate from the feature gate threshold and remain `NOT_PREDECLARED` until the P6 protocol freezes them.
"""
    (p6 / "component_loading_contract.md").write_text(contract, encoding="utf-8")
    readme = f"""# P6 handoff — trạng thái `P6_HANDOFF_READY`

Checkpoint 100M đã được verify và evaluation đã hoàn tất. `candidates_computational.csv` là ranking theo contribution/utilization. Trạng thái behavioral screening là `{evaluation['behavioral_screening']['status']}`; file risk screening chỉ là shortlist quan sát, không phải causal result.

Không copy U/V sang nguồn sự thật thứ hai: checkpoint rank-0 là authoritative source, còn helper loading tái tạo đúng U/V, bias, threshold và normalization từ checkpoint.
"""
    (p6 / "README_vi.md").write_text(readme, encoding="utf-8")

    artifact_paths = [
        run / "config_frozen.yaml",
        run / "data_manifest.json",
        run / "training_metrics.jsonl",
        run / "training_summary.json",
        run / "checkpoint_verification_report.json",
        run / "latest_checkpoint.pt",
        run / f"checkpoint_{cfg.training.max_steps:08d}.rank0.pt",
        run / f"checkpoint_{cfg.training.max_steps:08d}.rank1.pt",
        *sorted(run.glob("runtime_rank*_steps_*.json")),
        *sorted(run.glob("producer_provenance_steps_*.json")),
        evaluation_root / "held_out_evaluation.json",
        evaluation_root / "component_metrics.pt",
        evaluation_root / "component_metrics.csv",
        delta_path,
        summary_path,
        *plot_paths,
        p6 / "candidates_computational.csv",
        parameter_index_path,
        p6 / "component_loading_contract.md",
        p6 / "README_vi.md",
    ]
    if risk_rows:
        artifact_paths.append(p6 / "candidates_risk_screening.csv")
    manifest = {
        "schema_version": 1,
        "status": "P6_HANDOFF_READY",
        "training_status": training["status"],
        "evaluation_status": "SCIENTIFIC_EVALUATION_COMPLETE",
        "risk_screening_status": evaluation["behavioral_screening"]["status"],
        "artifacts": [
            {
                "path": str(path.resolve().relative_to(run.resolve())).replace("\\", "/"),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    manifest_path = run / "result_package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
