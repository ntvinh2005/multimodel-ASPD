# Handoff khoa học — Qwen2.5 two-model ASPD 100M

Hướng dẫn thao tác đầy đủ nằm tại [qwen25_aspd_original_100m_runbook_vi.md](qwen25_aspd_original_100m_runbook_vi.md). Kết quả cuối cùng và các số liệu review chính nằm tại [results/qwen25_aspd_original_100m_final_report.md](results/qwen25_aspd_original_100m_final_report.md). Tài liệu này chốt thiết kế, trạng thái handoff và ranh giới kết luận.

## Thiết kế đã khóa

- Base: `unsloth/Qwen2.5-7B-Instruct@a75c9dc945567a9b6f568b8503a0307731607bee`.
- Target: `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports@6aa7e935c74c519c66555a111d1e6982059bd82d`, cùng pinned Base revision.
- Grounding `R_mid^(n)`: input của `model.layers.14.post_attention_layernorm`.
- Matrix input `X_O^(n)`: input riêng của `model.layers.14.self_attn.o_proj`.
- Chỉ phân rã `W_O^(n)` tại L14 thành các component `P_c^(n)=u_c^(n)v_c^(n)T`.
- Shared gate dùng mean aggregation qua đúng hai `R_mid`; parameter factors vẫn riêng theo model.
- 36,864 component, BatchTopK `K=32`, pool 2,048 token/rank; 2 DDP trainer ranks; global 4,096 positions/step.
- Main: 24,414 bước = 99,999,744 exposures. Không có supervised risk loss hoặc Delta-Crosscoder loss.
- Training corpus là pinned FineWeb general-domain. Behavioral risk/control/neutral prompts chỉ dùng held-out evaluation, không trộn vào train.
- L7–L13 experiment vẫn paused và không nằm trong workflow này.

## Workflow duy nhất

1. CPU dataset job đóng băng train/calibration/validation shards, provenance, duplicate/near-duplicate và contamination audit.
2. Một 3×B200 technical smoke: 0→50, checkpoint, restart, resume 50→100, evaluator-load test, GO/NO-GO.
3. Một 3×B200 main run 100M, chỉ được phép khi smoke là `GO`.
4. Một 1×B200 full held-out evaluation và export P6 package.

Các sbatch tương ứng:

- `slurm/multimodel/qwen25_aspd_original_100m_prepare_data.sbatch`
- `slurm/multimodel/qwen25_aspd_original_100m_benchmark.sbatch`
- `slurm/multimodel/qwen25_aspd_original_100m_train.sbatch`
- `slurm/multimodel/qwen25_aspd_original_100m_evaluate.sbatch`

Mỗi script ghi rõ đúng ba rationale: mục đích, lý do tài nguyên và runtime dự kiến. Không script nào tự submit job khác.

## Result contract

Training complete phải có final step 24,414, exposure 99,999,744, paired rank checkpoint + atomic marker, SHA-256 đúng, finite metrics, coherent cursor và DDP parameters giống nhau. Rank-local learned threshold/dead counters có thể khác vì recipe khóa `sync_statistics=false`; rank 0 là canonical cho inference/P6.

Evaluation dùng global sufficient statistics để báo cáo riêng:

- Residual FVU của Base và Target.
- L14 `o_proj` output FVU/RMS của Base và Target.
- `ΔY_parameter(X)=(W_Target-W_Base)X_Base` NSE/RMS/cosine.
- `ΔY_input=W_Target(X_Target-X_Base)` và total Base–Target difference.
- Numerical identity của true computation và ASPD reconstruction.
- Utilization, learned JumpReLU so với BatchTopK, contribution distribution của cả 36,864 component.
- Clean-forward observational screening trên locked risk/matched-control/neutral prompts.

Final rank-0 checkpoint là nguồn sự thật duy nhất cho `u`, `v`, `b_dec`, `b_out`, activation decoder/center và threshold. `component_metrics.csv` là bảng thống kê/ranking, không thay thế parameter tensors. Helper `extract_original_component_for_p6` lấy một component đã copy khỏi checkpoint để P6 không vô tình mutate decomposition.

## Trạng thái hoàn tất ngày 2026-10-25

- Actual three-process CPU/Gloo: `VERIFIED`.
- Frozen 100M dataset: `READY` — 195,312 train sequences/99,999,744 positions, 64 calibration sequences/32,768 positions và 3,906 validation sequences/1,999,872 positions.
- GPU smoke: `GO` — checkpoint/restart/resume, evaluator load và DDP equality đều đạt.
- 100M training: `100M_TRAINING_COMPLETE` — đúng 24,414 bước/99,999,744 exposures.
- Scientific evaluation: `SCIENTIFIC_EVALUATION_COMPLETE` — đủ 36,864 component.
- Behavioral screening: `VERIFIED` — 60 matched risk/control pairs.
- P6 handoff: `P6_HANDOFF_READY` — package 26 artifact đã kiểm lại size và SHA-256, không mismatch.

Checkpoint rank 0 vẫn là nguồn sự thật cho P6. Behavioral screening và candidate ranking chỉ là bằng chứng quan sát; không component nào được gọi là causal risk mechanism trước can thiệp P6.
