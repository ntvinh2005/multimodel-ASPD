# Runbook HiPerGator — two-model ASPD 100M

Tài liệu này là đường chạy chuẩn cho thí nghiệm Qwen2.5 Base ↔ extreme-sports Target tại duy nhất `model.layers.14.self_attn.o_proj`. Agent trên HiPerGator không cần sửa YAML. Chỉ checkout đúng commit, khai báo `MM_ASPD_ROOT`/`HF_HOME`, rồi submit các sbatch theo thứ tự dưới đây.

Researcher phải gửi commit SHA sau khi review/push. Agent dùng SHA đó, không chạy từ branch đang thay đổi:

```bash
cd "${MM_ASPD_ROOT}/repo"
git fetch origin
export RESEARCHER_COMMIT_SHA="paste-the-reviewed-commit-sha-here"
git checkout --detach "${RESEARCHER_COMMIT_SHA}"
test -z "$(git status --porcelain)"
```

## 0. Kiểm tra checkout và Python

```bash
cd "${MM_ASPD_ROOT}/repo"
source "${MM_ASPD_ROOT}/env.sh"
module load conda
export UV_PYTHON=3.13

git status --short
git rev-parse HEAD
.venv/bin/python --version
.venv/bin/python -m aspd.multimodel.cli.audit_original_recipe \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml
```

Python phải là `3.13.x`. Audit trước dataset có thể trả `dataset.status=NOT_RUN`; đó là trạng thái đúng nếu chưa materialize FineWeb. Không dùng `uv run` với Python 3.14 của module Conda.

Nếu checkout mới chưa có `.venv`, chạy đúng một lần trên login node:

```bash
uv python install 3.13
UV_PYTHON=3.13 uv sync --frozen --all-groups
.venv/bin/python --version
```

Sau đó mọi sbatch gọi thẳng `.venv/bin/python`; agent không cần activate Conda env và không để `uv` tự chọn Python 3.14.

Biến bắt buộc trong `${MM_ASPD_ROOT}/env.sh`:

```bash
export HF_HOME="/blue/<group>/<user>/hf-cache"
export HF_TOKEN="<token-if-required>"
```

Luôn chạy `mkdir -p slurm/logs` trước lần `sbatch` đầu. `squeue` chỉ hiển thị job đang queue/running; job đã xong phải xem bằng `sacct`.

### CPU verification bundle trên Linux

PyTorch Windows hiện tại không cung cấp Gloo device, nên agent chạy một CPU job trên HiPerGator để hoàn tất bằng chứng multiprocess Linux trước GPU:

```bash
VERIFY_JOB_ID=$(sbatch --parsable \
  slurm/multimodel/qwen25_aspd_original_100m_local_verification.sbatch)
echo "VERIFY_JOB_ID=${VERIFY_JOB_ID}"
```

Kỳ vọng `COMPLETED`, dòng cuối `LOCAL_VERIFICATION_COMPLETE`, và `out/multimodel/local_verification/job_<VERIFY_JOB_ID>/gloo_report.json` có `status=LOCALLY_TESTED`, exact model/optimizer/RNG/cursor comparison và `ddp_synchronized_state_equal=true`. Đây là một bundle duy nhất cho các kiểm tra nhỏ cùng mục đích; không phải scientific experiment.

## 1. Chuẩn bị và kiểm tra dataset

```bash
mkdir -p slurm/logs
DATA_JOB_ID=$(sbatch --parsable --export=ALL,ALLOW_DATA_PREP=1 \
  slurm/multimodel/qwen25_aspd_original_100m_prepare_data.sbatch)
echo "DATA_JOB_ID=${DATA_JOB_ID}"
squeue -j "${DATA_JOB_ID}"
```

Sau khi job biến mất khỏi `squeue`:

```bash
sacct -j "${DATA_JOB_ID}" --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS
tail -n 80 "slurm/logs/q25-aspd-data-100m-${DATA_JOB_ID}.out"
.venv/bin/python -m aspd.multimodel.cli.prepare_original_recipe_data \
  configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml \
  --validate-only
```

Kỳ vọng: `status=READY`, 195,312 train sequences/99,999,744 positions; 64 calibration sequences/32,768 positions; 3,906 validation sequences/1,999,872 positions. Manifest chuẩn là `out/multimodel/data/qwen25_aspd_original_100m/manifest.json`. Không dùng `--overwrite` trừ khi researcher chủ động quyết định bỏ artifact dataset lỗi.

## 2. Một smoke GPU 100 bước có resume thật

```bash
SMOKE_JOB_ID=$(sbatch --parsable --export=ALL,ALLOW_SMOKE=1 \
  slurm/multimodel/qwen25_aspd_original_100m_benchmark.sbatch)
echo "SMOKE_JOB_ID=${SMOKE_JOB_ID}"
squeue -j "${SMOKE_JOB_ID}"
```

Job chạy bước 0→50, ghi paired checkpoint, khởi động lại 3 process, resume 50→100, load checkpoint bằng evaluator trên 8 validation sequences, rồi tạo:

```text
out/multimodel/smoke/qwen25_aspd_original_100m/job_<SMOKE_JOB_ID>/smoke_verification.json
```

Kiểm tra:

```bash
sacct -j "${SMOKE_JOB_ID}" --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS
.venv/bin/python -c "import json; p='out/multimodel/smoke/qwen25_aspd_original_100m/job_${SMOKE_JOB_ID}/smoke_verification.json'; d=json.load(open(p)); print(json.dumps(d,indent=2))"
```

Chỉ chuyển sang main khi `status` đúng bằng `GO`, `final_step=100`, `evaluation_load_verified=true`, DDP parameters bằng nhau, optimizer đã thay đổi giữa bước 50 và 100, và sustained throughput ≥4,500 paired positions/s. `NO_GO_RUNTIME` là kết quả đo hợp lệ nhưng không cho phép main; gửi JSON và hai Slurm log về để chẩn đoán.

## 3. Một main run 100M duy nhất

```bash
MAIN_JOB_ID=$(sbatch --parsable \
  --export=ALL,ALLOW_100M=1,SMOKE_JOB_ID="${SMOKE_JOB_ID}" \
  slurm/multimodel/qwen25_aspd_original_100m_train.sbatch)
echo "MAIN_JOB_ID=${MAIN_JOB_ID}"
squeue -j "${MAIN_JOB_ID}"
```

Output cố định, tách hoàn toàn khỏi smoke:

```text
out/multimodel/runs/qwen25_aspd_original_100m/
```

Nếu job hết walltime sau khi đã có `latest_checkpoint.pt`, submit lại đúng lệnh trên: sbatch tự chọn `--resume auto`. Không đổi config, không dùng smoke checkpoint, không tạo run 10M/30M/50M. Nếu thư mục run đã có dữ liệu nhưng chưa có atomic marker, dừng và gửi log; pipeline cố ý không đoán trạng thái resume.

Main chỉ hoàn tất khi `checkpoint_verification_report.json` có `status=100M_TRAINING_COMPLETE`, step 24,414 và exposures 99,999,744.

## 4. Held-out evaluation và P6 package

```bash
EVAL_JOB_ID=$(sbatch --parsable --export=ALL,ALLOW_EVALUATION=1 \
  slurm/multimodel/qwen25_aspd_original_100m_evaluate.sbatch)
echo "EVAL_JOB_ID=${EVAL_JOB_ID}"
squeue -j "${EVAL_JOB_ID}"
```

Job từ chối chạy nếu main verification chưa complete. Kết quả chính:

```text
evaluation/held_out_evaluation.json
evaluation/summary_vi.md
evaluation/component_metrics.pt
evaluation/component_metrics.csv
evaluation/delta_computation_metrics.json
evaluation/plots/*.png
p6_handoff/candidates_computational.csv
p6_handoff/candidates_risk_screening.csv   # chỉ khi behavioral evidence VERIFIED
p6_handoff/component_parameter_index.json
p6_handoff/component_loading_contract.md
p6_handoff/README_vi.md
result_package_manifest.json
```

`component_metrics.csv` có đủ 36,864 component. Vector `u/v`, `b_dec/b_out`, activation decoder/center và learned threshold không bị copy thành nguồn sự thật thứ hai; chúng nằm trong final rank-0 checkpoint và được lấy bằng `load_original_decomposition_for_p6` + `extract_original_component_for_p6`.

## 5. Chuyển package về máy local

Chạy từ PowerShell tại repo local, sau khi evaluation job hoàn tất:

```powershell
.\scripts\multimodel\pull_hpg_original_100m.ps1 `
  -Remote 'vinhnguyen1@login.rc.ufl.edu' `
  -RemoteRepo '/blue/<group>/vinhnguyen1/multimodel-ASPD' `
  -LocalRepo (Get-Location).Path
```

Script từ chối overwrite, kéo cả checkpoint/training/evaluation/P6 package và kiểm SHA-256 của mọi artifact trong `result_package_manifest.json`. Nếu chỉ cần review nhanh trước khi tải checkpoint lớn, dùng `scp` riêng ba file nhỏ: `training_summary.json`, `checkpoint_verification_report.json`, `evaluation/summary_vi.md`.

## 6. Trạng thái khoa học

- Code/config được merge vào checkout không có nghĩa GPU smoke đã pass.
- `GO` của smoke không có nghĩa main 100M đã hoàn tất.
- `100M_TRAINING_COMPLETE` không có nghĩa tìm được risk mechanism.
- Ranking behavioral chỉ là observational candidate evidence. Kết luận causal phải chờ P6 intervention với frozen clean gate và matched controls.
