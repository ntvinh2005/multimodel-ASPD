# Vòng lặp kết quả HiPerGator → máy local

## Output mong đợi của smoke job

Job `qwen25_aspd_p6_random_control_smoke.sbatch` chỉ kiểm tra code bằng dữ liệu tổng hợp CPU. Nó không tạo kết quả steering trên Qwen và không có effect size để diễn giải khoa học.

Khi submit thành công, Slurm trả về một job ID, ví dụ `48123456`. Job tạo bundle:

```text
out/multimodel/diagnostics/p6_random_control_smoke/job_48123456/
├── smoke_summary.json
├── pytest.txt
├── ruff.txt
├── fixed_cli_help.txt
├── generation_cli_help.txt
└── SHA256SUMS
```

`smoke_summary.json` là file cần đọc trước. Một run thành công có:

```json
{
  "artifact_type": "ASPD_P6_RANDOM_CONTROL_CPU_SMOKE",
  "scientific_measurement": false,
  "status": "PASS",
  "exit_code": 0,
  "checks": {
    "pytest": "PASS",
    "ruff": "PASS",
    "fixed_cli_help": "PASS",
    "generation_cli_help": "PASS"
  }
}
```

Hai Slurm log vẫn nằm tại `slurm/logs/mm-p6-random-smoke-<JOB_ID>.out` và `.err`. Bundle được tạo cả khi job fail; check đang chạy lúc lỗi sẽ có trạng thái `RUNNING`, các check chưa đến sẽ là `NOT_RUN`.

## Chạy trên HiPerGator

Từ thư mục repo đã pull commit cần kiểm tra:

```bash
REMOTE_REPO=$(pwd)
echo "REMOTE_REPO=${REMOTE_REPO}"
JOB_ID=$(sbatch --parsable slurm/multimodel/qwen25_aspd_p6_random_control_smoke.sbatch)
JOB_ID=${JOB_ID%%;*}
echo "JOB_ID=${JOB_ID}"
squeue -j "${JOB_ID}"
```

Sau khi job kết thúc:

```bash
sacct -j "${JOB_ID}" --format=JobID,State,ExitCode,Elapsed,MaxRSS
cat "out/multimodel/diagnostics/p6_random_control_smoke/job_${JOB_ID}/smoke_summary.json"
```

## Kéo kết quả về local

Chạy từ PowerShell trên máy local, tại root của repo. `Remote` có thể là hostname hoặc SSH alias bạn vẫn dùng để vào HiPerGator. `RemoteRepo` phải là đường dẫn tuyệt đối tới repo trên HiPerGator.

```powershell
.\scripts\multimodel\pull_hpg_job.ps1 `
  -Remote 'vinhnguyen1@hpg.rc.ufl.edu' `
  -RemoteRepo '/absolute/path/to/repo' `
  -JobId '48123456'
```

Script từ chối ghi đè một `job_<ID>` đã có, kéo cả artifact bundle và hai Slurm log, kiểm tra SHA-256, rồi in status/exit code/commit. Thư mục `out/` đã được Git ignore; không commit artifact thí nghiệm vào source branch.

Sau khi kéo xong, gửi lại đường dẫn local dạng:

```text
out/multimodel/diagnostics/p6_random_control_smoke/job_48123456
```

Từ đó có thể đọc trực tiếp toàn bộ artifact để sửa code và lặp lại bằng một job ID mới.
