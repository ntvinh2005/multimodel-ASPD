# Kiểm toán và bàn giao two-model ASPD D0/S1 L14 O-projection

Ngày chốt cục bộ: 2026-10-25. Tài liệu này so sánh với [bài báo ASPD](https://arxiv.org/abs/2609.37731) và [mã nguồn ASPD chính thức](https://github.com/tue147/weights-read-write). Không có job GPU/HiPerGator nào được chạy trong lần chuẩn bị này.

## Kết luận khoa học ngắn

Mã cuối cùng vẫn là **two-model ASPD**: Base và Target dùng chung một mã thưa được đọc từ hai residual stream sạch, nhưng có decoder activation và các cặp vector hạng một đọc/ghi riêng cho từng model. Chỉ ma trận `model.layers.14.self_attn.o_proj` được phân rã. Gate đọc input của `model.layers.14.post_attention_layernorm`; input thật của ma trận là input riêng của `o_proj`, ký hiệu `X_O`. Hai tensor này không bị đánh tráo.

Training chưa thể chạy vì corpus khoa học chưa được đóng băng. Cấu hình cố ý để `dataset_status: BLOCKED`: hiện chỉ có nguồn general FineWeb đã ghim revision, chưa có nguồn độc lập, đủ đa dạng và đã kiểm tra trùng lặp cho extreme-sports risk, sports-safety control và neutral. Đây là blocker dữ liệu, không phải lý do để tự thay dataset hoặc giảm 100M token.

## Bảng so sánh phương pháp

| Aspect | Original ASPD | Current implementation trước sửa | Status | Required change / kết quả cuối |
|---|---|---|---|---|
| Model | Một Qwen3-8B | Base/Target Qwen2.5-7B | REQUIRED_MULTIMODEL_ADAPTATION | Giữ hai model và cùng token IDs; không đổi về single-model |
| Layer/matrix | Qwen3 L17 `o_proj` trong config gốc | Có thí nghiệm cũ L7–L13 | REQUIRED_MULTIMODEL_ADAPTATION | Cô lập config mới ở Qwen2.5 L14 `o_proj`; không tiếp tục L7–L13 |
| Gate site | resid-mid sau attention cho O | Hook có sẵn nhưng cần kiểm chứng thứ tự | MATCHES_ORIGINAL | Test Qwen2 xác nhận `o_proj` chạy trước input hook của `post_attention_layernorm` |
| Matrix input | Input thật của ma trận | Cache có `X`, nhưng phải ngăn dùng residual thay thế | MATCHES_ORIGINAL | Streaming truyền riêng `R_mid` và `X_O`; loss dùng `W_O X_O` |
| Shared gate | Encoder untied đọc residual, ReLU rồi BatchTopK | Encoder tuyến tính tổng hợp hai model | REQUIRED_MULTIMODEL_ADAPTATION | Mean theo trục model tạo một gate chung; đây là adaptation rõ ràng, không phải loss ΔY |
| Encoder centering | `relu((r-b_dec) W_enc)` | Bias encoder và bias reconstruction là hai parameter khác nhau | SCIENTIFIC_MISMATCH | Dùng đúng cùng một parameter cho centering và bias reconstruction của từng model |
| Encoder/decoder init | Cột encoder unit norm; decoder khởi tạo transpose rồi untied | Có unit norm nhưng bias sharing thiếu | SCIENTIFIC_MISMATCH | Unit-norm theo đúng quy ước tensor; decoder activation seed từ encoder, sau đó tối ưu untied |
| Rank-one component | `u_c v_c^T`, component `b_dec`, `b_out` | Có U/V nhưng P6 đọc `v^T x` | SCIENTIFIC_MISMATCH | Reconstruction và P6 đều đọc `v^T(x-b_dec)`; `b_out` chỉ cộng một lần |
| Internal loss | FVU của output ma trận | FVU theo từng model/matrix | MATCHES_ORIGINAL | Giữ nguyên, target là phép tính ma trận thật |
| Activation loss | Matryoshka prefix MSE chia variance | Mean FVU trên cùng các prefix | MATCHES_ORIGINAL | Hai biểu thức tương đương vì dùng cùng mẫu số variance; có cả prefix bias-only |
| AuxK | Top dead preactivations, residual không detach, hệ số 0.03125 | Residual từng bị detach | SCIENTIFIC_MISMATCH | Bỏ detach, xử lý nhánh auxiliary-zero như code gốc, K_aux=2048 |
| Hai-model loss scale | Không có trục model trong bản gốc | Tổng theo model làm gradient gấp đôi | REQUIRED_MULTIMODEL_ADAPTATION | Mean tường minh trên Base/Target; toy gradient chứng minh shared-gradient giữ cùng thang |
| BatchTopK | K=32 trong pool 2048 token | Có BatchTopK nhưng thiếu state threshold/dead parity | ENGINEERING_DEVIATION | Controller checkpoint hóa threshold EMA và dead counters; mỗi rank có đúng một pool 2048 |
| Inference gate | JumpReLU với threshold EMA 0.01 | Historical P6 dùng calibration riêng | ENGINEERING_DEVIATION | Checkpoint dùng learned threshold; P6 lịch sử vẫn bất biến và được ghi nhãn là policy khác |
| `sync_statistics` | `false`, mỗi DP rank có local statistics | Chưa mô tả canonical rank | MATCHES_ORIGINAL | Giữ false; rank 0 là checkpoint canonical để inference, đồng thời báo rõ rank-local buffer |
| `use_delta_component` | Cờ của framework decomposition/sampling; không phải Base–Target supervised loss | Dễ bị nhầm với Delta-Crosscoder | MATCHES_ORIGINAL | Giữ true trong fingerprint; không thêm DeltaY loss hay risk loss |
| ΔY parameter-only | Không phải objective ASPD gốc | Evaluator cũ có decomposition nhưng affine `b_out` bị cộng vào input-change | SCIENTIFIC_MISMATCH | Tính input-change bằng hiệu hai reconstruction Target để `b_out` triệt tiêu |
| Calibration | 8 batch train-only trong code gốc | 8 batch đầu rồi bị replay làm train | ENGINEERING_DEVIATION | Split calibration độc lập 32,768 token; không tái sử dụng trong 99,999,744 exposures |
| Resume | Checkpoint framework gốc | Pointer có thể xuất hiện trước khi rank 1 hoàn tất | ENGINEERING_DEVIATION | Commit marker nguyên tử chỉ sau barrier và SHA256 của cả hai rank; khôi phục RNG/cursor/state |
| Dataset | Pile streaming, 2B token ở thí nghiệm gốc | Mix 100M chưa có đủ nguồn thật | NOT_VERIFIED | Fail closed ở `BLOCKED`; cần manifest immutable, provenance từng sample, zero overlap/duplicates |
| Throughput | Không suy ra cho topology mới | Chưa đo cùng workload | NOT_VERIFIED | Benchmark 3×B200, 245 steps (~1.00M positions), GO khi ≥4,500 positions/s |

`FVU` là “fraction of variance unexplained”: 0 là tái tạo hoàn hảo, 1 tương đương dự đoán chỉ bằng trung bình. `NSE` là squared error chia năng lượng của target; nó đặc biệt phù hợp với ΔY nhỏ. “Read vector” `v` chọn đại lượng nào trong input; “write vector” `u` chỉ hướng ghi ra output; tích ngoài `u v^T` là một cơ chế ma trận hạng một.

## Cấu hình đóng băng

Nguồn duy nhất có thẩm quyền cho giá trị hiệu lực là `configs/multimodel/qwen25_extreme_sports/aspd_d0_s1_l14_o_100m_original.yaml`. CLI audit kiểm tra toàn bộ invariant thay vì chỉ tin YAML.

- Base: `unsloth/Qwen2.5-7B-Instruct@a75c9dc945567a9b6f568b8503a0307731607bee`.
- Target adapter: `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports@6aa7e935c74c519c66555a111d1e6982059bd82d`, ghép vào đúng Base revision.
- Qwen2, 28 layer, hidden size 3,584; tokenizer Base/Target có cùng SHA256 `9c5ae00e...7fdafa`.
- L14 `o_proj` có shape `[3584,3584]`. Audit local tính `W_target=W_base+11.3137085(B@A)`, với LoRA rank 32, α=64, RS-LoRA. Norm delta là `0.4237276`, bằng khoảng `0.00776247` norm Base.
- C=36,864; K=32; pool=2,048; sequence=512; 4 sequence/rank; 2 training rank; global batch=8.
- 24,414 optimizer updates × 4,096 positions = 99,999,744 exposures. Seed 0.
- AdamW LR `3e-4`, betas `(0.9,0.99)`, weight decay 0, constant schedule, warmup 0, không gradient clipping; autocast BF16, trainable parameters FP32.
- Activation loss coefficient 1; internal loss coefficient 1; AuxK coefficient 0.03125; AuxK top-k 2,048; dead threshold 4,882 pools; alive window 10,000,000 token.
- Matryoshka fractions `[0.0625,0.0625,0.125,0.25,0.5]`; threshold EMA LR 0.01; `sync_statistics=false`.
- `component_arch=transcoder`, `component_init=unit_norm`, `encoder_init=unit_norm`, `sampling=continuous`, `sigmoid_type=leaky_hard`, `n_mask_samples=1`, `use_delta_component=true`.

Phân bổ train theo sequence 512 để không có token lẻ: general 80,000,000; risk 9,999,872; sports-control 9,999,872. Held-out 1,999,872: general 1,500,160; risk 166,912; sports-control 166,400; neutral 166,400. Calibration độc lập 32,768. Các con số này mới là budget; nguồn risk/control/neutral vẫn chưa được duyệt.

## Implementation diff

- `config.py` và frozen YAML: schema original-recipe, topology, dtype/schedule, calibration budget và toàn bộ invariant 100M.
- `components.py`, `model.py`, `batch_topk.py`, `losses.py`: transcoder bias/centering, unit initialization, sparse local reconstruction, model-axis mean, AuxK, Matryoshka, threshold/dead/alive state.
- `training.py`: truyền original-recipe config khi dựng model và chặn generic cached trainer để không replay calibration hoặc phá topology.
- `original_recipe.py`: fail-closed method/data audit, exact cursor, local artifact/LoRA audit và resource arithmetic.
- `original_streaming.py`: producer + hai DDP trainer, blocking bounded stream, token/provenance identity, disjoint calibration, paired atomic checkpoints và exact resume.
- `original_evaluation.py`: held-out group metrics, ΔY decomposition, utilization và component norm/mass artifacts.
- `causal.py`, `fidelity.py`, `original_steering.py`: component input centering, affine-bias cancellation, raw clean gate cho recipe mới và canonical checkpoint adapter.
- Bốn CLI mới: audit, train, evaluate và benchmark GO/NO-GO.
- Ba Slurm script, P6 JSON schema, test original-recipe và test P6 centered subtraction.
- `pd_transcoder.py`: backward-compatible default cho legacy statistic-reducer mocks; config thật vẫn tôn trọng `sync_statistics=false`.

Không comment source hiện hữu nào bị xóa, dịch hoặc đổi nội dung; chỉ có comment giải thích mới được thêm vào nơi behavior parity cần làm rõ. Không file/artifact lịch sử nào bị xóa hay ghi đè.

## Streaming, distributed và resume

Rank toàn cục 0 giữ hai source LLM sạch và chạy forward tuần tự trên cùng tensor token IDs CPU được clone. Rank 1 và 2 chỉ train ASPD DDP. Mỗi rank nhận bốn sequence, tức một pool 2,048 token; global update có 4,096 token. `dist.send/recv` blocking tạo backpressure hữu hạn (effective depth 1), nên không có cache activation 100M bắt buộc.

Mỗi batch truyền `input_ids`, `sequence_ids`, `provenance_ids`, mask, hai `R_mid` và hai `X_O`. Trainer kiểm tra sequence ID đúng cursor trước mỗi update. Checkpoint chứa model, optimizer, step, stream cursor, sampler cursor cho cả hai rank, Python/Torch/CUDA RNG, threshold/dead/alive buffers và config SHA256. `latest_checkpoint.pt` chỉ được commit sau khi cả hai rank ghi xong; nó chứa SHA256 của cặp checkpoint.

Resume chính xác đã được kiểm tra bằng CPU synthetic interrupted-vs-uninterrupted. Khi chạy thật, dùng cùng output directory và đặt `RESUME=auto`; không chỉ định một checkpoint đơn lẻ. Nếu marker/cặp hash không khớp, resume fail closed.

## Evaluation và steering

Evaluator held-out dùng learned JumpReLU threshold và thống kê FP64 toàn bộ dataset, không lấy trung bình các FVU theo batch. Nó xuất:

- residual activation FVU cho Base và Target;
- `o_proj` computation FVU cho Base và Target;
- total ΔY, parameter-only ΔY và input-change ΔY với NSE/RMS;
- identity `total = parameter + input-change` cho true và reconstruction;
- true/predicted ΔY RMS và error cancellation;
- L0 JumpReLU, L0 BatchTopK, support Jaccard, active/dead fraction, effective utilization;
- full fire counts, read/write norms và mass `||u||·||v||` trong `component_metrics.pt`;
- kết quả riêng cho general/risk/sports-control/neutral.

Low FVU không chứng minh cơ chế risk. Kết luận risk cần P6 causal riêng với gate sạch đã freeze, gate-off, random same-L2 nhiều seed, active matched và orthogonal matched controls. Historical P6 dùng threshold calibration theo component; threshold đó không phải learned JumpReLU threshold, không phải criterion gán nhãn risk, và không phải ngưỡng PASS/FAIL của thí nghiệm. Schema record mới bắt buộc ba trường này riêng biệt tại `docs/multimodel/p6_original_recipe_record.schema.json`. Không artifact P6 lịch sử nào bị sửa.

Core P6 đã tương thích với component `b_dec`, có helper nạp canonical rank-0 checkpoint và snapshot gate sạch. Các orchestrator P6 lịch sử cố ý không được dùng trực tiếp cho checkpoint mới: chúng gắn với normalization cache, component IDs và provenance của thí nghiệm cũ. Một steering run mới phải chọn component từ checkpoint 36,864-feature mới và giữ artifact lịch sử bất biến.

## Tài nguyên và runtime

Tính toán tĩnh cho width 3,584:

- Trainable parameters: 1,057,022,976.
- Mỗi training rank: parameter FP32 3.94 GiB, gradient 3.94 GiB, Adam moments 7.88 GiB; mức sàn 15.75 GiB trước activation, kernels và communication buffers.
- Producer: hai bộ weight BF16 7B khoảng 26.08 GiB, chưa gồm activation và runtime workspace. B200 192 GB có biên dung lượng lớn, nhưng số đo benchmark vẫn là thẩm quyền.
- Một rank checkpoint model+Adam khoảng 11.81 GiB; một cặp khoảng 23.63 GiB; giữ ba cặp khoảng 70.88 GiB, cộng log/evaluation. Không có 100M activation cache.
- CPU RAM yêu cầu dự kiến: 256 GB cho benchmark/train node; 192 GB cho evaluation. Đây là request bảo thủ, chưa đo.
- Tốc độ tối thiểu toán học cho 7 giờ là 3,968.25 positions/s. GO provisional là ≥4,500 sustained positions/s trên đúng workload. 100M ở 4,500 positions/s mất khoảng 6.17 giờ training/capture, tức khoảng 18.5 B200 GPU-hours; toàn budget tối đa 10 giờ tương đương tối đa 30 B200 GPU-hours nếu cả ba GPU hoạt động suốt.
- Benchmark 245 update xử lý 1,003,520 positions và không phải scientific run. Nếu dưới 4,500, kết quả là `NO_GO_RUNTIME`; không tự giảm C/K/token budget.

## HiPerGator handoff (chỉ chuẩn bị, chưa chạy)

Scripts:

- `slurm/multimodel/qwen25_aspd_original_100m_benchmark.sbatch`: 3×B200, 45 phút, audit dataset/model trước khi chạy và tạo `benchmark_go.json`.
- `slurm/multimodel/qwen25_aspd_original_100m_train.sbatch`: 3×B200, 8 giờ, fresh hoặc `RESUME=auto`.
- `slurm/multimodel/qwen25_aspd_original_100m_evaluate.sbatch`: 1×B200, 1 giờ 15 phút, kiểm tra hai rank checkpoint rồi đánh giá held-out.

Output dự kiến:

- Benchmark: `out/multimodel/benchmarks/qwen25_aspd_original_100m/<job-id>/`.
- Main: `out/multimodel/runs/qwen25_aspd_original_100m/`.
- Log: `slurm/logs/q25-aspd-orig-{bench,100m,eval}-<job-id>.{out,err}` theo `%x-%j`.
- Evaluation: `out/multimodel/runs/qwen25_aspd_original_100m/evaluation/held_out_step_<step>/`.

Trước khi submit thủ công: chuyển dataset status sang READY chỉ sau khi manifest vượt `--require-dataset-ready`; lưu hash config/manifest; kiểm tra đủ dung lượng; chạy benchmark; chỉ chạy main nếu verdict GO. Sau job, kiểm tra `latest_checkpoint.pt`, SHA256 của cả cặp rank, `producer_provenance.json`, step cuối 24,414, exposures 99,999,744, finite metrics và source weights unchanged. Có thể dùng `sha256sum` cho config, manifest, checkpoints và evaluation artifacts; hash đã được ghi chéo trong marker/report.

## Bằng chứng test cục bộ

- Focused ASPD/multimodel regression sau thay đổi: `77 passed in 6.11s`.
- Toàn bộ suite trừ file server legacy: `788 passed, 15 skipped, 3 xpassed in 36.37s`.
- `tests/test_feature_report.py`: `4 passed in 9.21s`.
- `tests/test_feature_server.py`: ba test đầu pass; test `test_server_lists_sites_and_serves_one_feature` không hoàn tất trong timeout 60 giây của môi trường local. File này không đi qua code training/evaluation đã sửa. Không báo nó là pass.
- Ruff trên toàn bộ file thay đổi: pass. `bash -n` cho ba Slurm script: pass. `git diff --check`: pass. CLI config resolution: pass. Audit pinned artifacts: pass; dataset audit trả `BLOCKED` đúng thiết kế.
- Không chạy full-model Qwen forward, NCCL 2-rank, 3×B200 throughput, GPU memory profiling, generation, judge hay steering. Đây là các kiểm tra HPC còn lại.

## Trạng thái bàn giao

- `METHOD_AUDIT = PASS_LOCAL`
- `ORIGINAL_RECIPE_PARITY = PASS_LOCAL_WITH_REQUIRED_TWO_MODEL_ADAPTATION`
- `TWO_MODEL_IMPLEMENTATION = PASS_LOCAL`
- `GROUNDING_CORRECTNESS = PASS_CPU_ARCHITECTURE_TEST; FULL_MODEL_FORWARD_NOT_RUN`
- `DATA_INTEGRITY = BLOCKED_MISSING_APPROVED_RISK_CONTROL_NEUTRAL_CORPORA`
- `STREAMING = PASS_CPU_CONTRACTS; HPC_THROUGHPUT_NOT_VERIFIED`
- `DISTRIBUTED_TRAINING = PASS_MATHEMATICAL_AND_CONFIG_TESTS; 3_GPU_NOT_RUN`
- `EXACT_RESUME = PASS_CPU_SYNTHETIC; MULTI_GPU_NOT_RUN`
- `LOCAL_TESTS = PASS_FOCUSED; BROAD_788_PASS_15_SKIP_3_XPASS; ONE_LEGACY_SERVER_TEST_TIMEOUT`
- `HPC_BENCHMARK = NOT_RUN`
- `100M_TRAINING = NOT_RUN`
