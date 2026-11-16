# Báo cáo tổng hợp các giả thuyết về interpretability của ASPD v3.1

Trạng thái: `COMPLETE_WITH_NONHUMAN_HELDOUT_H5_UPDATE`

Ngày tổng hợp: 2026-11-15

Checkpoint được đánh giá: step `61,036`, `500,006,912` accepted aligned-anchor exposures, checkpoint-manifest SHA-256 `735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969`.

## 1. Kết luận điều hành

Điểm Intruder khoảng 37% hiện tại **không phải là một ước lượng đã được hiệu chuẩn về semantic interpretability**. Bằng chứng mạnh nhất cho thấy evaluation instrument có vấn đề: Llama-3.3-70B có position bias nghiêm trọng, còn Mistral-Nemo độc lập cũng không vượt qua các reliability gate đã định trước. Vì vậy không thể dùng headline score để kết luận trực tiếp rằng dictionary chỉ có khoảng 37% feature có nghĩa.

Các kết quả hiện tại phân tách thành ba tầng:

1. **Measurement problem — bằng chứng mạnh nhất.** Judge và protocol cũ làm méo phép đo. Chưa xác định được độ lệch so với đánh giá hợp lệ là hướng xuống hay hướng lên.
2. **Representation problem — pattern chủ yếu yếu hoặc chưa nhận diện được.** Nhiều feature có nhiều context cluster ổn định; audit H4 không-human tìm thấy cả semantic lẫn syntactic/formatting/token-boundary features. Tuy nhiên held-out H5 follow-up không validate được common rule hoặc multiple unrelated rules nào trong 160 components. Stable clusters cũng xuất hiện gần ngang matched controls.
3. **Training-objective/capacity problem — chưa được chứng minh.** `C=8192`, detached internal gradients và training plateau đều là giả thuyết hợp lý nhưng chưa có controlled ablation đủ để kết luận nhân quả.

Kết luận tổng quát là interpretability hạn chế hiện nay nhiều khả năng là tổ hợp của **phép đo không hợp lệ** và **representation không đồng nhất/polysemantic**. Chưa có bằng chứng đủ mạnh để quy nguyên nhân chính cho capacity hoặc detached internal supervision.

## 2. Protocol hiện tại có theo ASPD gốc không?

Quick evaluation ban đầu **không theo đúng protocol ASPD gốc**. Các khác biệt chính gồm:

| Thành phần | Quick evaluation cũ | Official ASPD / Stage 1 đã sửa |
|---|---|---|
| Context window | 65 token | 41 token, khoảng ±20 quanh firing |
| Trials/component | Một parent task × năm rotation | 10 trial được lấy mẫu độc lập |
| Donor | Hard-negative matching mạnh hơn | Firing-density matched, tolerance 0.05 |
| Intruder position | Năm cyclic rotations | Random uniform; rotation chỉ là diagnostic riêng |
| Components | 100 PRE + 100 MID | 200 PRE + 200 MID |
| Judge decoding | Greedy | Server-default sampling, không tự ép greedy |
| Split | Held-out | Official code harvest trên training; Stage 1 giữ held-out như một v3.1 adaptation đã khai báo |

Stage 1 đã sửa phần lớn task-construction mismatch và tạo 4,000 paper-style trials cùng 2,000 rotation-diagnostic trials. Tuy nhiên đây vẫn là adaptation cho shared PRE/MID dictionaries của v3.1, không phải reproduction tuyệt đối. Quan trọng hơn, score mới vẫn chưa hợp lệ về khoa học vì judge không qua reliability gate.

## 3. Tổng hợp verdict H1–H10

| Hypothesis | Verdict | Kết luận ngắn |
|---|---|---|
| H1 — Judge/protocol mismatch | `PARTIALLY_SUPPORTED` | Evaluation design và judge behavior làm méo score, nhưng chưa biết hướng và độ lớn bias so với ground truth hợp lệ. |
| H2 — Donor ambiguity | `INCONCLUSIVE` | Nhiều task có thể không có semantic intruder duy nhất, nhưng chưa có donor-factorial hợp lệ. |
| H3 — Weak natural firings | `PARTIALLY_SUPPORTED` | Held-out frozen-rule presence cao hơn 9.00 pp ở strong so với weak events, nhưng absolute presence vẫn chỉ khoảng 31%. |
| H4 — Non-topical structure | `PARTIALLY_SUPPORTED` | Audit không-human tìm thấy 18/59 feature coherent về syntax/formatting/token boundary. |
| H5 — Polysemanticity | `NOT_SUPPORTED` | 0/160 qua validated multiple-rule gate; 153/160 không có reliable linguistic rule. Clustering controls cho stable-group rate gần như natural firings. |
| H6 — Capacity bottleneck | `INCONCLUSIVE` | MID reconstruction khó hơn và contribution tập trung, nhưng chưa chứng minh `C=8192` là binding constraint. |
| H7 — Cross-model mismatch | `INCONCLUSIVE` | SmolLM2–Granite align, hai cặp có Gemma không align đáng tin cậy; chưa support mismatch hoặc universality. |
| H8 — Objective mismatch | `INCONCLUSIVE` | Detached gradient được xác minh, nhưng activation và internal contribution vẫn tương quan dương. |
| H9 — Interpretability plateau | `INCONCLUSIVE` | Loss tiếp tục giảm, nhưng checkpoint 13k không còn để chạy comparison matched. |
| H10 — Extreme winners | `PARTIALLY_SUPPORTED` | Feature firing nhiều mang contribution lớn, nhưng chưa chứng minh chúng kém specific hoặc polysemantic hơn. |

## 4. Kết quả chi tiết theo từng giả thuyết

### H1 — Judge và protocol mismatch

#### Thí nghiệm

- Xây dựng lại paper-style Intruder theo official implementation gần nhất có thể.
- 400 components: 200 PRE và 200 MID.
- 10 trial độc lập/component, tổng 4,000 trials.
- Chạy thêm balanced five-position rotations.
- Kiểm tra độc lập với Mistral-Nemo.

#### Kết quả

| Site | Components | Trials | Accuracy | Feature-clustered 95% CI |
|---|---:|---:|---:|---:|
| PRE | 200 | 2,000 | 33.60% | 30.85–36.45% |
| MID | 200 | 2,000 | 33.70% | 31.20–36.25% |
| Macro | 400 | 4,000 | 33.65% | — |

Llama predicted các vị trí 1–5 với tỷ lệ lần lượt 3.73%, 41.86%, 20.16%, 7.48% và 26.78%, thay vì gần 20% mỗi vị trí. Accuracy theo true position dao động từ 8.36% đến 53.55%. Chỉ 17/400 parent tasks, tương đương 4.25%, được trả lời đúng ở cả năm rotations.

Mistral-Nemo cũng không hợp lệ: positive-control accuracy 76.4%, dưới gate 80%; position-bias p-value `9.25e-6`; position-conditioned accuracy spread 58 điểm phần trăm; canonical consistency gate thất bại.

#### Diễn giải

Hai model family độc lập đều có reliability problem. Do đó không được chọn judge chỉ vì judge đó cho ASPD score cao hơn. Kết quả support việc **measurement bị distortion**, nhưng không xác định score 37% bị bias xuống hay bias lên bao nhiêu.

#### Verdict

`PARTIALLY_SUPPORTED`.

### H2 — Donor không phải semantic outlier duy nhất

#### Bằng chứng hiện có

- Trong phép đo bằng judge đã fail, donor-density gap gần như không liên hệ với accuracy: `rho=0.021`.
- Same-source matching liên hệ âm với accuracy: `rho=-0.239`.
- Trong H4 agent audit, chỉ 23/59 item được đánh dấu có unique intruder rõ ràng. Con số này không tách được donor ambiguity khỏi feature incoherence.

#### Giới hạn

Chưa chạy comparison được kiểm soát giữa paper-like donor, hard-negative donor và independently verified semantic outlier trên cùng target contexts. Các association hiện tại đều bị confound bởi judge invalid hoặc một single nonhuman evaluator.

#### Verdict

`INCONCLUSIVE`.

### H3 — Weak natural firing events làm bẩn context

#### Thí nghiệm

Threshold được đóng băng trên calibration split, sau đó đo coverage trên 77,854 test anchors. Chỉ natural selected firings được dùng; không forced activation.

| Threshold | PRE event coverage / eligible features | MID event coverage / eligible features |
|---|---:|---:|
| Natural `g^s > 0` | 100% / 8,188 | 100% / 8,186 |
| `g^s >= 0.06` | 52.30% / 8,155 | 73.66% / 8,179 |
| `g^s >= 0.08` | 27.43% / 8,030 | 30.49% / 8,014 |
| `g^s >= 0.12` | 12.81% / 7,117 | 13.46% / 7,098 |
| `g^s >= 0.20` | 3.85% / 3,884 | 4.18% / 4,107 |
| Within-feature q90 | 13.86% / 7,489 | 14.37% / 7,519 |

Median activation strength tương quan với failed-judge accuracy ở `rho=0.421`; mean activation có `rho=0.314`.

Held-out H5 follow-up sau đó đóng băng common rules trên discovery contexts và test trên documents mới. Common-rule presence ở strong natural firings là 31.10%, so với 22.18% ở weak firings. Trên 131 components có cả hai bins, paired difference là `+9.00` điểm phần trăm, component-bootstrap 95% CI `+4.99` đến `+13.07`.

#### Diễn giải

Strong firings chứa frozen proposed pattern thường xuyên hơn weak firings trên held-out documents. Tuy nhiên absolute presence vẫn chỉ khoảng 31%, và đây là nonhuman rule-presence validation chứ không phải human semantic ground truth. Trade-off coverage của thresholding cũng lớn: q90 chỉ giữ khoảng 14% firing events.

#### Verdict

`PARTIALLY_SUPPORTED`.

### H4 — Feature biểu diễn syntax, formatting hoặc structure thay vì topic

#### Thiết kế

Một blinded package 60 item được phân tầng theo PRE/MID và failed-judge score band. Theo quyết định không dùng human rater, Codex thực hiện một lượt nonhuman qualitative audit. Response file ghi rõ `human_evaluator=false`; không có independent inter-rater agreement.

Trong lúc kiểm tra schema, answer của `Feature Set 001` bị lộ. Item này được đánh dấu contaminated và loại khỏi mọi estimate. Responses và adjudication rule cho 59 item còn lại được commit trước khi mở answer key.

#### Kết quả

| Agent classification | Count | Fraction |
|---|---:|---:|
| Coherent, mọi category | 23/59 | 39.0% |
| Syntactic | 10/59 | 16.9% |
| Formatting/document structure | 6/59 | 10.2% |
| Token boundary | 2/59 | 3.4% |
| Semantic | 5/59 | 8.5% |
| Mixed hoặc không có pattern rõ | 36/59 | 61.0% |

Tổng cộng 18/59, tương đương 30.5%, có non-topical coherent pattern. Trong đó PRE có 10/30 và MID có 8/29. Cả 18 agent-selected unique non-topical intruders đều trùng construction answer key.

Tuy nhiên 17/18 non-topical coherent items thuộc old-judge high-score band; intermediate có 1/24 và zero band có 0/10. Vì vậy audit cho thấy structural features thực sự tồn tại, nhưng không cho thấy judge đã bỏ lỡ nhiều structural pattern rõ ràng trong nhóm score thấp.

#### Giới hạn

- Đây là single language-model evaluator, không phải human evidence.
- Construction-key correctness chỉ cho biết chọn đúng donor đã cài vào task, không tự chứng minh semantic validity.
- Agent có thể chia sẻ language-model priors với judge.

#### Verdict

`PARTIALLY_SUPPORTED`.

### H5 — Feature polysemanticity

#### Thí nghiệm

- 120 feature được lấy mẫu theo PRE/MID và firing-frequency strata.
- 10,311 natural-firing contexts, 50–100 contexts/feature.
- Mistral-Nemo embeddings.
- Candidate `k=2..5`, 10 restarts và bootstrap adjusted-Rand stability.

#### Kết quả

- 78/120 feature, tương đương 65%, qua toàn bộ silhouette, minimum-size và stability gates.
- PRE: 45/60, tương đương 75%.
- MID: 33/60, tương đương 55%.
- Selected `k`: 44 feature có `k=2`, 53 có `k=3`, 20 có `k=4`, và 3 có `k=5`.
- Tỷ lệ stable multi-cluster theo frequency: low 65%, medium 70%, high 60%.

Các gallery thực tế cho thấy những feature tách code/HTML/API khỏi prose, hoặc biomedical text khỏi numerical/log-like records.

Follow-up confirmatory dùng thêm 40 random components, discovery/held-out document split, 625 frozen common/group rules và independent Phi validation trên 67,225 tasks. Kết quả:

- 0/160 components đạt single-common-rule gate.
- 0/160 đạt multiple-unrelated-rule gate.
- 153/160 không có reliable linguistic rule; 7/160 inconclusive.
- Trong đúng 78 historical multi-cluster components: 76 no-rule, 2 inconclusive.
- Stable multi-cluster rate là 96/160 trên natural firings, gần như controls: 93/160 random source/length-matched, 93/160 topic-matched và 94/160 frequency-matched other-component contexts.

#### Diễn giải

Stable context heterogeneity là phổ biến nhưng không đặc hiệu cho firing contexts. Nó không chuyển thành predictive evidence cho multiple unrelated linguistic rules. Kết quả phù hợp hơn với pattern yếu/chưa nhận diện được, đồng thời vẫn để ngỏ khả năng single proposer đã bỏ sót rule thật.

#### Verdict

`NOT_SUPPORTED` trong tested sample. Điều này không phải bằng chứng rằng polysemanticity không tồn tại ở toàn dictionary.

### H6 — `C=8192` là capacity bottleneck

#### Thí nghiệm quan sát

Held-out reconstruction được chạy trên 100,352 aligned anchors và toàn bộ 168 matrix targets.

#### Kết quả

- Activation FVU chuẩn hóa nằm trong khoảng 0.344–0.402 theo model/site.
- Mean matrix FVU của PRE q/k/v: 0.318–0.325.
- Mean matrix FVU của MID o/MLP: 0.439–0.470.
- Correlation giữa firing frequency và conditional matrix residual rất nhỏ: `rho=0.067` trên 16,384 features.
- Correlation giữa selected context-cluster count và frequency: `rho=-0.093`, exploratory `p=0.315`, `n=120`.

#### Diễn giải

MID targets khó reconstruct hơn PRE và contribution có concentration. Tuy nhiên pattern residual không chỉ ra rõ dictionary-wide capacity shortage. Cùng hiện tượng có thể đến từ objective, optimization hoặc target difficulty.

Chưa có controlled comparison `C=8192` versus `C=16384/32768`, nên không thể gán causal effect cho capacity.

#### Verdict

`INCONCLUSIVE`.

### H7 — Shared gates không tương ứng chức năng giữa các model

#### Thí nghiệm causal

- 24 components: 12 PRE và 12 MID; mỗi site có bốn low, bốn middle và bốn high-contribution components.
- Bốn natural active và bốn matched inactive canonical contexts/component.
- Cùng canonical prefix/continuation được giữ cố định qua Gemma, SmolLM2 và Granite.
- Primary signed metric: edited-minus-clean continuation log-probability trên mỗi canonical UTF-8 continuation byte.
- Mỗi model chọn learned operator có Frobenius norm lớn nhất trong site.
- Controls: ba norm-matched random rank-one seeds và một random learned-component control.
- 3,456 result rows; source-weight restoration và zero-strength invariants đều pass.

#### Kết quả

| Model pair | Spearman rho | Component-bootstrap 95% CI | Sign concordance |
|---|---:|---:|---:|
| Gemma–SmolLM2 | -0.082 | -0.439–0.367 | 41.7% |
| Gemma–Granite | 0.045 | -0.385–0.478 | 54.2% |
| SmolLM2–Granite | 0.532 | 0.090–0.823 | 70.8% |

All-three sign agreement là 33.3%, CI 16.7–54.2%, so với random-rank-one point mean 23.6%. Trên đúng cùng active canonical contexts, cả ba pairwise correlation CI đều cắt zero. Chỉ 4/24 components, tương đương 16.7%, chọn cùng strongest matrix-name class ở cả ba model.

#### Diễn giải

SmolLM2 và Granite có bằng chứng correspondence trong bounded sample. Gemma không align đáng tin cậy với hai model còn lại. Predeclared support rule cho mismatch và rule làm yếu mismatch đều không pass.

Shared feature ID không chứng minh universal mechanism; nhưng kết quả hiện tại cũng chưa đủ để kết luận các model có incompatible function.

#### Verdict

`INCONCLUSIVE`.

### H8 — Objective–interpretability mismatch và detached internal gradients

#### Audit implementation

Internal reconstruction dùng detached binary support:

\[
g_c=\mathbf{1}[g_c^s>0].
\]

Do đó internal loss không truyền gradient trực tiếp vào shared encoder. Đây là fact đã được xác minh từ implementation, không phải suy đoán.

#### Kết quả quan sát

- Activation-code energy và total internal contribution norm có `rho=0.649` trên 16,384 features.
- Firing frequency và internal contribution có `rho=0.815`.

#### Diễn giải

Kết quả làm yếu phiên bản cực đoan của giả thuyết rằng activation support hoàn toàn không liên quan tới parameter reconstruction. Nhưng contribution norm không phải signed causal specificity hay semantic coherence. Chưa có differentiable-internal-supervision ablation.

#### Verdict

`INCONCLUSIVE`.

### H9 — Interpretability plateau trước khi training kết thúc

#### Evidence còn tồn tại

| Step | Exposures | Activation loss | Internal loss |
|---|---:|---:|---:|
| 13,000 | 106,496,000 | 0.400998 | 0.419016 |
| 61,036 | 500,006,912 | 0.381607 | 0.393049 |

Training objectives tiếp tục cải thiện sau step 13k. Tuy nhiên đây là training-batch loss, không phải held-out FVU.

Checkpoint weights ở step 13k không còn. Các sample interpretability cũ và cuối chỉ trùng ba component IDs; protocol và judge cũng không matched. Vì vậy chênh lệch khoảng 37.53% tại 13k và 37.21% ở final không phải learning curve hợp lệ.

#### Verdict

`INCONCLUSIVE`. Exact checkpoint-matched H9 hiện không còn thực thi được trừ khi tìm thấy bản archive độc lập của checkpoint 13k.

### H10 — Extreme firing winners quá general

#### Kết quả

- Top 20 features chỉ chiếm 0.244% dictionary nhưng mang 12.00% tổng internal contribution ở PRE và 5.60% ở MID.
- Top 1% event share: 11.84% PRE và 9.82% MID.
- Top 1% model-specific contribution share: 17.77–23.16% PRE và 9.68–11.29% MID.
- Frequency–contribution correlation: `rho=0.815`.
- Stable multi-cluster fraction: low 65%, medium 70%, high 60%.
- Cluster count–frequency correlation: `rho=-0.093`.

#### Diễn giải

Extreme winners quan trọng một cách không cân xứng đối với reconstruction, đặc biệt ở PRE. Tuy nhiên high-frequency features không có multi-cluster rate cao hơn controls. Chưa có signed causal specificity hoặc valid coherence comparison chứng minh chúng broad hoặc undesirable hơn.

#### Verdict

`PARTIALLY_SUPPORTED`: support cho contribution concentration, chưa support cho phần “overly general/polysemantic”.

## 5. Trả lời các câu hỏi nghiên cứu chính

1. **Điểm Intruder 37% có bị distortion không?** Có. Nó không phải semantic estimate đã hiệu chuẩn. Chưa biết distortion làm score thấp đi hay cao lên bao nhiêu.
2. **Bao nhiêu feature có nghĩa dễ nhận biết?** Human-identifiable fraction vẫn chưa biết. Trong nonhuman audit, 23/59 (39.0%) coherent và 18/59 (30.5%) coherent theo non-topical pattern.
3. **Strong activations có coherent hơn không?** Có partial held-out evidence: frozen-rule presence tăng 9.00 pp từ weak sang strong events, nhưng absolute coverage vẫn thấp.
4. **Feature kém interpretable có thật sự polysemantic không?** Chưa có support. 0/160 qua multiple-rule gate; stable clustering xuất hiện gần ngang matched controls.
5. **`C=8192` có binding không?** Có thể, nhưng chưa được chứng minh.
6. **Shared gate có tìm cùng function ở ba model không?** Chưa xác định. SmolLM2–Granite align; Gemma pairs không align đáng tin cậy.
7. **Detached internal supervision có gây mismatch không?** Gradient route được xác minh; causal contribution vào interpretability chưa được test.
8. **Interpretability có plateau từ 13k đến final không?** Không thể kết luận từ artifact còn lại.
9. **Extreme winners là computation quan trọng hay concentration không mong muốn?** Chúng quan trọng cho reconstruction; chưa có bằng chứng chúng kém specific hơn.
10. **Hypothesis mạnh nhất là gì?** Measurement failure vẫn mạnh nhất. Follow-up mới cho thấy pattern chủ yếu yếu hoặc chưa nhận diện được, không support polysemanticity là lời giải thích chính. Capacity và objective mismatch vẫn chưa xác định.

## 6. Ưu tiên thí nghiệm tiếp theo

Nếu tiếp tục không dùng human rater, thứ tự có expected information gain cao nhất là:

1. Chạy contrastive dual-proposer rule induction: proposer xem discovery firings cùng matched discovery non-firings, freeze rule rồi test trên confirmatory package hiện có. Không chọn proposer dựa trên held-out score cao nhất.
2. Chạy donor factorial trên cùng target contexts để tách paper-like donor, hard negative và verified semantic outlier cho H2.
3. Chỉ cân nhắc `C=16384` hoặc objective ablation sau khi measurement validity và rule-discovery bottleneck được giải quyết. Các retraining ablation này chưa được ủy quyền bởi báo cáo hiện tại.

## 7. Provenance và artifact chính

- Báo cáo kỹ thuật chi tiết: `docs/multimodel/results/aspd_v3_1_interpretability_failure_investigation.md`.
- Machine-readable matrix: `docs/multimodel/analysis/aspd_v3_1_interpretability_failure_v1/evidence_matrix.json` và `.csv`.
- H4 nonhuman summary: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1/stage2/h4_agent_audit_git_5d6114a/summary.json`.
- H7 causal summary: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/interpretability_failure_v1/h7_functional_correspondence/analysis_git_3f028ee/functional_correspondence_summary.json`.
- H7 Slurm job: `51380041`; cả ba tasks `COMPLETED`, exit `0:0`.
- H5 held-out follow-up: `docs/multimodel/results/aspd_v3_1_feature_coherence_polysemanticity_report_vi.md`.
- H5 machine-readable analysis: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/feature_coherence_v1/rule_analysis_git_b9e1a4f`.
- Không có Qwen model nào được load hoặc train trong các experiment mới.
- Không có training mới, không sửa pretrained source weights và không sửa final ASPD checkpoint.

## 8. Kết luận cuối

Không nên phản ứng với score 37% bằng cách tăng dictionary capacity hoặc thay objective ngay lập tức. Dữ liệu hiện tại cho thấy phép đo semantic là điểm thất bại đã được xác minh rõ nhất. Đồng thời, representation không hoàn toàn vô nghĩa: có feature semantic, syntactic, formatting và token-boundary rõ; có nhiều stable context clusters; và reconstruction contribution được phân bố có cấu trúc. Vấn đề là những tín hiệu này chưa tạo thành một dictionary có semantic coherence cao và ổn định trên toàn bộ feature population.

Kết luận khoa học thận trọng nhất là: **ASPD v3.1 đã học representation hữu ích cho reconstruction, nhưng semantic/mechanistic interpretability đo được hiện bị giới hạn bởi evaluation invalidity và chủ yếu weak/unidentified linguistic patterns. Stable context clusters không đủ để kết luận polysemanticity. Chưa có bằng chứng nhân quả đủ để quy lỗi chính cho `C=8192`, detached internal supervision hoặc overtraining sau step 13k.**
