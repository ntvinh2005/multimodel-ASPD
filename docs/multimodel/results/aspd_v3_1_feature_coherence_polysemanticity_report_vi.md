# ASPD v3.1 — Kiểm tra held-out về feature coherence và polysemanticity

Trạng thái: `COMPLETE_NONHUMAN_HELDOUT_VALIDATION`

Ngày hoàn tất: 2026-11-15

Checkpoint: step `61,036`, `500,006,912` accepted aligned-anchor exposures, manifest SHA-256 `735cd57374090db7c0145315d01d9f447799a77a37941d66e4bbeafa412b4969`.

## 1. Kết luận điều hành

Kết quả phù hợp nhất với outcome **C — predominantly weak or unidentifiable patterns** trong mẫu đã test.

- Trong toàn bộ 160 components, **0/160** đạt tiêu chí đã đóng băng cho một common rule coherent, **0/160** đạt tiêu chí cho nhiều unrelated rules, **153/160** không có linguistic rule đáng tin theo phép đo này và **7/160** inconclusive.
- Trong đúng 78 components từng được H5 gọi là stable multi-cluster, **0/78** có một common rule được validate, **0/78** cần nhiều unrelated rules được validate, **76/78** không có rule đáng tin và **2/78** inconclusive.
- Mean held-out balanced accuracy của common rule là `0.5124` (component-bootstrap 95% CI `0.5067–0.5183`), gần chance. Multiple-rule OR classifier cũng gần chance: `0.5112` (`0.5052–0.5173`).
- Stable embedding clusters không đặc hiệu cho natural firings: `96/160` firing sets ổn định multi-cluster, so với `93/160`, `93/160` và `94/160` ở ba matched controls. Mọi paired 95% CI của chênh lệch đều cắt zero.
- Strong natural firings dễ khớp frozen common rule hơn weak firings: `31.10%` so với `22.18%`; paired component difference `+9.00` điểm phần trăm, 95% CI `+4.99` đến `+13.07`. Đây là evidence cho activation-strength effect, nhưng absolute rule coverage vẫn thấp.

Thí nghiệm này **không support genuine semantic polysemanticity** như lời giải thích chính cho low Intruder accuracy. Nó cũng không chứng minh components là mechanistically meaningless. Kết quả trực tiếp chỉ nói rằng các linguistic rules do discovery-only proposer tạo ra không dự đoán natural firing trên documents mới.

Vì vậy, chưa có đủ bằng chứng để đổi ASPD architecture, tăng capacity hay sửa training objective dựa trên polysemanticity. Thí nghiệm tiếp theo có information gain cao nhất là **contrastive rule discovery**: cho proposer xem discovery positives cùng matched discovery non-firings, đóng băng rule trước khi dùng lại confirmatory held-out split. Điều này kiểm tra trực tiếp liệu kết quả hiện tại phản ánh feature yếu hay proposal instrument bỏ sót predictive rule.

## 2. Câu hỏi trung tâm và decision rule

Câu hỏi không phải là “embedding có chia context thành nhiều topic hay không”, mà là:

> Một rule cụ thể, được đề xuất chỉ từ discovery data, có dự đoán được natural firing trên documents hoàn toàn mới và có generalize qua các context groups hay không?

Bốn outcome được dùng:

1. `single_coherent_rule`: common rule qua balanced-accuracy, CI, specificity, cross-group recall, coverage và complexity-advantage gates.
2. `multiple_unrelated_rules`: multiple-rule classifier tốt hơn common rule một cách có CI, ít nhất hai group rules có within-group specificity và common rule thất bại.
3. `no_reliable_linguistic_rule`: common và multiple rules đều có balanced accuracy không quá `0.60`, upper 95% CI dưới `0.70`.
4. `inconclusive`: mọi trường hợp còn lại, gồm thiếu group coverage hoặc gần threshold.

`UNCERTAIN` và invalid output được tính bảo thủ là “rule absent”. Đơn vị kết luận chính là component; matched pair và document grouping được giữ trong bootstrap.

## 3. Dữ liệu và provenance

### 3.1 Components và splits

- 120 H5 components cũ: 60 PRE + 60 MID.
- 40 components random mới: 20 PRE + 20 MID, lấy từ eligible population sau khi loại sample cũ.
- Tổng: 160 components, 80 PRE + 80 MID.
- Discovery dùng `test`; confirmatory dùng `validation` và loại toàn bộ 313 discovery document IDs.
- Discovery package sau dedup có 13,662 natural firing contexts.
- Confirmatory package có 7,627 natural positives, 7,627 ordinary matched non-firings và 7,627 hard non-firings.
- 138/160 components có ít nhất tám positive examples trong mọi discovered group; 139/160 đạt tối thiểu 80% group coverage.

Mọi firing đều là natural selected BatchTopK events của encoder thật. Không dùng forced support, positive pre-TopK candidates hoặc Qwen.

### 3.2 Candidate rules và independent validation

- Gemma-3-4B-IT tạo 625 frozen rules: 160 common rules và 465 group rules.
- Proposer không được xem confirmatory contexts; manifest ghi `heldout_contexts_exposed_to_proposer = 0`.
- Phi-4-mini-instruct đánh giá rule presence độc lập trên 67,225 held-out tasks.
- Validation có 67,223/67,225 valid outputs, 8 abstentions tính cả `UNCERTAIN`, và 2 invalid outputs.
- Không có human labels. Tất cả kết quả phải được gọi là **nonhuman independent validation**.

Calibration của Phi đạt đúng gate đã đóng băng: balanced accuracy `0.850`, sensitivity `1.000`, specificity `0.700`, 0 invalid và 0 abstention trên 80 synthetic controls. Specificity chỉ `0.70` cho thấy validator có xu hướng trả lời YES; vì vậy nó không được xem là semantic ground truth tuyệt đối.

### 3.3 Jobs đã xác minh

| Stage | Job / artifact | Trạng thái |
|---|---|---|
| Natural context extraction | Slurm `51402734` | `COMPLETED`, 3×B200, source weights unchanged |
| Confirmatory embeddings | Slurm `51418336` | `COMPLETED`, 1×B200, SHA `b1d582…e80a` |
| Rule proposer | Slurm `51482659` + bounded repair `51495469` | 625/625 frozen rules |
| Validator calibration | Slurm `51514807` | `COMPLETED`, reliability gate passed |
| Full held-out validation | Slurm `51515750` | `COMPLETED`, 67,225 tasks, source weights unchanged |
| Statistical analysis | `rule_analysis_git_b9e1a4f` | `ASPD_V3_1_FEATURE_COHERENCE_RULE_ANALYSIS_COMPLETE` |

Failed/superseded jobs và namespaces được giữ lại; không có artifact khoa học nào bị overwrite.

## 4. Kết quả chính

### 4.1 Required summary table

| Metric | PRE | MID | Overall |
|---|---:|---:|---:|
| Components evaluated | 80 | 80 | 160 |
| Components with one validated common rule | 0/80 | 0/80 | 0/160 |
| Components with validated multiple unrelated rules | 0/80 | 0/80 | 0/160 |
| Components without a reliable linguistic rule | 76/80 (95.0%) | 77/80 (96.25%) | 153/160 (95.63%, Wilson 95% CI 91.25–97.86%) |
| Inconclusive components | 4/80 | 3/80 | 7/160 (4.38%, 95% CI 2.14–8.75%) |
| Mean common-rule held-out balanced accuracy | 0.5187 (0.5100–0.5279) | 0.5060 (0.4992–0.5129) | 0.5124 (0.5067–0.5183) |
| Strong-firing common-rule presence | 0.3342 (0.2657–0.4020), n=1,483 | 0.2879 (0.2316–0.3449), n=1,356 | 0.3110 (0.2678–0.3559), n=2,839 |
| Natural-firing stable multi-cluster | 45/80 | 51/80 | 96/160 (60.0%, 52.26–67.27%) |
| Random source/length-matched control | 46/80 | 47/80 | 93/160 (58.13%, 50.38–65.49%) |

Các CI của balanced accuracy và activation-strength rate là component bootstrap; các CI của proportions là Wilson 95% CI.

### 4.2 Đúng 78 multi-cluster components cũ

| Outcome | Count | Fraction / 95% CI |
|---|---:|---:|
| One validated common rule | 0/78 | 0% |
| Validated multiple unrelated rules | 0/78 | 0% |
| No reliable linguistic rule | 76/78 | 97.44% / 91.12–99.29% |
| Inconclusive | 2/78 | 2.56% / 0.71–8.88% |

Mean common-rule balanced accuracy là `0.5181` (`0.5093–0.5272`); multiple-rule balanced accuracy là `0.5062` (`0.4988–0.5135`). Do đó H5 clustering result không chuyển thành held-out evidence cho multiple unrelated meanings.

### 4.3 Clustering controls

| Context set | Stable multi-cluster | Fraction | Mean silhouette |
|---|---:|---:|---:|
| Natural firing | 96/160 | 60.00% | 0.2842 |
| Random source/length-matched non-firing | 93/160 | 58.13% | 0.2784 |
| Topic-matched non-firing | 93/160 | 58.13% | 0.2937 |
| Frequency-matched other component | 94/160 | 58.75% | 0.2825 |

Paired natural-minus-control differences lần lượt là `+1.875` pp (95% CI `-8.125` đến `+11.875`), `+1.875` pp (`-7.50` đến `+10.625`) và `+1.25` pp (`-8.125` đến `+10.625`). Clustering pipeline tạo nhiều stable groups gần như ngang nhau trên controls. Vì vậy stable embedding clusters đơn lẻ là evidence rất yếu cho polysemanticity.

### 4.4 Activation strength

| Natural-firing bin | Components có mẫu | Positive contexts | Mean frozen-rule presence / 95% CI |
|---|---:|---:|---:|
| Weak | 131 | 2,255 | 0.2218 / 0.1822–0.2643 |
| Medium | 150 | 2,533 | 0.2395 / 0.2024–0.2782 |
| Strong | 160 | 2,839 | 0.3110 / 0.2678–0.3559 |

Trên 131 components có cả weak và strong examples, paired strong-minus-weak difference là `+0.0900` (95% CI `+0.0499` đến `+0.1307`). PRE có `+0.1128` (`+0.0541–+0.1743`); MID có `+0.0689` (`+0.0131–+0.1258`).

Điều này support phiên bản hẹp của giả thuyết weak firings: strong events thường chứa proposed pattern rõ hơn. Tuy nhiên ngay cả strong-bin presence cũng chỉ khoảng 31%, nên filtering strength không giải quyết phần lớn coherence failure.

### 4.5 PRE so với MID

PRE common-rule balanced accuracy cao hơn MID khoảng `1.27` điểm phần trăm. Sau khi điều chỉnh tuyến tính cho log discovery firing frequency và original/new cohort, estimate vẫn là `+1.28` pp, bootstrap 95% CI `+0.16` đến `+2.42` pp.

Đây là khác biệt nhỏ và exploratory. Cả PRE lẫn MID đều gần chance và không có component nào qua coherent/polysemantic gate, nên không nên diễn giải PRE là “interpretable” còn MID là “không interpretable”.

### 4.6 Liên hệ với Intruder cũ

Trên 120 H5 components, Spearman correlation giữa failed-judge Intruder accuracy và held-out common-rule balanced accuracy là `rho=-0.130`, exploratory `p=0.158`. Không có evidence rằng feature có Intruder score cao hơn cũng có rule predictive tốt hơn.

Intruder values này đến từ judge đã fail position-bias reliability checks; association chỉ là diagnostic, không phải validation của Intruder.

## 5. Mười component case studies

Các đoạn dưới đây là excerpt thật từ held-out natural firings; dấu `⟪ANCHOR⟫…⟪/ANCHOR⟫` là anchor thật. Không có ví dụ được sáng tác. Full contexts và mọi group rule nằm trong `case_studies.jsonl`.

| Component | Vai trò trong gallery | Frozen common rule | Common / multi BA | Held-out excerpt và diễn giải |
|---|---|---|---:|---|
| `pre/7429` | Best single-rule candidate; inconclusive | “`was`/`were` followed by a clause…” | 0.708 / 0.500 | iOS code context quanh `⟪ANCHOR⟫as⟪/ANCHOR⟫ possible`; rule không hiện diện ở group này. Feature gần pass overall BA nhưng specificity `0.625` và minimum group recall `0.438` fail gates. |
| `pre/3257` | Best candidate; inconclusive | Numerical data and patient counts | 0.635 / 0.521 | “Cash `⟪ANCHOR⟫$1,299.00⟪/ANCHOR⟫` … Rental-Purchase…” phù hợp numerical rule ở một group, nhưng minimum cross-group recall chỉ `0.25`. |
| `pre/3205` | Best candidate; inconclusive | Lists, software names or technical specifications | 0.604 / 0.521 | Eye-exam prose quanh `peripheral vision⟪ANCHOR⟫,⟪/ANCHOR⟫` và bibliography/code-like groups không được một rule đủ specific bao phủ. |
| `mid/2389` | Best remaining single candidate; no-rule | Conditional logic or comparisons | 0.594 / 0.490 | Java code có `if (!file.exists())`, và fitness prose có `⟪ANCHOR⟫if⟪/ANCHOR⟫`; pattern xuất hiện ở positives nhưng không phân biệt matched non-firings. |
| `mid/3778` | Largest multi-rule advantage; inconclusive | Anchor tags and legal disclaimers | 0.479 / 0.656 | Held-out groups gồm author page, PubMed citation và Scala stack trace. Multi-rule gain `+0.177` có CI dương, nhưng coverage `0.75` và chỉ một group rule đạt distinct-rule gate. |
| `pre/2759` | Second multi-rule candidate; inconclusive | “appears with anchor markers” | 0.521 / 0.688 | Biomedical title quanh `⟪ANCHOR⟫of⟪/ANCHOR⟫`, iOS prose quanh `off`, và public-health prose quanh `for`. Common rule vô nghĩa vì mọi task đều có display markers; specificity chỉ `0.104`; không group rule nào đủ distinct. |
| `mid/6466` | Multi-rule candidate; no-rule | Code identifiers such as `public` and class names | 0.448 / 0.573 | Legal opinion quanh `⟪ANCHOR⟫Submitted⟪/ANCHOR⟫`, biomedical abstract quanh `laser`, và iOS prose quanh `/`; frozen code rule không generalize. |
| `mid/7252` | Multi-rule boundary; inconclusive | Actions or processes | 0.500 / 0.615 | Patent prose quanh `greater than`, political prose quanh `as soon as`, và iOS layout prose quanh `as much as`; rule quá rộng và chỉ một group rule gần distinct. |
| `mid/6704` | Weakest common rule; no-rule | Anchor tags and lists | 0.417 / 0.479 | Personnel-board prose quanh `all of`, time-value article quanh `enough`, và patent prose quanh `sufficiently`; không có common list/tag pattern. |
| `pre/3298` | Weak common rule; no-rule | Anchor markers imply JSON/XML | 0.427 / 0.510 | Housing-policy prose quanh `can` và HTML question quanh `contents`; rule nhầm evaluation markup với source structure và không dự đoán firing. |

Không có “clearly coherent” hoặc “clearly polysemantic” case nào qua predeclared gates, nên gallery không gắn nhãn như vậy chỉ vì một excerpt trông hợp lý. Các best candidates được giữ là boundary/failure cases.

## 6. Trả lời tám câu hỏi bắt buộc

1. **Trong 78 multi-cluster components, bao nhiêu có một common rule?** 0/78 theo frozen held-out gates.
2. **Bao nhiêu thật sự cần multiple unrelated explanations?** 0/78; trong toàn sample cũng 0/160.
3. **Bao nhiêu uninterpretable hoặc inconclusive?** 76/78 no-rule và 2/78 inconclusive; overall 153/160 và 7/160.
4. **Context groups có meaningful hơn random controls không?** Không có evidence. Stable-cluster rates và silhouette tương tự controls; paired CIs đều cắt zero.
5. **Strong activation có tăng coherence không?** Có cho frozen-rule presence: paired `+9.00` pp. Absolute presence vẫn thấp, nên effect chỉ partial.
6. **PRE và MID có khác nhau không?** PRE cao hơn khoảng `1.28` pp sau adjustment, nhưng cả hai gần chance và khác biệt nhỏ.
7. **High Intruder có independently validated coherence cao hơn không?** Không thấy association: `rho=-0.130`, `p=0.158`; Intruder judge cũ cũng invalid.
8. **Polysemanticity có giải thích low Intruder không?** Không được support trong sample này. Clustering association không chuyển thành predictive multi-rule behavior; causation càng chưa được chứng minh.

## 7. Scientific decision

**Decision: C — Predominantly weak or unidentifiable patterns**, áp dụng cho 160-component tested sample, không tự động extrapolate ra toàn dictionary.

Không chọn B vì không có component nào thỏa validated multiple-rule criteria. Không chọn A vì không có common rule nào qua toàn bộ gates. Không chọn D ở cấp sample vì 153/160 có đủ data để bác bỏ chính các candidate rules đã freeze; tuy nhiên nguồn gốc của failure vẫn có uncertainty.

### Evidence không cho phép kết luận gì

- “No reliable linguistic rule” không đồng nghĩa với “không có computation có nghĩa”. Feature có thể encode một signal khó diễn đạt bằng text rule hoặc phụ thuộc joint representations của ba model.
- Một proposer có thể bỏ sót rule thật. Đặc biệt, proposer hiện nhìn positive discovery groups nhưng không được dùng matched negative contrasts để loại các topic-level descriptions.
- Phi calibration pass đúng ngưỡng tối thiểu, không vượt xa ngưỡng; specificity `0.70` là limitation quan trọng.
- Không có human evidence hoặc inter-rater agreement.
- Kết quả là association/prediction trên contexts, không phải causal intervention lên model behavior.

## 8. Có nên đổi architecture, capacity hoặc objective không?

**Chưa.** Genuine polysemanticity chưa được support đủ để biện minh retraining hoặc thay `C`, `K`, gate semantics hay gradient routing.

Thay đổi architecture lúc này sẽ trộn ít nhất hai lời giải thích:

1. representation thật sự yếu hoặc không có linguistic regularity;
2. rule-discovery instrument không contrast positives với non-firings nên đề xuất rule không predictive.

### Single most informative next experiment

Giữ nguyên checkpoint, components, document splits và Phi validator. Chỉ thay discovery stage bằng **contrastive dual-proposer rule induction**:

- mỗi proposer nhận natural discovery firings cùng source/topic/length-matched discovery non-firings;
- yêu cầu rule dự đoán khác biệt quanh anchor, không mô tả topic chung;
- freeze rules trước khi mở confirmatory data;
- validate từng proposer riêng và rule-union đã preregister trên đúng held-out package;
- không chọn proposer dựa trên score tốt nhất.

Nếu contrastive rules vẫn gần chance, evidence cho weak/unidentifiable linguistic patterns sẽ mạnh hơn đáng kể. Nếu balanced accuracy tăng rõ, bottleneck hiện tại chủ yếu nằm ở interpretation proposal, không phải ASPD representation.

## 9. Artifacts

- Protocol: `configs/multimodel/aspd_v3/main500m_feature_coherence_v1.yaml`
- Analysis contract: `configs/multimodel/aspd_v3/main500m_feature_coherence_analysis_v1.yaml`
- Frozen rules: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/feature_coherence_v1/candidate_rules_contract_09820f66658d`
- Frozen held-out tasks: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/feature_coherence_v1/rule_validation_contract_09820f66658d`
- Calibration: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/feature_coherence_v1/rule_validation/calibration_job_51514807`
- Validation: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/feature_coherence_v1/rule_validation/validation_job_51515750`
- Final machine-readable analysis: `/blue/thai/vinhnguyen1/aspd_v3/evaluation/feature_coherence_v1/rule_analysis_git_b9e1a4f`

Final analysis directory chứa `summary.json`, `per_component_results.jsonl`, `per_component_results.csv`, `activation_strength_results.jsonl`, `cluster_control_results.jsonl`, `case_studies.jsonl`, `experiment_manifest.json` và `COMPLETE`.

