# ASPD v3.1 500M storage reclamation proposal

Status: **DRY RUN ONLY — NO DELETION AUTHORIZED OR PERFORMED**

Audit timestamp: 2026-11-09T18:32:46-05:00

Machine-readable inventory:
`docs/multimodel/analysis/aspd_v3_500m_blocker_resolution/storage_inventory.json`.

## Live capacity and constraint

| Filesystem | Capacity | Used | Available | Inodes free | User byte limit | Group byte limit |
|---|---:|---:|---:|---:|---:|---:|
| `/orange/thai` | 2,199,023,255,552 B | 1,985,201,668,096 B | 213,821,587,456 B (200 GiB) | 13,422,163,115 | none reported | none reported |
| `/blue/thai` | 5,497,558,138,880 B | 5,116,136,005,632 B | 381,422,133,248 B (356 GiB) | 4,107,495,892 | none reported | none reported |

The effective allocation group is numeric GID 3059 (`thai`). The Lustre group query reports
1,938,677,900 KiB used on `/orange/thai` and 5,196,337,068 KiB used on `/blue/thai`, with no
explicit soft or hard group limits. Filesystem-wide free space, not a quota limit, is the current
constraint. The sandbox reports these external paths as non-writable, while their real ACL owner
entry is `rwx`; a bounded batch-environment write probe is still required before choosing the
production root.

The existing upper-bound budget is unchanged because model 3 is not selected:

| Item | Budget |
|---|---:|
| Two retained complete checkpoint bundles | 360–394 GB |
| One atomic in-progress checkpoint bundle | 180–197 GB |
| Live ledger and stream databases | 56 GB |
| Logs, manifests, profiler and staged evaluation | 25 GB |
| Recovery margin | 200 GB |
| **Required** | **821–872 GB; reserve at least 1 TiB free** |

Neither filesystem currently meets the reservation. `STORAGE_BLOCKED` remains correct.

## Candidate classification

Sizes are allocated bytes from `du -x -B1 -s`. A `REGENERABLE` label means the large tensor
payload can be regenerated from the preserved preflight report/config/source revision; it is not
authorization to rerun the experiment. A `REDUNDANT_VERIFIED` intermediate checkpoint has a later
complete checkpoint and a final report that names the later checkpoint. All deletions still need
approval of the exact paths.

| Path | Experiment | Size | Classification | Reason | Reproducible? | Estimated reclaimed space | Recommendation |
|---|---|---:|---|---|---|---:|---|
| `/blue/thai/vinhnguyen1/multimodel-ASPD/work/out/multimodel/aspd_v3/granite_normalized_target_v1/preflight/baseline/checkpoints` | Granite normalized-target baseline preflight | 127,280,418,816 B | `REGENERABLE` | Three-step preflight report, run manifest and embedded COMPLETE checkpoint manifest remain outside this directory. | Yes, conditional on a separately authorized bounded rerun. | 127,280,418,816 B | Tier A candidate; preserve report SHA `a28b22b8...5542ba3e`. |
| `/blue/thai/vinhnguyen1/multimodel-ASPD/work/out/multimodel/aspd_v3/granite_normalized_target_v1/preflight/normalized_target/checkpoints` | Granite normalized-target treatment preflight | 127,280,365,568 B | `REGENERABLE` | Same bounded preflight-only tensor payload; report and provenance remain. | Yes, conditional on authorization. | 127,280,365,568 B | Tier A candidate; preserve report SHA `00500532...de454`. |
| `/blue/thai/vinhnguyen1/multimodel-ASPD/work/out/multimodel/aspd_v3/two_model_scale_v1/preflight/baseline/checkpoints` | Gemma/SmolLM2 scale baseline preflight | 87,403,884,544 B | `REGENERABLE` | Two-step restore-probe checkpoint; preflight report and final 10M pilot remain. | Yes, conditional on authorization. | 87,403,884,544 B | Tier A candidate; preserve report SHA `d3d5c0d8...d472a`. |
| `/blue/thai/vinhnguyen1/multimodel-ASPD/work/out/multimodel/aspd_v3/two_model_scale_v1/preflight/treatment/checkpoints` | Gemma/SmolLM2 scale treatment preflight | 87,403,913,216 B | `REGENERABLE` | Two-step restore-probe checkpoint; preflight report and final 10M pilot remain. | Yes, conditional on authorization. | 87,403,913,216 B | Tier A candidate; preserve report SHA `96e3fb89...dd05`. |
| `/blue/thai/vinhnguyen1/multimodel-ASPD/work/out/multimodel/aspd_v3/granite_normalized_target_unconstrained_decoder_v1/preflight/normalized_target/checkpoints` | Unconstrained-decoder preflight | 127,280,353,280 B | `REGENERABLE` | The successful bounded preflight report and later reliability checkpoint remain. | Yes, conditional on authorization. | 127,280,353,280 B | Tier A candidate; preserve report SHA `8d3e58ef...97c8b`. |
| `/blue/thai/vinhnguyen1/multimodel-ASPD/work/out/multimodel/aspd_v3/two_model_scale_v1/baseline/job_49977094/checkpoints/step-000001000` | Scale baseline 10M pilot | 87,403,720,704 B | `REDUNDANT_VERIFIED` | Final COMPLETE step 1,221 exists and the PASS report names step 1,221. | Final state remains; exact step-1,000 recovery point would be lost. | 87,403,720,704 B | Tier A candidate after exact approval. |
| `/blue/thai/vinhnguyen1/multimodel-ASPD/work/out/multimodel/aspd_v3/two_model_scale_v1/treatment/job_49977095/checkpoints/step-000001000` | Scale treatment 10M pilot | 87,403,749,376 B | `REDUNDANT_VERIFIED` | Final COMPLETE step 1,221 exists and the PASS report names step 1,221. | Final state remains; exact step-1,000 recovery point would be lost. | 87,403,749,376 B | Tier A candidate after exact approval. |
| `/orange/thai/vinhnguyen1/aspd_v3/gemma_smol_matryoshka_ablation_v1/full_only/pilot/runs/job_50231651_1/checkpoints/step-000001000` | Full-only 10M pilot | 87,401,779,200 B | `REDUNDANT_VERIFIED` | Final COMPLETE step 1,221 exists and the PASS report names it. | Final state remains. | 87,401,779,200 B | Tier B; useful only if reclaiming `/orange`. |
| `/orange/thai/vinhnguyen1/aspd_v3/gemma_smol_matryoshka_ablation_v1/matryoshka/pilot/runs/job_50231651_0/checkpoints/step-000001000` | Matryoshka 10M pilot | 87,401,996,288 B | `REDUNDANT_VERIFIED` | Final COMPLETE step 1,221 exists and the PASS report names it. | Final state remains. | 87,401,996,288 B | Tier B; useful only if reclaiming `/orange`. |
| `/orange/thai/vinhnguyen1/aspd_v3/gemma_smol_gate_collapse_recovery_v1/per_anchor_zscore/preflight_r1/checkpoints` | Successful z-score arm preflight | 87,401,902,080 B | `REGENERABLE` | Report, run manifest, final 10M pilot and evaluation remain. | Yes, conditional on authorization. | 87,401,902,080 B | Tier B; do not remove pilot checkpoint. |
| `/orange/thai/vinhnguyen1/aspd_v3/gemma_smol_gate_collapse_recovery_v1/full_only_control/preflight_r1/checkpoints` | Matched control preflight | 87,401,979,904 B | `REGENERABLE` | Report, run manifest, final 10M pilot and evaluation remain. | Yes, conditional on authorization. | 87,401,979,904 B | Tier B; do not remove pilot checkpoint. |
| `/orange/thai/vinhnguyen1/aspd_v3/zscore_intruder_final_validation_v1/failures/job_50435027/core_dumps/core.pt_nccl_watchdg-6-1794242312-5268-3059-c0904a-s15.ufhpc.2917985` | Failed Intruder job core dump | 2,382,708,736 B | `INCOMPLETE_OR_TEMPORARY` | NCCL watchdog core, not a scientific result. | Not applicable; failure logs/reports are separate. | 2,382,708,736 B | Highest-safety exact-file deletion candidate. |
| `/orange/thai/vinhnguyen1/aspd_v3/zscore_intruder_final_validation_v1/failures/job_50435027/core_dumps/core.pt_nccl_watchdg-6-1794242108-5268-3059-c1003a-s5.ufhpc.1298597` | Failed Intruder job core dump | 2,352,959,488 B | `INCOMPLETE_OR_TEMPORARY` | NCCL watchdog core, not a scientific result. | Not applicable. | 2,352,959,488 B | Highest-safety exact-file deletion candidate. |
| `/blue/thai/vinhnguyen1/multimodel-ASPD/cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct` | Historical local semantic judge cache | 141,127,479,296 B | `RESEARCHER_REVIEW_REQUIRED` | Outputs are preserved, but this gated and expensive cache may still be useful to human-evaluation work. | Conditional on future access and a large redownload. | 141,127,479,296 B | Optional safety margin only after the human-evaluation owner confirms no dependency. |
| `/blue/thai/vinhnguyen1/multimodel-ASPD/work/out/multimodel/aspd_v3/main_v1/preflight/aspd_v3_qwen_gemma_phi_168targets_c8192_k32_preflight_v12_3gpu_b200_tensor_dcp/checkpoints` | Historical failed three-model main preflight | 150,938,775,552 B | `KEEP_UNTIL_REPRODUCED` | Qwen is forbidden in new experiments, so this exact historical checkpoint cannot currently be regenerated. | No under the current no-Qwen rule. | 0 B proposed | Keep; report SHA `5393cdfc...a4c7` is not a substitute for weights. |

## Protected inventory

The following are not deletion candidates:

| Path/category | Classification | Reason |
|---|---|---|
| Complete final pilot/reliability checkpoints under `/orange/thai/vinhnguyen1/aspd_v3` | `MUST_KEEP` | Primary scientific evidence for the z-score, decoder, factorial and Matryoshka comparisons. |
| `/orange/thai/vinhnguyen1/aspd_v3/human_interpretability_v1` | `MUST_KEEP` | Owned by the human-interpretability workflow and explicitly outside this workstream. |
| Gemma, SmolLM2 and Granite model-cache entries | `MUST_KEEP` | Retained sources and current model-3 review candidate; no cache deletion is proposed. |
| `/blue/thai/vinhnguyen1/multimodel-ASPD/work/out/multimodel/cache` | `KEEP_UNTIL_REPRODUCED` | Mostly historical Qwen activation caches; current policy forbids a new Qwen run to regenerate them. |
| Code repositories and Git worktrees | `MUST_KEEP` | Active/shared source history; never part of storage cleanup. |

## Quantified reclamation options

Tier A contains only the seven `/blue` preflight/intermediate checkpoint paths listed above:

- reclaimable: 731,456,405,504 B (681.2 GiB);
- projected `/blue` free space from the audit snapshot: 1,112,878,538,752 B (1.012 TiB);
- margin above 1 TiB: only 13,366,910,976 B (12.4 GiB).

Tier A technically crosses 1 TiB at the audit instant, but the margin is too small for concurrent
filesystem consumption. The recommended robust choices are either a dedicated allocation/storage
increase or Tier A plus a separately approved cache candidate. Adding the Llama-3.3-70B cache would
project 1,254,006,018,048 B (1.140 TiB) free, but it must not be removed until human-evaluation
dependencies are explicitly cleared.

The optional `/orange` candidates total 354,343,325,696 B and would project only
568,164,913,152 B free. `/orange` would remain unsuitable for the main run.

Therefore the preferred storage plan is:

1. obtain exact researcher approval for Tier A paths;
2. independently confirm the human workflow does not use the Llama cache, or request additional
   project capacity instead;
3. remeasure immediately after any approved cleanup;
4. only if one filesystem has at least 1 TiB free with a credible concurrent-use margin, update
   all versioned runtime/data/checkpoint paths and regenerate fingerprints;
5. keep the checked-in main configuration `STORAGE_BLOCKED` until then.

## Dry-run and approval protocol

No deletion command has been run. After the researcher approves exact targets, the operator must:

1. repeat `squeue`, `df -B1`, `df -i`, user/group `lfs quota`, and `du -x -B1 -s`;
2. fail closed if an approved target size or path changed, an active job references it, or a new
   worktree/output dependency appears;
3. verify every COMPLETE marker and compare the small manifest/report SHA-256 values with the
   machine-readable inventory;
4. preserve copies of the small checkpoint manifest, run manifest, preflight/final report,
   configuration SHA, dataset manifest SHA and source commit before removing any complete tensor
   payload;
5. print the exact target list and total bytes as a dry run;
6. request approval again if the resolved target list differs by even one path;
7. remove only the approved exact paths, never a glob or parent directory;
8. re-run `df`, `du`, and protected-artifact checks and append the actual reclaimed-byte result to
   this document.

Until those steps and the explicit target approval occur, the decision remains `STORAGE_BLOCKED`.
