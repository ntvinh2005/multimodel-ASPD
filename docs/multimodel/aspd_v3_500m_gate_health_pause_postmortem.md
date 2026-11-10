# ASPD v3.1 main gate-health pause postmortem

Status: `CONTROLLED_PAUSE_DIAGNOSED — MONITORING_FIX_IMPLEMENTED`

Date: 2026-11-10

## Outcome

Main job `50640347` made a complete checkpoint and paused at optimizer step 321
(`2,629,632` accepted aligned-anchor exposures). Slurm displayed `FAILED` because
`torchrun` wrapped the workers' controlled-pause exit, but the scientific trigger was
`SUSTAINED_GATE_HEALTH_FAILURE`; it was not an OOM, nonfinite event, duplicated exposure,
source-model mutation, or checkpoint failure.

The pause does not establish recurrence of the historical gate collapse. It exposes a
monitoring-policy calibration defect: three 64-step observations sampled a transient
effective-size valley that was already present in the successful matched z-score pilot.

## Metric and gate notation

For site \(t\in\{\mathrm{PRE},\mathrm{MID}\}\), the implemented gate remains

\[
\mu_{a,t}=\frac1C\sum_c z_{a,t,c},\qquad
\sigma_{a,t}^2=\frac1C\sum_c(z_{a,t,c}-\mu_{a,t})^2,
\]

\[
\tilde z_{a,t,c}=\frac{z_{a,t,c}-\mu_{a,t}}
{\sqrt{\sigma_{a,t}^2+10^{-5}}},\qquad
s_{a,t,c}=\operatorname{ReLU}(\tilde z_{a,t,c}).
\]

Pool-wide BatchTopK with \(C=8192\) and \(K=32\) produces the continuous sparse code
\(g^s\). Internal reconstruction continues to use detached support
\(g_{a,t,c}=\mathbf 1[g^s_{a,t,c}>0]\).

For one diagnostic pool, let \(n_{t,c}\) be the number of naturally selected firing
events for feature \(c\). The monitored entropy-effective dictionary size is

\[
p_{t,c}=\frac{n_{t,c}}{\sum_d n_{t,d}},\qquad
C_{\mathrm{eff},t}=\exp\!\left(-\sum_{c:p_{t,c}>0}p_{t,c}\log p_{t,c}\right).
\]

This is a concentration statistic. It is not the cumulative count of feature identities
that fire at least once over a large held-out corpus.

## Evidence

| Parameter step | Run | PRE \(C_{\mathrm{eff}}\) | MID \(C_{\mathrm{eff}}\) |
|---:|---|---:|---:|
| 64 | three-model main | 2478 | 2462 |
| 128 | three-model main | 1955 | 1913 |
| 192 | three-model main | 775 | 880 |
| 256 | three-model main | 808 | 808 |
| 320 | three-model main | 856 | 836 |
| 256 | matched two-model z-score pilot | 996 | 765 |
| 512 | matched two-model z-score pilot | 1432 | 1250 |
| 1220 | matched two-model z-score pilot | 2130 | 1808 |

At the main pause, PRE/MID positive candidates were approximately 6003/6015 per anchor,
the top-1% firing shares were 41.82%/42.47%, and 4241/4367 feature identities were active
in the diagnostic pool. Historical full collapse instead ended near 37--41 candidates,
effective sizes 46--52, and top-1% shares above 99%. The current state is a real
concentration valley, but it is not the historical absorbing collapse.

The reference pilot's diagnostic schedule sampled parameter steps
`0,64,128,256,512,768,1024,1220`. The main policy samples every 64 steps. Its three-strike
rule therefore observed steps 192, 256, and 320 below the provisional threshold 1024 and
paused before the first reference recovery observation at step 512.

## Corrective policy

The emitted `DICTIONARY_CONCENTRATION` alert is preserved at every diagnostic. Before
`4,202,496` accepted exposures (the row measuring parameter step 512), this alert alone is
not pause-eligible. Candidate starvation, top-1%/top-10% concentration, excessive recent
feature death, missing diagnostics, nonfinite values, repeated exposures, storage breach,
and checkpoint failure retain their existing escalation behavior.

At and after `4,202,496` exposures, dictionary concentration again contributes to the
three-consecutive-diagnostic pause rule. The implementation revision is
`v1.1_zscore_warmup_calibrated`. Legacy schema-v1 monitor state from the immutable
step-321 checkpoint is accepted; the first warm-up diagnostic resets the inherited
dictionary-only streak if no other gate-health condition fails.

No architecture or optimization semantics changed. In particular, this fix does not
alter the Transformer encoder, z-score, ReLU, BatchTopK, AuxK, loss weights, optimizer,
continuous activation code, detached binary support, rank-one operators, exposure ledger,
or source models.

## Resume boundary

The immutable step-321 checkpoint must be resumed into a new output namespace through the
existing external exact-resume path and a checkpoint-pinned continuation authorization.
The original run directory and checkpoint remain unchanged. The first required scientific
reviews are the diagnostics measuring parameter steps 512 and 768. Recovery is not assumed:
if the three-model trajectory stays below threshold after the calibrated boundary, the
sustained policy pauses it again.
