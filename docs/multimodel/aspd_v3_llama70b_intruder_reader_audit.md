# ASPD v3.1 Llama-70B Intruder Reader Audit

## Result

The raw Llama-70B Intruder run is genuinely position-biased. Correct positions were exactly
balanced at 40 per position, but the 194 valid outputs selected positions 1--5 with counts
8, 113, 32, 23, and 18. A chi-square test against a uniform response distribution gives
`chi2(4)=185.12`, `p=5.92e-39`. Position 2 was correct only 24 of the 113 times it was
selected (21.2%, approximately chance), so the preference cannot be explained by position 2
usually containing a uniquely clearer intruder.

The bias is not the only problem. A blinded reader audit froze decisions on Judge Sets 001--040
before loading the protected key. Only 9/40 items supported a defensible forced choice from the
compact contexts; 31/40 were marked indeterminate. Four of the nine forced choices matched the
constructed donor answer. This is an exploratory audit, not a substitute for the researcher's
human evaluation.

## Task-validity observations

The protected answer identifies which context was drawn from another feature. It does not ensure
that this context is the unique semantic outlier when the four target-feature contexts are
themselves incoherent. This produced three recognizable cases:

- Judge Set 009 contained four license/copyright passages and one generated resource-class passage.
  The reader selected the constructed intruder (5); Llama selected position 2 based on formatting.
- Judge Set 033 had four scientific citation-marker contexts and one ordinary prose anchor. The
  reader selected the constructed intruder (3); Llama selected position 2 based on scientific topic.
- Judge Set 039 visually grouped four scientific passages against a Naruto plot passage at position
  2. Both the reader and Llama selected position 2, but the constructed donor answer was position 4.
  This is evidence of target-feature incoherence, not a defensible semantic error by the judge.

Thus raw donor-identification accuracy mixes at least three effects: judge position preference,
judge topic-versus-structure preferences, and actual feature incoherence/non-unique intruders.

## Arm-level exploratory observation

The fixed 40-item slice contained ten items from each arm/site. The reader made 3 forced choices
and obtained 0 constructed answers in Control, versus 6 forced choices and 4 constructed answers
in Z-score (PRE 1/2; MID 3/4). These counts are too small and the fixed first-40 slice is not an
independent confirmatory sample. They are weak evidence that the Llama macro result may be hiding
more identifiable Z-score examples, especially structural features, and justify completing the
blinded human form.

## Scientific interpretation

The Llama raw score cannot support a confirmatory Control-versus-Z-score semantic claim. The audit
also does not support dismissing the low result as judge bias alone: most inspected feature sets
were genuinely difficult to interpret. Current status remains exploratory pending the frozen human
evaluation, with no evidence sufficient to claim shared causal mechanisms.
