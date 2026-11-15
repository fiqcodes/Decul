# Decul V6 — offline sequence-context research

## Completed experiment

V6 trains new classifiers and temporal models from existing IDSSE data. It does
not inherit V1's selected 60 features, XGBoost hyperparameters or HMM parameters.
All 84 common tracking measurements are available. Offline before/after context
is optional; another ablation adds training-fold-only unsupervised PCA/regimes.
No new external data was downloaded. Smart Tagging and BAR–ATH labels are unused.

## Nested match-group development results

| model | clean_spellF1 | dropout_spellF1 | objective | macroF1 | short_recall |
| --- | --- | --- | --- | --- | --- |
| causal_reference | 0.5961 | 0.4260 | 0.5110 | 0.8524 | 0.0962 |
| nested_v6 | 0.6301 | 0.4703 | 0.5502 | 0.8708 | 0.1538 |

Each outer match appears exactly once in evaluation. Its labels cannot fit that
outer model, feature normalizer, regime encoder, HMM or inner choice. The causal
reference is independently retrained on the same outer training matches. It is
not the frozen V1 model. Configurations and decoder weights are selected only from
inner match-group validation, using equal clean/dropout and match weights.
All seven fixtures were explored in earlier project versions, so these remain
development results, not a pristine external test or a guarantee of BAR–ATH accuracy.

## Per-match results

| match | stage | objective | clean | dropout |
| --- | --- | --- | --- | --- |
| J03WMX | causal_reference | 0.4455 | 0.5262 | 0.3648 |
| J03WMX | nested_v6 | 0.4915 | 0.5705 | 0.4125 |
| J03WN1 | causal_reference | 0.4742 | 0.5530 | 0.3953 |
| J03WN1 | nested_v6 | 0.5175 | 0.5960 | 0.4390 |
| J03WOH | causal_reference | 0.5387 | 0.6388 | 0.4387 |
| J03WOH | nested_v6 | 0.5717 | 0.6599 | 0.4835 |
| J03WOY | causal_reference | 0.4959 | 0.5805 | 0.4113 |
| J03WOY | nested_v6 | 0.5435 | 0.6168 | 0.4701 |
| J03WPY | causal_reference | 0.4995 | 0.5740 | 0.4250 |
| J03WPY | nested_v6 | 0.5535 | 0.6279 | 0.4791 |
| J03WQQ | causal_reference | 0.5813 | 0.6679 | 0.4947 |
| J03WQQ | nested_v6 | 0.6110 | 0.6930 | 0.5289 |
| J03WR9 | causal_reference | 0.5422 | 0.6322 | 0.4523 |
| J03WR9 | nested_v6 | 0.5629 | 0.6464 | 0.4793 |

## Comparison with frozen v1 on its three non-training fixtures

| model | clean_spellF1 | dropout_spellF1 |
| --- | --- | --- |
| frozen_v1_reference | 0.6147 | 0.4154 |
| nested_v6 | 0.6521 | 0.4928 |

Only J03WOY, J03WQQ and J03WR9 are used for this frozen-v1 comparison. The frozen
model is never treated as unseen on its own four training matches. V6's outer
training groups differ from V1's original four-match training pool, so this is a
paired input comparison with different training pools, not a pure architecture
ablation. The separate causal-reference comparison controls the training pool.

## Final all-seven-match model

Selected configuration: `context_deep`. Decoder: `prior_4`.
The final choice uses mean inner-validation scores, then refits on all seven
external matches. Outer scores estimate the nested selection procedure, not an
untouched test of this exact final fit. Full configuration, folds, source hashes,
checks and model hash are recorded in experiment.json.

Predictions run at a 5 Hz grid and expand to 25 Hz within each half. HMM estimates
use the same 5 Hz time scale. Offline context uses future tracking observations,
not future possession labels, and cannot cross a half or match boundary. This is
suitable for completed-match inference; do not describe it as a live model.

The final CSV, when present, is Decul_V6.csv. It is a separate candidate and does
not overwrite any earlier submission. Raw target reconstruction is performed only
after model selection and reads no target possession labels or tag teachers.
Target prediction accuracy remains unknown. Nothing is uploaded or submitted.

## Measured outcome — 20261108_053619_utc

The nested V6 procedure achieved **0.5502** combined development
SpellF1 objective versus **0.5110** for the independently refitted causal
reference, a difference of **+0.0392**. V6 had a higher
combined objective on **7 of 7** outer-evaluated matches.

| Metric | Causal reference | Nested V6 | Difference |
|---|---:|---:|---:|
| Clean SpellF1 | 0.5961 | 0.6301 | +0.0340 |
| Mean dropout SpellF1 | 0.4260 | 0.4703 | +0.0443 |
| Short H/A possession recall (<2s, IoU >=0.5) | 0.0962 | 0.1538 | +0.0575 |
| Transition precision (1s tolerance) | 0.6350 | 0.6260 | -0.0090 |
| Transition recall (1s tolerance) | 0.4265 | 0.4903 | +0.0638 |

Final configuration: **context_deep + prior_4**,
refitted on all seven IDSSE matches. This is a fresh model and fresh HMM, not a
renamed V1. The optional unsupervised candidate was evaluated, but final selection
is determined by the recorded inner-validation scores rather than a preference
for more complex methods.

Short-possession recall remains low in absolute terms despite the gain. Transition
recall improves, while transition precision decreases slightly: the model is
more sensitive, with a measurable precision trade-off. These are continuing
failure modes, not solved problems.

The comparison measures the complete selected pipeline; it does not isolate
context from tree regularization and decoder choice. The four XGBoost candidate
families and five decoder settings are defined in `src/run_v6.py`; the frozen
selection record is in `models/experiment.json`, with aggregate development
scores in `results/summary.csv` and `results/validation.csv`. The published
summary is not a replacement for every intermediate trial record.

These results support V6 as a stronger **IDSSE development candidate** relative
to the stated reference. They do not establish accuracy on BAR–ATH or superiority
over a concurrently developed V5. Previously inspected IDSSE matches are not an
untouched holdout, simulated dropout does not cover every real tracking failure,
and IDSSE possession semantics can differ from the competition.

Verification completed: 1,188 score rows; match-group isolation and selection
recomputed; final HMM reconstructed from IDSSE training labels; model/source
hashes checked; 199,319-row target CSV schema checked; separate-process inference
produced byte-identical predictions. The previous Decul submission and V3/V4
source files remain unchanged. No target possession labels were used and nothing
was submitted.
