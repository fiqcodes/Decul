# Decul

**More Than a Hack 2027 — Football Possession Prediction**  
**Technical methodology | Final model**  
**Team:** Decul  
**Model frozen:** 8 November 2026  
**Prediction output:** `Decul.csv` (`frame,label`)

## Executive summary

We infer the ball-possession state of each video frame using **tracking-only features** and a two-stage **XGBoost + Hidden Markov Model (HMM)** pipeline. XGBoost estimates the probabilities of **home possession (H), away possession (A), and dead ball (D)** from observed players, ball position and temporal context. An HMM then decodes the most plausible sequence of states. Frames outside the official evaluation periods receive **X**.

The model was trained with seven external **IDSSE** football matches. No target-match possession labels, Smart Tagging teacher channels, video recognition, or SkillCorner training data were used in the frozen submission model. The final model uses 84 tracking-derived base measurements, expanded to **172 context-aware inputs**, with predictions estimated at 5 Hz and returned to the original 25 Hz timeline. The selected learner is `context_deep` with the `prior_4` HMM decoder.

In a nested, match-grouped **development** evaluation, the final model-selection procedure obtained **0.6301 clean SpellF1**, **0.4703 mean dropout SpellF1**, and a **0.5502 balanced objective**, versus **0.5110** for an independently refitted causal reference. The resulting 199,319-row prediction file was reproduced byte-for-byte from raw target tracking; its SHA256 is `020ebd0cdac4f6bd8228cf1f27193c123b7c73fe872d5655a182300c4556c250`. These figures do **not** establish the unknown BAR–ATH ground-truth score.

## 1. Task definition

For every frame in the organiser's BAR–ATH tracking sequence, predict one of the following states:

| Label | Meaning | Where used |
|---|---|---|
| `H` | Home team in possession | Official match periods |
| `A` | Away team in possession | Official match periods |
| `D` | Ball out of play / dead-ball state | Official match periods |
| `X` | Frame outside configured match periods | Outside period masks only |

The primary assessment metric is **SpellF1**, which scores temporally coherent possession intervals rather than simply per-frame accuracy. The model therefore combines classification with sequence decoding instead of making independent framewise decisions. Target possession is never inferred from hidden labels or evaluated during prediction.

## 2. Data provenance and labeling

### External IDSSE football tracking

Seven existing local IDSSE match artifacts are used for learning: `J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WPY`, `J03WQQ`, and `J03WR9`. The underlying IDSSE collection contains spatial and event information from German professional football. Relevant source attribution: **Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025), _An integrated dataset of spatiotemporal and event data in elite soccer_, Scientific Data 12, 195**, DOI: `10.1038/s41597-025-04505-y`. The original publication DOI is provided above; obtain source data through the authorized IDSSE release or mirror under its applicable licence. Check the source license and third-party redistribution conditions before publishing any data.

Training labels originate from the IDSSE-provided `BallStatus` and `BallPossession` fields. In the project adapter, inactive ball status maps to `D`; valid active possession maps to `H` or `A`. Unknown mappings are rejected rather than recast as `D`. **Neither label field is used as a model feature.** The label convention is vendor-supplied, not independent human reannotation.

### Organiser's BAR–ATH target

The target provides tactical ATD tracking, metadata, pitch configuration and configured match periods. These sources are used **only at final inference** for tracking-derived features and output alignment; no target-match possession ground truth is accessed. The model reads coordinates of players and ball but does not use Smart Tagging outputs.

### Data access and limitations

The public repository contains code, frozen model, and aggregate scores, but intentionally excludes the submission prediction CSV, licensed tracking arrays, and per-frame IDSSE training labels. The CSV is delivered separately via the official submission channel. Reproducing training requires obtaining the IDSSE inputs and preparing the local adapter artifacts described in Section 12. Reproducing inference requires the permitted BAR–ATH tactical source and project-wide feature-building dependencies. The project supplies a record of source hashes and training provenance in `experiment.json`.

## 3. Preprocessing and quality control

The tracked player and ball coordinates are converted consistently into **metres**. IDSSE coordinates use a centered reference; BAR–ATH normalized coordinates are converted using the configured pitch dimensions. Missing coordinate pairs remain `NaN`, rather than being filled with a fictitious (0, 0) location. Feature construction uses player team-slot metadata, and the underlying feature builder guards against implausible velocities (including roughly >12 m/s player and >45 m/s ball motion in the original shared feature pipeline).

The feature extractor computes independent features for each match period. This prevents temporal history, symmetric future context and the HMM from crossing halftime. Input parity checks confirm that the independently reconstructed clean 84-feature values equal the cached IDSSE features; three observation-only measurements previously located in a legacy feature-building stage were rebuilt without reading tagging data.

Missingness is retained as an informative signal. For robustness, features are recomputed from tracked observations **after** synthetic ball and player dropout. Training labels themselves are never modified by dropout.

## 4. Feature engineering: 84 → 172

**Base feature set (84 measurements).** The model retains all 84 tracking-compatible measurements, covering ball positions, player proximity and team control indicators, ball and player movement, spatial spread, visibility and short temporal summaries. Eight explicitly constructed additions include ball missing-streak duration, home/away median player speed, home/away slow-player fraction, home/away team spread, and change in detected player count over five seconds. The 84-feature contract and exact column order are read from the IDSSE feature metadata.

**Sequence context (+88 = 172 measurements).** Twelve selected signals receive four temporal offsets (48 values) and two centered rolling summaries (24 values). Seven anchor signals are copied from the last and next visible ball observations (14 values), with elapsed distances to those observations (2 values). Thus **84 + 48 + 24 + 14 + 2 = 172**.

This is **offline sequence context**: selected features use observations from both before and after a frame, but only *tracking observations*, never future possession labels. It is appropriate for post-match analysis, not live deployment. All windows reset at match-period boundaries.

### Feature selection and dimensionality reduction

The model uses all 84 compatible base measurements instead of restricting the input to a previously selected smaller subset. The 172-input contextual formulation is selected in nested validation. Four predeclared XGBoost learner configurations were tested: `causal84`, `context_regularized`, `context_deep`, and `context_regimes`.

**PCA was tested but not selected.** The `context_regimes` candidate adds eight train-fold-fitted PCA components and eight MiniBatchKMeans tracking-regime distances (188-dimensional representation). No PCA or clustering transform is part of the frozen final `context_deep` model. The supervised learner's tree feature importance and optional SHAP explanations are interpretation tools, **not** an extra post-hoc feature-selection step.

## 5. Train/validation strategy

The experiment uses **three outer group folds split by complete matches**, so no training observations from a held-out match enter that fold's learner or HMM. Within each outer training set, **two additional match-disjoint inner folds** choose the learner and HMM decoder. The outer held-out labels do not influence their own model selection.

| Outer fold | Held-out matches |
|---|---|
| 0 | J03WMX, J03WOY |
| 1 | J03WN1, J03WQQ |
| 2 | J03WOH, J03WPY, J03WR9 |

Each candidate is evaluated on the same held-out matches under four tracking scenarios: clean, original dropout, short gaps, and long gaps. Evaluation uses match-equal weighting. The final model configuration is chosen using **pooled inner-fold development results**, then refitted using all seven IDSSE matches. Consequently, the outer-fold results describe a nested *selection procedure*, not an independent test of the final all-seven-match weights. These fixtures had also been studied in earlier project versions.

## 6. Data augmentation and sample weighting

Training consists of four observation views with per-view weights: **clean 0.40; original dropout 0.20; short gaps 0.20; long gaps 0.20**. The original dropout masks around 43% of ball frames; short-gap and long-gap scenarios mask approximately 25% and 60% respectively, with different contiguous missingness intervals. Player slots receive additional approximately 12% simulated dropout per training period. The source constructs repeatable masks using fixed seeds and periods. The model always receives recomputed features from the masked tracking, while the underlying H/A/D labels remain fixed.

The four views model the risk of intermittent missing ball observations, an important gap between training and the organiser's tracking conditions. Synthetic masks do not cover all real occlusion, detector, camera or identity-switch failure modes.

## 7. XGBoost classifier

The multiclass learner is `XGBClassifier(objective='multi:softprob', num_class=3)` with histogram training and probabilistic output. The base configuration uses seed 6006, `subsample=0.85`, `colsample_bytree=0.90`, `reg_alpha=0.10`, and four training jobs. Candidate-specific hyperparameters were selected on inner match folds.

| Candidate | Trees | Depth | Learning rate | min_child_weight | L2 regularization | Context | PCA/regimes |
|---|---:|---:|---:|---:|---:|---|---|
| `causal84` | 400 | 5 | 0.05 | 8 | 5 | No | No |
| `context_regularized` | 300 | 4 | 0.05 | 10 | 8 | Yes | No |
| **`context_deep` (selected)** | **500** | **6** | **0.04** | **5** | **5** | **Yes** | **No** |
| `context_regimes` | 400 | 5 | 0.04 | 8 | 5 | Yes | Yes |

The selected model outputs a three-class probability distribution at the model's 5 Hz grid, aligned to H/A/D. Training and inference use the same feature columns and order. This stage is trained **separately** from the HMM; it is not a jointly trained neural network.

## 8. Hidden Markov Model and decoding

The HMM learns a three-state first-order temporal model from **clean training-fold labels at 5 Hz**. State prior, initial-state and transition probabilities are calculated from the training labels with pseudocount smoothing, separately for each fold; none is inherited from an earlier fitted HMM. The final HMM is estimated again using all seven eligible IDSSE matches.

Let `p_t(s)` be the XGBoost probability for state `s` at model timestep `t`, `π(s)` the estimated training prior, `T(i,j)` the transition matrix and `q(s)` the initial-state distribution. The selected `prior_4` option uses emission evidence:

`e_t(s) = 4 × [log(max(p_t(s), ε)) − log(π(s))]`.

The Viterbi algorithm selects the state path maximizing the sum of initial-state evidence, emissions and log transition probabilities. Decoding restarts at every half. The candidate search included the raw XGBoost argmax, `prior_0.25`, `prior_1`, `prior_4`, and `posterior_1`. The selected strong evidence weight is a development choice; it should not be interpreted as a physical possession rule.

## 9. Evaluation and experimental results

We compute official-style **SpellF1** at 25 Hz after expanding 5 Hz predictions within each half. Additional metrics include macro F1, accuracy, recovery of H/A spells shorter than two seconds, transition precision/recall within one second and prediction quality on missing-ball frames.

### Main matched nested development comparison

| Metric | Causal reference | Decul | Difference |
|---|---:|---:|---:|
| Clean SpellF1 | 0.5961 | **0.6301** | **+0.0340** |
| Mean dropout SpellF1 | 0.4260 | **0.4703** | **+0.0443** |
| Balanced objective | 0.5110 | **0.5502** | **+0.0392** |
| Short H/A spell recall (<2 s; IoU ≥ 0.5) | 0.0962 | **0.1538** | **+0.0575** |
| Transition precision (±1 s) | **0.6350** | 0.6260 | −0.0090 |
| Transition recall (±1 s) | 0.4265 | **0.4903** | **+0.0638** |

The *balanced objective* averages clean SpellF1 and the mean of the three dropout scenario scores with equal 50% weighting. Decul improves the objective on all seven outer-evaluated IDSSE matches relative to the independently refitted causal baseline. The comparison reflects the full selected pipeline and does not attribute the entire gain solely to context features.

### Performance by tracking scenario

| Scenario | SpellF1 |
|---|---:|
| Clean | 0.6301 |
| Long gaps | 0.4137 |
| Original dropout | 0.4593 |
| Short gaps | 0.5380 |

The absolute recall of very short possession spells remains low (approximately 15.4%), and the improved transition recall comes with modestly lower transition precision. Missing ball conditions remain materially more difficult than clean tracking.

### Interpretation and feature importance

XGBoost feature importance can be inspected from the saved classifier, and the selected feature definitions are documented in the source and metadata contracts. Feature importance is an interpretation aid, **not** a causal explanation or validation metric. SHAP was not used to select the final hyperparameters or features, and it is not required for inference.

## 10. Final inference and file contract

Final target inference is performed only after the configuration is chosen and the model refitted. BAR–ATH tracking is read through the project's tactical ATD adapter, converted once to metres, split into configured official halves and transformed through the same 84 → 172 feature pipeline.

XGBoost probabilities are predicted at a period-aligned **5 Hz** grid. The HMM decodes each half separately and the resulting states expand to the original **25 Hz** frame timeline. The CSV contains exactly two columns, `frame,label`, covering 0 through 199318; official-period states use `H/A/D`, while outside-period frames use `X`.

| Label | Predicted frames |
|---|---:|
| H | 68,462 |
| A | 47,055 |
| D | 43,340 |
| X | 40,462 |
| **Total** | **199,319** |

The final `Decul.csv` and the original frozen-inference output are byte-identical. The established final readiness run reported the **official validator PASS** and matching SHA256. A separate-process raw-target reproduction also matched that exact fingerprint. **No BAR–ATH ground-truth performance is known.**

## 11. Reproducibility, artifacts and integrity

The frozen model originates from the documented training run on 8 November 2026. The repository includes `models/model.joblib`, `models/experiment.json`, the aggregate development tables under `results/`, and the training/inference implementation under `src/`. Only the feature-metadata file path was relocated for the public repository; its JSON contents and the frozen model weights are unchanged. The integrity checker records source hashes, while `models/experiment.json` retains the original provenance. Reproduction does not depend on an interactive analysis environment.

| Artifact | SHA256 |
|---|---|
| `Decul.csv` (separate form submission) | `020ebd0cdac4f6bd8228cf1f27193c123b7c73fe872d5655a182300c4556c250` |
| `models/model.joblib` | `69db493a71c99b53cfec4032f5b27429be67d1d9ba349af725ec28796bfb5282` |

**Review the methodology and development evidence without retraining:**

Read `docs/Decul.md`, `models/experiment.json`, `results/summary.csv`, `results/validation.csv` and `results/development_report.md`. The runnable training and inference source is in `src/`.

**Reproduce the final CSV from the frozen model and permitted raw BAR–ATH tracking:**

```bash
cd /path/to/Decul
conda activate mtah2027
export MTAH_DATA_DIR=/path/to/official/data
python scripts/check_package.py
python scripts/reproduce_submission.py
python evaluation/validate_submission.py output/Decul.csv
shasum -a 256 output/Decul.csv
```

The reproduction script generates `output/Decul.csv`, verifies its SHA256 against the frozen manifest, and refuses to overwrite an existing output. The original project owner ran this reproduction successfully and independently passed the official CSV-format validator. The public repository **does not include** the submission CSV, which is provided separately for the Google Form.

Full training requires seven legitimately obtained IDSSE match artifacts and their separately prepared adapter/label files under `data/derived/idsse/`. Historical reference comparisons additionally require separately retained research artifacts. Those licensed datasets and research artifacts are **not included** in the public repository. The training source under `src/` records the protocol and expected input contracts; recreating the full training data-preparation chain requires additional source/data access. Fresh training may not produce byte-identical model weights across platforms. **The verified bitwise claim is final inference with the supplied frozen model**, not end-to-end retraining from downloaded raw IDSSE files.

The copied organiser code (`games.py`, `starter/`, and `evaluation/`) retains its MIT licence notice in `third_party/METRICA_LICENSE.txt`. That licence does not grant permission to redistribute licensed match data.

## 12. Limitations, ethics and data handling

This is a tracking-only offline predictor and is not a live model. Results are bounded by annotation-policy differences between IDSSE and BAR–ATH, possible team/league/provider shift and synthetic rather than naturally occurring missingness. The seven development matches were previously explored; no claim of untouched external validation is made. The model can miss very short spells, and an HMM designed to reduce spurious transitions can also miss genuine possession changes.

Licensed raw football tracking, event labels, tactical XML and caches must **not** be committed to a public repository. The separately submitted `Decul.csv` is intended for the official submission channel; check competition restrictions before public distribution of reconstructive target predictions. Source attribution and third-party licenses must be retained. The submitted model uses only the IDSSE training labels and tactical tracking at inference; unrelated research candidates are not part of this pipeline.

---

**Conclusion.** Decul is a reproducible, frozen XGBoost + HMM model with measurable gains in nested IDSSE development experiments and a verified submission-file contract. Its actual possession accuracy on BAR–ATH remains unknown until official evaluation.
