# Decul — Football Possession Prediction

**More Than a Hack 2027 · Team Decul · Final submission pipeline**

Decul predicts possession as **H** (home), **A** (away), or **D** (dead ball) from tactical player-and-ball tracking. An XGBoost classifier uses 84 tracking-derived measurements plus 88 temporal-context measurements (172 inputs total). An HMM/Viterbi decoder produces coherent possession sequences. Frames outside official match periods are marked **X**.

## Repository structure

| Path | Purpose |
|---|---|
| `src/` | Feature construction, model implementation, training protocol, and HMM decoding |
| `local_modules/possession_features.py` | Tracking-based feature builder |
| `configs/` | Feature configuration and ordered metadata contract |
| `models/model.joblib` | Frozen XGBoost classifier and HMM parameters |
| `models/experiment.json` | Training protocol, development selections, hashes, and provenance |
| `results/` | Aggregate development scores and interpretation |
| `docs/Decul.md`, `docs/Decul.pdf` | Technical methodology and limitations |
| `scripts/reproduce_submission.py` | Exact inference from authorised tactical inputs |
| `scripts/check_package.py` | Integrity and dependency checks |
| `tests/` | Package tests |

All files needed to inspect or run the **frozen inference pipeline** are included. The supplied model contains a serialized reference to its original Python class location; its internal source filenames are retained for compatibility, **not** because multiple submission models are required.

## Training data and permitted inputs

The model was fitted and selected on **seven external IDSSE matches**, using provider-supplied `BallStatus` and `BallPossession` fields as training labels. Training and tuning did **not** use possession labels, video, or manually annotated data from the organiser's BAR–ATH game.

The organiser's BAR–ATH **tactical ATD tracking and metadata** are used **only for final inference**, including player/ball observations, team mapping, and match-period alignment. No video recognition or Smart Tagging teacher model is used in the final inference pipeline.

The repository includes the source module and four small feature/contract files required for inference:

- `local_modules/possession_features.py`
- `configs/bar_ath_feature_options.json`
- `configs/feature_metadata_public.json`
- `data/derived/idsse/J03WPY/idsse_feature_contract.json` — schema only; no tracking observations

Organiser-provided source modules (`games.py`, `starter/`, and `evaluation/`) are also included with the relevant licence notice at `third_party/METRICA_LICENSE.txt`. This notice **does not** licence redistribution of competition match data.

## Installation

Use Python 3.11 and a compatible environment, then install the directly required packages:

```bash
python -m pip install -r requirements.txt
```

For users of the original competition environment, activate it instead (for example, `conda activate mtah2027`). Transitive and platform-specific dependencies have not been fully locked across operating systems; this is not a guaranteed cross-platform lockfile.

## Reproduce the final possession CSV

With the organiser's legitimate tactical ATD bundle available locally, set `MTAH_DATA_DIR` to the parent directory containing `tactical/BAR-ATH_tactical_atd.xml` and `tactical/BAR-ATH_tactical_atd.txt`:

```bash
cd /path/to/Decul
export MTAH_DATA_DIR=/path/to/official/data
python scripts/check_package.py
python scripts/reproduce_submission.py
python evaluation/validate_submission.py output/Decul.csv
shasum -a 256 output/Decul.csv
```

`reproduce_submission.py` runs the frozen model; it **does not retrain**. It refuses to overwrite an existing `output/Decul.csv`, and checks that the generated file matches the recorded digest:

```text
020ebd0cdac4f6bd8228cf1f27193c123b7c73fe872d5655a182300c4556c250
```

The validated output has **199,319** rows and the header `frame,label`. The submission CSV is provided separately through the competition form; it is not committed to this public code repository. Exact raw-ATD inference has previously produced the matching digest in the original environment. A fresh-clone, cross-platform end-to-end run has not been independently established.

## Model development and performance

The selected model is XGBoost `context_deep` with the HMM decoder `prior_4`. The candidate configuration and decoder were chosen with **match-grouped nested cross-validation**, followed by a fit using all seven IDSSE matches.

| IDSSE development metric | Decul | Independently fitted causal reference |
|---|---:|---:|
| Clean SpellF1 | **0.6301** | 0.5961 |
| Mean dropout SpellF1 | **0.4703** | 0.4260 |
| Balanced objective | **0.5502** | 0.5110 |
| Short possession spell recall (<2 s) | **0.1538** | 0.0962 |

These are **external IDSSE development scores, not BAR–ATH competition scores**. The main model improves the balanced objective on all seven development matches; however, it remains sensitive to missing tracking observations and misses many possession intervals shorter than two seconds. Actual BAR–ATH possession performance is unknown because its ground truth is unavailable to participants.

For model design, training protocol, and full metrics, see `docs/Decul.pdf`, `docs/Decul.md`, `models/experiment.json`, and `results/development_report.md`. The experiment manifest and machine-readable result tables preserve original internal identifiers for auditability; they are not separate models to be submitted.

## Reproducibility scope and data rights

- **Supported exact reproduction:** inference from authorised BAR–ATH tactical ATD data using the supplied frozen model, with output verified by SHA256.
- **Not claimed:** independently repeating seven-match training from scratch using only this public repository. That requires separate licensed IDSSE inputs and their preprocessing adapters, and refitted weights are not guaranteed to be byte-identical across environments.
- **Never upload:** raw licensed match video, tracking, Smart Tagging data, private caches, credentials, or reconstructive datasets to this public repository or unauthorised external AI services.

External training data reference: Bassek, Rein, Weber & Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, *Scientific Data* 12, 195. https://doi.org/10.1038/s41597-025-04505-y

The organiser's MIT notice applies only to the included organiser code. It does not automatically licence Decul's original code, model, or third-party datasets.
