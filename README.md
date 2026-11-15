# Decul — Football Possession Prediction

**More Than a Hack 2027 · Team Decul · Final submitted-method candidate**

We predict home possession (**H**), away possession (**A**), and dead ball (**D**) using 84 tracking-compatible measurements, offline temporal context (172 XGBoost inputs), and an HMM sequence decoder. Frames outside the configured period receive **X**.

The saved model is the frozen, validated version used to produce `Decul.csv`. Historical implementation filenames `models_v6.py` / `run_v6.py` and the internal `version='V6'` are **intentionally retained** for serialized-model compatibility and scientific provenance; they are not separate submission versions.

## Repository contents

| Path | Purpose |
|---|---|
| `src/models_v6.py`, `src/run_v6.py` | Original model, feature, training and HMM source (one metadata-path relocation in `models_v6.py`) |
| `models/model.joblib` | Frozen XGBoost + HMM payload |
| `models/experiment.json` | Exact model architecture, splits, feature/data lineage, hashes |
| `results/` | Aggregate nested development evaluation (no per-frame licensed tracking) |
| `docs/Decul.md`, `docs/Decul.pdf` | Technical documentation |
| `scripts/reproduce_submission.py` | Inference from permitted BAR–ATH tactical inputs; exact checksum check |
| `scripts/check_package.py` | Integrity, required-module and data-access readiness check |
| `scripts/complete_repository.py` | One-time local collector for missing original shared source modules |

## Exact source dependencies and repository completion

This package **already includes** the four user-provided custom tracking-feature and feature-contract files:

- `local_modules/possession_features.py` (SHA256 matched to frozen source manifest)
- `configs/bar_ath_feature_options.json`
- `configs/feature_metadata_public.json`
- `data/derived/idsse/J03WPY/idsse_feature_contract.json` (84 column names and non-sensitive schema metadata; no tracking observations)

The V6 training/inference implementation also relies on the **organiser's official source files** `games.py`, `starter/`, `evaluation/possession_metrics.py` and `evaluation/validate_submission.py`. The original organiser MIT notice is copied to `third_party/METRICA_LICENSE.txt`. It covers the copied organiser code, **not** the restricted match data. These were not in the uploaded V6/auxiliary archives. Use the exact files from the official project checkout that was used for the frozen V6 reproduction:

```bash
cd ~/Documents/more-than-a-hack-2027
conda activate mtah2027
python Decul/scripts/complete_repository.py --project-root .
python Decul/scripts/check_package.py
```

`complete_repository.py` copies only necessary public-source Python modules, verifies that the four packaged custom files match the ones from the original project (when present), and writes `Decul_GitHub_Ready_v2.zip` in the parent project folder. It does **not** copy raw video/tracking/labels, model caches, or project-wide data. The script refuses to overwrite an existing ZIP.

**A completed ZIP only means the source has been assembled. It is not yet independently verified to reproduce predictions from a clean clone.** Before publication, perform the actual raw-ATD reproduction (using the official data access configuration) and check that `output/Decul.csv` has the frozen SHA256. Also review the copied organiser modules for attribution and redistributability, and exclude any local secrets.

## Environment

Use Python 3.11 and your pre-existing environment, or install the direct Python dependencies:

```bash
conda activate mtah2027
# or, in a fresh environment: python -m pip install -r requirements.txt
```

The direct dependency versions were recorded in existing package environment notes, but the transitive package tree and platform-specific ATD dependencies have **not been independently locked**. This is not a verified cross-platform lockfile.

## Reproduce the frozen submission

Once the exact shared source modules are collected and you have legitimate access to the organiser's tactical ATD inputs under the project's `games.py` data-access contract:

```bash
cd ~/Documents/more-than-a-hack-2027/Decul
conda activate mtah2027
# Set this to the directory with tactical/BAR-ATH_tactical_atd.{xml,txt}
export MTAH_DATA_DIR="$HOME/Documents/more-than-a-hack-2027/data"
python scripts/check_package.py
python scripts/reproduce_submission.py
python evaluation/validate_submission.py output/Decul.csv
shasum -a 256 output/Decul.csv
```

The script creates `output/Decul.csv` only after predicting from original raw tracking and verifying exact SHA256 `020ebd0cdac4f6bd8228cf1f27193c123b7c73fe872d5655a182300c4556c250`. It never re-trains the model or overwrites a frozen output. The final `Decul.csv` for the Google Form is also distributed separately from this public-source bundle.

Training the model from scratch additionally requires preparing seven IDSSE fixtures and their feature/label contracts under `data/derived/idsse/`, plus the original V1 reference artifacts under `data/derived/idsse/trained_model_v2/` for the historical comparison run. These are **training-only research dependencies**; frozen V6 inference does not require them. See `src/run_v6.py` for the exact protocol. Training refits are not guaranteed to yield byte-identical serialized model weights across environments; the supported **exact reproduction claim concerns inference with the frozen weights**.

## Review the methodology and development results

To understand how the model was trained, tuned and evaluated, read these tracked artifacts directly (no interactive environment required):

- `docs/Decul.md` and `docs/Decul.pdf`: methodological explanation, dataset provenance, limitations, metrics and output contract.
- `models/experiment.json`: saved model-selection protocol, candidate choices, features, checksums and provenance.
- `results/summary.csv`, `results/validation.csv`, `results/development_report.md`: aggregate development evidence and interpretation.
- `src/run_v6.py`, `src/models_v6.py`, `local_modules/possession_features.py`: implementation of feature construction, candidate fitting, HMM decoding and inference.

Inference is fully script-based. **PCA and SHAP are not part of the final prediction pipeline.** The confirmed exact reproduction claim applies to frozen-model inference, not to rerunning seven-match training without the licensed IDSSE input artifacts.

## Development performance (not competition scores)

- Clean SpellF1: **0.6301**
- Average dropout SpellF1: **0.4703**
- Balanced objective: **0.5502**
- Frozen-model CSV SHA256: `020ebd0cdac4f6bd8228cf1f27193c123b7c73fe872d5655a182300c4556c250`
- Frozen-model SHA256: `69db493a71c99b53cfec4032f5b27429be67d1d9ba349af725ec28796bfb5282`

These scores are match-grouped **IDSSE development** measurements, not a true independent BAR–ATH score. They do not prove generalization across tracking vendors. The shared SkillCorner audit and experimental V7 research are not part of this frozen submission model.

## License, attribution and safety

IDSSE: Bassek, Rein, Weber & Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, *Scientific Data* 12, 195 (https://doi.org/10.1038/s41597-025-04505-y). Comply with IDSSE and competition data rights. Never commit licensed raw tracking, event files, private caches, credentials, or forbidden derived tracking dumps. The model weights themselves are provided for inspectable competition inference.

## Public GitHub staging safeguards

The `.gitignore` blocks all local tracking under `data/` while allowing exactly one schema-only metadata file `data/derived/idsse/J03WPY/idsse_feature_contract.json`, which the frozen source loads by its canonical path. Confirm this one file appears in your Git staging area. **Never** force-add the remainder of `data/`.

For copied organiser source, retain `third_party/METRICA_LICENSE.txt` with the original MIT notice and scope disclaimer. Decul authors should separately choose any licence for their own code; the organiser MIT licence alone does not license Decul's model or external datasets.

Parts of `docs/Decul.md` describe historical workspace paths from the original model audit. For running this standalone public repository, follow the current commands in this README.
