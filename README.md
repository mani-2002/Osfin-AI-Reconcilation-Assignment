# recon-ml — reconciliation as cross-language retrieval

A working implementation of **Chew (ICAIF '20), *Unsupervised-Learning Financial Reconciliation: a Robust, Accurate Approach Inspired by Machine Translation***, extended into a hybrid matcher with calibrated confidence and a human-in-the-loop cycle.

Bank statements and the check register describe the same transactions in two different "languages" (`BP GAS #1775 / DEBIT` vs `Fill up / DR / Gas Station`). The system learns that translation from matches a human has already accepted, and gets better as more matches are validated.

```
            ┌──────────────┐   QC    ┌──────────────────────────┐   QC    ┌─────────┐
 bank  ───▶ │ unique-amount │ ──────▶ │ ML: SVD "cross-language" │ ──────▶ │ manual  │
 register ─▶│   matching    │         │ + learned date/amount LR │         │ mop-up  │
            └──────────────┘         └──────────────────────────┘         └─────────┘
                    │ accepted                     ▲  │ accepted                │ resolved
                    └──────────────▶ training store (parallel corpus) ◀────────┘
```

## Results at a glance

| Experiment | Precision | Recall | F1 |
|---|---|---|---|
| Provided data, full pipeline (unique 286 + ML 22) | **1.000** | **1.000** | **1.000** |
| Provided data, ML only, **amount hidden**, 240 training pairs | 0.996 | 0.844 | 0.913 |
| Synthetic 12 months (44% non-unique amounts), all automatic matches | 0.992 | 0.887 | 0.936 |
| Synthetic, ML stage only (non-unique pairs) — with learning carried over | 0.974 | 0.698 | 0.813 |
| … same, retraining from scratch each month | 0.968 | 0.596 | 0.738 |

The full analysis — learning curves, hardest transactions, ablations, design decisions and limitations — is in **[docs/ANALYSIS.md](docs/ANALYSIS.md)**.

## Quick start

```bash
python3 -m pip install -r requirements.txt

# Full cycle on the provided data, interactive QC in the terminal
python3 -m recon reconcile --state state/training.json --out out/matches.csv

# Same, non-interactive: a simulated reviewer uses the ground truth (ids share a suffix: B0047 <-> R0047)
python3 -m recon reconcile --review oracle --manual oracle --truth id --state state/training.json --out out/matches.csv

# What has the model learned?  (word alignments, DEBIT<->DR, lag and rounding profile)
python3 -m recon explain --state state/training.json

# Harder data: 12 months with many duplicate amounts and a bank format change in month 7
python3 -m recon generate --out data/synthetic
python3 -m recon simulate                 # month-by-month, learning carried forward
python3 -m recon simulate --no-carry      # baseline: no learning between months

# Tests and all experiments behind docs/ANALYSIS.md
python3 -m pytest -q
python3 scripts/run_experiments.py        # writes results/ (≈5 min)
```

**Review modes** (`--review`): `interactive` (accept/reject each proposal, `A` to accept the rest), `oracle` (simulated human), `auto:0.95` (scaled-back QC — auto-accept above a confidence). **Manual step** (`--manual`): `none`, `interactive`, `oracle`.

The **match → review → improve** loop is literal: `--state` loads the validated pairs before the run and saves them after, so the next batch (month) starts from a better model.

## How it works

1. **Unique-amount matching** (`recon/unique.py`). Amounts that occur once on each side are paired. Confidence = 1 − P(coincidental match), with the chance probability estimated from a KDE of the register's amount distribution, discounted for implausible lags or type mismatch. Flags: `DATE_LAG_EXCEEDS_THRESHOLD`, `REGISTER_DATED_AFTER_BANK`, `TYPE_MISMATCH`, `SOURCE_HAS_ANOMALY_NOTE`, `DESCRIPTIONS_DISAGREE`.
2. **Cross-language SVD model** (`recon/svd_model.py`) — the paper's Option A. Every transaction is a bag of source-prefixed terms (words, character trigrams, raw type code, amount magnitude). Mutual information between bank and register terms over the validated pairs → sparse alignment matrix `A`; the term × pair matrix `D` (log-entropy weighted) → `X = A·D` → SVD; new transactions are projected with `Uₖᵀ(A·t)` and compared by cosine.
3. **Hybrid scorer** (`recon/matcher.py`). Cosine, date lag, amount difference and shared raw tokens (e.g. cheque numbers) are combined as **learned likelihood ratios** (Fellegi–Sunter style): `P(feature | match)` from validated pairs (cross-fitted for the cosine), `P(feature | non-match)` from the unlabelled batch. Nothing like "lag ≤ 5 days" is hard-coded in the ML stage. A global 1:1 assignment (Hungarian) picks pairs; confidence is the posterior that a pair is the true one, so two indistinguishable candidates each get ~50% and go to a human instead of being guessed.
4. **Workflow** (`recon/workflow.py`). Only human-made or human-accepted matches enter the training store; rejected pairs are never re-proposed.

## Repository layout

```
recon/            data.py  features.py  svd_model.py  matcher.py  unique.py
                  workflow.py  metrics.py  synth.py  simulate.py  cli.py
tests/            unit + integration tests (pytest)
scripts/          run_experiments.py — reproduces every number in docs/ANALYSIS.md
data/             provided CSVs; data/synthetic/ generated 12-month dataset with truth.csv
results/          experiment outputs (CSV, PNG, summary.md)
docs/             ANALYSIS.md
```
