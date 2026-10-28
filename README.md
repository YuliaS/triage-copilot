# Northstar Triage Copilot

Decision-support prototype for a frontline support desk. For a new case it predicts
the case type, category, priority, suggested team and escalation risk, suggests tags,
and surfaces similar past cases with how they were resolved — and explains the call
in plain language. A Gradio app backed by scikit-learn models.

> Northstar Desk is a fictional company; the case data is synthetic.

## What it does

A single-page dashboard with a left sidebar and two views:

- **Triage (default)** — a wider **Case details** panel (dominant case-summary box
  with *Load example*, plus channel / plan / month / tenure / agent's priority /
  tags) and a compact **Triage prediction** panel showing case type, category, two
  priority estimates, suggested route and escalation risk as rows with confidence
  bars, a red risk pill and a *Why* line. Below both, a full-width **Similar cases**
  table (title, case type, category, priority, route, risk, similarity, created).
  *Run triage* also auto-suggests tags into the (optional) tags box.
- **View models** — the header button opens the exploration view: model cards, the
  normalised held-out metrics table, a held-out case inspector, a confusion matrix +
  per-class report per model, and a download button per training notebook.

Triage runs a **cascade**: case type and category are predicted from the summary,
then feed the priority and escalation models.

## Models

See `docs/MODELS.md` for full documentation. In short:

| ID | Shown as | Target | Held-out |
|----|----------|--------|----------|
| M1 | Case type | `case_type` | macro-F1 0.875 |
| M2 | Category | `category` | macro-F1 0.890 |
| M3-direct | Priority — direct | `priority` | macro-F1 0.587 |
| M3-review | Priority — review check | `priority` | macro-F1 0.502 (used as a flag) |
| M4 | Escalation risk | `escalated` | ROC-AUC 0.952 |
| M5 | Suggested route | `assigned_team` | macro-F1 0.805 |

Plus two helper tools: **similar-case search** (retrieval) and a **tag suggester**.

### Tags are leakage

The `tags` column is written by agents *after* they understand a case, so it encodes
the fields the models predict. **No predictive model uses tags as a feature.** The
tag suggester exists only to help the agent tag faster and more consistently; its
output is never fed back into the models.

### Honest evaluation, normalised

`case_pipeline.evaluate()` applies one metric set to every model (accuracy, macro-F1,
weighted-F1, Cohen's kappa; ROC-AUC/PR-AUC for the binary model), and each notebook
also stores a confusion matrix and per-class report. Text models use a group-aware
split (identical summaries never straddle train/test); sequential models use a
chronological split. `plan_tier`/`region_uk` are excluded everywhere except the
direct priority model; protected attributes are never used.

## Repository layout

```
case_pipeline.py         shared loading, splits, features, normalised evaluation,
                         similar-case retrieval and tag-suggestion helpers
training/                one notebook per model; each imports case_pipeline, trains,
                         evaluates on a held-out split, refits on all data, and saves
                         a bundle to models/
app/
  app.ipynb              the app — loads the bundles, trains nothing
  run_app.py             headless launcher
models/                  saved bundles (git-ignored; produced by the training notebooks)
data/
  cases_clean.xlsx       the cleaned dataset the pipeline reads
  raw/                   the original monthly Q*.csv exports (provenance)
docs/                    USER_JOURNEY.md, MODELS.md, pitch_slides.md
```

A model **bundle** is a dict (`model, target, features, classes, kind, label,
metrics, test, report, confusion, meta`); the `test` frame holds held-out rows with
precomputed predictions so the app can show predicted-vs-actual without retraining.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. build the models (writes models/*.joblib — these are git-ignored)
jupyter nbconvert --to notebook --execute --inplace training/train_*.ipynb

# 2. serve the app
python app/run_app.py        # http://127.0.0.1:7866
```

The model files are **not** committed (see `.gitignore`); run the training notebooks
once to generate them, then launch the app.

## Credits

Modelling approaches were contributed by several teammates during a hackathon
(anonymised as Source A–F). See `docs/MODELS.md`.
