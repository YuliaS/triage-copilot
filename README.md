# Northstar Triage Copilot

Decision-support prototype for a frontline support desk. For a new case it predicts
the case type, category, priority, suggested team, and escalation risk, and explains
the call in plain language. A Gradio app backed by six scikit-learn models.

> Northstar Desk is a fictional company; the case data is synthetic.

## What it does

Two screens:

- **Triage a case** — paste a case summary (plus a few known fields), press *Triage
  case*, and read one result card: case type, category, two priority estimates,
  suggested route, escalation risk, and a *Why* line. A *Load example* button fills a
  worked case; an *Explore how the models work →* button jumps to the second tab.
- **Explore the models** — model cards, the normalised held-out metrics table, a
  held-out case inspector (predicted vs actual on cases the model never trained on),
  a confusion matrix + per-class report per model, and a download button per
  training notebook.

The Triage tab runs a **cascade**: case type and category are predicted from the
summary, then feed the priority and escalation models.

## The six models

| Key | Shown as | Target | Design | Held-out |
|---|---|---|---|---|
| `m1_case_type` | Case type | `case_type` | summary-only, group split | macro-F1 0.875 |
| `m2_category` | Category | `category` | summary-only, group split | macro-F1 0.890 |
| `m3_priority_direct` | Priority — direct | `priority` | summary + plan + tenure + channel, time split | macro-F1 0.587 |
| `m3_priority_review` | Priority — review check | `priority` | double-check, time split | macro-F1 0.502 |
| `m5_route` | Suggested route | `assigned_team` | summary-only, time split, review gate | macro-F1 0.805 |
| `m4_escalation` | Escalation risk | `escalated` | recall-targeted threshold, time split | ROC-AUC 0.961 |

**Two priority models by design.** One predicts the priority directly; the other
refuses to trust an exact prediction — exact priority has a ~50% ceiling because the
historical labels are inconsistent — and instead *flags* cases where the agent's
priority is two or more levels from its estimate. The card shows both and marks
disagreement. See `docs/MODEL_COMPARISON.md` for the full comparison.

### Method

- **Honest evaluation, normalised.** `northstar_prep.evaluate()` applies one metric
  set to every model (accuracy, macro-F1, weighted-F1, Cohen's kappa; ROC-AUC/PR-AUC
  for the binary model), and each notebook also stores a confusion matrix and
  per-class report. Text models use a group-aware split (identical summaries never
  straddle train/test); sequential models use a chronological split.
- **Leakage / fairness.** Inbound fields only — no outcome or history columns.
  `plan_tier` and `region_uk` are excluded everywhere except the direct priority
  model, which keeps them to show the trade-off. Protected attributes are never used.

## Repository layout

```
northstar_prep.py        shared loading, splits, features, and normalised evaluation
training/                one notebook per model; each imports northstar_prep, trains,
                         evaluates on a held-out split, refits on all data, and saves
                         a bundle to models/
app/
  app.ipynb              the app — loads the six bundles, trains nothing
  run_app.py             headless launcher
models/                  saved bundles (git-ignored; produced by the training notebooks)
data/
  data_after_preprocessing_variables.xlsx   the cleaned dataset the pipeline reads
  raw/                   the original monthly Q*.csv exports (provenance)
docs/                    USER_JOURNEY.md, MODEL_COMPARISON.md, pitch_slides.md
```

A model **bundle** is a dict: `{model, target, features, classes, kind, label,
metrics, test, report, confusion, meta}` — the `test` frame holds held-out rows with
precomputed predictions so the app can show predicted-vs-actual without retraining.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. build the models (writes models/*.joblib — these are git-ignored)
jupyter nbconvert --to notebook --execute --inplace training/train_*.ipynb
#    (or open and run each notebook)

# 2. serve the app
python app/run_app.py        # http://127.0.0.1:7866
```

The model files are **not** committed (see `.gitignore`); run the training notebooks
once to generate them, then launch the app.

## Credits

Modelling approaches were contributed by six teammates during a hackathon and are
anonymised here as Source A–F (see `docs/MODEL_COMPARISON.md`).
