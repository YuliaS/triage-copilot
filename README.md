# Northstar Triage Copilot

Decision-support prototype for a frontline support desk. For a new case it predicts
the case type, category, priority, suggested team, and escalation risk, and explains
the call in plain language. Built as a Gradio app backed by six scikit-learn models.

> Northstar Desk is a fictional company; the case data is synthetic.

## What it does

The app has two screens:

- **Triage a case** — paste a case summary (plus a few known fields), press *Triage
  case*, and read one result card: case type, category, two priority estimates,
  suggested route, escalation risk, and a *Why* line. A *Load example* button fills
  a worked case.
- **How the models work** — model cards (algorithm, inputs, split, held-out
  metrics), a per-model held-out inspector (predicted vs actual on cases the model
  never trained on), the saved metrics table, and a download button per training
  notebook.

The Triage tab runs a **cascade**: case type and category are predicted from the
summary, then feed the priority and escalation models.

## The six models

| Key | Shown as | Target | Design | Held-out |
|---|---|---|---|---|
| `m1_case_type` | Case type | `case_type` | summary-only, group split | macro-F1 0.875 |
| `m2_category` | Category | `category` | summary-only, group split | macro-F1 0.890 |
| `m3_priority_tracy` | Priority — direct | `priority` | summary + plan + tenure + channel, time split | macro-F1 0.587 |
| `m3_priority_milind` | Priority — check | `priority` | summary + case_type + channel, double-check, time split | macro-F1 0.502 |
| `m5_route` | Suggested route | `assigned_team` | summary-only, time split, review gate | macro-F1 0.805 |
| `m4_escalation` | Escalation risk | `escalated` | summary + tags + category + …, tuned threshold, time split | ROC-AUC 0.961 |

**Two priority models by design.** `m3_priority_tracy` predicts the priority
directly (and uses `plan_tier`/`tenure`). `m3_priority_milind` refuses to trust an
exact prediction — exact priority has a ~50% ceiling because the historical labels
are inconsistent — and instead *flags* cases where the agent's priority is two or
more levels from its estimate. The result card shows both and marks disagreement.

### Method notes

- **Honest evaluation.** Classification uses a group-aware split (identical
  summaries never straddle train/test); the sequential models use a chronological
  split (train early months, score later months). Metrics are held-out.
- **Leakage / fairness.** Inbound fields only — no outcome or history columns.
  `plan_tier` and `region_uk` are excluded everywhere except Tracy's priority model,
  which keeps `plan_tier`/`tenure` to illustrate the trade-off.
- **Escalation threshold.** `m4_escalation` tunes its decision threshold on a
  validation set (maximising F1 of the rare escalation class); it favours precision
  over recall at the saved threshold.

## Repository layout

```
northstar_prep.py              shared data loading, splits, feature helpers
train_m1_case_type.ipynb       one training notebook per model; each imports
train_m2_category.ipynb        northstar_prep, trains, evaluates on a held-out
train_m3_priority_tracy.ipynb  split, refits on all data, and saves a bundle
train_m3_priority_milind.ipynb to models/*.joblib
train_m4_escalation.ipynb
train_m5_route.ipynb
triage-copilot-v2.ipynb        the app — loads the six bundles, trains nothing
run_copilot_v2.py              headless launcher for the app
models/                        saved model bundles (model + held-out test + metrics)
data/                          data_after_preprocessing_variables.xlsx (used) and
                               the raw monthly Q*.csv exports (provenance)
docs/                          user journey, model comparison, pitch slides
```

A model **bundle** is a dict: `{model, target, features, classes, kind, label,
metrics, test, meta}` — the `test` frame holds the held-out rows with precomputed
predictions so the app can show predicted-vs-actual without retraining.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (optional) retrain — run each train_*.ipynb; they write to models/
# serve the app:
python run_copilot_v2.py        # http://127.0.0.1:7866
```

The app loads the committed `models/*.joblib`, so it runs without retraining.

## Credits

The modelling approaches are adapted from a team hackathon: direct priority (Tracy),
priority double-check (Milind), escalation with tuned threshold (Anna), summary-only
routing with a review gate (Christian), and the case-type / category text models.
