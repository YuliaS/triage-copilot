# Models

Documentation for every model served by the app. Each predictive model is trained
in its own notebook under `training/` (importing `case_pipeline.py`), evaluated on a
held-out split with a **normalised** metric set, refit on all data, and saved as a
bundle in `models/`.

## Feature contract

All models use **inbound fields only** — information known when a case arrives.
Outcome and history fields (resolution time, status, CSAT) are never inputs.

**Tags are leakage.** The `tags` column is written by agents *after* they understand
a case, so it already encodes the fields we predict (category, team, priority,
escalation). **No predictive model uses `tags` (or the derived `n_tags`) as a
feature.** Tags are only *suggested* to the agent as a workflow aid (see the tag
suggester below) and are never fed back into the models.

For fairness, `plan_tier` and `region_uk` are also excluded from every model —
except the direct priority model, which keeps `plan_tier`/`tenure` to show the
trade-off. Protected attributes (gender, age) are never used.

## Predictive models

| ID | Predicts | Inputs | Split | Notebook |
|----|----------|--------|-------|----------|
| M1 | Case type | summary | group-on-summary | `train_m1_case_type.ipynb` |
| M2 | Category | summary | group-on-summary | `train_m2_category.ipynb` |
| M3-direct | Priority (direct) | summary + plan + tenure + channel | time | `train_m3_priority_direct.ipynb` |
| M3-review | Priority (review check) | summary + case_type + channel | time | `train_m3_priority_review.ipynb` |
| M4 | Escalation risk | summary + category + channel + month + tenure | time 70/15/15 | `train_m4_escalation.ipynb` |
| M5 | Suggested route | summary | time | `train_m5_route.ipynb` |

### Normalised evaluation

`case_pipeline.evaluate()` computes the same core for every model — **accuracy,
macro-F1, weighted-F1, Cohen's kappa**, plus **ROC-AUC and PR-AUC** for the binary
model — and each notebook also stores a **classification report** and **confusion
matrix**. Each model keeps a *signature* metric relevant to its design. Splits are
not forced to match: text models use a group-aware split; sequential models use a
chronological split — each the choice for its task, and recorded with the
metrics.

### Held-out results

| Model | split | accuracy | macro-F1 | weighted-F1 | kappa | signature |
|-------|-------|----------|----------|-------------|-------|-----------|
| M1 Case type | group | 0.875 | 0.875 | 0.875 | 0.750 | — |
| M2 Category | group | 0.901 | 0.890 | 0.902 | 0.889 | confidence gap ≈ 0.19 |
| M3 Priority — direct | time | 0.553 | 0.587 | 0.550 | 0.379 | within-1 ≈ 0.93 |
| M3 Priority — review | time | 0.444 | 0.502 | 0.440 | 0.241 | within-1 ≈ 0.93, flag ≈ 0.07 |
| M5 Suggested route | time | 0.848 | 0.805 | 0.854 | 0.776 | needs-review ≈ 0.16 |
| M4 Escalation risk | time | — | 0.819 | — | 0.638 | ROC-AUC 0.952 · PR-AUC 0.734 · recall 0.78 |

### Notes per model

- **M1 / M2 — text classifiers.** Summary-only, group-aware split so identical
  summaries cannot leak across train/test. M2 adds a confidence analysis to gate
  low-confidence category suggestions.
- **M3 — two priority models, one lesson.** The *direct* model predicts the
  priority outright (keeping `plan_tier`/`tenure`); the *review* model omits those
  proxies and, because exact priority sits near a ~0.5 ceiling while being ~93%
  within one level, is used to **flag** cases where the agent's priority is ≥2
  levels from its estimate rather than to assert a value.
- **M4 — escalation.** Missing an escalation is costlier than a false alarm, so the
  decision threshold is tuned on validation to a **minimum-recall = 0.80** target
  rather than to maximise F1. Tags are excluded (leakage), which lowers precision
  but keeps the model honest.
- **M5 — routing.** Summary-only (structured fields' link to the team drifts month
  to month); `operations` is merged into `support`; low-confidence suggestions are
  gated to human review.

## Helper tools: Similarity Seach and Tagging

- **Similar-case search** (`train_similar_cases.ipynb` → `models/similar_cases.joblib`).
  TF-IDF + cosine nearest neighbours over all historical summaries, keeping each
  case's resolution fields. At triage the app shows the most similar past cases and
  how they were resolved. No held-out metrics (retrieval, not classification).
- **Tag suggester** (`train_tag_suggester.ipynb` → `models/tag_suggester.joblib`).
  A multi-label one-vs-rest logistic regression that suggests tags from the summary
  (131 tags). **Its output is shown to the agent only; tags are never used as model
  features** (see the leakage note above).

## Contributors

The modelling approaches were contributed by several teammates during a hackathon
and are anonymised here as Source A–F.
