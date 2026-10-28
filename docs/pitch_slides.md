# Triage Copilot — pitch slides

What the tool does for the agent, the modelling discipline behind it, the app
architecture, and how the similarity score is computed.

---

## Slide 1 — One screen for the frontline agent

**Goal:** turn a new case into an action in one glance, without reading the whole
history.

For a pasted case summary (plus a few intake fields) the tool returns:
- **Case type** and **category** (text models on the summary).
- **Priority**, two ways: a direct estimate, and a *review check* that flags when the
  agent's own priority is far from the estimate.
- **Suggested route** (team), with a *needs human review* flag on low-confidence calls.
- **Escalation risk**, with a plain-language **Why**.
- **Suggested tags** (a workflow aid) and the most **similar past cases** with how
  they were resolved.

Everything shown is explained, and the agent always makes the final call.

---

## Slide 2 — The modelling discipline

**Inbound-only feature contract.** Models use only what is known when a case lands.
Outcome/history fields are never inputs.

**Tags are leakage.** Tags are written *after* an agent understands a case, so they
encode the very fields we predict. **No model uses tags as a feature.** Instead a
tag *suggester* proposes tags from the summary to speed up and standardise tagging —
its output is shown to the agent, never fed back into the models.

**Honest, normalised evaluation.** Every model is scored with the same metric set
(accuracy, macro-F1, weighted-F1, Cohen's kappa; ROC-AUC/PR-AUC for the binary
model). Text models use a group-aware split (identical summaries can't leak across
train/test); sequential models use a chronological split (train early months, score
later ones). `plan_tier`/`region_uk` are excluded except the direct priority model;
protected attributes are never used.

**Fit-for-purpose choices.**
- Exact priority has a ~50% ceiling but is ~93% within one level — so the review
  model *flags* gross mismatches rather than asserting a value.
- Escalation tunes its threshold to catch ≥80% of escalations, because a miss costs
  more than a false alarm.
- Routing is summary-only and gates low-confidence suggestions to human review.

---

## Slide 3 — App architecture

**Serve-time, not train-time.** The app loads pre-trained bundles from `models/`
and serves predictions; it trains nothing. A shared module (`case_pipeline.py`)
holds the single implementation of data loading, splits, feature transformers,
evaluation, retrieval, and tagging, used by both training and serving.

**Build → bundle → serve.** Each training notebook (`training/train_*.ipynb`)
writes a versioned `.joblib` bundle containing the fitted model plus its held-out
set, metrics, classification report, confusion matrix, and metadata. The app reads
those bundles, so what the agent sees and what the exploration view reports come
from the same artefact.

**Prediction cascade.** For one case summary and the intake fields: case type and
category are predicted first, then feed the priority and escalation models; routing
is predicted in parallel. Tags are never an input. The similar-case retriever and
tag suggester run alongside the classifiers.

**Two UI layers.** An operator dashboard (Input · Output · Similar cases) for daily
triage, and an admin-gated exploration view (model cards, normalised held-out
metrics, confusion matrix and per-class report, a browsable held-out table, and the
training notebooks themselves). A headless launcher (`app/run_app.py`) executes the
notebook and serves it on a local port.

---

## Slide 4 — The similarity score

**Purpose.** Retrieve the past cases most like the new summary and show how they
were resolved — precedent alongside the predictions, not another classifier.

**How it is computed.** Historical summaries are vectorised with TF-IDF (word
unigrams and bigrams, `min_df=2`, sublinear term frequency) and indexed for
nearest-neighbour search under cosine distance. A query summary is vectorised the
same way; the score is `similarity = 1 − cosine distance`, in the range 0–1 and
shown as a percentage.

**What the agent sees.** Every retrieved case at or above **50% similarity** (a
quality threshold, not a fixed count), each with its resolution fields and a
similarity bar; when nothing clears the threshold the panel says so.

**Scope and caveat.** Retrieval has no held-out accuracy metric — relevance is
judged by the agent. An identical past summary scores 100% (a self or duplicate
match), so the score ranks resemblance rather than certifying a correct answer.

---

## Slide 5 — Evaluation metrics, and what each adds

**Accuracy** — the fraction of cases predicted correctly. Intuitive, but inflated
when one class dominates, so it is read against its baseline.

**Baseline accuracy** — the accuracy of always predicting the most common class. It
sets the floor; accuracy only signals skill to the extent it clears this line.

**Macro-F1** — the unweighted mean of per-class F1. It counts every class equally,
so performance on rare classes cannot be hidden by the common ones; it is the
headline metric for the multiclass models.

**Weighted-F1** — F1 averaged by class frequency. It summarises typical-case
performance and sits closer to accuracy, complementing macro-F1's focus on rare
classes.

**Cohen's kappa** — agreement corrected for chance. It discounts the share of
correct predictions expected from guessing with the class frequencies, guarding
against scores that look high only because of imbalance.

**ROC-AUC** (binary escalation model) — the probability the model ranks a random
positive above a random negative, across all thresholds. It measures separability
independently of the chosen cut-off.

**PR-AUC** (average precision) — the precision/recall trade-off, focused on the
positive class. For a rare event such as escalation it is the more honest summary
than ROC-AUC, which heavy negatives can flatter.

**Baseline PR-AUC** — the positive-class prevalence on the test set, i.e. the score
a no-skill model would reach. It is the reference PR-AUC must beat (escalation:
0.734 against a 0.07 baseline).

**Operating-point metrics** — recall and precision of the positive class at the
deployed threshold. These describe behaviour in use rather than across all
thresholds (escalation is tuned to catch at least 80% of escalations).

**Signature metrics** — one task-specific measure per model: *within-1 accuracy*
for the ordinal priority models, the *flag rate* for the review model, and the
*needs-review rate* for routing. Each reports the quantity the model is actually
used for.

---

## Numbers reference (held-out)

| Model | metric | value |
|---|---|---|
| Case type | macro-F1 | 0.875 |
| Category | macro-F1 | 0.890 |
| Priority — direct | macro-F1 / within-1 | 0.587 / 0.93 |
| Priority — review | within-1 / flag rate | 0.93 / 0.07 |
| Suggested route | macro-F1 / needs-review | 0.805 / 0.16 |
| Escalation risk | ROC-AUC / PR-AUC / recall | 0.952 / 0.734 / 0.78 |

Tag suggester: 131 tags, suggestions only (not model inputs). Similar-case search:
TF-IDF + cosine nearest neighbours over all historical cases.
