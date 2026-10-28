# Triage Copilot — pitch slides

Two slides: what the tool does for the agent, and the modelling discipline behind it.

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
