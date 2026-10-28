# Model comparison

Six models were contributed independently (anonymised here as **Source A–F**) for
the same case dataset. This note records each one's design and original evaluation
method, and the **normalised** metric set now applied to all of them so they are
comparable. All metrics are held-out.

| Model | Source | Target | Design |
|---|---|---|---|
| M1 Case type | A | `case_type` | summary-only text classifier |
| M2 Category | B | `category` | summary-only text classifier |
| M3 Priority — direct | C | `priority` | direct predictor (uses plan tier + tenure) |
| M3 Priority — review | D | `priority` | double-check: flag ≥2-level mismatches |
| M4 Escalation | E | `escalated` | risk model, recall-targeted threshold |
| M5 Route | F | `assigned_team` | summary-only router + review gate |

## Original evaluation methods (as submitted)

- **A** — classification report + confusion matrix (random split).
- **B** — accuracy / macro-F1 / weighted-F1, majority baseline, classification
  report, confusion matrix, and a **confidence analysis** (mean confidence on
  correct vs incorrect predictions). Group-aware split on the summary.
- **C** — accuracy, classification report, normalised confusion matrix (random split).
- **D** — exact accuracy, macro-F1, **quadratic-weighted kappa**, within-1-level,
  and a **flag rate** for ≥2-level mismatches. Chronological split.
- **E** — ROC-AUC, **PR-AUC**, precision/recall/F1, confusion matrix, and a
  decision threshold tuned to a **minimum-recall target (0.80)** on a validation
  set. Chronological 70/15/15 split.
- **F** — accuracy, a **review-gate rate** (share of cases below 0.5 confidence)
  and the accuracy on the confident remainder. Chronological split.

## Normalised evaluation (applied to all)

`northstar_prep.evaluate()` computes the same core for every model —
**accuracy, macro-F1, weighted-F1, Cohen's kappa**, a majority **baseline**, plus
**ROC-AUC and PR-AUC** for the binary model — and every training notebook also
stores a **classification report** and **confusion matrix**. Each model keeps its
distinctive *signature* metric as an extra (confidence gap for B; within-1-level
and quadratic kappa for C/D; recall/precision at the tuned threshold for E;
review-gate rate for F). Splits are **not** forced to match: text classifiers use a
group-aware split, sequential models use a chronological split, because that is the
honest choice for each; the split is recorded alongside the metrics.

## Held-out results

| Model | split | accuracy | macro-F1 | weighted-F1 | kappa | other |
|---|---|---|---|---|---|---|
| M1 Case type | group | 0.875 | 0.875 | 0.875 | 0.750 | — |
| M2 Category | group | 0.901 | 0.890 | 0.902 | 0.889 | confidence gap ≈ 0.19 |
| M3 Priority — direct | time | 0.553 | 0.587 | 0.550 | 0.379 | within-1 ≈ 0.93 |
| M3 Priority — review | time | 0.444 | 0.502 | 0.440 | 0.241 | within-1 ≈ 0.93, flag ≈ 0.07 |
| M5 Route | time | 0.848 | 0.805 | 0.854 | 0.776 | needs-review ≈ 0.16 |
| M4 Escalation | time | 0.946 | 0.819 | 0.949 | 0.638 | ROC-AUC 0.961 · PR-AUC 0.783 |

## Findings

1. **Honest splits lower the numbers but tell the truth.** Text models use a
   group-aware split (identical summaries cannot leak across train/test);
   sequential models use a chronological split (train early months, score later
   months). Both are stricter than a plain random split.

2. **Two priority models, one lesson.** The direct predictor (Source C) beats the
   double-check estimator (Source D) by ~0.08 macro-F1 on the same chronological
   split — the gain comes from `plan_tier`/`tenure`, which Source D omits as proxies.
   But **both** sit near a ~0.5 exact-accuracy ceiling while being ~93% within one
   level. That is the case for Source D's design: don't assert an exact priority,
   just flag the gross mismatches.

3. **Escalation is a recall problem.** Source E tunes its threshold to catch at
   least 80% of escalations rather than to maximise F1, trading precision for the
   cheaper error. ROC-AUC 0.961 / PR-AUC 0.783 on unseen future months.

4. **Routing drifts.** Source F deliberately uses the summary only and gates
   low-confidence suggestions to human review (~16% of cases), because the link
   between structured fields and team shifts month to month.

5. **Category is tag-free here.** M2 reads only the summary, so its 0.89 macro-F1
   is honest; a naive model that also reads `tags` scores higher but leaks, because
   tags often name the category.

## Fairness

`plan_tier` and `region_uk` are excluded from every model except the direct
priority predictor (Source C), which keeps `plan_tier`/`tenure` to show the
trade-off. Protected attributes (gender, age) are never used.
