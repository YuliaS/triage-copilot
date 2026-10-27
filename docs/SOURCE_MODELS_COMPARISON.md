# Source models vs. the copilot models — comparison report

Scope: every artefact in `source_models/` compared against the matching model in
`triage-copilot-5models.ipynb` (M1 Case type, M2 Category, M3 Priority,
M4 Escalation, M5 Route-assign). "Mine" = the copilot model; "theirs" = the
teammate artefact.

## What is in `source_models/`

| File | Author | Target | Maps to |
|---|---|---|---|
| `_x_louis/case_type_model.ipynb` (+ `.pkl`) | Louis | `case_type` | **M1** |
| `Cleaned-DataSet-Category-Model.ipynb` | — | `category` | **M2** |
| `priority_milind.rtf` | Milind | `priority` (double-check) | **M3** |
| `priority_tracy.rtf` | Tracy | `priority` (direct) | **M3** |
| `escalation_binary_anna.rtfd` | Anna | `escalated` | **M4** |
| `christian.zip` (`data.py`, `model.py`) | Christian | `assigned_team` | **M5** |

All six share the same cleaned dataset (1,713 non-spam cases); several files
reproduce the same preprocessing I use (column-shift repair, spam filter,
dedupe, "February missing / `created_at` authoritative" finding).

---

## Headline: the single biggest difference

**My five models all use a random stratified 80/20 split. Four of the six source
models use a time-based (chronological) split, and the category model uses a
group-aware split.** This is a methodology gap, not a tuning detail:

- A random split lets near-duplicate summaries and the same month appear in both
  train and test, which **inflates** reported scores.
- A chronological split (train early months, test later months) measures what the
  tool will actually do in deployment — predict next month's cases — and exposes
  month-to-month concept drift.

Consequence: several of my higher numbers are not directly comparable to theirs.
Where a teammate reports a lower figure on a time split, that figure is usually
the more honest one.

---

## Per-model comparison

### M1 — Case type (vs Louis)
| | Mine (M1) | Louis |
|---|---|---|
| Inputs | summary + tags + channel/plan/region + SLA/tenure/month | **summary only** |
| Model | class-weighted LogReg | LogReg (no class weight) |
| Split | random 80/20 | random 80/20 |
| Reported | macro-F1 0.901 | (not recorded in file) |

Both are TF-IDF (1–2-gram) + logistic regression; case_type is near-balanced so
class weighting barely matters. Louis is the minimal honest version
(summary-only). Mine adds structured fields and tags for a marginal lift. Verdict:
**equivalent core**; Louis is cleaner, mine is slightly richer. Neither has the
leakage concerns of M2/M3.

### M2 — Category (vs the Category notebook)
| | Mine (M2) | Category notebook |
|---|---|---|
| Inputs | summary + **tags** + structured | **summary only** |
| Split | random 80/20 | **GroupShuffleSplit on summary** (no duplicate leaks across split) |
| Reported | macro-F1 **0.950** | macro-F1 **0.795**, acc 0.834 |

This is the most important finding in the report. My 0.950 **leans on `tags`**,
which agents add after intake and which frequently name the category — i.e.
leakage. The notebook's 0.795 is summary-only *and* uses a group-aware split that
stops identical summaries leaking across train/test. **The honest category number
is ~0.80, not 0.95.** Their methodology is stronger; I should either drop tags
from M2 or report the summary-only score alongside it. Their group-split idea
should be adopted across all my models.

### M3 — Priority (vs Milind and Tracy)
| | Mine (M3) | Milind | Tracy |
|---|---|---|---|
| Framing | **direct** predictor | **double-check** (flag if ≥2 levels off) | direct predictor |
| Inputs | summary + tags + **sla_target_hours** + … | summary + case_type + channel | summary + tenure + plan_tier + channel |
| `sla_target_hours` | **used (leakage)** | **excluded (leaks priority)** | excluded |
| Split | random 80/20 | **time split** (Jan–Sep train, Oct–Dec test) | random 80/20 |
| Reported | macro-F1 0.932 | — (optimised for flag recall) | accuracy only |

My 0.932 is **inflated by `sla_target_hours`**, which is derived from priority —
I flagged this as a caveat, Milind removes it entirely. More importantly, Milind
**reframes the problem**: because historic priority labels are inconsistent, no
model predicts exact priority reliably (~50%), so he does not try — he flags only
cases the agent set ≥2 levels away from the model's estimate ("a passenger who
speaks up only when you're clearly heading the wrong way"). That is a better
product decision than my confident top-2 priority. Tracy is a direct predictor
like mine but correctly drops `sla_target_hours`; she keeps `plan_tier`/`tenure`,
which Milind argues are age proxies. **Verdict: Milind's framing is the one to
adopt; my direct 0.93 is not trustworthy.**

### M4 — Escalation (vs Anna)
| | Mine (M4) | Anna |
|---|---|---|
| Inputs | summary + tags + structured | summary (TF-IDF) + tags (multi-hot) + category/subcategory + month + numerics |
| Split | random 80/20 | **chronological 70/15/15** (train/val/test) |
| Threshold | fixed 0.5 (then banded) | **tuned on validation** (0.569) |
| Metrics | macro-F1 0.836, ROC-AUC **0.974** | macro-F1 0.810, ROC-AUC **0.939**, PR-AUC 0.791, escalation-class recall 0.72 |

Both use class-weighted logistic regression and both treat the ~13% escalation
rate as imbalanced. Anna's work is **more rigorous**: a three-way chronological
split, an explicit validation set used to pick the decision threshold, and
PR-AUC + per-class recall (the right metrics for a rare positive). My ROC-AUC
0.974 is higher mainly because a random split is easier than her time split.
**Her 0.939 on unseen future months is the more deployable number.** I should
adopt threshold tuning and PR-AUC reporting.

### M5 — Route-assign (vs Christian)
| | Mine (M5) | Christian |
|---|---|---|
| Inputs | summary + **tags** + channel/plan/region + SLA/tenure/month | **summary only** (words 1–2-gram + char 3–5-gram) |
| Dropped on purpose | — | channel, plan_tier, tenure ("team-link shifted between months; hurt unseen-month routing"), tags (leakage), region (proxy) |
| Split | random 80/20 | **time split** |
| Extras | top-2 + risk band | **review threshold 0.5** ("needs human review"), per-word `explain()` crediting char n-grams |
| Reported | macro-F1 0.785 | accuracy ~0.45 below the review threshold (honest about the hard cases) |

Christian reached the opposite conclusion to me on features: he **removed**
channel/plan/tenure because their relationship to the team drifts across months,
which a random split hides but a time split reveals. His model is summary-only,
has a "needs human review" gate, and a more careful explainer (bigrams split
across their two words; char n-grams credited back to whole words). My M5 scores
higher partly because `tags` and the random split help it. **His is the more
robust routing design; mine is the higher — but softer — number.**

---

## Cross-cutting themes

1. **Leakage discipline.** The source models are consistently stricter than mine.
   They exclude `tags` (agent-added, names the category), `sla_target_hours`
   (derived from priority), and outcome fields. My copilot includes `tags` and
   `sla_target_hours` for better headline numbers, with caveats noted on the
   cards. Their stance is the defensible one.

2. **Validation method.** Time-based and group-aware splits (theirs) beat my
   random split for a tool meant to score future cases. This is the clearest
   thing to fix.

3. **Fairness / proxies.** Christian, Milind and Anna explicitly drop protected
   attributes *and proxies* (`region_uk`, `plan_tier` — noted as an age proxy).
   My copilot still uses `region_uk` and `plan_tier` as model inputs. This is a
   fairness gap in mine.

4. **Problem framing.** Milind's "double-check / flag only gross mismatches" and
   Christian's "needs human review" gate are better decision-support designs than
   my confident top-k outputs — they match the "help humans decide, don't
   replace them" brief.

5. **Where mine is competitive.** A single shared pipeline across five targets,
   one Gradio app with a model-inspection tab, held-out predicted-vs-actual, and
   the plain-language *Why* line. The source models are individually stronger on
   method but are separate scripts/notebooks, not one deployable tool.

---

## Recommended actions (merge the best of both)

1. **Switch to honest splits:** time-based split for M3/M4/M5 (drift matters),
   GroupShuffleSplit-on-summary everywhere else — adopted from the Category
   notebook and Christian/Milind/Anna.
2. **M2:** report the summary-only score (~0.80) as the headline; treat the
   tag-boosted 0.95 as "with confirmed tags only".
3. **M3:** replace the direct predictor with Milind's double-check framing and
   drop `sla_target_hours`.
4. **M4:** add validation-set threshold tuning and PR-AUC, per Anna.
5. **M5:** offer Christian's summary-only time-split variant as the robust
   default; keep the tag-aided version as "confirmed-tags only".
6. **Fairness:** drop `region_uk` and `plan_tier` from the model inputs; keep
   them for audit only.
7. Keep the copilot as the **integration layer** — one app — but back each tab
   with the teammate's more rigorous model where it is stronger.
