# Northstar Triage Copilot — pitch slides

Two slides on the modelling approach: how the routing model was trained, and what auto-tagging adds.

---

## Slide 1 — How the routing model was trained

**Goal:** given a new case, predict which of the 6 teams it should go to (`assigned_team`).

**Data & split**
- 1,713 cases after cleaning (de-duplicated, column-shift records repaired, spam removed).
- 80 / 20 stratified split → trained on 1,370, evaluated on **343 cases the model never saw**.

**Feature contract — inbound only (no leakage)**
- Uses only fields known at case creation: the **summary text**, `channel`, `plan_tier`, `region_uk`, `sla_target_hours`, `customer_tenure_months`, `month`.
- Excludes outcome/history fields (resolution time, status, CSAT) and protected attributes (gender, age) — the latter kept for audit only.

**Representation & model**
- Summary → TF-IDF of word **and character** n-grams (character grams catch product names, error codes, and typos).
- Categoricals one-hot encoded; numerics scaled. Linear classifier, class-weighted for the team imbalance.
- Scored with **macro-F1** (fair across uneven class sizes), compared to a majority-class baseline.

**Result on the 343 unseen cases**
- Majority baseline: macro-F1 0.34.
- **Summary-only model: macro-F1 ≈ 0.75, accuracy ≈ 0.78.**

---

## Slide 2 — What auto-tagging adds

**The tag signal.** Historic cases carry tags (e.g. `xero; invoice; vat`). Adding tags lifts routing to **macro-F1 0.79 / accuracy 0.81** — the single strongest feature. But tags are only legitimate if known *before* routing, so using raw historic tags risks leakage.

**The auto-tagger.** A multi-label model predicts ~4 tags from the summary (131 tags; **precision@3 ≈ 0.78**).

**Key finding — auto-tags are not a free accuracy boost.**
- Tags generated automatically from the summary, fed straight into routing, add almost nothing: **macro-F1 0.69 ≈ the no-tag 0.68**.
- Reason: the router already reads the summary, so summary-derived tags are redundant. The 0.79 comes from human/system knowledge in real tags that isn't in the raw text.

**What auto-tagging actually adds**
1. **Agent assist:** suggests tags, the agent **confirms in one click** — faster, consistent tagging; fixes the messy-tag data quality.
2. **Legitimises the high-accuracy path:** once a human confirms the tags at triage, they are valid inbound data, so the **0.79 router is usable without leakage**.
3. **Compounding value:** each confirmation is a fresh training label (active learning) that improves both models over time.

**One-line takeaway:** auto-tagging's value is workflow, data quality, and unlocking the 0.79 router legitimately — not a free jump in accuracy.

---

## Speaker notes

**Slide 1 (≈60s).**
"We frame routing as a prediction problem: new case in, team out. The discipline that matters is the feature contract — we only let the model use what an agent actually has when a case lands, never outcome fields, so the numbers are honest. We represent the summary with word and character n-grams, which matters because the text is messy — product names, error codes like 500, typos. We report macro-F1 because the teams are imbalanced and accuracy alone would flatter us. On 343 cases the model never saw, the summary-only model reaches about 0.75 macro-F1 against a 0.34 baseline."

**Slide 2 (≈75s).**
"Tags are the strongest signal and push us to ~0.79 — but only if they're known before routing, otherwise it's leakage. So we asked: can we generate tags from the summary? We built a tagger — precision@3 about 0.78. The honest finding is that auto-tags fed straight into routing add almost nothing, because the router already reads the summary; the tags are redundant. The real lift comes from human knowledge in the tags, not from the text. So auto-tagging earns its place three ways: it assists the agent — suggest, confirm in one click; that confirmation makes the tags legitimate inbound data, which unlocks the 0.79 model without leakage; and every confirmation becomes a new training label. The takeaway: auto-tagging is about workflow, data quality, and legitimising the high-accuracy path — not a free accuracy jump."

---

## Numbers reference (held-out 343 unseen cases)

| Routing policy | macro-F1 | accuracy |
|---|---|---|
| Majority baseline | 0.34 | — |
| Summary only (no tags) | ~0.75 | ~0.78 |
| Auto-tags (no confirmation) | 0.69 | 0.74 |
| Confirmed / true tags | 0.79 | 0.81 |

Auto-tagger quality: micro-F1 ≈ 0.71, precision@3 ≈ 0.78 (131 modelled tags).
