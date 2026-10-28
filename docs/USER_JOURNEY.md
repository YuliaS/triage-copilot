# Triage Copilot — User Journey

A design document for the Gradio application that serves the predictive models
(case type, category, two priority models, escalation risk, suggested route) plus
two helper tools (tag suggestions, similar-case search) behind one decision-support
app.

---

## 1. Who uses it

**Primary user — frontline support agent ("Priya").**
Opens each new case cold. Needs, in a few seconds: what kind of case this is, which
team should own it, whether it is likely to escalate, and how similar cases were
handled before. Not technical; will not read a confusion matrix; trusts the tool
only if it explains itself.

**Secondary user — team lead / reviewer ("Sam").**
Wants to judge whether the tool is safe to rely on: how accurate is each model,
where it fails, and what it uses to decide. Looks at numbers and at individual
predictions against known outcomes.

These two needs map to the two tabs.

---

## 2. The job to be done

> "A new case just landed. Tell me what it is, where it goes, and how risky it is —
> show me why, suggest the tags, and show me how similar cases were resolved — so I
> can act in one glance instead of reading the whole history and guessing."

Success = the agent accepts or overrides the suggestion in one action, and the
suggestion is right often enough that overriding is the exception.

---

## 3. Tab 1 — Triage a new case (the daily screen)

**Layout.** Two columns. Left: the case inputs. Right: the result card, suggested
tags, and similar past cases.

**Journey.**
1. Priya pastes the case summary and sets the few structured fields known at intake
   (channel, plan tier, month, tenure). She does **not** enter tags — tags are
   leakage (written after a case is understood), so the tool suggests them instead.
   Optionally she records the priority she would assign, to trigger a review check.
2. She presses **Triage case**.
3. The result card returns as label → value rows:

   | | |
   |---|---|
   | **Case type** | Incident — 94% |
   | **Category** | Integrations — 90% |
   | **Priority · direct** | Urgent — 61% |
   | **Priority · review** | Urgent — 42% |
   | **Suggested route** | Engineering |
   | **Risk** | High escalation risk — 92% |
   | **Why** | error, token, xero in summary + incident language + flagged above the escalation threshold |

4. Below the card, **suggested tags** appear as chips (clearly labelled a workflow
   aid, not model inputs), and the **most similar past cases** are listed with their
   team, resolution code, resolution time and status — the "what worked before" aid.
5. Risk is colour-coded (green / amber / red). The **Why** row names the real
   signals: top summary terms, case-type language, and the escalation decision.

**Two priority models.** The *direct* model predicts a priority; the *review* model
estimates one to **flag** cases where the agent is ≥2 levels off. The card shows both
and marks when they disagree.

**Prefill.** A **Load example** button fills every input with a realistic case so a
first-time user or a demo audience sees a full result in one click.

**Design principles.** One card, one glance; plain-language phrases, not class codes;
confidence as a percentage beside each label; risk is the only strong colour.

---

## 4. Tab 2 — Explore the models (the trust screen)

Reached from the triage tab via the **Explore how the models work →** button.

- **Model cards** — one per model: approach, inputs, split, the normalised held-out
  metrics, and each model's signature metric and caveats.
- **Normalised metrics table** — the same metric set across all models, side by side.
- **Inspect a held-out case** — pick a model, then a case it never trained on, and
  see predicted vs actual with a ✓/✗ marker.
- **Confusion matrix & per-class report** — per model, to see where errors cluster.
- **Training notebooks** — a download per model.

Everything on this tab is computed on held-out data and says which split it used.

---

## 5. Feature contract (shared)

All models use **inbound fields only**. **Tags are never an input** (leakage);
`plan_tier`/`region_uk` are excluded except the direct priority model; protected
attributes are never used. See `MODELS.md`.

---

## 6. What "good" looks like end-to-end

Priya pastes a case → presses Triage → reads the card, confirms the suggested tags,
and glances at how two similar cases were resolved → routes the case. If she doubts
it, she opens the exploration tab, finds a held-out case like hers, and sees the
model was right on that one. Sam, before sign-off, reads the metrics table and the
confusion matrices. One app, two audiences, no black box.
