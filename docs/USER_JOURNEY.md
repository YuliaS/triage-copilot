# Northstar Triage Copilot — User Journey

A design document for a single Gradio application that serves five models
(M1 Case type, M2 Category, M3 Priority, M4 Escalation risk, M5 Route-assign)
behind one decision-support screen.

---

## 1. Who uses it

**Primary user — frontline support agent ("Priya").**
Opens each new case cold. Needs, in a few seconds: what kind of case this is,
which team should own it, and whether it is likely to blow up. Not technical;
will not read a confusion matrix; trusts the tool only if it explains itself.

**Secondary user — team lead / reviewer ("Sam").**
Wants to judge whether the tool is safe to rely on: how accurate is each model,
where does it fail, what does it use to decide. Will look at numbers and at
individual predictions against known outcomes.

These two needs map to the two tabs.

---

## 2. The job to be done

> "A new case just landed. Tell me what it is, where it goes, and how risky it
> is — and show me why — so I can act in one glance instead of reading the whole
> history and guessing."

Success = the agent accepts or overrides the suggestion in one action, and the
suggestion is right often enough that overriding is the exception.

---

## 3. Tab 1 — Triage a new case (the daily screen)

**Layout.** Two columns. Left: the case inputs. Right: a single result card.

**Journey.**
1. Priya pastes the case summary and sets the few structured fields she already
   knows at intake (channel, plan tier, region, SLA target, tenure, month).
   Tags are optional — left blank if unknown.
2. She presses **Triage case**.
3. The result card returns, laid out as label → value rows:

   | | |
   |---|---|
   | **Case type** | Incident — 96% confidence |
   | **Category** | Integrations — 91% |
   | **Suggested route** | Engineering |
   | **Risk** | High escalation risk |
   | **Why** | Connector/authentication issue + incident language + historically high-escalation category |

4. Risk is colour-coded (green / amber / red) so the eye lands on it first.
5. The **Why** row names the actual signals behind the call — the top summary
   terms that drove it, the case-type language, and the historical escalation
   rate of the predicted category — so Priya can sanity-check rather than obey.

**Prefill.** A **Load example** button fills every input with a realistic
integrations-incident case, so a first-time user (or a demo audience) can press
one button and immediately see a full, meaningful result. This is the "show me
how to run it" affordance.

**Design principles for this tab.**
- One card, one glance. No tables of probabilities by default.
- Every output is a plain-language phrase, not a class code.
- Confidence shown as a percentage next to the label, not as a separate widget.
- The only strong colour on the screen is the risk band.

---

## 4. Tab 2 — How the models work (the trust screen)

For Sam, and for Priya on the day she stops trusting a suggestion. Three
sections, top to bottom.

**A. Model cards.** One card per model: what it predicts, the classes, the
held-out macro-F1 / accuracy (and ROC-AUC for the binary models), the baseline
it beats, and a one-line honest caveat (e.g. "priority is coupled to the SLA
target"). This is the spec sheet.

**B. Inspect a single case (predicted vs actual).** A dropdown of **held-out
cases the models never trained on**. Selecting one shows, for all five models
side by side, the predicted value, the confidence, the actual recorded value,
and a ✓/✗ match marker. This lets a sceptical user watch the tool be right — and
be wrong — on cases with a known answer.

**C. Run the full held-out set.** A button that scores all five models on the
entire held-out split and returns the metrics table. This is the "don't take my
word for it" button: the headline numbers, reproduced live, on data the models
did not see.

**Design principles for this tab.**
- Separation of concerns: cards = the claim, single case = the intuition, full
  set = the evidence.
- Only ever evaluates on the held-out split, never on training data, and says so.

---

## 5. The five models (shared contract)

All five consume the **same inbound feature set** (the sheet's Y columns):
case summary, tags, channel, plan tier, region, SLA target, tenure, month. No
outcome or history fields are used as inputs, so the numbers are honest. The
only difference between models is the target.

| Model | Predicts | Shown as |
|---|---|---|
| M1 | Case type | "Case type" |
| M2 | Category | "Category" |
| M3 | Priority | (feeds risk reasoning) |
| M4 | Escalation risk | "Risk" band |
| M5 | Assigned team | "Suggested route" |

---

## 6. What "good" looks like end-to-end

Priya pastes a case → presses Triage → reads five fields and one sentence of
reasoning → routes the case. If she doubts it, she switches to Tab 2, finds a
similar past case, and sees the model was right on that one. Sam, before
sign-off, presses **Run full held-out set** and reads the table. One app, two
audiences, no black box.
