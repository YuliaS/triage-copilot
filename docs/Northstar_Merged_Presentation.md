---
marp: true
theme: default
paginate: true
size: 16:9
---

# Northstar Desk — AI-Assisted Triage Copilot

### Human-in-the-loop decision support for frontline support agents

**Goal:** classify new support cases, highlight potential risk, find similar historical cases, and help agents act consistently and quickly.

> Classification: Restricted - External

---

# 1. Problem Statement

For every new support case, the agent needs to quickly answer four questions:

1. **What is this case?**
2. **How urgent or risky is it?**
3. **Who should own it?**
4. **Have we seen something similar before?**

### User
**Frontline support agent**

### Problem
Difficult to triage and act on new cases consistently.

### Solution
**AI-assisted classification + risk flagging + routing + similar-case retrieval**

### Value
- **Customer value:** better-informed, potentially faster handling
- **Business value:** more consistent routing and better reuse of historical case knowledge

---

# 2. Product Vision

## Northstar Triage Copilot

A decision-support tool that turns a new case into a reviewable recommendation.

```text
New case
   ↓
Classification
   ↓
Priority / risk review
   ↓
Suggested route
   ↓
Similar solved cases
   ↓
Human decision
```

The system does **not** replace the support agent.

It provides:
- predictions
- confidence
- explanations
- risk flags
- suggested tags
- relevant historical evidence

The **agent accepts or overrides** the recommendation.

---

# 3. Users

## Primary user — Priya, frontline support agent

Priya opens each new case cold. Within seconds, she needs to know:

- what kind of case it is
- which category it belongs to
- whether the priority looks reasonable
- whether it may escalate
- which team should own it
- how similar cases were handled before

Priya is not technical and will not interpret a confusion matrix during normal case handling.

She trusts the tool only when it is **fast, reviewable, and explainable**.

## Secondary user — Sam, team lead / reviewer

Sam wants to know:

- how accurate each model is
- where each model fails
- what features are used
- whether there is leakage
- whether low-confidence cases are sent to human review

---

# 4. Priya's User Journey

```text
1. PASTE
Case summary + intake fields
        ↓
2. TRIAGE
Run the models
        ↓
3. REVIEW
Type • Category • Priority • Route • Risk
        ↓
4. COMPARE
Tags + similar historical cases
        ↓
5. ACT
Accept / override / route
```

### Case inputs
- case summary
- channel
- plan
- month
- customer tenure
- optional agent-selected priority

### What Priya sees
One concise recommendation panel with:
- prediction
- confidence
- plain-language "Why"
- review flags
- historical evidence

---

# 5. Sam's User Journey

```text
Model cards
    ↓
Metrics table
    ↓
Held-out case inspector
    ↓
Confusion matrices
    ↓
Sign-off / review
```

All evaluation should be performed on data the models did **not** train on.

### Success for Sam
The system can be audited without turning the frontline workflow into a data-science interface.

One application serves two audiences:
- **Priya:** action at a glance
- **Sam:** evidence and model transparency

---

# 6. Application Architecture

## Runtime

```text
Gradio case-input interface
          ↓
      Orchestrator
          ↓
 ┌────────┼────────┬────────┬────────┬────────┐
 ↓        ↓        ↓        ↓        ↓        ↓
M1       M2       M3      M3b       M4       M5
Type   Category Priority Review  Escalation  Route

          ↓
Helper components
- Tag suggester
- Similar-case retriever
          ↓
Agent-facing outputs
```

### Agent-facing outputs
- predictions
- confidence
- explanations
- risk flags
- suggested tags
- similar historical cases

---

# 7. Offline Training Architecture

```text
Monthly CSV exports
        ↓
Data cleaning
        ↓
Feature preparation
        ↓
Train / validation / test split
        ↓
Model training
        ↓
Evaluation
        ↓
Saved model artefacts
        ↓
Runtime Gradio app
```

### Guardrail
Outcome and history fields can be used for **training labels and evaluation**, but not as runtime prediction inputs.

---

# 8. Data Preparation

### Source
12 monthly CSV exports containing about **1.85k rows** in a fictional, synthetic dataset.

Each row contains:
- structured triage fields
- outcomes
- timings
- free-text case summary
- tags

### Cleaning
- spam and duplicate rows removed
- final dataset: **1,713 cases**
- missing summaries and tags filled with empty strings

### Dates
`created_at` is treated as the authoritative date.

The monthly filenames are shifted by one month and February is missing, so:

```text
month_num = month(created_at)
```

### Team labels
`operations` and `support` are merged because they represent the same team under different labels.

### Priority
Ordered scale:

```text
Low < Medium < High < Urgent
```

---

# 9. Modelling Discipline

## Inbound-only feature contract

Models use only information available when the case arrives.

### Never use as runtime model inputs
- status
- resolution code
- resolution time
- escalation outcome
- CSAT
- other downstream outcomes

## Tags are leakage

Historical tags are written after an agent has interpreted a case.

Therefore:

```text
Historical tags → predictive model
```

is **not allowed**.

Instead:

```text
Case summary → tag suggester → agent confirms
```

Tags become a workflow aid, not a predictive input.

---

# 10. Model Selection Criteria

We chose models based on **fit for the problem**, not model complexity.

| Criterion | Why it matters |
|---|---|
| Dataset size | ~1.7k cases is small for deep learning |
| Feature type | Case summaries create high-dimensional sparse text features |
| Interpretability | Agents and reviewers need understandable outputs |
| Probability output | Confidence is needed for review thresholds |
| Calibration / thresholding | Risk models need controllable operating points |
| Speed | Models should train and run quickly |
| Deployment simplicity | Small `.joblib` pipelines are easy to serve |
| Leakage resistance | Input contract must be explicit |
| Temporal robustness | Features should generalise to later months |
| Human review | Low-confidence predictions should be reviewable |

### Resulting modelling philosophy

> Prefer the **simplest model that performs well enough, produces useful confidence, and can be explained to a reviewer.**

---

# 11. Why TF-IDF?

TF-IDF converts text into numerical features based on how informative words or character patterns are.

### Word TF-IDF captures
- "charged twice"
- "password reset"
- "stopped syncing"
- "server error"

### Character TF-IDF helps with
- spelling variation
- technical terms
- error codes
- product names
- short fragments such as `401`, `OAuth`, `SSO`

### Why it fits Northstar
Support tickets contain highly specific language, and the dataset is small enough that sparse linear text models are a strong baseline.

---

# 12. Why Logistic Regression?

Logistic regression is used repeatedly because it fits the data shape:

```text
~1,713 cases × thousands of sparse text features
```

### Advantages
- strong on sparse TF-IDF features
- works well on small datasets
- fast to train
- easy to deploy
- supports `predict_proba()`
- coefficients can be inspected
- supports class weighting
- naturally supports confidence-based review

### Why not default to neural networks / LLM classifiers?
They would add:
- more infrastructure
- more tuning
- less interpretability
- greater risk of overfitting

without guaranteed improvement on ~1.7k labelled cases.

---

# 13. Model 1 — Case Type Classification

## Purpose
Predict whether a case is:

- **Incident** — something has gone wrong
- **Service request** — the customer requests help or action

## Method

```text
Case summary
    ↓
Word + character TF-IDF
    ↓
Logistic regression
    ↓
Case type + confidence
```

### Split
80/20 **group-aware** split so identical summaries cannot appear in both train and test.

### Performance
- **Accuracy:** 87.5%
- **Macro-F1:** 0.875

### Why this model
Binary text classification has strong lexical signals and does not require a more complex model.

---

# 14. Model 2 — Category Prediction

## Target
Predict one of **10 support categories**, including:

- `billing_payments`
- `access_permissions`
- `integrations`
- `bug`
- `performance_outage`

## Pipeline

```text
Case summary
    ↓
Word TF-IDF + character TF-IDF
    ↓
Logistic regression
    ↓
Predicted category + confidence
```

### Performance

| Metric | C = 1.0 | C = 2.0 |
|---|---:|---:|
| Accuracy | 90.1% | **90.4%** |
| Macro-F1 | 0.890 | **0.892** |
| Weighted-F1 | 0.902 | **0.905** |
| Cohen's kappa | 0.889 | **0.892** |
| Confidence gap | 0.240 | **0.271** |

### Main confusion
`bug` vs `data_reporting`

### Why chosen
Strong multiclass performance, clear probabilities, and explainable text features.

---

# 15. Model 3 — Direct Priority Prediction

## Target
Predict:

```text
Low / Medium / High / Urgent
```

## Features
- customer tenure
- plan tier
- channel
- case summary

## Candidate models
- Logistic Regression
- Random Forest
- LinearSVC

### Selection
**Logistic Regression**

Random Forest result shown: **63.58%**

Hyperparameter tuning used **Optuna** with five-fold cross-validation.

### Result
**Accuracy improved from 64.45% to 71.34%.**

### Why Logistic Regression
- handles sparse TF-IDF text well
- combines text and structured features cleanly
- produces probabilities
- simpler to review and deploy than alternatives

---

# 16. Model 3b — Priority Review

Exact priority is noisy, so the second priority model solves a different problem:

> **Does the agent-selected priority look materially inconsistent with the case?**

## Pipeline

```text
Summary + Case type + Channel
            ↓
        Word TF-IDF
            ↓
    Logistic regression
            ↓
    Estimated priority
            ↓
Compare with agent priority
            ↓
Flag if difference ≥ 2 levels
```

### Excluded
- SLA hours — derived from priority and may leak the answer
- plan tier — excluded for fairness in this review model

### Performance

| Metric | Result |
|---|---:|
| Exact accuracy | 44.4% |
| Macro-F1 | 0.502 |
| Within one level | **92.6%** |
| Quadratic kappa | 0.313 |
| Flag rate | 7.4% |

### Why this design is useful
The model is rarely far from the recorded priority, even when exact accuracy is limited.

So it is better used as a **review flag** than an automatic priority setter.

---

# 17. Model 4 — Escalation Risk

## Purpose
Identify cases that may be more likely to escalate.

This is a **decision-support signal**, not an automated escalation decision.

## Predictors
- case summary text
- customer tenure
- category
- channel

## Model
TF-IDF + Logistic Regression

Best result shown with:

```text
C = 3.0
```

### Reported performance
- **Accuracy:** 92.2%
- **ROC-AUC:** 0.955
- **PR-AUC:** 0.744

### Why this model
- probabilities are essential for risk scoring
- coefficients help explain risk drivers
- threshold can be tuned to business cost

### Operating principle
Escalation is tuned to catch at least **80% of escalations** because missing a genuine escalation is more costly than generating some extra review.

> **Source consistency note:** the source deck states a 70% / 15% / 5% split, which does not sum to 100%. Verify the intended split before final submission.

---

# 18. Model 5 — Suggested Route

## Target
Predict the assigned team.

`operations` is merged into `support`.

## Inputs
**Case summary only**

## Pipeline

```text
Case summary
    ↓
Word + character TF-IDF
    ↓
Class-balanced logistic regression
    ↓
Top team + second-best team + confidence
```

### Split
Chronological:
- train: January–September
- test: October–December
- held-out set: 450 cases

### Results

| Metric | Held-out | Cross-validation |
|---|---:|---:|
| Accuracy | 0.77 | 0.86 |
| Macro-F1 | 0.72 | 0.81 |
| Top-2 accuracy | **0.91** | **0.97** |

### Review threshold
If top confidence is below **0.5**, send to human review.

### Why summary-only?
Channel, plan tier, and tenure were tested but their relationship with team assignment shifted over time and reduced later-month performance.

Summary text was more temporally stable.

---

# 19. Why Class-Balanced Logistic Regression for Routing?

Routing classes are not equally frequent.

Without class balancing, the model may over-predict the largest team.

```text
Common team     ███████████████
Rare team       ███
```

Class weighting gives rarer teams more influence during training.

### Why macro-F1 matters
Accuracy can hide weak performance on rare teams.

Macro-F1 weights each team equally, making it a better check of routing quality across the full operation.

### Why top-2 accuracy matters
Routing is a recommendation problem.

Showing:

```text
1. Engineering — 54%
2. Support     — 37%
```

is more useful than pretending one team is always certain.

---

# 20. Model 6 — Smart Resolution Recommendation

This component is not a classifier.

It is an **information-retrieval system**.

## Goal
Find historically similar solved cases and surface their outcomes.

## Method

```text
New case summary
       ↓
TF-IDF representation
       ↓
Cosine nearest neighbours
       ↓
Filter to solved cases
       ↓
Similarity threshold
       ↓
Top 3 qualifying cases
```

### Output
Previously solved cases and their resolution codes.

### Example threshold behaviour
At threshold **0.35**:
- 0.502
- 0.480
- 0.456

At threshold **0.60**:
- no qualifying matches

### Why chosen
- dataset is small
- no vector database is required
- retrieval is fast and deterministic
- similarity is understandable
- thresholding prevents obviously weak matches

### Watch-outs
- similar wording does not guarantee the same underlying issue
- thresholds are exploratory
- the feature is not yet integrated into Gradio

---

# 21. Suggested Tags — Helper Component

Historical tags are **not** used as predictive model inputs.

Instead, the app suggests tags from the new case summary.

Example:

```text
"xero connector returns 401 after token refresh"
                ↓
Suggested tags:
xero • api • authentication • sync
```

### Why separate tags from prediction?
Using historical tags as inputs would leak information that is created only after a human understands the ticket.

The tag suggester supports:
- faster handling
- more consistent tagging
- improved workflow

without contaminating the predictive models.

---

# 22. Evaluation Strategy

## Common metrics
- Accuracy
- Macro-F1
- Weighted-F1
- Cohen's kappa
- ROC-AUC for escalation
- PR-AUC for escalation

## Why not accuracy alone?

### Accuracy
Easy to understand, but can hide weak rare-class performance.

### Macro-F1
Treats every class equally.

### Weighted-F1
Reflects average performance weighted by class frequency.

### Cohen's kappa
Corrects observed agreement for agreement expected by chance.

### ROC-AUC
Measures ranking quality across thresholds.

### PR-AUC
Especially important when the positive event is rare.

---

# 23. Task-Specific Evaluation

Different models are used differently, so each needs a signature metric.

| Model | Signature metric | Why |
|---|---|---|
| Case type | Macro-F1 | balanced binary classification |
| Category | Macro-F1 | equal importance across 10 categories |
| Priority | Within-one-level accuracy | priority is ordinal |
| Priority review | Flag rate | measures operational intervention |
| Escalation | Recall / PR-AUC | positive class is rare and important |
| Routing | Top-2 accuracy / review rate | recommendation rather than automation |
| Retrieval | Relevance of top matches | no classification label exists |

### Key principle
Evaluation should reflect **how the model is actually used**.

---

# 24. Model Selection Summary

| Component | Chosen approach | Why |
|---|---|---|
| M1 Case type | TF-IDF + Logistic Regression | strong binary text baseline |
| M2 Category | Word + char TF-IDF + Logistic Regression | robust multiclass text classification |
| M3 Priority | TF-IDF + structured features + Logistic Regression | sparse text + probabilities |
| M3b Priority review | Logistic Regression + ordinal difference rule | safer than asserting exact priority |
| M4 Escalation | TF-IDF + Logistic Regression | probability + threshold control |
| M5 Route | Class-balanced TF-IDF + Logistic Regression | handles imbalance and sparse text |
| M6 Similar cases | TF-IDF + cosine similarity | transparent retrieval on small corpus |
| Tag helper | Rule / text-based suggestion | workflow support without leakage |

---

# 25. Human-in-the-Loop Guardrails

The prototype is designed around **assistance, not automation**.

### Guardrails
- all predictive models use inbound fields only
- protected attributes are not used
- tags are never predictive inputs
- confidence is surfaced
- low-confidence routing is sent to review
- priority review flags large mismatches rather than overwriting the agent
- escalation is a risk signal, not an automatic escalation
- weak similarity matches can return "no match"
- the agent makes the final decision

### Design principle

> **Predict where patterns are repeatable, retrieve where history is useful, and defer to a human when uncertainty is material.**

---

# 26. Frontline Agent Experience

For a pasted case summary and a few intake fields, the tool returns:

- **Case type**
- **Category**
- **Priority estimate**
- **Priority review flag**
- **Suggested route**
- **Escalation risk**
- **Suggested tags**
- **Similar historical solved cases**

Every major output includes:
- confidence
- a reason or explanation
- the option for human review

The goal is to turn a new case into an **action at a glance** without forcing the agent to search the full case history.

---

# 27. Example Agent Experience

### New case

> "Xero connector has stopped syncing and now returns a 401 error after the finance team refreshed the token."

### Copilot response

| Output | Example |
|---|---|
| Case type | Incident |
| Category | Integrations |
| Priority | High |
| Priority review | No large mismatch |
| Route | Engineering |
| Escalation risk | Elevated |
| Tags | xero, api, auth, sync |
| Similar cases | 3 solved integration/authentication cases |

### Human action
Priya reviews the evidence, accepts or overrides the recommendation, and routes the case.

---

# 28. What the Prototype Demonstrates

The prototype shows that Northstar can use its existing operational data to support four kinds of decision:

### Understand
Classify case type and category.

### Prioritise
Estimate seriousness and identify escalation risk.

### Route
Recommend the likely owning team.

### Resolve faster
Retrieve similar solved cases and historical outcomes.

The tool does not need to be "fully autonomous AI" to create value.

---

# 29. Key Limitations

### Data
- only ~1,713 cleaned cases
- synthetic dataset
- some historical labels are inconsistent
- monthly export filenames are unreliable

### Modelling
- simple linear models may miss deeper semantic relationships
- exact priority prediction remains difficult
- routing performance changes over time
- similarity does not prove identical root cause

### Product
- similar-case resolution recommendation is not yet fully integrated into Gradio
- retrieved cases provide historical outcomes, not guaranteed step-by-step fixes
- model confidence is not the same as certainty

---

# 30. Roadmap

## With another week

### Data
- validate date/export anomalies
- improve label consistency
- add richer resolution notes / KB links
- create explicit retrieval relevance labels

### Modelling
- calibrate probabilities
- compare sentence embeddings with TF-IDF retrieval
- investigate drift by month
- improve priority modelling
- run subgroup performance audits

### UX
- add accept / override feedback capture
- surface top contributing terms more clearly
- integrate smart resolution recommendations
- improve model-exploration view

### Governance
- threshold review process
- model cards
- monitoring
- drift alerts
- regular re-evaluation on recent cases

---

# 31. Final Message

Northstar already has useful operational knowledge.

The challenge is that this knowledge is distributed across historical cases.

The Triage Copilot puts that knowledge in front of the agent **at the moment a decision is needed**.

```text
Models suggest
Historical cases provide evidence
Confidence exposes uncertainty
Humans remain accountable
```

### Outcome

**Better-informed, more consistent, potentially faster case handling.**

---

# Appendix A — Exploratory Data Analysis

Suggested visuals:

- number of cases by month
- case volume by category
- case volume by priority
- cases by contact channel
- median resolution time by category
- missing-data summary
- escalation counts
- escalation rate by priority
- escalation rate by category
- resolution outcome frequencies
- average CSAT for escalated vs non-escalated cases

---

# Appendix B — Metric Definitions

### Accuracy
Share of predictions that are correct.

### Baseline accuracy
Accuracy from always predicting the most common class.

### Macro-F1
Unweighted mean of per-class F1.

### Weighted-F1
F1 averaged by class frequency.

### Cohen's kappa
Agreement corrected for chance.

### ROC-AUC
Probability that a random positive case is ranked above a random negative case.

### PR-AUC
Summarises precision-recall trade-off, especially useful for rare positive events.

### Baseline PR-AUC
Positive-class prevalence in the test set.

### Operating-point metrics
Precision and recall at the deployed threshold.

---

# Appendix C — Source Consistency Checks Before Final Submission

The merged deck preserves the reported model results, but two source-deck items should be verified before presenting externally:

1. **Escalation split:** the source states **70% train / 15% validation / 5% test**, which sums to 90%.
2. **Escalation PR-AUC:** one model slide reports **0.744**, while the evaluation-metrics section refers to **0.734**.

These should be reconciled against the final notebook / model artefact before sign-off.
