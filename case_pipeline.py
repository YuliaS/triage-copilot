"""Shared case-data pipeline: loading, splitting, features, evaluation, retrieval.

Every training notebook and the app import this one module, so preprocessing, the
feature contract and the evaluation method are defined once and applied identically.
Paths are anchored to this file's location, so notebooks run from any directory.

TAGS ARE LEAKAGE
----------------
The historical `tags` column is written by agents *after* they have understood a
case, so it already encodes the very fields the models predict (category, team,
priority, escalation). No predictive model in this project uses `tags` (or the
derived `n_tags`) as a feature. Tags are only *suggested* to the agent as a
workflow aid (see `fit_tag_suggester` / `suggest_tags`) and are never fed back into
M1–M5.
"""
from __future__ import annotations
import os
from collections import Counter

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score,
                             precision_score, recall_score)

# --- paths: anchored to this file so the working directory does not matter ----
ROOT       = os.path.dirname(os.path.abspath(__file__))   # repo root
DATA_DIR   = os.path.join(ROOT, "data")                   # cleaned data + raw/
MODELS_DIR = os.path.join(ROOT, "models")                 # saved model bundles
DATA_FILE  = "cases_clean.xlsx"                           # the cleaned export we read
SHEET      = "data_after_preprocessing"                   # sheet name inside DATA_FILE

# Ordinal priority scale (low -> high); RANK maps each level to an integer.
PRIORITIES    = ["Low", "Medium", "High", "Urgent"]
PRIORITY_RANK = {level: i for i, level in enumerate(PRIORITIES)}

# Two team names that refer to the same team in different months; merged on load.
TEAM_ALIASES = {"operations": "support"}


def _find(filename: str) -> str:
    """Return the first existing path for `filename` under data/, data/raw/ or root."""
    for candidate in (os.path.join(DATA_DIR, filename),
                      os.path.join(DATA_DIR, "raw", filename),
                      os.path.join(ROOT, filename)):
        if os.path.exists(candidate):
            return candidate
    return os.path.join(DATA_DIR, filename)


def load_clean(merge_team_aliases: bool = False) -> pd.DataFrame:
    """Load the audited, non-spam cases ready for modelling.

    `created_at` is treated as the authoritative date (the source filenames are
    month-shifted and February is missing). A reliable `month_num` and a `n_tags`
    count are derived. `n_tags` is provided for analysis only — it is NOT a model
    feature (it is derived from `tags`, which are leakage).
    """
    csv = _find("cases_clean.csv")
    df = (pd.read_csv(csv) if os.path.exists(csv)
          else pd.read_excel(_find(DATA_FILE), sheet_name=SHEET))

    # Drop spam/junk rows if the cleaning flag is present.
    if "is_spam" in df.columns:
        df = df[~df["is_spam"].fillna(False).astype(bool)].reset_index(drop=True)

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["month_num"]  = df["created_at"].dt.month                      # 1..12, reliable

    for text_col in ("case_summary", "tags"):
        df[text_col] = df[text_col].fillna("").astype(str)

    if "n_tags" not in df.columns:
        df["n_tags"] = df["tags"].apply(lambda s: 0 if not s.strip() else len(s.split(";")))
    df["n_tags"] = pd.to_numeric(df["n_tags"], errors="coerce").fillna(0).astype(int)

    if merge_team_aliases and "assigned_team" in df.columns:
        df["assigned_team"] = df["assigned_team"].replace(TEAM_ALIASES)

    return df.reset_index(drop=True)


# =====================================================================
# Splits — each model uses the split that is honest for its task
# =====================================================================
def time_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """Chronological train / validation / test split.

    Honest for models that will score *future* months: train on the earliest cases,
    validate on the next slice, test on the most recent.
    """
    ordered = df.sort_values("created_at").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))
    return ordered.iloc[:train_end].copy(), ordered.iloc[train_end:val_end].copy(), ordered.iloc[val_end:].copy()


def group_split(df: pd.DataFrame, group_col: str = "case_summary",
                test_size: float = 0.20, seed: int = 42):
    """Group-aware split: identical summaries never appear in both train and test.

    Prevents duplicate summaries from leaking an easy answer across the split.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


# =====================================================================
# Feature transformers — each returns a (name, transformer, columns) triple
# for a ColumnTransformer. All are picklable (no custom functions).
# =====================================================================
def t_word(col: str = "case_summary"):
    """Word 1–2-gram TF-IDF of a text column (captures phrases like 'double charged')."""
    return (f"{col}_w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), col)

def t_char(col: str = "case_summary"):
    """Character 3–5-gram TF-IDF (robust to typos, product names, error codes)."""
    return (f"{col}_c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True), col)

def t_cat(cols):
    """One-hot encode categorical columns; unseen categories at predict time are ignored."""
    return ("cat", OneHotEncoder(handle_unknown="ignore"), list(cols))

def t_num(cols):
    """Impute missing numerics with the median, then scale (sparse-safe)."""
    return ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler(with_mean=False))]), list(cols))


# def make_clf(transformers, C: float = 1.0) -> Pipeline:
def make_clf(transformers, C: float = 3.0) -> Pipeline:
    """Shared estimator: a ColumnTransformer feeding a class-balanced logistic regression."""
    return Pipeline([("features", ColumnTransformer(transformers)),
                     ("model", LogisticRegression(max_iter=4000, class_weight="balanced", C=C))])


# =====================================================================
# Evaluation — one normalised metric set applied to every classifier
# =====================================================================
def evaluate(y_true, y_pred, labels, proba=None, positive=None, baseline=None) -> dict:
    """Normalised core metrics shared by all models.

    y_true/y_pred : true and predicted labels on the held-out set.
    labels        : the full class list (for consistent ordering).
    proba         : probability of the positive class (binary models only).
    positive      : the positive label (binary models only) -> enables ROC/PR-AUC.
    baseline      : majority-class accuracy, for context.
    """
    metrics = {
        "n_test":      int(len(y_true)),
        "accuracy":    round(accuracy_score(y_true, y_pred), 3),
        "macro_F1":    round(f1_score(y_true, y_pred, average="macro"), 3),
        "weighted_F1": round(f1_score(y_true, y_pred, average="weighted"), 3),
        "cohen_kappa": round(cohen_kappa_score(y_true, y_pred), 3),
    }
    if baseline is not None:
        metrics["baseline_acc"] = round(float(baseline), 3)
    if proba is not None and positive is not None:
        y_bin = (np.asarray(y_true) == positive).astype(int)
        metrics["ROC_AUC"] = round(roc_auc_score(y_bin, proba), 3)
        metrics["PR_AUC"]  = round(average_precision_score(y_bin, proba), 3)
    return metrics


def class_report(y_true, y_pred, labels) -> dict:
    """Per-class precision / recall / F1 as a dict (shown on the exploration tab)."""
    return classification_report(y_true, y_pred, labels=list(labels), output_dict=True, zero_division=0)


def conf_matrix(y_true, y_pred, labels) -> dict:
    """Confusion matrix as {labels, matrix} (rows = actual, columns = predicted)."""
    return {"labels": list(labels),
            "matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist()}


def ordinal_extras(y_true, y_pred, order) -> dict:
    """Ordinal metrics for the priority target: within-1-level and quadratic kappa."""
    rank = {level: i for i, level in enumerate(order)}
    true_rank = np.array([rank[v] for v in y_true])
    pred_rank = np.array([rank[v] for v in y_pred])
    return {"within_1_level":  round(float(np.mean(np.abs(true_rank - pred_rank) <= 1)), 3),
            "quadratic_kappa": round(cohen_kappa_score(y_true, y_pred, weights="quadratic"), 3)}


def recall_targeted_threshold(y_true_bin, proba, min_recall: float = 0.80) -> float:
    """Pick the highest-precision threshold whose recall still meets `min_recall`.

    Used by the escalation model: missing an escalation is costlier than a false
    alarm, so we fix a recall floor instead of maximising F1. Falls back to 0.5.
    """
    y_bin = np.asarray(y_true_bin).astype(int)
    scores = np.asarray(proba)
    best_threshold, best_precision = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        predicted = scores >= threshold
        tp = int(((predicted == 1) & (y_bin == 1)).sum())
        fn = int(((predicted == 0) & (y_bin == 1)).sum())
        fp = int(((predicted == 1) & (y_bin == 0)).sum())
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        if recall >= min_recall and precision > best_precision:
            best_threshold, best_precision = float(threshold), precision
    return best_threshold


def binary_at_threshold(y_true, proba, positive, threshold) -> dict:
    """Recall and precision of the positive class at a chosen decision threshold."""
    y_bin = (np.asarray(y_true) == positive).astype(int)
    predicted = (np.asarray(proba) >= threshold).astype(int)
    return {"threshold":     round(float(threshold), 3),
            "recall_pos":    round(recall_score(y_bin, predicted, zero_division=0), 3),
            "precision_pos": round(precision_score(y_bin, predicted, zero_division=0), 3)}


# =====================================================================
# Similar-case retrieval (find past cases and how they were resolved)
# =====================================================================
def fit_retriever(df: pd.DataFrame, meta_cols, text_col: str = "case_summary", n_neighbors: int = 10) -> dict:
    """Build a TF-IDF + nearest-neighbour index over historical case summaries.

    meta_cols : columns to keep for each reference case (e.g. resolution fields),
                so the app can show how similar past cases were handled.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    matrix = vectorizer.fit_transform(df[text_col].fillna("").astype(str))
    index = NearestNeighbors(n_neighbors=min(n_neighbors, len(df)), metric="cosine").fit(matrix)
    reference = df[list(meta_cols)].reset_index(drop=True)
    return {"vectorizer": vectorizer, "index": index, "reference": reference, "text_col": text_col}


def query_similar(retriever: dict, summary: str, k: int = 3) -> pd.DataFrame:
    """Return the k most similar historical cases to `summary` with a similarity score."""
    query_vec = retriever["vectorizer"].transform([summary or ""])
    k = min(k, retriever["index"].n_samples_fit_)
    distances, indices = retriever["index"].kneighbors(query_vec, n_neighbors=k)
    neighbours = retriever["reference"].iloc[indices[0]].copy()
    neighbours["similarity"] = (1.0 - distances[0]).round(3)   # cosine distance -> similarity
    return neighbours


# =====================================================================
# Tag suggester (workflow aid only — suggested tags are NEVER model inputs)
# =====================================================================
def fit_tag_suggester(df: pd.DataFrame, text_col: str = "case_summary",
                      tag_col: str = "tags", min_support: int = 10) -> dict:
    """Train a multi-label model that suggests tags for a summary.

    Trained with `tags` as the LABEL and the summary as the input. This does not
    violate the no-leakage rule for M1–M5: the suggester's output is shown to the
    agent, never fed into the predictive models.
    """
    tag_lists = df[tag_col].fillna("").apply(
        lambda s: [t.strip() for t in s.split(";") if t.strip()])

    # Keep only tags seen at least `min_support` times (drop rare, noisy tags).
    counts = Counter(tag for lst in tag_lists for tag in lst)
    kept = sorted(tag for tag, c in counts.items() if c >= min_support)
    tag_lists = tag_lists.apply(lambda lst: [t for t in lst if t in kept])

    binariser = MultiLabelBinarizer(classes=kept)
    Y = binariser.fit_transform(tag_lists)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vectorizer.fit_transform(df[text_col].fillna("").astype(str))

    model = OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X, Y)
    return {"vectorizer": vectorizer, "model": model, "classes": list(binariser.classes_),
            "min_support": min_support}


def suggest_tags(suggester: dict, summary: str, k: int = 4, min_prob: float = 0.15):
    """Return up to k (tag, probability) suggestions above `min_prob`."""
    X = suggester["vectorizer"].transform([summary or ""])
    proba = suggester["model"].predict_proba(X)[0]
    order = proba.argsort()[::-1][:k]
    return [(suggester["classes"][i], float(proba[i])) for i in order if proba[i] >= min_prob]


# =====================================================================
# Persistence
# =====================================================================
def save_bundle(name: str, bundle: dict) -> str:
    """Pickle a model bundle to models/<name>.joblib and return the path."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(bundle, path)
    return path
