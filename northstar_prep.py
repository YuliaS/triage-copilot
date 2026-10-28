"""Shared data loading, splitting, feature and evaluation helpers.

Every training notebook imports this one module, so preprocessing, the feature
contract and the *evaluation method* are defined once and applied identically.
Paths are anchored to this file's location, so notebooks run from any directory.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score,
                             precision_score, recall_score)

# ---- paths (anchored to this file, so cwd does not matter) ----------------
ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
SHEET      = "data_after_preprocessing"

PRIORITIES    = ["Low", "Medium", "High", "Urgent"]
PRIORITY_RANK = {p: i for i, p in enumerate(PRIORITIES)}
TEAM_ALIASES  = {"operations": "support"}   # same team, two names across months


def _find(fname):
    for p in (os.path.join(DATA_DIR, fname), os.path.join(DATA_DIR, "raw", fname),
              os.path.join(ROOT, fname)):
        if os.path.exists(p):
            return p
    return os.path.join(DATA_DIR, fname)


def load_clean(merge_team_aliases: bool = False) -> pd.DataFrame:
    """Audited non-spam cases with a reliable month and tag count.

    created_at is authoritative (filenames are month-shifted; February is absent).
    """
    csv = _find("data_after_preprocessing.csv")
    df = (pd.read_csv(csv) if os.path.exists(csv)
          else pd.read_excel(_find("data_after_preprocessing_variables.xlsx"), sheet_name=SHEET))
    if "is_spam" in df.columns:
        df = df[~df["is_spam"].fillna(False).astype(bool)].reset_index(drop=True)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["month_num"]  = df["created_at"].dt.month
    for c in ("case_summary", "tags"):
        df[c] = df[c].fillna("").astype(str)
    if "n_tags" not in df.columns:
        df["n_tags"] = df["tags"].apply(lambda s: 0 if not s.strip() else len(s.split(";")))
    df["n_tags"] = pd.to_numeric(df["n_tags"], errors="coerce").fillna(0).astype(int)
    if merge_team_aliases and "assigned_team" in df.columns:
        df["assigned_team"] = df["assigned_team"].replace(TEAM_ALIASES)
    return df.reset_index(drop=True)


# ---- splits ---------------------------------------------------------------
def time_split(df, train_frac=0.70, val_frac=0.15):
    """Chronological train / validation / test (honest for future-month scoring)."""
    d = df.sort_values("created_at").reset_index(drop=True)
    n = len(d); a = int(n*train_frac); b = int(n*(train_frac+val_frac))
    return d.iloc[:a].copy(), d.iloc[a:b].copy(), d.iloc[b:].copy()


def group_split(df, group_col="case_summary", test_size=0.20, seed=42):
    """Group-aware split: identical summaries never straddle train and test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df[group_col]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


# ---- feature transformers (column-wise, all picklable) --------------------
def t_word(col="case_summary"):
    return (f"{col}_w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), col)
def t_char(col="case_summary"):
    return (f"{col}_c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True), col)
def t_tags(col="tags"):
    return (f"{col}_t", TfidfVectorizer(token_pattern=r"[A-Za-z0-9_]+", min_df=2, binary=True), col)
def t_cat(cols):
    return ("cat", OneHotEncoder(handle_unknown="ignore"), list(cols))
def t_num(cols):
    return ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler(with_mean=False))]), list(cols))


def make_clf(transformers, C=1.0):
    """Column transformer + class-balanced logistic regression (shared estimator)."""
    return Pipeline([("prep", ColumnTransformer(transformers)),
                     ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", C=C))])


# ---- normalised evaluation (same method for every model) ------------------
def evaluate(y_true, y_pred, labels, proba=None, positive=None, baseline=None):
    """One normalised metric set applied to every model."""
    m = {"n_test": int(len(y_true)),
         "accuracy":    round(accuracy_score(y_true, y_pred), 3),
         "macro_F1":    round(f1_score(y_true, y_pred, average="macro"), 3),
         "weighted_F1": round(f1_score(y_true, y_pred, average="weighted"), 3),
         "cohen_kappa": round(cohen_kappa_score(y_true, y_pred), 3)}
    if baseline is not None:
        m["baseline_acc"] = round(float(baseline), 3)
    if proba is not None and positive is not None:
        yb = (np.asarray(y_true) == positive).astype(int)
        m["ROC_AUC"] = round(roc_auc_score(yb, proba), 3)
        m["PR_AUC"]  = round(average_precision_score(yb, proba), 3)
    return m


def class_report(y_true, y_pred, labels):
    """Per-class precision/recall/F1 as a dict (for the exploration tab)."""
    return classification_report(y_true, y_pred, labels=list(labels), output_dict=True, zero_division=0)


def conf_matrix(y_true, y_pred, labels):
    return {"labels": list(labels),
            "matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist()}


def ordinal_extras(y_true, y_pred, order):
    """within-1-level and quadratic-weighted kappa for ordinal targets (priority)."""
    r = {o: i for i, o in enumerate(order)}
    t = np.array([r[x] for x in y_true]); p = np.array([r[x] for x in y_pred])
    return {"within_1_level": round(float(np.mean(np.abs(t - p) <= 1)), 3),
            "quadratic_kappa": round(cohen_kappa_score(y_true, y_pred, weights="quadratic"), 3)}


def recall_targeted_threshold(y_true_bin, proba, min_recall=0.80):
    """Among thresholds whose recall >= min_recall, return the one with highest precision.

    This matches the escalation source's "minimum recall" operating point (don't miss
    escalations). Falls back to 0.5 if the target is unreachable.
    """
    yb = np.asarray(y_true_bin).astype(int); pr = np.asarray(proba)
    best = (0.5, -1.0)
    for t in np.linspace(0.05, 0.95, 181):
        pred = pr >= t
        tp = int(((pred == 1) & (yb == 1)).sum())
        fn = int(((pred == 0) & (yb == 1)).sum())
        fp = int(((pred == 1) & (yb == 0)).sum())
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        if rec >= min_recall and prec > best[1]:
            best = (float(t), prec)
    return best[0]


def binary_at_threshold(y_true, proba, positive, thr):
    yb = (np.asarray(y_true) == positive).astype(int); pred = (np.asarray(proba) >= thr).astype(int)
    return {"threshold": round(float(thr), 3),
            "recall_pos":    round(recall_score(yb, pred, zero_division=0), 3),
            "precision_pos": round(precision_score(yb, pred, zero_division=0), 3)}


# ---- persistence ----------------------------------------------------------
def save_bundle(name: str, bundle: dict) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(bundle, path)
    return path
