"""Shared data loading, splitting and feature helpers for the Northstar copilot.

Every training notebook imports from this one module so the preprocessing and the
feature contract are defined once. The notebooks train and save models; the app
only loads them.
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

SHEET  = "data_after_preprocessing"
MODELS_DIR = "models"

def _find(fname):
    """Look for a data file in data/ first, then the working directory."""
    for p in (os.path.join("data", fname), fname):
        if os.path.exists(p):
            return p
    return fname

XLSX = _find("data_after_preprocessing_variables.xlsx")
CSV  = _find("data_after_preprocessing.csv")      # used if present, else the xlsx sheet

PRIORITIES   = ["Low", "Medium", "High", "Urgent"]
PRIORITY_RANK = {p: i for i, p in enumerate(PRIORITIES)}
# "operations" and "support" are the same team under two names across months (Christian's finding)
TEAM_ALIASES = {"operations": "support"}


def load_clean(merge_team_aliases: bool = False) -> pd.DataFrame:
    """Load the audited non-spam cases with reliable month and tag-count columns.

    created_at is authoritative (filenames are month-shifted; February is absent).
    """
    if os.path.exists(CSV):
        df = pd.read_csv(CSV)
    else:
        df = pd.read_excel(XLSX, sheet_name=SHEET)
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
def time_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    """Chronological train / validation / test split (honest for future-month scoring)."""
    d = df.sort_values("created_at").reset_index(drop=True)
    n = len(d); a = int(n * train_frac); b = int(n * (train_frac + val_frac))
    return d.iloc[:a].copy(), d.iloc[a:b].copy(), d.iloc[b:].copy()


def group_split(df: pd.DataFrame, group_col="case_summary", test_size=0.20, seed=42):
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
    """Column transformer + class-balanced logistic regression."""
    return Pipeline([("prep", ColumnTransformer(transformers)),
                     ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", C=C))])


# ---- persistence ----------------------------------------------------------
def save_bundle(name: str, bundle: dict) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(bundle, path)
    return path
