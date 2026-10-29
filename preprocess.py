"""Build the modelling dataset from the monthly raw exports.

Reconstructs the cleaning that produced `data/cases_clean.xlsx` directly from
`data/raw/*.csv`, keeping only the columns the models, retriever, tag suggester and
app consume, and writes `data/cases_clean.csv`. `case_pipeline.load_clean` reads this
CSV in preference to the workbook.

Recoverable steps are rebuilt from raw (concatenation with source file, deduplication
of the overlapping monthly snapshots, authoritative `created_at`). The two audit flags
that cannot be recomputed from the raw files — `is_spam` and `data_quality_flag` — are
carried over from the audited workbook. Columns that no model or view uses (PII such as
`gender`/`age_band`/`region_uk`, unused targets such as `subcategory`/`sentiment`,
history/leakage fields, and QA scratch columns) are dropped. `month_num` and `n_tags`
are derived at load time, so they are not stored here.

Run:  python preprocess.py
"""
import os
import glob
import pandas as pd

ROOT  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(ROOT, "data")
RAW   = os.path.join(DATA, "raw")
AUDIT_XLSX  = os.path.join(DATA, "cases_clean.xlsx")
AUDIT_SHEET = "data_after_preprocessing"
OUT_CSV     = os.path.join(DATA, "cases_clean.csv")

# Columns the models, retriever, tag suggester and app actually consume.
KEEP = ["case_id", "created_at", "channel", "case_type", "category", "priority",
        "status", "resolution_code", "resolution_time_hours", "escalated",
        "assigned_team", "customer_tenure_months", "plan_tier", "case_summary",
        "tags", "is_spam"]

# Audit flags that cannot be recomputed from the raw files.
AUDIT_FLAGS = ["is_spam", "data_quality_flag"]


def build() -> pd.DataFrame:
    """Return the lean, cleaned modelling frame (one row per case)."""
    files = sorted(glob.glob(os.path.join(RAW, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no raw CSVs under {RAW}")
    raw = pd.concat([pd.read_csv(f).assign(_file="data/" + os.path.basename(f)) for f in files],
                    ignore_index=True)

    # Authoritative date: created_at (the raw filenames are month-shifted).
    raw["created_at"] = pd.to_datetime(raw["created_at"], errors="coerce", utc=True)

    # Deduplicate the overlapping monthly snapshots by case_id, keeping the most
    # recent created_at (the Nov/Dec exports repeat ~135 cases; the later row wins).
    raw = (raw.sort_values("created_at")
              .drop_duplicates("case_id", keep="last")
              .reset_index(drop=True))

    # Carry the audit flags that are not recoverable from raw.
    audit = pd.read_excel(AUDIT_XLSX, sheet_name=AUDIT_SHEET)
    df = raw.merge(audit[["case_id"] + AUDIT_FLAGS], on="case_id", how="left")
    df["is_spam"] = df["is_spam"].fillna(False).astype(bool)

    # The column-shift repair rewrote the text cells (case_summary, tags) on the
    # flagged rows; those corrected values cannot be recovered from raw, so take them
    # from the audited export.
    repaired = df["data_quality_flag"].astype(str) == "col_shift_repaired"
    fixed = audit.set_index("case_id")
    for col in ("case_summary", "tags"):
        df.loc[repaired, col] = df.loc[repaired, "case_id"].map(fixed[col])

    out = df[KEEP].copy()
    out["escalated"] = out["escalated"].astype(bool)

    # Preserve the audited row order so order-sensitive splits (GroupShuffleSplit,
    # train_test_split) assign exactly the same rows to train/test as before.
    out = out.set_index("case_id").loc[audit["case_id"]].reset_index()
    return out


if __name__ == "__main__":
    out = build()
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}: {out.shape[0]} rows x {out.shape[1]} cols")
    print("columns:", list(out.columns))
