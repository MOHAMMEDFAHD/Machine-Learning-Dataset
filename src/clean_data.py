"""Reference cleaning solution (instructor copy).

Repairs every defect class that ``inject_defects.py`` puts into the raw files,
using only information a learner legitimately has: the raw file and the data
dictionary. The clean file is used ONLY afterwards, to grade the result.

Design rules this script follows (and that learners should follow too):

* Cleaning turns invalid values into NaN. It does NOT impute. Imputation is a
  learned step and belongs inside a Pipeline, fitted after the train/test split.
* Rules come from the data dictionary (allowed ranges and categories), not from
  peeking at the clean file.
* Every change is written to a repair log, so each decision can be audited.

Usage:
    python src/clean_data.py            # clean both datasets and print a grade
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID_COL = {"healthcare": "patient_id", "retail": "sale_id"}


# --------------------------------------------------------------------------- #
# Dictionary-driven rules
# --------------------------------------------------------------------------- #
def load_rules(name):
    """Read allowed numeric ranges and category sets from the data dictionary."""
    d = pd.read_csv(ROOT / f"data/metadata/{name}_dictionary.csv")
    ranges, categories, integer_cols = {}, {}, []
    for _, row in d.iterrows():
        allowed = str(row["allowed_values_or_range"])
        scale = str(row["measurement_scale"])
        if row["pandas_dtype"] == "int64":
            integer_cols.append(row["name"])
        if scale in ("identifier", "temporal"):
            continue  # ID and date lists in the dictionary are only samples
        match = re.fullmatch(r"\s*(-?[\d.]+)\s+to\s+(-?[\d.]+)\s*", allowed)
        if match:
            ranges[row["name"]] = (float(match.group(1)), float(match.group(2)))
        elif "|" in allowed:
            categories[row["name"]] = set(allowed.split("|"))
    return ranges, categories, integer_cols


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_raw(name):
    """Return (cleaned DataFrame, repair log DataFrame) for one dataset."""
    raw = pd.read_csv(ROOT / f"data/raw/{name}_raw.csv", dtype=str)
    df = raw.copy()
    ranges, categories, integer_cols = load_rules(name)
    id_col = ID_COL[name]
    log = []

    def record(mask, column, action):
        for i in df.index[mask]:
            log.append(dict(row=int(i), record=df.at[i, id_col], column=column,
                            raw_value=raw.at[i, column] if column in raw else "(whole row)",
                            action=action))

    # 1. Text categories: fix formatting, then validate against the dictionary.
    for col, allowed in categories.items():
        if not all(v.replace("_", "").isalpha() for v in allowed):
            continue  # numeric 0/1 flags are handled as numbers below
        stripped = df[col].str.strip().str.lower()
        changed = stripped.ne(df[col]) & df[col].notna()
        record(changed, col, "fix formatting (strip + lower case)")
        df[col] = stripped
        invalid = df[col].notna() & ~df[col].isin(allowed)
        record(invalid, col, "flag: value not in dictionary -> NaN")
        df.loc[invalid, col] = np.nan

    # 2. Numbers: coerce text, then enforce dictionary ranges.
    for col, (low, high) in ranges.items():
        numeric = pd.to_numeric(df[col], errors="coerce")
        not_a_number = df[col].notna() & numeric.isna()
        record(not_a_number, col, "flag: text in numeric column -> NaN")
        out_of_range = numeric.notna() & ~numeric.between(low, high)
        record(out_of_range, col, f"flag: outside dictionary range {low}-{high} -> NaN")
        numeric[out_of_range] = np.nan
        df[col] = numeric

    # 3. Dataset-specific logic.
    if name == "retail":
        # Dates arrive as ISO (YYYY-MM-DD) or day-first (DD/MM/YYYY).
        iso = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        day_first = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        record(iso.isna() & day_first.notna(), "date", "fix: day-first date -> ISO")
        df["date"] = iso.fillna(day_first).dt.strftime("%Y-%m-%d")
        # Returns can never exceed units sold.
        impossible = df["return_count"] > df["units_sold"]
        record(impossible, "return_count", "flag: returns > units sold -> NaN")
        df.loc[impossible, "return_count"] = np.nan

    # 4. Exact duplicate rows: keep the first copy.
    duplicated = df.duplicated(keep="first")
    record(duplicated, "__whole_row__", "drop: exact duplicate row")
    df = df[~duplicated].copy()

    # 5. ID collisions: different records sharing one ID. Keep the first,
    #    give the others a traceable surrogate key (the true ID is unknowable).
    collided = df[id_col].duplicated(keep="first")
    for n, i in enumerate(df.index[collided], start=1):
        log.append(dict(row=int(i), record=df.at[i, id_col], column=id_col,
                        raw_value=raw.at[i, id_col],
                        action="re-key: ID collision -> surrogate key"))
        df.at[i, id_col] = f"{df.at[i, id_col]}-DUP{n}"

    # 6. Restore integer dtypes (nullable, so NaN can stay NaN).
    for col in integer_cols:
        if col in df and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round().astype("Int64")

    return df, pd.DataFrame(log)


# --------------------------------------------------------------------------- #
# Grading against the answer key (instructor only)
# --------------------------------------------------------------------------- #
def grade(name, cleaned):
    """Compare a cleaned frame with the clean reference, cell by cell.

    Rows are matched by position: duplicates only replace the last two rows,
    so positions 0..len(cleaned)-1 line up with the clean file.
    """
    truth = pd.read_csv(ROOT / f"data/clean/{name}_clean.csv").iloc[: len(cleaned)]
    cleaned = cleaned.reset_index(drop=True)
    rows = []
    for col in truth.columns:
        t, c = truth[col], cleaned[col]
        missing = c.isna()
        if pd.api.types.is_numeric_dtype(t):
            same = pd.Series(np.isclose(pd.to_numeric(c, errors="coerce").astype(float),
                                        t.astype(float), atol=1e-6), index=t.index)
        else:
            same = c.astype(str).eq(t.astype(str))
        rows.append(dict(column=col, correct=int((same & ~missing).sum()),
                         left_missing=int(missing.sum()),
                         wrong=int((~same & ~missing).sum())))
    return pd.DataFrame(rows).set_index("column")


def main():
    for name in ["healthcare", "retail"]:
        cleaned, log = clean_raw(name)
        report = grade(name, cleaned)
        print(f"\n=== {name}: {len(cleaned)} rows after cleaning "
              f"({len(log)} logged repairs) ===")
        print(log["action"].value_counts().to_string())
        print("\nGrade against the answer key (columns with anything to report):")
        print(report[(report.left_missing > 0) | (report.wrong > 0)].to_string())
        print("Wrong values outside the ID column:",
              int(report.drop(index=ID_COL[name]).wrong.sum()))


if __name__ == "__main__":
    main()
