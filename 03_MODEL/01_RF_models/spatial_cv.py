#### Generates the spatial block splitting logic
# imports CSV where I manually define which country_code is in which test fold (with column per fold and 1 for in test fold - 20% of data)

import pandas as pd
from pathlib import Path

def load_fold_assignments(fold_csv_path: str) -> pd.DataFrame:
    """
    Load fold assignment CSV. Expects columns:
      - country_code (or similar identifier)
      - fold_1, fold_2, ... (binary 0/1)
    """
    folds = pd.read_csv(fold_csv_path)
    fold_cols = [c for c in folds.columns if c.startswith("fold_")]
    assert len(fold_cols) > 0, "No fold columns found — expected columns named fold_1, fold_2, ..."
    return folds, fold_cols


def make_spatial_folds(df: pd.DataFrame, config) -> list[dict]:
    """
    Returns a list of dicts, one per fold:
      {
        "fold":       fold name (e.g. "fold_1"),
        "train_idx":  pd.Index of training rows,
        "test_idx":   pd.Index of test rows,
        "test_countries": list of countries in test set
      }

    Countries with a 1 in a fold column are the TEST set for that fold.
    Countries with a 0 are the TRAIN set.
    Countries absent from the CSV entirely are excluded from both.
    """
    folds_df, fold_cols = load_fold_assignments(config.fold_assignments)

    # only keep rows whose country appears in the fold CSV
    valid_countries = set(folds_df["country_code"])
    df_filtered = df[df[config.spatial_block_col].isin(valid_countries)]

    n_dropped = len(df) - len(df_filtered)
    if n_dropped > 0:
        print(f"  Excluded {n_dropped:,} rows from countries not in fold CSV")

    splits = []
    for fold in fold_cols:
        test_countries  = folds_df.loc[folds_df[fold] == 1, "country_code"].tolist()
        train_countries = folds_df.loc[folds_df[fold] == 0, "country_code"].tolist()

        test_idx  = df_filtered[df_filtered[config.spatial_block_col].isin(test_countries)].index
        train_idx = df_filtered[df_filtered[config.spatial_block_col].isin(train_countries)].index

        assert len(test_idx)  > 0, f"{fold}: no test rows found"
        assert len(train_idx) > 0, f"{fold}: no train rows found"

        print(f"  {fold}: {len(train_countries)} train countries ({len(train_idx):,} rows) | "
              f"{len(test_countries)} test countries ({len(test_idx):,} rows)")

        splits.append({
            "fold":            fold,
            "train_idx":       train_idx,
            "test_idx":        test_idx,
            "test_countries":  test_countries,
            "train_countries": train_countries,
        })

    return splits