# Defines functions for setting up the spatial blocking for spatial CV
# requires CSV where I manually define which country_ID is in which fold 

import pandas as pd

# function for loading the CSV which assigns countries to folds for cross-validation 
# in the CSV, 1 = test countries, 0 = train countries 
def load_fold_assignments(fold_csv_path):
    folds = pd.read_csv(fold_csv_path)
    fold_cols = [c for c in folds.columns if c.startswith("fold_")] # find fold columns
    assert len(fold_cols) > 0, "Error: no fold columns found in fold CSV."
    return folds, fold_cols

# function to create a list of dictionaries which id's the train and test rows for each fold  
# countries missing in fold CSV get dropped from model altogether 
def make_spatial_folds(df: pd.DataFrame, config):
    
    # get filepath for fold assignments and run it through loading function
    folds_df, fold_cols = load_fold_assignments(config.fold_assignments) 

    # only keep rows from countries which appear in the fold CSV
    valid_countries = set(folds_df["country_ID"])
    df_filtered = df[df["country_ID"].isin(valid_countries)]

    n_dropped = len(df) - len(df_filtered)
    if n_dropped > 0:
        print(f"Excluded {n_dropped:,} rows from countries missing from fold CSV")

    splits = []
    for fold in fold_cols:
        test_countries  = folds_df.loc[folds_df[fold] == 1, "country_ID"].tolist()
        train_countries = folds_df.loc[folds_df[fold] == 0, "country_ID"].tolist()

        test_idx  = df_filtered[df_filtered["country_ID"].isin(test_countries)].index
        train_idx = df_filtered[df_filtered["country_ID"].isin(train_countries)].index

        assert len(test_idx)  > 0, f"{fold}: no test rows found"
        assert len(train_idx) > 0, f"{fold}: no train rows found"

        print(f"  {fold}: {len(train_countries)} train countries ({len(train_idx):,} rows) | "
              f"{len(test_countries)} test countries ({len(test_idx):,} rows)")

        # build dictionary of info for each split 
        splits.append({
            "fold":            fold,
            "train_idx":       train_idx,
            "test_idx":        test_idx,
            "test_countries":  test_countries,
            "train_countries": train_countries,
        })

    return splits