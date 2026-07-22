# Defines functions for setting up the spatial blocking for spatial CV
# requires CSV where folds are manually defined (split by by country_ID)

import pandas as pd

# function for loading the CSV which assigns countries to folds for cross-validation 
# in the CSV, 1 = test countries, 0 = train countries 
def load_fold_assignments(fold_csv_path):
    folds = pd.read_csv(fold_csv_path)
    fold_cols = [c for c in folds.columns if c.startswith("fold_")] 
    return folds, fold_cols

# function to create a list of dictionaries which id's the train and test rows for each fold  
def make_spatial_folds(df: pd.DataFrame, config):
    
    # get filepath for fold assignments and run it through loading function
    folds_df, fold_cols = load_fold_assignments(config.fold_assignments) 

    # create spatial CV splits 
    splits = []
    for fold in fold_cols:
        test_countries  = folds_df.loc[folds_df[fold] == 1, "country_ID"].tolist()
        train_countries = folds_df.loc[folds_df[fold] == 0, "country_ID"].tolist()

        test_idx  = df[df["country_ID"].isin(test_countries)].index
        train_idx = df[df["country_ID"].isin(train_countries)].index

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