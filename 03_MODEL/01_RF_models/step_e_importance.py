# Defines functions to measure feature importance across models

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

# Function to get impurity importance score 
# impurity importance = the more a feature is used in splits which reduce the impurity in nodes, the higher its importance
# NB: this measure is biased towards features with high-cardiality (i.e. non-binary features that offer more ways for model to split the data)
def get_impurity_importance(model, feature_cols):
    return pd.DataFrame({
        "feature":            feature_cols,
        "impurity_importance": model.feature_importances_,
    }).sort_values("impurity_importance", ascending=False)

# Function to get permutation importance 
# shuffle all of the values of each feature and see how model performance is impacted
# the most its impacted, the higher the importance of that feature 
def get_permutation_importance(model, X_test, y_test, config, n_repeats: int = 10) -> pd.DataFrame:
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats    = n_repeats,
        random_state = config.random_seed,
        n_jobs       = 1 if config.model_type == "qrf" else -1,
        scoring      = "neg_root_mean_squared_error",
    )
    return pd.DataFrame({
        "feature":                X_test.columns.tolist(),
        "permutation_importance": result.importances_mean,
        "permutation_std":        result.importances_std,
    }).sort_values("permutation_importance", ascending=False)

# Function to run impurity and permutation importance for each fold
# also gets aggregate stats across all folds
def get_feature_importance(results, df, config):
    X = df[results["feature_cols"]]
    y = df[config.target]

    fold_impurity     = []
    fold_permutation  = []

    for fold_result in results["fold_results"]:
        model    = fold_result["best_model"]
        test_idx = fold_result["test_idx"]

        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]

        imp  = get_impurity_importance(model, results["feature_cols"])
        perm = get_permutation_importance(model, X_test, y_test, config)

        imp["fold"]  = fold_result["fold"]
        perm["fold"] = fold_result["fold"]

        fold_impurity.append(imp)
        fold_permutation.append(perm)

    # aggregate across folds
    impurity_df    = pd.concat(fold_impurity)
    permutation_df = pd.concat(fold_permutation)

    # calculate aggregate metrics (mean and SD across all folds)
    impurity_agg = (
        impurity_df
        .groupby("feature")["impurity_importance"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "impurity_mean", "std": "impurity_std"})
        .reset_index()
    )

    permutation_agg = (
        permutation_df
        .groupby("feature")["permutation_importance"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "permutation_mean", "std": "permutation_std"})
        .reset_index()
    )

    combined = impurity_agg.merge(permutation_agg, on="feature")
    combined = combined.sort_values("permutation_mean", ascending=False).reset_index(drop=True)

    print("\n── Feature Importance (top 15) ──────────────────────────────")
    print(combined.head(15).to_string(index=False))

    return combined