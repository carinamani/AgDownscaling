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

# Function to get, for each fold, the top-N features by a given importance column
def get_top_n_by_fold(fold_importance_df, importance_col, n=5):
    return (
        fold_importance_df
        .groupby("fold", group_keys=False)
        .apply(lambda g: g.nlargest(n, importance_col))
        .reset_index(drop=True)
    )

# Function to measure how stable the top-N feature set is across folds
# "presence_rate" = fraction of folds in which a feature appears in that fold's top N
def compute_top_n_stability(fold_importance_df, importance_col, n=5):
    n_folds = fold_importance_df["fold"].nunique()

    top_n_df = get_top_n_by_fold(fold_importance_df, importance_col, n=n)

    # rank within each fold's top N (1 = most important that fold)
    top_n_df["rank_in_fold"] = (
        top_n_df.groupby("fold")[importance_col]
        .rank(ascending=False, method="first")
    )

    stability = (
        top_n_df.groupby("feature")
        .agg(
            times_in_top_n = ("fold", "count"),
            mean_rank_when_in_top_n = ("rank_in_fold", "mean"),
        )
        .reset_index()
    )
    stability["presence_rate"] = stability["times_in_top_n"] / n_folds
    stability = stability.sort_values(
        ["presence_rate", "mean_rank_when_in_top_n"], ascending=[False, True]
    ).reset_index(drop=True)

    return stability, top_n_df

# Function to run impurity and permutation importance for each fold
# also gets aggregate stats across all folds, and top-N stability
def get_feature_importance(results, df, config, top_n=5):
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

    # top-N stability, based on permutation importance (more reliable than impurity for this purpose)
    perm_stability, perm_top_n_by_fold = compute_top_n_stability(
        permutation_df, "permutation_importance", n=top_n
    )

    print(f"\n── Top-{top_n} feature stability across folds (permutation importance) ──")
    print(perm_stability.to_string(index=False))

    return {
        "combined":            combined,
        "top_n_stability":     perm_stability,
        "top_n_by_fold":       perm_top_n_by_fold,
    }