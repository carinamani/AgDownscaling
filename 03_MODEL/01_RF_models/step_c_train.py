import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from quantile_forest import RandomForestQuantileRegressor
from step_a_config import RunConfig


def get_model(config: RunConfig):
    if config.model_type == "rf":
        return RandomForestRegressor(random_state=config.random_seed, n_jobs=-1)
    elif config.model_type == "qrf":
        return RandomForestQuantileRegressor(random_state=config.random_seed, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def run_inner_search(model, X_train, y_train, config: RunConfig):
    """Randomized search over hyperparameters using non-spatial inner CV."""
    search = RandomizedSearchCV(
        estimator          = model,
        param_distributions = config.param_distributions,
        n_iter             = config.n_iter_search,
        cv                 = config.n_inner_folds,
        scoring            = "neg_root_mean_squared_error",
        random_state       = config.random_seed,
        n_jobs             = -1,
        refit              = True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.cv_results_


def predict(model, X_test, config: RunConfig):
    if config.model_type == "qrf":
        preds = model.predict(X_test, quantiles=config.quantiles)
        # returns array of shape (n_samples, n_quantiles)
        return pd.DataFrame(preds, columns=[f"q{int(q*100)}" for q in config.quantiles])
    else:
        return pd.Series(model.predict(X_test), name="prediction")


def train_model(df: pd.DataFrame, feature_cols: list, folds: list, config: RunConfig) -> dict:
    """
    Iterate over spatial folds, run inner search, predict on test set.
    Returns a results dict with predictions and metadata from all folds.
    """
    X = df[feature_cols]
    y = df[config.target]

    fold_results = []

    for fold in folds:
        print(f"\n── {fold['fold']} ──────────────────────────────")
        print(f"  test countries: {fold['test_countries']}")

        X_train = X.loc[fold["train_idx"]]
        y_train = y.loc[fold["train_idx"]]
        X_test  = X.loc[fold["test_idx"]]
        y_test  = y.loc[fold["test_idx"]]

        # fit
        model = get_model(config)
        best_model, best_params, cv_results = run_inner_search(model, X_train, y_train, config)
        print(f"  best params: {best_params}")

        # predict
        preds = predict(best_model, X_test, config)
        if isinstance(preds, pd.Series):
            preds = preds.to_frame()

        # assemble fold output
        fold_df = df.loc[fold["test_idx"], config.id_cols + [config.target]].copy().reset_index(drop=True)
        preds   = preds.reset_index(drop=True)
        fold_df = pd.concat([fold_df, preds], axis=1)
        fold_df["fold"] = fold["fold"]

        fold_results.append({
            "fold":        fold["fold"],
            "predictions": fold_df,
            "best_params": best_params,
            "best_model":  best_model,
            "cv_results":  cv_results,
        })

    # stack predictions across all folds
    all_predictions = pd.concat([r["predictions"] for r in fold_results], ignore_index=True)

    return {
        "config":       config,
        "feature_cols": feature_cols,
        "fold_results": fold_results,
        "predictions":  all_predictions,
    }