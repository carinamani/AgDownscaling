# Defines function for training the RF model 

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from quantile_forest import RandomForestQuantileRegressor
import numpy as np

# Function to initialize the model based on model type 
def get_model(config):
    if config.model_type == "rf":
        return RandomForestRegressor(random_state=config.random_seed, n_jobs=-1)
    elif config.model_type == "qrf":
        return RandomForestQuantileRegressor(random_state=config.random_seed, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
# Function to compute inverse-frequency sample weights based on country count in training set
# Computed per fold 
def compute_sample_weights(country_ids: pd.Series, method: str):
    if method == "none":
        return None

    counts = country_ids.value_counts()  

    if method == "inverse":
        raw_weights = 1.0 / counts
    elif method == "sqrt_inverse":
        raw_weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # normalize so mean weight 
    raw_weights = raw_weights / raw_weights.mean()

    return country_ids.map(raw_weights).values
    
# Function to train the model using grid search  
def run_inner_search(model, X_train, y_train, config, sample_weight=None):
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
    fit_params = {}
    if sample_weight is not None:
        fit_params["sample_weight"] = sample_weight

    search.fit(X_train, y_train, **fit_params)
    return search.best_estimator_, search.best_params_, search.cv_results_

# Function to test the model 
def predict(model, X_test, config):
    # if running QRF, return all quantiles 
    if config.model_type == "qrf":
        preds = model.predict(X_test, quantiles=config.quantiles)
        return pd.DataFrame(preds, columns=[f"q{int(q*100)}" for q in config.quantiles])
    # otherwise just run model
    else:
        return pd.Series(model.predict(X_test), name="prediction")

# Function which runs the training and testing functions for each fold of spatial CV 
def train_model(df, folds, config):

    feature_cols = config.feature_cols

    X = df[feature_cols]
    y = df[config.target]

    fold_results = []

    for fold in folds:
        print(f"\n── {fold['fold']} ──────────────────────────────")
        print(f"  test countries: {fold['test_countries']}")

        X_train = X.loc[fold["train_idx"]]
        y_train = y.loc[fold["train_idx"]]
        X_test  = X.loc[fold["test_idx"]]

        # compute sample weights from this fold's training country composition
        train_countries = df.loc[fold["train_idx"], "country_ID"]
        sample_weight   = compute_sample_weights(train_countries, config.weighting)

        # initialize and train model
        model = get_model(config)
        best_model, best_params, _ = run_inner_search(
            model, X_train, y_train, config, sample_weight=sample_weight
        )
        print(f"  best params: {best_params}")

        # predict on test and train
        preds       = predict(best_model, X_test,  config)
        train_preds = predict(best_model, X_train, config)
        if isinstance(preds, pd.Series):
            preds       = preds.to_frame()
            train_preds = train_preds.to_frame()

        # build test rows
        test_df = df.loc[fold["test_idx"], config.id_cols + [config.target]].copy().reset_index(drop=True)
        test_df = pd.concat([test_df, preds.reset_index(drop=True)], axis=1)
        test_df["fold"]  = fold["fold"]
        test_df["split"] = "test"

        # build train rows
        train_df = df.loc[fold["train_idx"], config.id_cols + [config.target]].copy().reset_index(drop=True)
        train_df = pd.concat([train_df, train_preds.reset_index(drop=True)], axis=1)
        train_df["fold"]  = fold["fold"]
        train_df["split"] = "train"

        fold_results.append({
            "fold":        fold["fold"],
            "test_df":     test_df,
            "train_df":    train_df,
            "best_params": best_params,
            "best_model":  best_model,
            "test_idx":    fold["test_idx"],
            "train_idx":   fold["train_idx"],
        })

    all_predictions = pd.concat(
        [pd.concat([r["test_df"], r["train_df"]]) for r in fold_results],
        ignore_index=True
    )

    return {
        "config":       config,
        "fold_results": fold_results,
        "predictions":  all_predictions,
    }