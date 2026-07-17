# Defines functions for training the RF model based on parameters set in run script and config

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from quantile_forest import RandomForestQuantileRegressor
import xgboost as xgb
import lightgbm as lgb

# Function to initialize the model based on model type defined in run script
# n_jobs = -1 means all available CPU cores are to be used for training the model
def get_model(config):
    if config.model_type == "rf":
        return RandomForestRegressor(random_state=config.random_seed, n_jobs=-1)
    elif config.model_type == "qrf":
        return RandomForestQuantileRegressor(random_state=config.random_seed, n_jobs=-1)
    elif config.model_type == "xgb":
        return xgb.XGBRegressor(random_state=config.random_seed, n_jobs=-1, tree_method="hist")
    elif config.model_type == "lgbm":
        return lgb.LGBMRegressor(random_state=config.random_seed, n_jobs=-1, verbosity=-1)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
# Function to train the model using the training data
# determines best hyperparameters using non-spatial CV within training data 
# uses random sampling of combination options rather than complete set of possible combinations 
def run_inner_search(model, X_train, y_train, config):
    search = RandomizedSearchCV(
        estimator          = model,
        param_distributions = config.param_distributions,
        n_iter             = config.n_iter_search,
        cv                 = config.n_inner_folds,
        scoring            = "neg_root_mean_squared_error",
        random_state       = config.random_seed,
        n_jobs             = -1,
        refit              = True, # automatically retrains the model on the best-fitting hyperparameter set 
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.cv_results_

# Function to test the model using the test data
def predict(model, X_test, config):
    # if running QRF, return all quantiles 
    if config.model_type == "qrf":
        preds = model.predict(X_test, quantiles=config.quantiles)
        return pd.DataFrame(preds, columns=[f"q{int(q*100)}" for q in config.quantiles])
    # otherwise just run model
    else:
        return pd.Series(model.predict(X_test), name="prediction")

# Function which runs the training and testing functions for each fold of spatial CV 
def train_model(df, feature_cols, folds, config):

    X = df[feature_cols] # set in run script to exclude ID columns and unused targets
    y = df[config.target]

    fold_results = []

    for fold in folds:
        print(f"\n── {fold['fold']} ──────────────────────────────")
        print(f"  test countries: {fold['test_countries']}")

        # split fold into test and train from fold id's
        X_train = X.loc[fold["train_idx"]]
        y_train = y.loc[fold["train_idx"]]
        X_test  = X.loc[fold["test_idx"]]
        y_test  = y.loc[fold["test_idx"]]

        # initialize and train model
        model = get_model(config)
        best_model, best_params, cv_results = run_inner_search(model, X_train, y_train, config)
        print(f"  best params: {best_params}")

        # predict on test
        preds = predict(best_model, X_test, config)
        if isinstance(preds, pd.Series):
            preds = preds.to_frame()

        # predict on train
        train_preds = predict(best_model, X_train, config)
        if isinstance(train_preds, pd.Series):
            train_preds = train_preds.to_frame()

        # create df of actual and predicted values for each test region
        fold_df = df.loc[fold["test_idx"], config.id_cols + [config.target]].copy().reset_index(drop=True)
        preds   = preds.reset_index(drop=True)
        fold_df = pd.concat([fold_df, preds], axis=1)
        fold_df["fold"] = fold["fold"]

        fold_results.append({
            "fold":             fold["fold"],
            "predictions":      fold_df,
            "train_predictions": pd.concat([
                df.loc[fold["train_idx"], config.id_cols + [config.target]].copy().reset_index(drop=True),
                train_preds.reset_index(drop=True)
            ], axis=1),
            "best_params":      best_params,
            "best_model":       best_model,
            "cv_results":       cv_results,
            "test_idx":         fold["test_idx"],
        })

    # stack predictions across all folds into 1 df
    all_predictions = pd.concat([r["predictions"] for r in fold_results], ignore_index=True)

    return {
        "config":       config,
        "feature_cols": feature_cols,
        "fold_results": fold_results,
        "predictions":  all_predictions,
    }