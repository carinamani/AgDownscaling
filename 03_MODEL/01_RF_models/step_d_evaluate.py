# Defines functions to evalutate model performance on test data

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score

# Function to calculate core metrics (RMSE, MAE, R2 on log scale, and RMSE on non-log scale)
def compute_point_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    return {
        "rmse":         root_mean_squared_error(y_true, y_pred),
        "mae":          np.mean(np.abs(residuals)),
        "r2":           r2_score(y_true, y_pred),
        "rmse_orig":    root_mean_squared_error(np.exp(y_true), np.exp(y_pred)),
        "mae_orig":     np.mean(np.abs(np.exp(y_true) - np.exp(y_pred))),
        "med_abs_err":  np.median(np.abs(residuals)),
    }

# Function to evaluate prediction intervals from QRF
# coverage: fraction of test points where true value falls within [q_low, q_high]
# mean_width: average interval width (on log scale)
# coverage_gap: how far coverage is from nominal (10-90th percentile should correctly capture 80% of data)
def compute_interval_metrics(y_true, q_low, q_high, nominal_coverage):
    covered    = (y_true >= q_low) & (y_true <= q_high)
    coverage   = covered.mean()
    mean_width = (q_high - q_low).mean()
    return {
        "coverage":         coverage,
        "nominal_coverage": nominal_coverage,
        "coverage_gap":     coverage - nominal_coverage,
        "mean_width":       mean_width,
        "mean_width_orig":  (np.exp(q_high) - np.exp(q_low)).mean(),
    }

# function to compute teh R2 and RMSE for each country for each fold 
def compute_country_metrics(preds_df, target_col, pred_col, country_col="country_ID"):
    rows = []
    for country, group in preds_df.groupby(country_col):
        y_true = group[target_col].values
        y_pred = group[pred_col].values
        if len(y_true) < 2:
            r2 = np.nan  # can't compute R2 with a single point
        else:
            r2 = r2_score(y_true, y_pred)
        rows.append({
            "country":  country,
            "n":        len(y_true),
            "r2":       r2,
            "rmse":     root_mean_squared_error(y_true, y_pred),
        })
    return pd.DataFrame(rows).sort_values("r2", ascending=False)

# Function to evaluate the results of an individual fold from the spatial CV
# runs functions from above based on if its RF or QRF
def evaluate_fold(fold_result, config):
    preds       = fold_result["predictions"]
    train_preds = fold_result["train_predictions"]
    y_true      = preds[config.target].values
    y_true_train = train_preds[config.target].values

    metrics = {"fold": fold_result["fold"]}

    pred_col = "q50" if config.model_type == "qrf" else "prediction"

    if config.model_type == "qrf":
        test_point  = compute_point_metrics(y_true,       preds[pred_col].values)
        train_point = compute_point_metrics(y_true_train, train_preds[pred_col].values)

        metrics.update({f"test_{k}":  v for k, v in test_point.items()})
        metrics.update({f"train_{k}": v for k, v in train_point.items()})

        quantile_pairs = [(q, 1 - q) for q in config.quantiles if q < 0.5]
        for q_lo, q_hi in quantile_pairs:
            q_lo_col = f"q{int(q_lo * 100)}"
            q_hi_col = f"q{int(q_hi * 100)}"
            if q_lo_col in preds.columns and q_hi_col in preds.columns:
                nominal = q_hi - q_lo
                interval_metrics = compute_interval_metrics(
                    y_true,
                    preds[q_lo_col].values,
                    preds[q_hi_col].values,
                    nominal_coverage=nominal,
                )
                label = f"{q_lo_col}_{q_hi_col}"
                metrics.update({f"{label}_{k}": v for k, v in interval_metrics.items()})
    else:
        test_point  = compute_point_metrics(y_true,       preds[pred_col].values)
        train_point = compute_point_metrics(y_true_train, train_preds[pred_col].values)

        metrics.update({f"test_{k}":  v for k, v in test_point.items()})
        metrics.update({f"train_{k}": v for k, v in train_point.items()})

    # per-country R2
    test_country_metrics  = compute_country_metrics(preds,       config.target, pred_col)
    train_country_metrics = compute_country_metrics(train_preds, config.target, pred_col)

    test_country_metrics["split"]  = "test"
    train_country_metrics["split"] = "train"
    test_country_metrics["fold"]   = fold_result["fold"]
    train_country_metrics["fold"]  = fold_result["fold"]

    metrics["country_metrics"] = pd.concat(
        [test_country_metrics, train_country_metrics], ignore_index=True
    )

    return metrics

# Function to run metrics actoss all folds and print into table
# includes row which aggregates results across all folds
def evaluate(results, config):
    fold_metrics = [evaluate_fold(f, config) for f in results["fold_results"]]

    # pull out country-level tables before flattening
    country_metrics_all = pd.concat(
        [f.pop("country_metrics") for f in fold_metrics],
        ignore_index=True
    )

    metrics_df = pd.DataFrame(fold_metrics)
    overall    = metrics_df.drop(columns="fold").mean().to_dict()
    overall["fold"] = "overall"
    metrics_df = pd.concat([metrics_df, pd.DataFrame([overall])], ignore_index=True)

    print("\n── Evaluation ───────────────────────────────")
    print(metrics_df.to_string(index=False))

    return metrics_df, country_metrics_all