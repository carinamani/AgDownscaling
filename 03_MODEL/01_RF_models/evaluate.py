import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score
from config import RunConfig


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE, R2 on log scale, plus RMSE on original scale."""
    residuals = y_true - y_pred
    return {
        "rmse":         root_mean_squared_error(y_true, y_pred),
        "mae":          np.mean(np.abs(residuals)),
        "r2":           r2_score(y_true, y_pred),
        "rmse_orig":    root_mean_squared_error(np.exp(y_true), np.exp(y_pred)),
        "mae_orig":     np.mean(np.abs(np.exp(y_true) - np.exp(y_pred))),
        "med_abs_err":  np.median(np.abs(residuals)),
    }


def compute_interval_metrics(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray, nominal_coverage: float) -> dict:
    """
    Coverage and width of prediction intervals for QRF.
    - coverage: fraction of test points where true value falls within [q_low, q_high]
    - mean_width: average interval width on log scale
    - coverage_gap: how far coverage is from nominal (e.g. 0.8 for 10-90 interval)
    """
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


def evaluate_fold(fold_result: dict, config: RunConfig) -> dict:
    preds  = fold_result["predictions"]
    y_true = preds[config.target].values

    metrics = {"fold": fold_result["fold"]}

    if config.model_type == "rf":
        metrics.update(compute_point_metrics(y_true, preds["prediction"].values))

    elif config.model_type == "qrf":
        # use median as point estimate
        metrics.update(compute_point_metrics(y_true, preds["q50"].values))

        # interval metrics for each pair of symmetric quantiles
        quantile_pairs = [
            (q, 1 - q) for q in config.quantiles if q < 0.5
        ]
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
                # prefix with interval name
                label = f"{q_lo_col}_{q_hi_col}"
                metrics.update({f"{label}_{k}": v for k, v in interval_metrics.items()})

    return metrics


def evaluate(results: dict, config: RunConfig) -> pd.DataFrame:
    """
    Evaluate all folds and return a DataFrame with one row per fold
    plus an overall row aggregated across all folds.
    """
    fold_metrics = [evaluate_fold(f, config) for f in results["fold_results"]]
    metrics_df   = pd.DataFrame(fold_metrics)

    # overall row: mean across folds
    overall      = metrics_df.drop(columns="fold").mean().to_dict()
    overall["fold"] = "overall"
    metrics_df   = pd.concat(
        [metrics_df, pd.DataFrame([overall])],
        ignore_index=True
    )

    print("\n── Evaluation ───────────────────────────────")
    print(metrics_df.to_string(index=False))

    return metrics_df