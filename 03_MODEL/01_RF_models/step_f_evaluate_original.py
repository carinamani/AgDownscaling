# Defines functions to convert rtv (country-relative) log predictions back into absolute intensity predictions, and evaluate R2/RMSE of those
# predictions against ground-truth subnational intensity data
# (in both log and raw scale)

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error

# map of version/unit -> column name
RAW_INTENSITY_COLS = {
    ("labor", "USD"):     "labor_intensity_jobs_per_million_USD",
    ("labor", "tonne"):   "labor_intensity_jobs_per_tonne",
    ("capital", "USD"):   "capital_intensity_USD_per_USD",
    ("capital", "tonne"): "capital_intensity_USD_per_tonne",
}

# rescaling factor needed 
SCALE_FACTOR = {
    ("labor", "USD"):     1,
    ("labor", "tonne"):   1e6,
    ("capital", "USD"):   1e6,
    ("capital", "tonne"): 1e6,
}

# list of numerators 
NUMERATOR = {"labor": "jobs", "capital": "USD"}

# files names for actual subnational data
SUBNATIONAL_FILES = {
    "labor":   "subnational_labor_intensity.csv",
    "capital": "subnational_capital_intensity.csv",
}

# function to set column name
def _intensity_col_name(version, unit):
    return f"{version}_intensity_{NUMERATOR[version]}_per_million_{unit}"

# Function to load and prep country-level actual intensities for a given version/unit
def load_country_actuals(country_intensity_path, version, unit):
    df = pd.read_csv(country_intensity_path)
    raw_col = RAW_INTENSITY_COLS[(version, unit)]
    out_col = f"country_{_intensity_col_name(version, unit)}"

    df = df.copy()
    df[out_col] = df[raw_col] * SCALE_FACTOR[(version, unit)]
    df["country_ID"] = df["ISO3"]

    return df[["country_ID", out_col]], out_col


# Function to load and prep subnational (region-level) ground-truth intensities
def load_subnational_actuals(intensity_dir, version, unit):
    df = pd.read_csv(intensity_dir / SUBNATIONAL_FILES[version])
    raw_col = RAW_INTENSITY_COLS[(version, unit)]
    out_col = f"region_{_intensity_col_name(version, unit)}"

    df = df.copy()
    df[out_col] = df[raw_col] * SCALE_FACTOR[(version, unit)]

    return df[["PROJ_ID", out_col]], out_col


# Function to reconstruct predicted absolute intensity from the model's prediction.
# If "rtv": adds the country's log intensity back onto the rtv log prediction
# If "abs": the prediction already IS the absolute log intensity, so it's passed through unchanged.
def add_absolute_intensity_predictions(predictions_df, country_actuals, country_col, config, pred_col):
    df = predictions_df.merge(country_actuals, on="country_ID", how="left")

    log_country_col = f"log_{country_col}"
    df[log_country_col] = np.log1p(df[country_col])

    base_name = _intensity_col_name(config.version, config.unit)
    pred_abs_log_col = f"predicted_{base_name}_log"
    pred_abs_col      = f"predicted_{base_name}"

    if config.variable_def == "rtv":
        df[pred_abs_log_col] = df[log_country_col] + df[pred_col]
    elif config.variable_def == "abs":
        df[pred_abs_log_col] = df[pred_col]
    else:
        raise ValueError(f"Unknown variable_def: {config.variable_def}")

    df[pred_abs_col] = np.expm1(df[pred_abs_log_col])

    return df, pred_abs_log_col, pred_abs_col

# Function to compute one row of R2/RMSE metrics (log + raw scale) for a group of rows
def _r2_row(label, group, actual_col, pred_abs_log_col, pred_abs_col):
    valid = group[[actual_col, pred_abs_log_col, pred_abs_col]].dropna().copy()
    log_actual_col = f"log_{actual_col}"
    valid[log_actual_col] = np.log1p(valid[actual_col])

    return {
        "fold":     label,
        "n":        len(valid),
        "r2_log":   r2_score(valid[log_actual_col], valid[pred_abs_log_col]),
        "rmse_log": root_mean_squared_error(valid[log_actual_col], valid[pred_abs_log_col]),
        "r2_raw":   r2_score(valid[actual_col], valid[pred_abs_col]),
        "rmse_raw": root_mean_squared_error(valid[actual_col], valid[pred_abs_col]),
    }

# Function to compute R2/RMSE per fold plus an overall row
def evaluate_intensity_r2(df, actual_col, pred_abs_log_col, pred_abs_col, fold_col="fold"):
    rows = [
        _r2_row(fold, group, actual_col, pred_abs_log_col, pred_abs_col)
        for fold, group in df.groupby(fold_col)
    ]
    rows.append(_r2_row("overall", df, actual_col, pred_abs_log_col, pred_abs_col))
    return pd.DataFrame(rows)


# Reconstructs absolute intensity predictions and scores them against ground-truth subnational intensity data, per fold and overall
def run_intensity_evaluation(results, config, country_intensity_path, intensity_dir):
    predictions = results["predictions"].copy()
    pred_col = "q50" if config.model_type == "qrf" else "prediction"

    country_actuals, country_col = load_country_actuals(
        country_intensity_path, config.version, config.unit
    )
    subnational_actuals, actual_col = load_subnational_actuals(
        intensity_dir, config.version, config.unit
    )

    predictions, pred_abs_log_col, pred_abs_col = add_absolute_intensity_predictions(
        predictions, country_actuals, country_col, config, pred_col
    )
    predictions = predictions.merge(subnational_actuals, on="PROJ_ID", how="left")

    metrics_model = evaluate_intensity_r2(predictions, actual_col, pred_abs_log_col, pred_abs_col)

    log_country_col = f"log_country_{_intensity_col_name(config.version, config.unit)}"
    country_col = f"country_{_intensity_col_name(config.version, config.unit)}"

    metrics_country_avg = evaluate_intensity_r2(predictions, actual_col, log_country_col, country_col)

    print("\n── Absolute intensity R² (reconstructed from predictions) ──────")
    print(metrics_model.to_string(index=False))

    print("\n── Absolute intensity R² (country average model) ──────")
    print(metrics_country_avg.to_string(index=False))

    return metrics_model, metrics_country_avg