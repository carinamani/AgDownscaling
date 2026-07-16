## Defines function to configure RF model run based on parameters set in main script

from dataclasses import dataclass, field
from typing import Literal

# lays out all of the inputs needed for each model run (some are defined here as default values, while others
# are set directly in the run script) and checks that they are the expected type (e.g. filepath should be a string)
# each class is then callable by config.target (for example)

# default hyperparameter search spaces, keyed by model type
DEFAULT_PARAM_DISTRIBUTIONS = {
    "rf": {
        "n_estimators":     [200, 500, 1000],
        "max_features":     ["sqrt", "log2", 0.3, 0.5],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_depth":        [None, 10, 20, 30],
    },
    "qrf": {
        "n_estimators":     [200, 500, 1000],
        "max_features":     ["sqrt", "log2", 0.3, 0.5],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_depth":        [None, 10, 20, 30],
    },
    "xgb": {
        "n_estimators":     [200, 500, 1000],
        "max_depth":        [3, 5, 7, 10],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 5, 10],
        "reg_alpha":        [0, 0.1, 1],
        "reg_lambda":       [1, 1.5, 2],
    },
    "lgbm": {
        "n_estimators":      [200, 500, 1000],
        "num_leaves":        [15, 31, 63, 127],
        "learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "subsample":         [0.6, 0.8, 1.0],
        "colsample_bytree":  [0.6, 0.8, 1.0],
        "min_child_samples": [5, 10, 20, 50],
        "reg_alpha":         [0, 0.1, 1],
        "reg_lambda":        [0, 0.1, 1],
    },
}

@dataclass
class RunConfig:
    run_name:            str
    target:              str
    dataset:             str
    fold_assignments:    str       
    model_type:          str
    id_cols:             list      = field(default_factory=list)
    n_inner_folds:       int       = 3
    n_iter_search:       int       = 25
    random_seed:         int       = 42
    quantiles:           list      = field(default_factory=lambda: [0.1, 0.5, 0.9])
    param_distributions: dict      = None  # if not set manually, filled in below based on model_type

    def __post_init__(self):
        if self.param_distributions is None:
            self.param_distributions = DEFAULT_PARAM_DISTRIBUTIONS[self.model_type]