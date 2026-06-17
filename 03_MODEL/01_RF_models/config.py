from dataclasses import dataclass, field
from typing import Literal

ModelType = Literal["rf", "qrf"]

@dataclass
class RunConfig:
    run_name:            str
    target:              str
    dataset:             str
    fold_assignments:    str       
    model_type:          ModelType = "rf"
    spatial_block_col:   str       = "country"
    id_cols:             list      = field(default_factory=list)
    n_inner_folds:       int       = 3
    n_iter_search:       int       = 25
    random_seed:         int       = 42
    quantiles:           list      = field(default_factory=lambda: [0.1, 0.5, 0.9])
    param_distributions: dict      = field(default_factory=lambda: {
        "n_estimators":     [200, 500, 1000],
        "max_features":     ["sqrt", "log2", 0.3, 0.5],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_depth":        [None, 10, 20, 30],
    })