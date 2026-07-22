## Defines function to configure RF model run based on parameters set in main 

from dataclasses import dataclass, field
from typing import Literal

# lays out all of the inputs needed for each model run (some are defined here as default values, while others
# are set directly in the run script) and checks that they are the expected type 

# set hyperparameter search space
GRID_SEARCH = {
    'n_estimators': [200, 500, 1000, 2000],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50],
    'max_features': ["sqrt", "log2", 0.3, 0.5, 0.75],
}

@dataclass
class RunConfig:
    run_name:            str
    target:              str
    dataset:             str
    fold_assignments:    str       
    model_type:          str
    feature_cols:        list
    weighting:           str       = 'sqrt_inverse'
    id_cols:             list      = field(default_factory=lambda: ["PROJ_ID", "country_ID"])
    n_inner_folds:       int       = 5
    n_iter_search:       int       = 100
    random_seed:         int       = 42
    quantiles:           list      = field(default_factory=lambda: [0.1, 0.5, 0.9])
    param_distributions: dict      = field(default_factory=lambda: GRID_SEARCH)
    version:             str       = None