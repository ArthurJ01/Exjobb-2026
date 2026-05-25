import optuna
from deepod.models.time_series import TranAD
from deepod.models.time_series import DeepSVDDTS
from deepod.models.time_series import COUTA
from deepod.models.time_series import TcnED
from aeon.anomaly_detection.series.distance_based import LOF
from utils.data_loader_hai import load_hai_optuna
from optuna.samplers import TPESampler
from sklearn.metrics import (
    average_precision_score,
)

def run_optuna_study_hai(
    model_class,
    hyperparameters,
    n_trials=10,
    study_name="optuna_hai_study",
    storage="sqlite:///optuna_hai_study.db",
    fixed_model_params=None,
    random_state=42,
):
    """
    HAI Optuna tuning function.

    Parameters
    ----------
    model_class : class
        Model class (e.g., TranAD)
    hyperparameters : dict
        hyperparam_dict = {
            "lr": ("float", 1e-5, 1e-3, True),                  # name: (type, low, high, log)
            "epochs": ("int", 5, 50, False) ,                   # name: (type, low, high, log)
            "batch_size": ("categorical", [32, 64, 128, 256])   # name: (type, choices)
        }
    fixed_model_params : dict
        Parameters always passed to model (e.g., seq_len, stride)
    n_trials : int
        Number of optuna trials in study
    study_name : str
        Name of the study
    storage : str
        where to store the study, ex: sqlite:///optuna_study.db
    random_state : int
        random seed
    """

    x_train, x_val, y_val = load_hai_optuna()

    if fixed_model_params is None:
        if model_class == TranAD or model_class == TcnED:
            fixed_model_params = {
                "seq_len": 50,
                "stride": 5,
            }
        elif model_class == COUTA:
            fixed_model_params = {
                "seq_len": 50,
                "stride": 5,
                "train_val_pc": 0,
            }
        elif model_class == DeepSVDDTS:
            fixed_model_params = {
                "seq_len": 50,
                "stride": 5,
                "prt_steps": 1,
            }
        elif model_class == LOF:
            fixed_model_params = {
                "window_size": 50,
                "stride": 5,
                "n_jobs": 1,
            }
        else:
            fixed_model_params = {}

    def objective(trial):

        params = {}
        for name, spec in hyperparameters.items():
            if spec[0] == "float":
                _, low, high, log = spec
                params[name] = trial.suggest_float(name, low, high, log=log)
            elif spec[0] == "int":
                _, low, high, log = spec
                params[name] = trial.suggest_int(name, low, high, log=log)
            elif spec[0] == "categorical":
                _, choices = spec
                params[name] = trial.suggest_categorical(name, choices)

        # Merge fixed params + tuned params
        model_params = {**fixed_model_params, **params}

        # Initialize model
        model = model_class(**model_params)

        if model_class == LOF:

            import numpy as np

            #x_combined = np.concatenate([x_train, x_val], axis=0)
            scores_val = model.fit_predict(x_val, axis=0)

            #scores_val = scores[len(x_train):]

            # Train & predict (non-novelty)
            #print(model.__dict__)
            #scores_val = LOF.fit_predict(x_val, axis=0)

        else:
            # Train
            model.fit(x_train)

            # Predict anomaly scores
            scores_val = model.decision_function(x_val)

        # Metrics
        pr_auc = average_precision_score(y_val, scores_val)

        trial.set_user_attr("pr_auc", float(pr_auc))
        return float(pr_auc)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=random_state),
    )

    study.optimize(objective, n_trials=n_trials)
    return study
