import optuna
import numpy as np
from deepod.models.time_series import TranAD
from deepod.models.time_series import DeepSVDDTS
from deepod.models.time_series import COUTA
from deepod.models.time_series import TcnED
from aeon.anomaly_detection.series.distance_based import LOF
from optuna.samplers import TPESampler
from sklearn.metrics import (
    average_precision_score,
)

def run_optuna_study(
    model_class,
    hyperparameters,
    train_df,
    validation_ratio = 0.25,
    n_trials=10,
    study_name="optuna_study",
    storage="sqlite:///optuna_study.db",
    fixed_model_params=None,
    random_state=42,
):
    """
    Volvo Optuna tuning function.

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
    train_df : pandas.DataFrame
        training data
    validation_ratio : float
        ratio of training set used as validation set
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

    FEATURES = ["duration", "time_since_last_timestamp"]
    LABEL_COL = "label"

    split_idx = int(len(train_df) * (1 - validation_ratio))

    df_train_optuna = train_df.iloc[:split_idx].copy()
    df_val_optuna = train_df.iloc[split_idx:].copy()

    x_train = df_train_optuna[FEATURES].to_numpy(dtype=np.float32)
    x_val = df_val_optuna[FEATURES].to_numpy(dtype=np.float32)
    y_val = df_val_optuna[LABEL_COL].to_numpy(dtype=int)

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
            # Train & predict (non-novelty)

            scores_val = model.fit_predict(x_val, axis=0)

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


'''
# OLD OPTUNA CODE BLOCK

RANDOM_STATE = 42
FEATURES = ["duration", "time_since_last_timestamp"]
LABEL_COL = "label"

SEQ_LEN = 50
STRIDE = 5
VAL_RATIO = 0.25


# Chronological split: first part for training, last part for validation
split_idx = int(len(x_train) * (1 - VAL_RATIO))

df_train_optuna = df_train.iloc[:split_idx].copy()
df_val_optuna = df_train.iloc[split_idx:].copy()

x_train_optuna = df_train_optuna[FEATURES].to_numpy(dtype=np.float32)
x_val_optuna = df_val_optuna[FEATURES].to_numpy(dtype=np.float32)
y_val_optuna = df_val_optuna[LABEL_COL].to_numpy(dtype=int)

def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    }

    optuna_model = TranAD(
        seq_len=SEQ_LEN,
        stride=STRIDE,
        lr=params["lr"],
        batch_size=params["batch_size"],
    )

    optuna_model.fit(x_train_optuna)
    optuna_scores_val = optuna_model.decision_function(x_val_optuna)

    optuna_roc_auc = roc_auc_score(y_val_optuna, optuna_scores_val)
    optuna_pr_auc = average_precision_score(y_val_optuna, optuna_scores_val)

    optuna_precision, optuna_recall, _ = precision_recall_curve(y_val_optuna, optuna_scores_val)
    optuna_f1 = 2 * optuna_precision * optuna_recall / (optuna_precision + optuna_recall + 1e-12)
    best_idx = np.argmax(optuna_f1)

    best_f1 = optuna_f1[best_idx]
    best_p = optuna_precision[best_idx]
    best_r = optuna_recall[best_idx]

    trial.set_user_attr("roc_auc", float(optuna_roc_auc))
    trial.set_user_attr("pr_auc", float(optuna_pr_auc))
    trial.set_user_attr("best_f1", float(best_f1))
    trial.set_user_attr("best_precision", float(best_p))
    trial.set_user_attr("best_recall", float(best_r))

    return float(optuna_pr_auc)

storage_db = "sqlite:///tranad_optuna.db"

study = optuna.create_study(
    study_name="tranad_test_study",
    storage=storage_db,
    load_if_exists=True,
    direction="maximize",
    sampler=TPESampler(seed=RANDOM_STATE),
)

study.optimize(objective, n_trials=10)

print("Best params:", study.best_params)
print("Best validation PR-AUC:", study.best_value)
print("Best trial extra metrics:", study.best_trial.user_attrs)

'''